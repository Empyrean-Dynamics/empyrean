//! The view a reader sees: a source, plus the column selection, the row
//! filter, and the row limit applied on top of it.
//!
//! Everything here is still streaming. Projection drops cells as rows go
//! past; the filter tests each row and forgets the ones it rejects; the
//! limit stops the pull. No stage accumulates rows.

use std::path::{Path, PathBuf};

use super::error::ShowError;
use super::render::{FloatFormat, render_cell};
use super::source::{self, Cell, Format, RowSource};

/// The column selection, filter, and limit a `show` invocation carries.
#[derive(Debug, Clone, Default)]
pub struct View {
    /// `--columns`: the subset to display, in the order given. `None`
    /// means every column in file order.
    pub columns: Option<Vec<String>>,
    /// `/` in the pager, applied as a case-insensitive substring test.
    pub filter: Option<String>,
    /// `--limit`: stop after this many matching rows.
    pub limit: Option<usize>,
    /// How floats are written.
    pub floats: FloatFormat,
}

/// A restartable stream of rendered rows over one file.
///
/// Held open across forward paging; dropped and rebuilt to page
/// backwards (see [`super::pager`]).
pub struct Table {
    path: PathBuf,
    format: Format,
    source: Box<dyn RowSource>,
    /// Indices into the source's columns, in display order.
    projection: Vec<usize>,
    header: Vec<String>,
    view: View,
    /// How many matching rows have been handed out.
    emitted: usize,
}

impl Table {
    /// Open `path` and resolve `--columns` against its real schema.
    pub fn open(path: &Path, view: View) -> Result<Self, ShowError> {
        let format = source::detect_format(path)?;
        Self::open_as(path, format, view)
    }

    pub fn open_as(path: &Path, format: Format, view: View) -> Result<Self, ShowError> {
        let source = source::open_as(path, format)?;
        let available = source.columns().to_vec();
        let projection = resolve_projection(path, &available, view.columns.as_deref())?;
        let header = projection.iter().map(|&i| available[i].clone()).collect();
        Ok(Self {
            path: path.to_path_buf(),
            format,
            source,
            projection,
            header,
            view,
            emitted: 0,
        })
    }

    /// Start again from the first row, with the same projection and view.
    pub fn restart(&self) -> Result<Self, ShowError> {
        Self::open_as(&self.path, self.format, self.view.clone())
    }

    /// A clone of the view, for building a restarted table with one field
    /// changed (the pager does this when the filter is edited).
    pub fn view(&self) -> &View {
        &self.view
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The displayed column names, after `--columns`.
    pub fn header(&self) -> &[String] {
        &self.header
    }

    /// The next row that passes the filter, already rendered to strings,
    /// or `None` at end of stream or once `--limit` is reached.
    pub fn next_rendered(&mut self) -> Result<Option<Vec<String>>, ShowError> {
        if self.view.limit.is_some_and(|n| self.emitted >= n) {
            return Ok(None);
        }
        loop {
            let Some(row) = self.source.next_row()? else {
                return Ok(None);
            };
            let rendered: Vec<String> = self
                .projection
                .iter()
                .map(|&i| render_cell(row.get(i).unwrap_or(&Cell::Null), self.view.floats))
                .collect();
            if matches(&rendered, self.view.filter.as_deref()) {
                self.emitted += 1;
                return Ok(Some(rendered));
            }
        }
    }

    /// Pull up to `n` rows.
    pub fn take(&mut self, n: usize) -> Result<Vec<Vec<String>>, ShowError> {
        let mut out = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            match self.next_rendered()? {
                Some(row) => out.push(row),
                None => break,
            }
        }
        Ok(out)
    }

    /// Discard `n` matching rows without rendering them into a buffer.
    ///
    /// This is how backward paging gets to its target row: the rows in
    /// between are streamed and dropped, not stored. Returns how many
    /// were actually skipped, which is fewer than `n` at end of stream.
    pub fn skip(&mut self, n: usize) -> Result<usize, ShowError> {
        let mut skipped = 0;
        for _ in 0..n {
            match self.next_rendered()? {
                Some(_) => skipped += 1,
                None => break,
            }
        }
        Ok(skipped)
    }
}

/// Does a rendered row satisfy the filter?
///
/// The test runs against the text on screen, so what you see is what you
/// search: `/NaN` finds not-a-number cells, `/568` finds the observatory
/// code as displayed. Case-insensitive, because typing an exact-case
/// designation into a pager is a chore.
pub fn matches(rendered: &[String], filter: Option<&str>) -> bool {
    let Some(needle) = filter else {
        return true;
    };
    if needle.is_empty() {
        return true;
    }
    let needle = needle.to_lowercase();
    rendered
        .iter()
        .any(|cell| cell.to_lowercase().contains(&needle))
}

/// Map `--columns` onto the file's real schema.
///
/// A name that is not in the file is an error listing what is, rather
/// than a silently narrower table.
fn resolve_projection(
    path: &Path,
    available: &[String],
    requested: Option<&[String]>,
) -> Result<Vec<usize>, ShowError> {
    let Some(requested) = requested else {
        return Ok((0..available.len()).collect());
    };
    requested
        .iter()
        .map(|name| {
            available
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| ShowError::UnknownColumn {
                    path: path.to_path_buf(),
                    name: name.clone(),
                    available: available.to_vec(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn no_selection_projects_every_column_in_file_order() {
        let available = names(&["a", "b", "c"]);
        let got = resolve_projection(Path::new("/f.csv"), &available, None).unwrap();
        assert_eq!(got, vec![0, 1, 2]);
    }

    /// `--columns` both subsets *and* reorders — the order given is the
    /// order shown.
    #[test]
    fn selection_subsets_and_reorders() {
        let available = names(&["a", "b", "c"]);
        let want = names(&["c", "a"]);
        let got = resolve_projection(Path::new("/f.csv"), &available, Some(&want)).unwrap();
        assert_eq!(got, vec![2, 0]);
    }

    /// A repeated column is allowed: pinning an identifier next to a far
    /// column is a legitimate way to read a 82-column table.
    #[test]
    fn a_column_may_be_selected_twice() {
        let available = names(&["a", "b"]);
        let want = names(&["a", "b", "a"]);
        let got = resolve_projection(Path::new("/f.csv"), &available, Some(&want)).unwrap();
        assert_eq!(got, vec![0, 1, 0]);
    }

    #[test]
    fn an_unknown_column_names_what_is_available() {
        let available = names(&["orbit_id", "x", "y"]);
        let want = names(&["z"]);
        let err = resolve_projection(Path::new("/f.parquet"), &available, Some(&want)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no column named `z`"), "{msg}");
        assert!(msg.contains("orbit_id, x, y"), "{msg}");
    }

    #[test]
    fn no_filter_matches_everything() {
        assert!(matches(&names(&["a", "b"]), None));
        assert!(matches(&names(&["a", "b"]), Some("")));
    }

    #[test]
    fn filter_is_a_case_insensitive_substring_over_the_whole_row() {
        let row = names(&["K23A00B", "568", "12.5"]);
        assert!(matches(&row, Some("k23")));
        assert!(matches(&row, Some("568")));
        assert!(matches(&row, Some("2.5")));
        assert!(!matches(&row, Some("999")));
    }

    /// The filter sees the text on screen, so it can find NaN cells and
    /// cannot find nulls (which render empty) by accident.
    #[test]
    fn filter_matches_rendered_text() {
        assert!(matches(&names(&["NaN", "1"]), Some("nan")));
        assert!(!matches(&names(&["", "1"]), Some("null")));
    }
}
