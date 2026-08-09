//! `empyrean show` — browse the tables the pipeline commands write.
//!
//! A `more`-style streaming pager over Parquet / CSV / JSON artifacts.
//! Two things shape the whole design:
//!
//! **It streams.** Residual tables run to millions of rows; a browser
//! that loaded one to show you the first screen would be useless on the
//! files that most need browsing. Every reader here pulls one record
//! batch / line / record at a time, so the first page draws immediately
//! and resident memory does not track file size. The cost of that choice
//! is paging backwards, which re-reads (see [`pager`]).
//!
//! **It needs no engine.** `show` never calls `libempyrean` — the
//! artifacts are ordinary Parquet and text, so reading them is a
//! client-side concern. `empyrean show` works on a machine that has the
//! CLI and no runtime library, and on files copied off a cluster.

mod catalog;
mod error;
mod pager;
mod render;
mod source;
mod table;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use clap::Args;

pub use error::ShowError;

use render::FloatFormat;
use table::{Table, View};

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// Artifact to display, or a directory to browse.
    ///
    /// A directory lists what it holds and asks which file to open.
    /// Defaults to the current directory.
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    /// Directory to browse. Equivalent to passing the directory
    /// positionally; accepted so `show` reads the same as the pipeline
    /// commands that wrote the files.
    #[arg(long, value_name = "DIR", conflicts_with = "path")]
    out_dir: Option<PathBuf>,

    /// Stop after this many rows.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,

    /// Show only these columns, in this order (comma separated).
    ///
    /// The way to read a wide table — an orbits Parquet carries 82
    /// columns, most of them a covariance block.
    #[arg(long, value_name = "A,B,C", value_delimiter = ',')]
    columns: Option<Vec<String>>,

    /// Show only rows containing this text (case-insensitive), matched
    /// against the row as displayed.
    #[arg(long, value_name = "TEXT")]
    filter: Option<String>,

    /// Omit the column header.
    #[arg(long)]
    no_header: bool,

    /// Print every digit of every float.
    ///
    /// The default shows 6 significant digits, which is readable but not
    /// the exact stored value. Use this when you need the bits — the
    /// output round-trips to the same `f64`.
    #[arg(long)]
    full_precision: bool,
}

pub fn run(args: ShowArgs) -> Result<()> {
    let target = args
        .path
        .or(args.out_dir)
        .unwrap_or_else(|| PathBuf::from("."));

    let view = View {
        columns: args.columns,
        filter: args.filter,
        limit: args.limit,
        floats: if args.full_precision {
            FloatFormat::Full
        } else {
            FloatFormat::default()
        },
    };

    if !target.exists() {
        return Err(ShowError::NotFound {
            path: target.clone(),
        })
        .context("empyrean show");
    }

    if target.is_dir() {
        browse_directory(&target, view, args.no_header)
    } else {
        show_file(&target, view, args.no_header)
    }
}

/// List a directory, then open whichever artifact is chosen.
fn browse_directory(dir: &Path, view: View, no_header: bool) -> Result<()> {
    let artifacts = catalog::list(dir).context("empyrean show")?;
    let interactive = pager::is_interactive();

    print!("{}", catalog::render_listing(dir, &artifacts, interactive));
    std::io::stdout().flush().ok();

    if !interactive {
        // Piped: the listing is the output. Nothing to pick with, and
        // guessing which file was meant would be worse than stopping.
        return Ok(());
    }
    println!();
    let Some(index) = catalog::prompt_choice(&artifacts).context("empyrean show")? else {
        return Ok(());
    };
    // The listing already established each file's format; reuse it rather
    // than re-sniffing the chosen one.
    let chosen = &artifacts[index];
    let table = Table::open_as(&chosen.path, chosen.format, view).context("empyrean show")?;
    page_or_stream(table, no_header)
}

/// Open one named file and display it.
fn show_file(path: &Path, view: View, no_header: bool) -> Result<()> {
    let table = Table::open(path, view).context("empyrean show")?;
    page_or_stream(table, no_header)
}

/// Page the table, or stream it plainly when stdout is not a terminal.
fn page_or_stream(table: Table, no_header: bool) -> Result<()> {
    if pager::is_interactive() {
        pager::run(table, !no_header).context("empyrean show")?;
    } else {
        stream_plain(table, no_header).context("empyrean show")?;
    }
    Ok(())
}

/// How many rows are held to measure column widths before printing
/// starts, when the output is a pipe.
///
/// A streaming writer cannot know the widest cell without reading
/// everything, so it learns from a bounded prefix. Large enough that a
/// realistic table lands aligned; small enough that memory stays flat on
/// a file of any size.
const WIDTH_SAMPLE_ROWS: usize = 512;

/// Write the whole table as aligned text, for a pipe or a file.
///
/// No truncation here, ever. A terminal has a width to respect; a pipe
/// does not, and clipping a number on its way into `awk` would corrupt
/// the value. Cells wider than their column simply widen the line.
fn stream_plain(mut table: Table, no_header: bool) -> Result<(), ShowError> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    let header: Vec<String> = table.header().to_vec();
    let sample = table.take(WIDTH_SAMPLE_ROWS)?;

    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for row in &sample {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }

    let write_row = |out: &mut dyn Write, row: &[String]| -> std::io::Result<()> {
        let mut line = String::new();
        for (i, cell) in row.iter().enumerate() {
            if i > 0 {
                line.push_str("  ");
            }
            let width = widths.get(i).copied().unwrap_or(0);
            let pad = width.saturating_sub(cell.chars().count());
            line.push_str(cell);
            line.push_str(&" ".repeat(pad));
        }
        writeln!(out, "{}", line.trim_end())
    };

    // Owned, so the error path does not hold a borrow of `table` across
    // the streaming loop's mutable calls.
    let source_path = table.path().to_path_buf();
    let io_err = |e| ShowError::io(&source_path, "write the table from", e);

    if !no_header && !header.is_empty() {
        write_row(&mut out, &header).map_err(io_err)?;
    }
    let mut rows = sample.len();
    for row in &sample {
        write_row(&mut out, row).map_err(io_err)?;
    }
    while let Some(row) = table.next_rendered()? {
        write_row(&mut out, &row).map_err(io_err)?;
        rows += 1;
    }

    // An empty table is a header and a count, not silence — otherwise a
    // zero-row file is indistinguishable from a command that did nothing.
    if rows == 0 {
        writeln!(out, "0 rows").map_err(io_err)?;
    }
    out.flush().map_err(io_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        show: ShowArgs,
    }

    fn parse(args: &[&str]) -> ShowArgs {
        let mut argv = vec!["show"];
        argv.extend_from_slice(args);
        Harness::try_parse_from(argv).expect("parse").show
    }

    #[test]
    fn a_bare_path_is_the_target() {
        assert_eq!(
            parse(&["out/states.parquet"]).path.unwrap(),
            PathBuf::from("out/states.parquet")
        );
    }

    #[test]
    fn out_dir_is_an_alias_for_the_positional_directory() {
        assert_eq!(
            parse(&["--out-dir", "./out"]).out_dir.unwrap(),
            PathBuf::from("./out")
        );
    }

    /// Passing both would leave two different targets with no rule for
    /// which wins, so clap rejects it up front.
    #[test]
    fn a_path_and_out_dir_together_are_rejected() {
        assert!(Harness::try_parse_from(["show", "./out", "--out-dir", "./other"]).is_err());
    }

    #[test]
    fn columns_split_on_commas_and_keep_their_order() {
        let args = parse(&["f.parquet", "--columns", "orbit_id,z,x"]);
        assert_eq!(
            args.columns.unwrap(),
            vec!["orbit_id".to_string(), "z".to_string(), "x".to_string()]
        );
    }

    /// Repeating the flag accumulates, so `--columns a --columns b` works
    /// the same as `--columns a,b`.
    #[test]
    fn columns_accumulate_across_repeats() {
        let args = parse(&["f.parquet", "--columns", "a", "--columns", "b,c"]);
        assert_eq!(
            args.columns.unwrap(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn flags_default_off() {
        let args = parse(&["f.parquet"]);
        assert!(!args.no_header);
        assert!(!args.full_precision);
        assert!(args.limit.is_none());
        assert!(args.filter.is_none());
    }

    #[test]
    fn limit_parses_as_a_row_count() {
        assert_eq!(parse(&["f.parquet", "--limit", "20"]).limit, Some(20));
    }
}
