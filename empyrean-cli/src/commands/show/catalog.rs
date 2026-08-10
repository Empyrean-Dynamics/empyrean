//! Listing an output directory, and picking a file from it.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use super::error::ShowError;
use super::render::fit;
use super::source::{Format, RowCount, count_rows, detect_format};

/// One artifact found in an output directory.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub path: PathBuf,
    pub format: Format,
    pub rows: RowCount,
    pub bytes: u64,
    pub description: &'static str,
}

impl Artifact {
    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// What each artifact the pipeline commands write actually contains.
///
/// Keyed on the file stem the writers use. Anything else gets a generic
/// label — `show` browses any table, not only the ones this CLI wrote.
fn describe(stem: &str) -> &'static str {
    match stem {
        "states" => "Propagated states — position, velocity, covariance at the target epoch",
        "events" => "Detected events — close approaches, impacts, and their geometry",
        "ephemeris" => "Predicted ephemeris — RA/Dec per observer and epoch",
        "fitted_orbit" => "Fitted orbit — solved state, covariance, and non-gravitational terms",
        "residuals" => "Per-observation residuals — RA/Dec, χ², and rejection outcome",
        _ => "Table",
    }
}

/// Every readable artifact in `dir`, sorted by filename.
///
/// Files whose format cannot be established are skipped here rather than
/// failing the listing — a stray `README.md` in an output directory should
/// not stop you browsing the Parquet next to it. Naming an unreadable
/// file *directly* is still a hard error.
pub fn list(dir: &Path) -> Result<Vec<Artifact>, ShowError> {
    let entries = std::fs::read_dir(dir).map_err(|e| ShowError::io(dir, "list", e))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| ShowError::io(dir, "list", e))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(format) = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(Format::from_extension)
        else {
            continue;
        };
        // Cross-check content, and skip anything mislabelled rather than
        // refusing to list the directory.
        if detect_format(&path).is_err() {
            continue;
        }
        let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
        let rows = count_rows(&path, format)?;
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.push(Artifact {
            path,
            format,
            rows,
            bytes,
            description: describe(&stem),
        });
    }
    out.sort_by_key(|a| a.file_name());
    if out.is_empty() {
        return Err(ShowError::NoArtifacts {
            dir: dir.to_path_buf(),
        });
    }
    Ok(out)
}

/// Human-readable byte size.
pub fn human_bytes(n: u64) -> String {
    const KIB: f64 = 1024.0;
    let n = n as f64;
    if n < KIB {
        return format!("{n:.0} B");
    }
    for (limit, unit) in [
        (KIB * KIB, "KiB"),
        (KIB * KIB * KIB, "MiB"),
        (KIB * KIB * KIB * KIB, "GiB"),
    ] {
        if n < limit {
            return format!("{:.1} {unit}", n / (limit / KIB));
        }
    }
    format!("{:.1} TiB", n / (KIB * KIB * KIB * KIB))
}

/// Render the listing as aligned text.
///
/// The same text goes to a terminal and to a pipe — the listing is useful
/// output in its own right, so `empyrean show ./out | grep residuals`
/// works.
pub fn render_listing(dir: &Path, artifacts: &[Artifact], numbered: bool) -> String {
    let name_w = artifacts
        .iter()
        .map(|a| a.file_name().chars().count())
        .max()
        .unwrap_or(4)
        .max(4);
    let rows_w = artifacts
        .iter()
        .map(|a| a.rows.render().chars().count())
        .max()
        .unwrap_or(4)
        .max(4);
    let size_w = artifacts
        .iter()
        .map(|a| human_bytes(a.bytes).chars().count())
        .max()
        .unwrap_or(4)
        .max(4);

    let mut out = format!("{}\n\n", dir.display());
    let index_w = if numbered { 4 } else { 0 };
    if numbered {
        out.push_str(&" ".repeat(index_w));
    }
    out.push_str(&format!(
        "{}  {}  {}  {}\n",
        fit("FILE", name_w),
        fit("ROWS", rows_w),
        fit("SIZE", size_w),
        "DESCRIPTION"
    ));
    for (i, a) in artifacts.iter().enumerate() {
        if numbered {
            out.push_str(&fit(&format!("{:>2}.", i + 1), index_w));
        }
        out.push_str(
            format!(
                "{}  {}  {}  {}\n",
                fit(&a.file_name(), name_w),
                fit(&a.rows.render(), rows_w),
                fit(&human_bytes(a.bytes), size_w),
                a.description
            )
            .trim_end(),
        );
        out.push('\n');
    }
    out
}

/// Ask which artifact to open.
///
/// Number entry plus enter — the baseline that works on every terminal,
/// over ssh, and inside a `screen` session. Returns `None` if the user
/// declined.
pub fn prompt_choice(artifacts: &[Artifact]) -> Result<Option<usize>, ShowError> {
    let mut stdout = std::io::stdout();
    let stdin = std::io::stdin();
    loop {
        write!(stdout, "Open [1-{}] (q to quit): ", artifacts.len())
            .and_then(|()| stdout.flush())
            .map_err(|e| ShowError::io(Path::new("<stdout>"), "prompt on", e))?;

        let mut line = String::new();
        let read = stdin
            .lock()
            .read_line(&mut line)
            .map_err(|e| ShowError::io(Path::new("<stdin>"), "read a selection from", e))?;
        if read == 0 {
            // stdin closed (^D).
            println!();
            return Ok(None);
        }
        match parse_choice(line.trim(), artifacts.len()) {
            Choice::Quit => return Ok(None),
            Choice::Pick(i) => return Ok(Some(i)),
            Choice::Retry(message) => println!("{message}"),
        }
    }
}

/// The outcome of parsing one line of picker input.
#[derive(Debug, PartialEq, Eq)]
pub enum Choice {
    Pick(usize),
    Quit,
    Retry(String),
}

/// Parse picker input into a zero-based index.
pub fn parse_choice(input: &str, count: usize) -> Choice {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Choice::Retry(format!("Enter a number from 1 to {count}, or q to quit."));
    }
    if trimmed.eq_ignore_ascii_case("q") || trimmed.eq_ignore_ascii_case("quit") {
        return Choice::Quit;
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n >= 1 && n <= count => Choice::Pick(n - 1),
        Ok(n) => Choice::Retry(format!("{n} is out of range — pick 1 to {count}.")),
        Err(_) => Choice::Retry(format!(
            "`{trimmed}` is not a number — pick 1 to {count}, or q."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_artifact_names_get_a_real_description() {
        for stem in ["states", "events", "ephemeris", "fitted_orbit", "residuals"] {
            assert_ne!(describe(stem), "Table", "{stem} must be described");
        }
    }

    /// `show` browses any table, not only ours, so an unknown name is
    /// labelled generically rather than rejected.
    #[test]
    fn unknown_names_get_a_generic_label() {
        assert_eq!(describe("my_export"), "Table");
        assert_eq!(describe(""), "Table");
    }

    #[test]
    fn byte_sizes_are_human_readable() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KiB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn picker_accepts_a_one_based_number() {
        assert_eq!(parse_choice("1", 3), Choice::Pick(0));
        assert_eq!(parse_choice("3", 3), Choice::Pick(2));
        assert_eq!(parse_choice("  2  ", 3), Choice::Pick(1));
    }

    #[test]
    fn picker_quits_on_q() {
        assert_eq!(parse_choice("q", 3), Choice::Quit);
        assert_eq!(parse_choice("Q", 3), Choice::Quit);
        assert_eq!(parse_choice("quit", 3), Choice::Quit);
    }

    /// Bad input re-prompts with a message; it never picks a file the
    /// user did not ask for.
    #[test]
    fn picker_rejects_out_of_range_and_non_numeric_input() {
        assert!(matches!(parse_choice("0", 3), Choice::Retry(_)));
        assert!(matches!(parse_choice("4", 3), Choice::Retry(_)));
        assert!(matches!(parse_choice("-1", 3), Choice::Retry(_)));
        assert!(matches!(parse_choice("residuals", 3), Choice::Retry(_)));
        assert!(matches!(parse_choice("", 3), Choice::Retry(_)));
    }

    fn artifact(name: &str, rows: RowCount, bytes: u64) -> Artifact {
        Artifact {
            path: PathBuf::from("/out").join(name),
            format: Format::Parquet,
            rows,
            bytes,
            description: describe(Path::new(name).file_stem().unwrap().to_str().unwrap()),
        }
    }

    #[test]
    fn listing_is_aligned_and_numbered_for_the_picker() {
        let artifacts = vec![
            artifact("residuals.parquet", RowCount::Exact(128), 4096),
            artifact("states.parquet", RowCount::Exact(1), 20537),
        ];
        let text = render_listing(Path::new("/out"), &artifacts, true);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "/out");
        assert_eq!(lines[1], "");
        assert_eq!(
            lines[2],
            "    FILE               ROWS  SIZE      DESCRIPTION"
        );
        assert_eq!(
            lines[3],
            " 1. residuals.parquet  128   4.0 KiB   Per-observation residuals — RA/Dec, χ², and rejection outcome"
        );
        assert_eq!(
            lines[4],
            " 2. states.parquet     1     20.1 KiB  Propagated states — position, velocity, covariance at the target epoch"
        );
    }

    /// Piped, the listing drops the picker numbers — there is nothing to
    /// pick with.
    #[test]
    fn listing_without_numbers_starts_at_the_filename() {
        let artifacts = vec![artifact("states.parquet", RowCount::Exact(1), 100)];
        let text = render_listing(Path::new("/out"), &artifacts, false);
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[2].starts_with("FILE"), "{:?}", lines[2]);
        assert!(lines[3].starts_with("states.parquet"), "{:?}", lines[3]);
    }

    /// An estimated row count must stay marked in the listing.
    #[test]
    fn approximate_counts_show_a_tilde_in_the_listing() {
        let artifacts = vec![artifact(
            "residuals.csv",
            RowCount::Approx(2_500_000),
            1 << 30,
        )];
        let text = render_listing(Path::new("/out"), &artifacts, true);
        assert!(text.contains("~2.5M"), "{text}");
    }
}
