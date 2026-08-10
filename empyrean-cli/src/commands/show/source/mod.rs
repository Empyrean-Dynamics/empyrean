//! Streaming row sources: one per on-disk format.
//!
//! Every reader here is a **pull** stream. A source holds at most one
//! decode buffer (a Parquet record batch, a CSV line, a JSON record) no
//! matter how large the file is, so a multi-million-row residuals table
//! costs the same resident memory as a ten-row one and the first page
//! renders without touching the tail of the file.
//!
//! `show` never calls into `libempyrean`. These readers are the reason:
//! the artifacts are plain Parquet / CSV / JSON, so browsing them is a
//! client-side concern and works on a machine with no engine installed.

mod csv;
mod json;
mod parquet_source;

use std::fmt;
use std::io::Read;
use std::path::Path;

use super::error::{ShowError, describe_head};

/// One table cell.
///
/// `Null` (absent) and `Float(NAN)` (present, not a number) are distinct
/// on purpose: in a residuals table an unevaluated rejection criterion is
/// NaN while an unset star catalog is null, and collapsing the two would
/// misreport both.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Text(String),
}

/// A pull stream of rows, all with the same width as [`RowSource::columns`].
pub trait RowSource {
    /// The column names, in file order. Fixed for the life of the source.
    fn columns(&self) -> &[String];

    /// The next row, or `None` at end of stream.
    fn next_row(&mut self) -> Result<Option<Vec<Cell>>, ShowError>;
}

/// The formats `show` can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Parquet,
    Csv,
    /// A JSON array of objects *or* newline-delimited JSON objects. The
    /// reader accepts both because `empyrean --format json` writes a
    /// pretty-printed array while other tooling in this space writes
    /// JSONL, and a browser that refused one of them would be a trap.
    Json,
}

impl Format {
    /// The extensions that name this format.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Self::Parquet => &["parquet"],
            Self::Csv => &["csv"],
            Self::Json => &["json", "jsonl", "ndjson"],
        }
    }

    /// Map an extension (without the dot, any case) to a format.
    pub fn from_extension(ext: &str) -> Option<Self> {
        let lower = ext.to_ascii_lowercase();
        [Self::Parquet, Self::Csv, Self::Json]
            .into_iter()
            .find(|f| f.extensions().contains(&lower.as_str()))
    }

    /// Guess a format from a file's leading bytes.
    ///
    /// Only used when the extension does not settle it. Parquet is exact
    /// (a magic number); the text formats are a guess, which is why a
    /// disagreement between extension and content is reported rather than
    /// silently resolved.
    pub fn sniff(head: &[u8]) -> Option<Self> {
        if head.starts_with(b"PAR1") {
            return Some(Self::Parquet);
        }
        let first = head.iter().find(|b| !b.is_ascii_whitespace())?;
        match first {
            b'[' | b'{' => Some(Self::Json),
            _ => {
                // A header line with a comma in it is the only remaining
                // shape this tool writes.
                let line: &[u8] = head.split(|&b| b == b'\n').next()?;
                line.contains(&b',').then_some(Self::Csv)
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Csv => "csv",
            Self::Json => "json",
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Read the first bytes of a file, for sniffing and for error messages.
fn read_head(path: &Path) -> Result<Vec<u8>, ShowError> {
    let mut f = std::fs::File::open(path).map_err(|e| ShowError::io(path, "open", e))?;
    let mut buf = vec![0_u8; 512];
    let mut filled = 0;
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(ShowError::io(path, "read", e)),
        }
    }
    buf.truncate(filled);
    Ok(buf)
}

/// Establish a file's format from its extension, cross-checked against
/// its content.
///
/// The extension decides when it is one we know, but a Parquet magic
/// number that disagrees is an error rather than an override: a
/// `residuals.csv` that is actually Parquet means something upstream
/// wrote the wrong file, and quietly reading it anyway would hide that.
/// When the extension says nothing, the content decides.
pub fn detect_format(path: &Path) -> Result<Format, ShowError> {
    if !path.exists() {
        return Err(ShowError::NotFound {
            path: path.to_path_buf(),
        });
    }
    let head = read_head(path)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);

    if let Some(claimed) = ext.as_deref().and_then(Format::from_extension) {
        let is_parquet_magic = head.starts_with(b"PAR1");
        // An empty file is allowed through on its extension alone: an
        // empty CSV is a real (if useless) artifact, and the reader
        // reports "0 rows" rather than a parse failure.
        if !head.is_empty() && (claimed == Format::Parquet) != is_parquet_magic {
            return Err(ShowError::ContentMismatch {
                path: path.to_path_buf(),
                claimed: claimed.label(),
                head: describe_head(&head),
            });
        }
        return Ok(claimed);
    }

    Format::sniff(&head).ok_or_else(|| ShowError::UnknownFormat {
        path: path.to_path_buf(),
        extension: ext,
        head: describe_head(&head),
    })
}

/// Open a fresh stream over `path`, whose format is already established.
///
/// Called once per pass. The pager re-calls it to restart a stream (see
/// [`super::pager`] on backward paging), which is why it takes a path
/// rather than handing back a rewindable handle: Parquet row-group state
/// and CSV quote state are both cheaper to rebuild than to rewind
/// correctly.
pub fn open_as(path: &Path, format: Format) -> Result<Box<dyn RowSource>, ShowError> {
    match format {
        Format::Parquet => Ok(Box::new(parquet_source::ParquetSource::open(path)?)),
        Format::Csv => Ok(Box::new(csv::CsvSource::open(path)?)),
        Format::Json => Ok(Box::new(json::JsonSource::open(path)?)),
    }
}

/// How many rows a file holds, and whether that number was counted or
/// estimated.
///
/// The directory listing wants a row count for every artifact, and for a
/// multi-gigabyte CSV counting them means reading the whole file. Past a
/// size threshold the count is extrapolated from a sample instead — and
/// says so, with a `~`, rather than presenting a guess as a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowCount {
    Exact(u64),
    Approx(u64),
}

/// Files at or below this size are counted exactly for the listing.
const COUNT_EXACTLY_UP_TO_BYTES: u64 = 64 * 1024 * 1024;

/// How much of an oversized file is read to estimate its row count.
const COUNT_SAMPLE_BYTES: u64 = 1024 * 1024;

impl RowCount {
    /// Render for the listing: `1234`, or `~1.2M` when estimated.
    pub fn render(self) -> String {
        match self {
            Self::Exact(n) => n.to_string(),
            Self::Approx(n) => format!("~{}", human_count(n)),
        }
    }
}

fn human_count(n: u64) -> String {
    match n {
        0..=9_999 => n.to_string(),
        10_000..=999_999 => format!("{:.1}K", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// Count the rows in a file for the directory listing.
///
/// Parquet answers from its footer without decoding a single page. The
/// text formats have to be walked, so they are walked only up to
/// [`COUNT_EXACTLY_UP_TO_BYTES`]; past that the count is extrapolated
/// from the first [`COUNT_SAMPLE_BYTES`] and flagged approximate.
pub fn count_rows(path: &Path, format: Format) -> Result<RowCount, ShowError> {
    if format == Format::Parquet {
        return parquet_source::row_count(path).map(RowCount::Exact);
    }
    let size = std::fs::metadata(path)
        .map_err(|e| ShowError::io(path, "stat", e))?
        .len();

    if size <= COUNT_EXACTLY_UP_TO_BYTES {
        let mut source = open_as(path, format)?;
        let mut n = 0_u64;
        while source.next_row()?.is_some() {
            n += 1;
        }
        return Ok(RowCount::Exact(n));
    }

    // Sample: count the rows in the first megabyte and scale by size. The
    // sample is taken through the real reader, so a quoted CSV field with
    // an embedded newline counts as the one row it is.
    let mut source = open_as(path, format)?;
    let mut sampled = 0_u64;
    let mut bytes_seen = 0_u64;
    while bytes_seen < COUNT_SAMPLE_BYTES {
        match source.next_row()? {
            Some(row) => {
                sampled += 1;
                bytes_seen += row_bytes(&row);
            }
            None => return Ok(RowCount::Exact(sampled)),
        }
    }
    if bytes_seen == 0 {
        return Ok(RowCount::Approx(0));
    }
    let scaled = (sampled as f64) * (size as f64) / (bytes_seen as f64);
    Ok(RowCount::Approx(scaled.round().max(0.0) as u64))
}

/// A rough on-disk footprint for one decoded row, used only to scale the
/// row-count estimate. Exactness is not the point; proportionality is.
fn row_bytes(row: &[Cell]) -> u64 {
    row.iter()
        .map(|c| match c {
            Cell::Text(s) => s.len() as u64 + 3,
            Cell::Null => 4,
            _ => 8,
        })
        .sum::<u64>()
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_detection_is_case_insensitive() {
        assert_eq!(Format::from_extension("parquet"), Some(Format::Parquet));
        assert_eq!(Format::from_extension("PARQUET"), Some(Format::Parquet));
        assert_eq!(Format::from_extension("Csv"), Some(Format::Csv));
        assert_eq!(Format::from_extension("json"), Some(Format::Json));
        assert_eq!(Format::from_extension("jsonl"), Some(Format::Json));
        assert_eq!(Format::from_extension("ndjson"), Some(Format::Json));
        assert_eq!(Format::from_extension("txt"), None);
        assert_eq!(Format::from_extension(""), None);
    }

    #[test]
    fn sniff_recognizes_parquet_magic() {
        assert_eq!(Format::sniff(b"PAR1\x15\x04"), Some(Format::Parquet));
        // Truncated magic is not parquet.
        assert_eq!(Format::sniff(b"PAR"), None);
    }

    #[test]
    fn sniff_recognizes_json_array_and_jsonl() {
        assert_eq!(Format::sniff(b"[\n  {\"a\": 1}\n]"), Some(Format::Json));
        assert_eq!(
            Format::sniff(b"{\"a\": 1}\n{\"a\": 2}\n"),
            Some(Format::Json)
        );
        assert_eq!(Format::sniff(b"   \n\t [{}]"), Some(Format::Json));
    }

    #[test]
    fn sniff_recognizes_csv_by_a_comma_in_the_header_line() {
        assert_eq!(Format::sniff(b"a,b,c\n1,2,3\n"), Some(Format::Csv));
        // A single-column file with no comma is not distinguishable from
        // prose, and is reported as unknown rather than guessed at.
        assert_eq!(Format::sniff(b"just some text\nmore text\n"), None);
        assert_eq!(Format::sniff(b""), None);
    }

    /// The comma has to be in the *header* line. A log file whose tenth
    /// line happens to contain a comma is not a CSV.
    #[test]
    fn sniff_only_considers_the_first_line_for_csv() {
        assert_eq!(Format::sniff(b"header\nlater,line,here\n"), None);
    }

    #[test]
    fn human_count_scales() {
        assert_eq!(human_count(0), "0");
        assert_eq!(human_count(9_999), "9999");
        assert_eq!(human_count(12_345), "12.3K");
        assert_eq!(human_count(500_000), "500.0K");
        assert_eq!(human_count(2_500_000), "2.5M");
        assert_eq!(human_count(25_000_000), "25.0M");
    }

    #[test]
    fn approximate_counts_are_marked() {
        assert_eq!(RowCount::Exact(17).render(), "17");
        assert_eq!(RowCount::Approx(2_500_000).render(), "~2.5M");
    }
}
