//! Typed failures for `empyrean show`.
//!
//! Every variant names the file it was reading and what it was trying to
//! do with it. A reader that cannot understand a byte says so and stops —
//! it never skips the row, substitutes a default, or degrades the table
//! to the subset it happened to parse.

use std::fmt;
use std::path::{Path, PathBuf};

/// What went wrong while listing, opening, or reading a table.
#[derive(Debug)]
pub enum ShowError {
    /// The path does not exist.
    NotFound { path: PathBuf },

    /// The path exists but the format could not be established from
    /// either its extension or its leading bytes.
    UnknownFormat {
        path: PathBuf,
        extension: Option<String>,
        head: String,
    },

    /// The extension promised one format and the content is another —
    /// the file is mislabelled or truncated.
    ContentMismatch {
        path: PathBuf,
        claimed: &'static str,
        head: String,
    },

    /// The file could not be opened or read.
    Io {
        path: PathBuf,
        action: &'static str,
        source: std::io::Error,
    },

    /// The Parquet reader rejected the file.
    Parquet {
        path: PathBuf,
        action: &'static str,
        source: parquet::errors::ParquetError,
    },

    /// A Parquet column has an Arrow type this renderer has no cell
    /// representation for. Rendering it as a placeholder would be a
    /// silent substitution, so it is a hard error naming the column.
    UnsupportedColumnType {
        path: PathBuf,
        column: String,
        data_type: String,
    },

    /// A CSV record did not have the same field count as the header.
    CsvFieldCount {
        path: PathBuf,
        line: u64,
        expected: usize,
        found: usize,
    },

    /// A CSV file ended inside an open quoted field.
    CsvUnterminatedQuote { path: PathBuf, line: u64 },

    /// A JSON record failed to parse.
    Json {
        path: PathBuf,
        record: u64,
        detail: String,
    },

    /// A JSON record was not an object, so it has no columns.
    JsonNotAnObject {
        path: PathBuf,
        record: u64,
        found: &'static str,
    },

    /// A JSON record carried a key absent from the first record. The
    /// header is pinned by record 1 so it can stay on screen while the
    /// file streams, which leaves nowhere to put a late-arriving column —
    /// and dropping it would lose data.
    JsonSchemaDrift {
        path: PathBuf,
        record: u64,
        key: String,
    },

    /// `--columns` named a column the table does not have.
    UnknownColumn {
        path: PathBuf,
        name: String,
        available: Vec<String>,
    },

    /// A directory was given to browse and it holds no readable artifacts.
    NoArtifacts { dir: PathBuf },
}

impl ShowError {
    pub(crate) fn io(path: &Path, action: &'static str, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            action,
            source,
        }
    }

    pub(crate) fn parquet(
        path: &Path,
        action: &'static str,
        source: parquet::errors::ParquetError,
    ) -> Self {
        Self::Parquet {
            path: path.to_path_buf(),
            action,
            source,
        }
    }
}

/// Render the first few bytes of a file for an error message, with
/// non-printables escaped, so a "this isn't what it claims" error shows
/// what was actually there.
pub(crate) fn describe_head(bytes: &[u8]) -> String {
    let shown: String = bytes
        .iter()
        .take(16)
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                (b as char).to_string()
            } else {
                format!("\\x{b:02x}")
            }
        })
        .collect();
    shown
}

impl fmt::Display for ShowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { path } => {
                write!(f, "no such file or directory: {}", path.display())
            }
            Self::UnknownFormat {
                path,
                extension,
                head,
            } => {
                write!(f, "cannot show {}: ", path.display())?;
                match extension {
                    Some(ext) => write!(f, "unrecognized extension `.{ext}`")?,
                    None => write!(f, "no file extension")?,
                }
                write!(
                    f,
                    ", and the leading bytes ({head:?}) match none of parquet, csv, json. \
                     Supported: .parquet, .csv, .json, .jsonl, .ndjson"
                )
            }
            Self::ContentMismatch {
                path,
                claimed,
                head,
            } => write!(
                f,
                "{} is named like {claimed} but does not contain {claimed} \
                 (leading bytes: {head:?}) — the file is mislabelled or truncated",
                path.display()
            ),
            Self::Io {
                path,
                action,
                source,
            } => write!(f, "failed to {action} {}: {source}", path.display()),
            Self::Parquet {
                path,
                action,
                source,
            } => write!(f, "failed to {action} {}: {source}", path.display()),
            Self::UnsupportedColumnType {
                path,
                column,
                data_type,
            } => write!(
                f,
                "{}: column `{column}` has type {data_type}, which `show` cannot \
                 render. Select the other columns with --columns, or read the file \
                 with a full Arrow client",
                path.display()
            ),
            Self::CsvFieldCount {
                path,
                line,
                expected,
                found,
            } => write!(
                f,
                "{}:{line}: expected {expected} fields to match the header, found {found}",
                path.display()
            ),
            Self::CsvUnterminatedQuote { path, line } => write!(
                f,
                "{}: file ended inside a quoted field opened on line {line}",
                path.display()
            ),
            Self::Json {
                path,
                record,
                detail,
            } => write!(
                f,
                "{}: record {record} is not valid JSON: {detail}",
                path.display()
            ),
            Self::JsonNotAnObject {
                path,
                record,
                found,
            } => write!(
                f,
                "{}: record {record} is a JSON {found}, not an object — \
                 `show` renders one object per row",
                path.display()
            ),
            Self::JsonSchemaDrift { path, record, key } => write!(
                f,
                "{}: record {record} carries key `{key}`, which the first record \
                 does not. The column header is pinned by the first record so it \
                 can stay on screen while the file streams, so this key has no \
                 column to go in and `show` will not drop it",
                path.display()
            ),
            Self::UnknownColumn {
                path,
                name,
                available,
            } if available.is_empty() => write!(
                f,
                "{}: cannot select `{name}` — the file declares no columns at all \
                 (an empty CSV has no header row, and an empty JSON array has no \
                 record to read a schema from). Without --columns it shows as 0 rows",
                path.display()
            ),
            Self::UnknownColumn {
                path,
                name,
                available,
            } => write!(
                f,
                "{}: no column named `{name}`. Available: {}",
                path.display(),
                available.join(", ")
            ),
            Self::NoArtifacts { dir } => write!(
                f,
                "{} holds no readable artifacts (looked for .parquet, .csv, .json, \
                 .jsonl, .ndjson)",
                dir.display()
            ),
        }
    }
}

impl std::error::Error for ShowError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parquet { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_escapes_non_printables() {
        assert_eq!(describe_head(b"PAR1"), "PAR1");
        assert_eq!(describe_head(&[0x00, 0x01, b'a']), "\\x00\\x01a");
    }

    /// Only the first 16 bytes go in the message — an error should not
    /// dump a megabyte of a binary file into the terminal.
    #[test]
    fn head_is_bounded() {
        let long = vec![b'x'; 4096];
        assert_eq!(describe_head(&long).len(), 16);
    }

    /// Every message must name the file. A pager error that says only
    /// "invalid record 3" is useless when browsing a directory.
    #[test]
    fn every_message_names_the_file() {
        let p = PathBuf::from("/out/residuals.csv");
        let cases: Vec<ShowError> = vec![
            ShowError::NotFound { path: p.clone() },
            ShowError::UnknownFormat {
                path: p.clone(),
                extension: Some("bin".into()),
                head: "??".into(),
            },
            ShowError::ContentMismatch {
                path: p.clone(),
                claimed: "parquet",
                head: "{".into(),
            },
            ShowError::Io {
                path: p.clone(),
                action: "open",
                source: std::io::Error::other("boom"),
            },
            ShowError::UnsupportedColumnType {
                path: p.clone(),
                column: "c".into(),
                data_type: "Map".into(),
            },
            ShowError::CsvFieldCount {
                path: p.clone(),
                line: 3,
                expected: 4,
                found: 2,
            },
            ShowError::CsvUnterminatedQuote {
                path: p.clone(),
                line: 9,
            },
            ShowError::Json {
                path: p.clone(),
                record: 2,
                detail: "eof".into(),
            },
            ShowError::JsonNotAnObject {
                path: p.clone(),
                record: 2,
                found: "array",
            },
            ShowError::JsonSchemaDrift {
                path: p.clone(),
                record: 2,
                key: "k".into(),
            },
            ShowError::UnknownColumn {
                path: p.clone(),
                name: "nope".into(),
                available: vec!["a".into()],
            },
            ShowError::NoArtifacts {
                dir: PathBuf::from("/out"),
            },
        ];
        for e in cases {
            let msg = e.to_string();
            assert!(
                msg.contains("/out"),
                "error must name the path it was reading: {msg}"
            );
        }
    }
}
