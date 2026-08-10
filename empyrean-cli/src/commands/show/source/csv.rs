//! CSV row source — one record resident at a time.
//!
//! RFC 4180 shape: comma separated, `"` quoted, `""` for a literal quote
//! inside a quoted field, and newlines permitted inside quotes. The
//! parser is a byte-at-a-time state machine over a [`BufReader`] rather
//! than a line splitter precisely so an embedded newline stays inside its
//! record instead of tearing the row in half.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use super::{Cell, RowSource, ShowError};

pub struct CsvSource {
    path: PathBuf,
    columns: Vec<String>,
    reader: BufReader<std::fs::File>,
    /// 1-based line number the *next* record starts on, for errors.
    line: u64,
    done: bool,
}

impl CsvSource {
    pub fn open(path: &Path) -> Result<Self, ShowError> {
        let file = std::fs::File::open(path).map_err(|e| ShowError::io(path, "open", e))?;
        let mut reader = BufReader::new(file);
        let mut line = 1_u64;
        let header = read_record(path, &mut reader, &mut line)?;
        Ok(Self {
            path: path.to_path_buf(),
            // An empty file has no header and therefore no columns; it
            // renders as a zero-column, zero-row table rather than failing.
            columns: header.unwrap_or_default(),
            reader,
            line,
            done: false,
        })
    }
}

impl RowSource for CsvSource {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Vec<Cell>>, ShowError> {
        if self.done || self.columns.is_empty() {
            return Ok(None);
        }
        let started_on = self.line;
        let Some(fields) = read_record(&self.path, &mut self.reader, &mut self.line)? else {
            self.done = true;
            return Ok(None);
        };
        if fields.len() != self.columns.len() {
            return Err(ShowError::CsvFieldCount {
                path: self.path.clone(),
                line: started_on,
                expected: self.columns.len(),
                found: fields.len(),
            });
        }
        Ok(Some(fields.into_iter().map(parse_field).collect()))
    }
}

/// Read one CSV record, or `None` at end of file.
///
/// Fields accumulate as bytes and are decoded as UTF-8 once complete —
/// pushing each byte as a `char` would mangle every non-ASCII designation
/// in the file (`Faye`, `Encke`, and every observatory name with an
/// accent) into mojibake.
///
/// `line` is advanced past every newline consumed, including those inside
/// quoted fields, so an error message points at a real line in the file.
fn read_record(
    path: &Path,
    reader: &mut BufReader<std::fs::File>,
    line: &mut u64,
) -> Result<Option<Vec<String>>, ShowError> {
    let opened_quote_on = *line;
    let mut fields: Vec<String> = Vec::new();
    let mut field: Vec<u8> = Vec::new();
    let mut in_quotes = false;
    let mut saw_any = false;

    loop {
        let byte = match read_byte(path, reader)? {
            Some(b) => b,
            None => {
                if in_quotes {
                    return Err(ShowError::CsvUnterminatedQuote {
                        path: path.to_path_buf(),
                        line: opened_quote_on,
                    });
                }
                if !saw_any {
                    return Ok(None);
                }
                fields.push(decode(path, field, *line)?);
                return Ok(Some(fields));
            }
        };
        saw_any = true;

        if in_quotes {
            match byte {
                b'"' => match peek_byte(path, reader)? {
                    // "" inside a quoted field is one literal quote.
                    Some(b'"') => {
                        let _ = read_byte(path, reader)?;
                        field.push(b'"');
                    }
                    _ => in_quotes = false,
                },
                b'\n' => {
                    *line += 1;
                    field.push(b'\n');
                }
                b => field.push(b),
            }
            continue;
        }

        match byte {
            b'"' => in_quotes = true,
            b',' => fields.push(decode(path, std::mem::take(&mut field), *line)?),
            b'\r' => { /* swallowed; the \n that follows ends the record */ }
            b'\n' => {
                *line += 1;
                fields.push(decode(path, field, *line - 1)?);
                return Ok(Some(fields));
            }
            b => field.push(b),
        }
    }
}

/// Decode one accumulated field as UTF-8.
fn decode(path: &Path, bytes: Vec<u8>, line: u64) -> Result<String, ShowError> {
    String::from_utf8(bytes).map_err(|e| {
        ShowError::io(
            path,
            "decode a UTF-8 field from",
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("line {line}: {e}")),
        )
    })
}

fn read_byte(path: &Path, reader: &mut BufReader<std::fs::File>) -> Result<Option<u8>, ShowError> {
    let mut b = [0_u8; 1];
    loop {
        return match reader.read(&mut b) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(b[0])),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => Err(ShowError::io(path, "read", e)),
        };
    }
}

fn peek_byte(path: &Path, reader: &mut BufReader<std::fs::File>) -> Result<Option<u8>, ShowError> {
    match reader.fill_buf() {
        Ok([]) => Ok(None),
        Ok(buf) => Ok(Some(buf[0])),
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => peek_byte(path, reader),
        Err(e) => Err(ShowError::io(path, "read", e)),
    }
}

/// Classify a raw CSV field.
///
/// CSV carries no types, so the cell type is inferred from the text: an
/// empty field is null, `true`/`false` is a boolean, anything Rust's
/// float parser accepts (including `NaN` and `inf`) is a float, and the
/// rest is text. Integers stay integers so an `obs_id` of `007` is not
/// reformatted into `7`.
fn parse_field(raw: String) -> Cell {
    if raw.is_empty() {
        return Cell::Null;
    }
    match raw.as_str() {
        "true" => return Cell::Bool(true),
        "false" => return Cell::Bool(false),
        _ => {}
    }
    // Only text that *looks* numeric is parsed. Without this guard a
    // leading-zero identifier or a `+`-prefixed code would silently
    // change shape on screen.
    if looks_numeric(&raw) {
        if let Ok(i) = raw.parse::<i64>() {
            return Cell::Int(i);
        }
        if let Ok(v) = raw.parse::<f64>() {
            return Cell::Float(v);
        }
    }
    Cell::Text(raw)
}

/// Whether a field should be offered to the numeric parsers.
///
/// Rust's `f64::from_str` accepts `NaN`, `inf`, `infinity` in any case,
/// which would turn an observatory named `Inf` into a float. Require
/// either a digit or an explicit float keyword with numeric punctuation.
fn looks_numeric(s: &str) -> bool {
    let body = s.strip_prefix(['+', '-']).unwrap_or(s);
    if body.is_empty() {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    if lower == "nan" || lower == "inf" || lower == "infinity" {
        return true;
    }
    // A leading zero followed by another digit is an identifier
    // convention (`007`), not a number.
    let mut chars = body.chars();
    if chars.next() == Some('0') && chars.next().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }
    body.starts_with(|c: char| c.is_ascii_digit() || c == '.')
        && body
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(s: &str) -> Cell {
        parse_field(s.to_string())
    }

    #[test]
    fn empty_field_is_null_not_empty_text() {
        assert_eq!(field(""), Cell::Null);
    }

    #[test]
    fn numeric_fields_become_numbers() {
        assert_eq!(field("42"), Cell::Int(42));
        assert_eq!(field("-7"), Cell::Int(-7));
        assert_eq!(field("60000.0"), Cell::Float(60000.0));
        assert_eq!(field("1e-12"), Cell::Float(1e-12));
        assert_eq!(field("-3.5E+02"), Cell::Float(-350.0));
    }

    #[test]
    fn nan_and_infinity_survive_the_round_trip() {
        let Cell::Float(v) = field("NaN") else {
            panic!("NaN must parse as a float cell, not text")
        };
        assert!(v.is_nan());
        assert_eq!(field("inf"), Cell::Float(f64::INFINITY));
        assert_eq!(field("-inf"), Cell::Float(f64::NEG_INFINITY));
    }

    /// Identifiers must not be reshaped by the parser. `007` is an
    /// obsID, not the number seven, and `568` *is* a number but must
    /// still render as `568`.
    #[test]
    fn leading_zero_identifiers_stay_text() {
        assert_eq!(field("007"), Cell::Text("007".into()));
        assert_eq!(field("0"), Cell::Int(0));
        assert_eq!(field("568"), Cell::Int(568));
    }

    /// A word that Rust's float parser happens to accept is still a word.
    #[test]
    fn words_that_look_like_float_keywords_stay_text() {
        assert_eq!(
            field("Infinity_Station"),
            Cell::Text("Infinity_Station".into())
        );
        assert_eq!(field("EclipticJ2000"), Cell::Text("EclipticJ2000".into()));
        assert_eq!(field("TDB"), Cell::Text("TDB".into()));
    }

    #[test]
    fn booleans_are_typed() {
        assert_eq!(field("true"), Cell::Bool(true));
        assert_eq!(field("false"), Cell::Bool(false));
        assert_eq!(field("True"), Cell::Text("True".into()));
    }

    // Record-level parsing is exercised end-to-end against real files in
    // `tests/show.rs`; these cover the byte-level state machine through a
    // temporary file, which is the only way to reach `read_record`.
    fn records(body: &str) -> Vec<Vec<String>> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("empyrean-show-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.csv", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, body).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut reader = BufReader::new(file);
        let mut line = 1;
        let mut out = Vec::new();
        while let Some(rec) = read_record(&path, &mut reader, &mut line).unwrap() {
            out.push(rec);
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn quoted_commas_stay_in_their_field() {
        assert_eq!(
            records("a,b\n\"x,y\",z\n"),
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["x,y".to_string(), "z".to_string()],
            ]
        );
    }

    #[test]
    fn doubled_quotes_are_one_literal_quote() {
        assert_eq!(
            records("a\n\"he said \"\"hi\"\"\"\n"),
            vec![vec!["a".to_string()], vec!["he said \"hi\"".to_string()]]
        );
    }

    /// The reason this is a byte state machine and not `lines()`.
    #[test]
    fn newlines_inside_quotes_do_not_split_the_record() {
        assert_eq!(
            records("a,b\n\"line1\nline2\",z\n"),
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["line1\nline2".to_string(), "z".to_string()],
            ]
        );
    }

    #[test]
    fn crlf_endings_do_not_leak_into_the_last_field() {
        assert_eq!(
            records("a,b\r\n1,2\r\n"),
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["1".to_string(), "2".to_string()],
            ]
        );
    }

    /// A file that does not end in a newline still yields its last row.
    #[test]
    fn final_record_without_a_trailing_newline_is_returned() {
        assert_eq!(
            records("a,b\n1,2"),
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["1".to_string(), "2".to_string()],
            ]
        );
    }

    #[test]
    fn empty_file_yields_no_records() {
        assert!(records("").is_empty());
    }
}
