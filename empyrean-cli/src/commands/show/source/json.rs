//! JSON row source — one record resident at a time.
//!
//! Handles both shapes that turn up in an output directory:
//!
//! * a **pretty-printed array** of objects, which is what
//!   `empyrean … --format json` writes; and
//! * **newline-delimited** objects (JSONL / NDJSON).
//!
//! Neither can be streamed by `serde_json`'s own readers at row
//! granularity — `Deserializer::into_iter::<Value>()` treats a top-level
//! array as one enormous value, which would pull a multi-million-row file
//! entirely into memory. So the bytes are walked here with a depth /
//! string-state scanner that hands out one top-level record's raw text at
//! a time, and only that record is parsed. Resident cost is one record.
//!
//! Nested values (the `elements` vector, the 6×6 `covariance` block) have
//! no column of their own in a flat table, so they are rendered into
//! their cell as compact JSON. They are never dropped and never
//! flattened into invented column names.

use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::{Cell, RowSource, ShowError};

pub struct JsonSource {
    path: PathBuf,
    columns: Vec<String>,
    scanner: RecordScanner,
    /// The first record, parsed during `open` to establish the header and
    /// held back so it is not consumed twice.
    pending: Option<Map<String, Value>>,
    /// 1-based index of the record `next_row` will return.
    record: u64,
    done: bool,
}

impl JsonSource {
    pub fn open(path: &Path) -> Result<Self, ShowError> {
        let file = std::fs::File::open(path).map_err(|e| ShowError::io(path, "open", e))?;
        let mut scanner = RecordScanner::new(BufReader::new(file));

        // The header is the first record's keys, in its own order. That
        // pins the column set for the whole stream, which is what lets
        // the header stay on screen while the file pages past it.
        let first = match scanner.next_record(path)? {
            Some(raw) => Some(parse_object(path, 1, &raw)?),
            None => None,
        };
        let columns = first
            .as_ref()
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();

        Ok(Self {
            path: path.to_path_buf(),
            columns,
            scanner,
            pending: first,
            record: 0,
            done: false,
        })
    }
}

impl RowSource for JsonSource {
    fn columns(&self) -> &[String] {
        &self.columns
    }

    fn next_row(&mut self) -> Result<Option<Vec<Cell>>, ShowError> {
        if self.done {
            return Ok(None);
        }
        self.record += 1;
        let obj = match self.pending.take() {
            Some(obj) => obj,
            None => match self.scanner.next_record(&self.path)? {
                Some(raw) => parse_object(&self.path, self.record, &raw)?,
                None => {
                    self.done = true;
                    return Ok(None);
                }
            },
        };

        // A key the header does not have would have nowhere to go. Say so
        // rather than dropping the value.
        for key in obj.keys() {
            if !self.columns.iter().any(|c| c == key) {
                return Err(ShowError::JsonSchemaDrift {
                    path: self.path.clone(),
                    record: self.record,
                    key: key.clone(),
                });
            }
        }

        // A key the header has but this record lacks is a genuine
        // absence, and renders as null.
        let row = self
            .columns
            .iter()
            .map(|c| obj.get(c).map_or(Cell::Null, value_to_cell))
            .collect();
        Ok(Some(row))
    }
}

fn parse_object(path: &Path, record: u64, raw: &str) -> Result<Map<String, Value>, ShowError> {
    let value: Value = serde_json::from_str(raw).map_err(|e| ShowError::Json {
        path: path.to_path_buf(),
        record,
        detail: e.to_string(),
    })?;
    match value {
        Value::Object(map) => Ok(map),
        other => Err(ShowError::JsonNotAnObject {
            path: path.to_path_buf(),
            record,
            found: match other {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => unreachable!(),
            },
        }),
    }
}

fn value_to_cell(value: &Value) -> Cell {
    match value {
        Value::Null => Cell::Null,
        Value::Bool(b) => Cell::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Cell::Int(i)
            } else if let Some(u) = n.as_u64() {
                Cell::UInt(u)
            } else {
                // serde_json never produces a non-finite number, so the
                // only remaining case is a finite float.
                Cell::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => Cell::Text(s.clone()),
        // Compact so a 6×6 covariance stays in one cell rather than
        // spilling the pretty-printer's newlines through the table.
        other => Cell::Text(other.to_string()),
    }
}

/// Hands out the raw text of one top-level JSON record at a time.
///
/// Tracks brace/bracket depth and string state so that punctuation inside
/// a string literal (`"obs,\"id\""`, a designation containing `}`) never
/// ends a record early. An enclosing top-level `[ … ]` is stepped over;
/// bare concatenated or newline-delimited objects work with no array at
/// all.
struct RecordScanner {
    reader: BufReader<std::fs::File>,
    /// Set once the opening `[` of an array wrapper has been consumed.
    entered_array: bool,
    finished: bool,
}

impl RecordScanner {
    fn new(reader: BufReader<std::fs::File>) -> Self {
        Self {
            reader,
            entered_array: false,
            finished: false,
        }
    }

    fn next_record(&mut self, path: &Path) -> Result<Option<String>, ShowError> {
        if self.finished {
            return Ok(None);
        }
        // Skip whitespace, commas, and (once) the array's opening bracket.
        let start = loop {
            match self.byte(path)? {
                None => {
                    self.finished = true;
                    return Ok(None);
                }
                Some(b) if b.is_ascii_whitespace() || b == b',' => {}
                Some(b'[') if !self.entered_array => self.entered_array = true,
                Some(b']') if self.entered_array => {
                    self.finished = true;
                    return Ok(None);
                }
                Some(b) => break b,
            }
        };

        let mut raw = Vec::new();
        raw.push(start);
        let mut depth: i32 = match start {
            b'{' | b'[' => 1,
            _ => 0,
        };
        let mut in_string = start == b'"';
        let mut escaped = false;

        loop {
            // A scalar record (a bare number or `null`) ends at the first
            // delimiter rather than at depth zero.
            if depth == 0 && !in_string {
                match self.peek(path)? {
                    None => break,
                    Some(b) if b.is_ascii_whitespace() || b == b',' || b == b']' => break,
                    _ => {}
                }
            }
            let Some(b) = self.byte(path)? else {
                break;
            };
            raw.push(b);

            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            }
            match b {
                b'"' => in_string = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }

        String::from_utf8(raw).map(Some).map_err(|e| {
            ShowError::io(
                path,
                "decode UTF-8 from",
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            )
        })
    }

    fn byte(&mut self, path: &Path) -> Result<Option<u8>, ShowError> {
        let mut b = [0_u8; 1];
        loop {
            return match self.reader.read(&mut b) {
                Ok(0) => Ok(None),
                Ok(_) => Ok(Some(b[0])),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => Err(ShowError::io(path, "read", e)),
            };
        }
    }

    fn peek(&mut self, path: &Path) -> Result<Option<u8>, ShowError> {
        use std::io::BufRead;
        loop {
            return match self.reader.fill_buf() {
                Ok([]) => Ok(None),
                Ok(buf) => Ok(Some(buf[0])),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => Err(ShowError::io(path, "read", e)),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(body: &str) -> Vec<String> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!("empyrean-show-json-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{}.json", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, body).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let mut scanner = RecordScanner::new(BufReader::new(file));
        let mut out = Vec::new();
        while let Some(rec) = scanner.next_record(&path).unwrap() {
            out.push(rec);
        }
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn splits_a_pretty_printed_array_into_records() {
        let got = scan("[\n  {\"a\": 1},\n  {\"a\": 2}\n]\n");
        assert_eq!(got, vec!["{\"a\": 1}", "{\"a\": 2}"]);
    }

    #[test]
    fn splits_newline_delimited_objects() {
        let got = scan("{\"a\": 1}\n{\"a\": 2}\n");
        assert_eq!(got, vec!["{\"a\": 1}", "{\"a\": 2}"]);
    }

    /// The whole reason for a hand-written scanner: punctuation inside a
    /// string literal must not end a record.
    #[test]
    fn braces_and_commas_inside_strings_do_not_end_a_record() {
        let got = scan(r#"[{"id": "K23,A0{1}", "n": 2}]"#);
        assert_eq!(got, vec![r#"{"id": "K23,A0{1}", "n": 2}"#]);
    }

    #[test]
    fn escaped_quotes_inside_strings_are_handled() {
        let got = scan(r#"[{"id": "he said \"hi\""}]"#);
        assert_eq!(got, vec![r#"{"id": "he said \"hi\""}"#]);
    }

    /// Nested arrays (the covariance block) belong to their record.
    #[test]
    fn nested_arrays_stay_inside_their_record() {
        let got = scan("[{\"cov\": [[1,2],[3,4]]},{\"cov\": [[5]]}]");
        assert_eq!(got, vec!["{\"cov\": [[1,2],[3,4]]}", "{\"cov\": [[5]]}"]);
    }

    #[test]
    fn empty_array_yields_no_records() {
        assert!(scan("[]").is_empty());
        assert!(scan("[\n]\n").is_empty());
        assert!(scan("").is_empty());
    }

    #[test]
    fn nested_values_render_compactly_in_one_cell() {
        let v: Value = serde_json::from_str("[[1.0, 2.0], [3.0, 4.0]]").unwrap();
        assert_eq!(
            value_to_cell(&v),
            Cell::Text("[[1.0,2.0],[3.0,4.0]]".to_string())
        );
    }

    #[test]
    fn scalars_map_to_typed_cells() {
        assert_eq!(value_to_cell(&Value::Null), Cell::Null);
        assert_eq!(value_to_cell(&serde_json::json!(true)), Cell::Bool(true));
        assert_eq!(value_to_cell(&serde_json::json!(7)), Cell::Int(7));
        assert_eq!(value_to_cell(&serde_json::json!(-7)), Cell::Int(-7));
        assert_eq!(value_to_cell(&serde_json::json!(1.5)), Cell::Float(1.5));
        assert_eq!(
            value_to_cell(&serde_json::json!("x")),
            Cell::Text("x".into())
        );
    }
}
