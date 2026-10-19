//! Per-orbit delivery status for the batch commands (`propagate`,
//! `ephemeris`).
//!
//! The engine carries a per-input-orbit delivery verdict: one orbit's
//! failure no longer ends a batch — the failing orbit is recorded and
//! every other orbit delivers its rows. These helpers surface that verdict
//! the way `determine` surfaces its fit summary: a status table the
//! operator reads on stderr, and a status file written beside the
//! states / ephemeris output, so a partially-successful run reads as such
//! rather than as a success with a shorter output file.
//!
//! The status file is JSON. The CLI's whole I/O surface goes through the
//! wrapper's C-ABI writers (orbits / events / ephemeris / fit summary), and
//! there is no C-ABI writer for this per-orbit table; a hand-written JSON
//! array needs neither a new C export nor a new crate dependency. It rides
//! the `parquet` and `json` output formats and is omitted for `csv`, which
//! stays state-only.

use std::path::Path;

use anyhow::{Context, Result};
use empyrean::OrbitOutcome;

use crate::io::output::OutputFormat;

/// One row of a batch command's per-orbit delivery status, one per INPUT
/// orbit in input order.
pub struct OrbitStatusRow {
    /// The input orbit's identifier.
    pub orbit_id: String,
    /// Zero-based index into the input orbit batch.
    pub orbit_index: usize,
    /// `"delivered"` (`EMPYREAN_ORBIT_OUTCOME_DELIVERED`) or `"failed"`
    /// (`EMPYREAN_ORBIT_OUTCOME_FAILED`).
    pub status: &'static str,
    /// The `EMPYREAN_*_FAILURE_*` classification on a failed orbit; `None`
    /// when delivered.
    pub error_code: Option<i32>,
    /// The engine's failure text on a failed orbit; `None` when delivered.
    pub error_message: Option<String>,
    /// Offset of the orbit's first delivered row in the flat output; `None`
    /// when the orbit failed (it occupies no row block).
    pub first_row: Option<usize>,
    /// Delivered row count for this orbit (`0` for a failed orbit, and also
    /// `0` for a delivered orbit whose marker stop fired before its first
    /// requested epoch — delivery is `status`, never this count).
    pub num_rows: usize,
}

/// Build the per-orbit status rows from the engine's outcome vector and the
/// input batch's identifiers.
pub fn status_rows(outcomes: &[OrbitOutcome], orbit_ids: &[String]) -> Vec<OrbitStatusRow> {
    outcomes
        .iter()
        .enumerate()
        .map(|(k, o)| {
            let orbit_id = orbit_ids
                .get(k)
                .cloned()
                .unwrap_or_else(|| format!("orbit_{k}"));
            match o {
                OrbitOutcome::Delivered {
                    first_row,
                    num_rows,
                } => OrbitStatusRow {
                    orbit_id,
                    orbit_index: k,
                    status: "delivered",
                    error_code: None,
                    error_message: None,
                    first_row: Some(*first_row),
                    num_rows: *num_rows,
                },
                OrbitOutcome::Failed { code, message } => OrbitStatusRow {
                    orbit_id,
                    orbit_index: k,
                    status: "failed",
                    error_code: Some(*code),
                    error_message: Some(message.clone()),
                    first_row: None,
                    num_rows: 0,
                },
            }
        })
        .collect()
}

/// The number of delivered orbits in a status table.
pub fn delivered_count(rows: &[OrbitStatusRow]) -> usize {
    rows.iter().filter(|r| r.status == "delivered").count()
}

/// Print the per-orbit status table to stderr, then list the failures in
/// full with their messages. The table is the command's primary accounting
/// output: one line per input orbit, so a partial run reads as such.
pub fn report_status(rows: &[OrbitStatusRow]) {
    eprintln!("\n  {:<24} {:>9} {:>8}", "Orbit", "Status", "Rows");
    eprintln!("  {}", "-".repeat(43));
    for r in rows {
        eprintln!("  {:<24} {:>9} {:>8}", r.orbit_id, r.status, r.num_rows);
    }

    let failures: Vec<&OrbitStatusRow> = rows.iter().filter(|r| r.status == "failed").collect();
    if !failures.is_empty() {
        eprintln!("\n  {} of {} orbit(s) failed:", failures.len(), rows.len());
        for r in &failures {
            eprintln!(
                "    {}: {}",
                r.orbit_id,
                r.error_message.as_deref().unwrap_or("")
            );
        }
    }
}

/// Minimal JSON string escaping — quotes, backslash, and the control
/// characters a failure message can carry. Enough for the engine's
/// single-line failure text; the CLI has no JSON serializer dependency.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Write the per-orbit status table as `<dir>/<stem>.json`.
///
/// Carried by the `parquet` and `json` output formats; `csv` stays
/// state-only and writes nothing here. The JSON is an array of objects, one
/// per input orbit, with `error_code` / `error_message` / `first_row` null
/// on a delivered orbit and `first_row` null on a failed one.
pub fn write_status(
    dir: &Path,
    stem: &str,
    rows: &[OrbitStatusRow],
    fmt: OutputFormat,
) -> Result<()> {
    if matches!(fmt, OutputFormat::Csv) {
        return Ok(());
    }
    std::fs::create_dir_all(dir).context("failed to create output directory")?;
    let path = dir.join(format!("{stem}.json"));

    let mut body = String::from("[\n");
    for (i, r) in rows.iter().enumerate() {
        let code = match r.error_code {
            Some(c) => c.to_string(),
            None => "null".to_string(),
        };
        let message = match &r.error_message {
            Some(m) => format!("\"{}\"", json_escape(m)),
            None => "null".to_string(),
        };
        let first_row = match r.first_row {
            Some(f) => f.to_string(),
            None => "null".to_string(),
        };
        body.push_str(&format!(
            "  {{\"orbit_id\": \"{}\", \"orbit_index\": {}, \"outcome\": \"{}\", \
             \"error_code\": {}, \"error_message\": {}, \"first_row\": {}, \"num_rows\": {}}}",
            json_escape(&r.orbit_id),
            r.orbit_index,
            r.status,
            code,
            message,
            first_row,
            r.num_rows,
        ));
        body.push_str(if i + 1 < rows.len() { ",\n" } else { "\n" });
    }
    body.push_str("]\n");

    std::fs::write(&path, body).with_context(|| format!("failed to write {}", path.display()))?;
    eprintln!("  {} ({} rows)", path.display(), rows.len());
    Ok(())
}
