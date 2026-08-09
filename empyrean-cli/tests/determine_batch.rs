//! `empyrean determine` is batch-first (end-to-end).
//!
//! Runs the real `empyrean` binary against a **synthetic** two-object
//! ADES PSV and asserts the batch contract that the single-fit surface
//! used to violate: every input object gets a `fit_summary` row whether
//! or not it produced an orbit, the summary is written in BOTH parquet
//! and CSV, the exit code distinguishes "all delivered" from anything
//! else, and the output is a deterministic function of the observation
//! set rather than of the order its rows arrived in.
//!
//! The fixture is fabricated, not a catalog object: two short tracklets
//! that the pipeline cannot fit. That is deliberate — the property under
//! test is that a failed object is *reported*, and a failure needs no
//! real astrometry to produce. A converging multi-object fit belongs to
//! the validation catalog, not to a unit fixture.
//!
//! Needs kernels (`EMPYREAN_DATA_DIR` or `~/.empyrean/data`) and a
//! `libempyrean` matching this source tree. Either missing => skip.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Resolve a data dir that actually exists: `EMPYREAN_DATA_DIR` (CI) else
/// `~/.empyrean/data` (local). `None` => skip.
fn resolve_data_dir() -> Option<PathBuf> {
    let candidates = [
        std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".empyrean/data")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|d| d.join("de440.bsp").exists())
}

fn run_cli(args: &[&str]) -> Output {
    let bin = env!("CARGO_BIN_EXE_empyrean");
    let dylib_dir = Path::new(bin).parent().unwrap();
    Command::new(bin)
        .args(args)
        .env("DYLD_LIBRARY_PATH", dylib_dir)
        .env("LD_LIBRARY_PATH", dylib_dir)
        .output()
        .expect("spawn empyrean CLI")
}

/// The engine dylib must both load and match this build. `version`
/// exercises the load path without needing kernels, so a stale or absent
/// libempyrean shows up here as a skip rather than as a spurious
/// assertion failure downstream.
fn engine_loads() -> bool {
    run_cli(&["version"]).status.success()
}

/// Two synthetic objects, two rows each. `interleaved` decides whether
/// the rows arrive grouped by object or alternating between them — the
/// outputs must not be able to tell the difference.
///
/// Two rows is below the three IOD needs, so each object fails cleanly
/// and identically. That is the point: the assertions are about a failed
/// object being *reported*, and a deliberate short arc produces that
/// without fabricating sky positions whose geometry the engine would
/// have to take seriously.
fn two_object_psv(interleaved: bool) -> String {
    let header =
        "# version=2022\npermID|provID|trkSub|mode|stn|obsTime|ra|dec|rmsRA|rmsDec|astCat\n";
    let row = |obj: &str, hour: u32, ra: f64, dec: f64| {
        format!("|{obj}||CCD|703|2024-01-10T{hour:02}:00:00.000Z|{ra:.6}|{dec:.6}|0.5|0.5|Gaia2\n")
    };
    let a = [
        row("SYNTH01", 8, 120.0, 15.0),
        row("SYNTH01", 9, 120.01, 15.004),
    ];
    let b = [
        row("SYNTH02", 8, 200.0, -8.0),
        row("SYNTH02", 9, 200.02, -8.006),
    ];

    let mut out = String::from(header);
    if interleaved {
        for (ra, rb) in a.iter().zip(b.iter()) {
            out.push_str(ra);
            out.push_str(rb);
        }
    } else {
        for r in a.iter().chain(b.iter()) {
            out.push_str(r);
        }
    }
    out
}

fn determine_into(dir: &Path, psv: &str, data_dir: &Path) -> Output {
    let psv_path = dir.join("obs.psv");
    std::fs::write(&psv_path, psv).expect("write synthetic PSV");
    run_cli(&[
        "--data-dir",
        &data_dir.display().to_string(),
        "determine",
        &psv_path.display().to_string(),
        "--out-dir",
        &dir.display().to_string(),
        "--format",
        "csv",
    ])
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "empyrean-determine-batch-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&d).expect("create temp dir");
    d
}

/// Every input object is accounted for: both appear in `fit_summary`,
/// which is written in both formats, and neither is silently dropped.
#[test]
fn every_input_object_gets_a_fit_summary_row_in_both_formats() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping: no data dir with de440.bsp");
        return;
    };
    if !engine_loads() {
        eprintln!("skipping: libempyrean does not load for this build");
        return;
    }

    let dir = temp_dir("both-formats");
    let out = determine_into(&dir, &two_object_psv(false), &data_dir);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // BOTH formats, always — the CSV is what makes the run readable at a
    // terminal without a parquet tool.
    let csv_path = dir.join("fit_summary.csv");
    let parquet_path = dir.join("fit_summary.parquet");
    assert!(
        csv_path.exists(),
        "fit_summary.csv must always be written\nstderr:\n{stderr}"
    );
    assert!(
        parquet_path.exists(),
        "fit_summary.parquet must always be written\nstderr:\n{stderr}"
    );

    let csv = std::fs::read_to_string(&csv_path).expect("read fit_summary.csv");
    let data_rows: Vec<&str> = csv
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .collect();
    assert_eq!(
        data_rows.len(),
        2,
        "one row per INPUT object, delivered or not\n{csv}"
    );
    assert!(csv.contains("SYNTH01"), "first object must appear\n{csv}");
    assert!(csv.contains("SYNTH02"), "second object must appear\n{csv}");

    // The per-object table reaches the operator, and every object is on it.
    assert!(stderr.contains("SYNTH01"), "stderr table:\n{stderr}");
    assert!(stderr.contains("SYNTH02"), "stderr table:\n{stderr}");

    // fitted_orbits carries the delivered objects; residuals carry the
    // object_id column. Both files exist even when nothing delivered.
    assert!(dir.join("fitted_orbits.csv").exists(), "stderr:\n{stderr}");
    let residuals = std::fs::read_to_string(dir.join("residuals.csv")).expect("read residuals.csv");
    if let Some(header) = residuals.lines().next() {
        assert!(
            header.contains("object_id"),
            "residuals must carry an object_id column\n{residuals}"
        );
    }
    // (No object delivered here, so the file is empty. The column itself
    // is asserted against written rows in the wrapper's
    // `residual_object_id` integration test.)

    // The two-row arcs are below the IOD minimum, so both objects fail —
    // and each says so in its own row rather than vanishing.
    let statuses: Vec<&str> = data_rows
        .iter()
        .map(|r| r.split(',').nth(1).unwrap_or_default())
        .collect();
    assert_eq!(
        statuses,
        vec!["failed", "failed"],
        "both objects must be marked failed\n{csv}"
    );
    for row in &data_rows {
        assert!(
            row.contains("NaN"),
            "a failed object's measurements are NaN, never 0.0\n{row}"
        );
    }
    assert!(
        stderr.contains("produced no orbit"),
        "failures must be listed loudly at the end\nstderr:\n{stderr}"
    );

    // Exit code is the batch's verdict, never a bare success when
    // objects are missing from the orbit file.
    assert_eq!(
        out.status.code(),
        Some(4),
        "no object delivered => exit 4\nstderr:\n{stderr}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// The outputs describe the observation SET, not the order its rows
/// happened to arrive in: grouping the two objects' rows or interleaving
/// them must produce a byte-identical summary.
#[test]
fn output_is_independent_of_input_row_order() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping: no data dir with de440.bsp");
        return;
    };
    if !engine_loads() {
        eprintln!("skipping: libempyrean does not load for this build");
        return;
    }

    let grouped_dir = temp_dir("grouped");
    let interleaved_dir = temp_dir("interleaved");
    determine_into(&grouped_dir, &two_object_psv(false), &data_dir);
    determine_into(&interleaved_dir, &two_object_psv(true), &data_dir);

    let grouped = std::fs::read_to_string(grouped_dir.join("fit_summary.csv"))
        .expect("read grouped fit_summary.csv");
    let interleaved = std::fs::read_to_string(interleaved_dir.join("fit_summary.csv"))
        .expect("read interleaved fit_summary.csv");
    assert_eq!(
        grouped, interleaved,
        "fit_summary must be a function of the observation set, not of row order"
    );

    std::fs::remove_dir_all(&grouped_dir).ok();
    std::fs::remove_dir_all(&interleaved_dir).ok();
}
