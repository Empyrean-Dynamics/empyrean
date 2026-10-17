//! `empyrean determine --out-dir` writes the fitted orbits in every format.
//!
//! Regression for the orbit-batch keepalive use-after-free: a converging
//! fit's packed-joint triangle was freed before the C writer ran, so the
//! writer read a dangling pointer. The state block came back as garbage
//! that disagreed with the inline flat 6x6, and the orbit-write guard
//! aborted `--format parquet` and `--format csv` by name ("the packed
//! joint and the flat state covariance disagree on the state block");
//! `--format json`, whose row shape serializes only the inline flat 6x6,
//! wrote anyway. This fits a real catalog object end to end and asserts
//! that all three formats write and that the parquet carries the joint.
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
        // Strict offline: resolve every kernel from --data-dir, never the
        // network (same floor as EMPYREAN_OFFLINE=1).
        .env("EMPYREAN_OFFLINE", "1")
        .output()
        .expect("spawn empyrean CLI")
}

/// The engine dylib must both load and match this build; exercised without
/// kernels so a stale or absent libempyrean shows up as a skip.
fn engine_loads() -> bool {
    run_cli(&["version"]).status.success()
}

/// The cached 433 Eros ADES fixture — a real arc that converges in a
/// fraction of a second, so the fit delivers an orbit to write.
fn eros_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../empyrean-c/tests/fixtures/433_eros.psv")
}

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "empyrean-determine-write-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create temp dir");
    d
}

/// A converging fit writes `fitted_orbits.<fmt>` in all three formats and
/// exits "all delivered" (0) for every one — the use-after-free aborted
/// parquet and CSV with the state-block disagreement.
#[test]
fn a_converging_fit_writes_every_format_without_the_state_block_abort() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping: no data dir with de440.bsp");
        return;
    };
    if !engine_loads() {
        eprintln!("skipping: libempyrean does not load for this build");
        return;
    }

    let fixture = eros_fixture();
    for (fmt, ext) in [("parquet", "parquet"), ("csv", "csv"), ("json", "json")] {
        let dir = temp_dir(fmt);
        let out = run_cli(&[
            "--data-dir",
            &data_dir.display().to_string(),
            "determine",
            &fixture.display().to_string(),
            "--out-dir",
            &dir.display().to_string(),
            "--format",
            fmt,
        ]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(
            out.status.code(),
            Some(0),
            "determine --format {fmt} must deliver and write (exit 0)\nstderr:\n{stderr}"
        );
        assert!(
            !stderr.contains("disagree on the state block"),
            "the write must not abort on the state-block guard\nstderr:\n{stderr}"
        );
        let written = dir.join(format!("fitted_orbits.{ext}"));
        assert!(
            written.exists(),
            "fitted_orbits.{ext} must be written\nstderr:\n{stderr}"
        );
        assert!(
            std::fs::metadata(&written).map(|m| m.len()).unwrap_or(0) > 0,
            "fitted_orbits.{ext} must be non-empty\nstderr:\n{stderr}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The parquet the fit wrote round-trips its packed joint: reading it back
/// yields a covariance whose state block is a real posterior (finite,
/// positive diagonal), never the freed-triangle garbage (~1e108) the
/// use-after-free produced.
#[test]
fn the_written_parquet_round_trips_the_packed_joint() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping: no data dir with de440.bsp");
        return;
    };
    if !engine_loads() {
        eprintln!("skipping: libempyrean does not load for this build");
        return;
    }

    let dir = temp_dir("roundtrip");
    let out = run_cli(&[
        "--data-dir",
        &data_dir.display().to_string(),
        "determine",
        &eros_fixture().display().to_string(),
        "--out-dir",
        &dir.display().to_string(),
        "--format",
        "parquet",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "fit must deliver\nstderr:\n{stderr}"
    );

    let batch = empyrean::read_orbits_parquet(dir.join("fitted_orbits.parquet"))
        .expect("read back the fitted-orbits parquet");
    assert_eq!(batch.len(), 1, "one delivered object");
    let joint = batch.orbits[0]
        .covariance
        .as_ref()
        .expect("the fitted orbit must carry its packed joint after a parquet round trip");
    assert!(
        joint.width >= 6,
        "a fitted joint is at least the 6x6 state block"
    );
    let sb = joint.state_block();
    for (i, row) in sb.iter().enumerate() {
        let v = row[i];
        assert!(
            v.is_finite() && v > 0.0 && v < 1.0,
            "state-block variance [{i}][{i}] = {v} is not a real posterior — the \
             freed-triangle garbage would be ~1e108",
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
