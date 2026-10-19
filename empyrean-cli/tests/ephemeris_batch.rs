//! `empyrean ephemeris` is batch-first with a per-orbit delivery verdict
//! (end-to-end).
//!
//! Same contract as `propagate_batch.rs`: a three-orbit input whose MIDDLE
//! orbit is stamped at MJD 400000.0 fails per-orbit without ending the batch.
//! The exit code is the batch's verdict — 0 / 3 / 4 — and the failed orbit is
//! named on stderr and recorded in `status.json`.
//!
//! Needs kernels (`EMPYREAN_DATA_DIR` or `~/.empyrean/data`) and a
//! `libempyrean` matching this source tree. Either missing => skip.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use empyrean::{CoordinateState, Epoch, Frame, Orbit, OrbitBatch, Origin};

const GOOD_EPOCH: f64 = 59000.0;
const BAD_EPOCH: f64 = 400000.0;

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

fn engine_loads() -> bool {
    run_cli(&["version"]).status.success()
}

fn skips() -> bool {
    resolve_data_dir().is_none() || !engine_loads()
}

fn tmp_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "empyrean_cli_ephemeris_batch_{}_{}_{}",
        std::process::id(),
        tag,
        n
    ));
    std::fs::create_dir_all(&dir).expect("mkdir tmp");
    dir
}

fn write_orbits(dir: &Path, epochs: &[f64]) -> PathBuf {
    let mut orbits = Vec::new();
    let mut ids = Vec::new();
    let mut obj_ids = Vec::new();
    for (i, &epoch) in epochs.iter().enumerate() {
        let id = format!("orb_{i}");
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(epoch),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::ICRF,
            Origin::SUN,
        );
        orbits.push(Orbit::new(state).with_orbit_id(&id).with_object_id(&id));
        ids.push(id.clone());
        obj_ids.push(Some(id));
    }
    let batch = OrbitBatch::new(orbits, ids, obj_ids).expect("batch");
    let path = dir.join("orbits.json");
    empyrean::write_orbits_json(&path, &batch).expect("write orbits json");
    path
}

fn ephemeris(input: &Path, out: &Path) -> Output {
    let data_dir = resolve_data_dir().expect("data dir checked by caller");
    run_cli(&[
        "--data-dir",
        &data_dir.display().to_string(),
        "ephemeris",
        "--input",
        &input.display().to_string(),
        "--observers",
        "500",
        "--epoch",
        "59000.5",
        "--out-dir",
        &out.display().to_string(),
        "--format",
        "json",
    ])
}

#[test]
fn partial_batch_exits_three_and_names_the_failed_orbit() {
    if skips() {
        eprintln!("skipping: no data dir or libempyrean does not load");
        return;
    }
    let dir = tmp_dir("partial");
    let input = write_orbits(&dir, &[GOOD_EPOCH, BAD_EPOCH, GOOD_EPOCH]);
    let out = dir.join("out");
    let result = ephemeris(&input, &out);
    let stderr = String::from_utf8_lossy(&result.stderr);

    assert_eq!(
        result.status.code(),
        Some(3),
        "a partial batch (one orbit failed) exits 3\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("orb_1") && stderr.contains("failed"),
        "stderr must name the failed orbit and its status\n{stderr}"
    );
    assert!(
        out.join("ephemeris.json").exists(),
        "ephemeris.json written"
    );
    assert!(out.join("status.json").exists(), "status.json written");
    let status = std::fs::read_to_string(out.join("status.json")).expect("read status.json");
    assert!(
        status.contains("orb_1") && status.contains("\"failed\""),
        "{status}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn total_failure_exits_four() {
    if skips() {
        eprintln!("skipping: no data dir or libempyrean does not load");
        return;
    }
    let dir = tmp_dir("none");
    let input = write_orbits(&dir, &[BAD_EPOCH, BAD_EPOCH]);
    let out = dir.join("out");
    let result = ephemeris(&input, &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(
        result.status.code(),
        Some(4),
        "a batch in which no orbit delivered exits 4\nstderr:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn all_delivered_exits_zero() {
    if skips() {
        eprintln!("skipping: no data dir or libempyrean does not load");
        return;
    }
    let dir = tmp_dir("all");
    let input = write_orbits(&dir, &[GOOD_EPOCH, GOOD_EPOCH]);
    let out = dir.join("out");
    let result = ephemeris(&input, &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert_eq!(
        result.status.code(),
        Some(0),
        "every orbit delivered => exit 0\nstderr:\n{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
