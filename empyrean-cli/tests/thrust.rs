//! Continuous-thrust input for the `propagate` command (end-to-end).
//!
//! Runs the real `empyrean` binary (`CARGO_BIN_EXE_empyrean`) with a JSON
//! thrust file (`--thrust-arcs`) and asserts the burn reaches the engine
//! and changes propagation: the thrusted trajectory diverges from the
//! identical ballistic orbit. A second test asserts the engine's
//! `dv_corrections` / `correction_covariances` length contract surfaces
//! as a loud, non-zero-exit CLI error — never silently repaired.
//!
//! Needs kernels (`EMPYREAN_DATA_DIR` or `~/.empyrean/data`) + the
//! `libempyrean` dylib. If the ballistic reference propagation can't run
//! (missing kernels), the test logs and returns (skips).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use empyrean::{CoordinateState, Epoch, Frame, Orbit, OrbitBatch, Origin};

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

/// Run the CLI binary, returning the full captured output. The
/// DYLD/LD_LIBRARY_PATH env is belt-and-suspenders (the binary dlopens
/// libempyrean by resolved absolute path); `--data-dir` is passed
/// explicitly so the child never depends on ambient environment.
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

/// A heliocentric Cartesian input orbit with a tight diagonal covariance,
/// written as a `.json` orbit file the CLI can `--input`.
fn write_input_orbit(dir: &Path) -> PathBuf {
    let mut cov = [[0.0_f64; 6]; 6];
    for (i, row) in cov.iter_mut().enumerate() {
        row[i] = 1.0e-16;
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(59000.0),
        [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
        Frame::ICRF,
        Origin::SUN,
    )
    .with_covariance(cov);
    let orbit = Orbit::new(state)
        .with_orbit_id("thrust-cli")
        .with_object_id("thrust-cli");
    let batch = OrbitBatch::new(
        vec![orbit],
        vec!["thrust-cli".to_string()],
        vec![Some("thrust-cli".to_string())],
    )
    .expect("batch");
    let path = dir.join("thrust_in.json");
    empyrean::write_orbits_json(&path, &batch).expect("write input json");
    path
}

/// Final propagated Cartesian position read back from a `states.json`
/// written by the CLI (uses the wrapper's own reader — no hand-parsing).
fn final_position(states_json: &Path) -> [f64; 3] {
    let batch = empyrean::read_orbits_json(states_json).expect("read states.json");
    let e = batch
        .orbits
        .last()
        .expect("at least one output state")
        .state
        .elements;
    [e[0], e[1], e[2]]
}

/// Unique temp dir per test invocation.
fn tmp_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "empyrean_cli_thrust_it_{}_{}_{}",
        std::process::id(),
        tag,
        n
    ));
    std::fs::create_dir_all(&dir).expect("mkdir tmp");
    dir
}

/// A ConstantRTN thrust file (strong burn) with a Δv correction + 3×3
/// covariance — the full input schema, mirroring the wrapper's
/// `propagate_with_thrust_perturbs_trajectory` scenario.
const THRUST_JSON: &str = r#"{
    "arcs": [{
        "start_mjd_tdb": 59000.0,
        "end_mjd_tdb": 59010.0,
        "thrust_n": 1000.0,
        "mass_kg": 1000.0,
        "steering": {"type": "constant_rtn", "alpha_rad": 0.1, "beta_rad": 0.2},
        "sharpness": 100.0,
        "central_body": 10
    }],
    "dv_corrections": [[0.0, 0.0, 0.0]],
    "correction_covariances": [[[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]]
}"#;

#[test]
fn propagate_with_thrust_json_perturbs_trajectory() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping propagate_with_thrust_json_perturbs_trajectory: no data dir");
        return;
    };
    let data_dir = data_dir.to_str().unwrap().to_string();

    let dir = tmp_dir("perturb");
    let input = write_input_orbit(&dir);
    let input = input.to_str().unwrap().to_string();
    let thrust_path = dir.join("thrust.json");
    std::fs::write(&thrust_path, THRUST_JSON).expect("write thrust json");

    let ballistic_out = dir.join("ballistic");
    let thrust_out = dir.join("thrust");

    // Ballistic reference. Failure here => missing kernels / context init
    // => skip (can't distinguish from a real error otherwise).
    let ballistic = run_cli(&[
        "--data-dir",
        &data_dir,
        "propagate",
        "--input",
        &input,
        "--epoch",
        "59012.0",
        "--out-dir",
        ballistic_out.to_str().unwrap(),
        "--format",
        "json",
    ]);
    if !ballistic.status.success() {
        eprintln!(
            "skipping propagate_with_thrust_json_perturbs_trajectory: ballistic propagate failed \
             (likely missing kernels)\n{}",
            String::from_utf8_lossy(&ballistic.stderr)
        );
        return;
    }

    // Identical orbit, now with the thrust file attached.
    let thrusted = run_cli(&[
        "--data-dir",
        &data_dir,
        "propagate",
        "--input",
        &input,
        "--thrust-arcs",
        thrust_path.to_str().unwrap(),
        "--epoch",
        "59012.0",
        "--out-dir",
        thrust_out.to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert!(
        thrusted.status.success(),
        "thrust propagate must succeed; stderr:\n{}",
        String::from_utf8_lossy(&thrusted.stderr)
    );

    let states = thrust_out.join("states.json");
    assert!(states.exists(), "thrust run must write states.json");

    let a = final_position(&states);
    let b = final_position(&ballistic_out.join("states.json"));
    let delta = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    assert!(
        delta > 1.0e-3,
        "thrust arc must perturb the trajectory vs ballistic (Δposition = {delta:e} AU)"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn thrust_correction_covariance_mismatch_fails_cli() {
    let Some(data_dir) = resolve_data_dir() else {
        eprintln!("skipping thrust_correction_covariance_mismatch_fails_cli: no data dir");
        return;
    };
    let data_dir = data_dir.to_str().unwrap().to_string();

    let dir = tmp_dir("mismatch");
    let input = write_input_orbit(&dir);
    let input = input.to_str().unwrap().to_string();

    // One arc, zero Δv corrections, but a correction covariance: the
    // engine's ThrustParams contract requires the covariance length to
    // match the Δv-correction length. Must reach the CLI as a non-zero
    // exit + named error — never silently dropped/repaired.
    let mismatch_json = r#"{
        "arcs": [{
            "start_mjd_tdb": 59000.0,
            "end_mjd_tdb": 59010.0,
            "thrust_n": 1000.0,
            "mass_kg": 1000.0,
            "steering": {"type": "constant_rtn", "alpha_rad": 0.1, "beta_rad": 0.2},
            "sharpness": 100.0,
            "central_body": 10
        }],
        "correction_covariances": [[[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]]
    }"#;
    let thrust_path = dir.join("thrust_mismatch.json");
    std::fs::write(&thrust_path, mismatch_json).expect("write thrust json");

    // Guard: confirm a plain ballistic run works here (kernels present),
    // so a failure below is genuinely the contract rejection, not a skip.
    let ballistic = run_cli(&[
        "--data-dir",
        &data_dir,
        "propagate",
        "--input",
        &input,
        "--epoch",
        "59005.0",
        "--out-dir",
        dir.join("ballistic").to_str().unwrap(),
        "--format",
        "json",
    ]);
    if !ballistic.status.success() {
        eprintln!("skipping thrust_correction_covariance_mismatch_fails_cli: no kernels");
        return;
    }

    let out = run_cli(&[
        "--data-dir",
        &data_dir,
        "propagate",
        "--input",
        &input,
        "--thrust-arcs",
        thrust_path.to_str().unwrap(),
        "--epoch",
        "59005.0",
        "--out-dir",
        dir.join("thrust").to_str().unwrap(),
        "--format",
        "json",
    ]);
    assert!(
        !out.status.success(),
        "mismatched correction_covariances must fail the CLI, not degrade silently"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("correction_covariances") || stderr.contains("propagation failed"),
        "CLI error must name the offending field / surface the propagation failure; stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
