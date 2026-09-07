//! A failed batch names the orbit it failed on, end to end through the
//! C ABI.
//!
//! The shape of the gap these close: a downstream caller handed the
//! engine a 4096-state chunk, got back a message with no index in it,
//! and re-ran the chunk one state at a time to find out which member was
//! bad. `Error` now carries the index, the id and the epoch, so the
//! offending row is read off the failure instead of bisected for.

use empyrean::{
    Context, CoordinateState, Epoch, Frame, Orbit, Origin, PropagationConfig, UncertaintyMethod,
};
use std::path::PathBuf;

/// Resolve a usable data dir: `EMPYREAN_DATA_DIR` (CI) else
/// `~/.empyrean/data` (local). Returns `None` to skip when neither
/// yields a working Context — except under `EMPYREAN_DATA_DIR`, where
/// the kernels are supposed to be present and a skip would hide a
/// regression rather than tolerate a bare machine.
fn try_context() -> Option<Context> {
    let declared = std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from);
    let candidates = [
        declared.clone(),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".empyrean/data")),
    ];
    for dir in candidates.into_iter().flatten() {
        match Context::from_data_dir(Some(&dir)) {
            Ok(ctx) => return Some(ctx),
            Err(e) if declared.is_some() => {
                panic!("EMPYREAN_DATA_DIR is set, so this test must not skip: {e}")
            }
            Err(_) => {}
        }
    }
    eprintln!("skipping: no data directory");
    None
}

/// 99942 Apophis, heliocentric ecliptic Cartesian at MJD 61000 TDB.
const APOPHIS: [f64; 6] = [
    -0.078_526_491_490_690_46,
    -0.819_748_051_902_064_6,
    0.041_893_951_532_339_09,
    0.019_875_102_496_888_46,
    0.001_322_088_445_361_402,
    0.000_399_496_044_422_352_2,
];
const T0_MJD: f64 = 61000.0;

fn orbit(id: &str, elements: [f64; 6]) -> Orbit {
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        elements,
        Frame::EclipticJ2000,
        Origin::Sun,
    );
    Orbit::new(state).with_orbit_id(id)
}

/// Three good orbits and one whose z-velocity is NaN, at a known index.
fn batch_with_one_bad_orbit() -> Vec<Orbit> {
    let mut bad = APOPHIS;
    bad[5] = f64::NAN;
    vec![
        orbit("good-0", APOPHIS),
        orbit("good-1", APOPHIS),
        orbit("the-bad-one", bad),
        orbit("good-3", APOPHIS),
    ]
}

fn config() -> PropagationConfig {
    PropagationConfig {
        uncertainty_method: UncertaintyMethod::FirstOrder,
        ..PropagationConfig::default()
    }
}

fn epochs() -> Vec<Epoch> {
    vec![Epoch::from_mjd_tdb(T0_MJD + 30.0)]
}

/// The headline: one bad member in a batch, and the failure says which.
#[test]
fn a_bad_batch_member_is_named_by_index_id_and_epoch() {
    let Some(ctx) = try_context() else { return };

    let err = ctx
        .propagate(&batch_with_one_bad_orbit(), &epochs(), &config())
        .expect_err("a NaN velocity component must fail the call");

    assert_eq!(
        err.orbit_index(),
        Some(2),
        "the failure must name the offending row: {err}"
    );
    assert_eq!(err.orbit_id(), Some("the-bad-one"));
    assert_eq!(err.epoch_mjd_tdb(), Some(T0_MJD));
    assert!(
        err.message.contains("elements[5]"),
        "the message must still say what was wrong, not only where: {}",
        err.message
    );
    // The rendered form carries the position too, for a caller that only
    // logs the error.
    let rendered = err.to_string();
    assert!(
        rendered.contains("orbit 2") && rendered.contains("the-bad-one"),
        "Display must carry the position: {rendered}"
    );
}

/// The positive control. The same batch with every element finite
/// succeeds — so the test above is failing on the NaN, and not on
/// something incidental about the batch, the config or the epochs.
#[test]
fn the_same_batch_without_the_nan_succeeds() {
    let Some(ctx) = try_context() else { return };

    let orbits: Vec<Orbit> = ["good-0", "good-1", "the-bad-one", "good-3"]
        .into_iter()
        .map(|id| orbit(id, APOPHIS))
        .collect();
    let result = ctx
        .propagate(&orbits, &epochs(), &config())
        .expect("an all-finite batch must propagate");
    assert_eq!(result.states.len(), 4);
}

/// The other positive control. A failure that belongs to no single
/// orbit reports no position at all, rather than defaulting to the first
/// row — an index that is always present is an index that means nothing.
#[test]
fn a_failure_with_no_offending_orbit_reports_no_position() {
    let Some(ctx) = try_context() else { return };

    // An empty epoch grid is a batch-wide refusal: every orbit in the
    // call is equally implicated, so there is no row to name.
    let err = ctx
        .propagate(&[orbit("only", APOPHIS)], &[], &config())
        .expect_err("an empty epoch grid must fail the call");

    assert_eq!(err.orbit_index(), None, "no row is at fault: {err}");
    assert_eq!(err.orbit_id(), None);
    assert_eq!(err.epoch_mjd_tdb(), None);
    let rendered = err.to_string();
    assert!(
        !rendered.contains("[orbit "),
        "Display must not invent a position: {rendered}"
    );
}

/// The position survives the `BuiltSystem` handle path too — the entry
/// point a caller reaches for when it propagates the same batch shape
/// many times, and the one where the engine hands the boundary its own
/// typed error rather than a rendered string.
#[test]
fn the_builtsystem_path_names_the_offending_orbit() {
    let Some(ctx) = try_context() else { return };

    let system = ctx
        .built_system(
            config().force_model,
            config().frame,
            config().advanced.encounter_timescale_divisor,
        )
        .expect("build a force-model handle");
    let err = system
        .propagate(&ctx, &batch_with_one_bad_orbit(), &epochs(), &config())
        .expect_err("a NaN velocity component must fail the call");

    assert_eq!(err.orbit_index(), Some(2), "{err}");
    assert_eq!(err.orbit_id(), Some("the-bad-one"));
}
