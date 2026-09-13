//! The boxed state transition tensor still carries the engine's numbers,
//! and `into_states` still hands back the states the result held.
//!
//! `PropagatedState::stt` moved behind a `Box` to take 1728 bytes off
//! every state that never asked for second order. The size win is
//! asserted in the crate's unit tests; what those cannot see is whether
//! the tensor still *arrives* — a marshalling change that quietly
//! dropped it would leave every size assertion green. These run a real
//! second-order propagation and read it.

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

/// Apophis with an input covariance — second order needs one to have
/// anything to propagate.
fn apophis() -> Orbit {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-14;
        cov[i + 3][i + 3] = 1.0e-20;
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    )
    .with_covariance(cov);
    Orbit::new(state).with_orbit_id("apophis")
}

fn epochs() -> Vec<Epoch> {
    vec![
        Epoch::from_mjd_tdb(T0_MJD + 30.0),
        Epoch::from_mjd_tdb(T0_MJD + 60.0),
    ]
}

fn second_order() -> PropagationConfig {
    PropagationConfig {
        uncertainty_method: UncertaintyMethod::SecondOrder,
        ..PropagationConfig::default()
    }
}

/// The tensor arrives, and it holds real second-order structure — not a
/// box of zeros, which is what a marshalling drop would leave behind.
#[test]
fn a_second_order_request_still_returns_a_populated_tensor() {
    let Some(ctx) = try_context() else { return };

    let result = ctx
        .propagate(&[apophis()], &epochs(), &second_order())
        .expect("second-order propagation");

    for (k, state) in result.states.iter().enumerate() {
        let stt = state
            .stt
            .as_ref()
            .unwrap_or_else(|| panic!("state {k} carries no STT under SecondOrder"));
        let mut nonzero = 0usize;
        for a in 0..6 {
            for b in 0..6 {
                for c in 0..6 {
                    let v = stt[a][b][c];
                    assert!(
                        v.is_finite(),
                        "stt[{a}][{b}][{c}] is not finite at state {k}"
                    );
                    if v != 0.0 {
                        nonzero += 1;
                    }
                }
            }
        }
        assert!(
            nonzero > 0,
            "state {k}'s STT is all zeros — the tensor was dropped, not propagated"
        );
    }
}

/// The positive control for the assertion above: a first-order request
/// returns no tensor at all. Without this, an implementation that
/// fabricated a tensor for every request would pass the first test.
#[test]
fn a_first_order_request_returns_no_tensor() {
    let Some(ctx) = try_context() else { return };

    let config = PropagationConfig {
        uncertainty_method: UncertaintyMethod::FirstOrder,
        ..PropagationConfig::default()
    };
    let result = ctx
        .propagate(&[apophis()], &epochs(), &config)
        .expect("first-order propagation");

    for (k, state) in result.states.iter().enumerate() {
        assert!(
            state.stt.is_none(),
            "state {k} carries an STT the request never asked for"
        );
    }
}

/// `into_states` is a release, not a filter: the states it hands back
/// are the ones the result held, tensor and all.
#[test]
fn into_states_hands_back_the_states_unchanged() {
    let Some(ctx) = try_context() else { return };

    let held = ctx
        .propagate(&[apophis()], &epochs(), &second_order())
        .expect("second-order propagation")
        .states
        .clone();
    let taken = ctx
        .propagate(&[apophis()], &epochs(), &second_order())
        .expect("second-order propagation")
        .into_states();

    assert_eq!(taken.len(), held.len());
    assert_eq!(taken, held, "into_states must not alter the states");
    assert!(
        taken.iter().all(|s| s.stt.is_some()),
        "the tensor must survive the release"
    );
}
