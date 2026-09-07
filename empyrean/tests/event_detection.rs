//! Switching event detection off is honoured, costs nothing in
//! accuracy, and is the reason to reach for it.
//!
//! The five per-type flags on `EventConfig` filter what is *emitted*.
//! They do not stop the detectors running: the C ABI used to drop
//! `detection_enabled` entirely, so a caller who turned every
//! detector off through the wrapper still paid per-substep detection on
//! every accepted integrator step and had no way to say otherwise.
//!
//! Four things are worth pinning: the switch reaches the engine (no
//! events come back), it changes no propagated number (state, covariance
//! and STM are bit-identical either way), it is worth having (the call is
//! measurably faster), and the two uncertainty methods that resolve
//! themselves from detector output are refused rather than served
//! silently degraded.

use empyrean::{
    Context, CoordinateState, Epoch, EventConfig, Frame, Orbit, Origin, PropagationConfig,
    UncertaintyMethod,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

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
/// Chosen because it produces Earth close approaches on a multi-year
/// arc, so the detectors have something to find.
const APOPHIS: [f64; 6] = [
    -0.078_526_491_490_690_46,
    -0.819_748_051_902_064_6,
    0.041_893_951_532_339_09,
    0.019_875_102_496_888_46,
    0.001_322_088_445_361_402,
    0.000_399_496_044_422_352_2,
];
const T0_MJD: f64 = 61000.0;
/// Just over one orbital period (~324 days), so the arc contains a
/// periapsis and the detectors have something to find, without reaching
/// the 2029 Earth encounter — which is far more expensive to integrate
/// and would make this a slow test for no extra coverage.
const T1_MJD: f64 = 61400.0;
const N_ORBITS: usize = 64;

/// A covariance-free batch — the shape a caller uses when it wants
/// states and nothing else, and the shape the cost was measured on.
fn batch() -> Vec<Orbit> {
    (0..N_ORBITS)
        .map(|i| {
            let mut e = APOPHIS;
            // Spread the batch so the members are genuinely different
            // trajectories rather than 64 copies of one cache line.
            e[0] += (i as f64) * 1.0e-7;
            let state = CoordinateState::cartesian(
                Epoch::from_mjd_tdb(T0_MJD),
                e,
                Frame::EclipticJ2000,
                Origin::Sun,
            );
            Orbit::new(state).with_orbit_id(format!("o{i}"))
        })
        .collect()
}

/// The same batch with an input covariance, so the identity check has a
/// covariance and an STM to compare and not only a position.
fn batch_with_covariance() -> Vec<Orbit> {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-14;
        cov[i + 3][i + 3] = 1.0e-20;
    }
    batch()
        .into_iter()
        .enumerate()
        .map(|(i, _)| {
            let mut e = APOPHIS;
            e[0] += (i as f64) * 1.0e-7;
            let state = CoordinateState::cartesian(
                Epoch::from_mjd_tdb(T0_MJD),
                e,
                Frame::EclipticJ2000,
                Origin::Sun,
            )
            .with_covariance(cov);
            Orbit::new(state).with_orbit_id(format!("o{i}"))
        })
        .collect()
}

fn epochs() -> Vec<Epoch> {
    vec![
        Epoch::from_mjd_tdb(T1_MJD - 30.0),
        Epoch::from_mjd_tdb(T1_MJD),
    ]
}

fn config(detection_enabled: bool) -> PropagationConfig {
    PropagationConfig {
        uncertainty_method: UncertaintyMethod::FirstOrder,
        events: EventConfig {
            detection_enabled,
            ..EventConfig::default()
        },
        // One thread, so the two arms are compared on equal footing and
        // the ratio does not move with whatever else the machine is
        // doing across its cores.
        num_threads: std::num::NonZeroUsize::new(1),
        ..PropagationConfig::default()
    }
}

/// Detection off means no events, and detection on means some — the
/// second half is the positive control without which the first proves
/// only that the batch is dull.
#[test]
fn detection_off_yields_no_events_and_on_yields_some() {
    let Some(ctx) = try_context() else { return };
    let orbits = batch();

    let on = ctx
        .propagate(&orbits, &epochs(), &config(true))
        .expect("propagation with detection on");
    assert!(
        !on.events.is_empty(),
        "the fixture must produce events with detection on, or the off-case proves nothing"
    );

    let off = ctx
        .propagate(&orbits, &epochs(), &config(false))
        .expect("propagation with detection off");
    assert!(
        off.events.is_empty(),
        "detection off must yield no events, got {}",
        off.events.len()
    );
}

/// Detection is observation, not dynamics. Switching it off must move no
/// propagated number — and "number" has to mean the covariance and the
/// STM too, not just the position: the covariance is exactly where a
/// silent degradation would hide, and a covariance-free batch cannot see
/// one. Run on a covariance-carrying batch with the STM requested, so
/// all three are present to compare.
#[test]
fn detection_off_changes_no_propagated_number() {
    let Some(ctx) = try_context() else { return };
    let orbits = batch_with_covariance();

    let with_stm = |detection_enabled: bool| PropagationConfig {
        compute_stm: true,
        ..config(detection_enabled)
    };
    let on = ctx
        .propagate(&orbits, &epochs(), &with_stm(true))
        .expect("propagation with detection on");
    let off = ctx
        .propagate(&orbits, &epochs(), &with_stm(false))
        .expect("propagation with detection off");

    assert_eq!(on.states.len(), off.states.len());
    // The fixture has to actually carry the things being compared, or
    // the loop below asserts nothing about them.
    assert!(
        on.states.iter().all(|s| s.covariance.is_some()),
        "the fixture must propagate a covariance for the comparison to mean anything"
    );
    assert!(
        on.states.iter().all(|s| s.stm.is_some()),
        "the fixture must produce an STM for the comparison to mean anything"
    );

    for (k, (a, b)) in on.states.iter().zip(off.states.iter()).enumerate() {
        assert_eq!(
            a.position, b.position,
            "state {k} position moved when detection was switched off"
        );
        assert_eq!(
            a.velocity, b.velocity,
            "state {k} velocity moved when detection was switched off"
        );
        assert_eq!(a.epoch, b.epoch, "state {k} epoch moved");
        assert_eq!(
            a.covariance, b.covariance,
            "state {k} covariance moved when detection was switched off"
        );
        assert_eq!(
            a.stm, b.stm,
            "state {k} STM moved when detection was switched off"
        );
        assert_eq!(
            a.resolved_kind, b.resolved_kind,
            "state {k} resolved covariance kind moved when detection was switched off"
        );
    }
}

/// Time one propagation.
fn time_once(ctx: &Context, orbits: &[Orbit], detection_enabled: bool) -> Duration {
    let t = Instant::now();
    let r = ctx
        .propagate(orbits, &epochs(), &config(detection_enabled))
        .expect("propagation");
    let elapsed = t.elapsed();
    // Keep the result alive until after the clock is read so the timing
    // never accidentally measures a call the optimizer elided.
    std::hint::black_box(&r);
    elapsed
}

/// And the reason to have the switch at all: it is measurably faster.
///
/// Timed as a ratio of alternating runs rather than as two absolute
/// numbers, and judged on the median of three pairs, because a shared
/// machine moves absolute wall-clocks around far more than it moves the
/// ratio between two calls run back to back. The bound is deliberately
/// slack against the effect it guards — the measured saving is around a
/// third of the call, and the assertion only requires a tenth — so this
/// catches the switch being dropped again without failing under load.
#[test]
fn detection_off_is_measurably_faster() {
    let Some(ctx) = try_context() else { return };
    let orbits = batch();

    // Warm the kernels and the allocator; the first call through a
    // context pays one-time costs that belong to neither arm.
    let _ = time_once(&ctx, &orbits, true);

    let mut ratios = Vec::new();
    for _ in 0..3 {
        let on = time_once(&ctx, &orbits, true);
        let off = time_once(&ctx, &orbits, false);
        eprintln!(
            "detection on {:.0} ms, off {:.0} ms ({:.2}x)",
            on.as_secs_f64() * 1e3,
            off.as_secs_f64() * 1e3,
            on.as_secs_f64() / off.as_secs_f64()
        );
        ratios.push(off.as_secs_f64() / on.as_secs_f64());
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    let median = ratios[1];
    assert!(
        median < 0.90,
        "detection off must be measurably faster; median off/on ratio was {median:.3}"
    );
}

/// The blocker this switch would otherwise ship: `Auto` resolves itself
/// from detected close approaches, so with detection off it does not
/// fail — it quietly returns `Linear` everywhere and no impact
/// probabilities. Refused by name instead.
#[test]
fn auto_under_detection_off_is_refused() {
    let Some(ctx) = try_context() else { return };

    let config = PropagationConfig {
        uncertainty_method: UncertaintyMethod::auto(),
        events: EventConfig {
            detection_enabled: false,
            ..EventConfig::default()
        },
        ..PropagationConfig::default()
    };
    let err = ctx
        .propagate(&batch_with_covariance(), &epochs(), &config)
        .expect_err("Auto with detection off must be refused, not served degraded");
    assert!(
        err.message.contains("detection_enabled = false")
            && err.message.contains("uncertainty_method = Auto")
            && err.message.contains("close approaches"),
        "the refusal must name both halves and why: {}",
        err.message
    );
}

/// The mixture splits at detected close approaches, so with none it
/// never splits and a caller receives a single Gaussian under a name
/// that promises a mixture. Refused on the same grounds.
#[test]
fn the_mixture_under_detection_off_is_refused() {
    let Some(ctx) = try_context() else { return };

    let config = PropagationConfig {
        uncertainty_method: UncertaintyMethod::gaussian_mixture(),
        events: EventConfig {
            detection_enabled: false,
            ..EventConfig::default()
        },
        ..PropagationConfig::default()
    };
    let err = ctx
        .propagate(&batch_with_covariance(), &epochs(), &config)
        .expect_err("the mixture with detection off must be refused");
    assert!(
        err.message.contains("GaussianMixture") && err.message.contains("close approaches"),
        "the refusal must name the method and why: {}",
        err.message
    );
}

/// The positive control for both refusals, and the reason they mean
/// something: every method that computes its covariance along the
/// trajectory is served under detection off, and both refused methods
/// are served with detection on. A blanket refusal would pass the two
/// tests above and fail this one.
#[test]
fn every_other_pairing_is_served() {
    let Some(ctx) = try_context() else { return };
    let orbits = batch_with_covariance();

    // Detection off is legal for the methods that do not read detector
    // output.
    for method in [
        UncertaintyMethod::FirstOrder,
        UncertaintyMethod::SecondOrder,
    ] {
        let config = PropagationConfig {
            uncertainty_method: method.clone(),
            events: EventConfig {
                detection_enabled: false,
                ..EventConfig::default()
            },
            ..PropagationConfig::default()
        };
        ctx.propagate(&orbits, &epochs(), &config)
            .unwrap_or_else(|e| panic!("{method:?} with detection off must be served: {e}"));
    }

    // And the two refused methods are refused only for the pairing, not
    // outright.
    for method in [
        UncertaintyMethod::auto(),
        UncertaintyMethod::gaussian_mixture(),
    ] {
        let config = PropagationConfig {
            uncertainty_method: method.clone(),
            ..PropagationConfig::default()
        };
        ctx.propagate(&orbits, &epochs(), &config)
            .unwrap_or_else(|e| panic!("{method:?} with detection ON must be served: {e}"));
    }
}
