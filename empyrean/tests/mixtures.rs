//! The AGM mixture read-back, end to end through the C ABI.
//!
//! The wrapper used to copy exactly three arrays off the FFI result —
//! states, object_ids and events — and never touch `mixtures` /
//! `num_mixtures`, so every retained component died at that one
//! function while the C ABI happily populated them. These tests are the
//! shape of that gap: a real propagation with a genuinely nonlinear
//! Earth close approach, asserting the components arrive.

use empyrean::propagate::MixtureComponent;
use empyrean::{
    Context, CoordinateState, Epoch, EventConfig, Frame, Orbit, Origin, PropagationConfig,
    UncertaintyMethod,
};
use std::path::PathBuf;

/// Resolve a usable data dir: `EMPYREAN_DATA_DIR` (CI) else
/// `~/.empyrean/data` (local). Returns `None` to skip when neither
/// yields a working Context.
fn try_context() -> Option<Context> {
    let candidates = [
        std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from),
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".empyrean/data")),
    ];
    for dir in candidates.into_iter().flatten() {
        if let Ok(ctx) = Context::from_data_dir(Some(&dir)) {
            return Some(ctx);
        }
    }
    None
}

/// 99942 Apophis, heliocentric ecliptic Cartesian at MJD 61000 TDB —
/// upstream of the 2029-04-13 Earth encounter.
// Shortest round-tripping f64 literals of the JPL SBDB state; the
// source values carry more digits than an f64 can hold, and these
// decode to the same bits.
const APOPHIS: [f64; 6] = [
    -0.078_526_491_490_690_46,
    -0.819_748_051_902_064_6,
    0.041_893_951_532_339_09,
    0.019_875_102_496_888_46,
    0.001_322_088_445_361_402,
    0.000_399_496_044_422_352_2,
];

const T0_MJD: f64 = 61000.0;
/// Straddles the 2029 Earth close approach (MJD ≈ 62239).
const T1_MJD: f64 = 62400.0;

/// Apophis with a deliberately loose covariance — loose enough that the
/// mapping through the 2029 Earth encounter is genuinely nonlinear, which
/// is the regime AGM exists for. σ_pos ≈ 1.5e3 km, σ_vel ≈ 1.5e-4 km/s.
fn apophis_with_loose_covariance() -> Orbit {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-10;
        cov[i + 3][i + 3] = 1.0e-16;
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    )
    .with_covariance(cov);
    Orbit::new(state)
        .with_orbit_id("apophis-mixture")
        .with_object_id("99942")
}

/// A tight-covariance twin of the same object: well-determined enough
/// that the splitter has no reason to fire.
fn apophis_tight() -> Orbit {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-18;
        cov[i + 3][i + 3] = 1.0e-24;
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    )
    .with_covariance(cov);
    Orbit::new(state).with_orbit_id("apophis-tight")
}

/// The same object with NO covariance of any kind — a perfectly
/// ordinary ballistic input, and the one batch shape for which the
/// engine returns no mixture chains at all.
fn apophis_ballistic(id: &str) -> Orbit {
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    );
    Orbit::new(state).with_orbit_id(id)
}

fn epochs() -> Vec<Epoch> {
    let mut out = Vec::new();
    let mut t = T0_MJD;
    while t <= T1_MJD {
        out.push(Epoch::from_mjd_tdb(t));
        t += 100.0;
    }
    out
}

fn mixture_config() -> PropagationConfig {
    PropagationConfig {
        uncertainty_method: UncertaintyMethod::gaussian_mixture(),
        events: EventConfig {
            body_filter: vec![Origin::EARTH, Origin::MOON],
            ..EventConfig::default()
        },
        ..PropagationConfig::default()
    }
}

/// THE test that would have caught the drop.
///
/// A covariance-bearing NEO through a nonlinear Earth close approach
/// under `GaussianMixture` must come back with retained components: one
/// chain per input orbit, at least one of them non-empty, every weight
/// finite and positive, and every basis tag decoded to a real enum
/// rather than erroring out of the marshal.
#[test]
fn gaussian_mixture_retains_components_through_an_earth_encounter() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping gaussian_mixture_retains_components_...: no data dir");
        return;
    };
    let orbits = [apophis_with_loose_covariance()];
    let result = ctx
        .propagate(&orbits, &epochs(), &mixture_config())
        .expect("mixture propagation must succeed");

    assert_eq!(
        result.mixtures.len(),
        orbits.len(),
        "one chain per input orbit, always — the positional join with the input \
         batch is the C ABI's documented contract"
    );

    let chain = &result.mixtures[0];
    assert_eq!(
        chain.ca_epochs_mjd_tdb.len(),
        chain.components.len(),
        "the epoch list and the component groups are index-aligned"
    );
    assert!(
        !chain.components.is_empty(),
        "the 2029 Apophis encounter with a loose covariance must fire the splitter; \
         an empty chain here means the components were dropped between the C ABI \
         and the wrapper (the original defect) or the fixture stopped being nonlinear"
    );

    let mut total_components = 0usize;
    for (k, group) in chain.components.iter().enumerate() {
        let t = chain.ca_epochs_mjd_tdb[k];
        assert!(
            (T0_MJD..=T1_MJD).contains(&t),
            "retained CA epoch {t} must lie inside the propagation span"
        );
        for c in group {
            total_components += 1;
            assert!(
                c.weight.is_finite() && c.weight > 0.0,
                "component weight must be finite and positive, got {}",
                c.weight
            );
            assert!(
                c.mean.iter().all(|v| v.is_finite()),
                "component mean must be finite"
            );
            assert!(
                c.covariance.iter().flatten().all(|v| v.is_finite()),
                "component covariance must be finite"
            );
            // The basis decoded rather than defaulted — an unknown tag
            // errors out of the marshal, so reaching here proves it.
            assert!(matches!(
                c.frame,
                Frame::ICRF | Frame::EclipticJ2000 | Frame::ITRF93
            ));
            let _ = c.origin.naif_id();
        }
    }
    assert!(
        total_components > 0,
        "a non-empty group must hold components"
    );

    // The retained weights are NOT renormalized on this path, so they
    // may sum to less than 1 (a sub-Gaussian that missed the CA
    // contributes nothing). They must never sum to MORE than 1.
    for group in &chain.components {
        let sum: f64 = group.iter().map(|c| c.weight).sum();
        assert!(
            sum <= 1.0 + 1e-9,
            "retained weights must never exceed 1, got {sum}"
        );
    }
}

/// The chains are aligned with the object ids by index, and a
/// FirstOrder run still yields one chain per orbit — empty, not absent.
#[test]
fn mixture_chains_are_positional_and_first_order_yields_empty_chains() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping mixture_chains_are_positional_...: no data dir");
        return;
    };
    let orbits = [apophis_with_loose_covariance(), apophis_tight()];
    let result = ctx
        .propagate(&orbits, &epochs(), &mixture_config())
        .expect("mixture propagation must succeed");
    assert_eq!(result.mixtures.len(), orbits.len());
    for (i, chain) in result.mixtures.iter().enumerate() {
        assert_eq!(
            Some(chain.orbit_id.as_str()),
            orbits[i].orbit_id.as_deref(),
            "chain {i} must carry input orbit {i}'s own id — the positional join \
             the C ABI documents"
        );
    }

    let first_order = PropagationConfig {
        uncertainty_method: UncertaintyMethod::FirstOrder,
        ..PropagationConfig::default()
    };
    let fo = ctx
        .propagate(&orbits, &epochs(), &first_order)
        .expect("first-order propagation must succeed");
    assert_eq!(
        fo.mixtures.len(),
        orbits.len(),
        "FirstOrder still emits one row per orbit — empty, not absent"
    );
    for chain in &fo.mixtures {
        assert!(
            chain.ca_epochs_mjd_tdb.is_empty() && chain.components.is_empty(),
            "FirstOrder retains no mixture components"
        );
    }
}

/// A batch whose orbits carry no covariance at all is the one case the
/// engine answers with an EMPTY mixtures vector rather than one row per
/// orbit — it zeroes the vector wholesale when no orbit produced
/// sensitivity tensors. The wrapper pads it back to one empty chain per
/// orbit, so `mixtures` is positional with the input batch for every
/// method and every batch.
///
/// FAILS on the unpadded marshal with `mixtures.len() == 0` for two
/// input orbits — under which `zip(&orbits, &result.mixtures)` iterates
/// zero times and reads as "every orbit processed", and
/// `&result.mixtures[i]` panics.
#[test]
fn a_covariance_less_batch_still_yields_one_empty_chain_per_orbit() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping a_covariance_less_batch_...: no data dir");
        return;
    };
    let orbits = [
        apophis_ballistic("ballistic-a"),
        apophis_ballistic("ballistic-b"),
    ];
    // A short grid: this is about the marshal's shape, not the dynamics.
    let grid = vec![
        Epoch::from_mjd_tdb(T0_MJD),
        Epoch::from_mjd_tdb(T0_MJD + 10.0),
    ];

    for (label, method) in [
        ("FirstOrder", UncertaintyMethod::FirstOrder),
        ("Mixture", UncertaintyMethod::gaussian_mixture()),
    ] {
        let config = PropagationConfig {
            uncertainty_method: method,
            ..PropagationConfig::default()
        };
        let result = ctx
            .propagate(&orbits, &grid, &config)
            .expect("a covariance-less propagation must succeed");
        assert_eq!(
            result.mixtures.len(),
            orbits.len(),
            "{label}: one row per input orbit, even when the engine produced no chains \
             at all — otherwise a positional join silently covers nothing"
        );
        for (i, chain) in result.mixtures.iter().enumerate() {
            assert!(
                chain.ca_epochs_mjd_tdb.is_empty() && chain.components.is_empty(),
                "{label}: chain {i} must be empty — nothing split"
            );
        }
        assert_eq!(
            orbits.iter().zip(&result.mixtures).count(),
            orbits.len(),
            "{label}: the documented positional zip must cover every input orbit"
        );
    }
}

/// The exposed components are the same objects the collapsed readback is
/// derived from: recomputing the moment collapse
/// \\(\Sigma = \sum_k \bar{w}_k (\Sigma_k + d_k d_k^\top)\\) from them
/// reproduces `covariance_at_cartesian` on the Mixture-tagged row.
///
/// Catches a future divergence between the two surfaces — the whole
/// point of exposing components beside a collapsed tag.
#[test]
fn exposed_components_reproduce_the_collapsed_covariance() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping exposed_components_reproduce_...: no data dir");
        return;
    };
    let orbits = [apophis_with_loose_covariance()];
    let grid = epochs();
    let result = ctx
        .propagate(&orbits, &grid, &mixture_config())
        .expect("mixture propagation must succeed");

    // Queried per epoch rather than through `covariance_series_cartesian`:
    // the series accessor fails the WHOLE orbit if any single epoch
    // resolves to Mixture with nothing retained there, which is a real
    // engine-side condition on this fixture and is not what this test is
    // about.
    let tolerance_days = 50.0;
    let mut checked = 0usize;
    for k in 0..grid.len() {
        let Ok(tagged) = result.covariance_at_cartesian(0, k) else {
            continue;
        };
        if tagged.kind != empyrean::CovarianceKind::Mixture {
            continue;
        }
        let epoch_mjd = tagged.epoch.mjd_tdb().expect("finite epoch");
        let Some(components) = result.mixture_at(0, epoch_mjd, tolerance_days) else {
            continue;
        };
        if components.is_empty() {
            continue;
        }
        let collapsed = collapse(components);
        let reference = tagged;
        // Compare relative to the diagonal scale: the collapse is a sum
        // of products of ~1e-10-scale numbers, so an absolute tolerance
        // would be meaningless.
        let scale: f64 = (0..6)
            .map(|i| reference.matrix[i][i].abs())
            .fold(0.0, f64::max);
        for (r, (got_row, want_row)) in collapsed.iter().zip(reference.matrix).enumerate() {
            for (c, (got, want)) in got_row.iter().zip(want_row).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-6 * scale.max(f64::MIN_POSITIVE),
                    "recomputed collapse[{r}][{c}] = {got} disagrees with the tagged \
                     readback {want} (scale {scale:e}) — the exposed components and \
                     the collapsed covariance have diverged"
                );
            }
        }
        checked += 1;
        break;
    }
    assert!(
        checked > 0,
        "no Mixture-tagged epoch co-located with retained components — the \
         consistency claim was never exercised"
    );
}

/// Moment collapse of a component set, weights renormalized to sum to 1
/// (which is what the engine's own collapse does, and why the retained
/// deficit is invisible on the collapsed path).
fn collapse(components: &[MixtureComponent]) -> [[f64; 6]; 6] {
    let wsum: f64 = components.iter().map(|c| c.weight).sum();
    let mut mean = [0.0_f64; 6];
    for c in components {
        for (m, x) in mean.iter_mut().zip(c.mean) {
            *m += (c.weight / wsum) * x;
        }
    }
    let mut out = [[0.0_f64; 6]; 6];
    for c in components {
        let w = c.weight / wsum;
        let d: Vec<f64> = c.mean.iter().zip(mean).map(|(m, mu)| m - mu).collect();
        for (r, out_row) in out.iter_mut().enumerate() {
            for (col, v) in out_row.iter_mut().enumerate() {
                *v += w * (c.covariance[r][col] + d[r] * d[col]);
            }
        }
    }
    out
}
