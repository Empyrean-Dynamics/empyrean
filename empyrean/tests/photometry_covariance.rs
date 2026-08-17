//! Photometric parameter covariance through the distribution.
//!
//! The engine has computed σ_V = sqrt(σ²_photo + σ²_state) since v1.0.0,
//! but the distribution gave the σ_photo term no input slot: the C ABI's
//! `EmpyreanOrbit` had no covariance field and
//! `empyrean_orbit_photometric_params` built every `PhotometricParams`
//! with `covariance: None`, so shipped `mag_sigma` was always the state
//! contribution alone.
//!
//! These tests pin the input slot from the wrapper down, using the one
//! exact oracle the model provides: V = H + 5·log₁₀(rΔ) + φ(α) gives
//! ∂V/∂H ≡ 1, so a photometric covariance of diag(σ_H², 0, 0) with no
//! state covariance must report σ_V = σ_H to machine precision.

use empyrean::{
    Context, CoordinateState, EphemerisConfig, Epoch, Frame, Observer, Orbit, Origin, PhaseFunction,
};
use std::path::PathBuf;

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

/// 99942 Apophis, heliocentric ecliptic Cartesian at MJD 61000 TDB.
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
const OBS_MJD: f64 = 61030.5;

/// 1σ on H, in magnitudes — a realistic short-arc photometric fit.
const SIGMA_H: f64 = 0.3;

fn base_orbit() -> Orbit {
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    );
    Orbit::new(state)
        .with_orbit_id("phot-cov")
        .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
}

fn state_covariance() -> [[f64; 6]; 6] {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-14;
        cov[i + 3][i + 3] = 1.0e-20;
    }
    cov
}

fn with_state_covariance(orbit: Orbit) -> Orbit {
    let state = orbit.state.with_covariance(state_covariance());
    Orbit { state, ..orbit }
}

fn observers(ctx: &Context) -> Vec<Observer> {
    ctx.get_observers(
        &["500"],
        &[Epoch::from_mjd_tdb(OBS_MJD)],
        Frame::ICRF,
        Origin::SSB,
    )
    .expect("geocenter observer state")
}

fn mag_sigma(ctx: &Context, orbit: Orbit) -> f64 {
    let result = ctx
        .generate_ephemeris(&[orbit], &observers(ctx), &EphemerisConfig::default())
        .expect("ephemeris generation must succeed");
    assert_eq!(result.entries.len(), 1);
    result.entries[0].mag_sigma
}

/// The exact oracle. With a photometric covariance of diag(σ_H², 0, 0)
/// and NO state covariance, σ_V must equal σ_H to machine precision,
/// because ∂V/∂H is identically 1.
///
/// FAILS on the pre-fix code with `mag_sigma = NaN`: without a state
/// covariance there was no uncertainty path at all, and the photometric
/// covariance had nowhere to enter from.
#[test]
fn photometric_covariance_alone_reports_sigma_h_exactly() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping photometric_covariance_alone_...: no data dir");
        return;
    };
    let orbit = base_orbit().with_photometry_covariance(Some([
        [SIGMA_H * SIGMA_H, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]));
    let sigma_v = mag_sigma(&ctx, orbit);
    assert!(
        (sigma_v - SIGMA_H).abs() <= 1e-12,
        "sigma_V must equal sigma_H exactly when the state contributes nothing \
         (dV/dH == 1); got {sigma_v} vs {SIGMA_H}"
    );
}

/// The quadrature sum. With BOTH a state covariance and a photometric
/// one, σ_V must be sqrt(σ_state² + σ_H²) against the same row's
/// state-only value — the two terms are summed as independent.
#[test]
fn photometric_and_state_terms_add_in_quadrature() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping photometric_and_state_terms_add_in_quadrature: no data dir");
        return;
    };
    let state_only = mag_sigma(&ctx, with_state_covariance(base_orbit()));
    assert!(
        state_only.is_finite() && state_only > 0.0,
        "the state-only sigma must be finite and positive to compare against, got \
         {state_only}"
    );

    let both = mag_sigma(
        &ctx,
        with_state_covariance(base_orbit()).with_photometry_covariance(Some([
            [SIGMA_H * SIGMA_H, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])),
    );
    let expected = (state_only * state_only + SIGMA_H * SIGMA_H).sqrt();
    assert!(
        (both - expected).abs() <= 1e-12 * expected,
        "sigma_V must be the quadrature sum: got {both}, expected {expected} \
         (state-only {state_only}, sigma_H {SIGMA_H})"
    );
    assert!(
        both > state_only,
        "attaching an H uncertainty must never tighten sigma_V"
    );
}

/// Absence stays absence: an orbit with no photometric covariance and no
/// state covariance reports NaN, not a fabricated zero.
#[test]
fn no_covariance_of_either_kind_reports_nan() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping no_covariance_of_either_kind_reports_nan: no data dir");
        return;
    };
    let sigma_v = mag_sigma(&ctx, base_orbit());
    assert!(
        sigma_v.is_nan(),
        "with neither covariance there is no sigma to report; got {sigma_v}"
    );
}

/// The slope block is read too, not just the H diagonal: a covariance
/// carrying only a slope variance still produces a finite, positive σ_V,
/// which an σ_H-only marshal could not.
#[test]
fn slope_variance_alone_still_contributes() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping slope_variance_alone_still_contributes: no data dir");
        return;
    };
    let orbit = base_orbit().with_photometry_covariance(Some([
        [0.0, 0.0, 0.0],
        [0.0, 0.05 * 0.05, 0.0],
        [0.0, 0.0, 0.0],
    ]));
    let sigma_v = mag_sigma(&ctx, orbit);
    assert!(
        sigma_v.is_finite() && sigma_v > 0.0,
        "the full 3x3 is consumed, not only its (H, H) entry; got {sigma_v}"
    );
    assert!(
        sigma_v < SIGMA_H,
        "a 0.05-mag slope sigma must contribute far less than a 0.3-mag H sigma"
    );
}

/// σ_V = σ_H holds for the H-ONLY shape and nothing else.
///
/// The docs said "an orbit with a photometric covariance and no state
/// covariance reports σ_V = σ_H exactly" without that qualifier, which
/// is false for every SBDB-queried orbit — SBDB publishes σ_H AND σ_G,
/// so the ingested 3×3 is diag(σ_H², σ_G², 0) and the slope term
/// contracts against ∂V/∂slope. This pins the qualified claim from both
/// sides.
#[test]
fn a_slope_variance_makes_sigma_v_exceed_sigma_h() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping a_slope_variance_makes_sigma_v_exceed_sigma_h: no data dir");
        return;
    };
    // The published SBDB Apophis photometric sigmas.
    const SIGMA_H_SBDB: f64 = 0.19;
    const SIGMA_G_SBDB: f64 = 0.11;

    let h_only = mag_sigma(
        &ctx,
        base_orbit().with_photometry_covariance(Some([
            [SIGMA_H_SBDB * SIGMA_H_SBDB, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])),
    );
    assert!(
        (h_only - SIGMA_H_SBDB).abs() <= 1e-12,
        "the H-only shape is the one that gives the identity; got {h_only}"
    );

    let with_slope = mag_sigma(
        &ctx,
        base_orbit().with_photometry_covariance(Some([
            [SIGMA_H_SBDB * SIGMA_H_SBDB, 0.0, 0.0],
            [0.0, SIGMA_G_SBDB * SIGMA_G_SBDB, 0.0],
            [0.0, 0.0, 0.0],
        ])),
    );
    assert!(
        with_slope > SIGMA_H_SBDB + 1e-9,
        "SBDB's published diag(sigma_H^2, sigma_G^2, 0) must report sigma_V strictly \
         above sigma_H — the slope variance contracts against dV/dslope, it does not \
         drop out; got {with_slope} vs {SIGMA_H_SBDB}"
    );
}

/// A covariance that is present but contracts to zero variance still
/// reports NaN — the "finite iff a covariance was carried" reading is
/// false in the reverse direction.
#[test]
fn a_zero_photometric_covariance_still_reports_nan() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping a_zero_photometric_covariance_still_reports_nan: no data dir");
        return;
    };
    let sigma_v = mag_sigma(
        &ctx,
        base_orbit().with_photometry_covariance(Some([[0.0; 3]; 3])),
    );
    assert!(
        sigma_v.is_nan(),
        "a carried covariance that contracts to zero variance has no sigma to report; \
         got {sigma_v}"
    );
}

/// The H–slope correlation is honored: the same diagonal with a strongly
/// negative off-diagonal (the shape a real HG fit produces) gives a
/// different σ_V than the diagonal alone. A marshal that copied only the
/// diagonal would give the same number twice.
#[test]
fn off_diagonal_terms_change_sigma_v() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping off_diagonal_terms_change_sigma_v: no data dir");
        return;
    };
    let diag = [
        [SIGMA_H * SIGMA_H, 0.0, 0.0],
        [0.0, 0.05 * 0.05, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let mut correlated = diag;
    // rho = -0.9 between H and the slope.
    correlated[0][1] = -0.9 * SIGMA_H * 0.05;
    correlated[1][0] = correlated[0][1];

    let a = mag_sigma(&ctx, base_orbit().with_photometry_covariance(Some(diag)));
    let b = mag_sigma(
        &ctx,
        base_orbit().with_photometry_covariance(Some(correlated)),
    );
    assert!(a.is_finite() && b.is_finite());
    assert!(
        (a - b).abs() > 1e-9,
        "the off-diagonal must reach the engine: diagonal-only gave {a}, correlated \
         gave {b}"
    );
}

/// The full round trip the ask names verbatim, minus the network: a
/// photometric covariance attached to an orbit, carried out through
/// `write_orbits_parquet`, read back, and re-fed to ephemeris generation
/// still reports the same σ_V.
///
/// FAILS on the pre-fix code at BOTH ends — the write marshal dropped
/// photometry outright and the read marshal had no covariance slot.
#[test]
fn photometric_covariance_survives_a_parquet_round_trip_into_sigma_v() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping photometric_covariance_survives_a_parquet_round_trip: no data dir");
        return;
    };
    let phot_cov = [
        [SIGMA_H * SIGMA_H, -0.9 * SIGMA_H * 0.05, 0.0],
        [-0.9 * SIGMA_H * 0.05, 0.05 * 0.05, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let orbit = base_orbit().with_photometry_covariance(Some(phot_cov));
    let direct = mag_sigma(&ctx, orbit.clone());
    assert!(direct.is_finite());

    let dir = std::env::temp_dir().join(format!("empyrean-photcov-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("orbits.parquet");
    let batch = empyrean::OrbitBatch::new(
        vec![orbit],
        vec!["phot-cov".to_string()],
        vec![Some("99942".to_string())],
    )
    .expect("a one-orbit batch with matching id arrays is well-formed");
    empyrean::write_orbits_parquet(path.to_str().unwrap(), &batch).expect("write");
    let read_back = empyrean::read_orbits_parquet(path.to_str().unwrap()).expect("read");
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(read_back.orbits.len(), 1);
    assert_eq!(
        read_back.orbits[0].phot_covariance,
        Some(phot_cov),
        "the photometric 3x3 must survive the file round trip verbatim"
    );
    let after = mag_sigma(&ctx, read_back.orbits[0].clone());
    assert!(
        (after - direct).abs() <= 1e-12 * direct,
        "sigma_V after the file round trip ({after}) must match the direct value \
         ({direct})"
    );
}
