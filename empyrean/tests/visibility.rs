//! `Context::compute_visibility` end to end — through the C ABI and the
//! wrapper's `from_ffi`, against a real catalog orbit (99942 Apophis) and a
//! ground observatory plus a horizonless one.
//!
//! The night product, the anchors, and the Moon circumstances only exist
//! once the survey has run a real propagation and ephemeris generation, so
//! these are gated on a usable data directory and skip (not fail) without
//! one.

use empyrean::{
    AnchorPolicy, Context, CoordinateState, Epoch, Frame, ObservatoryConfig, Orbit, Origin,
    PhaseFunction, PlanningConfig, TwilightThresholds,
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
const APOPHIS: [f64; 6] = [
    -0.078_526_491_490_690_46,
    -0.819_748_051_902_064_6,
    0.041_893_951_532_339_09,
    0.019_875_102_496_888_46,
    0.001_322_088_445_361_402,
    0.000_399_496_044_422_352_2,
];
const T0_MJD: f64 = 61000.0;

fn apophis() -> Orbit {
    let mut cov = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        cov[i][i] = 1.0e-12;
        cov[i + 3][i + 3] = 1.0e-18;
    }
    let state = CoordinateState::cartesian(
        Epoch::from_mjd_tdb(T0_MJD),
        APOPHIS,
        Frame::EclipticJ2000,
        Origin::Sun,
    )
    .with_covariance(cov);
    Orbit::new(state)
        .with_orbit_id("apophis-visibility")
        .with_object_id("99942")
        // Apophis's absolute magnitude (HG, H = 19.7, G = 0.15) so apparent
        // magnitude is computed from the geometry rather than left NaN —
        // without it the magnitude gate can never pass and `observable`
        // (the four-gate conjunction) is structurally all-False.
        .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
}

fn ground_and_geocenter() -> PlanningConfig {
    // F51 (Pan-STARRS 1, Haleakala) — a ground site with a local horizon.
    // 500 — the geocentre, a horizonless observer.
    let f51 = ObservatoryConfig {
        obs_code: "F51".to_string(),
        sigma_arcsec: [0.2, 0.2],
        max_apparent_mag: 24.0,
        min_elongation_deg: 60.0,
        min_elevation_deg: 20.0,
        max_sun_altitude_deg: None,
        twilight_thresholds: None,
        // An epoch inside the surveyed window → every night it lands in
        // carries a fourth, ExplicitUTC, anchor.
        explicit_anchor_utc: Some(Epoch::from_mjd_tdb(T0_MJD + 1.0)),
    };
    let geocentre = ObservatoryConfig {
        obs_code: "500".to_string(),
        explicit_anchor_utc: None,
        ..f51.clone()
    };
    PlanningConfig {
        observatories: vec![f51, geocentre],
        ..PlanningConfig::default()
    }
}

#[test]
fn a_ground_site_gets_nights_anchors_and_moon_a_horizonless_one_does_not() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping compute_visibility from-engine test: no data dir");
        return;
    };
    let config = ground_and_geocenter();
    let result = ctx
        .compute_visibility(
            &apophis(),
            Some("apophis-visibility"),
            Epoch::from_mjd_tdb(T0_MJD),
            Epoch::from_mjd_tdb(T0_MJD + 2.0),
            0.02,
            &config,
        )
        .expect("the survey runs over a valid window and observatory set");

    assert_eq!(result.orbit_id, "apophis-visibility");
    assert_eq!(result.observatory_visibility.len(), 2, "two observatories");

    let f51 = result
        .observatory_visibility
        .iter()
        .find(|o| o.obs_code == "F51")
        .expect("F51 is in the result");
    let geo = result
        .observatory_visibility
        .iter()
        .find(|o| o.obs_code == "500")
        .expect("500 is in the result");

    // The orbit carries photometry (HG, H = 19.7), so the engine predicts an
    // apparent magnitude on every point and the magnitude gate passes where
    // the target clears the configured limit (here ~mag 21.3 vs a 24.0 limit).
    // Until the C ABI carried the orbit's photometry into the survey
    // (`empyrean_orbit_to_orbits`), `mag` was `None` on every point and the
    // magnitude gate — hence `observable` — could never pass. The engine
    // computes `mag` independently of the elongation geometry, so this holds
    // even though this near-conjunction window has no fully observable point.
    assert!(
        f51.points.iter().any(|p| p.mag.is_some() && p.magnitude_ok),
        "a photometry-bearing orbit yields a computed apparent magnitude and a passing \
         magnitude gate"
    );

    // Ground site: nights exist, with the three site-derived anchors always
    // present and (because an in-window explicit epoch was supplied) a fourth
    // on the night it lands in.
    assert!(!f51.nights.is_empty(), "a 2-day window spans local nights");
    let a_night = &f51.nights[0];
    let policies: Vec<_> = a_night.anchors.iter().map(|a| a.policy).collect();
    assert!(
        policies.contains(&AnchorPolicy::Midnight)
            && policies.contains(&AnchorPolicy::Transit)
            && policies.contains(&AnchorPolicy::NearestObservableToMidnight),
        "the three site-derived anchor policies are always present, got {policies:?}"
    );
    let has_four_anchor_night = f51.nights.iter().any(|n| {
        n.anchors
            .iter()
            .any(|a| matches!(a.policy, AnchorPolicy::ExplicitUTC(_)))
            && n.anchors.len() >= 4
    });
    assert!(
        has_four_anchor_night,
        "the in-window explicit anchor gives some night four anchors"
    );

    // Moon circumstances ride the per-epoch points on a site with a horizon.
    assert!(
        f51.points
            .iter()
            .any(|p| p.moon_altitude_deg.is_some() && p.moon_illuminated_fraction.is_some()),
        "Moon altitude and illuminated fraction are populated for a ground site"
    );

    // The orbit carries a covariance, so the Midnight anchor (which always has
    // an epoch) carries a sky covariance under the resolved method.
    let midnight = a_night
        .anchors
        .iter()
        .find(|a| a.policy == AnchorPolicy::Midnight && a.epoch.is_some())
        .expect("the Midnight anchor has an epoch");
    let circ = midnight
        .circumstances
        .as_ref()
        .expect("an anchor with an epoch carries circumstances");
    assert!(
        circ.sky_covariance.is_some() && circ.sky_covariance_kind.is_some(),
        "a covariance-bearing orbit yields a sky covariance at the anchor epoch"
    );

    // Horizonless observer: no nights, with the structural note; the per-epoch
    // rows still exist (elevation not applicable).
    assert!(geo.nights.is_empty(), "the geocentre has no local night");
    assert!(
        geo.note.is_some(),
        "the horizonless observer's empty nights carry a reason"
    );
    assert!(
        geo.points.iter().all(|p| p.elevation_deg.is_none()),
        "a horizonless observer reports no elevation"
    );
}

#[test]
fn a_non_descending_twilight_set_is_refused_by_the_observatory_code() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping twilight-validation test: no data dir");
        return;
    };
    // Civil ABOVE sunset — not strictly descending. The engine's group
    // validator refuses the survey, naming the observatory.
    let bad = ObservatoryConfig {
        obs_code: "F51".to_string(),
        sigma_arcsec: [0.2, 0.2],
        max_apparent_mag: 24.0,
        min_elongation_deg: 60.0,
        min_elevation_deg: 20.0,
        max_sun_altitude_deg: None,
        twilight_thresholds: Some(TwilightThresholds {
            sunset_sun_altitude_deg: -6.0,
            civil_twilight_sun_altitude_deg: -0.833, // above sunset → invalid
            nautical_twilight_sun_altitude_deg: -12.0,
            astronomical_twilight_sun_altitude_deg: -18.0,
        }),
        explicit_anchor_utc: None,
    };
    let config = PlanningConfig {
        observatories: vec![bad],
        ..PlanningConfig::default()
    };
    let err = ctx
        .compute_visibility(
            &apophis(),
            None,
            Epoch::from_mjd_tdb(T0_MJD),
            Epoch::from_mjd_tdb(T0_MJD + 1.0),
            0.05,
            &config,
        )
        .expect_err("a non-descending twilight set must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("F51"),
        "the refusal names the offending observatory, got: {msg}"
    );
}
