//! End-to-end observation-noise-model surface tests through the safe Rust
//! wrapper.
//!
//! These fit a real arc, so they need the ephemeris kernels
//! (`EMPYREAN_DATA_DIR` or `~/.empyrean/data`) and reuse the C-ABI test
//! fixture `433_eros.psv`; each test skips cleanly when the data directory
//! is absent, matching the other kernel-gated suites in this crate.
//!
//! The Gaussian references the calibration columns are checked against are
//! the χ²₂ degenerations: median `2 ln 2 ≈ 1.3863` and tail
//! `e^{-4.5} ≈ 0.0111`.

use std::path::PathBuf;

use empyrean::{
    Context, CovarianceInformation, DeliveredNoiseModel, DeliveredRejectionKind,
    DetermineFailureKind, NoiseModel, ODConfig, ODError, Observations, RejectionConfig,
    RejectionKind,
};

const GAUSSIAN_MEDIAN_D2: f64 = 1.386_294_361_119_890_6; // 2 ln 2
const GAUSSIAN_TAIL: f64 = 0.011_108_996_538_242_306; // e^-4.5

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

fn eros(ctx: &Context) -> Observations {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../empyrean-c/tests/fixtures/433_eros.psv");
    ctx.read_ades(fixture.to_str().expect("fixture path is UTF-8"))
        .expect("read the 433 Eros ADES fixture")
}

/// A base config that fits the fixture quickly and deterministically: the
/// default weighting/debiasing hot path, held fixed so only the noise-model
/// axis varies between the runs a test compares.
fn base_config() -> ODConfig {
    ODConfig::default()
}

/// Fit the fixture and return the single delivered object's result, or the
/// failure message when it did not deliver.
fn fit(ctx: &Context, cfg: &ODConfig) -> Result<empyrean::DetermineResult, String> {
    let obs = eros(ctx);
    let results = ctx.determine(&obs, None, cfg).expect("determine call");
    let entry = results.iter().next().expect("one object in the fixture");
    entry.outcome.clone().map_err(|f| f.message)
}

#[test]
fn default_law_is_bit_identical_to_explicit_gaussian() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    let defaulted = fit(&ctx, &base_config()).expect("default-law fit delivers");

    let mut explicit = base_config();
    explicit.weighting.noise_model = NoiseModel::Gaussian;
    let explicit = fit(&ctx, &explicit).expect("explicit-Gaussian fit delivers");

    // The law is the only thing that changed, and it is the same law, so
    // the fit is bit-identical — state and posterior covariance alike.
    assert_eq!(
        defaulted.covariance, explicit.covariance,
        "covariance must be bit-identical between defaulted and explicit Gaussian"
    );
    assert_eq!(
        defaulted.state().position,
        explicit.state().position,
        "fitted position must be bit-identical"
    );
    assert_eq!(
        defaulted.state().velocity,
        explicit.state().velocity,
        "fitted velocity must be bit-identical"
    );
    // Under the Gaussian law every kept row is fully trusted.
    for r in defaulted.residuals.iter().filter(|r| r.selected) {
        assert_eq!(
            r.noise_model,
            DeliveredNoiseModel::Known(NoiseModel::Gaussian)
        );
        assert_eq!(r.robust_weight, 1.0);
    }
}

#[test]
fn student_t_with_blunder_gate_reports_the_law_and_calibration() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    let mut cfg = base_config();
    cfg.weighting.noise_model = NoiseModel::StudentT { nu: 4.0 };
    cfg.rejection = RejectionConfig {
        enabled: Some(true),
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    let f = fit(&ctx, &cfg).expect("Student-t + blunder fit delivers");

    let selected: Vec<_> = f.residuals.iter().filter(|r| r.selected).collect();
    assert!(!selected.is_empty(), "some rows were selected");
    for r in &selected {
        assert_eq!(
            r.noise_model,
            DeliveredNoiseModel::Known(NoiseModel::StudentT { nu: 4.0 }),
            "every fitted row reports the Student-t law"
        );
        assert!(
            r.robust_weight > 0.0 && r.robust_weight <= 1.0,
            "robust_weight in (0, 1], got {}",
            r.robust_weight
        );
        assert!(
            r.sigma_eff_ra_arcsec + 1e-12 >= r.sigma_ra_used_arcsec,
            "sigma_eff ({}) >= sigma_used ({})",
            r.sigma_eff_ra_arcsec,
            r.sigma_ra_used_arcsec
        );
    }
    // At least one row is genuinely down-weighted, so n_eff is a real
    // effective count below the selected total.
    assert!(
        f.summary.n_eff > 0.0 && f.summary.n_eff <= f.summary.num_selected as f64 + 1e-9,
        "n_eff in (0, num_selected], got {}",
        f.summary.n_eff
    );
    // The calibration references are the law's, not the Gaussian χ²₂ ones.
    assert!(f.summary.expected_median_d2.is_finite());
    assert!(f.summary.expected_tail.is_finite());
    assert!(
        (f.summary.expected_median_d2 - GAUSSIAN_MEDIAN_D2).abs() > 1e-6,
        "Student-t expected_median_d2 must differ from the Gaussian 2 ln 2"
    );
    assert!(
        (f.summary.expected_tail - GAUSSIAN_TAIL).abs() > 1e-6,
        "Student-t expected_tail must differ from the Gaussian e^-4.5"
    );
    assert_eq!(
        f.rejection_kind,
        DeliveredRejectionKind::Known(RejectionKind::Blunder)
    );
    assert!(f.rejection_enabled);
}

#[test]
fn student_t_with_rejection_unset_is_refused() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    // A non-Gaussian law with rejection left at the engine default (None)
    // is refused rather than silently chosen for. This refusal is a
    // fit-time ODError (DetermineError::OD(NoiseModelRejectionUnset)), so
    // it surfaces as a per-object DetermineFailure carrying the specific
    // OD-error variant — NOT a config-build error (the nu refusals below
    // take that other path).
    let mut unset = base_config();
    unset.weighting.noise_model = NoiseModel::StudentT { nu: 4.0 };
    unset.rejection = RejectionConfig {
        enabled: None,
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    let obs = eros(&ctx);
    let results = ctx.determine(&obs, None, &unset).expect("determine call");
    let entry = results.iter().next().expect("one object in the fixture");
    let failure = entry
        .failure()
        .expect("the object must fail under rejection-unset");
    assert_eq!(failure.kind, DetermineFailureKind::OD);
    assert_eq!(
        failure.od_error,
        Some(ODError::NoiseModelRejectionUnset),
        "the specific OD error must reach the wrapper, got {:?}",
        failure.od_error
    );

    // The same law with rejection declared on does deliver — proving the
    // refusal is specifically about the unset pairing, not the law.
    let mut declared = base_config();
    declared.weighting.noise_model = NoiseModel::StudentT { nu: 4.0 };
    declared.rejection = RejectionConfig {
        enabled: Some(true),
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    assert!(
        fit(&ctx, &declared).is_ok(),
        "the same law with rejection declared delivers"
    );
}

#[test]
fn a_per_station_rule_reports_its_law_on_that_station_alone() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    // Pick a station that appears in the fixture.
    let obs = eros(&ctx);
    let Some(target) = obs.iter().find_map(|o| {
        let c = o.obs_code.clone();
        (!c.trim().is_empty()).then_some(c)
    }) else {
        eprintln!("skipping: fixture carries no station code");
        return;
    };

    let mut cfg = base_config();
    cfg.weighting.noise_model = NoiseModel::Gaussian;
    cfg.weighting.additional_layers = vec![
        empyrean::WeightingLayer::ObservatoryRule {
            obs_code: target.clone(),
            sigma: [0.5, 0.5],
            start_epoch: None,
            end_epoch: None,
            scale: 1.0,
            noise_model: Some(NoiseModel::StudentT { nu: 4.0 }),
        },
        empyrean::WeightingLayer::NightlyDeweighting { max_gap_days: 0.5 },
    ];
    cfg.rejection = RejectionConfig {
        enabled: Some(true),
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    let f = fit(&ctx, &cfg).expect("per-station-law fit delivers");

    let mut saw_student_t = false;
    for r in f.residuals.iter().filter(|r| r.selected) {
        if r.obs_code == target {
            assert_eq!(
                r.noise_model,
                DeliveredNoiseModel::Known(NoiseModel::StudentT { nu: 4.0 }),
                "the target station's rows report the per-station law"
            );
            saw_student_t = true;
        } else {
            assert_eq!(
                r.noise_model,
                DeliveredNoiseModel::Known(NoiseModel::Gaussian),
                "every other station keeps the global Gaussian law"
            );
        }
    }
    assert!(
        saw_student_t,
        "the target station contributed selected rows"
    );
}

#[test]
fn expected_information_is_accepted_under_a_student_t_law() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    let mut cfg = base_config();
    cfg.weighting.noise_model = NoiseModel::StudentT { nu: 4.0 };
    cfg.weighting.covariance_information = CovarianceInformation::Expected;
    cfg.rejection = RejectionConfig {
        enabled: Some(true),
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    let f = fit(&ctx, &cfg).expect("expected-information Student-t fit delivers");
    // A real posterior came back (finite diagonal).
    assert!(
        f.covariance[0][0].is_finite() && f.covariance[0][0] > 0.0,
        "expected-information posterior covariance is populated"
    );
    for r in f.residuals.iter().filter(|r| r.selected) {
        assert_eq!(
            r.noise_model,
            DeliveredNoiseModel::Known(NoiseModel::StudentT { nu: 4.0 })
        );
    }
}

#[test]
fn nu_at_or_below_two_is_refused_at_the_boundary() {
    let Some(ctx) = try_context() else {
        eprintln!("skipping: no ephemeris data directory");
        return;
    };
    // ν ≤ 2 has no finite variance and is refused by name at the FFI
    // boundary — the determine call itself errors.
    let mut cfg = base_config();
    cfg.weighting.noise_model = NoiseModel::StudentT { nu: 2.0 };
    cfg.rejection = RejectionConfig {
        enabled: Some(true),
        kind: RejectionKind::Blunder,
        ..RejectionConfig::default()
    };
    let obs = eros(&ctx);
    let err = ctx
        .determine(&obs, None, &cfg)
        .expect_err("nu <= 2 must be refused");
    let msg = err.message.to_ascii_lowercase();
    assert!(
        msg.contains("nu") || msg.contains("variance"),
        "the refusal names the degrees-of-freedom problem, got: {}",
        err.message
    );
}
