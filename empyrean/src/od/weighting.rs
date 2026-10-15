//! Per-observation weighting pipeline used by orbit determination.
//!
//! The default chain is the production hot path: VFCC2017 station floors
//! (Vereš, Farnocchia, Chesley & Chamberlin 2017) plus a nightly de-weighting
//! layer (1/√N within 0.5 days). Set `enabled = false` for uniform 1″
//! weighting, pick a different preset, or append layers for custom
//! pipelines.

/// How a weighting layer's `sigma` combines with a per-observation
/// reported `sigma`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SigmaPolicy {
    /// Default — preserves the reported per-obs σ as the source of
    /// truth; rule's σ used only when the observation has no
    /// reported value.
    #[default]
    DefaultOnly,
    /// `σ = max(reported, rule)` — rule acts as a noise floor.
    /// VFCC2017 / NEODyS production policy.
    Floor,
}

/// The observation-error **law** — the shape axis of the error model.
///
/// [`Gaussian`](Self::Gaussian) is the default, under which the fit is
/// bit-identical to the pre-noise-model engine: no iteratively-reweighted
/// least-squares pass runs and every per-row `robust_weight` is `1.0`.
/// [`StudentT`](Self::StudentT) is the heavy-tailed law fitted by
/// iteratively-reweighted least squares — a residual at squared distance
/// \\(\delta^2\\) carries robust weight \\(\nu/(\nu+\delta^2) \in (0,1]\\),
/// so a large residual is down-weighted rather than removed, and is best
/// paired with [`RejectionKind::Blunder`](super::rejection::RejectionKind::Blunder).
///
/// \\(\nu \le 2\\) has no finite variance and is refused by the engine at
/// the FFI boundary; a stray `nu` on [`Gaussian`](Self::Gaussian) is
/// likewise refused rather than dropped.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NoiseModel {
    /// The classical Gaussian (normal) error law. Default.
    #[default]
    Gaussian,
    /// A Student-t error law with \\(\nu > 2\\) degrees of freedom.
    StudentT {
        /// Degrees of freedom \\(\nu > 2\\).
        nu: f64,
    },
    /// A per-station-night latent-scale mixture with \\(\nu > 2\\). Accepted
    /// by the config surface so the type is stable, but a fit configured
    /// with it is refused by the engine at fit time (its calibration is
    /// downstream work) — use [`StudentT`](Self::StudentT) with the nightly
    /// de-weighting layer until then.
    NightScale {
        /// Degrees of freedom \\(\nu > 2\\).
        nu: f64,
    },
}

impl NoiseModel {
    /// Pack into the C-ABI tagged value. The Gaussian law carries no ν
    /// (0.0, which the engine reads as unset).
    pub(crate) fn to_ffi(self) -> empyrean_sys::EmpyreanNoiseModel {
        let (tag, nu) = match self {
            NoiseModel::Gaussian => (empyrean_sys::EMPYREAN_NOISE_MODEL_GAUSSIAN as u8, 0.0),
            NoiseModel::StudentT { nu } => (empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T as u8, nu),
            NoiseModel::NightScale { nu } => {
                (empyrean_sys::EMPYREAN_NOISE_MODEL_NIGHT_SCALE as u8, nu)
            }
        };
        empyrean_sys::EmpyreanNoiseModel { tag, nu }
    }
}

/// Which information matrix the fit inverts for the posterior covariance
/// under a non-Gaussian [`NoiseModel`]. Both reduce to the Gauss–Newton
/// normal matrix under [`NoiseModel::Gaussian`], so this only bites for a
/// heavy-tailed law.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CovarianceInformation {
    /// **Observed** information — the negative Hessian of the robust
    /// log-likelihood at the solution (Efron & Hinkley 1978); the default.
    /// Conditions on which rows this fit down-weighted; can be
    /// non-positive-definite, which the engine refuses by name.
    #[default]
    Observed,
    /// **Expected** (Fisher) information — a scalar multiple of the Gaussian
    /// normal matrix, always positive definite when that is, but blind to
    /// which rows were down-weighted.
    Expected,
}

impl CovarianceInformation {
    pub(super) fn to_ffi(self) -> u8 {
        match self {
            CovarianceInformation::Observed => {
                empyrean_sys::EMPYREAN_COVARIANCE_INFORMATION_OBSERVED as u8
            }
            CovarianceInformation::Expected => {
                empyrean_sys::EMPYREAN_COVARIANCE_INFORMATION_EXPECTED as u8
            }
        }
    }
}

/// One element of the weighting pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightingLayer {
    /// Assign default sigma (and optional scale + time range) for a
    /// specific MPC observatory code.
    ObservatoryRule {
        /// MPC observatory code (e.g. "F51").
        obs_code: String,
        /// 1σ (RA·cos(δ), Dec) in arcseconds.
        sigma: [f64; 2],
        /// Start of applicable time range (MJD TDB). `None` = unbounded.
        start_epoch_mjd_tdb: Option<f64>,
        /// End of applicable time range (MJD TDB). `None` = unbounded.
        end_epoch_mjd_tdb: Option<f64>,
        /// Scale factor on the final weight. Default 1.0.
        scale: f64,
        /// Per-station error **law**, or `None` to inherit the global
        /// [`WeightingConfig::noise_model`]. Resolved by first match on
        /// `obs_code` + time range, on a pass separate from σ resolution.
        noise_model: Option<NoiseModel>,
    },
    /// Down-weight correlated observations from the same observing
    /// night (same station within `max_gap_days`).
    NightlyDeweighting {
        /// Maximum gap (days) between observations to count as the
        /// same night. Default 0.5.
        max_gap_days: f64,
    },
}

/// Preset selector for [`WeightingConfig`]. Picking a preset seeds the
/// layer chain with curated layers; entries in
/// [`WeightingConfig::additional_layers`] are placed ahead of the
/// preset's rules (sigma resolution is first-match-wins, so they
/// override the preset for their stations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightingPreset {
    /// No preset — only `additional_layers` apply.
    None,
    /// VFCC2017 — Vereš, Farnocchia, Chesley & Chamberlin 2017 station floors
    /// + nightly de-weighting at floor-σ policy. Production default.
    #[default]
    VFCC2017,
    /// NEODyS production preset.
    Neodys,
}

/// Observation weighting pipeline.
///
/// Default = `enabled` + [`WeightingPreset::VFCC2017`] + a
/// [`WeightingLayer::NightlyDeweighting`] layer (the production
/// combination — VFCC2017 station floors with 1/√N nightly down-weighting
/// chained on top).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightingConfig {
    /// `true` → run weighting (default), `false` → uniform 1″.
    pub enabled: bool,
    /// Preset selector. Default [`WeightingPreset::VFCC2017`].
    pub preset: WeightingPreset,
    /// Default 1σ (arcsec) when no rule applies. Used only when
    /// `preset = None`. Default 1.0.
    pub default_sigma_arcsec: f64,
    /// Sigma combination policy override. `None` = use the preset's policy.
    pub sigma_policy: Option<SigmaPolicy>,
    /// Layers placed ahead of the preset's chain — first-match-wins,
    /// so they override preset rules for their stations; the preset
    /// is the fallback.
    ///
    /// NOTE: assigning this field REPLACES the default
    /// `[NightlyDeweighting]` list — include a
    /// [`WeightingLayer::NightlyDeweighting`] entry explicitly to
    /// keep the production per-night 1/√N de-weighting. At most one
    /// nightly layer is accepted per chain.
    pub additional_layers: Vec<WeightingLayer>,
    /// The global observation-error **law** for every row a per-station
    /// [`WeightingLayer::ObservatoryRule`] does not override. Default
    /// [`NoiseModel::Gaussian`], under which the fit is bit-identical to the
    /// pre-noise-model engine.
    pub noise_model: NoiseModel,
    /// Which information matrix the fit inverts for the posterior covariance
    /// under a non-Gaussian [`noise_model`](Self::noise_model). Default
    /// [`CovarianceInformation::Observed`]; irrelevant under the Gaussian
    /// law.
    pub covariance_information: CovarianceInformation,
}

impl Default for WeightingConfig {
    fn default() -> Self {
        // VFCC2017 station floors WITH a NightlyDeweighting layer appended.
        // The preset alone is just the per-station σ table; nightly
        // de-weighting (1/√N within 0.5 days) is a separate layer that
        // production defaults chain on top, and the FFI surface has to
        // ship the same combination or determine() fits diverge from
        // the direct-engine path on objects with same-night-same-station
        // observation clusters.
        Self {
            enabled: true,
            preset: WeightingPreset::VFCC2017,
            default_sigma_arcsec: 1.0,
            sigma_policy: None,
            additional_layers: vec![WeightingLayer::NightlyDeweighting { max_gap_days: 0.5 }],
            noise_model: NoiseModel::Gaussian,
            covariance_information: CovarianceInformation::Observed,
        }
    }
}

/// Convert a [`WeightingLayer`] into its C-ABI tagged-union mirror.
///
/// `idx` is the layer's position in
/// [`WeightingConfig::additional_layers`], used to name the offending
/// layer in validation errors.
///
/// The `obs_code` is validated against the C-ABI field before packing:
/// codes longer than 4 bytes, non-ASCII codes, and codes containing
/// whitespace are rejected (station matching is exact and
/// case-sensitive engine-side — a silently truncated or trimmed code
/// would match the wrong station, or none).
pub(super) fn weighting_layer_to_ffi(
    idx: usize,
    layer: &WeightingLayer,
) -> crate::error::Result<empyrean_sys::EmpyreanWeightingLayer> {
    let mut ffi = empyrean_sys::EmpyreanWeightingLayer {
        kind: 0,
        obs_code: [0u8; 4],
        sigma_ra_arcsec: 0.0,
        sigma_dec_arcsec: 0.0,
        start_epoch_mjd_tdb: f64::NAN,
        end_epoch_mjd_tdb: f64::NAN,
        scale: 0.0,
        max_gap_days: 0.0,
        has_noise_model: 0,
        noise_model: NoiseModel::Gaussian.to_ffi(),
    };
    match layer {
        WeightingLayer::ObservatoryRule {
            obs_code,
            sigma,
            start_epoch_mjd_tdb,
            end_epoch_mjd_tdb,
            scale,
            noise_model,
        } => {
            let bytes = obs_code.as_bytes();
            if bytes.len() > 4 {
                return Err(crate::error::Error::invalid_input(format!(
                    "weighting.additional_layers[{idx}]: obs_code {obs_code:?} is longer than \
                     the 4-byte MPC station field; truncating would match the wrong station"
                )));
            }
            if obs_code.is_empty() {
                return Err(crate::error::Error::invalid_input(format!(
                    "weighting.additional_layers[{idx}]: ObservatoryRule has empty obs_code"
                )));
            }
            if !obs_code.chars().all(|ch| ch.is_ascii_graphic()) {
                return Err(crate::error::Error::invalid_input(format!(
                    "weighting.additional_layers[{idx}]: obs_code {obs_code:?} must be \
                     printable ASCII with no whitespace — MPC station matching is exact and \
                     case-sensitive"
                )));
            }
            ffi.kind = empyrean_sys::EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE as i32;
            for (i, &b) in bytes.iter().enumerate() {
                ffi.obs_code[i] = b;
            }
            ffi.sigma_ra_arcsec = sigma[0];
            ffi.sigma_dec_arcsec = sigma[1];
            ffi.start_epoch_mjd_tdb = start_epoch_mjd_tdb.unwrap_or(f64::NAN);
            ffi.end_epoch_mjd_tdb = end_epoch_mjd_tdb.unwrap_or(f64::NAN);
            ffi.scale = *scale;
            // Per-station error law: a presence flag guards it so a rule
            // that sets no law INHERITS the global one rather than forcing
            // Gaussian on its station.
            if let Some(nm) = noise_model {
                ffi.has_noise_model = 1;
                ffi.noise_model = nm.to_ffi();
            }
        }
        WeightingLayer::NightlyDeweighting { max_gap_days } => {
            ffi.kind = empyrean_sys::EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING as i32;
            ffi.max_gap_days = *max_gap_days;
        }
    }
    Ok(ffi)
}

#[cfg(test)]
mod weighting_default_tests {
    use super::{WeightingConfig, WeightingLayer, WeightingPreset};

    /// The wrapper's OD weighting default mirrors the production default the C
    /// surface reports: VFCC2017 station floors WITH a nightly-deweighting
    /// layer, enabled, a 1″ fallback σ.
    ///
    /// The wrapper has no engine access, so this is a wrapper-vs-C comparison
    /// rather than wrapper-vs-engine: it pins the wrapper-side default against
    /// the same production default the C ruling test pins engine-side (that
    /// test compares the C preset default against
    /// `empyrean_core::determination::WeightingConfig::default()`). The two
    /// must agree, or `determine()` diverges from the direct-engine path on
    /// same-night / same-station observation clusters.
    #[test]
    fn the_default_weighting_mirrors_the_production_default() {
        let d = WeightingConfig::default();
        assert!(d.enabled, "weighting is on by default");
        assert_eq!(d.preset, WeightingPreset::VFCC2017);
        assert_eq!(d.default_sigma_arcsec, 1.0);
        assert_eq!(
            d.additional_layers,
            vec![WeightingLayer::NightlyDeweighting { max_gap_days: 0.5 }],
            "production chains nightly de-weighting on top of the preset"
        );
    }
}

#[cfg(test)]
mod noise_model_tests {
    use super::*;

    #[test]
    fn noise_model_packs_tag_and_nu() {
        let g = NoiseModel::Gaussian.to_ffi();
        assert_eq!(g.tag as u32, empyrean_sys::EMPYREAN_NOISE_MODEL_GAUSSIAN);
        assert_eq!(g.nu, 0.0, "the Gaussian law carries no nu");

        let t = NoiseModel::StudentT { nu: 4.0 }.to_ffi();
        assert_eq!(t.tag as u32, empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T);
        assert_eq!(t.nu, 4.0);

        let n = NoiseModel::NightScale { nu: 6.0 }.to_ffi();
        assert_eq!(n.tag as u32, empyrean_sys::EMPYREAN_NOISE_MODEL_NIGHT_SCALE);
        assert_eq!(n.nu, 6.0);
    }

    #[test]
    fn covariance_information_packs_its_selector() {
        assert_eq!(
            CovarianceInformation::Observed.to_ffi() as u32,
            empyrean_sys::EMPYREAN_COVARIANCE_INFORMATION_OBSERVED
        );
        assert_eq!(
            CovarianceInformation::Expected.to_ffi() as u32,
            empyrean_sys::EMPYREAN_COVARIANCE_INFORMATION_EXPECTED
        );
    }

    #[test]
    fn a_law_only_station_rule_sets_the_presence_flag() {
        let layer = WeightingLayer::ObservatoryRule {
            obs_code: "F51".to_string(),
            sigma: [0.2, 0.2],
            start_epoch_mjd_tdb: None,
            end_epoch_mjd_tdb: None,
            scale: 1.0,
            noise_model: Some(NoiseModel::StudentT { nu: 4.0 }),
        };
        let ffi = weighting_layer_to_ffi(0, &layer).unwrap();
        assert_eq!(ffi.has_noise_model, 1);
        assert_eq!(
            ffi.noise_model.tag as u32,
            empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T
        );
        assert_eq!(ffi.noise_model.nu, 4.0);

        // A rule with no per-station law inherits the global one: the
        // presence flag stays 0 rather than forcing Gaussian.
        let inherit = WeightingLayer::ObservatoryRule {
            obs_code: "F51".to_string(),
            sigma: [0.2, 0.2],
            start_epoch_mjd_tdb: None,
            end_epoch_mjd_tdb: None,
            scale: 1.0,
            noise_model: None,
        };
        assert_eq!(
            weighting_layer_to_ffi(0, &inherit).unwrap().has_noise_model,
            0
        );
    }
}
