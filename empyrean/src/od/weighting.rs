//! Per-observation weighting pipeline used by orbit determination.
//!
//! The default chain is the production hot path: VFC17 station floors
//! (Vereš, Farnocchia, Chesley et al. 2017) plus a nightly de-weighting
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
    /// VFC17 / NEODyS production policy.
    Floor,
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
/// [`WeightingConfig::additional_layers`] are appended in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightingPreset {
    /// No preset — only `additional_layers` apply.
    None,
    /// VFC17 — Vereš, Farnocchia, Chesley et al. 2017 station floors
    /// + nightly de-weighting at floor-σ policy. Production default.
    #[default]
    Vfc17,
    /// NEODyS production preset.
    Neodys,
}

/// Observation weighting pipeline.
///
/// Default = `enabled` + [`WeightingPreset::Vfc17`] + a
/// [`WeightingLayer::NightlyDeweighting`] layer (the production
/// combination — VFC17 station floors with 1/√N nightly down-weighting
/// chained on top).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightingConfig {
    /// `true` → run weighting (default), `false` → uniform 1″.
    pub enabled: bool,
    /// Preset selector. Default [`WeightingPreset::Vfc17`].
    pub preset: WeightingPreset,
    /// Default 1σ (arcsec) when no rule applies. Used only when
    /// `preset = None`. Default 1.0.
    pub default_sigma_arcsec: f64,
    /// Sigma combination policy override. `None` = use the preset's policy.
    pub sigma_policy: Option<SigmaPolicy>,
    /// Layers appended to the preset's chain.
    pub additional_layers: Vec<WeightingLayer>,
}

impl Default for WeightingConfig {
    fn default() -> Self {
        // VFC17 station floors WITH a NightlyDeweighting layer appended.
        // The preset alone is just the per-station σ table; nightly
        // de-weighting (1/√N within 0.5 days) is a separate layer that
        // production defaults chain on top, and the FFI surface has to
        // ship the same combination or determine() fits diverge from
        // the direct-engine path on objects with same-night-same-station
        // observation clusters.
        Self {
            enabled: true,
            preset: WeightingPreset::Vfc17,
            default_sigma_arcsec: 1.0,
            sigma_policy: None,
            additional_layers: vec![WeightingLayer::NightlyDeweighting { max_gap_days: 0.5 }],
        }
    }
}

/// Convert a [`WeightingLayer`] into its C-ABI tagged-union mirror.
pub(super) fn weighting_layer_to_ffi(
    layer: &WeightingLayer,
) -> empyrean_sys::EmpyreanWeightingLayer {
    let mut ffi = empyrean_sys::EmpyreanWeightingLayer {
        kind: 0,
        obs_code: [0u8; 4],
        sigma_ra_arcsec: 0.0,
        sigma_dec_arcsec: 0.0,
        start_epoch_mjd_tdb: f64::NAN,
        end_epoch_mjd_tdb: f64::NAN,
        scale: 0.0,
        max_gap_days: 0.0,
    };
    match layer {
        WeightingLayer::ObservatoryRule {
            obs_code,
            sigma,
            start_epoch_mjd_tdb,
            end_epoch_mjd_tdb,
            scale,
        } => {
            ffi.kind = empyrean_sys::EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE as i32;
            let bytes = obs_code.as_bytes();
            for (i, &b) in bytes.iter().enumerate().take(4) {
                ffi.obs_code[i] = b;
            }
            ffi.sigma_ra_arcsec = sigma[0];
            ffi.sigma_dec_arcsec = sigma[1];
            ffi.start_epoch_mjd_tdb = start_epoch_mjd_tdb.unwrap_or(f64::NAN);
            ffi.end_epoch_mjd_tdb = end_epoch_mjd_tdb.unwrap_or(f64::NAN);
            ffi.scale = *scale;
        }
        WeightingLayer::NightlyDeweighting { max_gap_days } => {
            ffi.kind = empyrean_sys::EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING as i32;
            ffi.max_gap_days = *max_gap_days;
        }
    }
    ffi
}
