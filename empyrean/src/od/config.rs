//! Top-level [`ODConfig`] plus the small config bundles unique to it
//! (initial-orbit-determination tuning, auto-escalation policy,
//! acceptability thresholds).
//!
//! The weighting / debiasing / rejection / per-station nuisance
//! sub-configs live in their own sibling modules — see
//! [`super::weighting`], [`super::debiasing`], [`super::rejection`],
//! [`super::nuisance`].

pub use crate::propagate::ForceModelTier;

use super::debiasing::{DebiasingConfig, DebiasingResolution};
use super::nuisance::StationRaDecConfig;
use super::rejection::{RejectionConfig, RejectionKind};
use super::result::{OriginPolicy, OutputEpoch, SolveForParams};
use super::weighting::{SigmaPolicy, WeightingConfig, WeightingPreset, weighting_layer_to_ffi};

/// IOD ranging tuning.
///
/// Sentinel rule: every numeric field uses `0` / `0.0` for "engine
/// default"; `opposition_gap_days < 0.0` disables opposition splitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IODConfig {
    /// Maximum Gauss triplet attempts. 0 → engine default (10).
    pub max_triplet_attempts: u32,
    /// Maximum time span (days) within a Gauss triplet. 0.0 → engine default (30).
    pub max_triplet_span_days: f64,
    /// Minimum gap (days) for splitting an arc into oppositions.
    /// `< 0.0` disables splitting; `0.0` → engine default (90).
    pub opposition_gap_days: f64,
    /// Maximum arc length (days) used for IOD. 0.0 → engine default (30).
    pub max_iod_arc_days: f64,
    /// Curvature S/N threshold. 0.0 → engine default.
    pub curvature_snr_threshold: f64,
    /// Maximum σₐ/|a| from the IOD two-body covariance above which the
    /// IOD result is rejected as poorly constrained. 0.0 → engine
    /// default.
    pub max_iod_fractional_sigma_a: f64,
}

impl Default for IODConfig {
    fn default() -> Self {
        Self {
            max_triplet_attempts: 10,
            max_triplet_span_days: 30.0,
            opposition_gap_days: 90.0,
            max_iod_arc_days: 30.0,
            curvature_snr_threshold: 3.0,
            max_iod_fractional_sigma_a: 1.0,
        }
    }
}

/// Tuning for [`SolveForParams::Auto`] automatic escalation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoEscalationPolicy {
    /// Reduced-χ² tuning parameter.
    pub reduced_chi2: f64,
    /// Along/cross-track residual-ratio tuning parameter.
    pub at_ct_ratio: f64,
    /// Minimum arc length (days).
    pub min_arc_days: f64,
    /// Minimum observation count.
    pub min_n_obs: u32,
}

impl Default for AutoEscalationPolicy {
    fn default() -> Self {
        Self {
            reduced_chi2: 10.0,
            at_ct_ratio: 3.0,
            min_arc_days: 30.0,
            min_n_obs: 50,
        }
    }
}

/// Thresholds for the post-DC fit-acceptability sub-checks.
///
/// Defaults are tuned for production NEO survey work; tighten for
/// impact-monitoring orbits, loosen
/// for short-arc discovery fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptabilityThresholds {
    /// Maximum acceptable reduced χ².
    pub reduced_chi2: f64,
    /// Maximum acceptable RMS (arcsec).
    pub rms_arcsec: f64,
    /// Maximum acceptable AT/CT residual ratio.
    pub at_ct_ratio: f64,
    /// Minimum acceptable arc length (days).
    pub min_arc_days: f64,
    /// Maximum acceptable σₐ / |a|.
    pub fractional_sigma_a: f64,
}

impl Default for AcceptabilityThresholds {
    fn default() -> Self {
        Self {
            reduced_chi2: 3.0,
            rms_arcsec: 1.0,
            at_ct_ratio: 3.0,
            min_arc_days: 7.0,
            fractional_sigma_a: 0.1,
        }
    }
}

/// Orbit-determination configuration.
///
/// Shared knobs at the top, [`iod`](Self::iod) for IOD ranging, with
/// `output_epoch` / `auto_escalation` / `acceptability` / `rejection`
/// / `station_radec` as nested config bundles.
///
/// # Sentinel rule (footgun)
///
/// Numeric fields here are FFI sentinels — `0` / `0.0` does **not**
/// mean "tightest possible". It means "use the engine default" (e.g.
/// `convergence_tol = 0.0` resolves to `1e-5`, not 0; `epsilon = 0.0`
/// resolves to `1e-9`). To force a specific value, set it explicitly
/// — including the engine default if you want to lock that in
/// against a future default change.
///
/// Documented negative-value sentinels
/// (`iod.opposition_gap_days < 0`, `rejection.lambda < 0`) retain
/// their disable-this-feature meanings.
#[derive(Debug, Clone, PartialEq)]
pub struct ODConfig {
    // ── Shared (all OD entry points) ────────────────────────────────
    /// Force-model tier.
    pub force_model: ForceModelTier,
    /// Integrator truncation-error tolerance, interpreted by the
    /// active integrator backend:
    ///
    /// - [`IntegratorChoice::GR15`](crate::IntegratorChoice::GR15)
    ///   (default): relative b₆ truncation tolerance.
    /// - [`IntegratorChoice::DOP853`](crate::IntegratorChoice::DOP853):
    ///   relative tolerance, paired with a fixed `atol = 1e-14`.
    ///
    /// `0.0` → engine default (1e-9). The OD pipeline uses whatever
    /// integrator the underlying propagation config selects; if you
    /// switch backends, re-validate against your accuracy budget.
    pub epsilon: f64,
    /// Maximum light-time iterations. 0 → engine default (3).
    pub max_light_time_iterations: usize,
    /// Threads for batch operations. 0 → all available cores.
    pub num_threads: usize,
    /// Output reference frame.
    pub frame: crate::coordinate::Frame,
    /// Observation weighting pipeline. Default = `enabled` + VFC17
    /// preset (production hot path).
    pub weighting: WeightingConfig,
    /// Catalog-bias-correction configuration. Default = `enabled` +
    /// EFCC2020 standard resolution (production hot path).
    pub debiasing: DebiasingConfig,
    /// Bodies to exclude from the perturber set (for self-determination
    /// of perturbed asteroids — e.g. fitting Eros while excluding
    /// [`Origin::asteroid(433)`](crate::Origin::asteroid)). Empty = use all.
    pub excluded_perturbers: Vec<crate::coordinate::Origin>,
    /// Origin-policy selector. Default [`OriginPolicy::Auto`]
    /// (heliocentric → geocentric Earth cascade). Set to
    /// [`OriginPolicy::Explicit(Origin::Earth)`](OriginPolicy::Explicit)
    /// to pin the pipeline to a specific central body for catalog
    /// satellites or regime-classified workflows.
    pub origin: OriginPolicy,

    // ── IOD (determine only) ────────────────────────────────────────
    /// Initial orbit determination tuning.
    pub iod: IODConfig,

    // ── Differential correction ─────────────────────────────────────
    /// Output epoch policy for the fitted orbit.
    pub output_epoch: OutputEpoch,
    /// Maximum DC iterations. 0 → engine default (100).
    pub max_iterations: u32,
    /// DC convergence tolerance on Δx^T N Δx. 0.0 → engine default (1e-5).
    pub convergence_tol: f64,
    /// Use STM-cached ephemeris updates for iterations 2+. Default `true`.
    pub use_stm_cache: bool,
    /// Solve-for parameter set.
    pub solve_for: SolveForParams,
    /// Trigger thresholds for [`SolveForParams::Auto`] escalation.
    pub auto_escalation: AutoEscalationPolicy,
    /// Thresholds for the post-DC acceptability gate.
    pub acceptability: AcceptabilityThresholds,
    /// Schur-eliminate per-station RA/Dec biases.
    pub fit_station_biases: bool,
    /// Per-station RA/Dec bias config. Honored only when
    /// `fit_station_biases` is true.
    pub station_radec: StationRaDecConfig,
    /// Use span-grouped Jacobian reuse.
    pub use_span_grouping: bool,

    // ── Rejection ──────────────────────────────────────────────────
    /// Outlier-rejection strategy + tuning.
    pub rejection: RejectionConfig,
    /// Auto-select force-model tier from IOD elements.
    pub auto_force_model: bool,
    /// Output coordinate representation for the fitted orbit + covariance.
    pub output_representation: crate::coordinate::Representation,
}

impl Default for ODConfig {
    fn default() -> Self {
        // Production hot path:
        //   - VFC17 station weighting + nightly de-weighting on
        //   - EFCC2020 catalog debiasing on (if `bias.dat` is on disk)
        //   - SolveForParams::Auto (escalates 6→9 params on poor fit)
        //   - Adaptive rejection enabled
        Self {
            force_model: ForceModelTier::Standard,
            epsilon: 1e-9,
            max_light_time_iterations: 3,
            num_threads: 0,
            frame: crate::coordinate::Frame::ICRF,
            weighting: WeightingConfig::default(),
            debiasing: DebiasingConfig::default(),
            excluded_perturbers: Vec::new(),
            origin: OriginPolicy::default(),
            iod: IODConfig::default(),
            output_epoch: OutputEpoch::default(),
            max_iterations: 100,
            // 1e-5 on \\(\Delta\mathbf{x}^\top \mathcal{N} \, \Delta\mathbf{x}\\) —
            // tighter than survey-grade so the FFI path and the direct-engine
            // path agree at sub-meter level rather than the noise floor a
            // looser tol leaves behind.
            convergence_tol: 1e-5,
            use_stm_cache: true,
            solve_for: SolveForParams::Auto,
            auto_escalation: AutoEscalationPolicy::default(),
            acceptability: AcceptabilityThresholds::default(),
            fit_station_biases: false,
            station_radec: StationRaDecConfig::default(),
            use_span_grouping: false,
            rejection: RejectionConfig::default(),
            auto_force_model: false,
            output_representation: crate::coordinate::Representation::Cartesian,
        }
    }
}

/// Keepalive backing storage for [`ODConfig::to_ffi_with`]. The returned
/// FFI struct holds raw pointers into this struct's owned `Vec`s /
/// `CString`; drop the keepalive only after the FFI call has returned.
//
// Fields aren't "read" in Rust — the FFI struct points into them and
// they exist purely for lifetime extension. `#[allow(dead_code)]`
// suppresses the false-positive dead-code warning.
#[allow(dead_code)]
pub(crate) struct ODConfigKeepalive {
    pub perturbers: Vec<i32>,
    pub weighting_layers: Vec<empyrean_sys::EmpyreanWeightingLayer>,
    pub bias_dat_path: Option<std::ffi::CString>,
}

impl ODConfig {
    pub(crate) fn to_ffi_with(&self) -> (empyrean_sys::EmpyreanODConfig, ODConfigKeepalive) {
        let perturbers: Vec<i32> = self
            .excluded_perturbers
            .iter()
            .map(|o| o.naif_id())
            .collect();
        let (mode, explicit) = match self.output_epoch {
            OutputEpoch::MidArc => (0, 0.0),
            OutputEpoch::LastObservation => (1, 0.0),
            OutputEpoch::Epoch(t) => (2, t),
            OutputEpoch::IODEpoch => (3, 0.0),
        };
        // Materialize weighting layers
        let weighting_layers: Vec<empyrean_sys::EmpyreanWeightingLayer> = self
            .weighting
            .additional_layers
            .iter()
            .map(weighting_layer_to_ffi)
            .collect();
        let weighting = empyrean_sys::EmpyreanWeightingConfig {
            enabled: u8::from(self.weighting.enabled),
            preset: match self.weighting.preset {
                WeightingPreset::None => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_NONE as u8,
                WeightingPreset::Vfc17 => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_VFC17 as u8,
                WeightingPreset::Neodys => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_NEODYS as u8,
            },
            default_sigma_arcsec: self.weighting.default_sigma_arcsec,
            sigma_policy: match self.weighting.sigma_policy {
                None => -1,
                Some(SigmaPolicy::DefaultOnly) => {
                    empyrean_sys::EMPYREAN_SIGMA_POLICY_DEFAULT_ONLY as i32
                }
                Some(SigmaPolicy::Floor) => empyrean_sys::EMPYREAN_SIGMA_POLICY_FLOOR as i32,
            },
            additional_layers: if weighting_layers.is_empty() {
                std::ptr::null()
            } else {
                weighting_layers.as_ptr()
            },
            num_additional_layers: weighting_layers.len(),
        };
        // Materialize debiasing config (CString for the optional path).
        let bias_dat_path = self
            .debiasing
            .bias_dat_path
            .as_ref()
            .and_then(|p| std::ffi::CString::new(p.to_string_lossy().as_bytes()).ok());
        let debiasing = empyrean_sys::EmpyreanDebiasingConfig {
            enabled: u8::from(self.debiasing.enabled),
            table_id: empyrean_sys::EMPYREAN_DEBIASING_TABLE_EFCC2020 as i32,
            resolution: match self.debiasing.resolution {
                DebiasingResolution::Standard => {
                    empyrean_sys::EMPYREAN_DEBIASING_RESOLUTION_STANDARD as i32
                }
                DebiasingResolution::Hires => {
                    empyrean_sys::EMPYREAN_DEBIASING_RESOLUTION_HIRES as i32
                }
            },
            bias_dat_path: bias_dat_path
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(std::ptr::null()),
        };
        let cfg = empyrean_sys::EmpyreanODConfig {
            force_model: self.force_model as i32,
            epsilon: self.epsilon,
            max_light_time_iterations: self.max_light_time_iterations,
            num_threads: self.num_threads,
            frame: self.frame as i32,
            weighting,
            debiasing,
            num_excluded_perturbers: perturbers.len(),
            excluded_perturbers_naif: if perturbers.is_empty() {
                std::ptr::null()
            } else {
                perturbers.as_ptr()
            },
            origin: {
                let (policy, explicit_naif) = match self.origin {
                    OriginPolicy::Auto => (empyrean_sys::EMPYREAN_ORIGIN_POLICY_AUTO as i32, 0),
                    OriginPolicy::Explicit(o) => (
                        empyrean_sys::EMPYREAN_ORIGIN_POLICY_EXPLICIT as i32,
                        o.naif_id(),
                    ),
                };
                empyrean_sys::EmpyreanOriginPolicy {
                    policy,
                    explicit_naif,
                }
            },
            iod: empyrean_sys::EmpyreanIODConfig {
                max_triplet_attempts: self.iod.max_triplet_attempts,
                max_triplet_span_days: self.iod.max_triplet_span_days,
                opposition_gap_days: self.iod.opposition_gap_days,
                max_iod_arc_days: self.iod.max_iod_arc_days,
                curvature_snr_threshold: self.iod.curvature_snr_threshold,
                max_iod_fractional_sigma_a: self.iod.max_iod_fractional_sigma_a,
            },
            output_epoch: empyrean_sys::EmpyreanOutputEpoch {
                mode,
                explicit_mjd_tdb: explicit,
            },
            max_iterations: self.max_iterations,
            convergence_tol: self.convergence_tol,
            use_stm_cache: u8::from(self.use_stm_cache),
            solve_for: self.solve_for.to_int(),
            auto_escalation: empyrean_sys::EmpyreanAutoEscalationPolicy {
                reduced_chi2: self.auto_escalation.reduced_chi2,
                at_ct_ratio: self.auto_escalation.at_ct_ratio,
                min_arc_days: self.auto_escalation.min_arc_days,
                min_n_obs: self.auto_escalation.min_n_obs,
            },
            acceptability: empyrean_sys::EmpyreanAcceptabilityThresholds {
                reduced_chi2: self.acceptability.reduced_chi2,
                rms_arcsec: self.acceptability.rms_arcsec,
                at_ct_ratio: self.acceptability.at_ct_ratio,
                min_arc_days: self.acceptability.min_arc_days,
                fractional_sigma_a: self.acceptability.fractional_sigma_a,
            },
            fit_station_biases: u8::from(self.fit_station_biases),
            station_radec: empyrean_sys::EmpyreanStationRaDecConfig {
                sigma_prior_arcsec: self.station_radec.sigma_prior_arcsec,
                min_obs_per_station: self.station_radec.min_obs_per_station,
            },
            use_span_grouping: u8::from(self.use_span_grouping),
            rejection: empyrean_sys::EmpyreanRejectionConfig {
                enabled: u8::from(self.rejection.enabled),
                kind: match self.rejection.kind {
                    RejectionKind::Adaptive => empyrean_sys::EMPYREAN_REJECTION_KIND_ADAPTIVE as u8,
                    RejectionKind::CMC2003 => empyrean_sys::EMPYREAN_REJECTION_KIND_CMC2003 as u8,
                },
                chi2_base: self.rejection.chi2_base,
                lambda: self.rejection.lambda,
                max_threshold: self.rejection.max_threshold,
                chi2_rej: self.rejection.chi2_rej,
                chi2_rec: self.rejection.chi2_rec,
                max_passes: self.rejection.max_passes,
            },
            auto_force_model: u8::from(self.auto_force_model),
            output_representation: self.output_representation as i32,
        };
        let keep = ODConfigKeepalive {
            perturbers,
            weighting_layers,
            bias_dat_path,
        };
        (cfg, keep)
    }

    /// Convenience builder: a config carrying just the requested force
    /// model, defaults for everything else.
    pub fn with_force_model(force_model: ForceModelTier) -> Self {
        Self {
            force_model,
            ..Self::default()
        }
    }
}
