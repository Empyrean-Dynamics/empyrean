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
use super::nuisance::StationRADecConfig;
use super::rejection::{RejectionConfig, RejectionKind};
use super::result::{OriginPolicy, OutputEpoch, PhotometryModel, SolveForParams};
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
    /// Observation weighting pipeline. Default = `enabled` + VFCC2017
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
    /// DC convergence tolerance, tested on the **undamped** Gauss–Newton
    /// step's quadratic form — the number published back as
    /// [`DetermineResult::gn_step_qnorm`](crate::DetermineResult::gn_step_qnorm),
    /// NOT the μ-damped
    /// [`update_norm`](crate::DetermineResult::update_norm), which this
    /// tolerance does not bound. 0.0 → engine default (1e-5).
    pub convergence_tol: f64,
    /// Allow the outward-expansion pipeline to truncate a sub-arc it
    /// cannot fit as one piece. Default `true` (the engine default).
    ///
    /// Setting `false` makes an arc that spans a dynamical discontinuity
    /// FAIL loudly instead of delivering a fit of the reconcilable
    /// sub-arc with the rest tagged
    /// [`RejectionReason::OutsideArc`](crate::RejectionReason::OutsideArc).
    /// Two interactions matter before relying on it: per-observation
    /// [`rejection`](Self::rejection) is orthogonal and still runs (turn
    /// it off too to fit the whole arc or fail), and under
    /// [`OriginPolicy::Auto`] the refusal feeds the origin cascade rather
    /// than surfacing — pin the origin to get a pure loud failure.
    pub allow_arc_truncation: bool,
    /// Master switch for the co-orbital IOD lane. Default `true` (the
    /// engine default).
    ///
    /// Enabling it does not route ordinary objects through the lane — it
    /// still fires only when every co-orbitality gate passes. Set `false`
    /// to force the historical cascade.
    pub coorbital_enabled: bool,
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
    pub station_radec: StationRADecConfig,
    /// Use span-grouped Jacobian reuse.
    pub use_span_grouping: bool,

    // ── Rejection ──────────────────────────────────────────────────
    /// Outlier-rejection strategy + tuning.
    pub rejection: RejectionConfig,
    /// Auto-select force-model tier from IOD elements.
    pub auto_force_model: bool,
    /// Output coordinate representation for the fitted orbit + covariance.
    pub output_representation: crate::coordinate::Representation,
    /// Permit solving a thrust Δv segment whose burn window is not
    /// bracketed by observations (the state absorbs it otherwise; the
    /// Gates prior then carries it). Default `false` — refuse loudly.
    pub allow_unbracketed_maneuvers: bool,
    /// Post-OD photometric fit. `None` (default) disables it; the fit
    /// runs after the orbit is solved and never touches the state.
    pub photometry: Option<PhotometryConfig>,
    /// Uncertainty method for the fitted covariance. The OD fit is
    /// first-order by construction, so
    /// [`UncertaintyMethod::FirstOrder`](crate::UncertaintyMethod::FirstOrder)
    /// (the default) is delivered and
    /// [`UncertaintyMethod::auto`](crate::UncertaintyMethod::auto) resolves
    /// to first order — reported on
    /// [`DetermineResult::resolved_method`](crate::DetermineResult::resolved_method)
    /// and, for `Auto`, on
    /// [`DetermineResult::warnings`](crate::DetermineResult::warnings).
    /// Every richer method (`SecondOrder` / `SigmaPoint` / `MonteCarlo` /
    /// `Mixture`) is refused by name rather than silently downgraded.
    pub uncertainty_method: crate::UncertaintyMethod,
}

/// One band-to-Johnson-V conversion: the additive offset to V
/// (\\( V = m_\text{band} + \text{offset\_to\_v} \\)) and the systematic
/// 1σ of that conversion (the solar-colour assumption floor, not a
/// statistical error).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandConversion {
    /// Additive offset to Johnson V (mag).
    pub offset_to_v: f64,
    /// Systematic 1σ of the conversion (mag).
    pub sigma: f64,
}

/// Caller-supplied band→V conversions for the post-OD photometric fit.
///
/// [`default`](Self::default) is the MPC Magnitude Bands table unchanged
/// — the fit converts every reported survey-band magnitude to Johnson V
/// with it. [`with`](Self::with) supplies an entry: it replaces a code the
/// MPC table carries or adds one it does not (the LSST `Lu Lg Lr Li Lz
/// Ly`, Gaia `Gb`/`Gr`, the wide `VR`, …). The LSST codes carry no MPC
/// default and are refused by name until supplied — their magnitudes are
/// then excluded from the photometric fit with a report, and the
/// astrometry is unaffected. The comet magnitude flags `T`/`N` are refused
/// regardless of any supplied entry. Each converted band's offset is
/// reported back on the result's per-band `offset_applied`, so a fit is
/// auditable against the table it ran with.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BandConversions {
    supplied: std::collections::BTreeMap<String, BandConversion>,
}

impl BandConversions {
    /// Supply an entry for `code`: replaces a code the MPC table carries
    /// or adds one it does not. Builder-style:
    ///
    /// ```
    /// use empyrean::{BandConversion, BandConversions};
    /// let table = BandConversions::default()
    ///     .with("Lu", BandConversion { offset_to_v: -1.41, sigma: 0.30 });
    /// ```
    pub fn with(mut self, code: &str, conversion: BandConversion) -> Self {
        self.supplied.insert(code.to_string(), conversion);
        self
    }

    /// The supplied entries, in sorted code order. Empty = the MPC table
    /// unchanged.
    pub fn entries(&self) -> impl Iterator<Item = (&str, BandConversion)> {
        self.supplied
            .iter()
            .map(|(code, conv)| (code.as_str(), *conv))
    }

    /// True when no entry is supplied — the fit runs on the MPC table.
    pub fn is_empty(&self) -> bool {
        self.supplied.is_empty()
    }

    /// Marshal the supplied entries into FFI rows plus the owned `CString`
    /// backing their `code` pointers. Both must outlive the FFI call; they
    /// ride the [`ODConfigKeepalive`].
    fn to_ffi_rows(
        &self,
    ) -> (
        Vec<std::ffi::CString>,
        Vec<empyrean_sys::EmpyreanBandConversion>,
    ) {
        let codes: Vec<std::ffi::CString> = self
            .supplied
            .keys()
            .map(|c| std::ffi::CString::new(c.as_str()).expect("band code has no interior NUL"))
            .collect();
        let rows: Vec<empyrean_sys::EmpyreanBandConversion> = self
            .supplied
            .values()
            .zip(codes.iter())
            .map(|(conv, code)| empyrean_sys::EmpyreanBandConversion {
                code: code.as_ptr(),
                offset_to_v: conv.offset_to_v,
                sigma: conv.sigma,
            })
            .collect();
        (codes, rows)
    }
}

/// Post-OD photometric-fit configuration (mirrors the engine's
/// photometry config). Attach via [`ODConfig::photometry`]. Sentinel
/// rule: `0` / `0.0` on a tuning field requests the engine default.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotometryConfig {
    /// Model to fit. Default [`PhotometryModel::Auto`].
    pub model: PhotometryModel,
    /// 1σ lightcurve scatter floor (mag). 0.0 → engine default (0.2).
    pub sigma_lightcurve: f64,
    /// Include astrometrically-rejected observations' magnitudes.
    /// Default `false`.
    pub include_rejected: bool,
    /// Max Huber-IRLS iterations. 0 → engine default (30).
    pub max_irls_iterations: u32,
    /// Huber tuning constant. 0.0 → engine default (1.5).
    pub huber_k: f64,
    /// Band→V conversion table the fit converts with. Default = the MPC
    /// Magnitude Bands table; supply entries with
    /// [`BandConversions::with`] to add or replace a code. See
    /// [`BandConversions`].
    pub band_conversions: BandConversions,
}

impl Default for PhotometryConfig {
    fn default() -> Self {
        Self {
            model: PhotometryModel::Auto,
            sigma_lightcurve: 0.0,
            include_rejected: false,
            max_irls_iterations: 0,
            huber_k: 0.0,
            band_conversions: BandConversions::default(),
        }
    }
}

impl PhotometryConfig {
    fn to_ffi(
        &self,
        band_conversions: *const empyrean_sys::EmpyreanBandConversion,
        band_conversions_len: usize,
    ) -> empyrean_sys::EmpyreanPhotometryConfig {
        empyrean_sys::EmpyreanPhotometryConfig {
            model: self.model.to_int(),
            sigma_lightcurve: self.sigma_lightcurve,
            include_rejected: u8::from(self.include_rejected),
            max_irls_iterations: self.max_irls_iterations,
            huber_k: self.huber_k,
            band_conversions,
            band_conversions_len,
        }
    }
}

impl Default for ODConfig {
    fn default() -> Self {
        // Production hot path:
        //   - VFCC2017 station weighting + nightly de-weighting on
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
            allow_arc_truncation: true,
            coorbital_enabled: true,
            solve_for: SolveForParams::Auto,
            auto_escalation: AutoEscalationPolicy::default(),
            acceptability: AcceptabilityThresholds::default(),
            fit_station_biases: false,
            station_radec: StationRADecConfig::default(),
            use_span_grouping: false,
            rejection: RejectionConfig::default(),
            auto_force_model: false,
            output_representation: crate::coordinate::Representation::Cartesian,
            allow_unbracketed_maneuvers: false,
            photometry: None,
            uncertainty_method: crate::UncertaintyMethod::FirstOrder,
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
    /// Owned `code` strings the photometry band rows point into.
    pub band_codes: Vec<std::ffi::CString>,
    /// Photometry band→V conversion rows; the FFI photometry config points
    /// into this `Vec`.
    pub band_conversions: Vec<empyrean_sys::EmpyreanBandConversion>,
}

impl ODConfig {
    pub(crate) fn to_ffi_with(
        &self,
    ) -> crate::error::Result<(empyrean_sys::EmpyreanODConfig, ODConfigKeepalive)> {
        let perturbers: Vec<i32> = self
            .excluded_perturbers
            .iter()
            .map(|o| o.naif_id())
            .collect();
        let (mode, explicit) = match self.output_epoch {
            OutputEpoch::MidArc => (0, 0.0),
            OutputEpoch::LastObservation => (1, 0.0),
            // Explicit output epoch is a scale-carrying input; convert to
            // TDB once here, at the FFI boundary.
            OutputEpoch::Epoch(e) => (2, e.mjd_tdb()?),
            OutputEpoch::IODEpoch => (3, 0.0),
        };
        // Materialize the photometry band→V rows (and their owned code
        // strings) before the FFI struct that points into them. Empty
        // unless a photometry config supplies entries; empty leaves the
        // fit on the MPC Magnitude Bands table.
        let (band_codes, band_conversions) = match &self.photometry {
            Some(p) => p.band_conversions.to_ffi_rows(),
            None => (Vec::new(), Vec::new()),
        };
        let band_ptr = if band_conversions.is_empty() {
            std::ptr::null()
        } else {
            band_conversions.as_ptr()
        };
        // Materialize weighting layers (validates obs_codes against
        // the 4-byte C-ABI field — no silent truncation).
        let weighting_layers: Vec<empyrean_sys::EmpyreanWeightingLayer> = self
            .weighting
            .additional_layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| weighting_layer_to_ffi(idx, layer))
            .collect::<crate::error::Result<Vec<_>>>()?;
        let weighting = empyrean_sys::EmpyreanWeightingConfig {
            enabled: u8::from(self.weighting.enabled),
            preset: match self.weighting.preset {
                WeightingPreset::None => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_NONE as u8,
                WeightingPreset::VFCC2017 => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_VFCC2017 as u8,
                WeightingPreset::NEODYS => empyrean_sys::EMPYREAN_WEIGHTING_PRESET_NEODYS as u8,
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
            noise_model: self.weighting.noise_model.to_ffi(),
            covariance_information: self.weighting.covariance_information.to_ffi(),
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
            // Tri-state at the ABI: this wrapper always states a value,
            // so the engine-default branch (negative) is never taken from
            // here — the Rust surface's own default carries it instead.
            allow_arc_truncation: i8::from(self.allow_arc_truncation),
            coorbital_enabled: i8::from(self.coorbital_enabled),
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
            station_radec: empyrean_sys::EmpyreanStationRADecConfig {
                sigma_prior_arcsec: self.station_radec.sigma_prior_arcsec,
                min_obs_per_station: self.station_radec.min_obs_per_station,
            },
            use_span_grouping: u8::from(self.use_span_grouping),
            rejection: empyrean_sys::EmpyreanRejectionConfig {
                // Tri-state via the ABI's DEFAULT/ON/OFF convention:
                // engine default (None) is the zero-initialized value, never
                // a spelling of off.
                enabled: match self.rejection.enabled {
                    None => empyrean_sys::EMPYREAN_REJECTION_ENABLED_DEFAULT as u8,
                    Some(true) => empyrean_sys::EMPYREAN_REJECTION_ENABLED_ON as u8,
                    Some(false) => empyrean_sys::EMPYREAN_REJECTION_ENABLED_OFF as u8,
                },
                kind: match self.rejection.kind {
                    RejectionKind::Adaptive => empyrean_sys::EMPYREAN_REJECTION_KIND_ADAPTIVE as u8,
                    RejectionKind::CMC2003 => empyrean_sys::EMPYREAN_REJECTION_KIND_CMC2003 as u8,
                    RejectionKind::Blunder => empyrean_sys::EMPYREAN_REJECTION_KIND_BLUNDER as u8,
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
            // Per-axis flags — read by the C ABI only when solve_for == 3
            // (EXPLICIT); harmless for the coarse codes.
            solve_for_flags: self.solve_for.flags().to_ffi(),
            allow_unbracketed_maneuvers: u8::from(self.allow_unbracketed_maneuvers),
            has_photometry: u8::from(self.photometry.is_some()),
            photometry: self
                .photometry
                .as_ref()
                .map(|p| p.to_ffi(band_ptr, band_conversions.len()))
                .unwrap_or_default(),
            uncertainty_method: self.uncertainty_method.to_ffi(),
        };
        let keep = ODConfigKeepalive {
            perturbers,
            weighting_layers,
            bias_dat_path,
            band_codes,
            band_conversions,
        };
        Ok((cfg, keep))
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

#[cfg(test)]
mod uncertainty_method_tests {
    use super::*;

    /// The default config marshals the first-order tag across the FFI
    /// boundary — the OD seam's historical behaviour, now explicit.
    /// Mutation guard: dropping `uncertainty_method` from `to_ffi_with`
    /// would leave the FFI struct's `memset(0)` tag at 0 and pass this one
    /// but fail the SecondOrder case below.
    #[test]
    fn default_od_config_marshals_first_order() {
        let cfg = ODConfig::default();
        let (ffi, _keep) = cfg.to_ffi_with().expect("default config marshals");
        assert_eq!(ffi.uncertainty_method.tag, 0);
    }

    #[test]
    fn second_order_marshals_the_second_order_tag() {
        let cfg = ODConfig {
            uncertainty_method: crate::UncertaintyMethod::SecondOrder,
            ..ODConfig::default()
        };
        let (ffi, _keep) = cfg.to_ffi_with().expect("config marshals");
        assert_eq!(ffi.uncertainty_method.tag, 1);
    }
}

#[cfg(test)]
mod band_conversion_tests {
    use super::*;

    /// No photometry → no band rows marshalled; the FFI photometry struct
    /// is zero-init (null table pointer), leaving the fit on the MPC table.
    #[test]
    fn no_photometry_marshals_no_band_rows() {
        let cfg = ODConfig::default();
        assert!(cfg.photometry.is_none());
        let (ffi, keep) = cfg.to_ffi_with().expect("default config marshals");
        assert_eq!(ffi.has_photometry, 0);
        assert!(keep.band_conversions.is_empty());
        assert!(ffi.photometry.band_conversions.is_null());
        assert_eq!(ffi.photometry.band_conversions_len, 0);
    }

    /// Photometry with the default (empty) table marshals no rows — the
    /// engine applies the MPC table, not an empty one.
    #[test]
    fn default_table_marshals_no_rows() {
        let cfg = ODConfig {
            photometry: Some(PhotometryConfig::default()),
            ..ODConfig::default()
        };
        assert!(cfg.photometry.as_ref().unwrap().band_conversions.is_empty());
        let (ffi, keep) = cfg.to_ffi_with().expect("config marshals");
        assert_eq!(ffi.has_photometry, 1);
        assert!(keep.band_conversions.is_empty());
        assert!(ffi.photometry.band_conversions.is_null());
        assert_eq!(ffi.photometry.band_conversions_len, 0);
    }

    /// A supplied entry marshals one row, with the code string, offset and
    /// σ carried verbatim and the FFI pointer live into the keepalive.
    #[test]
    fn supplied_entry_marshals_one_row() {
        let cfg = ODConfig {
            photometry: Some(PhotometryConfig {
                band_conversions: BandConversions::default().with(
                    "Lu",
                    BandConversion {
                        offset_to_v: -1.41,
                        sigma: 0.30,
                    },
                ),
                ..PhotometryConfig::default()
            }),
            ..ODConfig::default()
        };
        let (ffi, keep) = cfg.to_ffi_with().expect("config marshals");
        assert_eq!(ffi.photometry.band_conversions_len, 1);
        assert_eq!(keep.band_conversions.len(), 1);
        assert_eq!(keep.band_codes.len(), 1);
        assert!(!ffi.photometry.band_conversions.is_null());
        // The row the FFI pointer addresses carries the supplied values and
        // its code string.
        let row = unsafe { &*ffi.photometry.band_conversions };
        assert_eq!(row.offset_to_v, -1.41);
        assert_eq!(row.sigma, 0.30);
        let code = unsafe { std::ffi::CStr::from_ptr(row.code) };
        assert_eq!(code.to_str().unwrap(), "Lu");
    }

    /// `entries()` reports the supplied set in sorted code order; a second
    /// `with` on the same code replaces it.
    #[test]
    fn with_replaces_and_entries_are_sorted() {
        let table = BandConversions::default()
            .with(
                "r",
                BandConversion {
                    offset_to_v: 0.25,
                    sigma: 0.05,
                },
            )
            .with(
                "Lu",
                BandConversion {
                    offset_to_v: -1.41,
                    sigma: 0.30,
                },
            )
            .with(
                "r",
                BandConversion {
                    offset_to_v: 0.30,
                    sigma: 0.06,
                },
            );
        let entries: Vec<_> = table.entries().collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "Lu");
        assert_eq!(entries[1].0, "r");
        assert_eq!(entries[1].1.offset_to_v, 0.30);
    }
}

#[cfg(test)]
mod noise_model_config_tests {
    use super::*;
    use crate::od::weighting::{CovarianceInformation, NoiseModel};

    #[test]
    fn rejection_enabled_tristate_maps_to_the_ffi_sentinels() {
        let cases = [
            (None, empyrean_sys::EMPYREAN_REJECTION_ENABLED_DEFAULT as u8),
            (
                Some(true),
                empyrean_sys::EMPYREAN_REJECTION_ENABLED_ON as u8,
            ),
            (
                Some(false),
                empyrean_sys::EMPYREAN_REJECTION_ENABLED_OFF as u8,
            ),
        ];
        for (enabled, want) in cases {
            let mut cfg = ODConfig::default();
            cfg.rejection.enabled = enabled;
            let (ffi, _keep) = cfg.to_ffi_with().unwrap();
            assert_eq!(ffi.rejection.enabled, want, "enabled {enabled:?}");
        }
    }

    #[test]
    fn rejection_kind_blunder_crosses_the_ffi() {
        let mut cfg = ODConfig::default();
        cfg.rejection.kind = RejectionKind::Blunder;
        let (ffi, _keep) = cfg.to_ffi_with().unwrap();
        assert_eq!(
            ffi.rejection.kind as u32,
            empyrean_sys::EMPYREAN_REJECTION_KIND_BLUNDER
        );
    }

    #[test]
    fn global_noise_model_and_covariance_information_cross_the_ffi() {
        let mut cfg = ODConfig::default();
        cfg.weighting.noise_model = NoiseModel::StudentT { nu: 4.0 };
        cfg.weighting.covariance_information = CovarianceInformation::Expected;
        let (ffi, _keep) = cfg.to_ffi_with().unwrap();
        assert_eq!(
            ffi.weighting.noise_model.tag as u32,
            empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T
        );
        assert_eq!(ffi.weighting.noise_model.nu, 4.0);
        assert_eq!(
            ffi.weighting.covariance_information as u32,
            empyrean_sys::EMPYREAN_COVARIANCE_INFORMATION_EXPECTED
        );
    }

    #[test]
    fn the_default_config_carries_the_gaussian_law() {
        let cfg = ODConfig::default();
        assert_eq!(cfg.weighting.noise_model, NoiseModel::Gaussian);
        let (ffi, _keep) = cfg.to_ffi_with().unwrap();
        assert_eq!(
            ffi.weighting.noise_model.tag as u32,
            empyrean_sys::EMPYREAN_NOISE_MODEL_GAUSSIAN
        );
    }
}
