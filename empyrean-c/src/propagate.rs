use std::ffi::{CStr, CString};
use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::constants::KM_PER_AU;
use empyrean_core::convert::{
    coordinate_state_to_coordinates, frame_to_int, representation_to_int,
};
use empyrean_core::nongrav::{
    GFunction, NonGravModel, NonGravParams, SteeringLaw, ThrustArc, ThrustParams,
};
use empyrean_core::orbits::Orbits;
use empyrean_core::photometry::PhotometricParams;
use empyrean_core::propagation::events::DynamicalEvent;
use empyrean_core::propagation::{
    AdvancedIntegratorConfig, DiagnosticsConfig, EventConfig, IntegratorChoice,
    OriginSwitchingConfig, PropagationConfig, PropagationResult, UncertaintyMethod, propagate,
};
use empyrean_core::time::Epoch;

/// Build a [`PhotometricParams`] from an [`EmpyreanOrbit`] when the
/// caller supplied a phase function. Returns `None` when
/// `phot_system == EMPYREAN_PHASE_FUNCTION_NONE` so callers can
/// `set_photometric_params(i, None)` to leave H/G unset.
pub(crate) fn empyrean_orbit_photometric_params(
    orbit: &EmpyreanOrbit,
) -> Option<PhotometricParams> {
    if !orbit.h_mag.is_finite() {
        return None;
    }
    match orbit.phot_system {
        EMPYREAN_PHASE_FUNCTION_HG => Some(PhotometricParams::hg(orbit.h_mag, orbit.slope1)),
        EMPYREAN_PHASE_FUNCTION_HG1G2 => Some(PhotometricParams::hg1g2(
            orbit.h_mag,
            orbit.slope1,
            orbit.slope2,
        )),
        EMPYREAN_PHASE_FUNCTION_HG12 => Some(PhotometricParams::hg12(orbit.h_mag, orbit.slope1)),
        EMPYREAN_PHASE_FUNCTION_NONE => None,
        _ => None,
    }
}

use crate::{CoordinateState, EmpyreanContext, set_last_error};

// ── C-compatible types ──────────────────────────────────────

/// Phase-function model selector for [`EmpyreanOrbit`] photometry.
///
/// `EMPYREAN_PHASE_FUNCTION_NONE` disables photometric magnitude
/// computation for that orbit; the corresponding ephemeris row gets
/// `mag = NaN`.
pub const EMPYREAN_PHASE_FUNCTION_NONE: i32 = -1;
pub const EMPYREAN_PHASE_FUNCTION_HG: i32 = 0;
pub const EMPYREAN_PHASE_FUNCTION_HG1G2: i32 = 1;
pub const EMPYREAN_PHASE_FUNCTION_HG12: i32 = 2;

/// Steering-law selector tags for [`EmpyreanThrustArc::steering_law`].
///
/// Mirrors [`empyrean_core::nongrav::SteeringLaw`]. The value is read as a
/// plain `int32_t` at the marshaling boundary and validated loudly — an
/// unrecognized value is an error, never a silent default (the same tag
/// convention used by `EMPYREAN_PHASE_FUNCTION_*` / `EMPYREAN_INTEGRATOR_*`).
/// Which `EmpyreanThrustArc` steering-parameter slots are read depends on the
/// selected law:
///
/// | value | Steering law | Active fields |
/// |-------|--------------|---------------|
/// | 0 | `CONSTANT_RTN`     | `steering_alpha_rad`, `steering_beta_rad` |
/// | 1 | `VELOCITY_TANGENT` | (none) |
/// | 2 | `INERTIAL_FIXED`   | `steering_direction` |
///
/// Constant RTN angles relative to the central body: fixed in-plane
/// (\\(\alpha\\)) and out-of-plane (\\(\beta\\)) angles.
pub const EMPYREAN_STEERING_LAW_CONSTANT_RTN: i32 = 0;
/// Thrust aligned with velocity relative to the central body.
pub const EMPYREAN_STEERING_LAW_VELOCITY_TANGENT: i32 = 1;
/// Fixed direction in the inertial frame (normalized internally).
pub const EMPYREAN_STEERING_LAW_INERTIAL_FIXED: i32 = 2;

/// A single continuous-thrust arc, mirroring
/// [`empyrean_core::nongrav::ThrustArc`] as a flat `#[repr(C)]` record.
///
/// Arcs are supplied through [`EmpyreanOrbit::thrust_arcs`] as a
/// caller-owned side array borrowed read-only for the duration of the call.
///
/// The acceleration during the arc is
/// \\[ \mathbf{a}(t) = \sigma(t)\,\frac{F}{m(t)}\,\hat{d} \\]
/// with a smooth \\(\tanh\\) switch \\(\sigma(t)\\) of width set by
/// `sharpness`, steering direction \\(\hat{d}\\) from `steering_law`, and
/// mass \\(m(t)\\) that depletes when `isp_s` is finite.
#[repr(C)]
pub struct EmpyreanThrustArc {
    /// Arc start epoch (MJD TDB).
    pub start_mjd_tdb: f64,
    /// Arc end epoch (MJD TDB).
    pub end_mjd_tdb: f64,
    /// Engine thrust force in Newtons.
    pub thrust_n: f64,
    /// Spacecraft mass at arc start in kilograms.
    pub mass_kg: f64,
    /// Specific impulse in seconds. Any non-finite value (`NaN` or `±∞`)
    /// selects constant mass (no depletion); a finite value depletes mass at
    /// \\(\dot m = F/(I_{sp} g_0)\\). This is the `Option<f64>` NaN-sentinel
    /// convention shared with [`EmpyreanOrbit::non_grav_dt`].
    pub isp_s: f64,
    /// Steering-law selector — see the `EMPYREAN_STEERING_LAW_*` tag
    /// constants. An unrecognized value errors loudly.
    pub steering_law: i32,
    /// `CONSTANT_RTN` in-plane angle α from radial toward transverse
    /// (radians). Read only when `steering_law == CONSTANT_RTN`.
    pub steering_alpha_rad: f64,
    /// `CONSTANT_RTN` out-of-plane angle β toward orbit normal (radians).
    /// Read only when `steering_law == CONSTANT_RTN`.
    pub steering_beta_rad: f64,
    /// `INERTIAL_FIXED` direction vector (normalized internally). Read only
    /// when `steering_law == INERTIAL_FIXED`.
    pub steering_direction: [f64; 3],
    /// \\(\tanh\\) switching sharpness (1/days). Higher = sharper on/off.
    pub sharpness: f64,
    /// Central-body NAIF id for the RTN / velocity-tangent frame reference
    /// (same encoding as [`EmpyreanOrbit`]'s `state.origin`, e.g. `399` for
    /// Earth, `10` for the Sun). An unknown NAIF id errors loudly.
    pub central_body_naif_id: i32,
}

/// An orbit with optional non-gravitational parameters, continuous-thrust
/// arcs, and photometry.
///
/// Thrust: when `n_thrust_arcs > 0`, [`thrust_arcs`](Self::thrust_arcs)
/// (plus the optional [`dv_corrections`](Self::dv_corrections) /
/// [`correction_covariances`](Self::correction_covariances) side arrays)
/// build a [`ThrustParams`] for this orbit. All-zero `a1/a2/a3` and
/// `n_thrust_arcs == 0` is a pure-gravity orbit.
///
/// `a1/a2/a3` are the Marsden–Sekanina RTN coefficients (radial,
/// transverse, normal) in AU/day². `ng_alpha … ng_k` parameterize the
/// g(r) distance-dependent scaling:
/// \\[ g(r) = \alpha \\, (r/r_0)^{-m} \\, (1 + (r/r_0)^n)^{-k} \\].
/// All-zeros for the g-function fields selects the inverse-square
/// default (Yarkovsky / SRP asteroid case): \\(\alpha = r_0 = 1\\),
/// \\(m = 2\\), \\(n = k = 0\\). Comets typically pass SBDB's
/// Marsden water-ice values
/// (\\(\alpha = 0.1113, r_0 = 2.808, m = 2.15, n = 5.093, k = 4.6142\\)).
///
/// Photometry: when `phot_system` is one of the `EMPYREAN_PHASE_FUNCTION_*`
/// non-`NONE` constants, ephemeris generation produces apparent magnitude
/// using the (`H`, `slope1`, `slope2`) triple per the chosen phase function:
///
/// | Model | `H` | `slope1` | `slope2` |
/// |-------|-----|----------|----------|
/// | `HG`     | absolute magnitude | G    | unused (0)  |
/// | `HG1G2`  | absolute magnitude | G₁   | G₂          |
/// | `HG12`   | absolute magnitude | G₁₂  | unused (0)  |
#[repr(C)]
pub struct EmpyreanOrbit {
    pub state: CoordinateState,
    /// Caller-supplied orbit identifier — primary key for joining
    /// outputs (propagated states, events, ephemeris, B-planes, IP) to
    /// the input row. NUL-terminated UTF-8. A null pointer or empty
    /// string causes the C ABI to fabricate a positional `"orbit_{i}"`
    /// tag instead — set this whenever you want stable joins.
    ///
    /// **Ownership:** read-only borrow from the caller; the C ABI does
    /// not free it.
    pub orbit_id: *const std::ffi::c_char,
    /// Caller-supplied object identifier (e.g. SBDB designation).
    /// Distinct from `orbit_id` so multiple orbit hypotheses for the
    /// same object can share an `object_id`. NUL-terminated UTF-8.
    /// Empty string or null = no object identifier.
    ///
    /// **Ownership:** read-only borrow from the caller; the C ABI does
    /// not free it.
    pub object_id: *const std::ffi::c_char,
    /// Marsden-Sekanina A1 coefficient (radial). 0 ⇒ no non-grav.
    pub a1: f64,
    /// Marsden-Sekanina A2 coefficient (transverse).
    pub a2: f64,
    /// Marsden-Sekanina A3 coefficient (normal).
    pub a3: f64,
    /// g(r) normalizing constant α. All-zero g-fields → inverse_square.
    pub ng_alpha: f64,
    /// g(r) reference distance r₀ (AU).
    pub ng_r0: f64,
    /// g(r) inner power-law exponent m.
    pub ng_m: f64,
    /// g(r) outer power-law exponent n.
    pub ng_n: f64,
    /// g(r) outer damping exponent k.
    pub ng_k: f64,
    /// SBDB non-grav time delay in days. Use NaN for no delay (the
    /// asteroid default and the Marsden water-ice default for comets
    /// SBDB doesn't fit a delay for). Set to a finite value (positive
    /// or negative) when SBDB's `model_pars[]` exposes a `DT` field —
    /// e.g. 67P (+45.7d), 46P/Wirtanen (−14.1d), 2I/Borisov (−65.1d).
    pub non_grav_dt: f64,
    /// Prior variance on the non-grav time delay DT (days²). NaN or ≤0 = no
    /// prior; a finite positive value opens + priors the DT column in a
    /// StateAndNonGravAndDT fit.
    pub non_grav_dt_variance: f64,
    /// 1 when `non_grav_covariance` carries a non-grav prior covariance; 0
    /// otherwise. Set by the OD output path (a fitted orbit) so it re-feeds
    /// into a StateAndNonGrav refine without losing its non-grav prior;
    /// leave 0 for hand-built / SBDB / propagate inputs.
    pub has_non_grav_covariance: u8,
    /// Non-grav 3×3 covariance for (A1, A2, A3), row-major. Only read when
    /// `has_non_grav_covariance = 1`.
    pub non_grav_covariance: [[f64; 3]; 3],
    /// Phase-function model. `EMPYREAN_PHASE_FUNCTION_NONE` disables
    /// magnitude computation; the other values map to villeneuve's
    /// `PhaseFunction` enum.
    pub phot_system: i32,
    /// Absolute magnitude H. Ignored when `phot_system == NONE`.
    pub h_mag: f64,
    /// Slope parameter slot 1 — G (HG), G₁ (HG1G2), or G₁₂ (HG12).
    pub slope1: f64,
    /// Slope parameter slot 2 — G₂ (HG1G2 only); unused (0) for HG / HG12.
    pub slope2: f64,
    /// Continuous-thrust arcs (variable length). Null / `n_thrust_arcs == 0`
    /// ⇒ no thrust (gravity + non-grav only).
    ///
    /// **Ownership:** caller-owned; borrowed read-only for the duration of
    /// the call. The C ABI never frees it. A null pointer with
    /// `n_thrust_arcs > 0` is a loud argument error, not a silent skip; a
    /// non-null pointer with `n_thrust_arcs == 0` is treated as absent (the
    /// pointer is not read).
    pub thrust_arcs: *const EmpyreanThrustArc,
    /// Number of arcs in [`thrust_arcs`](Self::thrust_arcs).
    pub n_thrust_arcs: usize,
    /// Optional per-arc Δv corrections (AU/day), positional with
    /// [`thrust_arcs`](Self::thrust_arcs). Null / `n_dv_corrections == 0` ⇒
    /// no targeting corrections. Supplying corrections without any arc is a
    /// loud argument error.
    ///
    /// **Ownership:** caller-owned; borrowed read-only for the call.
    pub dv_corrections: *const [f64; 3],
    /// Number of corrections in [`dv_corrections`](Self::dv_corrections).
    pub n_dv_corrections: usize,
    /// Optional 3×3 covariance (AU/day)² per Δv correction, positional with
    /// [`dv_corrections`](Self::dv_corrections). When non-empty its length
    /// MUST equal `n_dv_corrections`; a non-empty covariance triggers the
    /// wide-Jet burn-sensitivity propagation. The engine's higher-order
    /// (third-order) propagation path does not implement thrust-correction
    /// covariance and rejects that combination loudly upstream
    /// (`Not implemented: … ThirdOrder with thrust-correction covariance …`,
    /// surfaced as a propagation error) rather than silently dropping the
    /// covariance.
    ///
    /// **Ownership:** caller-owned; borrowed read-only for the call.
    pub correction_covariances: *const [[f64; 3]; 3],
    /// Number of entries in
    /// [`correction_covariances`](Self::correction_covariances).
    pub n_correction_covariances: usize,
}

/// Origin-switching configuration for trajectory splitting at body
/// Laplace spheres of influence (Amato/Baù/Bombardelli 2017 §6).
///
/// Default **enabled** (the `_DEFAULT` sentinel resolves to
/// `enabled = 1`). When enabled, integration switches to a
/// body-centric frame inside the SOI of each eligible body and back
/// to SSB on exit. Set `enabled = 0` explicitly to opt out.
///
/// At the C-ABI surface, the per-body opt-in list (`bodies` in
/// villeneuve) is not yet exposed — `EMPYREAN_ORIGIN_SWITCHING_ON`
/// selects all monitored bodies. File a request if per-body scoping
/// is needed from the C-ABI surface.
#[repr(C)]
pub struct EmpyreanOriginSwitchingConfig {
    /// Trajectory-splitting policy. Tri-state, see
    /// `EMPYREAN_ORIGIN_SWITCHING_*` constants:
    /// - `0 = DEFAULT` — use the upstream default (currently `ON`).
    ///   This is what `memset(0)` gives, matching the same sentinel
    ///   convention used for `dt_min` / `dt_initial` / `epsilon` /
    ///   `hysteresis` etc. — external C consumers can zero-init the
    ///   config struct and get the right behavior automatically.
    /// - `1 = ON`  — force trajectory-splitting on (matches the prior
    ///   semantic for `enabled = 1`, backward-compatible).
    /// - `2 = OFF` — force trajectory-splitting off. (The prior
    ///   semantic for `enabled = 0` is now reachable only via this
    ///   explicit value, since 0 now means DEFAULT.)
    ///
    /// Unknown values fall back to the upstream default.
    pub enabled: u8,
    /// Fractional band around the acceleration-ratio crossover used
    /// to suppress chatter (default 0.2 = ±20 %). 0.0 → upstream
    /// default.
    pub hysteresis: f64,
}

/// Integrator backend tag for [`EmpyreanAdvancedIntegratorConfig::integrator`].
///
/// - `0` = GR15 (Gauss-Radau 15, derived solely from Everhart 1985 and
///   Rein & Spiegel 2015; default)
/// - `1` = DOP853 (Dormand-Prince 8(5,3), derived solely from Hairer,
///   Nørsett & Wanner 1993; ~1.4× faster than GR15 with looser median
///   accuracy ~358 m vs Horizons)
///
/// Any other value is treated as GR15 with no warning. IAS15 is not
/// exposed through this surface and is not available downstream.
/// Exported for C consumers; the `0` value is also the zero-sentinel the
/// translator reads directly, so it has no in-crate use site.
#[allow(dead_code)]
pub const EMPYREAN_INTEGRATOR_GR15: i32 = 0;
pub const EMPYREAN_INTEGRATOR_DOP853: i32 = 1;

/// Tri-state tags for [`EmpyreanOriginSwitchingConfig::enabled`].
///
/// `memset(0)` of a fresh `EmpyreanPropagationConfig` gives
/// `DEFAULT`, which resolves to whatever the upstream Rust
/// [`OriginSwitchingConfig::default()`] reports as `enabled`
/// (currently `true`). Callers who want explicit ON or OFF set the
/// other constants. This matches how `dt_min` / `dt_initial` /
/// `epsilon` etc. use sentinel values for "use upstream default."
#[allow(dead_code)]
pub const EMPYREAN_ORIGIN_SWITCHING_DEFAULT: u8 = 0;
pub const EMPYREAN_ORIGIN_SWITCHING_ON: u8 = 1;
pub const EMPYREAN_ORIGIN_SWITCHING_OFF: u8 = 2;

/// Integrator tuning. Mirrors
/// [`villeneuve::propagation::AdvancedIntegratorConfig`]. The default
/// integrator backend is Gauss-Radau 15 (GR15), derived solely from the
/// published papers; `epsilon` is interpreted as the relative b₆
/// truncation tolerance.
///
/// Sentinel rule: `0.0` requests the upstream default for `epsilon` /
/// `encounter_timescale_divisor`; `dt_initial` / `dt_min` use NaN to
/// mean "auto-compute"; `0` requests the upstream defaults for
/// `max_steps` and `max_dense_steps`.
#[repr(C)]
pub struct EmpyreanAdvancedIntegratorConfig {
    /// Integrator backend (0 = GR15 default, 1 = DOP853). See
    /// `EMPYREAN_INTEGRATOR_*` constants.
    pub integrator: i32,
    /// Truncation-error tolerance (relative b₆ for GR15, rtol for
    /// DOP853 paired with a fixed atol = 1e-14). 0.0 → upstream
    /// default (1e-9).
    pub epsilon: f64,
    /// Initial step size in days. NaN → auto from orbital timescale.
    pub dt_initial: f64,
    /// Minimum allowed step size in days. NaN → auto.
    pub dt_min: f64,
    /// Encounter dynamical-timescale step floor divisor. 0.0 → upstream default (1000.0).
    pub encounter_timescale_divisor: f64,
    /// Maximum integration steps before aborting. 0 → upstream default (10_000_000).
    pub max_steps: usize,
    /// Memory cap on the per-step b-coefficient cache. 0 → upstream default (100_000).
    pub max_dense_steps: usize,
    /// Cache the integrator's per-step b-coefficients for fast
    /// interpolation (light-time iteration, dense output around close
    /// approaches, arbitrary-epoch state queries). 1 = on, 0 = off
    /// (default).
    pub cache_integrator_steps: u8,
    /// Origin-switching trajectory-splitting configuration. Default
    /// enabled.
    pub origin_switching: EmpyreanOriginSwitchingConfig,
}

/// Per-trajectory diagnostic outputs (sensitivity, nonlinearity, …).
/// Mirrors [`villeneuve::diagnostics::DiagnosticsConfig`].
///
/// All metrics default to off. Sentinel: `0` for `sample_stride` →
/// upstream default (1); a NaN threshold means "no event emission for
/// that metric".
#[repr(C)]
pub struct EmpyreanDiagnosticsConfig {
    /// Frobenius norm of position-STM block. 1 = on, 0 = off.
    pub sensitivity: u8,
    /// Hessian/gradient ratio (Jet2 only). 1 = on, 0 = off.
    pub nonlinearity: u8,
    /// Finite-time Lyapunov exponent. 1 = on, 0 = off.
    pub lyapunov: u8,
    /// Keyhole detection at close approaches (Jet1+). 1 = on, 0 = off.
    pub keyholes: u8,
    /// Bifurcation detection at close approaches (Jet2 only). 1 = on, 0 = off.
    pub bifurcations: u8,
    /// Timeseries sampling stride: every Nth integration step. 0 → upstream default (1).
    pub sample_stride: usize,
    /// Threshold for `HighSensitivity` events. NaN → no emission.
    pub sensitivity_threshold: f64,
    /// Threshold for `ChaoticRegion` events. NaN → no emission.
    pub lyapunov_threshold: f64,
    /// Threshold for `HighNonlinearity` events (Jet2 only). NaN → no emission.
    pub nonlinearity_threshold: f64,
}

/// Uncertainty-method tag values for [`EmpyreanUncertaintyMethod::tag`].
pub const EMPYREAN_UNCERTAINTY_FIRST: u8 = 0;
pub const EMPYREAN_UNCERTAINTY_SECOND: u8 = 1;
pub const EMPYREAN_UNCERTAINTY_SIGMA_POINT: u8 = 2;
pub const EMPYREAN_UNCERTAINTY_MONTE_CARLO: u8 = 3;
pub const EMPYREAN_UNCERTAINTY_AUTO: u8 = 4;

/// Uncertainty-propagation method, mirroring
/// [`empyrean_core::propagation::UncertaintyMethod`] as a flat C struct.
/// Fields outside the active variant are ignored — set them to zero /
/// NaN.
///
/// | tag | Variant | Active fields |
/// |-----|---------|---------------|
/// | 0   | First       | (none) |
/// | 1   | Second      | (none) |
/// | 2   | SigmaPoint  | `sp_n_sigma`, `sp_samples_per_plane` |
/// | 3   | MonteCarlo  | `mc_n_samples`, `mc_seed_some` (1 = use `mc_seed`) |
/// | 4   | Auto        | `auto_threshold_first`, `auto_threshold_mixture`, `auto_threshold_ip_skip`, `auto_gmm_max_depth`, `auto_gmm_components_per_split` |
#[repr(C)]
pub struct EmpyreanUncertaintyMethod {
    /// Variant tag — see `EMPYREAN_UNCERTAINTY_*` constants.
    pub tag: u8,
    /// SigmaPoint: number of sigma deviations (default 1.0).
    pub sp_n_sigma: f64,
    /// SigmaPoint: points per coordinate-plane pair (default 8 → 120 total).
    pub sp_samples_per_plane: u64,
    /// MonteCarlo: number of random samples.
    pub mc_n_samples: u64,
    /// MonteCarlo: 1 if `mc_seed` is set, 0 to draw from `thread_rng`.
    pub mc_seed_some: u8,
    /// MonteCarlo: RNG seed when `mc_seed_some == 1`.
    pub mc_seed: u64,
    /// Auto: first-order regime tuning parameter.
    pub auto_threshold_first: f64,
    /// Auto: adaptive-mixture regime tuning parameter.
    pub auto_threshold_mixture: f64,
    /// Auto: impact-probability tuning parameter for the higher-order pass.
    pub auto_threshold_ip_skip: f64,
    /// Auto: adaptive-Gaussian-mixture maximum recursion depth.
    pub auto_gmm_max_depth: u64,
    /// Auto: adaptive-Gaussian-mixture components per split (odd).
    pub auto_gmm_components_per_split: u64,
}

/// Event-detection configuration. Mirrors
/// [`villeneuve::events::EventConfig`] (less the `enrichment` sub-config,
/// which carries internal nested data that doesn't translate cleanly
/// through C — it always uses upstream defaults).
///
/// `body_filter_naif` is non-owning: caller must keep the array alive
/// for the duration of the propagation call. Pass `null` /
/// `num_body_filter = 0` to monitor all bodies.
#[repr(C)]
pub struct EmpyreanEventConfig {
    pub close_approaches: u8,
    pub impacts: u8,
    pub atmospheric: u8,
    pub possible_impacts: u8,
    pub shadow_events: u8,
    /// Number of NAIF IDs in [`body_filter_naif`]; 0 = monitor all bodies.
    pub num_body_filter: usize,
    /// Pointer to `num_body_filter` NAIF IDs to restrict event monitoring.
    pub body_filter_naif: *const i32,
    /// Insert dense output points around close approaches. 1 = on (auto-enables
    /// `cache_integrator_steps`), 0 = off (default).
    pub dense_output: u8,
    /// Cadence (days) of dense output. 0.0 → upstream default (5 minutes).
    pub dense_output_cadence_days: f64,
}

/// Propagation configuration.
///
/// Mirrors [`villeneuve::propagation::PropagationConfig`] structurally:
/// shared scalar fields at the top, nested config bundles for events,
/// diagnostics, and integrator tuning. Sentinel rule for primitive
/// fields: `0` / `0.0` requests the upstream default; documented
/// negative-value sentinels retain their special meanings.
///
/// `stop_condition` and `events.enrichment` are not exposed at this
/// surface — both carry internal nested data that doesn't translate
/// cleanly. They use upstream defaults; reach for empyrean-core's
/// Rust API when they need to be overridden.
#[repr(C)]
pub struct EmpyreanPropagationConfig {
    // ── Force model ─────────────────────────────────────────
    /// Force-model tier: 0=Approximate, 1=Basic, 2=Standard.
    pub force_model: i32,
    /// Number of `excluded_perturbers` in [`excluded_perturbers_naif`]; 0 = none.
    pub num_excluded_perturbers: usize,
    /// NAIF IDs to exclude (self-perturbation avoidance for SB441-N16
    /// bodies). Non-owning. Pass `null` / count 0 to use the full
    /// perturber set.
    pub excluded_perturbers_naif: *const i32,

    // ── Uncertainty & STM ──────────────────────────────────
    /// Uncertainty-propagation method (tag + per-variant params).
    pub uncertainty_method: EmpyreanUncertaintyMethod,
    /// Force `Jet1<6>` integration even without input covariance.
    /// 1 = on, 0 = off (default).
    pub compute_stm: u8,

    // ── Frame, events, diagnostics ─────────────────────────
    /// Output reference frame: 0=ICRF, 1=EclipticJ2000.
    pub frame: i32,
    pub events: EmpyreanEventConfig,
    pub diagnostics: EmpyreanDiagnosticsConfig,

    // ── Parallelism ────────────────────────────────────────
    /// Threads for multi-orbit propagation. 0 = use all cores
    /// (Rayon default); positive N = exactly N cores.
    pub num_threads: usize,

    // ── Integrator calibration ─────────────────────────────
    pub advanced: EmpyreanAdvancedIntegratorConfig,
}

/// Resolved-kind tag values for [`EmpyreanPropagatedState::resolved_kind`].
///
/// Mirrors [`empyrean_core::propagation::CovarianceKind`]; carries the
/// kind that was actually produced for this (orbit, epoch). Always
/// concrete — never `Auto` — and tagged
/// `EMPYREAN_COVARIANCE_KIND_LINEAR` everywhere except inside
/// [`UncertaintyMethod::Auto`]'s CA windows.
pub const EMPYREAN_COVARIANCE_KIND_LINEAR: u8 = 0;
pub const EMPYREAN_COVARIANCE_KIND_SECOND_ORDER: u8 = 1;
pub const EMPYREAN_COVARIANCE_KIND_THIRD_ORDER: u8 = 2;
pub const EMPYREAN_COVARIANCE_KIND_MIXTURE: u8 = 3;
/// Monte Carlo sample covariance. The run's RNG seed is carried in the
/// Monte-Carlo request config ([`EmpyreanUncertaintyMethod::mc_seed`]),
/// not in this per-row tag.
pub const EMPYREAN_COVARIANCE_KIND_MONTE_CARLO: u8 = 4;
/// Sigma-point sample covariance: the second moment of the propagated
/// canonical 2N+1 sigma-point set. Deterministic and parameter-free —
/// no per-row payload.
pub const EMPYREAN_COVARIANCE_KIND_SIGMA_POINT: u8 = 5;

// ── Covariance definiteness (TaggedCovariance.quality) ──────────────
/// All eigenvalues positive within round-off; `quality_min_eig` is NaN.
pub const EMPYREAN_COVARIANCE_QUALITY_POSITIVE_DEFINITE: u8 = 0;
/// At least one meaningfully negative eigenvalue; `quality_min_eig` is
/// that smallest (negative) eigenvalue.
pub const EMPYREAN_COVARIANCE_QUALITY_INDEFINITE: u8 = 1;
/// Explicitly repaired to positive semi-definite; `quality_min_eig` is
/// the smallest eigenvalue *before* repair. Never emitted by the
/// `_cartesian` accessor (which only classifies, never repairs).
pub const EMPYREAN_COVARIANCE_QUALITY_REPAIRED: u8 = 2;

// ── Target functional (TaggedCovariance.target_functional) ──────────
/// Generic Cartesian-state second moment.
pub const EMPYREAN_TARGET_FUNCTIONAL_CARTESIAN_STATE: u8 = 0;
/// Tied to the close-approach miss-distance functional — NOT a generic
/// state σ. Never emitted by the `_cartesian` accessor.
pub const EMPYREAN_TARGET_FUNCTIONAL_CLOSE_APPROACH_MISS_DISTANCE: u8 = 1;

// ── Tagged-covariance accessor return codes ─────────────────────────
/// Success.
pub const EMPYREAN_TAGGED_COV_OK: i32 = 0;
/// A required pointer argument was null.
pub const EMPYREAN_TAGGED_COV_NULL_POINTER: i32 = -1;
/// `orbit_index` is out of range.
pub const EMPYREAN_TAGGED_COV_ORBIT_INDEX_OUT_OF_RANGE: i32 = -2;
/// The input orbit carried no covariance (nothing to propagate forward).
pub const EMPYREAN_TAGGED_COV_NO_INITIAL_COVARIANCE: i32 = -3;
/// No dense trajectory — re-run with `advanced.cache_integrator_steps =
/// true` (or `events.dense_output = true`).
pub const EMPYREAN_TAGGED_COV_NO_DENSE_TRAJECTORY: i32 = -4;
/// No state at a requested epoch (or the series / state grids disagree).
pub const EMPYREAN_TAGGED_COV_STATE_MISSING: i32 = -5;
/// Element-space transform failed (reserved; not hit by the Cartesian path).
pub const EMPYREAN_TAGGED_COV_TRANSFORM: i32 = -6;
/// Nested uncertainty error (e.g. order / Monte-Carlo unavailable), or a
/// basis origin that has no NAIF id (an observatory).
pub const EMPYREAN_TAGGED_COV_UNCERTAINTY: i32 = -7;
/// `epoch_index` is out of range (point accessor only).
pub const EMPYREAN_TAGGED_COV_EPOCH_INDEX_OUT_OF_RANGE: i32 = -8;
/// A sample-based epoch (sigma-point) has no stored covariance on its
/// propagated state — an internal bookkeeping error, surfaced rather
/// than degraded.
pub const EMPYREAN_TAGGED_COV_SAMPLE_COVARIANCE_MISSING: i32 = -9;
/// A panic was caught at the boundary.
pub const EMPYREAN_TAGGED_COV_PANIC: i32 = -99;

/// A single propagated Cartesian state.
#[repr(C)]
pub struct EmpyreanPropagatedState {
    pub epoch_mjd_tdb: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub origin: i32,
    pub frame: i32,
    pub covariance: [[f64; 6]; 6],
    pub has_covariance: u8,
    /// State Transition Matrix Φ(t, t₀). Zero-filled when `has_stm` is 0.
    pub stm: [[f64; 6]; 6],
    pub has_stm: u8,
    /// State Transition Tensor Ψ(t, t₀). Zero-filled when `has_stt` is 0.
    pub stt: [[[f64; 6]; 6]; 6],
    pub has_stt: u8,
    /// Resolved [`CovarianceKind`](empyrean_core::propagation::CovarianceKind)
    /// at this output epoch — see `EMPYREAN_COVARIANCE_KIND_*`.
    /// Defaults to `LINEAR` for non-Auto methods and for Auto epochs
    /// outside CA windows.
    pub resolved_kind: u8,
}

/// A detected dynamical event from propagation.
///
/// For `event_type = "covariance_regime_change"` the audit-trail
/// fields (`previous_kind`, `resolved_kind`, `kappa`,
/// `threshold_below`, `threshold_above`) carry the
/// [`CovarianceRegimeChange`](empyrean_core::propagation::events::DynamicalEvent::CovarianceRegimeChange)
/// payload and the body / distance / velocity fields are zeroed
/// (regime changes don't carry a body or geometry — they're audit
/// markers from
/// [`UncertaintyMethod::Auto`](empyrean_core::propagation::UncertaintyMethod::Auto)).
/// For all other events these fields are sentinel-filled
/// (`0xFF` / NaN).
///
/// Field names mirror empyrean-core's variant fields for the
/// regime-change payload — no per-variant prefix.
#[repr(C)]
pub struct EmpyreanEvent {
    pub event_type: *mut std::ffi::c_char,
    pub orbit_id: *mut std::ffi::c_char,
    /// Object identifier looked up by `orbit_id` from the input batch.
    /// Empty string when the input had no `object_id`. Owning C string —
    /// caller frees via `*_result_free`.
    pub object_id: *mut std::ffi::c_char,
    pub body: *mut std::ffi::c_char,
    pub body_naif_id: i32,
    pub epoch_mjd_tdb: f64,
    pub distance_au: f64,
    pub distance_km: f64,
    pub relative_velocity_au_day: f64,
    /// `CovarianceRegimeChange` payload: previous resolved kind.
    /// Sentinel `0xFF` for non-regime events.
    pub previous_kind: u8,
    /// `CovarianceRegimeChange` payload: new resolved kind.
    /// Sentinel `0xFF` for non-regime events.
    pub resolved_kind: u8,
    /// `CovarianceRegimeChange` payload: local nonlinearity κ
    /// recorded at the CA. NaN for non-regime events.
    pub kappa: f64,
    /// `CovarianceRegimeChange` payload: lower κ value recorded in
    /// this audit payload. NaN for non-regime events.
    pub threshold_below: f64,
    /// `CovarianceRegimeChange` payload: upper κ value recorded in
    /// this audit payload. NaN for non-regime events.
    pub threshold_above: f64,
    /// `CaptureStart` / `CaptureEnd` payload: two-body energy w.r.t. the
    /// capturing body (AU²/day²). NaN for non-capture events.
    pub two_body_energy: f64,
    /// `CaptureStart` / `CaptureEnd` payload: Jacobi constant in the
    /// CR3BP. NaN for non-capture events or when unavailable.
    pub jacobi_constant: f64,
    /// `CaptureStart` / `CaptureEnd` payload: 1σ uncertainty on the
    /// Jacobi constant. NaN when unavailable.
    pub jacobi_constant_sigma: f64,
    /// `CaptureStart` / `CaptureEnd` payload: Jacobi constant at the L1
    /// gateway. NaN when unavailable.
    pub jacobi_constant_l1: f64,
    /// `CaptureStart` / `CaptureEnd` payload: Jacobi constant at the L2
    /// gateway. NaN when unavailable.
    pub jacobi_constant_l2: f64,
    /// `CaptureEnd` payload: number of periapsis passages during the
    /// temporary capture. `-1` sentinel for non-`CaptureEnd` events.
    pub n_periapses: i32,
    /// `Impact` payload: planetodetic latitude of the surface intercept
    /// (degrees). NaN for non-impact events or when unresolved.
    pub impact_latitude_deg: f64,
    /// `Impact` payload: planetodetic longitude of the surface intercept
    /// (degrees). NaN for non-impact events or when unresolved.
    pub impact_longitude_deg: f64,
    /// `Impact` payload: altitude of the surface intercept above the
    /// reference ellipsoid (km). NaN for non-impact events or when
    /// unresolved.
    pub impact_altitude_km: f64,
    /// `ShadowEntry` / `ShadowExit` payload: fraction of the Sun's disk
    /// occulted by the body (0 = none, 1 = full umbra). NaN for
    /// non-shadow events.
    pub shadow_fraction: f64,
    /// `ShadowEntry` / `ShadowExit` payload: fraction of incident
    /// sunlight reaching the particle (1 = full sun, 0 = total eclipse).
    /// NaN for non-shadow events.
    pub illumination: f64,
    /// `Periapsis` payload: relative position w.r.t. the approached body
    /// (AU), components x/y/z. NaN for non-periapsis events.
    pub relative_x: f64,
    pub relative_y: f64,
    pub relative_z: f64,
    /// `Periapsis` payload: relative velocity w.r.t. the approached body
    /// (AU/day), components vx/vy/vz. NaN for non-periapsis events.
    pub relative_vx: f64,
    pub relative_vy: f64,
    pub relative_vz: f64,
    /// `PossibleImpact` payload: effective capture radius with
    /// gravitational focusing (AU / km). NaN for non-possible-impact events.
    pub effective_radius_au: f64,
    pub effective_radius_km: f64,
    /// `PossibleImpact` payload: 1σ uncertainty along the miss direction
    /// (AU). NaN for non-possible-impact events.
    pub sigma_distance_au: f64,
    /// `PossibleImpact` payload: linear (STM-mapped) impact probability.
    /// NaN for non-possible-impact events.
    pub ip_linear: f64,
    /// `PossibleImpact` payload: second-order (Edgeworth) impact
    /// probability. NaN for non-possible-impact / first-order runs.
    pub ip_second_order: f64,
    /// `PossibleImpact` payload: local nonlinearity κ at the encounter.
    /// NaN when unavailable.
    pub nonlinearity: f64,
    /// `PossibleImpact` payload: adaptive-Gaussian-mixture impact
    /// probability. NaN when not an AGM run.
    pub ip_agm: f64,
    /// `PossibleImpact` payload: Monte-Carlo impact probability. NaN when
    /// not a Monte-Carlo run.
    pub ip_mc: f64,
}

/// One Gaussian sub-component of an AGM mixture decomposition.
///
/// Mirrors
/// [`empyrean_core::propagation::MixtureComponent`]. The mean carries
/// the *propagated* sub-Gaussian centroid at the CA epoch (f64 from
/// the Jet2 integrator); the covariance is the linearly-mapped
/// \\(\Phi \, \Sigma_k \, \Phi^\top\\) over the same propagation
/// segment. A consumer can evaluate
/// \\(\sum_k w_k \\, \mathcal{N}(x \mid \mu_k, \Sigma_k)\\) directly
/// at the CA epoch.
#[repr(C)]
pub struct EmpyreanMixtureComponent {
    pub weight: f64,
    pub mean: [f64; 6],
    pub covariance: [[f64; 6]; 6],
}

/// Per-orbit AGM mixture decomposition retained by Auto / Mixture.
///
/// Mirrors [`empyrean_core::propagation::MixtureChain`] in flat form.
/// `ca_epochs_mjd_tdb` and `components_per_epoch` are
/// `num_ca_epochs`-length parallel arrays. Components are flattened
/// into a single `components` array — for CA index `k`, the slice is
/// `components[components_offset[k] .. components_offset[k] +
/// components_per_epoch[k]]`. `components_offset[k]` is the
/// prefix-sum of `components_per_epoch[0..k]` (so
/// `components_offset[0] == 0`).
///
/// Only orbits whose AGM splits actually fired carry a chain;
/// non-mixture orbits get one entry whose `num_ca_epochs == 0` and
/// whose pointers are null.
#[repr(C)]
pub struct EmpyreanMixtureChain {
    /// Owning C string — caller frees via `*_result_free`.
    pub orbit_id: *mut std::ffi::c_char,
    /// `num_ca_epochs`-length array of CA epochs (mjd_tdb).
    pub ca_epochs_mjd_tdb: *mut f64,
    pub num_ca_epochs: usize,
    /// `num_ca_epochs`-length array of per-epoch component counts.
    pub components_per_epoch: *mut usize,
    /// `num_ca_epochs`-length array of starting offsets into the
    /// flattened `components` array. `components_offset[k]` is the
    /// prefix-sum of `components_per_epoch[0..k]`.
    pub components_offset: *mut usize,
    /// Flattened components — total length =
    /// sum(components_per_epoch).
    pub components: *mut EmpyreanMixtureComponent,
    pub num_components_total: usize,
}

/// Propagation result containing states, events, and object identifiers.
///
/// `mixtures` parallels `object_ids` only when populated — it is
/// per-orbit (not per-state), so its length is the distinct orbit
/// count, not `num_states`.
#[repr(C)]
pub struct EmpyreanPropagationResult {
    pub states: *mut EmpyreanPropagatedState,
    pub num_states: usize,
    pub object_ids: *mut *mut std::ffi::c_char,
    pub events: *mut EmpyreanEvent,
    pub num_events: usize,
    /// One [`EmpyreanMixtureChain`] per input orbit (positional with
    /// the input orbit batch). Empty / null when no orbits produced
    /// mixtures (the typical case for FirstOrder / SecondOrder /
    /// SigmaPoint / MonteCarlo).
    pub mixtures: *mut EmpyreanMixtureChain,
    pub num_mixtures: usize,
    /// Opaque retained handle to the rich propagation result, enabling
    /// the on-demand tagged-covariance accessors
    /// (`empyrean_propagation_covariance_series_cartesian` /
    /// `_covariance_at_cartesian`). **Do not dereference from C** — it is
    /// freed by `empyrean_propagation_result_free`. Null when the
    /// propagation produced no retainable result.
    pub lazy_handle: *mut std::ffi::c_void,
}

// ── Helpers ─────────────────────────────────────────────────

pub(crate) fn int_to_force_model(val: i32) -> Result<empyrean_core::ForceModelTier, String> {
    match val {
        0 => Ok(empyrean_core::ForceModelTier::Approximate),
        1 => Ok(empyrean_core::ForceModelTier::Basic),
        2 => Ok(empyrean_core::ForceModelTier::Standard),
        3 => Err("ForceModelTier::Full is not exposed in v0.7.0 — pass 2 for Standard tier".into()),
        _ => Err(format!("unknown force model tier: {val}")),
    }
}

/// Map [`empyrean_core::propagation::CovarianceKind`] to the C ABI
/// `EMPYREAN_COVARIANCE_KIND_*` numeric tag.
pub(crate) fn covariance_kind_to_u8(k: empyrean_core::propagation::CovarianceKind) -> u8 {
    use empyrean_core::propagation::CovarianceKind as K;
    match k {
        K::Linear => EMPYREAN_COVARIANCE_KIND_LINEAR,
        K::SecondOrder => EMPYREAN_COVARIANCE_KIND_SECOND_ORDER,
        K::ThirdOrder => EMPYREAN_COVARIANCE_KIND_THIRD_ORDER,
        K::Mixture => EMPYREAN_COVARIANCE_KIND_MIXTURE,
        K::MonteCarlo { .. } => EMPYREAN_COVARIANCE_KIND_MONTE_CARLO,
        K::SigmaPoint => EMPYREAN_COVARIANCE_KIND_SIGMA_POINT,
    }
}

pub(crate) fn flat_to_uncertainty_method(
    c: &EmpyreanUncertaintyMethod,
) -> Result<UncertaintyMethod, String> {
    match c.tag {
        EMPYREAN_UNCERTAINTY_FIRST => Ok(UncertaintyMethod::First),
        EMPYREAN_UNCERTAINTY_SECOND => Ok(UncertaintyMethod::Second),
        EMPYREAN_UNCERTAINTY_SIGMA_POINT => Ok(UncertaintyMethod::SigmaPoint {
            n_sigma: c.sp_n_sigma,
            samples_per_plane: c.sp_samples_per_plane as usize,
        }),
        EMPYREAN_UNCERTAINTY_MONTE_CARLO => Ok(UncertaintyMethod::MonteCarlo {
            n_samples: c.mc_n_samples as usize,
            seed: if c.mc_seed_some != 0 {
                Some(c.mc_seed)
            } else {
                None
            },
        }),
        EMPYREAN_UNCERTAINTY_AUTO => Ok(UncertaintyMethod::Auto {
            threshold_first: c.auto_threshold_first,
            threshold_mixture: c.auto_threshold_mixture,
            threshold_ip_skip: c.auto_threshold_ip_skip,
            gmm_max_depth: c.auto_gmm_max_depth as usize,
            gmm_components_per_split: c.auto_gmm_components_per_split as usize,
        }),
        other => Err(format!("unknown uncertainty method tag: {other}")),
    }
}

fn build_advanced_from_c(c: &EmpyreanAdvancedIntegratorConfig) -> AdvancedIntegratorConfig {
    let mut a = AdvancedIntegratorConfig::default();
    a.integrator = match c.integrator {
        EMPYREAN_INTEGRATOR_DOP853 => IntegratorChoice::DOP853,
        // GR15 default. Anything else (including the GR15 tag itself
        // and any unrecognised value) falls through here. IAS15 is
        // intentionally not exposed at the C-ABI surface.
        _ => IntegratorChoice::GR15,
    };
    if c.epsilon > 0.0 {
        a.epsilon = c.epsilon;
    }
    if c.dt_initial.is_finite() {
        a.dt_initial = Some(c.dt_initial);
    }
    if c.dt_min.is_finite() {
        a.dt_min = Some(c.dt_min);
    }
    if c.encounter_timescale_divisor > 0.0 {
        a.encounter_timescale_divisor = c.encounter_timescale_divisor;
    }
    if c.max_steps > 0 {
        a.max_steps = c.max_steps;
    }
    if c.max_dense_steps > 0 {
        a.max_dense_steps = c.max_dense_steps;
    }
    a.cache_integrator_steps = c.cache_integrator_steps != 0;
    // Tri-state origin-switching enabled tag: 0 = DEFAULT (resolves to
    // upstream default — `memset(0)` should give the right behavior;
    // matches the dt_min / hysteresis / epsilon sentinel convention),
    // 1 = ON, 2 = OFF. Unknown values fall back to DEFAULT.
    // Previously the C ABI used `enabled !=
    // 0` which made `memset(0)` mean "explicitly off," contradicting
    // the wrapper's default = true and silently disabling switching
    // for any external C consumer who didn't know to set the field.
    let enabled = match c.origin_switching.enabled {
        EMPYREAN_ORIGIN_SWITCHING_ON => true,
        EMPYREAN_ORIGIN_SWITCHING_OFF => false,
        _ => OriginSwitchingConfig::default().enabled, // DEFAULT + unknown
    };
    a.origin_switching = OriginSwitchingConfig {
        enabled,
        hysteresis: if c.origin_switching.hysteresis > 0.0 {
            c.origin_switching.hysteresis
        } else {
            // Use the upstream default (0.2) when the C caller passed
            // 0.0 — matches the sentinel convention used elsewhere.
            OriginSwitchingConfig::default().hysteresis
        },
        // Per-body opt-in not yet exposed at the C-ABI surface;
        // EMPYREAN_ORIGIN_SWITCHING_ON selects all monitored bodies.
        bodies: None,
    };
    a
}

fn build_diagnostics_from_c(c: &EmpyreanDiagnosticsConfig) -> DiagnosticsConfig {
    let mut d = DiagnosticsConfig::default();
    d.sensitivity = c.sensitivity != 0;
    d.nonlinearity = c.nonlinearity != 0;
    d.lyapunov = c.lyapunov != 0;
    d.keyholes = c.keyholes != 0;
    d.bifurcations = c.bifurcations != 0;
    if c.sample_stride > 0 {
        d.sample_stride = c.sample_stride;
    }
    d.sensitivity_threshold = if c.sensitivity_threshold.is_finite() {
        Some(c.sensitivity_threshold)
    } else {
        None
    };
    d.lyapunov_threshold = if c.lyapunov_threshold.is_finite() {
        Some(c.lyapunov_threshold)
    } else {
        None
    };
    d.nonlinearity_threshold = if c.nonlinearity_threshold.is_finite() {
        Some(c.nonlinearity_threshold)
    } else {
        None
    };
    d
}

fn build_event_config_from_c(c: &EmpyreanEventConfig) -> Result<EventConfig, String> {
    let mut e = EventConfig::default();
    e.close_approaches = c.close_approaches != 0;
    e.impacts = c.impacts != 0;
    e.atmospheric = c.atmospheric != 0;
    e.possible_impacts = c.possible_impacts != 0;
    e.shadow_events = c.shadow_events != 0;
    e.dense_output = c.dense_output != 0;
    if c.dense_output_cadence_days > 0.0 {
        e.dense_output_cadence_days = c.dense_output_cadence_days;
    }
    if c.num_body_filter > 0 && !c.body_filter_naif.is_null() {
        let slice = unsafe { std::slice::from_raw_parts(c.body_filter_naif, c.num_body_filter) };
        let mut filter = Vec::with_capacity(slice.len());
        for &naif in slice {
            let origin = Origin::from_naif_id(naif)
                .ok_or_else(|| format!("unknown NAIF id in body_filter: {naif}"))?;
            filter.push(origin);
        }
        e.body_filter = Some(filter);
    } else {
        e.body_filter = None;
    }
    Ok(e)
}

/// Build a [`PropagationConfig`] from a C-ABI config, honouring the
/// shared sentinel rules (0 / 0.0 → upstream default; NaN ↔ None for
/// optional float fields).
pub(crate) fn build_propagation_config_from_c(
    c: &EmpyreanPropagationConfig,
) -> Result<PropagationConfig, String> {
    let mut cfg = PropagationConfig::default();

    cfg.force_model = int_to_force_model(c.force_model)?.into();
    if c.num_excluded_perturbers > 0 && !c.excluded_perturbers_naif.is_null() {
        let slice = unsafe {
            std::slice::from_raw_parts(c.excluded_perturbers_naif, c.num_excluded_perturbers)
        };
        let mut filter = Vec::with_capacity(slice.len());
        for &naif in slice {
            let origin = Origin::from_naif_id(naif)
                .ok_or_else(|| format!("unknown NAIF id in excluded_perturbers: {naif}"))?;
            filter.push(origin);
        }
        cfg.excluded_perturbers = filter;
    }

    cfg.uncertainty_method = flat_to_uncertainty_method(&c.uncertainty_method)?.into();
    cfg.compute_stm = c.compute_stm != 0;

    cfg.frame = empyrean_core::convert::int_to_frame(c.frame).map_err(|e| e.to_string())?;
    cfg.events = build_event_config_from_c(&c.events)?;
    cfg.diagnostics = build_diagnostics_from_c(&c.diagnostics);

    cfg.num_threads = std::num::NonZeroUsize::new(c.num_threads);
    cfg.advanced = build_advanced_from_c(&c.advanced);

    Ok(cfg)
}

/// Build the non-gravitational parameters carried by an [`EmpyreanOrbit`],
/// or `None` when all of A1/A2/A3 are zero (pure gravity).
///
/// Shared by every C-ABI path that turns an `EmpyreanOrbit` into an
/// `Orbits<AU>` (propagation, orbit determination, ephemeris) so that none
/// of them silently drops the caller's non-grav model. All-zero g(r) fields
/// (`ng_alpha`..`ng_k`) select the inverse-square default (asteroid
/// Yarkovsky); any non-zero value selects the explicit SBDB Marsden–Sekanina
/// g(r). A non-finite `non_grav_dt` means "no thermal-lag delay".
pub(crate) fn empyrean_orbit_non_grav_params(orbit: &EmpyreanOrbit) -> Option<NonGravParams> {
    if orbit.a1 == 0.0 && orbit.a2 == 0.0 && orbit.a3 == 0.0 {
        return None;
    }
    let g_func = if orbit.ng_alpha == 0.0
        && orbit.ng_r0 == 0.0
        && orbit.ng_m == 0.0
        && orbit.ng_n == 0.0
        && orbit.ng_k == 0.0
    {
        GFunction::inverse_square()
    } else {
        GFunction::from_sbdb(
            orbit.ng_alpha,
            orbit.ng_r0,
            orbit.ng_m,
            orbit.ng_n,
            orbit.ng_k,
        )
    };
    Some(NonGravParams {
        a1: orbit.a1,
        a2: orbit.a2,
        a3: orbit.a3,
        model: NonGravModel::MarsdenSekanina(g_func),
        // Carry the non-grav prior covariance back in when present so a
        // fitted orbit re-feeds into a StateAndNonGrav refine.
        covariance: if orbit.has_non_grav_covariance != 0 {
            Some(orbit.non_grav_covariance)
        } else {
            None
        },
        dt: if orbit.non_grav_dt.is_finite() {
            Some(orbit.non_grav_dt)
        } else {
            None
        },
        // DT is a fittable axis in v1.20.0; propagation input carries the DT
        // value (above); carry its prior variance too when supplied so a
        // re-fed orbit opens + priors the DT column in a StateAndNonGravAndDT
        // refine.
        dt_variance: if orbit.non_grav_dt_variance.is_finite() && orbit.non_grav_dt_variance > 0.0 {
            Some(orbit.non_grav_dt_variance)
        } else {
            None
        },
    })
}

/// Validate a caller-owned side-array pointer/length pair: a non-zero
/// length paired with a null pointer (in either direction) is a loud
/// argument error rather than a silent empty read. A zero length is always
/// treated as "no array" (the pointer is ignored).
fn validate_side_array_ptr<T>(ptr: *const T, len: usize, field: &str) -> Result<(), String> {
    if len > 0 && ptr.is_null() {
        return Err(format!(
            "{field}: non-zero length ({len}) with a null pointer"
        ));
    }
    Ok(())
}

/// Build the continuous-thrust parameters carried by an [`EmpyreanOrbit`],
/// or `None` when the orbit supplies no thrust arcs.
///
/// Shared by every C-ABI path that turns an `EmpyreanOrbit` into an
/// `Orbits<AU>` (propagation, ephemeris, radar/planning, impact, IO
/// round-trip) so none of them silently drops the caller's burn model.
///
/// Marshals the [`thrust_arcs`](EmpyreanOrbit::thrust_arcs) side array into
/// [`ThrustArc`]s (mapping [`isp_s`](EmpyreanThrustArc::isp_s) `NaN → None`,
/// the [`steering_law`](EmpyreanThrustArc::steering_law) tag into a
/// [`SteeringLaw`], and `central_body_naif_id` into an [`Origin`]), and
/// carries the optional `dv_corrections` / `correction_covariances` side
/// arrays through unchanged. Validates loudly:
/// - any pointer/length mismatch (non-zero length with a null pointer),
/// - `dv_corrections` / `correction_covariances` supplied with no arcs,
/// - `correction_covariances` length not matching `dv_corrections`,
/// - an unknown `steering_law` tag or an unknown `central_body_naif_id`.
pub(crate) fn empyrean_orbit_thrust_params(
    orbit: &EmpyreanOrbit,
) -> Result<Option<ThrustParams>, String> {
    validate_side_array_ptr(orbit.thrust_arcs, orbit.n_thrust_arcs, "thrust_arcs")?;
    validate_side_array_ptr(
        orbit.dv_corrections,
        orbit.n_dv_corrections,
        "dv_corrections",
    )?;
    validate_side_array_ptr(
        orbit.correction_covariances,
        orbit.n_correction_covariances,
        "correction_covariances",
    )?;

    if orbit.n_thrust_arcs == 0 {
        // No arcs: corrections without an arc to attach them to are a loud
        // error, not a silent drop.
        if orbit.n_dv_corrections != 0 || orbit.n_correction_covariances != 0 {
            return Err(
                "dv_corrections / correction_covariances supplied without any thrust_arcs"
                    .to_string(),
            );
        }
        return Ok(None);
    }

    // villeneuve's ThrustParams contract: when correction_covariances is
    // non-empty its length must match dv_corrections.
    if orbit.n_correction_covariances != 0
        && orbit.n_correction_covariances != orbit.n_dv_corrections
    {
        return Err(format!(
            "correction_covariances length ({}) must match dv_corrections length ({})",
            orbit.n_correction_covariances, orbit.n_dv_corrections
        ));
    }

    let arc_slice = unsafe { std::slice::from_raw_parts(orbit.thrust_arcs, orbit.n_thrust_arcs) };
    let mut arcs = Vec::with_capacity(arc_slice.len());
    for (i, a) in arc_slice.iter().enumerate() {
        let steering = match a.steering_law {
            EMPYREAN_STEERING_LAW_CONSTANT_RTN => SteeringLaw::ConstantRTN {
                alpha_rad: a.steering_alpha_rad,
                beta_rad: a.steering_beta_rad,
            },
            EMPYREAN_STEERING_LAW_VELOCITY_TANGENT => SteeringLaw::VelocityTangent,
            EMPYREAN_STEERING_LAW_INERTIAL_FIXED => SteeringLaw::InertialFixed {
                direction: a.steering_direction,
            },
            other => {
                return Err(format!("thrust arc {i}: unknown steering_law tag {other}"));
            }
        };
        let central_body = Origin::from_naif_id(a.central_body_naif_id).ok_or_else(|| {
            format!(
                "thrust arc {i}: unknown central_body NAIF id {}",
                a.central_body_naif_id
            )
        })?;
        arcs.push(ThrustArc {
            start_mjd_tdb: a.start_mjd_tdb,
            end_mjd_tdb: a.end_mjd_tdb,
            thrust_n: a.thrust_n,
            mass_kg: a.mass_kg,
            isp_s: if a.isp_s.is_finite() {
                Some(a.isp_s)
            } else {
                None
            },
            steering,
            sharpness: a.sharpness,
            central_body,
        });
    }

    let dv_corrections = if orbit.n_dv_corrections > 0 {
        unsafe { std::slice::from_raw_parts(orbit.dv_corrections, orbit.n_dv_corrections) }.to_vec()
    } else {
        Vec::new()
    };
    let correction_covariances = if orbit.n_correction_covariances > 0 {
        unsafe {
            std::slice::from_raw_parts(orbit.correction_covariances, orbit.n_correction_covariances)
        }
        .to_vec()
    } else {
        Vec::new()
    };

    Ok(Some(ThrustParams {
        arcs,
        dv_corrections,
        correction_covariances,
    }))
}

// ── empyrean_propagate ──────────────────────────────────────

/// Propagate orbits to the requested target times.
///
/// Returns 0 on success, negative error code on failure.
/// On success, `result_out` is populated with the propagated states.
/// The caller must free the result with `empyrean_propagation_result_free()`.
///
/// States are flat in orbit-major order; within each orbit, rows are in
/// **ascending epoch order, always** (engine guarantee since villeneuve
/// v1.18.0), regardless of request order. Positional pairing against an
/// ascending, duplicate-free request grid is exact; for any other
/// request shape, join on `epoch_mjd_tdb`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_propagate(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    times_ptr: *const f64,
    num_times: usize,
    config: *const EmpyreanPropagationConfig,
    result_out: *mut EmpyreanPropagationResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbits_ptr.is_null()
            || times_ptr.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let config_ref = unsafe { &*config };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let times_slice = unsafe { std::slice::from_raw_parts(times_ptr, num_times) };

        // Build Orbits<AU> row by row (non-grav + thrust attached). Shared
        // with the BuiltSystem handle path so both marshal the caller's
        // dynamics model through the identical code.
        let (orbits, input_orbit_ids, input_object_ids) =
            match build_orbits_for_propagation(orbit_slice) {
                Ok(t) => t,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            };

        let cfg = match build_propagation_config_from_c(config_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let times: Vec<Epoch> = times_slice
            .iter()
            .map(|&t| Epoch::from_mjd_tdb(t))
            .collect();

        let prop_result = match propagate(ctx_ref, &orbits, &times, &cfg) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return -2;
            }
        };

        marshal_propagation_result(
            prop_result,
            times.len(),
            &input_orbit_ids,
            &input_object_ids,
            result_out,
        )
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_propagate");
            -99
        }
    }
}

/// Build an `Orbits<AU>` batch (plus per-orbit `orbit_id` / `object_id`
/// vectors) from a C-ABI orbit slice, attaching each row's non-grav and
/// thrust parameters. Shared by the one-shot [`empyrean_propagate`] and the
/// handle-based
/// [`empyrean_builtsystem_propagate`](crate::built_system::empyrean_builtsystem_propagate)
/// so both marshal the caller's dynamics model through the identical code
/// (no silent divergence between the two entry points). The returned tuple is
/// `(orbits, orbit_ids, object_ids)`.
#[allow(clippy::type_complexity)]
pub(crate) fn build_orbits_for_propagation(
    orbit_slice: &[EmpyreanOrbit],
) -> Result<
    (
        Orbits<empyrean_core::coordinates::AU>,
        Vec<String>,
        Vec<String>,
    ),
    String,
> {
    let mut orbits: Orbits<empyrean_core::coordinates::AU> = Orbits::empty();
    // Per-orbit identifiers. We use these to:
    //   1. Push the user-supplied orbit_id into villeneuve's batch
    //      (instead of fabricating "orbit_N"), so events emerge
    //      tagged with the user's string.
    //   2. Look up object_id by orbit_id when constructing C ABI
    //      result rows (events / states), since villeneuve doesn't
    //      track object_id internally.
    let mut input_orbit_ids: Vec<String> = Vec::with_capacity(orbit_slice.len());
    let mut input_object_ids: Vec<String> = Vec::with_capacity(orbit_slice.len());
    for (i, orbit) in orbit_slice.iter().enumerate() {
        let state = orbit.state.to_empyrean();
        let coords =
            coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit {i}: {e}"))?;
        // Use the caller-supplied orbit_id when present; fall back
        // to the positional fabrication only if null/empty so older
        // callers that haven't set the field keep working.
        let id = c_str_to_string(orbit.orbit_id).unwrap_or_default();
        let id = if id.is_empty() {
            format!("orbit_{i}")
        } else {
            id
        };
        let obj = c_str_to_string(orbit.object_id).unwrap_or_default();
        input_orbit_ids.push(id.clone());
        input_object_ids.push(obj);
        orbits
            .push(id, coords.into_radians())
            .map_err(|e| format!("orbit {i}: {e}"))?;
        if let Some(params) = empyrean_orbit_non_grav_params(orbit) {
            orbits.set_non_grav_params(i, Some(params));
        }
        match empyrean_orbit_thrust_params(orbit) {
            Ok(Some(tp)) => orbits.set_thrust_params(i, Some(tp)),
            Ok(None) => {}
            Err(e) => return Err(format!("orbit {i}: {e}")),
        }
    }
    Ok((orbits, input_orbit_ids, input_object_ids))
}

/// Marshal a [`PropagationResult`] into the C-ABI
/// [`EmpyreanPropagationResult`] (states + object ids + events + mixture
/// chains + the lazy tagged-covariance handle). Shared verbatim by the
/// one-shot [`empyrean_propagate`] and the handle-based
/// [`empyrean_builtsystem_propagate`](crate::built_system::empyrean_builtsystem_propagate)
/// so both emit byte-identical result buffers. Returns 0 on success or a
/// negative allocation-failure code. Runs inside the caller's
/// `catch_unwind`, so it installs no panic guard of its own.
pub(crate) fn marshal_propagation_result(
    prop_result: PropagationResult,
    n_times: usize,
    input_orbit_ids: &[String],
    input_object_ids: &[String],
    result_out: *mut EmpyreanPropagationResult,
) -> i32 {
    let n = prop_result.states.len();

    let states_layout = std::alloc::Layout::array::<EmpyreanPropagatedState>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanPropagatedState>());
    let states_ptr = if n > 0 {
        let ptr = unsafe { std::alloc::alloc(states_layout) } as *mut EmpyreanPropagatedState;
        if ptr.is_null() {
            set_last_error("allocation failed for states array");
            return -5;
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    let ids_ptr = if n > 0 {
        let layout = std::alloc::Layout::array::<*mut std::ffi::c_char>(n)
            .unwrap_or(std::alloc::Layout::new::<*mut std::ffi::c_char>());
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut std::ffi::c_char;
        if ptr.is_null() {
            set_last_error("allocation failed for object_ids array");
            unsafe { std::alloc::dealloc(states_ptr as *mut u8, states_layout) };
            return -5;
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    for (i, (orbit_id, coord, cov)) in prop_result.states.iter().enumerate() {
        let (covariance, has_covariance) = match cov {
            Some(c) => (*c, 1u8),
            None => ([[0.0; 6]; 6], 0u8),
        };

        let orbit_idx = if n_times > 0 { i / n_times } else { 0 };
        let time_idx = if n_times > 0 { i % n_times } else { 0 };
        let chain = prop_result.sensitivity.get(orbit_idx);
        let (stm, has_stm) = match chain.and_then(|c| c.stm(time_idx)) {
            Some(m) => (m.matrix, 1u8),
            None => ([[0.0; 6]; 6], 0u8),
        };
        let (stt, has_stt) = match chain.and_then(|c| c.stt(time_idx)) {
            Some(t) => (t.tensor, 1u8),
            None => ([[[0.0; 6]; 6]; 6], 0u8),
        };
        let resolved_kind = chain
            .and_then(|c| c.resolved_kinds().get(time_idx).copied())
            .map(covariance_kind_to_u8)
            .unwrap_or(EMPYREAN_COVARIANCE_KIND_LINEAR);

        let out_state = EmpyreanPropagatedState {
            epoch_mjd_tdb: coord.t.mjd_tdb(),
            x: coord.x,
            y: coord.y,
            z: coord.z,
            vx: coord.vx,
            vy: coord.vy,
            vz: coord.vz,
            origin: coord.origin.naif_id(),
            frame: frame_to_int(coord.frame),
            covariance,
            has_covariance,
            stm,
            has_stm,
            stt,
            has_stt,
            resolved_kind,
        };

        unsafe {
            states_ptr.add(i).write(out_state);
        }

        let c_id = CString::new(orbit_id).unwrap_or_else(|_| CString::new("?").unwrap());
        unsafe {
            ids_ptr.add(i).write(c_id.into_raw());
        }
    }

    // Pre-flatten through `event_to_c`. Variants the v0.7.0
    // distribution channel intentionally gates off (HighSensitivity,
    // ChaoticRegion, HighNonlinearity, Keyhole, Bifurcation —
    // diagnostic-only, payloads not yet shaped for the C ABI)
    // become `None` and are dropped here rather than collapsed
    // into a generic `event_type = "other"` row.
    let events_emitted: Vec<EmpyreanEvent> = {
        let object_id_map: std::collections::HashMap<String, String> = input_orbit_ids
            .iter()
            .zip(input_object_ids.iter())
            .map(|(o, x)| (o.clone(), x.clone()))
            .collect();
        prop_result
            .events
            .iter()
            .filter_map(|ev| event_to_c(ev, &object_id_map))
            .collect()
    };
    let n_ev = events_emitted.len();
    let events_ptr = if n_ev > 0 {
        let layout = std::alloc::Layout::array::<EmpyreanEvent>(n_ev)
            .unwrap_or(std::alloc::Layout::new::<EmpyreanEvent>());
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanEvent;
        if ptr.is_null() {
            set_last_error("allocation failed for events array");
            if !states_ptr.is_null() && n > 0 {
                unsafe { std::alloc::dealloc(states_ptr as *mut u8, states_layout) };
            }
            return -5;
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    for (i, out_ev) in events_emitted.into_iter().enumerate() {
        unsafe {
            events_ptr.add(i).write(out_ev);
        }
    }

    // Build per-orbit mixture chains. The empyrean-core /
    // villeneuve `result.mixtures` parallels `result.sensitivity`
    // — one entry per input orbit, `Some(MixtureChain)` only
    // when AGM actually fired. We surface ALL orbits (each gets
    // an `EmpyreanMixtureChain` row) so the consumer can do
    // a positional join with the input orbit batch; orbits
    // without mixtures get a row with `num_ca_epochs == 0`.
    let mixtures_count = prop_result.mixtures.len();
    let mixtures_ptr = if mixtures_count > 0 {
        let layout = std::alloc::Layout::array::<EmpyreanMixtureChain>(mixtures_count)
            .unwrap_or(std::alloc::Layout::new::<EmpyreanMixtureChain>());
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanMixtureChain;
        if ptr.is_null() {
            set_last_error("allocation failed for mixtures array");
            if !states_ptr.is_null() && n > 0 {
                let l = std::alloc::Layout::array::<EmpyreanPropagatedState>(n).unwrap();
                unsafe { std::alloc::dealloc(states_ptr as *mut u8, l) };
            }
            return -5;
        }
        for (mi, mixture_opt) in prop_result.mixtures.iter().enumerate() {
            // Look up this orbit's id by indexing into the
            // sensitivity vec — they're positionally aligned
            // with `mixtures` per villeneuve's contract.
            let orbit_id = input_orbit_ids.get(mi).map(String::as_str).unwrap_or("");
            let chain = match mixture_opt {
                Some(c) => c,
                None => {
                    unsafe {
                        ptr.add(mi).write(EmpyreanMixtureChain {
                            orbit_id: to_c_str(orbit_id),
                            ca_epochs_mjd_tdb: std::ptr::null_mut(),
                            num_ca_epochs: 0,
                            components_per_epoch: std::ptr::null_mut(),
                            components_offset: std::ptr::null_mut(),
                            components: std::ptr::null_mut(),
                            num_components_total: 0,
                        });
                    }
                    continue;
                }
            };
            let n_epochs = chain.epochs.len();
            let total_components: usize = chain.components.iter().map(|v| v.len()).sum();
            // Allocate the three parallel arrays + the flattened
            // components.
            let epochs_layout = std::alloc::Layout::array::<f64>(n_epochs).unwrap();
            let counts_layout = std::alloc::Layout::array::<usize>(n_epochs).unwrap();
            let offsets_layout = std::alloc::Layout::array::<usize>(n_epochs).unwrap();
            let comps_layout =
                std::alloc::Layout::array::<EmpyreanMixtureComponent>(total_components).unwrap();
            let epochs_p: *mut f64 = if n_epochs > 0 {
                (unsafe { std::alloc::alloc(epochs_layout) }) as *mut f64
            } else {
                std::ptr::null_mut()
            };
            let counts_p: *mut usize = if n_epochs > 0 {
                (unsafe { std::alloc::alloc(counts_layout) }) as *mut usize
            } else {
                std::ptr::null_mut()
            };
            let offsets_p: *mut usize = if n_epochs > 0 {
                (unsafe { std::alloc::alloc(offsets_layout) }) as *mut usize
            } else {
                std::ptr::null_mut()
            };
            let comps_p: *mut EmpyreanMixtureComponent = if total_components > 0 {
                (unsafe { std::alloc::alloc(comps_layout) }) as *mut EmpyreanMixtureComponent
            } else {
                std::ptr::null_mut()
            };
            let mut running_offset = 0usize;
            for (k, t) in chain.epochs.iter().enumerate() {
                let comps = &chain.components[k];
                unsafe {
                    epochs_p.add(k).write(*t);
                    counts_p.add(k).write(comps.len());
                    offsets_p.add(k).write(running_offset);
                }
                for (j, comp) in comps.iter().enumerate() {
                    unsafe {
                        comps_p
                            .add(running_offset + j)
                            .write(EmpyreanMixtureComponent {
                                weight: comp.weight,
                                mean: comp.mean,
                                covariance: comp.covariance,
                            });
                    }
                }
                running_offset += comps.len();
            }
            unsafe {
                ptr.add(mi).write(EmpyreanMixtureChain {
                    orbit_id: to_c_str(orbit_id),
                    ca_epochs_mjd_tdb: epochs_p,
                    num_ca_epochs: n_epochs,
                    components_per_epoch: counts_p,
                    components_offset: offsets_p,
                    components: comps_p,
                    num_components_total: total_components,
                });
            }
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    // Retain the rich result behind an opaque handle so the
    // on-demand tagged-covariance accessors can recompute the
    // resolved-kind readback and co-locate the per-epoch nominal
    // state. `prop_result` is moved here after all flattening
    // borrows have ended; freed by `empyrean_propagation_result_free`.
    let handle = Box::new(PropagationResultHandle {
        result: prop_result,
        n_times,
    });
    let handle_ptr = Box::into_raw(handle) as *mut std::ffi::c_void;

    unsafe {
        (*result_out).states = states_ptr;
        (*result_out).num_states = n;
        (*result_out).object_ids = ids_ptr;
        (*result_out).events = events_ptr;
        (*result_out).num_events = n_ev;
        (*result_out).mixtures = mixtures_ptr;
        (*result_out).num_mixtures = mixtures_count;
        (*result_out).lazy_handle = handle_ptr;
    }

    let _ = Origin::Earth; // reference Origin to suppress unused-import warnings
    0
}

/// Free a propagation result previously returned by `empyrean_propagate()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_propagation_result_free(result: *mut EmpyreanPropagationResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }

        let res = unsafe { &*result };
        let n = res.num_states;

        if !res.object_ids.is_null() {
            for i in 0..n {
                let ptr = unsafe { *res.object_ids.add(i) };
                if !ptr.is_null() {
                    drop(unsafe { CString::from_raw(ptr) });
                }
            }
            if n > 0 {
                let layout = std::alloc::Layout::array::<*mut std::ffi::c_char>(n).unwrap();
                unsafe {
                    std::alloc::dealloc(res.object_ids as *mut u8, layout);
                }
            }
        }

        if !res.states.is_null() && n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanPropagatedState>(n).unwrap();
            unsafe {
                std::alloc::dealloc(res.states as *mut u8, layout);
            }
        }

        let n_ev = res.num_events;
        if !res.events.is_null() && n_ev > 0 {
            for i in 0..n_ev {
                let ev = unsafe { &*res.events.add(i) };
                unsafe {
                    free_c_str(ev.event_type);
                    free_c_str(ev.orbit_id);
                    free_c_str(ev.object_id);
                    free_c_str(ev.body);
                }
            }
            let layout = std::alloc::Layout::array::<EmpyreanEvent>(n_ev).unwrap();
            unsafe {
                std::alloc::dealloc(res.events as *mut u8, layout);
            }
        }

        let n_mix = res.num_mixtures;
        if !res.mixtures.is_null() && n_mix > 0 {
            for i in 0..n_mix {
                let mc = unsafe { &*res.mixtures.add(i) };
                unsafe {
                    free_c_str(mc.orbit_id);
                }
                let n_epochs = mc.num_ca_epochs;
                if n_epochs > 0 {
                    if !mc.ca_epochs_mjd_tdb.is_null() {
                        let l = std::alloc::Layout::array::<f64>(n_epochs).unwrap();
                        unsafe {
                            std::alloc::dealloc(mc.ca_epochs_mjd_tdb as *mut u8, l);
                        }
                    }
                    if !mc.components_per_epoch.is_null() {
                        let l = std::alloc::Layout::array::<usize>(n_epochs).unwrap();
                        unsafe {
                            std::alloc::dealloc(mc.components_per_epoch as *mut u8, l);
                        }
                    }
                    if !mc.components_offset.is_null() {
                        let l = std::alloc::Layout::array::<usize>(n_epochs).unwrap();
                        unsafe {
                            std::alloc::dealloc(mc.components_offset as *mut u8, l);
                        }
                    }
                }
                let total = mc.num_components_total;
                if total > 0 && !mc.components.is_null() {
                    let l = std::alloc::Layout::array::<EmpyreanMixtureComponent>(total).unwrap();
                    unsafe {
                        std::alloc::dealloc(mc.components as *mut u8, l);
                    }
                }
            }
            let layout = std::alloc::Layout::array::<EmpyreanMixtureChain>(n_mix).unwrap();
            unsafe {
                std::alloc::dealloc(res.mixtures as *mut u8, layout);
            }
        }

        // Drop the retained rich result behind the opaque handle.
        if !res.lazy_handle.is_null() {
            drop(unsafe { Box::from_raw(res.lazy_handle as *mut PropagationResultHandle) });
        }

        unsafe {
            (*result).states = std::ptr::null_mut();
            (*result).object_ids = std::ptr::null_mut();
            (*result).num_states = 0;
            (*result).events = std::ptr::null_mut();
            (*result).num_events = 0;
            (*result).mixtures = std::ptr::null_mut();
            (*result).num_mixtures = 0;
            (*result).lazy_handle = std::ptr::null_mut();
        }
    }));
}

// ─────────────────────────────────────────────────────────────────────
// Provenance-tagged covariance readback
// ─────────────────────────────────────────────────────────────────────

/// Rich propagation result retained behind
/// [`EmpyreanPropagationResult::lazy_handle`], plus the per-orbit epoch
/// count needed to co-locate the nominal state by index. Boxed; freed by
/// [`empyrean_propagation_result_free`].
pub(crate) struct PropagationResultHandle {
    pub(crate) result: empyrean_core::propagation::PropagationResult,
    pub(crate) n_times: usize,
}

/// Flattened, owned form of `empyrean_core::propagation::TaggedCovariance`
/// (villeneuve's provenance-tagged covariance). Self-describing: the 6×6
/// matrix together with the co-located nominal state, how the covariance
/// was derived, its definiteness, its basis, the solved-for parameters,
/// and the functional it describes — so a consumer can never read a
/// second-order CA ellipsoid as a linear one.
///
/// NOTE on the `_cartesian` accessor's reachable range: that accessor
/// classifies but never repairs (`quality` ∈ {POSITIVE_DEFINITE,
/// INDEFINITE}), is always state-only (`solved_width == 6`,
/// `target_functional == CARTESIAN_STATE`, `has_mean_shift_input == 0`),
/// and resolves Monte-Carlo epochs to an error rather than a row. The
/// constant fields are carried for the future element-space / off-grid /
/// CA-window accessors that *do* vary them.
#[repr(C)]
pub struct EmpyreanTaggedCovariance {
    /// Epoch of this covariance (TDB).
    pub epoch_mjd_tdb: f64,
    /// Propagated nominal state [x, y, z, vx, vy, vz] (AU, AU/day),
    /// co-located engine-side. The CORRECTED mean is
    /// `state + (has_mean_shift_prop ? mean_shift_prop : 0)
    ///        + (has_mean_shift_input ? mean_shift_input : 0)`.
    pub state: [f64; 6],
    /// The 6×6 covariance (AU², AU²/day, AU²/day² blocks).
    pub matrix: [[f64; 6]; 6],
    /// `EMPYREAN_COVARIANCE_KIND_*`.
    pub kind: u8,
    /// RNG seed — valid iff `has_mc_seed == 1` (kind == MONTE_CARLO).
    pub mc_seed: u64,
    /// Disambiguates a real `mc_seed == 0` from "no seed". 0 on the
    /// `_cartesian` accessor (MC resolves to an error there).
    pub has_mc_seed: u8,
    /// Second-order propagation mean shift δμ_prop (zero at t₀).
    /// Zero-filled when `has_mean_shift_prop == 0`.
    pub mean_shift_prop: [f64; 6],
    pub has_mean_shift_prop: u8,
    /// OD-estimator mean shift δμ₀ (nonzero at t₀). Zero-filled when
    /// `has_mean_shift_input == 0` (always 0 on the `_cartesian` accessor).
    pub mean_shift_input: [f64; 6],
    pub has_mean_shift_input: u8,
    /// `EMPYREAN_COVARIANCE_QUALITY_*`.
    pub quality: u8,
    /// `min_eig` for INDEFINITE / REPAIRED; `f64::NAN` for
    /// POSITIVE_DEFINITE. **Read-only provenance** — `isnan`-guard before
    /// any arithmetic; never feed to a clamp.
    pub quality_min_eig: f64,
    /// `EMPYREAN_REPRESENTATION_*` (always CARTESIAN=0 on this accessor).
    pub representation: i32,
    /// Frame, same encoding as `EmpyreanPropagatedState.frame`.
    pub frame: i32,
    /// Origin NAIF id, same encoding as `EmpyreanPropagatedState.origin`.
    pub origin: i32,
    /// [A1, A2, A3] non-grav solved flags (0/1). The returned 6×6 is the
    /// MARGINALIZED state block of a possibly-wider fit.
    pub non_grav: [u8; 3],
    /// Thrust Δv segments solved for.
    pub thrust_segments: u32,
    /// `TaggedCovariance::solved_width()` (6 / 9 / 12 / …) — the scalar
    /// IP consumers key on (6 on this accessor).
    pub solved_width: u32,
    /// `EMPYREAN_TARGET_FUNCTIONAL_*` (always CARTESIAN_STATE on this accessor).
    pub target_functional: u8,
}

/// Owned series of [`EmpyreanTaggedCovariance`], one entry per output
/// epoch. Free with [`empyrean_tagged_covariance_series_free`].
#[repr(C)]
pub struct EmpyreanTaggedCovarianceSeries {
    pub entries: *mut EmpyreanTaggedCovariance,
    pub num_entries: usize,
}

/// Flatten one villeneuve `TaggedCovariance` (+ its co-located nominal
/// state) into the C struct. Errors loudly on a non-body basis origin
/// (an observatory has no NAIF id) rather than letting `naif_id()` panic.
fn flatten_tagged_covariance(
    epoch_mjd_tdb: f64,
    state: [f64; 6],
    tc: &empyrean_core::propagation::TaggedCovariance,
) -> Result<EmpyreanTaggedCovariance, String> {
    use empyrean_core::propagation::{CovarianceKind, CovarianceQuality, TargetFunctional};

    let (kind, mc_seed, has_mc_seed) = match tc.kind {
        CovarianceKind::Linear => (EMPYREAN_COVARIANCE_KIND_LINEAR, 0u64, 0u8),
        CovarianceKind::SecondOrder => (EMPYREAN_COVARIANCE_KIND_SECOND_ORDER, 0, 0),
        CovarianceKind::ThirdOrder => (EMPYREAN_COVARIANCE_KIND_THIRD_ORDER, 0, 0),
        CovarianceKind::Mixture => (EMPYREAN_COVARIANCE_KIND_MIXTURE, 0, 0),
        CovarianceKind::MonteCarlo { seed } => (EMPYREAN_COVARIANCE_KIND_MONTE_CARLO, seed, 1),
        CovarianceKind::SigmaPoint => (EMPYREAN_COVARIANCE_KIND_SIGMA_POINT, 0, 0),
    };

    let (quality, quality_min_eig) = match tc.quality {
        CovarianceQuality::PositiveDefinite => {
            (EMPYREAN_COVARIANCE_QUALITY_POSITIVE_DEFINITE, f64::NAN)
        }
        CovarianceQuality::Indefinite { min_eig } => {
            (EMPYREAN_COVARIANCE_QUALITY_INDEFINITE, min_eig)
        }
        CovarianceQuality::Repaired { min_eig } => (EMPYREAN_COVARIANCE_QUALITY_REPAIRED, min_eig),
    };

    let (mean_shift_prop, has_mean_shift_prop) = match tc.mean_shift_prop {
        Some(v) => (v, 1u8),
        None => ([0.0; 6], 0u8),
    };
    let (mean_shift_input, has_mean_shift_input) = match tc.mean_shift_input {
        Some(v) => (v, 1u8),
        None => ([0.0; 6], 0u8),
    };

    // Basis origin guard: `naif_id()` panics on an observatory origin.
    if tc.basis.origin.is_observatory() {
        return Err("covariance basis origin is an observatory (no NAIF id)".to_string());
    }
    let origin = tc.basis.origin.naif_id();
    let frame = frame_to_int(tc.basis.frame);
    let representation = representation_to_int(tc.basis.representation);

    let non_grav = [
        tc.param_list.non_grav[0] as u8,
        tc.param_list.non_grav[1] as u8,
        tc.param_list.non_grav[2] as u8,
    ];
    let thrust_segments = tc.param_list.thrust_segments as u32;
    let solved_width = tc.solved_width() as u32;

    let target_functional = match tc.target_functional {
        TargetFunctional::CartesianState => EMPYREAN_TARGET_FUNCTIONAL_CARTESIAN_STATE,
        TargetFunctional::CloseApproachMissDistance => {
            EMPYREAN_TARGET_FUNCTIONAL_CLOSE_APPROACH_MISS_DISTANCE
        }
    };

    Ok(EmpyreanTaggedCovariance {
        epoch_mjd_tdb,
        state,
        matrix: tc.matrix,
        kind,
        mc_seed,
        has_mc_seed,
        mean_shift_prop,
        has_mean_shift_prop,
        mean_shift_input,
        has_mean_shift_input,
        quality,
        quality_min_eig,
        representation,
        frame,
        origin,
        non_grav,
        thrust_segments,
        solved_width,
        target_functional,
    })
}

/// Map a `CovarianceSeriesError` to an `EMPYREAN_TAGGED_COV_*` code.
fn cov_series_err_code(e: &empyrean_core::propagation::CovarianceSeriesError) -> i32 {
    use empyrean_core::propagation::CovarianceSeriesError as E;
    match e {
        E::OrbitIndexOutOfRange { .. } => EMPYREAN_TAGGED_COV_ORBIT_INDEX_OUT_OF_RANGE,
        E::NoInitialCovariance => EMPYREAN_TAGGED_COV_NO_INITIAL_COVARIANCE,
        E::NoDenseTrajectory => EMPYREAN_TAGGED_COV_NO_DENSE_TRAJECTORY,
        E::StateMissing { .. } => EMPYREAN_TAGGED_COV_STATE_MISSING,
        E::Transform(_) => EMPYREAN_TAGGED_COV_TRANSFORM,
        E::Uncertainty(_) => EMPYREAN_TAGGED_COV_UNCERTAINTY,
        E::SampleCovarianceMissing { .. } => EMPYREAN_TAGGED_COV_SAMPLE_COVARIANCE_MISSING,
    }
}

/// Co-locate the propagated nominal state for `(orbit_index, epoch_index)`
/// and verify its epoch matches the covariance entry — the engine-side
/// join the consumer cannot do safely across the two arrays.
fn nominal_state_at(
    handle: &PropagationResultHandle,
    orbit_index: usize,
    epoch_index: usize,
    expected_epoch_mjd_tdb: f64,
) -> Result<[f64; 6], i32> {
    let flat_idx = orbit_index * handle.n_times + epoch_index;
    let coord = handle
        .result
        .states
        .iter()
        .nth(flat_idx)
        .map(|(_, c, _)| c)
        .ok_or(EMPYREAN_TAGGED_COV_STATE_MISSING)?;
    if (coord.t.mjd_tdb() - expected_epoch_mjd_tdb).abs() > 1e-9 {
        set_last_error("covariance-series epoch does not align with the propagated state grid");
        return Err(EMPYREAN_TAGGED_COV_STATE_MISSING);
    }
    Ok([coord.x, coord.y, coord.z, coord.vx, coord.vy, coord.vz])
}

/// Resolved-kind tagged covariance at every output epoch for one orbit,
/// Cartesian basis. On success `out_series` owns the array; free with
/// [`empyrean_tagged_covariance_series_free`]. On error `out_series` is
/// left null and the detail is on `empyrean_last_error()`.
///
/// # Safety
/// `result` must be a valid pointer returned by `empyrean_propagate`;
/// `out_series` must be a valid pointer to write the result into.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_propagation_covariance_series_cartesian(
    result: *const EmpyreanPropagationResult,
    orbit_index: usize,
    out_series: *mut *mut EmpyreanTaggedCovarianceSeries,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() || out_series.is_null() {
            set_last_error("null pointer argument");
            return EMPYREAN_TAGGED_COV_NULL_POINTER;
        }
        unsafe {
            *out_series = std::ptr::null_mut();
        }
        let res = unsafe { &*result };
        if res.lazy_handle.is_null() {
            set_last_error("propagation result carries no retained handle");
            return EMPYREAN_TAGGED_COV_NULL_POINTER;
        }
        let handle = unsafe { &*(res.lazy_handle as *const PropagationResultHandle) };

        let series = match handle.result.covariance_series_cartesian(orbit_index) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e.to_string());
                return cov_series_err_code(&e);
            }
        };

        let mut entries: Vec<EmpyreanTaggedCovariance> = Vec::with_capacity(series.len());
        for (k, (epoch, tc)) in series.iter().enumerate() {
            let state = match nominal_state_at(handle, orbit_index, k, *epoch) {
                Ok(s) => s,
                Err(code) => return code,
            };
            match flatten_tagged_covariance(*epoch, state, tc) {
                Ok(e) => entries.push(e),
                Err(msg) => {
                    set_last_error(&msg);
                    return EMPYREAN_TAGGED_COV_UNCERTAINTY;
                }
            }
        }

        let num_entries = entries.len();
        let entries_ptr = if num_entries > 0 {
            let layout =
                std::alloc::Layout::array::<EmpyreanTaggedCovariance>(num_entries).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) as *mut EmpyreanTaggedCovariance };
            if ptr.is_null() {
                set_last_error("allocation failed");
                return EMPYREAN_TAGGED_COV_PANIC;
            }
            for (i, e) in entries.into_iter().enumerate() {
                unsafe { ptr.add(i).write(e) };
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        let series_box = Box::new(EmpyreanTaggedCovarianceSeries {
            entries: entries_ptr,
            num_entries,
        });
        unsafe {
            *out_series = Box::into_raw(series_box);
        }
        EMPYREAN_TAGGED_COV_OK
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_propagation_covariance_series_cartesian");
            EMPYREAN_TAGGED_COV_PANIC
        }
    }
}

/// Resolved-kind tagged covariance at a single `(orbit_index,
/// epoch_index)`, Cartesian basis (the gm-free point query). `out` is
/// written on success.
///
/// # Safety
/// `result` and `out` must be valid pointers; `result` from `empyrean_propagate`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_propagation_covariance_at_cartesian(
    result: *const EmpyreanPropagationResult,
    orbit_index: usize,
    epoch_index: usize,
    out: *mut EmpyreanTaggedCovariance,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return EMPYREAN_TAGGED_COV_NULL_POINTER;
        }
        let res = unsafe { &*result };
        if res.lazy_handle.is_null() {
            set_last_error("propagation result carries no retained handle");
            return EMPYREAN_TAGGED_COV_NULL_POINTER;
        }
        let handle = unsafe { &*(res.lazy_handle as *const PropagationResultHandle) };

        // `uncertainty(o, e)` returns None for an out-of-range index; the
        // borrowed handle's `covariance()` is the gm-free resolved-kind
        // readback at that single epoch.
        let uncertainty = match handle.result.uncertainty(orbit_index, epoch_index) {
            Some(u) => u,
            None => {
                set_last_error("orbit or epoch index out of range");
                return EMPYREAN_TAGGED_COV_EPOCH_INDEX_OUT_OF_RANGE;
            }
        };
        let tc = match uncertainty.covariance() {
            Ok(tc) => tc,
            Err(e) => {
                set_last_error(&e.to_string());
                return EMPYREAN_TAGGED_COV_UNCERTAINTY;
            }
        };
        let epoch = uncertainty.epoch_mjd_tdb();
        let state = match nominal_state_at(handle, orbit_index, epoch_index, epoch) {
            Ok(s) => s,
            Err(code) => return code,
        };
        match flatten_tagged_covariance(epoch, state, &tc) {
            Ok(flat) => {
                unsafe { out.write(flat) };
                EMPYREAN_TAGGED_COV_OK
            }
            Err(msg) => {
                set_last_error(&msg);
                EMPYREAN_TAGGED_COV_UNCERTAINTY
            }
        }
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_propagation_covariance_at_cartesian");
            EMPYREAN_TAGGED_COV_PANIC
        }
    }
}

/// Free a series returned by
/// [`empyrean_propagation_covariance_series_cartesian`].
///
/// # Safety
/// `series` must be null or a pointer returned by that accessor, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_tagged_covariance_series_free(
    series: *mut EmpyreanTaggedCovarianceSeries,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if series.is_null() {
            return;
        }
        let s = unsafe { Box::from_raw(series) };
        if !s.entries.is_null() && s.num_entries > 0 {
            // `EmpyreanTaggedCovariance` is POD (no Drop), so dealloc the
            // backing array directly — matching the states-array pattern.
            let layout =
                std::alloc::Layout::array::<EmpyreanTaggedCovariance>(s.num_entries).unwrap();
            unsafe {
                std::alloc::dealloc(s.entries as *mut u8, layout);
            }
        }
        // `s` (the Box<EmpyreanTaggedCovarianceSeries>) drops here.
    }));
}

pub(crate) unsafe fn free_c_str(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

pub(crate) fn to_c_str(s: &str) -> *mut std::ffi::c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("?").unwrap())
        .into_raw()
}

/// Convert a borrowed `*const c_char` into an owned [`String`].
/// Returns `None` for null pointers; lossy-decodes invalid UTF-8.
pub(crate) fn c_str_to_string(ptr: *const std::ffi::c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Look up an object_id by orbit_id. Returns the empty string for
/// orbits the caller didn't tag with an `object_id`. Used by
/// `event_to_c` and the ephemeris / B-plane emit paths so every C
/// ABI row carries the same join key the input orbit batch did.
pub(crate) fn lookup_object_id<'a>(
    map: &'a std::collections::HashMap<String, String>,
    orbit_id: &str,
) -> &'a str {
    map.get(orbit_id).map(|s| s.as_str()).unwrap_or("")
}

/// Project a [`DynamicalEvent`] into the C ABI representation.
///
/// Returns `None` for variants the v0.7.0 distribution channel
/// intentionally does not yet expose (`HighSensitivity`,
/// `ChaoticRegion`, `HighNonlinearity`, `Keyhole`, `Bifurcation` —
/// diagnostic outputs whose payloads aren't yet shaped for the FFI
/// boundary). The caller filters those out so they never appear as
/// `event_type = "other"` rows.
///
/// Adding a new `DynamicalEvent` variant in villeneuve will fail to
/// compile here until an explicit decision is made about whether to
/// emit it (`Some(...)`) or gate it (`None`).
/// Euclidean norm of a 3-vector.
fn vec3_norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Neutral sentinel defaults for the variant-specific payload fields of
/// [`EmpyreanEvent`]. Each `event_to_c` arm overrides the four owning
/// string pointers plus whatever scalar fields its variant actually
/// carries; the trailing `..event_defaults()` then fills every remaining
/// field with the documented "not applicable" sentinel (`0xFF` kind
/// codes, `-1` counts, `NaN` floats). The pointer fields are null here
/// and MUST be set by every arm.
fn event_defaults() -> EmpyreanEvent {
    EmpyreanEvent {
        event_type: std::ptr::null_mut(),
        orbit_id: std::ptr::null_mut(),
        object_id: std::ptr::null_mut(),
        body: std::ptr::null_mut(),
        body_naif_id: 0,
        epoch_mjd_tdb: f64::NAN,
        distance_au: f64::NAN,
        distance_km: f64::NAN,
        relative_velocity_au_day: f64::NAN,
        previous_kind: 0xFF,
        resolved_kind: 0xFF,
        kappa: f64::NAN,
        threshold_below: f64::NAN,
        threshold_above: f64::NAN,
        two_body_energy: f64::NAN,
        jacobi_constant: f64::NAN,
        jacobi_constant_sigma: f64::NAN,
        jacobi_constant_l1: f64::NAN,
        jacobi_constant_l2: f64::NAN,
        n_periapses: -1,
        impact_latitude_deg: f64::NAN,
        impact_longitude_deg: f64::NAN,
        impact_altitude_km: f64::NAN,
        shadow_fraction: f64::NAN,
        illumination: f64::NAN,
        relative_x: f64::NAN,
        relative_y: f64::NAN,
        relative_z: f64::NAN,
        relative_vx: f64::NAN,
        relative_vy: f64::NAN,
        relative_vz: f64::NAN,
        effective_radius_au: f64::NAN,
        effective_radius_km: f64::NAN,
        sigma_distance_au: f64::NAN,
        ip_linear: f64::NAN,
        ip_second_order: f64::NAN,
        nonlinearity: f64::NAN,
        ip_agm: f64::NAN,
        ip_mc: f64::NAN,
    }
}

fn event_to_c(
    ev: &DynamicalEvent,
    object_id_map: &std::collections::HashMap<String, String>,
) -> Option<EmpyreanEvent> {
    let nan = f64::NAN;
    let obj = |orbit_id: &str| to_c_str(lookup_object_id(object_id_map, orbit_id));
    Some(match ev {
        DynamicalEvent::CloseApproachStart {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            distance_km,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("close_approach_start"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        DynamicalEvent::CloseApproachEnd {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            distance_km,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("close_approach_end"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        DynamicalEvent::Periapsis {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            distance_km,
            relative_velocity_au_day,
            relative_position,
            relative_velocity_vec,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("periapsis"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: *relative_velocity_au_day,
            relative_x: relative_position[0],
            relative_y: relative_position[1],
            relative_z: relative_position[2],
            relative_vx: relative_velocity_vec[0],
            relative_vy: relative_velocity_vec[1],
            relative_vz: relative_velocity_vec[2],
            ..event_defaults()
        },
        DynamicalEvent::Impact {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            relative_position,
            relative_velocity,
            planetodetic,
            ..
        } => {
            // Body-centric geometry at the surface intercept: |r| ≈ the
            // impacting body's radius, |v| the impact speed. These slots
            // were previously hard-zeroed/NaN'd, discarding real data.
            let d_au = vec3_norm(relative_position);
            EmpyreanEvent {
                event_type: to_c_str("impact"),
                orbit_id: to_c_str(orbit_id),
                object_id: obj(orbit_id),
                body: to_c_str(body_name),
                body_naif_id: body_origin.naif_id(),
                epoch_mjd_tdb: epoch.mjd_tdb(),
                distance_au: d_au,
                distance_km: d_au * KM_PER_AU,
                relative_velocity_au_day: vec3_norm(relative_velocity),
                impact_latitude_deg: planetodetic.as_ref().map(|s| s.latitude).unwrap_or(nan),
                impact_longitude_deg: planetodetic.as_ref().map(|s| s.longitude).unwrap_or(nan),
                impact_altitude_km: planetodetic.as_ref().map(|s| s.altitude_km).unwrap_or(nan),
                ..event_defaults()
            }
        }
        DynamicalEvent::PossibleImpact(data) => EmpyreanEvent {
            event_type: to_c_str("possible_impact"),
            orbit_id: to_c_str(&data.orbit_id),
            object_id: obj(&data.orbit_id),
            body: to_c_str(&data.body_name),
            body_naif_id: data.body_origin.naif_id(),
            epoch_mjd_tdb: data.epoch.mjd_tdb(),
            distance_au: data.miss_distance_au,
            distance_km: data.miss_distance_km,
            relative_velocity_au_day: data.relative_velocity_au_day,
            effective_radius_au: data.effective_radius_au,
            effective_radius_km: data.effective_radius_km,
            sigma_distance_au: data.sigma_distance_au,
            ip_linear: data.ip_linear,
            ip_second_order: data.ip_second_order.unwrap_or(nan),
            nonlinearity: data.nonlinearity.unwrap_or(nan),
            ip_agm: data.ip_agm.unwrap_or(nan),
            ip_mc: data.ip_mc.unwrap_or(nan),
            ..event_defaults()
        },
        DynamicalEvent::AtmosphericEntry {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            relative_velocity,
            planetodetic,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("atmospheric_entry"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            // distance_au/_km are the body-CENTER crossing distance (the
            // Karman radius), NOT an altitude. The true altitude above the
            // reference ellipsoid + the surface lat/lon ride in the
            // planetodetic block (NaN when the ground track is unresolved).
            distance_au: *distance_au,
            distance_km: *distance_au * KM_PER_AU,
            relative_velocity_au_day: vec3_norm(relative_velocity),
            impact_latitude_deg: planetodetic.as_ref().map(|s| s.latitude).unwrap_or(nan),
            impact_longitude_deg: planetodetic.as_ref().map(|s| s.longitude).unwrap_or(nan),
            impact_altitude_km: planetodetic.as_ref().map(|s| s.altitude_km).unwrap_or(nan),
            ..event_defaults()
        },
        DynamicalEvent::AtmosphericExit {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("atmospheric_exit"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: nan,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        DynamicalEvent::CaptureStart {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            distance_km,
            relative_velocity_au_day,
            two_body_energy,
            jacobi_constant,
            jacobi_constant_sigma,
            jacobi_constant_l1,
            jacobi_constant_l2,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("capture_start"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: *relative_velocity_au_day,
            two_body_energy: *two_body_energy,
            jacobi_constant: (*jacobi_constant).unwrap_or(nan),
            jacobi_constant_sigma: (*jacobi_constant_sigma).unwrap_or(nan),
            jacobi_constant_l1: (*jacobi_constant_l1).unwrap_or(nan),
            jacobi_constant_l2: (*jacobi_constant_l2).unwrap_or(nan),
            ..event_defaults()
        },
        DynamicalEvent::CaptureEnd {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            distance_au,
            distance_km,
            relative_velocity_au_day,
            two_body_energy,
            jacobi_constant,
            jacobi_constant_sigma,
            jacobi_constant_l1,
            jacobi_constant_l2,
            n_periapses,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("capture_end"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: *relative_velocity_au_day,
            two_body_energy: *two_body_energy,
            jacobi_constant: (*jacobi_constant).unwrap_or(nan),
            jacobi_constant_sigma: (*jacobi_constant_sigma).unwrap_or(nan),
            jacobi_constant_l1: (*jacobi_constant_l1).unwrap_or(nan),
            jacobi_constant_l2: (*jacobi_constant_l2).unwrap_or(nan),
            n_periapses: *n_periapses as i32,
            ..event_defaults()
        },
        DynamicalEvent::ShadowEntry {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            shadow_fraction,
            illumination,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("shadow_entry"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: nan,
            distance_km: nan,
            relative_velocity_au_day: nan,
            shadow_fraction: *shadow_fraction,
            illumination: *illumination,
            ..event_defaults()
        },
        DynamicalEvent::ShadowExit {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            shadow_fraction,
            illumination,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("shadow_exit"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: nan,
            distance_km: nan,
            relative_velocity_au_day: nan,
            shadow_fraction: *shadow_fraction,
            illumination: *illumination,
            ..event_defaults()
        },
        // ── Lagrange-point region entry/exit ────────────────────
        // event_type encodes both the L-point identifier and direction:
        // "lagrange_point_l1_entry" … "lagrange_point_l5_exit".
        // `body` carries the human-readable pair label
        // ("Sun–Earth L1") and `body_naif_id` is the secondary's NAIF
        // ID. `distance_*` is the rotating-frame distance to the
        // L-point.
        DynamicalEvent::LagrangePointEntry {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            point,
            rotating_frame_distance_au,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str(&format!(
                "lagrange_point_{}_entry",
                point.name().to_lowercase()
            )),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *rotating_frame_distance_au,
            distance_km: *rotating_frame_distance_au * KM_PER_AU,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        DynamicalEvent::LagrangePointExit {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            point,
            rotating_frame_distance_au,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str(&format!(
                "lagrange_point_{}_exit",
                point.name().to_lowercase()
            )),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *rotating_frame_distance_au,
            distance_km: *rotating_frame_distance_au * KM_PER_AU,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        // ── Ephemeris overlap ───────────────────────────────────
        DynamicalEvent::EphemerisOverlap {
            orbit_id,
            body_name,
            body_origin,
            epoch,
            position_delta_au,
            velocity_delta_au_day,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("ephemeris_overlap"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(body_name),
            body_naif_id: body_origin.naif_id(),
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *position_delta_au,
            distance_km: *position_delta_au * KM_PER_AU,
            relative_velocity_au_day: *velocity_delta_au_day,
            ..event_defaults()
        },
        // ── Periapsis relative to integration origin ────────────
        // No `body` — it's defined relative to whatever frame the
        // integrator was using, which the caller already knows.
        DynamicalEvent::CentralPeriapse {
            orbit_id,
            epoch,
            distance_au,
            distance_km,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("central_periapse"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: std::ptr::null_mut(),
            body_naif_id: -1,
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: *distance_au,
            distance_km: *distance_km,
            relative_velocity_au_day: nan,
            ..event_defaults()
        },
        // Diagnostic variants gated off in v0.7.0 distribution.
        // Returning `None` here so they never reach the C ABI.
        // Listed explicitly so adding a new variant requires a
        // deliberate decision (`HighSensitivity`, `ChaoticRegion`,
        // `HighNonlinearity`, `Keyhole`, `Bifurcation` — diagnostic
        // outputs whose payloads aren't yet shaped for the FFI
        // boundary).
        DynamicalEvent::HighSensitivity { .. }
        | DynamicalEvent::ChaoticRegion { .. }
        | DynamicalEvent::HighNonlinearity { .. }
        | DynamicalEvent::Keyhole { .. }
        | DynamicalEvent::Bifurcation { .. } => return None,

        // CovarianceRegimeChange is Auto's audit-trail event at each
        // non-Linear CA window boundary. Body / distance / velocity
        // aren't applicable so they're zeroed; the κ + band fields
        // carry the payload.
        DynamicalEvent::CovarianceRegimeChange {
            orbit_id,
            epoch,
            previous_kind,
            resolved_kind,
            kappa,
            threshold_below,
            threshold_above,
            ..
        } => EmpyreanEvent {
            event_type: to_c_str("covariance_regime_change"),
            orbit_id: to_c_str(orbit_id),
            object_id: obj(orbit_id),
            body: to_c_str(""),
            body_naif_id: 0,
            epoch_mjd_tdb: epoch.mjd_tdb(),
            distance_au: 0.0,
            distance_km: 0.0,
            relative_velocity_au_day: 0.0,
            previous_kind: covariance_kind_to_u8(*previous_kind),
            resolved_kind: covariance_kind_to_u8(*resolved_kind),
            kappa: *kappa,
            threshold_below: *threshold_below,
            threshold_above: *threshold_above,
            ..event_defaults()
        },
    })
}

#[cfg(test)]
mod tagged_covariance_tests {
    use super::*;
    use empyrean_core::coordinates::{CoordinateRepresentation, Frame, Origin};
    use empyrean_core::propagation::{
        Basis, CovarianceKind, CovarianceQuality, SolvedParameters, TaggedCovariance,
        TargetFunctional,
    };

    fn sample_tagged() -> TaggedCovariance {
        TaggedCovariance {
            matrix: [[0.0; 6]; 6],
            kind: CovarianceKind::SecondOrder,
            mean_shift_prop: Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            mean_shift_input: None,
            quality: CovarianceQuality::PositiveDefinite,
            basis: Basis {
                representation: CoordinateRepresentation::Cartesian,
                frame: Frame::EclipticJ2000,
                origin: Origin::Sun,
            },
            param_list: SolvedParameters::state_only(),
            target_functional: TargetFunctional::CartesianState,
        }
    }

    #[test]
    fn flatten_maps_kind_quality_shifts_and_basis() {
        let tc = sample_tagged();
        let f = flatten_tagged_covariance(60000.0, [7.0; 6], &tc).unwrap();

        assert_eq!(f.epoch_mjd_tdb, 60000.0);
        assert_eq!(f.state, [7.0; 6]);
        assert_eq!(f.kind, EMPYREAN_COVARIANCE_KIND_SECOND_ORDER);
        assert_eq!(f.has_mc_seed, 0);

        // Prop shift present, input shift absent.
        assert_eq!(f.has_mean_shift_prop, 1);
        assert_eq!(f.mean_shift_prop[0], 1.0);
        assert_eq!(f.has_mean_shift_input, 0);
        assert_eq!(f.mean_shift_input, [0.0; 6]);

        // PositiveDefinite ⟹ NaN sentinel (dynamicist check 3).
        assert_eq!(f.quality, EMPYREAN_COVARIANCE_QUALITY_POSITIVE_DEFINITE);
        assert!(f.quality_min_eig.is_nan());

        // Lossless basis round-trip (dynamicist check 4).
        assert_eq!(f.frame, frame_to_int(Frame::EclipticJ2000));
        assert_eq!(f.origin, Origin::Sun.naif_id());
        assert_eq!(
            f.representation,
            representation_to_int(CoordinateRepresentation::Cartesian)
        );

        // State-only on this path.
        assert_eq!(f.solved_width, 6);
        assert_eq!(f.non_grav, [0, 0, 0]);
        assert_eq!(
            f.target_functional,
            EMPYREAN_TARGET_FUNCTIONAL_CARTESIAN_STATE
        );
    }

    #[test]
    fn flatten_indefinite_carries_finite_negative_min_eig() {
        let mut tc = sample_tagged();
        tc.quality = CovarianceQuality::Indefinite { min_eig: -1.0e-9 };
        let f = flatten_tagged_covariance(60000.0, [0.0; 6], &tc).unwrap();
        assert_eq!(f.quality, EMPYREAN_COVARIANCE_QUALITY_INDEFINITE);
        assert!(f.quality_min_eig.is_finite() && f.quality_min_eig < 0.0);
    }

    #[test]
    fn flatten_monte_carlo_carries_seed_with_flag() {
        let mut tc = sample_tagged();
        tc.kind = CovarianceKind::MonteCarlo { seed: 42 };
        let f = flatten_tagged_covariance(60000.0, [0.0; 6], &tc).unwrap();
        assert_eq!(f.kind, EMPYREAN_COVARIANCE_KIND_MONTE_CARLO);
        assert_eq!(f.has_mc_seed, 1);
        assert_eq!(f.mc_seed, 42);
    }

    #[test]
    fn flatten_rejects_observatory_origin_instead_of_panicking() {
        let mut tc = sample_tagged();
        tc.basis.origin = Origin::observatory("W84");
        let r = flatten_tagged_covariance(60000.0, [0.0; 6], &tc);
        assert!(r.is_err(), "observatory origin must error, not panic");
    }

    #[test]
    fn accessors_reject_null_pointers() {
        let series_code = unsafe {
            empyrean_propagation_covariance_series_cartesian(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(series_code, EMPYREAN_TAGGED_COV_NULL_POINTER);

        let point_code = unsafe {
            empyrean_propagation_covariance_at_cartesian(
                std::ptr::null(),
                0,
                0,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(point_code, EMPYREAN_TAGGED_COV_NULL_POINTER);

        // Freeing null is a no-op, not a crash.
        unsafe { empyrean_tagged_covariance_series_free(std::ptr::null_mut()) };
    }
}

#[cfg(test)]
mod thrust_input_tests {
    use super::*;

    /// A gravity-only heliocentric Cartesian orbit with all thrust side
    /// arrays empty. Tests attach caller-owned arrays by overwriting the
    /// pointer/length pairs (keeping the backing storage alive in scope).
    fn base_orbit() -> EmpyreanOrbit {
        EmpyreanOrbit {
            state: crate::CoordinateState {
                epoch_mjd_tdb: 59000.0,
                // Plausible heliocentric Cartesian state (AU, AU/day).
                elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
                covariance: [[0.0; 6]; 6],
                has_covariance: 0,
                representation: 0, // Cartesian
                frame: 0,          // ICRF
                origin: 10,        // Sun (NAIF)
            },
            orbit_id: std::ptr::null(),
            object_id: std::ptr::null(),
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            non_grav_dt: f64::NAN,
            non_grav_dt_variance: f64::NAN,
            has_non_grav_covariance: 0,
            non_grav_covariance: [[0.0; 3]; 3],
            phot_system: -1,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
            thrust_arcs: std::ptr::null(),
            n_thrust_arcs: 0,
            dv_corrections: std::ptr::null(),
            n_dv_corrections: 0,
            correction_covariances: std::ptr::null(),
            n_correction_covariances: 0,
        }
    }

    /// A one-arc ConstantRTN burn about the Sun, constant mass (`isp NaN`).
    fn rtn_arc() -> EmpyreanThrustArc {
        EmpyreanThrustArc {
            start_mjd_tdb: 59000.0,
            end_mjd_tdb: 59010.0,
            thrust_n: 1000.0,
            mass_kg: 1000.0,
            isp_s: f64::NAN, // constant mass
            steering_law: EMPYREAN_STEERING_LAW_CONSTANT_RTN,
            steering_alpha_rad: 0.1,
            steering_beta_rad: 0.2,
            steering_direction: [0.0; 3],
            sharpness: 100.0,
            central_body_naif_id: 10, // Sun
        }
    }

    // ── Marshaling correctness (no context needed) ──────────────

    #[test]
    fn marshals_one_arc_constant_rtn_with_correction() {
        let arcs = [rtn_arc()];
        let dvs = [[1.0e-6, 2.0e-6, 3.0e-6]];
        let mut orbit = base_orbit();
        orbit.thrust_arcs = arcs.as_ptr();
        orbit.n_thrust_arcs = 1;
        orbit.dv_corrections = dvs.as_ptr();
        orbit.n_dv_corrections = 1;

        let tp = empyrean_orbit_thrust_params(&orbit)
            .expect("valid thrust params")
            .expect("Some(ThrustParams) for a one-arc orbit");
        assert_eq!(tp.arcs.len(), 1);
        let a = &tp.arcs[0];
        assert_eq!(a.start_mjd_tdb, 59000.0);
        assert_eq!(a.end_mjd_tdb, 59010.0);
        assert_eq!(a.thrust_n, 1000.0);
        assert_eq!(a.mass_kg, 1000.0);
        assert!(a.isp_s.is_none(), "NaN isp_s must map to None");
        assert_eq!(a.sharpness, 100.0);
        assert_eq!(a.central_body, Origin::Sun);
        match &a.steering {
            SteeringLaw::ConstantRTN {
                alpha_rad,
                beta_rad,
            } => {
                assert_eq!(*alpha_rad, 0.1);
                assert_eq!(*beta_rad, 0.2);
            }
            other => panic!("expected ConstantRTN, got {other:?}"),
        }
        assert_eq!(tp.dv_corrections, vec![[1.0e-6, 2.0e-6, 3.0e-6]]);
        assert!(tp.correction_covariances.is_empty());
    }

    #[test]
    fn isp_finite_maps_to_some_and_steering_variants_map() {
        let mut a0 = rtn_arc();
        a0.isp_s = 320.0; // finite → Some
        a0.steering_law = EMPYREAN_STEERING_LAW_VELOCITY_TANGENT;
        let mut a1 = rtn_arc();
        a1.steering_law = EMPYREAN_STEERING_LAW_INERTIAL_FIXED;
        a1.steering_direction = [0.0, 0.0, 1.0];
        let arcs = [a0, a1];
        let mut orbit = base_orbit();
        orbit.thrust_arcs = arcs.as_ptr();
        orbit.n_thrust_arcs = 2;

        let tp = empyrean_orbit_thrust_params(&orbit).unwrap().unwrap();
        assert_eq!(tp.arcs[0].isp_s, Some(320.0));
        assert!(matches!(tp.arcs[0].steering, SteeringLaw::VelocityTangent));
        assert!(matches!(
            tp.arcs[1].steering,
            SteeringLaw::InertialFixed { direction } if direction == [0.0, 0.0, 1.0]
        ));
    }

    #[test]
    fn no_arcs_yields_none() {
        let orbit = base_orbit();
        assert!(empyrean_orbit_thrust_params(&orbit).unwrap().is_none());
    }

    #[test]
    fn null_ptr_with_nonzero_len_errors_loudly() {
        let mut orbit = base_orbit();
        orbit.thrust_arcs = std::ptr::null();
        orbit.n_thrust_arcs = 2; // lie about the length
        let err = empyrean_orbit_thrust_params(&orbit).unwrap_err();
        assert!(err.contains("thrust_arcs"), "err was: {err}");
        assert!(err.contains("null"), "err was: {err}");
    }

    #[test]
    fn corrections_without_arcs_error() {
        let dvs = [[0.0, 0.0, 0.0]];
        let mut orbit = base_orbit();
        orbit.dv_corrections = dvs.as_ptr();
        orbit.n_dv_corrections = 1;
        let err = empyrean_orbit_thrust_params(&orbit).unwrap_err();
        assert!(err.contains("without any thrust_arcs"), "err was: {err}");
    }

    #[test]
    fn correction_covariance_length_mismatch_errors() {
        let arcs = [rtn_arc()];
        let dvs = [[0.0, 0.0, 0.0]];
        let eye = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let covs = [eye, eye]; // 2 covariances vs 1 correction
        let mut orbit = base_orbit();
        orbit.thrust_arcs = arcs.as_ptr();
        orbit.n_thrust_arcs = 1;
        orbit.dv_corrections = dvs.as_ptr();
        orbit.n_dv_corrections = 1;
        orbit.correction_covariances = covs.as_ptr();
        orbit.n_correction_covariances = 2;
        let err = empyrean_orbit_thrust_params(&orbit).unwrap_err();
        assert!(
            err.contains("must match dv_corrections length"),
            "err was: {err}"
        );
    }

    #[test]
    fn unknown_steering_law_tag_errors() {
        let mut a = rtn_arc();
        a.steering_law = 7;
        let arcs = [a];
        let mut orbit = base_orbit();
        orbit.thrust_arcs = arcs.as_ptr();
        orbit.n_thrust_arcs = 1;
        let err = empyrean_orbit_thrust_params(&orbit).unwrap_err();
        assert!(err.contains("unknown steering_law tag 7"), "err was: {err}");
    }

    #[test]
    fn unknown_central_body_naif_errors() {
        let mut a = rtn_arc();
        a.central_body_naif_id = 12_345_678;
        let arcs = [a];
        let mut orbit = base_orbit();
        orbit.thrust_arcs = arcs.as_ptr();
        orbit.n_thrust_arcs = 1;
        let err = empyrean_orbit_thrust_params(&orbit).unwrap_err();
        assert!(
            err.contains("unknown central_body NAIF id"),
            "err was: {err}"
        );
    }

    // ── Full C-ABI propagate round-trip (gated on ephemeris data) ──

    fn try_context() -> Option<EmpyreanContext> {
        empyrean_core::Context::from_data_dir(None).ok()
    }

    fn last_err() -> String {
        let p = crate::empyrean_last_error();
        if p.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    /// End-to-end: a thrust arc (ConstantRTN) + a Δv correction with its
    /// covariance reaches the dynamics through the real `empyrean_propagate`
    /// C ABI and perturbs the trajectory relative to a ballistic run, and the
    /// tagged-covariance readback stays callable on the thrusting result —
    /// closing the input→output loop. Asserts the input EFFECT rather than a
    /// specific `thrust_segments`/`solved_width` count: whether the wide-Jet
    /// burn-sensitivity segment is retained at readback is an engine/config
    /// detail (a FirstOrder no-non-grav config marginalizes to the 6×6 state
    /// block), so the version-robust contract at this layer is that thrust
    /// moves the dynamics and the output plumbing works.
    #[test]
    fn propagate_with_thrust_arc_runs_and_reports_thrust_segments() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping propagate_with_thrust_arc_*: no ephemeris data dir available");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        let times = [59000.0f64, 59005.0, 59012.0];

        // Standard-tier, first-order config. `zeroed` gives the right
        // sentinels for every field except dt_initial / dt_min, whose
        // NaN-means-auto sentinel is not zero.
        let mut cfg: EmpyreanPropagationConfig = unsafe { std::mem::zeroed() };
        cfg.force_model = 2; // Standard
        cfg.uncertainty_method.tag = EMPYREAN_UNCERTAINTY_FIRST;
        cfg.advanced.dt_initial = f64::NAN;
        cfg.advanced.dt_min = f64::NAN;
        cfg.advanced.cache_integrator_steps = 1;

        // Read the final-epoch position of a single-orbit propagation.
        let final_position = |orbit: EmpyreanOrbit| -> [f64; 3] {
            let orbits = [orbit];
            let mut result: EmpyreanPropagationResult = unsafe { std::mem::zeroed() };
            let code = unsafe {
                empyrean_propagate(
                    ctx_ptr,
                    orbits.as_ptr(),
                    1,
                    times.as_ptr(),
                    times.len(),
                    &cfg,
                    &mut result,
                )
            };
            assert_eq!(code, 0, "propagate must succeed: {}", last_err());
            assert_eq!(
                result.num_states,
                times.len(),
                "expected one state per requested epoch"
            );
            // Tagged-covariance readback must stay callable on this result.
            let mut tc: EmpyreanTaggedCovariance = unsafe { std::mem::zeroed() };
            let rc = unsafe {
                empyrean_propagation_covariance_at_cartesian(&result, 0, times.len() - 1, &mut tc)
            };
            assert_eq!(
                rc,
                EMPYREAN_TAGGED_COV_OK,
                "tagged-covariance readback failed: {}",
                last_err()
            );
            let last = unsafe { &*result.states.add(times.len() - 1) };
            let pos = [last.x, last.y, last.z];
            unsafe { empyrean_propagation_result_free(&mut result) };
            pos
        };

        // Input covariance so the wide-Jet burn-sensitivity path engages.
        let mut cov = [[0.0f64; 6]; 6];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0e-16;
        }

        // Ballistic baseline — same orbit, no thrust arcs.
        let mut ballistic = base_orbit();
        ballistic.state.covariance = cov;
        ballistic.state.has_covariance = 1;
        let pos_ballistic = final_position(ballistic);

        // Thrusting run — one ConstantRTN arc + a Δv correction with covariance.
        let arcs = [rtn_arc()];
        let dvs = [[0.0f64, 0.0, 0.0]];
        let eye_small = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 1.0e-20, 0.0],
            [0.0, 0.0, 1.0e-20],
        ];
        let covs = [eye_small];
        let mut thrusting = base_orbit();
        thrusting.state.covariance = cov;
        thrusting.state.has_covariance = 1;
        thrusting.thrust_arcs = arcs.as_ptr();
        thrusting.n_thrust_arcs = 1;
        thrusting.dv_corrections = dvs.as_ptr();
        thrusting.n_dv_corrections = 1;
        thrusting.correction_covariances = covs.as_ptr();
        thrusting.n_correction_covariances = 1;
        let pos_thrust = final_position(thrusting);
        // `arcs` / `dvs` / `covs` outlive the borrow above — dropped here.

        // The thrust arc must have moved the final position.
        let dr = ((pos_thrust[0] - pos_ballistic[0]).powi(2)
            + (pos_thrust[1] - pos_ballistic[1]).powi(2)
            + (pos_thrust[2] - pos_ballistic[2]).powi(2))
        .sqrt();
        assert!(
            dr > 1.0e-3,
            "thrust arc must perturb the trajectory (Δposition = {dr} AU)"
        );
    }
}
