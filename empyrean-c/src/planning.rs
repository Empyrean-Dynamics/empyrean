//! Observation-planning C ABI — information-gain analysis of candidate
//! follow-up observations against a prior orbit covariance.
//!
//! A thin shell over [`empyrean_core::planning::evaluate_plan_single`]. Given a
//! fitted orbit that carries a 6×6 Cartesian covariance (the information prior)
//! and a list of candidate `(observatory, epoch)` observations — optical or
//! radar — the planner reports how much each observation would shrink the orbit
//! uncertainty: the prior / posterior covariance metrics and a per-candidate
//! marginal-information-gain analysis.
//!
//! Radar candidates fold a predicted delay (range) / Doppler (range-rate)
//! measurement: the line-of-sight information that angles-only optical cannot
//! supply. The measurement σ is the Cramér-Rao bound from the waveform
//! bandwidth and the effective SNR (supplied, or computed from the link budget
//! with no silent default for the leverage-heavy target properties).
//!
//! The encounter-B-plane characterization that `scott::planning` can also emit
//! is intentionally not exposed here yet — it is only meaningful for a close
//! encounter and will be added additively when that showcase is built.

use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use empyrean_core::planning::{
    CandidateInfo, CandidateKind, CovarianceMetrics, ObservatoryConfig, PlanResult,
    PlannedObservation, PlanningConfig, RadarMode, RadarPlanSpec, Station, TargetRadarProperties,
    evaluate_plan_single,
};
use empyrean_core::time::Epoch;

use crate::od::empyrean_orbit_to_orbits;
use crate::propagate::{EmpyreanOrbit, int_to_force_model};
use crate::{EmpyreanContext, set_last_error};

// ── Input structs ───────────────────────────────────────────────────

/// Per-observatory assumptions: astrometric σ + observability filters.
#[repr(C)]
pub struct EmpyreanObservatoryConfig {
    /// MPC observatory code (null-terminated UTF-8).
    pub obs_code: *const c_char,
    /// 1σ (RA·cosδ) astrometric uncertainty (arcsec).
    pub sigma_ra_arcsec: f64,
    /// 1σ Dec astrometric uncertainty (arcsec).
    pub sigma_dec_arcsec: f64,
    /// Limiting apparent magnitude.
    pub max_apparent_mag: f64,
    /// Minimum solar elongation (degrees).
    pub min_elongation_deg: f64,
}

/// A single planned (candidate) observation: an epoch plus either an optical or
/// a radar measurement spec. Optical fields are read when `kind == 0`, radar
/// fields when `kind == 1`; absent optional radar target properties use NaN
/// (the link budget refuses, rather than silently defaults, a missing value).
#[repr(C)]
pub struct EmpyreanPlannedObservation {
    /// Planned (receive, for radar) epoch — MJD, TDB.
    pub epoch_mjd_tdb: f64,
    /// 0 = optical, 1 = radar.
    pub kind: u8,

    /// Optical: registered MPC station code (null-terminated UTF-8).
    pub optical_code: *const c_char,
    /// Optical 1σ (RA·cosδ) uncertainty (arcsec).
    pub optical_sigma_ra_arcsec: f64,
    /// Optical 1σ Dec uncertainty (arcsec).
    pub optical_sigma_dec_arcsec: f64,

    /// Radar transmit dish preset: 0 = Goldstone DSS-14, 1 = Green Bank
    /// (receive-only), 2 = Arecibo (defunct — rejected loudly, never scheduled).
    pub radar_transmit_station: u8,
    /// Radar receive dish preset (same encoding; equal to transmit = monostatic).
    pub radar_receive_station: u8,
    /// Radar mode: 0 = Delay, 1 = Doppler, 2 = Both.
    pub radar_mode: u8,
    /// Waveform bandwidth (Hz) — sets the delay (range) σ.
    pub radar_bandwidth_hz: f64,
    /// Doppler frequency resolution (Hz) — sets the Doppler (range-rate) σ.
    pub radar_freq_resolution_hz: f64,
    /// Supplied effective SNR (linear, not dB). NaN → compute from the link
    /// budget using the target properties + integration time below.
    pub radar_snr: f64,
    /// Link-budget target absolute magnitude H (mag). NaN = absent.
    pub radar_target_h_mag: f64,
    /// Link-budget target visual geometric albedo. NaN = absent.
    pub radar_target_visual_albedo: f64,
    /// Link-budget target radar (OC) albedo. NaN = absent.
    pub radar_target_radar_albedo: f64,
    /// Link-budget target effective diameter (km). NaN = absent.
    pub radar_target_diameter_km: f64,
    /// Link-budget target rotation period (hours), caps coherent integration.
    /// NaN = absent.
    pub radar_target_spin_period_hours: f64,
    /// Coherent integration time (s) for the link-budget SNR.
    pub radar_integration_s: f64,
}

/// Planning configuration: force model, integration tolerance, the
/// per-observatory astrometric assumptions, and threading.
#[repr(C)]
pub struct EmpyreanPlanningConfig {
    /// Force-model tier: 0 = Approximate, 1 = Basic, 2 = Standard.
    pub force_model: i32,
    /// GR15 / IAS15 integration tolerance.
    pub epsilon: f64,
    /// Per-observatory config array.
    pub observatories: *const EmpyreanObservatoryConfig,
    /// Number of entries in `observatories`.
    pub num_observatories: usize,
    /// Threads for batch operations; -1 = all CPUs.
    pub num_threads: i32,
}

// ── Output structs ──────────────────────────────────────────────────

/// Covariance summary metrics (prior, posterior, or per-candidate cumulative).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanCovarianceMetrics {
    /// RSS position 1σ (km).
    pub position_sigma_km: f64,
    /// RSS velocity 1σ (m/s) — the rendezvous-Δv-knowledge term.
    pub velocity_sigma_m_s: f64,
    /// Semi-major axis of the 1σ position ellipsoid (km).
    pub semi_major_km: f64,
    /// Semi-minor axis of the 1σ position ellipsoid (km).
    pub semi_minor_km: f64,
    /// ln(det(Σ)) — D-optimality criterion.
    pub log_det: f64,
}

/// Per-candidate information-gain analysis.
#[repr(C)]
pub struct EmpyreanPlanCandidate {
    /// Index into the planned-observation list.
    pub index: usize,
    /// Observatory / receive-station code (owned; freed by
    /// `empyrean_plan_result_free`).
    pub obs_code: *mut c_char,
    /// 0 = optical, 1 = radar.
    pub kind: u8,
    /// 1 if observable at this epoch (passes the filters / has positive SNR).
    pub observable: u8,
    /// Sky-plane along-track 1σ (arcsec) — optical geometry (NaN for radar).
    pub along_track_sigma_arcsec: f64,
    /// Sky-plane cross-track 1σ (arcsec).
    pub cross_track_sigma_arcsec: f64,
    /// Predicted RA·cosδ 1σ (arcsec).
    pub ra_sigma_arcsec: f64,
    /// Predicted Dec 1σ (arcsec).
    pub dec_sigma_arcsec: f64,
    /// Position angle of the sky-plane uncertainty ellipse (degrees).
    pub position_angle_deg: f64,
    /// Marginal covariance-volume reduction factor from this observation (≤ 1).
    pub marginal_volume_reduction: f64,
    /// Fractional position-σ improvement from this observation (∈ [0, 1]).
    pub marginal_position_improvement: f64,
    /// Post-observation along-track 1σ (arcsec).
    pub post_along_track_sigma_arcsec: f64,
    /// Post-observation cross-track 1σ (arcsec).
    pub post_cross_track_sigma_arcsec: f64,
    /// Covariance metrics after folding this observation and all prior ones.
    pub cumulative: EmpyreanCovarianceMetrics,
    /// Active solve-for width (6 = state-only).
    pub active_width: usize,
}

/// Result of a plan evaluation. The caller allocates the struct; the
/// `orbit_id` string and the `candidates` array (with their `obs_code`
/// strings) are heap-allocated here and must be released with
/// [`empyrean_plan_result_free`].
#[repr(C)]
pub struct EmpyreanPlanResult {
    /// Orbit identifier (owned).
    pub orbit_id: *mut c_char,
    /// Prior (pre-observation) covariance metrics.
    pub prior: EmpyreanCovarianceMetrics,
    /// Posterior (all candidates folded) covariance metrics.
    pub posterior: EmpyreanCovarianceMetrics,
    /// Per-candidate analysis array (owned).
    pub candidates: *mut EmpyreanPlanCandidate,
    /// Number of entries in `candidates`.
    pub num_candidates: usize,
    /// Active solve-for width (6 = state-only).
    pub active_width: usize,
}

// ── Conversions ─────────────────────────────────────────────────────

fn nan_to_opt(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

fn cstr_to_string(p: *const c_char, what: &str) -> Result<String, String> {
    if p.is_null() {
        return Err(format!("{what}: null string pointer"));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("{what}: invalid UTF-8: {e}"))
}

fn radar_station_preset(code: u8, role: &str) -> Result<Station, String> {
    match code {
        0 => Ok(Station::goldstone_dss14()),
        1 => Ok(Station::green_bank_receive()),
        2 => Ok(Station::arecibo_defunct()),
        other => Err(format!(
            "unknown radar {role} station preset {other} (0=Goldstone, 1=GreenBank, 2=Arecibo)"
        )),
    }
}

fn radar_mode_from_u8(m: u8) -> Result<RadarMode, String> {
    match m {
        0 => Ok(RadarMode::Delay),
        1 => Ok(RadarMode::Doppler),
        2 => Ok(RadarMode::Both),
        other => Err(format!(
            "unknown radar mode {other} (0=Delay, 1=Doppler, 2=Both)"
        )),
    }
}

fn build_planned(c: &EmpyreanPlannedObservation) -> Result<PlannedObservation, String> {
    let epoch = Epoch::from_mjd_tdb(c.epoch_mjd_tdb);
    match c.kind {
        0 => {
            let code = cstr_to_string(c.optical_code, "optical station code")?;
            Ok(PlannedObservation::optical(
                Station::optical(
                    code,
                    [c.optical_sigma_ra_arcsec, c.optical_sigma_dec_arcsec],
                ),
                epoch,
            ))
        }
        1 => {
            let tx = radar_station_preset(c.radar_transmit_station, "transmit")?;
            let rx = radar_station_preset(c.radar_receive_station, "receive")?;
            let mode = radar_mode_from_u8(c.radar_mode)?;
            let spec = if c.radar_snr.is_nan() {
                let target = TargetRadarProperties {
                    diameter_km: nan_to_opt(c.radar_target_diameter_km),
                    h_mag: nan_to_opt(c.radar_target_h_mag),
                    visual_albedo: nan_to_opt(c.radar_target_visual_albedo),
                    radar_albedo: nan_to_opt(c.radar_target_radar_albedo),
                    spin_period_hours: nan_to_opt(c.radar_target_spin_period_hours),
                };
                RadarPlanSpec::link_budget(
                    tx,
                    rx,
                    target,
                    c.radar_integration_s,
                    mode,
                    c.radar_bandwidth_hz,
                    c.radar_freq_resolution_hz,
                )
            } else {
                RadarPlanSpec::given(
                    tx,
                    rx,
                    mode,
                    c.radar_bandwidth_hz,
                    c.radar_freq_resolution_hz,
                    c.radar_snr,
                )
            };
            Ok(PlannedObservation::radar(spec, epoch))
        }
        other => Err(format!(
            "unknown planned-observation kind {other} (0=optical, 1=radar)"
        )),
    }
}

fn build_planning_config(c: &EmpyreanPlanningConfig) -> Result<PlanningConfig, String> {
    let force_model = int_to_force_model(c.force_model)?.into();

    let obs_slice: &[EmpyreanObservatoryConfig] =
        if c.observatories.is_null() || c.num_observatories == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(c.observatories, c.num_observatories) }
        };
    let mut observatories = Vec::with_capacity(obs_slice.len());
    for o in obs_slice {
        let code = cstr_to_string(o.obs_code, "observatory code")?;
        observatories.push(ObservatoryConfig {
            obs_code: code,
            sigma_arcsec: [o.sigma_ra_arcsec, o.sigma_dec_arcsec],
            max_apparent_mag: o.max_apparent_mag,
            min_elongation_deg: o.min_elongation_deg,
        });
    }

    Ok(PlanningConfig {
        force_model,
        epsilon: c.epsilon,
        observatories,
        num_threads: if c.num_threads < 0 {
            None
        } else {
            Some(c.num_threads as usize)
        },
        encounter: None,
    })
}

fn metrics_to_c(m: &CovarianceMetrics) -> EmpyreanCovarianceMetrics {
    EmpyreanCovarianceMetrics {
        position_sigma_km: m.position_sigma_km,
        velocity_sigma_m_s: m.velocity_sigma_m_s,
        semi_major_km: m.semi_major_km,
        semi_minor_km: m.semi_minor_km,
        log_det: m.log_det,
    }
}

fn owned_cstr(s: &str) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("").unwrap())
        .into_raw()
}

fn candidate_to_c(c: &CandidateInfo) -> EmpyreanPlanCandidate {
    let kind = match c.kind {
        CandidateKind::Optical => 0u8,
        CandidateKind::Radar { .. } => 1u8,
    };
    EmpyreanPlanCandidate {
        index: c.index,
        obs_code: owned_cstr(&c.obs_code),
        kind,
        observable: u8::from(c.observable),
        along_track_sigma_arcsec: c.along_track_sigma_arcsec,
        cross_track_sigma_arcsec: c.cross_track_sigma_arcsec,
        ra_sigma_arcsec: c.ra_sigma_arcsec,
        dec_sigma_arcsec: c.dec_sigma_arcsec,
        position_angle_deg: c.position_angle_deg,
        marginal_volume_reduction: c.marginal_volume_reduction,
        marginal_position_improvement: c.marginal_position_improvement,
        post_along_track_sigma_arcsec: c.post_along_track_sigma_arcsec,
        post_cross_track_sigma_arcsec: c.post_cross_track_sigma_arcsec,
        cumulative: metrics_to_c(&c.cumulative),
        active_width: c.active_width,
    }
}

fn plan_result_to_c(plan: &PlanResult, out: *mut EmpyreanPlanResult) {
    let candidates: Vec<EmpyreanPlanCandidate> =
        plan.candidates.iter().map(candidate_to_c).collect();
    let num_candidates = candidates.len();
    let cand_ptr = if num_candidates == 0 {
        std::ptr::null_mut()
    } else {
        Box::into_raw(candidates.into_boxed_slice()) as *mut EmpyreanPlanCandidate
    };
    unsafe {
        (*out).orbit_id = owned_cstr(&plan.orbit_id);
        (*out).prior = metrics_to_c(&plan.prior);
        (*out).posterior = metrics_to_c(&plan.posterior);
        (*out).candidates = cand_ptr;
        (*out).num_candidates = num_candidates;
        (*out).active_width = plan.active_width;
    }
}

// ── Entry points ────────────────────────────────────────────────────

/// Evaluate an observation plan: how much each candidate observation would
/// tighten the prior orbit covariance.
///
/// `orbit` must carry a 6×6 Cartesian covariance (e.g. a `determine` result).
/// `planned` is an array of `num_planned` candidate observations. `orbit_id`
/// may be null (defaults to `"orbit_0"`). On success populates `*result_out`
/// (caller-allocated); free with [`empyrean_plan_result_free`].
///
/// Returns 0 on success, -1 for a null/invalid argument, -3 if planning fails
/// (missing/singular prior covariance, an infeasible or invalid candidate, an
/// ephemeris-generation error), -99 on an internal panic. The error message is
/// retrievable via `empyrean_last_error()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_evaluate_plan(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    orbit_id: *const c_char,
    planned: *const EmpyreanPlannedObservation,
    num_planned: usize,
    config: *const EmpyreanPlanningConfig,
    result_out: *mut EmpyreanPlanResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbit.is_null()
            || planned.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let orbit_ref = unsafe { &*orbit };
        let cfg_ref = unsafe { &*config };
        let planned_slice = unsafe { std::slice::from_raw_parts(planned, num_planned) };

        let id = if orbit_id.is_null() {
            "orbit_0".to_string()
        } else {
            match cstr_to_string(orbit_id, "orbit_id") {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        };

        let orbits = match empyrean_orbit_to_orbits(orbit_ref, &id) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let mut planned_vec = Vec::with_capacity(planned_slice.len());
        for p in planned_slice {
            match build_planned(p) {
                Ok(po) => planned_vec.push(po),
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        }

        let plan_cfg = match build_planning_config(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let plan = match evaluate_plan_single(ctx_ref, &orbits, &planned_vec, &plan_cfg) {
            Ok(p) => p,
            Err(e) => {
                set_last_error(&format!("planning failed: {e}"));
                return -3;
            }
        };

        plan_result_to_c(&plan, result_out);
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_evaluate_plan");
            -99
        }
    }
}

/// Free the heap allocations inside a plan result populated by
/// [`empyrean_evaluate_plan`]. Does not free the caller-allocated struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_plan_result_free(result: *mut EmpyreanPlanResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &*result };
        if !res.orbit_id.is_null() {
            drop(unsafe { CString::from_raw(res.orbit_id) });
        }
        if !res.candidates.is_null() && res.num_candidates > 0 {
            let cands = unsafe {
                Vec::from_raw_parts(res.candidates, res.num_candidates, res.num_candidates)
            };
            for c in &cands {
                if !c.obs_code.is_null() {
                    drop(unsafe { CString::from_raw(c.obs_code) });
                }
            }
            drop(cands);
        }
        unsafe {
            (*result).orbit_id = std::ptr::null_mut();
            (*result).candidates = std::ptr::null_mut();
            (*result).num_candidates = 0;
        }
    }));
}
