//! C ABI exports for stateful orbit-determination sessions.
//!
//! Wraps [`empyrean_core::determination::Session`] (re-exported from
//! `scott::session`). A session owns an observation set, a mask of
//! currently-disabled observations, and a fit history. The typical
//! workflow is *fit → look at residuals → mask one bad night → re-fit
//! → compare χ²* — see the upstream Session docs for the rationale.
//!
//! This is the only place in the C ABI where a long-lived mutable
//! handle (other than `EmpyreanContext`) is exposed. All mutations
//! return `i32` codes (0 = success); queries return values directly.

use std::panic::AssertUnwindSafe;

use empyrean_core::convert::{coordinates_to_coordinate_state, frame_to_int};
use empyrean_core::coordinates::{AU, Coordinates};
use empyrean_core::determination::{ODConfig, ODResult, Session, SessionDiff};

use crate::od::{
    EmpyreanODConfig, EmpyreanODResult, EmpyreanObservation, c_observations_to_optical,
    observation_results_to_c, summary_to_c,
};
use crate::propagate::{EmpyreanPropagatedState, int_to_force_model};
use crate::{EmpyreanContext, set_last_error};

// ────────────────────────────────────────────────────────────────────
// Opaque handle + result types
// ────────────────────────────────────────────────────────────────────

/// Opaque handle to an `empyrean_core::determination::Session`.
///
/// Owns observations, the mask state, and the fit history. Construct
/// with [`empyrean_session_new`]; release with
/// [`empyrean_session_free`]. The handle is **not** thread-safe — do
/// not share between threads without external synchronization.
pub type EmpyreanSession = Session;

/// Pairwise diagnostic returned by [`empyrean_session_diff`].
#[repr(C)]
pub struct EmpyreanSessionDiff {
    /// Δ reduced χ² (positive ⇒ current fit is worse than prior).
    pub reduced_chi2_delta: f64,
    /// Δ iteration count.
    pub iterations_delta: i64,
    /// Δ number of observations used (negative ⇒ observations were
    /// masked between prior and current).
    pub n_observations_delta: i64,
    /// Final update-norm convergence metric on the current fit.
    pub update_norm_current: f64,
    /// Final update-norm convergence metric on the prior fit.
    pub update_norm_prior: f64,
}

impl EmpyreanSessionDiff {
    fn from_upstream(d: &SessionDiff) -> Self {
        Self {
            reduced_chi2_delta: d.reduced_chi2_delta,
            iterations_delta: d.iterations_delta,
            n_observations_delta: d.n_observations_delta,
            update_norm_current: d.update_norm_current,
            update_norm_prior: d.update_norm_prior,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Construction / destruction
// ────────────────────────────────────────────────────────────────────

/// Construct a new orbit-determination session over a fixed
/// observation set.
///
/// Returns a heap-allocated handle on success, or null on error.
/// The caller owns the returned pointer and must free it with
/// [`empyrean_session_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_new(
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
) -> *mut EmpyreanSession {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if observations.is_null() || config.is_null() {
            set_last_error("null pointer argument");
            return std::ptr::null_mut();
        }
        let cfg_ref = unsafe { &*config };
        let obs_slice = unsafe { std::slice::from_raw_parts(observations, num_observations) };
        let obs_vec = match c_observations_to_optical(obs_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return std::ptr::null_mut();
            }
        };
        let cfg = match build_od_config_from_c_local(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return std::ptr::null_mut();
            }
        };
        let session = Session::new(obs_vec, cfg);
        Box::into_raw(Box::new(session))
    }));
    match result {
        Ok(p) => p,
        Err(_) => {
            set_last_error("panic in empyrean_session_new");
            std::ptr::null_mut()
        }
    }
}

/// Free a session previously returned by [`empyrean_session_new`].
/// Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_free(session: *mut EmpyreanSession) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if !session.is_null() {
            unsafe { drop(Box::from_raw(session)) };
        }
    }));
}

// ────────────────────────────────────────────────────────────────────
// Mask state
// ────────────────────────────────────────────────────────────────────

/// Total number of observations in the session (masked or not).
/// Returns 0 if `session` is null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_n_observations(session: *const EmpyreanSession) -> usize {
    if session.is_null() {
        return 0;
    }
    unsafe { (*session).n_observations() }
}

/// Number of observations currently masked.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_n_masked(session: *const EmpyreanSession) -> usize {
    if session.is_null() {
        return 0;
    }
    unsafe { (*session).n_masked() }
}

/// Number of observations active (not masked) in the next refine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_n_active(session: *const EmpyreanSession) -> usize {
    if session.is_null() {
        return 0;
    }
    unsafe { (*session).n_active() }
}

/// Mask the observation at `idx`. Returns 0 on success, -1 on null
/// or out-of-bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_mask(session: *mut EmpyreanSession, idx: usize) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() {
            set_last_error("null session");
            return -1;
        }
        let s = unsafe { &mut *session };
        if idx >= s.n_observations() {
            set_last_error(&format!(
                "mask index {idx} out of bounds (session has {} observations)",
                s.n_observations()
            ));
            return -1;
        }
        s.mask(idx);
        0
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_session_mask");
            -99
        }
    }
}

/// Unmask the observation at `idx`. Returns 0 on success, -1 on null
/// or out-of-bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_unmask(session: *mut EmpyreanSession, idx: usize) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() {
            set_last_error("null session");
            return -1;
        }
        let s = unsafe { &mut *session };
        if idx >= s.n_observations() {
            set_last_error(&format!(
                "unmask index {idx} out of bounds (session has {} observations)",
                s.n_observations()
            ));
            return -1;
        }
        s.unmask(idx);
        0
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_session_unmask");
            -99
        }
    }
}

/// Clear all masks. Returns 0 on success, -1 on null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_unmask_all(session: *mut EmpyreanSession) -> i32 {
    if session.is_null() {
        set_last_error("null session");
        return -1;
    }
    unsafe { (*session).unmask_all() };
    0
}

/// Whether the observation at `idx` is masked. Returns 1 = masked,
/// 0 = active, 255 (-1 cast) on null/out-of-bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_is_masked(
    session: *const EmpyreanSession,
    idx: usize,
) -> u8 {
    if session.is_null() {
        return 255;
    }
    let s = unsafe { &*session };
    if idx >= s.n_observations() {
        return 255;
    }
    if s.is_masked(idx) { 1 } else { 0 }
}

// ────────────────────────────────────────────────────────────────────
// Refine + history
// ────────────────────────────────────────────────────────────────────

/// Run an OD refine using the current mask state.
///
/// On the first call, runs the full IOD → DC pipeline. On subsequent
/// calls, uses the previously-fit orbit as the IOD seed (skipping the
/// IOD step). Pushes the new fit onto the session's history.
///
/// On success populates `result_out` with the latest fit. The caller
/// must free `result_out` with [`empyrean_od_result_free`](crate::od::empyrean_od_result_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_refine(
    session: *mut EmpyreanSession,
    ctx: *const EmpyreanContext,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() || ctx.is_null() || result_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let s = unsafe { &mut *session };
        let c = unsafe { &*ctx };
        let od = match s.refine(c.ephemeris_data()) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&format!("session.refine failed: {e}"));
                return -2;
            }
        };
        match write_od_result(od, result_out) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e);
                -3
            }
        }
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_session_refine");
            -99
        }
    }
}

/// Number of fits in the session history.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_history_len(session: *const EmpyreanSession) -> usize {
    if session.is_null() {
        return 0;
    }
    unsafe { (*session).history().len() }
}

/// Copy the i-th history entry into `result_out`. Returns 0 on
/// success, -1 on null/out-of-bounds. Caller frees `result_out` with
/// [`empyrean_od_result_free`](crate::od::empyrean_od_result_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_get_history(
    session: *const EmpyreanSession,
    idx: usize,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() || result_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let s = unsafe { &*session };
        let history = s.history();
        let entry = match history.get(idx) {
            Some(e) => e,
            None => {
                set_last_error(&format!(
                    "history index {idx} out of bounds (history has {})",
                    history.len()
                ));
                return -1;
            }
        };
        match write_od_result(entry, result_out) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e);
                -3
            }
        }
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_session_get_history");
            -99
        }
    }
}

/// Diff the current fit against an earlier history entry. Returns 0
/// on success, -1 if there is no current fit or `prior_idx` is out
/// of bounds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_session_diff(
    session: *const EmpyreanSession,
    prior_idx: usize,
    diff_out: *mut EmpyreanSessionDiff,
) -> i32 {
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if session.is_null() || diff_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let s = unsafe { &*session };
        let prior = match s.history().get(prior_idx) {
            Some(p) => p,
            None => {
                set_last_error(&format!(
                    "prior_idx {prior_idx} out of bounds (history has {})",
                    s.history().len()
                ));
                return -1;
            }
        };
        let diff = match s.diff_against(prior) {
            Some(d) => d,
            None => {
                set_last_error("session has no current fit (call empyrean_session_refine first)");
                return -1;
            }
        };
        unsafe {
            *diff_out = EmpyreanSessionDiff::from_upstream(&diff);
        }
        0
    }));
    match r {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_session_diff");
            -99
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Helpers (local to this module)
// ────────────────────────────────────────────────────────────────────

fn build_od_config_from_c_local(c: &EmpyreanODConfig) -> Result<ODConfig, String> {
    let fm = int_to_force_model(c.force_model)?;
    let mut cfg = ODConfig::default();
    cfg.force_model = fm.into();
    if c.max_iterations > 0 {
        cfg.max_iterations = c.max_iterations as usize;
    }
    if c.convergence_tol > 0.0 {
        cfg.convergence_tol = c.convergence_tol;
    }
    if c.epsilon > 0.0 {
        cfg.epsilon = c.epsilon;
    }
    if c.max_light_time_iterations > 0 {
        cfg.max_light_time_iterations = c.max_light_time_iterations;
    }
    cfg.num_threads = std::num::NonZeroUsize::new(c.num_threads);
    Ok(cfg)
}

fn write_od_result(od: &ODResult, result_out: *mut EmpyreanODResult) -> Result<(), String> {
    let prop_state = od_orbit_to_propagated_local(&od.orbit, &od.covariance)?;
    let (obs_ptr, obs_n) = observation_results_to_c(&od.observations);
    let summary = summary_to_c(&od.summary);
    unsafe {
        (*result_out).orbit = prop_state;
        (*result_out).observations = obs_ptr;
        (*result_out).num_observations = obs_n;
        (*result_out).summary = summary;
        (*result_out).iterations = od.iterations as u32;
        (*result_out).converged = if od.acceptability.fit_acceptable {
            1
        } else {
            0
        };
    }
    Ok(())
}

fn od_orbit_to_propagated_local(
    orbit: &empyrean_core::orbits::Orbits<AU>,
    covariance: &[[f64; 6]; 6],
) -> Result<EmpyreanPropagatedState, String> {
    let (_id, coord) = orbit
        .get(0)
        .ok_or_else(|| "session OD result orbit is empty".to_string())?;
    let (epoch, x, y, z, vx, vy, vz, frame, origin) = match coord {
        Coordinates::Cartesian(c, _, _) => {
            (c.t, c.x, c.y, c.z, c.vx, c.vy, c.vz, c.frame, c.origin)
        }
        _ => {
            // Convert to Cartesian via empyrean-core's converter so we
            // get a uniform [x y z vx vy vz] regardless of whether the
            // session emitted Keplerian etc.
            // Coordinates from session are in radians by upstream
            // convention; convert to degrees for FFI parity with the
            // rest of the C ABI.
            let coord_deg = coord.into_angular::<empyrean_core::coordinates::Degrees>();
            let cs = coordinates_to_coordinate_state(&coord_deg);
            return Ok(EmpyreanPropagatedState {
                epoch_mjd_tdb: cs.epoch_mjd_tdb,
                x: cs.elements[0],
                y: cs.elements[1],
                z: cs.elements[2],
                vx: cs.elements[3],
                vy: cs.elements[4],
                vz: cs.elements[5],
                origin: cs.origin,
                frame: cs.frame,
                covariance: *covariance,
                has_covariance: 1,
                stm: [[0.0; 6]; 6],
                has_stm: 0,
                stt: [[[0.0; 6]; 6]; 6],
                has_stt: 0,
                resolved_kind: 0,
            });
        }
    };
    Ok(EmpyreanPropagatedState {
        epoch_mjd_tdb: epoch.mjd_tdb(),
        x,
        y,
        z,
        vx,
        vy,
        vz,
        origin: origin.naif_id(),
        frame: frame_to_int(frame),
        covariance: *covariance,
        has_covariance: 1,
        stm: [[0.0; 6]; 6],
        has_stm: 0,
        stt: [[[0.0; 6]; 6]; 6],
        has_stt: 0,
        resolved_kind: 0,
    })
}
