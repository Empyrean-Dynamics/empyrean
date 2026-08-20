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
use empyrean_core::determination::{ODResult, Session, SessionDiff};

use crate::od::{
    EmpyreanODConfig, EmpyreanODResult, EmpyreanObservation, build_od_config_from_c,
    c_observations_to_optical,
};
use crate::propagate::EmpyreanPropagatedState;
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
    /// Δ iteration count, over
    /// [`EmpyreanODResult::iterations`](crate::od::EmpyreanODResult::iterations)
    /// — a TOTAL across each fit's solves. Two fits that reached the
    /// same orbit by different ladders differ here without either being
    /// worse.
    pub iterations_delta: i64,
    /// Δ number of observations used (negative ⇒ observations were
    /// masked between prior and current).
    pub n_observations_delta: i64,
    /// [`EmpyreanODResult::update_norm`](crate::od::EmpyreanODResult::update_norm)
    /// on the current fit — the μ-DAMPED last accepted step, **not** a
    /// convergence metric and **not** comparable to
    /// [`EmpyreanODConfig::convergence_tol`](crate::od::EmpyreanODConfig::convergence_tol).
    /// It is here as a damping-trajectory diagnostic between two fits of
    /// the same arc. For whether either fit converged, and on what, read
    /// that fit's `termination` / `gn_step_qnorm`.
    pub update_norm_current: f64,
    /// The same quantity on the prior fit, with the same caveat.
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
/// `config` is parsed by the same builder the one-shot
/// `empyrean_determine` / `empyrean_evaluate` / `empyrean_refine`
/// entry points use, so every field — weighting preset and additional
/// layers, debiasing, rejection, solve-for, origin and output-epoch
/// policy — resolves identically on both surfaces. A malformed
/// weighting layer is rejected here, before any fitting: the call
/// returns null with the reason in `empyrean_last_error`.
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
        // Parsed by the SAME builder the one-shot determine / evaluate /
        // refine entry points use. A session-local parser existed here
        // and read only the force model and the four convergence knobs,
        // so a caller who configured weighting or debiasing on a session
        // silently got scott's `ODConfig::default()` instead — no
        // diagnostic, no parity with the one-shot surface. Every layer
        // validation error the shared builder raises now propagates out
        // of this constructor as a null return + `set_last_error`, which
        // the wrapper turns into a typed error.
        let cfg = match build_od_config_from_c(cfg_ref) {
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
                // Release the partial write before reporting failure —
                // by the time the joint marshal can fail, the writer has
                // already published the observation and station-bias
                // arrays onto `result_out`, and a caller told `-3` will
                // not call the free function for a result it never
                // received. Matches the determine and refine arms.
                unsafe { crate::od::free_od_result_fields(result_out) };
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
                // Release the partial write before reporting failure —
                // by the time the joint marshal can fail, the writer has
                // already published the observation and station-bias
                // arrays onto `result_out`, and a caller told `-3` will
                // not call the free function for a result it never
                // received. Matches the determine and refine arms.
                unsafe { crate::od::free_od_result_fields(result_out) };
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

fn write_od_result(od: &ODResult, result_out: *mut EmpyreanODResult) -> Result<(), String> {
    let prop_state = od_orbit_to_propagated_local(&od.orbit, &od.covariance)?;
    unsafe {
        (*result_out).orbit = prop_state;
        // Same writer as the one-shot entry points: this
        // path used to populate a subset by hand, so a session caller got
        // an all-zero `covariance`, `NaN` non-gravitational parameters, a
        // `solve_for_used` that disagreed with the fit that had actually
        // run, and `converged` read off a zeroed acceptability block. It
        // also owns the pointer surfaces (observations, station biases,
        // photometry) that `empyrean_od_result_free` frees
        // unconditionally, so nothing is left indeterminate for a caller
        // who did not zero-initialize the out-struct.
        // A session holds one object's observations, so the per-row
        // grouping key is the caller's, not the engine's: null object_id.
        crate::od::write_od_result_fields(result_out, od, None)?;
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
            // Fallible since the engine stopped panicking on an origin
            // with no NAIF id (an MPC site code). Surface it rather than
            // invent a sentinel: a made-up integer read back through the
            // ABI would name a different body.
            let cs = coordinates_to_coordinate_state(&coord_deg)
                .map_err(|e| format!("session OD result orbit: {e}"))?;
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
                // Filled by `write_od_result_fields` below.
                orbit_cov: crate::joint::empty_orbit_covariance(),
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
        // Filled by `write_od_result_fields` below.
        orbit_cov: crate::joint::empty_orbit_covariance(),
    })
}

/// The basis the joint marshal reads, which the session path is the only
/// consumer of.
///
/// The one-shot entry points refuse a non-Cartesian fitted orbit, and
/// Cartesian has no angular element rows — so degrees and radians are
/// indistinguishable there and the marshal's basis choice is
/// unobservable. The session is the only path that delivers Keplerian /
/// Cometary / Spherical results, and it shipped once with the marshal
/// reading DEGREES while `ODResult::covariance` is read off the stored
/// RADIAN coordinate: a factor of 180/π on three rows, inside one struct
/// whose documented contract is that the two share a basis.
///
/// # Why this pins the accessor rather than driving a session fit
///
/// A live Keplerian-output session cannot reach the code under test. A
/// Marsden solve needs a Cartesian seed — `empyrean_session_refine` with
/// `output_representation = KEPLERIAN` and Marsden solved fails in IOD
/// with "initial orbit must be Cartesian" before any joint is marshaled
/// — and without a Marsden solve there is no border to check. An
/// end-to-end test would therefore skip every time while appearing to
/// cover the path, which is the failure mode this delivery has already
/// been bitten by twice. So this pins the decision itself: on a
/// non-Cartesian coordinate the two read-outs genuinely differ, and the
/// marshal must take the stored one.
#[cfg(test)]
mod session_joint_basis_tests {
    use empyrean_core::coordinates::{AU, Degrees, ExtendedCovariance};
    use empyrean_core::orbits::Orbits;

    /// On a Keplerian orbit the stored and degree read-outs of a border
    /// differ by 180/π on the angular rows — so "which one the marshal
    /// takes" is a real choice with a wrong answer, not a formality.
    ///
    /// `write_orbit_covariance` takes the STORED one, because that is
    /// where `ODResult::covariance` comes from. This test fails if
    /// anyone reintroduces the angular read-out there: the two arrays it
    /// compares would become the same array.
    #[test]
    fn the_stored_and_degree_read_outs_of_a_border_really_differ() {
        let cs = empyrean_core::convert::CoordinateState {
            epoch_mjd_tdb: 59000.0,
            // a, e, i, raan, argperi, M — i/raan/argperi are angles.
            elements: [1.5, 0.2, 10.0, 20.0, 30.0, 40.0],
            covariance: {
                let mut c = [[0.0f64; 6]; 6];
                for (i, row) in c.iter_mut().enumerate() {
                    row[i] = 1.0e-8 * (i as f64 + 1.0);
                }
                c
            },
            has_covariance: 1,
            representation: crate::od::EMPYREAN_REPRESENTATION_KEPLERIAN,
            frame: 0,
            origin: 10,
        };
        let coord = empyrean_core::convert::coordinate_state_to_coordinates(&cs)
            .expect("a well-formed Keplerian state");
        let params: [[f64; 3]; 3] = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 1.0e-20, 0.0],
            [0.0, 0.0, 1.0e-20],
        ];
        let cross = [[1.0e-14f64; 3]; 6];
        let bordered = crate::joint::coordinates_with_extended(
            coord,
            Some(ExtendedCovariance::new(cross, params)),
        );

        let mut orbits: Orbits<AU> = Orbits::empty();
        orbits
            .push("kep".to_string(), bordered.into_radians())
            .expect("push");

        let stored = *orbits.coordinates()[0]
            .extended_covariance()
            .expect("the border survived the push");
        let (degree_coord, _) = orbits
            .coordinates_angular::<Degrees>(0)
            .expect("row 0 exists");
        let degrees = *degree_coord
            .extended_covariance()
            .expect("the degree read-out carries it too");

        // Keplerian's angular element indices are 2, 3, 4 AND 5:
        // a, e, i, raan, argperi, M — the mean anomaly is an angle too.
        // (Cometary's list stops at 4 because its sixth element is a
        // time of perihelion, not an angle. Using that list here is how
        // this test failed on its first run.)
        let to_deg = 180.0 / std::f64::consts::PI;
        for r in 0..6 {
            for c in 0..3 {
                if (2..6).contains(&r) {
                    assert!(
                        (degrees.cross[r][c] / stored.cross[r][c] - to_deg).abs() < 1e-6,
                        "angular row {r} must differ between the two read-outs by 180/pi; \
                         taking the wrong one is a 57x error on this row"
                    );
                } else {
                    assert!(
                        (degrees.cross[r][c] - stored.cross[r][c]).abs() < 1e-30,
                        "non-angular row {r} is identical in both read-outs"
                    );
                }
            }
        }
    }

    /// The marshal's contract, stated as an assertion: what it publishes
    /// must equal the STORED border, because `ODResult::covariance` is
    /// the stored 6x6 and the two have to describe one matrix.
    #[test]
    fn the_marshal_publishes_the_stored_border_not_the_degree_one() {
        let cs = empyrean_core::convert::CoordinateState {
            epoch_mjd_tdb: 59000.0,
            elements: [1.5, 0.2, 10.0, 20.0, 30.0, 40.0],
            covariance: {
                let mut c = [[0.0f64; 6]; 6];
                for (i, row) in c.iter_mut().enumerate() {
                    row[i] = 1.0e-8 * (i as f64 + 1.0);
                }
                c
            },
            has_covariance: 1,
            representation: crate::od::EMPYREAN_REPRESENTATION_KEPLERIAN,
            frame: 0,
            origin: 10,
        };
        let coord = empyrean_core::convert::coordinate_state_to_coordinates(&cs).unwrap();
        let params: [[f64; 3]; 3] = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 1.0e-20, 0.0],
            [0.0, 0.0, 1.0e-20],
        ];
        let bordered = crate::joint::coordinates_with_extended(
            coord,
            Some(ExtendedCovariance::new([[1.0e-14f64; 3]; 6], params)),
        );
        let mut orbits: Orbits<AU> = Orbits::empty();
        orbits
            .push("kep".to_string(), bordered.into_radians())
            .unwrap();

        let stored = *orbits.coordinates()[0].extended_covariance().unwrap();
        let published = crate::joint::joint_to_c(
            orbits.coordinates()[0].extended_covariance(),
            orbits.wide_cross(0),
            "test",
        )
        .expect("marshals");

        assert_eq!(published.has_non_grav_cross, 1);
        assert_eq!(
            published.non_grav_cross, stored.cross,
            "the published border must be the STORED one — the basis ODResult::covariance \
             is read in. Publishing the degree read-out beside a radian 6x6 puts a \
             180/pi error on three rows of one joint."
        );
    }
}
