use std::ffi::CString;
use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::{coordinate_state_to_coordinates, frame_to_int};
use empyrean_core::coordinates::{AU, CartesianCoordinates, Frame};
use empyrean_core::ephemeris::{
    EphemerisConfig, EphemerisPropagationConfig, EphemerisResult, generate_ephemeris,
};
use empyrean_core::observers::Observer;
use empyrean_core::orbits::Orbits;
use empyrean_core::time::Epoch;

use crate::observers::EmpyreanObserver;
use crate::propagate::{
    EmpyreanOrbit, EmpyreanPropagationConfig, empyrean_orbit_photometric_params,
    empyrean_orbit_srp_params, empyrean_orbit_thrust_params,
};
use crate::{EmpyreanContext, set_last_error};

// ────────────────────────────────────────────────────────────────────
// Observation-sensitivity row order
// ────────────────────────────────────────────────────────────────────
//
// Row indices into the `[6][n_params]` Jacobian and the
// `[6][n_params][n_params]` Hessian on
// [`EmpyreanObservationSensitivity`]. Both carry the same six output
// rows in the same order.
//
// The angles arrive in degrees and the range in AU, so a wrong row is
// wrong in unit as well as in observable — reading row 0 as RA yields a
// range partial in AU, which is finite, plausible, and silently wrong.
// These constants exist so no caller has to remember that.
//
// They are a caller-side contract: this crate marshals the Jacobian as
// an opaque block and never indexes it by observable, so each one is
// `dead_code` on the Rust side while being exactly what the generated C
// header is for.

/// Row of the range (topocentric distance) partials, in AU per input unit.
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_RANGE: usize = 0;
/// Row of the right-ascension partials, in degrees per input unit.
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_RA: usize = 1;
/// Row of the declination partials, in degrees per input unit.
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_DEC: usize = 2;
/// Row of the range-rate partials, in AU/day per input unit.
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_VRANGE: usize = 3;
/// Row of the RA-rate partials, in deg/day per input unit. The rate is
/// dRA/dt, not scaled by cos(Dec).
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_VRA: usize = 4;
/// Row of the Dec-rate partials, in deg/day per input unit.
#[allow(dead_code)]
pub const EMPYREAN_SENSITIVITY_ROW_VDEC: usize = 5;

// ── C-compatible types ──────────────────────────────────────

/// A single predicted ephemeris entry.
#[repr(C)]
pub struct EmpyreanEphemerisEntry {
    /// Orbit identifier (heap-allocated, freed by empyrean_ephemeris_result_free).
    pub orbit_id: *mut std::ffi::c_char,
    /// Epoch as MJD TDB.
    pub epoch_mjd_tdb: f64,
    /// Right ascension (degrees).
    pub ra_deg: f64,
    /// Declination (degrees).
    pub dec_deg: f64,
    /// Topocentric distance (AU).
    pub rho_au: f64,
    /// Radial velocity (AU/day).
    pub vrho_au_day: f64,
    /// RA rate (deg/day).
    pub vra_deg_day: f64,
    /// Dec rate (deg/day).
    pub vdec_deg_day: f64,
    /// One-way light time (days). NaN if unavailable.
    pub light_time_days: f64,
    /// Phase angle (degrees). NaN if unavailable.
    pub phase_angle_deg: f64,
    /// Solar elongation (degrees). NaN if unavailable.
    pub elongation_deg: f64,
    /// Heliocentric distance (AU). NaN if unavailable.
    pub heliocentric_distance_au: f64,
    /// Predicted apparent magnitude. NaN if unavailable.
    pub mag: f64,
    /// Magnitude uncertainty (1σ). Finite only when photometry is
    /// enabled AND the input orbit carried at least one of a state
    /// covariance or a photometric covariance
    /// (`EmpyreanOrbit::phot_covariance`) AND what it carried contracts
    /// to a strictly positive variance; NaN otherwise. A carried
    /// covariance is not sufficient on its own: an all-zero 3×3, or a
    /// non-PSD one that contracts to ≤ 0, still reports NaN.
    ///
    /// Both contributions are summed in quadrature:
    /// σ_V = sqrt(σ²_photo + σ²_state), where σ_state is the state
    /// contribution and σ_photo contracts the orbit's photometric 3×3
    /// over (H, slope1, slope2) against the FULL magnitude Jacobian:
    /// σ²_photo = J Σ_p Jᵀ with J = [∂V/∂H, ∂V/∂slope1, ∂V/∂slope2].
    ///
    /// V = H + 5·log10(r·Δ) + φ(α) gives ∂V/∂H ≡ 1, so an orbit with NO
    /// state covariance and a photometric covariance of the H-only shape
    /// diag(σ_H², 0, 0) — what an H-only fit emits — reports σ_V = σ_H
    /// exactly. The slope terms do not drop out of any other shape:
    /// slope variances and H–slope covariances contract against
    /// ∂V/∂slope, which vanishes only at zero phase angle, so any
    /// covariance carrying them reports σ_V > σ_H. An SBDB-queried orbit
    /// is the common case — its published diag(σ_H², σ_G², 0) makes σ_V
    /// strictly larger than σ_H.
    ///
    /// The two terms are combined as independent. They are not strictly
    /// independent — a fitted σ_H is conditional on the fitted state,
    /// because the photometric fit holds the geometry (r, Δ, α) exact —
    /// and no joint state↔photometry covariance is computed anywhere in
    /// the stack, so there is no cross term to add. The resulting σ_V is
    /// therefore mildly conservative, which is the safe direction.
    pub mag_sigma: f64,
    /// Topocentric zenith angle (degrees). NaN if unavailable (e.g. no
    /// observer geodetic position).
    pub zenith_angle_deg: f64,
    /// Topocentric azimuth, East of North (degrees). NaN if unavailable.
    pub azimuth_deg: f64,
    /// Local hour angle (degrees). NaN if unavailable.
    pub hour_angle_deg: f64,
    /// Angular separation from the Moon (degrees). NaN if unavailable.
    pub lunar_elongation_deg: f64,
    /// Position angle of motion, East of North (degrees). NaN if unavailable.
    pub position_angle_deg: f64,
    /// Total apparent sky-plane rate of motion (degrees/day). NaN if
    /// unavailable.
    pub sky_rate_deg_day: f64,
    /// Observer code, null-terminated (4 bytes).
    pub obs_code: [u8; 4],
    /// 1 when `covariance` / `aberrated_state` / `aberrated_covariance`
    /// below are populated — i.e. the input orbit carried a state
    /// covariance and the STM/uncertainty path ran; 0 otherwise.
    pub has_covariance: u8,
    /// 6×6 sky-plane covariance over (rho, RA, Dec, vrho, vRA, vDec) in
    /// (AU, deg) units, row-major. All-NaN when `has_covariance == 0`.
    pub covariance: [[f64; 6]; 6],
    /// Aberrated (light-time corrected) barycentric ICRF Cartesian state
    /// `[x, y, z, vx, vy, vz]` (AU, AU/day) at the photon-emission epoch
    /// (`epoch_mjd_tdb - light_time_days`). Populated independently of
    /// covariance; NaN-filled in the (never-observed-today) case where
    /// the engine produced no aberrated state for the row.
    pub aberrated_state: [f64; 6],
    /// 1 when `aberrated_covariance` is populated; 0 otherwise.
    pub has_aberrated_covariance: u8,
    /// 6×6 Cartesian covariance of the aberrated state, row-major.
    /// All-NaN when `has_aberrated_covariance == 0`.
    pub aberrated_covariance: [[f64; 6]; 6],
}

/// One observation-sensitivity row — the partial derivatives of the
/// sky-plane observable w.r.t. the input state, for a single
/// `(orbit, observer, epoch)`. One row per observation epoch within each
/// `(orbit_id, obs_code)` chain. Owning struct: free
/// the whole result with [`empyrean_ephemeris_result_free`].
///
/// The Jacobian composes d(obs)/d(state at t_obs) * Phi(t_obs, t0) and
/// omits the light-time terms (the -v * dtau/dx partial; the STM is
/// sampled at t_obs rather than emission t_obs - tau): both are O(tau),
/// landing in the velocity columns of the angle rows with fractional
/// error ~ tau/dt (tau ~ 0.006-0.017 d) — negligible for multi-night
/// arcs, growing as the arc shrinks toward intra-night.
///
/// The six output rows are the topocentric spherical observable, in the
/// order given by the `EMPYREAN_SENSITIVITY_ROW_*` constants. Index with
/// those rather than by hand — the row order is part of this ABI and the
/// range row sits ahead of the angles, so a hand-written `0` reads range
/// where RA was meant.
#[repr(C)]
pub struct EmpyreanObservationSensitivity {
    /// Orbit identifier. Owning C string.
    pub orbit_id: *mut std::ffi::c_char,
    /// Object identifier (owning C string) or null when absent.
    pub object_id: *mut std::ffi::c_char,
    /// MPC observatory code, null-terminated (4 bytes).
    pub obs_code: [u8; 4],
    /// Observation epoch (MJD TDB).
    pub epoch_mjd_tdb: f64,
    /// Solved-parameter dimension: 6 (state) or 9 (state + non-grav).
    pub n_params: u8,
    /// Jacobian ∂(observable)/∂(input), row-major `[6][n_params]` flattened
    /// (length `6 * n_params`). Null when this epoch carries no Jacobian.
    ///
    /// Element `(row, col)` is `jacobian[row * n_params + col]`. Columns
    /// `0..6` are the input Cartesian state, in the frame and origin the
    /// `frame` / `origin` fields tag; any further columns are the extra
    /// solved-for parameters `n_params` counts.
    ///
    /// The six rows, in order — see `EMPYREAN_SENSITIVITY_ROW_*`:
    ///
    /// - `0` range, AU per input unit
    /// - `1` RA, deg per input unit
    /// - `2` Dec, deg per input unit
    /// - `3` range rate, AU/day per input unit
    /// - `4` RA rate, deg/day per input unit (dRA/dt, NOT scaled by cos Dec)
    /// - `5` Dec rate, deg/day per input unit
    pub jacobian: *mut f64,
    /// Length of `jacobian` (`6 * n_params`), 0 when null.
    pub jacobian_len: usize,
    /// Hessian ∂²(observable)/∂(input)², row-major `[6][n_params][n_params]`
    /// flattened (length `6 * n_params * n_params`). Null unless a
    /// second-order method (Jet2) ran.
    ///
    /// Leading index is the observable, in the same order and the same
    /// units-per-input-unit as `jacobian` — index it with the same
    /// `EMPYREAN_SENSITIVITY_ROW_*` constants. Element `(row, i, j)` is
    /// `hessian[(row * n_params + i) * n_params + j]`.
    pub hessian: *mut f64,
    /// Length of `hessian` (`6 * n_params²`), 0 when null.
    pub hessian_len: usize,
    /// Frame of the input axis (Frame enum as int).
    pub frame: i32,
    /// Origin of the input axis (NAIF id).
    pub origin: i32,
}

/// Result containing an array of ephemeris entries and, when an
/// uncertainty method traced the STM, the observation-sensitivity chains.
#[repr(C)]
pub struct EmpyreanEphemerisResult {
    pub entries: *mut EmpyreanEphemerisEntry,
    pub num_entries: usize,
    /// Per-`(orbit, observer, epoch)` sensitivity rows. Null / 0 when no
    /// STM was traced (e.g. an f64-only path).
    pub sensitivity: *mut EmpyreanObservationSensitivity,
    pub num_sensitivity: usize,
    /// Non-fatal generation warnings: conditions the generator handled
    /// but the caller should be aware of — e.g. Earth-orientation kernel
    /// coverage gaps at one or more requested epochs (the analytic
    /// IAU 2006 fallback was used), or ephemeris rows whose
    /// observation-sensitivity chain could not be built (the row's
    /// astrometry is still present; only its partials are missing).
    /// Heap array of `num_warnings` NUL-terminated UTF-8 strings; null
    /// when `num_warnings == 0` (a clean run). One list per call, not
    /// per row; each message names the affected orbit id, observatory
    /// code, and epoch (MJD TDB) where applicable. Freed by
    /// `empyrean_ephemeris_result_free`.
    pub warnings: *mut *mut std::ffi::c_char,
    /// Number of warning strings. 0 when the run had nothing to report.
    pub num_warnings: usize,
}

/// Ephemeris-generation configuration.
///
/// Wraps the inner [`EmpyreanPropagationConfig`] plus the light-time
/// iteration controls and a diagnostics toggle. The propagation runs
/// internally to bring each orbit to its observation epoch, so the
/// propagation-level knobs the integrator consults apply here too:
/// `force_model`, `excluded_perturbers_naif`, `uncertainty_method`,
/// `compute_stm`, `frame`, `num_threads`, `ephemeris_overlap_policy`, and the
/// whole `advanced` block.
///
/// Two blocks do **not** apply: `events` and `diagnostics`. Ephemeris
/// generation runs with event detection and timeseries diagnostics off
/// and [`EmpyreanEphemerisResult`] carries no channel for either, so
/// setting a field in either block is refused with an error naming it
/// rather than accepted and dropped. Leave both zeroed (a `memset(0)`
/// config is valid); use `empyrean_propagate` when you need them.
///
/// # Generating for an SB441-N16 body
///
/// `ephemeris_overlap_policy` matters more here than on `empyrean_propagate`.
/// Under the default `EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK` the engine
/// skips integration for a target that coincides with one of its own
/// perturbers — and ephemeris generation reads the dense trajectory that
/// integration would have produced, so the call **fails** for any
/// SB441-N16 body at Standard tier. Pass
/// `EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE` (or exclude the body
/// via `excluded_perturbers_naif`) to generate ephemerides for one.
#[repr(C)]
pub struct EmpyreanEphemerisConfig {
    /// Inner propagation configuration applied to the trajectory that
    /// brings each orbit to its observation epoch. The `events` and
    /// `diagnostics` sub-blocks must be left zeroed — see the struct
    /// docs.
    pub propagation: EmpyreanPropagationConfig,
    /// Maximum iterations for light-time convergence. 0 → use the
    /// upstream default (3).
    pub max_light_time_iterations: usize,
    /// Tolerance (days) for light-time convergence. 0.0 → upstream
    /// default (1e-10).
    pub light_time_tolerance_days: f64,
    /// 1 = compute phase angle / elongation / heliocentric distance /
    /// apparent magnitude. 0 = skip (faster; appropriate for OD inner
    /// loops that only need RA/Dec).
    pub compute_diagnostics: u8,
}

// ── empyrean_generate_ephemeris ──────────────────────────────

/// Generate predicted ephemeris for orbits and observers.
///
/// Returns 0 on success, negative error code on failure.
/// On success, `result_out` is populated with ephemeris entries:
/// `num_orbits * num_observers` rows, orbit-major, and within each orbit
/// in **observer-input order** (sensitivity rows follow the same order).
/// Each observer carries its own epoch, so positional pairing against
/// the input observers is safe within an orbit block.
/// The caller must free the result with `empyrean_ephemeris_result_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_generate_ephemeris(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    observers_ptr: *const EmpyreanObserver,
    num_observers: usize,
    config: *const EmpyreanEphemerisConfig,
    result_out: *mut EmpyreanEphemerisResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbits_ptr.is_null()
            || observers_ptr.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let observer_slice = unsafe { std::slice::from_raw_parts(observers_ptr, num_observers) };

        let orbits = match build_orbits_for_ephemeris(orbit_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let observers = match build_observers_from_c(observer_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let config = match build_ephemeris_config_from_c(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let eph_result = match generate_ephemeris(ctx_ref, &orbits, &observers, &config) {
            Ok(e) => e,
            Err(e) => {
                set_last_error(&e.to_string());
                return -4;
            }
        };

        marshal_ephemeris_result(&eph_result, result_out)
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_generate_ephemeris");
            -99
        }
    }
}

/// Build an `Orbits<AU>` batch from a C-ABI orbit slice for ephemeris
/// generation, attaching each row's non-grav, photometric, and thrust
/// parameters. Orbit ids are the positional `"orbit_{i}"` fabrication the
/// optical forward model expects. Shared by the one-shot
/// [`empyrean_generate_ephemeris`] and the handle-based
/// [`empyrean_builtsystem_generate_ephemeris`](crate::built_system::empyrean_builtsystem_generate_ephemeris).
pub(crate) fn build_orbits_for_ephemeris(
    orbit_slice: &[EmpyreanOrbit],
) -> Result<Orbits<AU>, String> {
    let mut orbits: Orbits<AU> = Orbits::empty();
    for (i, orbit) in orbit_slice.iter().enumerate() {
        let state = orbit.state.to_empyrean();
        let coords =
            coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit {i}: {e}"))?;
        let id = format!("orbit_{i}");
        crate::joint::push_orbit_with_joint(&mut orbits, id, coords, orbit)
            .map_err(|e| format!("orbit {i}: {e}"))?;
        // Routed through the shared helper rather than the inline copy
        // this path used to carry: that copy hardcoded `covariance:
        // None`, dropping the Marsden 3×3 the ABI has carried since
        // v0.9.0 — and with the border now supplied, a dropped 3×3 is
        // also a border with no parameter block to sit against.
        if let Some(params) = crate::propagate::empyrean_orbit_non_grav_params(orbit) {
            orbits.set_non_grav_params(i, Some(params));
        }
        if let Some(ph) =
            empyrean_orbit_photometric_params(orbit).map_err(|e| format!("orbit {i}: {e}"))?
        {
            orbits.set_photometric_params(i, Some(ph));
        }
        match empyrean_orbit_thrust_params(orbit) {
            Ok(Some(tp)) => orbits.set_thrust_params(i, Some(tp)),
            Ok(None) => {}
            Err(e) => return Err(format!("orbit {i}: {e}")),
        }
        match empyrean_orbit_srp_params(orbit) {
            Ok(Some(srp)) => orbits.set_srp_params(i, Some(srp)),
            Ok(None) => {}
            Err(e) => return Err(format!("orbit {i}: {e}")),
        }
    }
    Ok(orbits)
}

/// Build a `Vec<Observer>` from a C-ABI observer slice. Shared by the
/// one-shot and handle ephemeris paths.
///
/// # Basis
///
/// Ephemeris generation requires observers in the **construction basis**
/// — ICRF, solar-system barycenter — which is what
/// `empyrean_get_observers` returns for `frame = 0, origin = 0`. A row
/// tagged with any other basis is refused by name rather than read as if
/// it were ICRF/SSB: the numbers would be a rotated and/or translated
/// station state fed to a forward model that assumes otherwise, and the
/// resulting astrometry would be wrong with no diagnostic. A `memset(0)`
/// observer is ICRF/SSB, so a caller who never touched the two tail
/// fields is unaffected.
pub(crate) fn build_observers_from_c(
    observer_slice: &[EmpyreanObserver],
) -> Result<Vec<Observer>, String> {
    let mut observers: Vec<Observer> = Vec::with_capacity(observer_slice.len());
    for (i, obs) in observer_slice.iter().enumerate() {
        let obs_frame = empyrean_core::convert::int_to_frame(obs.frame)
            .map_err(|e| format!("observer {i}: {e}"))?;
        if obs_frame != Frame::ICRF || obs.origin != Origin::SolarSystemBarycenter.naif_id() {
            return Err(format!(
                "observer {i}: ephemeris generation requires observers in the \
                 construction basis (frame = 0 / ICRF, origin = 0 / solar-system \
                 barycenter); got frame = {}, origin = {}. Re-request the observers \
                 from empyrean_get_observers with that basis — reading a rotated or \
                 translated station state as if it were ICRF/SSB would produce wrong \
                 astrometry with no diagnostic.",
                obs.frame, obs.origin
            ));
        }
        // The engine's observatory registry keys 3-byte MPC codes. A
        // 4th byte must not be dropped: the 3-byte prefix would silently
        // resolve to a DIFFERENT observatory (wrong topocentric geometry,
        // no diagnostic).
        if obs.obs_code[3] != 0 {
            return Err(format!(
                "observer {i}: observatory code {:?} uses all 4 bytes; \
                 4-character MPC codes are not yet supported by the \
                 engine's observatory registry",
                String::from_utf8_lossy(&obs.obs_code)
            ));
        }
        let mut code = [b' '; 3];
        for (c, &b) in code.iter_mut().zip(obs.obs_code.iter()) {
            *c = if b == 0 { b' ' } else { b };
        }
        let epoch = Epoch::from_mjd_tdb(obs.epoch_mjd_tdb);
        let state = CartesianCoordinates::new(
            epoch,
            obs.x,
            obs.y,
            obs.z,
            obs.vx,
            obs.vy,
            obs.vz,
            Frame::ICRF,
            Origin::SolarSystemBarycenter,
        );
        let observing_night = if obs.observing_night >= 0 {
            Some(obs.observing_night as u32)
        } else {
            None
        };
        observers.push(Observer {
            code,
            state,
            observing_night,
        });
    }
    Ok(observers)
}

/// Propagation fields the C ABI advertises on an ephemeris config that
/// [`EphemerisPropagationConfig`] has no home for, named individually so
/// a rejection can say which ones the caller set.
///
/// Ephemeris generation builds its inner `PropagationConfig` with
/// `events` and `diagnostics` pinned to their upstream defaults, and
/// [`EmpyreanEphemerisResult`] carries neither an event list nor a
/// diagnostics timeseries. A value set in either block therefore cannot
/// be honoured *and* cannot be observed. Taking it and dropping it is
/// exactly the failure mode this converter was rewritten to close, so
/// the request is refused instead.
///
/// Zero is the "not requested" state for both blocks: a `memset(0)`
/// config passes, and the flags map 1:1 to booleans elsewhere in the
/// ABI, so a zeroed block is also what the ephemeris path would have
/// produced anyway.
fn ephemeris_unsupported_propagation_fields(c: &EmpyreanPropagationConfig) -> Vec<&'static str> {
    // A float knob counts as requested only when it is finite and
    // non-zero. `memset(0)` is the documented way to build a C config, so
    // 0.0 is indistinguishable from "untouched" on this ABI, and NaN is
    // the ABI's explicit `None`. Both are inert here regardless.
    fn is_set(v: f64) -> bool {
        v.is_finite() && v != 0.0
    }
    let mut set = Vec::new();
    let e = &c.events;
    for (name, on) in [
        ("events.close_approaches", e.close_approaches != 0),
        ("events.impacts", e.impacts != 0),
        ("events.atmospheric", e.atmospheric != 0),
        ("events.possible_impacts", e.possible_impacts != 0),
        ("events.shadow_events", e.shadow_events != 0),
        ("events.dense_output", e.dense_output != 0),
        (
            "events.dense_output_cadence_days",
            is_set(e.dense_output_cadence_days),
        ),
        (
            "events.body_filter_naif",
            e.num_body_filter > 0 && !e.body_filter_naif.is_null(),
        ),
    ] {
        if on {
            set.push(name);
        }
    }
    let d = &c.diagnostics;
    for (name, on) in [
        ("diagnostics.sensitivity", d.sensitivity != 0),
        ("diagnostics.nonlinearity", d.nonlinearity != 0),
        ("diagnostics.lyapunov", d.lyapunov != 0),
        ("diagnostics.keyholes", d.keyholes != 0),
        ("diagnostics.bifurcations", d.bifurcations != 0),
        ("diagnostics.sample_stride", d.sample_stride > 0),
        (
            "diagnostics.sensitivity_threshold",
            is_set(d.sensitivity_threshold),
        ),
        (
            "diagnostics.lyapunov_threshold",
            is_set(d.lyapunov_threshold),
        ),
        (
            "diagnostics.nonlinearity_threshold",
            is_set(d.nonlinearity_threshold),
        ),
    ] {
        if on {
            set.push(name);
        }
    }
    set
}

/// Build an [`EphemerisConfig`] from the C-ABI ephemeris config, honouring
/// the shared sentinel rules. The `{force_model, frame, divisor}` triple it
/// carries is exactly what a [`BuiltSystem`](empyrean_core::propagation::BuiltSystem)'s
/// frozen key is compared against, so a handle built with the matching key
/// serves this config identically to the one-shot path.
///
/// The propagation block is converted by
/// [`build_propagation_config_from_c`](crate::propagate::build_propagation_config_from_c) —
/// the same converter every other C-ABI entry point uses — and then
/// narrowed to the ephemeris subset field by field. Both halves of that
/// are load-bearing: hand-rolling the narrow struct here is what silently
/// discarded `excluded_perturbers_naif`, `compute_stm`, `num_threads`,
/// `ephemeris_overlap_policy` and the whole `advanced` block from every ephemeris
/// call, and the narrowing is written without a `..Default::default()`
/// tail so that a field added to [`EphemerisPropagationConfig`] upstream
/// breaks this build rather than starting a fresh silent drop.
pub(crate) fn build_ephemeris_config_from_c(
    cfg_ref: &EmpyreanEphemerisConfig,
) -> Result<EphemerisConfig, String> {
    let unsupported = ephemeris_unsupported_propagation_fields(&cfg_ref.propagation);
    if !unsupported.is_empty() {
        return Err(format!(
            "ephemeris config sets propagation field(s) ephemeris generation cannot honour: \
             {}. Ephemeris generation runs with event detection and timeseries diagnostics \
             off and returns no channel for either, so these would be accepted and \
             discarded. Leave the `events` and `diagnostics` blocks zeroed on an ephemeris \
             config; call empyrean_propagate when you need them.",
            unsupported.join(", ")
        ));
    }

    let full = crate::propagate::build_propagation_config_from_c(&cfg_ref.propagation)?;
    // Field by field, no `..Default::default()` tail. The tail is what
    // turns an upstream field addition into a fresh silent drop; without
    // it the addition is a compile error here instead.
    let prop = EphemerisPropagationConfig {
        force_model: full.force_model,
        excluded_perturbers: full.excluded_perturbers,
        detect_ephemeris_overlap: full.detect_ephemeris_overlap,
        ephemeris_overlap_policy: full.ephemeris_overlap_policy,
        uncertainty_method: full.uncertainty_method,
        compute_stm: full.compute_stm,
        frame: full.frame,
        num_threads: full.num_threads,
        advanced: full.advanced,
    };
    let mut config = EphemerisConfig {
        propagation: prop,
        ..EphemerisConfig::default()
    };
    if cfg_ref.max_light_time_iterations > 0 {
        config.max_light_time_iterations = cfg_ref.max_light_time_iterations;
    }
    if cfg_ref.light_time_tolerance_days > 0.0 {
        config.light_time_tolerance_days = cfg_ref.light_time_tolerance_days;
    }
    config.compute_diagnostics = cfg_ref.compute_diagnostics != 0;
    Ok(config)
}

/// Marshal an [`EphemerisResult`] into the C-ABI [`EmpyreanEphemerisResult`]
/// (per-row optical entries + observation-sensitivity chains). Shared verbatim
/// by the one-shot [`empyrean_generate_ephemeris`] and the handle-based
/// [`empyrean_builtsystem_generate_ephemeris`](crate::built_system::empyrean_builtsystem_generate_ephemeris)
/// so both emit byte-identical result buffers. Returns 0 on success or a
/// negative allocation-failure code. Runs inside the caller's `catch_unwind`.
pub(crate) fn marshal_ephemeris_result(
    eph_result: &EphemerisResult<AU, empyrean_core::coordinates::Degrees>,
    result_out: *mut EmpyreanEphemerisResult,
) -> i32 {
    let ephemeris = &eph_result.ephemeris;
    let n = ephemeris.iter().count();

    // ── Generation warnings ──
    // Run-level (one list per call, not per row); Display-serialized so
    // the C ABI stays stable while the engine's warning taxonomy grows.
    // Marshaled FIRST so its allocation-failure return leaks nothing.
    let warn_src = &eph_result.diagnostics.warnings;
    let (warn_ptr, num_warnings) = if warn_src.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let nw = warn_src.len();
        let layout = std::alloc::Layout::array::<*mut std::ffi::c_char>(nw).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut std::ffi::c_char;
        if ptr.is_null() {
            set_last_error("allocation failed for warnings array");
            return -5;
        }
        for (k, w) in warn_src.iter().enumerate() {
            let s = CString::new(w.to_string()).unwrap_or_else(|_| CString::new("?").unwrap());
            unsafe {
                ptr.add(k).write(s.into_raw());
            }
        }
        (ptr, nw)
    };

    let out_ptr = if n > 0 {
        let layout = std::alloc::Layout::array::<EmpyreanEphemerisEntry>(n)
            .unwrap_or(std::alloc::Layout::new::<EmpyreanEphemerisEntry>());
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanEphemerisEntry;
        if ptr.is_null() {
            set_last_error("allocation failed for ephemeris entries array");
            return -5;
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    for (i, (orbit_id, coord, cov, obs_opt, light_time, aberrated)) in ephemeris.iter().enumerate()
    {
        // Uncertainty outputs the engine computes but the C ABI previously
        // dropped: the sky covariance, the aberrated
        // Cartesian state, and its covariance. Populate them; NaN-fill the
        // covariances when the input orbit carried none.
        let (has_covariance, covariance) = match cov {
            Some(c) => (1u8, c.matrix),
            None => (0u8, [[f64::NAN; 6]; 6]),
        };
        let aberrated_state = match aberrated {
            Some(a) => [a.x, a.y, a.z, a.vx, a.vy, a.vz],
            None => [f64::NAN; 6],
        };
        let (has_aberrated_covariance, aberrated_covariance) =
            match ephemeris.aberrated_covariance(i) {
                Some(c) => (1u8, c.matrix),
                None => (0u8, [[f64::NAN; 6]; 6]),
            };
        // SphericalCoordinates: r, lon (= RA), lat (= Dec), vr, vlon, vlat.
        // The ephemeris is generated with `Degrees` angular unit at the
        // facade layer (`Ephemeris::to_degrees()`), which converts EVERY
        // angular field — coordinates AND phase/elongation/zenith/azimuth/
        // hour_angle/lunar_elongation/position_angle/sky_rate — from
        // radians to degrees. So all of these are read straight through;
        // applying `.to_degrees()` here would double-convert (the
        // long-standing phase_angle/elongation degree bug, fixed below).
        let entry = EmpyreanEphemerisEntry {
            orbit_id: CString::new(orbit_id)
                .unwrap_or_else(|_| CString::new("?").unwrap())
                .into_raw(),
            epoch_mjd_tdb: coord.t.mjd_tdb(),
            ra_deg: coord.lon,
            dec_deg: coord.lat,
            rho_au: coord.r,
            vrho_au_day: coord.vr,
            vra_deg_day: coord.vlon,
            vdec_deg_day: coord.vlat,
            light_time_days: light_time.unwrap_or(f64::NAN),
            phase_angle_deg: ephemeris.phase_angle(i).unwrap_or(f64::NAN),
            elongation_deg: ephemeris.elongation(i).unwrap_or(f64::NAN),
            heliocentric_distance_au: ephemeris.heliocentric_distance(i).unwrap_or(f64::NAN),
            mag: ephemeris.mag(i).unwrap_or(f64::NAN),
            mag_sigma: ephemeris.sigma_mag(i).unwrap_or(f64::NAN),
            zenith_angle_deg: ephemeris.zenith_angle(i).unwrap_or(f64::NAN),
            azimuth_deg: ephemeris.azimuth(i).unwrap_or(f64::NAN),
            hour_angle_deg: ephemeris.hour_angle(i).unwrap_or(f64::NAN),
            lunar_elongation_deg: ephemeris.lunar_elongation(i).unwrap_or(f64::NAN),
            position_angle_deg: ephemeris.position_angle(i).unwrap_or(f64::NAN),
            sky_rate_deg_day: ephemeris.sky_rate(i).unwrap_or(f64::NAN),
            obs_code: {
                let mut c = [0u8; 4];
                if let Some(o) = obs_opt {
                    c[0] = o.code[0];
                    c[1] = o.code[1];
                    c[2] = o.code[2];
                }
                c[3] = 0;
                c
            },
            has_covariance,
            covariance,
            aberrated_state,
            has_aberrated_covariance,
            aberrated_covariance,
        };
        unsafe {
            out_ptr.add(i).write(entry);
        }
    }

    // ── Observation sensitivity chains ──
    // One row per (orbit, observer, epoch). Prefer the wide (state +
    // non-grav, 9-param) Jacobian/Hessian when present, else the 6-param.
    let mut sens_rows: Vec<EmpyreanObservationSensitivity> = Vec::new();
    for chain in &eph_result.sensitivity {
        // Engine observatory codes are 3 bytes today (the 4th C-field
        // byte is the NUL terminator). If the engine ever emits a longer
        // code, surface it instead of clipping to a prefix that names a
        // different observatory — the same contract the input paths
        // enforce.
        if chain.obs_code().len() > 3 {
            set_last_error(&format!(
                "sensitivity chain observatory code \"{}\" is longer than \
                 3 bytes; 4-character MPC codes are not yet supported",
                chain.obs_code()
            ));
            return -1;
        }
        let mut obs_code = [0u8; 4];
        for (k, b) in chain.obs_code().bytes().take(3).enumerate() {
            obs_code[k] = b;
        }
        let frame = frame_to_int(chain.frame());
        let origin = chain.origin().naif_id();
        let epochs = chain.epochs();
        for (i, &epoch_mjd_tdb) in epochs.iter().enumerate() {
            let (jac, n_params) = if let Some((jw, active_width)) = chain.jacobian_wide(i) {
                // v1.20.0: jacobian_wide is width-tagged — the wide STM
                // spans 7..=17 meaningful columns now (DT / AMRAT / thrust),
                // no longer a fixed 9, and is STORED at the MAX_WIDE stride.
                // Emit exactly the active_width columns so the buffer length
                // stays 6 * n_params.
                (
                    flatten_2d_active(&jw.matrix, active_width),
                    active_width as u8,
                )
            } else if let Some(j) = chain.jacobian(i) {
                (flatten_2d(&j.matrix), 6u8)
            } else {
                (Vec::new(), 6u8)
            };
            let hess = if let Some(hw) = chain.hessian_wide(i) {
                flatten_3d(&hw.tensor)
            } else if let Some(h) = chain.hessian(i) {
                flatten_3d(&h.tensor)
            } else {
                Vec::new()
            };
            let (jacobian, jacobian_len) = box_f64_array(&jac);
            let (hessian, hessian_len) = box_f64_array(&hess);
            let object_id = chain
                .object_id()
                .and_then(|s| CString::new(s).ok())
                .map(CString::into_raw)
                .unwrap_or(std::ptr::null_mut());
            sens_rows.push(EmpyreanObservationSensitivity {
                orbit_id: CString::new(chain.orbit_id())
                    .unwrap_or_else(|_| CString::new("?").unwrap())
                    .into_raw(),
                object_id,
                obs_code,
                epoch_mjd_tdb,
                n_params,
                jacobian,
                jacobian_len,
                hessian,
                hessian_len,
                frame,
                origin,
            });
        }
    }
    let (sens_ptr, num_sens) = if sens_rows.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let ns = sens_rows.len();
        let layout = std::alloc::Layout::array::<EmpyreanObservationSensitivity>(ns).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservationSensitivity;
        if ptr.is_null() {
            set_last_error("allocation failed for sensitivity array");
            return -5;
        }
        for (k, row) in sens_rows.into_iter().enumerate() {
            unsafe {
                ptr.add(k).write(row);
            }
        }
        (ptr, ns)
    };

    unsafe {
        (*result_out).entries = out_ptr;
        (*result_out).num_entries = n;
        (*result_out).sensitivity = sens_ptr;
        (*result_out).num_sensitivity = num_sens;
        (*result_out).warnings = warn_ptr;
        (*result_out).num_warnings = num_warnings;
    }

    0
}

/// Free an ephemeris result previously returned by `empyrean_generate_ephemeris()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_ephemeris_result_free(result: *mut EmpyreanEphemerisResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }

        let res = unsafe { &*result };
        let n = res.num_entries;

        if !res.entries.is_null() && n > 0 {
            for i in 0..n {
                let entry = unsafe { &*res.entries.add(i) };
                if !entry.orbit_id.is_null() {
                    drop(unsafe { CString::from_raw(entry.orbit_id) });
                }
            }
            let layout = std::alloc::Layout::array::<EmpyreanEphemerisEntry>(n).unwrap();
            unsafe {
                std::alloc::dealloc(res.entries as *mut u8, layout);
            }
        }

        if !res.sensitivity.is_null() && res.num_sensitivity > 0 {
            for i in 0..res.num_sensitivity {
                let row = unsafe { &*res.sensitivity.add(i) };
                if !row.orbit_id.is_null() {
                    drop(unsafe { CString::from_raw(row.orbit_id) });
                }
                if !row.object_id.is_null() {
                    drop(unsafe { CString::from_raw(row.object_id) });
                }
                free_f64_array(row.jacobian, row.jacobian_len);
                free_f64_array(row.hessian, row.hessian_len);
            }
            let layout =
                std::alloc::Layout::array::<EmpyreanObservationSensitivity>(res.num_sensitivity)
                    .unwrap();
            unsafe {
                std::alloc::dealloc(res.sensitivity as *mut u8, layout);
            }
        }

        if !res.warnings.is_null() && res.num_warnings > 0 {
            for i in 0..res.num_warnings {
                let p = unsafe { *res.warnings.add(i) };
                if !p.is_null() {
                    drop(unsafe { CString::from_raw(p) });
                }
            }
            let layout =
                std::alloc::Layout::array::<*mut std::ffi::c_char>(res.num_warnings).unwrap();
            unsafe {
                std::alloc::dealloc(res.warnings as *mut u8, layout);
            }
        }

        unsafe {
            (*result).entries = std::ptr::null_mut();
            (*result).num_entries = 0;
            (*result).sensitivity = std::ptr::null_mut();
            (*result).num_sensitivity = 0;
            (*result).warnings = std::ptr::null_mut();
            (*result).num_warnings = 0;
        }
    }));
}

// ── Sensitivity flattening + FFI heap helpers ──

/// Row-major flatten of a `6 × N` matrix.
fn flatten_2d<const N: usize>(m: &[[f64; N]; 6]) -> Vec<f64> {
    let mut v = Vec::with_capacity(6 * N);
    for row in m {
        v.extend_from_slice(row);
    }
    v
}

/// Row-major flatten of the leading `active_width` columns of the `6`
/// state rows. The width-tagged wide Jacobian is STORED at the `MAX_WIDE`
/// stride but only its first `active_width` columns are meaningful, so
/// emitting the full stride while reporting `n_params = active_width`
/// would misalign every non-`MAX_WIDE` Jacobian. Slicing to `active_width`
/// keeps the buffer at `6 * active_width`, matching the reported count.
fn flatten_2d_active<const N: usize>(m: &[[f64; N]; 6], active_width: usize) -> Vec<f64> {
    let w = active_width.min(N);
    let mut v = Vec::with_capacity(6 * w);
    for row in m {
        v.extend_from_slice(&row[..w]);
    }
    v
}

/// Row-major flatten of `6` symmetric `N × N` tensors.
fn flatten_3d<const N: usize>(t: &[[[f64; N]; N]; 6]) -> Vec<f64> {
    let mut v = Vec::with_capacity(6 * N * N);
    for mat in t {
        for row in mat {
            v.extend_from_slice(row);
        }
    }
    v
}

/// Copy a slice into a freshly heap-allocated C array. Returns
/// `(null, 0)` for an empty slice. Free with [`free_f64_array`].
fn box_f64_array(data: &[f64]) -> (*mut f64, usize) {
    if data.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<f64>(data.len()).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut f64;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    (ptr, data.len())
}

fn free_f64_array(ptr: *mut f64, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = std::alloc::Layout::array::<f64>(len).unwrap();
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observer_with_code(code: [u8; 4]) -> EmpyreanObserver {
        EmpyreanObserver {
            obs_code: code,
            epoch_mjd_tdb: 61000.0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
            vx: 0.0,
            vy: 0.01,
            vz: 0.0,
            observing_night: -1,
            frame: 0,  // ICRF
            origin: 0, // solar-system barycenter
        }
    }

    /// An observer tagged with any basis other than the construction one
    /// is refused by name. The two tail fields landed at ABI 3; before
    /// them the ICRF/SSB assumption was implicit, and a rotated station
    /// state fed to the optical forward model produced wrong astrometry
    /// with no diagnostic.
    #[test]
    fn a_non_construction_basis_observer_is_rejected() {
        let mut obs = observer_with_code(*b"W68\0");
        obs.frame = 1; // EclipticJ2000
        let err = build_observers_from_c(&[obs]).expect_err("ecliptic observer must not convert");
        assert!(err.contains("construction basis"), "{err}");
        assert!(err.contains("frame = 1"), "error names what it got: {err}");

        let mut obs = observer_with_code(*b"W68\0");
        obs.origin = 399; // Earth
        let err = build_observers_from_c(&[obs]).expect_err("geocentric observer must not convert");
        assert!(
            err.contains("origin = 399"),
            "error names what it got: {err}"
        );
    }

    /// A `memset(0)` observer is ICRF/SSB, so a caller who never touched
    /// the two new tail fields is unaffected by the ABI-3 widening.
    #[test]
    fn a_zeroed_basis_is_the_construction_basis() {
        let mut obs: EmpyreanObserver = unsafe { std::mem::zeroed() };
        obs.obs_code = *b"W68\0";
        build_observers_from_c(&[obs]).expect("a zeroed basis must convert");
    }

    /// A 4-byte observatory code must be a loud error at the C boundary:
    /// clipped to its 3-byte prefix it would silently alias a different
    /// observatory.
    #[test]
    fn four_byte_obs_code_is_rejected() {
        let err = build_observers_from_c(&[observer_with_code(*b"W68a")])
            .expect_err("4-byte observatory code must not convert");
        assert!(err.contains("W68a"), "error names the code: {err}");
        assert!(err.contains("4 bytes"), "error states the contract: {err}");
    }

    /// NUL-terminated 3-byte codes convert, space-padded to the engine's
    /// 3-byte registry key.
    #[test]
    fn three_byte_obs_code_converts() {
        let observers =
            build_observers_from_c(&[observer_with_code(*b"W68\0")]).expect("3-byte code");
        assert_eq!(observers[0].code, *b"W68");
    }
}

#[cfg(test)]
mod sensitivity_row_tests {
    use super::*;

    const HEADER: &str = include_str!("../../include/empyrean.h");

    /// Value of a `#define NAME <int>` in the generated header.
    fn header_define(name: &str) -> Option<usize> {
        HEADER.lines().find_map(|line| {
            let rest = line.strip_prefix("#define ")?;
            let (defined, value) = rest.split_once(' ')?;
            (defined == name).then(|| value.trim().parse().ok())?
        })
    }

    /// The row order is an ABI contract, so the values are pinned rather
    /// than merely self-consistent: a reorder here is a breaking change
    /// for every compiled consumer, and must be seen as one.
    #[test]
    fn row_constants_have_their_abi_values() {
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_RANGE, 0);
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_RA, 1);
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_DEC, 2);
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_VRANGE, 3);
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_VRA, 4);
        assert_eq!(EMPYREAN_SENSITIVITY_ROW_VDEC, 5);
    }

    /// The checked-in header is what C consumers actually compile
    /// against. cbindgen regenerates it from this file on build, so a
    /// stale committed header — the case where the Rust side moved and
    /// the shipped contract did not — shows up here.
    #[test]
    fn the_shipped_header_agrees_with_the_rust_constants() {
        for (name, want) in [
            (
                "EMPYREAN_SENSITIVITY_ROW_RANGE",
                EMPYREAN_SENSITIVITY_ROW_RANGE,
            ),
            ("EMPYREAN_SENSITIVITY_ROW_RA", EMPYREAN_SENSITIVITY_ROW_RA),
            ("EMPYREAN_SENSITIVITY_ROW_DEC", EMPYREAN_SENSITIVITY_ROW_DEC),
            (
                "EMPYREAN_SENSITIVITY_ROW_VRANGE",
                EMPYREAN_SENSITIVITY_ROW_VRANGE,
            ),
            ("EMPYREAN_SENSITIVITY_ROW_VRA", EMPYREAN_SENSITIVITY_ROW_VRA),
            (
                "EMPYREAN_SENSITIVITY_ROW_VDEC",
                EMPYREAN_SENSITIVITY_ROW_VDEC,
            ),
        ] {
            let got = header_define(name)
                .unwrap_or_else(|| panic!("{name} is missing from include/empyrean.h"));
            assert_eq!(
                got, want,
                "{name} disagrees between the header and the source"
            );
        }
    }
}

/// Contract tests for [`build_ephemeris_config_from_c`] — the seam that
/// used to hand-roll a three-field subset and silently drop the rest.
/// Every field the C struct advertises either arrives in the
/// [`EphemerisConfig`] the engine is handed, or is refused by name;
/// nothing in between.
#[cfg(test)]
mod ephemeris_config_conversion_tests {
    use super::*;
    use crate::propagate::{
        EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE, EMPYREAN_INTEGRATOR_DOP853,
        EMPYREAN_UNCERTAINTY_SECOND, EmpyreanPropagationConfig,
    };
    use empyrean_core::propagation::EphemerisOverlapPolicy;

    /// The state a C caller gets from `memset(&cfg, 0, sizeof cfg)`. All
    /// fields are plain data or raw pointers, so the zero pattern is a
    /// valid value of the type — and it is the documented way to build a
    /// config, so it must convert.
    fn zeroed_config() -> EmpyreanEphemerisConfig {
        unsafe { std::mem::zeroed::<EmpyreanEphemerisConfig>() }
    }

    fn prop_of(cfg: &mut EmpyreanEphemerisConfig) -> &mut EmpyreanPropagationConfig {
        &mut cfg.propagation
    }

    /// `EphemerisConfig` carries no `Debug`, so `expect_err` cannot name
    /// the unexpected success — take the message the long way.
    fn expect_refusal(cfg: &EmpyreanEphemerisConfig, what: &str) -> String {
        match build_ephemeris_config_from_c(cfg) {
            Ok(_) => panic!("{what} must not convert"),
            Err(e) => e,
        }
    }

    /// `memset(0)` is a valid ephemeris config: every sentinel resolves
    /// and neither unsupported block reads as "requested".
    #[test]
    fn a_zeroed_config_converts() {
        build_ephemeris_config_from_c(&zeroed_config()).expect("memset(0) config must convert");
    }

    /// The headline regression: `excluded_perturbers_naif` is the only
    /// way a caller can keep an SB441-N16 body out of its own perturber
    /// set, and no distribution layer could reach the ephemeris path with
    /// it because this converter never read the field.
    #[test]
    fn excluded_perturbers_reach_the_engine_config() {
        // 2000007 = Origin::Asteroid(7), Iris — an N16 self-perturber.
        let naif = [2_000_007i32, 2_000_004];
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).num_excluded_perturbers = naif.len();
        prop_of(&mut cfg).excluded_perturbers_naif = naif.as_ptr();

        let built = build_ephemeris_config_from_c(&cfg).expect("config converts");
        assert_eq!(
            built.propagation.excluded_perturbers,
            vec![Origin::Asteroid(7), Origin::Asteroid(4)],
            "excluded_perturbers_naif must reach EphemerisPropagationConfig in input order"
        );
    }

    /// An unknown NAIF id is an error, not a skipped entry — a dropped
    /// exclusion would leave the body self-perturbing with no diagnostic.
    #[test]
    fn an_unknown_excluded_perturber_is_rejected() {
        let naif = [424_242_424i32];
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).num_excluded_perturbers = naif.len();
        prop_of(&mut cfg).excluded_perturbers_naif = naif.as_ptr();

        let err = expect_refusal(&cfg, "an unknown NAIF id");
        assert!(err.contains("424242424"), "error names the id: {err}");
    }

    /// The other blocks that were dropped alongside it: `compute_stm`,
    /// `num_threads`, `frame`, and the whole `advanced` bag (the divisor
    /// in which is half of a `BuiltSystem`'s frozen key).
    #[test]
    fn the_remaining_dropped_fields_reach_the_engine_config() {
        let mut cfg = zeroed_config();
        {
            let p = prop_of(&mut cfg);
            p.force_model = 1; // Basic
            p.frame = 0; // ICRF
            p.compute_stm = 1;
            p.num_threads = 3;
            p.uncertainty_method.tag = EMPYREAN_UNCERTAINTY_SECOND;
            p.advanced.integrator = EMPYREAN_INTEGRATOR_DOP853;
            p.advanced.epsilon = 1e-11;
            p.advanced.dt_initial = f64::NAN;
            p.advanced.dt_min = 1e-6;
            p.advanced.encounter_timescale_divisor = 500.0;
            p.advanced.max_steps = 12_345;
            p.advanced.max_dense_steps = 6_789;
        }

        let built = build_ephemeris_config_from_c(&cfg).expect("config converts");
        let p = &built.propagation;
        assert!(p.compute_stm, "compute_stm must survive");
        assert_eq!(
            p.num_threads.map(|n| n.get()),
            Some(3),
            "num_threads must survive"
        );
        assert_eq!(p.frame, Frame::ICRF, "frame must survive");
        assert_eq!(p.advanced.epsilon, 1e-11, "advanced.epsilon must survive");
        assert_eq!(
            p.advanced.dt_min,
            Some(1e-6),
            "advanced.dt_min must survive"
        );
        assert_eq!(
            p.advanced.encounter_timescale_divisor, 500.0,
            "the divisor is compared against a BuiltSystem's frozen key — it must survive"
        );
        assert_eq!(p.advanced.max_steps, 12_345);
        assert_eq!(p.advanced.max_dense_steps, 6_789);
        assert_eq!(
            format!("{:?}", p.uncertainty_method),
            "SecondOrder",
            "uncertainty_method must survive"
        );
    }

    /// `ephemeris_overlap_policy` is the ABI-3 tail field, and the ephemeris path
    /// is where it matters most: the default policy skips integration,
    /// and ephemeris generation reads the dense trajectory integration
    /// would have produced. It must reach the narrowed config — the
    /// narrowing has no `..Default::default()` tail precisely so this
    /// cannot be forgotten.
    #[test]
    fn the_overlap_policy_reaches_the_engine_config() {
        let mut cfg = zeroed_config();
        let built = build_ephemeris_config_from_c(&cfg).expect("zeroed config converts");
        assert_eq!(
            built.propagation.ephemeris_overlap_policy,
            EphemerisOverlapPolicy::SubstituteSpk,
            "memset(0) must keep meaning the historical behaviour"
        );

        prop_of(&mut cfg).ephemeris_overlap_policy =
            EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE;
        let built = build_ephemeris_config_from_c(&cfg).expect("config converts");
        assert_eq!(
            built.propagation.ephemeris_overlap_policy,
            EphemerisOverlapPolicy::ExcludeAndIntegrate,
            "EXCLUDE_AND_INTEGRATE must reach EphemerisPropagationConfig"
        );
    }

    /// An unknown policy value is refused rather than resolved to the
    /// default — and the refusal reaches the ephemeris path because its
    /// converter routes through the shared one.
    #[test]
    fn an_unknown_overlap_policy_is_refused_on_the_ephemeris_path() {
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).ephemeris_overlap_policy = 7;
        let err = expect_refusal(&cfg, "an unknown overlap policy");
        assert!(err.contains('7'), "error names the value: {err}");
    }

    /// Event detection has no home on `EphemerisPropagationConfig` and no
    /// output channel on `EmpyreanEphemerisResult`. Asking for it is
    /// refused by name rather than accepted and dropped.
    #[test]
    fn an_event_request_is_refused_by_name() {
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).events.close_approaches = 1;
        prop_of(&mut cfg).events.dense_output = 1;

        let err = expect_refusal(&cfg, "an event request");
        assert!(
            err.contains("events.close_approaches") && err.contains("events.dense_output"),
            "error names every field the caller set: {err}"
        );
        assert!(
            err.contains("empyrean_propagate"),
            "error says where event detection does live: {err}"
        );
    }

    /// A body filter is a request even when every detection flag is off.
    #[test]
    fn an_event_body_filter_is_refused_by_name() {
        let bodies = [399i32];
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).events.num_body_filter = bodies.len();
        prop_of(&mut cfg).events.body_filter_naif = bodies.as_ptr();

        let err = expect_refusal(&cfg, "an event body filter");
        assert!(
            err.contains("events.body_filter_naif"),
            "error names the field: {err}"
        );
    }

    /// Same for the diagnostics timeseries.
    #[test]
    fn a_diagnostics_request_is_refused_by_name() {
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).diagnostics.lyapunov = 1;
        prop_of(&mut cfg).diagnostics.sample_stride = 5;
        prop_of(&mut cfg).diagnostics.sensitivity_threshold = 1e3;

        let err = expect_refusal(&cfg, "a diagnostics request");
        for field in [
            "diagnostics.lyapunov",
            "diagnostics.sample_stride",
            "diagnostics.sensitivity_threshold",
        ] {
            assert!(err.contains(field), "error names {field}: {err}");
        }
    }

    /// NaN is the ABI's `None` for the diagnostics thresholds; it is not
    /// a request and must not trip the refusal.
    #[test]
    fn nan_diagnostics_thresholds_are_not_a_request() {
        let mut cfg = zeroed_config();
        prop_of(&mut cfg).diagnostics.sensitivity_threshold = f64::NAN;
        prop_of(&mut cfg).diagnostics.lyapunov_threshold = f64::NAN;
        prop_of(&mut cfg).diagnostics.nonlinearity_threshold = f64::NAN;
        build_ephemeris_config_from_c(&cfg).expect("NaN thresholds must convert");
    }

    /// The ephemeris-specific knobs keep their own sentinel behaviour.
    #[test]
    fn light_time_sentinels_still_resolve() {
        let mut cfg = zeroed_config();
        cfg.max_light_time_iterations = 7;
        cfg.light_time_tolerance_days = 1e-12;
        cfg.compute_diagnostics = 1;

        let built = build_ephemeris_config_from_c(&cfg).expect("config converts");
        assert_eq!(built.max_light_time_iterations, 7);
        assert_eq!(built.light_time_tolerance_days, 1e-12);
        assert!(built.compute_diagnostics);
    }
}

/// End-to-end proof that the propagation-level knobs set on a C ephemeris
/// config reach the force model and change the answer — the behaviour the
/// hand-rolled converter made unreachable from every distribution layer.
///
/// These live in empyrean-c rather than in the Rust wrapper on purpose:
/// `empyrean-sys` dlopens a *released* `libempyrean`, so a wrapper-level
/// test would exercise whatever engine binary is cached rather than the
/// code in this tree, and would keep passing (wrongly) or failing
/// (spuriously) independent of this crate. Here `empyrean-core` is linked
/// directly, so the engine under test is the one being built.
///
/// Gated on a real data directory; skipped without one.
#[cfg(test)]
mod ephemeris_config_end_to_end_tests {
    use super::*;
    use crate::propagate::EMPYREAN_UNCERTAINTY_FIRST;

    /// A long enough arc that Jupiter's pull moves the predicted sky
    /// position well clear of round-off.
    const IC_EPOCH: f64 = 59000.0;
    const OBS_EPOCH: f64 = 59200.0;
    /// NAIF id of the Jupiter system barycenter.
    const JUPITER_BARYCENTER: i32 = 5;

    fn orbit() -> EmpyreanOrbit {
        let mut o: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        o.state = crate::CoordinateState {
            epoch_mjd_tdb: IC_EPOCH,
            elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: 0, // Cartesian
            frame: 0,          // ICRF
            origin: 10,        // Sun
            has_non_grav_cross: 0,
            non_grav_cross: [[0.0; 3]; 6],
        };
        o.non_grav_dt = f64::NAN;
        o.non_grav_dt_variance = f64::NAN;
        o.phot_system = -1;
        o.h_mag = f64::NAN;
        o.srp_amrat_variance = f64::NAN;
        o
    }

    fn observer() -> EmpyreanObserver {
        EmpyreanObserver {
            obs_code: *b"500\0",
            epoch_mjd_tdb: OBS_EPOCH,
            x: 0.9,
            y: -0.42,
            z: -0.18,
            vx: 0.0075,
            vy: 0.0148,
            vz: 0.0064,
            observing_night: -1,
            frame: 0,
            origin: 0,
        }
    }

    /// Standard tier, ICRF output. `zeroed` gives the right sentinels
    /// everywhere except the NaN-means-auto integrator slots.
    fn config(excluded: &[i32]) -> EmpyreanEphemerisConfig {
        let mut cfg: EmpyreanEphemerisConfig = unsafe { std::mem::zeroed() };
        cfg.propagation.force_model = 2; // Standard
        cfg.propagation.frame = 0; // ICRF
        cfg.propagation.uncertainty_method.tag = EMPYREAN_UNCERTAINTY_FIRST;
        cfg.propagation.advanced.dt_initial = f64::NAN;
        cfg.propagation.advanced.dt_min = f64::NAN;
        cfg.compute_diagnostics = 1;
        if !excluded.is_empty() {
            cfg.propagation.num_excluded_perturbers = excluded.len();
            cfg.propagation.excluded_perturbers_naif = excluded.as_ptr();
        }
        cfg
    }

    /// Run the full C-side ephemeris path (config conversion → engine)
    /// and return the single row's (RA, Dec) in degrees.
    fn ra_dec(ctx: &empyrean_core::Context, cfg: &EmpyreanEphemerisConfig) -> (f64, f64) {
        let built = build_ephemeris_config_from_c(cfg).expect("config converts");
        let orbits = build_orbits_for_ephemeris(&[orbit()]).expect("orbits convert");
        let observers = build_observers_from_c(&[observer()]).expect("observers convert");
        let result =
            generate_ephemeris(ctx, &orbits, &observers, &built).expect("ephemeris generates");
        let row = result
            .ephemeris
            .iter()
            .next()
            .expect("exactly one ephemeris row");
        (row.1.lon, row.1.lat)
    }

    /// Dropping Jupiter from the perturber set must move the answer. Two
    /// bit-identical results mean the exclusion never reached the force
    /// model — which is what every distribution layer produced before the
    /// fix, and what this test exists to catch if it regresses.
    #[test]
    fn excluding_jupiter_changes_the_predicted_sky_position() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping excluding_jupiter_changes_the_predicted_sky_position: no data dir");
            return;
        };
        let (ra_full, dec_full) = ra_dec(&ctx, &config(&[]));
        let excluded = [JUPITER_BARYCENTER];
        let (ra_excl, dec_excl) = ra_dec(&ctx, &config(&excluded));

        let d_ra = (ra_full - ra_excl).abs();
        let d_dec = (dec_full - dec_excl).abs();
        assert!(
            d_ra > 1e-9 || d_dec > 1e-9,
            "excluding Jupiter left the prediction unchanged \
             (dRA = {d_ra:e}, dDec = {d_dec:e}) — excluded_perturbers_naif did not \
             reach the ephemeris force model"
        );
    }

    /// An explicitly empty exclusion list must reproduce the shipped
    /// default bit-for-bit: routing the config through the shared
    /// converter must not have changed numbers for callers who asked for
    /// nothing.
    #[test]
    fn an_empty_exclusion_list_is_the_shipped_default() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping an_empty_exclusion_list_is_the_shipped_default: no data dir");
            return;
        };
        let (ra_a, dec_a) = ra_dec(&ctx, &config(&[]));
        let (ra_b, dec_b) = ra_dec(&ctx, &config(&[]));
        assert_eq!(ra_a, ra_b, "RA is deterministic");
        assert_eq!(dec_a, dec_b, "Dec is deterministic");
    }

    /// `compute_stm` reaching the engine, proved by its observable
    /// consequence rather than by re-reading the converter: with the flag
    /// set the run traces `Jet1<6>` and the observation-sensitivity chain
    /// carries a Jacobian at every epoch; without it (and without an
    /// input covariance) there is no STM to compose, and the chain comes
    /// back present-but-empty. The converter routing is the only thing
    /// between the C field and this behaviour — sabotage it and this test
    /// goes red.
    ///
    /// Counting *Jacobian-bearing epochs*, not chains: a placeholder
    /// chain is emitted per `(orbit, observer)` either way, so a chain
    /// count discriminates nothing.
    #[test]
    fn compute_stm_reaches_the_engine() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping compute_stm_reaches_the_engine: no data dir");
            return;
        };
        let partials = |compute_stm: u8| {
            let mut cfg = config(&[]);
            cfg.propagation.compute_stm = compute_stm;
            let built = build_ephemeris_config_from_c(&cfg).expect("config converts");
            let orbits = build_orbits_for_ephemeris(&[orbit()]).expect("orbits convert");
            let observers = build_observers_from_c(&[observer()]).expect("observers convert");
            let result =
                generate_ephemeris(&ctx, &orbits, &observers, &built).expect("ephemeris generates");
            result
                .sensitivity
                .iter()
                .map(|chain| {
                    (0..chain.epochs().len())
                        .filter(|&i| {
                            chain.jacobian(i).is_some() || chain.jacobian_wide(i).is_some()
                        })
                        .count()
                })
                .sum::<usize>()
        };
        assert_eq!(
            partials(0),
            0,
            "without compute_stm and without an input covariance there is no STM to \
             compose, so no epoch should carry a Jacobian"
        );
        assert!(
            partials(1) > 0,
            "compute_stm = 1 must reach the engine and force Jet1<6> integration, which \
             fills the observation-sensitivity Jacobians — zero partials means the C \
             field was dropped between build_ephemeris_config_from_c and \
             EphemerisPropagationConfig"
        );
    }

    // ── The ABI boundary itself ──────────────────────────────────
    //
    // The tests above call the internal converter and
    // `empyrean_core::generate_ephemeris` directly, so nothing yet
    // exercises the exported symbol plus `marshal_ephemeris_result` for
    // the no-input-covariance case — which is exactly the path a C
    // consumer sees, and exactly the claim ("sensitivity without an input
    // covariance") that has to be provable from outside.

    // A context, or `None` when the host genuinely has no kernels. The
    // crate-wide helper panics instead of skipping when the environment
    // says the kernels are supposed to be here — a data-gated test that
    // turns itself off on the one runner configured to run it is a green
    // no-op, not a test.
    use crate::testing::context_or_skip;

    fn observer_at(epoch_mjd_tdb: f64) -> EmpyreanObserver {
        EmpyreanObserver {
            epoch_mjd_tdb,
            ..observer()
        }
    }

    /// The three observer epochs the boundary tests share. A short arc:
    /// the point is the marshalled buffer, not a long integration.
    fn observers_three() -> [EmpyreanObserver; 3] {
        [
            observer_at(OBS_EPOCH),
            observer_at(OBS_EPOCH + 10.0),
            observer_at(OBS_EPOCH + 20.0),
        ]
    }

    /// Call the exported symbol and hand back the populated result. The
    /// caller frees it.
    fn generate_through_the_abi(
        ctx: &empyrean_core::Context,
        orbit: &EmpyreanOrbit,
        cfg: &EmpyreanEphemerisConfig,
        observers: &[EmpyreanObserver],
    ) -> EmpyreanEphemerisResult {
        let ctx_ptr: *const EmpyreanContext = ctx;
        // SAFETY: `#[repr(C)]` scalars and pointers; the exported entry
        // point overwrites every field on success.
        let mut out: EmpyreanEphemerisResult = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            empyrean_generate_ephemeris(
                ctx_ptr,
                orbit,
                1,
                observers.as_ptr(),
                observers.len(),
                cfg,
                &mut out,
            )
        };
        assert_eq!(rc, 0, "empyrean_generate_ephemeris returned {rc}");
        out
    }

    /// Collect each row's Jacobian as an owned buffer, asserting the
    /// pointer/length contract on the way out.
    fn jacobians(result: &EmpyreanEphemerisResult) -> Vec<Vec<f64>> {
        assert!(!result.sensitivity.is_null(), "sensitivity array is null");
        (0..result.num_sensitivity)
            .map(|i| {
                let row = unsafe { &*result.sensitivity.add(i) };
                assert_eq!(row.n_params, 6, "state-only fit is six columns wide");
                assert!(
                    !row.jacobian.is_null(),
                    "row {i} carries no Jacobian through the ABI"
                );
                assert_eq!(
                    row.jacobian_len,
                    6 * row.n_params as usize,
                    "row {i}: jacobian_len must be 6 * n_params"
                );
                unsafe { std::slice::from_raw_parts(row.jacobian, row.jacobian_len) }.to_vec()
            })
            .collect()
    }

    /// An orbit carrying a small diagonal covariance — the seeded-prior
    /// shim shape this API is meant to make unnecessary.
    fn orbit_with_dummy_covariance() -> EmpyreanOrbit {
        let mut o = orbit();
        o.state.has_covariance = 1;
        for i in 0..6 {
            o.state.covariance[i][i] = 1e-12;
        }
        o
    }

    /// The whole A6 claim, asserted at the surface a C consumer touches:
    /// `compute_stm = 1` with **no** input covariance produces the six
    /// observation-sensitivity rows, populated, finite, and physically
    /// sized — and the free path nulls them.
    #[test]
    fn sensitivity_survives_the_abi_with_no_input_covariance() {
        let Some(ctx) = context_or_skip("sensitivity_survives_the_abi_with_no_input_covariance")
        else {
            return;
        };
        let mut cfg = config(&[]);
        cfg.propagation.compute_stm = 1;
        let o = orbit();
        assert_eq!(o.state.has_covariance, 0, "the fixture carries no Σ₀");
        let observers = observers_three();

        let mut out = generate_through_the_abi(&ctx, &o, &cfg, &observers);
        assert_eq!(
            out.num_sensitivity,
            observers.len(),
            "one sensitivity row per observer epoch"
        );

        for (i, jac) in jacobians(&out).iter().enumerate() {
            assert!(
                jac.iter().all(|v| v.is_finite()),
                "row {i} carries a non-finite partial"
            );
            // Position partials of the two angles: a defaulted or
            // zero-filled buffer fails here, and so does a wrong-units or
            // wrong-row read — ∂RA/∂x₀ at ~1 AU is O(10) deg/AU, so the
            // true value sits comfortably inside these decades.
            for (name, row) in [
                ("RA", EMPYREAN_SENSITIVITY_ROW_RA),
                ("Dec", EMPYREAN_SENSITIVITY_ROW_DEC),
            ] {
                let peak = (0..3)
                    .map(|c| jac[row * 6 + c].abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    peak > 1e-2 && peak < 1e4,
                    "row {i}: |∂{name}/∂r₀| peaked at {peak:e} deg/AU, outside 1e-2..1e4"
                );
            }
        }

        unsafe { empyrean_ephemeris_result_free(&mut out) };
        assert!(
            out.sensitivity.is_null(),
            "free nulls the sensitivity array"
        );
        assert_eq!(out.num_sensitivity, 0);
        assert!(out.entries.is_null(), "free nulls the entry array");
        assert_eq!(out.num_entries, 0);
    }

    /// The shim-removal warrant: the partials produced with **no**
    /// covariance and `compute_stm = 1` are not merely present, they are
    /// the same numbers the dummy-covariance workaround was producing.
    ///
    /// Strict `f64` equality is deliberate and must stay strict. Both runs
    /// take the same `Jet1<6>` arm and Σ₀ feeds only the covariance
    /// composition, never the integration — so a red here does not mean
    /// "the tolerance is too tight", it means the seeded Σ₀ has started
    /// influencing step selection, and the two paths no longer produce the
    /// same dynamics. Do not "fix" this with a tolerance.
    #[test]
    fn sensitivity_is_bit_identical_with_and_without_the_covariance_shim() {
        let Some(ctx) =
            context_or_skip("sensitivity_is_bit_identical_with_and_without_the_covariance_shim")
        else {
            return;
        };
        let observers = observers_three();

        let mut stm_cfg = config(&[]);
        stm_cfg.propagation.compute_stm = 1;
        let mut without = generate_through_the_abi(&ctx, &orbit(), &stm_cfg, &observers);

        // The shim: a dummy Σ₀ switched the partials on, with the flag off.
        let shim_cfg = config(&[]);
        assert_eq!(shim_cfg.propagation.compute_stm, 0);
        let mut with_shim =
            generate_through_the_abi(&ctx, &orbit_with_dummy_covariance(), &shim_cfg, &observers);

        let a = jacobians(&without);
        let b = jacobians(&with_shim);
        assert_eq!(a.len(), b.len(), "same number of sensitivity rows");
        assert!(!a.is_empty(), "the comparison must not be vacuous");
        for (i, (ja, jb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ja.len(), jb.len(), "row {i}: same Jacobian length");
            for (k, (x, y)) in ja.iter().zip(jb.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "row {i}, element {k}: {x:e} vs {y:e} — the no-covariance partials \
                     must be bit-identical to the dummy-covariance shim's"
                );
            }
        }

        unsafe {
            empyrean_ephemeris_result_free(&mut without);
            empyrean_ephemeris_result_free(&mut with_shim);
        }
    }

    /// The negative control: with the flag off and no covariance there is
    /// no STM to compose, and the ABI must say so with a null pointer and
    /// a zero length — never a zero-filled buffer, which a caller cannot
    /// tell apart from a real all-zero Jacobian.
    #[test]
    fn compute_stm_zero_leaves_the_jacobian_null_through_the_abi() {
        let Some(ctx) =
            context_or_skip("compute_stm_zero_leaves_the_jacobian_null_through_the_abi")
        else {
            return;
        };
        let cfg = config(&[]);
        assert_eq!(cfg.propagation.compute_stm, 0);
        let observers = observers_three();
        let mut out = generate_through_the_abi(&ctx, &orbit(), &cfg, &observers);

        for i in 0..out.num_sensitivity {
            let row = unsafe { &*out.sensitivity.add(i) };
            assert!(
                row.jacobian.is_null(),
                "row {i}: no STM was traced, so the Jacobian must be null"
            );
            assert_eq!(row.jacobian_len, 0, "row {i}: a null Jacobian has length 0");
        }

        unsafe { empyrean_ephemeris_result_free(&mut out) };
    }
}
