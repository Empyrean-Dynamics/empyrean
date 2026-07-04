use std::ffi::CString;
use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::{coordinate_state_to_coordinates, frame_to_int};
use empyrean_core::coordinates::{AU, CartesianCoordinates, Frame};
use empyrean_core::ephemeris::{EphemerisConfig, EphemerisPropagationConfig, generate_ephemeris};
use empyrean_core::nongrav::{GFunction, NonGravModel, NonGravParams};
use empyrean_core::observers::Observer;
use empyrean_core::orbits::Orbits;
use empyrean_core::time::Epoch;

use crate::observers::EmpyreanObserver;
use crate::propagate::{
    EmpyreanOrbit, EmpyreanPropagationConfig, empyrean_orbit_photometric_params, int_to_force_model,
};
use crate::{EmpyreanContext, set_last_error};

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
    /// Magnitude uncertainty. NaN if unavailable.
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
}

/// One observation-sensitivity row — the partial derivatives of the
/// sky-plane observable w.r.t. the input state, for a single
/// `(orbit, observer, epoch)`. One row per observation epoch within each
/// `(orbit_id, obs_code)` chain. Owning struct: free
/// the whole result with [`empyrean_ephemeris_result_free`].
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
    pub jacobian: *mut f64,
    /// Length of `jacobian` (`6 * n_params`), 0 when null.
    pub jacobian_len: usize,
    /// Hessian ∂²(observable)/∂(input)², row-major `[6][n_params][n_params]`
    /// flattened (length `6 * n_params * n_params`). Null unless a
    /// second-order method (Jet2) ran.
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
}

/// Ephemeris-generation configuration.
///
/// Wraps the inner [`EmpyreanPropagationConfig`] (force model,
/// uncertainty method, output frame) plus the light-time iteration
/// controls and a diagnostics toggle. The propagation runs internally
/// to bring each orbit to its observation epoch, so every
/// propagation-level knob applies here too.
#[repr(C)]
pub struct EmpyreanEphemerisConfig {
    /// Inner propagation configuration applied to the trajectory that
    /// brings each orbit to its observation epoch.
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
/// On success, `result_out` is populated with ephemeris entries.
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

        let fm = match int_to_force_model(cfg_ref.propagation.force_model) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let frame = match empyrean_core::convert::int_to_frame(cfg_ref.propagation.frame) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };

        // Build Orbits<AU>.
        let mut orbits: Orbits<AU> = Orbits::empty();
        for (i, orbit) in orbit_slice.iter().enumerate() {
            let state = orbit.state.to_empyrean();
            let coords = match coordinate_state_to_coordinates(&state) {
                Ok(c) => c,
                Err(e) => {
                    set_last_error(&format!("orbit {i}: {e}"));
                    return -1;
                }
            };
            let id = format!("orbit_{i}");
            if let Err(e) = orbits.push(id, coords.into_radians()) {
                set_last_error(&format!("orbit {i}: {e}"));
                return -1;
            }
            if orbit.a1 != 0.0 || orbit.a2 != 0.0 || orbit.a3 != 0.0 {
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
                let params = NonGravParams {
                    a1: orbit.a1,
                    a2: orbit.a2,
                    a3: orbit.a3,
                    model: NonGravModel::MarsdenSekanina(g_func),
                    covariance: None,
                    dt: if orbit.non_grav_dt.is_finite() {
                        Some(orbit.non_grav_dt)
                    } else {
                        None
                    },
                };
                orbits.set_non_grav_params(i, Some(params));
            }
            if let Some(ph) = empyrean_orbit_photometric_params(orbit) {
                orbits.set_photometric_params(i, Some(ph));
            }
        }

        // Build Vec<Observer> from raw position/velocity vectors.
        let mut observers: Vec<Observer> = Vec::with_capacity(num_observers);
        for obs in observer_slice {
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

        let mut prop = EphemerisPropagationConfig {
            force_model: fm.into(),
            frame,
            ..EphemerisPropagationConfig::default()
        };
        match crate::propagate::flat_to_uncertainty_method(&cfg_ref.propagation.uncertainty_method)
        {
            Ok(m) => prop.uncertainty_method = m.into(),
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        }
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

        let eph_result = match generate_ephemeris(ctx_ref, &orbits, &observers, &config) {
            Ok(e) => e,
            Err(e) => {
                set_last_error(&e.to_string());
                return -4;
            }
        };

        let ephemeris = &eph_result.ephemeris;
        let n = ephemeris.iter().count();

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

        for (i, (orbit_id, coord, _cov, obs_opt, light_time, _aberrated)) in
            ephemeris.iter().enumerate()
        {
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
            let mut obs_code = [0u8; 4];
            for (k, b) in chain.obs_code().bytes().take(3).enumerate() {
                obs_code[k] = b;
            }
            let frame = frame_to_int(chain.frame());
            let origin = chain.origin().naif_id();
            let epochs = chain.epochs();
            for (i, &epoch_mjd_tdb) in epochs.iter().enumerate() {
                let (jac, n_params) = if let Some(jw) = chain.jacobian_wide(i) {
                    (flatten_2d(&jw.matrix), 9u8)
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
        }

        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_generate_ephemeris");
            -99
        }
    }
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

        unsafe {
            (*result).entries = std::ptr::null_mut();
            (*result).num_entries = 0;
            (*result).sensitivity = std::ptr::null_mut();
            (*result).num_sensitivity = 0;
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
