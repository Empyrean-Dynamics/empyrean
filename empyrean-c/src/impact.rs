//! C ABI for the multi-method impact-probability and B-plane wrappers
//! exposed by [`empyrean_core::impact::compute_impact_probabilities`]
//! and [`empyrean_core::impact::compute_b_planes`].
//!
//! Both entry points take an array of orbits, an end epoch, and a
//! caller-supplied list of [`UncertaintyMethod`](crate::propagate::EmpyreanUncertaintyMethod)
//! variants. They run one full propagation per method (the "easy
//! path" — a single-pass refactor is planned) and emit
//! flat result arrays tagged with the method that produced each row.
//!
//! Memory ownership: the result-out struct is allocated by the
//! caller; pointer fields inside it are allocated by Rust and must be
//! freed via the matching `*_result_free` entry points after use.

use std::ffi::CString;
use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::coordinate_state_to_coordinates;
use empyrean_core::impact::{compute_b_planes, compute_impact_probabilities};
use empyrean_core::nongrav::{GFunction, NonGravModel, NonGravParams};
use empyrean_core::orbits::Orbits;
use empyrean_core::time::Epoch;

use crate::propagate::{EmpyreanOrbit, EmpyreanUncertaintyMethod, flat_to_uncertainty_method};
use crate::{EmpyreanContext, set_last_error};

// ── C-compatible result types ──────────────────────────────

/// One impact-probability record emitted by
/// [`empyrean_compute_impact_probabilities`]. Mirrors
/// [`empyrean_core::impact::ImpactProbability`] flattened to primitives;
/// nullable fields use NaN sentinels (Option<f64>) or 0 / null
/// (Option<usize>).
///
/// `method_tag` matches the `EMPYREAN_UNCERTAINTY_*` constants in
/// `propagate.rs` and identifies which propagation produced this row.
#[repr(C)]
pub struct EmpyreanImpactProbability {
    /// Variant tag of the [`UncertaintyMethod`] that produced this row.
    pub method_tag: u8,
    /// Owning C string — caller frees via `*_result_free`.
    pub orbit_id: *mut std::ffi::c_char,
    pub object_id: *mut std::ffi::c_char,
    pub body: *mut std::ffi::c_char,
    pub body_naif_id: i32,
    pub epoch_mjd_tdb: f64,
    pub miss_distance_au: f64,
    pub miss_distance_km: f64,
    pub effective_radius_au: f64,
    pub effective_radius_km: f64,
    pub sigma_distance_au: f64,
    pub sigma_distance_km: f64,
    pub ip_linear: f64,
    pub relative_velocity_au_day: f64,
    /// `ip_second_order`, NaN when not available.
    pub ip_second_order: f64,
    /// Jet2 nonlinearity κ, NaN when not available.
    pub nonlinearity: f64,
    /// AGM-mixed IP, NaN when not available.
    pub ip_agm: f64,
    /// MC-sampled IP, NaN when not available.
    pub ip_mc: f64,
    /// Number of MC samples drawn (0 when MC was not used).
    pub mc_n_samples: u64,
    /// Number of MC samples that impacted (0 when MC was not used).
    pub mc_n_impacts: u64,
}

/// One B-plane breakdown emitted by [`empyrean_compute_b_planes`].
/// Mirrors [`empyrean_core::impact::BPlaneData`] flattened to primitives.
/// Nullable f64 fields use NaN sentinels.
#[repr(C)]
pub struct EmpyreanBPlane {
    /// Variant tag of the [`UncertaintyMethod`] that produced this row.
    pub method_tag: u8,
    /// Owning C string — caller frees via `*_result_free`.
    pub body: *mut std::ffi::c_char,
    pub epoch_mjd_tdb: f64,
    pub b_dot_t_km: f64,
    pub b_dot_r_km: f64,
    pub b_mag_km: f64,
    pub v_inf_km_s: f64,
    pub effective_radius_km: f64,
    pub body_radius_km: f64,
    /// `[σ_TT, σ_TR, σ_RR]` covariance elements (km²); NaN when no
    /// projected covariance is available.
    pub cov_b_plane: [f64; 3],
    pub semi_major_3sig_km: f64,
    pub semi_minor_3sig_km: f64,
    pub ellipse_angle_rad: f64,
    pub ip_linear: f64,
}

#[repr(C)]
pub struct EmpyreanImpactProbabilitiesResult {
    pub records: *mut EmpyreanImpactProbability,
    pub num_records: usize,
}

#[repr(C)]
pub struct EmpyreanBPlanesResult {
    pub records: *mut EmpyreanBPlane,
    pub num_records: usize,
}

// ── Helpers ────────────────────────────────────────────────

fn to_c_str(s: &str) -> *mut std::ffi::c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("?").unwrap())
        .into_raw()
}

unsafe fn free_c_str(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

/// Build [`Orbits<AU>`] from a slice of [`EmpyreanOrbit`], mirroring
/// the logic in [`crate::propagate::empyrean_propagate`]. Lifted out
/// into a helper so both impact and propagate paths share one
/// implementation.
fn build_orbits_from_c(
    orbit_slice: &[EmpyreanOrbit],
) -> Result<Orbits<empyrean_core::coordinates::AU>, String> {
    let mut orbits: Orbits<empyrean_core::coordinates::AU> = Orbits::empty();
    for (i, orbit) in orbit_slice.iter().enumerate() {
        let state = orbit.state.to_empyrean();
        let coords =
            coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit {i}: {e}"))?;
        let id = format!("orbit_{i}");
        orbits
            .push(id, coords.into_radians())
            .map_err(|e| format!("orbit {i}: {e}"))?;
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
                dt: None,
                dt_variance: None,
            };
            orbits.set_non_grav_params(i, Some(params));
        }
        if let Some(tp) = crate::propagate::empyrean_orbit_thrust_params(orbit)
            .map_err(|e| format!("orbit {i}: {e}"))?
        {
            orbits.set_thrust_params(i, Some(tp));
        }
    }
    Ok(orbits)
}

fn build_body_filter(
    body_filter_naif: *const i32,
    num_body_filter: usize,
) -> Result<Option<Vec<Origin>>, String> {
    if num_body_filter == 0 || body_filter_naif.is_null() {
        return Ok(None);
    }
    let slice = unsafe { std::slice::from_raw_parts(body_filter_naif, num_body_filter) };
    let mut filter = Vec::with_capacity(slice.len());
    for &naif in slice {
        let origin = Origin::from_naif_id(naif)
            .ok_or_else(|| format!("unknown NAIF id in body_filter: {naif}"))?;
        filter.push(origin);
    }
    Ok(Some(filter))
}

// ── Entry points ───────────────────────────────────────────

/// Run [`empyrean_core::impact::compute_impact_probabilities`] over a
/// caller-supplied set of [`UncertaintyMethod`] variants and return
/// the flattened IP records tagged by method.
///
/// Caller is responsible for freeing the result via
/// [`empyrean_compute_impact_probabilities_result_free`].
///
/// # Returns
/// `0` on success, negative on failure (see [`crate::set_last_error`]).
///
/// # Safety
/// All non-null pointer arguments must point to valid arrays of the
/// indicated length for the duration of the call. The output struct
/// is allocated by the caller; pointer fields inside it are allocated
/// here and owned by the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_compute_impact_probabilities(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    end_mjd_tdb: f64,
    methods_ptr: *const EmpyreanUncertaintyMethod,
    num_methods: usize,
    body_filter_naif: *const i32,
    num_body_filter: usize,
    result_out: *mut EmpyreanImpactProbabilitiesResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || orbits_ptr.is_null() || methods_ptr.is_null() || result_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_methods == 0 {
            set_last_error("methods array must be non-empty");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let methods_slice = unsafe { std::slice::from_raw_parts(methods_ptr, num_methods) };

        let orbits = match build_orbits_from_c(orbit_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let body_filter = match build_body_filter(body_filter_naif, num_body_filter) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let mut methods = Vec::with_capacity(num_methods);
        for m in methods_slice {
            match flat_to_uncertainty_method(m) {
                Ok(um) => methods.push(um),
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        }

        let end_epoch = Epoch::from_mjd_tdb(end_mjd_tdb);
        let by_method = match compute_impact_probabilities(
            ctx_ref,
            &orbits,
            end_epoch,
            &methods,
            body_filter,
        ) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&e.to_string());
                return -2;
            }
        };

        // Flatten into a single tagged array.
        let mut records: Vec<EmpyreanImpactProbability> = Vec::new();
        for (entry, m) in by_method.iter().zip(methods_slice.iter()) {
            for ip in &entry.impact_probabilities {
                records.push(EmpyreanImpactProbability {
                    method_tag: m.tag,
                    orbit_id: to_c_str(&ip.orbit_id),
                    object_id: to_c_str(ip.object_id.as_deref().unwrap_or("")),
                    body: to_c_str(&ip.body_name),
                    body_naif_id: ip.body_origin.naif_id(),
                    epoch_mjd_tdb: ip.epoch.mjd_tdb(),
                    miss_distance_au: ip.miss_distance_au,
                    miss_distance_km: ip.miss_distance_km,
                    effective_radius_au: ip.effective_radius_au,
                    effective_radius_km: ip.effective_radius_km,
                    sigma_distance_au: ip.sigma_distance_au,
                    sigma_distance_km: ip.sigma_distance_km,
                    ip_linear: ip.ip_linear,
                    relative_velocity_au_day: ip.relative_velocity_au_day,
                    ip_second_order: ip.ip_second_order.unwrap_or(f64::NAN),
                    nonlinearity: ip.nonlinearity.unwrap_or(f64::NAN),
                    ip_agm: ip.ip_agm.unwrap_or(f64::NAN),
                    ip_mc: ip.ip_mc.unwrap_or(f64::NAN),
                    mc_n_samples: ip.mc_n_samples.unwrap_or(0) as u64,
                    mc_n_impacts: ip.mc_n_impacts.unwrap_or(0) as u64,
                });
            }
        }

        let n = records.len();
        let records_ptr = if n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanImpactProbability>(n).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanImpactProbability;
            if ptr.is_null() {
                set_last_error("allocation failed for IP records");
                return -5;
            }
            for (i, rec) in records.into_iter().enumerate() {
                unsafe { ptr.add(i).write(rec) };
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        unsafe {
            (*result_out).records = records_ptr;
            (*result_out).num_records = n;
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in compute_impact_probabilities");
            -3
        }
    }
}

/// Free the records array allocated by
/// [`empyrean_compute_impact_probabilities`]. After calling, the
/// struct's `records` is reset to null and `num_records` to 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_compute_impact_probabilities_result_free(
    result: *mut EmpyreanImpactProbabilitiesResult,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &mut *result };
        if !res.records.is_null() && res.num_records > 0 {
            for i in 0..res.num_records {
                let rec = unsafe { &*res.records.add(i) };
                unsafe {
                    free_c_str(rec.orbit_id);
                    free_c_str(rec.object_id);
                    free_c_str(rec.body);
                }
            }
            let layout =
                std::alloc::Layout::array::<EmpyreanImpactProbability>(res.num_records).unwrap();
            unsafe { std::alloc::dealloc(res.records as *mut u8, layout) };
        }
        res.records = std::ptr::null_mut();
        res.num_records = 0;
    }));
}

/// Run [`empyrean_core::impact::compute_b_planes`] over a caller-
/// supplied set of [`UncertaintyMethod`] variants and return the
/// flattened B-plane records tagged by method.
///
/// Caller is responsible for freeing the result via
/// [`empyrean_compute_b_planes_result_free`].
///
/// # Safety
/// Same contract as
/// [`empyrean_compute_impact_probabilities`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_compute_b_planes(
    ctx: *const EmpyreanContext,
    orbits_ptr: *const EmpyreanOrbit,
    num_orbits: usize,
    end_mjd_tdb: f64,
    methods_ptr: *const EmpyreanUncertaintyMethod,
    num_methods: usize,
    body_filter_naif: *const i32,
    num_body_filter: usize,
    result_out: *mut EmpyreanBPlanesResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || orbits_ptr.is_null() || methods_ptr.is_null() || result_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_methods == 0 {
            set_last_error("methods array must be non-empty");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits_ptr, num_orbits) };
        let methods_slice = unsafe { std::slice::from_raw_parts(methods_ptr, num_methods) };

        let orbits = match build_orbits_from_c(orbit_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let body_filter = match build_body_filter(body_filter_naif, num_body_filter) {
            Ok(b) => b,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let mut methods = Vec::with_capacity(num_methods);
        for m in methods_slice {
            match flat_to_uncertainty_method(m) {
                Ok(um) => methods.push(um),
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        }

        let end_epoch = Epoch::from_mjd_tdb(end_mjd_tdb);
        let by_method = match compute_b_planes(ctx_ref, &orbits, end_epoch, &methods, body_filter) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&e.to_string());
                return -2;
            }
        };

        let mut records: Vec<EmpyreanBPlane> = Vec::new();
        for (entry, m) in by_method.iter().zip(methods_slice.iter()) {
            for bp in &entry.b_planes {
                records.push(EmpyreanBPlane {
                    method_tag: m.tag,
                    body: to_c_str(&bp.body),
                    epoch_mjd_tdb: bp.epoch.mjd_tdb(),
                    b_dot_t_km: bp.b_dot_t,
                    b_dot_r_km: bp.b_dot_r,
                    b_mag_km: bp.b_mag,
                    v_inf_km_s: bp.v_inf_km_s,
                    effective_radius_km: bp.effective_radius_km,
                    body_radius_km: bp.body_radius_km,
                    cov_b_plane: bp.cov_b_plane.unwrap_or([f64::NAN; 3]),
                    semi_major_3sig_km: bp.semi_major_3sig_km.unwrap_or(f64::NAN),
                    semi_minor_3sig_km: bp.semi_minor_3sig_km.unwrap_or(f64::NAN),
                    ellipse_angle_rad: bp.ellipse_angle.unwrap_or(f64::NAN),
                    ip_linear: bp.ip_linear.unwrap_or(f64::NAN),
                });
            }
        }

        let n = records.len();
        let records_ptr = if n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanBPlane>(n).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanBPlane;
            if ptr.is_null() {
                set_last_error("allocation failed for B-plane records");
                return -5;
            }
            for (i, rec) in records.into_iter().enumerate() {
                unsafe { ptr.add(i).write(rec) };
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        unsafe {
            (*result_out).records = records_ptr;
            (*result_out).num_records = n;
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in compute_b_planes");
            -3
        }
    }
}

/// Free the records array allocated by [`empyrean_compute_b_planes`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_compute_b_planes_result_free(result: *mut EmpyreanBPlanesResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &mut *result };
        if !res.records.is_null() && res.num_records > 0 {
            for i in 0..res.num_records {
                let rec = unsafe { &*res.records.add(i) };
                unsafe { free_c_str(rec.body) };
            }
            let layout = std::alloc::Layout::array::<EmpyreanBPlane>(res.num_records).unwrap();
            unsafe { std::alloc::dealloc(res.records as *mut u8, layout) };
        }
        res.records = std::ptr::null_mut();
        res.num_records = 0;
    }));
}
