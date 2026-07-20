//! External small-body queries.

use crate::ephemeris::EphemerisEntry;
use crate::error::{Error, Result};
use crate::io::OrbitBatch;
use crate::od::{Observations, RadarObservation};
use std::ffi::CString;
use std::path::Path;

/// Query the JPL Small-Body Database for one or more orbits by
/// designation, name, or SPK ID.
///
/// `cache_dir` is an optional on-disk cache directory; when provided,
/// SBDB JSON responses are cached there and re-used on subsequent
/// queries for the same object. Pass `None` to skip caching.
///
/// Returns an [`OrbitBatch`] with one entry per object ID in input
/// order. Throws if any object cannot be found or the SBDB API returns
/// an error.
pub fn query_sbdb(object_ids: &[&str], cache_dir: Option<&Path>) -> Result<OrbitBatch> {
    if object_ids.is_empty() {
        return OrbitBatch::new(Vec::new(), Vec::new(), Vec::new());
    }
    let id_cstrings: Vec<CString> = object_ids
        .iter()
        .map(|id| {
            CString::new(id.as_bytes())
                .map_err(|_| Error::invalid_input(format!("object_id contains a NUL byte: {id}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let id_ptrs: Vec<*const std::ffi::c_char> = id_cstrings.iter().map(|c| c.as_ptr()).collect();

    let cache_cstring = cache_path_to_cstring(cache_dir)?;
    let cache_ptr = cache_cstring
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut batch = empyrean_sys::EmpyreanOrbitBatch {
        orbits: std::ptr::null_mut(),
        orbit_ids: std::ptr::null_mut(),
        object_ids: std::ptr::null_mut(),
        num_orbits: 0,
    };
    let code = unsafe {
        empyrean_sys::empyrean_query_sbdb(id_ptrs.as_ptr(), id_ptrs.len(), cache_ptr, &mut batch)
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    let owned = ffi_batch_to_owned(&batch);
    unsafe { empyrean_sys::empyrean_orbits_batch_free(&mut batch) };
    owned
}

fn ffi_batch_to_owned(batch: &empyrean_sys::EmpyreanOrbitBatch) -> Result<OrbitBatch> {
    use crate::orbit::Orbit;
    let n = batch.num_orbits;
    let mut orbits = Vec::with_capacity(n);
    let mut orbit_ids = Vec::with_capacity(n);
    let mut object_ids = Vec::with_capacity(n);
    for i in 0..n {
        let ffi_orbit = unsafe { batch.orbits.add(i).read() };
        let phot_system = match ffi_orbit.phot_system {
            0 => Some(crate::orbit::PhaseFunction::HG),
            1 => Some(crate::orbit::PhaseFunction::HG1G2),
            2 => Some(crate::orbit::PhaseFunction::HG12),
            _ => None,
        };
        let orbit_id = if ffi_orbit.orbit_id.is_null() {
            None
        } else {
            Some(
                unsafe { std::ffi::CStr::from_ptr(ffi_orbit.orbit_id) }
                    .to_string_lossy()
                    .into_owned(),
            )
            .filter(|s| !s.is_empty())
        };
        let object_id = if ffi_orbit.object_id.is_null() {
            None
        } else {
            Some(
                unsafe { std::ffi::CStr::from_ptr(ffi_orbit.object_id) }
                    .to_string_lossy()
                    .into_owned(),
            )
            .filter(|s| !s.is_empty())
        };
        orbits.push(Orbit {
            orbit_id,
            object_id,
            state: crate::coordinate::CoordinateState::from_ffi(&ffi_orbit.state)?,
            a1: ffi_orbit.a1,
            a2: ffi_orbit.a2,
            a3: ffi_orbit.a3,
            ng_alpha: ffi_orbit.ng_alpha,
            ng_r0: ffi_orbit.ng_r0,
            ng_m: ffi_orbit.ng_m,
            ng_n: ffi_orbit.ng_n,
            ng_k: ffi_orbit.ng_k,
            non_grav_dt: if ffi_orbit.non_grav_dt.is_finite() {
                Some(ffi_orbit.non_grav_dt)
            } else {
                None
            },
            non_grav_dt_variance: if ffi_orbit.non_grav_dt_variance.is_finite()
                && ffi_orbit.non_grav_dt_variance > 0.0
            {
                Some(ffi_orbit.non_grav_dt_variance)
            } else {
                None
            },
            // Non-grav covariance is an OD-output concept; SBDB/query orbits
            // don't carry it.
            ng_covariance: None,
            phot_system,
            h_mag: ffi_orbit.h_mag,
            slope1: ffi_orbit.slope1,
            slope2: ffi_orbit.slope2,
            // SBDB / Horizons queries return ballistic orbits; thrust is a
            // caller-supplied input, never reconstructed from a query.
            thrust: None,
        });
        let id_ptr = unsafe { *batch.orbit_ids.add(i) };
        let id = if id_ptr.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(id_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        orbit_ids.push(id);
        let obj_ptr = if batch.object_ids.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { *batch.object_ids.add(i) }
        };
        let obj = if obj_ptr.is_null() {
            None
        } else {
            Some(
                unsafe { std::ffi::CStr::from_ptr(obj_ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        object_ids.push(obj);
    }
    OrbitBatch::new(orbits, orbit_ids, object_ids)
}

/// Query JPL Horizons for predicted ephemeris records at a single
/// observatory across a list of MJD TDB epochs.
///
/// Returns one [`EphemerisEntry`] per `(object_id × epoch)`. Optionally
/// caches the JSON response on disk under `cache_dir`.
pub fn query_horizons(
    object_ids: &[&str],
    obs_code: &str,
    times_mjd_tdb: &[f64],
    cache_dir: Option<&Path>,
) -> Result<Vec<EphemerisEntry>> {
    if object_ids.is_empty() {
        return Ok(Vec::new());
    }
    let id_cstrings: Vec<CString> = object_ids
        .iter()
        .map(|id| {
            CString::new(id.as_bytes())
                .map_err(|_| Error::invalid_input(format!("object_id contains a NUL byte: {id}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let id_ptrs: Vec<*const std::ffi::c_char> = id_cstrings.iter().map(|c| c.as_ptr()).collect();
    let obs_code_c = CString::new(obs_code.as_bytes())
        .map_err(|_| Error::invalid_input("obs_code contains a NUL byte"))?;
    let cache_cstring = cache_path_to_cstring(cache_dir)?;
    let cache_ptr = cache_cstring
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut result = empyrean_sys::EmpyreanEphemerisResult {
        entries: std::ptr::null_mut(),
        num_entries: 0,
        sensitivity: std::ptr::null_mut(),
        num_sensitivity: 0,
    };
    let code = unsafe {
        empyrean_sys::empyrean_query_horizons(
            id_ptrs.as_ptr(),
            id_ptrs.len(),
            obs_code_c.as_ptr(),
            times_mjd_tdb.as_ptr(),
            times_mjd_tdb.len(),
            cache_ptr,
            &mut result,
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    let entries: Vec<EphemerisEntry> = unsafe {
        std::slice::from_raw_parts(result.entries, result.num_entries)
            .iter()
            .map(EphemerisEntry::from_ffi)
            .collect()
    };
    unsafe { empyrean_sys::empyrean_ephemeris_result_free(&mut result) };
    Ok(entries)
}

/// Query JPL Horizons for a Cartesian state vector at a single epoch.
///
/// `command` is the Horizons COMMAND string (e.g. `"99942;"`,
/// `"DES=C/2019 Q4;"`); `epoch_mjd_tdb` is the epoch in MJD TDB.
/// Returns `(position, velocity)` in AU and AU/day — both solar-system
/// barycenter (SSB) centered, ICRF. Optionally caches the JSON
/// response on disk under `cache_dir`; pass `None` to skip caching.
pub fn query_horizons_vectors(
    command: &str,
    epoch_mjd_tdb: f64,
    cache_dir: Option<&Path>,
) -> Result<([f64; 3], [f64; 3])> {
    let command_c = CString::new(command.as_bytes())
        .map_err(|_| Error::invalid_input(format!("command contains a NUL byte: {command}")))?;
    let cache_cstring = cache_path_to_cstring(cache_dir)?;
    let cache_ptr = cache_cstring
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut pos = [0.0_f64; 3];
    let mut vel = [0.0_f64; 3];
    let code = unsafe {
        empyrean_sys::empyrean_query_horizons_vectors(
            command_c.as_ptr(),
            epoch_mjd_tdb,
            cache_ptr,
            pos.as_mut_ptr(),
            vel.as_mut_ptr(),
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok((pos, vel))
}

/// Query the MPC observations API for ADES records of one or more
/// designations.
///
/// Returns an [`Observations`] set parsed from the MPC ADES_DF JSON.
/// Optionally caches the JSON on disk under `cache_dir`.
pub fn query_observations(designations: &[&str], cache_dir: Option<&Path>) -> Result<Observations> {
    if designations.is_empty() {
        return Ok(Observations::default_empty());
    }
    let id_cstrings: Vec<CString> = designations
        .iter()
        .map(|id| {
            CString::new(id.as_bytes())
                .map_err(|_| Error::invalid_input(format!("designation contains a NUL byte: {id}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let id_ptrs: Vec<*const std::ffi::c_char> = id_cstrings.iter().map(|c| c.as_ptr()).collect();
    let cache_cstring = cache_path_to_cstring(cache_dir)?;
    let cache_ptr = cache_cstring
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut out_ptr: *mut empyrean_sys::EmpyreanObservation = std::ptr::null_mut();
    let mut out_num: usize = 0;
    let code = unsafe {
        empyrean_sys::empyrean_query_observations(
            id_ptrs.as_ptr(),
            id_ptrs.len(),
            cache_ptr,
            &mut out_ptr,
            &mut out_num,
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    // `empyrean_query_observations` is the optical MPC-fetch path and
    // returns no radar array; pass null/0 for the radar slots.
    Ok(Observations::from_raw_parts(
        out_ptr,
        out_num,
        std::ptr::null_mut(),
        0,
    ))
}

/// Query the JPL `sb_radar` API for delay/Doppler radar astrometry of one
/// or more designations.
///
/// Asteroid radar astrometry is a JPL SSD product — it is **not** served by
/// the MPC observations API ([`query_observations`] returns only optical /
/// occultation records), so radar ships as its own live-query entry point,
/// parallel to [`query_observations`]. Returns a [`RadarObservation`]
/// vector in **ADES-native** units (delay value in seconds, its σ in
/// microseconds, Doppler in Hz, frequency in MHz). An object with no radar
/// astrometry contributes no records (it is not an error, so the returned
/// vector is simply empty). Optionally caches the JSON on disk under
/// `cache_dir`; pass `None` to skip caching.
///
/// Fold the result into a fit by passing both tables through
/// [`Observations::from_arrays`] before
/// [`Context::determine`](crate::Context::determine).
pub fn query_radar(
    designations: &[&str],
    cache_dir: Option<&Path>,
) -> Result<Vec<RadarObservation>> {
    if designations.is_empty() {
        return Ok(Vec::new());
    }
    let id_cstrings: Vec<CString> = designations
        .iter()
        .map(|id| {
            CString::new(id.as_bytes())
                .map_err(|_| Error::invalid_input(format!("designation contains a NUL byte: {id}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let id_ptrs: Vec<*const std::ffi::c_char> = id_cstrings.iter().map(|c| c.as_ptr()).collect();
    let cache_cstring = cache_path_to_cstring(cache_dir)?;
    let cache_ptr = cache_cstring
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut out_ptr: *mut empyrean_sys::EmpyreanRadarObservation = std::ptr::null_mut();
    let mut out_num: usize = 0;
    let code = unsafe {
        empyrean_sys::empyrean_query_radar(
            id_ptrs.as_ptr(),
            id_ptrs.len(),
            cache_ptr,
            &mut out_ptr,
            &mut out_num,
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    // Wrap the FFI-owned radar array as an optical-less `Observations` so
    // its `Drop` releases the allocation via the C ABI; materialize the
    // typed snapshots out of it.
    let owned = Observations::from_raw_parts(std::ptr::null_mut(), 0, out_ptr, out_num);
    Ok(owned.radar())
}

fn cache_path_to_cstring(cache_dir: Option<&Path>) -> Result<Option<CString>> {
    match cache_dir {
        Some(d) => {
            let bytes = d
                .to_str()
                .ok_or_else(|| Error::invalid_input("cache_dir is not valid UTF-8"))?
                .as_bytes();
            Ok(Some(CString::new(bytes).map_err(|_| {
                Error::invalid_input("cache_dir contains a NUL byte")
            })?))
        }
        None => Ok(None),
    }
}
