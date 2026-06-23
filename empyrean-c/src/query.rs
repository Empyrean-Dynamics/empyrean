//! C ABI exports for external small-body queries.
//!
//! - [`empyrean_query_sbdb`] — JPL Small-Body Database orbit lookup,
//!   returning a flat [`EmpyreanOrbitBatch`].
//! - [`empyrean_query_horizons`] — JPL Horizons ephemeris lookup,
//!   returning an [`EmpyreanEphemerisResult`].
//! - [`empyrean_query_horizons_vectors`] — JPL Horizons Cartesian state
//!   vector lookup (SSB-centered, ICRF), written to caller-provided
//!   out-pointers.
//! - [`empyrean_query_observations`] — MPC observation records lookup,
//!   parsed into an [`EmpyreanObservation`] array.
//! - [`empyrean_query_radar`] — JPL `sb_radar` delay/Doppler radar
//!   astrometry lookup, parsed into an [`EmpyreanRadarObservation`] array.
//!
//! Caller-allocated results are released with
//! [`empyrean_orbits_batch_free`](crate::io::empyrean_orbits_batch_free)
//! / [`empyrean_ephemeris_result_free`](crate::ephemeris::empyrean_ephemeris_result_free)
//! / [`empyrean_observations_free`](crate::od::empyrean_observations_free)
//! / [`empyrean_radar_observations_free`](crate::od::empyrean_radar_observations_free).

use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use crate::ephemeris::{EmpyreanEphemerisEntry, EmpyreanEphemerisResult};
use crate::io::{EmpyreanOrbitBatch, orbits_to_batch};
use crate::od::{EmpyreanObservation, EmpyreanRadarObservation, scott_radar_to_c};
use crate::set_last_error;
use empyrean_core::determination::RadarObservation;

/// Query the JPL Small-Body Database for one or more orbits.
///
/// `object_ids` is an array of `num_object_ids` null-terminated UTF-8
/// designations / names / SPK IDs (e.g. `"apophis"`, `"99942"`,
/// `"2024 YR4"`, `"67P"`). `cache_dir` may be null to skip caching, or
/// a directory path where SBDB JSON responses are cached on disk.
///
/// On success the populated [`EmpyreanOrbitBatch`] must be released with
/// [`empyrean_orbits_batch_free`](crate::io::empyrean_orbits_batch_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_query_sbdb(
    object_ids: *const *const c_char,
    num_object_ids: usize,
    cache_dir: *const c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if object_ids.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_object_ids == 0 {
            set_last_error("at least one object_id is required");
            return -1;
        }
        let mut ids: Vec<String> = Vec::with_capacity(num_object_ids);
        for i in 0..num_object_ids {
            let p = unsafe { *object_ids.add(i) };
            if p.is_null() {
                set_last_error(&format!("null object_id at index {i}"));
                return -1;
            }
            match unsafe { CStr::from_ptr(p) }.to_str() {
                Ok(s) => ids.push(s.to_string()),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in object_id[{i}]: {e}"));
                    return -1;
                }
            }
        }
        let cache_path = if cache_dir.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
                Ok(s) => Some(std::path::PathBuf::from(s)),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in cache_dir: {e}"));
                    return -1;
                }
            }
        };

        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let mut cache_owner = cache_path.map(empyrean_core::data::DiskCache::new);
        let cache_ref = cache_owner.as_mut();

        let orbits = match empyrean_core::query::query_sbdb(&id_refs, cache_ref) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&format!("SBDB query failed: {e}"));
                return -2;
            }
        };

        match orbits_to_batch(&orbits) {
            Ok(batch) => {
                unsafe { *out = batch };
                0
            }
            Err(e) => {
                set_last_error(&e);
                -3
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_query_sbdb");
            -99
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// query_horizons
// ────────────────────────────────────────────────────────────────────

/// Query JPL Horizons for predicted ephemeris records.
///
/// `object_ids` is an array of `num_object_ids` null-terminated UTF-8
/// designations / names / SPK IDs. `obs_code` is the MPC observatory
/// code as a null-terminated string. `times_mjd_tdb` carries
/// `num_times` epochs in MJD TDB.
///
/// On success populates an [`EmpyreanEphemerisResult`] with one entry
/// per `(object_id × epoch)`. Free with
/// [`empyrean_ephemeris_result_free`](crate::ephemeris::empyrean_ephemeris_result_free).
///
/// All angular values are converted to **degrees** at the FFI boundary
/// (Horizons natively returns radians).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_query_horizons(
    object_ids: *const *const c_char,
    num_object_ids: usize,
    obs_code: *const c_char,
    times_mjd_tdb: *const f64,
    num_times: usize,
    cache_dir: *const c_char,
    out: *mut EmpyreanEphemerisResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if object_ids.is_null() || obs_code.is_null() || times_mjd_tdb.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_object_ids == 0 {
            set_last_error("at least one object_id is required");
            return -1;
        }
        let mut ids: Vec<String> = Vec::with_capacity(num_object_ids);
        for i in 0..num_object_ids {
            let p = unsafe { *object_ids.add(i) };
            if p.is_null() {
                set_last_error(&format!("null object_id at index {i}"));
                return -1;
            }
            match unsafe { CStr::from_ptr(p) }.to_str() {
                Ok(s) => ids.push(s.to_string()),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in object_id[{i}]: {e}"));
                    return -1;
                }
            }
        }
        let obs_code_str = match unsafe { CStr::from_ptr(obs_code) }.to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in obs_code: {e}"));
                return -1;
            }
        };
        let cache_path = if cache_dir.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
                Ok(s) => Some(std::path::PathBuf::from(s)),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in cache_dir: {e}"));
                    return -1;
                }
            }
        };
        let times = unsafe { std::slice::from_raw_parts(times_mjd_tdb, num_times) }.to_vec();

        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let mut cache_owner = cache_path.map(empyrean_core::data::DiskCache::new);
        let cache_ref = cache_owner.as_mut();

        let records = match empyrean_core::query::query_horizons(
            &id_refs,
            &obs_code_str,
            &times,
            cache_ref,
        ) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("Horizons query failed: {e}"));
                return -2;
            }
        };

        let n = records.len();
        if n == 0 {
            unsafe {
                (*out).entries = std::ptr::null_mut();
                (*out).num_entries = 0;
            }
            return 0;
        }
        let layout = std::alloc::Layout::array::<EmpyreanEphemerisEntry>(n).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanEphemerisEntry;
        if ptr.is_null() {
            set_last_error("allocation failed for ephemeris array");
            return -5;
        }

        let r2d = 180.0 / std::f64::consts::PI;
        for (i, rec) in records.iter().enumerate() {
            let orbit_id = CString::new(rec.orbit_id.as_str())
                .unwrap_or_default()
                .into_raw();
            let mut obs_code_bytes = [0u8; 4];
            let bytes = rec.obs_code.as_bytes();
            let m = bytes.len().min(3);
            obs_code_bytes[..m].copy_from_slice(&bytes[..m]);
            let entry = EmpyreanEphemerisEntry {
                orbit_id,
                epoch_mjd_tdb: rec.epoch.mjd_tdb(),
                ra_deg: rec.ra * r2d,
                dec_deg: rec.dec * r2d,
                rho_au: rec.rho,
                vrho_au_day: rec.vrho,
                vra_deg_day: rec.vra * r2d,
                vdec_deg_day: rec.vdec * r2d,
                light_time_days: rec.light_time.unwrap_or(f64::NAN),
                phase_angle_deg: rec.phase_angle.map(|v| v * r2d).unwrap_or(f64::NAN),
                elongation_deg: rec.elongation.map(|v| v * r2d).unwrap_or(f64::NAN),
                heliocentric_distance_au: rec.heliocentric_distance.unwrap_or(f64::NAN),
                mag: rec.apparent_magnitude.unwrap_or(f64::NAN),
                mag_sigma: f64::NAN,
                // Local-horizon / sky-motion angles are not part of the
                // Horizons observer-table record this path ingests, so
                // they are honestly unavailable (NaN) here. The native
                // generate_ephemeris path (ephemeris.rs) populates them.
                zenith_angle_deg: f64::NAN,
                azimuth_deg: f64::NAN,
                hour_angle_deg: f64::NAN,
                lunar_elongation_deg: f64::NAN,
                position_angle_deg: f64::NAN,
                sky_rate_deg_day: f64::NAN,
                obs_code: obs_code_bytes,
            };
            unsafe { ptr.add(i).write(entry) };
        }
        unsafe {
            (*out).entries = ptr;
            (*out).num_entries = n;
        }
        0
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_query_horizons");
            -99
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// query_horizons_vectors
// ────────────────────────────────────────────────────────────────────

/// Query JPL Horizons for a Cartesian state vector at a single epoch.
///
/// `command` is the Horizons COMMAND string as a null-terminated UTF-8
/// string (e.g. `"99942;"`, `"DES=C/2019 Q4;"`). `epoch_mjd_tdb` is
/// the epoch in MJD TDB. `cache_dir` may be null to skip caching, or
/// a directory path where Horizons JSON responses are cached on disk.
///
/// On success writes the position (AU) to `out_pos` (length 3) and the
/// velocity (AU/day) to `out_vel` (length 3) — both solar-system
/// barycenter (SSB) centered, ICRF.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_query_horizons_vectors(
    command: *const c_char,
    epoch_mjd_tdb: f64,
    cache_dir: *const c_char,
    out_pos: *mut f64,
    out_vel: *mut f64,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if command.is_null() || out_pos.is_null() || out_vel.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let command_str = match unsafe { CStr::from_ptr(command) }.to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in command: {e}"));
                return -1;
            }
        };
        let cache_path = if cache_dir.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
                Ok(s) => Some(std::path::PathBuf::from(s)),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in cache_dir: {e}"));
                    return -1;
                }
            }
        };

        let mut cache_owner = cache_path.map(empyrean_core::data::DiskCache::new);
        let cache_ref = cache_owner.as_mut();

        let (pos, vel) = match empyrean_core::query::query_horizons_vectors(
            &command_str,
            epoch_mjd_tdb,
            cache_ref,
        ) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("Horizons vectors query failed: {e}"));
                return -2;
            }
        };
        unsafe {
            std::ptr::copy_nonoverlapping(pos.as_ptr(), out_pos, 3);
            std::ptr::copy_nonoverlapping(vel.as_ptr(), out_vel, 3);
        }
        0
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_query_horizons_vectors");
            -99
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// query_observations
// ────────────────────────────────────────────────────────────────────

/// Query the MPC observations API for ADES records of one or more
/// designations.
///
/// `designations` is an array of `num_designations` null-terminated
/// UTF-8 designations (e.g. `"99942"`, `"2024 YR4"`, `"67P"`). The
/// MPC API returns ADES_DF JSON; this function parses each row into
/// the C-ABI [`EmpyreanObservation`] struct, filling the full ADES
/// schema (perm_id / prov_id / trk_sub / mode / sys / ctr / pos1-3 /
/// rms_corr / mag / rms_mag / band / ast_cat / phot_cat / phot_ap /
/// log_snr / seeing / exp / rms_fit / n_stars / notes / remarks) when
/// present in the JSON.
///
/// On success `*out_ptr` carries a heap-allocated array of length
/// `*out_num`. Free with
/// [`empyrean_observations_free`](crate::od::empyrean_observations_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_query_observations(
    designations: *const *const c_char,
    num_designations: usize,
    cache_dir: *const c_char,
    out_ptr: *mut *mut EmpyreanObservation,
    out_num: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if designations.is_null() || out_ptr.is_null() || out_num.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_designations == 0 {
            set_last_error("at least one designation is required");
            return -1;
        }
        unsafe {
            *out_ptr = std::ptr::null_mut();
            *out_num = 0;
        }
        let mut ids: Vec<String> = Vec::with_capacity(num_designations);
        for i in 0..num_designations {
            let p = unsafe { *designations.add(i) };
            if p.is_null() {
                set_last_error(&format!("null designation at index {i}"));
                return -1;
            }
            match unsafe { CStr::from_ptr(p) }.to_str() {
                Ok(s) => ids.push(s.to_string()),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in designation[{i}]: {e}"));
                    return -1;
                }
            }
        }
        let cache_path = if cache_dir.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
                Ok(s) => Some(std::path::PathBuf::from(s)),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in cache_dir: {e}"));
                    return -1;
                }
            }
        };

        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let mut cache_owner = cache_path.map(empyrean_core::data::DiskCache::new);
        let cache_ref = cache_owner.as_mut();

        let records = match empyrean_core::query::query_observations(&id_refs, cache_ref) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("MPC query failed: {e}"));
                return -2;
            }
        };

        let n = records.len();
        if n == 0 {
            return 0;
        }
        let layout = std::alloc::Layout::array::<EmpyreanObservation>(n).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservation;
        if ptr.is_null() {
            set_last_error("allocation failed for observations array");
            return -5;
        }

        for (i, rec) in records.iter().enumerate() {
            let entry = mpc_record_to_empyrean_observation(rec);
            unsafe { ptr.add(i).write(entry) };
        }
        unsafe {
            *out_ptr = ptr;
            *out_num = n;
        }
        0
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_query_observations");
            -99
        }
    }
}

/// Parse an MPC ADES_DF JSON record into a [`EmpyreanObservation`].
fn mpc_record_to_empyrean_observation(rec: &serde_json::Value) -> EmpyreanObservation {
    fn opt_str(v: &serde_json::Value, key: &str) -> *mut c_char {
        match v.get(key).and_then(|x| x.as_str()) {
            Some(s) if !s.is_empty() => CString::new(s).unwrap_or_default().into_raw(),
            _ => std::ptr::null_mut(),
        }
    }
    fn opt_f64(v: &serde_json::Value, key: &str) -> f64 {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(f64::NAN)
    }
    fn opt_i32(v: &serde_json::Value, key: &str) -> i32 {
        v.get(key)
            .and_then(|x| x.as_i64())
            .map(|n| n as i32)
            .unwrap_or(-1)
    }

    let mut obs_code = [0u8; 4];
    if let Some(stn) = rec.get("stn").and_then(|x| x.as_str()) {
        let bytes = stn.as_bytes();
        let m = bytes.len().min(3);
        obs_code[..m].copy_from_slice(&bytes[..m]);
    }

    let obs_time_str = rec
        .get("obstime")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let obs_time = CString::new(obs_time_str).unwrap_or_default().into_raw();

    EmpyreanObservation {
        // Identification
        perm_id: opt_str(rec, "permid"),
        prov_id: opt_str(rec, "provid"),
        trk_sub: opt_str(rec, "trksub"),
        obs_id: opt_str(rec, "obsid"),
        obs_sub_id: opt_str(rec, "obssubid"),
        trk_id: opt_str(rec, "trkid"),

        // Observer
        obs_code,
        mode: opt_str(rec, "mode"),
        prog: opt_str(rec, "prog"),

        // Observer location (roving / spacecraft)
        sys: opt_str(rec, "sys"),
        ctr: opt_f64(rec, "ctr"),
        pos1: opt_f64(rec, "pos1"),
        pos2: opt_f64(rec, "pos2"),
        pos3: opt_f64(rec, "pos3"),

        // Astrometry
        obs_time,
        ra_deg: opt_f64(rec, "ra"),
        dec_deg: opt_f64(rec, "dec"),

        // Uncertainties
        rms_ra_arcsec: opt_f64(rec, "rmsra"),
        rms_dec_arcsec: opt_f64(rec, "rmsdec"),
        rms_corr: opt_f64(rec, "rmscorr"),

        // Astrometric catalog
        ast_cat: opt_str(rec, "astcat"),

        // Photometry
        mag: opt_f64(rec, "mag"),
        rms_mag: opt_f64(rec, "rmsmag"),
        band: opt_str(rec, "band"),
        phot_cat: opt_str(rec, "photcat"),
        phot_ap: opt_f64(rec, "photap"),

        // Supplementary diagnostics
        log_snr: opt_f64(rec, "logsnr"),
        seeing: opt_f64(rec, "seeing"),
        exp: opt_f64(rec, "exp"),
        rms_fit: opt_f64(rec, "rmsfit"),
        n_stars: opt_i32(rec, "nstars"),
        notes: opt_str(rec, "notes"),
        remarks: opt_str(rec, "remarks"),
    }
}

// ────────────────────────────────────────────────────────────────────
// query_radar
// ────────────────────────────────────────────────────────────────────

/// Query the JPL `sb_radar` API for delay/Doppler radar astrometry of one
/// or more designations.
///
/// `designations` is an array of `num_designations` null-terminated UTF-8
/// designations (e.g. `"99942"`, `"2024 YR4"`). Asteroid radar astrometry
/// is a JPL SSD product — it is **not** served by the MPC observations API
/// (`empyrean_query_observations` returns only optical / occultation
/// records), so radar ships as its own live-query entry point. JPL
/// `sb_radar` JSON records are converted to ADES-native scott
/// `RadarObservation`s and packed into the C-ABI
/// [`EmpyreanRadarObservation`] struct (the same layout
/// [`empyrean_read_ades`](crate::od::empyrean_read_ades) emits): the delay
/// value is in seconds, its σ in microseconds, Doppler in Hz, frequency in
/// MHz, and the `com` flag is a tri-state i8. `cache_dir` may be null to
/// skip caching, or a directory path where `sb_radar` JSON responses are
/// cached on disk.
///
/// An object with no radar astrometry contributes no records (it is not an
/// error). A JPL record that fails to parse (missing required field, or an
/// unrecognised DSN station code) is rejected loudly rather than silently
/// dropped — the whole call fails so no radar quietly goes missing.
///
/// On success `*out_ptr` carries a heap-allocated array of length
/// `*out_num`. Free with
/// [`empyrean_radar_observations_free`](crate::od::empyrean_radar_observations_free).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_query_radar(
    designations: *const *const c_char,
    num_designations: usize,
    cache_dir: *const c_char,
    out_ptr: *mut *mut EmpyreanRadarObservation,
    out_num: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if designations.is_null() || out_ptr.is_null() || out_num.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if num_designations == 0 {
            set_last_error("at least one designation is required");
            return -1;
        }
        unsafe {
            *out_ptr = std::ptr::null_mut();
            *out_num = 0;
        }
        let mut ids: Vec<String> = Vec::with_capacity(num_designations);
        for i in 0..num_designations {
            let p = unsafe { *designations.add(i) };
            if p.is_null() {
                set_last_error(&format!("null designation at index {i}"));
                return -1;
            }
            match unsafe { CStr::from_ptr(p) }.to_str() {
                Ok(s) => ids.push(s.to_string()),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in designation[{i}]: {e}"));
                    return -1;
                }
            }
        }
        let cache_path = if cache_dir.is_null() {
            None
        } else {
            match unsafe { CStr::from_ptr(cache_dir) }.to_str() {
                Ok(s) => Some(std::path::PathBuf::from(s)),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in cache_dir: {e}"));
                    return -1;
                }
            }
        };

        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let mut cache_owner = cache_path.map(empyrean_core::data::DiskCache::new);
        let cache_ref = cache_owner.as_mut();

        let records = match empyrean_core::query::query_radar(&id_refs, cache_ref) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(&format!("JPL sb_radar query failed: {e}"));
                return -2;
            }
        };

        // Convert each JPL `sb_radar` JSON record to a scott
        // `RadarObservation`. A record that fails to parse is a loud
        // error — no observation is silently dropped.
        let mut radar: Vec<RadarObservation> = Vec::with_capacity(records.len());
        for (i, rec) in records.iter().enumerate() {
            match RadarObservation::from_jpl_radar(rec) {
                Some(r) => radar.push(r),
                None => {
                    set_last_error(&format!(
                        "failed to parse JPL sb_radar record {i} (missing required field or unrecognised station code)"
                    ));
                    return -2;
                }
            }
        }

        let n = radar.len();
        if n == 0 {
            return 0;
        }
        let layout = std::alloc::Layout::array::<EmpyreanRadarObservation>(n).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanRadarObservation;
        if ptr.is_null() {
            set_last_error("allocation failed for radar observations array");
            return -5;
        }

        for (i, r) in radar.iter().enumerate() {
            unsafe { ptr.add(i).write(scott_radar_to_c(r)) };
        }
        unsafe {
            *out_ptr = ptr;
            *out_num = n;
        }
        0
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_query_radar");
            -99
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Marshaling-level error paths only — no network. The C entry
    // point must reject bad pointers / encodings before any HTTP
    // request is attempted.

    #[test]
    fn horizons_vectors_null_command_is_rejected() {
        let mut pos = [0.0_f64; 3];
        let mut vel = [0.0_f64; 3];
        let code = unsafe {
            empyrean_query_horizons_vectors(
                std::ptr::null(),
                60000.0,
                std::ptr::null(),
                pos.as_mut_ptr(),
                vel.as_mut_ptr(),
            )
        };
        assert_eq!(code, -1);
    }

    #[test]
    fn horizons_vectors_null_out_pointers_are_rejected() {
        let command = CString::new("99942;").unwrap();
        let mut vel = [0.0_f64; 3];
        let code = unsafe {
            empyrean_query_horizons_vectors(
                command.as_ptr(),
                60000.0,
                std::ptr::null(),
                std::ptr::null_mut(),
                vel.as_mut_ptr(),
            )
        };
        assert_eq!(code, -1);

        let mut pos = [0.0_f64; 3];
        let code = unsafe {
            empyrean_query_horizons_vectors(
                command.as_ptr(),
                60000.0,
                std::ptr::null(),
                pos.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, -1);
    }

    #[test]
    fn horizons_vectors_invalid_utf8_command_is_rejected() {
        // 0xFF is never valid UTF-8.
        let bad = [0xFFu8, 0x00];
        let mut pos = [0.0_f64; 3];
        let mut vel = [0.0_f64; 3];
        let code = unsafe {
            empyrean_query_horizons_vectors(
                bad.as_ptr() as *const c_char,
                60000.0,
                std::ptr::null(),
                pos.as_mut_ptr(),
                vel.as_mut_ptr(),
            )
        };
        assert_eq!(code, -1);
    }

    #[test]
    fn horizons_vectors_invalid_utf8_cache_dir_is_rejected() {
        let command = CString::new("99942;").unwrap();
        let bad = [0xFFu8, 0x00];
        let mut pos = [0.0_f64; 3];
        let mut vel = [0.0_f64; 3];
        let code = unsafe {
            empyrean_query_horizons_vectors(
                command.as_ptr(),
                60000.0,
                bad.as_ptr() as *const c_char,
                pos.as_mut_ptr(),
                vel.as_mut_ptr(),
            )
        };
        assert_eq!(code, -1);
    }
}
