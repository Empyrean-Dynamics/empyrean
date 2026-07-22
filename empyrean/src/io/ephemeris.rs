//! Ephemeris I/O — write across parquet, JSON, and CSV.
//!
//! Read paths are not exposed: the propagator's
//! [`Context::generate_ephemeris`](crate::Context::generate_ephemeris)
//! is the canonical producer.

use std::ffi::CString;
use std::path::Path;

use crate::ephemeris::EphemerisEntry;
use crate::error::{Error, Result};

use super::path_to_cstring;

fn ephemeris_to_ffi_array(
    entries: &[EphemerisEntry],
) -> Result<(Vec<empyrean_sys::EmpyreanEphemerisEntry>, Vec<CString>)> {
    let mut keep = Vec::with_capacity(entries.len());
    let ffi: Vec<empyrean_sys::EmpyreanEphemerisEntry> = entries
        .iter()
        .map(|e| {
            let id_c = CString::new(e.orbit_id.as_bytes()).unwrap_or_default();
            let id_ptr = id_c.as_ptr() as *mut std::ffi::c_char;
            keep.push(id_c);
            let mut obs_code = [0u8; 4];
            let bytes = e.obs_code.as_bytes();
            let n = bytes.len().min(3);
            obs_code[..n].copy_from_slice(&bytes[..n]);
            Ok(empyrean_sys::EmpyreanEphemerisEntry {
                orbit_id: id_ptr,
                epoch_mjd_tdb: e.epoch.mjd_tdb()?,
                ra_deg: e.ra_deg,
                dec_deg: e.dec_deg,
                rho_au: e.rho_au,
                vrho_au_day: e.vrho_au_day,
                vra_deg_day: e.vra_deg_day,
                vdec_deg_day: e.vdec_deg_day,
                light_time_days: e.light_time_days,
                phase_angle_deg: e.phase_angle_deg,
                elongation_deg: e.elongation_deg,
                heliocentric_distance_au: e.heliocentric_distance_au,
                mag: e.mag,
                mag_sigma: e.mag_sigma,
                zenith_angle_deg: e.zenith_angle_deg,
                azimuth_deg: e.azimuth_deg,
                hour_angle_deg: e.hour_angle_deg,
                lunar_elongation_deg: e.lunar_elongation_deg,
                position_angle_deg: e.position_angle_deg,
                sky_rate_deg_day: e.sky_rate_deg_day,
                obs_code,
                has_covariance: u8::from(e.covariance.is_some()),
                covariance: e.covariance.unwrap_or([[f64::NAN; 6]; 6]),
                aberrated_state: e.aberrated_state,
                has_aberrated_covariance: u8::from(e.aberrated_covariance.is_some()),
                aberrated_covariance: e.aberrated_covariance.unwrap_or([[f64::NAN; 6]; 6]),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((ffi, keep))
}

fn write_ephemeris_via<F>(path: &Path, entries: &[EphemerisEntry], c_call: F) -> Result<()>
where
    F: FnOnce(*const std::ffi::c_char, *const empyrean_sys::EmpyreanEphemerisEntry, usize) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let (ffi, _keep) = ephemeris_to_ffi_array(entries)?;
    let code = c_call(path_c.as_ptr(), ffi.as_ptr(), ffi.len());
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok(())
}

/// Write ephemeris entries to a parquet file.
pub fn write_ephemeris_parquet(path: impl AsRef<Path>, entries: &[EphemerisEntry]) -> Result<()> {
    write_ephemeris_via(path.as_ref(), entries, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_ephemeris_write_parquet(p, ptr, n)
    })
}

/// Write ephemeris entries to JSON.
pub fn write_ephemeris_json(path: impl AsRef<Path>, entries: &[EphemerisEntry]) -> Result<()> {
    write_ephemeris_via(path.as_ref(), entries, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_ephemeris_write_json(p, ptr, n)
    })
}

/// Write ephemeris entries to CSV.
pub fn write_ephemeris_csv(path: impl AsRef<Path>, entries: &[EphemerisEntry]) -> Result<()> {
    write_ephemeris_via(path.as_ref(), entries, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_ephemeris_write_csv(p, ptr, n)
    })
}
