//! Event I/O — write across parquet, JSON, and CSV.
//!
//! The propagator (via [`Context::propagate`](crate::Context::propagate))
//! is the canonical producer of events; only a write surface is exposed
//! here.

use std::ffi::CString;
use std::path::Path;

use crate::error::{Error, Result};
use crate::propagate::Event;

use super::path_to_cstring;

fn events_to_ffi_array(
    events: &[Event],
) -> Result<(Vec<empyrean_sys::EmpyreanEvent>, Vec<CString>)> {
    let mut keep: Vec<CString> = Vec::with_capacity(events.len() * 3);
    let ffi: Vec<empyrean_sys::EmpyreanEvent> = events
        .iter()
        .map(|e| {
            let body_str = e.body.as_ref().map(|o| o.to_string()).unwrap_or_default();
            let body_naif_id = e.body.map(|o| o.naif_id()).unwrap_or(-1);
            let etype_c = CString::new(e.event_type.as_bytes()).unwrap_or_default();
            let oid_c = CString::new(e.orbit_id.as_bytes()).unwrap_or_default();
            let obj_c = CString::new(e.object_id.as_bytes()).unwrap_or_default();
            let body_c = CString::new(body_str).unwrap_or_default();
            let etype_ptr = etype_c.as_ptr() as *mut std::ffi::c_char;
            let oid_ptr = oid_c.as_ptr() as *mut std::ffi::c_char;
            let obj_ptr = obj_c.as_ptr() as *mut std::ffi::c_char;
            let body_ptr = body_c.as_ptr() as *mut std::ffi::c_char;
            keep.push(etype_c);
            keep.push(oid_c);
            keep.push(obj_c);
            keep.push(body_c);
            Ok(empyrean_sys::EmpyreanEvent {
                event_type: etype_ptr,
                orbit_id: oid_ptr,
                object_id: obj_ptr,
                body: body_ptr,
                body_naif_id,
                epoch_mjd_tdb: e.epoch.mjd_tdb()?,
                distance_au: e.distance_au,
                distance_km: e.distance_km,
                relative_velocity_au_day: e.relative_velocity_au_day,
                two_body_energy: e.two_body_energy,
                jacobi_constant: e.jacobi_constant,
                jacobi_constant_sigma: e.jacobi_constant_sigma,
                jacobi_constant_l1: e.jacobi_constant_l1,
                jacobi_constant_l2: e.jacobi_constant_l2,
                n_periapses: e.n_periapses.map(|n| n as i32).unwrap_or(-1),
                impact_latitude_deg: e.impact_latitude_deg,
                impact_longitude_deg: e.impact_longitude_deg,
                impact_altitude_km: e.impact_altitude_km,
                shadow_fraction: e.shadow_fraction,
                illumination: e.illumination,
                relative_x: e.relative_x,
                relative_y: e.relative_y,
                relative_z: e.relative_z,
                relative_vx: e.relative_vx,
                relative_vy: e.relative_vy,
                relative_vz: e.relative_vz,
                effective_radius_au: e.effective_radius_au,
                effective_radius_km: e.effective_radius_km,
                sigma_distance_au: e.sigma_distance_au,
                ip_linear: e.ip_linear,
                ip_second_order: e.ip_second_order,
                nonlinearity: e.nonlinearity,
                ip_agm: e.ip_agm,
                ip_mc: e.ip_mc,
                // `0xFF` = "not a regime event"; faithfully round-trip the
                // CovarianceRegimeChange payload the wrapper Event carries.
                previous_kind: e.previous_kind.map(|k| k.to_u8()).unwrap_or(0xFF),
                resolved_kind: e.regime_resolved_kind.map(|k| k.to_u8()).unwrap_or(0xFF),
                kappa: e.kappa,
                threshold_below: e.threshold_below,
                threshold_above: e.threshold_above,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((ffi, keep))
}

fn write_events_via<F>(path: &Path, events: &[Event], c_call: F) -> Result<()>
where
    F: FnOnce(*const std::ffi::c_char, *const empyrean_sys::EmpyreanEvent, usize) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let (ffi, _keep) = events_to_ffi_array(events)?;
    let code = c_call(path_c.as_ptr(), ffi.as_ptr(), ffi.len());
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok(())
}

/// Write events to a parquet file.
pub fn write_events_parquet(path: impl AsRef<Path>, events: &[Event]) -> Result<()> {
    write_events_via(path.as_ref(), events, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_events_write_parquet(p, ptr, n)
    })
}

/// Write events to JSON.
pub fn write_events_json(path: impl AsRef<Path>, events: &[Event]) -> Result<()> {
    write_events_via(path.as_ref(), events, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_events_write_json(p, ptr, n)
    })
}

/// Write events to CSV.
pub fn write_events_csv(path: impl AsRef<Path>, events: &[Event]) -> Result<()> {
    write_events_via(path.as_ref(), events, |p, ptr, n| unsafe {
        empyrean_sys::empyrean_events_write_csv(p, ptr, n)
    })
}
