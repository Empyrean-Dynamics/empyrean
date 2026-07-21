//! Orbit batch I/O — read + write across parquet, JSON, and CSV.

use std::ffi::{CStr, CString};
use std::path::Path;

use crate::error::{Error, Result};
use crate::orbit::Orbit;

use super::path_to_cstring;

/// A batch of orbits paired with their orbit / object identifiers.
///
/// Returned by every `read_orbits_*` and consumed by every
/// `write_orbits_*`. `orbit_ids` is parallel to `orbits` (same length);
/// each `object_ids[i]` may be `None` when the underlying record had no
/// physical-object designation (e.g. multiple-hypothesis sampling).
#[derive(Debug, Clone)]
pub struct OrbitBatch {
    /// Orbital states + non-gravitational parameters.
    pub orbits: Vec<Orbit>,
    /// Orbit-hypothesis identifiers. Unique per orbit by convention.
    pub orbit_ids: Vec<String>,
    /// Optional physical-object identifiers (multiple orbits may share).
    pub object_ids: Vec<Option<String>>,
}

impl OrbitBatch {
    /// Construct a batch from parallel vectors. Lengths must match.
    pub fn new(
        orbits: Vec<Orbit>,
        orbit_ids: Vec<String>,
        object_ids: Vec<Option<String>>,
    ) -> Result<Self> {
        if orbits.len() != orbit_ids.len() || orbits.len() != object_ids.len() {
            return Err(Error::invalid_input(
                "orbits, orbit_ids, and object_ids must have the same length",
            ));
        }
        Ok(Self {
            orbits,
            orbit_ids,
            object_ids,
        })
    }

    /// Number of orbits in the batch.
    pub fn len(&self) -> usize {
        self.orbits.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.orbits.is_empty()
    }
}

// ── FFI batch conversion ───────────────────────────────────────────

#[allow(clippy::type_complexity)]
fn batch_to_ffi(
    batch: &OrbitBatch,
) -> crate::error::Result<(
    Vec<empyrean_sys::EmpyreanOrbit>,
    Vec<*mut std::ffi::c_char>,
    Vec<*mut std::ffi::c_char>,
    Vec<CString>,
    Vec<Option<CString>>,
)> {
    let n = batch.len();
    let mut orbit_keep: Vec<crate::orbit::OrbitFfiKeep> = Vec::with_capacity(n);
    let ffi_orbits: Vec<empyrean_sys::EmpyreanOrbit> = batch
        .orbits
        .iter()
        .map(|o| {
            let (ffi, keep) = o.to_ffi_with_keep()?;
            orbit_keep.push(keep);
            Ok(ffi)
        })
        .collect::<crate::error::Result<Vec<_>>>()?;
    let id_cstrings: Vec<CString> = batch
        .orbit_ids
        .iter()
        .map(|s| CString::new(s.as_bytes()).unwrap_or_default())
        .collect();
    let obj_cstrings: Vec<Option<CString>> = batch
        .object_ids
        .iter()
        .map(|o| {
            o.as_ref()
                .map(|s| CString::new(s.as_bytes()).unwrap_or_default())
        })
        .collect();
    let id_ptrs: Vec<*mut std::ffi::c_char> = id_cstrings
        .iter()
        .map(|c| c.as_ptr() as *mut std::ffi::c_char)
        .collect();
    let obj_ptrs: Vec<*mut std::ffi::c_char> = obj_cstrings
        .iter()
        .map(|o| match o {
            Some(c) => c.as_ptr() as *mut std::ffi::c_char,
            None => std::ptr::null_mut(),
        })
        .collect();
    let _ = n;
    Ok((ffi_orbits, id_ptrs, obj_ptrs, id_cstrings, obj_cstrings))
}

fn ffi_batch_to_owned(batch: &empyrean_sys::EmpyreanOrbitBatch) -> Result<OrbitBatch> {
    let n = batch.num_orbits;
    let mut orbits = Vec::with_capacity(n);
    let mut orbit_ids = Vec::with_capacity(n);
    let mut object_ids = Vec::with_capacity(n);
    for i in 0..n {
        let ffi_orbit = unsafe { batch.orbits.add(i).read() };
        orbits.push(ffi_orbit_to_owned(&ffi_orbit)?);
        let id_ptr = unsafe { *batch.orbit_ids.add(i) };
        let id = if id_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(id_ptr) }
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
                unsafe { CStr::from_ptr(obj_ptr) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        object_ids.push(obj);
    }
    Ok(OrbitBatch {
        orbits,
        orbit_ids,
        object_ids,
    })
}

fn ffi_orbit_to_owned(o: &empyrean_sys::EmpyreanOrbit) -> Result<Orbit> {
    let phot_system = match o.phot_system {
        0 => Some(crate::orbit::PhaseFunction::HG),
        1 => Some(crate::orbit::PhaseFunction::HG1G2),
        2 => Some(crate::orbit::PhaseFunction::HG12),
        _ => None,
    };
    let orbit_id = if o.orbit_id.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(o.orbit_id) }
                .to_string_lossy()
                .into_owned(),
        )
        .filter(|s| !s.is_empty())
    };
    let object_id = if o.object_id.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(o.object_id) }
                .to_string_lossy()
                .into_owned(),
        )
        .filter(|s| !s.is_empty())
    };
    Ok(Orbit {
        orbit_id,
        object_id,
        state: crate::coordinate::CoordinateState::from_ffi(&o.state)?,
        a1: o.a1,
        a2: o.a2,
        a3: o.a3,
        ng_alpha: o.ng_alpha,
        ng_r0: o.ng_r0,
        ng_m: o.ng_m,
        ng_n: o.ng_n,
        ng_k: o.ng_k,
        non_grav_dt: if o.non_grav_dt.is_finite() {
            Some(o.non_grav_dt)
        } else {
            None
        },
        non_grav_dt_variance: if o.non_grav_dt_variance.is_finite() && o.non_grav_dt_variance > 0.0
        {
            Some(o.non_grav_dt_variance)
        } else {
            None
        },
        // Non-grav covariance is an OD-output concept; orbit reads don't carry it.
        ng_covariance: None,
        phot_system,
        h_mag: o.h_mag,
        slope1: o.slope1,
        slope2: o.slope2,
        // Thrust is a caller-owned input side array; the OrbitRow schema
        // doesn't carry it, so orbit reads never reconstruct it.
        thrust: None,
        // SRP slot carried through from the C EmpyreanOrbit when present.
        srp: crate::orbit::SrpParams::from_ffi(
            o.srp_amrat,
            o.srp_cr,
            o.has_srp,
            o.srp_amrat_variance,
        ),
    })
}

// ── Read / write helpers — batch lifetime managed via FFI free ────

fn read_orbits_via<F>(path: &Path, c_call: F) -> Result<OrbitBatch>
where
    F: FnOnce(*const std::ffi::c_char, *mut empyrean_sys::EmpyreanOrbitBatch) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let mut batch = empyrean_sys::EmpyreanOrbitBatch {
        orbits: std::ptr::null_mut(),
        orbit_ids: std::ptr::null_mut(),
        object_ids: std::ptr::null_mut(),
        num_orbits: 0,
    };
    let code = c_call(path_c.as_ptr(), &mut batch);
    if code != 0 {
        return Err(Error::capture(code));
    }
    let owned = ffi_batch_to_owned(&batch);
    unsafe { empyrean_sys::empyrean_orbits_batch_free(&mut batch) };
    owned
}

fn write_orbits_via<F>(path: &Path, batch: &OrbitBatch, c_call: F) -> Result<()>
where
    F: FnOnce(*const std::ffi::c_char, *const empyrean_sys::EmpyreanOrbitBatch) -> i32,
{
    let path_c = path_to_cstring(path)?;
    let (ffi_orbits, id_ptrs, obj_ptrs, _ids_keep, _objs_keep) = batch_to_ffi(batch)?;
    let ffi_batch = empyrean_sys::EmpyreanOrbitBatch {
        orbits: ffi_orbits.as_ptr() as *mut empyrean_sys::EmpyreanOrbit,
        orbit_ids: id_ptrs.as_ptr() as *mut *mut std::ffi::c_char,
        object_ids: obj_ptrs.as_ptr() as *mut *mut std::ffi::c_char,
        num_orbits: batch.len(),
    };
    let code = c_call(path_c.as_ptr(), &ffi_batch);
    if code != 0 {
        return Err(Error::capture(code));
    }
    Ok(())
}

/// Read an orbits parquet file.
pub fn read_orbits_parquet(path: impl AsRef<Path>) -> Result<OrbitBatch> {
    read_orbits_via(path.as_ref(), |p, out| unsafe {
        empyrean_sys::empyrean_orbits_read_parquet(p, out)
    })
}

/// Write an orbit batch to a parquet file.
pub fn write_orbits_parquet(path: impl AsRef<Path>, batch: &OrbitBatch) -> Result<()> {
    write_orbits_via(path.as_ref(), batch, |p, b| unsafe {
        empyrean_sys::empyrean_orbits_write_parquet(p, b)
    })
}

/// Read an orbits JSON file.
pub fn read_orbits_json(path: impl AsRef<Path>) -> Result<OrbitBatch> {
    read_orbits_via(path.as_ref(), |p, out| unsafe {
        empyrean_sys::empyrean_orbits_read_json(p, out)
    })
}

/// Write an orbit batch to JSON.
pub fn write_orbits_json(path: impl AsRef<Path>, batch: &OrbitBatch) -> Result<()> {
    write_orbits_via(path.as_ref(), batch, |p, b| unsafe {
        empyrean_sys::empyrean_orbits_write_json(p, b)
    })
}

/// Read an orbits CSV file.
pub fn read_orbits_csv(path: impl AsRef<Path>) -> Result<OrbitBatch> {
    read_orbits_via(path.as_ref(), |p, out| unsafe {
        empyrean_sys::empyrean_orbits_read_csv(p, out)
    })
}

/// Write an orbit batch to CSV.
pub fn write_orbits_csv(path: impl AsRef<Path>, batch: &OrbitBatch) -> Result<()> {
    write_orbits_via(path.as_ref(), batch, |p, b| unsafe {
        empyrean_sys::empyrean_orbits_write_csv(p, b)
    })
}
