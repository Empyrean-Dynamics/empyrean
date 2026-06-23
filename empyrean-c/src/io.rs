//! C ABI exports for file I/O.
//!
//! Three formats × four data types:
//!
//! - **Orbits** (read + write): parquet, JSON, CSV.
//! - **Ephemeris** (write): parquet, JSON, CSV.
//! - **Events** (write): parquet, JSON, CSV.
//! - **Residuals** (write): parquet, JSON, CSV.
//!
//! All readers populate a flat C-ABI struct or array; writers consume
//! the same types. Caller-allocated results (e.g. `EmpyreanOrbitBatch`
//! produced by a reader) must be released with the matching
//! `*_free()` helper. JSON / CSV use the schemas documented inline; the
//! parquet schemas are villeneuve-native (round-trips with the rest of
//! the empyrean ecosystem).

use std::ffi::{CStr, CString, c_char};
use std::fs::File;
use std::panic::AssertUnwindSafe;
use std::path::Path;

use empyrean_core::convert::{
    coordinate_state_to_coordinates, coordinates_to_coordinate_state, frame_to_int, int_to_frame,
    int_to_representation, representation_to_int,
};
use empyrean_core::coordinates::AU;
use empyrean_core::nongrav::{GFunction, NonGravModel, NonGravParams};
use empyrean_core::orbits::Orbits;
use empyrean_core::propagation::events::DynamicalEvent;
use serde::{Deserialize, Serialize};

use crate::ephemeris::EmpyreanEphemerisEntry;
use crate::od::EmpyreanObservationResult;
use crate::propagate::{EmpyreanEvent, EmpyreanOrbit};
use crate::{CoordinateState, set_last_error};

// ────────────────────────────────────────────────────────────────────
// Orbit batch type
// ────────────────────────────────────────────────────────────────────

/// A batch of orbits with their identifiers.
///
/// Returned by every `empyrean_orbits_read_*` and consumed by every
/// `empyrean_orbits_write_*`. `orbit_ids` and `object_ids` are parallel
/// to `orbits` (same length); each `object_ids[i]` may be null when the
/// underlying orbit had no object designation.
///
/// Free with [`empyrean_orbits_batch_free`] when done.
#[repr(C)]
pub struct EmpyreanOrbitBatch {
    /// Heap-allocated array of `EmpyreanOrbit`. Null when `num_orbits == 0`.
    pub orbits: *mut EmpyreanOrbit,
    /// Heap-allocated array of orbit identifiers (null-terminated UTF-8).
    /// Each `orbit_ids[i]` is non-null when `i < num_orbits`.
    pub orbit_ids: *mut *mut c_char,
    /// Heap-allocated array of optional object identifiers
    /// (null-terminated UTF-8 or null pointer when absent).
    pub object_ids: *mut *mut c_char,
    /// Number of orbits in the batch.
    pub num_orbits: usize,
}

impl EmpyreanOrbitBatch {
    fn empty() -> Self {
        Self {
            orbits: std::ptr::null_mut(),
            orbit_ids: std::ptr::null_mut(),
            object_ids: std::ptr::null_mut(),
            num_orbits: 0,
        }
    }
}

/// Free a batch previously returned by an `empyrean_orbits_read_*`
/// function. Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_batch_free(batch: *mut EmpyreanOrbitBatch) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if batch.is_null() {
            return;
        }
        let b = unsafe { &mut *batch };
        let n = b.num_orbits;
        if !b.orbit_ids.is_null() && n > 0 {
            for i in 0..n {
                let p = unsafe { *b.orbit_ids.add(i) };
                if !p.is_null() {
                    drop(unsafe { CString::from_raw(p) });
                }
            }
            let layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
            unsafe { std::alloc::dealloc(b.orbit_ids as *mut u8, layout) };
        }
        if !b.object_ids.is_null() && n > 0 {
            for i in 0..n {
                let p = unsafe { *b.object_ids.add(i) };
                if !p.is_null() {
                    drop(unsafe { CString::from_raw(p) });
                }
            }
            let layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
            unsafe { std::alloc::dealloc(b.object_ids as *mut u8, layout) };
        }
        if !b.orbits.is_null() && n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanOrbit>(n).unwrap();
            unsafe { std::alloc::dealloc(b.orbits as *mut u8, layout) };
        }
        b.orbits = std::ptr::null_mut();
        b.orbit_ids = std::ptr::null_mut();
        b.object_ids = std::ptr::null_mut();
        b.num_orbits = 0;
    }));
}

// ────────────────────────────────────────────────────────────────────
// Internal Rust row types (used for serde + transit).
//
// Mirror of the EmpyreanOrbit / EmpyreanEvent / EmpyreanEphemerisEntry
// flat structs with serde derives — chosen as the row-level type for
// JSON serialization and for round-tripping through villeneuve's
// parquet I/O.
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrbitRow {
    orbit_id: String,
    object_id: Option<String>,
    epoch_mjd_tdb: f64,
    elements: [f64; 6],
    representation: String,
    frame: String,
    origin: i32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    covariance: Option<[[f64; 6]; 6]>,
    #[serde(default)]
    a1: f64,
    #[serde(default)]
    a2: f64,
    #[serde(default)]
    a3: f64,
    #[serde(default)]
    ng_alpha: f64,
    #[serde(default)]
    ng_r0: f64,
    #[serde(default)]
    ng_m: f64,
    #[serde(default)]
    ng_n: f64,
    #[serde(default)]
    ng_k: f64,
}

fn orbit_to_row(orbit: &EmpyreanOrbit, orbit_id: &str, object_id: Option<&str>) -> OrbitRow {
    OrbitRow {
        orbit_id: orbit_id.to_string(),
        object_id: object_id.map(str::to_string),
        epoch_mjd_tdb: orbit.state.epoch_mjd_tdb,
        elements: orbit.state.elements,
        representation: rep_int_to_str(orbit.state.representation).to_string(),
        frame: frame_int_to_str(orbit.state.frame).to_string(),
        origin: orbit.state.origin,
        covariance: (orbit.state.has_covariance != 0).then_some(orbit.state.covariance),
        a1: orbit.a1,
        a2: orbit.a2,
        a3: orbit.a3,
        ng_alpha: orbit.ng_alpha,
        ng_r0: orbit.ng_r0,
        ng_m: orbit.ng_m,
        ng_n: orbit.ng_n,
        ng_k: orbit.ng_k,
    }
}

fn row_to_orbit(row: &OrbitRow) -> Result<(EmpyreanOrbit, String, Option<String>), String> {
    let representation = rep_str_to_int(&row.representation)?;
    let frame = frame_str_to_int(&row.frame)?;
    let has_covariance: u8 = if row.covariance.is_some() { 1 } else { 0 };
    let covariance = row.covariance.unwrap_or([[0.0; 6]; 6]);
    let state = CoordinateState {
        epoch_mjd_tdb: row.epoch_mjd_tdb,
        elements: row.elements,
        covariance,
        has_covariance,
        representation,
        frame,
        origin: row.origin,
    };
    // The IO path returns the row's orbit_id / object_id in the tuple
    // alongside the EmpyreanOrbit; the caller binds them via
    // EmpyreanOrbitBatch's parallel arrays. The orbit struct's own
    // orbit_id / object_id pointers stay null here — they're for the
    // direct-call path (propagate, generate_ephemeris) where the caller
    // owns the CString storage.
    let orbit = EmpyreanOrbit {
        state,
        orbit_id: std::ptr::null(),
        object_id: std::ptr::null(),
        a1: row.a1,
        a2: row.a2,
        a3: row.a3,
        ng_alpha: row.ng_alpha,
        ng_r0: row.ng_r0,
        ng_m: row.ng_m,
        ng_n: row.ng_n,
        ng_k: row.ng_k,
        // OrbitRow JSON/parquet schema does not carry the SBDB
        // non-grav DT yet; round-trip restores NaN (no delay) until
        // villeneuve::io::orbit_row gets a `non_grav_dt` field.
        non_grav_dt: f64::NAN,
        // Non-grav covariance is an OD-output concept (a fitted prior); the
        // orbit-read paths don't carry it (empyrean-wo4n).
        has_non_grav_covariance: 0,
        non_grav_covariance: [[0.0; 3]; 3],
        // OrbitRow JSON/parquet schema does not carry photometry yet;
        // round-tripped orbits come back without it.
        phot_system: -1,
        h_mag: f64::NAN,
        slope1: 0.0,
        slope2: 0.0,
    };
    Ok((orbit, row.orbit_id.clone(), row.object_id.clone()))
}

fn rep_int_to_str(val: i32) -> &'static str {
    match val {
        0 => "cartesian",
        1 => "keplerian",
        2 => "cometary",
        3 => "spherical",
        _ => "cartesian",
    }
}

fn rep_str_to_int(s: &str) -> Result<i32, String> {
    match s.to_ascii_lowercase().as_str() {
        "cartesian" => Ok(0),
        "keplerian" => Ok(1),
        "cometary" => Ok(2),
        "spherical" => Ok(3),
        other => Err(format!("unknown representation: {other}")),
    }
}

fn frame_int_to_str(val: i32) -> &'static str {
    match val {
        0 => "icrf",
        1 => "ecliptic_j2000",
        _ => "icrf",
    }
}

fn frame_str_to_int(s: &str) -> Result<i32, String> {
    match s.to_ascii_lowercase().as_str() {
        "icrf" => Ok(0),
        "ecliptic_j2000" | "eclipticj2000" | "ecliptic" => Ok(1),
        other => Err(format!("unknown frame: {other}")),
    }
}

// ────────────────────────────────────────────────────────────────────
// Helpers — assemble batches from row collections / rich types
// ────────────────────────────────────────────────────────────────────

fn rows_to_batch(rows: Vec<OrbitRow>) -> Result<EmpyreanOrbitBatch, String> {
    let n = rows.len();
    if n == 0 {
        return Ok(EmpyreanOrbitBatch::empty());
    }
    let orbits_layout = std::alloc::Layout::array::<EmpyreanOrbit>(n).unwrap();
    let ids_layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
    let orbits_ptr = unsafe { std::alloc::alloc(orbits_layout) } as *mut EmpyreanOrbit;
    let orbit_ids_ptr = unsafe { std::alloc::alloc(ids_layout) } as *mut *mut c_char;
    let object_ids_ptr = unsafe { std::alloc::alloc(ids_layout) } as *mut *mut c_char;
    if orbits_ptr.is_null() || orbit_ids_ptr.is_null() || object_ids_ptr.is_null() {
        return Err("allocation failed for orbit batch".into());
    }
    for (i, row) in rows.iter().enumerate() {
        let (orbit, orbit_id, object_id) = row_to_orbit(row)?;
        unsafe { orbits_ptr.add(i).write(orbit) };
        let id_c = CString::new(orbit_id.as_bytes()).unwrap_or_default();
        unsafe { orbit_ids_ptr.add(i).write(id_c.into_raw()) };
        let obj_c = match object_id {
            Some(s) => CString::new(s.as_bytes()).unwrap_or_default().into_raw(),
            None => std::ptr::null_mut(),
        };
        unsafe { object_ids_ptr.add(i).write(obj_c) };
    }
    Ok(EmpyreanOrbitBatch {
        orbits: orbits_ptr,
        orbit_ids: orbit_ids_ptr,
        object_ids: object_ids_ptr,
        num_orbits: n,
    })
}

fn batch_to_rows(batch: &EmpyreanOrbitBatch) -> Result<Vec<OrbitRow>, String> {
    if batch.num_orbits == 0 {
        return Ok(Vec::new());
    }
    if batch.orbits.is_null() || batch.orbit_ids.is_null() {
        return Err("null pointer in orbit batch".into());
    }
    let mut rows = Vec::with_capacity(batch.num_orbits);
    for i in 0..batch.num_orbits {
        let orbit = unsafe { &*batch.orbits.add(i) };
        let id_ptr = unsafe { *batch.orbit_ids.add(i) };
        if id_ptr.is_null() {
            return Err(format!("null orbit_id at index {i}"));
        }
        let orbit_id = unsafe { CStr::from_ptr(id_ptr) }
            .to_str()
            .map_err(|e| format!("invalid UTF-8 in orbit_id[{i}]: {e}"))?;
        let object_id = if !batch.object_ids.is_null() {
            let obj_ptr = unsafe { *batch.object_ids.add(i) };
            if obj_ptr.is_null() {
                None
            } else {
                Some(
                    unsafe { CStr::from_ptr(obj_ptr) }
                        .to_str()
                        .map_err(|e| format!("invalid UTF-8 in object_id[{i}]: {e}"))?
                        .to_string(),
                )
            }
        } else {
            None
        };
        rows.push(orbit_to_row(orbit, orbit_id, object_id.as_deref()));
    }
    Ok(rows)
}

/// Convert an [`EmpyreanOrbitBatch`] into a villeneuve `Orbits<AU>`,
/// preserving non-grav parameters and non-Cartesian representations.
pub(crate) fn batch_to_orbits(batch: &EmpyreanOrbitBatch) -> Result<Orbits<AU>, String> {
    let mut out: Orbits<AU> = Orbits::empty();
    for i in 0..batch.num_orbits {
        let orbit = unsafe { &*batch.orbits.add(i) };
        let id_ptr = unsafe { *batch.orbit_ids.add(i) };
        let orbit_id = unsafe { CStr::from_ptr(id_ptr) }
            .to_str()
            .map_err(|e| format!("invalid UTF-8 in orbit_id[{i}]: {e}"))?
            .to_string();
        let object_id = if !batch.object_ids.is_null() {
            let p = unsafe { *batch.object_ids.add(i) };
            if p.is_null() {
                None
            } else {
                Some(
                    unsafe { CStr::from_ptr(p) }
                        .to_str()
                        .map_err(|e| format!("invalid UTF-8 in object_id[{i}]: {e}"))?
                        .to_string(),
                )
            }
        } else {
            None
        };
        let state = orbit.state.to_empyrean();
        let coords = coordinate_state_to_coordinates(&state)
            .map_err(|e| format!("orbit {i}: {e}"))?
            .into_radians();
        out.push_with_object(orbit_id, object_id, coords)
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
                dt: if orbit.non_grav_dt.is_finite() {
                    Some(orbit.non_grav_dt)
                } else {
                    None
                },
            };
            out.set_non_grav_params(i, Some(params));
        }
    }
    Ok(out)
}

/// Convert a villeneuve `Orbits<AU>` into an [`EmpyreanOrbitBatch`].
pub(crate) fn orbits_to_batch(orbits: &Orbits<AU>) -> Result<EmpyreanOrbitBatch, String> {
    let n = orbits.len();
    if n == 0 {
        return Ok(EmpyreanOrbitBatch::empty());
    }
    let orbits_layout = std::alloc::Layout::array::<EmpyreanOrbit>(n).unwrap();
    let ids_layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
    let orbits_ptr = unsafe { std::alloc::alloc(orbits_layout) } as *mut EmpyreanOrbit;
    let orbit_ids_ptr = unsafe { std::alloc::alloc(ids_layout) } as *mut *mut c_char;
    let object_ids_ptr = unsafe { std::alloc::alloc(ids_layout) } as *mut *mut c_char;
    if orbits_ptr.is_null() || orbit_ids_ptr.is_null() || object_ids_ptr.is_null() {
        return Err("allocation failed for orbit batch".into());
    }
    for i in 0..n {
        let coord = orbits.coordinates()[i].into_angular::<empyrean_core::coordinates::Degrees>();
        let cs = coordinates_to_coordinate_state(&coord);
        let mut orbit = EmpyreanOrbit {
            state: CoordinateState::from_empyrean(&cs),
            // Same as the read path: per-orbit id pointers stay null in
            // the IO context; the orbit_id / object_id strings live in
            // the parallel `orbit_ids` / `object_ids` arrays of the
            // batch instead.
            orbit_id: std::ptr::null(),
            object_id: std::ptr::null(),
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            non_grav_dt: f64::NAN,
            // Non-grav covariance is an OD-output concept; the read-orbits
            // path doesn't carry it (empyrean-wo4n).
            has_non_grav_covariance: 0,
            non_grav_covariance: [[0.0; 3]; 3],
            // Photometry is not currently carried through the read-orbits
            // path; ephemeris generation downstream will see no H/G and
            // emit `mag = NaN`.
            phot_system: -1,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
        };
        if let Some(ph) = orbits.photometric_params(i) {
            orbit.h_mag = ph.h();
            orbit.phot_system = match ph.phase_function {
                empyrean_core::photometry::PhaseFunction::HG => 0,
                empyrean_core::photometry::PhaseFunction::HG1G2 => 1,
                empyrean_core::photometry::PhaseFunction::HG12 => 2,
            };
            orbit.slope1 = ph.p2;
            orbit.slope2 = ph.p3;
        }
        if let Some(ng) = orbits.non_grav_params(i) {
            orbit.a1 = ng.a1;
            orbit.a2 = ng.a2;
            orbit.a3 = ng.a3;
            orbit.non_grav_dt = ng.dt.unwrap_or(f64::NAN);
            if let NonGravModel::MarsdenSekanina(g) = &ng.model {
                orbit.ng_alpha = g.alpha;
                orbit.ng_r0 = g.r0;
                orbit.ng_m = g.m;
                orbit.ng_n = g.n;
                orbit.ng_k = g.k;
            }
        }
        unsafe { orbits_ptr.add(i).write(orbit) };
        let id_c =
            CString::new(orbits.orbit_ids()[i].as_str()).unwrap_or_else(|_| CString::default());
        unsafe { orbit_ids_ptr.add(i).write(id_c.into_raw()) };
        let obj_ptr = match orbits.object_ids()[i].as_ref() {
            Some(s) => CString::new(s.as_str())
                .unwrap_or_else(|_| CString::default())
                .into_raw(),
            None => std::ptr::null_mut(),
        };
        unsafe { object_ids_ptr.add(i).write(obj_ptr) };
    }
    Ok(EmpyreanOrbitBatch {
        orbits: orbits_ptr,
        orbit_ids: orbit_ids_ptr,
        object_ids: object_ids_ptr,
        num_orbits: n,
    })
}

// ────────────────────────────────────────────────────────────────────
// Orbit I/O — parquet
// ────────────────────────────────────────────────────────────────────

/// Read an orbits parquet file. Caller frees the result with
/// [`empyrean_orbits_batch_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_read_parquet(
    path: *const c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    file_op(path, out, |p, o| {
        let orbits: Orbits<AU> = empyrean_core::io::parquet::read_orbits(p)
            .map_err(|e| format!("parquet read failed: {e:?}"))?;
        *o = orbits_to_batch(&orbits)?;
        Ok(())
    })
}

/// Write an orbit batch to a parquet file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_write_parquet(
    path: *const c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    file_in_op(path, batch, |p, b| {
        let orbits = batch_to_orbits(b)?;
        empyrean_core::io::parquet::write_orbits(p, &orbits)
            .map_err(|e| format!("parquet write failed: {e:?}"))
    })
}

// ────────────────────────────────────────────────────────────────────
// Orbit I/O — JSON
// ────────────────────────────────────────────────────────────────────

/// Read an orbits JSON file (array of orbit-row objects). Caller frees
/// with [`empyrean_orbits_batch_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_read_json(
    path: *const c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    file_op(path, out, |p, o| {
        let f = File::open(p).map_err(|e| format!("open: {e}"))?;
        let rows: Vec<OrbitRow> =
            serde_json::from_reader(f).map_err(|e| format!("json parse: {e}"))?;
        *o = rows_to_batch(rows)?;
        Ok(())
    })
}

/// Write an orbit batch to JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_write_json(
    path: *const c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    file_in_op(path, batch, |p, b| {
        let rows = batch_to_rows(b)?;
        let f = File::create(p).map_err(|e| format!("create: {e}"))?;
        serde_json::to_writer_pretty(f, &rows).map_err(|e| format!("json write: {e}"))
    })
}

// ────────────────────────────────────────────────────────────────────
// Orbit I/O — CSV
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct OrbitCsvRow {
    orbit_id: String,
    object_id: String,
    epoch_mjd_tdb: f64,
    e0: f64,
    e1: f64,
    e2: f64,
    e3: f64,
    e4: f64,
    e5: f64,
    representation: String,
    frame: String,
    origin: i32,
    a1: f64,
    a2: f64,
    a3: f64,
    ng_alpha: f64,
    ng_r0: f64,
    ng_m: f64,
    ng_n: f64,
    ng_k: f64,
}

impl From<&OrbitRow> for OrbitCsvRow {
    fn from(r: &OrbitRow) -> Self {
        Self {
            orbit_id: r.orbit_id.clone(),
            object_id: r.object_id.clone().unwrap_or_default(),
            epoch_mjd_tdb: r.epoch_mjd_tdb,
            e0: r.elements[0],
            e1: r.elements[1],
            e2: r.elements[2],
            e3: r.elements[3],
            e4: r.elements[4],
            e5: r.elements[5],
            representation: r.representation.clone(),
            frame: r.frame.clone(),
            origin: r.origin,
            a1: r.a1,
            a2: r.a2,
            a3: r.a3,
            ng_alpha: r.ng_alpha,
            ng_r0: r.ng_r0,
            ng_m: r.ng_m,
            ng_n: r.ng_n,
            ng_k: r.ng_k,
        }
    }
}

impl From<OrbitCsvRow> for OrbitRow {
    fn from(r: OrbitCsvRow) -> Self {
        Self {
            orbit_id: r.orbit_id,
            object_id: if r.object_id.is_empty() {
                None
            } else {
                Some(r.object_id)
            },
            epoch_mjd_tdb: r.epoch_mjd_tdb,
            elements: [r.e0, r.e1, r.e2, r.e3, r.e4, r.e5],
            representation: r.representation,
            frame: r.frame,
            origin: r.origin,
            covariance: None,
            a1: r.a1,
            a2: r.a2,
            a3: r.a3,
            ng_alpha: r.ng_alpha,
            ng_r0: r.ng_r0,
            ng_m: r.ng_m,
            ng_n: r.ng_n,
            ng_k: r.ng_k,
        }
    }
}

/// Read an orbits CSV file.
///
/// CSV does not carry covariance (use parquet for covariance round-trip).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_read_csv(
    path: *const c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    file_op(path, out, |p, o| {
        let mut reader = csv::Reader::from_path(p).map_err(|e| format!("csv open: {e}"))?;
        let mut rows = Vec::new();
        for rec in reader.deserialize::<OrbitCsvRow>() {
            let r = rec.map_err(|e| format!("csv parse: {e}"))?;
            rows.push(OrbitRow::from(r));
        }
        *o = rows_to_batch(rows)?;
        Ok(())
    })
}

/// Write an orbit batch to CSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_write_csv(
    path: *const c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    file_in_op(path, batch, |p, b| {
        let rows = batch_to_rows(b)?;
        let mut wtr = csv::Writer::from_path(p).map_err(|e| format!("csv create: {e}"))?;
        for row in &rows {
            wtr.serialize(OrbitCsvRow::from(row))
                .map_err(|e| format!("csv write: {e}"))?;
        }
        wtr.flush().map_err(|e| format!("csv flush: {e}"))
    })
}

// ────────────────────────────────────────────────────────────────────
// Ephemeris I/O — write only (parquet/JSON/CSV)
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct EphemerisRow {
    orbit_id: String,
    obs_code: String,
    epoch_mjd_tdb: f64,
    ra_deg: f64,
    dec_deg: f64,
    rho_au: f64,
    vrho_au_day: f64,
    vra_deg_day: f64,
    vdec_deg_day: f64,
    light_time_days: f64,
    phase_angle_deg: f64,
    elongation_deg: f64,
    heliocentric_distance_au: f64,
    mag: f64,
    mag_sigma: f64,
    // Topocentric / sky-motion angles — present on EmpyreanEphemerisEntry
    // (the wrapper fills them) but previously omitted by every file writer
    // (bd empyrean-i7u5).
    zenith_angle_deg: f64,
    azimuth_deg: f64,
    hour_angle_deg: f64,
    lunar_elongation_deg: f64,
    position_angle_deg: f64,
    sky_rate_deg_day: f64,
}

fn ephemeris_to_rows(entries: &[EmpyreanEphemerisEntry]) -> Vec<EphemerisRow> {
    entries
        .iter()
        .map(|e| {
            let orbit_id = if e.orbit_id.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(e.orbit_id) }
                    .to_string_lossy()
                    .into_owned()
            };
            let obs_code = obs_code_bytes_to_string(&e.obs_code);
            EphemerisRow {
                orbit_id,
                obs_code,
                epoch_mjd_tdb: e.epoch_mjd_tdb,
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
            }
        })
        .collect()
}

fn obs_code_bytes_to_string(bytes: &[u8; 4]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Write ephemeris entries to parquet using the villeneuve schema.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_ephemeris_write_parquet(
    path: *const c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    array_in_op(path, entries_ptr, num_entries, |p, slice| {
        let rows = ephemeris_to_rows(slice);
        write_rows_parquet_generic(p, &rows, &EPHEMERIS_PARQUET_FIELDS, |row, builders| {
            ephemeris_append(row, builders)
        })
    })
}

/// Write ephemeris entries to JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_ephemeris_write_json(
    path: *const c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    array_in_op(path, entries_ptr, num_entries, |p, slice| {
        let rows = ephemeris_to_rows(slice);
        write_json(p, &rows)
    })
}

/// Write ephemeris entries to CSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_ephemeris_write_csv(
    path: *const c_char,
    entries_ptr: *const EmpyreanEphemerisEntry,
    num_entries: usize,
) -> i32 {
    array_in_op(path, entries_ptr, num_entries, |p, slice| {
        let rows = ephemeris_to_rows(slice);
        write_csv(p, &rows)
    })
}

// ────────────────────────────────────────────────────────────────────
// Events I/O — write only
// ────────────────────────────────────────────────────────────────────

// Carries the full per-event-type payload, mirroring `EmpyreanEvent` —
// previously common-fields-only, which silently dropped every type-specific
// field on the file path (bd empyrean-i7u5). Inapplicable fields are NaN /
// `-1` sentinels for a given event type, matching the in-memory Event.
#[derive(Debug, Serialize, Deserialize)]
struct EventRow {
    orbit_id: String,
    event_type: String,
    body: String,
    body_naif_id: i32,
    epoch_mjd_tdb: f64,
    distance_au: f64,
    distance_km: f64,
    relative_velocity_au_day: f64,
    // capture
    two_body_energy: f64,
    jacobi_constant: f64,
    jacobi_constant_sigma: f64,
    jacobi_constant_l1: f64,
    jacobi_constant_l2: f64,
    n_periapses: i32,
    // impact / atmospheric planetodetic
    impact_latitude_deg: f64,
    impact_longitude_deg: f64,
    impact_altitude_km: f64,
    // shadow
    shadow_fraction: f64,
    illumination: f64,
    // periapsis relative state
    relative_x: f64,
    relative_y: f64,
    relative_z: f64,
    relative_vx: f64,
    relative_vy: f64,
    relative_vz: f64,
    // possible-impact probability payload
    effective_radius_au: f64,
    effective_radius_km: f64,
    sigma_distance_au: f64,
    ip_linear: f64,
    ip_second_order: f64,
    nonlinearity: f64,
    ip_agm: f64,
    ip_mc: f64,
    // covariance-regime-change (kind codes: -1 = N/A, else 0..4)
    previous_kind: i32,
    resolved_kind: i32,
    kappa: f64,
    threshold_below: f64,
    threshold_above: f64,
}

/// Covariance-kind u8 (`0xFF` sentinel) -> i32 code (`-1` = N/A).
fn kind_code(k: u8) -> i32 {
    if k == 0xFF { -1 } else { k as i32 }
}

fn events_to_rows(events: &[EmpyreanEvent]) -> Vec<EventRow> {
    events
        .iter()
        .map(|e| EventRow {
            orbit_id: cstr_to_string(e.orbit_id),
            event_type: cstr_to_string(e.event_type),
            body: cstr_to_string(e.body),
            body_naif_id: e.body_naif_id,
            epoch_mjd_tdb: e.epoch_mjd_tdb,
            distance_au: e.distance_au,
            distance_km: e.distance_km,
            relative_velocity_au_day: e.relative_velocity_au_day,
            two_body_energy: e.two_body_energy,
            jacobi_constant: e.jacobi_constant,
            jacobi_constant_sigma: e.jacobi_constant_sigma,
            jacobi_constant_l1: e.jacobi_constant_l1,
            jacobi_constant_l2: e.jacobi_constant_l2,
            n_periapses: e.n_periapses,
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
            previous_kind: kind_code(e.previous_kind),
            resolved_kind: kind_code(e.resolved_kind),
            kappa: e.kappa,
            threshold_below: e.threshold_below,
            threshold_above: e.threshold_above,
        })
        .collect()
}

fn cstr_to_string(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// Write events to parquet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_events_write_parquet(
    path: *const c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    array_in_op(path, events_ptr, num_events, |p, slice| {
        let rows = events_to_rows(slice);
        write_rows_parquet_generic(p, &rows, &EVENT_PARQUET_FIELDS, |row, builders| {
            event_append(row, builders)
        })
    })
}

/// Write events to JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_events_write_json(
    path: *const c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    array_in_op(path, events_ptr, num_events, |p, slice| {
        let rows = events_to_rows(slice);
        write_json(p, &rows)
    })
}

/// Write events to CSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_events_write_csv(
    path: *const c_char,
    events_ptr: *const EmpyreanEvent,
    num_events: usize,
) -> i32 {
    array_in_op(path, events_ptr, num_events, |p, slice| {
        let rows = events_to_rows(slice);
        write_csv(p, &rows)
    })
}

// ────────────────────────────────────────────────────────────────────
// Residuals I/O — write only
// ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct ResidualRow {
    ra_residual_arcsec: f64,
    dec_residual_arcsec: f64,
    chi2: f64,
    probability: f64,
    selected: bool,
}

fn residuals_to_rows(observations: &[EmpyreanObservationResult]) -> Vec<ResidualRow> {
    observations
        .iter()
        .map(|o| ResidualRow {
            ra_residual_arcsec: o.ra_residual_arcsec,
            dec_residual_arcsec: o.dec_residual_arcsec,
            chi2: o.chi2,
            probability: o.probability,
            selected: o.selected != 0,
        })
        .collect()
}

/// Write OD residuals to parquet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_residuals_write_parquet(
    path: *const c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    array_in_op(path, obs_ptr, num_obs, |p, slice| {
        let rows = residuals_to_rows(slice);
        write_rows_parquet_generic(p, &rows, &RESIDUAL_PARQUET_FIELDS, |row, builders| {
            residual_append(row, builders)
        })
    })
}

/// Write OD residuals to JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_residuals_write_json(
    path: *const c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    array_in_op(path, obs_ptr, num_obs, |p, slice| {
        let rows = residuals_to_rows(slice);
        write_json(p, &rows)
    })
}

/// Write OD residuals to CSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_residuals_write_csv(
    path: *const c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    array_in_op(path, obs_ptr, num_obs, |p, slice| {
        let rows = residuals_to_rows(slice);
        write_csv(p, &rows)
    })
}

// ────────────────────────────────────────────────────────────────────
// Generic JSON / CSV / parquet helpers
// ────────────────────────────────────────────────────────────────────

fn write_json<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("create: {e}"))?;
    serde_json::to_writer_pretty(f, rows).map_err(|e| format!("json write: {e}"))
}

fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), String> {
    let mut wtr = csv::Writer::from_path(path).map_err(|e| format!("csv create: {e}"))?;
    for row in rows {
        wtr.serialize(row).map_err(|e| format!("csv write: {e}"))?;
    }
    wtr.flush().map_err(|e| format!("csv flush: {e}"))
}

// Per-row-type parquet plumbing. Rather than introduce a row trait we
// inline the schema descriptors and append closures — same effect, no
// extra abstractions.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanBuilder, Float64Builder, Int32Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

struct ParquetField {
    name: &'static str,
    data_type: DataType,
    nullable: bool,
}

enum Builder {
    F64(Float64Builder),
    Bool(BooleanBuilder),
    I32(Int32Builder),
    Str(StringBuilder),
}

impl Builder {
    fn finish(self) -> ArrayRef {
        match self {
            Builder::F64(mut b) => Arc::new(b.finish()),
            Builder::Bool(mut b) => Arc::new(b.finish()),
            Builder::I32(mut b) => Arc::new(b.finish()),
            Builder::Str(mut b) => Arc::new(b.finish()),
        }
    }
}

fn make_builder(f: &ParquetField, capacity: usize) -> Builder {
    match &f.data_type {
        DataType::Float64 => Builder::F64(Float64Builder::with_capacity(capacity)),
        DataType::Boolean => Builder::Bool(BooleanBuilder::with_capacity(capacity)),
        DataType::Int32 => Builder::I32(Int32Builder::with_capacity(capacity)),
        DataType::Utf8 => Builder::Str(StringBuilder::with_capacity(capacity, capacity * 16)),
        _ => unreachable!("unsupported parquet column type {:?}", f.data_type),
    }
}

fn write_rows_parquet_generic<T>(
    path: &Path,
    rows: &[T],
    fields: &[ParquetField],
    mut append: impl FnMut(&T, &mut [Builder]) -> Result<(), String>,
) -> Result<(), String> {
    let schema_fields: Vec<Field> = fields
        .iter()
        .map(|f| Field::new(f.name, f.data_type.clone(), f.nullable))
        .collect();
    let schema = Arc::new(Schema::new(schema_fields));
    let mut builders: Vec<Builder> = fields.iter().map(|f| make_builder(f, rows.len())).collect();
    for row in rows {
        append(row, &mut builders)?;
    }
    let cols: Vec<ArrayRef> = builders.into_iter().map(|b| b.finish()).collect();
    let batch =
        RecordBatch::try_new(schema.clone(), cols).map_err(|e| format!("record batch: {e}"))?;
    let f = File::create(path).map_err(|e| format!("create: {e}"))?;
    let mut writer =
        ArrowWriter::try_new(f, schema, None).map_err(|e| format!("parquet writer: {e}"))?;
    writer
        .write(&batch)
        .map_err(|e| format!("parquet write: {e}"))?;
    writer.close().map_err(|e| format!("parquet close: {e}"))?;
    Ok(())
}

// Schema descriptors + append fns per row type.

const EPHEMERIS_PARQUET_FIELDS: [ParquetField; 21] = [
    ParquetField {
        name: "orbit_id",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "obs_code",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "epoch_mjd_tdb",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "ra_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "dec_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "rho_au",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "vrho_au_day",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "vra_deg_day",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "vdec_deg_day",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "light_time_days",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "phase_angle_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "elongation_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "heliocentric_distance_au",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "mag",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "mag_sigma",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "zenith_angle_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "azimuth_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "hour_angle_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "lunar_elongation_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "position_angle_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "sky_rate_deg_day",
        data_type: DataType::Float64,
        nullable: false,
    },
];

fn ephemeris_append(row: &EphemerisRow, b: &mut [Builder]) -> Result<(), String> {
    if let Builder::Str(s) = &mut b[0] {
        s.append_value(&row.orbit_id);
    }
    if let Builder::Str(s) = &mut b[1] {
        s.append_value(&row.obs_code);
    }
    if let Builder::F64(f) = &mut b[2] {
        f.append_value(row.epoch_mjd_tdb);
    }
    if let Builder::F64(f) = &mut b[3] {
        f.append_value(row.ra_deg);
    }
    if let Builder::F64(f) = &mut b[4] {
        f.append_value(row.dec_deg);
    }
    if let Builder::F64(f) = &mut b[5] {
        f.append_value(row.rho_au);
    }
    if let Builder::F64(f) = &mut b[6] {
        f.append_value(row.vrho_au_day);
    }
    if let Builder::F64(f) = &mut b[7] {
        f.append_value(row.vra_deg_day);
    }
    if let Builder::F64(f) = &mut b[8] {
        f.append_value(row.vdec_deg_day);
    }
    if let Builder::F64(f) = &mut b[9] {
        f.append_value(row.light_time_days);
    }
    if let Builder::F64(f) = &mut b[10] {
        f.append_value(row.phase_angle_deg);
    }
    if let Builder::F64(f) = &mut b[11] {
        f.append_value(row.elongation_deg);
    }
    if let Builder::F64(f) = &mut b[12] {
        f.append_value(row.heliocentric_distance_au);
    }
    if let Builder::F64(f) = &mut b[13] {
        f.append_value(row.mag);
    }
    if let Builder::F64(f) = &mut b[14] {
        f.append_value(row.mag_sigma);
    }
    if let Builder::F64(f) = &mut b[15] {
        f.append_value(row.zenith_angle_deg);
    }
    if let Builder::F64(f) = &mut b[16] {
        f.append_value(row.azimuth_deg);
    }
    if let Builder::F64(f) = &mut b[17] {
        f.append_value(row.hour_angle_deg);
    }
    if let Builder::F64(f) = &mut b[18] {
        f.append_value(row.lunar_elongation_deg);
    }
    if let Builder::F64(f) = &mut b[19] {
        f.append_value(row.position_angle_deg);
    }
    if let Builder::F64(f) = &mut b[20] {
        f.append_value(row.sky_rate_deg_day);
    }
    Ok(())
}

const EVENT_PARQUET_FIELDS: [ParquetField; 38] = [
    ParquetField {
        name: "orbit_id",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "event_type",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "body",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "body_naif_id",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "epoch_mjd_tdb",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "distance_au",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "distance_km",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_velocity_au_day",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "two_body_energy",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "jacobi_constant",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "jacobi_constant_sigma",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "jacobi_constant_l1",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "jacobi_constant_l2",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "n_periapses",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "impact_latitude_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "impact_longitude_deg",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "impact_altitude_km",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "shadow_fraction",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "illumination",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_x",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_y",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_z",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_vx",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_vy",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "relative_vz",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "effective_radius_au",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "effective_radius_km",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "sigma_distance_au",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "ip_linear",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "ip_second_order",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "nonlinearity",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "ip_agm",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "ip_mc",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "previous_kind",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "resolved_kind",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "kappa",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "threshold_below",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "threshold_above",
        data_type: DataType::Float64,
        nullable: false,
    },
];

fn event_append(row: &EventRow, b: &mut [Builder]) -> Result<(), String> {
    macro_rules! f64_at {
        ($idx:expr, $val:expr) => {
            if let Builder::F64(f) = &mut b[$idx] {
                f.append_value($val);
            }
        };
    }
    macro_rules! i32_at {
        ($idx:expr, $val:expr) => {
            if let Builder::I32(i) = &mut b[$idx] {
                i.append_value($val);
            }
        };
    }
    if let Builder::Str(s) = &mut b[0] {
        s.append_value(&row.orbit_id);
    }
    if let Builder::Str(s) = &mut b[1] {
        s.append_value(&row.event_type);
    }
    if let Builder::Str(s) = &mut b[2] {
        s.append_value(&row.body);
    }
    i32_at!(3, row.body_naif_id);
    f64_at!(4, row.epoch_mjd_tdb);
    f64_at!(5, row.distance_au);
    f64_at!(6, row.distance_km);
    f64_at!(7, row.relative_velocity_au_day);
    f64_at!(8, row.two_body_energy);
    f64_at!(9, row.jacobi_constant);
    f64_at!(10, row.jacobi_constant_sigma);
    f64_at!(11, row.jacobi_constant_l1);
    f64_at!(12, row.jacobi_constant_l2);
    i32_at!(13, row.n_periapses);
    f64_at!(14, row.impact_latitude_deg);
    f64_at!(15, row.impact_longitude_deg);
    f64_at!(16, row.impact_altitude_km);
    f64_at!(17, row.shadow_fraction);
    f64_at!(18, row.illumination);
    f64_at!(19, row.relative_x);
    f64_at!(20, row.relative_y);
    f64_at!(21, row.relative_z);
    f64_at!(22, row.relative_vx);
    f64_at!(23, row.relative_vy);
    f64_at!(24, row.relative_vz);
    f64_at!(25, row.effective_radius_au);
    f64_at!(26, row.effective_radius_km);
    f64_at!(27, row.sigma_distance_au);
    f64_at!(28, row.ip_linear);
    f64_at!(29, row.ip_second_order);
    f64_at!(30, row.nonlinearity);
    f64_at!(31, row.ip_agm);
    f64_at!(32, row.ip_mc);
    i32_at!(33, row.previous_kind);
    i32_at!(34, row.resolved_kind);
    f64_at!(35, row.kappa);
    f64_at!(36, row.threshold_below);
    f64_at!(37, row.threshold_above);
    Ok(())
}

const RESIDUAL_PARQUET_FIELDS: [ParquetField; 5] = [
    ParquetField {
        name: "ra_residual_arcsec",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "dec_residual_arcsec",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "chi2",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "probability",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "selected",
        data_type: DataType::Boolean,
        nullable: false,
    },
];

fn residual_append(row: &ResidualRow, b: &mut [Builder]) -> Result<(), String> {
    if let Builder::F64(f) = &mut b[0] {
        f.append_value(row.ra_residual_arcsec);
    }
    if let Builder::F64(f) = &mut b[1] {
        f.append_value(row.dec_residual_arcsec);
    }
    if let Builder::F64(f) = &mut b[2] {
        f.append_value(row.chi2);
    }
    if let Builder::F64(f) = &mut b[3] {
        f.append_value(row.probability);
    }
    if let Builder::Bool(b_) = &mut b[4] {
        b_.append_value(row.selected);
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────
// Wrapper helpers — null-check, panic-catch, error propagation.
// ────────────────────────────────────────────────────────────────────

fn file_op<F>(path: *const c_char, out: *mut EmpyreanOrbitBatch, op: F) -> i32
where
    F: FnOnce(&Path, &mut EmpyreanOrbitBatch) -> Result<(), String>,
{
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => Path::new(s).to_path_buf(),
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in path: {e}"));
                return -1;
            }
        };
        let out_ref = unsafe { &mut *out };
        *out_ref = EmpyreanOrbitBatch::empty();
        match op(&path, out_ref) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e);
                -2
            }
        }
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in I/O");
            -99
        }
    }
}

fn file_in_op<F>(path: *const c_char, batch: *const EmpyreanOrbitBatch, op: F) -> i32
where
    F: FnOnce(&Path, &EmpyreanOrbitBatch) -> Result<(), String>,
{
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() || batch.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => Path::new(s).to_path_buf(),
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in path: {e}"));
                return -1;
            }
        };
        match op(&path, unsafe { &*batch }) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e);
                -2
            }
        }
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in I/O");
            -99
        }
    }
}

fn array_in_op<T, F>(path: *const c_char, array: *const T, n: usize, op: F) -> i32
where
    F: FnOnce(&Path, &[T]) -> Result<(), String>,
{
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            set_last_error("null path argument");
            return -1;
        }
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(s) => Path::new(s).to_path_buf(),
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in path: {e}"));
                return -1;
            }
        };
        let slice = if n == 0 || array.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(array, n) }
        };
        match op(&path, slice) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e);
                -2
            }
        }
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in I/O");
            -99
        }
    }
}

// Reference imports — `int_to_frame` / `int_to_representation` /
// `frame_to_int` / `representation_to_int` are part of the conversion
// boundary used elsewhere in this module via `coordinate_state_to_*`;
// keeping the imports avoids accidental drift when the row schemas grow
// to handle non-Cartesian or non-ICRF cases.
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = int_to_frame(0);
    let _ = int_to_representation(0);
    let _ = frame_to_int as fn(_) -> _;
    let _ = representation_to_int as fn(_) -> _;
    let _ = std::mem::size_of::<DynamicalEvent>();
}
