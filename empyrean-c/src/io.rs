//! C ABI exports for file I/O.
//!
//! Three formats × five data types:
//!
//! - **Orbits** (read + write): parquet, JSON, CSV.
//! - **Ephemeris** (write): parquet, JSON, CSV.
//! - **Events** (write): parquet, JSON, CSV.
//! - **Residuals** (write): parquet, JSON, CSV.
//! - **Fit summary** (write): parquet, JSON, CSV — one row per object a
//!   batch orbit determination attempted, delivered or failed.
//!
//! All readers populate a flat C-ABI struct or array; writers consume
//! the same types. Caller-allocated results (e.g. `EmpyreanOrbitBatch`
//! produced by a reader) must be released with the matching
//! `*_free()` helper. JSON / CSV use the schemas documented inline; the
//! parquet schemas are villeneuve-native (round-trips with the rest of
//! the empyrean ecosystem).
//!
//! # The covariance, by orbit format
//!
//! **Parquet and JSON both carry the full packed joint** — the whole
//! \\((6+P) \times (6+P)\\) covariance, state↔parameter and
//! parameter↔parameter cross terms included — so a fitted orbit round-trips
//! through either with the joint the fit computed rather than its diagonal
//! blocks. Parquet carries it in villeneuve's tagged `cov_*` columns; JSON
//! carries the same value in `OrbitRow`'s `cov_*` fields (the packed lower
//! triangle `cov_tri` plus the scalar tags), under the same names.
//!
//! **CSV carries the state 6×6 only** — the joint's leading block — and a
//! joint carrying solved-parameter axes is **refused by name** by the
//! engine's own CSV writer, pointing at parquet, rather than written short:
//! a file written short reads back as a block-diagonal joint, a tighter claim
//! than the caller held, with nothing in the round trip to signal it.
//!
//! A carrier holding continuous-thrust \\(\Delta v\\) terms is refused
//! wherever it is offered, because no orbit-file format can serialize the
//! thrust arcs those terms describe. Note the engine additionally has a
//! **JSONL** orbit format keyed by a widened `cov_dim`; that format is not
//! exposed through this ABI, and this crate's `JSON` is a different, flat
//! shape — do not describe this boundary in the engine's `cov_dim` terms.

use std::ffi::{CStr, CString, c_char};
use std::fs::File;
use std::panic::AssertUnwindSafe;
use std::path::Path;

use empyrean_core::convert::{
    coordinate_state_to_coordinates, coordinates_to_coordinate_state, frame_to_int, int_to_frame,
    int_to_representation, representation_to_int,
};
use empyrean_core::coordinates::AU;
use empyrean_core::nongrav::{GFunction, NonGravModel};
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
/// # Ownership of the packed-joint covariance
///
/// On a batch the library **hands you**, each orbit's
/// [`covariance`](EmpyreanOrbit::covariance) packed joint owns its `tri`
/// triangle, and [`empyrean_orbits_batch_free`] releases it with
/// everything else. That is the opposite of the same field on an
/// `EmpyreanOrbit` you **build**, where the triangle is caller-owned and
/// merely borrowed for the call — the identical asymmetry `orbit_id`
/// already has. A caller re-feeding a read orbit into a propagate/OD call
/// before freeing the batch may pass the joint straight through; one that
/// frees the batch first must copy the triangle.
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
    pub(crate) fn empty() -> Self {
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
            // Each orbit's packed-joint triangle is library-owned on a
            // batch the library produced, so it is released here rather
            // than left to the caller. Freed before the orbit array
            // itself, since the triangle lives inside it.
            for i in 0..n {
                let o = unsafe { &mut *b.orbits.add(i) };
                unsafe { crate::joint::free_packed_joint(&mut o.covariance) };
            }
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

/// The schema version of the JSON orbit covariance block, mirroring the
/// parquet covariance wire's `cov_schema` column (villeneuve's
/// `COV_WIRE_SCHEMA`). Written on every joint-bearing row and refused by name
/// on read if a future version ever disagrees — never silently accepted.
const JSON_COV_SCHEMA: i32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OrbitRow {
    orbit_id: String,
    object_id: Option<String>,
    epoch_mjd_tdb: f64,
    elements: [f64; 6],
    representation: String,
    frame: String,
    origin: i32,
    // The removed flat 6×6 `covariance` field. Files written before 0.11.0
    // carried the state block here, as a `[[f64; 6]; 6]`; 0.11.0 carries the
    // full packed joint in the `cov_*` fields below and derives the state 6×6
    // from the joint's leading block. This slot stays only to DETECT such an
    // old file and refuse it by name on read (there is no migration shim); it
    // is never written (`skip_serializing`).
    #[serde(default, skip_serializing)]
    covariance: Option<serde_json::Value>,
    // ── The packed joint — one JSON representation of the orbit's whole
    //    (6+P)×(6+P) covariance, the same value parquet's `cov_*` columns
    //    carry, with the same names, for every width the joint can have.
    //    `cov_tri` is the packed lower triangle in parquet's order; the scalar
    //    tags ride beside it. Absent (every field `None`) when the orbit has
    //    no covariance. The flat 6×6 is the joint's leading block, derived on
    //    read, never stored twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_schema: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_state_width: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_layout_present: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_layout_considered: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_layout_marginalized: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_tri: Option<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_kind: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_quality: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_rank: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_null_axes: Option<u32>,
    // The minimum correlation-space eigenvalue is `NaN` (the ABI's
    // not-available sentinel) for a positive-(semi)definite joint; JSON carries
    // that absence as a missing field and restores it to `NaN` on read, rather
    // than writing a bare `NaN` serde renders as `null`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_min_eig_corr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_functional: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_provenance: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_asserted_source: Option<u8>,
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

/// The packed-joint fields of an [`OrbitRow`], in parquet's `cov_*` names. All
/// `None` is "no covariance"; every field `Some` is a present joint.
#[derive(Default)]
struct CovWire {
    schema: Option<i32>,
    state_width: Option<u8>,
    layout_present: Option<u32>,
    layout_considered: Option<u32>,
    layout_marginalized: Option<u32>,
    tri: Option<Vec<f64>>,
    kind: Option<u8>,
    quality: Option<u8>,
    rank: Option<u8>,
    null_axes: Option<u32>,
    min_eig_corr: Option<f64>,
    functional: Option<u8>,
    provenance: Option<u8>,
    asserted_source: Option<u8>,
}

/// The dense symmetric 6×6 state block read off a packed triangle's leading
/// 21 entries and mirrored across the diagonal. The caller guarantees
/// `tri.len() ≥ 21` (a present joint is width ≥ 6).
#[allow(clippy::needless_range_loop)]
fn state_block_from_tri(tri: &[f64]) -> [[f64; 6]; 6] {
    let mut m = [[0.0f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let v = tri[i * (i + 1) / 2 + j];
            m[i][j] = v;
            m[j][i] = v;
        }
    }
    m
}

/// Read a C [`EmpyreanPackedJoint`]'s fields into the JSON wire form: the
/// packed triangle (copied), the layout bitsets, and every tag read straight
/// off the joint. The leading state-block width is the joint width less its
/// present parameter axes (mirroring parquet's `cov_state_width`); the minimum
/// correlation-space eigenvalue rides only when finite (a PSD joint's NaN
/// sentinel is carried as the absent field, restored to NaN on read).
///
/// # Safety
///
/// `joint.tri` must be non-null and valid for `joint.tri_len` doubles.
unsafe fn cov_wire_from_c_joint(joint: &crate::joint::EmpyreanPackedJoint) -> CovWire {
    let tri = unsafe { std::slice::from_raw_parts(joint.tri, joint.tri_len) }.to_vec();
    CovWire {
        schema: Some(JSON_COV_SCHEMA),
        state_width: Some(
            joint
                .width
                .saturating_sub(joint.layout_present.count_ones() as u8),
        ),
        layout_present: Some(joint.layout_present),
        layout_considered: Some(joint.layout_considered),
        layout_marginalized: Some(joint.layout_marginalized),
        tri: Some(tri),
        kind: Some(joint.kind),
        quality: Some(joint.quality),
        rank: Some(joint.cov_rank),
        null_axes: Some(joint.cov_null_axes),
        min_eig_corr: joint
            .cov_min_eig_corr
            .is_finite()
            .then_some(joint.cov_min_eig_corr),
        functional: Some(joint.functional),
        provenance: Some(joint.provenance),
        asserted_source: Some(joint.asserted_source),
    }
}

/// Read the orbit's covariance as the JSON packed-joint wire fields.
///
/// The packed joint on [`EmpyreanOrbit::covariance`] is the one covariance
/// home: a present joint (width > 0) is serialized whole — its packed triangle
/// and every tag, the same value parquet carries.
///
/// An orbit with only the flat state 6×6 and no joint (a hand-built C orbit) is
/// converted through the SAME engine path the parquet / CSV writers use
/// ([`coordinate_state_to_coordinates`] → `state_uncertainty_from_flat`), so
/// its tags are identical across all three formats: an externally supplied
/// flat matrix is an ASSERTED moment from `AssertedSource::ExternalFile`, its
/// quality is the engine's definiteness check, and an asymmetric block is
/// refused BY NAME exactly as on a parquet / CSV write — not stamped a
/// fabricated `Derived` / positive-definite. An orbit with neither covariance
/// leaves every `cov_*` field absent.
fn cov_wire_from_orbit(orbit: &EmpyreanOrbit) -> Result<CovWire, String> {
    let joint = &orbit.covariance;
    if joint.width > 0 && !joint.tri.is_null() {
        return Ok(unsafe { cov_wire_from_c_joint(joint) });
    }
    if orbit.state.has_covariance == 0 {
        return Ok(CovWire::default());
    }
    // Flat-only: route the state 6×6 through the engine conversion the
    // parquet / CSV writers use, so provenance (Asserted / ExternalFile),
    // quality and the by-name asymmetry refusal are byte-for-byte identical
    // across the three orbit formats. The resulting joint is marshaled back to
    // the ABI's degree basis, read into the wire form, and its library-owned
    // triangle freed.
    let state = orbit.state.to_empyrean();
    let (_coords, state_cov) =
        coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit covariance: {e}"))?;
    match state_cov {
        Some(unc) => {
            let mut c = crate::joint::orbit_uncertainty_to_c(&unc);
            let wire = unsafe { cov_wire_from_c_joint(&c) };
            unsafe { crate::joint::free_packed_joint(&mut c) };
            Ok(wire)
        }
        // `has_covariance != 0` but the engine produced no uncertainty: the
        // state carried no representable covariance, so carry none rather than
        // fabricate one.
        None => Ok(CovWire::default()),
    }
}

fn orbit_to_row(
    orbit: &EmpyreanOrbit,
    orbit_id: &str,
    object_id: Option<&str>,
) -> Result<OrbitRow, String> {
    let cov = cov_wire_from_orbit(orbit)?;
    Ok(OrbitRow {
        orbit_id: orbit_id.to_string(),
        object_id: object_id.map(str::to_string),
        epoch_mjd_tdb: orbit.state.epoch_mjd_tdb,
        elements: orbit.state.elements,
        representation: rep_int_to_str(orbit.state.representation).to_string(),
        frame: frame_int_to_str(orbit.state.frame).to_string(),
        origin: orbit.state.origin,
        covariance: None,
        cov_schema: cov.schema,
        cov_state_width: cov.state_width,
        cov_layout_present: cov.layout_present,
        cov_layout_considered: cov.layout_considered,
        cov_layout_marginalized: cov.layout_marginalized,
        cov_tri: cov.tri,
        cov_kind: cov.kind,
        cov_quality: cov.quality,
        cov_rank: cov.rank,
        cov_null_axes: cov.null_axes,
        cov_min_eig_corr: cov.min_eig_corr,
        cov_functional: cov.functional,
        cov_provenance: cov.provenance,
        cov_asserted_source: cov.asserted_source,
        a1: orbit.a1,
        a2: orbit.a2,
        a3: orbit.a3,
        ng_alpha: orbit.ng_alpha,
        ng_r0: orbit.ng_r0,
        ng_m: orbit.ng_m,
        ng_n: orbit.ng_n,
        ng_k: orbit.ng_k,
    })
}

/// Reconstruct the orbit's covariance from an [`OrbitRow`]'s packed-joint
/// fields: the library-owned [`EmpyreanPackedJoint`], its derived flat 6×6
/// state block, and the `has_covariance` flag.
///
/// A row carrying the removed flat `covariance` field (a file written before
/// 0.11.0, when the row held the state block alone) is refused BY NAME — there
/// is no migration shim. A present joint (`cov_tri` set) is restored verbatim:
/// the triangle is copied into a library-owned buffer (released with the
/// batch), the tags are the bytes the file carried (never re-derived), and the
/// flat 6×6 is the joint's leading block. A row with no covariance yields the
/// absent joint and no flat block. A declared width that disagrees with the
/// triangle length, or an unknown schema version, is refused by name rather
/// than zero-filled.
#[allow(clippy::type_complexity)]
fn joint_from_row(
    row: &OrbitRow,
) -> Result<(crate::joint::EmpyreanPackedJoint, [[f64; 6]; 6], u8), String> {
    if row.covariance.is_some() {
        return Err(
            "the JSON orbit file carries the removed flat 6×6 `covariance` field; \
             0.11.0 carries the full packed joint as `cov_tri` plus its `cov_*` tags \
             and derives the state 6×6 from the joint's leading block. Re-export the \
             file from parquet or re-run the fit — there is no migration shim for the \
             old flat shape."
                .to_string(),
        );
    }
    let Some(tri) = row.cov_tri.as_ref() else {
        // No joint and (guarded above) no flat field: the orbit has no
        // covariance of any kind.
        return Ok((crate::joint::empty_packed_joint(), [[0.0; 6]; 6], 0));
    };
    if let Some(schema) = row.cov_schema
        && schema != JSON_COV_SCHEMA
    {
        return Err(format!(
            "the JSON orbit covariance declares cov_schema {schema}, which this build \
             does not read (it writes and reads cov_schema {JSON_COV_SCHEMA})"
        ));
    }
    let present = row.cov_layout_present.unwrap_or(0);
    let state_width = row.cov_state_width.unwrap_or(6) as usize;
    let width = state_width + present.count_ones() as usize;
    let expected = width * (width + 1) / 2;
    if width < 6 || tri.len() != expected {
        return Err(format!(
            "the JSON orbit covariance declares a width-{width} joint (state_width \
             {state_width} + {} present axes), which needs a {expected}-entry lower \
             triangle, but `cov_tri` carries {}",
            present.count_ones(),
            tri.len(),
        ));
    }
    // The leading 6×6 is the joint's state block — the flat view, derived, not
    // a second stored covariance.
    let flat = state_block_from_tri(tri);
    let joint = crate::joint::packed_joint_from_json_wire(
        present,
        row.cov_layout_considered.unwrap_or(0),
        row.cov_layout_marginalized.unwrap_or(0),
        width as u8,
        tri.clone(),
        row.cov_kind
            .unwrap_or(crate::propagate::EMPYREAN_COVARIANCE_KIND_LINEAR),
        row.cov_quality
            .unwrap_or(crate::joint::wire_tags::EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE),
        row.cov_rank.unwrap_or(0),
        row.cov_null_axes.unwrap_or(0),
        row.cov_min_eig_corr.unwrap_or(f64::NAN),
        row.cov_functional
            .unwrap_or(crate::joint::wire_tags::EMPYREAN_JOINT_FUNCTIONAL_STATE),
        row.cov_provenance
            .unwrap_or(crate::joint::wire_tags::EMPYREAN_JOINT_PROVENANCE_DERIVED),
        row.cov_asserted_source
            .unwrap_or(crate::joint::wire_tags::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE),
    );
    Ok((joint, flat, 1))
}

fn row_to_orbit(row: &OrbitRow) -> Result<(EmpyreanOrbit, String, Option<String>), String> {
    let representation = rep_str_to_int(&row.representation)?;
    let frame = frame_str_to_int(&row.frame)?;
    // The packed joint is the one covariance home; the flat 6×6 is its derived
    // leading block. An old-shape flat `covariance` field is refused here.
    let (covariance, flat_covariance, has_covariance) = joint_from_row(row)?;
    let state = CoordinateState {
        epoch_mjd_tdb: row.epoch_mjd_tdb,
        elements: row.elements,
        covariance: flat_covariance,
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
        // OrbitRow JSON/parquet schema does not carry photometry yet;
        // round-tripped orbits come back without it.
        phot_system: -1,
        h_mag: f64::NAN,
        slope1: 0.0,
        slope2: 0.0,
        // OrbitRow JSON/parquet schema does not carry continuous-thrust
        // arcs; round-tripped orbits come back gravity/non-grav only. The
        // side arrays are caller-owned on input, so the read path leaves
        // them null/0.
        thrust_arcs: std::ptr::null(),
        n_thrust_arcs: 0,
        dv_corrections: std::ptr::null(),
        n_dv_corrections: 0,
        // OrbitRow JSON/parquet schema does not carry the SRP slot yet (same
        // documented limitation as DT / thrust / photometry above); round-trip
        // restores no SRP. Tracked for a unified lossless-orbit-file follow-up.
        has_srp: 0,
        srp_amrat: 0.0,
        srp_cr: 0.0,
        // The orbit's full covariance as one packed joint — reconstructed
        // from the row's `cov_*` fields (`joint_from_row`), library-owned and
        // released with the batch. 0.11.0 carries the whole joint in JSON, so
        // a round-tripped orbit comes back with the joint it was written with,
        // not an absent one; the flat `state.covariance` above is its derived
        // leading block.
        covariance,
        // This crate's own flat JSON row shape carries no photometry at
        // all (see `phot_system` above), so there is no covariance to
        // carry either. The engine's own parquet / CSV formats DO carry
        // both, and reach the batch through `orbits_to_batch`.
        has_phot_covariance: 0,
        phot_covariance: [[0.0; 3]; 3],
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
    // Each row now reconstructs a library-owned packed-joint triangle, so a
    // failure partway through the batch has live allocations (triangles + id
    // CStrings) to release. `written` is what the unwind walks; returning
    // straight out would leak every joint and string already marshaled plus
    // the three arrays.
    let mut written = 0usize;
    for row in rows.iter() {
        match row_to_orbit(row) {
            Ok((orbit, orbit_id, object_id)) => unsafe {
                orbits_ptr.add(written).write(orbit);
                let id_c = CString::new(orbit_id.as_bytes()).unwrap_or_default();
                orbit_ids_ptr.add(written).write(id_c.into_raw());
                let obj_c = match object_id {
                    Some(s) => CString::new(s.as_bytes()).unwrap_or_default().into_raw(),
                    None => std::ptr::null_mut(),
                };
                object_ids_ptr.add(written).write(obj_c);
                written += 1;
            },
            Err(e) => {
                unsafe {
                    free_partial_batch(
                        orbits_ptr,
                        orbit_ids_ptr,
                        object_ids_ptr,
                        written,
                        orbits_layout,
                        ids_layout,
                    );
                }
                return Err(e);
            }
        }
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
        // 0.11.0: this format carries the FULL packed joint at every width —
        // the same value parquet stores (`cov_tri` plus the `cov_*` tags) — so
        // a joint carrying solved-parameter axes is no longer refused here.
        // `orbit_to_row` serializes the whole joint; a width wider than the
        // state is written, not dropped short. (CSV still refuses a wider
        // joint by name, through the engine's own writer.)
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
        rows.push(orbit_to_row(orbit, orbit_id, object_id.as_deref())?);
    }
    Ok(rows)
}

/// Convert an [`EmpyreanOrbitBatch`] into a villeneuve `Orbits<AU>`,
/// preserving non-grav parameters and non-Cartesian representations.
pub(crate) fn batch_to_orbits(
    batch: &EmpyreanOrbitBatch,
) -> Result<Orbits<AU>, crate::joint::AbiError> {
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
        let (coords, state_cov) =
            coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit {i}: {e}"))?;
        // The write direction is the fifth site that turns a C-ABI orbit
        // into an engine `Orbits`, and it lost the covariance in exactly
        // the same way the four reader paths did. It gets the same
        // treatment: the joint is attached, and the writers then decide
        // what they can represent — parquet carries the carrier, the
        // engine's CSV writer refuses a wider-than-state joint by name, and a
        // thrust-bearing carrier is refused wherever it is offered, because no
        // orbit format can serialize a thrust arc to hang it on. (The JSON
        // path does not come through here at all: it is this crate's own flat
        // row shape, and `batch_to_rows` serializes the full joint directly —
        // parquet and JSON are the two formats that carry it whole.) Refusing
        // at the writer is the point: a carrier dropped here would be written
        // short and read back as a block-diagonal joint.
        crate::joint::push_orbit_with_joint(&mut out, orbit_id, coords, state_cov, orbit)
            .map_err(|e| e.prefixed(&format!("orbit {i}")))?;
        out.set_object_id(i, object_id);
        if let Some(params) = crate::propagate::empyrean_orbit_non_grav_params(orbit) {
            out.set_non_grav_params(i, Some(params));
        }
        // Photometry, including its optional 3×3 covariance. Every other
        // marshal into an engine `Orbits` sets this; this one never did,
        // so `empyrean_orbits_write_parquet` / `_write_csv` wrote NULL
        // photometry for a caller-supplied batch while the CSV writer's
        // own doc claimed to carry it.
        if let Some(ph) = crate::propagate::empyrean_orbit_photometric_params(orbit)
            .map_err(|e| format!("orbit {i}: {e}"))?
        {
            out.set_photometric_params(i, Some(ph));
        }
        if let Some(tp) = crate::propagate::empyrean_orbit_thrust_params(orbit)
            .map_err(|e| format!("orbit {i}: {e}"))?
        {
            out.set_thrust_params(i, Some(tp));
        }
        if let Some(srp) = crate::propagate::empyrean_orbit_srp_params(orbit)
            .map_err(|e| format!("orbit {i}: {e}"))?
        {
            out.set_srp_params(i, Some(srp));
        }
    }
    Ok(out)
}

/// Release a partially-built batch: the carrier arrays and identifier
/// strings of the first `written` rows, then the three arrays.
///
/// The batch's own free function cannot be used here because the batch
/// struct does not exist yet — the rows are still loose in raw arrays,
/// and only the first `written` of them are initialized. Freeing the
/// arrays alone would leak everything they point at.
///
/// # Safety
///
/// The three pointers must be the allocations described by the two
/// layouts, with exactly `written` initialized rows.
unsafe fn free_partial_batch(
    orbits_ptr: *mut EmpyreanOrbit,
    orbit_ids_ptr: *mut *mut c_char,
    object_ids_ptr: *mut *mut c_char,
    written: usize,
    orbits_layout: std::alloc::Layout,
    ids_layout: std::alloc::Layout,
) {
    for k in 0..written {
        unsafe {
            let o = &mut *orbits_ptr.add(k);
            crate::joint::free_packed_joint(&mut o.covariance);
            let id = *orbit_ids_ptr.add(k);
            if !id.is_null() {
                drop(CString::from_raw(id));
            }
            let obj = *object_ids_ptr.add(k);
            if !obj.is_null() {
                drop(CString::from_raw(obj));
            }
        }
    }
    unsafe {
        std::alloc::dealloc(orbits_ptr as *mut u8, orbits_layout);
        std::alloc::dealloc(orbit_ids_ptr as *mut u8, ids_layout);
        std::alloc::dealloc(object_ids_ptr as *mut u8, ids_layout);
    }
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
    // Rows are written one at a time and each may own two heap arrays,
    // so every failure past the first write has live allocations to
    // release. `written` is what the unwind below walks; returning
    // straight out of the loop would leak every carrier already
    // marshaled plus the three arrays.
    let mut written = 0usize;
    let build = |i: usize| -> Result<(EmpyreanOrbit, *mut c_char, *mut c_char), String> {
        // Read the stored coordinate (engine radian basis) and convert its
        // angular elements to the ABI's degree basis. The orbit's full
        // uncertainty is ONE packed joint read off the orbit; the flat
        // `CoordinateState.covariance` is DERIVED from that joint's state
        // block below, never a second stored covariance.
        let coord_ref = orbits
            .coordinates()
            .get(i)
            .ok_or_else(|| format!("orbit {i}: index out of range reading the coordinate"))?;
        let coord = (*coord_ref).into_angular::<empyrean_core::coordinates::Degrees>();
        let uncertainty = orbits.uncertainty(i);
        // Fallible since the engine stopped panicking on an origin with
        // no NAIF id (an MPC site code). Surface it — a fabricated
        // sentinel written into a Parquet/JSON/CSV orbit file would read
        // back as a different body. The flat 6×6 is the joint's state
        // block scaled to degrees when the orbit carries a joint, absent
        // otherwise.
        let cs = coordinates_to_coordinate_state(&coord, uncertainty)
            .map_err(|e| format!("orbit {i}: {e}"))?;
        // The full solved-parameter joint — the single covariance home —
        // marshaled from the orbit's `OrbitUncertainty` (a library-owned
        // triangle, released by `empyrean_orbits_batch_free`), or the
        // absent joint when the orbit carries none.
        let covariance = match uncertainty {
            Some(u) => crate::joint::orbit_uncertainty_to_c(u),
            None => crate::joint::empty_packed_joint(),
        };
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
            // Absent defaults; back-filled from the villeneuve orbit
            // below when it carries photometry. An orbit with none keeps
            // `phot_system = -1` and ephemeris generation downstream
            // emits `mag = NaN`.
            phot_system: -1,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
            // Thrust arcs are a caller-owned input side array; the
            // orbits→batch output path emits null/0 (the burn-prior
            // covariance rides the packed joint's thrust axes).
            thrust_arcs: std::ptr::null(),
            n_thrust_arcs: 0,
            dv_corrections: std::ptr::null(),
            n_dv_corrections: 0,
            // SRP slot carried from the villeneuve orbit below (parity with
            // non-grav / photometry), when present.
            has_srp: 0,
            srp_amrat: 0.0,
            srp_cr: 0.0,
            // The single covariance home — the whole (6+P)-wide joint.
            // Library-owned on this direction (see the batch's own docs)
            // and released by `empyrean_orbits_batch_free`.
            covariance,
            // Carried from the villeneuve orbit below when present. Every
            // engine-side producer of a photometric covariance — the
            // post-OD photometry fit, the SBDB `phys_par` ingest, and
            // the orbit-file readers — reaches the C ABI through this
            // one function, so dropping it here would strip the H/G
            // uncertainty from all three at once.
            has_phot_covariance: 0,
            phot_covariance: [[0.0; 3]; 3],
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
            if let Some(cov) = ph.covariance {
                orbit.has_phot_covariance = 1;
                orbit.phot_covariance = cov;
            }
        }
        if let Some(ng) = orbits.non_grav_params(i) {
            orbit.a1 = ng.a1;
            orbit.a2 = ng.a2;
            orbit.a3 = ng.a3;
            orbit.non_grav_dt = ng.dt.unwrap_or(f64::NAN);
            // The non-grav prior covariance and the DT prior variance ride
            // the packed joint's Marsden / DT axes now, not scalar fields —
            // there is nothing to copy here beyond the model values.
            // NonGravModel is Marsden-only in v1.20.0 — irrefutable.
            let NonGravModel::MarsdenSekanina(g) = &ng.model;
            orbit.ng_alpha = g.alpha;
            orbit.ng_r0 = g.r0;
            orbit.ng_m = g.m;
            orbit.ng_n = g.n;
            orbit.ng_k = g.k;
        }
        if let Some(srp) = orbits.srp_params(i) {
            orbit.has_srp = 1;
            orbit.srp_amrat = srp.amrat;
            orbit.srp_cr = srp.cr;
            // The AMRAT prior variance rides the packed joint's AMRAT axis.
        }
        let id_c =
            CString::new(orbits.orbit_ids()[i].as_str()).unwrap_or_else(|_| CString::default());
        let obj_ptr = match orbits.object_ids()[i].as_ref() {
            Some(s) => CString::new(s.as_str())
                .unwrap_or_else(|_| CString::default())
                .into_raw(),
            None => std::ptr::null_mut(),
        };
        Ok((orbit, id_c.into_raw(), obj_ptr))
    };
    for i in 0..n {
        match build(i) {
            Ok((orbit, id_ptr, obj_ptr)) => unsafe {
                orbits_ptr.add(i).write(orbit);
                orbit_ids_ptr.add(i).write(id_ptr);
                object_ids_ptr.add(i).write(obj_ptr);
                written += 1;
            },
            Err(e) => {
                unsafe {
                    free_partial_batch(
                        orbits_ptr,
                        orbit_ids_ptr,
                        object_ids_ptr,
                        written,
                        orbits_layout,
                        ids_layout,
                    );
                }
                return Err(e);
            }
        }
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
        // The `_with_non_grav` reader, for the same reason the CSV path
        // uses it: the plain reader attaches the `ExtendedCovariance` to
        // every `cov_dim = 9` row but leaves `NonGravParams` unset, so
        // the file's own A1/A2/A3 are dropped AND the Marsden 3×3 the
        // border needs is never back-filled. The row's own g(r)
        // exponents win; the model passed here is only the fallback for
        // a row that carried none, and an all-zero exponent set IS the
        // inverse-square asteroid default.
        let orbits: Orbits<AU> = empyrean_core::io::parquet::read_orbits_with_non_grav(
            p,
            NonGravModel::MarsdenSekanina(GFunction::inverse_square()),
        )
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
            .map_err(|e| crate::joint::AbiError::from(format!("parquet write failed: {e:?}")))
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
///
/// This is **not** the engine's orbit schema — it is this crate's own
/// flat row shape (`OrbitRow`). It carries the state, the full packed
/// joint covariance, and the Marsden coefficients with their g(r)
/// exponents.
///
/// 0.11.0: the covariance is the **full packed joint** at every width —
/// the same value parquet stores, in the same `cov_*` field names (the
/// packed lower triangle `cov_tri` plus the scalar tags). A joint carrying
/// solved-parameter axes is written, not refused; the flat 6×6 is the
/// joint's leading block, derived on read, never stored twice. (CSV still
/// refuses a wider-than-state joint by name — see
/// [`empyrean_orbits_write_csv`].)
///
/// Fields this format still drops, because they predate the joint surface
/// and are not covariance: the non-grav DT value, the SRP slot and the
/// photometric block. Their covariance axes (DT / AMRAT / thrust) ride the
/// packed joint and are carried; their model values do not — round-trip
/// through parquet if you need those.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_write_json(
    path: *const c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    file_in_op(path, batch, |p, b| {
        let rows = batch_to_rows(b)?;
        let f = File::create(p).map_err(|e| format!("create: {e}"))?;
        serde_json::to_writer_pretty(f, &rows)
            .map_err(|e| crate::joint::AbiError::from(format!("json write: {e}")))
    })
}

// ────────────────────────────────────────────────────────────────────
// Orbit I/O — CSV
// ────────────────────────────────────────────────────────────────────

/// Read an orbits CSV file.
///
/// Uses the engine's own CSV reader, so the file is the same schema the
/// parquet path round-trips — covariance included. The
/// `_with_non_grav` reader is the one used: the plain reader drops the
/// Marsden A1/A2/A3 block, and an orbit that loses its non-gravitational
/// parameters on a round trip is silently a different orbit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_read_csv(
    path: *const c_char,
    out: *mut EmpyreanOrbitBatch,
) -> i32 {
    file_op(path, out, |p, o| {
        // The row's own g(r) exponents win; the model passed here is only
        // the fallback for a row that carried none, and an all-zero
        // exponent set IS the inverse-square asteroid default.
        let orbits: Orbits<AU> = empyrean_core::io::read_orbits_csv_with_non_grav(
            p,
            NonGravModel::MarsdenSekanina(GFunction::inverse_square()),
        )
        .map_err(|e| format!("csv read failed: {e:?}"))?;
        *o = orbits_to_batch(&orbits)?;
        Ok(())
    })
}

/// Write an orbit batch to CSV.
///
/// Routed through the engine's writer — the same `Orbits<AU>` the
/// parquet path writes — so CSV carries the full column set (state,
/// covariance as the packed joint, non-grav including `dt`, photometry,
/// SRP) rather than a flattened projection of it. A batch whose joint is
/// wider than the state-only row schema can express is refused by name
/// before the file is created rather than written short.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_orbits_write_csv(
    path: *const c_char,
    batch: *const EmpyreanOrbitBatch,
) -> i32 {
    file_in_op(path, batch, |p, b| {
        let orbits = batch_to_orbits(b)?;
        empyrean_core::io::write_orbits_csv(p, &orbits)
            .map_err(|e| crate::joint::AbiError::from(format!("csv write failed: {e:?}")))
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
    // (the wrapper fills them) but previously omitted by every file writer.
    zenith_angle_deg: f64,
    azimuth_deg: f64,
    hour_angle_deg: f64,
    lunar_elongation_deg: f64,
    position_angle_deg: f64,
    sky_rate_deg_day: f64,
    // The row's covariance-kind wire code, mirroring the events writer's
    // `resolved_kind`: `kind_code(e.cov_kind)` — `-1` for the `0xFF` sentinel,
    // else the `EMPYREAN_COVARIANCE_KIND_*` value (a no-covariance row carries
    // `Linear`, so this is a real kind, never the sentinel, in practice).
    // Appended last so the file schemas grow append-only. The packed joint is
    // NOT carried in the files (the propagation file writers carry none
    // either), so the bare 6×6 angular columns stay the covariance a file row
    // holds.
    cov_kind: i32,
    // The engine's pre-retention Gaussian-mixture accounting for the row's
    // covariance, appended after `cov_kind` (so the schema still grows only at
    // the tail). Absent on every non-mixture row: each count is the `-1`
    // "not carried" sentinel and `mix_weight_delivered` is NaN, never a
    // fabricated `0`. The engine's i64 counts narrow to i32 here (component
    // counts are small), mirroring `cov_kind`'s Int32 file column.
    mix_n_components_total: i32,
    mix_weight_delivered: f64,
    mix_n_failed: i32,
    mix_n_unresolved: i32,
    mix_n_curvature_refused: i32,
    mix_n_sky_linearization_refused: i32,
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
            // The engine's pre-retention mixture accounting rides the row when
            // its covariance was a mixture; absent (every count -1, weight NaN)
            // otherwise — never fabricated zeros.
            let (
                mix_n_components_total,
                mix_weight_delivered,
                mix_n_failed,
                mix_n_unresolved,
                mix_n_curvature_refused,
                mix_n_sky_linearization_refused,
            ) = if e.has_mixture_tally != 0 {
                let t = e.mixture_tally;
                (
                    t.n_components_total as i32,
                    t.weight_delivered,
                    t.n_failed as i32,
                    t.n_unresolved as i32,
                    t.n_curvature_refused as i32,
                    t.n_sky_linearization_refused as i32,
                )
            } else {
                (-1, f64::NAN, -1, -1, -1, -1)
            };
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
                cov_kind: kind_code(e.cov_kind),
                mix_n_components_total,
                mix_weight_delivered,
                mix_n_failed,
                mix_n_unresolved,
                mix_n_curvature_refused,
                mix_n_sky_linearization_refused,
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
// field on the file path. Inapplicable fields are NaN /
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

/// One per-observation residual row, owned and format-independent.
///
/// Mirrors [`EmpyreanObservationResult`] field for field — the whole
/// surface reaches disk, not a projection of it. Field names here are
/// storage; the **wire** names and their order live in exactly one
/// place, [`RESIDUAL_COLUMNS`], which drives all three writers so the
/// formats cannot disagree about what a residual file contains.
#[derive(Debug, Clone)]
struct ResidualRow {
    object_id: String,
    obs_id: String,
    obs_code: String,
    ast_cat: String,
    epoch_mjd_tdb: f64,
    ra_residual_arcsec: f64,
    dec_residual_arcsec: f64,
    chi2: f64,
    dof: i32,
    probability: f64,
    selected: bool,
    residual_cov_ra: f64,
    residual_cov_dec: f64,
    residual_cov_corr: f64,
    rejection_reason: String,
    rejection_criterion: f64,
    rejection_threshold: f64,
    rejection_effective_threshold: f64,
    rejection_information_loss: f64,
    cooks_distance: f64,
    leverage: f64,
    fractional_information: f64,
    along_track_arcsec: f64,
    cross_track_arcsec: f64,
    along_track_error_arcsec: f64,
    cross_track_error_arcsec: f64,
    track_position_angle_deg: f64,
    influence_information_loss: f64,
    along_cross_covariance_arcsec2: f64,
    radar_residual: f64,
    radar_chi2: f64,
    radar_probability: f64,
    radar_variance: f64,
    radar_dof: i32,
    has_radar: bool,
    radar_kind: String,
}

/// The stable name of a rejection code, for the file's attribution
/// taxonomy. An integer on disk would make the column unreadable without
/// the header; the names match the Python `ObservationResults`
/// `rejection_reason` column exactly.
fn rejection_reason_str(code: i32) -> &'static str {
    match code {
        crate::od::EMPYREAN_REJECTION_ACCEPTED => "accepted",
        crate::od::EMPYREAN_REJECTION_CHI_SQUARED => "chi_squared",
        crate::od::EMPYREAN_REJECTION_SIGMA_CLIP => "sigma_clip",
        crate::od::EMPYREAN_REJECTION_COOKS_DISTANCE => "cooks_distance",
        crate::od::EMPYREAN_REJECTION_ADAPTIVE => "adaptive",
        crate::od::EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY => "unsupported_observatory",
        crate::od::EMPYREAN_REJECTION_CMC2003 => "cmc2003",
        crate::od::EMPYREAN_REJECTION_RADAR_UNSUPPORTED => "radar_observations_unsupported",
        crate::od::EMPYREAN_REJECTION_OCCULTATION_UNSUPPORTED => {
            "occultation_observations_unsupported"
        }
        crate::od::EMPYREAN_REJECTION_OUTSIDE_ARC => "outside_arc",
        crate::od::EMPYREAN_REJECTION_NON_FINITE_CHI2 => "non_finite_chi2",
        crate::od::EMPYREAN_REJECTION_MISSING_JACOBIAN => "missing_jacobian",
        crate::od::EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING => "spacecraft_kernel_missing",
        crate::od::EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED => {
            "observer_construction_failed"
        }
        crate::od::EMPYREAN_REJECTION_NEVER_ABSORBED => "never_absorbed",
        crate::od::EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED => {
            "per_observation_site_required"
        }
        _ => "not_evaluated",
    }
}

fn residuals_to_rows(observations: &[EmpyreanObservationResult]) -> Vec<ResidualRow> {
    observations
        .iter()
        .map(|o| ResidualRow {
            object_id: cstr_or_empty(o.object_id),
            obs_id: cstr_or_empty(o.obs_id),
            // Fixed 4-byte NUL-padded field, not a pointer.
            obs_code: String::from_utf8_lossy(&o.obs_code)
                .trim_end_matches('\0')
                .to_string(),
            ast_cat: cstr_or_empty(o.ast_cat),
            epoch_mjd_tdb: o.epoch_mjd_tdb,
            ra_residual_arcsec: o.ra_residual_arcsec,
            dec_residual_arcsec: o.dec_residual_arcsec,
            chi2: o.chi2,
            dof: o.dof as i32,
            probability: o.probability,
            selected: o.selected != 0,
            residual_cov_ra: o.residual_cov_ra,
            residual_cov_dec: o.residual_cov_dec,
            residual_cov_corr: o.residual_cov_corr,
            rejection_reason: rejection_reason_str(o.rejection_reason).to_string(),
            rejection_criterion: o.rejection_criterion,
            rejection_threshold: o.rejection_threshold,
            rejection_effective_threshold: o.rejection_effective_threshold,
            rejection_information_loss: o.rejection_information_loss,
            cooks_distance: o.cooks_distance,
            leverage: o.leverage,
            fractional_information: o.fractional_information,
            along_track_arcsec: o.along_track_arcsec,
            cross_track_arcsec: o.cross_track_arcsec,
            along_track_error_arcsec: o.along_track_error_arcsec,
            cross_track_error_arcsec: o.cross_track_error_arcsec,
            track_position_angle_deg: o.track_position_angle_deg,
            influence_information_loss: o.influence_information_loss,
            along_cross_covariance_arcsec2: o.along_cross_covariance_arcsec2,
            radar_residual: o.radar_residual,
            radar_chi2: o.radar_chi2,
            radar_probability: o.radar_probability,
            radar_variance: o.radar_variance,
            radar_dof: o.radar_dof as i32,
            has_radar: o.has_radar != 0,
            // Empty on optical rows: the discriminator is `has_radar`,
            // and naming a kind for a row that has none would be a lie.
            radar_kind: if o.has_radar == 0 {
                String::new()
            } else if o.radar_kind == crate::od::EMPYREAN_RADAR_KIND_DOPPLER {
                "doppler".to_string()
            } else {
                "delay".to_string()
            },
        })
        .collect()
}

/// Owned `String` from an owned C string pointer; empty for null.
fn cstr_or_empty(p: *mut c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// One cell of a residual row, in the only three storage classes the
/// residual surface uses.
enum Cell<'a> {
    F64(f64),
    I32(i32),
    Bool(bool),
    Str(&'a str),
}

/// A residual file column: its wire name, its parquet type, and how to
/// read it off a row.
struct ResidualColumn {
    name: &'static str,
    data_type: DataType,
    get: fn(&ResidualRow) -> Cell<'_>,
}

/// **The** residual file schema. Parquet, CSV, and JSON are all emitted
/// from this list, so "the three formats carry the same columns" is a
/// property of the code rather than something a test has to police —
/// the test then only has to confirm it.
///
/// Every field [`EmpyreanObservationResult`] carries appears here.
/// Non-computable numbers are NaN (JSON writes `null`, its established
/// encoding for the same thing, because JSON has no NaN literal).
static RESIDUAL_COLUMNS: &[ResidualColumn] = &[
    ResidualColumn {
        name: "object_id",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.object_id),
    },
    ResidualColumn {
        name: "obs_id",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.obs_id),
    },
    ResidualColumn {
        name: "obs_code",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.obs_code),
    },
    ResidualColumn {
        name: "ast_cat",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.ast_cat),
    },
    ResidualColumn {
        name: "epoch_mjd_tdb",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.epoch_mjd_tdb),
    },
    ResidualColumn {
        name: "ra_residual_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.ra_residual_arcsec),
    },
    ResidualColumn {
        name: "dec_residual_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.dec_residual_arcsec),
    },
    ResidualColumn {
        name: "chi2",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.chi2),
    },
    ResidualColumn {
        name: "dof",
        data_type: DataType::Int32,
        get: |r| Cell::I32(r.dof),
    },
    ResidualColumn {
        name: "probability",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.probability),
    },
    ResidualColumn {
        name: "selected",
        data_type: DataType::Boolean,
        get: |r| Cell::Bool(r.selected),
    },
    ResidualColumn {
        name: "residual_cov_ra",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.residual_cov_ra),
    },
    ResidualColumn {
        name: "residual_cov_dec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.residual_cov_dec),
    },
    ResidualColumn {
        name: "residual_cov_corr",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.residual_cov_corr),
    },
    ResidualColumn {
        name: "rejection_reason",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.rejection_reason),
    },
    ResidualColumn {
        name: "rejection_criterion",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.rejection_criterion),
    },
    ResidualColumn {
        name: "rejection_threshold",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.rejection_threshold),
    },
    ResidualColumn {
        name: "rejection_effective_threshold",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.rejection_effective_threshold),
    },
    ResidualColumn {
        name: "rejection_information_loss",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.rejection_information_loss),
    },
    ResidualColumn {
        name: "cooks_distance",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.cooks_distance),
    },
    ResidualColumn {
        name: "leverage",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.leverage),
    },
    ResidualColumn {
        name: "fractional_information",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.fractional_information),
    },
    ResidualColumn {
        name: "along_track_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.along_track_arcsec),
    },
    ResidualColumn {
        name: "cross_track_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.cross_track_arcsec),
    },
    ResidualColumn {
        name: "along_track_error_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.along_track_error_arcsec),
    },
    ResidualColumn {
        name: "cross_track_error_arcsec",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.cross_track_error_arcsec),
    },
    ResidualColumn {
        name: "track_position_angle_deg",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.track_position_angle_deg),
    },
    ResidualColumn {
        name: "influence_information_loss",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.influence_information_loss),
    },
    ResidualColumn {
        name: "along_cross_covariance_arcsec2",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.along_cross_covariance_arcsec2),
    },
    ResidualColumn {
        name: "radar_residual",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.radar_residual),
    },
    ResidualColumn {
        name: "radar_chi2",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.radar_chi2),
    },
    ResidualColumn {
        name: "radar_probability",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.radar_probability),
    },
    ResidualColumn {
        name: "radar_variance",
        data_type: DataType::Float64,
        get: |r| Cell::F64(r.radar_variance),
    },
    ResidualColumn {
        name: "radar_dof",
        data_type: DataType::Int32,
        get: |r| Cell::I32(r.radar_dof),
    },
    ResidualColumn {
        name: "has_radar",
        data_type: DataType::Boolean,
        get: |r| Cell::Bool(r.has_radar),
    },
    ResidualColumn {
        name: "radar_kind",
        data_type: DataType::Utf8,
        get: |r| Cell::Str(&r.radar_kind),
    },
];

fn write_residual_rows_parquet(path: &Path, rows: &[ResidualRow]) -> Result<(), String> {
    let fields: Vec<ParquetField> = RESIDUAL_COLUMNS
        .iter()
        .map(|c| ParquetField {
            name: c.name,
            data_type: c.data_type.clone(),
            nullable: false,
        })
        .collect();
    write_rows_parquet_generic(path, rows, &fields, |row, builders| {
        for (i, col) in RESIDUAL_COLUMNS.iter().enumerate() {
            match ((col.get)(row), &mut builders[i]) {
                (Cell::F64(v), Builder::F64(b)) => b.append_value(v),
                (Cell::I32(v), Builder::I32(b)) => b.append_value(v),
                (Cell::Bool(v), Builder::Bool(b)) => b.append_value(v),
                (Cell::Str(v), Builder::Str(b)) => b.append_value(v),
                _ => return Err(format!("residual column {} type mismatch", col.name)),
            }
        }
        Ok(())
    })
}

fn write_residual_rows_csv(path: &Path, rows: &[ResidualRow]) -> Result<(), String> {
    let mut wtr = csv::Writer::from_path(path).map_err(|e| format!("csv create: {e}"))?;
    wtr.write_record(RESIDUAL_COLUMNS.iter().map(|c| c.name))
        .map_err(|e| format!("csv header: {e}"))?;
    for row in rows {
        // CSV keeps the literal `NaN` — it has no null, and an empty
        // cell would be indistinguishable from a missing column.
        let record: Vec<String> = RESIDUAL_COLUMNS
            .iter()
            .map(|c| match (c.get)(row) {
                Cell::F64(v) => v.to_string(),
                Cell::I32(v) => v.to_string(),
                Cell::Bool(v) => v.to_string(),
                Cell::Str(v) => v.to_string(),
            })
            .collect();
        wtr.write_record(&record)
            .map_err(|e| format!("csv write: {e}"))?;
    }
    wtr.flush().map_err(|e| format!("csv flush: {e}"))
}

fn write_residual_rows_json(path: &Path, rows: &[ResidualRow]) -> Result<(), String> {
    let values: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            let mut map = serde_json::Map::with_capacity(RESIDUAL_COLUMNS.len());
            for col in RESIDUAL_COLUMNS {
                let v = match (col.get)(row) {
                    // JSON has no NaN literal; `null` is the established
                    // encoding for the same "not computable" state.
                    Cell::F64(v) => serde_json::Number::from_f64(v)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null),
                    Cell::I32(v) => serde_json::Value::Number(v.into()),
                    Cell::Bool(v) => serde_json::Value::Bool(v),
                    Cell::Str(v) => serde_json::Value::String(v.to_string()),
                };
                map.insert(col.name.to_string(), v);
            }
            serde_json::Value::Object(map)
        })
        .collect();
    let f = File::create(path).map_err(|e| format!("create: {e}"))?;
    serde_json::to_writer_pretty(f, &values).map_err(|e| format!("json write: {e}"))
}

/// Write OD residuals to parquet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_residuals_write_parquet(
    path: *const c_char,
    obs_ptr: *const EmpyreanObservationResult,
    num_obs: usize,
) -> i32 {
    array_in_op(path, obs_ptr, num_obs, |p, slice| {
        write_residual_rows_parquet(p, &residuals_to_rows(slice))
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
        write_residual_rows_json(p, &residuals_to_rows(slice))
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
        write_residual_rows_csv(p, &residuals_to_rows(slice))
    })
}

// ────────────────────────────────────────────────────────────────────
// Fit summary I/O — write only
// ────────────────────────────────────────────────────────────────────

/// One row of the per-object fit summary: what a batch orbit
/// determination did with **one** input object, delivered or not.
///
/// This is the row shape of the `fit_summary` table every distribution
/// channel emits, so the file the CLI writes and the table the Python
/// API returns describe a fit with the same column names.
///
/// A failed object still gets a row — that is the point of the table.
/// Its numeric columns are NaN (never 0.0, which would read as a
/// measurement), its `_ok` booleans are false, and `error` carries the
/// reason. `status` is `"delivered"` or `"failed"`.
///
/// Both string fields are borrowed for the duration of the call: the
/// writer copies what it needs and frees nothing.
#[repr(C)]
pub struct EmpyreanFitSummary {
    /// ADES object identifier. Never null.
    pub object_id: *const c_char,
    /// `"delivered"` or `"failed"`. Never null.
    pub status: *const c_char,
    /// 1 when the differential correction reached its stopping
    /// criterion. 0 on a failed object.
    pub converged: u8,
    /// DC iterations used. 0 on a failed object.
    pub iterations: u32,
    /// Observations this object contributed.
    pub n_obs: usize,
    /// Observations the fit retained.
    pub n_selected: usize,
    pub rms_ra_arcsec: f64,
    pub rms_dec_arcsec: f64,
    pub reduced_chi2: f64,
    pub fit_acceptable: u8,
    pub extrapolation_acceptable: u8,
    pub selection_fraction_ok: u8,
    pub selection_fraction: f64,
    pub selection_fraction_threshold: f64,
    pub selected_arc_coverage_ok: u8,
    pub selected_arc_days: f64,
    pub selected_arc_fraction: f64,
    pub selected_arc_fraction_threshold: f64,
    pub trailing_gap_ok: u8,
    pub trailing_gap_days: f64,
    pub trailing_gap_threshold_days: f64,
    pub fractional_sigma_a_ok: u8,
    pub fractional_sigma_a: f64,
    pub fractional_sigma_a_threshold: f64,
    /// Width of the solved-parameter set (6 for a state-only fit). 0 on
    /// a failed object.
    pub solve_for_width: u32,
    /// Failure message. Null on a delivered object.
    pub error: *const c_char,
}

#[derive(Debug, Serialize, Deserialize)]
struct FitSummaryRow {
    object_id: String,
    status: String,
    converged: bool,
    iterations: i32,
    n_obs: i32,
    n_selected: i32,
    rms_ra_arcsec: f64,
    rms_dec_arcsec: f64,
    reduced_chi2: f64,
    fit_acceptable: bool,
    extrapolation_acceptable: bool,
    selection_fraction_ok: bool,
    selection_fraction: f64,
    selection_fraction_threshold: f64,
    selected_arc_coverage_ok: bool,
    selected_arc_days: f64,
    selected_arc_fraction: f64,
    selected_arc_fraction_threshold: f64,
    trailing_gap_ok: bool,
    trailing_gap_days: f64,
    trailing_gap_threshold_days: f64,
    fractional_sigma_a_ok: bool,
    fractional_sigma_a: f64,
    fractional_sigma_a_threshold: f64,
    solve_for_width: i32,
    error: String,
}

fn fit_summaries_to_rows(summaries: &[EmpyreanFitSummary]) -> Vec<FitSummaryRow> {
    summaries
        .iter()
        .map(|s| FitSummaryRow {
            object_id: cstr_const_or_empty(s.object_id),
            status: cstr_const_or_empty(s.status),
            converged: s.converged != 0,
            iterations: s.iterations as i32,
            n_obs: s.n_obs as i32,
            n_selected: s.n_selected as i32,
            rms_ra_arcsec: s.rms_ra_arcsec,
            rms_dec_arcsec: s.rms_dec_arcsec,
            reduced_chi2: s.reduced_chi2,
            fit_acceptable: s.fit_acceptable != 0,
            extrapolation_acceptable: s.extrapolation_acceptable != 0,
            selection_fraction_ok: s.selection_fraction_ok != 0,
            selection_fraction: s.selection_fraction,
            selection_fraction_threshold: s.selection_fraction_threshold,
            selected_arc_coverage_ok: s.selected_arc_coverage_ok != 0,
            selected_arc_days: s.selected_arc_days,
            selected_arc_fraction: s.selected_arc_fraction,
            selected_arc_fraction_threshold: s.selected_arc_fraction_threshold,
            trailing_gap_ok: s.trailing_gap_ok != 0,
            trailing_gap_days: s.trailing_gap_days,
            trailing_gap_threshold_days: s.trailing_gap_threshold_days,
            fractional_sigma_a_ok: s.fractional_sigma_a_ok != 0,
            fractional_sigma_a: s.fractional_sigma_a,
            fractional_sigma_a_threshold: s.fractional_sigma_a_threshold,
            solve_for_width: s.solve_for_width as i32,
            error: cstr_const_or_empty(s.error),
        })
        .collect()
}

/// Owned `String` from a borrowed C string pointer; empty for null.
fn cstr_const_or_empty(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// Write the per-object fit summary to parquet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_fit_summary_write_parquet(
    path: *const c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    array_in_op(path, summaries_ptr, num_summaries, |p, slice| {
        let rows = fit_summaries_to_rows(slice);
        write_rows_parquet_generic(p, &rows, &FIT_SUMMARY_PARQUET_FIELDS, |row, builders| {
            fit_summary_append(row, builders)
        })
    })
}

/// Write the per-object fit summary to JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_fit_summary_write_json(
    path: *const c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    array_in_op(path, summaries_ptr, num_summaries, |p, slice| {
        let rows = fit_summaries_to_rows(slice);
        write_json(p, &rows)
    })
}

/// Write the per-object fit summary to CSV.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_fit_summary_write_csv(
    path: *const c_char,
    summaries_ptr: *const EmpyreanFitSummary,
    num_summaries: usize,
) -> i32 {
    array_in_op(path, summaries_ptr, num_summaries, |p, slice| {
        let rows = fit_summaries_to_rows(slice);
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

const EPHEMERIS_PARQUET_FIELDS: [ParquetField; 28] = [
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
    // Appended last (the 22nd field) so a reader of an older 21-column file
    // still parses — the schema grows only at the tail, every prior column
    // keeps its name, type and position.
    ParquetField {
        name: "cov_kind",
        data_type: DataType::Int32,
        nullable: false,
    },
    // The six mixture tallies, appended after cov_kind so the schema still
    // grows only at the tail; every prior column keeps its name, type and
    // position. Absent on a non-mixture row (counts -1, weight NaN).
    ParquetField {
        name: "mix_n_components_total",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "mix_weight_delivered",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "mix_n_failed",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "mix_n_unresolved",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "mix_n_curvature_refused",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "mix_n_sky_linearization_refused",
        data_type: DataType::Int32,
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
    if let Builder::I32(i) = &mut b[21] {
        i.append_value(row.cov_kind);
    }
    if let Builder::I32(i) = &mut b[22] {
        i.append_value(row.mix_n_components_total);
    }
    if let Builder::F64(f) = &mut b[23] {
        f.append_value(row.mix_weight_delivered);
    }
    if let Builder::I32(i) = &mut b[24] {
        i.append_value(row.mix_n_failed);
    }
    if let Builder::I32(i) = &mut b[25] {
        i.append_value(row.mix_n_unresolved);
    }
    if let Builder::I32(i) = &mut b[26] {
        i.append_value(row.mix_n_curvature_refused);
    }
    if let Builder::I32(i) = &mut b[27] {
        i.append_value(row.mix_n_sky_linearization_refused);
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

const FIT_SUMMARY_PARQUET_FIELDS: [ParquetField; 26] = [
    ParquetField {
        name: "object_id",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "status",
        data_type: DataType::Utf8,
        nullable: false,
    },
    ParquetField {
        name: "converged",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "iterations",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "n_obs",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "n_selected",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "rms_ra_arcsec",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "rms_dec_arcsec",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "reduced_chi2",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "fit_acceptable",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "extrapolation_acceptable",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "selection_fraction_ok",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "selection_fraction",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "selection_fraction_threshold",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "selected_arc_coverage_ok",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "selected_arc_days",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "selected_arc_fraction",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "selected_arc_fraction_threshold",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "trailing_gap_ok",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "trailing_gap_days",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "trailing_gap_threshold_days",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "fractional_sigma_a_ok",
        data_type: DataType::Boolean,
        nullable: false,
    },
    ParquetField {
        name: "fractional_sigma_a",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "fractional_sigma_a_threshold",
        data_type: DataType::Float64,
        nullable: false,
    },
    ParquetField {
        name: "solve_for_width",
        data_type: DataType::Int32,
        nullable: false,
    },
    ParquetField {
        name: "error",
        data_type: DataType::Utf8,
        nullable: false,
    },
];

fn fit_summary_append(row: &FitSummaryRow, b: &mut [Builder]) -> Result<(), String> {
    macro_rules! str_at {
        ($idx:expr, $val:expr) => {
            if let Builder::Str(s) = &mut b[$idx] {
                s.append_value($val);
            }
        };
    }
    macro_rules! bool_at {
        ($idx:expr, $val:expr) => {
            if let Builder::Bool(x) = &mut b[$idx] {
                x.append_value($val);
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
    macro_rules! f64_at {
        ($idx:expr, $val:expr) => {
            if let Builder::F64(f) = &mut b[$idx] {
                f.append_value($val);
            }
        };
    }
    str_at!(0, &row.object_id);
    str_at!(1, &row.status);
    bool_at!(2, row.converged);
    i32_at!(3, row.iterations);
    i32_at!(4, row.n_obs);
    i32_at!(5, row.n_selected);
    f64_at!(6, row.rms_ra_arcsec);
    f64_at!(7, row.rms_dec_arcsec);
    f64_at!(8, row.reduced_chi2);
    bool_at!(9, row.fit_acceptable);
    bool_at!(10, row.extrapolation_acceptable);
    bool_at!(11, row.selection_fraction_ok);
    f64_at!(12, row.selection_fraction);
    f64_at!(13, row.selection_fraction_threshold);
    bool_at!(14, row.selected_arc_coverage_ok);
    f64_at!(15, row.selected_arc_days);
    f64_at!(16, row.selected_arc_fraction);
    f64_at!(17, row.selected_arc_fraction_threshold);
    bool_at!(18, row.trailing_gap_ok);
    f64_at!(19, row.trailing_gap_days);
    f64_at!(20, row.trailing_gap_threshold_days);
    bool_at!(21, row.fractional_sigma_a_ok);
    f64_at!(22, row.fractional_sigma_a);
    f64_at!(23, row.fractional_sigma_a_threshold);
    i32_at!(24, row.solve_for_width);
    str_at!(25, &row.error);
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
    F: FnOnce(&Path, &EmpyreanOrbitBatch) -> Result<(), crate::joint::AbiError>,
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
        // A malformed packed-joint wire row at this file-write boundary is the
        // SAME refusal as at a direct push, so it returns the same per-variant
        // code (−40…−49) carried on the `AbiError`; any other op error keeps
        // the writer's own default (−2).
        match op(&path, unsafe { &*batch }) {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&e.message);
                e.code.unwrap_or(-2)
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

#[cfg(test)]
mod residual_writer_tests {
    use super::*;
    use crate::od::EmpyreanObservationResult;

    /// The 4-byte obs-code field is read back in full: a 4-character code
    /// (no terminator) returns all four characters; a shorter code is read
    /// up to its NUL. This is the read side of the query / ephemeris-entry
    /// carrier. (Narrowing the reader to a 3-byte prefix reddens this.)
    #[test]
    fn obs_code_bytes_to_string_reads_the_full_four_byte_field() {
        assert_eq!(obs_code_bytes_to_string(b"W68a"), "W68a");
        assert_eq!(obs_code_bytes_to_string(b"703\0"), "703");
        assert_eq!(obs_code_bytes_to_string(b"W68\0"), "W68");
    }

    /// Every field the residual surface carries, by wire name. This list
    /// is the contract: it is written out here independently of
    /// [`RESIDUAL_COLUMNS`], so dropping a column from the schema fails
    /// the test rather than quietly shortening the file.
    const EXPECTED_COLUMNS: [&str; 36] = [
        "object_id",
        "obs_id",
        "obs_code",
        "ast_cat",
        "epoch_mjd_tdb",
        "ra_residual_arcsec",
        "dec_residual_arcsec",
        "chi2",
        "dof",
        "probability",
        "selected",
        "residual_cov_ra",
        "residual_cov_dec",
        "residual_cov_corr",
        "rejection_reason",
        "rejection_criterion",
        "rejection_threshold",
        "rejection_effective_threshold",
        "rejection_information_loss",
        "cooks_distance",
        "leverage",
        "fractional_information",
        "along_track_arcsec",
        "cross_track_arcsec",
        "along_track_error_arcsec",
        "cross_track_error_arcsec",
        "track_position_angle_deg",
        "influence_information_loss",
        "along_cross_covariance_arcsec2",
        "radar_residual",
        "radar_chi2",
        "radar_probability",
        "radar_variance",
        "radar_dof",
        "has_radar",
        "radar_kind",
    ];

    pub(super) fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "empyrean-residual-writer-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&d).expect("create temp dir");
        d
    }

    /// One optical row with a rejection decision and a NaN in a field
    /// that is genuinely not computable, plus one radar row.
    fn sample_rows() -> Vec<EmpyreanObservationResult> {
        let mut optical: EmpyreanObservationResult = unsafe { std::mem::zeroed() };
        optical.obs_id = crate::od::alloc_cstring_for_test("obs-1");
        optical.object_id = crate::od::alloc_cstring_for_test("2024 YR4");
        optical.obs_code = *b"703\0";
        optical.ast_cat = crate::od::alloc_cstring_for_test("Gaia3");
        optical.epoch_mjd_tdb = 60320.5;
        optical.ra_residual_arcsec = 0.31;
        optical.dec_residual_arcsec = -0.12;
        optical.chi2 = 1.4;
        optical.dof = 2;
        optical.probability = 0.49;
        optical.selected = 1;
        optical.rejection_reason = crate::od::EMPYREAN_REJECTION_CHI_SQUARED;
        optical.rejection_criterion = 9.1;
        optical.rejection_threshold = 8.0;
        // No AT/CT decomposition available for this row.
        optical.along_track_arcsec = f64::NAN;
        optical.cross_track_arcsec = f64::NAN;
        optical.radar_residual = f64::NAN;
        optical.radar_chi2 = f64::NAN;
        optical.radar_probability = f64::NAN;
        optical.radar_variance = f64::NAN;

        let mut radar: EmpyreanObservationResult = unsafe { std::mem::zeroed() };
        radar.obs_id = crate::od::alloc_cstring_for_test("obs-2");
        radar.object_id = crate::od::alloc_cstring_for_test("2024 YR4");
        radar.obs_code = *b"251\0";
        radar.epoch_mjd_tdb = 60321.0;
        radar.ra_residual_arcsec = f64::NAN;
        radar.dec_residual_arcsec = f64::NAN;
        radar.chi2 = 0.8;
        radar.has_radar = 1;
        radar.radar_kind = crate::od::EMPYREAN_RADAR_KIND_DOPPLER;
        radar.radar_residual = -0.4;
        radar.radar_dof = 1;
        radar.rejection_reason = crate::od::EMPYREAN_REJECTION_ACCEPTED;

        vec![optical, radar]
    }

    fn write_all(dir: &std::path::Path, rows: &[EmpyreanObservationResult]) {
        write_residual_rows_parquet(&dir.join("r.parquet"), &residuals_to_rows(rows))
            .expect("parquet");
        write_residual_rows_csv(&dir.join("r.csv"), &residuals_to_rows(rows)).expect("csv");
        write_residual_rows_json(&dir.join("r.json"), &residuals_to_rows(rows)).expect("json");
    }

    fn parquet_columns(path: &std::path::Path) -> Vec<String> {
        let file = File::open(path).expect("open parquet");
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("parquet reader");
        reader
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    fn csv_columns(path: &std::path::Path) -> Vec<String> {
        let text = std::fs::read_to_string(path).expect("read csv");
        text.lines()
            .next()
            .expect("header")
            .split(',')
            .map(|s| s.to_string())
            .collect()
    }

    fn json_rows(path: &std::path::Path) -> Vec<serde_json::Map<String, serde_json::Value>> {
        let text = std::fs::read_to_string(path).expect("read json");
        let v: Vec<serde_json::Value> = serde_json::from_str(&text).expect("parse json");
        v.into_iter()
            .map(|x| x.as_object().expect("object row").clone())
            .collect()
    }

    /// The whole in-memory residual surface reaches disk. A projection
    /// is what this replaced: five columns out of thirty-six.
    #[test]
    fn every_residual_field_reaches_disk() {
        let dir = tmp_dir("all-fields");
        let rows = sample_rows();
        write_all(&dir, &rows);

        assert_eq!(
            csv_columns(&dir.join("r.csv")),
            EXPECTED_COLUMNS.to_vec(),
            "the CSV schema is the full residual surface, in order"
        );
        // The schema table and the expectation list agree, so neither can
        // drift alone.
        let from_table: Vec<&str> = RESIDUAL_COLUMNS.iter().map(|c| c.name).collect();
        assert_eq!(from_table, EXPECTED_COLUMNS.to_vec());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Parquet, CSV, and JSON describe the same rows with the same
    /// columns. Format-specific value encodings are allowed; the column
    /// set is not.
    #[test]
    fn all_three_formats_emit_the_same_columns() {
        let dir = tmp_dir("parity");
        let rows = sample_rows();
        write_all(&dir, &rows);

        let pq = parquet_columns(&dir.join("r.parquet"));
        let csv = csv_columns(&dir.join("r.csv"));
        let json = json_rows(&dir.join("r.json"));

        assert_eq!(pq, csv, "parquet and CSV must carry the same columns");
        assert_eq!(json.len(), rows.len(), "one JSON object per row");
        for row in &json {
            let mut names: Vec<&str> = row.keys().map(|s| s.as_str()).collect();
            names.sort_unstable();
            let mut expected: Vec<&str> = csv.iter().map(|s| s.as_str()).collect();
            expected.sort_unstable();
            assert_eq!(names, expected, "JSON must carry the same columns");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A non-computable number is NaN in CSV (which has no null) and
    /// `null` in JSON (which has no NaN). Both mean "not computable" —
    /// neither is ever written as `0.0`.
    #[test]
    fn non_computable_numbers_encode_per_format_never_as_zero() {
        let dir = tmp_dir("nan");
        let rows = sample_rows();
        write_all(&dir, &rows);

        let csv_text = std::fs::read_to_string(dir.join("r.csv")).expect("read csv");
        let header = csv_columns(&dir.join("r.csv"));
        let at_idx = header
            .iter()
            .position(|c| c == "along_track_arcsec")
            .expect("column present");
        let first_data = csv_text.lines().nth(1).expect("a data row");
        assert_eq!(
            first_data.split(',').nth(at_idx),
            Some("NaN"),
            "CSV keeps the literal NaN\n{csv_text}"
        );

        let json = json_rows(&dir.join("r.json"));
        assert!(
            json[0]["along_track_arcsec"].is_null(),
            "JSON encodes NaN as null: {}",
            json[0]["along_track_arcsec"]
        );
        assert!(!json[0]["along_track_arcsec"].is_number());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The rejection code is written as its name, so the attribution
    /// taxonomy is readable without a lookup table, and the radar
    /// discriminator never labels an optical row.
    #[test]
    fn rejection_reason_and_radar_kind_are_written_as_names() {
        let dir = tmp_dir("names");
        let rows = sample_rows();
        write_all(&dir, &rows);

        let json = json_rows(&dir.join("r.json"));
        assert_eq!(json[0]["rejection_reason"], "chi_squared");
        assert_eq!(json[1]["rejection_reason"], "accepted");
        // Optical row: no radar, so no kind is claimed.
        assert_eq!(json[0]["has_radar"], false);
        assert_eq!(json[0]["radar_kind"], "");
        assert_eq!(json[1]["has_radar"], true);
        assert_eq!(json[1]["radar_kind"], "doppler");
        // The grouping key survives to every row.
        assert_eq!(json[0]["object_id"], "2024 YR4");
        assert_eq!(json[1]["object_id"], "2024 YR4");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A one-orbit batch with a populated covariance, for the orbit
    /// writers.
    fn sample_orbit_batch() -> EmpyreanOrbitBatch {
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 60320.0,
            elements: [1.1, 0.2, 0.03, -0.004, 0.017, 0.0006],
            covariance: {
                let mut c = [[0.0f64; 6]; 6];
                for (i, row) in c.iter_mut().enumerate() {
                    row[i] = 1.0e-9 * (i as f64 + 1.0);
                }
                c
            },
            has_covariance: 1,
            representation: crate::od::EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 0,
            origin: 10,
        };
        orbit.non_grav_dt = f64::NAN;
        orbit.h_mag = f64::NAN;
        orbit.slope1 = f64::NAN;
        orbit.slope2 = f64::NAN;

        let id = crate::od::alloc_cstring_for_test("test-orbit");
        let orbits = Box::into_raw(Box::new(orbit));
        let ids = Box::into_raw(Box::new(id));
        let object_ids = Box::into_raw(Box::new(std::ptr::null_mut::<c_char>()));
        EmpyreanOrbitBatch {
            orbits,
            orbit_ids: ids,
            object_ids,
            num_orbits: 1,
        }
    }

    /// Both orbit writers carry the state covariance — neither is the
    /// covariance-free projection CSV once was, when `--format csv`
    /// silently discarded the uncertainty the whole engine exists to
    /// propagate.
    ///
    /// The two schemas are the engine's own and differ by design in
    /// 0.11.0: parquet carries the packed joint wire (`cov_tri`, the
    /// layout bitsets, and the full second-order / mixture / Monte-Carlo
    /// metadata), while CSV carries the explicit lower-triangle cells
    /// (`covt_00`..`covt_55`) with the core tags. So this pins that each
    /// FORMAT carries a real covariance, not a column-set parity the
    /// engine never promised. (A CSV write of a joint carrying
    /// solved-parameter axes is refused upstream by name — the
    /// packed-joint round-trip contract tests exercise that.)
    #[test]
    fn orbit_csv_and_parquet_both_carry_the_covariance() {
        let dir = tmp_dir("orbit-parity");
        let batch = sample_orbit_batch();
        let pq_path = dir.join("o.parquet");
        let csv_path = dir.join("o.csv");
        let c_pq = CString::new(pq_path.display().to_string()).unwrap();
        let c_csv = CString::new(csv_path.display().to_string()).unwrap();

        let rc_pq = unsafe { empyrean_orbits_write_parquet(c_pq.as_ptr(), &batch) };
        let rc_csv = unsafe { empyrean_orbits_write_csv(c_csv.as_ptr(), &batch) };
        assert_eq!(rc_pq, 0, "parquet write");
        assert_eq!(rc_csv, 0, "csv write");

        let pq_cols = parquet_columns(&pq_path);
        let csv_cols = csv_columns(&csv_path);

        // Each format carries a real covariance. Parquet carries the packed
        // joint triangle `cov_tri`; CSV carries the explicit lower-triangle
        // cells `covt_00`..`covt_55`. The column SETS legitimately differ
        // (parquet additionally carries the layout bitsets and the
        // second-order / mixture / Monte-Carlo metadata CSV omits), so this
        // asserts presence of the covariance in each, not set equality.
        assert!(
            pq_cols.iter().any(|c| c == "cov_tri"),
            "parquet carries the packed joint triangle: {pq_cols:?}"
        );
        assert!(
            csv_cols.iter().any(|c| c == "covt_00"),
            "CSV carries the state covariance cells: {csv_cols:?}"
        );

        // The covariance actually reached the CSV, with a value.
        let text = std::fs::read_to_string(&csv_path).expect("read csv");
        let first = text.lines().nth(1).expect("a data row");
        let idx = csv_cols
            .iter()
            .position(|c| c == "covt_00")
            .expect("covt_00 column");
        let cell = first.split(',').nth(idx).expect("covt_00 cell");
        assert!(
            cell.parse::<f64>().is_ok_and(|v| v > 0.0),
            "CSV must carry a real covariance, got {cell:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
        unsafe { empyrean_orbits_batch_free(&batch as *const _ as *mut _) };
    }

    /// Every rejection code maps to a distinct name — a collision would
    /// merge two causes in the attribution census.
    #[test]
    fn rejection_reason_names_are_distinct() {
        let codes = [
            crate::od::EMPYREAN_REJECTION_ACCEPTED,
            crate::od::EMPYREAN_REJECTION_CHI_SQUARED,
            crate::od::EMPYREAN_REJECTION_SIGMA_CLIP,
            crate::od::EMPYREAN_REJECTION_COOKS_DISTANCE,
            crate::od::EMPYREAN_REJECTION_ADAPTIVE,
            crate::od::EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY,
            crate::od::EMPYREAN_REJECTION_CMC2003,
            crate::od::EMPYREAN_REJECTION_RADAR_UNSUPPORTED,
            crate::od::EMPYREAN_REJECTION_OCCULTATION_UNSUPPORTED,
            crate::od::EMPYREAN_REJECTION_OUTSIDE_ARC,
            crate::od::EMPYREAN_REJECTION_NON_FINITE_CHI2,
            crate::od::EMPYREAN_REJECTION_MISSING_JACOBIAN,
            crate::od::EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING,
            crate::od::EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED,
            crate::od::EMPYREAN_REJECTION_NEVER_ABSORBED,
            crate::od::EMPYREAN_REJECTION_NOT_EVALUATED,
        ];
        let mut names: Vec<&str> = codes.iter().map(|c| rejection_reason_str(*c)).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "each code needs its own name");
    }
}

// DELETED (0.11.0): module `orbit_file_joint_tests` (`bordered_batch`,
// `a_bordered_orbit_round_trips_through_parquet_with_its_3x3`,
// `a_border_reaching_the_batch_without_its_3x3_is_impossible`,
// `an_orbit_read_from_parquet_propagates`). It built a state+Marsden joint
// through the removed `CoordinateState` border + `EmpyreanOrbit`
// `has_non_grav_covariance`/`non_grav_covariance` C fields and
// `ExtendedCovariance`/`coordinates_with_extended`. RULING-REMOVED:
// `a_border_reaching_the_batch_without_its_3x3_is_impossible` pinned the
// border/3x3 belt-and-braces, which no longer exists (the packed joint is
// ONE atomic matrix — a border cannot reach a batch without its block).
// LIVE SUCCESSOR for the lead to re-add on the FILE seam: write an orbit
// whose `covariance` is a width-9 Marsden joint to parquet, read it back,
// assert layout_present/width/tri survive, and that it propagates — the
// parquet dual of the packed-joint push round-trip ruling test.

#[cfg(test)]
mod photometry_batch_tests {
    use super::residual_writer_tests::tmp_dir;
    use super::*;

    /// The 3×3 the fixture carries: a real HG fit's shape, with the
    /// strong negative H–G correlation such fits produce, so a
    /// transposed or partially-copied round trip is visible.
    const PHOT_COV: [[f64; 3]; 3] = [
        [0.0900, -0.0042, 0.0],
        [-0.0042, 0.0025, 0.0],
        [0.0, 0.0, 0.0],
    ];

    /// A one-orbit batch carrying HG photometry with its 3×3.
    fn photometric_batch() -> EmpyreanOrbitBatch {
        // SAFETY: `#[repr(C)]` POD plus pointers that zero-init to null.
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 60320.0,
            elements: [1.1, 0.2, 0.03, -0.004, 0.017, 0.0006],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: crate::od::EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 0,
            origin: 10,
        };
        orbit.non_grav_dt = f64::NAN;
        orbit.phot_system = crate::propagate::EMPYREAN_PHASE_FUNCTION_HG;
        orbit.h_mag = 19.7;
        orbit.slope1 = 0.15;
        orbit.slope2 = 0.0;
        orbit.has_phot_covariance = 1;
        orbit.phot_covariance = PHOT_COV;

        let id = crate::od::alloc_cstring_for_test("photometric-orbit");
        EmpyreanOrbitBatch {
            orbits: Box::into_raw(Box::new(orbit)),
            orbit_ids: Box::into_raw(Box::new(id)),
            object_ids: Box::into_raw(Box::new(std::ptr::null_mut::<c_char>())),
            num_orbits: 1,
        }
    }

    /// A one-orbit batch built the `memset(0)` way, with only the state
    /// filled in — no photometry of any kind.
    fn zero_initialized_batch() -> EmpyreanOrbitBatch {
        // SAFETY: `#[repr(C)]` POD plus pointers that zero-init to null.
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 60320.0,
            elements: [1.1, 0.2, 0.03, -0.004, 0.017, 0.0006],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: crate::od::EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 0,
            origin: 10,
        };
        // The three NaN sentinels a C caller sets for "absent"; leaving
        // them zero is a separate (pre-existing) question, and this test
        // is about photometry.
        orbit.non_grav_dt = f64::NAN;

        let id = crate::od::alloc_cstring_for_test("zero-init-orbit");
        EmpyreanOrbitBatch {
            orbits: Box::into_raw(Box::new(orbit)),
            orbit_ids: Box::into_raw(Box::new(id)),
            object_ids: Box::into_raw(Box::new(std::ptr::null_mut::<c_char>())),
            num_orbits: 1,
        }
    }

    fn assert_photometry_survived(got: &EmpyreanOrbit, what: &str) {
        assert_eq!(
            got.phot_system,
            crate::propagate::EMPYREAN_PHASE_FUNCTION_HG,
            "{what}: the phase function must survive"
        );
        assert!(
            (got.h_mag - 19.7).abs() < 1e-12,
            "{what}: H must survive (got {})",
            got.h_mag
        );
        assert!(
            (got.slope1 - 0.15).abs() < 1e-12,
            "{what}: the slope must survive (got {})",
            got.slope1
        );
        assert_eq!(
            got.has_phot_covariance, 1,
            "{what}: the photometric 3×3 must survive"
        );
        for (i, (got_row, want_row)) in got.phot_covariance.iter().zip(PHOT_COV).enumerate() {
            for (j, (g, w)) in got_row.iter().zip(want_row).enumerate() {
                assert!(
                    (g - w).abs() < 1e-15,
                    "{what}: phot_covariance[{i}][{j}] must round-trip exactly"
                );
            }
        }
    }

    /// The in-memory marshal pair, both directions in one call.
    ///
    /// This FAILS on the pre-fix code at BOTH ends: `batch_to_orbits`
    /// never called `set_photometric_params` at all (so nothing reached
    /// the engine `Orbits`), and `orbits_to_batch` dropped
    /// `ph.covariance` on the way back out.
    #[test]
    fn photometry_and_its_covariance_round_trip_through_the_marshal_pair() {
        let batch = photometric_batch();
        let orbits = batch_to_orbits(&batch).expect("batch → orbits");

        let ph = orbits
            .photometric_params(0)
            .expect("batch_to_orbits must attach photometry, not drop it");
        assert!((ph.h() - 19.7).abs() < 1e-12);
        assert_eq!(
            ph.covariance,
            Some(PHOT_COV),
            "the 3×3 must reach the engine orbit — this is the input that makes \
             magnitude_uncertainty's σ_photo term reachable at all"
        );

        let back = orbits_to_batch(&orbits).expect("orbits → batch");
        assert_photometry_survived(unsafe { &*back.orbits }, "marshal pair");

        unsafe { empyrean_orbits_batch_free(&back as *const _ as *mut _) };
        unsafe { empyrean_orbits_batch_free(&batch as *const _ as *mut _) };
    }

    /// Both engine orbit-file formats carry photometry + its covariance.
    ///
    /// The write side is what `batch_to_orbits` used to break: the CSV
    /// writer's own doc claims to carry photometry, and it wrote NULL.
    #[test]
    fn photometry_and_its_covariance_round_trip_through_parquet_and_csv() {
        for (label, ext, write, read) in [
            (
                "parquet",
                "parquet",
                empyrean_orbits_write_parquet
                    as unsafe extern "C" fn(*const c_char, *const EmpyreanOrbitBatch) -> i32,
                empyrean_orbits_read_parquet
                    as unsafe extern "C" fn(*const c_char, *mut EmpyreanOrbitBatch) -> i32,
            ),
            (
                "csv",
                "csv",
                empyrean_orbits_write_csv
                    as unsafe extern "C" fn(*const c_char, *const EmpyreanOrbitBatch) -> i32,
                empyrean_orbits_read_csv
                    as unsafe extern "C" fn(*const c_char, *mut EmpyreanOrbitBatch) -> i32,
            ),
        ] {
            let dir = tmp_dir(&format!("photometric-{label}"));
            let path = dir.join(format!("photometric.{ext}"));
            let c_path = CString::new(path.display().to_string()).unwrap();

            let written = photometric_batch();
            let rc = unsafe { write(c_path.as_ptr(), &written) };
            let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
            assert_eq!(rc, 0, "{label}: writing must succeed: {err}");

            let mut read_back: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
            let rc = unsafe { read(c_path.as_ptr(), &mut read_back) };
            let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
            assert_eq!(rc, 0, "{label}: reading back must succeed: {err}");
            assert_eq!(read_back.num_orbits, 1);

            assert_photometry_survived(unsafe { &*read_back.orbits }, label);

            unsafe { empyrean_orbits_batch_free(&mut read_back) };
            unsafe { empyrean_orbits_batch_free(&written as *const _ as *mut _) };
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// Absence survives the write path too.
    ///
    /// `EmpyreanOrbit o = {0}` names a phase function (HG is 0) with a
    /// finite H (0.0), so the moment `batch_to_orbits` started attaching
    /// photometry, a zero-initialized orbit began WRITING HG H = 0 into
    /// orbit files — an absolute magnitude ~20 mag too bright, in a file
    /// that previously carried NULL and that reads back as fact.
    ///
    /// FAILS with `phot_system == HG` and `h_mag == 0.0` on the
    /// finite-only presence predicate.
    #[test]
    fn a_zero_initialized_orbit_writes_no_photometry() {
        let dir = tmp_dir("photometric-zero-init");
        let path = dir.join("zero-init.parquet");
        let c_path = CString::new(path.display().to_string()).unwrap();

        // Zero-init, state filled in, photometry never touched — the
        // idiom `EmpyreanOrbit`'s own docs present as supported.
        let written = zero_initialized_batch();
        assert_eq!(
            unsafe { (*written.orbits).phot_system },
            crate::propagate::EMPYREAN_PHASE_FUNCTION_HG,
            "the trap this guards: memset(0) leaves phot_system naming HG"
        );

        let rc = unsafe { empyrean_orbits_write_parquet(c_path.as_ptr(), &written) };
        let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
        assert_eq!(rc, 0, "writing must succeed: {err}");

        let mut read_back: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
        let rc = unsafe { empyrean_orbits_read_parquet(c_path.as_ptr(), &mut read_back) };
        let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
        assert_eq!(rc, 0, "reading back must succeed: {err}");
        assert_eq!(read_back.num_orbits, 1);

        let got = unsafe { &*read_back.orbits };
        assert_eq!(
            got.phot_system,
            crate::propagate::EMPYREAN_PHASE_FUNCTION_NONE,
            "an orbit that was never given photometry must read back with none, not with \
             a fabricated HG H = 0"
        );
        assert_eq!(
            got.has_phot_covariance, 0,
            "and no photometric covariance either"
        );

        unsafe { empyrean_orbits_batch_free(&mut read_back) };
        unsafe { empyrean_orbits_batch_free(&written as *const _ as *mut _) };
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod ephemeris_cov_kind_writer_tests {
    use super::residual_writer_tests::tmp_dir;
    use super::*;
    use arrow::array::Int32Array;

    /// The 21 ephemeris columns that predate the `cov_kind` append, in order.
    /// The writers must grow only at the tail, so a reader of an older file
    /// still parses every one of these by name and position.
    const PRIOR_EPHEMERIS_COLUMNS: [&str; 21] = [
        "orbit_id",
        "obs_code",
        "epoch_mjd_tdb",
        "ra_deg",
        "dec_deg",
        "rho_au",
        "vrho_au_day",
        "vra_deg_day",
        "vdec_deg_day",
        "light_time_days",
        "phase_angle_deg",
        "elongation_deg",
        "heliocentric_distance_au",
        "mag",
        "mag_sigma",
        "zenith_angle_deg",
        "azimuth_deg",
        "hour_angle_deg",
        "lunar_elongation_deg",
        "position_angle_deg",
        "sky_rate_deg_day",
    ];

    /// A zeroed ephemeris entry carrying a chosen covariance-kind wire value.
    fn entry(epoch: f64, cov_kind: u8) -> EmpyreanEphemerisEntry {
        let mut e: EmpyreanEphemerisEntry = unsafe { std::mem::zeroed() };
        e.epoch_mjd_tdb = epoch;
        e.ra_deg = 123.5;
        e.dec_deg = -4.25;
        e.cov_kind = cov_kind;
        e
    }

    /// Each writer carries a per-row `cov_kind` column that round-trips
    /// through the matching reader, and the file schemas grow ONLY by that
    /// appended column: the 21 prior columns keep their names and order.
    ///
    /// Mutation: dropping the `cov_kind` field from `EphemerisRow` /
    /// `EPHEMERIS_PARQUET_FIELDS` / `ephemeris_append` drops the column and
    /// every `cov_kind` lookup below goes red.
    #[test]
    fn each_writer_carries_cov_kind_appended_and_round_trips() {
        let dir = tmp_dir("ephemeris-cov-kind");
        // Linear, SecondOrder, Mixture — three distinct wire codes.
        let entries = [entry(60000.0, 0), entry(60001.0, 1), entry(60002.0, 3)];
        let expected: Vec<i32> = vec![0, 1, 3];

        let pq = dir.join("e.parquet");
        let js = dir.join("e.json");
        let cs = dir.join("e.csv");
        let c_pq = std::ffi::CString::new(pq.to_str().unwrap()).unwrap();
        let c_js = std::ffi::CString::new(js.to_str().unwrap()).unwrap();
        let c_cs = std::ffi::CString::new(cs.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe {
                empyrean_ephemeris_write_parquet(c_pq.as_ptr(), entries.as_ptr(), entries.len())
            },
            0
        );
        assert_eq!(
            unsafe {
                empyrean_ephemeris_write_json(c_js.as_ptr(), entries.as_ptr(), entries.len())
            },
            0
        );
        assert_eq!(
            unsafe { empyrean_ephemeris_write_csv(c_cs.as_ptr(), entries.as_ptr(), entries.len()) },
            0
        );

        // Parquet: the schema is the 21 prior columns in order, then cov_kind,
        // and the Int32 column carries the per-row kinds.
        let file = File::open(&pq).expect("open parquet");
        let mut reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("parquet reader")
                .build()
                .expect("build reader");
        let batch = reader.next().expect("one batch").expect("batch ok");
        let names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            &names[..21],
            &PRIOR_EPHEMERIS_COLUMNS,
            "the 21 prior columns keep their names and order"
        );
        assert_eq!(
            names[21], "cov_kind",
            "cov_kind keeps its position after the 21 prior columns"
        );
        assert_eq!(
            names.len(),
            28,
            "cov_kind plus the six appended mixture-tally columns"
        );
        let idx = batch
            .schema()
            .index_of("cov_kind")
            .expect("cov_kind column");
        let col = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("cov_kind is Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, expected, "parquet cov_kind round-trips per row");

        // CSV: cov_kind is the last header column and each data row ends with
        // its kind.
        let csv_text = std::fs::read_to_string(&cs).expect("read csv");
        let mut lines = csv_text.lines();
        let header: Vec<&str> = lines.next().expect("header").split(',').collect();
        assert_eq!(&header[..21], &PRIOR_EPHEMERIS_COLUMNS);
        assert_eq!(header[21], "cov_kind");
        for (line, want) in lines.zip(expected.iter()) {
            let field = line.split(',').nth(21).expect("cov_kind field");
            assert_eq!(field, want.to_string(), "csv cov_kind round-trips per row");
        }

        // JSON: every object carries the cov_kind key with its kind.
        let json_text = std::fs::read_to_string(&js).expect("read json");
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json_text).expect("json");
        for (row, want) in rows.iter().zip(expected.iter()) {
            let v = row
                .get("cov_kind")
                .and_then(|x| x.as_i64())
                .expect("cov_kind key present");
            assert_eq!(v as i32, *want, "json cov_kind round-trips per row");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An entry whose sky covariance was a mixture, carrying the engine's six
    /// tallies (with `n_unresolved` / `n_sky_linearization_refused` engine-None
    /// → `-1`).
    fn mixture_entry(epoch: f64) -> EmpyreanEphemerisEntry {
        let mut e = entry(epoch, 3);
        e.has_mixture_tally = 1;
        e.mixture_tally = crate::joint::EmpyreanMixtureTally {
            n_components_total: 7,
            weight_delivered: 0.9,
            n_failed: 2,
            n_unresolved: -1,
            n_curvature_refused: 3,
            n_sky_linearization_refused: -1,
        };
        e
    }

    /// The ephemeris writers carry the six mixture-tally columns, appended
    /// after `cov_kind`: a mixture row reports the engine's tallies and a
    /// non-mixture row reports the absent sentinel (`-1` counts, NaN weight),
    /// never fabricated zeros.
    ///
    /// Mutation: writing `0` instead of the `-1` / NaN sentinel for a
    /// non-mixture row in `ephemeris_to_rows` turns the absent-row asserts red;
    /// dropping any `mix_*` field from `EphemerisRow` / the parquet field array
    /// / `ephemeris_append` drops the column and the lookups go red.
    #[test]
    fn each_writer_carries_mixture_tally_appended_and_round_trips() {
        use arrow::array::Float64Array;
        let dir = tmp_dir("ephemeris-mixture-tally");
        // Row 0: a mixture. Row 1: no mixture (absent tally).
        let entries = [mixture_entry(60000.0), entry(60001.0, 0)];

        let pq = dir.join("e.parquet");
        let js = dir.join("e.json");
        let cs = dir.join("e.csv");
        let c_pq = std::ffi::CString::new(pq.to_str().unwrap()).unwrap();
        let c_js = std::ffi::CString::new(js.to_str().unwrap()).unwrap();
        let c_cs = std::ffi::CString::new(cs.to_str().unwrap()).unwrap();
        assert_eq!(
            unsafe {
                empyrean_ephemeris_write_parquet(c_pq.as_ptr(), entries.as_ptr(), entries.len())
            },
            0
        );
        assert_eq!(
            unsafe {
                empyrean_ephemeris_write_json(c_js.as_ptr(), entries.as_ptr(), entries.len())
            },
            0
        );
        assert_eq!(
            unsafe { empyrean_ephemeris_write_csv(c_cs.as_ptr(), entries.as_ptr(), entries.len()) },
            0
        );

        // Parquet: the six mixture columns follow cov_kind (positions 22..28).
        let file = File::open(&pq).expect("open parquet");
        let mut reader =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("parquet reader")
                .build()
                .expect("build reader");
        let batch = reader.next().expect("one batch").expect("batch ok");
        let names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            &names[22..28],
            &[
                "mix_n_components_total",
                "mix_weight_delivered",
                "mix_n_failed",
                "mix_n_unresolved",
                "mix_n_curvature_refused",
                "mix_n_sky_linearization_refused",
            ],
            "the six mixture columns follow cov_kind, in order"
        );
        let i32_col = |name: &str| -> Vec<i32> {
            let idx = batch.schema().index_of(name).expect("column");
            let c = batch
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32");
            (0..c.len()).map(|i| c.value(i)).collect()
        };
        assert_eq!(
            i32_col("mix_n_components_total"),
            vec![7, -1],
            "mixture row 7, absent row -1"
        );
        assert_eq!(i32_col("mix_n_failed"), vec![2, -1]);
        assert_eq!(
            i32_col("mix_n_unresolved"),
            vec![-1, -1],
            "engine-None on the mixture row is -1, and the absent row is -1"
        );
        assert_eq!(i32_col("mix_n_curvature_refused"), vec![3, -1]);
        assert_eq!(
            i32_col("mix_n_sky_linearization_refused"),
            vec![-1, -1],
            "sky-linearization refusals are carried as -1 (absent), never 0"
        );
        let widx = batch.schema().index_of("mix_weight_delivered").unwrap();
        let wcol = batch
            .column(widx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64");
        assert_eq!(wcol.value(0), 0.9, "mixture row carries the delivered mass");
        assert!(
            wcol.value(1).is_nan(),
            "an absent weight is NaN, never a fabricated 0"
        );

        // JSON: the absent row carries -1 counts (never 0) by name.
        let json_text = std::fs::read_to_string(&js).expect("read json");
        let rows: Vec<serde_json::Value> = serde_json::from_str(&json_text).expect("json");
        assert_eq!(
            rows[0].get("mix_n_components_total").unwrap().as_i64(),
            Some(7)
        );
        assert_eq!(
            rows[1]
                .get("mix_n_sky_linearization_refused")
                .unwrap()
                .as_i64(),
            Some(-1),
            "an absent row reads -1 by name, not 0"
        );

        // CSV: the six columns are present in the header, after cov_kind.
        let csv_text = std::fs::read_to_string(&cs).expect("read csv");
        let header: Vec<&str> = csv_text
            .lines()
            .next()
            .expect("header")
            .split(',')
            .collect();
        assert_eq!(header[22], "mix_n_components_total");
        assert_eq!(header[27], "mix_n_sky_linearization_refused");

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod orbit_json_joint_tests {
    use super::residual_writer_tests::tmp_dir;
    use super::*;

    /// A one-orbit batch whose covariance is a width-9 Cartesian joint (6
    /// state + the 3 Marsden axes) with a distinctive, strictly increasing
    /// triangle — so a transposed, truncated or mis-tagged round trip is
    /// visible. The joint is library-owned (built through the JSON-wire
    /// helper), so `empyrean_orbits_batch_free` releases it.
    fn wide_joint_batch() -> EmpyreanOrbitBatch {
        // SAFETY: `#[repr(C)]` POD plus pointers that zero-init to null.
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 60320.0,
            elements: [1.1, 0.2, 0.03, -0.004, 0.017, 0.0006],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: crate::od::EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 0,
            origin: 10,
        };
        orbit.non_grav_dt = f64::NAN;
        orbit.h_mag = f64::NAN;
        let tri: Vec<f64> = (0..45).map(|i| 1.0 + i as f64 * 0.5).collect();
        orbit.covariance = crate::joint::packed_joint_from_json_wire(
            0b111, // A1/A2/A3 present → width 6 + 3 = 9
            0,
            0,
            9,
            tri,
            crate::propagate::EMPYREAN_COVARIANCE_KIND_LINEAR,
            crate::joint::wire_tags::EMPYREAN_JOINT_QUALITY_POSITIVE_DEFINITE,
            0,
            0,
            f64::NAN,
            crate::joint::wire_tags::EMPYREAN_JOINT_FUNCTIONAL_STATE,
            crate::joint::wire_tags::EMPYREAN_JOINT_PROVENANCE_DERIVED,
            crate::joint::wire_tags::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE,
        );

        let id = crate::od::alloc_cstring_for_test("wide-joint-orbit");
        EmpyreanOrbitBatch {
            orbits: Box::into_raw(Box::new(orbit)),
            orbit_ids: Box::into_raw(Box::new(id)),
            object_ids: Box::into_raw(Box::new(std::ptr::null_mut::<c_char>())),
            num_orbits: 1,
        }
    }

    /// The JSON orbit format now carries a joint wider than the state 6×6 —
    /// no refusal — and restores its packed triangle and every tag
    /// bit-identically. Before 0.11.0 this width was refused by name at the
    /// writer; the state-only 6×6 was all the row could hold.
    #[test]
    fn json_carries_and_restores_a_width9_joint() {
        let dir = tmp_dir("json-wide-joint");
        let path = dir.join("wide.json");
        let c_path = CString::new(path.display().to_string()).unwrap();

        let written = wide_joint_batch();
        let rc = unsafe { empyrean_orbits_write_json(c_path.as_ptr(), &written) };
        let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
        assert_eq!(
            rc, 0,
            "a width-9 joint must now WRITE to JSON, not be refused: {err}"
        );

        let mut back: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
        let rc = unsafe { empyrean_orbits_read_json(c_path.as_ptr(), &mut back) };
        let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy();
        assert_eq!(rc, 0, "reading the JSON back must succeed: {err}");
        assert_eq!(back.num_orbits, 1);

        let got = unsafe { &*back.orbits };
        let want = unsafe { &*written.orbits };
        assert_eq!(
            got.covariance.width, 9,
            "the wide joint's width must survive"
        );
        assert_eq!(
            got.covariance.layout_present, 0b111,
            "the Marsden axes survive"
        );
        assert_eq!(got.covariance.tri_len, want.covariance.tri_len);
        assert_eq!(got.covariance.kind, want.covariance.kind);
        assert_eq!(got.covariance.quality, want.covariance.quality);
        assert_eq!(got.covariance.functional, want.covariance.functional);
        assert_eq!(got.covariance.provenance, want.covariance.provenance);
        assert_eq!(
            got.covariance.asserted_source,
            want.covariance.asserted_source
        );
        assert!(
            got.covariance.cov_min_eig_corr.is_nan(),
            "a positive-definite joint restores its NaN min-eig sentinel, not 0"
        );
        let got_tri =
            unsafe { std::slice::from_raw_parts(got.covariance.tri, got.covariance.tri_len) };
        let want_tri =
            unsafe { std::slice::from_raw_parts(want.covariance.tri, want.covariance.tri_len) };
        for (i, (g, w)) in got_tri.iter().zip(want_tri).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "cov_tri[{i}] must round-trip bit-identical"
            );
        }
        // The flat 6×6 is the joint's derived leading block.
        assert_eq!(
            got.state.has_covariance, 1,
            "the derived flat 6×6 is present"
        );
        assert_eq!(got.state.covariance[0][0].to_bits(), want_tri[0].to_bits());

        unsafe { empyrean_orbits_batch_free(&mut back) };
        unsafe { empyrean_orbits_batch_free(&written as *const _ as *mut _) };
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A JSON file written before 0.11.0 — carrying the removed flat 6×6
    /// `covariance` field — is refused BY NAME on read, with a message naming
    /// the old shape. No silent acceptance, no migration shim.
    #[test]
    fn json_refuses_the_removed_flat_covariance_field_by_name() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/orbit_legacy_flat_covariance.json");
        let c_path = CString::new(fixture.display().to_string()).unwrap();
        let mut back: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
        let rc = unsafe { empyrean_orbits_read_json(c_path.as_ptr(), &mut back) };
        assert_ne!(
            rc, 0,
            "the legacy flat-covariance file must be refused, not read"
        );
        let err = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }
            .to_string_lossy()
            .into_owned();
        assert!(
            err.contains("flat") && err.contains("covariance") && err.contains("packed joint"),
            "the refusal must name the removed flat covariance shape: {err}"
        );
        assert_eq!(back.num_orbits, 0, "nothing is read on refusal");
    }

    /// A flat state 6×6 symmetric covariance (the shape `state_block_guard`
    /// uses): positive-definite, with one symmetric off-diagonal pair.
    fn symmetric_state_6x6() -> [[f64; 6]; 6] {
        let diag = [4.8e-10, 2.0e-10, 1.9e-10, 1.4e-14, 1.5e-13, 6.0e-15];
        let mut m = [[0.0f64; 6]; 6];
        for (i, d) in diag.iter().enumerate() {
            m[i][i] = *d;
        }
        m[0][1] = 3.0e-11;
        m[1][0] = 3.0e-11;
        m
    }

    /// A one-orbit Cartesian batch carrying ONLY a flat state 6×6 (no packed
    /// joint): the hand-built C orbit the reviewer flagged. `std::mem::zeroed`
    /// leaves `covariance` the width-0 absent joint.
    fn flat_only_batch(cov: [[f64; 6]; 6]) -> EmpyreanOrbitBatch {
        // SAFETY: `#[repr(C)]` POD plus pointers that zero-init to null.
        let mut orbit: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        orbit.state = CoordinateState {
            epoch_mjd_tdb: 60320.0,
            elements: [1.1, 0.2, 0.03, -0.004, 0.017, 0.0006],
            covariance: cov,
            has_covariance: 1,
            representation: crate::od::EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 1,
            origin: 10,
        };
        orbit.non_grav_dt = f64::NAN;
        orbit.h_mag = f64::NAN;
        let id = crate::od::alloc_cstring_for_test("flat-only-orbit");
        EmpyreanOrbitBatch {
            orbits: Box::into_raw(Box::new(orbit)),
            orbit_ids: Box::into_raw(Box::new(id)),
            object_ids: Box::into_raw(Box::new(std::ptr::null_mut::<c_char>())),
            num_orbits: 1,
        }
    }

    /// A flat-only orbit (no joint, just a state 6×6) is converted through the
    /// SAME engine path the parquet writer uses, so JSON stamps the covariance
    /// `provenance = ASSERTED` from `asserted_source = EXTERNAL_FILE` — never a
    /// fabricated `DERIVED` — and its tags equal the parquet writer's for the
    /// identical batch.
    #[test]
    fn flat_only_orbit_json_stamps_asserted_external_file_matching_parquet() {
        let dir = tmp_dir("flat-only-parity");
        let jpath = dir.join("flat.json");
        let ppath = dir.join("flat.parquet");
        let cj = CString::new(jpath.display().to_string()).unwrap();
        let cp = CString::new(ppath.display().to_string()).unwrap();

        let batch = flat_only_batch(symmetric_state_6x6());
        assert_eq!(
            unsafe { empyrean_orbits_write_json(cj.as_ptr(), &batch) },
            0,
            "flat-only JSON write"
        );
        assert_eq!(
            unsafe { empyrean_orbits_write_parquet(cp.as_ptr(), &batch) },
            0,
            "flat-only parquet write"
        );

        let mut from_json: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
        let mut from_pq: EmpyreanOrbitBatch = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { empyrean_orbits_read_json(cj.as_ptr(), &mut from_json) },
            0
        );
        assert_eq!(
            unsafe { empyrean_orbits_read_parquet(cp.as_ptr(), &mut from_pq) },
            0
        );

        let jc = &unsafe { &*from_json.orbits }.covariance;
        let pc = &unsafe { &*from_pq.orbits }.covariance;
        assert_eq!(
            jc.provenance,
            crate::joint::wire_tags::EMPYREAN_JOINT_PROVENANCE_ASSERTED,
            "a flat external 6×6 is an ASSERTED moment, never fabricated as DERIVED"
        );
        assert_eq!(
            jc.asserted_source,
            crate::joint::wire_tags::EMPYREAN_JOINT_ASSERTED_SOURCE_EXTERNAL_FILE,
            "its source is EXTERNAL_FILE, matching the engine's flat path"
        );
        // Every tag equals the parquet writer's for the identical input.
        assert_eq!(jc.kind, pc.kind, "kind parity json vs parquet");
        assert_eq!(jc.quality, pc.quality, "quality parity json vs parquet");
        assert_eq!(jc.functional, pc.functional, "functional parity");
        assert_eq!(jc.provenance, pc.provenance, "provenance parity");
        assert_eq!(
            jc.asserted_source, pc.asserted_source,
            "asserted-source parity"
        );
        assert_eq!(jc.width, pc.width, "both are the width-6 state joint");

        unsafe { empyrean_orbits_batch_free(&mut from_json) };
        unsafe { empyrean_orbits_batch_free(&mut from_pq) };
        unsafe { empyrean_orbits_batch_free(&batch as *const _ as *mut _) };
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An asymmetric flat 6×6 is refused BY NAME on the JSON write exactly as
    /// on the parquet write — both route through the engine's flat conversion,
    /// which refuses an `AsymmetricStateBlock`. JSON no longer accepts a
    /// matrix parquet would reject.
    #[test]
    fn an_asymmetric_flat_6x6_is_refused_by_name_on_json_like_parquet() {
        let mut cov = symmetric_state_6x6();
        cov[1][0] = -cov[0][1]; // break symmetry on an off-diagonal pair

        let dir = tmp_dir("flat-asym");
        let jpath = dir.join("asym.json");
        let ppath = dir.join("asym.parquet");
        let cj = CString::new(jpath.display().to_string()).unwrap();
        let cp = CString::new(ppath.display().to_string()).unwrap();

        let batch = flat_only_batch(cov);
        let rc_j = unsafe { empyrean_orbits_write_json(cj.as_ptr(), &batch) };
        let err_j = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }
            .to_string_lossy()
            .into_owned();
        let rc_p = unsafe { empyrean_orbits_write_parquet(cp.as_ptr(), &batch) };
        let err_p = unsafe { CStr::from_ptr(crate::empyrean_last_error()) }
            .to_string_lossy()
            .into_owned();

        assert_ne!(rc_j, 0, "JSON must refuse an asymmetric 6×6: {err_j}");
        assert_ne!(rc_p, 0, "parquet must refuse an asymmetric 6×6: {err_p}");
        assert!(
            err_j.contains("AsymmetricStateBlock"),
            "JSON refuses by name: {err_j}"
        );
        assert!(
            err_p.contains("AsymmetricStateBlock"),
            "parquet refuses by the same name: {err_p}"
        );
        assert!(!jpath.exists(), "no JSON file is written on refusal");

        unsafe { empyrean_orbits_batch_free(&batch as *const _ as *mut _) };
        std::fs::remove_dir_all(&dir).ok();
    }
}
