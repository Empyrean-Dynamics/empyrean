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
    Vec<crate::orbit::OrbitFfiKeep>,
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
    // `orbit_keep` owns every orbit's heap allocations — critically the packed
    // joint's triangle `Vec`, into which `EmpyreanOrbit::covariance.tri` holds a
    // borrowed raw pointer. It MUST be returned and held across the C call: the
    // id / object `CString`s are kept alive below, but if the orbit keepalive is
    // dropped here the triangle is freed and the writer reads a dangling pointer
    // — the state block then comes back as garbage that disagrees with the
    // inline flat 6×6 and the orbit-write guard refuses the (now contradictory)
    // pair. The flat `state.covariance` is inline in the struct, so it alone was
    // unaffected; JSON, which serializes only that flat 6×6 and never the joint
    // triangle, wrote correctly while parquet / CSV aborted.
    Ok((
        ffi_orbits,
        id_ptrs,
        obj_ptrs,
        id_cstrings,
        obj_cstrings,
        orbit_keep,
    ))
}

pub(crate) fn ffi_batch_to_owned(batch: &empyrean_sys::EmpyreanOrbitBatch) -> Result<OrbitBatch> {
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
        phot_system,
        h_mag: o.h_mag,
        slope1: o.slope1,
        slope2: o.slope2,
        // Carried through when the file (parquet / CSV, whose six
        // lower-triangle photometry columns hold it) supplied one.
        phot_covariance: (o.has_phot_covariance != 0).then_some(o.phot_covariance),
        // Thrust is a caller-owned input side array; the OrbitRow schema
        // doesn't carry it, so orbit reads never reconstruct it.
        thrust: None,
        // SRP slot carried through from the C EmpyreanOrbit when present.
        srp: crate::orbit::SRPParams::from_ffi(o.srp_amrat, o.srp_cr, o.has_srp),
        // The orbit's covariance as one packed joint. The batch a read
        // produces owns the triangle, so it is copied into an owned Vec here —
        // the wrapper's `Orbit` outlives the batch it came from. A width-0
        // joint reads back as `None`.
        covariance: unsafe { crate::PackedJoint::from_ffi(&o.covariance) }?,
    })
}

// ── Read / write helpers — batch lifetime managed via FFI free ────

fn read_orbits_via<F>(path: &Path, c_call: F) -> Result<OrbitBatch>
where
    F: FnOnce(*const std::ffi::c_char, *mut empyrean_sys::EmpyreanOrbitBatch) -> i32,
{
    crate::context::ensure_engine_loaded()?;
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
    // `_orbit_keep` owns the packed-joint triangles the FFI orbits borrow; it
    // must stay bound until after `c_call` returns or the writer reads a freed
    // triangle (see `batch_to_ffi`).
    let (ffi_orbits, id_ptrs, obj_ptrs, _ids_keep, _objs_keep, _orbit_keep) = batch_to_ffi(batch)?;
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
    crate::context::ensure_engine_loaded()?;
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
    crate::context::ensure_engine_loaded()?;
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
    crate::context::ensure_engine_loaded()?;
    write_orbits_via(path.as_ref(), batch, |p, b| unsafe {
        empyrean_sys::empyrean_orbits_write_csv(p, b)
    })
}

#[cfg(test)]
mod push_code_tests {
    use crate::{
        AxisLayout, CoordinateState, CovarianceKind, Epoch, Frame, JointFunctional,
        JointProvenance, JointQuality, Orbit, Origin, PackedJoint, PushError,
    };

    /// A well-formed state-only (width-6) Cartesian joint with a distinctive
    /// positive-definite diagonal, paired with the identical flat 6×6 on the
    /// coordinate — exactly the shape a converging `determine` returns. The
    /// flat view and the joint's state block agree by construction, so the
    /// orbit-write guard must accept the pair: any disagreement here is a
    /// distribution bug, not a contradictory caller input.
    fn roundtrip_orbit() -> (Orbit, [f64; 6]) {
        let diag = [4.8e-10, 2.0e-10, 1.9e-10, 1.4e-14, 1.5e-13, 6.0e-15];
        let mut dense = [[0.0f64; 6]; 6];
        let mut tri = vec![0.0f64; 21];
        for i in 0..6 {
            dense[i][i] = diag[i];
            tri[i * (i + 1) / 2 + i] = diag[i];
        }
        let joint = PackedJoint {
            layout: AxisLayout {
                present: 0,
                considered: 0,
                marginalized: 0,
            },
            width: 6,
            tri,
            kind: CovarianceKind::Linear,
            quality: JointQuality::PositiveDefinite,
            functional: JointFunctional::State,
            provenance: JointProvenance::Derived,
            asserted_source: None,
        };
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        )
        .with_covariance(dense);
        let orbit = Orbit::new(state)
            .with_orbit_id("roundtrip")
            .with_covariance(joint);
        (orbit, diag)
    }

    /// Regression for the keepalive use-after-free in `batch_to_ffi`: the
    /// per-orbit `OrbitFfiKeep` owns the packed joint's triangle `Vec`, which
    /// the FFI `EmpyreanOrbit::covariance.tri` borrows by raw pointer. The bag
    /// was built in a local that was dropped before the C writer ran, so the
    /// writer read a freed triangle; the state block came back as garbage that
    /// disagreed with the inline flat 6×6, and the orbit-write guard aborted
    /// parquet and CSV (JSON survived only because its row shape serializes the
    /// inline flat 6×6 and never the joint triangle). With the keepalive held
    /// across the call, all three formats write and the parquet round-trips the
    /// joint's state block bit-for-bit (Cartesian: no angular scaling).
    #[test]
    fn a_fitted_joint_survives_every_writer_and_round_trips_through_parquet() {
        let (orbit, diag) = roundtrip_orbit();
        let batch = super::OrbitBatch::new(vec![orbit], vec!["roundtrip".into()], vec![None])
            .expect("batch");

        let pq = tmp_path("parquet");
        let csv = tmp_path("csv");
        let json = tmp_path("json");
        super::write_orbits_parquet(&pq, &batch).expect("parquet write must succeed");
        super::write_orbits_csv(&csv, &batch).expect("csv write must succeed");
        super::write_orbits_json(&json, &batch).expect("json write must succeed");

        let back = super::read_orbits_parquet(&pq).expect("parquet read-back");
        let joint = back.orbits[0]
            .covariance
            .as_ref()
            .expect("the written joint must survive the parquet round trip");
        assert_eq!(joint.width, 6, "a state-only fit round-trips as width 6");
        let sb = joint.state_block();
        for i in 0..6 {
            assert!(
                (sb[i][i] - diag[i]).abs() <= 1e-18_f64.max(diag[i].abs() * 1e-12),
                "state-block variance [{i}][{i}] must round-trip ({} vs {}), never the \
                 freed-triangle garbage the use-after-free produced",
                sb[i][i],
                diag[i]
            );
        }

        let _ = std::fs::remove_file(&pq);
        let _ = std::fs::remove_file(&csv);
        let _ = std::fs::remove_file(&json);
    }

    /// A malformed packed joint — `width = 6` but a triangle too short for it
    /// (6 needs 21 doubles) — so the engine's ingestion refuses it with the
    /// triangle-length wire code. Constructible through the safe API because
    /// `width` and `tri` are independent fields.
    fn orbit_with_short_triangle() -> Orbit {
        let joint = PackedJoint {
            // State-only (width = 6 + popcount(0) = 6); the layout implies a
            // 21-double triangle, so a 3-double triangle is a length mismatch.
            layout: AxisLayout {
                present: 0,
                considered: 0,
                marginalized: 0,
            },
            width: 6,
            tri: vec![1.0e-8; 3], // far short of 6·7/2 = 21
            kind: CovarianceKind::Linear,
            quality: JointQuality::PositiveDefinite,
            functional: JointFunctional::State,
            provenance: JointProvenance::Derived,
            asserted_source: None,
        };
        Orbit::new(crate::CoordinateState::cartesian(
            crate::Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        ))
        .with_orbit_id("bad")
        .with_covariance(joint)
    }

    fn tmp_path(ext: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("empyrean_push_{nanos}_{ext}.{ext}"))
    }

    /// A packed-joint wire refusal reaches every push boundary as the SAME
    /// typed variant: writing the malformed orbit to parquet and to CSV both
    /// fail with the same [`PushError`]. The engine validates the joint on
    /// ingestion before either format serializes, so the two channels agree —
    /// the refusal is never collapsed to one generic code at one boundary and
    /// a different one at another.
    #[test]
    fn the_same_push_variant_reaches_both_the_parquet_and_csv_writers() {
        let batch = super::OrbitBatch::new(
            vec![orbit_with_short_triangle()],
            vec!["bad".into()],
            vec![None],
        )
        .expect("batch");

        let pq = tmp_path("parquet");
        let csv = tmp_path("csv");
        let parquet_err = super::write_orbits_parquet(&pq, &batch)
            .expect_err("a short triangle must be refused on parquet write");
        let csv_err = super::write_orbits_csv(&csv, &batch)
            .expect_err("a short triangle must be refused on csv write");
        let _ = std::fs::remove_file(&pq);
        let _ = std::fs::remove_file(&csv);

        let pv = parquet_err.push_error();
        let cv = csv_err.push_error();
        assert!(
            matches!(
                pv,
                Some(PushError::TriangleLength | PushError::WidthMismatch)
            ),
            "parquet refusal must be a packed-joint wire code, got {pv:?} ({})",
            parquet_err.message
        );
        assert_eq!(
            pv, cv,
            "the same malformed joint must yield the same variant at both writers"
        );
    }
}
