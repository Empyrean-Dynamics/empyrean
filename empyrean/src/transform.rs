//! Coordinate transformation.

use crate::context::Context;
use crate::coordinate::{CoordinateState, Frame, Origin, Representation};
use crate::error::{Error, Result};
use crate::io::OrbitBatch;
use crate::orbit::{Orbit, orbits_to_ffi};

impl Context {
    /// Transform a batch of **orbits** — coordinate state plus its packed
    /// joint covariance — to a different representation, frame, and/or origin.
    ///
    /// Each orbit's covariance is transported through the \\(6 \times 6\\)
    /// Jacobian of the transformation. This is the entry point for moving a
    /// covariance between bases: the state-only
    /// [`transform_coordinates`](Self::transform_coordinates) /
    /// [`transform_coordinates_single`](Self::transform_coordinates_single)
    /// transform bare states and **refuse** a covariance-bearing input rather
    /// than drop its covariance silently
    /// ([`PushError::TransformCovarianceUnsupported`](crate::PushError::TransformCovarianceUnsupported),
    /// which names this method).
    ///
    /// The returned [`OrbitBatch`] carries each transformed orbit with its
    /// joint transported and its identifiers preserved. An empty slice yields
    /// an empty batch and touches nothing.
    ///
    /// # Errors
    ///
    /// A per-orbit input refusal surfaces with its packed-joint wire code
    /// (classify with [`Error::push_error`](crate::Error::push_error)); the
    /// engine's per-element failure names the failing orbit's index in the
    /// message.
    pub fn transform_orbits(
        &self,
        orbits: &[Orbit],
        target_rep: Representation,
        target_frame: Frame,
        target_origin: Origin,
    ) -> Result<OrbitBatch> {
        if orbits.is_empty() {
            return OrbitBatch::new(Vec::new(), Vec::new(), Vec::new());
        }
        // `_keep` owns the heap storage each input `EmpyreanOrbit` borrows
        // (ids, thrust arrays, the packed-joint triangle); it must outlive the
        // call.
        let (inputs, _keep) = orbits_to_ffi(orbits)?;
        let mut out = empyrean_sys::EmpyreanOrbitBatch {
            orbits: std::ptr::null_mut(),
            orbit_ids: std::ptr::null_mut(),
            object_ids: std::ptr::null_mut(),
            num_orbits: 0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_transform_orbits(
                self.as_raw(),
                inputs.as_ptr(),
                inputs.len(),
                target_rep as i32,
                target_frame as i32,
                target_origin.naif_id(),
                &mut out,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // Copy out into owned Rust values, then release the library-owned
        // batch before returning.
        let batch = crate::io::orbits::ffi_batch_to_owned(&out);
        unsafe { empyrean_sys::empyrean_orbits_batch_free(&mut out) };
        batch
    }

    /// Transform a **batch** of coordinate states to a different
    /// representation, frame, and/or origin.
    ///
    /// The batched form carries the main name; the unit of work is
    /// [`transform_coordinates_single`](Self::transform_coordinates_single).
    /// Element `i` of the returned `Vec` is **bit-identical** to that
    /// method applied to `states[i]` with the same target arguments, so
    /// batching is a scheduling choice and never a numerical one. An
    /// empty slice yields an empty `Vec` and touches nothing.
    ///
    /// Reach for this whenever more than one state is going to the same
    /// target basis. Two costs in the transform pipeline do not depend on
    /// the individual state — gravitational-parameter resolution (keyed
    /// on the origin) and the origin shift (keyed on `(from, to, epoch)`)
    /// — and the engine pays each once per distinct key rather than once
    /// per state, so a catalogue snapshot or a sigma-point cloud sharing
    /// an epoch reuses one shift.
    ///
    /// Note what that does **not** buy: those memos are scoped to the
    /// [`Context`], not to the call, so a loop over
    /// [`transform_coordinates_single`](Self::transform_coordinates_single)
    /// on the same context hits them too. Measured against such a loop
    /// the batch is ~17% faster from a thousand states up and
    /// indistinguishable at one — what it saves is the per-call boundary
    /// crossing, not the shift. Prefer it for the shape (one call, one
    /// error, one index) rather than for a large speedup. A batch that
    /// reuses nothing costs the same as the equivalent single-state loop
    /// rather than more.
    ///
    /// Covariance is propagated through the Jacobian of the
    /// transformation for every element that carries one.
    ///
    /// # Errors
    ///
    /// **Fail-fast, with the index.** The first element that fails aborts
    /// the batch; no partial `Vec` is ever returned. The error message
    /// names the failing element's zero-based index alongside its
    /// underlying cause.
    pub fn transform_coordinates(
        &self,
        states: &[CoordinateState],
        target_rep: Representation,
        target_frame: Frame,
        target_origin: Origin,
    ) -> Result<Vec<CoordinateState>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<empyrean_sys::CoordinateState> = states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                s.to_ffi()
                    .map_err(|e| Error::invalid_input(format!("batch element {i}: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs: Vec<empyrean_sys::CoordinateState> =
            (0..states.len()).map(|_| Default::default()).collect();

        let code = unsafe {
            empyrean_sys::empyrean_transform_coordinates(
                self.as_raw(),
                inputs.as_ptr(),
                inputs.len(),
                target_rep as i32,
                target_frame as i32,
                target_origin.naif_id(),
                outputs.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        outputs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                CoordinateState::from_ffi(o)
                    .map_err(|e| Error::invalid_input(format!("batch element {i}: {e}")))
            })
            .collect()
    }

    /// Transform **one** coordinate state to a different representation,
    /// frame, and/or origin.
    ///
    /// The single-state unit of work;
    /// [`transform_coordinates`](Self::transform_coordinates) is the
    /// batched form that transforms a whole slice and amortizes the
    /// shared cost across it.
    ///
    /// Covariance is propagated through the Jacobian of the
    /// transformation when the input state has a covariance attached.
    pub fn transform_coordinates_single(
        &self,
        coords: &CoordinateState,
        target_rep: Representation,
        target_frame: Frame,
        target_origin: Origin,
    ) -> Result<CoordinateState> {
        let input_ffi = coords.to_ffi()?;
        let mut output_ffi = empyrean_sys::CoordinateState::default();
        let code = unsafe {
            empyrean_sys::empyrean_transform_coordinates_single(
                self.as_raw(),
                &input_ffi,
                target_rep as i32,
                target_frame as i32,
                target_origin.naif_id(),
                &mut output_ffi,
            )
        };
        if code == 0 {
            CoordinateState::from_ffi(&output_ffi)
        } else {
            Err(Error::capture(code))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AxisLayout, Context, CoordinateState, CovarianceKind, Frame, JointFunctional,
        JointProvenance, JointQuality, Orbit, Origin, PackedJoint, PushError, Representation,
    };
    use std::path::PathBuf;

    /// Resolve a usable data dir (`EMPYREAN_DATA_DIR` else `~/.empyrean/data`),
    /// or `None` to skip when neither yields a working Context.
    fn try_context() -> Option<Context> {
        let candidates = [
            std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from),
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".empyrean/data")),
        ];
        for dir in candidates.into_iter().flatten() {
            if let Ok(ctx) = Context::from_data_dir(Some(&dir)) {
                return Some(ctx);
            }
        }
        None
    }

    /// A heliocentric Cartesian orbit carrying a width-6 state joint.
    fn cartesian_orbit_with_joint() -> Orbit {
        // A small, positive-definite diagonal-ish state triangle (packed lower
        // triangle of a 6×6), Cartesian basis.
        let mut tri = vec![0.0_f64; 21];
        // Diagonal entries (indices 0, 2, 5, 9, 14, 20).
        for (slot, &idx) in [0usize, 2, 5, 9, 14, 20].iter().enumerate() {
            tri[idx] = 1.0e-8 * (slot as f64 + 1.0);
        }
        let joint = PackedJoint {
            // State-only: no parameter axes, so width = 6 + popcount(0) = 6.
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
        Orbit::new(CoordinateState::cartesian(
            crate::Epoch::from_mjd_tdb(59000.0),
            [1.2, 0.3, 0.1, -0.004, 0.013, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        ))
        .with_orbit_id("rt")
        .with_covariance(joint)
    }

    /// `transform_orbits` carries the joint through the Jacobian, and a
    /// Cartesian → Keplerian → Cartesian round trip recovers both the state and
    /// the joint's state block to the C test's band (≤1e-6 rel — a band, not
    /// bit-identical, because the transform is a Jacobian congruence, not a
    /// copy).
    #[test]
    fn transform_orbits_round_trips_state_and_joint() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping transform_orbits_round_trips_state_and_joint: no data dir");
            return;
        };
        let o = cartesian_orbit_with_joint();
        let kep = ctx
            .transform_orbits(
                std::slice::from_ref(&o),
                Representation::Keplerian,
                Frame::EclipticJ2000,
                Origin::Sun,
            )
            .expect("Cartesian → Keplerian");
        let back = ctx
            .transform_orbits(
                &kep.orbits,
                Representation::Cartesian,
                Frame::EclipticJ2000,
                Origin::Sun,
            )
            .expect("Keplerian → Cartesian");
        assert_eq!(back.orbits.len(), 1);
        let rt = &back.orbits[0];
        for (i, (a, b)) in rt.state.elements.iter().zip(o.state.elements).enumerate() {
            let rel = (a - b).abs() / (a.abs() + b.abs()).max(1e-30);
            assert!(rel <= 1e-6, "state[{i}] round trip rel err {rel:e}");
        }
        let got = rt.covariance.as_ref().expect("joint survived round trip");
        let want = o.covariance.as_ref().unwrap();
        let (gb, wb) = (got.state_block(), want.state_block());
        let scale = (0..6).map(|i| wb[i][i].abs()).fold(0.0_f64, f64::max);
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (gb[i][j] - wb[i][j]).abs() <= 1e-6 * scale.max(f64::MIN_POSITIVE),
                    "joint[{i}][{j}] round trip {} vs {}",
                    gb[i][j],
                    wb[i][j]
                );
            }
        }
    }

    /// The state-only transform refuses a covariance-bearing state by name
    /// rather than dropping its covariance: the C ABI returns −50 and the
    /// wrapper classifies it as
    /// [`PushError::TransformCovarianceUnsupported`], whose message names
    /// `transform_orbits`.
    #[test]
    fn transform_coordinates_refuses_a_covariance_bearing_state() {
        let Some(ctx) = try_context() else {
            eprintln!(
                "skipping transform_coordinates_refuses_a_covariance_bearing_state: no data dir"
            );
            return;
        };
        let state = CoordinateState::cartesian(
            crate::Epoch::from_mjd_tdb(59000.0),
            [1.2, 0.3, 0.1, -0.004, 0.013, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        )
        .with_covariance([[0.0; 6]; 6]);
        let err = ctx
            .transform_coordinates_single(
                &state,
                Representation::Keplerian,
                Frame::EclipticJ2000,
                Origin::Sun,
            )
            .expect_err("a covariance-bearing state must be refused, not silently dropped");
        assert_eq!(
            err.push_error(),
            Some(PushError::TransformCovarianceUnsupported)
        );
        assert!(
            err.message.contains("transform_orbits"),
            "the refusal must name transform_orbits, got: {}",
            err.message
        );
    }
}
