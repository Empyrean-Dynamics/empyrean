//! Coordinate transformation.

use crate::context::Context;
use crate::coordinate::{CoordinateState, Frame, Origin, Representation};
use crate::error::{Error, Result};

impl Context {
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
