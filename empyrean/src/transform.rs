//! Coordinate transformation.

use crate::context::Context;
use crate::coordinate::{CoordinateState, Frame, Origin, Representation};
use crate::error::{Error, Result};

impl Context {
    /// Transform a coordinate state to a different representation, frame,
    /// and/or origin.
    ///
    /// Covariance is propagated through the Jacobian of the transformation
    /// when the input state has a covariance attached.
    pub fn transform(
        &self,
        input: &CoordinateState,
        target_representation: Representation,
        target_frame: Frame,
        target_origin: Origin,
    ) -> Result<CoordinateState> {
        let input_ffi = input.to_ffi()?;
        let mut output_ffi = empyrean_sys::CoordinateState::default();
        let code = unsafe {
            empyrean_sys::empyrean_transform_coordinates(
                self.as_raw(),
                &input_ffi,
                target_representation as i32,
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
