use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::{
    coordinate_state_to_coordinates, coordinates_to_coordinate_state, int_to_frame,
    int_to_representation,
};

use crate::{CoordinateState, EmpyreanContext, set_last_error};

/// Transform a coordinate state to a new representation, frame, and/or origin.
///
/// Returns 0 on success or a negative error code on failure.
/// Call `empyrean_last_error()` to retrieve the error message on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_transform_coordinates(
    ctx: *const EmpyreanContext,
    input: *const CoordinateState,
    target_representation: i32,
    target_frame: i32,
    target_origin: i32,
    output: *mut CoordinateState,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || input.is_null() || output.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let input_state = unsafe { &*input }.to_empyrean();

        let target_rep = match int_to_representation(target_representation) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };
        let target_frm = match int_to_frame(target_frame) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };
        let target_orig = match Origin::from_naif_id(target_origin) {
            Some(o) => o,
            None => {
                set_last_error(&format!("unknown NAIF id: {target_origin}"));
                return -1;
            }
        };

        let coords_in = match coordinate_state_to_coordinates(&input_state) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };

        match empyrean_core::coordinates::transform(
            ctx_ref,
            coords_in,
            target_rep,
            target_frm,
            target_orig,
        ) {
            Ok(transformed) => {
                let flat = coordinates_to_coordinate_state(&transformed);
                unsafe {
                    *output = CoordinateState::from_empyrean(&flat);
                }
                0
            }
            Err(e) => {
                set_last_error(&e.to_string());
                -2
            }
        }
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_transform_coordinates");
            -99
        }
    }
}
