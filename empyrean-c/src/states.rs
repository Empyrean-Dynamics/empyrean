use std::panic::AssertUnwindSafe;

use empyrean_core::Origin;
use empyrean_core::convert::{frame_to_int, int_to_frame};
use empyrean_core::ephemeris::body_states;
use empyrean_core::time::Epoch;

use crate::{EmpyreanContext, set_last_error};

// ── C-compatible types ──────────────────────────────────────

/// A single body state for the C API.
#[repr(C)]
pub struct EmpyreanState {
    /// Epoch as MJD TDB.
    pub epoch_mjd_tdb: f64,
    /// Position (AU).
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Velocity (AU/day).
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    /// Reference frame (EMPYREAN_FRAME_*).
    pub frame: i32,
    /// Center body NAIF ID.
    pub origin: i32,
}

/// Result containing an array of body states.
#[repr(C)]
pub struct EmpyreanStateResult {
    pub states: *mut EmpyreanState,
    pub num_states: usize,
}

// ── empyrean_get_states ─────────────────────────────────────

/// Query body states relative to a center body at given epochs.
///
/// Returns 0 on success, negative error code on failure.
/// On success, `result_out` is populated with body states.
/// The caller must free the result with `empyrean_state_result_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_get_states(
    ctx: *const EmpyreanContext,
    target_naif_id: i32,
    center_naif_id: i32,
    epochs_mjd_tdb: *const f64,
    num_epochs: usize,
    frame: i32,
    result_out: *mut EmpyreanStateResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Null-pointer checks
        if ctx.is_null() {
            set_last_error("null context pointer");
            return -1;
        }
        if epochs_mjd_tdb.is_null() {
            set_last_error("null epochs pointer");
            return -1;
        }
        if result_out.is_null() {
            set_last_error("null result_out pointer");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let epochs_raw = unsafe { std::slice::from_raw_parts(epochs_mjd_tdb, num_epochs) };

        let target = match Origin::from_naif_id(target_naif_id) {
            Some(o) => o,
            None => {
                set_last_error(&format!("unknown target NAIF id: {target_naif_id}"));
                return -1;
            }
        };
        let center = match Origin::from_naif_id(center_naif_id) {
            Some(o) => o,
            None => {
                set_last_error(&format!("unknown center NAIF id: {center_naif_id}"));
                return -1;
            }
        };
        let e_frame = match int_to_frame(frame) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };

        let epochs: Vec<Epoch> = epochs_raw.iter().map(|&t| Epoch::from_mjd_tdb(t)).collect();

        let states = match body_states(ctx_ref, target, center, &epochs, e_frame) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e.to_string());
                return -2;
            }
        };

        let n = states.len();
        let frame_out = frame_to_int(e_frame);
        let center_out = center.naif_id();

        let out_ptr = if n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanState>(n)
                .unwrap_or(std::alloc::Layout::new::<EmpyreanState>());
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanState;
            if ptr.is_null() {
                set_last_error("allocation failed for states array");
                return -5;
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        for (i, (_id, coord, _cov)) in states.iter().enumerate() {
            let entry = EmpyreanState {
                epoch_mjd_tdb: coord.t.mjd_tdb(),
                x: coord.x,
                y: coord.y,
                z: coord.z,
                vx: coord.vx,
                vy: coord.vy,
                vz: coord.vz,
                frame: frame_out,
                origin: center_out,
            };
            unsafe {
                out_ptr.add(i).write(entry);
            }
        }

        unsafe {
            (*result_out).states = out_ptr;
            (*result_out).num_states = n;
        }

        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_get_states");
            -99
        }
    }
}

/// Free a state result previously returned by `empyrean_get_states()`.
///
/// Passing a zeroed/null result is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_state_result_free(result: *mut EmpyreanStateResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }

        let res = unsafe { &*result };
        let n = res.num_states;

        if !res.states.is_null() && n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanState>(n).unwrap();
            unsafe {
                std::alloc::dealloc(res.states as *mut u8, layout);
            }
        }

        unsafe {
            (*result).states = std::ptr::null_mut();
            (*result).num_states = 0;
        }
    }));
}
