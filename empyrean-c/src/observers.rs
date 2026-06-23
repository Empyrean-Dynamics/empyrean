use std::ffi::CStr;
use std::panic::AssertUnwindSafe;

use empyrean_core::time::Epoch;

use crate::{EmpyreanContext, set_last_error};

// ── C-compatible types ──────────────────────────────────────

/// A single observer state for the C API.
#[repr(C)]
pub struct EmpyreanObserver {
    /// MPC 3-character code, null-terminated (4 bytes).
    pub obs_code: [u8; 4],
    /// Epoch as MJD TDB.
    pub epoch_mjd_tdb: f64,
    /// Position in ICRF relative to SSB (AU).
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Velocity in ICRF relative to SSB (AU/day).
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    /// Observing night as YYYYMMDD integer, or -1 if unavailable.
    pub observing_night: i32,
}

/// Result containing an array of observer states.
#[repr(C)]
pub struct EmpyreanObserverResult {
    pub observers: *mut EmpyreanObserver,
    pub num_observers: usize,
}

// ── empyrean_get_observers ──────────────────────────────────

/// Compute observer states for given observatory codes and epochs.
///
/// Returns 0 on success, negative error code on failure.
/// On success, `result_out` is populated with observer states.
/// The caller must free the result with `empyrean_observer_result_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_get_observers(
    ctx: *const EmpyreanContext,
    obs_codes: *const *const std::ffi::c_char,
    num_codes: usize,
    epochs_mjd_tdb: *const f64,
    num_epochs: usize,
    result_out: *mut EmpyreanObserverResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Null-pointer checks
        if ctx.is_null() {
            set_last_error("null context pointer");
            return -1;
        }
        if obs_codes.is_null() {
            set_last_error("null obs_codes pointer");
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
        let codes_ptrs = unsafe { std::slice::from_raw_parts(obs_codes, num_codes) };
        let epochs_raw = unsafe { std::slice::from_raw_parts(epochs_mjd_tdb, num_epochs) };

        // Convert C strings to &str
        let mut code_strs: Vec<&str> = Vec::with_capacity(num_codes);
        for &ptr in codes_ptrs {
            if ptr.is_null() {
                set_last_error("null observatory code string");
                return -1;
            }
            let c_str = unsafe { CStr::from_ptr(ptr) };
            match c_str.to_str() {
                Ok(s) => code_strs.push(s),
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in observatory code: {e}"));
                    return -1;
                }
            }
        }

        let epochs: Vec<Epoch> = epochs_raw.iter().map(|&t| Epoch::from_mjd_tdb(t)).collect();

        let observers = match ctx_ref.get_observers(&code_strs, &epochs) {
            Ok(obs) => obs,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };

        let n = observers.len();

        let out_ptr = if n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanObserver>(n)
                .unwrap_or(std::alloc::Layout::new::<EmpyreanObserver>());
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObserver;
            if ptr.is_null() {
                set_last_error("allocation failed for observers array");
                return -5;
            }
            ptr
        } else {
            std::ptr::null_mut()
        };

        for (i, obs) in observers.iter().enumerate() {
            // Observer.code is [u8; 3]; pad to 4 with null terminator.
            let mut obs_code = [0u8; 4];
            obs_code[0] = obs.code[0];
            obs_code[1] = obs.code[1];
            obs_code[2] = obs.code[2];
            obs_code[3] = 0;

            let observing_night = obs.observing_night.map(|n| n as i32).unwrap_or(-1);
            let pos = obs.position();
            let vel = obs.velocity();

            let entry = EmpyreanObserver {
                obs_code,
                epoch_mjd_tdb: obs.epoch().mjd_tdb(),
                x: pos[0],
                y: pos[1],
                z: pos[2],
                vx: vel[0],
                vy: vel[1],
                vz: vel[2],
                observing_night,
            };

            unsafe {
                out_ptr.add(i).write(entry);
            }
        }

        unsafe {
            (*result_out).observers = out_ptr;
            (*result_out).num_observers = n;
        }

        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_get_observers");
            -99
        }
    }
}

/// Free an observer result previously returned by `empyrean_get_observers()`.
///
/// Passing a zeroed/null result is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_observer_result_free(result: *mut EmpyreanObserverResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }

        let res = unsafe { &*result };
        let n = res.num_observers;

        if !res.observers.is_null() && n > 0 {
            let layout = std::alloc::Layout::array::<EmpyreanObserver>(n).unwrap();
            unsafe {
                std::alloc::dealloc(res.observers as *mut u8, layout);
            }
        }

        unsafe {
            (*result).observers = std::ptr::null_mut();
            (*result).num_observers = 0;
        }
    }));
}
