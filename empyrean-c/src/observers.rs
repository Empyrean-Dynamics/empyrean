use std::ffi::CStr;
use std::panic::AssertUnwindSafe;

use empyrean_core::time::Epoch;

use crate::{EmpyreanContext, set_last_error};

// ── C-compatible types ──────────────────────────────────────

/// A single observer state for the C API.
///
/// The state is expressed in the `(frame, origin)` basis carried in the
/// two tail fields. Before ABI 3 the basis was an undeclared ICRF / SSB
/// assumption baked into the field docs; it is now on the struct, so a
/// consumer reading a row never has to infer which basis produced it.
#[repr(C)]
pub struct EmpyreanObserver {
    /// MPC 3-character code, null-terminated (4 bytes).
    pub obs_code: [u8; 4],
    /// Epoch as MJD TDB.
    pub epoch_mjd_tdb: f64,
    /// Position in the `(frame, origin)` basis below (AU).
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Velocity in the `(frame, origin)` basis below (AU/day).
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    /// Observing night as YYYYMMDD integer, or -1 if unavailable.
    pub observing_night: i32,
    /// Reference frame the state is expressed in: 0=ICRF,
    /// 1=EclipticJ2000 (same encoding as
    /// `EmpyreanPropagationConfig::frame`).
    ///
    /// Appended at the tail of the struct deliberately: growing a
    /// `#[repr(C)]` record at the end is the only way to extend it
    /// without moving an existing field's offset.
    pub frame: i32,
    /// NAIF id of the body the state is relative to (0 = solar-system
    /// barycenter, 399 = Earth, …), same encoding as
    /// `EmpyreanPropagatedState::origin`.
    pub origin: i32,
}

/// Result containing an array of observer states.
#[repr(C)]
pub struct EmpyreanObserverResult {
    pub observers: *mut EmpyreanObserver,
    pub num_observers: usize,
}

// ── empyrean_get_observers ──────────────────────────────────

/// Compute observer states for given observatory codes and epochs, in a
/// caller-chosen `(frame, origin)` basis.
///
/// The result is the Cartesian product `obs_codes × epochs`, **code-major**:
/// all epochs for `obs_codes[0]`, then all epochs for `obs_codes[1]`, so
/// `observers[i * num_epochs + j]` is `(obs_codes[i], epochs[j])`.
///
/// # Choosing a basis
///
/// `frame = 0` (ICRF) with `origin = 0` (solar-system barycenter) is the
/// **construction basis** — the one every consumer of an observer state
/// requires, and the one to pass when the observers are headed for
/// `empyrean_generate_ephemeris`,
/// `empyrean_builtsystem_generate_ephemeris`, or orbit determination.
/// Requesting it takes no transform at all: the observers come back
/// exactly as constructed, bit for bit. Any other basis rotates and/or
/// translates them, which is what a consumer plotting observer geometry
/// wants (e.g. heliocentric ecliptic site positions via
/// `frame = 1`, `origin = 10`).
///
/// Each returned row carries the basis it is expressed in
/// ([`EmpyreanObserver::frame`] / [`EmpyreanObserver::origin`]), so a
/// row is never ambiguous about which basis produced it.
///
/// # Returns
///
/// `0` on success; `result_out` is populated and the caller must free it
/// with `empyrean_observer_result_free()`.
///
/// Every rejection aborts the whole call — there is no partial output and
/// no silently untransformed entry. The failure codes are split by
/// **remedy**, not flattened, because a channel that collapses them tells
/// an operator to fix their input when the fix is on disk:
///
/// - `-1` — retry with different arguments: an observatory code is not in
///   the registry; `frame` is one an observer cannot be rotated into
///   (anything outside ICRF ↔ EclipticJ2000); `origin` names an MPC site
///   rather than an SPK body; a null pointer or non-UTF-8 code.
/// - `-2` — load or fetch data, the arguments are fine: the code names a
///   space telescope whose SPK is not loaded; `origin` has no SPK
///   coverage at a requested epoch; an epoch falls outside the loaded
///   BPC's window, or this context carries no BPC / no observatory
///   registry at all.
/// - `-5` — allocation failure. `-99` — a panic was caught at the boundary.
///
/// Call `empyrean_last_error()` for the message in every failing case.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_get_observers(
    ctx: *const EmpyreanContext,
    obs_codes: *const *const std::ffi::c_char,
    num_codes: usize,
    epochs_mjd_tdb: *const f64,
    num_epochs: usize,
    frame: i32,
    origin: i32,
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

        let frame_enum = match empyrean_core::convert::int_to_frame(frame) {
            Ok(f) => f,
            Err(e) => {
                set_last_error(&e.to_string());
                return -1;
            }
        };
        let origin_enum = match empyrean_core::Origin::from_naif_id(origin) {
            Some(o) => o,
            None => {
                set_last_error(&format!("unknown NAIF id for observer origin: {origin}"));
                return -1;
            }
        };

        let observers = match ctx_ref.get_observers(&code_strs, &epochs, frame_enum, origin_enum) {
            Ok(obs) => obs,
            // The engine splits its rejections across two axes by remedy
            // (bad argument vs absent data) and carries the split in the
            // error's own category code. Forward that code instead of
            // flattening every failure onto -1 — the two have different
            // fixes, and a caller told "invalid argument" for an
            // unfetched spacecraft kernel goes looking in the wrong place.
            Err(e) => {
                set_last_error(&e.to_string());
                return e.error_code();
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

            // Read the components straight off the state, NOT through
            // `Observer::position` / `Observer::velocity`. Those two
            // accessors are documented to *panic* on anything that is not
            // ICRF / SSB — they exist for the ephemeris pipeline, which
            // requires the construction basis. This entry point is
            // basis-taking by definition, so any non-default request
            // would abort inside the accessor and surface as a -99 panic.
            // The state's own fields carry the same numbers in whatever
            // basis the engine produced, and the tags below say which.
            //
            // The basis is read off the returned state, not echoed from
            // the request: the row reports the basis it is actually
            // expressed in, so a future engine-side deviation surfaces as
            // a mismatched tag rather than a correct-looking label on the
            // wrong numbers.
            let entry = EmpyreanObserver {
                obs_code,
                epoch_mjd_tdb: obs.epoch().mjd_tdb(),
                x: obs.state.x,
                y: obs.state.y,
                z: obs.state.z,
                vx: obs.state.vx,
                vy: obs.state.vy,
                vz: obs.state.vz,
                observing_night,
                frame: empyrean_core::convert::frame_to_int(obs.state.frame),
                origin: match obs.state.origin.try_naif_id() {
                    Some(id) => id,
                    None => {
                        set_last_error(&format!(
                            "observer {} came back on an origin with no NAIF id \
                             (an MPC site code), which cannot be marshaled",
                            String::from_utf8_lossy(&obs.code)
                        ));
                        // The array is raw-allocated and partially
                        // written; release it rather than leak, then fail.
                        unsafe {
                            std::alloc::dealloc(
                                out_ptr as *mut u8,
                                std::alloc::Layout::array::<EmpyreanObserver>(n).unwrap(),
                            );
                        }
                        return -1;
                    }
                },
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

#[cfg(test)]
mod observer_basis_tests {
    use super::*;
    use std::ffi::CString;

    /// One marshaled row, reduced to what these tests compare.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Row {
        position: [f64; 3],
        velocity: [f64; 3],
        frame: i32,
        origin: i32,
    }

    /// Drive the real C entry point end to end, returning its code and
    /// whatever rows it produced. The result buffer is always freed.
    fn call(
        ctx: &EmpyreanContext,
        codes: &[&str],
        epochs: &[f64],
        frame: i32,
        origin: i32,
    ) -> (i32, Vec<Row>) {
        let cstrings: Vec<CString> = codes.iter().map(|c| CString::new(*c).unwrap()).collect();
        let ptrs: Vec<*const std::ffi::c_char> = cstrings.iter().map(|s| s.as_ptr()).collect();
        let mut result = EmpyreanObserverResult {
            observers: std::ptr::null_mut(),
            num_observers: 0,
        };
        let code = unsafe {
            empyrean_get_observers(
                ctx,
                ptrs.as_ptr(),
                ptrs.len(),
                epochs.as_ptr(),
                epochs.len(),
                frame,
                origin,
                &mut result,
            )
        };
        let rows = if code == 0 && !result.observers.is_null() {
            unsafe { std::slice::from_raw_parts(result.observers, result.num_observers) }
                .iter()
                .map(|o| Row {
                    position: [o.x, o.y, o.z],
                    velocity: [o.vx, o.vy, o.vz],
                    frame: o.frame,
                    origin: o.origin,
                })
                .collect()
        } else {
            Vec::new()
        };
        unsafe { empyrean_observer_result_free(&mut result) };
        (code, rows)
    }

    fn rows(
        ctx: &EmpyreanContext,
        codes: &[&str],
        epochs: &[f64],
        frame: i32,
        origin: i32,
    ) -> Vec<Row> {
        let (code, rows) = call(ctx, codes, epochs, frame, origin);
        assert_eq!(code, 0, "empyrean_get_observers failed: {code}");
        rows
    }

    /// A non-default basis must come back as numbers, not a panic.
    ///
    /// Regression: the first cut of the widened entry point marshaled
    /// through `Observer::position` / `Observer::velocity`, which are
    /// documented to **panic** on anything that is not ICRF / SSB. Every
    /// non-default request therefore unwound into the boundary's
    /// `catch_unwind` and came back as `-99`, making the widening
    /// unusable for the one thing it was widened for.
    #[test]
    fn a_non_default_basis_returns_states_not_a_panic() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping a_non_default_basis_returns_states_not_a_panic: no data dir");
            return;
        };
        // EclipticJ2000 (frame 1) about the Sun (NAIF 10): both axes move.
        let (code, rows) = call(&ctx, &["500"], &[60000.0], 1, 10);
        assert_eq!(
            code, 0,
            "a heliocentric-ecliptic request must succeed; -99 means the \
             marshaling went through an ICRF-asserting accessor"
        );
        assert_eq!(rows.len(), 1);
        assert!(rows[0].position.iter().all(|v| v.is_finite()));
        assert!(rows[0].velocity.iter().all(|v| v.is_finite()));
    }

    /// The construction basis is returned untouched, and any other basis
    /// genuinely moves the state — the request is honoured, not ignored.
    #[test]
    fn the_requested_basis_is_honoured_and_tagged() {
        let Ok(ctx) = empyrean_core::Context::from_data_dir(None) else {
            eprintln!("skipping the_requested_basis_is_honoured_and_tagged: no data dir");
            return;
        };
        let icrf_ssb = rows(&ctx, &["500", "W84"], &[60000.0, 60001.0], 0, 0);
        let ecl_sun = rows(&ctx, &["500", "W84"], &[60000.0, 60001.0], 1, 10);
        assert_eq!(icrf_ssb.len(), 4);
        assert_eq!(ecl_sun.len(), 4);

        for (i, r) in icrf_ssb.iter().enumerate() {
            assert_eq!(
                (r.frame, r.origin),
                (0, 0),
                "row {i}: ICRF/SSB must be tagged 0/0"
            );
        }
        for (i, r) in ecl_sun.iter().enumerate() {
            assert_eq!(
                (r.frame, r.origin),
                (1, 10),
                "row {i}: the row must report the basis it is expressed in"
            );
        }
        for (i, (a, b)) in icrf_ssb.iter().zip(ecl_sun.iter()).enumerate() {
            assert_ne!(
                a.position, b.position,
                "row {i}: a heliocentric-ecliptic request that returns the \
                 barycentric-ICRF numbers means the basis was ignored"
            );
        }
    }
}
