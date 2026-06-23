//! C ABI exports for ISO 8601 ↔ MJD conversion.
//!
//! Native (no astropy detour) round-trip between ISO 8601 UTC strings
//! and MJD values in either UTC or TDB. Backed by villeneuve's
//! leap-second + Fairhead & Bretagnon (1990) TDB-TT secular term.

use std::ffi::{CStr, c_char};
use std::panic::AssertUnwindSafe;

use empyrean_core::time::Epoch;

use crate::set_last_error;

const SCALE_UTC: i32 = 0;
const SCALE_TDB: i32 = 1;

/// Local time scale tag for FFI dispatch only. We deliberately do not
/// depend on `villeneuve::time::TimeScale` here — the C-ABI crate sees
/// only the (closed-source) `empyrean_core` re-exports.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scale {
    Utc,
    Tdb,
}

fn int_to_scale(val: i32) -> Result<Scale, String> {
    match val {
        SCALE_UTC => Ok(Scale::Utc),
        SCALE_TDB => Ok(Scale::Tdb),
        other => Err(format!(
            "unknown time scale: {other} (expected 0=UTC, 1=TDB)"
        )),
    }
}

/// Parse an ISO 8601 UTC string (e.g. ``"2024-08-01T00:00:00.000Z"``)
/// to MJD in the requested target scale.
///
/// `scale` is `0` for UTC, `1` for TDB.
///
/// On success writes the MJD value to `*out_mjd` and returns 0.
/// On failure returns a negative code; consult
/// [`empyrean_last_error`](crate::empyrean_last_error).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_iso_to_mjd(
    iso: *const c_char,
    scale: i32,
    out_mjd: *mut f64,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if iso.is_null() || out_mjd.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let iso_str = match unsafe { CStr::from_ptr(iso) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in iso: {e}"));
                return -1;
            }
        };
        let target = match int_to_scale(scale) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let utc_epoch = match Epoch::from_iso_utc(iso_str) {
            Ok(e) => e,
            Err(e) => {
                set_last_error(&format!("ISO parse failed: {e}"));
                return -2;
            }
        };
        let mjd = match target {
            Scale::Utc => utc_epoch.mjd(),
            Scale::Tdb => match utc_epoch.to_tdb() {
                Ok(e) => e.mjd(),
                Err(e) => {
                    set_last_error(&format!("UTC→TDB conversion failed: {e}"));
                    return -3;
                }
            },
        };
        unsafe { *out_mjd = mjd };
        0
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_iso_to_mjd");
            -99
        }
    }
}

/// Format an MJD value (in the given scale) as an ISO 8601 UTC string.
///
/// `scale` is `0` for UTC, `1` for TDB. Writes a null-terminated
/// string of length ≤ `buf_len-1` into `out_buf`. A 32-byte buffer is
/// always sufficient (typical output is 24 bytes:
/// ``"2024-08-01T00:00:00.000Z"``).
///
/// Returns 0 on success; negative on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_mjd_to_iso(
    mjd: f64,
    scale: i32,
    out_buf: *mut c_char,
    buf_len: usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if out_buf.is_null() || buf_len == 0 {
            set_last_error("null or zero-length output buffer");
            return -1;
        }
        let source = match int_to_scale(scale) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };
        let epoch = match source {
            Scale::Utc => Epoch::from_mjd_utc(mjd),
            Scale::Tdb => Epoch::from_mjd_tdb(mjd),
        };
        let iso = epoch.to_iso_utc();
        let bytes = iso.as_bytes();
        if bytes.len() + 1 > buf_len {
            set_last_error(&format!(
                "buffer too small: need {} bytes (incl. NUL), got {buf_len}",
                bytes.len() + 1,
            ));
            return -2;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buf as *mut u8, bytes.len());
            *out_buf.add(bytes.len()) = 0;
        }
        0
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_mjd_to_iso");
            -99
        }
    }
}
