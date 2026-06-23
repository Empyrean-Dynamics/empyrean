//! Version reporting for the empyrean stack.
//!
//! Mirrors `empyrean_core::{Versions, version_string}` across the C ABI
//! so callers can verify build provenance at runtime.

use crate::error::{Error, Result};
use std::ffi::CStr;

/// Per-crate versions reported by the empyrean stack.
///
/// `empyrean_core` is its own crate's semver (from `Cargo.toml`); the
/// upstream physics crates (`villeneuve`, `scott`, `nolan`) are
/// git-populated `<tag>+<sha>` strings precise to the commit that
/// the deployed cdylib was built against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Versions {
    /// `empyrean-core` crate version.
    pub empyrean_core: String,
    /// `villeneuve` crate version.
    pub villeneuve: String,
    /// `scott` crate version.
    pub scott: String,
    /// `nolan` crate version.
    pub nolan: String,
}

/// Multi-line version report — `empyrean-core <ver>\nvilleneuve <ver>\n…`.
///
/// Use for `--version`-style output and for verifying the build
/// provenance of a deployed cdylib.
pub fn version_string() -> Result<String> {
    let raw = unsafe { empyrean_sys::empyrean_version_string() };
    if raw.is_null() {
        return Err(Error::capture(-1));
    }
    let s = unsafe { CStr::from_ptr(raw) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|_| Error::invalid_input("version string is not valid UTF-8"));
    unsafe { empyrean_sys::empyrean_string_free(raw) };
    s
}

/// Per-crate version strings for the empyrean stack.
pub fn versions() -> Result<Versions> {
    let mut raw = empyrean_sys::EmpyreanVersions {
        empyrean_core: std::ptr::null_mut(),
        villeneuve: std::ptr::null_mut(),
        scott: std::ptr::null_mut(),
        nolan: std::ptr::null_mut(),
    };
    let code = unsafe { empyrean_sys::empyrean_versions(&mut raw) };
    if code != 0 {
        return Err(Error::capture(code));
    }
    // Owned struct — copy the strings out and let
    // `empyrean_versions_free` reclaim the heap allocations.
    let take = |p: *mut std::ffi::c_char| -> Result<String> {
        if p.is_null() {
            return Err(Error::invalid_input("version field is null"));
        }
        unsafe { CStr::from_ptr(p) }
            .to_str()
            .map(|s| s.to_string())
            .map_err(|_| Error::invalid_input("version field is not valid UTF-8"))
    };
    let result = (|| {
        Ok(Versions {
            empyrean_core: take(raw.empyrean_core)?,
            villeneuve: take(raw.villeneuve)?,
            scott: take(raw.scott)?,
            nolan: take(raw.nolan)?,
        })
    })();
    unsafe { empyrean_sys::empyrean_versions_free(&mut raw) };
    result
}
