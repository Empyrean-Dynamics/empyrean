//! Thread-safe empyrean context (SPK, GM, ephemeris state).

use crate::error::{Error, Result};
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

/// Handle to loaded SPICE kernels, gravitational parameters, and ephemeris
/// state required for every propagation, ephemeris, or OD call.
///
/// Construct with [`Context::from_data_dir`] for the production path
/// (loads the full Standard-tier kernel set). The underlying
/// libempyrean resources are released when the `Context` is dropped.
/// `Context` is `Send + Sync` — a single instance can be shared
/// across threads.
pub struct Context {
    raw: NonNull<empyrean_sys::EmpyreanContext>,
}

// Safety: libempyrean documents its Context as `Send + Sync` (read-only
// after construction; concurrent propagation calls are safe).
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Load a **minimal** `Context` from a DE440 SPK file and a GM TPC file.
    ///
    /// Loads ONLY the planetary ephemeris and gravitational
    /// parameters — no Earth/Moon BPC kernels, no asteroid perturbers,
    /// no MPC observatory codes, no Earth gravity field. Sufficient
    /// for coordinate transforms and basic propagation under the
    /// `Approximate` force model only. **Not enough** for production
    /// orbit propagation, OD, or topocentric ephemeris generation —
    /// most callers should use [`Context::from_data_dir`], which loads
    /// the full Standard-tier kernel set (downloading any missing
    /// files on first use).
    pub fn new_minimal(de440_path: impl AsRef<Path>, gm_path: impl AsRef<Path>) -> Result<Self> {
        let de440_c = path_to_cstring(de440_path.as_ref())?;
        let gm_c = path_to_cstring(gm_path.as_ref())?;
        let raw =
            unsafe { empyrean_sys::empyrean_context_new_minimal(de440_c.as_ptr(), gm_c.as_ptr()) };
        NonNull::new(raw)
            .map(|raw| Context { raw })
            .ok_or_else(Error::from_null_ptr)
    }

    /// Load a `Context` from a directory containing the kernel files.
    ///
    /// Loads the full Standard-tier kernel set (DE440, SB441-N16,
    /// Earth/Moon BPCs, GM, MPC observatory codes) — downloading any
    /// missing files. Pass `None` to use the platform XDG data directory
    /// (`~/.empyrean/data` on Linux/macOS).
    pub fn from_data_dir(data_dir: Option<&Path>) -> Result<Self> {
        let c_path = match data_dir {
            Some(d) => Some(path_to_cstring(d)?),
            None => None,
        };
        let raw_path = c_path
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());
        let raw = unsafe { empyrean_sys::empyrean_context_from_data_dir(raw_path) };
        NonNull::new(raw)
            .map(|raw| Context { raw })
            .ok_or_else(Error::from_null_ptr)
    }

    /// Load an additional SPK file in place, layering its body
    /// coverage on top of what is already loaded.
    ///
    /// Use to attach spacecraft SPK kernels (JWST, Gaia, custom
    /// probes) or asteroid perturber sets (SB441-N16) onto a context
    /// built by [`Context::new_minimal`] or
    /// [`Context::from_data_dir`].
    pub fn with_spk(&mut self, spk_path: impl AsRef<Path>) -> Result<()> {
        let c_path = path_to_cstring(spk_path.as_ref())?;
        let code =
            unsafe { empyrean_sys::empyrean_context_with_spk(self.raw.as_ptr(), c_path.as_ptr()) };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(())
    }

    /// Borrow the raw FFI context pointer (internal use).
    pub(crate) fn as_raw(&self) -> *const empyrean_sys::EmpyreanContext {
        self.raw.as_ptr()
    }
}

/// Return the platform XDG-compliant default data directory.
///
/// Honors `EMPYREAN_DATA_DIR` first, then falls back to the platform
/// XDG data dir — `~/.local/share/empyrean/data/` on Linux,
/// `~/Library/Application Support/empyrean/data/` on macOS,
/// `%APPDATA%\empyrean\data\` on Windows. Cheap (no filesystem I/O).
///
/// This is the same path [`Context::from_data_dir`] writes kernels
/// to when called with `None`.
pub fn default_data_dir() -> Result<std::path::PathBuf> {
    let raw = unsafe { empyrean_sys::empyrean_default_data_dir() };
    if raw.is_null() {
        return Err(Error::capture(-1));
    }
    let path = unsafe { CStr::from_ptr(raw) }
        .to_str()
        .map(std::path::PathBuf::from)
        .map_err(|_| Error::invalid_input("default data dir is not valid UTF-8"));
    unsafe { empyrean_sys::empyrean_string_free(raw) };
    path
}

/// Resolve the configured data directory.
///
/// Returns `data_dir` if provided, otherwise [`default_data_dir`]. Does
/// not download anything — kernel discovery is the caller's job.
pub fn download_data(data_dir: Option<&Path>) -> Result<std::path::PathBuf> {
    if let Some(d) = data_dir {
        return Ok(d.to_path_buf());
    }
    default_data_dir()
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { empyrean_sys::empyrean_context_free(self.raw.as_ptr()) }
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let bytes = path
        .to_str()
        .ok_or_else(|| Error::invalid_input("path is not valid UTF-8"))?
        .as_bytes();
    CString::new(bytes).map_err(|_| Error::invalid_input("path contains a NUL byte"))
}
