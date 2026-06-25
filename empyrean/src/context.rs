//! Thread-safe empyrean context (SPK, GM, ephemeris state).

use crate::error::{Error, Result, dedupe_io_prefix};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Handle to loaded SPICE kernels, gravitational parameters, and ephemeris
/// state required for every propagation, ephemeris, or OD call.
///
/// Construct with [`Context::from_data_dir`] for the production path
/// (loads the full Standard-tier kernel set). The underlying
/// libempyrean resources are released when the `Context` is dropped.
/// `Context` is `Send + Sync` — a single instance can be shared
/// across threads, and contexts may be constructed concurrently
/// (libempyrean serializes native construction at the C ABI).
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
        NonNull::new(raw).map(|raw| Context { raw }).ok_or_else(|| {
            let mut err = Error::from_null_ptr();
            err.message = dedupe_io_prefix(&err.message);
            err
        })
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
            .ok_or_else(|| construction_error(data_dir))
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
            let mut err = Error::capture(code);
            err.message = dedupe_io_prefix(&err.message);
            return Err(err);
        }
        Ok(())
    }

    /// Borrow the raw FFI context pointer (internal use).
    pub(crate) fn as_raw(&self) -> *const empyrean_sys::EmpyreanContext {
        self.raw.as_ptr()
    }
}

/// Stable-named core kernels every Standard-tier load needs. Used only to turn
/// an opaque construction failure into an actionable one — NOT an authoritative
/// manifest (the engine owns that, and the dated Earth/Moon orientation kernels
/// are intentionally omitted because their filenames change each release). This
/// is a best-effort "is this directory provisioned at all?" probe.
const CORE_KERNELS: &[&str] = &[
    "de440.bsp",
    "gm_de440.tpc",
    "sb441-n16.bsp",
    "obscodes_extended.json",
    "earth_latest_high_prec.bpc",
    "bias.dat",
];

/// First core kernel absent from `dir`, if any.
fn first_missing_core_kernel(dir: &Path) -> Option<&'static str> {
    CORE_KERNELS.iter().copied().find(|f| !dir.join(f).exists())
}

/// Append the resolved data directory (and, if a core kernel is absent, an
/// actionable remedy) to a construction error's (already de-duplicated) message.
///
/// The native cause is kept up front, so the real failure reason is never hidden
/// behind the remedy. Phrased non-committally — an absent core kernel is one
/// likely cause, not asserted as *the* cause.
fn augment_with_data_dir(base: &str, dir: &Path, missing: Option<&str>) -> String {
    match missing {
        Some(file) => format!(
            "{base} — empyrean data directory '{}' may be incompletely provisioned \
             (required kernel '{file}' is absent). Run `empyrean::download_data(None)` to \
             provision it, or set EMPYREAN_DATA_DIR to a directory that already contains \
             the kernels.",
            dir.display(),
        ),
        None => format!("{base} (data directory: '{}')", dir.display()),
    }
}

/// Build an actionable error for a failed native context construction:
/// de-duplicate the engine's doubly-wrapped `I/O error:` prefix and, when the
/// resolved data directory is missing required kernels, name the file and the
/// remedy instead of returning the path-less native message.
fn construction_error(data_dir: Option<&Path>) -> Error {
    let mut err = Error::from_null_ptr();
    err.message = dedupe_io_prefix(&err.message);
    let resolved = data_dir
        .map(Path::to_path_buf)
        .or_else(|| default_data_dir().ok());
    if let Some(dir) = resolved {
        let missing = first_missing_core_kernel(&dir);
        if missing.is_some() {
            // Categorize as missing-data rather than the generic invalid-argument
            // code `from_null_ptr` defaults to, so `err.code` matches the message.
            err.code = -2;
        }
        err.message = augment_with_data_dir(&err.message, &dir, missing);
    }
    err
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

/// Provision the complete Standard-tier kernel set into `data_dir` (or the
/// platform [`default_data_dir`] when `None`) and return the resolved directory.
///
/// **Idempotent:** files already present are kept; only missing files are
/// downloaded. After this returns, a [`Context::from_data_dir`] over the same
/// directory loads with no further downloads. Safe to call concurrently —
/// construction is serialized internally.
///
/// This drives provisioning by briefly building (and discarding) a full
/// Standard-tier context, so it loads the kernel set into memory once. A
/// lightweight download-only path is a planned follow-up.
///
/// ```no_run
/// # fn main() -> Result<(), empyrean::Error> {
/// let dir = empyrean::download_data(None)?; // ensures a usable data directory
/// let _ctx = empyrean::Context::from_data_dir(Some(&dir))?;
/// # Ok(())
/// # }
/// ```
pub fn download_data(data_dir: Option<&Path>) -> Result<PathBuf> {
    // The engine downloads any missing kernels as part of building a
    // Standard-tier context. Build one to drive provisioning, then discard it
    // and return the resolved directory.
    let _ctx = Context::from_data_dir(data_dir)?;
    match data_dir {
        Some(d) => Ok(d.to_path_buf()),
        None => default_data_dir(),
    }
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

#[cfg(test)]
mod tests {
    use super::{CORE_KERNELS, augment_with_data_dir, first_missing_core_kernel};
    use std::path::Path;

    #[test]
    fn construction_message_assembly() {
        // Missing kernel: message names the file, the dir, and both remedies,
        // and keeps the native cause up front.
        let m = augment_with_data_dir("I/O error: nope", Path::new("/tmp/dd"), Some("bias.dat"));
        assert!(
            m.starts_with("I/O error: nope"),
            "native cause kept up front: {m}"
        );
        assert!(m.contains("bias.dat"), "names the missing kernel: {m}");
        assert!(m.contains("/tmp/dd"), "names the data directory: {m}");
        assert!(
            m.contains("download_data"),
            "hints the download remedy: {m}"
        );
        assert!(
            m.contains("EMPYREAN_DATA_DIR"),
            "hints the env-var remedy: {m}"
        );

        // Nothing missing: just the dir, no (possibly-wrong) kernel remedy.
        let g = augment_with_data_dir("boom", Path::new("/tmp/dd"), None);
        assert!(g.contains("/tmp/dd"));
        assert!(!g.contains("download_data"));
    }

    #[test]
    fn missing_core_kernel_probe() {
        let tmp = std::env::temp_dir().join(format!("empyrean-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Empty directory: the first core kernel is reported missing.
        assert_eq!(first_missing_core_kernel(&tmp), Some(CORE_KERNELS[0]));

        // All present: nothing missing.
        for k in CORE_KERNELS {
            std::fs::write(tmp.join(k), b"x").unwrap();
        }
        assert_eq!(first_missing_core_kernel(&tmp), None);

        // Remove one: that exact file is reported.
        std::fs::remove_file(tmp.join("sb441-n16.bsp")).unwrap();
        assert_eq!(first_missing_core_kernel(&tmp), Some("sb441-n16.bsp"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
