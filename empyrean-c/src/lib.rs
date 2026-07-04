// This crate is the C ABI: every entry point is `unsafe extern "C"` taking
// caller-provided pointers whose validity is the documented contract of the C
// header (`include/empyrean.h`), not a per-fn `# Safety` section. The C->Rust
// config translators also build their structs by reassigning fields on a
// `Default` value (sentinel-aware), which is intentional, not a missed
// struct-update.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::field_reassign_with_default)]

mod built_system;
mod ephemeris;
mod impact;
mod io;
mod math;
mod observers;
mod od;
mod planning;
mod propagate;
mod query;
mod session;
mod states;
mod time;
mod transform;

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::sync::Mutex;

/// Serializes native context CONSTRUCTION across the C ABI.
///
/// The engine's first-init kernel provisioning does writable-cache file I/O
/// that is not safe to run concurrently — building several contexts at once
/// raced and surfaced as a path-less `I/O error: ... (os error 2)`. Guarding
/// the constructors here (rather than in a higher-level wrapper) makes the C ABI
/// itself thread-safe, so every consumer — the Rust wrapper, the Python package,
/// and direct C SDK users — is protected. It guards construction / in-place
/// kernel loading ONLY; propagation, ephemeris, and OD on a built context are
/// concurrency-safe and never take this lock.
static CONSTRUCT_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the construction lock, recovering from a poisoned mutex (a panic in
/// one constructor must not wedge all future construction).
fn construct_lock() -> std::sync::MutexGuard<'static, ()> {
    CONSTRUCT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// C FFI opaque handle. Internally an [`empyrean_core::Context`]; the
/// C header forward-declares `struct EmpyreanContext` — callers only
/// see the pointer.
pub type EmpyreanContext = empyrean_core::Context;

/// Flat C-ABI compatible coordinate state.
///
/// Field-identical to [`empyrean_core::convert::CoordinateState`]; the
/// duplicate definition exists so cbindgen (which has `parse_deps =
/// false`) can emit the matching C struct in `empyrean.h` without
/// traversing into the empyrean-core crate.
#[repr(C)]
pub struct CoordinateState {
    pub epoch_mjd_tdb: f64,
    pub elements: [f64; 6],
    pub covariance: [[f64; 6]; 6],
    pub has_covariance: u8,
    pub representation: i32,
    pub frame: i32,
    pub origin: i32,
}

impl CoordinateState {
    /// Convert this C-ABI state to an [`empyrean_core::convert::CoordinateState`].
    ///
    /// Field-by-field copy — both structs are `#[repr(C)]` with
    /// identical layouts, but we copy explicitly for clarity instead
    /// of `transmute`.
    pub fn to_empyrean(&self) -> empyrean_core::convert::CoordinateState {
        empyrean_core::convert::CoordinateState {
            epoch_mjd_tdb: self.epoch_mjd_tdb,
            elements: self.elements,
            covariance: self.covariance,
            has_covariance: self.has_covariance,
            representation: self.representation,
            frame: self.frame,
            origin: self.origin,
        }
    }

    /// Build a C-ABI state from an [`empyrean_core::convert::CoordinateState`].
    pub fn from_empyrean(s: &empyrean_core::convert::CoordinateState) -> Self {
        Self {
            epoch_mjd_tdb: s.epoch_mjd_tdb,
            elements: s.elements,
            covariance: s.covariance,
            has_covariance: s.has_covariance,
            representation: s.representation,
            frame: s.frame,
            origin: s.origin,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

pub(crate) fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() =
            CString::new(msg).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    });
}

/// Return a pointer to the last error message (thread-local, null-terminated).
///
/// The pointer is valid until the next call that sets an error on the same
/// thread.
#[unsafe(no_mangle)]
pub extern "C" fn empyrean_last_error() -> *const c_char {
    std::panic::catch_unwind(|| LAST_ERROR.with(|e| e.borrow().as_ptr()))
        .unwrap_or(std::ptr::null())
}

/// Create a **minimal** `EmpyreanContext` from a DE440 SPK file and a
/// GM TPC file.
///
/// Loads ONLY the planetary ephemeris and gravitational parameters —
/// no Earth/Moon BPC kernels, no SB441-N16 asteroid perturbers, no
/// MPC observatory codes, no Earth gravity field. This is sufficient
/// for coordinate transforms and basic propagation under the
/// `Approximate` force model, but is **not** enough for production
/// orbit propagation, orbit determination, or topocentric ephemeris
/// generation. Most callers should use
/// [`empyrean_context_from_data_dir`] instead, which loads the full
/// Standard-tier kernel set (downloading any missing files).
///
/// Use [`empyrean_context_with_spk`] to chain additional SPK kernels
/// (e.g. SB441-N16) onto a context built by this function.
///
/// Returns a heap-allocated pointer on success, or null on error.
/// Call `empyrean_last_error()` to retrieve the error message when null is
/// returned.  The caller owns the returned pointer and must free it with
/// `empyrean_context_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_context_new_minimal(
    de440_path: *const c_char,
    gm_path: *const c_char,
) -> *mut EmpyreanContext {
    let result = std::panic::catch_unwind(|| {
        if de440_path.is_null() || gm_path.is_null() {
            set_last_error("null path argument");
            return std::ptr::null_mut();
        }

        let de440 = unsafe { CStr::from_ptr(de440_path) };
        let gm = unsafe { CStr::from_ptr(gm_path) };

        let de440_str = match de440.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in de440_path: {e}"));
                return std::ptr::null_mut();
            }
        };
        let gm_str = match gm.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in gm_path: {e}"));
                return std::ptr::null_mut();
            }
        };

        let outcome = {
            let _guard = construct_lock();
            empyrean_core::Context::new(Path::new(de440_str), Path::new(gm_str))
        };
        match outcome {
            Ok(ctx) => Box::into_raw(Box::new(ctx)),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("panic in empyrean_context_new_minimal");
            std::ptr::null_mut()
        }
    }
}

/// Load an additional SPK kernel into an existing context, in place.
///
/// Useful for layering SB441-N16 asteroid perturbers or spacecraft
/// SPK kernels (JWST, Gaia, custom probes) on top of a context built
/// by [`empyrean_context_new_minimal`] or [`empyrean_context_from_data_dir`].
/// The merged context picks up the new kernel's body coverage on top
/// of what was already loaded.
///
/// Returns 0 on success; negative error code on failure. The context
/// pointer remains valid and unchanged when this function returns
/// non-zero — failure does not invalidate `ctx`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_context_with_spk(
    ctx: *mut EmpyreanContext,
    spk_path: *const c_char,
) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if ctx.is_null() || spk_path.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        let path_str = match unsafe { CStr::from_ptr(spk_path) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8 in spk_path: {e}"));
                return -1;
            }
        };
        let ctx_ref = unsafe { &mut *ctx };
        let outcome = {
            // In-place kernel loading mutates the shared native pool — serialize
            // it with the other constructors.
            let _guard = construct_lock();
            ctx_ref.load_spk(Path::new(path_str))
        };
        match outcome {
            Ok(()) => 0,
            Err(e) => {
                set_last_error(&format!("load_spk failed: {e}"));
                -2
            }
        }
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_context_with_spk");
            -99
        }
    }
}

/// Create a new `EmpyreanContext` from a data directory.
///
/// Loads the full Standard-tier kernel set (DE440, SB441-N16, Earth/Moon
/// BPCs, GM, MPC observatory codes) from `data_dir`, downloading any
/// missing files. Pass null for `data_dir` to use the platform XDG
/// data directory (`~/.empyrean/data` on Linux/macOS).
///
/// Returns a heap-allocated pointer on success, or null on error.
/// Call `empyrean_last_error()` to retrieve the error message when null is
/// returned. The caller owns the returned pointer and must free it with
/// `empyrean_context_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_context_from_data_dir(
    data_dir: *const c_char,
) -> *mut EmpyreanContext {
    let result = std::panic::catch_unwind(|| {
        let dir_buf;
        let dir_opt: Option<&Path> = if data_dir.is_null() {
            None
        } else {
            let cstr = unsafe { CStr::from_ptr(data_dir) };
            match cstr.to_str() {
                Ok(s) => {
                    dir_buf = std::path::PathBuf::from(s);
                    Some(dir_buf.as_path())
                }
                Err(e) => {
                    set_last_error(&format!("invalid UTF-8 in data_dir: {e}"));
                    return std::ptr::null_mut();
                }
            }
        };

        let outcome = {
            let _guard = construct_lock();
            empyrean_core::Context::from_data_dir(dir_opt)
        };
        match outcome {
            Ok(ctx) => Box::into_raw(Box::new(ctx)),
            Err(e) => {
                set_last_error(&e.to_string());
                std::ptr::null_mut()
            }
        }
    });

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("panic in empyrean_context_from_data_dir");
            std::ptr::null_mut()
        }
    }
}

/// Free an `EmpyreanContext` previously returned by `empyrean_context_new()`.
///
/// Passing null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_context_free(ctx: *mut EmpyreanContext) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !ctx.is_null() {
            unsafe {
                drop(Box::from_raw(ctx));
            }
        }
    }));
}

/// Return the platform XDG-compliant default data directory as a
/// heap-allocated, NUL-terminated UTF-8 string.
///
/// Mirrors villeneuve's [`DataManager::new`] resolution: honors
/// `EMPYREAN_DATA_DIR` first, then falls back to `dirs::data_dir()` —
/// `~/.local/share/empyrean/data/` on Linux, `~/Library/Application
/// Support/empyrean/data/` on macOS, `%APPDATA%\empyrean\data\` on
/// Windows. Cheap (no filesystem I/O).
///
/// Returns null on failure (non-UTF-8 path, NUL byte in path, panic).
/// Call `empyrean_last_error()` for details.
///
/// **The caller owns the returned pointer and must release it with
/// [`empyrean_string_free`].**
#[unsafe(no_mangle)]
pub extern "C" fn empyrean_default_data_dir() -> *mut c_char {
    let result = std::panic::catch_unwind(|| {
        let path = empyrean_core::data::default_data_dir();
        let path_str = match path.to_str() {
            Some(s) => s,
            None => {
                set_last_error("default data dir contains non-UTF-8 bytes");
                return std::ptr::null_mut();
            }
        };
        match CString::new(path_str) {
            Ok(c) => c.into_raw(),
            Err(_) => {
                set_last_error("default data dir contains an embedded NUL byte");
                std::ptr::null_mut()
            }
        }
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("panic in empyrean_default_data_dir");
            std::ptr::null_mut()
        }
    }
}

/// Free a string returned by an empyrean C API function (e.g.,
/// [`empyrean_default_data_dir`], [`empyrean_version_string`]).
///
/// Passing null is a no-op. Passing any pointer not obtained from an
/// empyrean string-returning function is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_string_free(s: *mut c_char) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !s.is_null() {
            unsafe {
                drop(CString::from_raw(s));
            }
        }
    }));
}

/// Multi-line version report — `empyrean-core <ver>\nvilleneuve <ver>\n…`.
///
/// Mirrors [`empyrean_core::version_string`]. Useful for `--version`-style
/// output and for verifying the build provenance of a deployed cdylib.
/// Returns null on allocation failure (extremely unlikely — the strings
/// are short and `&'static` underneath); call `empyrean_last_error()` if
/// it does.
///
/// **The caller owns the returned pointer and must release it with
/// [`empyrean_string_free`].**
#[unsafe(no_mangle)]
pub extern "C" fn empyrean_version_string() -> *mut c_char {
    let result = std::panic::catch_unwind(|| {
        let s = empyrean_core::version_string();
        match CString::new(s) {
            Ok(c) => c.into_raw(),
            Err(_) => {
                set_last_error("version string contains an embedded NUL byte");
                std::ptr::null_mut()
            }
        }
    });
    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("panic in empyrean_version_string");
            std::ptr::null_mut()
        }
    }
}

/// Per-crate version strings reported by the empyrean stack.
///
/// Mirrors [`empyrean_core::Versions`]. Each pointer is a heap-allocated
/// NUL-terminated UTF-8 string owned by [`EmpyreanVersions`]; release
/// the whole struct with [`empyrean_versions_free`] (do not free the
/// individual fields with [`empyrean_string_free`]).
#[repr(C)]
pub struct EmpyreanVersions {
    /// `empyrean-core` crate version (semver string from `Cargo.toml`).
    pub empyrean_core: *mut c_char,
    /// `villeneuve` crate version (`<tag>+<sha>` git-populated).
    pub villeneuve: *mut c_char,
    /// `scott` crate version (`<tag>+<sha>` git-populated).
    pub scott: *mut c_char,
    /// `nolan` crate version (`<tag>+<sha>` git-populated).
    pub nolan: *mut c_char,
}

/// Populate `out` with the per-crate versions of the empyrean stack.
///
/// Returns 0 on success, non-zero on failure (`empyrean_last_error()`
/// has the details). On failure `out` is left zero-initialized — no
/// allocation needs freeing.
///
/// **The caller owns the strings inside `out` and must release the
/// whole struct with [`empyrean_versions_free`] when done.**
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_versions(out: *mut EmpyreanVersions) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out.is_null() {
            set_last_error("empyrean_versions: `out` is null");
            return -1;
        }
        let v = empyrean_core::versions();
        let make = |s: &str| -> Result<*mut c_char, ()> {
            CString::new(s).map(|c| c.into_raw()).map_err(|_| ())
        };
        let core = match make(v.empyrean_core) {
            Ok(p) => p,
            Err(_) => {
                set_last_error("empyrean_versions: empyrean-core version contains NUL");
                return -1;
            }
        };
        let villeneuve = match make(v.villeneuve) {
            Ok(p) => p,
            Err(_) => {
                unsafe { drop(CString::from_raw(core)) };
                set_last_error("empyrean_versions: villeneuve version contains NUL");
                return -1;
            }
        };
        let scott = match make(v.scott) {
            Ok(p) => p,
            Err(_) => {
                unsafe {
                    drop(CString::from_raw(core));
                    drop(CString::from_raw(villeneuve));
                }
                set_last_error("empyrean_versions: scott version contains NUL");
                return -1;
            }
        };
        let nolan = match make(v.nolan) {
            Ok(p) => p,
            Err(_) => {
                unsafe {
                    drop(CString::from_raw(core));
                    drop(CString::from_raw(villeneuve));
                    drop(CString::from_raw(scott));
                }
                set_last_error("empyrean_versions: nolan version contains NUL");
                return -1;
            }
        };
        unsafe {
            (*out).empyrean_core = core;
            (*out).villeneuve = villeneuve;
            (*out).scott = scott;
            (*out).nolan = nolan;
        }
        0
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_versions");
            -1
        }
    }
}

/// Free the version strings inside `versions` (each was heap-allocated
/// by a previous successful [`empyrean_versions`] call). After this
/// returns, `versions` itself is zero-initialized — safe to drop on
/// the caller's stack.
///
/// Passing null is a no-op. Calling this twice on the same struct, or
/// passing a struct that wasn't populated by [`empyrean_versions`], is
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_versions_free(versions: *mut EmpyreanVersions) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if versions.is_null() {
            return;
        }
        unsafe {
            let v = &mut *versions;
            for slot in [
                &mut v.empyrean_core,
                &mut v.villeneuve,
                &mut v.scott,
                &mut v.nolan,
            ] {
                if !slot.is_null() {
                    drop(CString::from_raw(*slot));
                    *slot = std::ptr::null_mut();
                }
            }
        }
    }));
}
