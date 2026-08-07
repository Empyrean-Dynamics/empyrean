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
///
/// # Thread safety
///
/// A built `EmpyreanContext` is **read-only and safe to share across
/// threads**: one pointer may be handed to any number of concurrent
/// `empyrean_propagate` / `empyrean_generate_ephemeris` /
/// `empyrean_determine` calls without external locking. Build it once
/// and share it — construction is the expensive part (it loads the whole
/// kernel set), and every constructor here is serialized internally
/// through [`CONSTRUCT_LOCK`] because the engine's first-init kernel
/// provisioning does writable-cache file I/O that is not concurrency
/// safe.
///
/// The two `&mut`-shaped entry points — [`empyrean_context_with_spk`] and
/// [`empyrean_context_free`] — are the exceptions: they mutate or destroy
/// the context, so the caller must guarantee no other thread is using the
/// pointer for the duration of the call. `with_spk` takes the same
/// construction lock; `free` cannot.
pub type EmpyreanContext = empyrean_core::Context;

/// Compile-time canary for the thread-safety claim in
/// [`EmpyreanContext`]'s docs and in `include/empyrean.h`.
///
/// Mirrors the identical assertion on `EmpyreanBuiltSystem`. The C header
/// promises callers they may share one context pointer across threads;
/// the promise is only as good as the underlying type staying `Send +
/// Sync`, and nothing else in this crate would notice if an engine-side
/// change (an interior-mutability cache, a non-atomic memo) took that
/// away — the C ABI has no borrow checker to catch it at the boundary.
/// This fails the build instead.
#[allow(dead_code)]
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EmpyreanContext>();
};

/// Flat C-ABI compatible coordinate state.
///
/// Field-identical to [`empyrean_core::convert::CoordinateState`]; the
/// duplicate definition exists so cbindgen (which has `parse_deps =
/// false`) can emit the matching C struct in `empyrean.h` without
/// traversing into the empyrean-core crate.
///
/// `Copy` is a Rust-side convenience only (the batched transform writes
/// whole rows into a caller-owned array); the C layout is unaffected.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// File names carried by the most recent
    /// [`EmpyreanError::MissingDataFiles`](empyrean_core::Error::MissingDataFiles),
    /// drained by [`empyrean_missing_data_files`]. Empty whenever the
    /// last error was anything else — [`set_last_error`] clears it, so a
    /// stale list can never be read against an unrelated failure.
    static LAST_MISSING_DATA_FILES: RefCell<Vec<CString>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() =
            CString::new(msg).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    });
    LAST_MISSING_DATA_FILES.with(|f| f.borrow_mut().clear());
}

/// Record an engine error, keeping the structured payload the message
/// only renders as text.
///
/// `MissingDataFiles` is the one variant that carries an actionable list
/// rather than prose: the caller can fetch exactly those files. Parsing
/// them back out of `"Missing data files: a, b, c"` would be a
/// comma-in-a-filename bug waiting to happen, so the list is stashed for
/// [`empyrean_missing_data_files`] alongside the human-readable message.
/// Every other variant behaves exactly like [`set_last_error`].
pub(crate) fn set_last_error_from(e: &empyrean_core::Error) -> i32 {
    set_last_error(&e.to_string());
    if let empyrean_core::Error::MissingDataFiles(files) = e {
        LAST_MISSING_DATA_FILES.with(|slot| {
            *slot.borrow_mut() = files
                .iter()
                .map(|f| CString::new(f.as_str()).unwrap_or_else(|_| CString::new("?").unwrap()))
                .collect();
        });
    }
    e.error_code()
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
/// missing files. Pass null for `data_dir` to use the platform data
/// directory (`~/.local/share/empyrean/data` on Linux,
/// `~/Library/Application Support/empyrean/data` on macOS).
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

// ── Data-directory options (empyrean_context_from_data_dir_with) ─────
//
// Both fields follow the ABI's shared sentinel rule: **0 means "the
// upstream default"**, so a `memset(0)` options struct is byte-for-byte
// equivalent to passing `NULL`, which in turn is equivalent to calling
// `empyrean_context_from_data_dir`. That equivalence is the whole point
// of the encoding — an options struct whose zero value differed from
// `NULL` would be a trap for exactly the callers who zero their structs
// because the header told them to.

/// Acquire kernels as `empyrean_context_from_data_dir` always has:
/// download anything the tier requires and is missing, and re-download
/// anything whose upstream copy moved. The `memset(0)` default.
pub const EMPYREAN_DATA_REFRESH_DEFAULT: u8 = 0;
/// Same as [`EMPYREAN_DATA_REFRESH_DEFAULT`], spelled explicitly.
pub const EMPYREAN_DATA_REFRESH_ON: u8 = 1;
/// **Strict offline.** Resolve the tier's kernels from the data
/// directory alone — no HTTP HEAD, no download, no staleness check — and
/// fail with `-2` (missing data), naming every absent file through
/// [`empyrean_missing_data_files`], if any is not there.
///
/// Nothing is degraded to make an incomplete directory work: there is no
/// try-the-network-and-tolerate path, no fall back to a lower tier, and
/// no partially-loaded context. The call either produces a context
/// carrying the whole requested tier or it fails naming what is absent.
pub const EMPYREAN_DATA_REFRESH_OFF: u8 = 2;

/// Acquire and load the default tier (Standard). The `memset(0)` default.
///
/// # Why these are not the `force_model` integers
///
/// [`EmpyreanPropagationConfig::force_model`](crate::propagate::EmpyreanPropagationConfig::force_model)
/// encodes Approximate as `0`, because a propagation config has no
/// "unset tier" state — it always propagates under some tier. A data
/// directory does: `NULL` options must mean "what `from_data_dir` does",
/// and a zeroed struct must agree with `NULL`. So this field spends `0`
/// on DEFAULT and shifts the ladder by one rather than making
/// `memset(0)` silently select the Approximate kernel set.
pub const EMPYREAN_DATA_TIER_DEFAULT: i32 = 0;
/// Point-mass planets + Moon + Pluto.
pub const EMPYREAN_DATA_TIER_APPROXIMATE: i32 = 1;
/// Approximate + EIH GR + Sun J2.
pub const EMPYREAN_DATA_TIER_BASIC: i32 = 2;
/// Production tier — Basic + the 16 SB441-N16 asteroid perturbers +
/// Earth J2–J4 + non-gravitational forces. Same as
/// [`EMPYREAN_DATA_TIER_DEFAULT`], spelled explicitly.
pub const EMPYREAN_DATA_TIER_STANDARD: i32 = 3;
/// Standard + planetary spherical harmonics (~290 MB of extra kernels).
///
/// **Reserved, not accepted.** The propagation surface exposes no Full
/// tier in this release, so acquiring its kernels would buy a kernel set
/// nothing at this ABI can integrate under. Passing it is a `-1` naming
/// the reason rather than a silent downgrade to Standard.
pub const EMPYREAN_DATA_TIER_FULL: i32 = 4;

/// Options for [`empyrean_context_from_data_dir_with`].
///
/// `memset(0)` — or passing a `NULL` pointer instead of the struct —
/// selects exactly what [`empyrean_context_from_data_dir`] does. Set only
/// the fields you are changing.
#[repr(C)]
pub struct EmpyreanDataDirOptions {
    /// Whether the constructor may reach the network — one of the
    /// `EMPYREAN_DATA_REFRESH_*` constants. `0` = DEFAULT (on).
    pub refresh: u8,
    /// Force-model tier whose kernel set is acquired and loaded — one of
    /// the `EMPYREAN_DATA_TIER_*` constants. `0` = DEFAULT (Standard).
    ///
    /// Note the deliberate offset from
    /// `EmpyreanPropagationConfig::force_model`; see
    /// [`EMPYREAN_DATA_TIER_DEFAULT`].
    pub tier: i32,
}

/// Resolve an [`EmpyreanDataDirOptions`] (or the `NULL` stand-in) into
/// the engine's own options bag.
fn build_data_dir_options(
    options: *const EmpyreanDataDirOptions,
) -> Result<empyrean_core::data::DataDirOptions, String> {
    use empyrean_core::data::{DataDirOptions, UpstreamForceModelTier as Tier};

    // NULL == memset(0) == today's behaviour. Written as one early
    // return rather than a defaulted struct so the two spellings cannot
    // drift apart.
    if options.is_null() {
        return Ok(DataDirOptions::default());
    }
    let o = unsafe { &*options };

    let refresh = match o.refresh {
        EMPYREAN_DATA_REFRESH_DEFAULT | EMPYREAN_DATA_REFRESH_ON => true,
        EMPYREAN_DATA_REFRESH_OFF => false,
        other => {
            return Err(format!(
                "unknown data-dir refresh mode: {other} (expected \
                 {EMPYREAN_DATA_REFRESH_DEFAULT} = DEFAULT, \
                 {EMPYREAN_DATA_REFRESH_ON} = ON, or \
                 {EMPYREAN_DATA_REFRESH_OFF} = OFF / strict offline)"
            ));
        }
    };

    let tier = match o.tier {
        EMPYREAN_DATA_TIER_DEFAULT | EMPYREAN_DATA_TIER_STANDARD => Tier::Standard,
        EMPYREAN_DATA_TIER_APPROXIMATE => Tier::Approximate,
        EMPYREAN_DATA_TIER_BASIC => Tier::Basic,
        EMPYREAN_DATA_TIER_FULL => {
            return Err(
                "data-dir tier FULL is not exposed at this ABI: the propagation surface \
                 has no Full force-model tier in this release, so its ~290 MB of \
                 spherical-harmonics kernels could be acquired but never integrated \
                 under. Pass EMPYREAN_DATA_TIER_STANDARD (or 0 for the default)."
                    .to_string(),
            );
        }
        other => {
            return Err(format!(
                "unknown data-dir tier: {other} (expected {EMPYREAN_DATA_TIER_DEFAULT} = \
                 DEFAULT, {EMPYREAN_DATA_TIER_APPROXIMATE} = APPROXIMATE, \
                 {EMPYREAN_DATA_TIER_BASIC} = BASIC, or \
                 {EMPYREAN_DATA_TIER_STANDARD} = STANDARD)"
            ));
        }
    };

    Ok(DataDirOptions { refresh, tier })
}

/// Create a new `EmpyreanContext` from a data directory under explicit
/// options — the superset of [`empyrean_context_from_data_dir`].
///
/// Pass `options = NULL` (or a `memset(0)` struct) for exactly the
/// behaviour of [`empyrean_context_from_data_dir`]: acquire and load the
/// full Standard-tier kernel set, downloading anything missing.
///
/// # Strict offline
///
/// `options->refresh = EMPYREAN_DATA_REFRESH_OFF` is the reason this
/// entry point exists. It resolves the tier's kernels from `data_dir`
/// alone and fails if any is absent, naming **every** missing file
/// through [`empyrean_missing_data_files`] — villeneuve's kernels plus
/// the catalog-debiasing table — so an offline context can never come up
/// silently missing part of its data. No lower-tier fallback, no
/// download-just-this-one, no partially-loaded context.
///
/// This entry point reads **no environment variable** to decide
/// `refresh`. `EMPYREAN_OFFLINE` is a *channel*-level floor applied by
/// the layers above (the Rust wrapper, the CLI, the Python package); the
/// C ABI honours exactly what the caller passed.
///
/// Returns a heap-allocated pointer on success, or null on error. Call
/// `empyrean_last_error()` for the message, and
/// [`empyrean_missing_data_files`] for the structured file list when the
/// failure was a strict-offline shortfall. The caller owns the returned
/// pointer and must free it with `empyrean_context_free()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_context_from_data_dir_with(
    data_dir: *const c_char,
    options: *const EmpyreanDataDirOptions,
) -> *mut EmpyreanContext {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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

        let opts = match build_data_dir_options(options) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return std::ptr::null_mut();
            }
        };

        let outcome = {
            let _guard = construct_lock();
            empyrean_core::Context::from_data_dir_with(dir_opt, opts)
        };
        match outcome {
            Ok(ctx) => Box::into_raw(Box::new(ctx)),
            Err(e) => {
                // Keeps the missing-file list, not just its rendering.
                set_last_error_from(&e);
                std::ptr::null_mut()
            }
        }
    }));

    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            set_last_error("panic in empyrean_context_from_data_dir_with");
            std::ptr::null_mut()
        }
    }
}

/// The data files a strict-offline construction found absent.
///
/// Populated by [`empyrean_missing_data_files`]; release it with
/// [`empyrean_missing_data_files_free`].
#[repr(C)]
pub struct EmpyreanMissingDataFiles {
    /// Heap array of `num_files` NUL-terminated UTF-8 file names, or
    /// null when `num_files == 0`.
    pub files: *mut *mut c_char,
    /// Number of missing files. `0` when the last error on this thread
    /// was not a missing-data-files failure.
    pub num_files: usize,
}

/// Retrieve the structured file list from the most recent
/// missing-data-files failure on this thread.
///
/// The companion to `empyrean_last_error()`: that returns the rendered
/// message, this returns the list the message was rendered from, so a
/// caller can fetch or report exactly those names instead of splitting a
/// string (file names may contain the separator).
///
/// Returns 0 and fills `out` on success. `out->num_files == 0` — with
/// `out->files` null — means the last error on this thread was not a
/// missing-data-files failure; it is not itself an error. Returns `-1`
/// for a null `out`, `-5` on allocation failure, `-99` on a caught panic.
///
/// The list is thread-local and is cleared by the next call that records
/// an error on this thread, so read it immediately after the failing
/// call. **The caller owns `out` and must release it with
/// [`empyrean_missing_data_files_free`].**
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_missing_data_files(out: *mut EmpyreanMissingDataFiles) -> i32 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out.is_null() {
            return -1;
        }
        let files: Vec<CString> =
            LAST_MISSING_DATA_FILES.with(|slot| slot.borrow().iter().cloned().collect());
        if files.is_empty() {
            unsafe {
                (*out).files = std::ptr::null_mut();
                (*out).num_files = 0;
            }
            return 0;
        }
        let n = files.len();
        let layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut c_char;
        if ptr.is_null() {
            return -5;
        }
        for (i, f) in files.into_iter().enumerate() {
            unsafe { ptr.add(i).write(f.into_raw()) };
        }
        unsafe {
            (*out).files = ptr;
            (*out).num_files = n;
        }
        0
    }));
    // A panic here leaves the thread-local list untouched and `out`
    // whatever the caller passed in — nothing was handed over, so there
    // is nothing to free.
    result.unwrap_or(-99)
}

/// Free an [`EmpyreanMissingDataFiles`] populated by
/// [`empyrean_missing_data_files`]. Passing a null or zeroed struct is a
/// no-op; the struct is left zeroed so a double free is safe.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_missing_data_files_free(out: *mut EmpyreanMissingDataFiles) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out.is_null() {
            return;
        }
        let res = unsafe { &*out };
        if !res.files.is_null() && res.num_files > 0 {
            for i in 0..res.num_files {
                let p = unsafe { *res.files.add(i) };
                if !p.is_null() {
                    drop(unsafe { CString::from_raw(p) });
                }
            }
            let layout = std::alloc::Layout::array::<*mut c_char>(res.num_files).unwrap();
            unsafe { std::alloc::dealloc(res.files as *mut u8, layout) };
        }
        unsafe {
            (*out).files = std::ptr::null_mut();
            (*out).num_files = 0;
        }
    }));
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
/// Mirrors the engine's `DataManager::new` resolution: honors
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

/// The `EmpyreanDataDirOptions` sentinel contract.
///
/// The whole encoding exists to make three spellings of "do what
/// `empyrean_context_from_data_dir` does" identical: a `NULL` options
/// pointer, a `memset(0)` struct, and the explicit constants. An options
/// struct whose zero value differed from `NULL` would be a trap for
/// exactly the callers who zero their structs because the header told
/// them to — so that equivalence is pinned here rather than left to the
/// prose.
#[cfg(test)]
mod data_dir_options_tests {
    use super::*;
    use empyrean_core::data::{DataDirOptions, UpstreamForceModelTier as Tier};

    fn opts(refresh: u8, tier: i32) -> EmpyreanDataDirOptions {
        EmpyreanDataDirOptions { refresh, tier }
    }

    fn resolve(o: &EmpyreanDataDirOptions) -> Result<DataDirOptions, String> {
        build_data_dir_options(o as *const _)
    }

    /// NULL == memset(0) == the engine's own default.
    #[test]
    fn null_and_zeroed_both_mean_the_default() {
        let from_null = build_data_dir_options(std::ptr::null()).expect("NULL resolves");
        assert_eq!(from_null, DataDirOptions::default());

        let zeroed: EmpyreanDataDirOptions = unsafe { std::mem::zeroed() };
        let from_zero = resolve(&zeroed).expect("memset(0) resolves");
        assert_eq!(
            from_zero,
            DataDirOptions::default(),
            "a zeroed options struct must be indistinguishable from NULL"
        );
        assert!(
            from_zero.refresh,
            "the default acquires kernels, as from_data_dir always has"
        );
    }

    /// The explicit spellings of the default agree with the zero one.
    #[test]
    fn the_explicit_default_spellings_agree() {
        let explicit = resolve(&opts(EMPYREAN_DATA_REFRESH_ON, EMPYREAN_DATA_TIER_STANDARD))
            .expect("explicit defaults resolve");
        assert_eq!(explicit, DataDirOptions::default());
    }

    /// OFF is the reason the entry point exists: it must actually turn
    /// the network off, not merely be accepted.
    #[test]
    fn refresh_off_is_strict_offline() {
        let o = resolve(&opts(EMPYREAN_DATA_REFRESH_OFF, EMPYREAN_DATA_TIER_DEFAULT))
            .expect("OFF resolves");
        assert!(!o.refresh, "REFRESH_OFF must reach the engine as offline");
        assert_eq!(o.tier, Tier::Standard, "the tier default is unaffected");
    }

    /// Every tier constant maps to the tier it names, one step up from
    /// the `force_model` numbering.
    #[test]
    fn the_tier_ladder_maps_as_documented() {
        for (c, want) in [
            (EMPYREAN_DATA_TIER_DEFAULT, Tier::Standard),
            (EMPYREAN_DATA_TIER_APPROXIMATE, Tier::Approximate),
            (EMPYREAN_DATA_TIER_BASIC, Tier::Basic),
            (EMPYREAN_DATA_TIER_STANDARD, Tier::Standard),
        ] {
            let o = resolve(&opts(EMPYREAN_DATA_REFRESH_DEFAULT, c)).expect("tier resolves");
            assert_eq!(o.tier, want, "tier constant {c}");
        }
    }

    /// FULL is reserved, and reserved means refused by name — a silent
    /// downgrade to Standard would hand back a context that quietly is
    /// not the one that was asked for.
    #[test]
    fn the_full_tier_is_refused_by_name() {
        let err = resolve(&opts(
            EMPYREAN_DATA_REFRESH_DEFAULT,
            EMPYREAN_DATA_TIER_FULL,
        ))
        .expect_err("FULL must not resolve");
        assert!(err.contains("FULL"), "error names the tier: {err}");
        assert!(
            err.contains("STANDARD"),
            "error names what to pass instead: {err}"
        );
    }

    /// Unknown values on either field are refused by value rather than
    /// resolved to the default.
    #[test]
    fn unknown_values_are_refused_by_value() {
        let err = resolve(&opts(9, EMPYREAN_DATA_TIER_DEFAULT))
            .expect_err("an unknown refresh mode must not resolve");
        assert!(err.contains('9'), "error names the value: {err}");

        let err = resolve(&opts(EMPYREAN_DATA_REFRESH_DEFAULT, 42))
            .expect_err("an unknown tier must not resolve");
        assert!(err.contains("42"), "error names the value: {err}");
    }
}

/// The structured missing-data-files channel.
///
/// `empyrean_last_error()` renders `MissingDataFiles` as
/// `"Missing data files: a, b, c"`. A caller that wants to *act* on the
/// names — fetch exactly those — must not have to split that string, so
/// the list is carried separately. These pin the two halves of that
/// contract: the list is there after a missing-files failure, and it is
/// gone after anything else (a stale list read against an unrelated
/// error would send an operator after files that are not the problem).
#[cfg(test)]
mod missing_data_files_tests {
    use super::*;

    fn drain() -> Vec<String> {
        let mut out = EmpyreanMissingDataFiles {
            files: std::ptr::null_mut(),
            num_files: 0,
        };
        assert_eq!(unsafe { empyrean_missing_data_files(&mut out) }, 0);
        let names = (0..out.num_files)
            .map(|i| unsafe {
                CStr::from_ptr(*out.files.add(i))
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        unsafe { empyrean_missing_data_files_free(&mut out) };
        assert!(out.files.is_null(), "free must zero the struct");
        assert_eq!(out.num_files, 0);
        names
    }

    #[test]
    fn a_missing_files_error_carries_its_list_and_its_code() {
        let e = empyrean_core::Error::MissingDataFiles(vec![
            "de440.bsp".to_string(),
            "bias.dat".to_string(),
        ]);
        let code = set_last_error_from(&e);
        assert_eq!(code, -2, "missing data is the -2 axis");
        assert_eq!(drain(), vec!["de440.bsp", "bias.dat"]);
    }

    /// Reading the list twice returns it twice — draining it is the
    /// caller's copy, not a move — but any later error clears it.
    #[test]
    fn any_other_error_clears_the_list() {
        set_last_error_from(&empyrean_core::Error::MissingDataFiles(vec![
            "de440.bsp".to_string(),
        ]));
        assert_eq!(drain().len(), 1);
        assert_eq!(drain().len(), 1, "reading does not consume the list");

        set_last_error("something else went wrong");
        assert!(
            drain().is_empty(),
            "an unrelated error must not leave a stale file list behind"
        );

        set_last_error_from(&empyrean_core::Error::MissingDataFiles(vec![
            "gm_de440.tpc".to_string(),
        ]));
        assert_eq!(drain(), vec!["gm_de440.tpc"]);
        set_last_error_from(&empyrean_core::Error::InvalidArgument("nope".into()));
        assert!(
            drain().is_empty(),
            "a non-missing-files engine error must also clear it"
        );
    }

    /// A null `out` is refused, not dereferenced.
    #[test]
    fn a_null_out_is_refused() {
        assert_eq!(
            unsafe { empyrean_missing_data_files(std::ptr::null_mut()) },
            -1
        );
        unsafe { empyrean_missing_data_files_free(std::ptr::null_mut()) };
    }
}
