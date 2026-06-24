//! Raw FFI bindings to `libempyrean`, the C shared library for empyrean's
//! astrodynamics engine.
//!
//! Prefer the safe wrapper crate [`empyrean`](https://docs.rs/empyrean)
//! unless you need direct access to the C ABI.
//!
//! # How `libempyrean` is loaded
//!
//! The library is opened at run time with [`libloading`] — there is **no**
//! link-time native dependency and no `install_name_tool` / `patchelf` / rpath /
//! `LD_LIBRARY_PATH` setup. Its path is resolved on first use, in order:
//!
//! 1. `EMPYREAN_LIB` — an explicit path to the library, passed straight to
//!    `dlopen` (override / offline use / a locally built engine). A bare name
//!    defers to the OS loader search; pass a full path for an exact file.
//! 2. A `libempyrean.{dylib,so}` sitting next to the **currently loaded module**
//!    — the `.so`/`.dylib`/executable that statically links this crate — located
//!    via `dladdr`. This makes prebuilt, relocatable artifacts self-contained:
//!    a Python wheel can bundle the engine beside its extension and it is found
//!    with no build-machine path baked in.
//! 3. The absolute path recorded at build time (`LIB_PATH`): a sibling
//!    `../target/release` build, an `EMPYREAN_LIB_DIR` override, or a
//!    version-matched, checksum-pinned prebuilt downloaded (in pure Rust — no
//!    `curl`/`tar`) into `~/.cache/empyrean`. This covers `cargo add empyrean`,
//!    where the build script runs on the consumer's own machine.
//!
//! The bindings are pre-generated and committed, so building needs no C header
//! and no `libclang` / `bindgen`. Callers use the free `empyrean_*` functions
//! exactly as with a statically linked library; each delegates to the loaded
//! [`EmpyreanLib`].
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
// The generated dynamic-loading methods call the loaded fn pointers directly in
// their (unsafe) bodies; this is generated FFI, so allow the 2024 granularity lint.
#![allow(unsafe_op_in_unsafe_fn)]
// The generated bindings and the free-function shims are `pub unsafe fn`s without
// per-function `# Safety` sections, and mirror the C ABI's wide argument lists;
// they are generated FFI, not hand-authored API.
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// Dynamic-loading bindings: `struct EmpyreanLib` + per-function methods + the
// shared type/const definitions.
include!("bindings.rs");

// Absolute path to libempyrean recorded by the build script — resolution
// fallback #3 (see the module docs).
include!(concat!(env!("OUT_DIR"), "/lib_path.rs"));

static LIB: OnceLock<EmpyreanLib> = OnceLock::new();

/// Platform file name of the engine library.
const LIB_FILENAME: &str = if cfg!(target_os = "macos") {
    "libempyrean.dylib"
} else if cfg!(target_os = "windows") {
    "empyrean.dll"
} else {
    "libempyrean.so"
};

// A data symbol whose address lands in *this* module, so `dladdr` reports the
// shared object / executable that statically links empyrean-sys.
#[cfg(unix)]
static SELF_MARKER: u8 = 0;

/// Directory of the currently loaded module — the `.so`/`.dylib`/executable that
/// links this crate — via `dladdr`. `None` if it cannot be determined.
#[cfg(unix)]
fn self_module_dir() -> Option<PathBuf> {
    use std::ffi::CStr;
    use std::os::raw::c_void;

    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let addr = &SELF_MARKER as *const u8 as *const c_void;
    // SAFETY: `info` is a valid out-pointer; `dli_fname` is a loader-owned C
    // string we only dereference when dladdr reports success (non-zero).
    if unsafe { libc::dladdr(addr, &mut info) } == 0 || info.dli_fname.is_null() {
        return None;
    }
    let cstr = unsafe { CStr::from_ptr(info.dli_fname) };
    let dir = Path::new(cstr.to_str().ok()?).parent()?;
    // Only trust an ABSOLUTE module directory. `dladdr` may report a relative or
    // bare path (e.g. glibc echoes a relative `argv[0]` for the main executable),
    // whose `parent()` can be `""`; joining the library name onto that would
    // resolve against the current working directory and let a cwd-planted
    // `libempyrean` load ahead of the checksum-pinned build-time path. Treat that
    // as "cannot determine" so resolution falls through to the absolute LIB_PATH.
    dir.is_absolute().then(|| dir.to_path_buf())
}

#[cfg(not(unix))]
fn self_module_dir() -> Option<PathBuf> {
    None
}

/// Resolve the engine library path at run time (see the module docs for the
/// full resolution order).
fn resolve_lib_path() -> PathBuf {
    // 1. Explicit override.
    if let Some(p) = std::env::var_os("EMPYREAN_LIB") {
        return PathBuf::from(p);
    }
    // 2. Bundled next to the currently loaded module (relocatable artifacts).
    if let Some(dir) = self_module_dir() {
        let candidate = dir.join(LIB_FILENAME);
        if candidate.exists() {
            return candidate;
        }
    }
    // 3. The absolute path recorded at build time.
    PathBuf::from(LIB_PATH)
}

/// The loaded `libempyrean`, opened lazily on first use.
///
/// Panics if the library cannot be opened. The path is resolved from the host
/// environment and the build, so a failure here means a broken or incomplete
/// install (e.g. a prebuilt artifact missing its bundled engine) — surfaced
/// loudly rather than papered over.
pub fn lib() -> &'static EmpyreanLib {
    LIB.get_or_init(|| {
        let path = resolve_lib_path();
        // SAFETY: opening a shared library by the path resolved above.
        unsafe {
            EmpyreanLib::new(&path)
                .unwrap_or_else(|e| panic!("failed to load libempyrean from {path:?}: {e}"))
        }
    })
}

mod shims;
pub use shims::*;
