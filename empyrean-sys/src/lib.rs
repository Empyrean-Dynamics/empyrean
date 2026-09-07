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
//! `LD_LIBRARY_PATH` setup. Every location below is tried in order, and the
//! first one that opens wins:
//!
//! 1. `EMPYREAN_LIB` — an explicit path to the library, passed straight to
//!    `dlopen` (override / offline use / a locally built engine). A bare name
//!    defers to the OS loader search; pass a full path for an exact file.
//! 2. A `libempyrean.{dylib,so}` sitting next to the **currently loaded module**
//!    — the `.so`/`.dylib`/executable that statically links this crate — located
//!    via `dladdr`. This makes prebuilt, relocatable artifacts self-contained:
//!    a Python wheel can bundle the engine beside its extension and it is found
//!    with no build-machine path baked in.
//! 3. A `libempyrean.{dylib,so}` sitting next to the **running executable**,
//!    from [`std::env::current_exe`] (`/proc/self/exe` on Linux,
//!    `_NSGetExecutablePath` on macOS), canonicalized. Rule 2 cannot cover this
//!    on glibc: `dladdr` echoes a relative `argv[0]` for the main executable, so
//!    a binary started as `./myapp` or by bare name through `PATH` gets a
//!    relative module directory, which rule 2 deliberately refuses rather than
//!    resolve against the working directory. The kernel's answer has no such
//!    ambiguity, which is what makes a container image that bakes the engine
//!    beside the binary work with no environment set at all.
//! 4. An **explicitly set, absolute `EMPYREAN_DATA_DIR`** — first the directory
//!    itself, then a `lib/` sibling of it. A deployment that already stages
//!    kernels under one mount can stage the engine with them.
//!
//!    Narrower than the engine's own data-directory resolution, on purpose. The
//!    platform default is not searched: it is where kernels are *downloaded
//!    into*, user-writable, and never a place an engine is intentionally put —
//!    treating it as one would rank a download target ahead of the build-time
//!    path, the only candidate whose contents were checksum-pinned. A relative
//!    value is refused by name for the reason rule 2 refuses a relative module
//!    directory: it would resolve against the working directory and let a
//!    library planted there load first.
//! 5. The absolute path recorded at build time (`LIB_PATH`): a sibling
//!    `../target/release` build, an `EMPYREAN_LIB_DIR` override, or a
//!    version-matched, checksum-pinned prebuilt downloaded (in pure Rust — no
//!    `curl`/`tar`) into `~/.cache/empyrean`. This covers `cargo add empyrean`,
//!    where the build script runs on the consumer's own machine and the cache
//!    it downloaded into is the engine's real home. Always tried, in every
//!    profile — but **last**, because it is the one candidate that describes
//!    the build machine rather than this one, so anything that travelled with
//!    the binary gets to answer first.
//!
//! When every location fails, [`try_lib`] returns a [`LoadError`] naming each
//! one in order with the reason it did not yield the engine. The safe wrapper
//! reports that from `Context` construction, so a misplaced engine surfaces as
//! an ordinary error rather than a panic deep in a later call.
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

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// Dynamic-loading bindings: `struct EmpyreanLib` + per-function methods + the
// shared type/const definitions.
include!("bindings.rs");

// Absolute path to libempyrean recorded by the build script — resolution
// rule #5 (see the module docs).
include!(concat!(env!("OUT_DIR"), "/lib_path.rs"));

static LIB: OnceLock<Result<EmpyreanLib, LoadError>> = OnceLock::new();

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

/// A rule in the engine-lookup order (see the module docs for the full order
/// and why each rule is there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Source {
    /// The `EMPYREAN_LIB` environment variable.
    EnvLib,
    /// Beside the currently loaded module, located with `dladdr`.
    ModuleDir,
    /// Beside the running executable, from `std::env::current_exe`.
    ExecutableDir,
    /// An explicitly set, absolute `EMPYREAN_DATA_DIR`.
    DataDir,
    /// A `lib/` sibling of that directory.
    DataDirSiblingLib,
    /// The absolute path recorded at build time.
    BuildTime,
}

impl Source {
    /// Short human label used in [`LoadError`]'s message.
    fn label(self) -> &'static str {
        match self {
            Self::EnvLib => "EMPYREAN_LIB",
            Self::ModuleDir => "beside the loaded module (dladdr)",
            Self::ExecutableDir => "beside the running executable",
            Self::DataDir => "EMPYREAN_DATA_DIR",
            Self::DataDirSiblingLib => "a lib/ sibling of EMPYREAN_DATA_DIR",
            Self::BuildTime => "the path recorded at build time",
        }
    }
}

/// One rule of the lookup, and what came of it.
///
/// `path` is the location the rule produced, absent when the rule could not
/// produce one at all. `reason` is why this step did not hand back the engine —
/// filled in by the loader for a path it opened and failed on, and carried from
/// the start for a rule that produced no path, or produced one this crate
/// declines to load code from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Step {
    /// The rule this step came from.
    pub source: Source,
    /// The location the rule named, when it named one.
    pub path: Option<PathBuf>,
    /// Why this step did not yield the engine. `None` means "open this path",
    /// and is what the loader replaces with the open failure.
    pub reason: Option<String>,
}

impl Step {
    /// A step with a path to open.
    fn open(source: Source, path: PathBuf) -> Self {
        Self {
            source,
            path: Some(path),
            reason: None,
        }
    }

    /// A step that cannot yield the engine, and why.
    fn blocked(source: Source, path: Option<PathBuf>, reason: impl Into<String>) -> Self {
        Self {
            source,
            path,
            reason: Some(reason.into()),
        }
    }
}

/// Every location the loader looked for `libempyrean`, and why none of them
/// produced it.
///
/// The whole point of this type is that it names **every** candidate, in the
/// order they were tried, with a distinct reason each. A single-path message
/// cannot be acted on when the resolution order has five rules in it: the
/// reader has to see that `EMPYREAN_LIB` was unset, that nothing sat beside the
/// executable, and which directory the build recorded, to know which of those
/// to fix.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoadError {
    /// Every rule of the lookup order, in order, each with its outcome.
    steps: Vec<Step>,
}

impl LoadError {
    /// Every rule of the lookup order, in the order it was tried.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not load {LIB_FILENAME}, the empyrean engine library. \
             {} locations were tried, in order:",
            self.steps.len()
        )?;
        for (i, step) in self.steps.iter().enumerate() {
            let reason = step.reason.as_deref().unwrap_or("not attempted");
            match &step.path {
                Some(p) => write!(
                    f,
                    "\n  {}. {} — {} — {reason}",
                    i + 1,
                    step.source.label(),
                    p.display(),
                )?,
                None => write!(f, "\n  {}. {} — {reason}", i + 1, step.source.label())?,
            }
        }
        write!(
            f,
            "\n\nSet EMPYREAN_LIB to the engine library, or place {LIB_FILENAME} beside the \
             executable or under an absolute EMPYREAN_DATA_DIR."
        )
    }
}

impl std::error::Error for LoadError {}

/// The directory holding the module `dladdr` named, or the reason it is not
/// usable.
///
/// Split out of [`self_module_dir`] so the glibc case — a relative `argv[0]`
/// echoed back for the main executable — can be exercised without arranging a
/// real process to be launched that way.
///
/// Only an ABSOLUTE module directory is trusted. `dladdr` may report a relative
/// or bare path, whose `parent()` can be `""`; joining the library name onto
/// that would resolve against the current working directory and let a
/// cwd-planted `libempyrean` load ahead of everything below it. That refusal is
/// exactly why the executable-relative rule exists: `current_exe` answers the
/// same question from the kernel, where there is no relative form to guard
/// against.
fn module_dir_from_dli_fname(fname: &str) -> Result<PathBuf, String> {
    let Some(dir) = Path::new(fname).parent() else {
        return Err(format!(
            "the loader named '{fname}', which has no directory"
        ));
    };
    if !dir.is_absolute() {
        return Err(format!(
            "the loader named '{fname}', a relative path — glibc echoes a relative argv[0] for \
             the main executable, and resolving it against the working directory would let a \
             library planted there load first"
        ));
    }
    Ok(dir.to_path_buf())
}

/// Directory of the currently loaded module — the `.so`/`.dylib`/executable that
/// links this crate — via `dladdr`, or why it could not be determined.
#[cfg(unix)]
fn self_module_dir() -> Result<PathBuf, String> {
    use std::ffi::CStr;
    use std::os::raw::c_void;

    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let addr = &SELF_MARKER as *const u8 as *const c_void;
    // SAFETY: `info` is a valid out-pointer; `dli_fname` is a loader-owned C
    // string we only dereference when dladdr reports success (non-zero).
    if unsafe { libc::dladdr(addr, &mut info) } == 0 || info.dli_fname.is_null() {
        return Err("dladdr could not name the module that links empyrean-sys".to_string());
    }
    let cstr = unsafe { CStr::from_ptr(info.dli_fname) };
    let fname = cstr
        .to_str()
        .map_err(|_| "the loader named a module path that is not UTF-8".to_string())?;
    module_dir_from_dli_fname(fname)
}

#[cfg(not(unix))]
fn self_module_dir() -> Result<PathBuf, String> {
    Err("locating the loaded module is a unix-only facility".to_string())
}

/// Directory of the running executable, canonicalized, or why it could not be
/// determined.
///
/// `current_exe` reads `/proc/self/exe` on Linux and `_NSGetExecutablePath` on
/// macOS, so unlike `dladdr` it is absolute however the process was started —
/// `./myapp`, a bare name resolved through `PATH`, or a full path all give the
/// same answer. Canonicalizing resolves a symlinked launcher back to the
/// directory the real binary (and any engine shipped beside it) lives in.
fn executable_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .map_err(|e| format!("the running executable could not be located: {e}"))?;
    let exe = exe.canonicalize().unwrap_or(exe);
    exe.parent().map(Path::to_path_buf).ok_or_else(|| {
        format!(
            "the running executable '{}' has no directory",
            exe.display()
        )
    })
}

/// The data directory to look for a staged engine in, or why there is none.
///
/// Deliberately narrower than the engine's own data-directory resolution.
/// Two constraints, both about trust rather than convenience:
///
/// **Explicit only.** An unset `EMPYREAN_DATA_DIR` yields nothing. The
/// platform default (`~/.local/share/empyrean/data` and friends) is where the
/// engine *downloads kernels into* over the network; it is user-writable, and
/// nothing ever intentionally stages an engine there. Treating a download
/// target as a place to load code from would put it ahead of the build-time
/// path, the one candidate whose contents were checksum-pinned. The
/// single-mount deployment this rule exists for always sets the variable.
///
/// **Absolute only.** A relative value would be joined onto the process's
/// working directory, which is exactly the cwd-relative load
/// [`module_dir_from_dli_fname`] refuses in so many words: a `libempyrean`
/// planted in whatever directory the process happens to be started from would
/// load ahead of everything below it. Refused by name rather than ignored, so
/// an operator who set it and expected it to work is told why it did not.
fn data_dir() -> Result<PathBuf, String> {
    let Some(dir) = non_empty_env("EMPYREAN_DATA_DIR") else {
        return Err(
            "not set (the platform default is where kernels are downloaded to, not a place to \
             load the engine from, so it is deliberately not searched)"
                .to_string(),
        );
    };
    let dir = PathBuf::from(dir);
    if !dir.is_absolute() {
        return Err(format!(
            "EMPYREAN_DATA_DIR is '{}', a relative path; it would resolve against the working \
             directory, letting a library planted there load first. Set it to an absolute path",
            dir.display()
        ));
    }
    Ok(dir)
}

/// An environment variable's value, treating unset and empty alike.
fn non_empty_env(key: &str) -> Option<std::ffi::OsString> {
    std::env::var_os(key).filter(|v| !v.is_empty())
}

/// The build-time recorded path, or why there is none.
///
/// Empty only where the build script never resolved an engine — a docs.rs
/// build, which is network-isolated and never loads the library.
fn build_time_candidate(lib_path: &str) -> Result<PathBuf, String> {
    if lib_path.is_empty() {
        return Err("no path was recorded at build time".to_string());
    }
    Ok(PathBuf::from(lib_path))
}

/// Everything the lookup order is computed from, each already resolved to a
/// directory or to the reason there is none.
///
/// Every field is an input rather than something [`plan`] goes and reads, so
/// the order — the part that broke — is exercised by tests without a container,
/// a relocated binary, or a mutated environment.
struct Inputs {
    /// Platform file name of the engine library.
    lib_filename: String,
    /// The `EMPYREAN_LIB` override — a full path, or a bare name for the OS
    /// loader search.
    env_lib: Result<PathBuf, String>,
    /// Rule 2: the `dladdr` module directory.
    module_dir: Result<PathBuf, String>,
    /// Rule 3: the executable's directory.
    exe_dir: Result<PathBuf, String>,
    /// Rule 4: an explicitly set, absolute `EMPYREAN_DATA_DIR`.
    data_dir: Result<PathBuf, String>,
    /// Rule 5: the build-time recorded path.
    build_time: Result<PathBuf, String>,
}

impl Inputs {
    /// Read every input from this host and this build.
    fn from_host() -> Self {
        Self {
            lib_filename: LIB_FILENAME.to_string(),
            env_lib: non_empty_env("EMPYREAN_LIB")
                .map(PathBuf::from)
                .ok_or_else(|| "not set".to_string()),
            module_dir: self_module_dir(),
            exe_dir: executable_dir(),
            data_dir: data_dir(),
            build_time: build_time_candidate(LIB_PATH),
        }
    }
}

/// The ordered lookup plan: every rule, with the path it named or the reason it
/// named none, and with a path that is not on disk already marked as such.
///
/// `exists` is injected so a test can lay out a filesystem in a few lines and
/// assert which candidate wins. The one path never checked against it is
/// `EMPYREAN_LIB`: a bare name there is a deliberate hand-off to the OS loader
/// search, which has its own path list, so the only honest test of it is to
/// open it.
fn plan(inputs: &Inputs, exists: impl Fn(&Path) -> bool) -> Vec<Step> {
    let file = inputs.lib_filename.as_str();
    let mut steps = Vec::with_capacity(6);

    /// Add the step for a rule that named a directory: the library inside it,
    /// marked for opening when it is there and skipped with a reason when not.
    fn push(
        steps: &mut Vec<Step>,
        exists: &impl Fn(&Path) -> bool,
        source: Source,
        dir: &Path,
        file: &str,
    ) {
        let candidate = dir.join(file);
        if exists(&candidate) {
            steps.push(Step::open(source, candidate));
        } else {
            steps.push(Step::blocked(source, Some(candidate), "no file there"));
        }
    }

    match &inputs.env_lib {
        // Straight to the loader, existence unchecked: a bare name is meant
        // to defer to the OS search path.
        Ok(p) => steps.push(Step::open(Source::EnvLib, p.clone())),
        Err(why) => steps.push(Step::blocked(Source::EnvLib, None, why.clone())),
    }
    match &inputs.module_dir {
        Ok(d) => push(&mut steps, &exists, Source::ModuleDir, d, file),
        Err(why) => steps.push(Step::blocked(Source::ModuleDir, None, why.clone())),
    }
    match &inputs.exe_dir {
        // On macOS `dladdr` names the main executable by its absolute path,
        // so rules 2 and 3 can land on the same directory. Say so once rather
        // than printing the identical path twice as if they were two misses.
        Ok(d) if inputs.module_dir.as_ref() == Ok(d) => steps.push(Step::blocked(
            Source::ExecutableDir,
            None,
            "the same directory as the loaded module, already tried above",
        )),
        Ok(d) => push(&mut steps, &exists, Source::ExecutableDir, d, file),
        Err(why) => steps.push(Step::blocked(Source::ExecutableDir, None, why.clone())),
    }
    match &inputs.data_dir {
        Ok(d) => {
            push(&mut steps, &exists, Source::DataDir, d, file);
            // A `lib/` sibling makes sense inside a deployment tree and
            // nowhere else. At the filesystem root it would name the system
            // library directory, which is not what a single-mount deployment
            // meant and not somewhere this should be reaching.
            match d.parent().filter(|p| p.parent().is_some()) {
                Some(parent) => push(
                    &mut steps,
                    &exists,
                    Source::DataDirSiblingLib,
                    &parent.join("lib"),
                    file,
                ),
                None => steps.push(Step::blocked(
                    Source::DataDirSiblingLib,
                    None,
                    format!(
                        "'{}' sits at the filesystem root, so a lib/ sibling of it would be the \
                         system library directory",
                        d.display()
                    ),
                )),
            }
        }
        Err(why) => {
            steps.push(Step::blocked(Source::DataDir, None, why.clone()));
            steps.push(Step::blocked(Source::DataDirSiblingLib, None, why.clone()));
        }
    }
    match &inputs.build_time {
        Ok(p) => steps.push(Step::open(Source::BuildTime, p.clone())),
        Err(why) => steps.push(Step::blocked(Source::BuildTime, None, why.clone())),
    }

    steps
}

/// Walk an ordered plan, opening each candidate until one succeeds.
///
/// On success the winning path comes back with the library: with six candidates
/// rather than three, a later diagnosis — the ABI handshake above all — has to
/// be able to say which file actually answered.
///
/// On total failure every step comes back in the [`LoadError`], in order, with
/// the reason it did not produce the engine — the ones that were never opened
/// carrying the reason they were not, and the ones that were carrying the
/// loader's own message.
///
/// `open` is injected so the failure path — the one a user in a container hits
/// and the one whose message has to be right — is testable without a broken
/// install to hand.
fn load_from_plan<L>(
    mut steps: Vec<Step>,
    open: impl Fn(&Path) -> Result<L, String>,
) -> Result<(L, PathBuf), LoadError> {
    for step in &mut steps {
        if step.reason.is_some() {
            continue;
        }
        let path = step.path.clone().expect("an open step carries a path");
        match open(&path) {
            Ok(lib) => return Ok((lib, path)),
            Err(e) => step.reason = Some(e),
        }
    }
    Err(LoadError { steps })
}

/// The loaded `libempyrean`, or every location that was tried and why none of
/// them produced it.
///
/// This is the fallible way in, and the one a caller with an error channel
/// should use: the safe wrapper calls it when a `Context` is built, so a
/// misplaced engine reaches the caller as an ordinary error naming the whole
/// lookup order, instead of a panic from whichever `empyrean_*` call happened
/// to be first.
///
/// The outcome is memoized, failure included. Opening the engine is a
/// process-wide, once-only act, and a second attempt would re-derive the same
/// answer from the same environment — so an `EMPYREAN_LIB` set *after* a failed
/// load does not take effect. Set it before the first call.
///
/// A library that opens but reports a different [`EMPYREAN_ABI_VERSION`] still
/// panics; see [`lib`] for why that failure is deliberately not survivable.
pub fn try_lib() -> Result<&'static EmpyreanLib, LoadError> {
    LIB.get_or_init(|| {
        let steps = plan(&Inputs::from_host(), |p| p.exists());
        let (lib, path) = load_from_plan(steps, |path| {
            // SAFETY: opening a shared library by a path from the resolution
            // order above.
            unsafe { EmpyreanLib::new(path) }.map_err(|e| e.to_string())
        })?;
        // SAFETY: the symbol resolved when the library opened
        // (`--dynamic-link-require-all`), and takes no arguments, so it is
        // safe to call regardless of which ABI version answers it.
        let loaded = unsafe { lib.empyrean_abi_version() };
        assert_eq!(
            loaded, EMPYREAN_ABI_VERSION,
            "libempyrean ABI mismatch: {path:?} reports ABI version {loaded}, but this build of \
             empyrean-sys was compiled against version {EMPYREAN_ABI_VERSION}. \
             Symbol names are shared across versions but signatures and struct layouts are not, \
             so continuing would read arguments through the wrong shape. Point EMPYREAN_LIB at a \
             matching engine, or rebuild it."
        );
        Ok(lib)
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// The loaded `libempyrean`, opened lazily on first use.
///
/// The infallible way in, for the generated `empyrean_*` shims: they mirror the
/// C ABI's signatures and have no channel to report a failure to open the
/// library through. Panics if the library cannot be opened — with the same
/// every-location-tried diagnosis [`try_lib`] returns — or if the library that
/// opened is built to a different [`EMPYREAN_ABI_VERSION`] than this crate.
///
/// Prefer [`try_lib`] anywhere an error can be returned.
///
/// # The version handshake
///
/// `dlsym` matches on symbol **name** only, so a mismatched engine resolves
/// every symbol and then reads the caller's arguments through the wrong
/// signature. The check is an equality: [`EMPYREAN_ABI_VERSION`] carries
/// the distribution release that built this crate and advances with every
/// release, so a different value is a different release and nothing about
/// the layouts behind the shared names can be assumed.
///
/// The damage a tolerated mismatch does has grown with the ABI. Under the
/// retired counter, up to its version 2, it could only ever produce wrong
/// values, because every bump appended struct fields; version 3 changed the
/// parameter lists of `empyrean_transform_coordinates` and
/// `empyrean_get_observers` in place, where the same mismatch writes through
/// an integer the caller passed by value; the 0.10.0 ABI shrinks
/// `EmpyreanSolveFor` and shifts the tail of the `EmpyreanODConfig` that
/// embeds it. Checking [`empyrean_abi_version`] the moment the library opens
/// is what that accessor exists for, and it is cheap — one call, once per
/// process.
pub fn lib() -> &'static EmpyreanLib {
    try_lib().unwrap_or_else(|e| panic!("{e}"))
}

mod shims;
pub use shims::*;

// The header/prebuilt pin the build script enforces. Compiled only for
// tests here — at build time `build.rs` pulls the same file in directly,
// so the parse the tests cover is the parse the guard runs.
#[cfg(test)]
mod header_pin;

#[cfg(test)]
mod lookup_tests {
    use super::*;
    use std::collections::HashSet;

    /// Inputs with every rule supplying a directory, so a test can knock out
    /// exactly the one it is about.
    fn inputs() -> Inputs {
        Inputs {
            lib_filename: "libempyrean.so".to_string(),
            env_lib: Ok(PathBuf::from("/env/libempyrean.so")),
            module_dir: Ok(PathBuf::from("/module")),
            exe_dir: Ok(PathBuf::from("/app/bin")),
            data_dir: Ok(PathBuf::from("/data/empyrean/data")),
            build_time: Ok(PathBuf::from("/build/host/libempyrean.so")),
        }
    }

    /// An `exists` predicate over a fixed set of paths.
    fn present(paths: &[&str]) -> impl Fn(&Path) -> bool + use<> {
        let set: HashSet<PathBuf> = paths.iter().map(PathBuf::from).collect();
        move |p: &Path| set.contains(p)
    }

    /// The sources of a plan, in order.
    fn sources(steps: &[Step]) -> Vec<Source> {
        steps.iter().map(|s| s.source).collect()
    }

    /// The paths a plan would actually open, in order.
    fn openable(steps: &[Step]) -> Vec<PathBuf> {
        steps
            .iter()
            .filter(|s| s.reason.is_none())
            .map(|s| s.path.clone().expect("an open step carries a path"))
            .collect()
    }

    #[test]
    fn plan_visits_every_rule_in_the_documented_order() {
        let steps = plan(&inputs(), present(&[]));
        assert_eq!(
            sources(&steps),
            vec![
                Source::EnvLib,
                Source::ModuleDir,
                Source::ExecutableDir,
                Source::DataDir,
                Source::DataDirSiblingLib,
                Source::BuildTime,
            ],
            "the lookup order is a contract; the plan is what documents it"
        );
    }

    #[test]
    fn every_rule_contributes_its_own_candidate_path() {
        // Positive control for the order test above: with every candidate on
        // disk, the plan names one path per rule, each derived from that
        // rule's directory. A plan that silently dropped a rule would pass
        // the order assertion only by also failing here.
        let all = [
            "/env/libempyrean.so",
            "/module/libempyrean.so",
            "/app/bin/libempyrean.so",
            "/data/empyrean/data/libempyrean.so",
            "/data/empyrean/lib/libempyrean.so",
            "/build/host/libempyrean.so",
        ];
        let steps = plan(&inputs(), present(&all));
        assert_eq!(
            openable(&steps),
            all.iter().map(PathBuf::from).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_candidate_that_is_not_on_disk_is_not_opened() {
        // Only the executable-relative library is present; every other rule
        // still appears in the plan, marked with why it cannot serve.
        let steps = plan(&inputs(), present(&["/app/bin/libempyrean.so"]));
        assert_eq!(
            openable(&steps),
            vec![
                // EMPYREAN_LIB is never existence-checked: a bare name there
                // defers to the OS loader search.
                PathBuf::from("/env/libempyrean.so"),
                PathBuf::from("/app/bin/libempyrean.so"),
                // The build-time path is a recorded absolute file, not a
                // directory to probe, so it too goes straight to the loader.
                PathBuf::from("/build/host/libempyrean.so"),
            ]
        );
        let module = &steps[1];
        assert_eq!(module.source, Source::ModuleDir);
        assert_eq!(module.reason.as_deref(), Some("no file there"));
    }

    #[test]
    fn a_relative_module_path_does_not_defeat_the_executable_candidate() {
        // The container case a downstream caller reported: glibc echoes a
        // relative argv[0] for the
        // main executable, so `dladdr` cannot give a trustworthy directory.
        // Drive the real `dladdr` post-processing with that answer.
        let module_dir = module_dir_from_dli_fname("myapp");
        let why = module_dir
            .clone()
            .expect_err("a bare argv[0] must not yield a module directory");
        assert!(why.contains("relative"), "reason should say why: {why}");

        // Positive control: the same code path accepts an absolute answer,
        // which is what keeps the Python extension finding its own bundled
        // engine.
        assert_eq!(
            module_dir_from_dli_fname("/wheel/empyrean/_empyrean_rs.so"),
            Ok(PathBuf::from("/wheel/empyrean"))
        );

        // With the module rule knocked out and nothing but the engine beside
        // the binary, the executable-relative rule is what loads.
        let steps = plan(
            &Inputs {
                env_lib: Err("not set".to_string()),
                module_dir,
                build_time: Err("no path was recorded at build time".to_string()),
                ..inputs()
            },
            present(&["/app/bin/libempyrean.so"]),
        );
        assert_eq!(
            openable(&steps),
            vec![PathBuf::from("/app/bin/libempyrean.so")]
        );
    }

    #[test]
    fn the_build_time_path_is_last_and_is_still_tried() {
        // `cargo add empyrean` has build.rs download a checksum-pinned engine
        // into the cache, and this candidate is what finds it again — so it is
        // offered whatever the profile. Being LAST is what makes it safe: an
        // engine that travelled with the binary answers first.
        let cache = "/home/builder/.cache/empyrean/libempyrean.so";
        assert_eq!(build_time_candidate(cache), Ok(PathBuf::from(cache)));

        let steps = plan(&inputs(), present(&[]));
        let last = steps.last().expect("the plan is not empty");
        assert_eq!(last.source, Source::BuildTime);
        assert!(
            last.reason.is_none(),
            "the recorded path is opened, not skipped: {last:?}"
        );
        assert_eq!(
            openable(&steps).last(),
            Some(&PathBuf::from("/build/host/libempyrean.so"))
        );
    }

    #[test]
    fn a_build_that_recorded_no_path_offers_none() {
        // docs.rs builds in a network-isolated sandbox and records an empty
        // path; it must read as "nothing recorded", not as the root directory.
        assert!(build_time_candidate("").is_err());
    }

    #[test]
    fn a_total_failure_names_every_location_in_order() {
        let steps = plan(&inputs(), present(&[]));
        let expected: Vec<PathBuf> = openable(&steps);
        let err = load_from_plan(steps, |_| Err("no such file".to_string()))
            .map(|_: ((), PathBuf)| ())
            .expect_err("no candidate can open when every open fails");

        // Every rule is accounted for, each with a reason.
        assert_eq!(err.steps().len(), 6);
        assert!(err.steps().iter().all(|s| s.reason.is_some()));

        // Every path the loader touched appears in the message, in the order
        // it was touched.
        let text = err.to_string();
        let mut cursor = 0usize;
        for path in &expected {
            let needle = path.display().to_string();
            let at = text[cursor..].find(&needle).unwrap_or_else(|| {
                panic!("'{needle}' is missing from the error:\n{text}");
            });
            cursor += at + needle.len();
        }

        // And so do the rules that never produced a path, with their reasons.
        assert!(text.contains("beside the loaded module"), "{text}");
        assert!(text.contains("no such file"), "{text}");
    }

    #[test]
    fn the_first_candidate_that_opens_wins_and_the_rest_are_untouched() {
        // Positive control for the failure test: the same walk succeeds, and
        // stops where it succeeded.
        let steps = plan(&inputs(), present(&["/app/bin/libempyrean.so"]));
        let opened = std::cell::RefCell::new(Vec::new());
        let (got, winner) = load_from_plan(steps, |p| {
            opened.borrow_mut().push(p.to_path_buf());
            if p == Path::new("/app/bin/libempyrean.so") {
                Ok(p.to_path_buf())
            } else {
                Err("no such file".to_string())
            }
        })
        .expect("the executable-relative engine opens");
        assert_eq!(got, PathBuf::from("/app/bin/libempyrean.so"));
        assert_eq!(
            winner,
            PathBuf::from("/app/bin/libempyrean.so"),
            "the winning path is carried out for the ABI handshake to name"
        );
        assert_eq!(
            *opened.borrow(),
            vec![
                PathBuf::from("/env/libempyrean.so"),
                PathBuf::from("/app/bin/libempyrean.so"),
            ],
            "the build-time path must not be opened once an earlier rule wins"
        );
    }

    /// Run `data_dir()` with `EMPYREAN_DATA_DIR` set to `value` (or unset).
    ///
    /// Serialized against the other environment-reading tests: `set_var` is
    /// process-global, and cargo runs tests on threads of one process.
    fn data_dir_with(value: Option<&str>) -> Result<PathBuf, String> {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var_os("EMPYREAN_DATA_DIR");
        // SAFETY: the lock above is the only writer of this variable in this
        // process, and no other thread reads it while the guard is held.
        unsafe {
            match value {
                Some(v) => std::env::set_var("EMPYREAN_DATA_DIR", v),
                None => std::env::remove_var("EMPYREAN_DATA_DIR"),
            }
        }
        let out = data_dir();
        // SAFETY: as above.
        unsafe {
            match previous {
                Some(v) => std::env::set_var("EMPYREAN_DATA_DIR", v),
                None => std::env::remove_var("EMPYREAN_DATA_DIR"),
            }
        }
        out
    }

    #[test]
    fn a_relative_data_directory_is_refused_by_name() {
        // A relative value would be joined onto the working directory, and a
        // libempyrean planted in whatever directory the process was started
        // from would then load ahead of the build-time path — the only
        // candidate whose contents were checksum-pinned. This is the same
        // cwd-relative load the module rule refuses.
        for relative in ["data", "d", "./data", "../data"] {
            let why = data_dir_with(Some(relative))
                .expect_err("a relative data directory must not become a load path");
            assert!(
                why.contains(relative),
                "the refused value must be named: {why}"
            );
            assert!(
                why.contains("working directory"),
                "and the reason given: {why}"
            );
        }

        // Positive control: an absolute value is accepted, so the test above
        // is not passing because `data_dir` refuses everything.
        assert_eq!(
            data_dir_with(Some("/opt/empyrean/data")),
            Ok(PathBuf::from("/opt/empyrean/data"))
        );
    }

    #[test]
    fn the_platform_default_data_directory_is_never_probed() {
        // The platform default is where kernels are DOWNLOADED into: user
        // writable, and never somewhere an engine is deliberately staged.
        // Ranking it ahead of the build-time path would make a download
        // target outrank the checksum-pinned one.
        let why = data_dir_with(None)
            .expect_err("an unset EMPYREAN_DATA_DIR must contribute no candidate");
        assert!(
            !why.contains(".local/share") && !why.contains("Application Support"),
            "no platform path may appear as a candidate: {why}"
        );

        // The plan still accounts for both data steps, so the operator can
        // see they were considered and why they gave nothing.
        let steps = plan(
            &Inputs {
                data_dir: Err(why),
                ..inputs()
            },
            present(&[]),
        );
        let data: Vec<Source> = steps
            .iter()
            .filter(|s| matches!(s.source, Source::DataDir | Source::DataDirSiblingLib))
            .map(|s| s.source)
            .collect();
        assert_eq!(data, vec![Source::DataDir, Source::DataDirSiblingLib]);
        assert!(
            steps
                .iter()
                .filter(|s| matches!(s.source, Source::DataDir | Source::DataDirSiblingLib))
                .all(|s| s.path.is_none() && s.reason.is_some()),
            "neither data step may name a path when the variable is unset"
        );
    }

    #[test]
    fn a_data_directory_at_the_filesystem_root_grows_no_lib_sibling() {
        // `/data` would make the sibling `/lib`, the system library
        // directory. That is not what a single-mount deployment meant.
        let steps = plan(
            &Inputs {
                data_dir: Ok(PathBuf::from("/data")),
                ..inputs()
            },
            present(&["/lib/libempyrean.so"]),
        );
        let sibling = steps
            .iter()
            .find(|s| s.source == Source::DataDirSiblingLib)
            .expect("the step is still reported");
        assert_eq!(sibling.path, None);
        assert!(
            sibling
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("system library directory"),
            "{sibling:?}"
        );
        assert!(!openable(&steps).contains(&PathBuf::from("/lib/libempyrean.so")));
    }

    #[test]
    fn an_executable_directory_matching_the_module_is_reported_once() {
        // On macOS `dladdr` names the main executable absolutely, so both
        // rules land on the same directory. Printing the identical path twice
        // reads as a bug in the message.
        let steps = plan(
            &Inputs {
                module_dir: Ok(PathBuf::from("/app/bin")),
                exe_dir: Ok(PathBuf::from("/app/bin")),
                ..inputs()
            },
            present(&["/app/bin/libempyrean.so"]),
        );
        assert_eq!(
            openable(&steps)
                .iter()
                .filter(|p| *p == &PathBuf::from("/app/bin/libempyrean.so"))
                .count(),
            1,
            "the shared directory is opened once, not twice"
        );
        let exe = steps
            .iter()
            .find(|s| s.source == Source::ExecutableDir)
            .expect("the rule is still reported");
        assert!(
            exe.reason
                .as_deref()
                .unwrap_or_default()
                .contains("same directory as the loaded module"),
            "{exe:?}"
        );
    }

    #[test]
    fn the_data_directory_contributes_itself_and_a_lib_sibling() {
        let steps = plan(
            &Inputs {
                data_dir: Ok(PathBuf::from("/opt/empyrean/data")),
                ..inputs()
            },
            present(&["/opt/empyrean/lib/libempyrean.so"]),
        );
        let data = &steps[3];
        assert_eq!(data.source, Source::DataDir);
        assert_eq!(
            data.path.as_deref(),
            Some(Path::new("/opt/empyrean/data/libempyrean.so"))
        );
        let sibling = &steps[4];
        assert_eq!(sibling.source, Source::DataDirSiblingLib);
        assert_eq!(
            sibling.path.as_deref(),
            Some(Path::new("/opt/empyrean/lib/libempyrean.so"))
        );
        assert!(sibling.reason.is_none(), "the staged engine must be opened");
    }
}
