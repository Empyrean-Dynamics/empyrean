//! Thread-safe empyrean context (SPK, GM, ephemeris state).

use crate::error::{Error, Result, dedupe_io_prefix};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Handle to loaded SPICE kernels, gravitational parameters, and ephemeris
/// state required for every propagation, ephemeris, or OD call.
///
/// Construct with [`Context::from_data_dir`] for the production path
/// (loads the full Standard-tier kernel set), or
/// [`Context::from_data_dir_with`] when you need explicit
/// [`DataDirOptions`] — strict offline above all. The underlying
/// libempyrean resources are released when the `Context` is dropped.
///
/// # Build once, share many
///
/// `Context` is `Send + Sync` and read-only after construction, so one
/// instance serves any number of concurrent calls. Construction is the
/// expensive part — it loads the whole kernel set — so build it once and
/// share it behind an [`Arc`](std::sync::Arc) rather than constructing
/// per task or per thread:
///
/// ```no_run
/// use std::sync::Arc;
/// use std::thread;
///
/// # fn main() -> Result<(), empyrean::Error> {
/// // Load the kernels exactly once.
/// let ctx = Arc::new(empyrean::Context::from_data_dir(None)?);
///
/// let handles: Vec<_> = (0..4)
///     .map(|i| {
///         // Cheap: bumps a refcount, loads nothing.
///         let ctx = Arc::clone(&ctx);
///         thread::spawn(move || {
///             let epochs = [empyrean::Epoch::from_mjd_tdb(60000.0 + i as f64)];
///             ctx.get_observers(
///                 &["568"],
///                 &epochs,
///                 empyrean::Frame::ICRF,
///                 empyrean::Origin::SSB,
///             )
///         })
///     })
///     .collect();
///
/// for h in handles {
///     let _observers = h.join().expect("worker panicked")?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// Contexts may also be *constructed* concurrently — libempyrean
/// serializes native construction at the C ABI — but that serialization
/// means concurrent construction buys nothing over the pattern above.
pub struct Context {
    raw: NonNull<empyrean_sys::EmpyreanContext>,
}

/// Force-model tier whose kernel set a data-directory constructor
/// acquires and loads.
///
/// Distinct from [`ForceModelTier`](crate::ForceModelTier), which selects
/// the physics a *propagation* runs under: this one selects which files
/// have to be on disk. The two ladders line up (a context loaded at
/// `DataTier::Standard` can propagate at `ForceModelTier::Standard`), and
/// the engine's `Full` tier is not exposed at either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataTier {
    /// Point-mass planets + Moon + Pluto.
    Approximate,
    /// Approximate + EIH general relativity + Sun J2.
    Basic,
    /// Production tier — Basic + the 16 SB441-N16 asteroid perturbers +
    /// Earth J2–J4 + non-gravitational forces. Default.
    #[default]
    Standard,
}

/// Options for [`Context::from_data_dir_with`].
///
/// The defaults are what [`Context::from_data_dir`] has always done —
/// `refresh: true`, `tier: DataTier::Standard` — so
/// `Context::from_data_dir_with(dir, DataDirOptions::default())` is
/// exactly `Context::from_data_dir(dir)`.
///
/// Construct it with functional update, spelling out only the fields you
/// are changing:
///
/// ```
/// # use empyrean::DataDirOptions;
/// let offline = DataDirOptions { refresh: false, ..DataDirOptions::default() };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataDirOptions {
    /// Whether the constructor may reach the network.
    ///
    /// * `true` *(default)* — download any kernel the tier requires and
    ///   is missing, and re-download any whose upstream copy moved.
    /// * `false` — **strict offline**. Resolve the tier's kernels from
    ///   the data directory alone and fail, naming every absent file,
    ///   if any is not there. There is no try-the-network-and-tolerate
    ///   path and no degrade-to-a-lower-tier path: an offline context
    ///   either has the full requested tier on disk or it does not get
    ///   built. The names come back through
    ///   [`Error::missing_data_files`].
    ///
    /// # `EMPYREAN_OFFLINE` is a floor, not an override
    ///
    /// `EMPYREAN_OFFLINE=1` downgrades a `true` here to `false` and says
    /// so on stderr; it can never turn a `false` into a `true`. See
    /// [`Context::from_data_dir_with`] for the full precedence table.
    pub refresh: bool,
    /// Force-model tier whose kernel set is acquired and loaded.
    /// Default: [`DataTier::Standard`].
    pub tier: DataTier,
}

impl Default for DataDirOptions {
    fn default() -> Self {
        Self {
            refresh: true,
            tier: DataTier::Standard,
        }
    }
}

/// Environment variable that can only ever turn network access **off**.
const OFFLINE_ENV: &str = "EMPYREAN_OFFLINE";

/// Whether `EMPYREAN_OFFLINE` is asserted.
///
/// Only the exact value `1` counts. Anything else — including `0`,
/// `true`, `yes`, or the empty string — leaves the floor unset, so a
/// caller cannot half-set it and get a surprise.
fn offline_env_is_set() -> bool {
    std::env::var(OFFLINE_ENV)
        .map(|v| v == "1")
        .unwrap_or(false)
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
    ///
    /// **This constructor ignores `EMPYREAN_OFFLINE`.** It predates the
    /// variable, and quietly reinterpreting it would change the meaning
    /// of code written before the variable existed — so on a host where
    /// the operator has asserted the floor, this still reaches the
    /// network, including for the staleness checks a fully populated
    /// data directory would otherwise skip. Use
    /// [`from_data_dir_with`](Self::from_data_dir_with) with an explicit
    /// [`DataDirOptions`] when the variable should apply; that is the
    /// constructor Python's `initialize()` and the CLI both route
    /// through.
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

    /// Load a `Context` from a data directory under explicit
    /// [`DataDirOptions`] — the superset of [`Context::from_data_dir`],
    /// which is exactly this call with `DataDirOptions::default()`.
    ///
    /// # Strict offline
    ///
    /// `options.refresh == false` is the reason this constructor exists.
    /// It resolves the tier's kernels from `data_dir` alone — no HTTP
    /// HEAD, no download, no staleness check — and fails naming **every**
    /// file the tier needs and the directory does not have, retrievable
    /// as a list from [`Error::missing_data_files`]. Nothing is degraded
    /// to make an incomplete directory work: no lower-tier fallback, no
    /// download-just-this-one, no partially-loaded context.
    ///
    /// # `EMPYREAN_OFFLINE` is a floor, not an override
    ///
    /// When the environment variable `EMPYREAN_OFFLINE` is set to `1`,
    /// this constructor may only turn `refresh` **off** — it can never
    /// turn it on. Precedence, in full:
    ///
    /// | `options.refresh` | `EMPYREAN_OFFLINE=1` | effective |
    /// |---|---|---|
    /// | `true`  | unset | `true`  |
    /// | `true`  | set   | `false` — floored, announced on stderr |
    /// | `false` | unset | `false` |
    /// | `false` | set   | `false` |
    ///
    /// The asymmetry is the whole point: an operator exporting
    /// `EMPYREAN_OFFLINE=1` is asserting "this machine must not reach the
    /// network", and that assertion outranks a library caller's
    /// `refresh: true` — but it can never *grant* network access that
    /// `refresh: false` withheld.
    ///
    /// The floor is loud, not silent. `refresh` is a plain `bool`
    /// (matching the engine's own options bag), so a `true` that came
    /// from [`DataDirOptions::default`] is indistinguishable from one a
    /// caller wrote by hand — there is no "defaulted versus explicit"
    /// state to branch on, and inventing a third state here would put
    /// this layer's options bag out of step with the engine's. So the
    /// floor treats every `true` alike and, whenever it actually
    /// downgrades one, says so on stderr naming the variable. A caller
    /// that genuinely must reach the network on such a machine unsets
    /// `EMPYREAN_OFFLINE` for that process; nothing is decided quietly.
    ///
    /// Note that `Context::from_data_dir` does **not** consult the
    /// variable: quietly reinterpreting the older, options-less
    /// constructor would change the meaning of code written before the
    /// variable existed. Reach for this constructor — with an explicit
    /// `refresh` — when the variable should apply.
    pub fn from_data_dir_with(data_dir: Option<&Path>, options: DataDirOptions) -> Result<Self> {
        let c_path = match data_dir {
            Some(d) => Some(path_to_cstring(d)?),
            None => None,
        };
        let raw_path = c_path
            .as_ref()
            .map(|c| c.as_ptr())
            .unwrap_or(std::ptr::null());

        let refresh = apply_offline_floor(options.refresh);
        let ffi_options = empyrean_sys::EmpyreanDataDirOptions {
            refresh: if refresh {
                empyrean_sys::EMPYREAN_DATA_REFRESH_ON
            } else {
                empyrean_sys::EMPYREAN_DATA_REFRESH_OFF
            },
            tier: match options.tier {
                DataTier::Approximate => empyrean_sys::EMPYREAN_DATA_TIER_APPROXIMATE,
                DataTier::Basic => empyrean_sys::EMPYREAN_DATA_TIER_BASIC,
                DataTier::Standard => empyrean_sys::EMPYREAN_DATA_TIER_STANDARD,
            },
        };

        let raw =
            unsafe { empyrean_sys::empyrean_context_from_data_dir_with(raw_path, &ffi_options) };
        NonNull::new(raw).map(|raw| Context { raw }).ok_or_else(|| {
            let mut err = Error::from_null_ptr();
            err.message = dedupe_io_prefix(&err.message);
            // Drain the structured file list before anything else can
            // record an error and clear it. A non-empty list is also the
            // authoritative signal that this was a missing-data failure,
            // so it sets the category too — a null-returning constructor
            // has no return code to carry one.
            err.missing_data_files = drain_missing_data_files();
            if !err.missing_data_files.is_empty() {
                err.code = -2;
                return err;
            }
            // Not a missing-files failure: fall back to the same
            // directory-probing diagnosis `from_data_dir` uses.
            augment_construction_error(err, data_dir)
        })
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
    augment_construction_error(err, data_dir)
}

/// Add the data-directory diagnosis to an already-captured construction
/// error. Split out of [`construction_error`] so
/// [`Context::from_data_dir_with`] can reach it after it has decided the
/// failure was *not* a structured missing-files one.
fn augment_construction_error(mut err: Error, data_dir: Option<&Path>) -> Error {
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

/// Whether the `EMPYREAN_OFFLINE` floor is in force for this process.
///
/// The floor itself is applied inside [`Context::from_data_dir_with`],
/// which is where it belongs — but a caller that does network work of its
/// own *before* building a context (fetching a kernel set, say) has to be
/// able to ask, or the floor covers only half of what the process does.
/// This is that question, and it is the only way to ask it: the variable's
/// name and its accepted values stay in one place.
///
/// Silent by design. The announcement belongs to the moment a request is
/// actually downgraded, not to every poll.
pub fn offline_floor_is_active() -> bool {
    offline_env_is_set()
}

/// Apply the `EMPYREAN_OFFLINE` floor to a requested `refresh`.
///
/// A floor, never an override: it can only turn network access **off**.
/// `refresh: false` is already at the floor and passes through
/// untouched; `refresh: true` — whether written by hand or inherited
/// from [`DataDirOptions::default`], which a `bool` cannot tell apart —
/// is downgraded.
///
/// Announces on stderr whenever it actually downgrades, so a run that
/// stopped downloading is never a mystery. It stays quiet when it
/// changes nothing.
fn apply_offline_floor(requested_refresh: bool) -> bool {
    if requested_refresh && offline_env_is_set() {
        eprintln!(
            "empyrean: {OFFLINE_ENV}=1 — building the context in strict-offline mode \
             (no downloads). Kernels must already be present in the data directory; \
             the constructor will fail naming any that are not."
        );
        return false;
    }
    requested_refresh
}

/// Take the structured missing-file list recorded by the most recent
/// failing native call on this thread.
///
/// Empty for every failure that was not a missing-data-files one — the C
/// ABI clears the list on each error, so a non-empty result always
/// belongs to the call that just failed.
fn drain_missing_data_files() -> Vec<String> {
    let mut out = empyrean_sys::EmpyreanMissingDataFiles {
        files: std::ptr::null_mut(),
        num_files: 0,
    };
    let code = unsafe { empyrean_sys::empyrean_missing_data_files(&mut out) };
    if code != 0 || out.files.is_null() || out.num_files == 0 {
        return Vec::new();
    }
    let files = unsafe {
        std::slice::from_raw_parts(out.files, out.num_files)
            .iter()
            .map(|&p| {
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            })
            .collect()
    };
    unsafe { empyrean_sys::empyrean_missing_data_files_free(&mut out) };
    files
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
/// Provisions through the engine's download-only path
/// (`empyrean_download_data`): the kernel set is downloaded and cached, but no
/// context is built or loaded, so this does not pay for a full Standard-tier
/// context assembly it would immediately discard.
///
/// ```no_run
/// # fn main() -> Result<(), empyrean::Error> {
/// let dir = empyrean::download_data(None)?; // ensures a usable data directory
/// let _ctx = empyrean::Context::from_data_dir(Some(&dir))?;
/// # Ok(())
/// # }
/// ```
pub fn download_data(data_dir: Option<&Path>) -> Result<PathBuf> {
    // Provision without building a context: the C ABI's
    // `empyrean_download_data` runs the engine's download-and-cache pass and
    // stops at the resolved kernel paths — no ephemeris load, no context to
    // discard.
    let c_path = match data_dir {
        Some(d) => Some(path_to_cstring(d)?),
        None => None,
    };
    let raw_path = c_path
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null());
    let code = unsafe { empyrean_sys::empyrean_download_data(raw_path) };
    if code != 0 {
        let mut err = Error::capture(code);
        err.message = dedupe_io_prefix(&err.message);
        // A strict-offline-style missing-data shortfall carries a structured
        // file list; keep it (not just the rendered message) as the other
        // data-dir entry points do.
        err.missing_data_files = drain_missing_data_files();
        return Err(augment_construction_error(err, data_dir));
    }
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

/// The `EMPYREAN_OFFLINE` floor and the [`DataDirOptions`] defaults.
///
/// The precedence rule is the whole point: the variable may only ever
/// turn network access **off**, never on, and only the exact value `1`
/// asserts it. These exercise [`apply_offline_floor`] directly rather
/// than through a constructor, so they need no kernels and no network.
#[cfg(test)]
mod offline_floor_tests {
    use super::{DataDirOptions, DataTier, OFFLINE_ENV, apply_offline_floor, offline_env_is_set};

    /// `EMPYREAN_OFFLINE` is process-global; serialize the tests that
    /// set it so they cannot observe each other's value.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_offline<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var(OFFLINE_ENV).ok();
        // Safety: guarded by ENV_LOCK, and the rest of this crate's tests
        // do not read EMPYREAN_OFFLINE.
        unsafe {
            match value {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
        }
        let out = f();
        unsafe {
            match prior {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
        }
        out
    }

    /// `DataDirOptions::default()` must be what `from_data_dir` has
    /// always done, or `from_data_dir_with(dir, default())` is not the
    /// superset it is documented to be.
    #[test]
    fn the_defaults_are_todays_behaviour() {
        let d = DataDirOptions::default();
        assert!(d.refresh, "the default acquires kernels");
        assert_eq!(d.tier, DataTier::Standard);
        assert_eq!(DataTier::default(), DataTier::Standard);
    }

    /// The floor only ever turns refresh off — all four cells of the
    /// documented precedence table, including the one that matters:
    /// `refresh: false` with the variable unset must stay `false`, or
    /// the "floor" would be granting network access nobody asked for.
    #[test]
    fn the_floor_never_turns_the_network_on() {
        with_offline(None, || {
            assert!(apply_offline_floor(true), "unset + true = true");
            assert!(!apply_offline_floor(false), "unset + false = false");
        });
        with_offline(Some("1"), || {
            assert!(!apply_offline_floor(true), "set + true = floored to false");
            assert!(!apply_offline_floor(false), "set + false = false");
        });
    }

    /// Only the exact value `1` asserts the floor. A half-set variable
    /// must not silently stop downloads.
    #[test]
    fn only_the_exact_value_one_sets_the_floor() {
        for v in ["0", "true", "yes", "", "1 ", "01"] {
            with_offline(Some(v), || {
                assert!(
                    !offline_env_is_set(),
                    "{OFFLINE_ENV}={v:?} must not assert the floor"
                );
                assert!(apply_offline_floor(true), "{OFFLINE_ENV}={v:?}");
            });
        }
        with_offline(Some("1"), || {
            assert!(offline_env_is_set());
        });
    }
}
