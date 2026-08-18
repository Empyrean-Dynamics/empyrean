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
    /// **`EMPYREAN_OFFLINE=1` applies here too.** The variable is a floor
    /// on the process, not a knob on one constructor: an operator who has
    /// asserted "this machine must not reach the network" is not reached
    /// past by the older, options-less entry point. When the floor is in
    /// force this call behaves exactly like
    /// [`from_data_dir_with`](Self::from_data_dir_with) with
    /// `refresh: false` — it resolves the Standard tier from `data_dir`
    /// alone and fails naming every file the tier needs and the directory
    /// does not have. The downgrade is announced on stderr naming the
    /// variable, so a run that stopped downloading is never a mystery.
    /// Nothing else about this constructor changes.
    pub fn from_data_dir(data_dir: Option<&Path>) -> Result<Self> {
        // The floor, applied where the network request is actually made.
        // `DataDirOptions::default()` is documented as exactly this
        // constructor's behaviour, so routing the floored case through the
        // options constructor is the same call with `refresh` at the floor
        // — no second announcement (it only speaks when it downgrades a
        // `true`), no tier change, no other divergence.
        if !apply_offline_floor(true) {
            return Self::from_data_dir_with(
                data_dir,
                DataDirOptions {
                    refresh: false,
                    ..DataDirOptions::default()
                },
            );
        }
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
    /// # What the floor covers
    ///
    /// Every **data-provisioning** entry point in this crate applies it:
    /// this constructor, [`Context::from_data_dir`] (which is this call
    /// with [`DataDirOptions::default`]), and
    /// [`download_data`](crate::download_data), which refuses outright
    /// under the floor rather than downloading — reaching the network is
    /// its entire purpose, so there is nothing left to downgrade.
    ///
    /// It does **not** cover the catalog query helpers
    /// ([`query_sbdb`](crate::query_sbdb),
    /// [`query_horizons`](crate::query_horizons),
    /// [`query_horizons_vectors`](crate::query_horizons_vectors),
    /// [`query_observations`](crate::query_observations),
    /// [`query_radar`](crate::query_radar)): those reach JPL and the MPC
    /// and are ungated. Use [`offline_floor_is_active`] to gate them
    /// yourself on a host where the variable must mean "no outbound
    /// requests at all".
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

/// Why a data directory that *exists* cannot be used as one.
///
/// Distinct from "a kernel is absent", and it has to be: the engine's
/// provisioning pass `mkdir`s the data directory, and `mkdir` never follows
/// a trailing symbolic link — so a data dir that is a link to nowhere (or
/// to a file, or to itself) fails with a bare `File exists (os error 17)`.
/// Probing *that* path for kernels finds none of them, so the first entry
/// of [`CORE_KERNELS`] would be blamed for a failure it had no part in, and
/// the caller would be sent to `download_data`, which cannot fix a link.
struct UnusableDataDir {
    /// The link's target, when the data-dir path is itself a symbolic link.
    symlink_target: Option<PathBuf>,
    /// What is wrong with the path, in the filesystem's own terms.
    reason: String,
}

/// Inspect the data directory itself, before any kernel is probed.
///
/// `Ok(())` means either that the path resolves to a directory (so a
/// kernel-level diagnosis is meaningful) or that nothing is there at all
/// (the ordinary un-provisioned case, whose remedy really is
/// `download_data`). `Err` means the path exists but is not a usable
/// directory, and no kernel may be named for it.
fn inspect_data_dir(dir: &Path) -> std::result::Result<(), UnusableDataDir> {
    // lstat: does not follow the link, so a dangling one is still "there".
    let link_meta = match std::fs::symlink_metadata(dir) {
        Ok(m) => m,
        // Nothing at this path — not an obstruction, just unprovisioned.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(UnusableDataDir {
                symlink_target: None,
                reason: format!("could not be inspected: {source}"),
            });
        }
    };
    let symlink_target = if link_meta.file_type().is_symlink() {
        std::fs::read_link(dir).ok()
    } else {
        None
    };
    // stat: follows the link, which is the resolution `mkdir` needs and
    // does not perform itself.
    match std::fs::metadata(dir) {
        Ok(m) if m.is_dir() => Ok(()),
        Ok(_) => Err(UnusableDataDir {
            symlink_target,
            reason: "exists but is not a directory".to_string(),
        }),
        Err(source) => Err(UnusableDataDir {
            symlink_target,
            reason: format!("does not resolve to a directory: {source}"),
        }),
    }
}

/// Append the directory-level diagnosis to a construction error's message.
///
/// Names the path, the link target when there is one, and the real reason —
/// and deliberately names **no** kernel, because none of them is why this
/// failed.
fn augment_with_unusable_data_dir(base: &str, dir: &Path, bad: &UnusableDataDir) -> String {
    let what = match &bad.symlink_target {
        Some(target) => format!(
            "the empyrean data directory '{}' is a symbolic link to '{}' that {}",
            dir.display(),
            target.display(),
            bad.reason,
        ),
        None => format!(
            "the empyrean data directory '{}' {}",
            dir.display(),
            bad.reason,
        ),
    };
    format!(
        "{base} — {what}. Repoint or remove that path, or set EMPYREAN_DATA_DIR to a \
         directory that already contains the kernels."
    )
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
///
/// The structured missing-files payload is drained first, exactly as
/// [`Context::from_data_dir_with`] drains it. The C ABI records that
/// payload for **both** constructors, so which one a caller reached the
/// failure through must not decide whether
/// [`Error::missing_data_files`] is populated — an error contract that
/// changes shape with an environment variable is not a contract.
fn construction_error(data_dir: Option<&Path>) -> Error {
    let mut err = Error::from_null_ptr();
    err.message = dedupe_io_prefix(&err.message);
    err.missing_data_files = drain_missing_data_files();
    if !err.missing_data_files.is_empty() {
        // A non-empty list is the authoritative signal that this was a
        // missing-data failure, and a null-returning constructor has no
        // return code to carry the category otherwise.
        err.code = -2;
        return err;
    }
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
        // The directory itself comes first. A path that does not resolve to
        // a directory makes every kernel look absent, so a kernel-level
        // guess made here would be manufactured, not observed — and its
        // remedy (`download_data`) cannot repair a broken link. Only a
        // resolving directory may reach `first_missing_core_kernel`.
        if let Err(bad) = inspect_data_dir(&dir) {
            // Not a missing-data failure: leave the captured code alone
            // rather than recategorizing an I/O fault as absent data.
            err.message = augment_with_unusable_data_dir(&err.message, &dir, &bad);
            return err;
        }
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

/// Refuse a download-only call while the `EMPYREAN_OFFLINE` floor is in
/// force.
///
/// [`apply_offline_floor`] downgrades a *construction* from "refresh" to
/// "resolve what is already here". A provisioning call has no such second
/// mode — the network access is the whole call — so the floor cannot
/// downgrade it and must not be ignored either. Factored out so the tests
/// can exercise the decision without a network or a data directory.
fn refuse_download_under_offline_floor() -> Result<()> {
    if offline_env_is_set() {
        return Err(Error::invalid_input(format!(
            "{OFFLINE_ENV}=1 is set — refusing to provision kernels, because downloading \
             them is the entire operation and there is no offline form of it. Build the \
             context against an already-provisioned directory with `refresh: false` \
             (Python `refresh=False`, CLI `--no-refresh`), or unset {OFFLINE_ENV} for the \
             process that must provision."
        )));
    }
    Ok(())
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
/// # `EMPYREAN_OFFLINE=1` refuses this call
///
/// The [`EMPYREAN_OFFLINE`](offline_floor_is_active) floor downgrades a
/// context construction from "refresh" to "resolve what is already here".
/// This function has no such second mode — reaching the network *is* the
/// call — so under the floor it fails with an error naming the variable
/// rather than provisioning anyway. Build against an already-provisioned
/// directory with `refresh: false` (Python `refresh=False`, CLI
/// `--no-refresh`), or unset the variable for the process that must
/// provision.
///
/// ```no_run
/// # fn main() -> Result<(), empyrean::Error> {
/// let dir = empyrean::download_data(None)?; // ensures a usable data directory
/// let _ctx = empyrean::Context::from_data_dir(Some(&dir))?;
/// # Ok(())
/// # }
/// ```
pub fn download_data(data_dir: Option<&Path>) -> Result<PathBuf> {
    refuse_download_under_offline_floor()?;
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
/// asserts it. Most of these exercise [`apply_offline_floor`] directly,
/// so they need no kernels and no network — but the two that matter most
/// go through the real public entry points, because a decision function
/// nobody calls is not a floor.
#[cfg(test)]
mod offline_floor_tests {
    use super::{
        Context, DataDirOptions, DataTier, OFFLINE_ENV, apply_offline_floor, download_data,
        offline_env_is_set, offline_floor_is_active, refuse_download_under_offline_floor,
    };

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

    /// `from_data_dir` is `from_data_dir_with(dir, default())`, so under
    /// the floor it must take the same floored path. Asserted through the
    /// decision function the constructor branches on, so the test needs
    /// neither kernels nor a network: `apply_offline_floor(true)` is
    /// exactly the predicate `from_data_dir` evaluates, and it is what the
    /// options constructor evaluates for `DataDirOptions::default()`.
    #[test]
    fn the_older_constructor_takes_the_same_floored_path() {
        with_offline(Some("1"), || {
            assert!(
                !apply_offline_floor(DataDirOptions::default().refresh),
                "the floor downgrades the default `refresh: true` the old constructor uses"
            );
            assert!(offline_floor_is_active());
        });
        with_offline(None, || {
            assert!(
                apply_offline_floor(DataDirOptions::default().refresh),
                "unset, the old constructor still refreshes"
            );
        });
    }

    /// Provisioning has no offline form, so the floor refuses it outright
    /// instead of downgrading it — and the refusal names the variable and
    /// the way out, because an error a caller cannot act on is its own
    /// failure.
    #[test]
    fn provisioning_is_refused_under_the_floor() {
        with_offline(None, || {
            assert!(
                refuse_download_under_offline_floor().is_ok(),
                "unset: provisioning proceeds"
            );
        });
        with_offline(Some("1"), || {
            let err = refuse_download_under_offline_floor()
                .expect_err("the floor must refuse a download-only call");
            assert!(
                err.message.contains(OFFLINE_ENV),
                "names the variable: {err}"
            );
            assert!(
                err.message.contains("refresh: false") || err.message.contains("refresh=False"),
                "points at the offline construction path: {err}"
            );
        });
        with_offline(Some("0"), || {
            assert!(
                refuse_download_under_offline_floor().is_ok(),
                "only the exact value 1 asserts the floor"
            );
        });
    }

    /// A scratch directory that does not exist yet, so the constructor has
    /// to either provision it or refuse.
    fn empty_scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "empyrean-offline-{}-{}-{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch data dir");
        dir
    }

    fn entries(dir: &std::path::Path) -> usize {
        std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
    }

    /// The floor through the real constructor, end to end.
    ///
    /// The decision function being right is necessary and not sufficient:
    /// delete the branch in [`Context::from_data_dir`] that consults it
    /// and every predicate test above still passes. This calls the
    /// shipped constructor over an empty directory and asserts the three
    /// things a floored construction must be — it fails, it fails as a
    /// named data shortfall (`code == -2` with the file list, the same
    /// contract `from_data_dir_with` carries), and it downloads
    /// **nothing**: the directory is still empty afterwards, and the
    /// whole call is over in well under the time the first kernel alone
    /// takes to fetch.
    #[test]
    fn the_older_constructor_is_floored_end_to_end() {
        let dir = empty_scratch("ctor");
        let started = std::time::Instant::now();
        let err = with_offline(Some("1"), || {
            Context::from_data_dir(Some(&dir))
                .err()
                .expect("an empty directory under the floor cannot produce a context")
        });
        let elapsed = started.elapsed();

        assert_eq!(
            err.code, -2,
            "a floored construction is a missing-data failure: {err}"
        );
        assert!(
            !err.missing_data_files().is_empty(),
            "the floored path must carry the structured file list, not just prose: {err}"
        );
        assert_eq!(
            entries(&dir),
            0,
            "the floor must not have downloaded anything into {}",
            dir.display()
        );
        assert!(
            elapsed < std::time::Duration::from_secs(20),
            "the floored constructor resolved in {elapsed:?} — that is long enough to \
             have gone to the network"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The same, for the provisioning call — which has no offline form, so
    /// the floor refuses it outright. Through the public function, for the
    /// same reason: the private helper being right proves nothing about
    /// whether anything calls it.
    #[test]
    fn provisioning_is_refused_through_the_public_function() {
        let dir = empty_scratch("download");
        let err = with_offline(Some("1"), || {
            download_data(Some(&dir)).expect_err("provisioning under the floor must refuse")
        });
        assert!(
            err.message.contains(OFFLINE_ENV),
            "the refusal must name the variable: {err}"
        );
        assert_eq!(
            entries(&dir),
            0,
            "nothing may be fetched into {}",
            dir.display()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The directory-level diagnosis, which must run *before* any kernel is
/// named.
///
/// A data dir that is a symlink to nowhere makes every kernel look absent,
/// so the old probe always blamed `de440.bsp` — the first entry of
/// `CORE_KERNELS` — and sent the caller to `download_data`, which cannot
/// repair a link. These pin the message assembly directly, so they need no
/// engine, no kernels and no network.
#[cfg(test)]
mod data_dir_shape_tests {
    use super::{augment_construction_error, augment_with_unusable_data_dir, inspect_data_dir};
    use crate::error::Error;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "empyrean-datadir-{}-{}-{tag}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A real directory, and an absent path, both leave the kernel-level
    /// diagnosis in charge — the first because it resolves, the second
    /// because "nothing is provisioned yet" really is a `download_data`
    /// case.
    #[test]
    fn a_resolving_or_absent_path_is_not_a_directory_level_fault() {
        let root = scratch("ok");
        assert!(inspect_data_dir(&root).is_ok(), "a real directory resolves");
        assert!(
            inspect_data_dir(&root.join("not-created-yet")).is_ok(),
            "an absent path is unprovisioned, not obstructed"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A symlink that DOES resolve to a directory is fine — the guard must
    /// not reject the ordinary "data dir lives behind a link" layout.
    #[cfg(unix)]
    #[test]
    fn a_symlink_to_a_real_directory_resolves() {
        let root = scratch("goodlink");
        let real = root.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = root.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(inspect_data_dir(&link).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The regression: a dangling data-dir symlink must produce the
    /// directory-level diagnosis, naming the link and its target, and must
    /// NOT accuse `de440.bsp` of being absent.
    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_names_the_link_not_a_kernel() {
        let root = scratch("dangling");
        let link = root.join("data");
        std::os::unix::fs::symlink(root.join("nowhere"), &link).unwrap();

        let bad = inspect_data_dir(&link).expect_err("a dangling link is not a usable data dir");
        assert_eq!(
            bad.symlink_target.as_deref(),
            Some(root.join("nowhere").as_path()),
            "the link target is read and reported"
        );

        let msg =
            augment_with_unusable_data_dir("I/O error: File exists (os error 17)", &link, &bad);
        assert!(
            msg.starts_with("I/O error: File exists (os error 17)"),
            "the native cause stays up front: {msg}"
        );
        assert!(
            msg.contains("symbolic link"),
            "names the shape of the fault: {msg}"
        );
        assert!(
            msg.contains("nowhere"),
            "names the unresolved target: {msg}"
        );
        assert!(
            !msg.contains("de440.bsp"),
            "no kernel may be blamed for a directory-level fault: {msg}"
        );
        assert!(
            !msg.contains("download_data"),
            "download_data cannot repair a broken link: {msg}"
        );

        // And through the real assembly path the constructors use.
        let err = augment_construction_error(
            Error::invalid_input("I/O error: File exists (os error 17)"),
            Some(&link),
        );
        assert!(!err.message.contains("de440.bsp"), "{}", err.message);
        assert!(err.message.contains("symbolic link"), "{}", err.message);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A data dir that is a regular file (or a link to one) is named as
    /// such — the same class of fault, a different filesystem shape.
    #[test]
    fn a_regular_file_at_the_data_dir_path_is_named() {
        let root = scratch("file");
        let path = root.join("data");
        std::fs::write(&path, b"not a directory").unwrap();
        let bad = inspect_data_dir(&path).expect_err("a file is not a usable data dir");
        assert!(bad.symlink_target.is_none());
        let msg = augment_with_unusable_data_dir("boom", &path, &bad);
        assert!(msg.contains("is not a directory"), "{msg}");
        assert!(!msg.contains("de440.bsp"), "{msg}");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The older constructor's error builder drains the structured
    /// missing-files payload, exactly as its `_with` sibling does.
    ///
    /// The C ABI records that payload for both constructors, so the
    /// question is only whether this side reads it. If it does not, the
    /// same directory yields a file list through one constructor and an
    /// empty one through the other, and a caller branching on
    /// `missing_data_files().is_empty()` takes a different path depending
    /// on which entry point they happened to call.
    ///
    /// Drives the engine to a real missing-files failure with no network
    /// (`refresh` off over an empty directory), which leaves the payload
    /// on this thread, then asks [`construction_error`] what it makes of
    /// it — the same function `Context::from_data_dir` builds its error
    /// with.
    #[test]
    fn the_older_constructors_error_builder_carries_the_file_list() {
        let dir = scratch("payload");
        let c_dir = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
        let options = empyrean_sys::EmpyreanDataDirOptions {
            refresh: empyrean_sys::EMPYREAN_DATA_REFRESH_OFF,
            tier: empyrean_sys::EMPYREAN_DATA_TIER_STANDARD,
        };
        let raw =
            unsafe { empyrean_sys::empyrean_context_from_data_dir_with(c_dir.as_ptr(), &options) };
        assert!(
            raw.is_null(),
            "an empty directory cannot resolve the Standard tier offline"
        );

        let err = super::construction_error(Some(&dir));
        assert!(
            !err.missing_data_files().is_empty(),
            "the older constructor must carry the file list too: {err}"
        );
        assert_eq!(
            err.code, -2,
            "a named data shortfall is the missing-data category: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
