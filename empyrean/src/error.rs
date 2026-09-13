//! Error type for empyrean FFI calls.

use std::ffi::CStr;
use std::fmt;

/// Error returned from an empyrean FFI call.
///
/// Carries the integer error code from libempyrean and the thread-local
/// error message captured at the time of the failure.
#[derive(Debug, Clone)]
pub struct Error {
    /// Numeric error code returned from libempyrean. Zero means success;
    /// negative values indicate the error category:
    /// -1 invalid argument, -2 missing data, -3 convergence failure,
    /// -4 propagation error, -5 I/O error.
    ///
    /// -2 is the missing-data **category**, not one failure: it covers a
    /// named set of files the data directory does not have (no fetch was
    /// attempted), a fetch that was attempted and failed — a 404 from an
    /// upstream that rotated or withdrew a pinned kernel, a refused
    /// connection, a mid-transfer failure — and the category's other
    /// residents (a kernel that read but would not parse, an
    /// Earth-orientation coverage gap, a force-model tier whose inputs
    /// are not loaded). Two implications DO hold and are the ones to
    /// key on: a non-empty
    /// [`missing_data_files`](Self::missing_data_files) list means the
    /// named-files shape, and a message starting `"Data download
    /// failed: "` means a failed acquisition. Anything else under -2:
    /// read the message.
    ///
    /// -5 is local I/O — a read or write against the filesystem. A
    /// failed acquisition is **not** -5: its remedy is connectivity or
    /// a stale kernel pin, never local file repair.
    ///
    /// -6 is the one code libempyrean never returns, because it is the
    /// code for libempyrean not being there: the engine library could
    /// not be found or opened, so no call was made and no engine error
    /// exists to report. Its
    /// [`message`](Self::message) names every location the lookup tried,
    /// in order, with the reason each did not serve. See
    /// [`Context::from_data_dir`](crate::Context::from_data_dir).
    pub code: i32,
    /// Error message captured from `empyrean_last_error()` at the time
    /// of the failure.
    pub message: String,
    /// Data files a construction found absent.
    ///
    /// Non-empty whenever the engine reported a missing-data shortfall by
    /// name — [`Context::from_data_dir_with`](crate::Context::from_data_dir_with)
    /// under `refresh: false`,
    /// [`Context::from_data_dir`](crate::Context::from_data_dir) (which
    /// resolves that way under the `EMPYREAN_OFFLINE` floor, and reports
    /// the same payload when the engine names files on any other path),
    /// and [`download_data`](crate::download_data). It names **every**
    /// file the requested tier needs and the directory does not have, so
    /// a caller can fetch or report exactly that set in one pass instead
    /// of splitting [`message`](Self::message) back apart on a separator
    /// a filename may itself contain. Empty for every failure that is not
    /// a named data shortfall. Prefer
    /// [`missing_data_files`](Self::missing_data_files) to reading the
    /// field.
    pub missing_data_files: Vec<String>,
    /// Zero-based index of the offending orbit in the batch the failing
    /// call was given, when the failure belongs to one orbit.
    ///
    /// A batch propagation takes \\(N\\) orbits × \\(M\\) epochs and
    /// fails as a whole, so without this the only route from "the call
    /// failed" to "orbit 2317 failed" is to re-run the batch one orbit
    /// at a time. Read it through
    /// [`orbit_index`](Self::orbit_index).
    pub orbit_index: Option<usize>,
    /// The `orbit_id` of the offending orbit — the caller's own string,
    /// or the positional `"orbit_{i}"` the boundary substitutes when the
    /// caller left the field unset. Read it through
    /// [`orbit_id`](Self::orbit_id).
    pub orbit_id: Option<String>,
    /// The epoch that identifies the failure, MJD TDB — the requested
    /// output epoch when the failure is tied to one, otherwise the
    /// offending orbit's own epoch. Read it through
    /// [`epoch_mjd_tdb`](Self::epoch_mjd_tdb).
    pub epoch_mjd_tdb: Option<f64>,
}

impl Error {
    /// Capture the current thread-local error from libempyrean.
    ///
    /// Drains the engine's positional payload in the same breath as the
    /// message. The two are thread-local and the next failing call
    /// overwrites both, so they are read together or not at all —
    /// capturing the message now and the position later would risk
    /// pairing one failure's prose with another's index.
    pub(crate) fn capture(code: i32) -> Self {
        let message = unsafe {
            let ptr = empyrean_sys::empyrean_last_error();
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        let (orbit_index, orbit_id, epoch_mjd_tdb) = capture_location();
        Error {
            code,
            message,
            missing_data_files: Vec::new(),
            orbit_index,
            orbit_id,
            epoch_mjd_tdb,
        }
    }

    /// Build an error for a null pointer / failed constructor.
    pub(crate) fn from_null_ptr() -> Self {
        Self::capture(-1)
    }

    /// Build an error for an invalid input (path contains nul byte, etc.).
    ///
    /// Raised by the wrapper itself, before or after the boundary, so it
    /// carries no engine position.
    pub(crate) fn invalid_input(msg: impl Into<String>) -> Self {
        Error {
            code: -1,
            message: msg.into(),
            missing_data_files: Vec::new(),
            orbit_index: None,
            orbit_id: None,
            epoch_mjd_tdb: None,
        }
    }

    /// Build an error for an engine library that could not be found or
    /// opened.
    ///
    /// Carries [`ENGINE_NOT_LOADED`] rather than one of the engine's own
    /// codes: those come back from a call into libempyrean, and here
    /// there is no libempyrean to call. `message` is the loader's
    /// full account — every location tried, in order, with the reason
    /// each did not serve — because a single path is not enough to act
    /// on when the lookup has five rules in it.
    ///
    /// Takes the diagnosis as anything printable rather than the loader's
    /// own type, so the mapping — which code, and that the whole account
    /// survives into the message — is exercisable without a broken
    /// install to hand.
    pub(crate) fn engine_not_loaded(diagnosis: impl fmt::Display) -> Self {
        Error {
            code: ENGINE_NOT_LOADED,
            message: diagnosis.to_string(),
            missing_data_files: Vec::new(),
            orbit_index: None,
            orbit_id: None,
            epoch_mjd_tdb: None,
        }
    }

    /// Zero-based index of the offending orbit in the batch the failing
    /// call was given, or `None` when the failure names no single orbit.
    ///
    /// This is the field that lets a failed batch be repaired instead of
    /// bisected: drop or fix the row it names and re-issue the call. It
    /// is populated only from a position the boundary or the engine
    /// supplied directly — never inferred from
    /// [`message`](Self::message) — so `None` means "not known", never
    /// "not applicable".
    ///
    /// Populated today by every failure the C boundary raises while
    /// marshaling the caller's orbit array (a non-finite element, an
    /// unconvertible coordinate, a malformed thrust or SRP block, a
    /// joint that fails its definiteness gate), by the retained-result
    /// accessors, which are addressed by index, and by every engine
    /// propagation failure that names an orbit — which on the
    /// [`BuiltSystem`](crate::BuiltSystem) path is most of them.
    pub fn orbit_index(&self) -> Option<usize> {
        self.orbit_index
    }

    /// The `orbit_id` of the offending orbit, or `None` when the failure
    /// names no single orbit.
    ///
    /// The caller's own string when it supplied one, and otherwise the
    /// positional `"orbit_{i}"` the boundary substitutes — the same id
    /// the results and events of a successful call carry, so it joins
    /// against them directly.
    ///
    /// An id may be present with no [`orbit_index`](Self::orbit_index)
    /// beside it: the engine names the orbit, and resolving that name to
    /// an index needs an exact, unique match against the batch. A batch
    /// carrying duplicate ids resolves to the id alone rather than to a
    /// guessed row.
    pub fn orbit_id(&self) -> Option<&str> {
        self.orbit_id.as_deref()
    }

    /// The epoch that identifies the failure, MJD TDB, or `None` when
    /// the failure is tied to no epoch.
    ///
    /// The requested output epoch when the failure belongs to one,
    /// otherwise the offending orbit's own input epoch.
    pub fn epoch_mjd_tdb(&self) -> Option<f64> {
        self.epoch_mjd_tdb
    }

    /// The data files a construction found absent, or an empty slice for
    /// any failure that is not a named data shortfall.
    ///
    /// The actionable form of a missing-data failure: fetch exactly these
    /// and the same call succeeds. Population is unconditional — whenever
    /// the engine names files, the list is here, whatever path reached the
    /// failure and whether or not the `EMPYREAN_OFFLINE` floor is in
    /// effect. Both context constructors report it
    /// ([`Context::from_data_dir_with`](crate::Context::from_data_dir_with)
    /// and [`Context::from_data_dir`](crate::Context::from_data_dir)), as
    /// does [`download_data`](crate::download_data). An error contract
    /// that changes shape with an environment variable is not a contract.
    ///
    /// A non-empty list from either constructor carries
    /// `self.code == -2`, the missing-data category. The converse does
    /// not hold — -2 is a category with several residents, and only two
    /// of them have a shape this method can key on:
    ///
    /// * **Files absent, no fetch attempted** — the list names every one
    ///   of them and [`message`](Self::message) reads
    ///   `"Missing data files: <names>"`. Fetch or stage exactly those.
    /// * **A fetch was attempted and failed** — the list is **empty**,
    ///   and the message starts `"Data download failed: "`, naming the
    ///   failing kernel by its URL. The remedy is
    ///   connectivity, or — when the URL 404s because an upstream
    ///   rotated or withdrew a pinned kernel — staging that file by hand
    ///   or moving to an engine whose pin is still served.
    ///
    /// An empty list beside a -2 with any other message is one of the
    /// category's other residents — a kernel that read but would not
    /// parse, a coverage gap, an unloaded force-model input — whose
    /// remedy the message itself states. So: a non-empty list always
    /// means named files; the download prefix always means a failed
    /// acquisition; for anything else under -2, read the message.
    pub fn missing_data_files(&self) -> &[String] {
        &self.missing_data_files
    }
}

/// Drain the engine's positional payload for the failure just reported
/// on this thread.
///
/// Returns all-`None` when the engine recorded no position, which is not
/// itself an error — most failures are not tied to one orbit.
fn capture_location() -> (Option<usize>, Option<String>, Option<f64>) {
    let mut out = empyrean_sys::EmpyreanErrorLocation {
        orbit_id: std::ptr::null_mut(),
        orbit_index: 0,
        epoch_mjd_tdb: 0.0,
        has_orbit_index: 0,
        has_epoch: 0,
    };
    // A non-zero return means the position could not be read (a null
    // out, an allocation failure, a caught panic). Nothing was handed
    // over in that case, so there is nothing to free and nothing to
    // report — the message still stands on its own.
    if unsafe { empyrean_sys::empyrean_error_location(&mut out) } != 0 {
        return (None, None, None);
    }
    let orbit_id = (!out.orbit_id.is_null()).then(|| {
        unsafe { CStr::from_ptr(out.orbit_id) }
            .to_string_lossy()
            .into_owned()
    });
    let index = (out.has_orbit_index != 0).then_some(out.orbit_index);
    let epoch = (out.has_epoch != 0).then_some(out.epoch_mjd_tdb);
    unsafe { empyrean_sys::empyrean_error_location_free(&mut out) };
    (index, orbit_id, epoch)
}

impl fmt::Display for Error {
    /// Renders the position after the message when the failure carries
    /// one, so a caller that only logs the error still sees which orbit
    /// it was — the structured fields are for a caller that acts on it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "empyrean error (code {})", self.code)?;
        } else {
            write!(f, "{} (code {})", self.message, self.code)?;
        }
        match (self.orbit_index, self.orbit_id.as_deref()) {
            (Some(i), Some(id)) => write!(f, " [orbit {i}: {id}]")?,
            (Some(i), None) => write!(f, " [orbit {i}]")?,
            (None, Some(id)) => write!(f, " [orbit {id}]")?,
            (None, None) => {}
        }
        if let Some(epoch) = self.epoch_mjd_tdb {
            write!(f, " [epoch {epoch} MJD TDB]")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// The code for an engine library that could not be found or opened.
///
/// Outside the engine's own -1..-5 range on purpose: those are categories
/// libempyrean reports, and this is the failure of libempyrean to be there
/// at all. Nothing that reaches the engine can ever carry it.
pub const ENGINE_NOT_LOADED: i32 = -6;

/// The engine's category prefix on a failed acquisition — a fetch that was
/// attempted and errored, as opposed to files that are simply absent. The
/// rest of the message is the request context, naming the kernel by URL.
pub(crate) const DOWNLOAD_FAILED_PREFIX: &str = "Data download failed: ";

/// The engine's category prefix on a local I/O failure.
pub(crate) const IO_ERROR_PREFIX: &str = "I/O error: ";

/// Collapse a repeated category prefix down to a single copy.
///
/// The native engine wraps an already-formatted `io::Error` string inside
/// another `io::Error`, so a missing-file failure arrives as
/// `"I/O error: I/O error: No such file or directory (os error 2)"`. Keep one
/// prefix so the message reads cleanly.
///
/// Failed downloads no longer lead with `"I/O error: "` — they carry
/// [`DOWNLOAD_FAILED_PREFIX`] and the missing-data code — so a test anchored
/// at position 0 against the I/O prefix alone would stop collapsing anything
/// the moment a doubling sat behind the download prefix. Walk the leading
/// prefixes instead, dropping only exact consecutive repeats: a distinct
/// prefix is always kept, and a URL or filename further into the message is
/// never touched.
pub(crate) fn dedupe_io_prefix(msg: &str) -> String {
    let mut kept = String::new();
    let mut rest = msg;
    while let Some((prefix, tail)) = [DOWNLOAD_FAILED_PREFIX, IO_ERROR_PREFIX]
        .into_iter()
        .find_map(|p| rest.strip_prefix(p).map(|tail| (p, tail)))
    {
        if !kept.ends_with(prefix) {
            kept.push_str(prefix);
        }
        rest = tail;
    }
    kept.push_str(rest);
    kept
}

/// Result type for empyrean FFI calls.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::{ENGINE_NOT_LOADED, Error, dedupe_io_prefix};

    #[test]
    fn collapses_doubled_io_prefix() {
        assert_eq!(
            dedupe_io_prefix("I/O error: I/O error: No such file or directory (os error 2)"),
            "I/O error: No such file or directory (os error 2)"
        );
        // single prefix and unrelated messages are untouched
        assert_eq!(dedupe_io_prefix("I/O error: nope"), "I/O error: nope");
        assert_eq!(dedupe_io_prefix("convergence failed"), "convergence failed");
        // triple collapses to one
        assert_eq!(
            dedupe_io_prefix("I/O error: I/O error: I/O error: boom"),
            "I/O error: boom"
        );
    }

    #[test]
    fn collapses_behind_the_download_prefix() {
        // A failed acquisition leads with its own category prefix; a
        // doubling behind it still collapses.
        assert_eq!(
            dedupe_io_prefix("Data download failed: I/O error: I/O error: boom"),
            "Data download failed: I/O error: boom"
        );
        // A doubled download prefix collapses the same way.
        assert_eq!(
            dedupe_io_prefix("Data download failed: Data download failed: GET https://x/y: 404"),
            "Data download failed: GET https://x/y: 404"
        );
        // The ordinary shape — one prefix, a request line naming the
        // kernel by URL — is passed through untouched.
        assert_eq!(
            dedupe_io_prefix(
                "Data download failed: GET https://naif.jpl.nasa.gov/x.bpc: http status: 404"
            ),
            "Data download failed: GET https://naif.jpl.nasa.gov/x.bpc: http status: 404"
        );
        // Distinct prefixes are both kept; only repeats are dropped.
        assert_eq!(
            dedupe_io_prefix("I/O error: Data download failed: nope"),
            "I/O error: Data download failed: nope"
        );
    }

    #[test]
    fn a_missing_engine_keeps_its_own_code_and_the_whole_diagnosis() {
        // The loader's account is multi-line and names every location it
        // tried. Truncating it, or filing it under one of the engine's own
        // categories, is what left the container report with nothing to act
        // on.
        let diagnosis = "could not load libempyrean.so, the empyrean engine library. \
                         2 locations were tried, in order:\n  \
                         1. EMPYREAN_LIB — not set\n  \
                         2. beside the running executable — /app/libempyrean.so — no file there";
        let err = Error::engine_not_loaded(diagnosis);

        assert_eq!(err.code, ENGINE_NOT_LOADED);
        assert_ne!(
            err.code, -2,
            "a missing engine is not a missing-data failure; the remedies are unrelated"
        );
        assert!(err.missing_data_files().is_empty());
        assert!(
            err.message.contains("/app/libempyrean.so"),
            "{}",
            err.message
        );
        assert!(err.message.contains("EMPYREAN_LIB"), "{}", err.message);
        // Display appends the code, and keeps the account intact.
        let shown = err.to_string();
        assert!(shown.contains("2 locations were tried"), "{shown}");
        assert!(shown.ends_with("(code -6)"), "{shown}");
    }

    #[test]
    fn the_engine_load_code_sits_outside_the_engine_s_own_range() {
        // Positive control for the test above: -6 has to be a code the
        // engine can never return, or a caller cannot tell "libempyrean
        // said no" from "there was no libempyrean to ask".
        assert!(
            !(-5..=0).contains(&ENGINE_NOT_LOADED),
            "the engine reports -1..-5; this one means the engine is absent"
        );
    }
}
