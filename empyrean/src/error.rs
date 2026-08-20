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
}

impl Error {
    /// Capture the current thread-local error from libempyrean.
    pub(crate) fn capture(code: i32) -> Self {
        let message = unsafe {
            let ptr = empyrean_sys::empyrean_last_error();
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error {
            code,
            message,
            missing_data_files: Vec::new(),
        }
    }

    /// Build an error for a null pointer / failed constructor.
    pub(crate) fn from_null_ptr() -> Self {
        Self::capture(-1)
    }

    /// Build an error for an invalid input (path contains nul byte, etc.).
    pub(crate) fn invalid_input(msg: impl Into<String>) -> Self {
        Error {
            code: -1,
            message: msg.into(),
            missing_data_files: Vec::new(),
        }
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            write!(f, "empyrean error (code {})", self.code)
        } else {
            write!(f, "{} (code {})", self.message, self.code)
        }
    }
}

impl std::error::Error for Error {}

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
    use super::dedupe_io_prefix;

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
}
