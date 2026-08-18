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
    /// `self.code == -2`, the missing-data category; `download_data`
    /// keeps whatever code the engine returned, so read the list rather
    /// than the code to decide whether a failure named files.
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

/// Collapse a doubly-wrapped `"I/O error: I/O error: ..."` prefix down to a
/// single `"I/O error: "`.
///
/// The native engine wraps an already-formatted `io::Error` string inside
/// another `io::Error`, so a missing-file failure arrives as
/// `"I/O error: I/O error: No such file or directory (os error 2)"`. Keep one
/// prefix so the message reads cleanly.
pub(crate) fn dedupe_io_prefix(msg: &str) -> String {
    let mut s = msg.to_string();
    while s.starts_with("I/O error: I/O error: ") {
        // Drop the first prefix, leaving exactly one.
        s = s.replacen("I/O error: ", "", 1);
    }
    s
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
}
