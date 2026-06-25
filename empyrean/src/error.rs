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
        Error { code, message }
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
        }
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
