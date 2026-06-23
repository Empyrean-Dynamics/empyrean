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

/// Result type for empyrean FFI calls.
pub type Result<T> = std::result::Result<T, Error>;
