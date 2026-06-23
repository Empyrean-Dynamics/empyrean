//! Time scale conversions: ISO 8601 ↔ MJD in UTC or TDB.
//!
//! Native (no astropy required) round-trip via villeneuve's leap-second
//! table and Fairhead & Bretagnon (1990) TDB-TT secular term.

use std::ffi::{CStr, CString};

use crate::error::{Error, Result};

/// Time scale identifier matching the C ABI.
///
/// `0` is UTC, `1` is TDB. Other values are rejected at the FFI layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TimeScale {
    /// Coordinated Universal Time.
    UTC = 0,
    /// Barycentric Dynamical Time.
    TDB = 1,
}

impl TimeScale {
    /// Parse a string scale name (case-insensitive).
    ///
    /// Accepts `"utc"`, `"UTC"`, `"tdb"`, `"TDB"`. Returns an
    /// [`Error`] with `invalid_input` on anything else.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "utc" => Ok(Self::UTC),
            "tdb" => Ok(Self::TDB),
            other => Err(Error::invalid_input(format!(
                "unknown time scale: {other:?} (expected \"utc\" or \"tdb\")"
            ))),
        }
    }
}

/// A timestamp tagged with its time scale.
///
/// Mirrors `villeneuve::time::Epoch` at the public-API layer so callers
/// can pass time values without bare-`f64` ambiguity. Carries an MJD
/// value plus the [`TimeScale`] it's expressed in; convert to a known
/// scale via [`mjd_tdb`](Self::mjd_tdb) / [`mjd_utc`](Self::mjd_utc).
///
/// # Examples
///
/// ```no_run
/// use empyrean::{Epoch, TimeScale};
/// let e = Epoch::from_mjd_tdb(60523.0);
/// assert_eq!(e.scale(), TimeScale::TDB);
/// assert_eq!(e.mjd(), 60523.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epoch {
    mjd: f64,
    scale: TimeScale,
}

impl Epoch {
    /// Construct an epoch from a raw MJD value and the scale it's in.
    pub fn new(mjd: f64, scale: TimeScale) -> Self {
        Self { mjd, scale }
    }

    /// Construct an epoch from MJD in TDB.
    pub fn from_mjd_tdb(mjd: f64) -> Self {
        Self {
            mjd,
            scale: TimeScale::TDB,
        }
    }

    /// Construct an epoch from MJD in UTC.
    pub fn from_mjd_utc(mjd: f64) -> Self {
        Self {
            mjd,
            scale: TimeScale::UTC,
        }
    }

    /// Construct an epoch from an ISO 8601 UTC string (e.g.
    /// `"2024-08-01T00:00:00.000Z"`). The trailing `Z` is required.
    /// The returned epoch carries [`TimeScale::UTC`].
    pub fn from_iso_utc(iso: &str) -> Result<Self> {
        Ok(Self::from_mjd_utc(iso_to_mjd(iso, TimeScale::UTC)?))
    }

    /// Return the raw MJD value in this epoch's scale.
    pub fn mjd(&self) -> f64 {
        self.mjd
    }

    /// Return the time scale.
    pub fn scale(&self) -> TimeScale {
        self.scale
    }

    /// Return the MJD value converted to TDB.
    ///
    /// No-op when the epoch is already TDB; otherwise round-trips
    /// through the wrapper's ISO conversion path so the leap-second
    /// table and TDB-TT secular term are applied consistently.
    pub fn mjd_tdb(&self) -> Result<f64> {
        match self.scale {
            TimeScale::TDB => Ok(self.mjd),
            TimeScale::UTC => {
                let iso = mjd_to_iso(self.mjd, TimeScale::UTC)?;
                iso_to_mjd(&iso, TimeScale::TDB)
            }
        }
    }

    /// Return the MJD value converted to UTC.
    pub fn mjd_utc(&self) -> Result<f64> {
        match self.scale {
            TimeScale::UTC => Ok(self.mjd),
            TimeScale::TDB => {
                let iso = mjd_to_iso(self.mjd, TimeScale::TDB)?;
                iso_to_mjd(&iso, TimeScale::UTC)
            }
        }
    }
}

/// Parse an ISO 8601 UTC string to MJD in the requested target scale.
///
/// `iso` must be a UTC string (e.g. `"2024-08-01T00:00:00.000Z"`); the
/// `Z` suffix is required. Returns the corresponding MJD value in
/// either UTC or TDB.
///
/// # Examples
///
/// ```no_run
/// use empyrean::time::{iso_to_mjd, TimeScale};
/// let mjd_tdb = iso_to_mjd("2024-08-01T00:00:00.000Z", TimeScale::TDB)?;
/// # Ok::<(), empyrean::Error>(())
/// ```
pub fn iso_to_mjd(iso: &str, scale: TimeScale) -> Result<f64> {
    let c_iso = CString::new(iso)
        .map_err(|_| Error::invalid_input("ISO string contains an interior NUL byte"))?;
    let mut out: f64 = 0.0;
    let code = unsafe {
        empyrean_sys::empyrean_iso_to_mjd(c_iso.as_ptr(), scale as i32, &mut out as *mut f64)
    };
    if code == 0 {
        Ok(out)
    } else {
        Err(Error::capture(code))
    }
}

/// Format an MJD value (in the given scale) as an ISO 8601 UTC string.
///
/// The output is always a UTC ISO string with the trailing `Z`; `scale`
/// only describes how to interpret the input MJD.
///
/// # Examples
///
/// ```no_run
/// use empyrean::time::{mjd_to_iso, TimeScale};
/// let iso = mjd_to_iso(60523.0, TimeScale::TDB)?;
/// # Ok::<(), empyrean::Error>(())
/// ```
pub fn mjd_to_iso(mjd: f64, scale: TimeScale) -> Result<String> {
    let mut buf = vec![0u8; 64];
    let code = unsafe {
        empyrean_sys::empyrean_mjd_to_iso(
            mjd,
            scale as i32,
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
        )
    };
    if code != 0 {
        return Err(Error::capture(code));
    }
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const std::os::raw::c_char) };
    Ok(cstr.to_string_lossy().into_owned())
}
