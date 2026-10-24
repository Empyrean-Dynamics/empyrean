//! Time scale conversions: ISO 8601 ↔ MJD in UTC or TDB.
//!
//! Native (no astropy required) round-trip via villeneuve's leap-second
//! table and the full periodic Fairhead & Bretagnon (1990) TDB-TT
//! series — the annual term is carried, not truncated away.

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
    /// through the wrapper's ISO conversion path (nanosecond-precision
    /// formatting, so the round trip is not the precision limiter) so
    /// the leap-second table and the periodic TDB-TT series are applied
    /// consistently.
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

    /// Return this epoch expressed in `scale`.
    ///
    /// A no-op (same value, same scale) when already in `scale`.
    /// Otherwise converts through the leap-second + Fairhead/Bretagnon
    /// path shared with [`mjd_tdb`](Self::mjd_tdb) / [`mjd_utc`](Self::mjd_utc).
    /// This is the convert-once path for an **engine-found** instant (a
    /// close approach, event, impact, rise or set time): it is handed back
    /// in the scale the caller requested. A **requested** instant is not
    /// converted here. The caller's own `Epoch` is echoed straight back
    /// (same value, same scale), so an output epoch the caller asked for is
    /// the caller's own bytes, never a formatter round-trip.
    pub fn to_scale(&self, scale: TimeScale) -> Result<Self> {
        if self.scale == scale {
            return Ok(*self);
        }
        match scale {
            TimeScale::TDB => Ok(Self::from_mjd_tdb(self.mjd_tdb()?)),
            TimeScale::UTC => Ok(Self::from_mjd_utc(self.mjd_utc()?)),
        }
    }
}

/// The single request scale shared by a slice of input epochs.
///
/// Output epochs are expressed in this scale: a requested instant is
/// echoed back as the caller's own `Epoch`, an engine-found instant is
/// converted into it. A slice that mixes UTC and TDB is refused by name
/// rather than silently picking one. One call carries one scale. An
/// empty slice yields [`TimeScale::TDB`] (the engine scale), which only
/// ever tags an empty result.
pub(crate) fn request_scale(epochs: &[Epoch]) -> Result<TimeScale> {
    let mut it = epochs.iter();
    let Some(first) = it.next() else {
        return Ok(TimeScale::TDB);
    };
    let scale = first.scale();
    for e in it {
        if e.scale() != scale {
            return Err(Error::invalid_input(
                "epochs mix time scales (UTC and TDB) in one call; one call carries one scale",
            ));
        }
    }
    Ok(scale)
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
    crate::context::ensure_engine_loaded()?;
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
    crate::context::ensure_engine_loaded()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_scale_reads_the_batch_and_refuses_a_mix() {
        // The request scale is the single scale a batch carries: a UTC batch
        // yields UTC, a TDB batch TDB, and an empty batch the engine scale.
        assert_eq!(
            request_scale(&[Epoch::from_mjd_utc(60000.0)]).unwrap(),
            TimeScale::UTC
        );
        assert_eq!(
            request_scale(&[Epoch::from_mjd_tdb(60000.0)]).unwrap(),
            TimeScale::TDB
        );
        assert_eq!(request_scale(&[]).unwrap(), TimeScale::TDB);
        // A batch that mixes UTC and TDB is refused by name, never silently
        // reduced to one scale.
        let err = request_scale(&[Epoch::from_mjd_utc(60000.0), Epoch::from_mjd_tdb(60001.0)])
            .unwrap_err();
        assert!(err.to_string().contains("mix time scales"));
    }

    #[test]
    fn to_scale_within_scale_is_a_value_preserving_noop() {
        // A requested instant handed back in its own scale is the caller's
        // exact value, with no conversion applied.
        let e = Epoch::from_mjd_utc(60000.123456789);
        let same = e.to_scale(TimeScale::UTC).unwrap();
        assert_eq!(same.scale(), TimeScale::UTC);
        assert_eq!(same.mjd(), 60000.123456789);
    }
}
