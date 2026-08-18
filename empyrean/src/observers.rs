//! MPC observatory code → observer state queries.

use crate::context::Context;
use crate::coordinate::{Frame, Origin, int_to_frame};
use crate::error::{Error, Result};
use std::ffi::CString;

/// Observer state at one epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Observer {
    /// MPC observatory code (3 characters).
    pub obs_code: String,
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Position in the [`frame`](Self::frame) / [`origin`](Self::origin)
    /// basis (AU).
    pub position: [f64; 3],
    /// Velocity in the [`frame`](Self::frame) / [`origin`](Self::origin)
    /// basis (AU/day).
    pub velocity: [f64; 3],
    /// Observing night as YYYYMMDD integer, or -1 if unavailable
    /// (space-based observers, or a site without a longitude).
    ///
    /// The night is the local calendar date on which the observing night
    /// began: the UTC epoch is shifted by the site's east longitude to
    /// local mean solar time, and epochs before local noon stamp the
    /// previous date. MPC east-of-Greenwich longitudes in \\( [0, 360) \\)
    /// are wrapped to signed \\( [-180, 180] \\) before the fold, so a
    /// site west of Greenwich (e.g. \\( 289^\circ \\)E in Chile) shifts by
    /// \\( -71^\circ \\), not \\( +19 \\) hours.
    pub observing_night: i32,
    /// Reference frame [`position`](Self::position) and
    /// [`velocity`](Self::velocity) are expressed in.
    ///
    /// Read off the returned state rather than echoed from the request,
    /// so a row always reports the basis that actually produced it.
    pub frame: Frame,
    /// Body the state is relative to.
    pub origin: Origin,
}

impl Observer {
    pub(crate) fn from_ffi(o: &empyrean_sys::EmpyreanObserver) -> Result<Self> {
        Ok(Self {
            obs_code: obs_code_from_bytes(&o.obs_code),
            epoch: crate::Epoch::from_mjd_tdb(o.epoch_mjd_tdb),
            position: [o.x, o.y, o.z],
            velocity: [o.vx, o.vy, o.vz],
            observing_night: o.observing_night,
            frame: int_to_frame(o.frame)?,
            origin: Origin::from_naif_id(o.origin).ok_or_else(|| {
                Error::invalid_input(format!(
                    "libempyrean returned an unknown origin NAIF id: {}",
                    o.origin
                ))
            })?,
        })
    }
}

pub(crate) fn obs_code_from_bytes(bytes: &[u8; 4]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl Context {
    /// Compute observer states (cross product: codes × epochs) in a
    /// caller-chosen `(frame, origin)` basis.
    ///
    /// Returns `obs_codes.len() * epochs.len()` observer entries in
    /// code-major order: all epochs for `obs_codes[0]`, then all epochs
    /// for `obs_codes[1]`, so `result[i * epochs.len() + j]` is
    /// `(obs_codes[i], epochs[j])`.
    ///
    /// # Choosing a basis
    ///
    /// [`Frame::ICRF`] with [`Origin::SSB`] is the **construction
    /// basis** — the one every consumer of an observer state requires,
    /// and the one to pass when the observers are headed for ephemeris
    /// generation or orbit determination. Requesting it takes no
    /// transform at all: the observers come back exactly as constructed,
    /// bit for bit. Any other basis rotates and/or translates them, which
    /// is what a consumer plotting observer geometry wants (e.g.
    /// heliocentric ecliptic site positions via
    /// `(Frame::EclipticJ2000, Origin::SUN)`).
    ///
    /// # Errors
    ///
    /// Every rejection aborts the whole call — there is no partial output
    /// and no silently untransformed entry. The failures keep the
    /// engine's two axes rather than being flattened onto one:
    /// [`Error::code`] is `-1` when the remedy is different arguments (an
    /// unknown observatory code, a frame an observer cannot be rotated
    /// into, an origin that is an MPC site rather than an SPK body) and
    /// `-2` when the remedy is to load or fetch data (an unfetched
    /// spacecraft kernel, an SPK coverage gap, a BPC window that ends
    /// before the requested epoch).
    pub fn get_observers(
        &self,
        obs_codes: &[&str],
        epochs: &[crate::Epoch],
        frame: Frame,
        origin: Origin,
    ) -> Result<Vec<Observer>> {
        let cstrings: Vec<CString> = obs_codes
            .iter()
            .map(|&c| {
                CString::new(c).map_err(|_| Error::invalid_input("observatory code has NUL byte"))
            })
            .collect::<Result<Vec<_>>>()?;
        let ptrs: Vec<*const std::ffi::c_char> = cstrings.iter().map(|s| s.as_ptr()).collect();
        let epochs_mjd_tdb: Vec<f64> = epochs
            .iter()
            .map(|e| e.mjd_tdb())
            .collect::<Result<Vec<_>>>()?;

        let mut result = empyrean_sys::EmpyreanObserverResult {
            observers: std::ptr::null_mut(),
            num_observers: 0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_get_observers(
                self.as_raw(),
                ptrs.as_ptr(),
                ptrs.len(),
                epochs_mjd_tdb.as_ptr(),
                epochs_mjd_tdb.len(),
                frame as i32,
                origin.naif_id(),
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let observers: Result<Vec<Observer>> = unsafe {
            std::slice::from_raw_parts(result.observers, result.num_observers)
                .iter()
                .map(Observer::from_ffi)
                .collect()
        };
        // Free before propagating: the native array is owned regardless
        // of whether marshaling every row succeeded.
        unsafe { empyrean_sys::empyrean_observer_result_free(&mut result) };
        observers
    }
}
