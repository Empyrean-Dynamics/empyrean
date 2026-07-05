//! MPC observatory code → observer state queries.

use crate::context::Context;
use crate::error::{Error, Result};
use std::ffi::CString;

/// Observer state at one epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct Observer {
    /// MPC observatory code (3 characters).
    pub obs_code: String,
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Position in ICRF relative to SSB (AU).
    pub position: [f64; 3],
    /// Velocity in ICRF relative to SSB (AU/day).
    pub velocity: [f64; 3],
    /// Observing night as YYYYMMDD integer, or -1 if unavailable.
    pub observing_night: i32,
}

impl Observer {
    pub(crate) fn from_ffi(o: &empyrean_sys::EmpyreanObserver) -> Self {
        Self {
            obs_code: obs_code_from_bytes(&o.obs_code),
            epoch: crate::Epoch::from_mjd_tdb(o.epoch_mjd_tdb),
            position: [o.x, o.y, o.z],
            velocity: [o.vx, o.vy, o.vz],
            observing_night: o.observing_night,
        }
    }
}

pub(crate) fn obs_code_from_bytes(bytes: &[u8; 4]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl Context {
    /// Compute observer states (cross product: codes × epochs).
    ///
    /// Returns `obs_codes.len() * epochs.len()` observer entries in
    /// code-major order.
    pub fn get_observers(
        &self,
        obs_codes: &[&str],
        epochs: &[crate::Epoch],
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
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let observers = unsafe {
            std::slice::from_raw_parts(result.observers, result.num_observers)
                .iter()
                .map(Observer::from_ffi)
                .collect()
        };
        unsafe { empyrean_sys::empyrean_observer_result_free(&mut result) };
        Ok(observers)
    }
}
