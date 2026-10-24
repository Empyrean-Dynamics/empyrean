//! SPK body state queries.

use crate::context::Context;
use crate::coordinate::{Frame, Origin};
use crate::error::{Error, Result};

/// A single body state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State {
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Position (AU).
    pub position: [f64; 3],
    /// Velocity (AU/day).
    pub velocity: [f64; 3],
    /// Reference frame.
    pub frame: Frame,
    /// Center body NAIF ID.
    pub origin: Origin,
}

impl State {
    fn from_ffi(s: &empyrean_sys::EmpyreanState, request_epoch: crate::Epoch) -> Result<Self> {
        let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for origin: {}",
                s.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(s.frame)?;
        Ok(Self {
            // One state per requested epoch, in order: echo the caller's own
            // Epoch straight back (same value, same scale), rather than
            // round-tripping the engine's TDB return through a conversion.
            epoch: request_epoch,
            position: [s.x, s.y, s.z],
            velocity: [s.vx, s.vy, s.vz],
            frame,
            origin,
        })
    }
}

impl Context {
    /// Query SPK body states (target relative to center) at the given epochs.
    pub fn get_states(
        &self,
        target: Origin,
        center: Origin,
        epochs: &[crate::Epoch],
        frame: Frame,
    ) -> Result<Vec<State>> {
        // Validates the batch carries one scale (refuses a UTC/TDB mix); the
        // echoed output epochs come from the caller's own `epochs` below.
        crate::time::request_scale(epochs)?;
        let epochs_mjd_tdb: Vec<f64> = epochs
            .iter()
            .map(|e| e.mjd_tdb())
            .collect::<Result<Vec<_>>>()?;
        let mut result = empyrean_sys::EmpyreanStateResult {
            states: std::ptr::null_mut(),
            num_states: 0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_get_states(
                self.as_raw(),
                target.naif_id(),
                center.naif_id(),
                epochs_mjd_tdb.as_ptr(),
                epochs_mjd_tdb.len(),
                frame as i32,
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let raw = unsafe { std::slice::from_raw_parts(result.states, result.num_states) };
        if raw.len() != epochs.len() {
            unsafe { empyrean_sys::empyrean_state_result_free(&mut result) };
            return Err(Error::invalid_input(format!(
                "get_states returned {} states for {} requested epochs; cannot echo \
                 the requested instants",
                raw.len(),
                epochs.len()
            )));
        }
        let states: Result<Vec<State>> = raw
            .iter()
            .zip(epochs.iter())
            .map(|(s, req)| State::from_ffi(s, *req))
            .collect();
        unsafe { empyrean_sys::empyrean_state_result_free(&mut result) };
        let states = states?;
        Ok(states)
    }
}
