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
    fn from_ffi(s: &empyrean_sys::EmpyreanState) -> Result<Self> {
        let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for origin: {}",
                s.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(s.frame)?;
        Ok(Self {
            epoch: crate::Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
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
        let states: Vec<State> = unsafe {
            std::slice::from_raw_parts(result.states, result.num_states)
                .iter()
                .map(State::from_ffi)
                .collect::<Result<_>>()?
        };
        unsafe { empyrean_sys::empyrean_state_result_free(&mut result) };
        Ok(states)
    }
}
