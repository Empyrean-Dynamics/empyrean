//! Orbit propagation and event detection.
//!
//! [`Context::propagate`] takes a batch of [`Orbit`](crate::Orbit)
//! values and a list of target epochs (MJD TDB) and returns a
//! [`PropagationResult`] carrying propagated states, detected events
//! (close approaches, periapses, SOI crossings, occultations, eclipses,
//! atmospheric entries, …), and — when the configured
//! [`UncertaintyMethod`] populates them — STMs and STTs on each state.
//!
//! # Example: multi-epoch propagation with events
//!
//! ```no_run
//! use empyrean::{Context, EventConfig, Origin, PropagationConfig};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let batch = empyrean::query_sbdb(&["99942"], None)?;
//!
//! // Apophis through the 2029 flyby. EventConfig defaults monitor
//! // every body; restrict to Earth + Moon for a focused readout.
//! let mut cfg = PropagationConfig::default();
//! cfg.events = EventConfig {
//!     body_filter: vec![Origin::EARTH, Origin::MOON],
//!     ..EventConfig::default()
//! };
//!
//! let epochs: Vec<empyrean::Epoch> = (60500..=63500)
//!     .step_by(180)
//!     .map(|d| empyrean::Epoch::from_mjd_tdb(d as f64))
//!     .collect();
//! let result = ctx.propagate(&batch.orbits, &epochs, &cfg)?;
//!
//! for e in result.events.iter().filter(|e| e.event_type == "close_approach_start") {
//!     println!("CA @ MJD {:.2}, distance {:.0} km", e.epoch.mjd_tdb()?, e.distance_km);
//! }
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! # Choosing an uncertainty method
//!
//! See [`UncertaintyMethod`]. The default is
//! [`UncertaintyMethod::FirstOrder`] (linear covariance via STM —
//! adequate for the bulk of NEO work). Use
//! [`UncertaintyMethod::SecondOrder`] near a planetary close approach
//! (for the second-order impact-probability correction);
//! [`UncertaintyMethod::SigmaPoint`] /
//! [`UncertaintyMethod::MonteCarlo`] when you need tail probabilities
//! or want to exercise the full distribution.
//!
//! # Composing STMs between non-initial epochs
//!
//! [`PropagatedState::stm`] is the cumulative state-transition matrix
//! \\(\\Phi(t_i, t_0)\\) at each output epoch \\(t_i\\), referenced
//! to the orbit's *initial* epoch \\(t_0\\). To get the segment STM
//! between two non-initial epochs, compose:
//!
//! \\[
//!   \\Phi(t_b, t_a) = \\Phi(t_b, t_0)\\, \\Phi(t_a, t_0)^{-1}
//! \\]

mod config;
mod result;

pub use config::{
    AdvancedIntegratorConfig, DiagnosticsConfig, EventConfig, ForceModelTier, IntegratorChoice,
    OriginSwitchingConfig, PropagationConfig, UncertaintyMethod,
};
pub use result::{
    CovarianceKind, CovarianceQuality, Event, PropagatedState, PropagationResult, TaggedCovariance,
    TargetFunctional,
};

// Internal types other modules in this crate reach for via the
// `crate::propagate::` path — re-exported here so the move into
// submodules does not break callers (ephemeris, od, session).
pub(crate) use config::PropConfigKeep;

use std::ffi::CStr;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::orbit::Orbit;

impl Context {
    /// Propagate one or more orbits to a list of target epochs.
    ///
    /// Returns a [`PropagationResult`] containing the propagated states,
    /// the object IDs (matched to the orbits input), and any detected
    /// events (close approach, periapsis, SOI entry, etc.) along each
    /// trajectory.
    ///
    /// # State ordering
    ///
    /// Within each orbit, states are **epoch-ordered, not request-ordered**:
    /// forward epochs first in ascending order, then backward epochs in
    /// descending order. When the requested `epochs` are not already in that
    /// order, pairing states to requests by position silently associates
    /// them with the wrong epochs — join on
    /// [`PropagatedState::epoch`](crate::propagate::PropagatedState::epoch)
    /// instead.
    pub fn propagate(
        &self,
        orbits: &[Orbit],
        epochs: &[crate::Epoch],
        config: &PropagationConfig,
    ) -> Result<PropagationResult> {
        // Hold the per-orbit identifier CStrings alive across the FFI
        // call — `EmpyreanOrbit.orbit_id` / `.object_id` borrow into
        // their backing storage.
        let (ffi_orbits, _orbit_keep) = crate::orbit::orbits_to_ffi(orbits)?;
        let (ffi_config, _config_keep) = config.to_ffi_with();
        let epochs_mjd_tdb: Vec<f64> = epochs
            .iter()
            .map(|e| e.mjd_tdb())
            .collect::<Result<Vec<_>>>()?;
        let mut ffi_result = empyrean_sys::EmpyreanPropagationResult {
            states: std::ptr::null_mut(),
            num_states: 0,
            object_ids: std::ptr::null_mut(),
            events: std::ptr::null_mut(),
            num_events: 0,
            mixtures: std::ptr::null_mut(),
            num_mixtures: 0,
            // Opaque retained-result handle, filled by empyrean_propagate
            // and freed by empyrean_propagation_result_free.
            lazy_handle: std::ptr::null_mut(),
        };
        let code = unsafe {
            empyrean_sys::empyrean_propagate(
                self.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                epochs_mjd_tdb.as_ptr(),
                epochs_mjd_tdb.len(),
                &ffi_config,
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        marshal_propagation_result(ffi_result, orbits.len())
    }
}

/// Marshal a populated FFI propagation result into the safe
/// [`PropagationResult`], retaining the raw result so the lazy
/// tagged-covariance accessors stay callable (it is freed when the returned
/// value drops).
///
/// Shared by the one-shot [`Context::propagate`] and the pre-built
/// [`BuiltSystem::propagate`](crate::BuiltSystem::propagate) so both produce
/// byte-identical output — the handle only changes *when* the force model is
/// assembled, never the marshaling. `num_orbits` sizes the object-id array
/// (one id per input orbit).
pub(crate) fn marshal_propagation_result(
    ffi_result: empyrean_sys::EmpyreanPropagationResult,
    num_orbits: usize,
) -> Result<PropagationResult> {
    // `slice::from_raw_parts` requires a non-null pointer even for
    // length 0, but the C ABI hands back a null pointer for an empty
    // array (e.g. no detected events), so guard each one.
    let states: Vec<PropagatedState> = if ffi_result.states.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ffi_result.states, ffi_result.num_states) }
            .iter()
            .map(PropagatedState::from_ffi)
            .collect::<Result<_>>()?
    };
    let object_ids = if ffi_result.object_ids.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ffi_result.object_ids, num_orbits) }
            .iter()
            .map(|&p| {
                if p.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
                }
            })
            .collect()
    };
    let events = if ffi_result.events.is_null() {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ffi_result.events, ffi_result.num_events) }
            .iter()
            .map(Event::from_ffi)
            .collect()
    };

    // Retain the FFI result (rather than freeing it here) so the lazy
    // tagged-covariance accessors stay callable; it is freed when the
    // returned `PropagationResult` drops. The owned `states` / `object_ids`
    // / `events` above are independent copies.
    Ok(PropagationResult::new(
        states, object_ids, events, ffi_result,
    ))
}
