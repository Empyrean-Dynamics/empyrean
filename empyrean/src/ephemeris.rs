//! Ephemeris generation (predicted RA/Dec for observers).

use crate::context::Context;
use crate::coordinate::Frame;
use crate::error::{Error, Result};
use crate::observers::{Observer, obs_code_from_bytes};
use crate::orbit::Orbit;
use crate::propagate::{ForceModelTier, PropagationConfig, UncertaintyMethod};
use std::ffi::CStr;

/// One row of predicted astrometry.
#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisEntry {
    /// Orbit identifier.
    pub orbit_id: String,
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Right ascension (degrees).
    pub ra_deg: f64,
    /// Declination (degrees).
    pub dec_deg: f64,
    /// Topocentric distance (AU).
    pub rho_au: f64,
    /// Radial velocity (AU/day).
    pub vrho_au_day: f64,
    /// RA rate (deg/day).
    pub vra_deg_day: f64,
    /// Dec rate (deg/day).
    pub vdec_deg_day: f64,
    /// One-way light time (days). NaN if unavailable.
    pub light_time_days: f64,
    /// Phase angle (degrees). NaN if unavailable.
    pub phase_angle_deg: f64,
    /// Solar elongation (degrees). NaN if unavailable.
    pub elongation_deg: f64,
    /// Heliocentric distance (AU). NaN if unavailable.
    pub heliocentric_distance_au: f64,
    /// Predicted apparent magnitude. NaN if unavailable.
    pub mag: f64,
    /// Magnitude uncertainty (1σ). Finite iff photometry is enabled AND
    /// the input orbit carried a state covariance; NaN otherwise. Today
    /// this reflects the **state contribution only** — an H-magnitude
    /// uncertainty is not yet an input, so `mag_sigma` under-reports σ_V
    /// when the H uncertainty is significant (the dominant photometric
    /// error source for short-arc orbits).
    pub mag_sigma: f64,
    /// Topocentric zenith angle (degrees). NaN if unavailable.
    pub zenith_angle_deg: f64,
    /// Topocentric azimuth, East of North (degrees). NaN if unavailable.
    pub azimuth_deg: f64,
    /// Local hour angle (degrees). NaN if unavailable.
    pub hour_angle_deg: f64,
    /// Angular separation from the Moon (degrees). NaN if unavailable.
    pub lunar_elongation_deg: f64,
    /// Position angle of motion, East of North (degrees). NaN if unavailable.
    pub position_angle_deg: f64,
    /// Total apparent sky-plane rate of motion (degrees/day). NaN if unavailable.
    pub sky_rate_deg_day: f64,
    /// MPC observatory code.
    pub obs_code: String,
}

impl EphemerisEntry {
    pub(crate) fn from_ffi(e: &empyrean_sys::EmpyreanEphemerisEntry) -> Self {
        let orbit_id = if e.orbit_id.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(e.orbit_id).to_string_lossy().into_owned() }
        };
        Self {
            orbit_id,
            epoch: crate::Epoch::from_mjd_tdb(e.epoch_mjd_tdb),
            ra_deg: e.ra_deg,
            dec_deg: e.dec_deg,
            rho_au: e.rho_au,
            vrho_au_day: e.vrho_au_day,
            vra_deg_day: e.vra_deg_day,
            vdec_deg_day: e.vdec_deg_day,
            light_time_days: e.light_time_days,
            phase_angle_deg: e.phase_angle_deg,
            elongation_deg: e.elongation_deg,
            heliocentric_distance_au: e.heliocentric_distance_au,
            mag: e.mag,
            mag_sigma: e.mag_sigma,
            zenith_angle_deg: e.zenith_angle_deg,
            azimuth_deg: e.azimuth_deg,
            hour_angle_deg: e.hour_angle_deg,
            lunar_elongation_deg: e.lunar_elongation_deg,
            position_angle_deg: e.position_angle_deg,
            sky_rate_deg_day: e.sky_rate_deg_day,
            obs_code: obs_code_from_bytes(&e.obs_code),
        }
    }
}

/// Observation-sensitivity row: the partial derivatives of the sky-plane
/// observable w.r.t. the input state, for one `(orbit, observer, epoch)`.
/// Produced when the ephemeris uncertainty method traced the STM.
///
/// The Jacobian composes ∂(obs)/∂(state at t_obs) · Φ(t_obs, t₀) and
/// omits the light-time terms: the −v·∂τ/∂x partial, and the STM is
/// sampled at t_obs rather than at emission (t_obs − τ). Both terms are
/// O(τ) and land in the velocity columns of the angle rows — fractional
/// error ≈ τ/Δt with τ ≈ 0.006–0.017 d, negligible for multi-night arcs
/// but growing as the arc shrinks toward intra-night.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationSensitivity {
    /// Orbit identifier.
    pub orbit_id: String,
    /// Object identifier, when present.
    pub object_id: Option<String>,
    /// MPC observatory code.
    pub obs_code: String,
    /// Observation epoch (MJD TDB).
    pub epoch_mjd_tdb: f64,
    /// Solved-parameter dimension: 6 (state) or 9 (state + non-grav).
    pub n_params: u8,
    /// Jacobian ∂(observable)/∂(input), row-major `[6][n_params]` flattened.
    /// Empty when this epoch carries no Jacobian.
    pub jacobian: Vec<f64>,
    /// Hessian ∂²(observable)/∂(input)², row-major `[6][n_params][n_params]`
    /// flattened. Empty unless a second-order method ran.
    pub hessian: Vec<f64>,
    /// Frame of the input axis (Frame enum as int).
    pub frame: i32,
    /// Origin of the input axis (NAIF id).
    pub origin: i32,
}

impl ObservationSensitivity {
    pub(crate) fn from_ffi(e: &empyrean_sys::EmpyreanObservationSensitivity) -> Self {
        let orbit_id = if e.orbit_id.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(e.orbit_id).to_string_lossy().into_owned() }
        };
        let object_id = if e.object_id.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(e.object_id).to_string_lossy().into_owned() })
        };
        let jacobian = if e.jacobian.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(e.jacobian, e.jacobian_len).to_vec() }
        };
        let hessian = if e.hessian.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(e.hessian, e.hessian_len).to_vec() }
        };
        Self {
            orbit_id,
            object_id,
            obs_code: obs_code_from_bytes(&e.obs_code),
            epoch_mjd_tdb: e.epoch_mjd_tdb,
            n_params: e.n_params,
            jacobian,
            hessian,
            frame: e.frame,
            origin: e.origin,
        }
    }
}

/// Result of [`Context::generate_ephemeris`]: the per-`(orbit, observer,
/// epoch)` ephemeris entries plus the observation-sensitivity rows (empty
/// unless the uncertainty method traced the STM).
#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisResult {
    /// Ephemeris entries (RA/Dec + diagnostics), one per observation.
    pub entries: Vec<EphemerisEntry>,
    /// Observation-sensitivity rows (Jacobian/Hessian). Empty on the
    /// f64-only path.
    pub sensitivity: Vec<ObservationSensitivity>,
}

/// Ephemeris-generation configuration.
///
/// Mirrors the upstream `EphemerisConfig`: drives the inner
/// propagation that brings each orbit to its observation epoch, plus
/// the light-time iteration loop and the diagnostics toggle that
/// gates phase-angle / elongation / heliocentric-distance / magnitude
/// computation. Use [`EphemerisConfig::default`] for sensible
/// production settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisConfig {
    /// Inner propagation configuration applied while bringing each
    /// orbit to its observation epoch. Sets force model, uncertainty
    /// method, and output frame for the trajectory.
    pub propagation: PropagationConfig,
    /// Maximum iterations for light-time convergence. 0 selects the
    /// upstream default (3).
    pub max_light_time_iterations: usize,
    /// Tolerance (days) for light-time convergence. 0.0 selects the
    /// upstream default (1e-10).
    pub light_time_tolerance_days: f64,
    /// Whether to compute phase-angle, elongation, heliocentric
    /// distance, and apparent magnitude. Skip these when only RA/Dec
    /// are needed (DC inner loop) for a small speedup.
    pub compute_diagnostics: bool,
}

impl Default for EphemerisConfig {
    fn default() -> Self {
        Self {
            propagation: PropagationConfig::default(),
            max_light_time_iterations: 0,
            light_time_tolerance_days: 0.0,
            compute_diagnostics: true,
        }
    }
}

impl EphemerisConfig {
    /// Build the C-ABI representation. Returns the FFI struct plus a
    /// keepalive that owns the raw arrays the FFI struct points into.
    pub(crate) fn to_ffi_with(
        &self,
    ) -> (
        empyrean_sys::EmpyreanEphemerisConfig,
        crate::propagate::PropConfigKeep,
    ) {
        let (prop_ffi, keep) = self.propagation.to_ffi_with();
        let cfg = empyrean_sys::EmpyreanEphemerisConfig {
            propagation: prop_ffi,
            max_light_time_iterations: self.max_light_time_iterations,
            light_time_tolerance_days: self.light_time_tolerance_days,
            compute_diagnostics: u8::from(self.compute_diagnostics),
        };
        (cfg, keep)
    }

    /// Convenience builder: a config carrying just the requested force
    /// model, defaults for everything else.
    ///
    /// The internal propagation runs in **EclipticJ2000** because
    /// villeneuve's ephemeris pipeline assumes that integration frame
    /// when it converts the propagated state to ICRF for the
    /// observer-relative geometry. The user-facing RA/Dec output is
    /// still in ICRF — only the integration frame is overridden here.
    pub fn with_force_model(force_model: ForceModelTier) -> Self {
        Self {
            propagation: PropagationConfig {
                force_model,
                frame: Frame::EclipticJ2000,
                ..PropagationConfig::default()
            },
            ..Self::default()
        }
    }
}

impl Context {
    /// Generate predicted ephemeris for orbits as seen by observers.
    ///
    /// Returns `num_orbits * num_observers` entries, orbit-major; within
    /// each orbit, entries (and [`ObservationSensitivity`] rows) follow
    /// the **observer-input order**. Each observer carries its own epoch,
    /// so there is no separate epoch axis — positional pairing against
    /// the input observers is safe within an orbit block.
    pub fn generate_ephemeris(
        &self,
        orbits: &[Orbit],
        observers: &[Observer],
        config: &EphemerisConfig,
    ) -> Result<EphemerisResult> {
        let _ = (Frame::ICRF, UncertaintyMethod::FirstOrder); // suppress unused-import in default-config branch
        let (ffi_orbits, _orbit_keep) = crate::orbit::orbits_to_ffi(orbits)?;
        let ffi_observers = observers_to_ffi(observers)?;

        let mut result = empyrean_sys::EmpyreanEphemerisResult {
            entries: std::ptr::null_mut(),
            num_entries: 0,
            sensitivity: std::ptr::null_mut(),
            num_sensitivity: 0,
        };
        let (ffi_config, _config_keep) = config.to_ffi_with();
        let code = unsafe {
            empyrean_sys::empyrean_generate_ephemeris(
                self.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                ffi_observers.as_ptr(),
                ffi_observers.len(),
                &ffi_config,
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(marshal_ephemeris_result(&mut result))
    }
}

/// Marshal an observer batch into the FFI representation. Shared by the
/// one-shot [`Context::generate_ephemeris`] and the pre-built
/// [`BuiltSystem::generate_ephemeris`](crate::BuiltSystem::generate_ephemeris)
/// so both feed the engine byte-identical observer rows.
pub(crate) fn observers_to_ffi(
    observers: &[Observer],
) -> Result<Vec<empyrean_sys::EmpyreanObserver>> {
    observers
        .iter()
        .map(|o| {
            let mut code_bytes = [0u8; 4];
            let src = o.obs_code.as_bytes();
            // The engine's observatory registry keys 3-byte MPC codes. A
            // longer code must not be truncated: its 3-byte prefix would
            // silently resolve to a DIFFERENT observatory (wrong
            // topocentric geometry, no diagnostic).
            if src.len() > 3 {
                return Err(Error::invalid_input(format!(
                    "observatory code \"{}\" is longer than 3 bytes; \
                     4-character MPC codes are not yet supported by the \
                     engine's observatory registry",
                    o.obs_code
                )));
            }
            code_bytes[..src.len()].copy_from_slice(src);
            Ok(empyrean_sys::EmpyreanObserver {
                obs_code: code_bytes,
                epoch_mjd_tdb: o.epoch.mjd_tdb()?,
                x: o.position[0],
                y: o.position[1],
                z: o.position[2],
                vx: o.velocity[0],
                vy: o.velocity[1],
                vz: o.velocity[2],
                observing_night: o.observing_night,
            })
        })
        .collect::<Result<Vec<_>>>()
}

/// Marshal a populated FFI ephemeris result into the safe
/// [`EphemerisResult`] and free the raw result. Shared by the one-shot
/// [`Context::generate_ephemeris`] and the pre-built
/// [`BuiltSystem::generate_ephemeris`](crate::BuiltSystem::generate_ephemeris)
/// so both produce byte-identical output.
pub(crate) fn marshal_ephemeris_result(
    result: &mut empyrean_sys::EmpyreanEphemerisResult,
) -> EphemerisResult {
    let entries = if result.entries.is_null() {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(result.entries, result.num_entries)
                .iter()
                .map(EphemerisEntry::from_ffi)
                .collect()
        }
    };
    let sensitivity = if result.sensitivity.is_null() {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(result.sensitivity, result.num_sensitivity)
                .iter()
                .map(ObservationSensitivity::from_ffi)
                .collect()
        }
    };
    unsafe { empyrean_sys::empyrean_ephemeris_result_free(result) };
    EphemerisResult {
        entries,
        sensitivity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observers::Observer;

    /// A 4-character observatory code must be a loud error at the FFI
    /// boundary: clipped to 3 bytes it would silently alias a different
    /// observatory (empyrean-agp9).
    #[test]
    fn four_char_obs_code_is_rejected() {
        let observer = Observer {
            obs_code: "W68a".to_string(),
            epoch: crate::Epoch::from_mjd_tdb(61000.0),
            position: [1.0, 0.0, 0.0],
            velocity: [0.0, 0.01, 0.0],
            observing_night: -1,
        };
        let err = observers_to_ffi(&[observer])
            .expect_err("4-character observatory code must not marshal");
        let msg = err.to_string();
        assert!(msg.contains("W68a"), "error names the code: {msg}");
        assert!(
            msg.contains("longer than 3 bytes"),
            "error states the contract: {msg}"
        );
    }

    /// 3-character (and shorter) codes still marshal, NUL-padded.
    #[test]
    fn three_char_obs_code_marshals() {
        let observer = Observer {
            obs_code: "W68".to_string(),
            epoch: crate::Epoch::from_mjd_tdb(61000.0),
            position: [1.0, 0.0, 0.0],
            velocity: [0.0, 0.01, 0.0],
            observing_night: -1,
        };
        let ffi = observers_to_ffi(&[observer]).expect("3-character code marshals");
        assert_eq!(&ffi[0].obs_code, b"W68\0");
    }
}
