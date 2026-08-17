//! Ephemeris generation (predicted RA/Dec for observers).

use crate::context::Context;
use crate::coordinate::Frame;
use crate::error::{Error, Result};
use crate::observers::{Observer, obs_code_from_bytes};
use crate::orbit::Orbit;
use crate::propagate::{
    DiagnosticsConfig, EventConfig, ForceModelTier, PropagationConfig, UncertaintyMethod,
};
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
    /// Magnitude uncertainty (1σ). Finite only when photometry is
    /// enabled AND the input orbit carried at least one of a state
    /// covariance or a photometric covariance
    /// ([`Orbit::with_photometry_covariance`](crate::Orbit::with_photometry_covariance))
    /// AND what it carried contracts to a strictly positive variance;
    /// NaN otherwise. A carried covariance is not sufficient on its own:
    /// an all-zero \\(3 \times 3\\), or a non-PSD one that contracts to
    /// \\(\le 0\\), still reports NaN.
    ///
    /// Both contributions are summed in quadrature,
    /// \\[
    ///   \sigma_V = \sqrt{\sigma^2_{\text{photo}} + \sigma^2_{\text{state}}},
    /// \\]
    /// where \\(\sigma_{\text{state}}\\) is the state contribution and
    /// \\(\sigma_{\text{photo}}\\) contracts the orbit's photometric
    /// \\(3 \times 3\\) over \\((H, \text{slope}_1, \text{slope}_2)\\)
    /// against the **full** magnitude Jacobian,
    /// \\[
    ///   \sigma^2_{\text{photo}} = J \Sigma_p J^\top, \quad
    ///   J = \left[\frac{\partial V}{\partial H},
    ///             \frac{\partial V}{\partial \text{slope}_1},
    ///             \frac{\partial V}{\partial \text{slope}_2}\right].
    /// \\]
    ///
    /// \\(V = H + 5\log_{10}(r\Delta) + \phi(\alpha)\\) gives
    /// \\(\partial V/\partial H \equiv 1\\), so an orbit with **no state
    /// covariance** and a photometric covariance of the H-only shape
    /// \\(\mathrm{diag}(\sigma_H^2, 0, 0)\\) — what an H-only fit emits —
    /// reports \\(\sigma_V = \sigma_H\\) exactly. The slope terms do not
    /// drop out of any other shape: slope variances and \\(H\\)–slope
    /// covariances contract against \\(\partial V/\partial\text{slope}\\),
    /// which vanishes only at zero phase angle, so any covariance
    /// carrying them reports \\(\sigma_V > \sigma_H\\). An SBDB-queried
    /// orbit is the common case — its published
    /// \\(\mathrm{diag}(\sigma_H^2, \sigma_G^2, 0)\\) makes
    /// \\(\sigma_V\\) strictly larger than \\(\sigma_H\\).
    ///
    /// The two terms are combined as independent. They are not strictly
    /// independent — a fitted \\(\sigma_H\\) is conditional on the
    /// fitted state, because the photometric fit holds the geometry
    /// \\((r, \Delta, \alpha)\\) exact — and no joint
    /// state↔photometry covariance is computed anywhere in the stack, so
    /// there is no cross term to add. The resulting \\(\sigma_V\\) is
    /// therefore mildly conservative, which is the safe direction.
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
    /// 6×6 sky-plane covariance over (rho, RA, Dec, vrho, vRA, vDec) in
    /// (AU, deg) units, row-major. `None` when the input orbit carried no
    /// state covariance (no uncertainty path ran).
    pub covariance: Option<[[f64; 6]; 6]>,
    /// Aberrated (light-time corrected) barycentric ICRF Cartesian state
    /// `[x, y, z, vx, vy, vz]` (AU, AU/day) at the photon-emission epoch.
    /// NaN-filled in the (never-observed-today) case where the engine
    /// produced no aberrated state for the row.
    pub aberrated_state: [f64; 6],
    /// 6×6 Cartesian covariance of the aberrated state, row-major. `None`
    /// when the input orbit carried no state covariance.
    pub aberrated_covariance: Option<[[f64; 6]; 6]>,
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
            covariance: (e.has_covariance != 0).then_some(e.covariance),
            aberrated_state: e.aberrated_state,
            aberrated_covariance: (e.has_aberrated_covariance != 0)
                .then_some(e.aberrated_covariance),
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Observation-sensitivity row order
// ────────────────────────────────────────────────────────────────────
//
// Row indices into [`ObservationSensitivity::jacobian`] and
// [`ObservationSensitivity::hessian`]; both carry the same six output
// rows in the same order.
//
// The angles are in degrees and the range in AU, so a wrong row is
// wrong in unit as well as in observable — reading row 0 as RA yields a
// range partial in AU, which is finite, plausible, and silently wrong.
// Use these instead of literals.

/// Row of the range (topocentric distance) partials, in AU per input unit.
pub const SENSITIVITY_ROW_RANGE: usize = 0;
/// Row of the right-ascension partials, in degrees per input unit.
pub const SENSITIVITY_ROW_RA: usize = 1;
/// Row of the declination partials, in degrees per input unit.
pub const SENSITIVITY_ROW_DEC: usize = 2;
/// Row of the range-rate partials, in AU/day per input unit.
pub const SENSITIVITY_ROW_VRANGE: usize = 3;
/// Row of the RA-rate partials, in deg/day per input unit. The rate is
/// dRA/dt, not scaled by cos(Dec).
pub const SENSITIVITY_ROW_VRA: usize = 4;
/// Row of the Dec-rate partials, in deg/day per input unit.
pub const SENSITIVITY_ROW_VDEC: usize = 5;

/// Observation-sensitivity row: the partial derivatives of the sky-plane
/// observable w.r.t. the input state, for one `(orbit, observer, epoch)`.
///
/// Produced whenever the ephemeris propagation traced the STM: from an
/// input covariance under an analytic uncertainty method, or from
/// [`PropagationConfig::compute_stm`](crate::PropagationConfig::compute_stm)
/// on its own. The flag reaches the engine on this path, so partials can
/// be requested for an orbit that carries no covariance at all.
///
/// The six output rows are the topocentric spherical observable, in the
/// order given by the `SENSITIVITY_ROW_*` constants — see
/// [`jacobian`](Self::jacobian).
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
    ///
    /// Element `(row, col)` is `jacobian[row * n_params as usize + col]`.
    /// Columns `0..6` are the input Cartesian state, in the frame and
    /// origin the [`frame`](Self::frame) / [`origin`](Self::origin)
    /// fields tag; any further columns are the extra solved-for
    /// parameters [`n_params`](Self::n_params) counts.
    ///
    /// The six rows, in order:
    ///
    /// | row | constant                                  | observable | unit per input unit |
    /// |-----|-------------------------------------------|------------|---------------------|
    /// | 0   | [`SENSITIVITY_ROW_RANGE`]                  | range      | AU                  |
    /// | 1   | [`SENSITIVITY_ROW_RA`]                     | RA         | deg                 |
    /// | 2   | [`SENSITIVITY_ROW_DEC`]                    | Dec        | deg                 |
    /// | 3   | [`SENSITIVITY_ROW_VRANGE`]                 | range rate | AU/day              |
    /// | 4   | [`SENSITIVITY_ROW_VRA`]                    | RA rate    | deg/day             |
    /// | 5   | [`SENSITIVITY_ROW_VDEC`]                   | Dec rate   | deg/day             |
    ///
    /// The RA rate is dRA/dt, **not** scaled by cos(Dec). Projecting an
    /// input covariance onto the sky plane therefore means contracting
    /// rows 1 and 2 — row 0 is range, and reading it as RA yields a
    /// number in the wrong observable and the wrong unit.
    pub jacobian: Vec<f64>,
    /// Hessian ∂²(observable)/∂(input)², row-major `[6][n_params][n_params]`
    /// flattened. Empty unless a second-order method ran.
    ///
    /// Leading index is the observable, in the same order and the same
    /// units-per-input-unit as [`jacobian`](Self::jacobian) — index it
    /// with the same `SENSITIVITY_ROW_*` constants. Element `(row, i, j)`
    /// is `hessian[(row * np + i) * np + j]` with `np = n_params as usize`.
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
/// unless the propagation traced the STM — see [`ObservationSensitivity`]
/// for what makes it trace).
#[derive(Debug, Clone, PartialEq)]
pub struct EphemerisResult {
    /// Ephemeris entries (RA/Dec + diagnostics), one per observation.
    pub entries: Vec<EphemerisEntry>,
    /// Observation-sensitivity rows (Jacobian/Hessian). Empty on the
    /// f64-only path — the path taken when the orbit carries no
    /// covariance *and*
    /// [`PropagationConfig::compute_stm`](crate::PropagationConfig::compute_stm)
    /// is unset. Setting that flag switches the run to hyperdual
    /// integration and populates these rows.
    pub sensitivity: Vec<ObservationSensitivity>,
    /// Non-fatal generation warnings, in the order the engine emitted
    /// them. Empty when the run completed with nothing to report. Each
    /// human-readable message names the affected orbit id / observatory
    /// code / epoch (MJD TDB) where applicable — e.g. Earth-orientation
    /// kernel coverage gaps handled by the analytic IAU 2006 fallback,
    /// or rows whose observation-sensitivity chain was skipped
    /// (astrometry present, partials absent).
    pub warnings: Vec<String>,
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
    /// Check that the config asks for nothing ephemeris generation
    /// cannot deliver.
    ///
    /// `propagation.events` and `propagation.diagnostics` have no meaning
    /// here: ephemeris generation runs with event detection and
    /// timeseries diagnostics off, and [`EphemerisResult`] carries
    /// neither. [`PropagationConfig`] is the shared struct — it is
    /// [`Default`]-constructed with event detection *on*, because that is
    /// right for `propagate` — so leaving those blocks at their defaults
    /// is not a request, and is simply not sent across the FFI boundary.
    /// *Changing* one is a request that could only be dropped, so it is
    /// an error instead.
    ///
    /// Called for you by [`Context::generate_ephemeris`]; exposed so a
    /// caller assembling a config from user input can surface the same
    /// message before doing the work.
    pub fn validate(&self) -> Result<()> {
        if self.propagation.events != EventConfig::default() {
            return Err(Error::invalid_input(
                "ephemeris config: propagation.events was modified, but ephemeris \
                 generation runs with event detection disabled and EphemerisResult \
                 carries no events — the request could only be dropped. Leave \
                 propagation.events at its default and call propagate() when you need \
                 event detection.",
            ));
        }
        if self.propagation.diagnostics != DiagnosticsConfig::default() {
            return Err(Error::invalid_input(
                "ephemeris config: propagation.diagnostics was modified, but ephemeris \
                 generation produces no diagnostics timeseries — the request could only \
                 be dropped. Leave propagation.diagnostics at its default and call \
                 propagate() when you need per-trajectory diagnostics.",
            ));
        }
        Ok(())
    }

    /// Build the C-ABI representation. Returns the FFI struct plus a
    /// keepalive that owns the raw arrays the FFI struct points into.
    /// Runs [`validate`](Self::validate) first.
    pub(crate) fn to_ffi_with(
        &self,
    ) -> Result<(
        empyrean_sys::EmpyreanEphemerisConfig,
        crate::propagate::PropConfigKeep,
    )> {
        self.validate()?;
        let (mut prop_ffi, keep) = self.propagation.to_ffi_with();
        // Send the "not requested" state the C ABI defines for these two
        // blocks (all-zero, NaN thresholds). Without this the wrapper's
        // own `EventConfig::default()` — every detection flag on, because
        // that is the propagate default — would arrive as an explicit
        // request the C layer is obliged to refuse.
        prop_ffi.events = empyrean_sys::EmpyreanEventConfig {
            close_approaches: 0,
            impacts: 0,
            atmospheric: 0,
            possible_impacts: 0,
            shadow_events: 0,
            num_body_filter: 0,
            body_filter_naif: std::ptr::null(),
            dense_output: 0,
            dense_output_cadence_days: 0.0,
        };
        prop_ffi.diagnostics = empyrean_sys::EmpyreanDiagnosticsConfig {
            sensitivity: 0,
            nonlinearity: 0,
            lyapunov: 0,
            keyholes: 0,
            bifurcations: 0,
            sample_stride: 0,
            sensitivity_threshold: f64::NAN,
            lyapunov_threshold: f64::NAN,
            nonlinearity_threshold: f64::NAN,
        };
        let cfg = empyrean_sys::EmpyreanEphemerisConfig {
            propagation: prop_ffi,
            max_light_time_iterations: self.max_light_time_iterations,
            light_time_tolerance_days: self.light_time_tolerance_days,
            compute_diagnostics: u8::from(self.compute_diagnostics),
        };
        Ok((cfg, keep))
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
            warnings: std::ptr::null_mut(),
            num_warnings: 0,
        };
        let (ffi_config, _config_keep) = config.to_ffi_with()?;
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
                // Carried through rather than pinned to ICRF/SSB: the C
                // ABI refuses any other basis for ephemeris generation,
                // so an observer that came back in a plotting basis
                // fails by name here instead of being read as if it
                // were the construction basis.
                frame: o.frame as i32,
                origin: o.origin.naif_id(),
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
    let warnings: Vec<String> = if result.warnings.is_null() {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(result.warnings, result.num_warnings)
                .iter()
                .map(|&p| {
                    if p.is_null() {
                        String::new()
                    } else {
                        CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                })
                .collect()
        }
    };
    unsafe { empyrean_sys::empyrean_ephemeris_result_free(result) };
    EphemerisResult {
        entries,
        sensitivity,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observers::Observer;

    /// The row order is a contract shared with the C ABI, so the values
    /// are pinned rather than merely self-consistent: rotating them
    /// silently re-points every consumer's index at a different
    /// observable, in a different unit.
    #[test]
    fn sensitivity_row_constants_have_their_contract_values() {
        assert_eq!(SENSITIVITY_ROW_RANGE, 0);
        assert_eq!(SENSITIVITY_ROW_RA, 1);
        assert_eq!(SENSITIVITY_ROW_DEC, 2);
        assert_eq!(SENSITIVITY_ROW_VRANGE, 3);
        assert_eq!(SENSITIVITY_ROW_VRA, 4);
        assert_eq!(SENSITIVITY_ROW_VDEC, 5);
    }

    /// Every row of the `[6][n_params]` Jacobian is addressable, and the
    /// six constants address six distinct rows — a duplicate or an
    /// out-of-range index would slice the wrong observable rather than
    /// fail.
    #[test]
    fn sensitivity_row_constants_cover_all_six_rows() {
        let mut rows = [
            SENSITIVITY_ROW_RANGE,
            SENSITIVITY_ROW_RA,
            SENSITIVITY_ROW_DEC,
            SENSITIVITY_ROW_VRANGE,
            SENSITIVITY_ROW_VRA,
            SENSITIVITY_ROW_VDEC,
        ];
        rows.sort_unstable();
        assert_eq!(rows, [0, 1, 2, 3, 4, 5]);
    }

    /// A 4-character observatory code must be a loud error at the FFI
    /// boundary: clipped to 3 bytes it would silently alias a different
    /// observatory.
    #[test]
    fn four_char_obs_code_is_rejected() {
        let observer = Observer {
            obs_code: "W68a".to_string(),
            epoch: crate::Epoch::from_mjd_tdb(61000.0),
            position: [1.0, 0.0, 0.0],
            velocity: [0.0, 0.01, 0.0],
            observing_night: -1,
            frame: crate::Frame::ICRF,
            origin: crate::Origin::SSB,
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
            frame: crate::Frame::ICRF,
            origin: crate::Origin::SSB,
        };
        let ffi = observers_to_ffi(&[observer]).expect("3-character code marshals");
        assert_eq!(&ffi[0].obs_code, b"W68\0");
    }
}

/// [`EphemerisConfig::validate`] and the FFI narrowing it guards.
/// `PropagationConfig` is shared with `propagate`, so its defaults are
/// the propagate defaults; the question each of these answers is which of
/// those defaults count as a request.
#[cfg(test)]
mod ephemeris_config_validation_tests {
    use super::*;
    use crate::coordinate::Origin;

    /// The default config must convert. `EventConfig::default()` has
    /// every detection flag on because that is right for `propagate`;
    /// treating that as an ephemeris event request would reject every
    /// call made without an explicit config.
    #[test]
    fn the_default_config_is_not_an_event_request() {
        EphemerisConfig::default()
            .validate()
            .expect("the default config must be usable for ephemeris");
        EphemerisConfig::with_force_model(ForceModelTier::Standard)
            .validate()
            .expect("with_force_model must be usable for ephemeris");
    }

    /// Setting the perturber exclusion is a supported request and stays
    /// one — it is the only way to keep an SB441-N16 body out of its own
    /// perturber set.
    #[test]
    fn excluding_a_perturber_is_a_supported_request() {
        let mut cfg = EphemerisConfig::default();
        cfg.propagation.excluded_perturbers = vec![Origin::asteroid(7)];
        cfg.validate().expect("excluded_perturbers is supported");

        let (ffi, _keep) = cfg.to_ffi_with().expect("config marshals");
        assert_eq!(ffi.propagation.num_excluded_perturbers, 1);
        assert!(!ffi.propagation.excluded_perturbers_naif.is_null());
        assert_eq!(
            unsafe { *ffi.propagation.excluded_perturbers_naif },
            2_000_007,
            "the NAIF id must reach the FFI struct"
        );
    }

    /// `compute_stm` is a propagation-level knob the ephemeris path
    /// honours, and the sentence in the config docs claiming so is only
    /// true if it survives the narrowing. Pinned on the wire.
    #[test]
    fn compute_stm_reaches_the_wire() {
        let mut cfg = EphemerisConfig::default();
        assert_eq!(
            cfg.to_ffi_with()
                .expect("marshals")
                .0
                .propagation
                .compute_stm,
            0,
            "off by default"
        );
        cfg.propagation.compute_stm = true;
        assert_eq!(
            cfg.to_ffi_with()
                .expect("marshals")
                .0
                .propagation
                .compute_stm,
            1,
            "compute_stm must reach the FFI struct — the C layer narrows it into \
             EphemerisPropagationConfig from there"
        );
    }

    /// The overlap policy reaches the wire too. It is the knob that
    /// decides whether an SB441-N16 body can be generated for at all.
    #[test]
    fn the_overlap_policy_reaches_the_wire() {
        let mut cfg = EphemerisConfig::default();
        assert_eq!(
            cfg.to_ffi_with()
                .expect("marshals")
                .0
                .propagation
                .ephemeris_overlap_policy,
            empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK
        );
        cfg.propagation.ephemeris_overlap_policy =
            crate::EphemerisOverlapPolicy::ExcludeAndIntegrate;
        assert_eq!(
            cfg.to_ffi_with()
                .expect("marshals")
                .0
                .propagation
                .ephemeris_overlap_policy,
            empyrean_sys::EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE
        );
    }

    /// Turning a detection flag *off* is still a modification, and still
    /// unanswerable — the point is that the caller reasoned about events
    /// on a call that has none.
    #[test]
    fn a_modified_event_block_is_refused() {
        let mut cfg = EphemerisConfig::default();
        cfg.propagation.events.close_approaches = false;
        let err = cfg
            .validate()
            .expect_err("a modified events block must fail");
        assert!(
            err.message.contains("propagation.events"),
            "error names the field: {}",
            err.message
        );
    }

    #[test]
    fn a_modified_diagnostics_block_is_refused() {
        let mut cfg = EphemerisConfig::default();
        cfg.propagation.diagnostics.lyapunov = true;
        let err = cfg
            .validate()
            .expect_err("a modified diagnostics block must fail");
        assert!(
            err.message.contains("propagation.diagnostics"),
            "error names the field: {}",
            err.message
        );
    }

    /// The wrapper sends the C ABI's "not requested" state for both
    /// blocks, so the C layer's refusal never fires on a valid wrapper
    /// config.
    #[test]
    fn the_unsupported_blocks_are_zeroed_on_the_wire() {
        let cfg = EphemerisConfig::default();
        let (ffi, _keep) = cfg.to_ffi_with().expect("config marshals");
        let e = &ffi.propagation.events;
        assert_eq!(
            (
                e.close_approaches,
                e.impacts,
                e.atmospheric,
                e.possible_impacts,
                e.shadow_events,
                e.dense_output
            ),
            (0, 0, 0, 0, 0, 0)
        );
        assert_eq!(e.num_body_filter, 0);
        assert!(e.body_filter_naif.is_null());
        assert_eq!(e.dense_output_cadence_days, 0.0);

        let d = &ffi.propagation.diagnostics;
        assert_eq!(
            (
                d.sensitivity,
                d.nonlinearity,
                d.lyapunov,
                d.keyholes,
                d.bifurcations
            ),
            (0, 0, 0, 0, 0)
        );
        assert_eq!(d.sample_stride, 0);
        // NaN is the ABI's explicit `None` for these thresholds.
        assert!(d.sensitivity_threshold.is_nan());
        assert!(d.lyapunov_threshold.is_nan());
        assert!(d.nonlinearity_threshold.is_nan());
    }
}
