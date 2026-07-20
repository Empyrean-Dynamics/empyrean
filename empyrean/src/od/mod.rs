//! Orbit determination — fit an orbit to ADES astrometric observations.
//!
//! Three entry points sit on [`Context`]:
//! [`Context::determine`] (full IOD → DC pipeline),
//! [`Context::evaluate`] (residuals only, no fit), and
//! [`Context::refine`] (Bayesian update against a prior orbit). All
//! three consume an [`Observations`] set and an [`ODConfig`], and
//! return either a [`DetermineResult`] (fit + diagnostics +
//! acceptability verdict) or an [`EvaluateResult`] (residuals only).
//!
//! For interactive workflows where you want to mask observations and
//! compare fits, see [`Session`](crate::Session).
//!
//! # Example: full pipeline with the production hot path
//!
//! ```no_run
//! use empyrean::{Context, ODConfig};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let obs = ctx.read_ades("apophis_2004_2021.psv")?;
//! let cfg = ODConfig::default(); // VFC17 weights + EFCC2020 debias + auto-escalate
//!
//! let fit = ctx.determine(&obs, None, &cfg)?;
//! println!(
//!     "converged={}, χ²_red={:.2}, fit_acceptable={}",
//!     fit.converged, fit.summary.reduced_chi2, fit.acceptability.fit_acceptable,
//! );
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! # Reading the acceptability verdict
//!
//! [`AcceptabilityReport::fit_acceptable`] is the AND of the
//! fit-quality gates (convergence, positive-definite covariance,
//! reduced χ², RMS, AT/CT residual isotropy).
//! [`AcceptabilityReport::extrapolation_acceptable`] is
//! `fit_acceptable` AND the trustworthy-forward-propagation gates
//! (arc length, fractional σₐ). Use the first to gate publication;
//! the second to gate forward propagation, ephemeris generation, or
//! impact-risk assessment. Tighten thresholds in
//! [`AcceptabilityThresholds`] for impact-monitoring orbits.

mod config;
mod debiasing;
mod nuisance;
mod observation;
mod rejection;
mod result;
mod weighting;

pub use config::{
    AcceptabilityThresholds, AutoEscalationPolicy, IODConfig, ODConfig, PhotometryConfig,
};
pub use debiasing::{DebiasingConfig, DebiasingResolution};
pub use nuisance::StationRaDecConfig;
pub use observation::{Observation, Observations, RadarMeasurement, RadarObservation};
pub use rejection::{RejectionConfig, RejectionKind};
pub use result::{
    AcceptabilityReport, BandStat, CovarianceRepresentation, DetermineResult, EvaluateResult,
    GateRecord, ObservationResidual, OriginPolicy, OutputEpoch, PhotometryModel, PhotometryResult,
    RejectionReason, ResidualSummary, SolveFor, SolveForParams, SolvedCovariance, StationBias,
};
pub use weighting::{SigmaPolicy, WeightingConfig, WeightingLayer, WeightingPreset};

use std::ffi::CString;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::orbit::Orbit;
use crate::propagate::ForceModelTier;

impl Context {
    /// Parse ADES PSV observations from a file path or a PSV string.
    ///
    /// If `path_or_content` has no newlines, it's treated as a file
    /// path. Otherwise, it's treated as the PSV content itself.
    pub fn read_ades(&self, path_or_content: &str) -> Result<Observations> {
        let _ = self; // reserved for future context-dependent parsing
        let c_input = CString::new(path_or_content)
            .map_err(|_| Error::invalid_input("input contains a NUL byte"))?;
        let mut ptr: *mut empyrean_sys::EmpyreanObservation = std::ptr::null_mut();
        let mut num: usize = 0;
        let mut radar_ptr: *mut empyrean_sys::EmpyreanRadarObservation = std::ptr::null_mut();
        let mut radar_num: usize = 0;
        let code = unsafe {
            empyrean_sys::empyrean_read_ades(
                c_input.as_ptr(),
                &mut ptr,
                &mut num,
                &mut radar_ptr,
                &mut radar_num,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        Ok(Observations::from_raw_parts(ptr, num, radar_ptr, radar_num))
    }

    /// Run the full orbit-determination pipeline (IOD → differential correction).
    ///
    /// Pass `None` for `initial_orbits` to use the internal IOD, or pass
    /// seed orbits to skip IOD and start the differential correction
    /// from the provided states.
    pub fn determine(
        &self,
        observations: &Observations,
        initial_orbits: Option<&[Orbit]>,
        config: &ODConfig,
    ) -> Result<DetermineResult> {
        let mut _orbit_keep: Vec<crate::orbit::OrbitFfiKeep> = Vec::new();
        let ffi_initial: Option<Vec<_>> = match initial_orbits {
            Some(orbs) => {
                _orbit_keep.reserve(orbs.len());
                let v: Vec<_> = orbs
                    .iter()
                    .map(|o| {
                        let (ffi, keep) = o.to_ffi_with_keep()?;
                        _orbit_keep.push(keep);
                        Ok(ffi)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Some(v)
            }
            None => None,
        };
        let (init_ptr, init_len) = match &ffi_initial {
            Some(v) => (v.as_ptr(), v.len()),
            None => (std::ptr::null(), 0),
        };
        let (obs_ptr, obs_len) = observations.as_ffi_slice();
        let (radar_ptr, radar_len) = observations.as_radar_ffi_slice();

        let mut result = empyrean_sys::EmpyreanODResult::default();
        let (ffi_config, _perturbers_keep) = config.to_ffi_with();
        let code = unsafe {
            empyrean_sys::empyrean_determine(
                self.as_raw(),
                obs_ptr,
                obs_len,
                radar_ptr,
                radar_len,
                init_ptr,
                init_len,
                &ffi_config,
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let det = ffi_od_result_to_rust(&result);
        unsafe { empyrean_sys::empyrean_od_result_free(&mut result) };
        det
    }

    /// Evaluate a candidate orbit against observations without fitting.
    pub fn evaluate(
        &self,
        orbit: &Orbit,
        observations: &Observations,
        config: &ODConfig,
    ) -> Result<EvaluateResult> {
        let (ffi_orbit, _orbit_keep) = orbit.to_ffi_with_keep()?;
        let (obs_ptr, obs_len) = observations.as_ffi_slice();

        let mut result = empyrean_sys::EmpyreanEvaluateResult::default();
        let (ffi_config, _perturbers_keep) = config.to_ffi_with();
        let code = unsafe {
            empyrean_sys::empyrean_evaluate(
                self.as_raw(),
                &ffi_orbit,
                obs_ptr,
                obs_len,
                &ffi_config,
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let residuals = unsafe {
            std::slice::from_raw_parts(result.observations, result.num_observations)
                .iter()
                .map(ObservationResidual::from_ffi)
                .collect()
        };
        let summary = ResidualSummary::from_ffi(&result.summary);
        unsafe { empyrean_sys::empyrean_evaluate_result_free(&mut result) };

        Ok(EvaluateResult { residuals, summary })
    }

    /// Refine an orbit with observations using a Bayesian prior.
    ///
    /// Requires the input orbit to carry a covariance matrix.
    pub fn refine(
        &self,
        orbit: &Orbit,
        observations: &Observations,
        config: &ODConfig,
    ) -> Result<DetermineResult> {
        let (ffi_orbit, _orbit_keep) = orbit.to_ffi_with_keep()?;
        let (obs_ptr, obs_len) = observations.as_ffi_slice();

        let mut result = empyrean_sys::EmpyreanODResult::default();
        let (ffi_config, _perturbers_keep) = config.to_ffi_with();
        let code = unsafe {
            empyrean_sys::empyrean_refine(
                self.as_raw(),
                &ffi_orbit,
                obs_ptr,
                obs_len,
                &ffi_config,
                &mut result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let det = ffi_od_result_to_rust(&result);
        unsafe { empyrean_sys::empyrean_od_result_free(&mut result) };
        det
    }
}

/// Internal converter shared with [`crate::session`].
pub(crate) fn ffi_od_result_to_rust_pub(
    result: &empyrean_sys::EmpyreanODResult,
) -> crate::error::Result<DetermineResult> {
    ffi_od_result_to_rust(result)
}

fn ffi_od_result_to_rust(
    result: &empyrean_sys::EmpyreanODResult,
) -> crate::error::Result<DetermineResult> {
    let orbit = ffi_od_result_orbit(result)?;
    let residuals = unsafe {
        std::slice::from_raw_parts(result.observations, result.num_observations)
            .iter()
            .map(ObservationResidual::from_ffi)
            .collect()
    };
    let summary = ResidualSummary::from_ffi(&result.summary);
    let acceptability = AcceptabilityReport::from_ffi(&result.acceptability);
    let force_model_used = match result.force_model_used {
        0 => ForceModelTier::Approximate,
        1 => ForceModelTier::Basic,
        _ => ForceModelTier::Standard,
    };
    let station_biases: Vec<StationBias> =
        if result.station_biases.is_null() || result.num_station_biases == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(result.station_biases, result.num_station_biases)
                    .iter()
                    .map(StationBias::from_ffi)
                    .collect()
            }
        };
    let solved_covariance = (result.has_solved_covariance != 0)
        .then(|| SolvedCovariance::from_ffi(&result.solved_covariance));
    let thrust_delta_m_per_s: Vec<[f64; 3]> =
        result.thrust_delta_m_per_s[..result.thrust_delta_count as usize].to_vec();
    // dv_frame is only meaningful when a thrust segment was solved.
    let dv_frame = (result.thrust_delta_count > 0)
        .then(|| crate::coordinate::int_to_frame(result.dv_frame).ok())
        .flatten();
    let photometry =
        (result.has_photometry != 0).then(|| PhotometryResult::from_ffi(&result.photometry));
    Ok(DetermineResult {
        orbit,
        residuals,
        summary,
        iterations: result.iterations,
        update_norm: result.update_norm,
        converged: result.converged != 0,
        covariance: result.covariance,
        covariance_representation: CovarianceRepresentation::from_int(
            result.covariance_representation,
        ),
        covariance_9x9: (result.has_covariance_9x9 != 0).then_some(result.covariance_9x9),
        non_grav_delta: (result.has_non_grav_delta != 0).then_some(result.non_grav_delta),
        rejection_passes: result.rejection_passes,
        num_oppositions_fit: result.num_oppositions_fit,
        force_model_used,
        // Reconstruct the solved axes: an Explicit fit's exact set is
        // recovered from the covariance slot tags, not the coarse code.
        solve_for_used: SolveForParams::from_result(
            result.solve_for_used,
            solved_covariance.as_ref(),
        ),
        acceptability,
        station_biases,
        solved_covariance,
        dt_delta: (result.has_dt_delta != 0).then_some(result.dt_delta),
        amrat_delta: (result.has_amrat_delta != 0).then_some(result.amrat_delta),
        thrust_delta_m_per_s,
        dv_frame,
        photometry,
    })
}

/// Build the re-feedable fitted [`Orbit`] from a C-ABI OD result: the
/// Cartesian state + covariance from `result.orbit`, plus the **absolute**
/// non-gravitational model from `result.non_grav` (when present). This orbit
/// is what `evaluate` / `refine` / `propagate` /
/// `compute_impact_probabilities` accept directly — no reconstruction, no
/// silently-dropped force model.
fn ffi_od_result_orbit(result: &empyrean_sys::EmpyreanODResult) -> crate::error::Result<Orbit> {
    use crate::coordinate::Origin;
    let s = &result.orbit;
    let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
        Error::invalid_input(format!(
            "C ABI returned unknown NAIF id for origin: {}",
            s.origin
        ))
    })?;
    let frame = crate::coordinate::int_to_frame(s.frame)?;
    let mut state = crate::CoordinateState::cartesian(
        crate::Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
        [s.x, s.y, s.z, s.vx, s.vy, s.vz],
        frame,
        origin,
    );
    if s.has_covariance != 0 {
        state = state.with_covariance(s.covariance);
    }
    let mut orbit = Orbit::new(state);
    if result.has_non_grav != 0 {
        let ng = &result.non_grav;
        orbit = orbit.with_nongrav(ng.a1, ng.a2, ng.a3);
        // Any non-zero exponent selects the explicit Marsden–Sekanina g(r);
        // all-zero is the inverse-square default `with_nongrav` already set.
        if ng.ng_alpha != 0.0
            || ng.ng_r0 != 0.0
            || ng.ng_m != 0.0
            || ng.ng_n != 0.0
            || ng.ng_k != 0.0
        {
            orbit = orbit.with_g_function(ng.ng_alpha, ng.ng_r0, ng.ng_m, ng.ng_n, ng.ng_k);
        }
        if ng.has_dt != 0 {
            orbit = orbit.with_non_grav_dt(Some(ng.non_grav_dt));
        }
        // Carry the fitted non-grav covariance so the orbit re-feeds into a
        // StateAndNonGrav refine without losing its non-grav prior.
        if ng.has_covariance != 0 {
            orbit = orbit.with_nongrav_covariance(Some(ng.covariance));
        }
    }
    Ok(orbit)
}
