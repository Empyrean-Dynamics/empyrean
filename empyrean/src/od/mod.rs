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
//! let cfg = ODConfig::default(); // VFCC2017 weights + EFCC2020 debias + auto-escalate
//!
//! // `determine` fits every object in the arc; `into_single` unwraps
//! // the one-object case and refuses (naming them) if there are more.
//! let fit = ctx.determine(&obs, None, &cfg)?.into_single()?;
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
//! `fit_acceptable` AND the four selection / coverage gates: the
//! fraction of observations the fit retained, the span the *selected*
//! observations still cover, whether the most-recent observations were
//! rejected, and fractional σₐ. Use the first to gate publication; the
//! second to gate forward propagation, ephemeris generation, or
//! impact-risk assessment — and read the individual axes to say *why*
//! a fit did not clear it. Tighten thresholds in
//! [`AcceptabilityThresholds`] for impact-monitoring orbits.
//!
//! # Batches
//!
//! [`Context::determine`] fits every object the observations group
//! into and returns a [`DetermineResults`] table; a failed object is an
//! entry carrying its reason, never a missing one. See that type for
//! iteration, by-object lookup, and the single-object convenience.

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
    AcceptabilityReport, BandStat, CovarianceRepresentation, CovarianceTrust, DetermineEntry,
    DetermineFailure, DetermineFailureKind, DetermineResult, DetermineResults, EvaluateResult,
    GateRecord, MAX_THRUST_SEGMENTS, ObservationResidual, OriginPolicy, OutputEpoch,
    PhotometryModel, PhotometryResult, RadarResidual, RadarResidualKind, RejectionReason,
    ResidualSummary, SolveFor, SolveForParams, SolvedCovariance, StationBias, TrustGateEvent,
};
pub use weighting::{SigmaPolicy, WeightingConfig, WeightingLayer, WeightingPreset};

use std::ffi::CString;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::orbit::Orbit;
use crate::propagate::ForceModelTier;

impl Context {
    /// Parse ADES PSV / MPC80 observations from a file path or from the
    /// content itself.
    ///
    /// A string with no newline that names an existing file is read from
    /// disk; anything else is parsed as content. No filename can contain
    /// a newline on POSIX or Windows, so inline multi-line PSV never
    /// reaches the filesystem. This matches the Python `read_ades`
    /// resolution rule exactly.
    pub fn read_ades(&self, path_or_content: &str) -> Result<Observations> {
        let _ = self; // reserved for future context-dependent parsing
        // The C ABI's `empyrean_read_ades` parses content, so a path has
        // to be resolved here. Without this the path string itself is
        // handed to the parser and fails as malformed astrometry.
        let owned;
        let input =
            if !path_or_content.contains('\n') && std::path::Path::new(path_or_content).is_file() {
                owned = std::fs::read_to_string(path_or_content).map_err(|e| {
                    Error::invalid_input(format!("failed to read {path_or_content}: {e}"))
                })?;
                owned.as_str()
            } else {
                path_or_content
            };
        let c_input =
            CString::new(input).map_err(|_| Error::invalid_input("input contains a NUL byte"))?;
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

    /// Run the full orbit-determination pipeline (IOD → differential
    /// correction) over **every object** in `observations`.
    ///
    /// The observations are grouped by ADES object identifier (permID /
    /// provID / trkSub) and each group is fitted independently, so one
    /// call determines a whole batch. The returned
    /// [`DetermineResults`] holds one entry per object, in `object_id`
    /// order, each carrying either the fit or a typed failure — one
    /// object failing never removes the others.
    ///
    /// Fitting a single object is the one-entry case, not a separate
    /// call: use [`DetermineResults::into_single`] to unwrap it, which
    /// refuses (loudly) if the batch turned out to hold more than one.
    ///
    /// Pass `None` for `initial_orbits` to use the internal IOD, or pass
    /// seed orbits to skip IOD and start the differential correction
    /// from the provided states. Seeds that match no observation group
    /// are reported in [`DetermineResults::unmatched_orbit_ids`].
    ///
    /// # Errors
    ///
    /// Returns `Err` only for a batch-level failure — malformed input, or
    /// a configuration the engine rejected before fitting anything. A
    /// batch in which *every* object failed still returns `Ok`: the
    /// per-object failures are the diagnosis, and
    /// [`DetermineResults::all_failed`] reports that state.
    pub fn determine(
        &self,
        observations: &Observations,
        initial_orbits: Option<&[Orbit]>,
        config: &ODConfig,
    ) -> Result<DetermineResults> {
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

        let mut results = empyrean_sys::EmpyreanDetermineResults::default();
        let (ffi_config, _perturbers_keep) = config.to_ffi_with()?;
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
                &mut results,
            )
        };
        // NONE_DELIVERED still populates (and hands us ownership of) the
        // table; every other nonzero code writes nothing.
        if code != 0 && code != empyrean_sys::EMPYREAN_DETERMINE_NONE_DELIVERED {
            return Err(Error::capture(code));
        }
        let batch = ffi_determine_results_to_rust(&results);
        unsafe { empyrean_sys::empyrean_determine_results_free(&mut results) };
        batch
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
        let (ffi_config, _perturbers_keep) = config.to_ffi_with()?;
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
        let (ffi_config, _perturbers_keep) = config.to_ffi_with()?;
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

/// Marshal the C-ABI batch table into the owned Rust one.
///
/// Reads every slot — delivered and failed — so no object is dropped on
/// the way across.
fn ffi_determine_results_to_rust(
    results: &empyrean_sys::EmpyreanDetermineResults,
) -> Result<DetermineResults> {
    let slots: &[empyrean_sys::EmpyreanODObjectResult] =
        if results.objects.is_null() || results.num_objects == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(results.objects, results.num_objects) }
        };

    let mut entries = Vec::with_capacity(slots.len());
    for slot in slots {
        let object_id = ffi_string(slot.object_id);
        let outcome = if slot.delivered != 0 {
            Ok(ffi_od_result_to_rust(&slot.result)?)
        } else {
            Err(DetermineFailure {
                object_id: object_id.clone(),
                message: ffi_string(slot.error),
                kind: DetermineFailureKind::from_code(slot.error_code),
            })
        };
        entries.push(DetermineEntry { object_id, outcome });
    }

    let unmatched = if results.unmatched_orbit_ids.is_null() || results.num_unmatched_orbit_ids == 0
    {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(results.unmatched_orbit_ids, results.num_unmatched_orbit_ids)
        }
        .iter()
        .map(|p| ffi_string(*p))
        .collect()
    };

    Ok(DetermineResults::new(entries, unmatched))
}

/// An owned `String` from a C-ABI string pointer; empty for null.
fn ffi_string(p: *mut std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
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
    // Both per-segment arrays are DECLARED-indexed and NaN-filled at a
    // segment the fit did not solve, so `None` is read off the NaN
    // rather than inferred from a count. Pairing a solved-order Δv with
    // a declared-order covariance would attribute a burn's correction to
    // another burn's uncertainty.
    let n_declared = result.n_thrust_segments as usize;
    let thrust_delta_m_per_s: Vec<Option<[f64; 3]>> = result.thrust_delta_m_per_s[..n_declared]
        .iter()
        .map(|dv| dv.iter().all(|v| v.is_finite()).then_some(*dv))
        .collect();
    let thrust_correction_covariances: Vec<Option<[[f64; 3]; 3]>> = result
        .thrust_correction_covariances[..n_declared]
        .iter()
        .map(|m| m.iter().flatten().all(|v| v.is_finite()).then_some(*m))
        .collect();
    // dv_frame is only meaningful when a thrust segment was solved.
    let dv_frame = (n_declared > 0)
        .then(|| crate::coordinate::int_to_frame(result.dv_frame).ok())
        .flatten();
    let dispositions = SolveFor::from_ffi(&result.dispositions)?;
    let warnings: Vec<String> = if result.warnings.is_null() || result.num_warnings == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(result.warnings, result.num_warnings) }
            .iter()
            .filter(|p| !p.is_null())
            .map(|p| {
                unsafe { std::ffi::CStr::from_ptr(*p) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    };
    let photometry =
        (result.has_photometry != 0).then(|| PhotometryResult::from_ffi(&result.photometry));
    let covariance_trust = CovarianceTrust::from_ffi(result);
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
        covariance_trust,
        thrust_correction_covariances,
        dispositions,
        warnings,
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
    // ── The two-convention seam, refused rather than passed through ──
    //
    // The fitted STATE is always Cartesian here — the C layer refuses to
    // flatten anything else. The fitted COVARIANCE is in the fit's
    // `covariance_representation`, and the session path can deliver
    // Keplerian / Cometary / Spherical. Attaching one to the other would
    // produce a `CoordinateState` labelled Cartesian carrying a
    // covariance that is not: not a unit mismatch but a mislabeled
    // matrix, which a re-feed would then propagate as if it were
    // Cartesian.
    //
    // The engine's angular convention differs too — a non-Cartesian
    // fitted covariance comes back in RADIANS, while every covariance
    // this crate accepts on input is in the coordinate's own units
    // (degrees for angular rows). Converting here would need the
    // representation Jacobian, which this layer does not have and must
    // not approximate.
    //
    // So the re-feedable orbit is refused with the remedy named, rather
    // than handed back wrong.
    if s.has_covariance != 0 {
        let rep = result.covariance_representation;
        if rep != empyrean_sys::EMPYREAN_REPRESENTATION_CARTESIAN as i32 {
            return Err(Error::invalid_input(format!(
                "this fit reports its covariance in representation {rep}, not Cartesian,                  while its state is Cartesian — the two cannot be assembled into one                  re-feedable orbit without the representation Jacobian, and the                  non-Cartesian covariance is additionally in radians rather than the                  degrees this crate's inputs use. Re-run the fit with a Cartesian                  output representation, or read `covariance` and                  `covariance_representation` off the result directly and transform                  them yourself."
            )));
        }
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
    // The DT posterior. It had no wire before ABI v4, so a solved-DT fit
    // used to round-trip with its DT column closed.
    if result.has_non_grav != 0 && result.non_grav.has_dt_variance != 0 {
        orbit = orbit.with_non_grav_dt_variance(Some(result.non_grav.dt_variance));
    }
    // Carry the fitted/absolute SRP slot so a solved AMRAT orbit re-feeds into
    // propagate / refine without silently dropping its SRP force. When the
    // AMRAT was solved, `amrat_variance` is the fitted posterior — chaining the
    // correct prior into a follow-on StateAndAMRAT refine.
    if result.has_srp != 0 {
        let srp = &result.srp;
        orbit = orbit.with_srp(srp.amrat, srp.cr);
        if srp.has_amrat_variance != 0 {
            orbit = orbit.with_srp_amrat_variance(Some(srp.amrat_variance));
        }
    }
    // The cross terms — the half of the posterior that is not a diagonal
    // block. Both halves land where a re-feed reads them: the border on
    // the coordinate beside its 6×6, the carrier on the orbit. Without
    // this the returned orbit carries a block-diagonal covariance, which
    // is a tighter claim than the fit made.
    let joint = unsafe { crate::JointCovariance::from_ffi(&result.orbit.orbit_cov) }?;
    orbit.state.non_grav_cross = joint.non_grav_cross;
    orbit.wide_cross = joint.wide_cross;
    Ok(orbit)
}
