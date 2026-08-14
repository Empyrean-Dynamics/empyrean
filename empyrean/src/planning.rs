//! Observation planning — how much would a candidate observation tighten
//! the orbit?
//!
//! [`Context::evaluate_plan`] takes a barycentric orbit that already
//! carries a 6×6 Cartesian covariance (the information prior) plus a
//! list of candidate observations, and reports what each one buys: the
//! prior and posterior covariance metrics, and a per-candidate marginal
//! information gain.
//!
//! Candidates are optical (RA/Dec astrometry from a registered
//! observatory) or radar. Radar folds a predicted delay (range) and/or
//! Doppler (range-rate) measurement — the line-of-sight information that
//! angles-only optical cannot supply. Its measurement σ is the
//! Cramér-Rao bound set by the waveform bandwidth and the effective SNR:
//! either supply the SNR directly ([`RadarPlanSpec::given`]) or let the
//! engine derive it from a link budget over the target's physical
//! properties ([`RadarPlanSpec::link_budget`]). The link-budget path
//! never silently defaults a missing property — whatever it had to
//! assume comes back on
//! [`CandidateKind::Radar::provenance`](CandidateKind::Radar).
//!
//! # Example: is one more night worth it?
//!
//! ```no_run
//! use empyrean::{
//!     Context, Epoch, Frame, Origin, PlannedObservation, PlanningConfig, Representation,
//! };
//!
//! let ctx = Context::from_data_dir(None)?;
//! let batch = empyrean::query_sbdb(&["99942"], None)?;
//!
//! // The planner evaluates in a barycentric basis. An origin shift is a
//! // pure translation, so the covariance comes across unchanged.
//! let mut orbit = batch.orbits[0].clone();
//! orbit.state = ctx.transform_coordinates_single(
//!     &orbit.state,
//!     Representation::Cartesian,
//!     Frame::EclipticJ2000,
//!     Origin::SSB,
//! )?;
//! let orbit = &orbit;
//!
//! // Two nights at Pan-STARRS 1 (F51), 0.2" per axis.
//! let t0 = orbit.state.epoch.mjd_tdb()?;
//! let planned = vec![
//!     PlannedObservation::optical("F51", [0.2, 0.2], Epoch::from_mjd_tdb(t0 + 30.0)),
//!     PlannedObservation::optical("F51", [0.2, 0.2], Epoch::from_mjd_tdb(t0 + 31.0)),
//! ];
//!
//! let plan = ctx.evaluate_plan(orbit, None, &planned, &PlanningConfig::default())?;
//! println!(
//!     "position σ {:.1} km → {:.1} km",
//!     plan.prior.position_sigma_km, plan.posterior.position_sigma_km,
//! );
//! for c in &plan.candidates {
//!     println!(
//!         "{}: {:.1}% position improvement",
//!         c.obs_code,
//!         100.0 * c.marginal_position_improvement,
//!     );
//! }
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! # Reading the result
//!
//! - [`PlanResult::prior`] / [`PlanResult::posterior`] bracket the
//!   campaign: the covariance before any of the candidates and after all
//!   of them. A candidate reported unobservable is **still folded** —
//!   `observable` does not gate the fold; see
//!   [`PlanCandidate::observable`].
//! - [`PlanCandidate::cumulative`] is the running covariance after that
//!   candidate and every one the engine folded before it, so the
//!   sequence shows where the campaign saturates.
//! - [`PlanCandidate::marginal_volume_reduction`] and
//!   [`PlanCandidate::marginal_position_improvement`] are what that one
//!   observation adds **given the ones already folded before it**, not a
//!   standalone score. Rank on either, but see the ordering note below.
//! - [`PlanCandidate::observable`] carries an engine verdict on an
//!   optical row and is always `true` on a radar row; the filters are
//!   engine-set and not caller-configurable. Read the per-kind detail on
//!   the field before using it as a feasibility gate.
//!
//! Ordering, conditional marginals, and the deliberate subsetting are
//! documented on [`Context::evaluate_plan`], where rustdoc renders them —
//! this module is private, so this block is not part of the public docs.

use crate::context::Context;
use crate::error::{Error, Result};
use crate::orbit::Orbit;
use crate::propagate::ForceModelTier;
use crate::time::Epoch;
use std::ffi::{CStr, CString};

// ── Input types ─────────────────────────────────────────────────────

/// A radar dish the planner can schedule against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadarStation {
    /// Goldstone DSS-14, the 70 m transmit/receive antenna.
    GoldstoneDSS14,
    /// The Green Bank Telescope, receive-only (bistatic with a
    /// transmitting dish).
    GreenBank,
    /// Arecibo. Collapsed in 2020 — accepted here only so a historical
    /// plan can be described; scheduling it is rejected by the engine.
    Arecibo,
}

impl RadarStation {
    fn to_ffi(self) -> u8 {
        match self {
            RadarStation::GoldstoneDSS14 => 0,
            RadarStation::GreenBank => 1,
            RadarStation::Arecibo => 2,
        }
    }
}

/// Which radar observable(s) a candidate would measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadarMode {
    /// Round-trip delay only — a range measurement.
    Delay,
    /// Doppler shift only — a range-rate measurement.
    Doppler,
    /// Both delay and Doppler.
    Both,
}

impl RadarMode {
    fn to_ffi(self) -> u8 {
        match self {
            RadarMode::Delay => 0,
            RadarMode::Doppler => 1,
            RadarMode::Both => 2,
        }
    }

    fn from_ffi(tag: i32) -> Option<Self> {
        match tag {
            0 => Some(RadarMode::Delay),
            1 => Some(RadarMode::Doppler),
            2 => Some(RadarMode::Both),
            _ => None,
        }
    }
}

/// Target physical properties the radar link budget needs to predict an
/// SNR. Every field is optional; `None` means "not known".
///
/// The link budget refuses rather than substitutes: it will not invent a
/// value for a property it needs, and any property it *derived* from the
/// others is reported on
/// [`CandidateKind::Radar::provenance`](CandidateKind::Radar). Only
/// [`RadarPlanSpec::link_budget`] reads these; a spec built with
/// [`RadarPlanSpec::given`] carries its SNR directly.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TargetRadarProperties {
    /// Absolute magnitude \\(H\\) (mag).
    pub h_mag: Option<f64>,
    /// Visual geometric albedo \\(p_V\\).
    pub visual_albedo: Option<f64>,
    /// Radar (OC) albedo.
    pub radar_albedo: Option<f64>,
    /// Effective diameter (km).
    pub diameter_km: Option<f64>,
    /// Rotation period (hours). Caps the coherent integration time.
    pub spin_period_hours: Option<f64>,
}

/// The radar measurement a candidate would make: the link (transmit and
/// receive dishes), the waveform, and how the effective SNR is sourced.
///
/// Build with [`given`](Self::given) when the SNR is already known (from
/// a scheduling tool or a previous apparition), or
/// [`link_budget`](Self::link_budget) to have it derived from the
/// target's physical properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarPlanSpec {
    /// Transmitting dish.
    pub transmit_station: RadarStation,
    /// Receiving dish. Equal to `transmit_station` for a monostatic
    /// observation.
    pub receive_station: RadarStation,
    /// Which observable(s) to measure.
    pub mode: RadarMode,
    /// Waveform bandwidth (Hz) — sets the delay (range) σ. Must be
    /// positive for [`RadarMode::Delay`] or [`RadarMode::Both`].
    pub bandwidth_hz: f64,
    /// Doppler frequency resolution (Hz) — sets the range-rate σ. Must be
    /// positive for [`RadarMode::Doppler`] or [`RadarMode::Both`].
    pub freq_resolution_hz: f64,
    /// Effective SNR as a linear power ratio, not dB. `None` derives it
    /// from the link budget over [`target`](Self::target) and
    /// [`integration_s`](Self::integration_s).
    pub snr: Option<f64>,
    /// Target properties for the link budget. **Refused** alongside a
    /// supplied [`snr`](Self::snr) rather than ignored — the two are
    /// different requests, and the supplied-SNR path has nowhere to
    /// carry them.
    pub target: TargetRadarProperties,
    /// Coherent integration time (s) for the link budget. **Refused**
    /// alongside a supplied [`snr`](Self::snr), like
    /// [`target`](Self::target).
    pub integration_s: f64,
}

impl RadarPlanSpec {
    /// A radar candidate whose effective SNR is supplied by the caller.
    pub fn given(
        transmit_station: RadarStation,
        receive_station: RadarStation,
        mode: RadarMode,
        bandwidth_hz: f64,
        freq_resolution_hz: f64,
        snr: f64,
    ) -> Self {
        Self {
            transmit_station,
            receive_station,
            mode,
            bandwidth_hz,
            freq_resolution_hz,
            snr: Some(snr),
            target: TargetRadarProperties::default(),
            integration_s: 0.0,
        }
    }

    /// A radar candidate whose effective SNR is derived from the link
    /// budget over the target's physical properties.
    pub fn link_budget(
        transmit_station: RadarStation,
        receive_station: RadarStation,
        target: TargetRadarProperties,
        integration_s: f64,
        mode: RadarMode,
        bandwidth_hz: f64,
        freq_resolution_hz: f64,
    ) -> Self {
        Self {
            transmit_station,
            receive_station,
            mode,
            bandwidth_hz,
            freq_resolution_hz,
            snr: None,
            target,
            integration_s,
        }
    }
}

/// The measurement a [`PlannedObservation`] would make.
#[derive(Debug, Clone, PartialEq)]
pub enum PlannedObservationKind {
    /// Optical astrometry from a registered observatory.
    Optical {
        /// Registered MPC observatory code.
        optical_code: String,
        /// Assumed 1σ `(RA·cosδ, Dec)` astrometric uncertainty (arcsec).
        optical_sigma_arcsec: [f64; 2],
    },
    /// Radar delay and/or Doppler.
    Radar(Box<RadarPlanSpec>),
}

/// One candidate observation: when it would be taken and what it would
/// measure.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedObservation {
    /// Planned epoch — the receive epoch for a radar candidate.
    pub epoch: Epoch,
    /// Optical or radar, with that kind's parameters.
    pub kind: PlannedObservationKind,
}

impl PlannedObservation {
    /// An optical candidate at a registered observatory, with the
    /// assumed 1σ `(RA·cosδ, Dec)` astrometric uncertainty in arcsec.
    pub fn optical(
        optical_code: impl Into<String>,
        optical_sigma_arcsec: [f64; 2],
        epoch: Epoch,
    ) -> Self {
        Self {
            epoch,
            kind: PlannedObservationKind::Optical {
                optical_code: optical_code.into(),
                optical_sigma_arcsec,
            },
        }
    }

    /// A radar candidate at the given receive epoch.
    pub fn radar(spec: RadarPlanSpec, epoch: Epoch) -> Self {
        Self {
            epoch,
            kind: PlannedObservationKind::Radar(Box::new(spec)),
        }
    }
}

/// Per-site astrometric assumptions and observability filters.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservatoryConfig {
    /// MPC observatory code.
    pub obs_code: String,
    /// Assumed 1σ `(RA·cosδ, Dec)` uncertainty (arcsec).
    pub sigma_arcsec: [f64; 2],
    /// Limiting apparent magnitude.
    pub max_apparent_mag: f64,
    /// Minimum solar elongation (degrees).
    pub min_elongation_deg: f64,
    /// Minimum geometric elevation of the target above the site's local
    /// horizon (degrees), ignoring atmospheric refraction.
    ///
    /// `0.0` is the geometric horizon and the engine's default — the
    /// least-opinionated statement the geometry can make, not an
    /// observing recommendation: airmass there is ≈ 38, and real
    /// programs cut between 20° and 30°.
    pub min_elevation_deg: f64,
    /// Solar altitude at or below which the site counts as dark
    /// (degrees). `None` takes the engine's default of −18°,
    /// astronomical twilight.
    ///
    /// An `Option` rather than a bare `f64` because `0.0` is a legal
    /// solar altitude — the Sun's centre on the geometric horizon — so a
    /// defaulted zero would quietly plan a campaign in daylight. The
    /// other conventions are civil (−6°) and nautical (−12°); above
    /// +90° disables the gate.
    pub max_sun_altitude_deg: Option<f64>,
}

/// Configuration for [`Context::evaluate_plan`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlanningConfig {
    /// Force-model tier for the planning propagation.
    pub force_model: ForceModelTier,
    /// Adaptive integrator truncation-error tolerance.
    pub epsilon: f64,
    /// Per-site astrometric assumptions.
    ///
    /// **Not consulted by** [`Context::evaluate_plan`], which reads each
    /// optical candidate's σ from that candidate's own
    /// [`PlannedObservation`] and applies engine-set observability
    /// filters that no field here reaches. A non-empty list is rejected
    /// rather than accepted and ignored. The field exists because it is
    /// part of the shared planning configuration; it becomes live if a
    /// surface that reads it is exposed.
    pub observatories: Vec<ObservatoryConfig>,
    /// Worker threads. `None` uses every available core.
    ///
    /// **Not consulted by** [`Context::evaluate_plan`], which evaluates
    /// one orbit and does not shard the work. Any value other than
    /// `None` is rejected rather than accepted and ignored.
    pub num_threads: Option<usize>,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            force_model: ForceModelTier::Standard,
            epsilon: 1e-9,
            observatories: Vec::new(),
            num_threads: None,
        }
    }
}

// ── Output types ────────────────────────────────────────────────────

/// Summary metrics for a state covariance — the prior, the posterior, or
/// a candidate's running total.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceMetrics {
    /// RSS position 1σ (km).
    pub position_sigma_km: f64,
    /// RSS velocity 1σ (m/s) at the orbit epoch.
    pub velocity_sigma_m_s: f64,
    /// Semi-major axis of the 1σ position ellipsoid (km).
    pub semi_major_km: f64,
    /// Semi-minor axis of the 1σ position ellipsoid (km).
    pub semi_minor_km: f64,
    /// \\(\\ln \\det \\Sigma\\) over the 6×6 state covariance — the
    /// D-optimality criterion.
    ///
    /// **In AU and AU·day⁻¹**, unlike the four fields above, which are
    /// rescaled to km and m/s. A log-determinant is dimensional, so the
    /// absolute value depends on that choice: the same covariance
    /// expressed in km and m/s gives a value larger by
    /// \\(6\\ln(\\mathrm{km/AU}) + 6\\ln(\\mathrm{(m/s)/(AU/day)}) \\approx
    /// 199.13\\). Differences between two `log_det` values on this
    /// surface are unit-invariant and can be compared directly.
    pub log_det: f64,
}

/// One predicted sky position from the plan's optical ephemeris.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanEphemerisPoint {
    /// Epoch of the prediction.
    pub epoch: Epoch,
    /// Predicted topocentric right ascension (degrees, ICRF).
    pub ra_deg: f64,
    /// Predicted topocentric declination (degrees, ICRF).
    pub dec_deg: f64,
}

/// What a candidate turned out to be, with the fields only that kind
/// reports.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateKind {
    /// Optical astrometry. The sky-plane geometry on
    /// [`PlanCandidate`] applies.
    Optical,
    /// Radar delay and/or Doppler. The sky-plane geometry on
    /// [`PlanCandidate`] does not apply and is reported as `None`.
    Radar {
        /// Observable(s) measured.
        mode: RadarMode,
        /// Effective SNR the measurement σ was derived from (linear
        /// power ratio, not dB) — either the supplied value or the one
        /// the link budget produced.
        snr: f64,
        /// One-way topocentric range to the target at the receive epoch
        /// (km), from the predicted round-trip delay.
        range_km: f64,
        /// Assumptions the link budget had to make to reach the SNR —
        /// for example a diameter derived from \\(H\\) and \\(p_V\\), or
        /// coherent integration left uncapped because the spin period
        /// is unknown, or an integration capped by a known spin period.
        /// Empty only when the SNR was supplied, or when every input was
        /// given *and* the requested integration already fit inside the
        /// speckle-decorrelation time.
        provenance: Vec<String>,
    },
}

/// What one candidate observation would buy.
///
/// The sky-plane geometry fields are `None` on a radar candidate: radar
/// measures line-of-sight range and range-rate, not a sky-plane
/// position, so there is no on-sky ellipse to report and a numeric zero
/// there would read as a measured zero.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanCandidate {
    /// This candidate's row in [`PlanResult::ephemeris`] for an optical
    /// candidate; its rank among the radar candidates, ordered by epoch,
    /// for a radar one.
    ///
    /// A radar row carries no epoch of its own, so this rank is the only
    /// key back to the input: sort the radar candidates you submitted by
    /// epoch and the *n*-th is the row with `index == n`.
    pub index: usize,
    /// Observatory code (optical) or receive-station code (radar).
    pub obs_code: String,
    /// Optical or radar, with that kind's reported fields.
    pub kind: CandidateKind,
    /// Whether the candidate passes its observability filters — with a
    /// different meaning per kind, so branch on
    /// [`kind`](Self::kind) before using it as a gate.
    ///
    /// On an **optical** row this is a real engine verdict, and today it
    /// is a solar-elongation test and nothing else: the limiting
    /// magnitude the engine would also apply cannot fire, because the
    /// target's absolute magnitude does not reach the planner. On a
    /// **radar** row it is always `true` — no radar feasibility test
    /// runs on this entry point, so `true` here means "not assessed",
    /// not "checked and cleared". In particular no antenna-elevation or
    /// horizon test is applied, so a track below the horizon still
    /// reports `true`.
    ///
    /// The filters are engine-set and not caller-configurable. An
    /// unobservable candidate is reported rather than dropped, **and is
    /// still folded** into [`cumulative`](Self::cumulative) and
    /// [`PlanResult::posterior`]: this flag does not gate the fold. Only
    /// a candidate whose observation partials the engine could not
    /// compute leaves the covariance untouched, and that shows as a
    /// [`marginal_volume_reduction`](Self::marginal_volume_reduction) of
    /// exactly 1.
    pub observable: bool,
    /// Prior along-track 1σ on the sky plane (arcsec), in the frame of
    /// the predicted sky motion. Optical only.
    ///
    /// "Prior" is literal: the campaign prior mapped to this candidate's
    /// epoch with **no** candidate folded — not even this one. Its
    /// partner [`post_along_track_sigma_arcsec`](Self::post_along_track_sigma_arcsec)
    /// is cumulative, so the pair is not a single-observation bracket.
    pub along_track_sigma_arcsec: Option<f64>,
    /// Prior cross-track 1σ on the sky plane (arcsec). Same "no
    /// candidate folded" basis as
    /// [`along_track_sigma_arcsec`](Self::along_track_sigma_arcsec).
    /// Optical only.
    ///
    /// Along- and cross-track are a projection onto the sky-motion
    /// frame, not the principal axes of the sky covariance, so
    /// cross-track may legitimately exceed along-track.
    pub cross_track_sigma_arcsec: Option<f64>,
    /// Prior RA·cosδ 1σ (arcsec), no candidate folded. Optical only.
    pub ra_sigma_arcsec: Option<f64>,
    /// Prior Dec 1σ (arcsec), no candidate folded. Optical only.
    pub dec_sigma_arcsec: Option<f64>,
    /// Position angle of the predicted **sky motion** (degrees, east of
    /// north) — the axis the along/cross-track σ above are projected
    /// onto. Optical only.
    ///
    /// This is kinematic and does not depend on the covariance: it is
    /// not the orientation of the sky-plane uncertainty ellipse. The
    /// range is \\((-180, 180]\\); add 360 **to negative values**
    /// (equivalently `pa.rem_euclid(360.0)`) for the conventional
    /// \\([0, 360)\\) position-angle convention.
    pub position_angle_deg: Option<f64>,
    /// Per-dimension generalized-variance ratio from this one
    /// observation, \\((\\det \\Sigma_\\text{post} / \\det
    /// \\Sigma_\\text{prior})^{1/6}\\) over the 6×6 state covariance
    /// (≤ 1) — a D-optimality score normalized to one dimension, so it
    /// reads as a linear scale factor and is comparable across plans.
    ///
    /// The 1σ ellipsoid *volume* ratio is this value **cubed**, and the
    /// raw determinant ratio is it to the **sixth** power. The value is
    /// conditional on the candidates folded before this one — see
    /// "Ordering, and what the marginals are conditional on" on
    /// [`Context::evaluate_plan`].
    pub marginal_volume_reduction: f64,
    /// Fractional position-σ improvement from this one observation,
    /// in \\([0, 1]\\). Conditional on the candidates folded before it,
    /// like [`marginal_volume_reduction`](Self::marginal_volume_reduction).
    pub marginal_position_improvement: f64,
    /// Along-track 1σ after folding this observation **and every one
    /// folded before it** (arcsec). Optical only.
    ///
    /// Cumulative, on the same basis as [`cumulative`](Self::cumulative)
    /// — not the far end of a single-observation bracket against
    /// [`along_track_sigma_arcsec`](Self::along_track_sigma_arcsec),
    /// which folds nothing.
    pub post_along_track_sigma_arcsec: Option<f64>,
    /// Cross-track 1σ after folding this observation and every one
    /// folded before it (arcsec). Cumulative, like
    /// [`post_along_track_sigma_arcsec`](Self::post_along_track_sigma_arcsec).
    /// Optical only.
    pub post_cross_track_sigma_arcsec: Option<f64>,
    /// Covariance metrics after folding this observation and every one
    /// folded before it — including any reported unobservable, since the
    /// fold does not consult [`observable`](Self::observable).
    pub cumulative: CovarianceMetrics,
    /// Width of the solve-for set this candidate folded into. Always 6
    /// (state-only) on this entry point — the non-gravitational solve is
    /// not exposed here; see "What this entry point does not expose" on
    /// [`Context::evaluate_plan`].
    pub active_width: usize,
}

/// The result of evaluating an observation plan.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanResult {
    /// Orbit identifier the plan was evaluated for.
    pub orbit_id: String,
    /// Covariance metrics before any of the candidates.
    pub prior: CovarianceMetrics,
    /// Covariance metrics after **every** candidate submitted.
    ///
    /// Not the observable subset: the fold does not consult
    /// [`PlanCandidate::observable`], so a candidate reported
    /// unobservable still contributes here. To price the observable
    /// subset, drop those candidates from `planned` and evaluate again.
    ///
    /// What *does* gate the fold is whether the engine could compute the
    /// candidate's observation partials. A candidate for which it could
    /// not leaves the covariance untouched, and its signature is a
    /// [`PlanCandidate::marginal_volume_reduction`] of exactly 1 with
    /// [`PlanCandidate::cumulative`] unchanged from the previous row.
    pub posterior: CovarianceMetrics,
    /// Per-candidate analysis, in ascending epoch order rather than the
    /// order the candidates were supplied in.
    pub candidates: Vec<PlanCandidate>,
    /// Width of the solve-for set. Always 6 (state-only) on this entry
    /// point; see "What this entry point does not expose" on
    /// [`Context::evaluate_plan`].
    pub active_width: usize,
    /// Predicted sky position for each optical candidate, in
    /// chronological order. An optical candidate's
    /// [`PlanCandidate::index`] is its row here. Empty for a radar-only
    /// plan.
    pub ephemeris: Vec<PlanEphemerisPoint>,
}

// ── FFI marshaling ──────────────────────────────────────────────────

unsafe fn cstr_to_string(p: *mut std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

/// `Option` → the NaN sentinel the C ABI reads for "absent".
fn nan_from_opt(v: Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

/// Sky-plane geometry is meaningful only for an optical candidate. A
/// radar row carries a structural placeholder there; surfacing it as a
/// number would read as a measured zero.
///
/// The candidate kind alone decides applicability. A non-finite value on
/// a row where the geometry *does* apply is passed through as
/// `Some(NaN)` rather than folded into the same `None` that means "not
/// applicable", so a Rust caller can tell the two apart. The Python
/// surface collapses them into one arrow null by its own convention that
/// a nullable float column carries nulls rather than NaN; it documents
/// the collapse and disambiguates on the `kind` column.
fn optical_only(is_optical: bool, v: f64) -> Option<f64> {
    if is_optical { Some(v) } else { None }
}

/// Reject a non-finite value a caller supplied for a slot where the C ABI
/// reads NaN as a *discriminant* rather than as a number.
///
/// `None` is the only spelling of "absent" / "derive it" on this surface.
/// Letting `Some(NaN)` through would silently re-interpret a caller-side
/// arithmetic bug as a different request.
fn reject_non_finite(value: Option<f64>, what: &str, index: usize) -> Result<()> {
    if let Some(v) = value
        && !v.is_finite()
    {
        return Err(Error::invalid_input(format!(
            "planned observation {index}: {what} must be finite, got {v}; pass None to leave it \
             absent"
        )));
    }
    Ok(())
}

fn metrics_from_ffi(m: empyrean_sys::EmpyreanCovarianceMetrics) -> CovarianceMetrics {
    let empyrean_sys::EmpyreanCovarianceMetrics {
        position_sigma_km,
        velocity_sigma_m_s,
        semi_major_km,
        semi_minor_km,
        log_det,
    } = m;
    CovarianceMetrics {
        position_sigma_km,
        velocity_sigma_m_s,
        semi_major_km,
        semi_minor_km,
        log_det,
    }
}

/// Keepalive for the `CString`s the planned-observation and observatory
/// FFI structs borrow as `*const c_char`. Must outlive the FFI call.
struct PlanFfiKeep {
    optical_codes: Vec<CString>,
    obs_codes: Vec<CString>,
    orbit_id: Option<CString>,
}

/// Refuse the [`PlanningConfig`] fields the plan-evaluation path does
/// not read.
///
/// A knob that accepts a value and silently does nothing is the failure
/// mode this crate refuses everywhere else, and both of these would
/// otherwise be quietly dropped somewhere below the FFI boundary. Delete
/// the corresponding arm here — and the field's doc caveat — if a
/// release ever wires one of them through.
fn reject_unread_config_knobs(config: &PlanningConfig) -> Result<()> {
    if !config.observatories.is_empty() {
        return Err(Error::invalid_input(format!(
            "PlanningConfig::observatories is not consulted by evaluate_plan (got {} \
             entries); each optical candidate's σ comes from its own PlannedObservation. \
             Observability filters are engine-set on this entry point and are not \
             caller-configurable.",
            config.observatories.len()
        )));
    }
    if let Some(n) = config.num_threads {
        return Err(Error::invalid_input(format!(
            "PlanningConfig::num_threads is not consulted by evaluate_plan (got {n}); it \
             evaluates a single orbit and does not shard the work. Leave it as None."
        )));
    }
    Ok(())
}

/// Refuse an orbit whose covariance the planner cannot consume as
/// written.
///
/// Two hard requirements, both silent failures if unchecked. The prior is
/// inverted directly into the Fisher accumulator as a **Cartesian** state
/// covariance — the engine tags a covariance by representation but never
/// converts it, so cometary or Keplerian elements would be reinterpreted
/// rather than rejected. And every sensitivity chain is built in the
/// **barycentric** basis; the engine's own origin check lives on the
/// optical chain, so a radar-only plan would slip past it entirely.
fn reject_unusable_orbit_basis(orbit: &Orbit) -> Result<()> {
    if orbit.state.representation != crate::Representation::Cartesian {
        return Err(Error::invalid_input(format!(
            "evaluate_plan requires a Cartesian orbit, got {:?}. The covariance is consumed as a \
             6×6 Cartesian state prior and is never converted, so elements in any other \
             representation would be reinterpreted rather than rejected. Convert with \
             Context::transform_coordinates_single(.., Representation::Cartesian, .., \
             Origin::SSB).",
            orbit.state.representation
        )));
    }
    if orbit.state.origin != crate::Origin::SSB {
        return Err(Error::invalid_input(format!(
            "evaluate_plan requires an orbit with its origin at the Solar System barycenter, got \
             {:?}. Convert with Context::transform_coordinates_single(.., \
             Representation::Cartesian, .., Origin::SSB) — an origin shift is a pure translation, \
             so the covariance and its metrics are unchanged.",
            orbit.state.origin
        )));
    }
    Ok(())
}

/// Pick the label a plan result carries: the caller's argument, else the
/// orbit's own identifier, else `None` so the engine assigns one.
///
/// Extracted so the fallback is testable without a [`Context`] — an
/// empty string is treated as absent at both levels, matching the C
/// ABI's "id absent" sentinel.
fn resolve_orbit_id<'a>(arg: Option<&'a str>, orbit_id: Option<&'a str>) -> Option<&'a str> {
    [arg, orbit_id]
        .into_iter()
        .flatten()
        .find(|id| !id.is_empty())
}

/// Refuse link-budget inputs supplied alongside an explicit SNR.
///
/// The engine's radar spec is a sum type: its supplied-SNR arm carries
/// neither the target properties nor the integration time, so the C ABI
/// drops all six on the floor. Refusing here keeps the Rust channel from
/// silently accepting a program Python rejects.
fn reject_link_budget_inputs_with_given_snr(spec: &RadarPlanSpec, index: usize) -> Result<()> {
    if spec.snr.is_none() {
        return Ok(());
    }
    let mut unused: Vec<String> = Vec::new();
    for (name, value) in [
        ("h_mag", spec.target.h_mag),
        ("visual_albedo", spec.target.visual_albedo),
        ("radar_albedo", spec.target.radar_albedo),
        ("diameter_km", spec.target.diameter_km),
        ("spin_period_hours", spec.target.spin_period_hours),
    ] {
        if let Some(v) = value {
            unused.push(format!("target.{name}={v}"));
        }
    }
    if spec.integration_s != 0.0 {
        unused.push(format!("integration_s={}", spec.integration_s));
    }
    if unused.is_empty() {
        return Ok(());
    }
    Err(Error::invalid_input(format!(
        "planned observation {index}: snr is supplied, so the link budget never runs and {} would \
         be dropped. Set snr to None to derive the SNR from those properties, or remove them.",
        unused.join(", ")
    )))
}

fn cstring_for(value: &str, what: &str) -> Result<CString> {
    CString::new(value).map_err(|_| {
        Error::invalid_input(format!("{what} contains an interior nul byte: {value:?}"))
    })
}

impl Context {
    /// Evaluate an observation plan: how much would each candidate
    /// observation tighten this orbit's covariance?
    ///
    /// `orbit` must carry a 6×6 Cartesian covariance and be referenced
    /// to the Solar System barycenter; the frame is free. A
    /// heliocentric fit converts with
    /// [`Context::transform_coordinates_single`] — an origin shift is a
    /// pure translation, so the covariance and every metric below it are
    /// unchanged. `orbit_id` labels the result; `None` uses the orbit's
    /// own [`Orbit::orbit_id`], or an engine-assigned label if it has
    /// none. `planned` is the candidate list, and must not be empty.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] when [`PlanningConfig::observatories`] or
    /// [`PlanningConfig::num_threads`] is set (neither is consulted on
    /// this entry point), when the orbit has no covariance, a singular
    /// one, or an origin other than the barycenter, when a candidate is
    /// invalid or infeasible (an
    /// unregistered observatory code, a defunct radar dish, a
    /// non-positive bandwidth for a delay measurement, a link budget
    /// missing a property it needs), or when the planning propagation
    /// fails.
    ///
    /// # Ordering, and what the marginals are conditional on
    ///
    /// The engine evaluates candidates in ascending epoch order
    /// regardless of the order they were supplied in, and folds each one
    /// into the covariance that already contains every earlier
    /// candidate. Two consequences:
    ///
    /// - A candidate's marginal gain depends on its position in that
    ///   sequence. Two identical observations do not score identically —
    ///   the later one is measured against a tighter covariance and
    ///   scores smaller. Ranking compares conditional contributions
    ///   **within one campaign**; to compare candidates head to head,
    ///   evaluate a separate one-candidate plan for each.
    /// - A row does not carry its input epoch. Join an optical candidate
    ///   to its predicted sky position — and hence its epoch — through
    ///   [`PlanCandidate::index`], its row in [`PlanResult::ephemeris`].
    ///
    /// # What this entry point does not expose
    ///
    /// Deliberate subsetting, recorded so the omissions are not mistaken
    /// for oversights. The engine also offers a non-gravitational
    /// planning variant that solves over state ⊕ (A1, A2, A3) and
    /// reports the σ(A2) tightening a radar campaign buys, a visibility
    /// survey over a time window, batch evaluation across many orbits,
    /// and an encounter-B-plane characterization. None of them is
    /// reachable from this crate.
    ///
    /// An orbit carrying non-gravitational parameters — a Yarkovsky fit,
    /// say — is accepted here and evaluated **state-only**. The
    /// non-gravitational acceleration still acts in the dynamics, so the
    /// predicted trajectory and sky positions account for it, but the
    /// solve-for set stays 6×6 ([`PlanResult::active_width`] reports
    /// `6`), the A1/A2/A3 columns are not folded, and no σ(A2) is
    /// reported. The plan prices what the observations buy for the
    /// *state* under that force model, not what they buy for the
    /// non-gravitational parameters themselves.
    ///
    /// # On the name
    ///
    /// This method is the **single-object** form, while `evaluate_plan`
    /// is the batch name one layer down in the engine. That is
    /// deliberate: the C symbol `empyrean_evaluate_plan` has carried
    /// single-object semantics since v0.7.0 and is frozen, and this
    /// crate mirrors the C ABI it wraps. If a batch form is exposed
    /// later it follows the migration precedent
    /// [`Context::transform_coordinates`] set — the batch takes the
    /// plain name and the single-object form gains a `_single` suffix —
    /// rather than renaming this method out from under callers now.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use empyrean::{
    ///     Context, Epoch, Frame, Origin, PlannedObservation, PlanningConfig, RadarMode,
    ///     RadarPlanSpec, RadarStation, Representation,
    /// };
    ///
    /// let ctx = Context::from_data_dir(None)?;
    /// let mut orbit = empyrean::query_sbdb(&["99942"], None)?.orbits[0].clone();
    /// orbit.state = ctx.transform_coordinates_single(
    ///     &orbit.state,
    ///     Representation::Cartesian,
    ///     Frame::EclipticJ2000,
    ///     Origin::SSB,
    /// )?;
    /// let orbit = &orbit;
    /// let t0 = orbit.state.epoch.mjd_tdb()?;
    ///
    /// // One optical night, then a Goldstone delay+Doppler run at a
    /// // known SNR.
    /// let planned = vec![
    ///     PlannedObservation::optical("F51", [0.2, 0.2], Epoch::from_mjd_tdb(t0 + 30.0)),
    ///     PlannedObservation::radar(
    ///         RadarPlanSpec::given(
    ///             RadarStation::GoldstoneDSS14,
    ///             RadarStation::GoldstoneDSS14,
    ///             RadarMode::Both,
    ///             1.0e5,
    ///             0.1,
    ///             50.0,
    ///         ),
    ///         Epoch::from_mjd_tdb(t0 + 45.0),
    ///     ),
    /// ];
    ///
    /// let plan = ctx.evaluate_plan(orbit, Some("apophis"), &planned, &PlanningConfig::default())?;
    /// assert!(plan.posterior.position_sigma_km <= plan.prior.position_sigma_km);
    /// # Ok::<(), empyrean::Error>(())
    /// ```
    pub fn evaluate_plan(
        &self,
        orbit: &Orbit,
        orbit_id: Option<&str>,
        planned: &[PlannedObservation],
        config: &PlanningConfig,
    ) -> Result<PlanResult> {
        reject_unread_config_knobs(config)?;
        reject_unusable_orbit_basis(orbit)?;

        let (ffi_orbit, _orbit_keep) = orbit.to_ffi_with_keep()?;

        // Every borrowed string must outlive the call; build them all
        // before any pointer is captured.
        let mut optical_codes: Vec<CString> = Vec::with_capacity(planned.len());
        for p in planned {
            let cs = match &p.kind {
                PlannedObservationKind::Optical { optical_code, .. } => {
                    cstring_for(optical_code, "optical station code")?
                }
                PlannedObservationKind::Radar(_) => CString::default(),
            };
            optical_codes.push(cs);
        }
        let mut obs_codes: Vec<CString> = Vec::with_capacity(config.observatories.len());
        for o in &config.observatories {
            obs_codes.push(cstring_for(&o.obs_code, "observatory code")?);
        }
        let orbit_id_cstr = match resolve_orbit_id(orbit_id, orbit.orbit_id.as_deref()) {
            Some(id) => Some(cstring_for(id, "orbit_id")?),
            None => None,
        };
        let keep = PlanFfiKeep {
            optical_codes,
            obs_codes,
            orbit_id: orbit_id_cstr,
        };

        let ffi_planned: Vec<empyrean_sys::EmpyreanPlannedObservation> = planned
            .iter()
            .zip(keep.optical_codes.iter())
            .enumerate()
            .map(|(i, (p, code))| planned_to_ffi(p, code, i))
            .collect::<Result<Vec<_>>>()?;

        let ffi_observatories: Vec<empyrean_sys::EmpyreanObservatoryConfig> = config
            .observatories
            .iter()
            .zip(keep.obs_codes.iter())
            .map(|(o, code)| empyrean_sys::EmpyreanObservatoryConfig {
                obs_code: code.as_ptr(),
                sigma_ra_arcsec: o.sigma_arcsec[0],
                sigma_dec_arcsec: o.sigma_arcsec[1],
                max_apparent_mag: o.max_apparent_mag,
                min_elongation_deg: o.min_elongation_deg,
                min_elevation_deg: o.min_elevation_deg,
                has_max_sun_altitude_deg: u8::from(o.max_sun_altitude_deg.is_some()),
                max_sun_altitude_deg: o.max_sun_altitude_deg.unwrap_or(0.0),
            })
            .collect();

        let ffi_config = empyrean_sys::EmpyreanPlanningConfig {
            force_model: config.force_model as i32,
            epsilon: config.epsilon,
            observatories: if ffi_observatories.is_empty() {
                std::ptr::null()
            } else {
                ffi_observatories.as_ptr()
            },
            num_observatories: ffi_observatories.len(),
            // C ABI convention: -1 = every available core.
            num_threads: match config.num_threads {
                None => -1,
                Some(n) => i32::try_from(n).map_err(|_| {
                    Error::invalid_input(format!("num_threads {n} exceeds the C ABI's i32 range"))
                })?,
            },
        };

        let mut ffi_result = empyrean_sys::EmpyreanPlanResult::default();
        let code = unsafe {
            empyrean_sys::empyrean_evaluate_plan(
                self.as_raw(),
                &ffi_orbit,
                keep.orbit_id
                    .as_ref()
                    .map_or(std::ptr::null(), |c| c.as_ptr()),
                ffi_planned.as_ptr(),
                ffi_planned.len(),
                &ffi_config,
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }

        // Copy every field out of the C result BEFORE releasing it. The
        // copy is fallible (unknown enum tags from a newer engine), so it
        // runs in a closure whose error propagates only after the free —
        // an early `?` here would leak the result's interiors.
        let copied = plan_result_from_ffi(&ffi_result);
        unsafe { empyrean_sys::empyrean_plan_result_free(&mut ffi_result) };
        drop(keep);
        copied
    }
}

fn planned_to_ffi(
    p: &PlannedObservation,
    optical_code: &CString,
    index: usize,
) -> Result<empyrean_sys::EmpyreanPlannedObservation> {
    let epoch_mjd_tdb = p.epoch.mjd_tdb()?;
    // A non-finite epoch reaches the engine's chronological sort and
    // panics its comparator, which comes back as an opaque internal-panic
    // code. Refuse it here, where the message can name the candidate.
    if !epoch_mjd_tdb.is_finite() {
        return Err(Error::invalid_input(format!(
            "planned observation {index}: epoch must be a finite MJD TDB, got {epoch_mjd_tdb}"
        )));
    }
    if let PlannedObservationKind::Radar(spec) = &p.kind {
        // Green Bank cannot transmit. On the link-budget path its zero
        // transmit power makes the engine refuse, but a supplied SNR
        // skips the link budget entirely, so the impossible link would
        // otherwise be accepted.
        if spec.transmit_station == RadarStation::GreenBank {
            return Err(Error::invalid_input(format!(
                "planned observation {index}: Green Bank is receive-only and cannot be the \
                 transmit station; pair it with a transmitting dish for a bistatic observation"
            )));
        }
        reject_link_budget_inputs_with_given_snr(spec, index)?;
        reject_non_finite(spec.snr, "radar snr", index)?;
        reject_non_finite(spec.target.h_mag, "radar target h_mag", index)?;
        reject_non_finite(
            spec.target.visual_albedo,
            "radar target visual_albedo",
            index,
        )?;
        reject_non_finite(spec.target.radar_albedo, "radar target radar_albedo", index)?;
        reject_non_finite(spec.target.diameter_km, "radar target diameter_km", index)?;
        reject_non_finite(
            spec.target.spin_period_hours,
            "radar target spin_period_hours",
            index,
        )?;
    }
    Ok(match &p.kind {
        PlannedObservationKind::Optical {
            optical_code: _,
            optical_sigma_arcsec,
        } => empyrean_sys::EmpyreanPlannedObservation {
            epoch_mjd_tdb,
            kind: 0,
            optical_code: optical_code.as_ptr(),
            optical_sigma_ra_arcsec: optical_sigma_arcsec[0],
            optical_sigma_dec_arcsec: optical_sigma_arcsec[1],
            // Radar slot unused for an optical candidate; the C side
            // reads it only when `kind == 1`.
            radar_transmit_station: 0,
            radar_receive_station: 0,
            radar_mode: 0,
            radar_bandwidth_hz: 0.0,
            radar_freq_resolution_hz: 0.0,
            radar_snr: f64::NAN,
            radar_target_h_mag: f64::NAN,
            radar_target_visual_albedo: f64::NAN,
            radar_target_radar_albedo: f64::NAN,
            radar_target_diameter_km: f64::NAN,
            radar_target_spin_period_hours: f64::NAN,
            radar_integration_s: 0.0,
        },
        PlannedObservationKind::Radar(spec) => empyrean_sys::EmpyreanPlannedObservation {
            epoch_mjd_tdb,
            kind: 1,
            // Optical slot unused for a radar candidate; the empty
            // CString keeps the pointer non-null, as every other C-ABI
            // string slot does.
            optical_code: optical_code.as_ptr(),
            optical_sigma_ra_arcsec: 0.0,
            optical_sigma_dec_arcsec: 0.0,
            radar_transmit_station: spec.transmit_station.to_ffi(),
            radar_receive_station: spec.receive_station.to_ffi(),
            radar_mode: spec.mode.to_ffi(),
            radar_bandwidth_hz: spec.bandwidth_hz,
            radar_freq_resolution_hz: spec.freq_resolution_hz,
            // NaN is the C ABI's "derive it from the link budget"
            // discriminant, not a missing value.
            radar_snr: nan_from_opt(spec.snr),
            radar_target_h_mag: nan_from_opt(spec.target.h_mag),
            radar_target_visual_albedo: nan_from_opt(spec.target.visual_albedo),
            radar_target_radar_albedo: nan_from_opt(spec.target.radar_albedo),
            radar_target_diameter_km: nan_from_opt(spec.target.diameter_km),
            radar_target_spin_period_hours: nan_from_opt(spec.target.spin_period_hours),
            radar_integration_s: spec.integration_s,
        },
    })
}

fn plan_result_from_ffi(ffi: &empyrean_sys::EmpyreanPlanResult) -> Result<PlanResult> {
    // Destructured with no `..` rest pattern so a new C-ABI field
    // becomes a compile error here, at the marshal boundary, instead of
    // a silent per-field drop.
    let empyrean_sys::EmpyreanPlanResult {
        orbit_id,
        prior,
        posterior,
        candidates,
        num_candidates,
        active_width,
        ephemeris,
        num_ephemeris,
    } = *ffi;

    let mut out_candidates = Vec::with_capacity(num_candidates);
    if !candidates.is_null() && num_candidates > 0 {
        for i in 0..num_candidates {
            out_candidates.push(candidate_from_ffi(unsafe { &*candidates.add(i) })?);
        }
    }

    let mut out_ephemeris = Vec::with_capacity(num_ephemeris);
    if !ephemeris.is_null() && num_ephemeris > 0 {
        for i in 0..num_ephemeris {
            let empyrean_sys::EmpyreanPlanEphemerisPoint {
                epoch_mjd_tdb,
                ra_deg,
                dec_deg,
            } = unsafe { *ephemeris.add(i) };
            out_ephemeris.push(PlanEphemerisPoint {
                epoch: Epoch::from_mjd_tdb(epoch_mjd_tdb),
                ra_deg,
                dec_deg,
            });
        }
    }

    Ok(PlanResult {
        orbit_id: unsafe { cstr_to_string(orbit_id) },
        prior: metrics_from_ffi(prior),
        posterior: metrics_from_ffi(posterior),
        candidates: out_candidates,
        active_width,
        ephemeris: out_ephemeris,
    })
}

fn candidate_from_ffi(c: &empyrean_sys::EmpyreanPlanCandidate) -> Result<PlanCandidate> {
    // Same no-`..` destructuring contract as `plan_result_from_ffi`.
    let empyrean_sys::EmpyreanPlanCandidate {
        index,
        obs_code,
        kind,
        observable,
        along_track_sigma_arcsec,
        cross_track_sigma_arcsec,
        ra_sigma_arcsec,
        dec_sigma_arcsec,
        position_angle_deg,
        marginal_volume_reduction,
        marginal_position_improvement,
        post_along_track_sigma_arcsec,
        post_cross_track_sigma_arcsec,
        cumulative,
        active_width,
        radar_snr,
        radar_range_km,
        radar_provenance,
        num_radar_provenance,
        radar_mode,
    } = *c;

    let out_kind = match kind {
        0 => CandidateKind::Optical,
        1 => {
            let mode = RadarMode::from_ffi(radar_mode).ok_or_else(|| {
                Error::invalid_input(format!(
                    "C ABI returned an unknown radar mode tag for a radar candidate: {radar_mode}"
                ))
            })?;
            let mut provenance = Vec::with_capacity(num_radar_provenance);
            if !radar_provenance.is_null() {
                for j in 0..num_radar_provenance {
                    provenance.push(unsafe { cstr_to_string(*radar_provenance.add(j)) });
                }
            }
            CandidateKind::Radar {
                mode,
                snr: radar_snr,
                range_km: radar_range_km,
                provenance,
            }
        }
        other => {
            return Err(Error::invalid_input(format!(
                "C ABI returned an unknown plan-candidate kind: {other}"
            )));
        }
    };
    let is_optical = matches!(out_kind, CandidateKind::Optical);

    Ok(PlanCandidate {
        index,
        obs_code: unsafe { cstr_to_string(obs_code) },
        kind: out_kind,
        observable: observable != 0,
        along_track_sigma_arcsec: optical_only(is_optical, along_track_sigma_arcsec),
        cross_track_sigma_arcsec: optical_only(is_optical, cross_track_sigma_arcsec),
        ra_sigma_arcsec: optical_only(is_optical, ra_sigma_arcsec),
        dec_sigma_arcsec: optical_only(is_optical, dec_sigma_arcsec),
        position_angle_deg: optical_only(is_optical, position_angle_deg),
        marginal_volume_reduction,
        marginal_position_improvement,
        post_along_track_sigma_arcsec: optical_only(is_optical, post_along_track_sigma_arcsec),
        post_cross_track_sigma_arcsec: optical_only(is_optical, post_cross_track_sigma_arcsec),
        cumulative: metrics_from_ffi(cumulative),
        active_width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optical_constructor_carries_code_and_sigma() {
        let p = PlannedObservation::optical("F51", [0.25, 0.3], Epoch::from_mjd_tdb(61000.0));
        match &p.kind {
            PlannedObservationKind::Optical {
                optical_code,
                optical_sigma_arcsec,
            } => {
                assert_eq!(optical_code, "F51");
                assert_eq!(*optical_sigma_arcsec, [0.25, 0.3]);
            }
            other => panic!("expected an optical candidate, got {other:?}"),
        }
    }

    #[test]
    fn given_snr_and_link_budget_specs_are_distinguishable() {
        let given = RadarPlanSpec::given(
            RadarStation::GoldstoneDSS14,
            RadarStation::GreenBank,
            RadarMode::Both,
            1.0e5,
            0.1,
            42.0,
        );
        assert_eq!(given.snr, Some(42.0));
        assert_eq!(given.target, TargetRadarProperties::default());

        let budget = RadarPlanSpec::link_budget(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            TargetRadarProperties {
                h_mag: Some(19.7),
                visual_albedo: Some(0.23),
                ..TargetRadarProperties::default()
            },
            600.0,
            RadarMode::Delay,
            1.0e5,
            0.1,
        );
        assert_eq!(budget.snr, None);
        assert_eq!(budget.integration_s, 600.0);
        assert_eq!(budget.target.h_mag, Some(19.7));
    }

    #[test]
    fn absent_link_budget_properties_lower_to_nan() {
        let spec = RadarPlanSpec::link_budget(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            TargetRadarProperties {
                diameter_km: Some(0.34),
                ..TargetRadarProperties::default()
            },
            600.0,
            RadarMode::Both,
            1.0e5,
            0.1,
        );
        let p = PlannedObservation::radar(spec, Epoch::from_mjd_tdb(61000.0));
        let code = CString::default();
        let ffi = planned_to_ffi(&p, &code, 0).expect("radar candidate lowers");
        assert_eq!(ffi.kind, 1);
        assert!(ffi.radar_snr.is_nan(), "absent SNR must lower to NaN");
        assert_eq!(ffi.radar_target_diameter_km, 0.34);
        assert!(ffi.radar_target_h_mag.is_nan());
        assert!(ffi.radar_target_spin_period_hours.is_nan());
    }

    #[test]
    fn radar_mode_tags_round_trip() {
        for mode in [RadarMode::Delay, RadarMode::Doppler, RadarMode::Both] {
            let tag = i32::from(mode.to_ffi());
            assert_eq!(RadarMode::from_ffi(tag), Some(mode));
        }
        // -1 is the C ABI's optical sentinel — never a radar mode.
        assert_eq!(RadarMode::from_ffi(-1), None);
        assert_eq!(RadarMode::from_ffi(3), None);
    }

    #[test]
    fn default_config_matches_the_engine_defaults() {
        let cfg = PlanningConfig::default();
        assert_eq!(cfg.num_threads, None);
        assert_eq!(cfg.epsilon, 1e-9);
        assert_eq!(cfg.force_model, ForceModelTier::Standard);
        assert!(cfg.observatories.is_empty());
        // The default must be the one config the guard below accepts.
        assert!(reject_unread_config_knobs(&cfg).is_ok());
    }

    #[test]
    fn config_knobs_the_planner_never_reads_are_refused() {
        // Both fields ride the shared planning config but are unread on
        // this entry point; accepting them silently would be a dead knob
        // carried across three layers.
        let mut cfg = PlanningConfig {
            observatories: vec![ObservatoryConfig {
                obs_code: "F51".to_string(),
                sigma_arcsec: [0.2, 0.2],
                max_apparent_mag: 22.0,
                min_elongation_deg: 45.0,
                min_elevation_deg: 0.0,
                max_sun_altitude_deg: None,
            }],
            ..PlanningConfig::default()
        };
        let err = reject_unread_config_knobs(&cfg).expect_err("observatories must be refused");
        assert!(err.message.contains("observatories"), "{}", err.message);
        assert!(
            err.message.contains("PlannedObservation"),
            "the refusal must name the alternative: {}",
            err.message
        );

        cfg = PlanningConfig {
            num_threads: Some(4),
            ..PlanningConfig::default()
        };
        let err = reject_unread_config_knobs(&cfg).expect_err("num_threads must be refused");
        assert!(err.message.contains("num_threads"), "{}", err.message);
    }

    #[test]
    fn sky_plane_geometry_is_none_on_a_radar_row() {
        assert_eq!(optical_only(true, 1.5), Some(1.5));
        // Radar rows carry a structural placeholder, not a measurement.
        assert_eq!(optical_only(false, 0.0), None);
        // A non-finite value on a row where the geometry DOES apply is a
        // different failure from "not applicable", so at this layer it
        // survives as Some(NaN). (Python collapses both to one arrow
        // null by convention and says so in the column docs.)
        assert!(
            optical_only(true, f64::NAN).is_some_and(f64::is_nan),
            "a NaN on an optical row must not collapse into the not-applicable None"
        );
    }
    /// Build a default optical candidate for the marshal-guard tests.
    fn optical_at(mjd: f64) -> PlannedObservation {
        PlannedObservation::optical("F51", [0.2, 0.2], Epoch::from_mjd_tdb(mjd))
    }

    #[test]
    fn a_non_finite_candidate_epoch_is_refused_before_the_engine() {
        // The engine sorts candidates chronologically with a partial_cmp
        // unwrap; a NaN epoch panics that comparator and returns an
        // opaque internal-panic code instead of naming the candidate.
        let code = CString::default();
        let err = planned_to_ffi(&optical_at(f64::NAN), &code, 3)
            .expect_err("a NaN epoch must be refused");
        assert!(err.message.contains("epoch"), "{}", err.message);
        assert!(
            err.message.contains('3'),
            "the message must name the candidate: {}",
            err.message
        );
    }

    #[test]
    fn green_bank_is_refused_as_a_transmit_station() {
        // Receive-only: zero transmit power. The link-budget path fails
        // engine-side on the resulting non-positive SNR, but a supplied
        // SNR skips the link budget, so only this guard catches it.
        let spec = RadarPlanSpec::given(
            RadarStation::GreenBank,
            RadarStation::GoldstoneDSS14,
            RadarMode::Both,
            1.0e5,
            0.1,
            50.0,
        );
        let p = PlannedObservation::radar(spec, Epoch::from_mjd_tdb(61000.0));
        let code = CString::default();
        let err = planned_to_ffi(&p, &code, 0).expect_err("Green Bank cannot transmit");
        assert!(err.message.contains("receive-only"), "{}", err.message);
    }

    #[test]
    fn a_non_finite_radar_value_is_refused_rather_than_read_as_absent() {
        // NaN is the C ABI's "absent" / "derive it" discriminant, so
        // Some(NaN) would silently become a different request.
        let code = CString::default();
        let mut spec = RadarPlanSpec::given(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            RadarMode::Both,
            1.0e5,
            0.1,
            f64::NAN,
        );
        let p = PlannedObservation::radar(spec.clone(), Epoch::from_mjd_tdb(61000.0));
        let err = planned_to_ffi(&p, &code, 1).expect_err("Some(NaN) snr must be refused");
        assert!(err.message.contains("snr"), "{}", err.message);

        spec.snr = Some(50.0);
        spec.target.h_mag = Some(f64::INFINITY);
        let p = PlannedObservation::radar(spec, Epoch::from_mjd_tdb(61000.0));
        let err = planned_to_ffi(&p, &code, 1).expect_err("a non-finite H must be refused");
        assert!(err.message.contains("h_mag"), "{}", err.message);
    }

    #[test]
    fn an_unknown_candidate_kind_from_the_c_abi_is_refused() {
        // A newer engine could add a candidate kind; relabelling it would
        // put a row of the wrong shape into the result.
        let c = empyrean_sys::EmpyreanPlanCandidate {
            kind: 2,
            ..Default::default()
        };
        let err = candidate_from_ffi(&c).expect_err("an unknown kind must be refused");
        assert!(err.message.contains("kind"), "{}", err.message);
    }

    #[test]
    fn an_unknown_radar_mode_from_the_c_abi_is_refused() {
        let c = empyrean_sys::EmpyreanPlanCandidate {
            kind: 1,
            radar_mode: 7,
            ..Default::default()
        };
        let err = candidate_from_ffi(&c).expect_err("an unknown radar mode must be refused");
        assert!(err.message.contains("radar mode"), "{}", err.message);
    }

    #[test]
    fn an_interior_nul_in_a_borrowed_string_is_refused() {
        let err = cstring_for("F5\0 1", "optical station code")
            .expect_err("an interior nul must be refused");
        assert!(
            err.message.contains("optical station code"),
            "{}",
            err.message
        );
    }
    #[test]
    fn a_non_cartesian_orbit_is_refused_before_the_engine() {
        // The prior is inverted as a Cartesian state covariance and never
        // converted, so cometary elements would be reinterpreted and the
        // run would return plausible finite nonsense.
        let mut orbit = Orbit::new(crate::CoordinateState::cometary(
            Epoch::from_mjd_tdb(61000.0),
            [0.746, 0.191, 3.339, 204.446, 126.687, 60159.0],
            crate::Frame::EclipticJ2000,
            crate::Origin::SSB,
        ));
        let err = reject_unusable_orbit_basis(&orbit).expect_err("cometary must be refused");
        assert!(err.message.contains("Cartesian"), "{}", err.message);
        assert!(
            err.message.contains("transform_coordinates_single"),
            "the refusal must name the conversion: {}",
            err.message
        );

        // Cartesian but heliocentric: the engine's own origin guard lives
        // on the optical chain, so a radar-only plan would slip past it.
        orbit = Orbit::new(crate::CoordinateState::cartesian(
            Epoch::from_mjd_tdb(61000.0),
            [1.0, 0.0, 0.0, 0.0, 0.017, 0.0],
            crate::Frame::EclipticJ2000,
            crate::Origin::SUN,
        ));
        let err = reject_unusable_orbit_basis(&orbit).expect_err("heliocentric must be refused");
        assert!(err.message.contains("barycenter"), "{}", err.message);
        assert!(
            err.message.contains("transform_coordinates_single"),
            "{}",
            err.message
        );

        // Cartesian + barycentric passes.
        orbit = Orbit::new(crate::CoordinateState::cartesian(
            Epoch::from_mjd_tdb(61000.0),
            [1.0, 0.0, 0.0, 0.0, 0.017, 0.0],
            crate::Frame::EclipticJ2000,
            crate::Origin::SSB,
        ));
        assert!(reject_unusable_orbit_basis(&orbit).is_ok());
    }

    #[test]
    fn the_result_label_falls_back_to_the_orbits_own_id() {
        assert_eq!(
            resolve_orbit_id(Some("explicit"), Some("orbit")),
            Some("explicit")
        );
        assert_eq!(resolve_orbit_id(None, Some("orbit")), Some("orbit"));
        // Empty is the C ABI's "id absent" sentinel at both levels, so it
        // falls through rather than labelling a result with "".
        assert_eq!(resolve_orbit_id(Some(""), Some("orbit")), Some("orbit"));
        assert_eq!(resolve_orbit_id(None, None), None);
        assert_eq!(resolve_orbit_id(Some(""), Some("")), None);
    }

    #[test]
    fn link_budget_inputs_are_refused_alongside_a_supplied_snr() {
        // The engine's spec is a sum type whose supplied-SNR arm carries
        // no target and no integration time, so the C ABI drops all six.
        // Python already refuses this program; the Rust channel must too.
        let mut spec = RadarPlanSpec::given(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            RadarMode::Both,
            1.0e5,
            0.1,
            50.0,
        );
        spec.target.h_mag = Some(19.7);
        spec.integration_s = 600.0;
        let err = reject_link_budget_inputs_with_given_snr(&spec, 2)
            .expect_err("link-budget inputs beside a supplied snr must be refused");
        assert!(err.message.contains("target.h_mag"), "{}", err.message);
        assert!(err.message.contains("integration_s"), "{}", err.message);
        assert!(
            err.message.contains("snr to None"),
            "the refusal must name the fix: {}",
            err.message
        );

        // The link-budget spec itself is untouched by the guard.
        let budget = RadarPlanSpec::link_budget(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            TargetRadarProperties {
                h_mag: Some(19.7),
                ..TargetRadarProperties::default()
            },
            600.0,
            RadarMode::Both,
            1.0e5,
            0.1,
        );
        assert!(reject_link_budget_inputs_with_given_snr(&budget, 0).is_ok());
    }
}
