//! OD result and result-shaped config types: per-observation residuals,
//! summary statistics, the acceptability verdict, fitted station
//! biases, and the [`DetermineResult`] / [`EvaluateResult`] returned
//! by the determine / evaluate / refine entry points.

use std::ffi::CStr;

use crate::observers::obs_code_from_bytes;
use crate::orbit::Orbit;
use crate::propagate::{CovarianceKind, ForceModelTier, PropagatedState};

/// Why an observation was kept or rejected. The `NotEvaluated` variant
/// is the safe-Rust analogue of the C ABI's "not evaluated" sentinel —
/// used on the evaluate path where rejection is not run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionReason {
    /// Observation passed all rejection criteria.
    Accepted,
    /// Rejected by Layer 1 chi-squared threshold.
    ChiSquared,
    /// Rejected by Layer 1 sigma clipping.
    SigmaClip,
    /// Rejected by Layer 2 Cook's distance.
    CooksDistance,
    /// Rejected by Layer 3 information-aware adaptive criterion.
    Adaptive,
    /// Observatory could not be resolved.
    UnsupportedObservatory,
    /// Rejected by Carpino–Milani–Chesley (2003) χ²-with-hysteresis scheme.
    CMC2003,
    /// Skipped because the observation mode is `RAD` (radar). The
    /// optical-only fitter cannot fold radar range / Doppler measurements.
    RadarObservationsUnsupported,
    /// Skipped because the observation mode is `OCC` (stellar
    /// occultation). The optical-only fitter cannot fold occultation
    /// chord timings.
    OccultationObservationsUnsupported,
    /// Tagged by `dc_pipeline`'s multi-arc orchestration: the
    /// observation sits in an opposition group that could not be
    /// reconciled with the in-family fit (sub-arc dynamical
    /// incompatibility, e.g. across a chaotic-capture interval).
    /// Distinct from noise-driven outliers — surfacing the dynamical
    /// regime mismatch separately lets downstream tooling skip past
    /// these without confusing them with measurement error.
    OutsideArc,
    /// Rejection was not evaluated (e.g. evaluate path).
    NotEvaluated,
}

impl RejectionReason {
    pub(super) fn from_int(v: i32) -> Self {
        match v {
            0 => RejectionReason::Accepted,
            1 => RejectionReason::ChiSquared,
            2 => RejectionReason::SigmaClip,
            3 => RejectionReason::CooksDistance,
            4 => RejectionReason::Adaptive,
            5 => RejectionReason::UnsupportedObservatory,
            6 => RejectionReason::CMC2003,
            7 => RejectionReason::RadarObservationsUnsupported,
            8 => RejectionReason::OccultationObservationsUnsupported,
            9 => RejectionReason::OutsideArc,
            _ => RejectionReason::NotEvaluated,
        }
    }
}

/// Per-observation result from orbit determination or evaluation.
///
/// The `obs_id` is what you cross-match to your input ADES rows. NaN
/// values mark stats that weren't computed for the call type (e.g.
/// evaluate doesn't run rejection or influence diagnostics).
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationResidual {
    /// ADES `obsID` (or auto-assigned). Use this to cross-match to
    /// your input observations.
    pub obs_id: String,
    /// MPC observatory code.
    pub obs_code: String,
    /// Star catalog used for astrometric reduction (ADES `astCat`).
    pub ast_cat: Option<String>,
    /// Observation epoch.
    pub epoch: crate::Epoch,
    /// RA·cos(Dec) residual (arcseconds).
    pub ra_residual_arcsec: f64,
    /// Dec residual (arcseconds).
    pub dec_residual_arcsec: f64,
    /// Mahalanobis χ² for this observation. NaN if combined covariance unavailable.
    pub chi2: f64,
    /// Degrees of freedom (number of non-NaN residual dimensions).
    pub dof: u32,
    /// χ² survival probability.
    pub probability: f64,
    /// Was this observation used in the fit?
    pub selected: bool,
    /// Combined obs+predicted covariance for RA·cos(Dec) (arcsec²). NaN if absent.
    pub residual_cov_ra: f64,
    /// Combined obs+predicted covariance for Dec (arcsec²). NaN if absent.
    pub residual_cov_dec: f64,
    /// RA-Dec correlation coefficient (dimensionless, [-1, 1]). NaN if absent.
    pub residual_cov_corr: f64,
    /// Why this observation was kept / rejected.
    pub rejection_reason: RejectionReason,
    /// The criterion value (χ², Cook's D, …) that was tested. NaN if not evaluated.
    pub rejection_criterion: f64,
    /// Static threshold the criterion was compared against. NaN if not evaluated.
    pub rejection_threshold: f64,
    /// Effective threshold for adaptive rejection (Layer 3). NaN otherwise.
    pub rejection_effective_threshold: f64,
    /// D-optimality information loss. NaN if no influence pass.
    pub rejection_information_loss: f64,
    /// Cook's distance. NaN if no influence pass.
    pub cooks_distance: f64,
    /// Scalar leverage \\(h_{ii}\\). NaN if no influence pass.
    pub leverage: f64,
    /// Fractional information contribution \\(f_i\\). NaN if no influence pass.
    pub fractional_information: f64,
    /// Along-track residual (arcsec). NaN when no sky-motion rates.
    pub along_track_arcsec: f64,
    /// Cross-track residual (arcsec). NaN when no sky-motion rates.
    pub cross_track_arcsec: f64,
    /// Along-track 1-sigma uncertainty (arcsec). NaN if unavailable.
    pub along_track_error_arcsec: f64,
    /// Cross-track 1-sigma uncertainty (arcsec). NaN if unavailable.
    pub cross_track_error_arcsec: f64,
    /// Position angle of sky motion (deg, East of North). NaN if unavailable.
    pub track_position_angle_deg: f64,
}

impl ObservationResidual {
    pub(super) fn from_ffi(r: &empyrean_sys::EmpyreanObservationResult) -> Self {
        let obs_id = if r.obs_id.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(r.obs_id) }
                .to_string_lossy()
                .into_owned()
        };
        let ast_cat = if r.ast_cat.is_null() {
            None
        } else {
            let s = unsafe { CStr::from_ptr(r.ast_cat) }
                .to_string_lossy()
                .into_owned();
            (!s.is_empty()).then_some(s)
        };
        Self {
            obs_id,
            obs_code: obs_code_from_bytes(&r.obs_code),
            ast_cat,
            epoch: crate::Epoch::from_mjd_tdb(r.epoch_mjd_tdb),
            ra_residual_arcsec: r.ra_residual_arcsec,
            dec_residual_arcsec: r.dec_residual_arcsec,
            chi2: r.chi2,
            dof: r.dof,
            probability: r.probability,
            selected: r.selected != 0,
            residual_cov_ra: r.residual_cov_ra,
            residual_cov_dec: r.residual_cov_dec,
            residual_cov_corr: r.residual_cov_corr,
            rejection_reason: RejectionReason::from_int(r.rejection_reason),
            rejection_criterion: r.rejection_criterion,
            rejection_threshold: r.rejection_threshold,
            rejection_effective_threshold: r.rejection_effective_threshold,
            rejection_information_loss: r.rejection_information_loss,
            cooks_distance: r.cooks_distance,
            leverage: r.leverage,
            fractional_information: r.fractional_information,
            along_track_arcsec: r.along_track_arcsec,
            cross_track_arcsec: r.cross_track_arcsec,
            along_track_error_arcsec: r.along_track_error_arcsec,
            cross_track_error_arcsec: r.cross_track_error_arcsec,
            track_position_angle_deg: r.track_position_angle_deg,
        }
    }
}

/// Summary statistics over a residual set. AT/CT RMS values are NaN
/// when no sky-motion rates were available.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualSummary {
    /// Total observations.
    pub num_obs: usize,
    /// Observations selected (used in fit).
    pub num_selected: usize,
    /// Observations rejected.
    pub num_rejected: usize,
    /// χ² of selected observations.
    pub chi2: f64,
    /// Effective degrees of freedom.
    pub dof: usize,
    /// Reduced χ² = chi2 / dof. NaN when dof ≤ 0.
    pub reduced_chi2: f64,
    /// RA·cos(Dec) RMS over selected obs (arcsec).
    pub rms_ra_arcsec: f64,
    /// Dec RMS over selected obs (arcsec).
    pub rms_dec_arcsec: f64,
    /// Combined RA·cos(Dec) + Dec residual RMS (arcsec). Single-number
    /// figure matching the find_orb / OrbFit `rms` reporting convention.
    pub rms_combined_arcsec: f64,
    /// Per-observation σ-weighted RA RMS (arcsec).
    pub weighted_rms_ra_arcsec: f64,
    /// Per-observation σ-weighted Dec RMS (arcsec).
    pub weighted_rms_dec_arcsec: f64,
    /// Combined weighted RA·cos(Dec) + Dec residual RMS (arcsec).
    pub weighted_rms_combined_arcsec: f64,
    /// Mean RA·cos(Dec) residual (arcsec).
    pub mean_ra_arcsec: f64,
    /// Mean Dec residual (arcsec).
    pub mean_dec_arcsec: f64,
    /// Standard deviation of RA·cos(Dec) residuals (arcsec).
    pub std_ra_arcsec: f64,
    /// Standard deviation of Dec residuals (arcsec).
    pub std_dec_arcsec: f64,
    /// RMS along-track residual (arcsec). NaN if no AT/CT.
    pub rms_along_track_arcsec: f64,
    /// RMS cross-track residual (arcsec). NaN if no AT/CT.
    pub rms_cross_track_arcsec: f64,
}

impl ResidualSummary {
    pub(super) fn from_ffi(s: &empyrean_sys::EmpyreanResidualSummary) -> Self {
        Self {
            num_obs: s.num_obs,
            num_selected: s.num_selected,
            num_rejected: s.num_rejected,
            chi2: s.chi2,
            dof: s.dof,
            reduced_chi2: s.reduced_chi2,
            rms_ra_arcsec: s.rms_ra_arcsec,
            rms_dec_arcsec: s.rms_dec_arcsec,
            rms_combined_arcsec: s.rms_combined_arcsec,
            weighted_rms_ra_arcsec: s.weighted_rms_ra_arcsec,
            weighted_rms_dec_arcsec: s.weighted_rms_dec_arcsec,
            weighted_rms_combined_arcsec: s.weighted_rms_combined_arcsec,
            mean_ra_arcsec: s.mean_ra_arcsec,
            mean_dec_arcsec: s.mean_dec_arcsec,
            std_ra_arcsec: s.std_ra_arcsec,
            std_dec_arcsec: s.std_dec_arcsec,
            rms_along_track_arcsec: s.rms_along_track_arcsec,
            rms_cross_track_arcsec: s.rms_cross_track_arcsec,
        }
    }
}

/// Coordinate basis tag for OD output covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceRepresentation {
    /// `(x, y, z, vx, vy, vz)` covariance.
    Cartesian,
    /// Keplerian-element covariance.
    Keplerian,
    /// Cometary-element covariance.
    Cometary,
    /// Spherical-coordinate covariance.
    Spherical,
}

impl CovarianceRepresentation {
    pub(super) fn from_int(v: i32) -> Self {
        match v {
            0 => Self::Cartesian,
            1 => Self::Keplerian,
            2 => Self::Cometary,
            _ => Self::Spherical,
        }
    }
}

/// Solve-for parameter set used by differential correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveForParams {
    /// Solve only for the 6-element state vector.
    StateOnly,
    /// Solve for state + (A1, A2, A3) non-grav coefficients (9 params).
    StateAndNonGrav,
    /// Start with state-only and escalate to the 9-parameter set
    /// automatically (see [`AutoEscalationPolicy`](super::AutoEscalationPolicy)).
    Auto,
}

impl SolveForParams {
    pub(super) fn from_int(v: i32) -> Self {
        match v {
            0 => Self::StateOnly,
            1 => Self::StateAndNonGrav,
            _ => Self::Auto,
        }
    }
    pub(super) fn to_int(self) -> i32 {
        match self {
            Self::StateOnly => 0,
            Self::StateAndNonGrav => 1,
            Self::Auto => 2,
        }
    }
}

/// Output epoch for the fitted orbit.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OutputEpoch {
    /// Midpoint of the observation arc (default). Resolved against
    /// the active observation set (not the full input arc) so multi-
    /// year arcs whose mid-arc target lies in a chaotic interval keep
    /// the integrator anchor inside the IOD opposition window.
    #[default]
    MidArc,
    /// Epoch of the last observation. Resolved against the active
    /// observation set.
    LastObservation,
    /// Anchor at the IOD-derived epoch — the state stays where the
    /// initial-orbit determination produced it. Matches OrbFit's
    /// `epoch.eq0` and find_orb's "anchor at most recent good fit"
    /// pattern.
    IODEpoch,
    /// Explicit MJD TDB.
    Epoch(f64),
}

/// Origin-policy selector for the OD pipeline.
///
/// Mirrors `scott::od::OriginPolicy` — controls whether the
/// determine / evaluate / refine pipeline tries a heliocentric →
/// body-centric Earth cascade ([`OriginPolicy::Auto`], default) or
/// pins to a specific central body
/// ([`OriginPolicy::Explicit`]). Auto handles TCOs / minimoons /
/// geocentric impactors / chaotic-capture interiors without per-
/// object regime classification by the caller; Explicit is required
/// for cataloged satellites where heliocentric Gauss is unphysical
/// and recommended for pipelines that already know the regime.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum OriginPolicy {
    /// Selects the central body (heliocentric vs Earth-centric)
    /// automatically. Default.
    #[default]
    Auto,
    /// Pin IOD + DC to a specific central body. Skips the cascade.
    Explicit(crate::coordinate::Origin),
}

impl From<crate::coordinate::Origin> for OriginPolicy {
    fn from(origin: crate::coordinate::Origin) -> Self {
        Self::Explicit(origin)
    }
}

/// Acceptability sub-checks reported on a [`DetermineResult`].
///
/// The `*_value` fields carry the measured statistic; the
/// `*_threshold` fields the bound it was compared against; the `*_ok`
/// flags the verdict on each individual gate.
///
/// # Aggregate verdicts
///
/// Two aggregate flags summarise the report:
///
/// - [`fit_acceptable`](Self::fit_acceptable) is the **AND** of the
///   fit-quality gates:
///   `converged_ok` ∧ `covariance_ok` ∧ `reduced_chi2_ok` ∧ `rms_ok` ∧
///   `residual_isotropy_ok`. It answers: did the differential
///   correction land on a numerically valid fit whose residuals look
///   like noise rather than signal?
/// - [`extrapolation_acceptable`](Self::extrapolation_acceptable) is
///   `fit_acceptable` **AND** the trustworthy-forward-propagation
///   gates: `arc_coverage_ok` ∧ `fractional_sigma_a_ok`. It answers:
///   on top of being a valid fit, is the arc long enough and the
///   semi-major-axis uncertainty tight enough that propagating the
///   orbit forward is meaningful?
///
/// Use `fit_acceptable` to gate publication-quality residuals; use
/// `extrapolation_acceptable` to gate forward propagation, ephemeris
/// generation, or impact-risk assessment. Tighten the underlying
/// thresholds in
/// [`AcceptabilityThresholds`](super::AcceptabilityThresholds) for
/// impact-monitoring orbits,
/// loosen for short-arc discovery fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptabilityReport {
    /// Aggregate verdict on fit quality.
    pub fit_acceptable: bool,
    /// Aggregate verdict on whether the orbit may be safely extrapolated forward.
    pub extrapolation_acceptable: bool,
    /// Did the differential correction converge?
    pub converged_ok: bool,
    /// Did the reduced χ² pass its threshold?
    pub reduced_chi2_ok: bool,
    /// Measured reduced χ².
    pub reduced_chi2_value: f64,
    /// Threshold the reduced χ² was compared against.
    pub reduced_chi2_threshold: f64,
    /// Did the residual RMS pass its threshold?
    pub rms_ok: bool,
    /// Measured RMS (arcsec).
    pub rms_value_arcsec: f64,
    /// Threshold the RMS was compared against (arcsec).
    pub rms_threshold_arcsec: f64,
    /// Did the AT/CT ratio pass the residual-isotropy gate?
    pub residual_isotropy_ok: bool,
    /// Measured AT/CT ratio.
    pub at_ct_ratio_value: f64,
    /// Threshold the AT/CT ratio was compared against.
    pub at_ct_ratio_threshold: f64,
    /// Is the fitted covariance positive-definite?
    pub covariance_ok: bool,
    /// Is the observation arc long enough?
    pub arc_coverage_ok: bool,
    /// Measured arc length (days).
    pub arc_days_value: f64,
    /// Minimum arc length threshold (days).
    pub arc_days_threshold: f64,
    /// Did σₐ / |a| pass its threshold?
    pub fractional_sigma_a_ok: bool,
    /// Measured σₐ / |a|.
    pub fractional_sigma_a_value: f64,
    /// Threshold for σₐ / |a|.
    pub fractional_sigma_a_threshold: f64,
}

impl AcceptabilityReport {
    pub(super) fn from_ffi(r: &empyrean_sys::EmpyreanAcceptabilityReport) -> Self {
        Self {
            fit_acceptable: r.fit_acceptable != 0,
            extrapolation_acceptable: r.extrapolation_acceptable != 0,
            converged_ok: r.converged_ok != 0,
            reduced_chi2_ok: r.reduced_chi2_ok != 0,
            reduced_chi2_value: r.reduced_chi2_value,
            reduced_chi2_threshold: r.reduced_chi2_threshold,
            rms_ok: r.rms_ok != 0,
            rms_value_arcsec: r.rms_value_arcsec,
            rms_threshold_arcsec: r.rms_threshold_arcsec,
            residual_isotropy_ok: r.residual_isotropy_ok != 0,
            at_ct_ratio_value: r.at_ct_ratio_value,
            at_ct_ratio_threshold: r.at_ct_ratio_threshold,
            covariance_ok: r.covariance_ok != 0,
            arc_coverage_ok: r.arc_coverage_ok != 0,
            arc_days_value: r.arc_days_value,
            arc_days_threshold: r.arc_days_threshold,
            fractional_sigma_a_ok: r.fractional_sigma_a_ok != 0,
            fractional_sigma_a_value: r.fractional_sigma_a_value,
            fractional_sigma_a_threshold: r.fractional_sigma_a_threshold,
        }
    }
}

/// One per-station fitted nuisance bias.
///
/// Populated rows correspond to stations that met the
/// `min_obs_per_station` threshold during the Schur-eliminated bias
/// fit. Significance is the maximum of `|bᵢ|/σᵢ` across the populated
/// components — a value ≥ 3 indicates the data is pushing against the
/// prior on at least one component.
#[derive(Debug, Clone, PartialEq)]
pub struct StationBias {
    /// MPC observatory code.
    pub obs_code: String,
    /// Number of observations from this station used in the fit.
    pub n_obs: usize,
    /// Fitted RA bias (arcsec).
    pub bias_ra_arcsec: f64,
    /// 1-σ uncertainty on the RA bias (arcsec).
    pub sigma_ra_arcsec: f64,
    /// Fitted Dec bias (arcsec).
    pub bias_dec_arcsec: f64,
    /// 1-σ uncertainty on the Dec bias (arcsec).
    pub sigma_dec_arcsec: f64,
    /// Fitted timing bias (seconds), when the timing nuisance is active.
    pub bias_timing_sec: Option<f64>,
    /// 1-σ uncertainty on the timing bias (seconds).
    pub sigma_timing_sec: Option<f64>,
    /// Maximum of `|bᵢ|/σᵢ` across the populated components — values
    /// ≥ 3 indicate the data is constraining the bias against the prior.
    pub significance: f64,
}

impl StationBias {
    pub(super) fn from_ffi(b: &empyrean_sys::EmpyreanStationBias) -> Self {
        let obs_code = if b.obs_code.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(b.obs_code) }
                .to_string_lossy()
                .into_owned()
        };
        let (bias_t, sigma_t) = if b.has_timing != 0 {
            (Some(b.bias_timing_sec), Some(b.sigma_timing_sec))
        } else {
            (None, None)
        };
        Self {
            obs_code,
            n_obs: b.n_obs,
            bias_ra_arcsec: b.bias_ra_arcsec,
            sigma_ra_arcsec: b.sigma_ra_arcsec,
            bias_dec_arcsec: b.bias_dec_arcsec,
            sigma_dec_arcsec: b.sigma_dec_arcsec,
            bias_timing_sec: bias_t,
            sigma_timing_sec: sigma_t,
            significance: b.significance,
        }
    }
}

/// Result of a differential-correction orbit determination.
#[derive(Debug, Clone, PartialEq)]
pub struct DetermineResult {
    /// Fitted orbit — a fully re-feedable [`Orbit`] carrying the fitted
    /// state, its covariance, and the **absolute** non-gravitational model
    /// (A1/A2/A3 + g(r) + thermal-lag `dt`). Pass it straight back into
    /// [`Context::evaluate`](crate::Context::evaluate),
    /// [`Context::refine`](crate::Context::refine),
    /// [`Context::propagate`](crate::Context::propagate), or
    /// [`Context::compute_impact_probabilities`](crate::Context::compute_impact_probabilities)
    /// with no reconstruction. For the bare state snapshot
    /// (position/velocity), use [`state`](Self::state).
    pub orbit: Orbit,
    /// Per-observation residuals + rejection / influence diagnostics.
    pub residuals: Vec<ObservationResidual>,
    /// Summary statistics.
    pub summary: ResidualSummary,
    /// Iterations used in the final DC pass.
    pub iterations: u32,
    /// Final iteration's convergence metric Δx^T N Δx.
    pub update_norm: f64,
    /// Did the DC reach its stopping criterion?
    pub converged: bool,
    /// Fitted 6×6 state covariance, in [`covariance_representation`](Self::covariance_representation).
    pub covariance: [[f64; 6]; 6],
    /// Coordinate basis of [`covariance`](Self::covariance) /
    /// [`covariance_9x9`](Self::covariance_9x9).
    pub covariance_representation: CovarianceRepresentation,
    /// Full 9×9 covariance over (state, A1, A2, A3) when solving for non-grav.
    pub covariance_9x9: Option<[[f64; 9]; 9]>,
    /// Cumulative non-grav parameter corrections (ΔA1, ΔA2, ΔA3) when present.
    pub non_grav_delta: Option<[f64; 3]>,
    /// Number of rejection-refit passes performed.
    pub rejection_passes: u32,
    /// Number of oppositions fit.
    pub num_oppositions_fit: u32,
    /// Force model tier actually used.
    pub force_model_used: ForceModelTier,
    /// Solve-for parameter set that produced this fit.
    pub solve_for_used: SolveForParams,
    /// Structured fit-quality verdict.
    pub acceptability: AcceptabilityReport,
    /// Per-station fitted RA/Dec biases when `fit_station_biases` was
    /// active. Empty otherwise.
    pub station_biases: Vec<StationBias>,
}

impl DetermineResult {
    /// The fitted orbit as a bare Cartesian state snapshot
    /// (epoch + position + velocity + covariance).
    ///
    /// Convenience for callers that want the flat numbers rather than the
    /// re-feedable [`orbit`](Self::orbit). The STM/STT are absent — orbit
    /// determination does not emit a state-transition matrix — and the
    /// resolved covariance kind is [`CovarianceKind::Linear`] (the fitted
    /// formal covariance). The covariance is reported in
    /// [`covariance_representation`](Self::covariance_representation).
    pub fn state(&self) -> PropagatedState {
        let st = &self.orbit.state;
        let e = st.elements;
        PropagatedState {
            epoch: st.epoch,
            position: [e[0], e[1], e[2]],
            velocity: [e[3], e[4], e[5]],
            origin: st.origin,
            frame: st.frame,
            covariance: Some(self.covariance),
            stm: None,
            stt: None,
            resolved_kind: CovarianceKind::Linear,
        }
    }
}

/// Result of evaluating a candidate orbit against observations (no fit).
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluateResult {
    /// Per-observation residuals.
    pub residuals: Vec<ObservationResidual>,
    /// Summary statistics.
    pub summary: ResidualSummary,
}
