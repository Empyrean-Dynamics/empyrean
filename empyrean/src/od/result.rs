//! OD result and result-shaped config types: per-observation residuals,
//! summary statistics, the acceptability verdict, fitted station
//! biases, and the [`DetermineResult`] / [`EvaluateResult`] returned
//! by the determine / evaluate / refine entry points.

use crate::joint::ParamDisposition;
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
    /// Rejected by the blunder gate — the CMC2003 hysteresis mechanics at
    /// blunder scale (χ² 100 / 81, a 10σ test). A data error, not a
    /// statistical outlier: distinct from [`CMC2003`](Self::CMC2003), which
    /// is the same mechanics at 3σ outlier scale.
    Blunder,
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
    /// The observation's χ² against the published orbit is non-finite, so
    /// it cannot participate in any fit statistic. Reported explicitly so
    /// the row reads as excluded rather than silently counting as used.
    NonFiniteChi2,
    /// The propagation retained no Jacobian / STM at this observation's
    /// epoch, so the observation contributed no row to the normal
    /// equations — it was never part of the fit.
    MissingJacobian,
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
            10 => RejectionReason::NonFiniteChi2,
            11 => RejectionReason::MissingJacobian,
            16 => RejectionReason::Blunder,
            _ => RejectionReason::NotEvaluated,
        }
    }
}

/// Which branch of the weighting pipeline supplied a delivered row's base
/// σ. Not recoverable from the σ itself: a floor destroys the value it
/// replaced, so the branch is recorded where it is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaSource {
    /// The record's own reported σ (ADES `rmsRA` / `rmsDec`), used
    /// verbatim. Also every delivered radar row.
    Reported,
    /// A station rule a preset constructor installed from a published
    /// floors table (VFCC2017 / NEODyS).
    Preset,
    /// A caller-written [`WeightingLayer::ObservatoryRule`](super::weighting::WeightingLayer::ObservatoryRule)
    /// carrying no preset stamp.
    Layer,
    /// [`WeightingConfig::default_sigma_arcsec`](super::weighting::WeightingConfig::default_sigma_arcsec)
    /// — no record or rule supplied a σ, or the floor fallback applied it.
    Default,
    /// The row carries no weighted σ at all (its σ columns are NaN): a
    /// skipped stub, an IOD output, or a span-grouped row never evaluated.
    NotWeighted,
    /// The engine reported a σ-provenance this build does not model (a
    /// variant added upstream after these bindings were generated).
    /// Distinct from [`NotWeighted`](Self::NotWeighted): a row weighted by
    /// an unknown branch was still weighted, never reported as unweighted.
    Unknown,
}

impl SigmaSource {
    fn from_int(v: u8) -> Self {
        match v as u32 {
            empyrean_sys::EMPYREAN_SIGMA_SOURCE_REPORTED => SigmaSource::Reported,
            empyrean_sys::EMPYREAN_SIGMA_SOURCE_PRESET => SigmaSource::Preset,
            empyrean_sys::EMPYREAN_SIGMA_SOURCE_LAYER => SigmaSource::Layer,
            empyrean_sys::EMPYREAN_SIGMA_SOURCE_DEFAULT => SigmaSource::Default,
            empyrean_sys::EMPYREAN_SIGMA_SOURCE_NOT_WEIGHTED => SigmaSource::NotWeighted,
            // Every unrecognised code (including EMPYREAN_SIGMA_SOURCE_UNKNOWN)
            // decodes to Unknown — never silently to NotWeighted.
            _ => SigmaSource::Unknown,
        }
    }
}

/// The error **law** a delivered row was fitted under.
///
/// A delivered-value type, distinct from the config-input
/// [`NoiseModel`](super::weighting::NoiseModel): the wrapper and
/// `libempyrean` are separately versioned and loaded at run time, so a
/// newer library can mint a law tag this build does not know. Such a tag
/// surfaces as [`Unknown`](Self::Unknown) carrying the raw wire value —
/// never silently as a valid law — mirroring
/// [`DetermineFailureKind::Unknown`] and [`SigmaSource::Unknown`]. The
/// config input enum stays constructible only with valid laws.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeliveredNoiseModel {
    /// A law this build knows.
    Known(super::weighting::NoiseModel),
    /// A law tag the loaded library minted that this wrapper does not
    /// model — the raw wire value, kept verbatim. Its appearance means the
    /// library is newer than the wrapper.
    Unknown(u8),
}

impl DeliveredNoiseModel {
    /// Pack into the C-ABI tagged value, round-tripping an
    /// [`Unknown`](Self::Unknown) tag verbatim (with `nu = NaN`).
    pub(crate) fn to_ffi(self) -> empyrean_sys::EmpyreanNoiseModel {
        match self {
            DeliveredNoiseModel::Known(nm) => nm.to_ffi(),
            DeliveredNoiseModel::Unknown(tag) => {
                empyrean_sys::EmpyreanNoiseModel { tag, nu: f64::NAN }
            }
        }
    }
}

/// The outlier-rejection gate a fit was configured with, as delivered.
///
/// A delivered-value type, distinct from the config-input
/// [`RejectionKind`](super::rejection::RejectionKind), for the same
/// separately-versioned reason as [`DeliveredNoiseModel`]: an unknown gate
/// tag surfaces as [`Unknown`](Self::Unknown) with the raw value, never as
/// a valid gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveredRejectionKind {
    /// A gate this build knows.
    Known(super::rejection::RejectionKind),
    /// A gate tag the loaded library minted that this wrapper does not
    /// model — the raw wire value, kept verbatim.
    Unknown(u8),
}

/// Decode the per-row noise-model tag + ν the engine delivered. An
/// unrecognised tag is kept as [`DeliveredNoiseModel::Unknown`], never
/// decoded to a valid law (the library may be newer than the wrapper).
fn noise_model_from_ffi(tag: u8, nu: f64) -> DeliveredNoiseModel {
    use super::weighting::NoiseModel;
    match tag as u32 {
        empyrean_sys::EMPYREAN_NOISE_MODEL_GAUSSIAN => {
            DeliveredNoiseModel::Known(NoiseModel::Gaussian)
        }
        empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T => {
            DeliveredNoiseModel::Known(NoiseModel::StudentT { nu })
        }
        empyrean_sys::EMPYREAN_NOISE_MODEL_NIGHT_SCALE => {
            DeliveredNoiseModel::Known(NoiseModel::NightScale { nu })
        }
        _ => DeliveredNoiseModel::Unknown(tag),
    }
}

/// Decode the configured outlier-rejection gate tag the engine delivered.
/// An unrecognised tag is kept as [`DeliveredRejectionKind::Unknown`],
/// never decoded to a valid gate.
pub(super) fn rejection_kind_from_ffi(v: u8) -> DeliveredRejectionKind {
    use super::rejection::RejectionKind;
    match v as u32 {
        empyrean_sys::EMPYREAN_REJECTION_KIND_ADAPTIVE => {
            DeliveredRejectionKind::Known(RejectionKind::Adaptive)
        }
        empyrean_sys::EMPYREAN_REJECTION_KIND_CMC2003 => {
            DeliveredRejectionKind::Known(RejectionKind::CMC2003)
        }
        empyrean_sys::EMPYREAN_REJECTION_KIND_BLUNDER => {
            DeliveredRejectionKind::Known(RejectionKind::Blunder)
        }
        _ => DeliveredRejectionKind::Unknown(v),
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
    /// ADES object identifier of the fit this row belongs to.
    ///
    /// `Some` for rows from [`Context::determine`](crate::Context::determine),
    /// which fits per object — so residuals from a whole batch can be
    /// concatenated into one table and stay attributable. `None` for
    /// [`Context::evaluate`](crate::Context::evaluate) /
    /// [`Context::refine`](crate::Context::refine), where the caller
    /// supplied the one orbit and there is no grouping key.
    pub object_id: Option<String>,
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
    /// D-optimality information loss on removal, from the influence
    /// pass: \\(\Delta_i = \log\det N - \log\det(N - I_i)\\). NaN if
    /// no influence pass was run; +∞ when removing this observation
    /// makes the normal matrix singular (the observation is
    /// indispensable).
    pub influence_information_loss: f64,
    /// Off-diagonal covariance of the (along-track, cross-track)
    /// residual pair (arcsec²). Together with the AT/CT 1σ fields this
    /// reconstructs the full symmetric 2×2. NaN when unavailable.
    pub along_cross_covariance_arcsec2: f64,
    /// Radar (delay/Doppler) residual block. `None` for optical
    /// observations; on radar rows the optical RA/Dec residual fields
    /// are NaN.
    pub radar: Option<RadarResidual>,
    /// The error **law** this row was fitted under — the global
    /// [`WeightingConfig::noise_model`](super::weighting::WeightingConfig::noise_model)
    /// or the per-station override a rule set for it.
    /// [`DeliveredNoiseModel::Known(NoiseModel::Gaussian)`](DeliveredNoiseModel)
    /// on a default fit and on every radar row;
    /// [`DeliveredNoiseModel::Unknown`] if the loaded library is newer than
    /// this wrapper.
    pub noise_model: DeliveredNoiseModel,
    /// The law's per-row **robust weight** \\(w = \nu/(\nu+\delta^2) \in
    /// (0,1]\\) — its down-weighting of this row alone. Exactly `1.0` under
    /// the Gaussian law; smaller for a large residual under a heavy-tailed
    /// law. The quantity the summary [`n_eff`](ResidualSummary::n_eff) is
    /// defined over.
    pub robust_weight: f64,
    /// The RA·cos(Dec) **1σ the fit actually weighted this row with**
    /// (arcsec), every weighting layer folded in. NaN on skipped stubs and
    /// radar rows.
    pub sigma_ra_used_arcsec: f64,
    /// The Dec **1σ the fit actually weighted this row with** (arcsec); see
    /// [`sigma_ra_used_arcsec`](Self::sigma_ra_used_arcsec).
    pub sigma_dec_used_arcsec: f64,
    /// The RA·cos(Dec) **effective 1σ** \\(\sigma_\text{eff} =
    /// \sigma_\text{used}/\sqrt{w}\\) (arcsec): the fitted σ inflated by the
    /// law's down-weighting [`robust_weight`](Self::robust_weight). Equals
    /// [`sigma_ra_used_arcsec`](Self::sigma_ra_used_arcsec) under the
    /// Gaussian law, and is `≥` it in general.
    pub sigma_eff_ra_arcsec: f64,
    /// The Dec effective 1σ (arcsec); see
    /// [`sigma_eff_ra_arcsec`](Self::sigma_eff_ra_arcsec).
    pub sigma_eff_dec_arcsec: f64,
    /// Which branch of the weighting pipeline supplied this row's base σ.
    pub sigma_source: SigmaSource,
    /// The single factor by which this row's base σ was inflated to reach
    /// [`sigma_ra_used_arcsec`](Self::sigma_ra_used_arcsec) — the product of
    /// the observatory-rule scale and the nightly-batch factor. `1.0` when
    /// no layer scaled the row. Excludes the floor.
    pub sigma_scale_applied: f64,
}

/// Which radar observable a [`RadarResidual`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadarResidualKind {
    /// Round-trip delay (seconds).
    Delay,
    /// Two-way Doppler shift (hertz).
    Doppler,
}

/// Radar residual block on an [`ObservationResidual`] row.
#[derive(Debug, Clone, PartialEq)]
pub struct RadarResidual {
    /// Which observable the residual refers to (sets the units).
    pub kind: RadarResidualKind,
    /// Observed − predicted: seconds for delay, hertz for Doppler.
    pub residual: f64,
    /// χ² of the radar residual.
    pub chi2: f64,
    /// Degrees of freedom (1 for radar).
    pub dof: u32,
    /// χ² survival probability.
    pub probability: f64,
    /// Combined observed+predicted variance (s² / Hz²), when available.
    pub variance: Option<f64>,
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
        let object_id = if r.object_id.is_null() {
            None
        } else {
            let s = unsafe { CStr::from_ptr(r.object_id) }
                .to_string_lossy()
                .into_owned();
            (!s.is_empty()).then_some(s)
        };
        Self {
            obs_id,
            object_id,
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
            influence_information_loss: r.influence_information_loss,
            along_cross_covariance_arcsec2: r.along_cross_covariance_arcsec2,
            radar: (r.has_radar != 0).then(|| RadarResidual {
                kind: if r.radar_kind == empyrean_sys::EMPYREAN_RADAR_KIND_DOPPLER as u8 {
                    RadarResidualKind::Doppler
                } else {
                    RadarResidualKind::Delay
                },
                residual: r.radar_residual,
                chi2: r.radar_chi2,
                dof: r.radar_dof,
                probability: r.radar_probability,
                variance: r.radar_variance.is_finite().then_some(r.radar_variance),
            }),
            noise_model: noise_model_from_ffi(r.noise_model, r.noise_model_nu),
            robust_weight: r.robust_weight,
            sigma_ra_used_arcsec: r.sigma_ra_used_arcsec,
            sigma_dec_used_arcsec: r.sigma_dec_used_arcsec,
            sigma_eff_ra_arcsec: r.sigma_eff_ra_arcsec,
            sigma_eff_dec_arcsec: r.sigma_eff_dec_arcsec,
            sigma_source: SigmaSource::from_int(r.sigma_source),
            sigma_scale_applied: r.sigma_scale_applied,
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
    /// **Effective** number of selected observations
    /// \\(n_\text{eff} = (\sum_i w_i)^2 / \sum_i w_i^2\\), with \\(w_i\\) the
    /// per-row [`robust_weight`](ObservationResidual::robust_weight). Equals
    /// [`num_selected`](Self::num_selected) under the Gaussian law; smaller
    /// under a heavy-tailed law. `0.0` when nothing was selected.
    pub n_eff: f64,
    /// Law-aware calibration reference: the median of the nominal 2-DOF
    /// Mahalanobis distance \\(d^2 = r^\top W r\\) the configured law
    /// predicts. \\(2\ln 2 \approx 1.386\\) under the Gaussian law.
    pub expected_median_d2: f64,
    /// Law-aware calibration reference: the fraction of rows the configured
    /// law predicts beyond the 3σ ellipse (\\(d^2 > 9\\)).
    /// \\(e^{-4.5} \approx 0.011\\) under the Gaussian law.
    pub expected_tail: f64,
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
            n_eff: s.n_eff,
            expected_median_d2: s.expected_median_d2,
            expected_tail: s.expected_tail,
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

/// The largest number of thrust Δv correction segments one fit can
/// declare, and the length of [`SolveFor::thrust`].
pub const MAX_THRUST_SEGMENTS: usize = 3;

/// What a fit does with each parameter axis (mirrors the engine's
/// `SolveFor`).
///
/// Each axis carries a [`ParamDisposition`] rather than a flag, because
/// the three treatments of a parameter are different operations: solved
/// is estimated, considered inflates the posterior through its
/// measurement partials without being estimated, and fixed is
/// marginalized out. Using one where another is meant is silent — each
/// produces a well-formed covariance.
///
/// Used to request an explicit multi-axis fit via
/// [`SolveForParams::Explicit`], and reported back on
/// [`DetermineResult::dispositions`] as what the fit actually ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveFor {
    /// Disposition of the Marsden A1/A2/A3 block. Solving or considering
    /// it requires the orbit to carry a non-grav covariance.
    pub marsden: ParamDisposition,
    /// Disposition of the non-grav time delay DT. Solving it requires
    /// `marsden` solved plus a DT value and prior variance.
    pub dt: ParamDisposition,
    /// Disposition of the SRP AMRAT. Requires an SRP slot carrying an
    /// AMRAT prior variance.
    pub amrat: ParamDisposition,
    /// Disposition of each **declared** thrust Δv segment, positional
    /// with the orbit's `correction_covariances`.
    ///
    /// Positional rather than a count: a considered or fixed segment
    /// sits between solved ones as readily as after them, and a count
    /// cannot say which burn is which. Entries beyond the declared
    /// count must be [`ParamDisposition::Fixed`].
    pub thrust: [ParamDisposition; MAX_THRUST_SEGMENTS],
}

impl Default for SolveFor {
    /// Every axis [`Fixed`](ParamDisposition::Fixed) — the state-only
    /// set, and what a `false` flag always meant.
    fn default() -> Self {
        Self {
            marsden: ParamDisposition::Fixed,
            dt: ParamDisposition::Fixed,
            amrat: ParamDisposition::Fixed,
            thrust: [ParamDisposition::Fixed; MAX_THRUST_SEGMENTS],
        }
    }
}

impl SolveFor {
    /// Solve the leading `segments` burns, leaving the rest fixed.
    ///
    /// Refuses an over-budget request rather than saturating: clamping
    /// would turn a four-segment request into a three-segment fit and
    /// marginalize the fourth burn without a word.
    pub fn with_leading_thrust(mut self, segments: usize) -> crate::error::Result<Self> {
        if segments > MAX_THRUST_SEGMENTS {
            return Err(crate::error::Error::invalid_input(format!(
                "requested {segments} thrust segments but the engine's \
                 maximum is {MAX_THRUST_SEGMENTS} — the 17-column wide budget \
                 is shared across the state, Marsden, DT, AMRAT and thrust \
                 axes"
            )));
        }
        for d in self.thrust.iter_mut().take(segments) {
            *d = ParamDisposition::Solved;
        }
        Ok(self)
    }

    /// How many thrust segments this fit solves.
    pub fn solved_thrust_segments(&self) -> usize {
        self.thrust.iter().filter(|d| d.is_solved()).count()
    }

    /// How many thrust segments this fit considers.
    pub fn considered_thrust_segments(&self) -> usize {
        self.thrust.iter().filter(|d| d.is_considered()).count()
    }

    /// Recover the solved axes from a tagged [`SolvedCovariance`].
    ///
    /// Only ever reports `Solved` or `Fixed`: a solved-covariance's slot
    /// tags record what occupied a column, and a considered axis
    /// occupies none. A result's true partition — including which axes
    /// were considered — is [`DetermineResult::dispositions`], which the engine
    /// reports directly.
    fn from_covariance(cov: &SolvedCovariance) -> Self {
        let d = |present: bool| {
            if present {
                ParamDisposition::Solved
            } else {
                ParamDisposition::Fixed
            }
        };
        let mut thrust = [ParamDisposition::Fixed; MAX_THRUST_SEGMENTS];
        for slot in thrust.iter_mut().take(cov.thrust_slots.len()) {
            *slot = ParamDisposition::Solved;
        }
        Self {
            marsden: d(cov.marsden_slot.is_some()),
            dt: d(cov.dt_slot.is_some()),
            amrat: d(cov.amrat_slot.is_some()),
            thrust,
        }
    }

    pub(crate) fn to_ffi(self) -> empyrean_sys::EmpyreanSolveFor {
        let mut thrust_dispositions = [ParamDisposition::Fixed.to_ffi(); MAX_THRUST_SEGMENTS];
        for (i, d) in self.thrust.iter().enumerate() {
            thrust_dispositions[i] = d.to_ffi();
        }
        empyrean_sys::EmpyreanSolveFor {
            marsden: self.marsden.to_ffi(),
            dt: self.dt.to_ffi(),
            amrat: self.amrat.to_ffi(),
            thrust_dispositions,
        }
    }

    pub(crate) fn from_ffi(f: &empyrean_sys::EmpyreanSolveFor) -> crate::error::Result<Self> {
        let mut thrust = [ParamDisposition::Fixed; MAX_THRUST_SEGMENTS];
        for (i, d) in f.thrust_dispositions.iter().enumerate() {
            thrust[i] = ParamDisposition::from_ffi(*d, &format!("thrust[{i}]"))?;
        }
        Ok(Self {
            marsden: ParamDisposition::from_ffi(f.marsden, "marsden")?,
            dt: ParamDisposition::from_ffi(f.dt, "dt")?,
            amrat: ParamDisposition::from_ffi(f.amrat, "amrat")?,
            thrust,
        })
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
    /// Solve an explicit multi-axis set (any of Marsden / DT / AMRAT /
    /// thrust) that the coarse variants can't name. On the config it
    /// carries the requested axes; on a result it is reconstructed from
    /// the solved covariance's slot tags.
    Explicit(SolveFor),
}

impl SolveForParams {
    /// Reconstruct the solve-for set that produced a fit. The C ABI
    /// reports a coarse integer code; an `Explicit` fit's exact axes are
    /// recovered from the solved covariance's slot tags.
    pub(super) fn from_result(code: i32, cov: Option<&SolvedCovariance>) -> Self {
        match code {
            0 => Self::StateOnly,
            1 => Self::StateAndNonGrav,
            3 => Self::Explicit(cov.map(SolveFor::from_covariance).unwrap_or_default()),
            _ => Self::Auto,
        }
    }
    pub(super) fn to_int(self) -> i32 {
        match self {
            Self::StateOnly => 0,
            Self::StateAndNonGrav => 1,
            Self::Auto => 2,
            Self::Explicit(_) => 3,
        }
    }
    /// The per-axis flags this request implies — set on the C config's
    /// `solve_for_flags` (read only when `to_int() == 3`).
    pub(super) fn flags(self) -> SolveFor {
        match self {
            Self::StateOnly | Self::Auto => SolveFor::default(),
            Self::StateAndNonGrav => SolveFor {
                marsden: ParamDisposition::Solved,
                ..SolveFor::default()
            },
            Self::Explicit(sf) => sf,
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
    /// Did the fit retain enough of its input? `false` means the
    /// residual bars above describe a heavily pruned subset.
    ///
    /// Reproduce the fraction from
    /// [`selection_fraction_value`](Self::selection_fraction_value), not
    /// from [`ResidualSummary`] — the summary counts merged radar /
    /// occultation stub rows that were never candidates for outlier
    /// pruning, so its ratio is a different (smaller) number.
    pub selection_fraction_ok: bool,
    /// Fraction of the SUBMITTED optical observations retained. The
    /// denominator is the optical rows the caller submitted (after the mode
    /// partition), so a row dropped at observer construction — an
    /// unresolvable site, an unsupported mode — counts as not selected,
    /// just like an outlier the fit pruned.
    pub selection_fraction_value: f64,
    /// Minimum retained fraction the gate required.
    pub selection_fraction_threshold: f64,
    /// Do the SELECTED observations still span enough of the arc to
    /// extrapolate across it? This is the coverage axis
    /// [`extrapolation_acceptable`](Self::extrapolation_acceptable)
    /// gates on — a strict tightening of the full-span
    /// [`arc_coverage_ok`](Self::arc_coverage_ok).
    pub selected_arc_coverage_ok: bool,
    /// Arc span (days) over the selected observations only. NaN when
    /// nothing was selected.
    pub selected_arc_days_value: f64,
    /// Selected-span / full-span ratio.
    pub selected_arc_fraction_value: f64,
    /// Minimum span ratio the gate required.
    pub selected_arc_fraction_threshold: f64,
    /// Were the most-recent observations kept? The asymmetric backstop
    /// the span ratio cannot provide: it catches a short recent tail
    /// rejected off a long arc, where the ratio still passes but the
    /// discarded rows are the ones nearest a forward extrapolation
    /// target.
    pub trailing_gap_ok: bool,
    /// Days between the last selected and the last full-arc
    /// observation. `0.0` when the last kept observation is the last
    /// observation; NaN when nothing was selected.
    pub trailing_gap_days_value: f64,
    /// Largest trailing gap the gate allowed (days).
    pub trailing_gap_threshold: f64,
    /// Radar astrometry joint-fit acceptability. `None` when no radar
    /// contributed to the fit — which is never the same as `Some(false)`.
    pub radar_fit_ok: Option<bool>,
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
            selection_fraction_ok: r.selection_fraction_ok != 0,
            selection_fraction_value: r.selection_fraction_value,
            selection_fraction_threshold: r.selection_fraction_threshold,
            selected_arc_coverage_ok: r.selected_arc_coverage_ok != 0,
            selected_arc_days_value: r.selected_arc_days_value,
            selected_arc_fraction_value: r.selected_arc_fraction_value,
            selected_arc_fraction_threshold: r.selected_arc_fraction_threshold,
            trailing_gap_ok: r.trailing_gap_ok != 0,
            trailing_gap_days_value: r.trailing_gap_days_value,
            trailing_gap_threshold: r.trailing_gap_threshold,
            radar_fit_ok: match r.radar_fit_ok {
                1 => Some(true),
                0 => Some(false),
                _ => None,
            },
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

/// Full tagged solved-parameter covariance from a wide OD fit.
///
/// The `matrix` is the real solved covariance, sized `width × width`
/// (row-major); parameters are located by the slot fields, never by
/// width (width 9 is Marsden-only OR one thrust segment). Canonical
/// order is `[state 6 | Marsden 3 | DT 1 | AMRAT 1 | thrust 3×k]`. The Δv
/// axes are integration-frame components (see
/// [`DetermineResult::dv_frame`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SolvedCovariance {
    /// The solved covariance, sized `width × width` (row-major).
    pub matrix: Vec<Vec<f64>>,
    /// Solved width (6..=17 under the current engine).
    pub width: usize,
    /// Column of the first Marsden coefficient, when Marsden was solved.
    pub marsden_slot: Option<usize>,
    /// Column of the DT scalar, when DT was solved.
    pub dt_slot: Option<usize>,
    /// Column of the AMRAT scalar, when AMRAT was solved.
    pub amrat_slot: Option<usize>,
    /// Column triples of each fitted thrust Δv segment (one `[i,i+1,i+2]`
    /// per solved segment).
    pub thrust_slots: Vec<[usize; 3]>,
}

impl SolvedCovariance {
    pub(super) fn from_ffi(c: &empyrean_sys::EmpyreanSolvedCovariance) -> Self {
        let width = c.width as usize;
        let matrix = (0..width)
            .map(|i| (0..width).map(|j| c.matrix[i][j]).collect())
            .collect();
        let slot = |v: u32| (v != empyrean_sys::EMPYREAN_SLOT_NONE).then_some(v as usize);
        let thrust_slots = (0..c.thrust_count as usize)
            .map(|i| {
                let r = c.thrust_slots[i];
                [r[0] as usize, r[1] as usize, r[2] as usize]
            })
            .collect();
        Self {
            matrix,
            width,
            marsden_slot: slot(c.marsden_slot),
            dt_slot: slot(c.dt_slot),
            amrat_slot: slot(c.amrat_slot),
            thrust_slots,
        }
    }
}

/// Photometric model for the post-OD phase-function fit.
///
/// In [`Auto`](Self::Auto) the fit climbs a model ladder —
/// H-only → HG12 → HG1G2 — admitting the richest model the arc's
/// phase-angle coverage and magnitude count support, and reports the
/// one it actually fit on [`PhotometryResult::model_used`] (never
/// `Auto`). An explicit value pins a specific model. HG12 / HG1G2
/// follow Muinonen et al. (2010); H-only holds the slope fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhotometryModel {
    /// Auto-select up the ladder (H-only → HG12 → HG1G2) by data richness.
    #[default]
    Auto,
    /// Fit only the absolute magnitude H (fixed slope).
    HOnly,
    /// Two-parameter H, G.
    HG,
    /// Two-parameter H, G12 (Muinonen et al. 2010).
    HG12,
    /// Three-parameter H, G1, G2 (Muinonen et al. 2010).
    HG1G2,
}

impl PhotometryModel {
    pub(super) fn to_int(self) -> i32 {
        (match self {
            Self::Auto => empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_AUTO,
            Self::HOnly => empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HONLY,
            Self::HG => empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG,
            Self::HG12 => empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG12,
            Self::HG1G2 => empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG1G2,
        }) as i32
    }
    fn from_int(v: i32) -> Self {
        match v as u32 {
            empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HONLY => Self::HOnly,
            empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG => Self::HG,
            empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG12 => Self::HG12,
            empyrean_sys::EMPYREAN_PHOTOMETRY_MODEL_HG1G2 => Self::HG1G2,
            _ => Self::Auto,
        }
    }
}

/// Per-band photometric fit statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct BandStat {
    /// Photometric band tag.
    pub band: String,
    /// Number of observations in this band.
    pub n: usize,
    /// Band→V offset applied (mag).
    pub offset_applied: f64,
    /// Mean residual in V (mag).
    pub mean_residual: f64,
    /// RMS residual in V (mag).
    pub rms: f64,
}

impl BandStat {
    fn from_ffi(b: &empyrean_sys::EmpyreanBandStat) -> Self {
        let band = if b.band.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(b.band) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            band,
            n: b.n,
            offset_applied: b.offset_applied,
            mean_residual: b.mean_residual,
            rms: b.rms,
        }
    }
}

/// One model-ladder gate decision from the photometric fit.
#[derive(Debug, Clone, PartialEq)]
pub struct GateRecord {
    /// Model the gate evaluated.
    pub model: PhotometryModel,
    /// Whether the model was admitted.
    pub passed: bool,
    /// Human-readable gate reason.
    pub reason: String,
}

impl GateRecord {
    fn from_ffi(g: &empyrean_sys::EmpyreanGateRecord) -> Self {
        let reason = if g.reason.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(g.reason) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            model: PhotometryModel::from_int(g.model),
            passed: g.passed != 0,
            reason,
        }
    }
}

/// Post-OD photometric solution — an H/G fit over the arc's magnitudes,
/// run after the orbit is solved (photometry has no astrometric
/// partials, so it never touches the state). H carries honest σ via
/// [`covariance`](Self::covariance).
#[derive(Debug, Clone, PartialEq)]
pub struct PhotometryResult {
    /// Fitted absolute magnitude H (mag).
    pub h: f64,
    /// First slope parameter (G / G12 / G1 by model).
    pub slope1: f64,
    /// Second slope parameter (G2 for HG1G2; unused otherwise).
    pub slope2: f64,
    /// Parameter covariance (H, slope1, slope2), when available.
    pub covariance: Option<[[f64; 3]; 3]>,
    /// Model actually fitted (never [`PhotometryModel::Auto`]).
    pub model_used: PhotometryModel,
    /// Reduced χ² of the photometric fit over its used magnitudes.
    pub reduced_chi2: f64,
    /// Whether a simplex constraint was active on the fitted slopes.
    pub constraint_active: bool,
    /// Magnitudes used in the fit.
    pub n_mags_used: usize,
    /// Magnitudes rejected by the photometric outlier pass.
    pub n_mags_rejected_photometric: usize,
    /// Observations carrying no magnitude.
    pub n_obs_without_mags: usize,
    /// Magnitudes drawn from astrometrically-selected observations.
    pub n_mags_from_astrometric_selected: usize,
    /// Magnitudes drawn from astrometrically-rejected observations.
    pub n_mags_from_astrometric_rejected: usize,
    /// Minimum phase angle of the fitted magnitudes (deg).
    pub alpha_min_deg: f64,
    /// Maximum phase angle of the fitted magnitudes (deg).
    pub alpha_max_deg: f64,
    /// Phase-angle span of the fitted magnitudes (deg).
    pub alpha_span_deg: f64,
    /// Per-band statistics.
    pub per_band: Vec<BandStat>,
    /// Model-ladder gate records.
    pub gates: Vec<GateRecord>,
    /// Magnitudes excluded from the fit because their photometric band
    /// has no adopted V-band conversion (unknown/unspecified band
    /// codes, comet total/nuclear magnitudes). The observations'
    /// astrometry is unaffected.
    pub n_mags_dropped_unconvertible: usize,
    /// Distinct band codes that were dropped, sorted.
    pub dropped_bands: Vec<String>,
}

impl PhotometryResult {
    pub(super) fn from_ffi(p: &empyrean_sys::EmpyreanODPhotometryResult) -> Self {
        let covariance = (p.has_covariance != 0).then_some(p.covariance);
        let per_band = if p.per_band.is_null() || p.num_per_band == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(p.per_band, p.num_per_band)
                    .iter()
                    .map(BandStat::from_ffi)
                    .collect()
            }
        };
        let gates = if p.gates.is_null() || p.num_gates == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(p.gates, p.num_gates)
                    .iter()
                    .map(GateRecord::from_ffi)
                    .collect()
            }
        };
        Self {
            h: p.h,
            slope1: p.slope1,
            slope2: p.slope2,
            covariance,
            model_used: PhotometryModel::from_int(p.model_used),
            reduced_chi2: p.reduced_chi2,
            constraint_active: p.constraint_active != 0,
            n_mags_used: p.n_mags_used,
            n_mags_rejected_photometric: p.n_mags_rejected_photometric,
            n_obs_without_mags: p.n_obs_without_mags,
            n_mags_from_astrometric_selected: p.n_mags_from_astrometric_selected,
            n_mags_from_astrometric_rejected: p.n_mags_from_astrometric_rejected,
            alpha_min_deg: p.alpha_min_deg,
            alpha_max_deg: p.alpha_max_deg,
            alpha_span_deg: p.alpha_span_deg,
            per_band,
            gates,
            n_mags_dropped_unconvertible: p.n_mags_dropped_unconvertible,
            dropped_bands: if p.dropped_bands.is_null() || p.num_dropped_bands == 0 {
                Vec::new()
            } else {
                unsafe {
                    std::slice::from_raw_parts(p.dropped_bands, p.num_dropped_bands)
                        .iter()
                        .map(|&b| {
                            if b.is_null() {
                                String::new()
                            } else {
                                CStr::from_ptr(b).to_string_lossy().into_owned()
                            }
                        })
                        .collect()
                }
            },
        }
    }
}

/// The intervening event named by a
/// [`CovarianceTrust::EncounterIntervenes`] verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum TrustGateEvent {
    /// A close approach to a massive body lies inside the covariance
    /// validity window.
    CloseApproach {
        /// Name of the body approached (e.g. `"Earth"`).
        body: String,
        /// Epoch of the close-approach signal.
        epoch: crate::Epoch,
        /// Approach distance at the signal (AU).
        distance_au: f64,
    },
    /// The nonlinearity ratio exceeded its threshold — the second
    /// moment is no longer well-described by the linear map.
    HighNonlinearity {
        /// Epoch of the threshold crossing.
        epoch: crate::Epoch,
        /// The nonlinearity ratio at the crossing.
        nonlinearity: f64,
        /// The threshold it exceeded.
        threshold: f64,
    },
}

/// Event-aware trust verdict on the delivered covariance, evaluated
/// over its validity window on the converged orbit. Absence
/// (`DetermineResult::covariance_trust == None`) means the call path
/// ran no gate — absence of a verdict is not trust.
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceTrust {
    /// No intervening close approach and a 6-state solve — the linear
    /// covariance may be used as delivered.
    Trusted,
    /// A close approach (or high-nonlinearity crossing) lies inside the
    /// window; do not extrapolate the linear covariance across it —
    /// escalate to nonlinear uncertainty propagation (second-order when
    /// `second_order_recoverable`, otherwise sampling).
    EncounterIntervenes {
        /// The earliest intervening event in the window.
        event: TrustGateEvent,
        /// The fit's solved-for width.
        solved_width: u32,
        /// Whether a second-order state-only correction can recover the
        /// encounter (solved width 6).
        second_order_recoverable: bool,
    },
    /// The fit solved more than the 6-state, so the delivered 6×6 is a
    /// marginal of a wider fit — a conservative flag.
    WeaklyDeterminedHighN {
        /// The fit's solved-for width (> 6).
        solved_width: u32,
    },
}

impl CovarianceTrust {
    /// Decode the C-ABI trust field group. `None` ⇔ `NOT_EVALUATED`.
    pub(super) fn from_ffi(r: &empyrean_sys::EmpyreanODResult) -> Option<Self> {
        match r.covariance_trust {
            empyrean_sys::EMPYREAN_COVARIANCE_TRUST_TRUSTED => Some(CovarianceTrust::Trusted),
            empyrean_sys::EMPYREAN_COVARIANCE_TRUST_ENCOUNTER_INTERVENES => {
                let event =
                    if r.trust_event_kind == empyrean_sys::EMPYREAN_TRUST_EVENT_HIGH_NONLINEARITY {
                        TrustGateEvent::HighNonlinearity {
                            epoch: crate::Epoch::from_mjd_tdb(r.trust_event_epoch_mjd_tdb),
                            nonlinearity: r.trust_event_nonlinearity,
                            threshold: r.trust_event_threshold,
                        }
                    } else {
                        TrustGateEvent::CloseApproach {
                            body: if r.trust_event_body.is_null() {
                                String::new()
                            } else {
                                unsafe { CStr::from_ptr(r.trust_event_body) }
                                    .to_string_lossy()
                                    .into_owned()
                            },
                            epoch: crate::Epoch::from_mjd_tdb(r.trust_event_epoch_mjd_tdb),
                            distance_au: r.trust_event_distance_au,
                        }
                    };
                Some(CovarianceTrust::EncounterIntervenes {
                    event,
                    solved_width: r.trust_solved_width,
                    second_order_recoverable: r.trust_second_order_recoverable != 0,
                })
            }
            empyrean_sys::EMPYREAN_COVARIANCE_TRUST_WEAKLY_DETERMINED_HIGH_N => {
                Some(CovarianceTrust::WeaklyDeterminedHighN {
                    solved_width: r.trust_solved_width,
                })
            }
            _ => None,
        }
    }
}

/// Which criterion ended the solve that produced a fit's published
/// state.
///
/// Read this rather than inferring convergence from
/// [`DetermineResult::update_norm`], which is the μ-damped accepted step
/// and is not comparable to `ODConfig::convergence_tol` in either
/// direction.
///
/// # Fieldless, deliberately
///
/// The engine's own enum carries a payload on two arms — the damping
/// value that exceeded the cap, and the trial count of the exhausted
/// iteration. Neither crosses the C ABI as part of the stop code: μ at
/// termination is published for every path as
/// [`DetermineResult::mu_final`], which is where a
/// [`DampingExhausted`](Self::DampingExhausted) fit's μ is read, and the
/// trial count has no ABI field. Carrying half a payload here would
/// invite matching on a value that is only sometimes present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStop {
    /// The scaled gradient at the accepted point fell below `gtol`
    /// (MINPACK cosine form). Tested before the step test, so it says
    /// nothing about how large the remaining Gauss-Newton step was.
    GradientTolerance,
    /// The **undamped** Gauss-Newton step's quadratic form fell below
    /// [`ODConfig::convergence_tol`](crate::ODConfig::convergence_tol).
    /// The only stop on the ordinary LM path under which
    /// [`DetermineResult::gn_step_qnorm`] is bounded by that tolerance.
    StepTolerance,
    /// Actual and predicted reduction of the last accepted step both
    /// fell below `ftol`·Φ under a consistent model.
    CostTolerance,
    /// The outer iteration budget ran out while the solve was still
    /// moving. **Not** a convergence.
    MaxIterations,
    /// Damping exceeded the driver's cap while rejecting trials: no
    /// improving step exists from that point at any μ. Not a
    /// convergence — but see [`StalledDelivered`](Self::StalledDelivered),
    /// the verdict when such a point is nonetheless statistically
    /// stationary. The μ that exceeded the cap is
    /// [`DetermineResult::mu_final`].
    DampingExhausted,
    /// The inner trial budget ran out inside one outer iteration without
    /// an acceptance. Not a convergence; same caveat as
    /// [`DampingExhausted`](Self::DampingExhausted).
    InnerTrialsExhausted,
    /// No solver criterion latched, and the iterate was delivered anyway
    /// by the stable-stall acceptance.
    /// [`DetermineResult::stall_delivery`] carries the numbers that
    /// earned it, including the solver's own verdict underneath.
    StalledDelivered,
    /// The Schur (nuisance-block) loop's own step test, under the Schur
    /// complement rather than the prior-augmented normal matrix.
    /// Deliberately not [`StepTolerance`](Self::StepTolerance): they are
    /// quantities of different systems.
    /// [`DetermineResult::gn_step_qnorm`] is published on this path only
    /// at solved width ≤ 6, where the loop runs undamped.
    SchurStepTolerance,
    /// The solver reported a termination this engine build cannot name —
    /// the optimizer grew a variant ahead of the mapping.
    ///
    /// Never folded into one of the arms above, and never reported as
    /// absent: a stop that fired but cannot be named is not the same as
    /// no stop at all.
    Unrecognized,
}

impl SolverStop {
    /// Decode the C-ABI `EMPYREAN_SOLVER_STOP_*` code. `None` ⇔
    /// `NOT_REPORTED` — the result did not come from a solver run.
    ///
    /// A code this build does not know maps to
    /// [`Unrecognized`](Self::Unrecognized) rather than to `None`, for
    /// the same reason the engine has that variant: an unnamed stop is
    /// not a missing one.
    pub(super) fn from_int(code: i32) -> Option<Self> {
        match code {
            empyrean_sys::EMPYREAN_SOLVER_STOP_NOT_REPORTED => None,
            empyrean_sys::EMPYREAN_SOLVER_STOP_GRADIENT_TOLERANCE => {
                Some(SolverStop::GradientTolerance)
            }
            empyrean_sys::EMPYREAN_SOLVER_STOP_STEP_TOLERANCE => Some(SolverStop::StepTolerance),
            empyrean_sys::EMPYREAN_SOLVER_STOP_COST_TOLERANCE => Some(SolverStop::CostTolerance),
            empyrean_sys::EMPYREAN_SOLVER_STOP_MAX_ITERATIONS => Some(SolverStop::MaxIterations),
            empyrean_sys::EMPYREAN_SOLVER_STOP_DAMPING_EXHAUSTED => {
                Some(SolverStop::DampingExhausted)
            }
            empyrean_sys::EMPYREAN_SOLVER_STOP_INNER_TRIALS_EXHAUSTED => {
                Some(SolverStop::InnerTrialsExhausted)
            }
            empyrean_sys::EMPYREAN_SOLVER_STOP_STALLED_DELIVERED => {
                Some(SolverStop::StalledDelivered)
            }
            empyrean_sys::EMPYREAN_SOLVER_STOP_SCHUR_STEP_TOLERANCE => {
                Some(SolverStop::SchurStepTolerance)
            }
            _ => Some(SolverStop::Unrecognized),
        }
    }

    /// Stable lowercase tag, matching the engine's own serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            SolverStop::GradientTolerance => "gradient_tolerance",
            SolverStop::StepTolerance => "step_tolerance",
            SolverStop::CostTolerance => "cost_tolerance",
            SolverStop::MaxIterations => "max_iterations",
            SolverStop::DampingExhausted => "damping_exhausted",
            SolverStop::InnerTrialsExhausted => "inner_trials_exhausted",
            SolverStop::StalledDelivered => "stalled_delivered",
            SolverStop::SchurStepTolerance => "schur_step_tolerance",
            SolverStop::Unrecognized => "unrecognized",
        }
    }

    /// Whether this stop is one of the solver's own convergence criteria
    /// — the solve ended because it arrived, not because it ran out of
    /// budget or damping.
    ///
    /// [`StalledDelivered`](Self::StalledDelivered) is deliberately
    /// `false`: the acceptance delivers such a fit, but no criterion
    /// latched, and a consumer asking this question is asking about the
    /// solver.
    pub fn is_solver_criterion(&self) -> bool {
        matches!(
            self,
            SolverStop::GradientTolerance
                | SolverStop::StepTolerance
                | SolverStop::CostTolerance
                | SolverStop::SchurStepTolerance
        )
    }
}

/// The numbers behind a fit delivered by the stable-stall acceptance —
/// one whose final solve latched no convergence criterion but was
/// stationary to a small fraction of its own formal σ with χ² clearing
/// the fit-quality bars.
///
/// Present as [`DetermineResult::stall_delivery`] exactly when
/// [`DetermineResult::termination`] is
/// [`SolverStop::StalledDelivered`]. `None` is the ordinary case; a
/// consumer that wants the strict convergence contract checks for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StallDelivery {
    /// What the SOLVER reported before the acceptance overrode the
    /// verdict — always [`SolverStop::DampingExhausted`] or
    /// [`SolverStop::InnerTrialsExhausted`], the only two stops the
    /// acceptance considers. Kept beside
    /// [`DetermineResult::termination`] so the delivery's verdict and
    /// the solver's own are both readable and neither is inferred from
    /// the other.
    pub underlying_stop: SolverStop,
    /// \\(q\\) of the undamped Gauss-Newton step at the delivered
    /// iterate — the quantity `convergence_tol` bounds and did not bound
    /// here. \\(\sqrt{q}\\) is the remaining step in the fit's own
    /// formal σ.
    pub gn_qnorm: f64,
    /// The **resolved** convergence tolerance the stall was judged
    /// against, delivered because `ODConfig::convergence_tol` carries a
    /// "use the engine default" sentinel and the enforced value is
    /// therefore not recoverable from the config that was passed in.
    pub convergence_tol: f64,
    /// Reduced χ² of the OPTICAL family alone at the delivered iterate.
    ///
    /// The full objective's reduced χ² and the bar both were held to are
    /// already on [`DetermineResult::acceptability`], and the delivery's
    /// μ and iteration count are already
    /// [`DetermineResult::mu_final`] and
    /// [`DetermineResult::final_solve_iterations`] — a stall-delivered
    /// fit's final solve IS the stalled solve.
    pub optical_reduced_chi2: f64,
}

impl StallDelivery {
    /// The remaining undamped Gauss-Newton step in units of the fit's
    /// own formal 1σ, \\(\sqrt{q}\\) — the number to quote when saying
    /// how far from stationary the delivered iterate is.
    pub fn step_sigmas(&self) -> f64 {
        self.gn_qnorm.sqrt()
    }

    /// Decode the C-ABI `stall_*` field group. `None` when the fit was
    /// not stall-delivered.
    pub(super) fn from_ffi(r: &empyrean_sys::EmpyreanODResult) -> Option<Self> {
        (r.stall_delivered == 1).then(|| StallDelivery {
            underlying_stop: SolverStop::from_int(r.stall_underlying_stop)
                .unwrap_or(SolverStop::Unrecognized),
            gn_qnorm: r.stall_gn_qnorm,
            convergence_tol: r.stall_convergence_tol,
            optical_reduced_chi2: r.stall_optical_reduced_chi2,
        })
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
    /// **NOT the quantity
    /// [`ODConfig::convergence_tol`](crate::ODConfig::convergence_tol)
    /// bounds.** The μ-DAMPED last ACCEPTED step's \\(q\\)-norm
    /// (Δx^T N Δx) — the size of the step the solver actually took,
    /// after Levenberg-Marquardt damping.
    ///
    /// The tolerance is tested on the *undamped* Gauss-Newton step,
    /// published as [`gn_step_qnorm`](Self::gn_step_qnorm). The two are
    /// incomparable in BOTH directions: a converged fit routinely
    /// reports an `update_norm` orders of magnitude ABOVE
    /// `convergence_tol` (damping is light near the optimum, so the step
    /// taken is large), while a solve thrashing to
    /// [`SolverStop::DampingExhausted`] reports one orders of magnitude
    /// BELOW it (μ has crushed every step to nothing). Read
    /// [`termination`](Self::termination) and
    /// [`gn_step_qnorm`](Self::gn_step_qnorm) for convergence, or
    /// [`acceptability`](Self::acceptability) for the fit-quality
    /// verdict.
    ///
    /// On the Schur path (station-bias / nuisance fits) this carries
    /// that loop's own step norm under the Schur complement instead —
    /// the quantity IT tests against `convergence_tol`.
    /// [`termination`](Self::termination) says which loop ran.
    ///
    /// Exactly `0.0` with [`accepted_steps`](Self::accepted_steps)` == 0`
    /// means the solver latched a criterion at its STARTING point and
    /// never accepted a step — the incoming iterate was already
    /// stationary. It does not mean "a step of size zero was taken".
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
    /// The outlier-rejection gate this fit was configured with — named
    /// whether or not it ran (read it beside
    /// [`rejection_enabled`](Self::rejection_enabled)).
    /// [`DeliveredRejectionKind::Unknown`] if the loaded library is newer
    /// than this wrapper.
    pub rejection_kind: DeliveredRejectionKind,
    /// Whether outlier rejection actually ran this fit — the engine's
    /// resolution of the [`RejectionConfig::enabled`](super::rejection::RejectionConfig::enabled)
    /// tri-state. When `false`, [`rejection_kind`](Self::rejection_kind)
    /// names a gate that never ran.
    pub rejection_enabled: bool,
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
    /// Full tagged solved-parameter covariance when the fit solved any
    /// wide axis (Marsden / DT / AMRAT / thrust). `None` for a state-only
    /// fit. The go-forward covariance for all solved widths; the legacy
    /// [`covariance_9x9`](Self::covariance_9x9) is kept for one window.
    pub solved_covariance: Option<SolvedCovariance>,
    /// Cumulative non-grav time-delay correction ΔDT (days), when DT was solved.
    pub dt_delta: Option<f64>,
    /// Cumulative SRP AMRAT correction (m²/kg), when AMRAT was solved.
    pub amrat_delta: Option<f64>,
    /// Per-segment fitted thrust Δv (m/s), expressed in
    /// [`dv_frame`](Self::dv_frame), indexed by **declared** segment.
    ///
    /// `None` at a segment this fit did not solve — that burn has no
    /// correction, and a zero there would read as a fitted Δv of exactly
    /// zero. Shares one index space with
    /// [`thrust_correction_covariances`](Self::thrust_correction_covariances)
    /// and [`dispositions`](Self::dispositions)`.thrust`.
    pub thrust_delta_m_per_s: Vec<Option<[f64; 3]>>,
    /// Per-segment fitted Δv correction covariances (AU/day)², indexed
    /// by **declared** segment.
    ///
    /// `None` at a segment this fit did not solve: a considered or fixed
    /// burn has no posterior, and echoing its prior here would republish
    /// a prior under a posterior's name. Read
    /// [`dispositions`](Self::dispositions)`.thrust[i]` before the
    /// block.
    pub thrust_correction_covariances: Vec<Option<[[f64; 3]; 3]>>,
    /// Integration frame the thrust Δv components are expressed in.
    /// `None` when no thrust was solved.
    pub dv_frame: Option<crate::coordinate::Frame>,
    /// Post-OD photometric solution when photometry was requested and ran.
    pub photometry: Option<PhotometryResult>,
    /// Event-aware trust verdict on the delivered covariance. `None`
    /// when the call path ran no trust gate — absence of a verdict is
    /// not trust.
    pub covariance_trust: Option<CovarianceTrust>,
    /// What the fit did with each parameter axis the orbit declared.
    ///
    /// Without it a covariance is ambiguous. An axis the fit
    /// **considered** already has its uncertainty inside the delivered
    /// 6×6, so re-attaching a prior to it double-counts; an axis held
    /// **fixed** contributed nothing, so attaching a prior is
    /// conservative and correct. Same covariance, opposite conclusions.
    ///
    /// Reports the request as resolved against the orbit, so under
    /// [`SolveForParams::Auto`] it names the width the fit actually ran
    /// at rather than the width requested.
    pub dispositions: SolveFor,
    /// Non-fatal conditions the fit reports about itself — chiefly
    /// supplied covariance it deliberately did not use.
    ///
    /// Empty on a fit that used everything it was given, which is the
    /// common case. These are delivered payload rather than log lines: a
    /// dropped prior cross term changes how the σ for that slot should
    /// be read.
    pub warnings: Vec<String>,
    /// Which criterion ended the solve that produced the published
    /// state. `None` only on a result that did not come from a solver
    /// run; every fit this crate delivers names its stop.
    pub termination: Option<SolverStop>,
    /// Quadratic form of the **undamped** Gauss-Newton step — the only
    /// number here comparable to
    /// [`ODConfig::convergence_tol`](crate::ODConfig::convergence_tol),
    /// and the one the solver's step test is decided on.
    ///
    /// \\(q = \mathbf{h}_{\text{GN}}^\top A\,\mathbf{h}_{\text{GN}}\\)
    /// with \\(A = \Sigma^{-1}\\), so \\(\sqrt{q}\\) is the remaining
    /// step measured in the fit's own formal σ.
    ///
    /// `q <= convergence_tol` is guaranteed only under
    /// [`SolverStop::StepTolerance`] (or
    /// [`SolverStop::SchurStepTolerance`]). Under the gradient, cost and
    /// stall verdicts the fit is delivered as converged with `q`
    /// legitimately above the tolerance.
    ///
    /// `None` when the undamped system was singular at the returned
    /// point, when the Schur loop ran at solved width > 6 (its only step
    /// norm there is λ-shrunken, which is what this field promises it is
    /// not), or on a non-solver result.
    pub gn_step_qnorm: Option<f64>,
    /// Levenberg-Marquardt damping at termination. Large μ beside a
    /// small [`update_norm`](Self::update_norm) is the signature of a
    /// stalled solve rather than a converged one.
    ///
    /// `None` on the Schur path, which runs its own λ schedule and forms
    /// no comparable quantity — so its absence, alongside
    /// [`SolverStop::SchurStepTolerance`], is what identifies that path
    /// — and on non-solver results.
    pub mu_final: Option<f64>,
    /// Steps the solve actually ACCEPTED — trials that passed the
    /// gain-ratio test and moved the iterate.
    ///
    /// `0` says the solver latched a criterion at its starting point and
    /// never moved: the incoming orbit was already stationary for this
    /// observation set. That is what disambiguates
    /// [`update_norm`](Self::update_norm)'s `0.0` sentinel.
    ///
    /// Counted per SOLVE, so it describes the same solve as
    /// [`final_solve_iterations`](Self::final_solve_iterations), never a
    /// total across the rejection–refit passes. The Schur path applies
    /// every step it computes rather than accepting some, so there this
    /// is applied steps and equals `final_solve_iterations`.
    pub accepted_steps: u32,
    /// Iterations spent by the solve that produced the published state —
    /// the one [`update_norm`](Self::update_norm),
    /// [`termination`](Self::termination),
    /// [`gn_step_qnorm`](Self::gn_step_qnorm) and
    /// [`accepted_steps`](Self::accepted_steps) all describe.
    ///
    /// Additive disambiguator for [`iterations`](Self::iterations),
    /// whose composition varies by entry point; this one never does.
    /// `None` on a result that did not come from a solver run.
    pub final_solve_iterations: Option<usize>,
    /// `Some` when this fit was delivered by the stable-stall acceptance
    /// rather than by a convergence criterion — see [`StallDelivery`]
    /// for the numbers that earned it.
    ///
    /// `None` is the ordinary case. A consumer that wants the strict
    /// convergence contract checks `stall_delivery.is_none()`.
    pub stall_delivery: Option<StallDelivery>,
    /// The kind of covariance this fit actually delivered. Always
    /// [`CovarianceKind::Linear`](crate::CovarianceKind::Linear): the
    /// differential-correction fit is first-order by construction, and a
    /// [`UncertaintyMethod::auto`](crate::UncertaintyMethod::auto) request
    /// on the config resolves to it — a resolution also named on
    /// [`warnings`](Self::warnings). Every richer method is refused by name
    /// before a fit is produced, so this never reads anything but the
    /// linear kind on a delivered result. The OD analogue of
    /// [`PlanResult::resolved_method`](crate::PlanResult::resolved_method):
    /// one vocabulary — the covariance-kind vocabulary — for what kind of
    /// covariance a result carries, while the requested method stays
    /// visible on the config's `uncertainty_method` knob.
    pub resolved_method: crate::CovarianceKind,
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
            // The fit's own packed joint, from the same orbit this snapshot
            // flattens — so the flat view and the re-feedable orbit describe
            // one joint rather than two.
            joint: self.orbit.covariance.clone(),
            stm: None,
            stt: None,
            resolved_kind: CovarianceKind::Linear,
            // An OD fit's covariance is linear, never a mixture.
            mixture_tally: None,
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

/// The specific differential-correction error behind a
/// [`DetermineFailureKind::OD`] failure.
///
/// One variant per engine `ODError`, names identical to the engine's, so a
/// caller can branch on the exact cause instead of matching the message
/// string. Codes only: variant payloads the engine carries (a count, a
/// threshold, an observatory code) are not reproduced here — read
/// [`DetermineFailure::message`] for them. A code newer than this build
/// decodes to [`Unknown`](Self::Unknown) keeping the raw value, never a
/// valid-looking variant (the wrapper and `libempyrean` are separately
/// versioned).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ODError {
    /// Initial orbit is not in Cartesian representation.
    NotCartesian,
    /// Initial orbit carries no covariance, required for Jet propagation.
    NoCovarianceOnOrbit,
    /// The requested uncertainty method has no honest implementation on
    /// this entry point; refused, not silently downgraded.
    UnsupportedUncertaintyMethod,
    /// The orbit collection did not contain exactly one orbit.
    NotSingleOrbit,
    /// Ephemeris generation failed during the fit.
    EphemerisGeneration,
    /// The normal matrix is singular — no state correction can be solved.
    SingularNormalMatrix,
    /// No observations carried valid Jacobians / STMs to fit against.
    NoValidObservations,
    /// A batch orbit's map key matched no observation group.
    NoObservationsForOrbit,
    /// The solve did not reach a deliverable iterate.
    NotConverged,
    /// Every rung of a retry ladder failed.
    RetryLadderExhausted,
    /// Consecutive non-finite trial costs with no underlying domain error.
    NonFiniteTrialCosts,
    /// A full evaluation at the initial point was non-finite or negative.
    NonFiniteEvaluation,
    /// Nearest-perihelion output epoch requested for a non-heliocentric fit.
    PerihelionEpochRequiresHeliocentric,
    /// Nearest-perihelion resolution produced a non-finite time (near-parabolic).
    PerihelionEpochNonFinite,
    /// The solver reported a contract violation (a bug in the problem mapping).
    SolverContractViolation,
    /// A user-settable solver knob holds an invalid value.
    InvalidSolverConfig,
    /// A solver knob was set that names a code path this engine no longer has.
    RetiredSolverKnob,
    /// The Bayesian prior could not be built from the orbit's covariance.
    PriorConstruction,
    /// A thrust Δv solve was requested but the orbit declares no burn arcs.
    ThrustParamsMissing,
    /// A thrust request reaches past the orbit's declared correction covariances.
    ThrustPriorMismatch,
    /// An axis is considered but the fit runs without an orbit prior.
    ConsiderPriorUnavailable,
    /// A solved thrust segment's burn window lies off the propagation path.
    ManeuverNotBracketed,
    /// A DT (thermal-lag) solve was requested but the orbit carries no DT value.
    NonGravDTMissing,
    /// A DT solve was requested but its variance is absent.
    DTPriorMissing,
    /// A solved covariance carried a width the representation transform does
    /// not support (a version skew).
    UnsupportedSolvedWidth,
    /// An AMRAT solve was requested but the orbit carries no SRP force.
    SRPParamsMissing,
    /// An AMRAT solve was requested but its variance is absent.
    AMRATPriorMissing,
    /// A converged AMRAT solve delivered a non-physical area-to-mass ratio.
    AMRATNotPhysical,
    /// Multi-opposition arc truncation was needed but disabled.
    ArcTruncationDisabled,
    /// An observation's reported σ is non-finite or non-positive.
    InvalidObservationUncertainty,
    /// Observation conversion failed (time parsing, malformed record).
    ObservationConversion,
    /// Observer construction failed (unknown obs code, epoch out of range).
    ObserverConstruction,
    /// The loaded Earth-orientation (BPC) coverage does not span the observations.
    EarthOrientationCoverageIncomplete,
    /// IOD returned a physically pathological orbit.
    PathologicalIOD,
    /// Propagation asked for a body position outside the loaded SPK coverage.
    PropagationOutOfCoverage,
    /// The requested output representation is unsupported for a fitted orbit.
    UnsupportedOutputRepresentation,
    /// Transforming the fit into the requested representation failed (degenerate).
    DegenerateRepresentation,
    /// The explicitly requested post-OD photometric fit failed.
    PhotometricFit,
    /// A non-Gaussian noise model was configured with outlier rejection left
    /// at the engine default — refused rather than a trimmed-t fit by accident.
    NoiseModelRejectionUnset,
    /// A fit was configured with the night-scale law, refused by name in v1.
    NightScaleNotYetValidated,
    /// A non-Gaussian noise model was combined with a fit feature its v1 IRLS
    /// engine does not yet wire.
    NoiseModelCompositionUnsupported,
    /// The Student-t observed information is not positive definite, so it has
    /// no honest covariance.
    ObservedInformationNotPositiveDefinite,
    /// The IRLS outer loop reached its cap without the weights and step settling.
    NoiseModelIrlsNotConverged,
    /// An observation named an observatory code in no loaded registry and
    /// matching no station table — refused by name, never silently skipped
    /// or placed at a fallback site. The offending code and its observation
    /// index are in [`DetermineFailure::message`].
    UnknownObservatoryCode,
    /// A weighting configuration was refused at construction: a σ or scale
    /// is non-finite, zero or negative. The offending layer, field and
    /// value are in [`DetermineFailure::message`].
    InvalidWeightingConfig,
    /// An observation's assembled 2×2 weight matrix was non-finite — a
    /// weighting defect refused before the solve's factorization.
    NonFiniteWeightMatrix,
    /// An observation's assembled 2×2 weight matrix was finite but not
    /// positive-definite, so it has no Cholesky factor.
    WeightMatrixNotPositiveDefinite,
    /// A code this build does not model — the raw wire value, kept verbatim.
    /// Its appearance means the loaded library is newer than the wrapper.
    Unknown(i32),
}

impl ODError {
    pub(super) fn from_code(code: i32) -> Self {
        match code as u32 {
            empyrean_sys::EMPYREAN_OD_ERROR_NOT_CARTESIAN => ODError::NotCartesian,
            empyrean_sys::EMPYREAN_OD_ERROR_NO_COVARIANCE_ON_ORBIT => ODError::NoCovarianceOnOrbit,
            empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_UNCERTAINTY_METHOD => {
                ODError::UnsupportedUncertaintyMethod
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NOT_SINGLE_ORBIT => ODError::NotSingleOrbit,
            empyrean_sys::EMPYREAN_OD_ERROR_EPHEMERIS_GENERATION => ODError::EphemerisGeneration,
            empyrean_sys::EMPYREAN_OD_ERROR_SINGULAR_NORMAL_MATRIX => ODError::SingularNormalMatrix,
            empyrean_sys::EMPYREAN_OD_ERROR_NO_VALID_OBSERVATIONS => ODError::NoValidObservations,
            empyrean_sys::EMPYREAN_OD_ERROR_NO_OBSERVATIONS_FOR_ORBIT => {
                ODError::NoObservationsForOrbit
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NOT_CONVERGED => ODError::NotConverged,
            empyrean_sys::EMPYREAN_OD_ERROR_RETRY_LADDER_EXHAUSTED => ODError::RetryLadderExhausted,
            empyrean_sys::EMPYREAN_OD_ERROR_NON_FINITE_TRIAL_COSTS => ODError::NonFiniteTrialCosts,
            empyrean_sys::EMPYREAN_OD_ERROR_NON_FINITE_EVALUATION => ODError::NonFiniteEvaluation,
            empyrean_sys::EMPYREAN_OD_ERROR_PERIHELION_EPOCH_REQUIRES_HELIOCENTRIC => {
                ODError::PerihelionEpochRequiresHeliocentric
            }
            empyrean_sys::EMPYREAN_OD_ERROR_PERIHELION_EPOCH_NON_FINITE => {
                ODError::PerihelionEpochNonFinite
            }
            empyrean_sys::EMPYREAN_OD_ERROR_SOLVER_CONTRACT_VIOLATION => {
                ODError::SolverContractViolation
            }
            empyrean_sys::EMPYREAN_OD_ERROR_INVALID_SOLVER_CONFIG => ODError::InvalidSolverConfig,
            empyrean_sys::EMPYREAN_OD_ERROR_RETIRED_SOLVER_KNOB => ODError::RetiredSolverKnob,
            empyrean_sys::EMPYREAN_OD_ERROR_PRIOR_CONSTRUCTION => ODError::PriorConstruction,
            empyrean_sys::EMPYREAN_OD_ERROR_THRUST_PARAMS_MISSING => ODError::ThrustParamsMissing,
            empyrean_sys::EMPYREAN_OD_ERROR_THRUST_PRIOR_MISMATCH => ODError::ThrustPriorMismatch,
            empyrean_sys::EMPYREAN_OD_ERROR_CONSIDER_PRIOR_UNAVAILABLE => {
                ODError::ConsiderPriorUnavailable
            }
            empyrean_sys::EMPYREAN_OD_ERROR_MANEUVER_NOT_BRACKETED => ODError::ManeuverNotBracketed,
            empyrean_sys::EMPYREAN_OD_ERROR_NON_GRAV_DT_MISSING => ODError::NonGravDTMissing,
            empyrean_sys::EMPYREAN_OD_ERROR_DT_PRIOR_MISSING => ODError::DTPriorMissing,
            empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_SOLVED_WIDTH => {
                ODError::UnsupportedSolvedWidth
            }
            empyrean_sys::EMPYREAN_OD_ERROR_SRP_PARAMS_MISSING => ODError::SRPParamsMissing,
            empyrean_sys::EMPYREAN_OD_ERROR_AMRAT_PRIOR_MISSING => ODError::AMRATPriorMissing,
            empyrean_sys::EMPYREAN_OD_ERROR_AMRAT_NOT_PHYSICAL => ODError::AMRATNotPhysical,
            empyrean_sys::EMPYREAN_OD_ERROR_ARC_TRUNCATION_DISABLED => {
                ODError::ArcTruncationDisabled
            }
            empyrean_sys::EMPYREAN_OD_ERROR_INVALID_OBSERVATION_UNCERTAINTY => {
                ODError::InvalidObservationUncertainty
            }
            empyrean_sys::EMPYREAN_OD_ERROR_OBSERVATION_CONVERSION => {
                ODError::ObservationConversion
            }
            empyrean_sys::EMPYREAN_OD_ERROR_OBSERVER_CONSTRUCTION => ODError::ObserverConstruction,
            empyrean_sys::EMPYREAN_OD_ERROR_EARTH_ORIENTATION_COVERAGE_INCOMPLETE => {
                ODError::EarthOrientationCoverageIncomplete
            }
            empyrean_sys::EMPYREAN_OD_ERROR_PATHOLOGICAL_IOD => ODError::PathologicalIOD,
            empyrean_sys::EMPYREAN_OD_ERROR_PROPAGATION_OUT_OF_COVERAGE => {
                ODError::PropagationOutOfCoverage
            }
            empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_OUTPUT_REPRESENTATION => {
                ODError::UnsupportedOutputRepresentation
            }
            empyrean_sys::EMPYREAN_OD_ERROR_DEGENERATE_REPRESENTATION => {
                ODError::DegenerateRepresentation
            }
            empyrean_sys::EMPYREAN_OD_ERROR_PHOTOMETRIC_FIT => ODError::PhotometricFit,
            empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_REJECTION_UNSET => {
                ODError::NoiseModelRejectionUnset
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NIGHT_SCALE_NOT_YET_VALIDATED => {
                ODError::NightScaleNotYetValidated
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_COMPOSITION_UNSUPPORTED => {
                ODError::NoiseModelCompositionUnsupported
            }
            empyrean_sys::EMPYREAN_OD_ERROR_OBSERVED_INFORMATION_NOT_POSITIVE_DEFINITE => {
                ODError::ObservedInformationNotPositiveDefinite
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_IRLS_NOT_CONVERGED => {
                ODError::NoiseModelIrlsNotConverged
            }
            empyrean_sys::EMPYREAN_OD_ERROR_UNKNOWN_OBSERVATORY_CODE => {
                ODError::UnknownObservatoryCode
            }
            empyrean_sys::EMPYREAN_OD_ERROR_INVALID_WEIGHTING_CONFIG => {
                ODError::InvalidWeightingConfig
            }
            empyrean_sys::EMPYREAN_OD_ERROR_NON_FINITE_WEIGHT_MATRIX => {
                ODError::NonFiniteWeightMatrix
            }
            empyrean_sys::EMPYREAN_OD_ERROR_WEIGHT_MATRIX_NOT_POSITIVE_DEFINITE => {
                ODError::WeightMatrixNotPositiveDefinite
            }
            // Every unrecognised code (including NONE) → Unknown(raw),
            // never a valid-looking variant.
            _ => ODError::Unknown(code),
        }
    }
}

/// Why one object's fit failed inside a batch
/// [`Context::determine`](crate::Context::determine).
///
/// Mirrors the engine's `DetermineError` discriminants, so a caller can
/// branch on the cause instead of matching on message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetermineFailureKind {
    /// Observation conversion failed (UTC → TDB, malformed record, …).
    ObservationConversion,
    /// Observer position could not be constructed — unknown observatory
    /// code, epoch outside the loaded kernels, unsupported spacecraft.
    ObserverConstruction,
    /// A roving-observer record used an unsupported coordinate system.
    UnsupportedCoordinateSystem,
    /// The loaded Earth-orientation (BPC) coverage does not span the
    /// observations. An engine-configuration failure, not a property of
    /// the data.
    EarthOrientationCoverage,
    /// Initial orbit determination produced no usable seed.
    IOD,
    /// The N-body differential correction failed.
    OD,
    /// Two observations of this object carried the same observation ID.
    DuplicateObsIds,
    /// Radar observations were supplied with no optical astrometry, which
    /// leaves the two plane-of-sky angular degrees of freedom
    /// unconstrained.
    RadarOnly,
    /// An explicit non-gravitational solve could not recover A1/A2/A3.
    /// Reported rather than silently degrading to a state-only fit.
    NonGravNotRecovered,
    /// An observation named an observatory code in no loaded registry and
    /// matching no station table. Split from
    /// [`ObserverConstruction`](Self::ObserverConstruction) — the one
    /// observer failure a caller fixes by registering the code. A hard
    /// refusal, never a deselect; the code and its observation index are in
    /// [`DetermineFailure::message`].
    UnknownObservatoryCode,
    /// A failure code this build of the wrapper does not know. Carries
    /// the raw code so a newer engine's cause is never flattened away.
    Unknown(i32),
}

impl DetermineFailureKind {
    pub(super) fn from_code(code: i32) -> Self {
        match code {
            empyrean_sys::EMPYREAN_OD_FAILURE_OBSERVATION_CONVERSION => Self::ObservationConversion,
            empyrean_sys::EMPYREAN_OD_FAILURE_OBSERVER_CONSTRUCTION => Self::ObserverConstruction,
            empyrean_sys::EMPYREAN_OD_FAILURE_UNSUPPORTED_COORDINATE_SYSTEM => {
                Self::UnsupportedCoordinateSystem
            }
            empyrean_sys::EMPYREAN_OD_FAILURE_EARTH_ORIENTATION_COVERAGE => {
                Self::EarthOrientationCoverage
            }
            empyrean_sys::EMPYREAN_OD_FAILURE_IOD => Self::IOD,
            empyrean_sys::EMPYREAN_OD_FAILURE_OD => Self::OD,
            empyrean_sys::EMPYREAN_OD_FAILURE_DUPLICATE_OBS_IDS => Self::DuplicateObsIds,
            empyrean_sys::EMPYREAN_OD_FAILURE_RADAR_ONLY => Self::RadarOnly,
            empyrean_sys::EMPYREAN_OD_FAILURE_NON_GRAV_NOT_RECOVERED => Self::NonGravNotRecovered,
            empyrean_sys::EMPYREAN_OD_FAILURE_UNKNOWN_OBSERVATORY_CODE => {
                Self::UnknownObservatoryCode
            }
            other => Self::Unknown(other),
        }
    }
}

/// One object's failure inside a batch determine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetermineFailure {
    /// ADES object identifier whose fit failed.
    pub object_id: String,
    /// The engine's message.
    pub message: String,
    /// Classified cause.
    pub kind: DetermineFailureKind,
    /// The specific differential-correction error, when
    /// [`kind`](Self::kind) is [`DetermineFailureKind::OD`] and the engine
    /// produced a typed [`ODError`]. `None` for every other kind, and for
    /// an OD-bucket failure raised distribution-side (its detail is in
    /// [`message`](Self::message)). A code newer than this build decodes to
    /// [`ODError::Unknown`], never a valid-looking variant.
    pub od_error: Option<ODError>,
}

impl std::fmt::Display for DetermineFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.object_id, self.message)
    }
}

impl std::error::Error for DetermineFailure {}

/// One object's slot in a batch determine: its identifier and either the
/// delivered fit or the typed failure.
#[derive(Debug, Clone, PartialEq)]
pub struct DetermineEntry {
    /// ADES object identifier (permID / provID / trkSub) the
    /// observations were grouped under.
    pub object_id: String,
    /// The fit, or why this object did not produce one.
    pub outcome: std::result::Result<DetermineResult, DetermineFailure>,
}

impl DetermineEntry {
    /// The fit, when this object delivered one.
    pub fn result(&self) -> Option<&DetermineResult> {
        self.outcome.as_ref().ok()
    }

    /// The failure, when this object did not deliver.
    pub fn failure(&self) -> Option<&DetermineFailure> {
        self.outcome.as_ref().err()
    }

    /// Did this object produce a fit?
    pub fn delivered(&self) -> bool {
        self.outcome.is_ok()
    }
}

/// The result of a batch [`Context::determine`](crate::Context::determine)
/// — one [`DetermineEntry`] per ADES object in the observations, ordered
/// by `object_id`.
///
/// determine is batch-first: fitting one object is the one-entry case of
/// this type, not a separate call. A failure never removes an object from
/// the table, so `len()` always equals the number of objects the
/// observations grouped into.
///
/// ```no_run
/// use empyrean::{Context, ODConfig};
///
/// let ctx = Context::from_data_dir(None)?;
/// let obs = ctx.read_ades("many_objects.psv")?;
/// let fits = ctx.determine(&obs, None, &ODConfig::default())?;
///
/// for entry in fits.iter() {
///     match &entry.outcome {
///         Ok(fit) => println!(
///             "{}: χ²_red={:.2} extrapolable={}",
///             entry.object_id,
///             fit.summary.reduced_chi2,
///             fit.acceptability.extrapolation_acceptable,
///         ),
///         Err(e) => eprintln!("{}: FAILED — {}", entry.object_id, e.message),
///     }
/// }
/// # Ok::<(), empyrean::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DetermineResults {
    entries: Vec<DetermineEntry>,
    unmatched_orbit_ids: Vec<String>,
}

impl DetermineResults {
    /// Assemble a batch from per-object entries.
    ///
    /// The engine builds this from a determine call; it is public so a
    /// caller that composes its own multi-pass fit (e.g. a seed solve
    /// followed by a per-object wide refine) can report the composed
    /// result through the same type, failures included, instead of
    /// inventing a parallel shape.
    ///
    /// Entries are expected in `object_id` order — that is the order the
    /// engine produces and the order every consumer writes rows in.
    pub fn from_entries(entries: Vec<DetermineEntry>, unmatched_orbit_ids: Vec<String>) -> Self {
        Self {
            entries,
            unmatched_orbit_ids,
        }
    }

    pub(crate) fn new(entries: Vec<DetermineEntry>, unmatched_orbit_ids: Vec<String>) -> Self {
        Self::from_entries(entries, unmatched_orbit_ids)
    }

    /// Number of objects the observations grouped into — delivered and
    /// failed alike.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Were there no objects at all?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every slot, in `object_id` order.
    pub fn iter(&self) -> std::slice::Iter<'_, DetermineEntry> {
        self.entries.iter()
    }

    /// Every object identifier, in table order.
    pub fn object_ids(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.object_id.as_str())
    }

    /// The slot for one object, or `None` when the observations carried
    /// no such identifier.
    pub fn get(&self, object_id: &str) -> Option<&DetermineEntry> {
        self.entries.iter().find(|e| e.object_id == object_id)
    }

    /// Only the objects that produced a fit, paired with their id.
    pub fn delivered(&self) -> impl Iterator<Item = (&str, &DetermineResult)> {
        self.entries
            .iter()
            .filter_map(|e| e.result().map(|r| (e.object_id.as_str(), r)))
    }

    /// How many objects produced a fit.
    pub fn delivered_count(&self) -> usize {
        self.entries.iter().filter(|e| e.delivered()).count()
    }

    /// Only the objects that failed.
    pub fn failures(&self) -> impl Iterator<Item = &DetermineFailure> {
        self.entries.iter().filter_map(|e| e.failure())
    }

    /// True when the batch ran but **no** object produced a fit. The
    /// per-object [`failures`](Self::failures) are the diagnosis.
    pub fn all_failed(&self) -> bool {
        !self.entries.is_empty() && self.delivered_count() == 0
    }

    /// Seed orbits from `initial_orbits` that matched no observation
    /// group.
    ///
    /// A non-empty list means those seeds' identities do not appear in
    /// the observations, so they constrained nothing. They are reported
    /// rather than dropped.
    pub fn unmatched_orbit_ids(&self) -> &[String] {
        &self.unmatched_orbit_ids
    }

    /// The one fit, for the common single-object call.
    ///
    /// Errors — loudly — when the batch holds anything other than
    /// exactly one delivered object: zero objects, more than one object
    /// (naming them, so the caller can pick with [`get`](Self::get)
    /// instead), or one object whose fit failed. It never chooses among
    /// several fits on the caller's behalf.
    pub fn into_single(self) -> crate::error::Result<DetermineResult> {
        match self.entries.len() {
            1 => {
                let entry = self.entries.into_iter().next().expect("length checked");
                entry.outcome.map_err(|f| {
                    crate::error::Error::invalid_input(format!(
                        "orbit determination failed for {}: {}",
                        f.object_id, f.message
                    ))
                })
            }
            0 => Err(crate::error::Error::invalid_input(
                "orbit determination produced no objects: the observations \
                 carried no rows to group",
            )),
            n => {
                let ids: Vec<&str> = self.object_ids().collect();
                Err(crate::error::Error::invalid_input(format!(
                    "orbit determination fitted {n} objects ({}); into_single \
                     refuses to choose one — iterate the batch or select with \
                     `get(object_id)`",
                    ids.join(", ")
                )))
            }
        }
    }
}

impl<'a> IntoIterator for &'a DetermineResults {
    type Item = &'a DetermineEntry;
    type IntoIter = std::slice::Iter<'a, DetermineEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl IntoIterator for DetermineResults {
    type Item = DetermineEntry;
    type IntoIter = std::vec::IntoIter<DetermineEntry>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::coordinate::{Frame, Origin};
    use crate::propagate::ForceModelTier;

    fn summary() -> ResidualSummary {
        ResidualSummary {
            num_obs: 42,
            num_selected: 40,
            num_rejected: 2,
            n_eff: 40.0,
            expected_median_d2: 1.3863,
            expected_tail: 0.0111,
            chi2: 38.0,
            dof: 34,
            reduced_chi2: 1.12,
            rms_ra_arcsec: 0.31,
            rms_dec_arcsec: 0.28,
            rms_combined_arcsec: 0.30,
            weighted_rms_ra_arcsec: 0.9,
            weighted_rms_dec_arcsec: 0.8,
            weighted_rms_combined_arcsec: 0.85,
            mean_ra_arcsec: 0.01,
            mean_dec_arcsec: -0.02,
            std_ra_arcsec: 0.31,
            std_dec_arcsec: 0.28,
            rms_along_track_arcsec: f64::NAN,
            rms_cross_track_arcsec: f64::NAN,
        }
    }

    fn acceptability() -> AcceptabilityReport {
        AcceptabilityReport {
            fit_acceptable: true,
            extrapolation_acceptable: true,
            converged_ok: true,
            reduced_chi2_ok: true,
            reduced_chi2_value: 1.12,
            reduced_chi2_threshold: 2.0,
            rms_ok: true,
            rms_value_arcsec: 0.31,
            rms_threshold_arcsec: 1.0,
            residual_isotropy_ok: true,
            at_ct_ratio_value: f64::NAN,
            at_ct_ratio_threshold: 3.0,
            covariance_ok: true,
            arc_coverage_ok: true,
            arc_days_value: 900.0,
            arc_days_threshold: 30.0,
            fractional_sigma_a_ok: true,
            fractional_sigma_a_value: 1.0e-6,
            fractional_sigma_a_threshold: 1.0e-3,
            selection_fraction_ok: true,
            selection_fraction_value: 40.0 / 42.0,
            selection_fraction_threshold: 0.7,
            selected_arc_coverage_ok: true,
            selected_arc_days_value: 890.0,
            selected_arc_fraction_value: 0.99,
            selected_arc_fraction_threshold: 0.8,
            trailing_gap_ok: true,
            trailing_gap_days_value: 0.0,
            trailing_gap_threshold: 30.0,
            radar_fit_ok: None,
        }
    }

    fn fit() -> DetermineResult {
        let state = crate::CoordinateState::cartesian(
            crate::Epoch::from_mjd_tdb(60000.0),
            [1.0, 0.1, 0.02, -0.003, 0.017, 0.0006],
            Frame::EclipticJ2000,
            Origin::Sun,
        );
        DetermineResult {
            orbit: crate::Orbit::new(state),
            residuals: Vec::new(),
            summary: summary(),
            iterations: 5,
            update_norm: 1.0e-12,
            converged: true,
            covariance: [[0.0; 6]; 6],
            covariance_representation: CovarianceRepresentation::Cartesian,
            covariance_9x9: None,
            non_grav_delta: None,
            rejection_passes: 1,
            rejection_kind: DeliveredRejectionKind::Known(
                crate::od::rejection::RejectionKind::Adaptive,
            ),
            rejection_enabled: true,
            num_oppositions_fit: 3,
            force_model_used: ForceModelTier::Standard,
            solve_for_used: SolveForParams::StateOnly,
            acceptability: acceptability(),
            station_biases: Vec::new(),
            solved_covariance: None,
            dt_delta: None,
            amrat_delta: None,
            thrust_delta_m_per_s: Vec::new(),
            thrust_correction_covariances: Vec::new(),
            dispositions: SolveFor::default(),
            warnings: Vec::new(),
            dv_frame: None,
            photometry: None,
            covariance_trust: None,
            // A hand-built fixture, not a solver run: `None` throughout
            // is exactly what that reads as on the real surface.
            termination: None,
            gn_step_qnorm: None,
            mu_final: None,
            accepted_steps: 0,
            final_solve_iterations: None,
            stall_delivery: None,
            resolved_method: crate::CovarianceKind::Linear,
        }
    }

    fn delivered(object_id: &str) -> DetermineEntry {
        DetermineEntry {
            object_id: object_id.to_string(),
            outcome: Ok(fit()),
        }
    }

    fn failed(object_id: &str, kind: DetermineFailureKind) -> DetermineEntry {
        DetermineEntry {
            object_id: object_id.to_string(),
            outcome: Err(DetermineFailure {
                object_id: object_id.to_string(),
                message: "no viable IOD seed".to_string(),
                kind,
                od_error: None,
            }),
        }
    }

    /// A failed object stays in the table. Dropping it would leave the
    /// caller with a shorter batch than they submitted and no signal.
    #[test]
    fn failures_stay_in_the_table() {
        let batch = DetermineResults::new(
            vec![
                delivered("2024 YR4"),
                failed("K25A00B", DetermineFailureKind::IOD),
                delivered("433"),
            ],
            Vec::new(),
        );
        assert_eq!(batch.len(), 3, "every input object keeps its row");
        assert_eq!(batch.delivered_count(), 2);
        assert_eq!(batch.failures().count(), 1);
        assert!(!batch.all_failed());
        assert_eq!(
            batch.object_ids().collect::<Vec<_>>(),
            vec!["2024 YR4", "K25A00B", "433"]
        );
    }

    #[test]
    fn lookup_finds_delivered_and_failed_objects() {
        let batch = DetermineResults::new(
            vec![
                delivered("2024 YR4"),
                failed("K25A00B", DetermineFailureKind::RadarOnly),
            ],
            Vec::new(),
        );

        let hit = batch.get("2024 YR4").expect("delivered object is findable");
        assert!(hit.delivered());
        assert!(hit.result().is_some());
        assert!(hit.failure().is_none());

        let miss = batch.get("K25A00B").expect("failed object is findable");
        assert!(!miss.delivered());
        assert!(miss.result().is_none());
        let f = miss.failure().expect("carries a typed failure");
        assert_eq!(f.kind, DetermineFailureKind::RadarOnly);
        assert_eq!(f.object_id, "K25A00B");
        assert!(f.to_string().contains("K25A00B"));

        assert!(batch.get("1997 XF11").is_none(), "absent object is None");
    }

    /// The whole point of the batch surface: `into_single` must refuse
    /// rather than pick, because picking is how N−1 fits used to vanish.
    #[test]
    fn into_single_refuses_a_multi_object_batch() {
        let batch =
            DetermineResults::new(vec![delivered("2024 YR4"), delivered("433")], Vec::new());
        let err = batch.into_single().expect_err("must refuse to choose");
        assert!(err.message.contains("2 objects"), "{}", err.message);
        // Both identities are named so the caller can select one.
        assert!(err.message.contains("2024 YR4"), "{}", err.message);
        assert!(err.message.contains("433"), "{}", err.message);
    }

    #[test]
    fn into_single_unwraps_the_one_object_case() {
        let batch = DetermineResults::new(vec![delivered("2024 YR4")], Vec::new());
        let fit = batch.into_single().expect("one delivered object unwraps");
        assert!(fit.converged);
        assert_eq!(fit.summary.num_obs, 42);
    }

    #[test]
    fn into_single_surfaces_the_failure_of_a_lone_failed_object() {
        let batch = DetermineResults::new(
            vec![failed("K25A00B", DetermineFailureKind::OD)],
            Vec::new(),
        );
        let err = batch.into_single().expect_err("a failed fit is not a fit");
        assert!(err.message.contains("K25A00B"), "{}", err.message);
        assert!(
            err.message.contains("no viable IOD seed"),
            "{}",
            err.message
        );
    }

    #[test]
    fn into_single_reports_an_empty_batch_as_such() {
        let batch = DetermineResults::new(Vec::new(), Vec::new());
        assert!(batch.is_empty());
        let err = batch.into_single().expect_err("nothing to unwrap");
        assert!(err.message.contains("no objects"), "{}", err.message);
    }

    /// Zero delivered is a state the caller can detect and diagnose, not
    /// an empty success.
    #[test]
    fn all_failed_is_detectable_with_per_object_reasons() {
        let batch = DetermineResults::new(
            vec![
                failed("A", DetermineFailureKind::IOD),
                failed("B", DetermineFailureKind::EarthOrientationCoverage),
            ],
            Vec::new(),
        );
        assert!(batch.all_failed());
        assert_eq!(batch.delivered_count(), 0);
        let kinds: Vec<_> = batch.failures().map(|f| f.kind).collect();
        assert_eq!(
            kinds,
            vec![
                DetermineFailureKind::IOD,
                DetermineFailureKind::EarthOrientationCoverage
            ]
        );
    }

    /// An empty batch is not "all failed" — there was nothing to fail.
    #[test]
    fn an_empty_batch_is_not_all_failed() {
        assert!(!DetermineResults::new(Vec::new(), Vec::new()).all_failed());
    }

    /// A seed that matched no observation group is reported, not
    /// silently ignored.
    #[test]
    fn unmatched_seeds_are_reported() {
        let batch =
            DetermineResults::new(vec![delivered("2024 YR4")], vec!["1997 XF11".to_string()]);
        assert_eq!(batch.unmatched_orbit_ids(), ["1997 XF11"]);
    }

    /// An engine failure code this build does not know keeps its raw
    /// value rather than collapsing into a generic bucket.
    #[test]
    fn unknown_failure_codes_keep_their_raw_value() {
        assert_eq!(
            DetermineFailureKind::from_code(9999),
            DetermineFailureKind::Unknown(9999)
        );
        assert_eq!(
            DetermineFailureKind::from_code(empyrean_sys::EMPYREAN_OD_FAILURE_IOD),
            DetermineFailureKind::IOD
        );
        assert_eq!(
            DetermineFailureKind::from_code(
                empyrean_sys::EMPYREAN_OD_FAILURE_NON_GRAV_NOT_RECOVERED
            ),
            DetermineFailureKind::NonGravNotRecovered
        );
    }

    /// Both iteration shapes reach every slot.
    #[test]
    fn iteration_covers_every_slot() {
        let batch = DetermineResults::new(
            vec![delivered("A"), failed("B", DetermineFailureKind::OD)],
            Vec::new(),
        );
        assert_eq!(batch.iter().count(), 2);
        assert_eq!((&batch).into_iter().count(), 2);
        assert_eq!(batch.into_iter().count(), 2);
    }

    /// The fit summary describes every INPUT object, so a failed fit is
    /// a row with NaN measurements and a reason — never an absence.
    #[test]
    fn fit_summary_rows_cover_failures_with_nan_not_zero() {
        let batch = DetermineResults::new(
            vec![
                delivered("2024 YR4"),
                failed("K25A00B", DetermineFailureKind::IOD),
            ],
            Vec::new(),
        );
        let rows = crate::FitSummaryRow::from_results(&batch);
        assert_eq!(rows.len(), 2, "one row per input object");

        assert_eq!(rows[0].status, "delivered");
        assert_eq!(rows[0].object_id, "2024 YR4");
        assert!(rows[0].converged);
        assert_eq!(rows[0].n_obs, 42);
        assert_eq!(rows[0].n_selected, 40);
        assert!(rows[0].error.is_none());
        assert!(rows[0].extrapolation_acceptable);
        assert_eq!(rows[0].trailing_gap_days, 0.0);
        // A state-only fit has no tagged solved covariance; its width is 6.
        assert_eq!(rows[0].solve_for_width, 6);

        assert_eq!(rows[1].status, "failed");
        assert_eq!(rows[1].object_id, "K25A00B");
        assert!(!rows[1].converged);
        assert!(rows[1].reduced_chi2.is_nan(), "no fit means no χ², not 0");
        assert!(rows[1].rms_ra_arcsec.is_nan());
        assert!(rows[1].selection_fraction.is_nan());
        assert!(rows[1].fractional_sigma_a.is_nan());
        assert!(!rows[1].fit_acceptable && !rows[1].extrapolation_acceptable);
        assert_eq!(rows[1].error.as_deref(), Some("no viable IOD seed"));
    }

    #[test]
    fn solver_stop_criterion_line_matches_the_engine() {
        // The four genuine convergence criteria and nothing else — the
        // same line scott draws; StalledDelivered explicitly means no
        // criterion latched.
        use super::SolverStop as S;
        let all = [
            S::GradientTolerance,
            S::StepTolerance,
            S::CostTolerance,
            S::MaxIterations,
            S::DampingExhausted,
            S::InnerTrialsExhausted,
            S::StalledDelivered,
            S::SchurStepTolerance,
            S::Unrecognized,
        ];
        let criteria: Vec<_> = all.iter().filter(|s| s.is_solver_criterion()).collect();
        assert_eq!(
            criteria,
            [
                &S::GradientTolerance,
                &S::StepTolerance,
                &S::CostTolerance,
                &S::SchurStepTolerance
            ]
        );
        // Tag strings are the stable wire vocabulary — all distinct.
        let mut tags: Vec<_> = all.iter().map(|s| s.as_str()).collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), all.len());
    }

    #[test]
    fn stall_step_sigmas_is_the_root_of_the_quadratic_form() {
        let stall = super::StallDelivery {
            underlying_stop: super::SolverStop::DampingExhausted,
            gn_qnorm: 4.0e-4,
            convergence_tol: 1.0e-5,
            optical_reduced_chi2: 1.1,
        };
        assert_eq!(stall.step_sigmas(), 2.0e-2);
    }
}

#[cfg(test)]
mod noise_model_decode_tests {
    use super::*;
    use crate::od::rejection::RejectionKind;
    use crate::od::weighting::NoiseModel;

    #[test]
    fn blunder_reason_arrives_as_code_16() {
        assert_eq!(empyrean_sys::EMPYREAN_REJECTION_BLUNDER, 16);
        assert_eq!(RejectionReason::from_int(16), RejectionReason::Blunder);
    }

    #[test]
    fn sigma_source_tags_decode() {
        assert_eq!(
            SigmaSource::from_int(empyrean_sys::EMPYREAN_SIGMA_SOURCE_REPORTED as u8),
            SigmaSource::Reported
        );
        assert_eq!(
            SigmaSource::from_int(empyrean_sys::EMPYREAN_SIGMA_SOURCE_PRESET as u8),
            SigmaSource::Preset
        );
        assert_eq!(
            SigmaSource::from_int(empyrean_sys::EMPYREAN_SIGMA_SOURCE_LAYER as u8),
            SigmaSource::Layer
        );
        assert_eq!(
            SigmaSource::from_int(empyrean_sys::EMPYREAN_SIGMA_SOURCE_DEFAULT as u8),
            SigmaSource::Default
        );
        assert_eq!(
            SigmaSource::from_int(empyrean_sys::EMPYREAN_SIGMA_SOURCE_NOT_WEIGHTED as u8),
            SigmaSource::NotWeighted
        );
    }

    #[test]
    fn per_row_noise_model_decodes() {
        assert_eq!(
            noise_model_from_ffi(empyrean_sys::EMPYREAN_NOISE_MODEL_GAUSSIAN as u8, f64::NAN),
            DeliveredNoiseModel::Known(NoiseModel::Gaussian)
        );
        assert_eq!(
            noise_model_from_ffi(empyrean_sys::EMPYREAN_NOISE_MODEL_STUDENT_T as u8, 4.0),
            DeliveredNoiseModel::Known(NoiseModel::StudentT { nu: 4.0 })
        );
        assert_eq!(
            noise_model_from_ffi(empyrean_sys::EMPYREAN_NOISE_MODEL_NIGHT_SCALE as u8, 6.0),
            DeliveredNoiseModel::Known(NoiseModel::NightScale { nu: 6.0 })
        );
    }

    #[test]
    fn an_unknown_law_tag_keeps_its_raw_value_and_is_not_gaussian() {
        // A library newer than this wrapper mints a fourth law tag: it must
        // surface as itself, never silently as Gaussian.
        let decoded = noise_model_from_ffi(99, 4.0);
        assert_eq!(decoded, DeliveredNoiseModel::Unknown(99));
        assert_ne!(decoded, DeliveredNoiseModel::Known(NoiseModel::Gaussian));
    }

    #[test]
    fn rejection_kind_tags_decode() {
        assert_eq!(
            rejection_kind_from_ffi(empyrean_sys::EMPYREAN_REJECTION_KIND_ADAPTIVE as u8),
            DeliveredRejectionKind::Known(RejectionKind::Adaptive)
        );
        assert_eq!(
            rejection_kind_from_ffi(empyrean_sys::EMPYREAN_REJECTION_KIND_CMC2003 as u8),
            DeliveredRejectionKind::Known(RejectionKind::CMC2003)
        );
        assert_eq!(
            rejection_kind_from_ffi(empyrean_sys::EMPYREAN_REJECTION_KIND_BLUNDER as u8),
            DeliveredRejectionKind::Known(RejectionKind::Blunder)
        );
    }

    #[test]
    fn an_unknown_gate_tag_keeps_its_raw_value_and_is_not_adaptive() {
        let decoded = rejection_kind_from_ffi(99);
        assert_eq!(decoded, DeliveredRejectionKind::Unknown(99));
        assert_ne!(
            decoded,
            DeliveredRejectionKind::Known(RejectionKind::Adaptive)
        );
    }
}

#[cfg(test)]
mod od_error_tests {
    use super::ODError;

    /// Every EMPYREAN_OD_ERROR_* code round-trips to the same-named wrapper
    /// variant, all 43 map to distinct variants, and none collides with
    /// the NONE sentinel.
    #[test]
    fn od_error_codes_round_trip_to_named_variants() {
        let table: [(i32, ODError); 43] = [
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOT_CARTESIAN as i32,
                ODError::NotCartesian,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NO_COVARIANCE_ON_ORBIT as i32,
                ODError::NoCovarianceOnOrbit,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_UNCERTAINTY_METHOD as i32,
                ODError::UnsupportedUncertaintyMethod,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOT_SINGLE_ORBIT as i32,
                ODError::NotSingleOrbit,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_EPHEMERIS_GENERATION as i32,
                ODError::EphemerisGeneration,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_SINGULAR_NORMAL_MATRIX as i32,
                ODError::SingularNormalMatrix,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NO_VALID_OBSERVATIONS as i32,
                ODError::NoValidObservations,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NO_OBSERVATIONS_FOR_ORBIT as i32,
                ODError::NoObservationsForOrbit,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOT_CONVERGED as i32,
                ODError::NotConverged,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_RETRY_LADDER_EXHAUSTED as i32,
                ODError::RetryLadderExhausted,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NON_FINITE_TRIAL_COSTS as i32,
                ODError::NonFiniteTrialCosts,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NON_FINITE_EVALUATION as i32,
                ODError::NonFiniteEvaluation,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PERIHELION_EPOCH_REQUIRES_HELIOCENTRIC as i32,
                ODError::PerihelionEpochRequiresHeliocentric,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PERIHELION_EPOCH_NON_FINITE as i32,
                ODError::PerihelionEpochNonFinite,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_SOLVER_CONTRACT_VIOLATION as i32,
                ODError::SolverContractViolation,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_INVALID_SOLVER_CONFIG as i32,
                ODError::InvalidSolverConfig,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_RETIRED_SOLVER_KNOB as i32,
                ODError::RetiredSolverKnob,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PRIOR_CONSTRUCTION as i32,
                ODError::PriorConstruction,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_THRUST_PARAMS_MISSING as i32,
                ODError::ThrustParamsMissing,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_THRUST_PRIOR_MISMATCH as i32,
                ODError::ThrustPriorMismatch,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_CONSIDER_PRIOR_UNAVAILABLE as i32,
                ODError::ConsiderPriorUnavailable,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_MANEUVER_NOT_BRACKETED as i32,
                ODError::ManeuverNotBracketed,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NON_GRAV_DT_MISSING as i32,
                ODError::NonGravDTMissing,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_DT_PRIOR_MISSING as i32,
                ODError::DTPriorMissing,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_SOLVED_WIDTH as i32,
                ODError::UnsupportedSolvedWidth,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_SRP_PARAMS_MISSING as i32,
                ODError::SRPParamsMissing,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_AMRAT_PRIOR_MISSING as i32,
                ODError::AMRATPriorMissing,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_AMRAT_NOT_PHYSICAL as i32,
                ODError::AMRATNotPhysical,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_ARC_TRUNCATION_DISABLED as i32,
                ODError::ArcTruncationDisabled,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_INVALID_OBSERVATION_UNCERTAINTY as i32,
                ODError::InvalidObservationUncertainty,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_OBSERVATION_CONVERSION as i32,
                ODError::ObservationConversion,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_OBSERVER_CONSTRUCTION as i32,
                ODError::ObserverConstruction,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_EARTH_ORIENTATION_COVERAGE_INCOMPLETE as i32,
                ODError::EarthOrientationCoverageIncomplete,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PATHOLOGICAL_IOD as i32,
                ODError::PathologicalIOD,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PROPAGATION_OUT_OF_COVERAGE as i32,
                ODError::PropagationOutOfCoverage,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_UNSUPPORTED_OUTPUT_REPRESENTATION as i32,
                ODError::UnsupportedOutputRepresentation,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_DEGENERATE_REPRESENTATION as i32,
                ODError::DegenerateRepresentation,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_PHOTOMETRIC_FIT as i32,
                ODError::PhotometricFit,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_REJECTION_UNSET as i32,
                ODError::NoiseModelRejectionUnset,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NIGHT_SCALE_NOT_YET_VALIDATED as i32,
                ODError::NightScaleNotYetValidated,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_COMPOSITION_UNSUPPORTED as i32,
                ODError::NoiseModelCompositionUnsupported,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_OBSERVED_INFORMATION_NOT_POSITIVE_DEFINITE as i32,
                ODError::ObservedInformationNotPositiveDefinite,
            ),
            (
                empyrean_sys::EMPYREAN_OD_ERROR_NOISE_MODEL_IRLS_NOT_CONVERGED as i32,
                ODError::NoiseModelIrlsNotConverged,
            ),
        ];
        let mut seen_codes = std::collections::BTreeSet::new();
        let mut seen_variants = std::collections::BTreeSet::new();
        for (code, variant) in table {
            assert_eq!(
                ODError::from_code(code),
                variant,
                "code {code} must decode to its variant"
            );
            assert_ne!(
                code,
                empyrean_sys::EMPYREAN_OD_ERROR_NONE as i32,
                "no variant code is NONE"
            );
            assert!(seen_codes.insert(code), "code {code} is not distinct");
            assert!(
                seen_variants.insert(format!("{variant:?}")),
                "variant {variant:?} is not distinct"
            );
        }
        assert_eq!(seen_codes.len(), 43);
    }

    /// A code this build does not know decodes to Unknown(raw), never a
    /// valid-looking variant.
    #[test]
    fn an_unknown_od_error_code_keeps_its_raw_value() {
        assert_eq!(ODError::from_code(9999), ODError::Unknown(9999));
        assert_ne!(ODError::from_code(9999), ODError::NotCartesian);
    }
}
