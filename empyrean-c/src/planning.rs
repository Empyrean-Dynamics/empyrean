//! Observation-planning C ABI — information-gain analysis of candidate
//! follow-up observations against a prior orbit covariance.
//!
//! A thin shell over [`empyrean_core::planning::evaluate_plan_single`]. Given a
//! fitted orbit that carries a 6×6 Cartesian covariance (the information prior)
//! and a list of candidate `(observatory, epoch)` observations — optical or
//! radar — the planner reports how much each observation would shrink the orbit
//! uncertainty: the prior / posterior covariance metrics and a per-candidate
//! marginal-information-gain analysis.
//!
//! Radar candidates fold a predicted delay (range) / Doppler (range-rate)
//! measurement: the line-of-sight information that angles-only optical cannot
//! supply. The measurement σ is the Cramér-Rao bound from the waveform
//! bandwidth and the effective SNR (supplied, or computed from the link budget
//! with no silent default for the leverage-heavy target properties).
//!
//! The encounter-B-plane characterization that `scott::planning` can also emit
//! is intentionally not exposed here yet — it is only meaningful for a close
//! encounter and will be added additively when that showcase is built.

use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use std::collections::HashMap;

use empyrean_core::planning::{
    Anchor, AnchorCircumstances, AnchorPolicy, CandidateInfo, CandidateKind, CovarianceMetrics,
    Gate, LogDetWithheld, Night, ObservatoryConfig, ObservatoryVisibility, PlanResult,
    PlannedObservation, PlanningConfig, PlanningError, RadarMode, RadarPlanSpec, Station,
    TargetRadarProperties, VisibilityPoint, VisibilityResult, VisibilityWindow, compute_visibility,
    evaluate_plan_single,
};
use empyrean_core::time::Epoch;

use crate::joint::{
    EmpyreanPackedJoint, empty_packed_joint, free_packed_joint, sky_uncertainty_to_c,
};
use crate::od::empyrean_orbit_to_orbits;
use crate::propagate::{
    EMPYREAN_COVARIANCE_KIND_UNKNOWN, EmpyreanOrbit, EmpyreanUncertaintyMethod, batch_orbit_id,
    covariance_kind_to_u8, flat_to_uncertainty_method, int_to_force_model,
};
use crate::{EmpyreanContext, set_last_error};

// ── Input structs ───────────────────────────────────────────────────

/// Per-observatory assumptions: astrometric σ + observability filters.
///
/// # Which filters this ABI applies
///
/// [`empyrean_evaluate_plan`] is the only exported function that reads
/// this struct, and it consults **two** of the four filters below:
/// `max_apparent_mag` and `min_elongation_deg`, the site-invariant pair.
/// A candidate's `observable` flag is their conjunction and nothing
/// else. In practice only the elongation test can fire, because the
/// target's absolute magnitude does not reach the planner on this entry
/// point, so the magnitude test passes vacuously.
///
/// [`min_elevation_deg`](EmpyreanObservatoryConfig::min_elevation_deg)
/// and
/// [`max_sun_altitude_deg`](EmpyreanObservatoryConfig::max_sun_altitude_deg)
/// are marshaled across this boundary in full — they reach the engine's
/// own observatory config with the values set here — but the gates that
/// read them belong to the engine's **visibility survey**, which this
/// ABI does not export. Setting either therefore changes no number
/// `empyrean_evaluate_plan` returns, on any release of this ABI so far.
/// They ride the struct now so that exposing the survey later needs no
/// further break, and it is said here rather than left to be discovered:
/// a plan whose candidates are `observable` may still include epochs at
/// which the target is under that site's horizon or the sky above it is
/// bright.
#[repr(C)]
pub struct EmpyreanObservatoryConfig {
    /// MPC observatory code (null-terminated UTF-8).
    pub obs_code: *const c_char,
    /// 1σ (RA·cosδ) astrometric uncertainty (arcsec).
    pub sigma_ra_arcsec: f64,
    /// 1σ Dec astrometric uncertainty (arcsec).
    pub sigma_dec_arcsec: f64,
    /// Limiting apparent magnitude.
    pub max_apparent_mag: f64,
    /// Minimum solar elongation (degrees).
    pub min_elongation_deg: f64,
    /// Minimum geometric elevation of the target above the site's local
    /// horizon, in degrees.
    ///
    /// The elevation is \\( h = \arcsin(\hat{u} \cdot \hat{s}) \\), with
    /// \\(\hat{u}\\) the site's geodetic zenith and \\(\hat{s}\\) the
    /// apparent — light-time- and aberration-corrected — topocentric
    /// direction to the target. **Atmospheric refraction is ignored**:
    /// near the horizon refraction lifts a source by roughly
    /// \\(0.5°\\), and the error falls below \\(0.1°\\) by
    /// \\(h \approx 10°\\).
    ///
    /// `0.0` — the zero-init value — is the geometric horizon, which is
    /// also the engine's default. That is the least-opinionated
    /// statement the geometry can make, *not* an observing
    /// recommendation: airmass at \\(h = 0°\\) is \\(\approx 38\\), and
    /// real programs cut between \\(20°\\) and \\(30°\\). The site's own
    /// pointing or airmass limit is the value to carry here — e.g.
    /// `30.0` for airmass \\(\le 2\\).
    ///
    /// **No exported entry point applies it.** The value is marshaled
    /// into the engine's observatory config, but the elevation gate that
    /// reads it lives in the visibility survey this ABI does not export;
    /// [`empyrean_evaluate_plan`] consults the site-invariant pair
    /// alone. See the struct-level docs.
    pub min_elevation_deg: f64,
    /// 1 when [`max_sun_altitude_deg`](Self::max_sun_altitude_deg)
    /// carries a darkness threshold; 0 to take the engine's default of
    /// −18° (astronomical twilight).
    ///
    /// This axis needs the switch and
    /// [`min_elevation_deg`](Self::min_elevation_deg) does not, for one
    /// reason: `0.0` is a legal and meaningful solar altitude (the Sun's
    /// centre on the geometric horizon), so a zero-init struct read as a
    /// literal `0.0` would quietly plan a campaign in daylight. The
    /// elevation default *is* `0.0`, so its zero-init value is already
    /// the engine's.
    pub has_max_sun_altitude_deg: u8,
    /// Solar altitude at or below which the site counts as dark, in
    /// degrees. Only read when
    /// [`has_max_sun_altitude_deg`](Self::has_max_sun_altitude_deg) is 1.
    ///
    /// Built from the same site zenith as `min_elevation_deg`, against
    /// the **geometric** topocentric direction to the Sun — no
    /// light-time, aberration or refraction correction, each of which is
    /// a few tens of arcseconds at most against a threshold that
    /// operates on degrees.
    ///
    /// The standard conventions are civil (\\(-6°\\)), nautical
    /// (\\(-12°\\)) and astronomical (\\(-18°\\), the engine default,
    /// where the sky background has fallen to its dark-time floor).
    /// Because refraction is ignored, `0.0` means the Sun's *centre* on
    /// the geometric horizon, about \\(0.83°\\) later than visible
    /// sunset. A value above \\(+90°\\) disables the darkness gate.
    ///
    /// **No exported entry point applies it**, on the same terms as
    /// [`min_elevation_deg`](Self::min_elevation_deg): the value reaches
    /// the engine's observatory config, and the darkness gate that reads
    /// it belongs to the unexported visibility survey. See the
    /// struct-level docs.
    pub max_sun_altitude_deg: f64,

    // ── Night-product twilight thresholds (0.11.0) ────────────────
    /// 1 when the four twilight thresholds below carry caller values; 0 to
    /// take the engine's defaults (−0.833° / −6° / −12° / −18°).
    ///
    /// A single switch gates all four because they are validated as a
    /// **group** — the engine refuses a set that does not strictly descend
    /// (sunset > civil > nautical > astronomical). A zero-init struct
    /// (`memset(0)`) would read `0, 0, 0, 0`, which is not strictly
    /// descending and would be refused, so — exactly as
    /// [`has_max_sun_altitude_deg`](Self::has_max_sun_altitude_deg) guards
    /// the darkness gate against a literal `0.0` — this switch lets a
    /// zero-init struct resolve to the engine's own defaults rather than to
    /// an invalid all-zero set. These thresholds structure the [night
    /// product](empyrean_compute_visibility) only; per-epoch darkness still
    /// gates on [`max_sun_altitude_deg`](Self::max_sun_altitude_deg).
    pub has_twilight_thresholds: u8,
    /// Sun altitude (degrees) that opens/closes the observable night on the
    /// refraction-corrected visible horizon (sunset/sunrise). Default
    /// `-0.833` — the Sun's upper limb on the apparent horizon. Read only
    /// when [`has_twilight_thresholds`](Self::has_twilight_thresholds) is 1.
    pub sunset_sun_altitude_deg: f64,
    /// Sun altitude (degrees) for civil twilight. Default `-6.0`. Read only
    /// when [`has_twilight_thresholds`](Self::has_twilight_thresholds) is 1.
    pub civil_twilight_sun_altitude_deg: f64,
    /// Sun altitude (degrees) for nautical twilight. Default `-12.0`. Read
    /// only when [`has_twilight_thresholds`](Self::has_twilight_thresholds)
    /// is 1.
    pub nautical_twilight_sun_altitude_deg: f64,
    /// Sun altitude (degrees) for astronomical twilight. Default `-18.0` —
    /// the same convention as the darkness gate's default, but an
    /// independent night-structure threshold. Read only when
    /// [`has_twilight_thresholds`](Self::has_twilight_thresholds) is 1.
    pub astronomical_twilight_sun_altitude_deg: f64,
    /// 1 when [`explicit_anchor_utc_mjd_tdb`](Self::explicit_anchor_utc_mjd_tdb)
    /// carries a caller-supplied anchor epoch; 0 for the three site-derived
    /// anchors only.
    pub has_explicit_anchor_utc: u8,
    /// Caller-supplied epoch (MJD, TDB) for the
    /// `EMPYREAN_ANCHOR_POLICY_EXPLICIT_UTC` anchor: every night whose
    /// noon-to-noon span contains it also carries an explicit anchor there,
    /// with the circumstances read off the ephemeris row. Read only when
    /// [`has_explicit_anchor_utc`](Self::has_explicit_anchor_utc) is 1.
    pub explicit_anchor_utc_mjd_tdb: f64,
}

/// A single planned (candidate) observation: an epoch plus either an optical or
/// a radar measurement spec. Optical fields are read when `kind == 0`, radar
/// fields when `kind == 1`; absent optional radar target properties use NaN
/// (the link budget refuses, rather than silently defaults, a missing value).
#[repr(C)]
pub struct EmpyreanPlannedObservation {
    /// Planned (receive, for radar) epoch — MJD, TDB.
    pub epoch_mjd_tdb: f64,
    /// 0 = optical, 1 = radar.
    pub kind: u8,

    /// Optical: registered MPC station code (null-terminated UTF-8).
    pub optical_code: *const c_char,
    /// Optical 1σ (RA·cosδ) uncertainty (arcsec).
    pub optical_sigma_ra_arcsec: f64,
    /// Optical 1σ Dec uncertainty (arcsec).
    pub optical_sigma_dec_arcsec: f64,

    /// Radar transmit dish preset: 0 = Goldstone DSS-14, 1 = Green Bank
    /// (receive-only), 2 = Arecibo (defunct — rejected loudly, never scheduled).
    pub radar_transmit_station: u8,
    /// Radar receive dish preset (same encoding; equal to transmit = monostatic).
    pub radar_receive_station: u8,
    /// Radar mode: 0 = Delay, 1 = Doppler, 2 = Both.
    pub radar_mode: u8,
    /// Waveform bandwidth (Hz) — sets the delay (range) σ.
    pub radar_bandwidth_hz: f64,
    /// Doppler frequency resolution (Hz) — sets the Doppler (range-rate) σ.
    pub radar_freq_resolution_hz: f64,
    /// Supplied effective SNR (linear, not dB). NaN → compute from the link
    /// budget using the target properties + integration time below.
    pub radar_snr: f64,
    /// Link-budget target absolute magnitude H (mag). NaN = absent.
    pub radar_target_h_mag: f64,
    /// Link-budget target visual geometric albedo. NaN = absent.
    pub radar_target_visual_albedo: f64,
    /// Link-budget target radar (OC) albedo. NaN = absent.
    pub radar_target_radar_albedo: f64,
    /// Link-budget target effective diameter (km). NaN = absent.
    pub radar_target_diameter_km: f64,
    /// Link-budget target rotation period (hours), caps coherent integration.
    /// NaN = absent.
    pub radar_target_spin_period_hours: f64,
    /// Coherent integration time (s) for the link-budget SNR.
    pub radar_integration_s: f64,
}

/// Planning configuration: force model, integration tolerance, the
/// per-observatory astrometric assumptions, and threading.
#[repr(C)]
pub struct EmpyreanPlanningConfig {
    /// Force-model tier: 0 = Approximate, 1 = Basic, 2 = Standard.
    pub force_model: i32,
    /// GR15 / IAS15 integration tolerance.
    pub epsilon: f64,
    /// Per-observatory config array.
    pub observatories: *const EmpyreanObservatoryConfig,
    /// Number of entries in `observatories`.
    pub num_observatories: usize,
    /// Threads for batch operations; -1 = all CPUs.
    pub num_threads: i32,

    // ── Uncertainty method (0.11.0) ───────────────────────────────
    /// Uncertainty method for the covariance fold — the same flat tag +
    /// per-variant parameter carrier the propagation config uses
    /// ([`EmpyreanUncertaintyMethod`](crate::propagate::EmpyreanUncertaintyMethod)).
    /// A zero-initialized struct selects
    /// [`EMPYREAN_UNCERTAINTY_FIRST`](crate::propagate::EMPYREAN_UNCERTAINTY_FIRST)
    /// — the single-linearization Gaussian fold every previous release of
    /// `empyrean_evaluate_plan` computed — so `memset(0)` is byte-identical
    /// to a pre-0.11.0 caller. The planning fold honours
    /// `FIRST` / `SECOND` / `SIGMA_POINT` / `MONTE_CARLO` / `AUTO`, each
    /// reported on [`EmpyreanPlanResult::resolved_method`] in the
    /// covariance-kind vocabulary; `MIXTURE` is refused by name.
    pub uncertainty_method: EmpyreanUncertaintyMethod,
}

// ── Output structs ──────────────────────────────────────────────────

/// [`EmpyreanCovarianceMetrics::log_det`] is delivered — the finite
/// \\(\ln\det\Sigma\\). [`EmpyreanCovarianceMetrics::log_det_withheld`]
/// carries this when `log_det` is a real number, never withheld.
pub const EMPYREAN_LOG_DET_WITHHELD_NONE: u8 = 0;
/// `log_det` is `NaN` because the metric was reported past a closed-form
/// second-order moment transport, where no epoch-invariant determinant
/// identity holds. The epoch-invariant
/// [`EmpyreanPlanResult::log_det_gain`] stays finite.
pub const EMPYREAN_LOG_DET_WITHHELD_SECOND_ORDER_MOMENT: u8 = 1;
/// `log_det` is `NaN` because the metric was reported past a sampled
/// transport whose products are flown-member sample moments; the sampled
/// transport kind rides beside it in
/// [`EmpyreanCovarianceMetrics::log_det_withheld_method`], an
/// `EMPYREAN_COVARIANCE_KIND_*` value (`SIGMA_POINT` or `MONTE_CARLO`).
pub const EMPYREAN_LOG_DET_WITHHELD_SAMPLED_MOMENT: u8 = 2;
/// `log_det` is `NaN` because the metric was reported past a
/// Gaussian-mixture moment transport. Not produced by this planning seam
/// (it runs the single-linearization fold), but named so the withholding
/// vocabulary is complete across the distribution's uncertainty methods.
pub const EMPYREAN_LOG_DET_WITHHELD_MIXTURE_MOMENT: u8 = 3;

/// Covariance summary metrics (prior, posterior, or per-candidate cumulative).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanCovarianceMetrics {
    /// RSS position 1σ (km).
    pub position_sigma_km: f64,
    /// RSS velocity 1σ (m/s) — the rendezvous-Δv-knowledge term.
    pub velocity_sigma_m_s: f64,
    /// Semi-major axis of the 1σ position ellipsoid (km).
    pub semi_major_km: f64,
    /// Semi-minor axis of the 1σ position ellipsoid (km).
    pub semi_minor_km: f64,
    /// ln(det(Σ)) — the absolute D-optimality criterion over the 6×6
    /// state block. A finite, delivered value at the prior epoch and at a
    /// report epoch reached under the single-linearization (`Linear`)
    /// transport — which is the only transport this planning seam runs,
    /// so on every result `empyrean_evaluate_plan` returns today this is
    /// finite. `NaN` **only** when
    /// [`log_det_withheld`](Self::log_det_withheld) names a nonlinear
    /// transport that made the absolute determinant ill-posed — never a
    /// NaN without a reason, and never a zero that would read as a
    /// unit-determinant. The two fields are complementary: exactly one of
    /// `log_det` (finite) / `log_det_withheld` (a non-`NONE` reason)
    /// carries the state.
    pub log_det: f64,
    /// Why [`log_det`](Self::log_det) is `NaN`, when it is — an
    /// `EMPYREAN_LOG_DET_WITHHELD_*` code. `EMPYREAN_LOG_DET_WITHHELD_NONE`
    /// (0) whenever `log_det` is delivered (finite); otherwise the
    /// nonlinear transport that withheld it, named rather than left as a
    /// bare NaN.
    pub log_det_withheld: u8,
    /// The sampled transport kind when
    /// [`log_det_withheld`](Self::log_det_withheld) is
    /// `EMPYREAN_LOG_DET_WITHHELD_SAMPLED_MOMENT` — an
    /// `EMPYREAN_COVARIANCE_KIND_*` value (`SIGMA_POINT` or
    /// `MONTE_CARLO`). `EMPYREAN_COVARIANCE_KIND_UNKNOWN` for every other
    /// `log_det_withheld` code, where no sampled method applies.
    pub log_det_withheld_method: u8,
}

/// One predicted sky-plane point of the plan's optical ephemeris: the
/// target's predicted topocentric right ascension and declination
/// (degrees, ICRF) at an optical candidate's epoch (MJD, TDB).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanPlanEphemerisPoint {
    /// Epoch — MJD, TDB.
    pub epoch_mjd_tdb: f64,
    /// Predicted topocentric right ascension (degrees, ICRF).
    pub ra_deg: f64,
    /// Predicted topocentric declination (degrees, ICRF).
    pub dec_deg: f64,
}

/// Per-candidate information-gain analysis.
#[repr(C)]
pub struct EmpyreanPlanCandidate {
    /// Row in the plan's optical `ephemeris` array for an optical
    /// candidate; position in the radar sub-sequence for a radar
    /// candidate.
    pub index: usize,
    /// Observatory / receive-station code (owned; freed by
    /// `empyrean_plan_result_free`).
    pub obs_code: *mut c_char,
    /// 0 = optical, 1 = radar.
    pub kind: u8,
    /// 1 if this candidate passes the **site-invariant** filters at this
    /// epoch — solar elongation and apparent magnitude, and nothing else
    /// (radar candidates report 1 unconditionally, and the magnitude test
    /// passes vacuously here because the target's absolute magnitude does
    /// not reach the planner).
    ///
    /// Read it as "not ruled out from Earth", not "schedulable from this
    /// site": the target's elevation above the site's horizon and the
    /// Sun's altitude there are not tested here. See
    /// [`EmpyreanObservatoryConfig`].
    pub observable: u8,
    /// Sky-plane along-track 1σ (arcsec) — optical geometry (NaN for radar).
    pub along_track_sigma_arcsec: f64,
    /// Sky-plane cross-track 1σ (arcsec).
    pub cross_track_sigma_arcsec: f64,
    /// Predicted RA·cosδ 1σ (arcsec).
    pub ra_sigma_arcsec: f64,
    /// Predicted Dec 1σ (arcsec).
    pub dec_sigma_arcsec: f64,
    /// Position angle of the predicted **sky motion** (degrees, east of
    /// north) — the axis the along/cross-track σ above are projected
    /// onto. Optical only.
    ///
    /// Kinematic, and not a function of the covariance: it is *not* the
    /// orientation of the sky-plane uncertainty ellipse. The range is
    /// \\((-180, 180]\\); add 360 **to negative values** for the
    /// conventional \\([0, 360)\\) convention.
    pub position_angle_deg: f64,
    /// Marginal covariance-volume reduction factor from this observation (≤ 1).
    pub marginal_volume_reduction: f64,
    /// Fractional position-σ improvement from this observation (∈ [0, 1]).
    pub marginal_position_improvement: f64,
    /// Post-observation along-track 1σ (arcsec).
    pub post_along_track_sigma_arcsec: f64,
    /// Post-observation cross-track 1σ (arcsec).
    pub post_cross_track_sigma_arcsec: f64,
    /// Covariance metrics after folding this observation and all prior ones.
    pub cumulative: EmpyreanCovarianceMetrics,
    /// Active solve-for width (6 = state-only).
    pub active_width: usize,
    /// Radar only: effective signal-to-noise ratio used for the
    /// delay/Doppler measurement uncertainty (linear power ratio, not
    /// dB). Always finite and positive for a radar candidate; NaN for
    /// an optical candidate.
    pub radar_snr: f64,
    /// Radar only: one-way topocentric range to the target at the
    /// receive epoch (km), derived from the predicted round-trip delay.
    /// NaN for an optical candidate.
    pub radar_range_km: f64,
    /// Radar only: human-readable notes recording assumptions made
    /// while deriving the SNR from the link budget (for example, a
    /// diameter derived from the absolute magnitude and visual albedo,
    /// or coherent integration left uncapped because the spin period is
    /// unknown). Owned, null-terminated UTF-8 strings, freed by
    /// `empyrean_plan_result_free`. Null/0 when there are no notes:
    /// optical candidates, a caller-supplied SNR, or a fully specified
    /// link budget.
    pub radar_provenance: *mut *mut c_char,
    /// Number of entries in `radar_provenance` (0 when null).
    pub num_radar_provenance: usize,
    /// Radar only: measurement mode — 0 = delay, 1 = Doppler,
    /// 2 = both (same encoding as the planned-observation input).
    /// -1 for an optical candidate.
    pub radar_mode: i32,
}

/// Result of a plan evaluation. The caller allocates the struct; the
/// `orbit_id` string, the `candidates` array (with their `obs_code`
/// and `radar_provenance` strings), and the `ephemeris` array are
/// heap-allocated here and must be released with
/// [`empyrean_plan_result_free`].
#[repr(C)]
pub struct EmpyreanPlanResult {
    /// Orbit identifier (owned).
    pub orbit_id: *mut c_char,
    /// Prior (pre-observation) covariance metrics.
    pub prior: EmpyreanCovarianceMetrics,
    /// Posterior (all candidates folded) covariance metrics.
    pub posterior: EmpyreanCovarianceMetrics,
    /// Epoch-invariant prior→posterior change in the absolute 6×6
    /// state-block ln-determinant, in nats:
    /// \\(\ln\det\Sigma_{\mathrm{post}} - \ln\det\Sigma_{\mathrm{prior}}\\)
    /// formed once at the prior's epoch, where both are well-posed.
    /// Negative when the planned observations shrink the covariance
    /// volume; \\(\exp(\text{gain}/6)\\) is the whole-plan 1σ-volume
    /// ratio. Always finite — it is reached at the prior epoch, before
    /// any transport, so it stays available even when
    /// [`prior`](Self::prior) / [`posterior`](Self::posterior) carry a
    /// withheld (`NaN`) `log_det` past a nonlinear transport.
    pub log_det_gain: f64,
    /// Per-candidate analysis array (owned).
    pub candidates: *mut EmpyreanPlanCandidate,
    /// Number of entries in `candidates`.
    pub num_candidates: usize,
    /// Active solve-for width (6 = state-only).
    pub active_width: usize,
    /// Predicted optical ephemeris, one point per optical candidate in
    /// chronological order; an optical candidate's `index` field gives
    /// its row in this array (radar candidates carry no sky-plane
    /// prediction). Owned; freed by `empyrean_plan_result_free`. Null
    /// with count 0 for a radar-only plan.
    pub ephemeris: *mut EmpyreanPlanEphemerisPoint,
    /// Number of entries in `ephemeris` (0 when null).
    pub num_ephemeris: usize,

    // ── Resolved uncertainty method (0.11.0) ──────────────────────
    /// The covariance kind the fold actually resolved and reported the
    /// prior / posterior / cumulative metrics under — an
    /// `EMPYREAN_COVARIANCE_KIND_*` value. `FirstOrder` resolves to
    /// [`EMPYREAN_COVARIANCE_KIND_LINEAR`](crate::propagate::EMPYREAN_COVARIANCE_KIND_LINEAR),
    /// `SecondOrder` to
    /// [`EMPYREAN_COVARIANCE_KIND_SECOND_ORDER`](crate::propagate::EMPYREAN_COVARIANCE_KIND_SECOND_ORDER),
    /// `SigmaPoint` to
    /// [`EMPYREAN_COVARIANCE_KIND_SIGMA_POINT`](crate::propagate::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT),
    /// `MonteCarlo` to
    /// [`EMPYREAN_COVARIANCE_KIND_MONTE_CARLO`](crate::propagate::EMPYREAN_COVARIANCE_KIND_MONTE_CARLO).
    /// `Auto` is reported as the kind it actually ran, never as `Auto`.
    /// Passed through verbatim from the engine's
    /// `PlanResult::resolved_method`: it is the box the search is sized to,
    /// so a sampled transport's `log_det_withheld` on the metrics beside
    /// it reads under the same resolved kind.
    pub resolved_method: u8,
}

// ── Visibility night product (0.11.0) ───────────────────────────────
//
// `empyrean_compute_visibility` surveys one or more orbits over a time
// window from a set of ground (or horizonless) observatories and reports,
// per observatory: the per-epoch observability `points`, the contiguous
// observable `windows`, and the local-night product `nights` — twilight
// epochs, the local solar midnight, the per-night observable intervals, and
// the anchors (a chosen epoch and its circumstances under each policy). The
// Moon's altitude, azimuth and illuminated fraction ride the points and the
// anchor circumstances. Every `Option<f64>` engine field is a `has_*` flag
// beside its value, exactly as `EmpyreanPlanEphemerisPoint` and
// `EmpyreanObservatoryConfig` already do (never a NaN sentinel that a
// measured `0.0` could be mistaken for).

/// Which gate closed a [`EmpyreanVisibilityWindow`] — the `closing_gate`
/// value set. `0` (`_NONE`) when the engine reported no closing gate; the
/// others mirror `scott::planning::Gate` one for one.
pub const EMPYREAN_GATE_NONE: u8 = 0;
/// Solar elongation fell below the configured minimum.
pub const EMPYREAN_GATE_ELONGATION: u8 = 1;
/// Apparent magnitude rose above the configured limit.
pub const EMPYREAN_GATE_MAGNITUDE: u8 = 2;
/// Target altitude fell below the configured minimum elevation.
pub const EMPYREAN_GATE_ELEVATION: u8 = 3;
/// The Sun rose above the configured darkness threshold (dawn).
pub const EMPYREAN_GATE_DARKNESS: u8 = 4;
/// No gate closed it — the window ran to the end of its night (or of the
/// surveyed span).
pub const EMPYREAN_GATE_END_OF_NIGHT: u8 = 5;

/// How a night anchor's epoch was chosen — the
/// [`EmpyreanAnchor::policy`] value set. Mirrors
/// `scott::planning::AnchorPolicy`; the `ExplicitUTC` epoch itself rides on
/// [`EmpyreanAnchor::epoch_mjd_tdb`].
pub const EMPYREAN_ANCHOR_POLICY_NEAREST_OBSERVABLE_TO_MIDNIGHT: u8 = 0;
/// The local solar midnight itself, observable or not.
pub const EMPYREAN_ANCHOR_POLICY_MIDNIGHT: u8 = 1;
/// The target's transit — its maximum altitude while the site is dark.
pub const EMPYREAN_ANCHOR_POLICY_TRANSIT: u8 = 2;
/// A caller-supplied UTC epoch
/// ([`EmpyreanObservatoryConfig::explicit_anchor_utc_mjd_tdb`]).
pub const EMPYREAN_ANCHOR_POLICY_EXPLICIT_UTC: u8 = 3;

/// A per-orbit [`EmpyreanVisibilityResult`] was delivered — its
/// `error_code` is this, and its observatory data is populated.
pub const EMPYREAN_PLANNING_ERROR_NONE: i32 = 0;
/// Invalid time window or step for the survey.
pub const EMPYREAN_PLANNING_ERROR_INVALID_TIME_WINDOW: i32 = 1;
/// The observer could not be constructed from its code.
pub const EMPYREAN_PLANNING_ERROR_OBSERVER_CONSTRUCTION: i32 = 2;
/// A station code could not be parsed as an MPC observatory code.
pub const EMPYREAN_PLANNING_ERROR_INVALID_STATION_CODE: i32 = 3;
/// Ephemeris generation failed at the call level.
pub const EMPYREAN_PLANNING_ERROR_EPHEMERIS_GENERATION: i32 = 4;
/// A single orbit's forward propagation was recorded as failed.
pub const EMPYREAN_PLANNING_ERROR_ORBIT_PROPAGATION_FAILED: i32 = 5;
/// A single orbit's optical-ephemeris generation was recorded as failed.
pub const EMPYREAN_PLANNING_ERROR_ORBIT_EPHEMERIS_FAILED: i32 = 6;
/// The orbit's prior covariance was withheld by the engine.
pub const EMPYREAN_PLANNING_ERROR_CANDIDATE_COVARIANCE_WITHHELD: i32 = 7;
/// A parallel worker-budget request was refused.
pub const EMPYREAN_PLANNING_ERROR_THREAD_BUDGET: i32 = 8;
/// No observatories were configured.
pub const EMPYREAN_PLANNING_ERROR_NO_OBSERVATORIES: i32 = 9;
/// The requested uncertainty method has no honest fold on this entry point.
pub const EMPYREAN_PLANNING_ERROR_UNSUPPORTED_UNCERTAINTY_METHOD: i32 = 10;
/// An Earth-orientation kernel the survey needed was unavailable.
pub const EMPYREAN_PLANNING_ERROR_EARTH_ORIENTATION_UNAVAILABLE: i32 = 11;
/// The Earth-orientation rotation could not be evaluated at a survey epoch.
pub const EMPYREAN_PLANNING_ERROR_EARTH_ORIENTATION_EVALUATION: i32 = 12;
/// A required body state was unavailable from the loaded kernels.
pub const EMPYREAN_PLANNING_ERROR_BODY_STATE_UNAVAILABLE: i32 = 13;
/// The site geometry was degenerate (e.g. a pole-coincident zenith).
pub const EMPYREAN_PLANNING_ERROR_DEGENERATE_SITE_GEOMETRY: i32 = 14;
/// The visibility grid did not match across observatories.
pub const EMPYREAN_PLANNING_ERROR_VISIBILITY_GRID_MISMATCH: i32 = 15;
/// An observability diagnostic the survey needed was missing.
pub const EMPYREAN_PLANNING_ERROR_MISSING_OBSERVABILITY_DIAGNOSTIC: i32 = 16;
/// The astrometry frame was not ICRF where ICRF was required.
pub const EMPYREAN_PLANNING_ERROR_ASTROMETRY_FRAME_NOT_ICRF: i32 = 17;
/// An SB441-N16 body was SPK-substituted and refused rather than served
/// without observation partials.
pub const EMPYREAN_PLANNING_ERROR_SPK_SUBSTITUTED_BODY: i32 = 18;
/// A prior covariance that had to reach the metrics helpers was not finite.
pub const EMPYREAN_PLANNING_ERROR_NON_FINITE_PRIOR_COVARIANCE: i32 = 19;
/// A `[ObservatoryConfig]`'s twilight thresholds were invalid — not finite,
/// or not strictly descending (sunset > civil > nautical > astronomical).
/// (NEW in 0.11.0.)
pub const EMPYREAN_PLANNING_ERROR_INVALID_OBSERVATORY_CONFIG: i32 = 20;
/// The local-solar-time conversion the night product keys on failed — an
/// epoch with no UTC representation. (NEW in 0.11.0.)
pub const EMPYREAN_PLANNING_ERROR_NIGHT_TIME_CONVERSION: i32 = 21;
/// A planning error outside the visibility survey's reachable set — a fold,
/// encounter-B-plane, output-epoch transport, second-order or sampled-rung
/// refusal that `empyrean_compute_visibility` does not exercise. The engine's
/// full typed cause is always carried in
/// [`EmpyreanVisibilityResult::error_message`]; this code only classes the
/// kind as "outside the survey".
pub const EMPYREAN_PLANNING_ERROR_OTHER: i32 = 99;

/// One epoch's observability from one observatory — mirrors
/// `scott::planning::VisibilityPoint` field for field. The four `*_ok`
/// flags are the individual gates and [`observable`](Self::observable) is
/// their conjunction, so a caller sees WHICH condition closed a window.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanVisibilityPoint {
    /// Epoch — MJD, TDB.
    pub epoch_mjd_tdb: f64,
    /// Solar elongation (degrees).
    pub solar_elongation_deg: f64,
    /// 1 when [`elevation_deg`](Self::elevation_deg) is a measured value; 0
    /// for an observer with no local horizon (geocentric `500`, a registered
    /// spacecraft), where the elevation gate passes vacuously.
    pub has_elevation_deg: u8,
    /// Target elevation above the site's local horizon (degrees, geodetic
    /// zenith, refraction ignored). Meaningful only when
    /// [`has_elevation_deg`](Self::has_elevation_deg) is 1.
    pub elevation_deg: f64,
    /// 1 when [`sun_altitude_deg`](Self::sun_altitude_deg) is a measured
    /// value; 0 for a horizonless observer, where the darkness gate passes
    /// vacuously.
    pub has_sun_altitude_deg: u8,
    /// Altitude of the Sun's geometric centre above the site's local horizon
    /// (degrees). Meaningful only when
    /// [`has_sun_altitude_deg`](Self::has_sun_altitude_deg) is 1.
    pub sun_altitude_deg: f64,
    /// 1 when all four gates pass.
    pub observable: u8,
    /// 1 when the solar-elongation gate passes.
    pub elongation_ok: u8,
    /// 1 when [`mag`](Self::mag) carries a predicted apparent magnitude; 0
    /// when the orbit has no photometry — the magnitude gate is then NOT
    /// applicable (and [`magnitude_ok`](Self::magnitude_ok) is 0, never a
    /// vacuous pass).
    pub has_mag: u8,
    /// Predicted apparent magnitude. Meaningful only when
    /// [`has_mag`](Self::has_mag) is 1.
    pub mag: f64,
    /// 1 when the target is brighter than the limit AND a magnitude is known.
    /// A not-applicable magnitude does NOT pass vacuously.
    pub magnitude_ok: u8,
    /// 1 when the target is at/above the minimum elevation, or the observer
    /// has no local horizon.
    pub elevation_ok: u8,
    /// 1 when the Sun is at/below the darkness threshold, or the observer has
    /// no local horizon.
    pub darkness_ok: u8,
    /// 1 when the Moon altitude is measured; 0 for a horizonless observer.
    pub has_moon_altitude_deg: u8,
    /// Moon altitude above the site's local horizon (degrees, villeneuve's
    /// geocentric vertical). Meaningful only when
    /// [`has_moon_altitude_deg`](Self::has_moon_altitude_deg) is 1.
    pub moon_altitude_deg: f64,
    /// 1 when the Moon azimuth is measured; 0 for a horizonless observer.
    pub has_moon_azimuth_deg: u8,
    /// Moon azimuth (degrees, East of North). Meaningful only when
    /// [`has_moon_azimuth_deg`](Self::has_moon_azimuth_deg) is 1.
    pub moon_azimuth_deg: f64,
    /// 1 when the Moon illuminated fraction is available; 0 only in the
    /// degenerate observer-at-Moon case.
    pub has_moon_illuminated_fraction: u8,
    /// Sunlit fraction of the Moon's disc, \\(k = \tfrac{1}{2}(1+\cos\alpha)\\)
    /// from its phase angle. Meaningful only when
    /// [`has_moon_illuminated_fraction`](Self::has_moon_illuminated_fraction)
    /// is 1.
    pub moon_illuminated_fraction: f64,
    /// Target–observer–Moon angle (degrees), independent of any horizon.
    pub lunar_elongation_deg: f64,
}

/// A contiguous window of observability — mirrors
/// `scott::planning::VisibilityWindow`. Used both for the whole-survey
/// windows and for a [`EmpyreanNight`]'s intervals (whose boundaries are
/// bisection-refined to one second).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmpyreanVisibilityWindow {
    /// Window start — MJD, TDB.
    pub start_epoch_mjd_tdb: f64,
    /// Window end — MJD, TDB.
    pub end_epoch_mjd_tdb: f64,
    /// Peak solar elongation during the window (degrees).
    pub peak_elongation_deg: f64,
    /// 1 when [`brightest_mag`](Self::brightest_mag) is set.
    pub has_brightest_mag: u8,
    /// Brightest predicted magnitude during the window. Meaningful only when
    /// [`has_brightest_mag`](Self::has_brightest_mag) is 1.
    pub brightest_mag: f64,
    /// Length of the window in hours (`end − start`).
    pub hours_observable: f64,
    /// 1 when [`max_altitude_deg`](Self::max_altitude_deg) is set (absent for
    /// a horizonless observer's whole-survey window).
    pub has_max_altitude_deg: u8,
    /// Maximum target altitude in the window (degrees, geodetic vertical).
    pub max_altitude_deg: f64,
    /// 1 when [`max_altitude_epoch_mjd_tdb`](Self::max_altitude_epoch_mjd_tdb)
    /// is set.
    pub has_max_altitude_epoch: u8,
    /// Epoch of the maximum altitude — MJD, TDB.
    pub max_altitude_epoch_mjd_tdb: f64,
    /// The gate that closed the window — an `EMPYREAN_GATE_*` value
    /// (`EMPYREAN_GATE_NONE` when the engine reported none).
    pub closing_gate: u8,
}

/// The circumstances at a night anchor epoch — mirrors
/// `scott::planning::AnchorCircumstances`. Read off the single ephemeris row
/// generated at the anchor epoch (never interpolated). The target and Moon
/// altitudes are villeneuve's geocentric vertical; the Sun altitude is
/// scott's geodetic value (consistent with the darkness gate).
#[repr(C)]
pub struct EmpyreanAnchorCircumstances {
    /// Target altitude above the horizon (degrees, geocentric vertical).
    pub altitude_deg: f64,
    /// Target azimuth (degrees, East of North, geocentric vertical).
    pub azimuth_deg: f64,
    /// Sun altitude at the site (degrees, geodetic vertical).
    pub sun_altitude_deg: f64,
    /// Solar elongation (degrees).
    pub solar_elongation_deg: f64,
    /// Solar phase angle (degrees).
    pub phase_angle_deg: f64,
    /// Target–observer–Moon angle (degrees).
    pub lunar_elongation_deg: f64,
    /// 1 when the Moon altitude is set; 0 for a horizonless observer.
    pub has_moon_altitude_deg: u8,
    /// Moon altitude above the horizon (degrees, geocentric vertical).
    pub moon_altitude_deg: f64,
    /// 1 when the Moon illuminated fraction is set.
    pub has_moon_illuminated_fraction: u8,
    /// Moon illuminated fraction (dimensionless, [0, 1]).
    pub moon_illuminated_fraction: f64,
    /// 1 when [`mag`](Self::mag) is set (0 when the orbit has no photometry).
    pub has_mag: u8,
    /// Predicted apparent magnitude.
    pub mag: f64,
    /// 1 when [`sigma_mag`](Self::sigma_mag) is set.
    pub has_sigma_mag: u8,
    /// 1σ magnitude uncertainty.
    pub sigma_mag: f64,
    /// 1 when [`sky_rate`](Self::sky_rate) is set.
    pub has_sky_rate: u8,
    /// Total on-sky angular rate (degrees/day).
    pub sky_rate: f64,
    /// 1 when [`position_angle_deg`](Self::position_angle_deg) is set.
    pub has_position_angle_deg: u8,
    /// Position angle of motion (degrees, East of North).
    pub position_angle_deg: f64,
    /// 1 when [`hour_angle`](Self::hour_angle) is set; 0 for a horizonless
    /// observer.
    pub has_hour_angle: u8,
    /// Hour angle (degrees).
    pub hour_angle: f64,
    /// 1 when [`sky_covariance`](Self::sky_covariance) carries the row's sky
    /// covariance; 0 when the orbit carries no covariance.
    pub has_sky_covariance: u8,
    /// Derivation tag of the sky covariance — an `EMPYREAN_COVARIANCE_KIND_*`
    /// value; `EMPYREAN_COVARIANCE_KIND_UNKNOWN` when absent.
    pub sky_covariance_kind: u8,
    /// The row's sky covariance under the requested uncertainty method, as
    /// ONE self-describing packed joint — the same `EmpyreanPackedJoint`
    /// struct and lower-triangle packing the propagation / ephemeris rows
    /// carry. The absent joint (`width == 0`, null `tri`) when
    /// [`has_sky_covariance`](Self::has_sky_covariance) is 0. Freed by
    /// [`empyrean_visibility_result_free`].
    pub sky_covariance: EmpyreanPackedJoint,
}

/// The structure of one local night at one observatory — mirrors
/// `scott::planning::Night`. The two solar-noon epochs always carry a value;
/// the nine twilight / midnight epochs are `has_*` + value (a `None` crossing
/// is a polar case, or a surveyed span that did not reach it, with the reason
/// in [`note`](Self::note)).
#[repr(C)]
pub struct EmpyreanNight {
    /// Local civil date year.
    pub local_date_year: i64,
    /// Local civil date month (1–12).
    pub local_date_month: u32,
    /// Local civil date day (1–31).
    pub local_date_day: u32,
    /// Local solar noon that opens the night — MJD, TDB.
    pub noon_start_mjd_tdb: f64,
    /// Local solar noon that closes the night — MJD, TDB.
    pub noon_end_mjd_tdb: f64,
    /// 1 when the evening sunset crossing was found.
    pub has_sunset: u8,
    /// Evening crossing of the sunset threshold — MJD, TDB.
    pub sunset_mjd_tdb: f64,
    /// 1 when the evening civil-twilight crossing was found.
    pub has_civil_dusk: u8,
    /// Evening civil-twilight crossing — MJD, TDB.
    pub civil_dusk_mjd_tdb: f64,
    /// 1 when the evening nautical-twilight crossing was found.
    pub has_nautical_dusk: u8,
    /// Evening nautical-twilight crossing — MJD, TDB.
    pub nautical_dusk_mjd_tdb: f64,
    /// 1 when the evening astronomical-twilight crossing was found.
    pub has_astronomical_dusk: u8,
    /// Evening astronomical-twilight crossing — MJD, TDB.
    pub astronomical_dusk_mjd_tdb: f64,
    /// 1 when the local solar midnight was found.
    pub has_local_solar_midnight: u8,
    /// Local solar midnight (Sun-altitude minimum) — MJD, TDB.
    pub local_solar_midnight_mjd_tdb: f64,
    /// 1 when the morning astronomical-twilight crossing was found.
    pub has_astronomical_dawn: u8,
    /// Morning astronomical-twilight crossing — MJD, TDB.
    pub astronomical_dawn_mjd_tdb: f64,
    /// 1 when the morning nautical-twilight crossing was found.
    pub has_nautical_dawn: u8,
    /// Morning nautical-twilight crossing — MJD, TDB.
    pub nautical_dawn_mjd_tdb: f64,
    /// 1 when the morning civil-twilight crossing was found.
    pub has_civil_dawn: u8,
    /// Morning civil-twilight crossing — MJD, TDB.
    pub civil_dawn_mjd_tdb: f64,
    /// 1 when the morning sunrise crossing was found.
    pub has_sunrise: u8,
    /// Morning sunrise crossing — MJD, TDB.
    pub sunrise_mjd_tdb: f64,
    /// Observable intervals within the night (boundaries refined to one
    /// second). Owned array; freed by [`empyrean_visibility_result_free`].
    pub intervals: *mut EmpyreanVisibilityWindow,
    /// Number of entries in [`intervals`](Self::intervals).
    pub num_intervals: usize,
    /// Anchors under each policy. Owned array; freed by
    /// [`empyrean_visibility_result_free`].
    pub anchors: *mut EmpyreanAnchor,
    /// Number of entries in [`anchors`](Self::anchors).
    pub num_anchors: usize,
    /// Edge annotations (which thresholds are absent and why). Owned,
    /// null-terminated UTF-8, or null when every threshold was found. Freed
    /// by [`empyrean_visibility_result_free`].
    pub note: *mut c_char,
}

/// A night anchor: the epoch chosen under one policy and the circumstances
/// there — mirrors `scott::planning::Anchor`.
#[repr(C)]
pub struct EmpyreanAnchor {
    /// The policy that chose the epoch — an `EMPYREAN_ANCHOR_POLICY_*` value.
    pub policy: u8,
    /// 1 when [`epoch_mjd_tdb`](Self::epoch_mjd_tdb) is set (0 when the policy
    /// has no epoch this night, e.g. nearest-observable with no observable
    /// interval).
    pub has_epoch: u8,
    /// The chosen epoch — MJD, TDB. Meaningful only when
    /// [`has_epoch`](Self::has_epoch) is 1.
    pub epoch_mjd_tdb: f64,
    /// 1 when the target passes all four gates at the anchor epoch; 0 when
    /// there is no epoch.
    pub observable: u8,
    /// 1 when [`circumstances`](Self::circumstances) is populated (0 when
    /// there is no epoch).
    pub has_circumstances: u8,
    /// The circumstances at the anchor epoch. Meaningful only when
    /// [`has_circumstances`](Self::has_circumstances) is 1; its
    /// `sky_covariance` packed joint is freed by
    /// [`empyrean_visibility_result_free`].
    pub circumstances: EmpyreanAnchorCircumstances,
}

/// Per-observatory visibility data — mirrors
/// `scott::planning::ObservatoryVisibility`.
#[repr(C)]
pub struct EmpyreanObservatoryVisibility {
    /// Observatory code (owned; freed by [`empyrean_visibility_result_free`]).
    pub obs_code: *mut c_char,
    /// Per-epoch observability detail. Owned; freed on release.
    pub points: *mut EmpyreanVisibilityPoint,
    /// Number of entries in [`points`](Self::points).
    pub num_points: usize,
    /// Contiguous observable windows across the whole surveyed span. Owned.
    pub windows: *mut EmpyreanVisibilityWindow,
    /// Number of entries in [`windows`](Self::windows).
    pub num_windows: usize,
    /// One night per local night the surveyed span overlaps. EMPTY for an
    /// observer with no local horizon, with the reason in
    /// [`note`](Self::note). Owned.
    pub nights: *mut EmpyreanNight,
    /// Number of entries in [`nights`](Self::nights).
    pub num_nights: usize,
    /// Why [`nights`](Self::nights) is empty, when it is for a structural
    /// reason (a horizonless or space-based observer). Owned, null-terminated
    /// UTF-8, or null on a ground site whose nights were built normally.
    pub note: *mut c_char,
}

/// One orbit's visibility survey outcome — mirrors
/// `scott::planning::VisibilityResult`, plus the per-orbit delivery channel
/// (the engine returns each orbit's survey as its own `Result`).
///
/// The survey-wide provenance — the resolved covariance method and the
/// Earth-orientation summary — is carried here (as in the engine), where it
/// belongs to the whole survey rather than to one observatory.
#[repr(C)]
pub struct EmpyreanVisibilityResult {
    /// Orbit identifier (owned; freed by [`empyrean_visibility_result_free`]).
    pub orbit_id: *mut c_char,
    /// 1 when this orbit's survey was delivered; 0 when it failed (see
    /// [`error_code`](Self::error_code) / [`error_message`](Self::error_message)).
    pub delivered: u8,
    /// `EMPYREAN_PLANNING_ERROR_*` code — `_NONE` (0) when delivered.
    pub error_code: i32,
    /// The engine's typed failure message when not delivered; null when
    /// delivered. Owned; freed on release.
    pub error_message: *mut c_char,
    /// The covariance method every sky search box was produced under — an
    /// `EMPYREAN_COVARIANCE_KIND_*` value (never `Auto`). Meaningful only when
    /// delivered.
    pub resolved_method: u8,
    /// 1 when the Earth-orientation summary is present (an observer with a
    /// local horizon consulted the Earth orientation); 0 when no observer in
    /// the survey had a local horizon.
    pub has_earth_orientation: u8,
    /// Survey epochs resolved from the Earth-orientation kernels.
    pub earth_orientation_bpc_epochs: usize,
    /// Survey epochs served by the analytic IAU 2006 model (below the
    /// 1962-02-01 mandate floor) — NON-ZERO means degraded site geometry.
    pub earth_orientation_analytic_iau_2006_epochs: usize,
    /// Per-observatory visibility. Owned array; freed on release. Null/0 when
    /// not delivered.
    pub observatories: *mut EmpyreanObservatoryVisibility,
    /// Number of entries in [`observatories`](Self::observatories).
    pub num_observatories: usize,
}

/// The batch result of [`empyrean_compute_visibility`]: one
/// [`EmpyreanVisibilityResult`] per input orbit, in input order. The caller
/// allocates this holder; [`results`](Self::results) is heap-allocated here
/// and the whole tree is released with [`empyrean_visibility_result_free`].
#[repr(C)]
pub struct EmpyreanVisibilityBatch {
    /// One result per input orbit, in input order (so a result can be joined
    /// back to the orbit that produced it positionally). Owned.
    pub results: *mut EmpyreanVisibilityResult,
    /// Number of entries in [`results`](Self::results).
    pub num_results: usize,
}

impl Default for EmpyreanVisibilityBatch {
    fn default() -> Self {
        Self {
            results: std::ptr::null_mut(),
            num_results: 0,
        }
    }
}

// ── Conversions ─────────────────────────────────────────────────────

fn nan_to_opt(v: f64) -> Option<f64> {
    if v.is_nan() { None } else { Some(v) }
}

fn cstr_to_string(p: *const c_char, what: &str) -> Result<String, String> {
    if p.is_null() {
        return Err(format!("{what}: null string pointer"));
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("{what}: invalid UTF-8: {e}"))
}

fn radar_station_preset(code: u8, role: &str) -> Result<Station, String> {
    match code {
        0 => Ok(Station::goldstone_dss14()),
        1 => Ok(Station::green_bank_receive()),
        2 => Ok(Station::arecibo_defunct()),
        other => Err(format!(
            "unknown radar {role} station preset {other} (0=Goldstone, 1=GreenBank, 2=Arecibo)"
        )),
    }
}

fn radar_mode_from_u8(m: u8) -> Result<RadarMode, String> {
    match m {
        0 => Ok(RadarMode::Delay),
        1 => Ok(RadarMode::Doppler),
        2 => Ok(RadarMode::Both),
        other => Err(format!(
            "unknown radar mode {other} (0=Delay, 1=Doppler, 2=Both)"
        )),
    }
}

fn build_planned(c: &EmpyreanPlannedObservation) -> Result<PlannedObservation, String> {
    let epoch = Epoch::from_mjd_tdb(c.epoch_mjd_tdb);
    match c.kind {
        0 => {
            let code = cstr_to_string(c.optical_code, "optical station code")?;
            Ok(PlannedObservation::optical(
                Station::optical(
                    code,
                    [c.optical_sigma_ra_arcsec, c.optical_sigma_dec_arcsec],
                ),
                epoch,
            ))
        }
        1 => {
            let tx = radar_station_preset(c.radar_transmit_station, "transmit")?;
            let rx = radar_station_preset(c.radar_receive_station, "receive")?;
            let mode = radar_mode_from_u8(c.radar_mode)?;
            let spec = if c.radar_snr.is_nan() {
                let target = TargetRadarProperties {
                    diameter_km: nan_to_opt(c.radar_target_diameter_km),
                    h_mag: nan_to_opt(c.radar_target_h_mag),
                    visual_albedo: nan_to_opt(c.radar_target_visual_albedo),
                    radar_albedo: nan_to_opt(c.radar_target_radar_albedo),
                    spin_period_hours: nan_to_opt(c.radar_target_spin_period_hours),
                };
                RadarPlanSpec::link_budget(
                    tx,
                    rx,
                    target,
                    c.radar_integration_s,
                    mode,
                    c.radar_bandwidth_hz,
                    c.radar_freq_resolution_hz,
                )
            } else {
                RadarPlanSpec::given(
                    tx,
                    rx,
                    mode,
                    c.radar_bandwidth_hz,
                    c.radar_freq_resolution_hz,
                    c.radar_snr,
                )
            };
            Ok(PlannedObservation::radar(spec, epoch))
        }
        other => Err(format!(
            "unknown planned-observation kind {other} (0=optical, 1=radar)"
        )),
    }
}

/// The engine's own darkness threshold, read off the engine rather than
/// restated here.
///
/// `EmpyreanObservatoryConfig::has_max_sun_altitude_deg = 0` means "use
/// the engine default", and the honest way to honour that is to ask the
/// engine what its default is. A literal `-18.0` in this crate would be
/// a second copy of a number the engine owns, with nothing linking them:
/// the day the convention moves, the C channel would keep planning to
/// the old one and say nothing.
fn engine_default_max_sun_altitude_deg() -> f64 {
    ObservatoryConfig::new("", [0.0, 0.0]).max_sun_altitude_deg
}

fn build_planning_config(c: &EmpyreanPlanningConfig) -> Result<PlanningConfig, String> {
    let force_model = int_to_force_model(c.force_model)?.into();

    let obs_slice: &[EmpyreanObservatoryConfig] =
        if c.observatories.is_null() || c.num_observatories == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(c.observatories, c.num_observatories) }
        };
    let mut observatories = Vec::with_capacity(obs_slice.len());
    for o in obs_slice {
        let code = cstr_to_string(o.obs_code, "observatory code")?;
        // Every field is set explicitly rather than by struct update
        // from a default, so a field the engine grows is a compile break
        // here — at the marshal boundary — instead of a silently
        // defaulted visibility gate.
        // The night-product twilight thresholds default — as a group — to
        // the engine's own values when the caller does not opt in, read off
        // a fresh engine config rather than restated as literals so a moved
        // convention is a build-time, not a silent, change. `min_elevation_deg`
        // and `max_sun_altitude_deg` are now READ by the visibility survey's
        // elevation and darkness gates (they were marshaled but gate-less on
        // the `evaluate_plan` seam).
        let engine_defaults =
            ObservatoryConfig::new(&code, [o.sigma_ra_arcsec, o.sigma_dec_arcsec]);
        let use_twilight = o.has_twilight_thresholds != 0;
        observatories.push(ObservatoryConfig {
            obs_code: code,
            sigma_arcsec: [o.sigma_ra_arcsec, o.sigma_dec_arcsec],
            max_apparent_mag: o.max_apparent_mag,
            min_elongation_deg: o.min_elongation_deg,
            min_elevation_deg: o.min_elevation_deg,
            max_sun_altitude_deg: if o.has_max_sun_altitude_deg != 0 {
                o.max_sun_altitude_deg
            } else {
                engine_default_max_sun_altitude_deg()
            },
            sunset_sun_altitude_deg: if use_twilight {
                o.sunset_sun_altitude_deg
            } else {
                engine_defaults.sunset_sun_altitude_deg
            },
            civil_twilight_sun_altitude_deg: if use_twilight {
                o.civil_twilight_sun_altitude_deg
            } else {
                engine_defaults.civil_twilight_sun_altitude_deg
            },
            nautical_twilight_sun_altitude_deg: if use_twilight {
                o.nautical_twilight_sun_altitude_deg
            } else {
                engine_defaults.nautical_twilight_sun_altitude_deg
            },
            astronomical_twilight_sun_altitude_deg: if use_twilight {
                o.astronomical_twilight_sun_altitude_deg
            } else {
                engine_defaults.astronomical_twilight_sun_altitude_deg
            },
            explicit_anchor_utc: if o.has_explicit_anchor_utc != 0 {
                Some(Epoch::from_mjd_tdb(o.explicit_anchor_utc_mjd_tdb))
            } else {
                None
            },
        });
    }

    // Five fields the villeneuve 1.26.0 / scott 1.19.0 pin added to the
    // engine `PlanningConfig` (`max_light_time_iterations`,
    // `excluded_perturbers`, `origin_switching`, `output_epoch`,
    // `transport_cache`). This ABI exposes no knob for any of them yet — the
    // same deferral as `uncertainty_method` / `weighting` below — so each
    // is pinned to the engine's own documented planning default, read
    // field-by-field from `PlanningConfig::default()`
    // rather than by a `..Default::default()` struct-update, so a field the
    // engine grows next is a compile break here (at the marshal boundary)
    // rather than a silently defaulted plan. Four of the five preserve the
    // behaviour of a seam that never surfaced them —
    // `max_light_time_iterations = 3`, no perturber exclusions, villeneuve's
    // library origin-switching, and no transport cache. The fifth,
    // `output_epoch = LastObservation`, is NOT behaviour-preserving relative
    // to the prior (IOD) epoch: it transports the plan's prior and posterior
    // metrics to the last planned observation, so the covariance-bearing
    // fields (and the per-candidate `cumulative`) are reported at that epoch,
    // not at the input orbit's. Under the `Linear` transport the engine
    // delivers the transported state-block `log_det` there; `log_det_gain`
    // stays the epoch-invariant prior→posterior difference, formed once at
    // the prior epoch.
    let engine_defaults = PlanningConfig::default();
    Ok(PlanningConfig {
        force_model,
        epsilon: c.epsilon,
        max_light_time_iterations: engine_defaults.max_light_time_iterations,
        excluded_perturbers: engine_defaults.excluded_perturbers,
        origin_switching: engine_defaults.origin_switching,
        output_epoch: engine_defaults.output_epoch,
        transport_cache: engine_defaults.transport_cache,
        observatories,
        num_threads: if c.num_threads < 0 {
            None
        } else {
            Some(c.num_threads as usize)
        },
        encounter: None,
        // `uncertainty_method` is now a C-ABI knob: the propagation seam's
        // flat tag carrier, decoded through the same `flat_to_uncertainty_method`
        // and converted to the engine enum. A zero-init carrier decodes to
        // `First`, the single-linearization Gaussian fold every previous
        // release of `empyrean_evaluate_plan` computed, so an unset field is
        // behaviour-preserving. A `Mixture` request is refused by name inside
        // the fold; every other method is honoured and its resolved kind rides
        // out on `EmpyreanPlanResult::resolved_method`.
        //
        // `weighting` stays pinned (not a `Default`, and no C knob yet):
        // `None` is the raw per-candidate σ this seam has always consumed
        // unmodified, and the engine's default may move.
        uncertainty_method: flat_to_uncertainty_method(&c.uncertainty_method)?.into(),
        weighting: None,
    })
}

fn metrics_to_c(m: &CovarianceMetrics) -> EmpyreanCovarianceMetrics {
    // `CovarianceMetrics` carries the absolute `log_det` under the
    // `Linear` transport (and at the prior epoch) and withholds it under a
    // nonlinear one, naming the transport on `log_det_withheld` rather
    // than leaving a bare `None` — exactly one of the two is `Some`. The
    // reason tag is authoritative: when it is present the value is `NaN`
    // and the tag says why; when it is absent the value is delivered. So
    // a `NaN` on this struct never arrives without a non-`NONE`
    // `log_det_withheld` beside it to explain it.
    let (log_det, withheld, withheld_method) = match m.log_det_withheld {
        None => (
            m.log_det.unwrap_or(f64::NAN),
            EMPYREAN_LOG_DET_WITHHELD_NONE,
            EMPYREAN_COVARIANCE_KIND_UNKNOWN,
        ),
        Some(LogDetWithheld::SecondOrderMoment) => (
            f64::NAN,
            EMPYREAN_LOG_DET_WITHHELD_SECOND_ORDER_MOMENT,
            EMPYREAN_COVARIANCE_KIND_UNKNOWN,
        ),
        Some(LogDetWithheld::SampledMoment { method }) => (
            f64::NAN,
            EMPYREAN_LOG_DET_WITHHELD_SAMPLED_MOMENT,
            covariance_kind_to_u8(method),
        ),
        Some(LogDetWithheld::MixtureMoment) => (
            f64::NAN,
            EMPYREAN_LOG_DET_WITHHELD_MIXTURE_MOMENT,
            EMPYREAN_COVARIANCE_KIND_UNKNOWN,
        ),
    };
    EmpyreanCovarianceMetrics {
        position_sigma_km: m.position_sigma_km,
        velocity_sigma_m_s: m.velocity_sigma_m_s,
        semi_major_km: m.semi_major_km,
        semi_minor_km: m.semi_minor_km,
        log_det,
        log_det_withheld: withheld,
        log_det_withheld_method: withheld_method,
    }
}

fn owned_cstr(s: &str) -> *mut c_char {
    CString::new(s)
        .unwrap_or_else(|_| CString::new("").unwrap())
        .into_raw()
}

fn candidate_to_c(c: &CandidateInfo) -> EmpyreanPlanCandidate {
    // Radar candidates carry the link-budget outputs the fit's
    // measurement σ was derived from; pass them through verbatim
    // (provenance notes as owned strings — a C-side code mapping would
    // silently drop future engine notes).
    let (kind, radar_snr, radar_range_km, radar_mode, prov) = match &c.kind {
        CandidateKind::Optical => (0u8, f64::NAN, f64::NAN, -1i32, None),
        CandidateKind::Radar {
            mode,
            snr,
            range_km,
            provenance,
        } => {
            let mode_int = match mode {
                RadarMode::Delay => 0i32,
                RadarMode::Doppler => 1i32,
                RadarMode::Both => 2i32,
            };
            (1u8, *snr, *range_km, mode_int, Some(provenance))
        }
    };
    let (radar_provenance, num_radar_provenance) = match prov {
        Some(p) if !p.is_empty() => {
            let v: Vec<*mut c_char> = p.iter().map(|s| owned_cstr(s)).collect();
            let n = v.len();
            (Box::into_raw(v.into_boxed_slice()) as *mut *mut c_char, n)
        }
        _ => (std::ptr::null_mut(), 0),
    };
    EmpyreanPlanCandidate {
        index: c.index,
        obs_code: owned_cstr(&c.obs_code),
        kind,
        observable: u8::from(c.observable),
        along_track_sigma_arcsec: c.along_track_sigma_arcsec,
        cross_track_sigma_arcsec: c.cross_track_sigma_arcsec,
        ra_sigma_arcsec: c.ra_sigma_arcsec,
        dec_sigma_arcsec: c.dec_sigma_arcsec,
        position_angle_deg: c.position_angle_deg,
        marginal_volume_reduction: c.marginal_volume_reduction,
        marginal_position_improvement: c.marginal_position_improvement,
        post_along_track_sigma_arcsec: c.post_along_track_sigma_arcsec,
        post_cross_track_sigma_arcsec: c.post_cross_track_sigma_arcsec,
        cumulative: metrics_to_c(&c.cumulative),
        active_width: c.active_width,
        radar_snr,
        radar_range_km,
        radar_provenance,
        num_radar_provenance,
        radar_mode,
    }
}

fn plan_result_to_c(plan: &PlanResult, out: *mut EmpyreanPlanResult) {
    let candidates: Vec<EmpyreanPlanCandidate> =
        plan.candidates.iter().map(candidate_to_c).collect();
    let num_candidates = candidates.len();
    let cand_ptr = if num_candidates == 0 {
        std::ptr::null_mut()
    } else {
        Box::into_raw(candidates.into_boxed_slice()) as *mut EmpyreanPlanCandidate
    };
    // Predicted optical ephemeris: one (epoch, RA, Dec) point per
    // optical candidate, chronological. Absent (radar-only plan) →
    // null / 0.
    let eph_points: Vec<EmpyreanPlanEphemerisPoint> = plan
        .ephemeris
        .as_ref()
        .map(|e| {
            e.coordinates()
                .iter()
                .map(|c| EmpyreanPlanEphemerisPoint {
                    epoch_mjd_tdb: c.t.mjd_tdb(),
                    ra_deg: c.lon,
                    dec_deg: c.lat,
                })
                .collect()
        })
        .unwrap_or_default();
    let num_ephemeris = eph_points.len();
    let eph_ptr = if num_ephemeris == 0 {
        std::ptr::null_mut()
    } else {
        Box::into_raw(eph_points.into_boxed_slice()) as *mut EmpyreanPlanEphemerisPoint
    };
    unsafe {
        (*out).orbit_id = owned_cstr(&plan.orbit_id);
        (*out).prior = metrics_to_c(&plan.prior);
        (*out).posterior = metrics_to_c(&plan.posterior);
        (*out).log_det_gain = plan.log_det_gain;
        (*out).candidates = cand_ptr;
        (*out).num_candidates = num_candidates;
        (*out).active_width = plan.active_width;
        (*out).ephemeris = eph_ptr;
        (*out).num_ephemeris = num_ephemeris;
        // The kind the fold resolved to, in the covariance-kind vocabulary
        // — `Auto` already collapsed to the kind it ran. Passed through
        // verbatim so the search box's resolved kind and the metrics'
        // `log_det_withheld` beside it never disagree.
        (*out).resolved_method = covariance_kind_to_u8(plan.resolved_method);
    }
}

// ── Entry points ────────────────────────────────────────────────────

/// Evaluate an observation plan: how much each candidate observation would
/// tighten the prior orbit covariance.
///
/// `orbit` must carry a 6×6 Cartesian covariance (e.g. a `determine` result).
/// `planned` is an array of `num_planned` candidate observations. `orbit_id`
/// may be null (defaults to `"orbit_0"`). On success populates `*result_out`
/// (caller-allocated); free with [`empyrean_plan_result_free`].
///
/// Returns 0 on success, -1 for a null/invalid argument, -3 if planning fails
/// (missing/singular prior covariance, an infeasible or invalid candidate, an
/// ephemeris-generation error), -99 on an internal panic. The error message is
/// retrievable via `empyrean_last_error()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_evaluate_plan(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    orbit_id: *const c_char,
    planned: *const EmpyreanPlannedObservation,
    num_planned: usize,
    config: *const EmpyreanPlanningConfig,
    result_out: *mut EmpyreanPlanResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbit.is_null()
            || planned.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let orbit_ref = unsafe { &*orbit };
        let cfg_ref = unsafe { &*config };
        let planned_slice = unsafe { std::slice::from_raw_parts(planned, num_planned) };

        let id = if orbit_id.is_null() {
            "orbit_0".to_string()
        } else {
            match cstr_to_string(orbit_id, "orbit_id") {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        };

        let orbits = match empyrean_orbit_to_orbits(orbit_ref, &id) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e.message);
                return e.code.unwrap_or(-1);
            }
        };

        let mut planned_vec = Vec::with_capacity(planned_slice.len());
        for p in planned_slice {
            match build_planned(p) {
                Ok(po) => planned_vec.push(po),
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        }

        let plan_cfg = match build_planning_config(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let plan = match evaluate_plan_single(ctx_ref, &orbits, &planned_vec, &plan_cfg) {
            Ok(p) => p,
            Err(e) => {
                set_last_error(&format!("planning failed: {e}"));
                return -3;
            }
        };

        plan_result_to_c(&plan, result_out);
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_evaluate_plan");
            -99
        }
    }
}

/// Free the heap allocations inside a plan result populated by
/// [`empyrean_evaluate_plan`]. Does not free the caller-allocated struct.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_plan_result_free(result: *mut EmpyreanPlanResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &*result };
        if !res.orbit_id.is_null() {
            drop(unsafe { CString::from_raw(res.orbit_id) });
        }
        if !res.candidates.is_null() && res.num_candidates > 0 {
            let cands = unsafe {
                Vec::from_raw_parts(res.candidates, res.num_candidates, res.num_candidates)
            };
            for c in &cands {
                if !c.obs_code.is_null() {
                    drop(unsafe { CString::from_raw(c.obs_code) });
                }
                if !c.radar_provenance.is_null() && c.num_radar_provenance > 0 {
                    let strs = unsafe {
                        Vec::from_raw_parts(
                            c.radar_provenance,
                            c.num_radar_provenance,
                            c.num_radar_provenance,
                        )
                    };
                    for s in strs {
                        if !s.is_null() {
                            drop(unsafe { CString::from_raw(s) });
                        }
                    }
                }
            }
            drop(cands);
        }
        if !res.ephemeris.is_null() && res.num_ephemeris > 0 {
            drop(unsafe {
                Vec::from_raw_parts(res.ephemeris, res.num_ephemeris, res.num_ephemeris)
            });
        }
        unsafe {
            (*result).orbit_id = std::ptr::null_mut();
            (*result).candidates = std::ptr::null_mut();
            (*result).num_candidates = 0;
            (*result).ephemeris = std::ptr::null_mut();
            (*result).num_ephemeris = 0;
        }
    }));
}

// ── Visibility marshaling ───────────────────────────────────────────

/// `Some(v)` → `(1, v)`, `None` → `(0, 0.0)`. The flag carries the
/// not-applicable-vs-measured distinction a NaN sentinel could not.
fn opt_f64_to_c(o: Option<f64>) -> (u8, f64) {
    match o {
        Some(v) => (1, v),
        None => (0, 0.0),
    }
}

/// An optional engine epoch → `(has, mjd_tdb)`.
fn opt_epoch_to_c(o: Option<Epoch>) -> (u8, f64) {
    match o {
        Some(e) => (1, e.mjd_tdb()),
        None => (0, 0.0),
    }
}

/// An owned C string for `Some`, null for `None`.
fn owned_cstr_opt(o: Option<&str>) -> *mut c_char {
    match o {
        Some(s) => owned_cstr(s),
        None => std::ptr::null_mut(),
    }
}

/// `Option<Gate>` → an `EMPYREAN_GATE_*` byte (`_NONE` for `None`).
fn gate_to_c(g: Option<Gate>) -> u8 {
    match g {
        None => EMPYREAN_GATE_NONE,
        Some(Gate::Elongation) => EMPYREAN_GATE_ELONGATION,
        Some(Gate::Magnitude) => EMPYREAN_GATE_MAGNITUDE,
        Some(Gate::Elevation) => EMPYREAN_GATE_ELEVATION,
        Some(Gate::Darkness) => EMPYREAN_GATE_DARKNESS,
        Some(Gate::EndOfNight) => EMPYREAN_GATE_END_OF_NIGHT,
    }
}

/// `AnchorPolicy` → an `EMPYREAN_ANCHOR_POLICY_*` byte (the `ExplicitUTC`
/// epoch itself rides on [`EmpyreanAnchor::epoch_mjd_tdb`]).
fn anchor_policy_to_c(p: &AnchorPolicy) -> u8 {
    match p {
        AnchorPolicy::NearestObservableToMidnight => {
            EMPYREAN_ANCHOR_POLICY_NEAREST_OBSERVABLE_TO_MIDNIGHT
        }
        AnchorPolicy::Midnight => EMPYREAN_ANCHOR_POLICY_MIDNIGHT,
        AnchorPolicy::Transit => EMPYREAN_ANCHOR_POLICY_TRANSIT,
        AnchorPolicy::ExplicitUTC(_) => EMPYREAN_ANCHOR_POLICY_EXPLICIT_UTC,
    }
}

fn point_to_c(p: &VisibilityPoint) -> EmpyreanVisibilityPoint {
    let (has_elevation_deg, elevation_deg) = opt_f64_to_c(p.elevation_deg);
    let (has_sun_altitude_deg, sun_altitude_deg) = opt_f64_to_c(p.sun_altitude_deg);
    let (has_mag, mag) = opt_f64_to_c(p.mag);
    let (has_moon_altitude_deg, moon_altitude_deg) = opt_f64_to_c(p.moon_altitude_deg);
    let (has_moon_azimuth_deg, moon_azimuth_deg) = opt_f64_to_c(p.moon_azimuth_deg);
    let (has_moon_illuminated_fraction, moon_illuminated_fraction) =
        opt_f64_to_c(p.moon_illuminated_fraction);
    EmpyreanVisibilityPoint {
        epoch_mjd_tdb: p.epoch.mjd_tdb(),
        solar_elongation_deg: p.solar_elongation_deg,
        has_elevation_deg,
        elevation_deg,
        has_sun_altitude_deg,
        sun_altitude_deg,
        observable: u8::from(p.observable),
        elongation_ok: u8::from(p.elongation_ok),
        has_mag,
        mag,
        magnitude_ok: u8::from(p.magnitude_ok),
        elevation_ok: u8::from(p.elevation_ok),
        darkness_ok: u8::from(p.darkness_ok),
        has_moon_altitude_deg,
        moon_altitude_deg,
        has_moon_azimuth_deg,
        moon_azimuth_deg,
        has_moon_illuminated_fraction,
        moon_illuminated_fraction,
        lunar_elongation_deg: p.lunar_elongation_deg,
    }
}

fn window_to_c(w: &VisibilityWindow) -> EmpyreanVisibilityWindow {
    let (has_brightest_mag, brightest_mag) = opt_f64_to_c(w.brightest_mag);
    let (has_max_altitude_deg, max_altitude_deg) = opt_f64_to_c(w.max_altitude_deg);
    let (has_max_altitude_epoch, max_altitude_epoch_mjd_tdb) = opt_epoch_to_c(w.max_altitude_epoch);
    EmpyreanVisibilityWindow {
        start_epoch_mjd_tdb: w.start_epoch.mjd_tdb(),
        end_epoch_mjd_tdb: w.end_epoch.mjd_tdb(),
        peak_elongation_deg: w.peak_elongation_deg,
        has_brightest_mag,
        brightest_mag,
        hours_observable: w.hours_observable,
        has_max_altitude_deg,
        max_altitude_deg,
        has_max_altitude_epoch,
        max_altitude_epoch_mjd_tdb,
        closing_gate: gate_to_c(w.closing_gate),
    }
}

/// Marshal an anchor's circumstances, moving the sky covariance into an
/// owned packed joint through the shared `sky_uncertainty_to_c` path (the
/// same struct and lower-triangle packing the ephemeris rows carry). The
/// `None` covariance is the absent packed joint.
fn circ_to_c(c: &AnchorCircumstances) -> EmpyreanAnchorCircumstances {
    let (has_moon_altitude_deg, moon_altitude_deg) = opt_f64_to_c(c.moon_altitude_deg);
    let (has_moon_illuminated_fraction, moon_illuminated_fraction) =
        opt_f64_to_c(c.moon_illuminated_fraction);
    let (has_mag, mag) = opt_f64_to_c(c.mag);
    let (has_sigma_mag, sigma_mag) = opt_f64_to_c(c.sigma_mag);
    let (has_sky_rate, sky_rate) = opt_f64_to_c(c.sky_rate);
    let (has_position_angle_deg, position_angle_deg) = opt_f64_to_c(c.position_angle_deg);
    let (has_hour_angle, hour_angle) = opt_f64_to_c(c.hour_angle);
    let (has_sky_covariance, sky_covariance_kind, sky_covariance) = match &c.sky_covariance {
        Some(unc) => (
            1u8,
            covariance_kind_to_u8(unc.kind()),
            sky_uncertainty_to_c(unc),
        ),
        None => (0u8, EMPYREAN_COVARIANCE_KIND_UNKNOWN, empty_packed_joint()),
    };
    EmpyreanAnchorCircumstances {
        altitude_deg: c.altitude_deg,
        azimuth_deg: c.azimuth_deg,
        sun_altitude_deg: c.sun_altitude_deg,
        solar_elongation_deg: c.solar_elongation_deg,
        phase_angle_deg: c.phase_angle_deg,
        lunar_elongation_deg: c.lunar_elongation_deg,
        has_moon_altitude_deg,
        moon_altitude_deg,
        has_moon_illuminated_fraction,
        moon_illuminated_fraction,
        has_mag,
        mag,
        has_sigma_mag,
        sigma_mag,
        has_sky_rate,
        sky_rate,
        has_position_angle_deg,
        position_angle_deg,
        has_hour_angle,
        hour_angle,
        has_sky_covariance,
        sky_covariance_kind,
        sky_covariance,
    }
}

/// The absent circumstances (an anchor with no epoch): every value zeroed,
/// every `has_*` flag off, and the absent packed joint.
fn empty_circ() -> EmpyreanAnchorCircumstances {
    EmpyreanAnchorCircumstances {
        altitude_deg: 0.0,
        azimuth_deg: 0.0,
        sun_altitude_deg: 0.0,
        solar_elongation_deg: 0.0,
        phase_angle_deg: 0.0,
        lunar_elongation_deg: 0.0,
        has_moon_altitude_deg: 0,
        moon_altitude_deg: 0.0,
        has_moon_illuminated_fraction: 0,
        moon_illuminated_fraction: 0.0,
        has_mag: 0,
        mag: 0.0,
        has_sigma_mag: 0,
        sigma_mag: 0.0,
        has_sky_rate: 0,
        sky_rate: 0.0,
        has_position_angle_deg: 0,
        position_angle_deg: 0.0,
        has_hour_angle: 0,
        hour_angle: 0.0,
        has_sky_covariance: 0,
        sky_covariance_kind: EMPYREAN_COVARIANCE_KIND_UNKNOWN,
        sky_covariance: empty_packed_joint(),
    }
}

fn anchor_to_c(a: &Anchor) -> EmpyreanAnchor {
    let (has_epoch, epoch_mjd_tdb) = opt_epoch_to_c(a.epoch);
    let (has_circumstances, circumstances) = match &a.circumstances {
        Some(c) => (1u8, circ_to_c(c)),
        None => (0u8, empty_circ()),
    };
    EmpyreanAnchor {
        policy: anchor_policy_to_c(&a.policy),
        has_epoch,
        epoch_mjd_tdb,
        observable: u8::from(a.observable),
        has_circumstances,
        circumstances,
    }
}

fn windows_to_c_ptr(ws: &[VisibilityWindow]) -> (*mut EmpyreanVisibilityWindow, usize) {
    if ws.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let v: Vec<EmpyreanVisibilityWindow> = ws.iter().map(window_to_c).collect();
    let n = v.len();
    (
        Box::into_raw(v.into_boxed_slice()) as *mut EmpyreanVisibilityWindow,
        n,
    )
}

fn night_to_c(n: &Night) -> EmpyreanNight {
    let (has_sunset, sunset_mjd_tdb) = opt_epoch_to_c(n.sunset);
    let (has_civil_dusk, civil_dusk_mjd_tdb) = opt_epoch_to_c(n.civil_dusk);
    let (has_nautical_dusk, nautical_dusk_mjd_tdb) = opt_epoch_to_c(n.nautical_dusk);
    let (has_astronomical_dusk, astronomical_dusk_mjd_tdb) = opt_epoch_to_c(n.astronomical_dusk);
    let (has_local_solar_midnight, local_solar_midnight_mjd_tdb) =
        opt_epoch_to_c(n.local_solar_midnight);
    let (has_astronomical_dawn, astronomical_dawn_mjd_tdb) = opt_epoch_to_c(n.astronomical_dawn);
    let (has_nautical_dawn, nautical_dawn_mjd_tdb) = opt_epoch_to_c(n.nautical_dawn);
    let (has_civil_dawn, civil_dawn_mjd_tdb) = opt_epoch_to_c(n.civil_dawn);
    let (has_sunrise, sunrise_mjd_tdb) = opt_epoch_to_c(n.sunrise);
    let (intervals, num_intervals) = windows_to_c_ptr(&n.intervals);
    let (anchors, num_anchors) = if n.anchors.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let v: Vec<EmpyreanAnchor> = n.anchors.iter().map(anchor_to_c).collect();
        let len = v.len();
        (
            Box::into_raw(v.into_boxed_slice()) as *mut EmpyreanAnchor,
            len,
        )
    };
    let (year, month, day) = n.local_date;
    EmpyreanNight {
        local_date_year: year,
        local_date_month: month,
        local_date_day: day,
        noon_start_mjd_tdb: n.noon_start.mjd_tdb(),
        noon_end_mjd_tdb: n.noon_end.mjd_tdb(),
        has_sunset,
        sunset_mjd_tdb,
        has_civil_dusk,
        civil_dusk_mjd_tdb,
        has_nautical_dusk,
        nautical_dusk_mjd_tdb,
        has_astronomical_dusk,
        astronomical_dusk_mjd_tdb,
        has_local_solar_midnight,
        local_solar_midnight_mjd_tdb,
        has_astronomical_dawn,
        astronomical_dawn_mjd_tdb,
        has_nautical_dawn,
        nautical_dawn_mjd_tdb,
        has_civil_dawn,
        civil_dawn_mjd_tdb,
        has_sunrise,
        sunrise_mjd_tdb,
        intervals,
        num_intervals,
        anchors,
        num_anchors,
        note: owned_cstr_opt(n.note.as_deref()),
    }
}

fn obs_vis_to_c(ov: &ObservatoryVisibility) -> EmpyreanObservatoryVisibility {
    let (points, num_points) = if ov.points.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let v: Vec<EmpyreanVisibilityPoint> = ov.points.iter().map(point_to_c).collect();
        let n = v.len();
        (
            Box::into_raw(v.into_boxed_slice()) as *mut EmpyreanVisibilityPoint,
            n,
        )
    };
    let (windows, num_windows) = windows_to_c_ptr(&ov.windows);
    let (nights, num_nights) = if ov.nights.is_empty() {
        (std::ptr::null_mut(), 0)
    } else {
        let v: Vec<EmpyreanNight> = ov.nights.iter().map(night_to_c).collect();
        let n = v.len();
        (Box::into_raw(v.into_boxed_slice()) as *mut EmpyreanNight, n)
    };
    EmpyreanObservatoryVisibility {
        obs_code: owned_cstr(&ov.obs_code),
        points,
        num_points,
        windows,
        num_windows,
        nights,
        num_nights,
        note: owned_cstr_opt(ov.note.as_deref()),
    }
}

/// Map a [`PlanningError`] variant to its stable `EMPYREAN_PLANNING_ERROR_*`
/// code. The match is EXHAUSTIVE and names every variant — a variant the
/// engine grows is a compile break here, not a silent miscode. Variants
/// outside the visibility survey's reachable set (the fold, encounter,
/// output-epoch, second-order and sampled-rung refusals that
/// `compute_visibility` does not exercise) collapse to
/// `EMPYREAN_PLANNING_ERROR_OTHER`; the full typed cause always rides on the
/// result's `error_message`, so nothing is hidden.
fn planning_error_code(e: &PlanningError) -> i32 {
    match e {
        PlanningError::InvalidTimeWindow(_) => EMPYREAN_PLANNING_ERROR_INVALID_TIME_WINDOW,
        PlanningError::ObserverConstruction { .. } => EMPYREAN_PLANNING_ERROR_OBSERVER_CONSTRUCTION,
        PlanningError::InvalidStationCode { .. } => EMPYREAN_PLANNING_ERROR_INVALID_STATION_CODE,
        PlanningError::EphemerisGeneration(_) => EMPYREAN_PLANNING_ERROR_EPHEMERIS_GENERATION,
        PlanningError::OrbitPropagationFailed { .. } => {
            EMPYREAN_PLANNING_ERROR_ORBIT_PROPAGATION_FAILED
        }
        PlanningError::OrbitEphemerisFailed { .. } => {
            EMPYREAN_PLANNING_ERROR_ORBIT_EPHEMERIS_FAILED
        }
        PlanningError::CandidateCovarianceWithheld { .. } => {
            EMPYREAN_PLANNING_ERROR_CANDIDATE_COVARIANCE_WITHHELD
        }
        PlanningError::ThreadBudget(_) => EMPYREAN_PLANNING_ERROR_THREAD_BUDGET,
        PlanningError::NoObservatories => EMPYREAN_PLANNING_ERROR_NO_OBSERVATORIES,
        PlanningError::UnsupportedUncertaintyMethod { .. } => {
            EMPYREAN_PLANNING_ERROR_UNSUPPORTED_UNCERTAINTY_METHOD
        }
        PlanningError::EarthOrientationUnavailable => {
            EMPYREAN_PLANNING_ERROR_EARTH_ORIENTATION_UNAVAILABLE
        }
        PlanningError::EarthOrientationEvaluation { .. } => {
            EMPYREAN_PLANNING_ERROR_EARTH_ORIENTATION_EVALUATION
        }
        PlanningError::BodyStateUnavailable { .. } => {
            EMPYREAN_PLANNING_ERROR_BODY_STATE_UNAVAILABLE
        }
        PlanningError::DegenerateSiteGeometry { .. } => {
            EMPYREAN_PLANNING_ERROR_DEGENERATE_SITE_GEOMETRY
        }
        PlanningError::VisibilityGridMismatch { .. } => {
            EMPYREAN_PLANNING_ERROR_VISIBILITY_GRID_MISMATCH
        }
        PlanningError::MissingObservabilityDiagnostic { .. } => {
            EMPYREAN_PLANNING_ERROR_MISSING_OBSERVABILITY_DIAGNOSTIC
        }
        PlanningError::AstrometryFrameNotICRF { .. } => {
            EMPYREAN_PLANNING_ERROR_ASTROMETRY_FRAME_NOT_ICRF
        }
        PlanningError::SPKSubstitutedBody { .. } => EMPYREAN_PLANNING_ERROR_SPK_SUBSTITUTED_BODY,
        PlanningError::NonFinitePriorCovariance { .. } => {
            EMPYREAN_PLANNING_ERROR_NON_FINITE_PRIOR_COVARIANCE
        }
        PlanningError::InvalidObservatoryConfig { .. } => {
            EMPYREAN_PLANNING_ERROR_INVALID_OBSERVATORY_CONFIG
        }
        PlanningError::NightTimeConversion { .. } => EMPYREAN_PLANNING_ERROR_NIGHT_TIME_CONVERSION,
        // Outside the survey's reachable set — a fold / encounter /
        // output-epoch / second-order / sampled-rung refusal. Named one by
        // one (never a `_` wildcard) so a new engine variant breaks the
        // build here; the full cause is on `error_message`.
        PlanningError::EncounterPropagation(_)
        | PlanningError::EncounterSTMUnavailable { .. }
        | PlanningError::NoCovarianceOnOrbit
        | PlanningError::SingularCovariance
        | PlanningError::SingularInformationMatrix
        | PlanningError::FrameMismatch { .. }
        | PlanningError::OriginMismatch { .. }
        | PlanningError::NonGravCovarianceRequired { .. }
        | PlanningError::MissingWideJacobian { .. }
        | PlanningError::RadarPredictionUnavailable { .. }
        | PlanningError::InvalidRadarSpec { .. }
        | PlanningError::NoObservations
        | PlanningError::NonCartesianPrior { .. }
        | PlanningError::OutputEpochRequiresHeliocentric { .. }
        | PlanningError::OutputEpochNonFinite { .. }
        | PlanningError::OutputEpochEphemeris { .. }
        | PlanningError::ReportEpochTransportSTMUnavailable { .. }
        | PlanningError::IndefiniteReportBlock { .. }
        | PlanningError::SigmaFlightFailed { .. }
        | PlanningError::MonteCarloMemberFlightFailed { .. }
        | PlanningError::PerPrefixMemberFlightFailed { .. }
        | PlanningError::TransportHandleKeyMismatch
        | PlanningError::TransportContextMissingPriorSet { .. }
        | PlanningError::TransportDispersionExceeded { .. }
        | PlanningError::NonCanonicalSigmaPoint { .. }
        | PlanningError::SigmaPointConstruction { .. }
        | PlanningError::MonteCarloConstruction { .. }
        | PlanningError::MonteCarloSampleFloor { .. }
        | PlanningError::SigmaFoldChartExceeded { .. }
        | PlanningError::MissingObservationHessian { .. }
        | PlanningError::NonFiniteSecondOrderMoment { .. }
        | PlanningError::SecondOrderCurvatureDominant { .. }
        | PlanningError::SecondOrderSingularObservedAxis { .. }
        | PlanningError::ObservationMoment(_) => EMPYREAN_PLANNING_ERROR_OTHER,
    }
}

/// Marshal one orbit's survey outcome (delivered or a typed `PlanningError`).
fn vis_result_to_c(
    id: &str,
    outcome: &Result<VisibilityResult, PlanningError>,
) -> EmpyreanVisibilityResult {
    match outcome {
        Ok(r) => {
            let obs: Vec<EmpyreanObservatoryVisibility> =
                r.observatory_visibility.iter().map(obs_vis_to_c).collect();
            let num_observatories = obs.len();
            let obs_ptr = if num_observatories == 0 {
                std::ptr::null_mut()
            } else {
                Box::into_raw(obs.into_boxed_slice()) as *mut EmpyreanObservatoryVisibility
            };
            let (has_earth_orientation, bpc, analytic) = match r.earth_orientation {
                Some(s) => (1u8, s.bpc_epochs, s.analytic_iau_2006_epochs),
                None => (0u8, 0, 0),
            };
            EmpyreanVisibilityResult {
                orbit_id: owned_cstr(&r.orbit_id),
                delivered: 1,
                error_code: EMPYREAN_PLANNING_ERROR_NONE,
                error_message: std::ptr::null_mut(),
                resolved_method: covariance_kind_to_u8(r.resolved_method),
                has_earth_orientation,
                earth_orientation_bpc_epochs: bpc,
                earth_orientation_analytic_iau_2006_epochs: analytic,
                observatories: obs_ptr,
                num_observatories,
            }
        }
        Err(e) => EmpyreanVisibilityResult {
            orbit_id: owned_cstr(id),
            delivered: 0,
            error_code: planning_error_code(e),
            error_message: owned_cstr(&format!("{e}")),
            resolved_method: EMPYREAN_COVARIANCE_KIND_UNKNOWN,
            has_earth_orientation: 0,
            earth_orientation_bpc_epochs: 0,
            earth_orientation_analytic_iau_2006_epochs: 0,
            observatories: std::ptr::null_mut(),
            num_observatories: 0,
        },
    }
}

// ── Visibility entry points ─────────────────────────────────────────

/// Survey the observability of one or more orbits over a time window from a
/// set of observatories: the per-epoch gates, the observable windows, and
/// the local-night product (twilight epochs, local solar midnight, observable
/// intervals, and the anchors with their Moon-aware circumstances).
///
/// `orbits` is an array of `n_orbits` orbits (each carrying its own id; a null
/// or empty id becomes `"orbit_{i}"`). `start` / `end` are MJD TDB and
/// `step_days` the survey grid step. `config.observatories` carries the
/// per-site gates and the night-product twilight thresholds. On success
/// populates `*out` (caller-allocated) with one result per input orbit, in
/// input order; free with [`empyrean_visibility_result_free`].
///
/// Returns 0 on success (per-orbit failures ride each result's `error_code`
/// / `error_message`, NOT the call status), -1 for a null/invalid argument,
/// -3 if the survey setup fails, -99 on an internal panic. The error message
/// is retrievable via `empyrean_last_error()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_compute_visibility(
    ctx: *const EmpyreanContext,
    orbits: *const EmpyreanOrbit,
    n_orbits: usize,
    start: f64,
    end: f64,
    step_days: f64,
    config: *const EmpyreanPlanningConfig,
    out: *mut EmpyreanVisibilityBatch,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || orbits.is_null() || config.is_null() || out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }
        if n_orbits == 0 {
            set_last_error("n_orbits must be non-zero");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let orbit_slice = unsafe { std::slice::from_raw_parts(orbits, n_orbits) };

        // Input-order ids, the canonical batch keys, so a result joins back
        // to its orbit positionally.
        let ids: Vec<String> = orbit_slice
            .iter()
            .enumerate()
            .map(|(i, o)| batch_orbit_id(o, i))
            .collect();

        let mut map: HashMap<String, _> = HashMap::with_capacity(n_orbits);
        for (o, id) in orbit_slice.iter().zip(ids.iter()) {
            match empyrean_orbit_to_orbits(o, id) {
                Ok(orbits) => {
                    map.insert(id.clone(), orbits);
                }
                Err(e) => {
                    set_last_error(&e.message);
                    return e.code.unwrap_or(-1);
                }
            }
        }

        let plan_cfg = match build_planning_config(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let start_epoch = Epoch::from_mjd_tdb(start);
        let end_epoch = Epoch::from_mjd_tdb(end);
        let surveyed =
            compute_visibility(ctx_ref, &map, start_epoch, end_epoch, step_days, &plan_cfg);

        // One result per INPUT orbit, in input order (HashMap order is not
        // stable). A key missing from the map is an internal invariant
        // violation (every id was inserted), surfaced as a failed orbit
        // rather than a silent drop.
        let results: Vec<EmpyreanVisibilityResult> = ids
            .iter()
            .map(|id| match surveyed.get(id) {
                Some(outcome) => vis_result_to_c(id, outcome),
                None => EmpyreanVisibilityResult {
                    orbit_id: owned_cstr(id),
                    delivered: 0,
                    error_code: EMPYREAN_PLANNING_ERROR_OTHER,
                    error_message: owned_cstr(
                        "internal error: orbit id missing from the survey result map",
                    ),
                    resolved_method: EMPYREAN_COVARIANCE_KIND_UNKNOWN,
                    has_earth_orientation: 0,
                    earth_orientation_bpc_epochs: 0,
                    earth_orientation_analytic_iau_2006_epochs: 0,
                    observatories: std::ptr::null_mut(),
                    num_observatories: 0,
                },
            })
            .collect();

        let num_results = results.len();
        let results_ptr =
            Box::into_raw(results.into_boxed_slice()) as *mut EmpyreanVisibilityResult;
        unsafe {
            (*out).results = results_ptr;
            (*out).num_results = num_results;
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_compute_visibility");
            -99
        }
    }
}

/// Free the heap allocations inside a visibility batch populated by
/// [`empyrean_compute_visibility`]. Does not free the caller-allocated
/// `EmpyreanVisibilityBatch` struct itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_visibility_result_free(result: *mut EmpyreanVisibilityBatch) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &*result };
        if res.results.is_null() || res.num_results == 0 {
            return;
        }
        let batch = unsafe { Vec::from_raw_parts(res.results, res.num_results, res.num_results) };
        for r in batch {
            free_visibility_result(r);
        }
        unsafe {
            (*result).results = std::ptr::null_mut();
            (*result).num_results = 0;
        }
    }));
}

/// Release one per-orbit result and everything it owns (strings, the
/// observatory array and, recursively, every point / window / night / anchor,
/// including each anchor circumstance's packed-joint triangle).
fn free_visibility_result(r: EmpyreanVisibilityResult) {
    if !r.orbit_id.is_null() {
        drop(unsafe { CString::from_raw(r.orbit_id) });
    }
    if !r.error_message.is_null() {
        drop(unsafe { CString::from_raw(r.error_message) });
    }
    if !r.observatories.is_null() && r.num_observatories > 0 {
        let obs = unsafe {
            Vec::from_raw_parts(r.observatories, r.num_observatories, r.num_observatories)
        };
        for ov in obs {
            free_observatory_visibility(ov);
        }
    }
}

fn free_observatory_visibility(ov: EmpyreanObservatoryVisibility) {
    if !ov.obs_code.is_null() {
        drop(unsafe { CString::from_raw(ov.obs_code) });
    }
    if !ov.note.is_null() {
        drop(unsafe { CString::from_raw(ov.note) });
    }
    if !ov.points.is_null() && ov.num_points > 0 {
        drop(unsafe { Vec::from_raw_parts(ov.points, ov.num_points, ov.num_points) });
    }
    if !ov.windows.is_null() && ov.num_windows > 0 {
        drop(unsafe { Vec::from_raw_parts(ov.windows, ov.num_windows, ov.num_windows) });
    }
    if !ov.nights.is_null() && ov.num_nights > 0 {
        let nights = unsafe { Vec::from_raw_parts(ov.nights, ov.num_nights, ov.num_nights) };
        for n in nights {
            free_night(n);
        }
    }
}

fn free_night(n: EmpyreanNight) {
    if !n.note.is_null() {
        drop(unsafe { CString::from_raw(n.note) });
    }
    if !n.intervals.is_null() && n.num_intervals > 0 {
        drop(unsafe { Vec::from_raw_parts(n.intervals, n.num_intervals, n.num_intervals) });
    }
    if !n.anchors.is_null() && n.num_anchors > 0 {
        let anchors = unsafe { Vec::from_raw_parts(n.anchors, n.num_anchors, n.num_anchors) };
        for mut a in anchors {
            // The packed-joint triangle in each anchor's circumstances is the
            // only heap inside an anchor; free it whether or not the
            // circumstances were populated (an absent joint is a no-op).
            unsafe { free_packed_joint(&mut a.circumstances.sky_covariance) };
        }
    }
}

/// The observatory visibility gates at the boundary.
///
/// Two fields with different zero-init stories, and the difference is
/// the whole point: `min_elevation_deg`'s engine default IS `0.0`, so a
/// zero-init struct already means it; `max_sun_altitude_deg`'s is
/// `-18.0` while `0.0` is a legal solar altitude, so a zero-init struct
/// read literally would quietly plan a campaign in daylight.
#[cfg(test)]
mod observatory_gate_tests {
    use super::*;

    fn zeroed_config() -> EmpyreanObservatoryConfig {
        // SAFETY: `#[repr(C)]`, scalars plus one pointer that zero-inits
        // to null — the caller-side `memset(0)` this pins.
        unsafe { std::mem::zeroed() }
    }

    /// The engine's own default is what the switch resolves to, read off
    /// the engine rather than restated here — so this test fails if the
    /// twilight convention ever moves, instead of enshrining a literal.
    #[test]
    fn the_darkness_default_is_read_off_the_engine() {
        let engine = ObservatoryConfig::new("F51", [0.5, 0.5]);
        assert_eq!(
            engine_default_max_sun_altitude_deg(),
            engine.max_sun_altitude_deg,
            "the fallback must BE the engine's default, not a copy of it"
        );
        assert!(
            engine.max_sun_altitude_deg < 0.0,
            "a darkness threshold at or above 0 would mean daylight; the engine's \
             default is astronomical twilight"
        );
    }

    /// `memset(0)` must resolve to the engine's darkness default, not to
    /// a literal 0.0 — the Sun's centre on the geometric horizon.
    #[test]
    fn a_zero_init_config_resolves_to_the_engine_darkness_default() {
        let c = zeroed_config();
        assert_eq!(
            c.has_max_sun_altitude_deg, 0,
            "zero-init declares no override"
        );
        assert_eq!(
            c.max_sun_altitude_deg, 0.0,
            "and the value slot is a bare zero"
        );

        // Drive the PRODUCTION marshal, not a copy of its branch: a
        // test that re-implements the resolution asserts only that the
        // test is self-consistent, and would keep passing if the real
        // site stopped honouring the switch.
        let code = std::ffi::CString::new("F51").unwrap();
        let mut c = c;
        c.obs_code = code.as_ptr();
        let mut cfg: EmpyreanPlanningConfig = unsafe { std::mem::zeroed() };
        let obs = [c];
        cfg.observatories = obs.as_ptr();
        cfg.num_observatories = 1;
        cfg.num_threads = -1;
        let built = build_planning_config(&cfg).expect("a zero-init observatory is well-formed");
        let resolved = built.observatories[0].max_sun_altitude_deg;

        assert_eq!(
            resolved,
            ObservatoryConfig::new("", [0.0, 0.0]).max_sun_altitude_deg,
            "a zero-init struct must plan under the engine's twilight default"
        );
        assert_ne!(
            resolved, 0.0,
            "resolving to a literal 0.0 would place the Sun's centre on the horizon — \
             a campaign planned in daylight, silently"
        );
        assert_eq!(
            built.observatories[0].min_elevation_deg, 0.0,
            "and the elevation gate stays at the geometric horizon"
        );
    }

    /// The elevation field needs no switch: its engine default IS 0.0,
    /// so zero-init already means the geometric horizon. This pins the
    /// asymmetry deliberately, so nobody "tidies" a switch onto it or
    /// off the other.
    #[test]
    fn the_elevation_field_needs_no_switch() {
        assert_eq!(
            ObservatoryConfig::new("", [0.0, 0.0]).min_elevation_deg,
            0.0,
            "the engine's elevation default is the geometric horizon, which IS the \
             zero-init value — a presence switch here would add a state with no meaning"
        );
        assert_eq!(zeroed_config().min_elevation_deg, 0.0);
    }

    /// An explicit override reaches the engine config unchanged, on both
    /// axes, and does not disturb the fields beside it.
    #[test]
    fn explicit_values_reach_the_engine_config() {
        let code = std::ffi::CString::new("F51").unwrap();
        let c = EmpyreanObservatoryConfig {
            obs_code: code.as_ptr(),
            sigma_ra_arcsec: 0.3,
            sigma_dec_arcsec: 0.4,
            max_apparent_mag: 24.5,
            min_elongation_deg: 70.0,
            min_elevation_deg: 30.0,
            has_max_sun_altitude_deg: 1,
            max_sun_altitude_deg: -12.0,
            has_twilight_thresholds: 0,
            sunset_sun_altitude_deg: 0.0,
            civil_twilight_sun_altitude_deg: 0.0,
            nautical_twilight_sun_altitude_deg: 0.0,
            astronomical_twilight_sun_altitude_deg: 0.0,
            has_explicit_anchor_utc: 0,
            explicit_anchor_utc_mjd_tdb: 0.0,
        };
        let mut cfg: EmpyreanPlanningConfig = unsafe { std::mem::zeroed() };
        let obs = [c];
        cfg.observatories = obs.as_ptr();
        cfg.num_observatories = 1;
        cfg.num_threads = -1;

        let built = build_planning_config(&cfg).expect("a well-formed observatory config");
        assert_eq!(built.observatories.len(), 1);
        let o = &built.observatories[0];
        assert_eq!(o.obs_code, "F51");
        assert_eq!(o.min_elevation_deg, 30.0, "airmass ≤ 2, as supplied");
        assert_eq!(
            o.max_sun_altitude_deg, -12.0,
            "nautical twilight, as supplied"
        );
        assert_eq!(o.max_apparent_mag, 24.5);
        assert_eq!(o.min_elongation_deg, 70.0);
        assert_eq!(o.sigma_arcsec, [0.3, 0.4]);
        // has_twilight_thresholds == 0 resolves to the engine's own defaults,
        // read off the engine so a moved convention fails this instead of a
        // literal being enshrined.
        let defaults = ObservatoryConfig::new("F51", [0.3, 0.4]);
        assert_eq!(o.sunset_sun_altitude_deg, defaults.sunset_sun_altitude_deg);
        assert_eq!(
            o.astronomical_twilight_sun_altitude_deg,
            defaults.astronomical_twilight_sun_altitude_deg
        );
        assert!(
            o.explicit_anchor_utc.is_none(),
            "no explicit anchor supplied"
        );
    }

    /// The twilight thresholds and the explicit-anchor epoch reach scott's
    /// `ObservatoryConfig` when the caller opts in — the fix the brief asks
    /// for: these fields are now READ by the visibility survey, not just
    /// marshaled. A scrambled set is refused by the engine's own validator.
    #[test]
    fn the_twilight_thresholds_and_explicit_anchor_round_trip() {
        let code = std::ffi::CString::new("X05").unwrap();
        let base = EmpyreanObservatoryConfig {
            obs_code: code.as_ptr(),
            sigma_ra_arcsec: 0.5,
            sigma_dec_arcsec: 0.5,
            max_apparent_mag: 23.0,
            min_elongation_deg: 60.0,
            min_elevation_deg: 20.0,
            has_max_sun_altitude_deg: 0,
            max_sun_altitude_deg: 0.0,
            has_twilight_thresholds: 1,
            sunset_sun_altitude_deg: -0.5,
            civil_twilight_sun_altitude_deg: -6.0,
            nautical_twilight_sun_altitude_deg: -12.0,
            astronomical_twilight_sun_altitude_deg: -18.0,
            has_explicit_anchor_utc: 1,
            explicit_anchor_utc_mjd_tdb: 61000.25,
        };
        let mut cfg: EmpyreanPlanningConfig = unsafe { std::mem::zeroed() };
        let obs = [base];
        cfg.observatories = obs.as_ptr();
        cfg.num_observatories = 1;
        cfg.num_threads = -1;
        let built = build_planning_config(&cfg).expect("a strictly-descending set is valid");
        let o = &built.observatories[0];
        assert_eq!(
            o.sunset_sun_altitude_deg, -0.5,
            "as supplied, not the default"
        );
        assert_eq!(o.civil_twilight_sun_altitude_deg, -6.0);
        assert_eq!(o.nautical_twilight_sun_altitude_deg, -12.0);
        assert_eq!(o.astronomical_twilight_sun_altitude_deg, -18.0);
        let anchor = o
            .explicit_anchor_utc
            .expect("an explicit anchor was supplied");
        assert_eq!(anchor.mjd_tdb(), 61000.25);
        // The engine's own group validator (non-descending → refused by the
        // observatory's code) runs inside the survey, which needs a Context;
        // it is exercised in the from-engine visibility test.
    }
}

/// `uncertainty_method` crosses the planning marshal verbatim (0.11.0).
///
/// Drives the production `build_planning_config`, not a copy of its arm.
/// Mutation guard: re-pinning `uncertainty_method` to `FirstOrder` in
/// `build_planning_config` makes the SecondOrder / SigmaPoint cases below
/// fold first order → red.
#[cfg(test)]
mod uncertainty_method_tests {
    use super::*;
    use crate::propagate::{
        EMPYREAN_UNCERTAINTY_FIRST, EMPYREAN_UNCERTAINTY_SECOND, EMPYREAN_UNCERTAINTY_SIGMA_POINT,
    };
    use empyrean_core::planning::EngineUncertaintyMethod;

    fn cfg_with_method(tag: u8) -> EmpyreanPlanningConfig {
        // SAFETY: `#[repr(C)]` scalars plus two null-initializing pointers.
        let mut cfg: EmpyreanPlanningConfig = unsafe { std::mem::zeroed() };
        cfg.num_threads = -1;
        cfg.uncertainty_method.tag = tag;
        cfg
    }

    #[test]
    fn a_zero_init_config_folds_first_order() {
        // `memset(0)` is tag 0 — the single-linearization fold every prior
        // release computed — so the default is behaviour-preserving.
        let built = build_planning_config(&cfg_with_method(EMPYREAN_UNCERTAINTY_FIRST))
            .expect("a zero-init planning config is well-formed");
        assert_eq!(
            built.uncertainty_method,
            EngineUncertaintyMethod::FirstOrder
        );
    }

    #[test]
    fn second_order_tag_folds_second_order() {
        let built = build_planning_config(&cfg_with_method(EMPYREAN_UNCERTAINTY_SECOND))
            .expect("a SecondOrder request marshals");
        assert_eq!(
            built.uncertainty_method,
            EngineUncertaintyMethod::SecondOrder
        );
    }

    #[test]
    fn sigma_point_tag_folds_sigma_point_with_its_params() {
        let mut cfg = cfg_with_method(EMPYREAN_UNCERTAINTY_SIGMA_POINT);
        cfg.uncertainty_method.sp_n_sigma = 1.0;
        cfg.uncertainty_method.sp_samples_per_plane = 8;
        let built = build_planning_config(&cfg).expect("a SigmaPoint request marshals");
        assert!(
            matches!(
                built.uncertainty_method,
                EngineUncertaintyMethod::SigmaPoint { .. }
            ),
            "expected SigmaPoint, got {:?}",
            built.uncertainty_method
        );
    }
}

/// The log-det delivered/withheld contract at the C boundary: a finite
/// value with the `_NONE` reason when delivered, a `NaN` paired with a
/// named reason under a nonlinear transport — never a `NaN` without a
/// reason beside it.
#[cfg(test)]
mod log_det_withheld_tests {
    use super::*;
    use crate::propagate::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT;
    use empyrean_core::propagation::CovarianceKind;

    fn metrics(log_det: Option<f64>, withheld: Option<LogDetWithheld>) -> CovarianceMetrics {
        CovarianceMetrics {
            position_sigma_km: 1.0,
            velocity_sigma_m_s: 2.0,
            semi_major_km: 3.0,
            semi_minor_km: 4.0,
            log_det,
            log_det_withheld: withheld,
        }
    }

    /// The withheld codes are frozen ABI wire values, pinned here against
    /// the core `LogDetWithheld` variants they encode. A renumber is an
    /// ABI break, caught here rather than silently reinterpreted downstream.
    #[test]
    fn withheld_codes_are_the_pinned_wire_values() {
        assert_eq!(EMPYREAN_LOG_DET_WITHHELD_NONE, 0);
        assert_eq!(EMPYREAN_LOG_DET_WITHHELD_SECOND_ORDER_MOMENT, 1);
        assert_eq!(EMPYREAN_LOG_DET_WITHHELD_SAMPLED_MOMENT, 2);
        assert_eq!(EMPYREAN_LOG_DET_WITHHELD_MIXTURE_MOMENT, 3);
    }

    /// A delivered log_det carries the finite value and the `_NONE`
    /// reason, with no sampled method — the delivered half of the pair.
    #[test]
    fn a_delivered_log_det_carries_the_value_and_no_reason() {
        let c = metrics_to_c(&metrics(Some(-221.5), None));
        assert_eq!(c.log_det, -221.5);
        assert_eq!(c.log_det_withheld, EMPYREAN_LOG_DET_WITHHELD_NONE);
        assert_eq!(c.log_det_withheld_method, EMPYREAN_COVARIANCE_KIND_UNKNOWN);
    }

    /// A SecondOrder (and a Mixture) transport withholds log_det as `NaN`
    /// AND names itself — assert the pair. Mapping either reason to
    /// `_NONE` (the mutation) fails the reason assertion here.
    #[test]
    fn a_second_order_plan_withholds_log_det_with_its_reason() {
        let second = metrics_to_c(&metrics(None, Some(LogDetWithheld::SecondOrderMoment)));
        assert!(second.log_det.is_nan(), "a withheld log_det is NaN");
        assert_eq!(
            second.log_det_withheld, EMPYREAN_LOG_DET_WITHHELD_SECOND_ORDER_MOMENT,
            "the NaN must carry the SecondOrder reason, never _NONE"
        );
        assert_eq!(
            second.log_det_withheld_method,
            EMPYREAN_COVARIANCE_KIND_UNKNOWN
        );

        let mixture = metrics_to_c(&metrics(None, Some(LogDetWithheld::MixtureMoment)));
        assert!(mixture.log_det.is_nan());
        assert_eq!(
            mixture.log_det_withheld,
            EMPYREAN_LOG_DET_WITHHELD_MIXTURE_MOMENT
        );
        assert_eq!(
            mixture.log_det_withheld_method,
            EMPYREAN_COVARIANCE_KIND_UNKNOWN
        );
    }

    /// A sampled transport withholds log_det and carries the sampled kind
    /// beside the reason, in the covariance-kind vocabulary.
    #[test]
    fn a_sampled_moment_carries_its_transport_kind() {
        let sampled = metrics_to_c(&metrics(
            None,
            Some(LogDetWithheld::SampledMoment {
                method: CovarianceKind::SigmaPoint,
            }),
        ));
        assert!(sampled.log_det.is_nan());
        assert_eq!(
            sampled.log_det_withheld,
            EMPYREAN_LOG_DET_WITHHELD_SAMPLED_MOMENT
        );
        assert_eq!(
            sampled.log_det_withheld_method,
            EMPYREAN_COVARIANCE_KIND_SIGMA_POINT
        );
    }
}
