use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use empyrean_core::ForceModelTier;
use empyrean_core::convert::{coordinate_state_to_coordinates, frame_to_int};
use empyrean_core::coordinates::{AU, CoordinateRepresentation, Coordinates, Origin};
use empyrean_core::determination::{
    AcceptabilityReport, AdaptiveRejectionConfig, BiasKind, BiasScope, CMC2003Config, ODConfig,
    ODResult, ObservationResidualSummary, ObservationResult, Observations, OriginPolicy,
    OutputEpoch, RadarMeasurement, RadarObservation, RejectionReason, SolveFor, SolveForParams,
    SolvedCovariance, UpstreamForceModelTier, determine, evaluate_single, refine_single,
};
use empyrean_core::io::{ADESObservations, parse_ades};
use empyrean_core::nongrav::NonGravModel;
use empyrean_core::orbits::Orbits;
use empyrean_core::photometry::{
    FittedPhotometryModel, PhotometryConfig, PhotometryModel, PhotometryResult,
};

use crate::propagate::{EmpyreanOrbit, EmpyreanPropagatedState, int_to_force_model};
use crate::{EmpyreanContext, set_last_error};

// ── C-compatible types ──────────────────────────────────────

/// A single optical observation for orbit determination — full ADES schema.
///
/// String fields are nullable (`null` pointer = absent). Float fields
/// use NaN as the absent sentinel. The `n_stars` integer uses `-1` as
/// the absent sentinel (since `u32::MAX` is a valid count). The
/// `obs_code` is fixed-size 4-byte null-padded to keep the common case
/// allocation-free.
///
/// Mirrors scott's `OpticalObservation` field-for-field — every named
/// PSV column round-trips losslessly except for ADES extension fields
/// not yet in the upstream schema.
#[repr(C)]
pub struct EmpyreanObservation {
    // ── Identification ────────────────────────────────────
    /// IAU permanent designation (nullable).
    pub perm_id: *mut c_char,
    /// MPC provisional designation (nullable).
    pub prov_id: *mut c_char,
    /// Observer-assigned tracklet identifier (nullable).
    pub trk_sub: *mut c_char,
    /// MPC-assigned observation identifier (`obsid`, nullable).
    pub obs_id: *mut c_char,
    /// Observer-assigned sub-identifier (`obsSubID`, nullable).
    pub obs_sub_id: *mut c_char,
    /// Track identifier (`trkID`, nullable).
    pub trk_id: *mut c_char,

    // ── Observer ──────────────────────────────────────────
    /// MPC observatory code, null-padded to 4 bytes.
    pub obs_code: [u8; 4],
    /// Observation mode (CCD, CMOS, etc.) (nullable).
    pub mode: *mut c_char,
    /// MPC program code (nullable).
    pub prog: *mut c_char,

    // ── Observer location (roving / spacecraft) ──────────
    /// Coordinate system for observer position (nullable).
    pub sys: *mut c_char,
    /// Center body NAIF ID. NaN if unset.
    pub ctr: f64,
    /// Position component 1 (lon for WGS84, X for ICRF_KM). NaN if unset.
    pub pos1: f64,
    /// Position component 2 (lat for WGS84, Y for ICRF_KM). NaN if unset.
    pub pos2: f64,
    /// Position component 3 (alt for WGS84, Z for ICRF_KM). NaN if unset.
    pub pos3: f64,

    // ── Core astrometry ──────────────────────────────────
    /// Observation time as ISO 8601 UTC string.
    pub obs_time: *mut c_char,
    /// Right ascension (degrees).
    pub ra_deg: f64,
    /// Declination (degrees).
    pub dec_deg: f64,

    // ── Uncertainties ────────────────────────────────────
    /// RA·cos(Dec) uncertainty (arcseconds). NaN if unavailable.
    pub rms_ra_arcsec: f64,
    /// Dec uncertainty (arcseconds). NaN if unavailable.
    pub rms_dec_arcsec: f64,
    /// RA-Dec correlation coefficient [-1, 1]. NaN if unavailable.
    pub rms_corr: f64,

    // ── Astrometric catalog ──────────────────────────────
    /// Star catalog used for astrometric reduction (nullable).
    pub ast_cat: *mut c_char,

    // ── Photometry ───────────────────────────────────────
    /// Apparent magnitude. NaN if unavailable.
    pub mag: f64,
    /// Magnitude uncertainty. NaN if unavailable.
    pub rms_mag: f64,
    /// Photometric passband (nullable).
    pub band: *mut c_char,
    /// Photometric catalog (nullable).
    pub phot_cat: *mut c_char,
    /// Photometric aperture (arcseconds). NaN if unavailable.
    pub phot_ap: f64,

    // ── Supplementary diagnostics ────────────────────────
    /// log10(SNR) of the detection. NaN if unavailable.
    pub log_snr: f64,
    /// Seeing FWHM (arcseconds). NaN if unavailable.
    pub seeing: f64,
    /// Exposure time (seconds). NaN if unavailable.
    pub exp: f64,
    /// RMS of astrometric fit (arcseconds). NaN if unavailable.
    pub rms_fit: f64,
    /// Number of reference stars in astrometric fit. -1 if unavailable.
    pub n_stars: i32,
    /// MPC note flags (nullable).
    pub notes: *mut c_char,
    /// Free-text observer remarks (nullable).
    pub remarks: *mut c_char,
}

// ── Radar measurement kinds (the ADES `RadarValue` choice discriminator) ──
//
// Pinned integer codes selecting which delay/Doppler pair on
// [`EmpyreanRadarObservation`] is live. A radar record carries a delay
// **XOR** a Doppler value — never both.

/// Round-trip time-delay measurement: `delay_seconds` / `rms_delay_microseconds`
/// are valid; the Doppler pair is `f64::NAN`.
pub const EMPYREAN_RADAR_KIND_DELAY: u8 = 0;
/// Doppler-shift measurement: `doppler_hz` / `rms_doppler_hz` are valid;
/// the delay pair is `f64::NAN`.
pub const EMPYREAN_RADAR_KIND_DOPPLER: u8 = 1;

/// A single radar (delay or Doppler) observation for orbit determination —
/// the ADES `<radar>` schema.
///
/// ADES models radar astrometry as its own top-level table, parallel to
/// `<optical>` (not as an optical `mode`). A record carries a round-trip
/// time **delay** *or* a **Doppler** shift (the ADES `RadarValue` is an
/// XSD `<choice>`), referred to a transmitting (`trx`) and receiving
/// (`rcv`) station — equal for a monostatic observation, distinct for a
/// bistatic one.
///
/// **Units are ADES-native through this FFI — the C ABI performs ZERO unit
/// conversion.** The delay *value* is in **seconds** while its uncertainty
/// `rms_delay_microseconds` is in **microseconds** (this asymmetry is
/// intentional in the ADES schema, verified against the IAU-ADES
/// `a4179_radar` reference data and the JPL SSD radar API); Doppler value
/// and uncertainty are both in **Hz**; `frq_mhz` is in **MHz**. The single
/// SI normalisation happens once downstream in scott's
/// `Observation::from_radar`, never here.
///
/// String fields are nullable (`null` pointer = absent). The `trx`/`rcv`
/// station codes are fixed-size 4-byte null-padded (like `obs_code`).
/// Absent f64 fields use `f64::NAN`. The `com` flag is a tri-state `i8`.
/// No field is silently zeroed or defaulted.
///
/// Mirrors scott's `RadarObservation` field-for-field.
#[repr(C)]
pub struct EmpyreanRadarObservation {
    // ── Identification ────────────────────────────────────
    /// IAU permanent designation (nullable).
    pub perm_id: *mut c_char,
    /// MPC provisional designation (nullable).
    pub prov_id: *mut c_char,
    /// Observer-assigned tracklet identifier (nullable).
    pub trk_sub: *mut c_char,

    // ── Bistatic geometry ─────────────────────────────────
    /// MPC station code of the **transmitting** antenna (ADES `trx`),
    /// null-padded to 4 bytes.
    pub trx: [u8; 4],
    /// MPC station code of the **receiving** antenna (ADES `rcv`),
    /// null-padded to 4 bytes. Equal to `trx` for a monostatic
    /// observation; differs for a bistatic one.
    pub rcv: [u8; 4],

    // ── Core measurement ──────────────────────────────────
    /// Observation epoch as an ISO 8601 UTC string. For radar this is the
    /// **receive** epoch (the time the returned signal is recorded).
    /// Required, non-null.
    pub obs_time: *mut c_char,
    /// Measurement kind: `EMPYREAN_RADAR_KIND_DELAY` (0) or
    /// `EMPYREAN_RADAR_KIND_DOPPLER` (1). Selects which value pair is live.
    pub kind: u8,
    /// Round-trip time delay in **seconds** (ADES-native). Valid iff
    /// `kind == EMPYREAN_RADAR_KIND_DELAY`, else `f64::NAN`.
    pub delay_seconds: f64,
    /// 1σ uncertainty of the delay in **microseconds** (ADES-native; the
    /// asymmetry vs `delay_seconds` is intentional). Valid iff
    /// `kind == EMPYREAN_RADAR_KIND_DELAY`, else `f64::NAN`.
    pub rms_delay_microseconds: f64,
    /// Doppler shift in **Hz** (ADES-native, signed). Valid iff
    /// `kind == EMPYREAN_RADAR_KIND_DOPPLER`, else `f64::NAN`.
    pub doppler_hz: f64,
    /// 1σ uncertainty of the Doppler shift in **Hz** (ADES-native). Valid
    /// iff `kind == EMPYREAN_RADAR_KIND_DOPPLER`, else `f64::NAN`.
    pub rms_doppler_hz: f64,

    // ── Reduction metadata ────────────────────────────────
    /// Transmit carrier reference frequency in **MHz** (ADES `frq`).
    /// Required; relates a Doppler shift to a range rate.
    pub frq_mhz: f64,
    /// Center-of-mass flag (ADES `com`), tri-state: `-1` = absent,
    /// `0` = false (peak-power / leading-edge reduction), `1` = true
    /// (reduced to target center of mass). Mirrors `Option<bool>`; a
    /// missing flag MUST map to `-1` (never `0`) — the ADES center-of-mass
    /// default is applied explicitly downstream, not silently here.
    pub com: i8,
    /// log10(SNR) of the echo, if reported. `f64::NAN` if absent.
    pub log_snr: f64,
    /// Free-text observer remarks (nullable).
    pub remarks: *mut c_char,
}

// ── Rejection-reason codes (mirrors scott::rejection::RejectionReason) ──
//
// Pinned integer codes so the Python layer can decode without needing
// the scott enum visible. Add new variants by appending; never reorder.

/// Observation passed all rejection criteria (or evaluate did not run rejection).
pub const EMPYREAN_REJECTION_ACCEPTED: i32 = 0;
/// Rejected by chi-squared threshold (Layer 1).
pub const EMPYREAN_REJECTION_CHI_SQUARED: i32 = 1;
/// Rejected by sigma-clipping (Layer 1).
pub const EMPYREAN_REJECTION_SIGMA_CLIP: i32 = 2;
/// Rejected by Cook's distance threshold (Layer 2).
pub const EMPYREAN_REJECTION_COOKS_DISTANCE: i32 = 3;
/// Rejected by information-aware adaptive criterion (Layer 3).
pub const EMPYREAN_REJECTION_ADAPTIVE: i32 = 4;
/// Observatory could not be resolved.
pub const EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY: i32 = 5;
/// Rejected by Carpino–Milani–Chesley (2003) χ²-with-hysteresis scheme.
pub const EMPYREAN_REJECTION_CMC2003: i32 = 6;
/// Skipped because the observation mode is `RAD` (radar). scott's
/// optical-only fitter can't fold radar range / Doppler measurements
/// — radar observations are surfaced with NaN residuals and this code.
pub const EMPYREAN_REJECTION_RADAR_UNSUPPORTED: i32 = 7;
/// Skipped because the observation mode is `OCC` (stellar
/// occultation). scott's optical-only fitter can't fold occultation
/// chord timings — occultation observations are surfaced with NaN
/// residuals and this code.
pub const EMPYREAN_REJECTION_OCCULTATION_UNSUPPORTED: i32 = 8;
/// The observation belongs to an opposition group / sub-arc that
/// could not be reconciled with the converged fit. The observation
/// is not necessarily noisy — it is incompatible with the dynamical
/// regime of the in-arc fit (e.g. cross-Hill-sphere transition,
/// chaotic-capture interior, regime change between pre- and
/// post-encounter geometry).
pub const EMPYREAN_REJECTION_OUTSIDE_ARC: i32 = 9;
/// Rejection was not evaluated for this observation (e.g. evaluate path).
pub const EMPYREAN_REJECTION_NOT_EVALUATED: i32 = -1;

// ── Rejection-strategy kinds (selects which fields of EmpyreanRejectionConfig apply) ──
//
// `kind` discriminator on [`EmpyreanRejectionConfig`]. Default `0` keeps
// backward compatibility with C callers that zero-init the struct.

/// Information-loss-weighted adaptive rejection. Uses
/// `chi2_base` / `lambda` / `max_threshold`. Default.
pub const EMPYREAN_REJECTION_KIND_ADAPTIVE: u8 = 0;
/// Carpino–Milani–Chesley (2003) χ²-with-hysteresis. Uses
/// `chi2_rej` / `chi2_rec` (the upper / lower hysteresis thresholds).
pub const EMPYREAN_REJECTION_KIND_CMC2003: u8 = 1;

// ── Weighting (mirrors scott::weighting::WeightingConfig) ─────────
//
// The C ABI exposes weighting as a preset selector + an optional
// list of additional layers. Presets seed the chain with scott's
// curated layer sets; additional_layers are appended in order.
// `preset = NONE` + non-empty additional_layers = build from scratch.

/// No weighting preset — only `additional_layers` apply.
pub const EMPYREAN_WEIGHTING_PRESET_NONE: u8 = 0;
/// VFC17 — Vereš, Farnocchia, Chesley et al. 2017 station floors +
/// nightly de-weighting. The production default.
pub const EMPYREAN_WEIGHTING_PRESET_VFC17: u8 = 1;
/// NEODyS production preset.
pub const EMPYREAN_WEIGHTING_PRESET_NEODYS: u8 = 2;

/// `\sigma = \text{reported}` if present, else `\sigma = \text{rule}`.
pub const EMPYREAN_SIGMA_POLICY_DEFAULT_ONLY: i32 = 0;
/// `\sigma = \max(\text{reported}, \text{rule})` (production presets).
pub const EMPYREAN_SIGMA_POLICY_FLOOR: i32 = 1;

/// `kind` discriminator on [`EmpyreanWeightingLayer`].
pub const EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE: i32 = 0;
pub const EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING: i32 = 1;

/// One element of [`EmpyreanWeightingConfig::additional_layers`].
/// Tagged-union shape: the active fields depend on `kind`.
#[repr(C)]
pub struct EmpyreanWeightingLayer {
    /// Layer kind discriminator — one of
    /// `EMPYREAN_WEIGHTING_LAYER_*`.
    pub kind: i32,
    // ── ObservatoryRule fields ─────────────────────────────────
    /// MPC observatory code, null-padded to 4 bytes.
    pub obs_code: [u8; 4],
    /// 1σ RA·cos(δ) in arcsec.
    pub sigma_ra_arcsec: f64,
    /// 1σ Dec in arcsec.
    pub sigma_dec_arcsec: f64,
    /// Start of applicable time range (MJD TDB). NaN = unbounded.
    pub start_epoch_mjd_tdb: f64,
    /// End of applicable time range (MJD TDB). NaN = unbounded.
    pub end_epoch_mjd_tdb: f64,
    /// Scale factor on the resulting weight. 0.0 → upstream default (1.0).
    pub scale: f64,
    // ── NightlyDeweighting fields ──────────────────────────────
    /// Maximum gap between observations to count as the same night
    /// (days). 0.0 → upstream default (0.5).
    pub max_gap_days: f64,
}

/// Weighting configuration. Mirrors
/// [`scott::weighting::WeightingConfig`] structurally; extends it
/// with an `enabled` toggle and a preset selector for the common
/// case of "use the production preset" without constructing layers
/// by hand.
///
/// `enabled = 0` runs OD with uniform 1″ weighting (the old
/// `use_weighting = 0` behavior). `enabled = 1` activates the
/// pipeline; the resulting layer chain is the preset's layers
/// followed by `additional_layers` (allows e.g. VFC17 + per-survey
/// override).
#[repr(C)]
pub struct EmpyreanWeightingConfig {
    /// 1 = enabled (default), 0 = uniform 1″ weighting.
    pub enabled: u8,
    /// Preset selector. One of `EMPYREAN_WEIGHTING_PRESET_*`.
    /// Default `0` (NONE) means "use additional_layers only";
    /// when `enabled = 1` and zero-init, the conversion code
    /// substitutes `VFC17` so default-zero structs keep the
    /// production behavior.
    pub preset: u8,
    /// Default 1σ used when no rule applies (arcsec). 0.0 →
    /// upstream default (1.0). Ignored when preset != NONE.
    pub default_sigma_arcsec: f64,
    /// Sigma combination policy. -1 = use the preset's policy;
    /// otherwise one of `EMPYREAN_SIGMA_POLICY_*`.
    pub sigma_policy: i32,
    /// Pointer to additional layers appended to the preset's chain.
    /// Non-owning — caller keeps the array alive for the OD call.
    pub additional_layers: *const EmpyreanWeightingLayer,
    pub num_additional_layers: usize,
}

// ── Debiasing (mirrors scott::debiasing::DebiasingTable) ──────────

/// Debiasing-table identity tag. Currently EFCC2020 only.
pub const EMPYREAN_DEBIASING_TABLE_EFCC2020: i32 = 0;

/// Healpix resolution of a debiasing table.
pub const EMPYREAN_DEBIASING_RESOLUTION_STANDARD: i32 = 0;
pub const EMPYREAN_DEBIASING_RESOLUTION_HIRES: i32 = 1;

/// Catalog-bias-correction configuration. Mirrors scott's
/// `Option<Arc<DebiasingTable>>` field on `ODConfig`.
///
/// `enabled = 0` runs OD with no catalog debiasing (matches the old
/// `use_debiasing = 0` behavior). `enabled = 1` activates the
/// EFCC2020 pipeline; the table is loaded from `bias_dat_path` if
/// non-NULL, otherwise from the DataManager-default location at the
/// requested `resolution`.
#[repr(C)]
pub struct EmpyreanDebiasingConfig {
    /// 1 = on (default), 0 = no debiasing.
    pub enabled: u8,
    /// Table identity. Currently `EFCC2020` only.
    pub table_id: i32,
    /// `EMPYREAN_DEBIASING_RESOLUTION_*` — Standard (~35 MB) or Hires (~567 MB).
    pub resolution: i32,
    /// Optional path to the bias.dat file. NULL = DataManager default.
    /// Non-owning.
    pub bias_dat_path: *const c_char,
}

// ── SolveForParams codes ──────────────────────────────────────────
pub const EMPYREAN_SOLVE_FOR_STATE_ONLY: i32 = 0;
pub const EMPYREAN_SOLVE_FOR_STATE_AND_NONGRAV: i32 = 1;
pub const EMPYREAN_SOLVE_FOR_AUTO: i32 = 2;
/// An explicit multi-axis solve (any of DT / AMRAT / thrust, or a
/// combination) that the three coarse codes above cannot name. The
/// exact axes travel in the `EmpyreanSolveFor` flag struct.
pub const EMPYREAN_SOLVE_FOR_EXPLICIT: i32 = 3;

// ── Wide solved-covariance freeze (ABI-FROZEN; NEVER grows) ────────
/// Frozen storage width of the solved-parameter covariance matrix. Set
/// once at v0.9.0-rc.0 and never widened — there is no runtime
/// `abi_version` negotiation, so the inline `matrix[W][W]` is baked into
/// the struct size. `20` is scott's STRUCTURAL maximum (6 state + 3
/// Marsden + 1 DT + 1 AMRAT + 3 thrust segments × 3). scott v1.14.0 today
/// caps the actually-producible width at 17 (`MAX_SOLVE_WIDTH`; its solve
/// guard rejects anything wider), so columns 17..20 are RESERVE — held for
/// whatever axis combination scott may later admit past width 17, and zero
/// until then. (A 3-segment thrust solve already fits below 17; the reserve
/// is for the widest joint solves, not a specific axis.) A parameter beyond
/// this structural max (e.g. a drag axis) takes a fresh
/// `EMPYREAN_ABI_VERSION`-guarded break, not a silent widening.
pub const EMPYREAN_SOLVE_WIDTH: usize = 20;
/// `u32` sentinel for an absent slot tag (C has no `Option`). Consumers
/// MUST read the slot tags — a width alone is ambiguous (width 9 is
/// Marsden OR one-segment thrust).
pub const EMPYREAN_SLOT_NONE: u32 = 0xFFFF_FFFF;
// The frozen width can never silently fall below scott's own maximum.
const _: () = assert!(EMPYREAN_SOLVE_WIDTH >= empyrean_core::determination::MAX_SOLVE_WIDTH);

// ── Photometry fit-model codes (config request + result report) ────
// In AUTO the post-OD fit climbs a model ladder — H-only → HG12 → HG1G2
// — admitting the richest model the arc's phase-angle coverage and
// magnitude count support, and reports the one it fit via
// `model_used` (never AUTO). An explicit code pins a specific model.
// HG12 / HG1G2 follow Muinonen et al. (2010); H-only holds the slope
// fixed at G = 0.15.
pub const EMPYREAN_PHOTOMETRY_MODEL_AUTO: i32 = 0;
pub const EMPYREAN_PHOTOMETRY_MODEL_HONLY: i32 = 1;
pub const EMPYREAN_PHOTOMETRY_MODEL_HG: i32 = 2;
pub const EMPYREAN_PHOTOMETRY_MODEL_HG12: i32 = 3;
pub const EMPYREAN_PHOTOMETRY_MODEL_HG1G2: i32 = 4;

/// Integer handshake on the frozen-ABI shape contract, distinct from the
/// per-crate semver strings in `EmpyreanVersions` (which are provenance).
/// Consumers compiled against a given `EMPYREAN_SOLVE_WIDTH` check this at
/// load; any additive change to the frozen structs bumps it.
pub const EMPYREAN_ABI_VERSION: u32 = 1;

/// Runtime accessor for [`EMPYREAN_ABI_VERSION`] — lets a dynamically
/// linked consumer confirm the loaded library's frozen-shape contract
/// matches what it compiled against.
#[unsafe(no_mangle)]
pub extern "C" fn empyrean_abi_version() -> u32 {
    EMPYREAN_ABI_VERSION
}

// ── Origin-policy modes ───────────────────────────────────────────
/// Auto: selects the central body (heliocentric vs Earth-centric)
/// automatically. Default for `EmpyreanODConfig::origin`.
pub const EMPYREAN_ORIGIN_POLICY_AUTO: i32 = 0;
/// Pin IOD + DC to the central body identified by
/// `EmpyreanOriginPolicy::explicit_naif`. Skips the cascade.
/// Required for cataloged satellites where heliocentric Gauss is
/// unphysical; recommended for pipelines that already know the regime.
pub const EMPYREAN_ORIGIN_POLICY_EXPLICIT: i32 = 1;

// ── OutputEpoch modes ─────────────────────────────────────────────
pub const EMPYREAN_OUTPUT_EPOCH_MID_ARC: i32 = 0;
pub const EMPYREAN_OUTPUT_EPOCH_LAST_OBSERVATION: i32 = 1;
pub const EMPYREAN_OUTPUT_EPOCH_EXPLICIT: i32 = 2;
/// Anchor the fitted orbit at the IOD epoch (the epoch the initial-
/// orbit determination produced). Matches OrbFit's `epoch.eq0` and
/// find_orb's "anchor at most recent good fit" pattern. Useful for
/// multi-year arcs whose mid-arc target lies in a chaotic interval —
/// keeps the integrator anchor inside the IOD opposition window.
pub const EMPYREAN_OUTPUT_EPOCH_IOD_EPOCH: i32 = 3;

// ── CoordinateRepresentation codes (matches the global C-ABI mapping) ─
pub const EMPYREAN_REPRESENTATION_CARTESIAN: i32 = 0;
pub const EMPYREAN_REPRESENTATION_KEPLERIAN: i32 = 1;
pub const EMPYREAN_REPRESENTATION_COMETARY: i32 = 2;
pub const EMPYREAN_REPRESENTATION_SPHERICAL: i32 = 3;

/// Per-observation result from orbit determination or evaluation.
///
/// Mirrors scott's [`ObservationResult`](scott::results::ObservationResult)
/// — every field upstream produces is carried across the C ABI. NaN /
/// `EMPYREAN_REJECTION_NOT_EVALUATED` mark fields that aren't populated
/// for the call type (e.g. evaluate doesn't compute rejection or
/// influence diagnostics).
///
/// `obs_id` is a heap-allocated NUL-terminated UTF-8 string; the
/// pointer is freed by [`empyrean_od_result_free`] /
/// [`empyrean_evaluate_result_free`] when the parent array is freed.
/// Do NOT free it manually.
#[repr(C)]
pub struct EmpyreanObservationResult {
    /// ADES `obsID` (or scott auto-assigned). Owned by the parent array
    /// — freed by the matching `*_result_free` call.
    pub obs_id: *mut c_char,
    /// MPC observatory code (3-byte + NUL).
    pub obs_code: [u8; 4],
    /// Star catalog used for astrometric reduction (ADES `astCat`).
    /// Heap-allocated; null when ADES did not carry one. Freed with the array.
    pub ast_cat: *mut c_char,
    /// Observation epoch (MJD TDB).
    pub epoch_mjd_tdb: f64,
    /// RA·cosδ residual (observed - predicted), arcsec.
    pub ra_residual_arcsec: f64,
    /// Dec residual, arcsec.
    pub dec_residual_arcsec: f64,
    /// Mahalanobis χ² of this observation. NaN if covariance unavailable.
    pub chi2: f64,
    /// Degrees of freedom (number of non-NaN residual dimensions).
    pub dof: u32,
    /// χ² survival probability.
    pub probability: f64,
    /// Whether this observation was used in the fit (1 = yes, 0 = no).
    pub selected: u8,
    /// Combined obs+predicted RA covariance (arcsec²). NaN if absent.
    pub residual_cov_ra: f64,
    /// Combined obs+predicted Dec covariance (arcsec²). NaN if absent.
    pub residual_cov_dec: f64,
    /// Off-diagonal correlation coefficient (dimensionless, [-1, 1]). NaN if absent.
    pub residual_cov_corr: f64,
    /// Reason this observation was kept / rejected. One of the
    /// `EMPYREAN_REJECTION_*` codes; `EMPYREAN_REJECTION_NOT_EVALUATED`
    /// when the call did not run rejection (e.g. `empyrean_evaluate`).
    pub rejection_reason: i32,
    /// Criterion value (chi², Cook's D, …) tested against the threshold. NaN if not evaluated.
    pub rejection_criterion: f64,
    /// Static threshold the criterion was compared against. NaN if not evaluated.
    pub rejection_threshold: f64,
    /// Effective threshold for adaptive rejection (Layer 3). NaN otherwise.
    pub rejection_effective_threshold: f64,
    /// D-optimality information loss from removing this observation. NaN if not computed.
    pub rejection_information_loss: f64,
    /// Cook's distance. NaN if no influence pass was run.
    pub cooks_distance: f64,
    /// Scalar leverage h_ii ∈ [0, 2]. NaN if no influence pass.
    pub leverage: f64,
    /// D-optimality fractional information contribution
    /// `f_i = tr(N⁻¹ I_i)`. NaN if no influence pass.
    pub fractional_information: f64,
    /// Along-track residual (arcsec). NaN if no sky-motion rates.
    pub along_track_arcsec: f64,
    /// Cross-track residual (arcsec). NaN if no sky-motion rates.
    pub cross_track_arcsec: f64,
    /// Along-track 1σ (arcsec). NaN if unavailable.
    pub along_track_error_arcsec: f64,
    /// Cross-track 1σ (arcsec). NaN if unavailable.
    pub cross_track_error_arcsec: f64,
    /// Position angle of sky motion (degrees, East of North). NaN if unavailable.
    pub track_position_angle_deg: f64,
}

/// Aggregate residual statistics.
///
/// All angular quantities in arcseconds. NaN entries indicate the stat
/// could not be computed (e.g. AT/CT RMS when no sky-motion rates were
/// available, or weighted RMS when no weighting layer was active).
#[repr(C)]
pub struct EmpyreanResidualSummary {
    pub num_obs: usize,
    pub num_selected: usize,
    pub num_rejected: usize,
    /// Total χ² over selected observations.
    pub chi2: f64,
    /// Effective degrees of freedom (after subtracting solve-for params).
    pub dof: usize,
    /// Reduced χ² = chi2 / dof. NaN when dof ≤ 0.
    pub reduced_chi2: f64,
    pub rms_ra_arcsec: f64,
    pub rms_dec_arcsec: f64,
    /// Combined RA·cosδ + Dec residual RMS (arcsec). Matches the
    /// find_orb / OrbFit `rms` reporting convention — a single
    /// number directly comparable across tools.
    pub rms_combined_arcsec: f64,
    /// RMS weighted by the per-observation σ (matches scott's `weighted_rms`).
    pub weighted_rms_ra_arcsec: f64,
    pub weighted_rms_dec_arcsec: f64,
    /// Combined weighted RA·cosδ + Dec residual RMS (arcsec).
    pub weighted_rms_combined_arcsec: f64,
    pub mean_ra_arcsec: f64,
    pub mean_dec_arcsec: f64,
    pub std_ra_arcsec: f64,
    pub std_dec_arcsec: f64,
    /// RMS along-track residual (arcsec). NaN if no AT/CT data.
    pub rms_along_track_arcsec: f64,
    /// RMS cross-track residual (arcsec). NaN if no AT/CT data.
    pub rms_cross_track_arcsec: f64,
}

/// Acceptability sub-checks computed post-DC.
///
/// Mirrors scott's [`AcceptabilityReport`](scott::od::AcceptabilityReport).
/// Boolean fields are encoded as `u8` (0/1). When a value is unavailable
/// (e.g. AT/CT ratio with no sky-motion rates), the `_value` is NaN and
/// the corresponding `_ok` flag is 0. Always populated on
/// [`EmpyreanODResult`]; on [`EmpyreanEvaluateResult`] the report is
/// filled with NaN/0 because evaluate does not produce a fitted orbit.
#[repr(C)]
pub struct EmpyreanAcceptabilityReport {
    pub fit_acceptable: u8,
    pub extrapolation_acceptable: u8,
    pub converged_ok: u8,
    pub reduced_chi2_ok: u8,
    pub reduced_chi2_value: f64,
    pub reduced_chi2_threshold: f64,
    pub rms_ok: u8,
    pub rms_value_arcsec: f64,
    pub rms_threshold_arcsec: f64,
    pub residual_isotropy_ok: u8,
    pub at_ct_ratio_value: f64,
    pub at_ct_ratio_threshold: f64,
    pub covariance_ok: u8,
    pub arc_coverage_ok: u8,
    pub arc_days_value: f64,
    pub arc_days_threshold: f64,
    pub fractional_sigma_a_ok: u8,
    pub fractional_sigma_a_value: f64,
    pub fractional_sigma_a_threshold: f64,
}

/// One per-station bias estimate from a Schur-eliminated nuisance fit.
///
/// Mirrors [`scott::results::StationBias`]. Populated rows in the
/// returned array correspond to stations that met the
/// `min_obs_per_station` threshold; under-observed stations are absent.
/// Timing fields are populated only when a `BiasKind::StationTiming`
/// nuisance was active (currently no surface to enable it from the C
/// ABI; reserved for a planned follow-up).
///
/// `obs_code` is heap-allocated and owned by the parent array — freed
/// by [`empyrean_od_result_free`] when the result is freed. Don't free
/// it manually.
#[repr(C)]
pub struct EmpyreanStationBias {
    pub obs_code: *mut c_char,
    /// Pre-rejection observation count from this station.
    pub n_obs: usize,
    pub bias_ra_arcsec: f64,
    pub sigma_ra_arcsec: f64,
    pub bias_dec_arcsec: f64,
    pub sigma_dec_arcsec: f64,
    /// 1 when the timing bias is populated; 0 otherwise. Reserved for
    /// the planned `BiasKind::StationTiming` follow-up.
    pub has_timing: u8,
    pub bias_timing_sec: f64,
    pub sigma_timing_sec: f64,
    /// Scalar significance: max of |bᵢ|/σᵢ across populated components.
    pub significance: f64,
}

/// A complete non-gravitational acceleration model, flattened for the C ABI.
///
/// Mirrors the fields the input [`EmpyreanOrbit`] carries, so a fitted orbit's
/// non-grav can be read back off [`EmpyreanODResult::non_grav`] and re-applied
/// to an `EmpyreanOrbit` with no loss: the radial/transverse/normal
/// coefficients (A1/A2/A3, AU/day²), the Marsden–Sekanina g(r) exponents
/// (`ng_alpha`..`ng_k`; all-zero = inverse-square default), and the optional
/// thermal-lag delay `non_grav_dt` (days, valid only when `has_dt = 1`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EmpyreanNonGravParams {
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub ng_alpha: f64,
    pub ng_r0: f64,
    pub ng_m: f64,
    pub ng_n: f64,
    pub ng_k: f64,
    /// 1 when `non_grav_dt` carries a thermal-lag delay; 0 otherwise.
    pub has_dt: u8,
    /// g(r) evaluation time delay (days); only meaningful when `has_dt = 1`.
    pub non_grav_dt: f64,
    /// 1 when `covariance` carries the fitted non-grav covariance; 0 otherwise.
    pub has_covariance: u8,
    /// Fitted non-grav 3×3 covariance for (A1, A2, A3), row-major. Only
    /// meaningful when `has_covariance = 1`. Re-feeding it onto an input
    /// orbit lets a fitted orbit flow into a StateAndNonGrav refine without
    /// losing its non-grav prior.
    pub covariance: [[f64; 3]; 3],
}

/// The fitted orbit's **absolute** solar-radiation-pressure slot, flattened
/// for the C ABI.
///
/// Mirrors the SRP fields the input [`EmpyreanOrbit`] carries (`srp_amrat`,
/// `srp_cr`, `srp_amrat_variance`) so a fitted orbit's SRP force can be read
/// back off [`EmpyreanODResult::srp`] and re-applied to an `EmpyreanOrbit`
/// (`has_srp = 1`) with no loss — whether the AMRAT was solved (fitted value +
/// posterior variance) or merely carried through the fit as a fixed force.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EmpyreanSRPParams {
    /// Absolute area-to-mass ratio AMRAT (m²/kg) — the input prior plus any
    /// fitted correction.
    pub amrat: f64,
    /// Radiation-pressure coefficient Cr, carried through unchanged (fixed,
    /// never fitted).
    pub cr: f64,
    /// 1 when `amrat_variance` carries a meaningful AMRAT variance (the fitted
    /// posterior when AMRAT was solved, else the carried-through prior); 0
    /// otherwise.
    pub has_amrat_variance: u8,
    /// AMRAT variance ((m²/kg)²). Only meaningful when `has_amrat_variance = 1`.
    /// Re-feeding it opens + priors the AMRAT column in a follow-on
    /// StateAndAMRAT / StateAndNonGravAndAMRAT refine.
    pub amrat_variance: f64,
}

/// Result of orbit determination (determine or refine).
///
/// Per-axis solve-for flags (mirrors scott's `SolveFor`). Read only when
/// [`EmpyreanODConfig::solve_for`] is [`EMPYREAN_SOLVE_FOR_EXPLICIT`]; the
/// three coarse `EMPYREAN_SOLVE_FOR_*` codes cover the common shapes
/// without it. Each flag turns on a wide-STM axis, subject to its own
/// precondition (a declared prior on the orbit) enforced by scott.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EmpyreanSolveFor {
    /// Solve the Marsden A1/A2/A3 block (requires a non-grav covariance).
    pub marsden: u8,
    /// Solve the non-grav time delay DT (requires `marsden` + a DT prior).
    pub dt: u8,
    /// Solve the SRP AMRAT (requires an SRP AMRAT prior).
    pub amrat: u8,
    /// Number of thrust Δv segments to solve (3 columns each; 0 = none).
    pub thrust_segments: u32,
}

/// Full solved-parameter covariance at the ABI-frozen width
/// [`EMPYREAN_SOLVE_WIDTH`] (mirrors scott's `SolvedCovariance`
/// tag-for-tag). The leading `width × width` block is meaningful; rows and
/// columns beyond `width` are zero (RESERVED, not defaulted covariance).
/// Consumers MUST read the slot tags to locate a parameter — the width
/// alone is ambiguous (width 9 is Marsden OR a one-segment thrust). An
/// absent tag carries [`EMPYREAN_SLOT_NONE`].
#[repr(C)]
pub struct EmpyreanSolvedCovariance {
    /// Covariance at fixed storage width; leading `width×width` meaningful.
    pub matrix: [[f64; EMPYREAN_SOLVE_WIDTH]; EMPYREAN_SOLVE_WIDTH],
    /// Real solved width — 6..=17 under scott v1.14.0 (`MAX_SOLVE_WIDTH`);
    /// the struct reserves storage to 20. The leading `width × width`
    /// block is meaningful.
    pub width: u32,
    /// Slot of the first Marsden coefficient, or [`EMPYREAN_SLOT_NONE`].
    pub marsden_slot: u32,
    /// Slot of the DT scalar, or [`EMPYREAN_SLOT_NONE`].
    pub dt_slot: u32,
    /// Slot of the AMRAT scalar, or [`EMPYREAN_SLOT_NONE`].
    pub amrat_slot: u32,
    /// Slots of each fitted thrust Δv segment (3 wide each); entries
    /// `0..thrust_count` meaningful. Δv axes are INTEGRATION-frame
    /// components (see [`EmpyreanODResult::dv_frame`]).
    pub thrust_slots: [[u32; 3]; 3],
    /// Number of fitted thrust segments (0..=3).
    pub thrust_count: u32,
}

/// Post-OD photometric-fit request (mirrors scott's `PhotometryConfig`).
/// Enabled by [`EmpyreanODConfig::has_photometry`]; the fit runs after the
/// orbit is solved and never touches the state (photometry has no
/// astrometric partials). Zero-init reproduces scott's defaults.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EmpyreanPhotometryConfig {
    /// Model to fit (`EMPYREAN_PHOTOMETRY_MODEL_*`). Default = Auto (0).
    /// In Auto the fit climbs a ladder — H-only → HG12 → HG1G2 —
    /// admitting the richest model the arc's phase-angle coverage and
    /// magnitude count support, and reports the one it fit via
    /// `model_used` (never Auto). An explicit code pins a model. HG12 /
    /// HG1G2 follow Muinonen et al. (2010); H-only holds the slope fixed.
    pub model: i32,
    /// 1σ lightcurve scatter floor (mag). 0.0 → upstream default (0.2).
    pub sigma_lightcurve: f64,
    /// Include astrometrically-rejected observations' magnitudes. 0 = off.
    pub include_rejected: u8,
    /// Max Huber-IRLS iterations. 0 → upstream default (30).
    pub max_irls_iterations: u32,
    /// Huber tuning constant. 0.0 → upstream default (1.5).
    pub huber_k: f64,
}

/// Per-band photometric statistics (mirrors scott's `BandStat`). Owned
/// heap entry, freed by [`empyrean_od_result_free`].
#[repr(C)]
pub struct EmpyreanBandStat {
    /// Photometric band tag (owned C string).
    pub band: *mut c_char,
    /// Number of observations in this band.
    pub n: usize,
    /// Band→V offset applied (mag).
    pub offset_applied: f64,
    /// Mean residual in V (mag).
    pub mean_residual: f64,
    /// RMS residual in V (mag).
    pub rms: f64,
}

/// One model-ladder gate decision (mirrors scott's `GateRecord`). Owned
/// heap entry, freed by [`empyrean_od_result_free`].
#[repr(C)]
pub struct EmpyreanGateRecord {
    /// Model the gate evaluated (`EMPYREAN_PHOTOMETRY_MODEL_*`, fitted).
    pub model: i32,
    /// 1 if the model was admitted.
    pub passed: u8,
    /// Human-readable gate reason (owned C string).
    pub reason: *mut c_char,
}

/// Post-OD photometric solution (mirrors scott's `PhotometryResult`).
/// Present only when photometry was requested and ran
/// ([`EmpyreanODResult::has_photometry`]). H carries honest σ via the
/// [`covariance`](EmpyreanODPhotometryResult::covariance) block.
#[repr(C)]
pub struct EmpyreanODPhotometryResult {
    /// Fitted absolute magnitude H (mag).
    pub h: f64,
    /// First slope parameter (G / G12 / G1 by model).
    pub slope1: f64,
    /// Second slope parameter (G2 for HG1G2; unused otherwise).
    pub slope2: f64,
    /// 1 when [`covariance`](EmpyreanODPhotometryResult::covariance) is populated.
    pub has_covariance: u8,
    /// Parameter covariance (H, slope1, slope2 order).
    pub covariance: [[f64; 3]; 3],
    /// Model actually fitted (`EMPYREAN_PHOTOMETRY_MODEL_*`; never Auto).
    pub model_used: i32,
    /// Reduced χ² of the photometric fit over its used magnitudes.
    pub reduced_chi2: f64,
    /// 1 when a simplex constraint was active on the fitted slopes.
    pub constraint_active: u8,
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
    /// Phase-angle coverage of the fitted magnitudes (deg).
    pub alpha_min_deg: f64,
    pub alpha_max_deg: f64,
    pub alpha_span_deg: f64,
    /// Owned per-band statistics array; freed by [`empyrean_od_result_free`].
    pub per_band: *mut EmpyreanBandStat,
    pub num_per_band: usize,
    /// Owned model-ladder gate records; freed by [`empyrean_od_result_free`].
    pub gates: *mut EmpyreanGateRecord,
    pub num_gates: usize,
}

/// Mirrors scott's [`ODResult`](scott::od::ODResult). Carries the fitted
/// orbit, the 6×6 (or 9×9 when non-grav was solved) formal covariance,
/// the per-observation result array, the summary, the structured
/// acceptability report, and the per-station nuisance-bias array
/// when `fit_station_biases` was active.
///
/// The fitted **absolute** non-gravitational model (when one was solved
/// or carried) is exposed via [`has_non_grav`](EmpyreanODResult::has_non_grav)
/// / [`non_grav`](EmpyreanODResult::non_grav) so the orbit can be re-fed into
/// propagation / evaluate / refine without losing the force model. This is
/// distinct from [`non_grav_delta`](EmpyreanODResult::non_grav_delta), which
/// is the *cumulative correction* the fit applied and is kept for inspection.
#[repr(C)]
pub struct EmpyreanODResult {
    pub orbit: EmpyreanPropagatedState,
    pub observations: *mut EmpyreanObservationResult,
    pub num_observations: usize,
    pub summary: EmpyreanResidualSummary,
    pub iterations: u32,
    /// Convergence metric at the final DC iteration (Δx^T N Δx).
    pub update_norm: f64,
    /// Solver reached its stopping criterion (1 = yes).
    /// Equivalent to `acceptability.converged_ok`; kept for backwards
    /// compatibility with the v0.7.0 surface that pre-dated the
    /// structured acceptability report.
    pub converged: u8,
    /// Fitted 6×6 state covariance in [`covariance_representation`].
    pub covariance: [[f64; 6]; 6],
    /// Coordinate basis the 6×6 / 9×9 covariance is reported in
    /// (`EMPYREAN_REPRESENTATION_*`).
    pub covariance_representation: i32,
    /// 1 when [`covariance_9x9`] is populated (non-grav was solved).
    pub has_covariance_9x9: u8,
    /// Full 9×9 covariance over (state, A1, A2, A3) when solving for non-grav.
    pub covariance_9x9: [[f64; 9]; 9],
    /// 1 when [`non_grav_delta`] is populated.
    pub has_non_grav_delta: u8,
    /// Cumulative non-grav parameter corrections (ΔA1, ΔA2, ΔA3) when solving for non-grav.
    pub non_grav_delta: [f64; 3],
    /// 1 when [`non_grav`] carries a fitted/absolute non-gravitational model.
    pub has_non_grav: u8,
    /// The fitted orbit's **absolute** non-gravitational model (A1/A2/A3 +
    /// g(r) exponents + optional thermal-lag `dt`). Re-feed this onto the
    /// orbit for propagation / evaluate / refine. Zeroed when the orbit is
    /// gravity-only (`has_non_grav = 0`).
    pub non_grav: EmpyreanNonGravParams,
    /// Number of rejection/refit passes performed.
    pub rejection_passes: u32,
    /// Number of oppositions successfully fit.
    pub num_oppositions_fit: u32,
    /// Force model tier actually used (0=Approximate, 1=Basic, 2=Standard).
    pub force_model_used: i32,
    /// Solve-for parameter set requested on the driving config
    /// (`EMPYREAN_SOLVE_FOR_*`). Together with `has_covariance_9x9`
    /// disambiguates Auto outcomes.
    pub solve_for_used: i32,
    /// Structured fit-quality verdict. The `acceptable` flags can be
    /// checked directly; per-check values + thresholds are exposed for
    /// reporting and downstream sub-classification.
    pub acceptability: EmpyreanAcceptabilityReport,
    /// Per-station fitted nuisance biases when [`EmpyreanODConfig::fit_station_biases`]
    /// was set. Owned heap allocation; freed by [`empyrean_od_result_free`].
    /// Null + `num_station_biases = 0` when no bias fit was configured.
    pub station_biases: *mut EmpyreanStationBias,
    pub num_station_biases: usize,

    // ── Wide fitting surface (v0.9.0) ───────────────────────────────
    /// 1 when [`solved_covariance`](EmpyreanODResult::solved_covariance)
    /// is populated (any solved width > 6). 0 for a pure state-only fit.
    pub has_solved_covariance: u8,
    /// Full tagged solved-parameter covariance at the frozen width. The
    /// go-forward field for ALL solved widths (including 9); the legacy
    /// `covariance_9x9` remains for one deprecation window.
    pub solved_covariance: EmpyreanSolvedCovariance,
    /// 1 when [`dt_delta`](EmpyreanODResult::dt_delta) is populated (DT solved).
    pub has_dt_delta: u8,
    /// Cumulative non-grav time-delay correction ΔDT (days).
    pub dt_delta: f64,
    /// 1 when [`amrat_delta`](EmpyreanODResult::amrat_delta) is populated (AMRAT solved).
    pub has_amrat_delta: u8,
    /// Cumulative SRP AMRAT correction (m²/kg).
    pub amrat_delta: f64,
    /// Number of fitted thrust Δv segments (0..=3); 0 = no thrust solve.
    pub thrust_delta_count: u32,
    /// Per-segment fitted Δv in m/s, expressed in
    /// [`dv_frame`](EmpyreanODResult::dv_frame). Entries
    /// `0..thrust_delta_count` meaningful.
    pub thrust_delta_m_per_s: [[f64; 3]; 3],
    /// Integration frame the Δv components are expressed in (0=ICRF,
    /// 1=EclipticJ2000). Only meaningful when `thrust_delta_count > 0`.
    pub dv_frame: i32,
    /// 1 when [`photometry`](EmpyreanODResult::photometry) carries a fitted H/G solution.
    pub has_photometry: u8,
    /// Post-OD photometric solution when photometry was requested + ran.
    /// Owns its per-band / gate arrays (freed by `empyrean_od_result_free`).
    pub photometry: EmpyreanODPhotometryResult,
    /// 1 when [`srp`] carries a fitted/carried absolute SRP slot.
    pub has_srp: u8,
    /// The fitted orbit's **absolute** SRP slot (AMRAT + Cr + optional AMRAT
    /// variance). Re-feed this onto the orbit (`has_srp = 1`) for propagation /
    /// evaluate / refine so a fitted orbit never silently drops its SRP force.
    /// Zeroed when the orbit carries no SRP (`has_srp = 0`).
    pub srp: EmpyreanSRPParams,
}

/// Result of orbit evaluation (residuals without fitting).
///
/// Same per-observation surface as [`EmpyreanODResult`] (rejection +
/// influence fields are NaN / `NOT_EVALUATED` because evaluate does
/// not run rejection or influence passes), but no fitted orbit or
/// acceptability report.
#[repr(C)]
pub struct EmpyreanEvaluateResult {
    pub observations: *mut EmpyreanObservationResult,
    pub num_observations: usize,
    pub summary: EmpyreanResidualSummary,
}

/// Output epoch specification (mirrors [`OutputEpoch`]).
///
/// The `mode` field determines which variant is active:
/// `EMPYREAN_OUTPUT_EPOCH_MID_ARC` / `_LAST_OBSERVATION` / `_IOD_EPOCH`
/// ignore `explicit_mjd_tdb`; `_EXPLICIT` reads the field as MJD TDB.
#[repr(C)]
pub struct EmpyreanOutputEpoch {
    pub mode: i32,
    pub explicit_mjd_tdb: f64,
}

/// Origin-policy selector for the OD pipeline (mirrors
/// [`OriginPolicy`]).
///
/// `policy = EMPYREAN_ORIGIN_POLICY_AUTO` ignores `explicit_naif`;
/// `_EXPLICIT` interprets the field as the NAIF body ID of the central
/// body to pin to (e.g. 10 = Sun, 399 = Earth, 4 = Mars-barycenter).
#[repr(C)]
pub struct EmpyreanOriginPolicy {
    pub policy: i32,
    pub explicit_naif: i32,
}

/// IOD ranging tuning (mirrors the IOD section of [`scott::od::ODConfig`]).
///
/// Nested rather than flattened so callers can pass the bundle around
/// as a single value and zero-init pulls the upstream defaults
/// uniformly. Sentinel rule: `0` / `0.0` requests the upstream default;
/// `opposition_gap_days < 0` disables opposition splitting.
#[repr(C)]
pub struct EmpyreanIODConfig {
    pub max_triplet_attempts: u32,
    pub max_triplet_span_days: f64,
    /// `-1.0` disables opposition splitting; `0.0` uses upstream default (90).
    pub opposition_gap_days: f64,
    pub max_iod_arc_days: f64,
    pub curvature_snr_threshold: f64,
    pub max_iod_fractional_sigma_a: f64,
}

/// Auto-escalation policy for [`SolveForParams::Auto`]
/// (mirrors [`scott::od::AutoEscalationPolicy`]). Sentinel: `0` /
/// `0.0` → upstream default.
#[repr(C)]
pub struct EmpyreanAutoEscalationPolicy {
    pub reduced_chi2: f64,
    pub at_ct_ratio: f64,
    pub min_arc_days: f64,
    pub min_n_obs: u32,
}

/// Acceptability thresholds for the post-DC fit-quality checks
/// (mirrors [`scott::od::AcceptabilityThresholds`]). Sentinel: `0.0` →
/// upstream default.
#[repr(C)]
pub struct EmpyreanAcceptabilityThresholds {
    pub reduced_chi2: f64,
    pub rms_arcsec: f64,
    pub at_ct_ratio: f64,
    pub min_arc_days: f64,
    pub fractional_sigma_a: f64,
}

/// Per-station RA/Dec bias-fit configuration (mirrors
/// [`scott::nuisance::BiasKind::StationRaDec`]).
///
/// Activated by [`EmpyreanODConfig::fit_station_biases`]. Per-station
/// sigma overrides and `BiasScope` filtering aren't carried across the
/// C ABI yet — every active station uses `sigma_prior_arcsec` and the
/// scope is always [`BiasScope::AllStations`]. Reach for the
/// empyrean-core Rust API when you need finer control.
#[repr(C)]
pub struct EmpyreanStationRaDecConfig {
    /// Default 1-sigma prior on the RA / Dec offset (arcsec). Default = 0.3.
    pub sigma_prior_arcsec: f64,
    /// Minimum observations per station for a bias parameter to be
    /// allocated. Stations below this threshold contribute observations
    /// at face value. 0 → upstream default (5).
    pub min_obs_per_station: usize,
}

/// Outlier rejection configuration. Selects between two strategies via
/// the [`kind`](Self::kind) discriminator:
///
/// - `kind = EMPYREAN_REJECTION_KIND_ADAPTIVE` (default): mirrors
///   [`scott::rejection::AdaptiveRejectionConfig`]. Reads
///   `chi2_base` / `lambda` / `max_threshold`. Sentinels:
///   `chi2_base = 0.0` → 9.21, `lambda < 0` → 1.0,
///   `max_threshold = 0.0` → 100.0.
/// - `kind = EMPYREAN_REJECTION_KIND_CMC2003`: mirrors
///   [`scott::rejection::CMC2003Config`]. Reads `chi2_rej` / `chi2_rec`
///   (the upper / lower hysteresis thresholds). Sentinels:
///   `chi2_rej = 0.0` → 8.0, `chi2_rec = 0.0` → 7.0.
///
/// `enabled = 0` runs OD without any rejection pass — the strategy
/// fields are ignored. `enabled = 1` activates rejection.
#[repr(C)]
pub struct EmpyreanRejectionConfig {
    /// 1 = run rejection (default), 0 = skip.
    pub enabled: u8,
    /// Strategy selector — one of the `EMPYREAN_REJECTION_KIND_*`
    /// constants. Default `0` (Adaptive) keeps existing C callers
    /// working without code changes.
    pub kind: u8,
    pub chi2_base: f64,
    /// `-1.0` selects the upstream default (1.0); negative values are
    /// otherwise valid and disable adaptation when 0.0.
    pub lambda: f64,
    pub max_threshold: f64,
    /// CMC2003 upper threshold (reject when χ² > chi2_rej). 0.0 →
    /// upstream default (8.0). Ignored unless `kind ==
    /// EMPYREAN_REJECTION_KIND_CMC2003`.
    pub chi2_rej: f64,
    /// CMC2003 lower threshold (recover when χ² < chi2_rec). 0.0 →
    /// upstream default (7.0). Must be strictly less than `chi2_rej`
    /// for hysteresis to break cycles. Ignored unless `kind ==
    /// EMPYREAN_REJECTION_KIND_CMC2003`.
    pub chi2_rec: f64,
    /// Maximum rejection-refit passes. 0 → upstream default (3).
    pub max_passes: u32,
}

/// Orbit-determination configuration.
///
/// Drives `empyrean_determine`, `empyrean_evaluate`, and `empyrean_refine`.
/// Mirrors [`scott::od::ODConfig`](scott::od::ODConfig) **structurally** —
/// where scott has a nested config (e.g. `auto_escalation`,
/// `acceptability`), this surface keeps the same nesting via
/// [`EmpyreanAutoEscalationPolicy`], [`EmpyreanAcceptabilityThresholds`],
/// etc., so the C-side caller's mental model matches the upstream Rust
/// type. Sentinel rule for primitive fields: `0` / `0.0` requests the
/// upstream default; only the few fields documented inline (e.g.
/// `opposition_gap_days < 0`, `lambda < 0`) carry their own special
/// values.
///
/// IOD strategy configs (Gauss / Herget / SystematicRanging /
/// Refinement) are not exposed here — those are tens of internal
/// tuning fields that don't translate cleanly. They always run with
/// their upstream defaults; reach for the empyrean-core Rust API when
/// you need to override them.
#[repr(C)]
pub struct EmpyreanODConfig {
    // ── Shared (all OD entry points) ────────────────────────────────
    /// Force-model tier: 0=Approximate, 1=Basic, 2=Standard.
    pub force_model: i32,
    /// Integrator truncation-error tolerance (interpreted by the
    /// active integrator backend — for the default GR15 this is the
    /// relative b₆ truncation tolerance). 0.0 → upstream default
    /// (1e-9).
    pub epsilon: f64,
    /// Maximum light-time iterations. 0 → upstream default (3).
    pub max_light_time_iterations: usize,
    /// Threads for batch operations. 0 → all available cores.
    pub num_threads: usize,
    /// Output reference frame: 0=ICRF, 1=EclipticJ2000.
    pub frame: i32,
    /// Observation weighting pipeline configuration. Zero-init = the
    /// production default (VFC17 + nightly de-weighting at floor-σ
    /// policy). See [`EmpyreanWeightingConfig`].
    pub weighting: EmpyreanWeightingConfig,
    /// Catalog-bias-correction configuration. Zero-init = the
    /// production default (EFCC2020 standard resolution, loaded from
    /// the DataManager default path). See [`EmpyreanDebiasingConfig`].
    pub debiasing: EmpyreanDebiasingConfig,
    /// Number of `excluded_perturbers` in [`excluded_perturbers_naif`]; 0 = none.
    pub num_excluded_perturbers: usize,
    /// Pointer to `num_excluded_perturbers` NAIF body IDs to exclude
    /// from the perturber set (for self-determination of SB441-N16
    /// bodies). Non-owning — caller must keep the array alive for the
    /// duration of the OD call.
    pub excluded_perturbers_naif: *const i32,
    /// Origin-policy selector. Zero-init = `Auto` (heliocentric → geo-
    /// centric Earth cascade). See [`EmpyreanOriginPolicy`].
    pub origin: EmpyreanOriginPolicy,

    // ── IOD (determine only) ────────────────────────────────────────
    pub iod: EmpyreanIODConfig,

    // ── Differential correction ─────────────────────────────────────
    pub output_epoch: EmpyreanOutputEpoch,
    /// Maximum DC iterations. 0 → upstream default (100).
    pub max_iterations: u32,
    /// DC convergence tolerance on Δx^T N Δx. 0.0 → upstream default (0.1).
    pub convergence_tol: f64,
    /// Use STM-cached ephemeris updates for iterations 2+. 1 = on (default).
    pub use_stm_cache: u8,
    /// Solve-for parameter set (`EMPYREAN_SOLVE_FOR_*`). Default = Auto.
    pub solve_for: i32,
    pub auto_escalation: EmpyreanAutoEscalationPolicy,
    pub acceptability: EmpyreanAcceptabilityThresholds,
    /// Schur-eliminate per-station RA/Dec biases. 1 = enable, 0 = off (default).
    pub fit_station_biases: u8,
    /// Per-station RA/Dec bias config. Honored only when
    /// [`fit_station_biases`] is non-zero.
    pub station_radec: EmpyreanStationRaDecConfig,
    /// Use span-grouped Jacobian reuse on cache iterations. 0 = off (default).
    pub use_span_grouping: u8,

    // ── Rejection ──────────────────────────────────────────────────
    pub rejection: EmpyreanRejectionConfig,
    /// Auto-select force-model tier from IOD elements. 0 = off (default).
    pub auto_force_model: u8,
    /// Output coordinate representation for the fitted orbit + covariance
    /// (`EMPYREAN_REPRESENTATION_*`). Default = Cartesian.
    pub output_representation: i32,

    // ── Wide fitting surface (v0.9.0) ───────────────────────────────
    /// Per-axis solve-for flags, read ONLY when
    /// [`solve_for`](EmpyreanODConfig::solve_for) is
    /// [`EMPYREAN_SOLVE_FOR_EXPLICIT`]. The three coarse `solve_for` codes
    /// ignore this field.
    pub solve_for_flags: EmpyreanSolveFor,
    /// Permit solving a thrust Δv segment whose burn window is not
    /// bracketed by observations (degenerate with the state; the Gates
    /// prior then carries it). 0 = refuse loudly (default).
    pub allow_unbracketed_maneuvers: u8,
    /// 1 to run the post-OD photometric fit; 0 = off (default). When 0,
    /// [`photometry`](EmpyreanODConfig::photometry) is ignored.
    pub has_photometry: u8,
    /// Post-OD photometric-fit configuration. Honored only when
    /// [`has_photometry`](EmpyreanODConfig::has_photometry) is non-zero.
    pub photometry: EmpyreanPhotometryConfig,
}

// ── Helpers ─────────────────────────────────────────────────

fn cstr_optional(p: *mut c_char, field: &str) -> Result<Option<String>, String> {
    if p.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|e| format!("invalid UTF-8 in {field}: {e}"))?;
    Ok((!s.is_empty()).then(|| s.to_string()))
}

/// Convert C `EmpyreanObservation` array to scott's `OpticalObservation`s.
///
/// Populates the full ADES surface — perm_id / prov_id / trk_sub /
/// mode / sys / ctr / pos1-3 / rms_corr / mag / rms_mag / band /
/// ast_cat round-trip on top of the core astrometry.
pub(crate) fn c_observations_to_optical(
    obs_slice: &[EmpyreanObservation],
) -> Result<Vec<ADESObservations>, String> {
    let mut out = Vec::with_capacity(obs_slice.len());
    for obs in obs_slice {
        if obs.obs_time.is_null() {
            return Err("null obs_time pointer in observation".to_string());
        }
        let obs_time = unsafe { CStr::from_ptr(obs.obs_time) }
            .to_str()
            .map_err(|e| format!("invalid UTF-8 in obs_time: {e}"))?
            .to_string();

        let stn = std::str::from_utf8(&obs.obs_code[..3])
            .unwrap_or("   ")
            .trim_end_matches('\0')
            .trim_end()
            .to_string();

        let mut o = ADESObservations::default();
        o.perm_id = cstr_optional(obs.perm_id, "perm_id")?;
        o.prov_id = cstr_optional(obs.prov_id, "prov_id")?;
        o.trk_sub = cstr_optional(obs.trk_sub, "trk_sub")?;
        o.obs_id = cstr_optional(obs.obs_id, "obs_id")?;
        o.obs_sub_id = cstr_optional(obs.obs_sub_id, "obs_sub_id")?;
        o.trk_id = cstr_optional(obs.trk_id, "trk_id")?;
        o.mode = cstr_optional(obs.mode, "mode")?;
        o.stn = stn;
        o.prog = cstr_optional(obs.prog, "prog")?;
        o.sys = cstr_optional(obs.sys, "sys")?;
        o.ctr = (!obs.ctr.is_nan()).then_some(obs.ctr);
        o.pos1 = (!obs.pos1.is_nan()).then_some(obs.pos1);
        o.pos2 = (!obs.pos2.is_nan()).then_some(obs.pos2);
        o.pos3 = (!obs.pos3.is_nan()).then_some(obs.pos3);
        o.obs_time = obs_time;
        o.ra = obs.ra_deg;
        o.dec = obs.dec_deg;
        o.rms_ra = (!obs.rms_ra_arcsec.is_nan()).then_some(obs.rms_ra_arcsec);
        o.rms_dec = (!obs.rms_dec_arcsec.is_nan()).then_some(obs.rms_dec_arcsec);
        o.rms_corr = (!obs.rms_corr.is_nan()).then_some(obs.rms_corr);
        o.ast_cat = cstr_optional(obs.ast_cat, "ast_cat")?;
        o.mag = (!obs.mag.is_nan()).then_some(obs.mag);
        o.rms_mag = (!obs.rms_mag.is_nan()).then_some(obs.rms_mag);
        o.band = cstr_optional(obs.band, "band")?;
        o.phot_cat = cstr_optional(obs.phot_cat, "phot_cat")?;
        o.phot_ap = (!obs.phot_ap.is_nan()).then_some(obs.phot_ap);
        o.log_snr = (!obs.log_snr.is_nan()).then_some(obs.log_snr);
        o.seeing = (!obs.seeing.is_nan()).then_some(obs.seeing);
        o.exp = (!obs.exp.is_nan()).then_some(obs.exp);
        o.rms_fit = (!obs.rms_fit.is_nan()).then_some(obs.rms_fit);
        o.n_stars = if obs.n_stars >= 0 {
            Some(obs.n_stars as u32)
        } else {
            None
        };
        o.notes = cstr_optional(obs.notes, "notes")?;
        o.remarks = cstr_optional(obs.remarks, "remarks")?;
        out.push(o);
    }
    Ok(out)
}

/// Convert a C `EmpyreanRadarObservation` array to scott's
/// `RadarObservation`s.
///
/// Carries the radar surface through ADES-native — no unit conversion:
/// the delay value stays in seconds, `rms_delay_microseconds` in
/// microseconds, Doppler in Hz, `frq_mhz` in MHz. The single SI
/// normalisation happens downstream in scott's `Observation::from_radar`.
/// The `com` tri-state `i8` maps back to `Option<bool>` (`1` → `Some(true)`,
/// `0` → `Some(false)`, anything else → `None`), preserving the ADES
/// "absent means apply the center-of-mass default downstream" contract.
fn c_radar_to_scott(slice: &[EmpyreanRadarObservation]) -> Result<Vec<RadarObservation>, String> {
    let mut out = Vec::with_capacity(slice.len());
    for r in slice {
        if r.obs_time.is_null() {
            return Err("null obs_time pointer in radar observation".to_string());
        }
        let obs_time = unsafe { CStr::from_ptr(r.obs_time) }
            .to_str()
            .map_err(|e| format!("invalid UTF-8 in radar obs_time: {e}"))?
            .to_string();

        let trx = std::str::from_utf8(&r.trx[..3])
            .unwrap_or("   ")
            .trim_end_matches('\0')
            .trim_end()
            .to_string();
        let rcv = std::str::from_utf8(&r.rcv[..3])
            .unwrap_or("   ")
            .trim_end_matches('\0')
            .trim_end()
            .to_string();

        let measurement = match r.kind {
            EMPYREAN_RADAR_KIND_DELAY => RadarMeasurement::Delay {
                delay_seconds: r.delay_seconds,
                rms_delay_microseconds: r.rms_delay_microseconds,
            },
            EMPYREAN_RADAR_KIND_DOPPLER => RadarMeasurement::Doppler {
                doppler_hz: r.doppler_hz,
                rms_doppler_hz: r.rms_doppler_hz,
            },
            other => {
                return Err(format!(
                    "unsupported radar kind = {other} (expected EMPYREAN_RADAR_KIND_DELAY = {EMPYREAN_RADAR_KIND_DELAY} or EMPYREAN_RADAR_KIND_DOPPLER = {EMPYREAN_RADAR_KIND_DOPPLER})"
                ));
            }
        };

        let com = match r.com {
            1 => Some(true),
            0 => Some(false),
            _ => None,
        };

        out.push(RadarObservation {
            perm_id: cstr_optional(r.perm_id, "radar perm_id")?,
            prov_id: cstr_optional(r.prov_id, "radar prov_id")?,
            trk_sub: cstr_optional(r.trk_sub, "radar trk_sub")?,
            trx,
            rcv,
            obs_time,
            measurement,
            frq_mhz: r.frq_mhz,
            com,
            log_snr: (!r.log_snr.is_nan()).then_some(r.log_snr),
            remarks: cstr_optional(r.remarks, "radar remarks")?,
        });
    }
    Ok(out)
}

/// Marshal a scott `RadarObservation` into the C-ABI
/// [`EmpyreanRadarObservation`] — the inverse of [`c_radar_to_scott`].
///
/// Packs the record ADES-native, performing **NO** unit conversion: the
/// delay value stays in seconds, its σ in microseconds, Doppler in Hz, and
/// frequency in MHz; SI normalisation happens downstream in scott's
/// `Observation::from_radar`. The ADES `RadarValue` choice is honoured by
/// emitting the live value pair and NaN-ing the inactive one, with `kind`
/// carrying the discriminator. `com` is emitted as the tri-state i8
/// (`None` → `-1`, never `0`). String fields are heap-allocated C strings
/// (null when absent); the returned struct owns them and must be released
/// with [`empyrean_radar_observations_free`]. No field is dropped or zeroed.
///
/// Shared by [`empyrean_read_ades`] (ADES-file radar) and
/// [`empyrean_query_radar`](crate::query::empyrean_query_radar) (JPL
/// `sb_radar` live radar) so both emit byte-identical layouts.
pub(crate) fn scott_radar_to_c(r: &RadarObservation) -> EmpyreanRadarObservation {
    let mut trx = [0u8; 4];
    for (j, b) in r.trx.as_bytes().iter().take(3).enumerate() {
        trx[j] = *b;
    }
    let mut rcv = [0u8; 4];
    for (j, b) in r.rcv.as_bytes().iter().take(3).enumerate() {
        rcv[j] = *b;
    }

    // The ADES RadarValue choice: emit the live pair, NaN the other.
    let (kind, delay_seconds, rms_delay_microseconds, doppler_hz, rms_doppler_hz) =
        match r.measurement {
            RadarMeasurement::Delay {
                delay_seconds,
                rms_delay_microseconds,
            } => (
                EMPYREAN_RADAR_KIND_DELAY,
                delay_seconds,
                rms_delay_microseconds,
                f64::NAN,
                f64::NAN,
            ),
            RadarMeasurement::Doppler {
                doppler_hz,
                rms_doppler_hz,
            } => (
                EMPYREAN_RADAR_KIND_DOPPLER,
                f64::NAN,
                f64::NAN,
                doppler_hz,
                rms_doppler_hz,
            ),
        };

    fn opt_cstr(s: Option<&String>) -> *mut c_char {
        match s {
            Some(v) if !v.is_empty() => CString::new(v.as_str())
                .unwrap_or_else(|_| CString::new("").unwrap())
                .into_raw(),
            _ => std::ptr::null_mut(),
        }
    }

    EmpyreanRadarObservation {
        perm_id: opt_cstr(r.perm_id.as_ref()),
        prov_id: opt_cstr(r.prov_id.as_ref()),
        trk_sub: opt_cstr(r.trk_sub.as_ref()),
        trx,
        rcv,
        obs_time: CString::new(r.obs_time.as_str())
            .unwrap_or_else(|_| CString::new("").unwrap())
            .into_raw(),
        kind,
        delay_seconds,
        rms_delay_microseconds,
        doppler_hz,
        rms_doppler_hz,
        frq_mhz: r.frq_mhz,
        com: match r.com {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        log_snr: r.log_snr.unwrap_or(f64::NAN),
        remarks: opt_cstr(r.remarks.as_ref()),
    }
}

/// Heap-allocate a NUL-terminated C string. Empty input returns null.
fn alloc_cstring(s: &str) -> *mut c_char {
    if s.is_empty() {
        return std::ptr::null_mut();
    }
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a `*mut c_char` previously returned by [`alloc_cstring`].
unsafe fn free_cstring(p: *mut c_char) {
    if !p.is_null() {
        unsafe {
            drop(CString::from_raw(p));
        }
    }
}

fn rejection_reason_to_c(reason: RejectionReason) -> i32 {
    match reason {
        RejectionReason::Accepted => EMPYREAN_REJECTION_ACCEPTED,
        RejectionReason::ChiSquared => EMPYREAN_REJECTION_CHI_SQUARED,
        RejectionReason::SigmaClip => EMPYREAN_REJECTION_SIGMA_CLIP,
        RejectionReason::CooksDistance => EMPYREAN_REJECTION_COOKS_DISTANCE,
        RejectionReason::AdaptiveInformationAware => EMPYREAN_REJECTION_ADAPTIVE,
        RejectionReason::UnsupportedObservatory => EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY,
        RejectionReason::CMC2003 => EMPYREAN_REJECTION_CMC2003,
        RejectionReason::RadarObservationsUnsupported => EMPYREAN_REJECTION_RADAR_UNSUPPORTED,
        RejectionReason::OccultationObservationsUnsupported => {
            EMPYREAN_REJECTION_OCCULTATION_UNSUPPORTED
        }
        RejectionReason::OutsideArc => EMPYREAN_REJECTION_OUTSIDE_ARC,
    }
}

/// Map scott's per-obs result vector into a heap-allocated C array.
///
/// Each entry's `obs_id` and `ast_cat` strings are heap-allocated and
/// owned by the array. The caller frees them with the matching
/// [`empyrean_od_result_free`] / [`empyrean_evaluate_result_free`].
pub(crate) fn observation_results_to_c(
    observations: &[ObservationResult],
) -> (*mut EmpyreanObservationResult, usize) {
    let n = observations.len();
    if n == 0 {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<EmpyreanObservationResult>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanObservationResult>());
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservationResult;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, obs) in observations.iter().enumerate() {
        // 3-byte obs code + NUL.
        let mut code = [0u8; 4];
        let bytes = obs.obs_code.as_bytes();
        let take = bytes.len().min(3);
        code[..take].copy_from_slice(&bytes[..take]);

        // residual.values is arcseconds: [Δα·cosδ, Δδ].
        let res_vals = obs.residual.values;
        let res_cov = obs.residual.covariance;
        let (cov_ra, cov_dec, cov_corr) = match res_cov {
            Some(m) => {
                let s_ra = m[0][0];
                let s_dec = m[1][1];
                let off = m[0][1];
                let denom = (s_ra * s_dec).sqrt();
                let corr = if denom > 0.0 && denom.is_finite() {
                    off / denom
                } else {
                    f64::NAN
                };
                (s_ra, s_dec, corr)
            }
            None => (f64::NAN, f64::NAN, f64::NAN),
        };

        // Rejection decision (None on the evaluate path).
        let (rej_reason, rej_crit, rej_thr, rej_eff, rej_loss) = match &obs.rejection {
            Some(d) => (
                rejection_reason_to_c(d.reason),
                d.criterion_value,
                d.threshold,
                d.effective_threshold.unwrap_or(f64::NAN),
                d.information_loss.unwrap_or(f64::NAN),
            ),
            None => (
                EMPYREAN_REJECTION_NOT_EVALUATED,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ),
        };

        // Influence diagnostics (None on evaluate path).
        let (cooks, lev, frac_info) = match &obs.influence {
            Some(inf) => (inf.cooks_distance, inf.leverage, inf.fractional_information),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };

        // Along/cross-track decomposition (None when no sky-motion rates).
        let (at, ct, at_err, ct_err, pa) = match &obs.along_cross_track {
            Some(act) => (
                act.along_track,
                act.cross_track,
                act.along_track_error,
                act.cross_track_error,
                act.position_angle,
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };

        let entry = EmpyreanObservationResult {
            obs_id: alloc_cstring(&obs.obs_id),
            obs_code: code,
            ast_cat: alloc_cstring(obs.ast_cat.as_deref().unwrap_or("")),
            epoch_mjd_tdb: obs.epoch_mjd_tdb,
            ra_residual_arcsec: res_vals[0],
            dec_residual_arcsec: res_vals[1],
            chi2: obs.residual.chi2,
            dof: obs.residual.dof as u32,
            probability: obs.residual.probability,
            selected: if obs.selected { 1 } else { 0 },
            residual_cov_ra: cov_ra,
            residual_cov_dec: cov_dec,
            residual_cov_corr: cov_corr,
            rejection_reason: rej_reason,
            rejection_criterion: rej_crit,
            rejection_threshold: rej_thr,
            rejection_effective_threshold: rej_eff,
            rejection_information_loss: rej_loss,
            cooks_distance: cooks,
            leverage: lev,
            fractional_information: frac_info,
            along_track_arcsec: at,
            cross_track_arcsec: ct,
            along_track_error_arcsec: at_err,
            cross_track_error_arcsec: ct_err,
            track_position_angle_deg: pa,
        };
        unsafe {
            ptr.add(i).write(entry);
        }
    }
    (ptr, n)
}

/// Free per-entry heap allocations and the array backing.
///
/// Used by both [`empyrean_od_result_free`] and
/// [`empyrean_evaluate_result_free`].
unsafe fn free_observation_results(ptr: *mut EmpyreanObservationResult, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    for i in 0..n {
        let entry = unsafe { &mut *ptr.add(i) };
        unsafe {
            free_cstring(entry.obs_id);
            free_cstring(entry.ast_cat);
        }
        entry.obs_id = std::ptr::null_mut();
        entry.ast_cat = std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::array::<EmpyreanObservationResult>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanObservationResult>());
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

pub(crate) fn summary_to_c(summary: &ObservationResidualSummary) -> EmpyreanResidualSummary {
    EmpyreanResidualSummary {
        num_obs: summary.num_obs,
        num_selected: summary.num_selected,
        num_rejected: summary.num_rejected,
        chi2: summary.chi2,
        dof: summary.dof,
        reduced_chi2: summary.reduced_chi2,
        rms_ra_arcsec: summary.rms_ra,
        rms_dec_arcsec: summary.rms_dec,
        rms_combined_arcsec: summary.rms_combined,
        weighted_rms_ra_arcsec: summary.weighted_rms_ra,
        weighted_rms_dec_arcsec: summary.weighted_rms_dec,
        weighted_rms_combined_arcsec: summary.weighted_rms_combined,
        mean_ra_arcsec: summary.mean_ra,
        mean_dec_arcsec: summary.mean_dec,
        std_ra_arcsec: summary.std_ra,
        std_dec_arcsec: summary.std_dec,
        rms_along_track_arcsec: summary.rms_along_track,
        rms_cross_track_arcsec: summary.rms_cross_track,
    }
}

/// Map scott's `Vec<StationBias>` into a heap-allocated C array.
///
/// Returns `(null, 0)` when `biases` is None or empty. The caller frees
/// the array (and the per-row `obs_code` strings) via
/// [`free_station_biases`].
pub(crate) fn station_biases_to_c(
    biases: &Option<Vec<empyrean_core::determination::StationBias>>,
) -> (*mut EmpyreanStationBias, usize) {
    let Some(list) = biases else {
        return (std::ptr::null_mut(), 0);
    };
    let n = list.len();
    if n == 0 {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<EmpyreanStationBias>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanStationBias>());
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanStationBias;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, b) in list.iter().enumerate() {
        let (has_timing, bias_t, sigma_t) = match (b.bias_timing_sec, b.sigma_timing_sec) {
            (Some(bt), Some(st)) => (1u8, bt, st),
            _ => (0u8, f64::NAN, f64::NAN),
        };
        let entry = EmpyreanStationBias {
            obs_code: alloc_cstring(&b.obs_code),
            n_obs: b.n_obs,
            bias_ra_arcsec: b.bias_ra_arcsec,
            sigma_ra_arcsec: b.sigma_ra_arcsec,
            bias_dec_arcsec: b.bias_dec_arcsec,
            sigma_dec_arcsec: b.sigma_dec_arcsec,
            has_timing,
            bias_timing_sec: bias_t,
            sigma_timing_sec: sigma_t,
            significance: b.significance,
        };
        unsafe {
            ptr.add(i).write(entry);
        }
    }
    (ptr, n)
}

unsafe fn free_station_biases(ptr: *mut EmpyreanStationBias, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    for i in 0..n {
        let entry = unsafe { &mut *ptr.add(i) };
        unsafe {
            free_cstring(entry.obs_code);
        }
        entry.obs_code = std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::array::<EmpyreanStationBias>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanStationBias>());
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

// ── Wide-fitting marshaling (v0.9.0) ────────────────────────────────

/// `FittedPhotometryModel` → an `EMPYREAN_PHOTOMETRY_MODEL_*` code
/// (never Auto — Auto is a request, not a result).
fn fitted_photometry_model_to_int(m: &FittedPhotometryModel) -> i32 {
    match m {
        FittedPhotometryModel::HOnly => EMPYREAN_PHOTOMETRY_MODEL_HONLY,
        FittedPhotometryModel::HG => EMPYREAN_PHOTOMETRY_MODEL_HG,
        FittedPhotometryModel::HG12 => EMPYREAN_PHOTOMETRY_MODEL_HG12,
        FittedPhotometryModel::HG1G2 => EMPYREAN_PHOTOMETRY_MODEL_HG1G2,
    }
}

/// `Option<usize>` slot tag → `u32` with the [`EMPYREAN_SLOT_NONE`] sentinel.
fn slot_to_c(s: Option<usize>) -> u32 {
    s.map(|v| v as u32).unwrap_or(EMPYREAN_SLOT_NONE)
}

/// Copy scott's `SolvedCovariance` into the ABI-frozen struct: the leading
/// scott-width block into the frozen `W×W` matrix (scott's `MAX_SOLVE_WIDTH`
/// ≤ `EMPYREAN_SOLVE_WIDTH`, so the whole scott block fits and rows beyond
/// the solved width are already zero), slot tags with sentinels.
fn solved_covariance_to_c(sc: &SolvedCovariance) -> EmpyreanSolvedCovariance {
    let mut matrix = [[0.0_f64; EMPYREAN_SOLVE_WIDTH]; EMPYREAN_SOLVE_WIDTH];
    // scott's MAX_SOLVE_WIDTH (17) ≤ EMPYREAN_SOLVE_WIDTH (20): zip copies
    // the whole scott block into the leading frozen block; rows / cols
    // beyond scott's stored width are already zero.
    for (dst_row, src_row) in matrix.iter_mut().zip(sc.matrix.iter()) {
        for (dst, src) in dst_row.iter_mut().zip(src_row.iter()) {
            *dst = *src;
        }
    }
    // Unused thrust rows (i >= thrust_count) carry the SLOT_NONE sentinel
    // rather than scott's raw [0, 0, 0]: slot 0 is a valid STATE slot, so a
    // consumer that forgets to gate on thrust_count must not be able to
    // misread an unused row as "thrust at state slots 0,1,2".
    let mut thrust_slots = [[EMPYREAN_SLOT_NONE; 3]; 3];
    for (i, seg) in sc.thrust_slots.iter().enumerate().take(sc.thrust_count) {
        for (j, &slot) in seg.iter().enumerate() {
            thrust_slots[i][j] = slot as u32;
        }
    }
    EmpyreanSolvedCovariance {
        matrix,
        width: sc.width as u32,
        marsden_slot: slot_to_c(sc.marsden_slot),
        dt_slot: slot_to_c(sc.dt_slot),
        amrat_slot: slot_to_c(sc.amrat_slot),
        thrust_slots,
        thrust_count: sc.thrust_count as u32,
    }
}

fn zeroed_solved_covariance() -> EmpyreanSolvedCovariance {
    EmpyreanSolvedCovariance {
        matrix: [[0.0; EMPYREAN_SOLVE_WIDTH]; EMPYREAN_SOLVE_WIDTH],
        width: 0,
        marsden_slot: EMPYREAN_SLOT_NONE,
        dt_slot: EMPYREAN_SLOT_NONE,
        amrat_slot: EMPYREAN_SLOT_NONE,
        thrust_slots: [[EMPYREAN_SLOT_NONE; 3]; 3],
        thrust_count: 0,
    }
}

fn band_stats_to_c(list: &[empyrean_core::photometry::BandStat]) -> (*mut EmpyreanBandStat, usize) {
    let n = list.len();
    if n == 0 {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<EmpyreanBandStat>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanBandStat>());
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanBandStat;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, b) in list.iter().enumerate() {
        let entry = EmpyreanBandStat {
            band: alloc_cstring(&b.band),
            n: b.n,
            offset_applied: b.offset_applied,
            mean_residual: b.mean_residual,
            rms: b.rms,
        };
        unsafe {
            ptr.add(i).write(entry);
        }
    }
    (ptr, n)
}

unsafe fn free_band_stats(ptr: *mut EmpyreanBandStat, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    for i in 0..n {
        let entry = unsafe { &mut *ptr.add(i) };
        unsafe {
            free_cstring(entry.band);
        }
        entry.band = std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::array::<EmpyreanBandStat>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanBandStat>());
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

fn gate_records_to_c(
    list: &[empyrean_core::photometry::GateRecord],
) -> (*mut EmpyreanGateRecord, usize) {
    let n = list.len();
    if n == 0 {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<EmpyreanGateRecord>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanGateRecord>());
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanGateRecord;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, g) in list.iter().enumerate() {
        let entry = EmpyreanGateRecord {
            model: fitted_photometry_model_to_int(&g.model),
            passed: g.passed as u8,
            reason: alloc_cstring(&g.reason),
        };
        unsafe {
            ptr.add(i).write(entry);
        }
    }
    (ptr, n)
}

unsafe fn free_gate_records(ptr: *mut EmpyreanGateRecord, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    for i in 0..n {
        let entry = unsafe { &mut *ptr.add(i) };
        unsafe {
            free_cstring(entry.reason);
        }
        entry.reason = std::ptr::null_mut();
    }
    let layout = std::alloc::Layout::array::<EmpyreanGateRecord>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanGateRecord>());
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout);
    }
}

fn photometry_result_to_c(p: &PhotometryResult) -> EmpyreanODPhotometryResult {
    let (has_covariance, covariance) = match p.params.covariance {
        Some(c) => (1u8, c),
        None => (0u8, [[0.0; 3]; 3]),
    };
    let (per_band, num_per_band) = band_stats_to_c(&p.per_band);
    let (gates, num_gates) = gate_records_to_c(&p.gates);
    EmpyreanODPhotometryResult {
        h: p.params.p1,
        slope1: p.params.p2,
        slope2: p.params.p3,
        has_covariance,
        covariance,
        model_used: fitted_photometry_model_to_int(&p.model_used),
        reduced_chi2: p.reduced_chi2,
        constraint_active: p.constraint_active as u8,
        n_mags_used: p.n_mags_used,
        n_mags_rejected_photometric: p.n_mags_rejected_photometric,
        n_obs_without_mags: p.n_obs_without_mags,
        n_mags_from_astrometric_selected: p.n_mags_from_astrometric_selected,
        n_mags_from_astrometric_rejected: p.n_mags_from_astrometric_rejected,
        alpha_min_deg: p.phase_coverage.alpha_min_deg,
        alpha_max_deg: p.phase_coverage.alpha_max_deg,
        alpha_span_deg: p.phase_coverage.span_deg,
        per_band,
        num_per_band,
        gates,
        num_gates,
    }
}

fn zeroed_photometry_result() -> EmpyreanODPhotometryResult {
    EmpyreanODPhotometryResult {
        h: 0.0,
        slope1: 0.0,
        slope2: 0.0,
        has_covariance: 0,
        covariance: [[0.0; 3]; 3],
        model_used: EMPYREAN_PHOTOMETRY_MODEL_AUTO,
        reduced_chi2: 0.0,
        constraint_active: 0,
        n_mags_used: 0,
        n_mags_rejected_photometric: 0,
        n_obs_without_mags: 0,
        n_mags_from_astrometric_selected: 0,
        n_mags_from_astrometric_rejected: 0,
        alpha_min_deg: 0.0,
        alpha_max_deg: 0.0,
        alpha_span_deg: 0.0,
        per_band: std::ptr::null_mut(),
        num_per_band: 0,
        gates: std::ptr::null_mut(),
        num_gates: 0,
    }
}

/// Populate the v0.9.0 wide-fitting fields on a result out-pointer from
/// scott's `ODResult`. ALWAYS writes every field (zeros / sentinels when
/// an axis was not solved) — no defaulted covariance presented as real,
/// per the full-population contract.
unsafe fn populate_wide_fitting_fields(result_out: *mut EmpyreanODResult, od: &ODResult) {
    unsafe {
        match &od.solved_covariance {
            Some(sc) => {
                (*result_out).has_solved_covariance = 1;
                (*result_out).solved_covariance = solved_covariance_to_c(sc);
            }
            None => {
                (*result_out).has_solved_covariance = 0;
                (*result_out).solved_covariance = zeroed_solved_covariance();
            }
        }
        match od.dt_delta {
            Some(d) => {
                (*result_out).has_dt_delta = 1;
                (*result_out).dt_delta = d;
            }
            None => {
                (*result_out).has_dt_delta = 0;
                (*result_out).dt_delta = 0.0;
            }
        }
        match od.amrat_delta {
            Some(a) => {
                (*result_out).has_amrat_delta = 1;
                (*result_out).amrat_delta = a;
            }
            None => {
                (*result_out).has_amrat_delta = 0;
                (*result_out).amrat_delta = 0.0;
            }
        }
        // Absolute fitted/carried SRP slot for lossless re-feed (parity with
        // `non_grav` above — the SRP force must survive a fitted-orbit round
        // trip, not just its correction `amrat_delta`).
        let (has_srp, srp) = od_result_srp_to_c(od);
        (*result_out).has_srp = has_srp;
        (*result_out).srp = srp;
        let mut thrust = [[0.0_f64; 3]; 3];
        let count = match od.thrust_delta_m_per_s() {
            Some(dvs) => {
                for (i, dv) in dvs.iter().take(3).enumerate() {
                    thrust[i] = *dv;
                }
                dvs.len().min(3) as u32
            }
            None => 0,
        };
        (*result_out).thrust_delta_count = count;
        (*result_out).thrust_delta_m_per_s = thrust;
        (*result_out).dv_frame = od.dv_frame.map(frame_to_int).unwrap_or(0);
        match &od.photometry {
            Some(p) => {
                (*result_out).has_photometry = 1;
                (*result_out).photometry = photometry_result_to_c(p);
            }
            None => {
                (*result_out).has_photometry = 0;
                (*result_out).photometry = zeroed_photometry_result();
            }
        }
    }
}

pub(crate) fn acceptability_to_c(r: &AcceptabilityReport) -> EmpyreanAcceptabilityReport {
    EmpyreanAcceptabilityReport {
        fit_acceptable: u8::from(r.fit_acceptable),
        extrapolation_acceptable: u8::from(r.extrapolation_acceptable),
        converged_ok: u8::from(r.converged_ok),
        reduced_chi2_ok: u8::from(r.reduced_chi2_ok),
        reduced_chi2_value: r.reduced_chi2_value,
        reduced_chi2_threshold: r.reduced_chi2_threshold,
        rms_ok: u8::from(r.rms_ok),
        rms_value_arcsec: r.rms_value_arcsec,
        rms_threshold_arcsec: r.rms_threshold_arcsec,
        residual_isotropy_ok: u8::from(r.residual_isotropy_ok),
        at_ct_ratio_value: r.at_ct_ratio_value,
        at_ct_ratio_threshold: r.at_ct_ratio_threshold,
        covariance_ok: u8::from(r.covariance_ok),
        arc_coverage_ok: u8::from(r.arc_coverage_ok),
        arc_days_value: r.arc_days_value,
        arc_days_threshold: r.arc_days_threshold,
        fractional_sigma_a_ok: u8::from(r.fractional_sigma_a_ok),
        fractional_sigma_a_value: r.fractional_sigma_a_value,
        fractional_sigma_a_threshold: r.fractional_sigma_a_threshold,
    }
}

fn force_model_tier_to_int(tier: ForceModelTier) -> i32 {
    match tier {
        ForceModelTier::Approximate => 0,
        ForceModelTier::Basic => 1,
        ForceModelTier::Standard => 2,
        // empyrean-core's enum is `#[non_exhaustive]` — defensively map
        // any future tier to Standard rather than panicking.
        _ => 2,
    }
}

/// scott's [`ODResult::force_model_used`] carries villeneuve's tier
/// directly (not the empyrean-core wrapper). Convert via the upstream
/// 1-to-1 mapping defined in `empyrean-core/src/data.rs`.
fn v_force_model_tier_to_int(t: UpstreamForceModelTier) -> i32 {
    match ForceModelTier::try_from(t) {
        Ok(tier) => force_model_tier_to_int(tier),
        // Defensively map an unknown villeneuve tier to Standard.
        Err(_) => 2,
    }
}

fn solve_for_to_int(s: &SolveForParams) -> i32 {
    match s {
        SolveForParams::Auto => EMPYREAN_SOLVE_FOR_AUTO,
        SolveForParams::Explicit(sf) => {
            let state_only = !sf.marsden && !sf.dt && !sf.amrat && sf.thrust_segments == 0;
            let non_grav_only = sf.marsden && !sf.dt && !sf.amrat && sf.thrust_segments == 0;
            if state_only {
                EMPYREAN_SOLVE_FOR_STATE_ONLY
            } else if non_grav_only {
                EMPYREAN_SOLVE_FOR_STATE_AND_NONGRAV
            } else {
                // DT / AMRAT / thrust (or a combination) — not nameable
                // by the coarse codes; the EmpyreanSolveFor flag struct
                // carries the exact axes.
                EMPYREAN_SOLVE_FOR_EXPLICIT
            }
        }
    }
}

fn coord_rep_to_int(r: CoordinateRepresentation) -> i32 {
    match r {
        CoordinateRepresentation::Cartesian => EMPYREAN_REPRESENTATION_CARTESIAN,
        CoordinateRepresentation::Keplerian => EMPYREAN_REPRESENTATION_KEPLERIAN,
        CoordinateRepresentation::Cometary => EMPYREAN_REPRESENTATION_COMETARY,
        CoordinateRepresentation::Spherical => EMPYREAN_REPRESENTATION_SPHERICAL,
    }
}

/// Pull the (single-row) Cartesian state out of an `Orbits<AU>` and
/// pack it into the C ABI's `EmpyreanPropagatedState`.
/// Read the fitted orbit's **absolute** non-gravitational model off an
/// [`ODResult`] and flatten it for the C ABI. Returns `(0, default)` for a
/// gravity-only orbit. The g(r) exponents are pulled straight off the
/// `GFunction` (its fields are public). `NonGravModel` is Marsden-only in
/// v1.20.0 — SRP is a separate first-class slot (`SRPForceParams`), no
/// longer a non-grav model variant.
fn od_result_non_grav_to_c(od: &ODResult) -> (u8, EmpyreanNonGravParams) {
    match od.orbit.non_grav_params(0) {
        Some(ng) => {
            let (ng_alpha, ng_r0, ng_m, ng_n, ng_k) = match &ng.model {
                // Normalize the inverse-square default (α=1, r0=1, m=2, n=0,
                // k=0) back to all-zeros so it matches the C-ABI **input**
                // convention (all-zero g(r) = inverse-square). Keeps the
                // round-trip lossless and the model label honest.
                NonGravModel::MarsdenSekanina(g)
                    if g.alpha == 1.0 && g.r0 == 1.0 && g.m == 2.0 && g.n == 0.0 && g.k == 0.0 =>
                {
                    (0.0, 0.0, 0.0, 0.0, 0.0)
                }
                NonGravModel::MarsdenSekanina(g) => (g.alpha, g.r0, g.m, g.n, g.k),
            };
            let (has_dt, non_grav_dt) = match ng.dt {
                Some(d) => (1u8, d),
                None => (0u8, f64::NAN),
            };
            // Fitted non-grav covariance: carry it out so the
            // re-feedable orbit keeps its non-grav prior for a StateAndNonGrav
            // refine. Source it from the authoritative 9×9 posterior block
            // `od.covariance_9x9[6..9][6..9]` rather than the orbit's own
            // non-grav covariance — the latter can still carry the escalation
            // seed prior on the determine path, whereas
            // `covariance_9x9` is always the final fitted posterior.
            let (has_covariance, covariance) = match od.covariance_9x9 {
                Some(c9) => {
                    let mut c = [[0.0_f64; 3]; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            c[i][j] = c9[6 + i][6 + j];
                        }
                    }
                    (1u8, c)
                }
                None => (0u8, [[0.0_f64; 3]; 3]),
            };
            (
                1,
                EmpyreanNonGravParams {
                    a1: ng.a1,
                    a2: ng.a2,
                    a3: ng.a3,
                    ng_alpha,
                    ng_r0,
                    ng_m,
                    ng_n,
                    ng_k,
                    has_dt,
                    non_grav_dt,
                    has_covariance,
                    covariance,
                },
            )
        }
        None => (0, EmpyreanNonGravParams::default()),
    }
}

/// Read the fitted orbit's **absolute** SRP slot off an [`ODResult`] and
/// flatten it for the C ABI. Returns `(0, default)` when the orbit carries no
/// SRP. The AMRAT variance prefers the fitted **posterior** from the tagged
/// solved covariance (`amrat_slot`, when AMRAT was solved) over the orbit's
/// carried-through prior — mirroring how [`od_result_non_grav_to_c`] sources
/// the Marsden covariance from the 9×9 posterior — so a re-fed orbit chains
/// the correct Bayesian prior into a follow-on StateAndAMRAT refine.
fn od_result_srp_to_c(od: &ODResult) -> (u8, EmpyreanSRPParams) {
    match od.orbit.srp_params(0) {
        Some(srp) => {
            let posterior = od
                .solved_covariance
                .as_ref()
                .and_then(|sc| sc.amrat_slot.map(|s| sc.matrix[s][s]));
            let (has_amrat_variance, amrat_variance) = match posterior.or(srp.amrat_variance) {
                Some(v) => (1u8, v),
                None => (0u8, f64::NAN),
            };
            (
                1,
                EmpyreanSRPParams {
                    amrat: srp.amrat,
                    cr: srp.cr,
                    has_amrat_variance,
                    amrat_variance,
                },
            )
        }
        None => (0, EmpyreanSRPParams::default()),
    }
}

fn od_orbit_to_propagated(
    orbit: &Orbits<AU>,
    covariance: &[[f64; 6]; 6],
) -> Result<EmpyreanPropagatedState, String> {
    let (_id, coord) = orbit
        .get(0)
        .ok_or_else(|| "OD result orbit is empty".to_string())?;
    let (epoch, x, y, z, vx, vy, vz, frame, origin) = match coord {
        Coordinates::Cartesian(c, _, _) => {
            (c.t, c.x, c.y, c.z, c.vx, c.vy, c.vz, c.frame, c.origin)
        }
        _ => return Err("OD result orbit is not in Cartesian representation".to_string()),
    };
    Ok(EmpyreanPropagatedState {
        epoch_mjd_tdb: epoch.mjd_tdb(),
        x,
        y,
        z,
        vx,
        vy,
        vz,
        origin: origin.naif_id(),
        frame: frame_to_int(frame),
        covariance: *covariance,
        has_covariance: 1,
        stm: [[0.0; 6]; 6],
        has_stm: 0,
        stt: [[[0.0; 6]; 6]; 6],
        has_stt: 0,
        resolved_kind: 0,
    })
}

fn int_to_solve_for(v: i32) -> Result<SolveForParams, String> {
    match v {
        EMPYREAN_SOLVE_FOR_STATE_ONLY => Ok(SolveForParams::state_only()),
        EMPYREAN_SOLVE_FOR_STATE_AND_NONGRAV => Ok(SolveForParams::state_and_non_grav()),
        EMPYREAN_SOLVE_FOR_AUTO => Ok(SolveForParams::Auto),
        EMPYREAN_SOLVE_FOR_EXPLICIT => Err(
            "solve_for = EXPLICIT (3) requires the per-axis solve_for flags \
             (marsden / dt / amrat / thrust_segments); pass them via the \
             EmpyreanSolveFor flag struct on EmpyreanODConfig"
                .to_string(),
        ),
        other => Err(format!("unknown solve_for code: {other}")),
    }
}

/// Map the config photometry-model code (`EMPYREAN_PHOTOMETRY_MODEL_*`)
/// to scott's [`PhotometryModel`].
fn photometry_model_from_int(v: i32) -> Result<PhotometryModel, String> {
    match v {
        EMPYREAN_PHOTOMETRY_MODEL_AUTO => Ok(PhotometryModel::Auto),
        EMPYREAN_PHOTOMETRY_MODEL_HONLY => Ok(PhotometryModel::HOnly),
        EMPYREAN_PHOTOMETRY_MODEL_HG => Ok(PhotometryModel::HG),
        EMPYREAN_PHOTOMETRY_MODEL_HG12 => Ok(PhotometryModel::HG12),
        EMPYREAN_PHOTOMETRY_MODEL_HG1G2 => Ok(PhotometryModel::HG1G2),
        other => Err(format!("unknown photometry model code: {other}")),
    }
}

/// Build a scott [`PhotometryConfig`] from the C request. Sentinel rule:
/// `0` / `0.0` on a tuning field requests the upstream default.
fn photometry_config_from_c(c: &EmpyreanPhotometryConfig) -> Result<PhotometryConfig, String> {
    let mut pc = PhotometryConfig {
        model: photometry_model_from_int(c.model)?,
        ..PhotometryConfig::default()
    };
    if c.sigma_lightcurve > 0.0 {
        pc.sigma_lightcurve = c.sigma_lightcurve;
    }
    pc.include_rejected = c.include_rejected != 0;
    if c.max_irls_iterations > 0 {
        pc.max_irls_iterations = c.max_irls_iterations as usize;
    }
    if c.huber_k > 0.0 {
        pc.huber_k = c.huber_k;
    }
    Ok(pc)
}

fn int_to_coord_rep(v: i32) -> Result<CoordinateRepresentation, String> {
    match v {
        EMPYREAN_REPRESENTATION_CARTESIAN => Ok(CoordinateRepresentation::Cartesian),
        EMPYREAN_REPRESENTATION_KEPLERIAN => Ok(CoordinateRepresentation::Keplerian),
        EMPYREAN_REPRESENTATION_COMETARY => Ok(CoordinateRepresentation::Cometary),
        EMPYREAN_REPRESENTATION_SPHERICAL => Ok(CoordinateRepresentation::Spherical),
        other => Err(format!("unknown output_representation code: {other}")),
    }
}

fn build_weighting_from_c(
    c: &EmpyreanWeightingConfig,
) -> Result<Option<empyrean_core::determination::WeightingConfig>, String> {
    use empyrean_core::determination::{SigmaPolicy, WeightingConfig, WeightingLayer};
    use empyrean_core::time::Epoch;

    if c.enabled == 0 {
        return Ok(None);
    }

    // Zero-init `preset = NONE` + zero-init layers list = "use the
    // production default" so callers that don't set anything keep
    // pre-structured-config behavior.
    let preset_is_none_zero_init =
        c.preset == EMPYREAN_WEIGHTING_PRESET_NONE && c.num_additional_layers == 0;
    let effective_preset = if preset_is_none_zero_init {
        EMPYREAN_WEIGHTING_PRESET_VFC17
    } else {
        c.preset
    };

    let mut wcfg = match effective_preset {
        EMPYREAN_WEIGHTING_PRESET_NONE => WeightingConfig {
            default_sigma_arcsec: if c.default_sigma_arcsec > 0.0 {
                c.default_sigma_arcsec
            } else {
                1.0
            },
            layers: Vec::new(),
            sigma_policy: SigmaPolicy::default(),
        },
        EMPYREAN_WEIGHTING_PRESET_VFC17 => WeightingConfig::veres_farnocchia_chesley_2017(),
        EMPYREAN_WEIGHTING_PRESET_NEODYS => WeightingConfig::neodys()
            .map_err(|e| format!("failed to load NEODyS weighting preset: {e}"))?,
        other => {
            return Err(format!(
                "unsupported weighting.preset = {other} (expected NONE = {} / VFC17 = {} / NEODYS = {})",
                EMPYREAN_WEIGHTING_PRESET_NONE,
                EMPYREAN_WEIGHTING_PRESET_VFC17,
                EMPYREAN_WEIGHTING_PRESET_NEODYS,
            ));
        }
    };

    if c.sigma_policy >= 0 {
        wcfg.sigma_policy = match c.sigma_policy {
            EMPYREAN_SIGMA_POLICY_DEFAULT_ONLY => SigmaPolicy::DefaultOnly,
            EMPYREAN_SIGMA_POLICY_FLOOR => SigmaPolicy::Floor,
            other => {
                return Err(format!("unsupported weighting.sigma_policy = {other}"));
            }
        };
    }

    if c.num_additional_layers > 0 && !c.additional_layers.is_null() {
        let slice =
            unsafe { std::slice::from_raw_parts(c.additional_layers, c.num_additional_layers) };
        for layer in slice {
            let parsed = match layer.kind {
                EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE => {
                    let code = String::from_utf8_lossy(
                        &layer.obs_code[..layer
                            .obs_code
                            .iter()
                            .position(|&b| b == 0)
                            .unwrap_or(layer.obs_code.len())],
                    )
                    .trim()
                    .to_string();
                    if code.is_empty() {
                        return Err("weighting.ObservatoryRule layer has empty obs_code".into());
                    }
                    let start_epoch = if layer.start_epoch_mjd_tdb.is_finite() {
                        Some(Epoch::from_mjd_tdb(layer.start_epoch_mjd_tdb))
                    } else {
                        None
                    };
                    let end_epoch = if layer.end_epoch_mjd_tdb.is_finite() {
                        Some(Epoch::from_mjd_tdb(layer.end_epoch_mjd_tdb))
                    } else {
                        None
                    };
                    let scale = if layer.scale > 0.0 { layer.scale } else { 1.0 };
                    WeightingLayer::ObservatoryRule {
                        obs_code: code,
                        sigma: [layer.sigma_ra_arcsec, layer.sigma_dec_arcsec],
                        start_epoch,
                        end_epoch,
                        scale,
                    }
                }
                EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING => {
                    let max_gap_days = if layer.max_gap_days > 0.0 {
                        layer.max_gap_days
                    } else {
                        0.5
                    };
                    WeightingLayer::NightlyDeweighting { max_gap_days }
                }
                other => {
                    return Err(format!(
                        "unsupported weighting layer kind = {other} (expected OBSERVATORY_RULE = {} / NIGHTLY_DEWEIGHTING = {})",
                        EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE,
                        EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING,
                    ));
                }
            };
            wcfg.layers.push(parsed);
        }
    }

    Ok(Some(wcfg))
}

/// Three-state debiasing decision from the C ABI surface.
///
/// The conversion has to differentiate "user said disable" (which should
/// override scott's default) from "user said use the default" (which
/// should leave scott's lazy-loaded default in place). A bare
/// `Result<Option<Arc<DebiasingTable>>, _>` collapses those two cases
/// onto `Ok(None)`, so the caller would silently disable debiasing when
/// the user expected the production default.
enum DebiasingChoice {
    /// `enabled = 1` + null `bias_dat_path` — leave `cfg.debiasing` at
    /// scott's `ODConfig::default()` value (lazy-loads `bias.dat` from
    /// the platform data directory, e.g.
    /// `~/.local/share/empyrean/data/bias.dat` on Linux).
    KeepDefault,
    /// `enabled = 0` — explicit disable.
    Disable,
    /// `enabled = 1` + explicit path — load the table from disk.
    Override(std::sync::Arc<empyrean_core::determination::DebiasingTable>),
}

fn build_debiasing_from_c(c: &EmpyreanDebiasingConfig) -> Result<DebiasingChoice, String> {
    use empyrean_core::determination::{DebiasingResolution, DebiasingTable};
    use std::ffi::CStr;
    use std::sync::Arc;

    if c.enabled == 0 {
        return Ok(DebiasingChoice::Disable);
    }

    if c.table_id != EMPYREAN_DEBIASING_TABLE_EFCC2020 {
        return Err(format!(
            "unsupported debiasing.table_id = {} (expected EFCC2020 = {})",
            c.table_id, EMPYREAN_DEBIASING_TABLE_EFCC2020,
        ));
    }
    let resolution = match c.resolution {
        EMPYREAN_DEBIASING_RESOLUTION_STANDARD => DebiasingResolution::Standard,
        EMPYREAN_DEBIASING_RESOLUTION_HIRES => DebiasingResolution::Hires,
        other => {
            return Err(format!(
                "unsupported debiasing.resolution = {other} (expected STANDARD = {} / HIRES = {})",
                EMPYREAN_DEBIASING_RESOLUTION_STANDARD, EMPYREAN_DEBIASING_RESOLUTION_HIRES,
            ));
        }
    };

    if c.bias_dat_path.is_null() {
        // Caveat: this path doesn't honor an explicit non-default
        // resolution. If callers need Hires they must pass an
        // explicit path; the DataManager-default lazy-load is
        // hard-coded to Standard.
        let _ = resolution;
        return Ok(DebiasingChoice::KeepDefault);
    }

    let path_cstr = unsafe { CStr::from_ptr(c.bias_dat_path) };
    let path = path_cstr
        .to_str()
        .map_err(|e| format!("debiasing.bias_dat_path is not valid UTF-8: {e}"))?;

    let dir = std::path::Path::new(path)
        .parent()
        .ok_or_else(|| format!("debiasing.bias_dat_path has no parent directory: {path}"))?;

    DebiasingTable::load(dir, resolution)
        .map(Arc::new)
        .map(DebiasingChoice::Override)
        .map_err(|e| format!("failed to load debiasing table from {path}: {e}"))
}

fn build_rejection_strategy_from_c(
    rej: &EmpyreanRejectionConfig,
) -> Result<Option<empyrean_core::determination::RejectionStrategy>, String> {
    if rej.enabled == 0 {
        return Ok(None);
    }
    Ok(Some(match rej.kind {
        EMPYREAN_REJECTION_KIND_ADAPTIVE => {
            let mut r = AdaptiveRejectionConfig::default();
            if rej.chi2_base > 0.0 {
                r.chi2_base = rej.chi2_base;
            }
            if rej.lambda >= 0.0 {
                r.lambda = rej.lambda;
            }
            if rej.max_threshold > 0.0 {
                r.max_threshold = rej.max_threshold;
            }
            empyrean_core::determination::RejectionStrategy::Adaptive(r)
        }
        EMPYREAN_REJECTION_KIND_CMC2003 => {
            let mut r = CMC2003Config::default();
            if rej.chi2_rej > 0.0 {
                r.chi2_rej = rej.chi2_rej;
            }
            if rej.chi2_rec > 0.0 {
                r.chi2_rec = rej.chi2_rec;
            }
            r.validate().map_err(|e| format!("CMC2003 config: {e}"))?;
            empyrean_core::determination::RejectionStrategy::CMC2003(r)
        }
        other => {
            return Err(format!(
                "unsupported rejection.kind = {other} (expected EMPYREAN_REJECTION_KIND_ADAPTIVE = {EMPYREAN_REJECTION_KIND_ADAPTIVE} or EMPYREAN_REJECTION_KIND_CMC2003 = {EMPYREAN_REJECTION_KIND_CMC2003})"
            ));
        }
    }))
}

fn build_od_config_from_c(c: &EmpyreanODConfig) -> Result<ODConfig, String> {
    let fm = int_to_force_model(c.force_model)?;
    let mut cfg = ODConfig::default();
    cfg.force_model = fm.into();

    // ── Shared ────────────────────────────────────────────────────
    if c.epsilon > 0.0 {
        cfg.epsilon = c.epsilon;
    }
    if c.max_light_time_iterations > 0 {
        cfg.max_light_time_iterations = c.max_light_time_iterations;
    }
    cfg.num_threads = std::num::NonZeroUsize::new(c.num_threads);
    cfg.weighting = build_weighting_from_c(&c.weighting)?;
    match build_debiasing_from_c(&c.debiasing)? {
        DebiasingChoice::KeepDefault => {
            // Leave `cfg.debiasing` at the value `ODConfig::default()`
            // installed (scott's lazy-loaded EFCC2020). Overriding here
            // would silently disable debiasing in the FFI path while
            // the direct-scott path keeps it on — exactly the kind of
            // distribution-vs-core parity bug the validation suite
            // catches.
        }
        DebiasingChoice::Disable => cfg.debiasing = None,
        DebiasingChoice::Override(t) => cfg.debiasing = Some(t),
    }
    if c.num_excluded_perturbers > 0 && !c.excluded_perturbers_naif.is_null() {
        let slice = unsafe {
            std::slice::from_raw_parts(c.excluded_perturbers_naif, c.num_excluded_perturbers)
        };
        let mut out: Vec<Origin> = Vec::with_capacity(slice.len());
        for &naif in slice {
            let origin = Origin::from_naif_id(naif)
                .ok_or_else(|| format!("unknown NAIF id in excluded_perturbers: {naif}"))?;
            out.push(origin);
        }
        cfg.excluded_perturbers = out;
    }

    // ── IOD ───────────────────────────────────────────────────────
    let iod = &c.iod;
    if iod.max_triplet_attempts > 0 {
        cfg.max_triplet_attempts = iod.max_triplet_attempts as usize;
    }
    if iod.max_triplet_span_days > 0.0 {
        cfg.max_triplet_span_days = iod.max_triplet_span_days;
    }
    if iod.opposition_gap_days < 0.0 {
        cfg.opposition_gap_days = None;
    } else if iod.opposition_gap_days > 0.0 {
        cfg.opposition_gap_days = Some(iod.opposition_gap_days);
    }
    if iod.max_iod_arc_days > 0.0 {
        cfg.max_iod_arc_days = iod.max_iod_arc_days;
    }
    if iod.curvature_snr_threshold > 0.0 {
        cfg.curvature_snr_threshold = iod.curvature_snr_threshold;
    }
    if iod.max_iod_fractional_sigma_a > 0.0 {
        cfg.max_iod_fractional_sigma_a = iod.max_iod_fractional_sigma_a;
    }

    // ── Origin policy ─────────────────────────────────────────────
    cfg.origin = match c.origin.policy {
        EMPYREAN_ORIGIN_POLICY_AUTO => OriginPolicy::Auto,
        EMPYREAN_ORIGIN_POLICY_EXPLICIT => {
            let origin = Origin::from_naif_id(c.origin.explicit_naif).ok_or_else(|| {
                format!(
                    "unknown NAIF body id for origin.explicit_naif: {}",
                    c.origin.explicit_naif
                )
            })?;
            OriginPolicy::Explicit(origin)
        }
        other => return Err(format!("unknown origin.policy: {other}")),
    };

    // ── DC ────────────────────────────────────────────────────────
    cfg.output_epoch = match c.output_epoch.mode {
        EMPYREAN_OUTPUT_EPOCH_MID_ARC => OutputEpoch::MidArc,
        EMPYREAN_OUTPUT_EPOCH_LAST_OBSERVATION => OutputEpoch::LastObservation,
        EMPYREAN_OUTPUT_EPOCH_IOD_EPOCH => OutputEpoch::IODEpoch,
        EMPYREAN_OUTPUT_EPOCH_EXPLICIT => OutputEpoch::Epoch(c.output_epoch.explicit_mjd_tdb),
        other => return Err(format!("unknown output_epoch.mode: {other}")),
    };
    if c.max_iterations > 0 {
        cfg.max_iterations = c.max_iterations as usize;
    }
    if c.convergence_tol > 0.0 {
        cfg.convergence_tol = c.convergence_tol;
    }
    cfg.use_stm_cache = c.use_stm_cache != 0;
    cfg.solve_for = if c.solve_for == EMPYREAN_SOLVE_FOR_EXPLICIT {
        // Explicit multi-axis request — the coarse code can't name it, so
        // read the per-axis flag struct.
        let f = &c.solve_for_flags;
        SolveForParams::Explicit(SolveFor {
            marsden: f.marsden != 0,
            dt: f.dt != 0,
            amrat: f.amrat != 0,
            thrust_segments: f.thrust_segments as usize,
        })
    } else {
        int_to_solve_for(c.solve_for)?
    };
    cfg.allow_unbracketed_maneuvers = c.allow_unbracketed_maneuvers != 0;
    cfg.photometry = if c.has_photometry != 0 {
        Some(photometry_config_from_c(&c.photometry)?)
    } else {
        None
    };

    // ── Auto-escalation ───────────────────────────────────────────
    let ae = &c.auto_escalation;
    if ae.reduced_chi2 > 0.0 {
        cfg.auto_escalation.reduced_chi2 = ae.reduced_chi2;
    }
    if ae.at_ct_ratio > 0.0 {
        cfg.auto_escalation.at_ct_ratio = ae.at_ct_ratio;
    }
    if ae.min_arc_days > 0.0 {
        cfg.auto_escalation.min_arc_days = ae.min_arc_days;
    }
    if ae.min_n_obs > 0 {
        cfg.auto_escalation.min_n_obs = ae.min_n_obs as usize;
    }

    // ── Acceptability ─────────────────────────────────────────────
    let ac = &c.acceptability;
    if ac.reduced_chi2 > 0.0 {
        cfg.acceptability.reduced_chi2 = ac.reduced_chi2;
    }
    if ac.rms_arcsec > 0.0 {
        cfg.acceptability.rms_arcsec = ac.rms_arcsec;
    }
    if ac.at_ct_ratio > 0.0 {
        cfg.acceptability.at_ct_ratio = ac.at_ct_ratio;
    }
    if ac.min_arc_days > 0.0 {
        cfg.acceptability.min_arc_days = ac.min_arc_days;
    }
    if ac.fractional_sigma_a > 0.0 {
        cfg.acceptability.fractional_sigma_a = ac.fractional_sigma_a;
    }

    if c.fit_station_biases != 0 {
        let sigma = if c.station_radec.sigma_prior_arcsec > 0.0 {
            c.station_radec.sigma_prior_arcsec
        } else {
            0.3
        };
        let min_obs = if c.station_radec.min_obs_per_station > 0 {
            c.station_radec.min_obs_per_station
        } else {
            5
        };
        cfg.nuisance.push(BiasKind::StationRaDec {
            sigma_prior_arcsec: sigma,
            per_station_sigma_arcsec: std::collections::HashMap::new(),
            scope: BiasScope::AllStations,
            min_obs_per_station: min_obs,
        });
    }
    cfg.use_span_grouping = c.use_span_grouping != 0;

    // ── Rejection ─────────────────────────────────────────────────
    let rej = &c.rejection;
    cfg.rejection = build_rejection_strategy_from_c(rej)?;
    if rej.max_passes > 0 {
        cfg.max_rejection_passes = rej.max_passes as usize;
    }
    cfg.auto_force_model = c.auto_force_model != 0;
    cfg.output_representation = int_to_coord_rep(c.output_representation)?;

    Ok(cfg)
}

/// Return the ADES object identifier of an observation, or "unknown"
/// when none is set. Used as the HashMap key for batch determine /
/// evaluate / refine calls.
fn ades_object_id(obs: &ADESObservations) -> String {
    obs.object_id()
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Build a single-row `Orbits<AU>` from an EmpyreanOrbit (orbit_id required).
pub(crate) fn empyrean_orbit_to_orbits(
    orbit: &EmpyreanOrbit,
    id: &str,
) -> Result<Orbits<AU>, String> {
    let state = orbit.state.to_empyrean();
    let coords =
        coordinate_state_to_coordinates(&state).map_err(|e| format!("orbit conversion: {e}"))?;
    let mut out = Orbits::empty();
    out.push(id.to_string(), coords.into_radians())
        .map_err(|e| format!("orbit push: {e}"))?;
    // Carry the caller's non-grav model onto the orbit. Without this the OD
    // entry points (evaluate / refine / seeded determine) would fit a
    // gravity-only orbit and silently discard A1/A2/A3 + g(r) + dt.
    if let Some(params) = crate::propagate::empyrean_orbit_non_grav_params(orbit) {
        out.set_non_grav_params(0, Some(params));
    }
    // Carry the caller's continuous-thrust model onto the orbit so the
    // radar/optical planning (evaluate_plan) and OD (evaluate / refine)
    // single-orbit paths never silently discard thrust arcs + corrections.
    if let Some(tp) = crate::propagate::empyrean_orbit_thrust_params(orbit)? {
        out.set_thrust_params(0, Some(tp));
    }
    // Carry the caller's SRP slot onto the orbit so refine / evaluate never
    // silently drop the AMRAT prior. `srp_amrat_variance` (finite, > 0) is the
    // trigger + Bayesian prior that opens the AMRAT column in a StateAndAMRAT /
    // StateAndNonGravAndAMRAT fit; without this the AMRAT solve errors loudly
    // upstream (SRPParamsMissing / AMRATPriorMissing) rather than fitting a
    // gravity-only orbit.
    if let Some(srp) = crate::propagate::empyrean_orbit_srp_params(orbit)? {
        out.set_srp_params(0, Some(srp));
    }
    Ok(out)
}

// ── empyrean_read_ades ──────────────────────────────────────

/// Read ADES PSV / MPC80 data from a string and pack into the C array.
///
/// `path_or_content` is a null-terminated UTF-8 string with the ADES
/// content directly (not a file path).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_read_ades(
    content: *const c_char,
    observations_out: *mut *mut EmpyreanObservation,
    num_observations_out: *mut usize,
    radar_out: *mut *mut EmpyreanRadarObservation,
    num_radar_out: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if content.is_null()
            || observations_out.is_null()
            || num_observations_out.is_null()
            || radar_out.is_null()
            || num_radar_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }
        let input_str = match unsafe { CStr::from_ptr(content) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&format!("invalid UTF-8: {e}"));
                return -1;
            }
        };
        let observations = match parse_ades(input_str) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&format!("ADES parse error: {e}"));
                return -2;
            }
        };

        fn opt_cstr(s: Option<&String>) -> *mut c_char {
            match s {
                Some(v) if !v.is_empty() => CString::new(v.as_str())
                    .unwrap_or_else(|_| CString::new("").unwrap())
                    .into_raw(),
                _ => std::ptr::null_mut(),
            }
        }

        // ── Optical table ──
        let n = observations.optical.len();
        if n == 0 {
            unsafe {
                *observations_out = std::ptr::null_mut();
                *num_observations_out = 0;
            }
        } else {
            let layout = std::alloc::Layout::array::<EmpyreanObservation>(n)
                .unwrap_or(std::alloc::Layout::new::<EmpyreanObservation>());
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservation;
            if ptr.is_null() {
                set_last_error("allocation failed for observations array");
                return -5;
            }

            for (i, obs) in observations.optical.iter().enumerate() {
                let mut obs_code = [0u8; 4];
                let stn_bytes = obs.stn.as_bytes();
                for (j, b) in stn_bytes.iter().take(3).enumerate() {
                    obs_code[j] = *b;
                }
                obs_code[3] = 0;

                let entry = EmpyreanObservation {
                    perm_id: opt_cstr(obs.perm_id.as_ref()),
                    prov_id: opt_cstr(obs.prov_id.as_ref()),
                    trk_sub: opt_cstr(obs.trk_sub.as_ref()),
                    obs_id: opt_cstr(obs.obs_id.as_ref()),
                    obs_sub_id: opt_cstr(obs.obs_sub_id.as_ref()),
                    trk_id: opt_cstr(obs.trk_id.as_ref()),
                    obs_code,
                    mode: opt_cstr(obs.mode.as_ref()),
                    prog: opt_cstr(obs.prog.as_ref()),
                    sys: opt_cstr(obs.sys.as_ref()),
                    ctr: obs.ctr.unwrap_or(f64::NAN),
                    pos1: obs.pos1.unwrap_or(f64::NAN),
                    pos2: obs.pos2.unwrap_or(f64::NAN),
                    pos3: obs.pos3.unwrap_or(f64::NAN),
                    obs_time: CString::new(obs.obs_time.as_str())
                        .unwrap_or_else(|_| CString::new("").unwrap())
                        .into_raw(),
                    ra_deg: obs.ra,
                    dec_deg: obs.dec,
                    rms_ra_arcsec: obs.rms_ra.unwrap_or(f64::NAN),
                    rms_dec_arcsec: obs.rms_dec.unwrap_or(f64::NAN),
                    rms_corr: obs.rms_corr.unwrap_or(f64::NAN),
                    ast_cat: opt_cstr(obs.ast_cat.as_ref()),
                    mag: obs.mag.unwrap_or(f64::NAN),
                    rms_mag: obs.rms_mag.unwrap_or(f64::NAN),
                    band: opt_cstr(obs.band.as_ref()),
                    phot_cat: opt_cstr(obs.phot_cat.as_ref()),
                    phot_ap: obs.phot_ap.unwrap_or(f64::NAN),
                    log_snr: obs.log_snr.unwrap_or(f64::NAN),
                    seeing: obs.seeing.unwrap_or(f64::NAN),
                    exp: obs.exp.unwrap_or(f64::NAN),
                    rms_fit: obs.rms_fit.unwrap_or(f64::NAN),
                    n_stars: obs.n_stars.map(|v| v as i32).unwrap_or(-1),
                    notes: opt_cstr(obs.notes.as_ref()),
                    remarks: opt_cstr(obs.remarks.as_ref()),
                };
                unsafe {
                    ptr.add(i).write(entry);
                }
            }

            unsafe {
                *observations_out = ptr;
                *num_observations_out = n;
            }
        }

        // ── Radar table ──
        //
        // Pack each `RadarObservation` ADES-native via the shared
        // `scott_radar_to_c` marshaler — see its doc comment for the
        // unit / tri-state contract.
        let nr = observations.radar.len();
        if nr == 0 {
            unsafe {
                *radar_out = std::ptr::null_mut();
                *num_radar_out = 0;
            }
        } else {
            let layout = std::alloc::Layout::array::<EmpyreanRadarObservation>(nr)
                .unwrap_or(std::alloc::Layout::new::<EmpyreanRadarObservation>());
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanRadarObservation;
            if ptr.is_null() {
                set_last_error("allocation failed for radar observations array");
                return -5;
            }
            for (i, r) in observations.radar.iter().enumerate() {
                unsafe {
                    ptr.add(i).write(scott_radar_to_c(r));
                }
            }
            unsafe {
                *radar_out = ptr;
                *num_radar_out = nr;
            }
        }

        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_read_ades");
            -99
        }
    }
}

/// Free an observation array previously returned by `empyrean_read_ades()`.
/// Copy a caller-owned array of [`EmpyreanObservation`] into a fresh
/// allocation that matches the layout produced by
/// [`empyrean_read_ades`].
///
/// The strings on the input observations (`perm_id` / `prov_id` /
/// `obs_time`) are duplicated into freshly-allocated `CString`s so the
/// returned array owns its own memory independent of the input.
///
/// On success populates `*out_ptr` with the new array and `*out_num`
/// with its length, both freeable with [`empyrean_observations_free`].
///
/// Returns 0 on success; negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_observations_from_array(
    input: *const EmpyreanObservation,
    num: usize,
    out_ptr: *mut *mut EmpyreanObservation,
    out_num: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if out_ptr.is_null() || out_num.is_null() {
            set_last_error("null output pointer");
            return -1;
        }
        unsafe {
            *out_ptr = std::ptr::null_mut();
            *out_num = 0;
        }
        if num == 0 {
            return 0;
        }
        if input.is_null() {
            set_last_error("null input pointer with num > 0");
            return -1;
        }
        let layout = std::alloc::Layout::array::<EmpyreanObservation>(num)
            .unwrap_or(std::alloc::Layout::new::<EmpyreanObservation>());
        let dst = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservation;
        if dst.is_null() {
            set_last_error("allocation failed for observations array");
            return -5;
        }

        let dup_cstr = |p: *mut c_char| -> *mut c_char {
            if p.is_null() {
                std::ptr::null_mut()
            } else {
                let s = unsafe { CStr::from_ptr(p) };
                CString::new(s.to_bytes())
                    .unwrap_or_else(|_| CString::new("").unwrap())
                    .into_raw()
            }
        };

        for i in 0..num {
            let src = unsafe { &*input.add(i) };
            let entry = EmpyreanObservation {
                perm_id: dup_cstr(src.perm_id),
                prov_id: dup_cstr(src.prov_id),
                trk_sub: dup_cstr(src.trk_sub),
                obs_id: dup_cstr(src.obs_id),
                obs_sub_id: dup_cstr(src.obs_sub_id),
                trk_id: dup_cstr(src.trk_id),
                obs_code: src.obs_code,
                mode: dup_cstr(src.mode),
                prog: dup_cstr(src.prog),
                sys: dup_cstr(src.sys),
                ctr: src.ctr,
                pos1: src.pos1,
                pos2: src.pos2,
                pos3: src.pos3,
                obs_time: dup_cstr(src.obs_time),
                ra_deg: src.ra_deg,
                dec_deg: src.dec_deg,
                rms_ra_arcsec: src.rms_ra_arcsec,
                rms_dec_arcsec: src.rms_dec_arcsec,
                rms_corr: src.rms_corr,
                ast_cat: dup_cstr(src.ast_cat),
                mag: src.mag,
                rms_mag: src.rms_mag,
                band: dup_cstr(src.band),
                phot_cat: dup_cstr(src.phot_cat),
                phot_ap: src.phot_ap,
                log_snr: src.log_snr,
                seeing: src.seeing,
                exp: src.exp,
                rms_fit: src.rms_fit,
                n_stars: src.n_stars,
                notes: dup_cstr(src.notes),
                remarks: dup_cstr(src.remarks),
            };
            unsafe { dst.add(i).write(entry) };
        }
        unsafe {
            *out_ptr = dst;
            *out_num = num;
        }
        0
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_observations_from_array");
            -99
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_observations_free(
    observations: *mut EmpyreanObservation,
    num: usize,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if observations.is_null() || num == 0 {
            return;
        }
        for i in 0..num {
            let obs = unsafe { &*observations.add(i) };
            for ptr in [
                obs.perm_id,
                obs.prov_id,
                obs.trk_sub,
                obs.obs_id,
                obs.obs_sub_id,
                obs.trk_id,
                obs.mode,
                obs.prog,
                obs.sys,
                obs.obs_time,
                obs.ast_cat,
                obs.band,
                obs.phot_cat,
                obs.notes,
                obs.remarks,
            ] {
                if !ptr.is_null() {
                    drop(unsafe { CString::from_raw(ptr) });
                }
            }
        }
        let layout = std::alloc::Layout::array::<EmpyreanObservation>(num).unwrap();
        unsafe {
            std::alloc::dealloc(observations as *mut u8, layout);
        }
    }));
}

// ── empyrean radar observation array (copy + free) ──────────

/// Copy a caller-owned array of [`EmpyreanRadarObservation`] into a fresh
/// allocation matching the layout produced by [`empyrean_read_ades`].
///
/// The nullable `*mut c_char` fields (`perm_id` / `prov_id` / `trk_sub` /
/// `obs_time` / `remarks`) are duplicated into freshly-allocated
/// `CString`s so the returned array owns its own memory independent of the
/// input. All scalar fields (including the ADES-native delay/Doppler
/// values) are copied verbatim — no unit conversion, nothing zeroed.
///
/// On success populates `*out_ptr` with the new array and `*out_num` with
/// its length, both freeable with [`empyrean_radar_observations_free`].
///
/// Returns 0 on success; negative error code on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_radar_observations_from_array(
    input: *const EmpyreanRadarObservation,
    num: usize,
    out_ptr: *mut *mut EmpyreanRadarObservation,
    out_num: *mut usize,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if out_ptr.is_null() || out_num.is_null() {
            set_last_error("null output pointer");
            return -1;
        }
        unsafe {
            *out_ptr = std::ptr::null_mut();
            *out_num = 0;
        }
        if num == 0 {
            return 0;
        }
        if input.is_null() {
            set_last_error("null input pointer with num > 0");
            return -1;
        }
        let layout = std::alloc::Layout::array::<EmpyreanRadarObservation>(num)
            .unwrap_or(std::alloc::Layout::new::<EmpyreanRadarObservation>());
        let dst = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanRadarObservation;
        if dst.is_null() {
            set_last_error("allocation failed for radar observations array");
            return -5;
        }

        let dup_cstr = |p: *mut c_char| -> *mut c_char {
            if p.is_null() {
                std::ptr::null_mut()
            } else {
                let s = unsafe { CStr::from_ptr(p) };
                CString::new(s.to_bytes())
                    .unwrap_or_else(|_| CString::new("").unwrap())
                    .into_raw()
            }
        };

        for i in 0..num {
            let src = unsafe { &*input.add(i) };
            let entry = EmpyreanRadarObservation {
                perm_id: dup_cstr(src.perm_id),
                prov_id: dup_cstr(src.prov_id),
                trk_sub: dup_cstr(src.trk_sub),
                trx: src.trx,
                rcv: src.rcv,
                obs_time: dup_cstr(src.obs_time),
                kind: src.kind,
                delay_seconds: src.delay_seconds,
                rms_delay_microseconds: src.rms_delay_microseconds,
                doppler_hz: src.doppler_hz,
                rms_doppler_hz: src.rms_doppler_hz,
                frq_mhz: src.frq_mhz,
                com: src.com,
                log_snr: src.log_snr,
                remarks: dup_cstr(src.remarks),
            };
            unsafe { dst.add(i).write(entry) };
        }
        unsafe {
            *out_ptr = dst;
            *out_num = num;
        }
        0
    }));
    match result {
        Ok(c) => c,
        Err(_) => {
            set_last_error("panic in empyrean_radar_observations_from_array");
            -99
        }
    }
}

/// Free a radar observation array previously returned by
/// [`empyrean_read_ades`] or [`empyrean_radar_observations_from_array`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_radar_observations_free(
    observations: *mut EmpyreanRadarObservation,
    num: usize,
) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if observations.is_null() || num == 0 {
            return;
        }
        for i in 0..num {
            let obs = unsafe { &*observations.add(i) };
            for ptr in [
                obs.perm_id,
                obs.prov_id,
                obs.trk_sub,
                obs.obs_time,
                obs.remarks,
            ] {
                if !ptr.is_null() {
                    drop(unsafe { CString::from_raw(ptr) });
                }
            }
        }
        let layout = std::alloc::Layout::array::<EmpyreanRadarObservation>(num).unwrap();
        unsafe {
            std::alloc::dealloc(observations as *mut u8, layout);
        }
    }));
}

// ── empyrean_determine ──────────────────────────────────────

/// Run the full orbit determination pipeline.
///
/// When `num_initial_orbits > 0`, the supplied orbits are used as DC
/// seeds (one per ADES object_id encountered in `observations`,
/// matched by orbit index). Pass `null, 0` to let the IOD pipeline
/// produce its own seeds.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_determine(
    ctx: *const EmpyreanContext,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    radar: *const EmpyreanRadarObservation,
    num_radar: usize,
    initial_orbits: *const EmpyreanOrbit,
    num_initial_orbits: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || observations.is_null() || config.is_null() || result_out.is_null() {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let obs_slice = unsafe { std::slice::from_raw_parts(observations, num_observations) };

        let obs_vec = match c_observations_to_optical(obs_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        // Radar table (optional). Optical-only callers pass `null, 0`.
        let radar_vec = if num_radar > 0 {
            if radar.is_null() {
                set_last_error("radar pointer is null but num_radar > 0");
                return -1;
            }
            let radar_slice = unsafe { std::slice::from_raw_parts(radar, num_radar) };
            match c_radar_to_scott(radar_slice) {
                Ok(r) => r,
                Err(e) => {
                    set_last_error(&e);
                    return -1;
                }
            }
        } else {
            Vec::new()
        };

        let cfg = match build_od_config_from_c(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        // Build the optional initial-orbit map. The HashMap key is the
        // ADES `object_id` of the matching observation group; we map
        // each input orbit to the i-th unique object_id encountered.
        let initial_map: Option<HashMap<String, Orbits<AU>>> = if num_initial_orbits > 0 {
            if initial_orbits.is_null() {
                set_last_error("initial_orbits pointer is null but num_initial_orbits > 0");
                return -1;
            }
            let init_slice =
                unsafe { std::slice::from_raw_parts(initial_orbits, num_initial_orbits) };
            // Collect unique object_ids in the order they first appear.
            let mut keys: Vec<String> = Vec::new();
            for obs in &obs_vec {
                let key = ades_object_id(obs);
                if !keys.iter().any(|k| k == &key) {
                    keys.push(key);
                }
            }
            let mut map: HashMap<String, Orbits<AU>> = HashMap::new();
            for (i, init) in init_slice.iter().enumerate() {
                let key = match keys.get(i) {
                    Some(k) => k.clone(),
                    None => format!("orbit_{i}"),
                };
                let orb = match empyrean_orbit_to_orbits(init, &key) {
                    Ok(o) => o,
                    Err(e) => {
                        set_last_error(&format!("initial orbit {i}: {e}"));
                        return -1;
                    }
                };
                map.insert(key, orb);
            }
            Some(map)
        } else {
            None
        };

        let determine_results = determine(
            ctx_ref,
            Observations::new(obs_vec, radar_vec),
            initial_map.as_ref(),
            &cfg,
            None,
        );

        // Pick the first acceptable fit, else the first overall.
        let best = determine_results
            .results()
            .iter()
            .find(|r| match r {
                Ok(d) => d.od.acceptability.fit_acceptable,
                Err(_) => false,
            })
            .or_else(|| determine_results.results().first());

        let det_result = match best {
            Some(Ok(d)) => d,
            Some(Err(e)) => {
                set_last_error(&format!("orbit determination failed: {e}"));
                return -3;
            }
            None => {
                set_last_error("orbit determination produced no results");
                return -3;
            }
        };

        let prop_state =
            match od_orbit_to_propagated(&det_result.od.orbit, &det_result.od.covariance) {
                Ok(s) => s,
                Err(e) => {
                    set_last_error(&e);
                    return -3;
                }
            };

        let od = &det_result.od;
        let (obs_ptr, obs_n) = observation_results_to_c(&od.observations);
        let summary = summary_to_c(&od.summary);
        let acceptability = acceptability_to_c(&od.acceptability);

        let (has_cov_9x9, covariance_9x9) = match &od.covariance_9x9 {
            Some(m) => (1u8, *m),
            None => (0u8, [[0.0f64; 9]; 9]),
        };
        let (has_ng_delta, non_grav_delta) = match &od.non_grav_delta {
            Some(d) => (1u8, *d),
            None => (0u8, [f64::NAN; 3]),
        };
        let (has_non_grav, non_grav) = od_result_non_grav_to_c(od);

        let (sb_ptr, sb_n) = station_biases_to_c(&od.station_biases);

        unsafe {
            (*result_out).orbit = prop_state;
            (*result_out).observations = obs_ptr;
            (*result_out).num_observations = obs_n;
            (*result_out).summary = summary;
            (*result_out).iterations = od.iterations as u32;
            (*result_out).update_norm = od.update_norm;
            (*result_out).converged = u8::from(od.acceptability.converged_ok);
            (*result_out).covariance = od.covariance;
            (*result_out).covariance_representation =
                coord_rep_to_int(od.covariance_representation);
            (*result_out).has_covariance_9x9 = has_cov_9x9;
            (*result_out).covariance_9x9 = covariance_9x9;
            (*result_out).has_non_grav_delta = has_ng_delta;
            (*result_out).non_grav_delta = non_grav_delta;
            (*result_out).has_non_grav = has_non_grav;
            (*result_out).non_grav = non_grav;
            (*result_out).rejection_passes = od.rejection_passes as u32;
            (*result_out).num_oppositions_fit = od.num_oppositions_fit as u32;
            (*result_out).force_model_used = v_force_model_tier_to_int(od.force_model_used);
            (*result_out).solve_for_used = solve_for_to_int(&od.solve_for);
            (*result_out).acceptability = acceptability;
            (*result_out).station_biases = sb_ptr;
            (*result_out).num_station_biases = sb_n;
            populate_wide_fitting_fields(result_out, od);
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_determine");
            -99
        }
    }
}

/// Free an OD result previously returned by `empyrean_determine()` or `empyrean_refine()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_od_result_free(result: *mut EmpyreanODResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &*result };
        let n = res.num_observations;
        let sb_n = res.num_station_biases;
        unsafe {
            free_observation_results(res.observations, n);
            free_station_biases(res.station_biases, sb_n);
            // Photometry owned arrays (null / 0 when no photometry ran).
            free_band_stats(res.photometry.per_band, res.photometry.num_per_band);
            free_gate_records(res.photometry.gates, res.photometry.num_gates);
            (*result).observations = std::ptr::null_mut();
            (*result).num_observations = 0;
            (*result).station_biases = std::ptr::null_mut();
            (*result).num_station_biases = 0;
            (*result).photometry.per_band = std::ptr::null_mut();
            (*result).photometry.num_per_band = 0;
            (*result).photometry.gates = std::ptr::null_mut();
            (*result).photometry.num_gates = 0;
        }
    }));
}

// ── empyrean_evaluate ───────────────────────────────────────

/// Evaluate residuals for a single orbit against observations.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_evaluate(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanEvaluateResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbit.is_null()
            || observations.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let obs_slice = unsafe { std::slice::from_raw_parts(observations, num_observations) };
        let orbit_ref = unsafe { &*orbit };

        let obs_vec = match c_observations_to_optical(obs_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let orbits_single = match empyrean_orbit_to_orbits(orbit_ref, "orbit_0") {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let cfg = match build_od_config_from_c(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        // Single-orbit evaluate: residuals of this one orbit against ALL the
        // supplied observations, with no object-identifier keying.
        let eval_result = match evaluate_single(ctx_ref, &orbits_single, &obs_vec, &cfg) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&format!("evaluate failed: {e}"));
                return -3;
            }
        };

        let (obs_ptr, obs_n) = observation_results_to_c(&eval_result.observations);
        let summary = summary_to_c(&eval_result.summary);

        unsafe {
            (*result_out).observations = obs_ptr;
            (*result_out).num_observations = obs_n;
            (*result_out).summary = summary;
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_evaluate");
            -99
        }
    }
}

/// Free an evaluate result previously returned by `empyrean_evaluate()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_evaluate_result_free(result: *mut EmpyreanEvaluateResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        let res = unsafe { &*result };
        let n = res.num_observations;
        unsafe {
            free_observation_results(res.observations, n);
            (*result).observations = std::ptr::null_mut();
            (*result).num_observations = 0;
        }
    }));
}

// ── empyrean_refine ─────────────────────────────────────────

/// Refine a single orbit estimate with new observations using a
/// Bayesian prior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_refine(
    ctx: *const EmpyreanContext,
    orbit: *const EmpyreanOrbit,
    observations: *const EmpyreanObservation,
    num_observations: usize,
    config: *const EmpyreanODConfig,
    result_out: *mut EmpyreanODResult,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null()
            || orbit.is_null()
            || observations.is_null()
            || config.is_null()
            || result_out.is_null()
        {
            set_last_error("null pointer argument");
            return -1;
        }

        let ctx_ref = unsafe { &*ctx };
        let cfg_ref = unsafe { &*config };
        let obs_slice = unsafe { std::slice::from_raw_parts(observations, num_observations) };
        let orbit_ref = unsafe { &*orbit };

        let obs_vec = match c_observations_to_optical(obs_slice) {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let orbits_single = match empyrean_orbit_to_orbits(orbit_ref, "orbit_0") {
            Ok(o) => o,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        let cfg = match build_od_config_from_c(cfg_ref) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(&e);
                return -1;
            }
        };

        // Single-orbit refine: Bayesian update of this one orbit against ALL
        // the supplied observations, with no object-identifier keying.
        let od_result: ODResult = match refine_single(ctx_ref, &orbits_single, &obs_vec, &cfg) {
            Ok(r) => r,
            Err(e) => {
                set_last_error(&format!("refine failed: {e}"));
                return -3;
            }
        };

        let prop_state = match od_orbit_to_propagated(&od_result.orbit, &od_result.covariance) {
            Ok(s) => s,
            Err(e) => {
                set_last_error(&e);
                return -3;
            }
        };

        let (obs_ptr, obs_n) = observation_results_to_c(&od_result.observations);
        let summary = summary_to_c(&od_result.summary);
        let acceptability = acceptability_to_c(&od_result.acceptability);

        let (has_cov_9x9, covariance_9x9) = match &od_result.covariance_9x9 {
            Some(m) => (1u8, *m),
            None => (0u8, [[0.0f64; 9]; 9]),
        };
        let (has_ng_delta, non_grav_delta) = match &od_result.non_grav_delta {
            Some(d) => (1u8, *d),
            None => (0u8, [f64::NAN; 3]),
        };
        let (has_non_grav, non_grav) = od_result_non_grav_to_c(&od_result);

        let (sb_ptr, sb_n) = station_biases_to_c(&od_result.station_biases);

        unsafe {
            (*result_out).orbit = prop_state;
            (*result_out).observations = obs_ptr;
            (*result_out).num_observations = obs_n;
            (*result_out).summary = summary;
            (*result_out).iterations = od_result.iterations as u32;
            (*result_out).update_norm = od_result.update_norm;
            (*result_out).converged = u8::from(od_result.acceptability.converged_ok);
            (*result_out).covariance = od_result.covariance;
            (*result_out).covariance_representation =
                coord_rep_to_int(od_result.covariance_representation);
            (*result_out).has_covariance_9x9 = has_cov_9x9;
            (*result_out).covariance_9x9 = covariance_9x9;
            (*result_out).has_non_grav_delta = has_ng_delta;
            (*result_out).non_grav_delta = non_grav_delta;
            (*result_out).has_non_grav = has_non_grav;
            (*result_out).non_grav = non_grav;
            (*result_out).rejection_passes = od_result.rejection_passes as u32;
            (*result_out).num_oppositions_fit = od_result.num_oppositions_fit as u32;
            (*result_out).force_model_used = v_force_model_tier_to_int(od_result.force_model_used);
            (*result_out).solve_for_used = solve_for_to_int(&od_result.solve_for);
            (*result_out).acceptability = acceptability;
            (*result_out).station_biases = sb_ptr;
            (*result_out).num_station_biases = sb_n;
            populate_wide_fitting_fields(result_out, &od_result);
        }
        0
    }));

    match result {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in empyrean_refine");
            -99
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use empyrean_core::determination::RejectionStrategy;

    fn rejection_config(kind: u8) -> EmpyreanRejectionConfig {
        EmpyreanRejectionConfig {
            enabled: 1,
            kind,
            chi2_base: 0.0,
            lambda: -1.0,
            max_threshold: 0.0,
            chi2_rej: 0.0,
            chi2_rec: 0.0,
            max_passes: 0,
        }
    }

    #[test]
    fn rejection_disabled_yields_none() {
        let mut c = rejection_config(EMPYREAN_REJECTION_KIND_ADAPTIVE);
        c.enabled = 0;
        let s = build_rejection_strategy_from_c(&c).unwrap();
        assert!(s.is_none());
    }

    #[test]
    fn rejection_kind_adaptive_default_sentinels() {
        // enabled with all sentinels => upstream defaults.
        let c = rejection_config(EMPYREAN_REJECTION_KIND_ADAPTIVE);
        let s = build_rejection_strategy_from_c(&c).unwrap();
        match s.unwrap() {
            RejectionStrategy::Adaptive(a) => {
                let d = AdaptiveRejectionConfig::default();
                assert_eq!(a.chi2_base, d.chi2_base);
                assert_eq!(a.lambda, d.lambda);
                assert_eq!(a.max_threshold, d.max_threshold);
            }
            other => panic!("expected Adaptive, got {other:?}"),
        }
    }

    #[test]
    fn rejection_kind_adaptive_overrides() {
        let mut c = rejection_config(EMPYREAN_REJECTION_KIND_ADAPTIVE);
        c.chi2_base = 12.5;
        c.lambda = 2.0;
        c.max_threshold = 50.0;
        let s = build_rejection_strategy_from_c(&c).unwrap();
        match s.unwrap() {
            RejectionStrategy::Adaptive(a) => {
                assert_eq!(a.chi2_base, 12.5);
                assert_eq!(a.lambda, 2.0);
                assert_eq!(a.max_threshold, 50.0);
            }
            other => panic!("expected Adaptive, got {other:?}"),
        }
    }

    #[test]
    fn rejection_kind_cmc2003_default_sentinels() {
        let c = rejection_config(EMPYREAN_REJECTION_KIND_CMC2003);
        let s = build_rejection_strategy_from_c(&c).unwrap();
        match s.unwrap() {
            RejectionStrategy::CMC2003(r) => {
                let d = CMC2003Config::default();
                assert_eq!(r.chi2_rej, d.chi2_rej);
                assert_eq!(r.chi2_rec, d.chi2_rec);
            }
            other => panic!("expected CMC2003, got {other:?}"),
        }
    }

    #[test]
    fn rejection_kind_cmc2003_overrides() {
        let mut c = rejection_config(EMPYREAN_REJECTION_KIND_CMC2003);
        c.chi2_rej = 9.0;
        c.chi2_rec = 6.5;
        let s = build_rejection_strategy_from_c(&c).unwrap();
        match s.unwrap() {
            RejectionStrategy::CMC2003(r) => {
                assert_eq!(r.chi2_rej, 9.0);
                assert_eq!(r.chi2_rec, 6.5);
            }
            other => panic!("expected CMC2003, got {other:?}"),
        }
    }

    #[test]
    fn rejection_kind_cmc2003_rejects_inverted_thresholds() {
        // CMC2003Config::validate requires chi2_rec < chi2_rej.
        let mut c = rejection_config(EMPYREAN_REJECTION_KIND_CMC2003);
        c.chi2_rej = 6.0;
        c.chi2_rec = 7.0;
        let err = build_rejection_strategy_from_c(&c).unwrap_err();
        assert!(err.contains("CMC2003 config:"), "got {err}");
        assert!(err.contains("hysteresis"), "got {err}");
    }

    #[test]
    fn rejection_unknown_kind_is_rejected() {
        let c = rejection_config(99);
        let err = build_rejection_strategy_from_c(&c).unwrap_err();
        assert!(err.contains("unsupported rejection.kind = 99"), "got {err}");
    }

    #[test]
    fn rejection_reason_cmc2003_maps_to_dedicated_code() {
        // Regression: CMC2003 was previously folded into ADAPTIVE.
        assert_eq!(
            rejection_reason_to_c(RejectionReason::CMC2003),
            EMPYREAN_REJECTION_CMC2003
        );
        assert_ne!(EMPYREAN_REJECTION_CMC2003, EMPYREAN_REJECTION_ADAPTIVE);
    }

    /// Round-trips a radar array through the C-ABI deep-copy + free path to
    /// guard the query_radar marshaling: every field must survive the copy
    /// (incl. NaN on the inactive measurement pair and the `com` tri-state,
    /// per the no-silent-fallback contract), and the free must reclaim every
    /// allocated string exactly once. A future drift — e.g. adding a sixth
    /// allocated string to the marshaler without updating the free loop —
    /// would leak/double-free here.
    #[test]
    fn radar_observations_round_trip_preserves_fields_and_frees() {
        use std::ffi::{CStr, CString};

        let mk = |s: &str| CString::new(s).unwrap().into_raw();
        let cstr = |p: *mut c_char| -> Option<String> {
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        };

        // A delay row (com=true, full strings) + a Doppler row (com absent,
        // sparse strings) — both measurement arms, both com edge values.
        let mut input = vec![
            EmpyreanRadarObservation {
                perm_id: mk("99942"),
                prov_id: std::ptr::null_mut(),
                trk_sub: std::ptr::null_mut(),
                trx: *b"253\0",
                rcv: *b"253\0",
                obs_time: mk("2021-03-11T08:20:00Z"),
                kind: EMPYREAN_RADAR_KIND_DELAY,
                delay_seconds: 120.5,
                rms_delay_microseconds: 0.25,
                doppler_hz: f64::NAN,
                rms_doppler_hz: f64::NAN,
                frq_mhz: 8560.0,
                com: 1,
                log_snr: 2.5,
                remarks: mk("note"),
            },
            EmpyreanRadarObservation {
                perm_id: std::ptr::null_mut(),
                prov_id: mk("2004 MN4"),
                trk_sub: std::ptr::null_mut(),
                trx: *b"253\0",
                rcv: *b"257\0",
                obs_time: mk("2021-03-08T02:50:00Z"),
                kind: EMPYREAN_RADAR_KIND_DOPPLER,
                delay_seconds: f64::NAN,
                rms_delay_microseconds: f64::NAN,
                doppler_hz: -5000.0,
                rms_doppler_hz: 0.2,
                frq_mhz: 2380.0,
                com: -1,
                log_snr: f64::NAN,
                remarks: std::ptr::null_mut(),
            },
        ];

        let mut out_ptr: *mut EmpyreanRadarObservation = std::ptr::null_mut();
        let mut out_num: usize = 0;
        let code = unsafe {
            empyrean_radar_observations_from_array(
                input.as_ptr(),
                input.len(),
                &mut out_ptr,
                &mut out_num,
            )
        };
        assert_eq!(code, 0);
        assert_eq!(out_num, 2);
        assert!(!out_ptr.is_null());

        let d = unsafe { &*out_ptr };
        assert_eq!(cstr(d.perm_id).as_deref(), Some("99942"));
        assert_eq!(cstr(d.prov_id), None);
        assert_eq!(&d.trx, b"253\0");
        assert_eq!(d.kind, EMPYREAN_RADAR_KIND_DELAY);
        assert_eq!(d.delay_seconds, 120.5);
        assert_eq!(d.rms_delay_microseconds, 0.25);
        assert!(d.doppler_hz.is_nan() && d.rms_doppler_hz.is_nan());
        assert_eq!(d.com, 1);
        assert_eq!(d.frq_mhz, 8560.0);
        assert_eq!(cstr(d.remarks).as_deref(), Some("note"));

        let dop = unsafe { &*out_ptr.add(1) };
        assert_eq!(dop.kind, EMPYREAN_RADAR_KIND_DOPPLER);
        assert!(dop.delay_seconds.is_nan() && dop.rms_delay_microseconds.is_nan());
        assert_eq!(dop.doppler_hz, -5000.0);
        assert_eq!(dop.com, -1); // absent stays -1, never silently 0
        assert!(dop.log_snr.is_nan());
        assert_eq!(cstr(dop.perm_id), None);
        assert_eq!(cstr(dop.prov_id).as_deref(), Some("2004 MN4"));

        // Free the deep copy via the C ABI (balances every dup'd string).
        unsafe { empyrean_radar_observations_free(out_ptr, out_num) };

        // Reclaim the hand-built input strings so the test itself is clean.
        for obs in input.drain(..) {
            for p in [
                obs.perm_id,
                obs.prov_id,
                obs.trk_sub,
                obs.obs_time,
                obs.remarks,
            ] {
                if !p.is_null() {
                    drop(unsafe { CString::from_raw(p) });
                }
            }
        }
    }

    // ── OD output-redesign acceptance tests (locks c37m + 833t at the ABI) ──
    //
    // The determine / evaluate / refine OUTPUT redesign (plug-and-play OD
    // outputs) shipped without acceptance tests. These two tests close that
    // gap at the C-ABI chokepoint — the single point every distribution
    // channel (Rust wrapper, Python, CLI) funnels through.
    //
    //   1. CONVERTER NON-GRAV CARRY (c37m): the input-side converter
    //      `empyrean_orbit_to_orbits` must carry the caller's ABSOLUTE
    //      non-grav (A1/A2/A3 + g(r) model) onto the `Orbits<AU>` it builds,
    //      so a fitted orbit re-fed through the ABI keeps fitting WITH its
    //      non-gravitational acceleration instead of silently reverting to a
    //      gravity-only orbit. A negative control proves a gravity-only orbit
    //      stays gravity-only (no fabricated non-grav).
    //
    //   2. FFI NO-KEYING SMOKE (833t): `empyrean_evaluate` / `empyrean_refine`
    //      evaluate the single supplied orbit against EVERY supplied
    //      observation with NO object-identifier ⇄ orbit-tag keying. The C ABI
    //      internally tags the orbit `"orbit_0"`; the observations carry a
    //      different designation (Eros = "433"). The mismatch must NOT collapse
    //      to a NoValidObservations failure — both calls must return code 0
    //      with `num_observations > 0`.

    /// Helper: a Cartesian `EmpyreanOrbit` carrying a known A2 (Yarkovsky,
    /// inverse-square g(r)) and otherwise-zero non-grav. The state itself is a
    /// throwaway heliocentric placeholder — the converter only touches the
    /// non-grav fields, which is all this test exercises.
    fn orbit_with_a2(a2: f64) -> EmpyreanOrbit {
        EmpyreanOrbit {
            state: crate::CoordinateState {
                epoch_mjd_tdb: 59000.0,
                // A plausible heliocentric Cartesian state (AU, AU/day).
                // Only origin=Sun(10) + representation=Cartesian(0) matter for
                // the converter; the numbers never reach an integrator here.
                elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
                covariance: [[0.0; 6]; 6],
                has_covariance: 0,
                representation: EMPYREAN_REPRESENTATION_CARTESIAN, // Cartesian
                frame: 0,                                          // ICRF
                origin: 10,                                        // Sun (NAIF)
            },
            orbit_id: std::ptr::null(),
            object_id: std::ptr::null(),
            a1: 0.0,
            a2,
            a3: 0.0,
            // All-zero g(r) fields ⇒ inverse-square model (Yarkovsky / SRP).
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            // NaN ⇒ no thermal-lag delay (asteroid default).
            non_grav_dt: f64::NAN,
            // NaN ⇒ no DT prior (DT column stays closed).
            non_grav_dt_variance: f64::NAN,
            has_non_grav_covariance: 0,
            non_grav_covariance: [[0.0; 3]; 3],
            phot_system: 0,
            h_mag: f64::NAN,
            slope1: f64::NAN,
            slope2: f64::NAN,
            // No continuous thrust (gravity + non-grav only).
            thrust_arcs: std::ptr::null(),
            n_thrust_arcs: 0,
            dv_corrections: std::ptr::null(),
            n_dv_corrections: 0,
            correction_covariances: std::ptr::null(),
            n_correction_covariances: 0,
            has_srp: 0,
            srp_amrat: 0.0,
            srp_cr: 0.0,
            srp_amrat_variance: f64::NAN,
        }
    }

    /// c37m: the input converter carries an ABSOLUTE A2 (with the
    /// inverse-square g(r) model) onto the `Orbits<AU>` it builds, so the OD
    /// entry points fit WITH the Yarkovsky acceleration instead of silently
    /// discarding it. Without the `set_non_grav_params` call in
    /// `empyrean_orbit_to_orbits`, `non_grav_params(0)` would be `None` and
    /// this assertion would fail — which is exactly the silent-fallback
    /// regression this test pins.
    #[test]
    fn converter_carries_absolute_a2_with_inverse_square_model() {
        let a2 = -2.9e-14; // Apophis-scale transverse Yarkovsky (AU/day²).
        let orbit = orbit_with_a2(a2);

        let orbits = empyrean_orbit_to_orbits(&orbit, "test")
            .expect("converter must build a single-row Orbits<AU>");

        let ng = orbits
            .non_grav_params(0)
            .expect("non-grav must survive the converter (c37m: no silent drop)");

        // The ABSOLUTE A-coefficients carry through verbatim.
        assert_eq!(ng.a1, 0.0, "A1 must stay zero");
        assert_eq!(ng.a2, a2, "A2 must carry through the converter unchanged");
        assert_eq!(ng.a3, 0.0, "A3 must stay zero");

        // The model must be Marsden-Sekanina with the inverse-square g(r)
        // (α=1, r0=1, m=2, n=0, k=0) selected by the all-zero g-fields.
        // NonGravModel is Marsden-only in v1.20.0 — irrefutable binding.
        let NonGravModel::MarsdenSekanina(g) = &ng.model;
        assert_eq!(g.alpha, 1.0, "inverse-square α");
        assert_eq!(g.r0, 1.0, "inverse-square r0");
        assert_eq!(g.m, 2.0, "inverse-square m");
        assert_eq!(g.n, 0.0, "inverse-square n");
        assert_eq!(g.k, 0.0, "inverse-square k");

        // No thermal-lag delay was requested (NaN input).
        assert!(ng.dt.is_none(), "non_grav_dt=NaN must map to dt=None");
    }

    /// c37m negative control: an all-zero-A orbit stays gravity-only. The
    /// converter must NOT fabricate a non-grav model out of thin air — that
    /// would silently turn every gravity-only re-feed into a (spurious)
    /// Yarkovsky fit.
    #[test]
    fn converter_leaves_gravity_only_orbit_without_non_grav() {
        let orbit = orbit_with_a2(0.0); // a1 = a2 = a3 = 0
        let orbits = empyrean_orbit_to_orbits(&orbit, "test")
            .expect("converter must build a single-row Orbits<AU>");
        assert!(
            orbits.non_grav_params(0).is_none(),
            "all-zero-A orbit must carry NO non-grav model (no fabrication)"
        );
    }

    // ── 833t FFI no-keying smoke (needs ephemeris; gated on data dir) ──

    /// Build a full Standard-tier context from the local data dir, or `None`
    /// when the ephemeris is unavailable (so CI without kernels skips the
    /// heavy smoke instead of failing). Resolves `EMPYREAN_DATA_DIR` / XDG
    /// exactly like the production constructor.
    fn try_context() -> Option<EmpyreanContext> {
        empyrean_core::Context::from_data_dir(None).ok()
    }

    /// Parse the bundled Eros ADES fixture into a freshly-allocated C
    /// `EmpyreanObservation` array via the real `empyrean_read_ades` ABI entry
    /// point (the same path a C caller uses). Returns the pointer + count;
    /// the caller frees with `empyrean_observations_free`.
    fn read_eros_observations() -> (*mut EmpyreanObservation, usize) {
        let psv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/433_eros.psv");
        let content = std::fs::read_to_string(psv).expect("read bundled Eros fixture");
        let c_content = CString::new(content).expect("fixture has no interior NUL");

        let mut obs_ptr: *mut EmpyreanObservation = std::ptr::null_mut();
        let mut obs_n: usize = 0;
        let mut radar_ptr: *mut EmpyreanRadarObservation = std::ptr::null_mut();
        let mut radar_n: usize = 0;
        let code = unsafe {
            empyrean_read_ades(
                c_content.as_ptr(),
                &mut obs_ptr,
                &mut obs_n,
                &mut radar_ptr,
                &mut radar_n,
            )
        };
        assert_eq!(code, 0, "empyrean_read_ades must parse the Eros fixture");
        assert!(
            obs_n > 0 && !obs_ptr.is_null(),
            "fixture must yield optical obs"
        );
        // The Eros optical-only fixture carries no radar rows.
        assert_eq!(radar_n, 0);
        (obs_ptr, obs_n)
    }

    /// A zero-initialized `EmpyreanODConfig` maps to upstream defaults; we
    /// only override the force-model tier to Standard so the fit is realistic
    /// (Approximate is too coarse for an OD smoke). `std::mem::zeroed` is sound
    /// here: every field is `#[repr(C)]` POD and the lone pointer
    /// (`excluded_perturbers_naif`) zero-inits to null with count 0.
    fn standard_od_config() -> EmpyreanODConfig {
        let mut cfg: EmpyreanODConfig = unsafe { std::mem::zeroed() };
        cfg.force_model = 2; // Standard tier
        cfg
    }

    /// Reconstruct a re-feedable `EmpyreanOrbit` from a fitted
    /// `EmpyreanPropagatedState` (the determine/refine output orbit). Mirrors
    /// the output→input re-feed a real caller performs: flatten the propagated
    /// Cartesian state + 6×6 covariance back into an input orbit.
    fn refeed_orbit(p: &EmpyreanPropagatedState) -> EmpyreanOrbit {
        EmpyreanOrbit {
            state: crate::CoordinateState {
                epoch_mjd_tdb: p.epoch_mjd_tdb,
                elements: [p.x, p.y, p.z, p.vx, p.vy, p.vz],
                covariance: p.covariance,
                has_covariance: p.has_covariance,
                representation: EMPYREAN_REPRESENTATION_CARTESIAN,
                frame: p.frame,
                origin: p.origin,
            },
            orbit_id: std::ptr::null(),
            object_id: std::ptr::null(),
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            non_grav_dt: f64::NAN,
            non_grav_dt_variance: f64::NAN,
            has_non_grav_covariance: 0,
            non_grav_covariance: [[0.0; 3]; 3],
            phot_system: 0,
            h_mag: f64::NAN,
            slope1: f64::NAN,
            slope2: f64::NAN,
            // Re-fed OD output orbits carry no thrust arcs.
            thrust_arcs: std::ptr::null(),
            n_thrust_arcs: 0,
            dv_corrections: std::ptr::null(),
            n_dv_corrections: 0,
            correction_covariances: std::ptr::null(),
            n_correction_covariances: 0,
            has_srp: 0,
            srp_amrat: 0.0,
            srp_cr: 0.0,
            srp_amrat_variance: f64::NAN,
        }
    }

    /// 833t at the ABI: a determine→evaluate→refine round-trip where the
    /// observations' designation ("433") never matches the orbit tag the C
    /// ABI assigns internally ("orbit_0"). `empyrean_evaluate` and
    /// `empyrean_refine` dispatch to the single-orbit path, which evaluates
    /// the orbit against ALL supplied observations with no keying — so the
    /// id mismatch must NOT collapse to NoValidObservations. Both calls must
    /// return code 0 with `num_observations > 0`.
    #[test]
    fn ffi_evaluate_refine_ignore_orbit_tag() {
        let ctx = match try_context() {
            Some(c) => c,
            None => {
                eprintln!("skipping ffi_evaluate_refine_ignore_orbit_tag: no ephemeris data dir");
                return;
            }
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        let (obs_ptr, obs_n) = read_eros_observations();
        let cfg = standard_od_config();

        // ── Fit Eros via the ABI to get a covariance-bearing orbit ──
        let mut od_result: EmpyreanODResult = unsafe { std::mem::zeroed() };
        let det_code = unsafe {
            empyrean_determine(
                ctx_ptr,
                obs_ptr,
                obs_n,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &cfg,
                &mut od_result,
            )
        };
        assert_eq!(
            det_code,
            0,
            "empyrean_determine must fit Eros (code {det_code}, last_error: {})",
            unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy()
        );
        assert!(
            od_result.num_observations > 0,
            "determine must report fitted observations"
        );
        assert_eq!(od_result.converged, 1, "Eros DC must converge");
        assert_eq!(
            od_result.orbit.has_covariance, 1,
            "fitted orbit must carry covariance for refine's prior"
        );

        // The re-feedable orbit, tagged INTERNALLY as "orbit_0" by the ABI —
        // which never matches the observations' designation ("433").
        let refed = refeed_orbit(&od_result.orbit);

        // ── evaluate: residuals of this orbit against ALL obs, no keying ──
        let mut eval_result: EmpyreanEvaluateResult = unsafe { std::mem::zeroed() };
        let eval_code =
            unsafe { empyrean_evaluate(ctx_ptr, &refed, obs_ptr, obs_n, &cfg, &mut eval_result) };
        assert_eq!(
            eval_code,
            0,
            "empyrean_evaluate must succeed despite the obs-id ⇄ orbit-tag \
             mismatch (833t); last_error: {}",
            unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy()
        );
        assert!(
            eval_result.num_observations > 0,
            "evaluate must report a NON-ZERO observation count, not collapse to \
             NoValidObservations (got {})",
            eval_result.num_observations
        );
        assert!(
            eval_result.summary.num_obs > 0 && eval_result.summary.num_selected > 0,
            "evaluate summary must show selected observations (num_obs={}, num_selected={})",
            eval_result.summary.num_obs,
            eval_result.summary.num_selected
        );
        assert!(
            eval_result.summary.rms_combined_arcsec.is_finite(),
            "evaluate combined RMS must be finite"
        );
        eprintln!(
            "ffi_evaluate (tag mismatch): num_obs={} num_selected={} rms_comb={:.3}\"",
            eval_result.summary.num_obs,
            eval_result.summary.num_selected,
            eval_result.summary.rms_combined_arcsec,
        );

        // ── refine: Bayesian update of this orbit against ALL obs, no keying ──
        let mut refine_result: EmpyreanODResult = unsafe { std::mem::zeroed() };
        let refine_code =
            unsafe { empyrean_refine(ctx_ptr, &refed, obs_ptr, obs_n, &cfg, &mut refine_result) };
        assert_eq!(
            refine_code,
            0,
            "empyrean_refine must succeed despite the obs-id ⇄ orbit-tag \
             mismatch (833t); last_error: {}",
            unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy()
        );
        assert!(
            refine_result.num_observations > 0,
            "refine must report a NON-ZERO observation count, not collapse to \
             NoValidObservations (got {})",
            refine_result.num_observations
        );
        assert!(
            refine_result.summary.num_obs > 0 && refine_result.summary.num_selected > 0,
            "refine summary must show selected observations (num_obs={}, num_selected={})",
            refine_result.summary.num_obs,
            refine_result.summary.num_selected
        );
        eprintln!(
            "ffi_refine (tag mismatch): num_obs={} num_selected={} converged={}",
            refine_result.summary.num_obs,
            refine_result.summary.num_selected,
            refine_result.converged,
        );

        // ── Free everything via the ABI free paths ──
        unsafe {
            empyrean_evaluate_result_free(&mut eval_result);
            empyrean_od_result_free(&mut refine_result);
            empyrean_od_result_free(&mut od_result);
            empyrean_observations_free(obs_ptr, obs_n);
        }
    }
}
