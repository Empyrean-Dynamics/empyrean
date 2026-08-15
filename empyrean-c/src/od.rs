use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;

use empyrean_core::ForceModelTier;
use empyrean_core::convert::{coordinate_state_to_coordinates, frame_to_int};
use empyrean_core::coordinates::{AU, CoordinateRepresentation, Coordinates, Origin};
use empyrean_core::determination::{
    AcceptabilityReport, AdaptiveRejectionConfig, BiasKind, BiasScope, CMC2003Config,
    CovarianceTrust, DetermineError, ODConfig, ODResult, ODWarning, ObservationResidualSummary,
    ObservationResult, Observations, OriginPolicy, OutputEpoch, ParamDisposition, RadarMeasurement,
    RadarObservation, RadarResidual, RejectionReason, SolveFor, SolveForParams, SolvedCovariance,
    TrustGateEvent, UpstreamForceModelTier, determine, evaluate_single, refine_single,
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

/// Covariance-trust verdict: the call path ran no trust gate — absence
/// of a verdict is NOT trust (e.g. `empyrean_refine`, session paths).
pub const EMPYREAN_COVARIANCE_TRUST_NOT_EVALUATED: i32 = 0;
/// Covariance-trust verdict: no intervening close approach and a
/// 6-state solve — the linear covariance may be used as delivered.
pub const EMPYREAN_COVARIANCE_TRUST_TRUSTED: i32 = 1;
/// Covariance-trust verdict: a close approach (or high-nonlinearity
/// crossing) lies inside the covariance validity window — do not
/// extrapolate the linear covariance across it.
pub const EMPYREAN_COVARIANCE_TRUST_ENCOUNTER_INTERVENES: i32 = 2;
/// Covariance-trust verdict: the fit solved more than the 6-state, so
/// the delivered 6×6 is a marginal of a wider fit (conservative flag).
pub const EMPYREAN_COVARIANCE_TRUST_WEAKLY_DETERMINED_HIGH_N: i32 = 3;

/// Intervening trust-gate event kind: none.
pub const EMPYREAN_TRUST_EVENT_NONE: i32 = 0;
/// Intervening trust-gate event kind: close approach to a body.
pub const EMPYREAN_TRUST_EVENT_CLOSE_APPROACH: i32 = 1;
/// Intervening trust-gate event kind: high-nonlinearity crossing.
pub const EMPYREAN_TRUST_EVENT_HIGH_NONLINEARITY: i32 = 2;

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
/// The observation's χ² against the published orbit is non-finite (NaN
/// or infinite residual / weight product), so it cannot participate in
/// any fit statistic. Every summary already excluded it from χ²/dof;
/// this code makes the exclusion visible instead of letting the row
/// read as used ("48 of 48 used" for a fit that used 42).
pub const EMPYREAN_REJECTION_NON_FINITE_CHI2: i32 = 10;
/// The propagation retained no Jacobian / STM at this observation's
/// epoch, so the observation contributed no row to the normal equations
/// — it was never part of the fit and its per-obs χ² is NaN.
pub const EMPYREAN_REJECTION_MISSING_JACOBIAN: i32 = 11;
/// The observation's observatory is a spacecraft whose SPK kernel is not
/// loaded, so its position could not be constructed. Distinct from
/// [`EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY`] (a code the engine does
/// not model at all): this one is a **data-provisioning** gap the caller
/// can close by loading the kernel, not a property of the observation.
pub const EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING: i32 = 12;
/// The observer position could not be constructed for a reason specific
/// to this row (bad roving-observer record, epoch outside the loaded
/// Earth-orientation coverage, …). The engine's per-row explanation is
/// not carried across the ABI as a string; consult the batch-level error
/// or the engine log for the detail.
pub const EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED: i32 = 13;
/// The observation was never absorbed into the fit: it reached no
/// iteration that could have used it. Distinct from a rejection — no
/// criterion was ever tested against it — and distinct from
/// [`EMPYREAN_REJECTION_NOT_EVALUATED`], which means the call path ran
/// no rejection pass at all.
pub const EMPYREAN_REJECTION_NEVER_ABSORBED: i32 = 14;
/// The observation names a **per-observation site** — the roving-observer
/// codes `247` / `270` and the occultation code `275` — whose position
/// travels with each observation, so the MPC publishes no planetodetic
/// constants for it and there is nothing to look up.
///
/// Distinct from [`EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY`]: the code
/// is perfectly well known, and what is missing is the observation's own
/// longitude / latitude / altitude, not a registry entry. Distinct from
/// [`EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING`] too: that is a
/// data-provisioning gap closed by loading a kernel, whereas this one is
/// closed by supplying the coordinates the ADES record already carries
/// per observation, which routes the row through the geodetic-observer
/// path and fits it.
pub const EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED: i32 = 15;
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
// curated layer sets; additional_layers go AHEAD of the preset's
// rules (their relative order preserved), so user rules win their
// stations under first-match-wins sigma resolution and the preset
// is the fallback. `preset = NONE` = build from scratch: only
// additional_layers contribute rules, and with an empty list the
// caller's `default_sigma_arcsec` applies uniformly.

/// No weighting preset — only `additional_layers` apply.
pub const EMPYREAN_WEIGHTING_PRESET_NONE: u8 = 0;
/// VFCC2017 — Vereš, Farnocchia, Chesley & Chamberlin 2017 station floors +
/// nightly de-weighting. The production default.
pub const EMPYREAN_WEIGHTING_PRESET_VFCC2017: u8 = 1;
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
/// Tagged-union shape: the active fields depend on `kind`, and the
/// inactive fields MUST be left at their unset values (zeroed bytes /
/// 0.0 / NaN epochs) — a layer carrying fields its kind does not read
/// is rejected with an error rather than silently ignored. In
/// particular a `NIGHTLY_DEWEIGHTING` layer reads only
/// `max_gap_days`: nightly de-weighting cannot be scoped by station
/// or time range.
#[repr(C)]
pub struct EmpyreanWeightingLayer {
    /// Layer kind discriminator — one of
    /// `EMPYREAN_WEIGHTING_LAYER_*`.
    pub kind: i32,
    // ── ObservatoryRule fields ─────────────────────────────────
    /// MPC observatory code: printable ASCII, no whitespace,
    /// left-aligned and NUL-padded to 4 bytes. Station matching is
    /// exact and case-sensitive; malformed codes are rejected with an
    /// error (never repaired or trimmed).
    pub obs_code: [u8; 4],
    /// 1σ RA·cos(δ) in arcsec.
    pub sigma_ra_arcsec: f64,
    /// 1σ Dec in arcsec.
    pub sigma_dec_arcsec: f64,
    /// Start of applicable time range (MJD TDB). NaN = unbounded.
    pub start_epoch_mjd_tdb: f64,
    /// End of applicable time range (MJD TDB). NaN = unbounded.
    pub end_epoch_mjd_tdb: f64,
    /// Scale factor on the resulting weight. Must be finite and > 0
    /// — use 1.0 for no scaling. Non-positive or non-finite values
    /// are rejected with an error (0.0 no longer silently maps to
    /// 1.0).
    pub scale: f64,
    // ── NightlyDeweighting fields ──────────────────────────────
    /// Maximum gap between observations to count as the same night
    /// (days). Must be finite and > 0 — the production value is 0.5.
    /// Non-positive or non-finite values are rejected with an error
    /// (0.0 no longer silently maps to 0.5).
    ///
    /// The de-weighting **law** is not selectable at this ABI: a
    /// `NIGHTLY_DEWEIGHTING` layer always applies Vereš, Farnocchia,
    /// Chesley & Chamberlin (2017) §3 — σ unchanged for a batch of
    /// N ≤ 4, then σ_eff = σ√(N/4). The pre-2017 σ_eff = σ√N law the
    /// engine still carries as a historical baseline is deliberately
    /// not exposed here.
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
/// pipeline; the resulting layer chain is `additional_layers`
/// followed by the preset's layers. Sigma resolution is
/// first-match-wins, so a user rule overrides the preset for its
/// station and the preset serves as the fallback (allows e.g. VFCC2017
/// + per-survey override).
///
/// A **zero-initialized struct is NOT the production default** — it
/// has `enabled = 0`, i.e. weighting disabled (uniform 1″). The
/// production combination (VFCC2017 station floors + nightly
/// de-weighting + Floor policy) must be requested explicitly:
/// `enabled = 1`, `preset = VFCC2017`, `sigma_policy = -1`, plus one
/// `NIGHTLY_DEWEIGHTING` additional layer.
#[repr(C)]
pub struct EmpyreanWeightingConfig {
    /// 1 = run the weighting pipeline, 0 = uniform 1″ weighting.
    /// Zero-init leaves weighting disabled.
    pub enabled: u8,
    /// Preset selector. One of `EMPYREAN_WEIGHTING_PRESET_*`.
    /// `0` (NONE) means no preset rules: `default_sigma_arcsec`
    /// applies uniformly and only `additional_layers` contribute
    /// rules. NONE is honored literally — there is no silent
    /// substitution of the production preset.
    pub preset: u8,
    /// Default 1σ used when no rule applies (arcsec). Exactly 0.0 is the
    /// zero-init sentinel and resolves to 1.0; negative or non-finite
    /// values are rejected with an error rather than silently read as
    /// 1.0. Ignored when preset != NONE.
    pub default_sigma_arcsec: f64,
    /// Sigma combination policy. -1 = use the preset's policy
    /// (VFCC2017 / NEODYS presets use Floor); otherwise one of
    /// `EMPYREAN_SIGMA_POLICY_*`. Note `0` is DEFAULT_ONLY — an
    /// **active override**, not "unset": a zero-initialized field
    /// replaces a preset's Floor policy with DefaultOnly. Callers
    /// who want the preset's own policy must set -1.
    pub sigma_policy: i32,
    /// Pointer to additional layers inserted AHEAD of the preset's
    /// chain (first-match-wins: they override preset rules for their
    /// stations; relative order within the array is preserved).
    /// Presets contribute station-sigma rules only — the production
    /// default chain includes exactly one `NIGHTLY_DEWEIGHTING`
    /// layer, so callers composing this array must include it
    /// explicitly or nightly de-weighting is off. At most one
    /// `NIGHTLY_DEWEIGHTING` layer is accepted per chain (duplicates
    /// compound the 1/√N de-weighting and are rejected).
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

// ── Per-axis parameter dispositions ───────────────────────────────
// The tri-state on `EmpyreanSolveFor` and the per-segment thrust
// disposition arrays. `0` and `1` are what the retired boolean flags
// meant, so `memset(0)` is unchanged; `2` is new.
/// The axis is marginalized out of the prior in covariance space. It
/// contributes nothing and changes no number. The zero-init default,
/// and what a boolean `false` always meant.
pub const EMPYREAN_PARAM_FIXED: u8 = 0;
/// The axis is estimated from the data: it occupies a solved slot and
/// comes back with a posterior variance. What a boolean `true` meant.
pub const EMPYREAN_PARAM_SOLVED: u8 = 1;
/// The axis is not estimated but is uncertain, and that uncertainty
/// reaches the state through its measurement partials — Schmidt–Kalman
/// consider analysis (Tapley, Byron D., Schutz, Bob E., and Born,
/// George H., *Statistical Orbit Determination*, Elsevier Academic
/// Press, 2004, ch. 6, "Consider Covariance Analysis").
pub const EMPYREAN_PARAM_CONSIDERED: u8 = 2;
/// The largest number of thrust Δv correction segments one fit can
/// declare, and the length of every per-segment thrust array on this
/// ABI — the dispositions, the Δv corrections, and their posterior
/// covariances.
///
/// A literal rather than an alias of the engine's own constant, because
/// cbindgen resolves array lengths textually and cannot see through a
/// crate it does not parse — an aliased value emits a header that names
/// a macro it never defines. A compile-time assertion in the library's
/// own source keeps the literal honest: it fails the build the day the
/// engine's maximum moves, which is exactly when this ABI needs a fresh
/// version bump rather than a silently truncated array.
pub const EMPYREAN_MAX_THRUST_SEGMENTS: usize = 3;
const _: () = assert!(
    EMPYREAN_MAX_THRUST_SEGMENTS == empyrean_core::determination::MAX_THRUST_SEGMENTS,
    "the engine's thrust-segment maximum moved: every per-segment array on \
     EmpyreanODResult and EmpyreanSolveFor is frozen at this width, so widening \
     it is an EMPYREAN_ABI_VERSION break, not a constant edit"
);
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
/// A consumer checks it the moment it opens the library and requires
/// **equality**, never an ordering or a range: the value names the
/// distribution release that built the library (see *The version scheme*
/// at the end of this comment), so any difference at all means a
/// different release, and the remedy is to rebuild against that
/// release's header or to repoint at the matching engine. A difference
/// is therefore no longer evidence that a frozen struct moved, and an
/// equal value is what licenses the layouts below — `dlsym` resolves on
/// symbol name alone, and the names are stable across releases while the
/// shapes behind them are not, so a mismatch allowed to proceed reads
/// the caller's arguments through the wrong layout instead of failing.
///
/// This release's ABI (0.10.0) is a single, batched break carrying the joint
/// solved-parameter covariance across the boundary in both directions,
/// plus the riders that were queued behind a version bump. Every change,
/// in full:
///
/// **The joint covariance, input side.**
///
/// - [`CoordinateState`](crate::CoordinateState) grows
///   `has_non_grav_cross` / `non_grav_cross[6][3]` — the state↔Marsden
///   border, placed beside the 6×6 it borders so
///   `empyrean_transform_coordinates` cannot move one without the
///   other;
/// - [`EmpyreanOrbit`] grows `state_param_cross` / `n_state_param_cross`
///   and `param_pair_cross` / `n_param_pair_cross` — the wide carrier,
///   as caller-owned side arrays following the `thrust_arcs` template;
/// - three new input structs — [`EmpyreanParamColumn`](crate::joint::EmpyreanParamColumn),
///   [`EmpyreanStateParamCross`](crate::joint::EmpyreanStateParamCross),
///   [`EmpyreanParamPairCross`](crate::joint::EmpyreanParamPairCross).
///
/// **Two new exported symbols** — the first movement of the parity
/// manifest in this release, and deliberate:
///
/// - [`empyrean_propagation_joint_at`](crate::propagate::empyrean_propagation_joint_at)
///   returns the propagated joint's cross terms at one
///   `(orbit_index, epoch_index)`;
/// - [`empyrean_orbit_covariance_free`](crate::propagate::empyrean_orbit_covariance_free)
///   releases what it wrote.
///
/// They exist as a separate call rather than as fields on
/// [`EmpyreanTaggedCovariance`](crate::propagate::EmpyreanTaggedCovariance)
/// **to preserve that struct's plain-old-data contract**. A caller
/// declares one on the stack and frees nothing; giving it owned arrays
/// would have turned every such caller — code correct today, and
/// recompiling without a diagnostic — into a leaking one, two
/// allocations per call, with no error and no wrong number to notice.
/// Opt-in ownership at a new entry point costs one extra symbol and
/// makes the acquisition explicit. `EmpyreanTaggedCovariance` is
/// therefore unchanged in the 0.10.0 ABI.
///
/// **New constants**, all nine of them:
///
/// - `EMPYREAN_PARAM_COLUMN_MARSDEN` / `_DT` / `_AMRAT` / `_THRUST` —
///   the parameter-column identity tags, the value set of
///   `EmpyreanParamColumn::kind`;
/// - `EMPYREAN_PARAM_FIXED` / `_SOLVED` / `_CONSIDERED` — the
///   disposition tri-state, the value set of every `u8` on
///   [`EmpyreanSolveFor`] and of its `thrust_dispositions` entries;
/// - [`EMPYREAN_MAX_THRUST_SEGMENTS`] — the length of every per-segment
///   array on this ABI;
/// - [`EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED`] (15).
///
/// **The joint covariance, output side.**
///
/// - [`EmpyreanPropagatedState`](crate::propagate::EmpyreanPropagatedState)
///   grows `orbit_cov`
///   ([`EmpyreanOrbitCovariance`](crate::joint::EmpyreanOrbitCovariance),
///   new), carrying the propagated border and carrier as library-owned
///   arrays. This is what closes leg chaining: `covariance` alone is the
///   state block, and the engine's propagated joint has non-zero
///   state↔parameter columns **even from a block-diagonal input**,
///   because propagation itself generates them. A caller who chained
///   legs on the 6×6 alone was quoting a tighter uncertainty than the
///   propagation supports;
/// - the same struct rides
///   [`EmpyreanODResult::orbit`], so a fitted orbit and a propagated
///   state expose their joint under one name with one ownership rule,
///   and `determine → propagate` and `propagate → propagate` are the
///   same field copy. The joint has exactly one home per result — there
///   is deliberately no second border on `EmpyreanODResult` itself that
///   could disagree with the one on its orbit;
/// - [`EmpyreanNonGravParams`] grows `has_dt_variance` / `dt_variance`,
///   which had no wire at all: a solved-DT fit used to round-trip with
///   its DT column closed;
/// - the non-grav 3×3, the DT variance and the AMRAT variance are now
///   sourced from the fitted **orbit** rather than from
///   `covariance_9x9`. The 9×9 is populated only for the width-9
///   Marsden fit while every carrier-bearing fit is wider, so the old
///   source reported an absent covariance for 100% of them.
///   `covariance_9x9` remains populated for its deprecation window and
///   stops being a source.
///
/// **The parameter partition.**
///
/// - [`EmpyreanSolveFor`]'s `marsden` / `dt` / `amrat` become a
///   tri-state — `0` fixed, `1` solved, `2` considered — validated
///   strictly at the boundary. `0` and `1` keep their exact former
///   meaning, so `memset(0)` and every value an older caller could
///   write are unchanged; a **semantic** break nonetheless, which is
///   why it rides this bump rather than passing unversioned;
/// - `EmpyreanSolveFor::thrust_segments` (a count) becomes
///   `thrust_dispositions[3]` (per declared segment). Two counts cannot
///   say WHICH burn is considered, and a three-segment orbit with only
///   the middle burn solved is now a routine case;
/// - [`EmpyreanODResult`] grows `dispositions`, echoing the partition
///   the fit actually ran — which is what makes an Auto escalation
///   readable after the fact, and what tells a caller whether
///   re-attaching a prior to an axis double-counts.
///
/// **Thrust, per declared segment.**
///
/// - [`EmpyreanODResult`] grows `thrust_correction_covariances` and
///   `n_thrust_segments`;
/// - `thrust_delta_m_per_s` is **re-indexed from solved to declared
///   order**, and `thrust_delta_count` becomes the declared count. This
///   is a semantic change to a shipped field. It is made here because
///   the alternative is freezing two incompatible index spaces into one
///   struct forever: a consumer pairing `thrust_delta_m_per_s[i]` with
///   `thrust_correction_covariances[i]` would otherwise attribute a Δv
///   to the wrong burn's covariance the moment any segment is
///   considered or fixed. An unsolved segment's Δv is NaN-filled
///   exactly as its covariance is.
///
/// **Riders.**
///
/// - [`EmpyreanODResult`] grows the `warnings` / `num_warnings` string
///   channel — supplied covariance a fit deliberately did not use;
/// - `EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED` (15) joins the
///   rejection codes;
/// - [`EmpyreanObservatoryConfig`](crate::planning::EmpyreanObservatoryConfig)
///   grows `min_elevation_deg` plus `has_max_sun_altitude_deg` /
///   `max_sun_altitude_deg`, matching the engine's own observatory
///   config. Both are marshaled across in full, and **no entry point
///   exported by this ABI applies them**: the gates that read them
///   belong to the engine's visibility survey, which has no C entry
///   point, while `empyrean_evaluate_plan` — the one exported consumer
///   of this struct — consults the site-invariant filters alone. They
///   ride the struct so that exposing the survey later needs no further
///   break;
/// - the impact and B-plane paths marshal the caller's non-grav DT
///   value and Marsden 3×3, both of which they silently dropped — the
///   DT drop meant those two entry points evaluated a DT comet's
///   \\(g(r)\\) at zero delay while honouring the DT prior variance
///   eleven lines away;
/// - the ephemeris path marshals the caller's Marsden 3×3 for the same
///   reason;
/// - and those three paths, plus the orbit-file writer, now share the
///   propagation path's non-grav **presence rule**: an orbit that
///   declares a non-grav covariance or a DT prior carries a non-grav
///   model even when its A coefficients are all zero. That opens the
///   Marsden columns those paths previously left closed — a behaviour
///   change on the way to fixing the drops above, not merely a
///   re-routing;
/// - the orbit-file read path carries the non-grav 3×3, the border and
///   the carrier onto [`EmpyreanOrbitBatch`](crate::io::EmpyreanOrbitBatch)
///   rather than reporting them absent.
///
/// **Layout: appended everywhere but one, and that one SHRINKS two
/// structs.** Every earlier release of this ABI could say "fields are
/// only ever appended, never reordered or removed". This one cannot, and
/// the exception is stated here rather than left for a consumer to
/// discover by corruption: replacing `EmpyreanSolveFor::thrust_segments`
/// (a `u32`) with `thrust_dispositions[3]` (three `u8`s) takes that
/// struct from 8 bytes to 6 and its alignment from 4 to 1, which shifts
/// every field after `solve_for_flags` inside the
/// [`EmpyreanODConfig`] that embeds it —
/// `allow_unbracketed_maneuvers` 392→390, `has_photometry` 393→391,
/// `photometry` 400→392 — and shrinks that config 432→424 in turn.
/// `EmpyreanSolveFor` and `EmpyreanODConfig` are the first structs on
/// this ABI to get SMALLER.
///
/// A consumer with a hand-mirrored `EmpyreanODConfig` must therefore
/// re-derive its whole layout, not just append: keeping an existing
/// prefix and writing `photometry` at its old offset lands eight bytes
/// past where the library reads it, corrupting the photometry config
/// and the two bytes before it with no diagnostic. Every other frozen
/// struct in this release grows by appending, and their sizes are
/// enumerated in the changelog.
///
/// The source-breaking changes are the two semantic ones
/// (`EmpyreanSolveFor`'s encoding and the thrust Δv index space) plus
/// the `thrust_segments` → `thrust_dispositions` replacement above — a
/// consumer built against an older header must recompile against this
/// one either way, and `empyrean_abi_version()` is what makes a
/// dynamically-loaded mismatch fail at the version check rather than in
/// the physics.
///
/// **The version scheme.** The C ABI carries the distribution's own
/// version, encoded \\(\text{major} \times 10000 + \text{minor} \times
/// 100 + \text{patch}\\). It advances with every distribution release
/// whether or not any boundary type changed: the ABI is versioned by the
/// release that ships it, not by an independent counter.
///
/// **The scheme begins with 0.10.0**, which reports `1000` — the
/// smallest value it can produce. Every release before it reported the
/// retired independent counter instead; its last published value is 2,
/// shipped by v0.9.0. That is why values below 1000 are counter-era and
/// are not release numbers.
///
/// **Only the base version is encoded.** The pre-release suffix is not:
/// `0.10.0-rc.1` and `0.10.0` both report `1000`. So this number
/// separates one version from another, and never a version from its own
/// pre-releases — if a boundary type moves inside a pre-release cycle,
/// the handshake will not catch the mismatch and both sides have to be
/// rebuilt together. Across the pre-releases of a single version, the
/// artifact or tag that was installed is the only thing that identifies
/// the exact build.
///
/// The distribution's own release string is not exported. A consumer
/// identifies the build it is running by the artifact or tag it
/// installed; `empyrean_version_string()` reports something else — the
/// build provenance of the closed-source engine crates behind this
/// boundary, not this distribution's version.
pub const EMPYREAN_ABI_VERSION: u32 = 1000;

/// The constant above is the crate's own version, encoded — and this
/// assertion is what keeps it that way. Under the retired counter the
/// value only had to move when a boundary type changed, which is a
/// reviewed event; it now has to move on every release, including the
/// release that changes nothing at the boundary and so gives nobody a
/// reason to open this file. A forgotten bump would leave
/// `empyrean_abi_version()` reporting a version the library is not,
/// which is undetectable from either side of the handshake — both sides
/// descend from this one literal.
///
/// `CARGO_PKG_VERSION_MAJOR` / `_MINOR` / `_PATCH` drop any pre-release
/// suffix, so a pre-release and its final release encode to the same
/// number and the assertion holds across a whole pre-release cycle
/// without either side moving. That is the mechanism behind the
/// base-version-only limitation stated on the constant above; it is a
/// property of this encoding, not an oversight in the check.
const fn encoded_crate_version() -> u32 {
    const fn parse_u32(s: &str) -> u32 {
        let bytes = s.as_bytes();
        let mut value = 0u32;
        let mut i = 0;
        while i < bytes.len() {
            assert!(
                bytes[i] >= b'0' && bytes[i] <= b'9',
                "a version component of this crate is not a plain integer"
            );
            value = value * 10 + (bytes[i] - b'0') as u32;
            i += 1;
        }
        value
    }
    parse_u32(env!("CARGO_PKG_VERSION_MAJOR")) * 10000
        + parse_u32(env!("CARGO_PKG_VERSION_MINOR")) * 100
        + parse_u32(env!("CARGO_PKG_VERSION_PATCH"))
}
const _: () = assert!(
    EMPYREAN_ABI_VERSION == encoded_crate_version(),
    "EMPYREAN_ABI_VERSION no longer encodes this crate's version: the ABI is \
     versioned by the distribution release that ships it, so a release bump \
     must carry the constant with it (major * 10000 + minor * 100 + patch)"
);

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
/// `obs_id`, `object_id` and `ast_cat` are heap-allocated NUL-terminated
/// UTF-8 strings; the pointers are freed by
/// [`empyrean_determine_results_free`] / [`empyrean_od_result_free`] /
/// [`empyrean_evaluate_result_free`] when the parent array is freed.
/// Do NOT free them manually.
#[repr(C)]
pub struct EmpyreanObservationResult {
    /// ADES `obsID` (or scott auto-assigned). Owned by the parent array
    /// — freed by the matching `*_result_free` call.
    pub obs_id: *mut c_char,
    /// ADES object identifier (permID / provID / trkSub) of the object
    /// this row was fitted against. Populated by
    /// [`empyrean_determine`], which groups by object — so a caller may
    /// concatenate every object's rows into one flat table and still
    /// know which fit each row belongs to.
    ///
    /// Null on the single-object paths (`empyrean_evaluate`,
    /// `empyrean_refine`), where the caller supplied the one orbit and
    /// no grouping key exists. Owned by the parent array.
    pub object_id: *mut c_char,
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
    /// D-optimality information loss on removal, from the influence
    /// pass: \\(\Delta_i = \log\det N - \log\det(N - I_i)\\). NaN
    /// if no influence pass was run; +∞ when removing this observation
    /// makes the normal matrix singular (the observation is
    /// indispensable).
    pub influence_information_loss: f64,
    /// Off-diagonal covariance of the (along-track, cross-track)
    /// residual pair (arcsec²). Symmetric 2×2: the diagonal is
    /// `along_track_error_arcsec`² / `cross_track_error_arcsec`².
    /// NaN when the AT/CT covariance is unavailable.
    pub along_cross_covariance_arcsec2: f64,
    /// Radar (delay/Doppler) residual: observed − predicted. Seconds
    /// for delay, hertz for Doppler (see `radar_kind`). NaN when
    /// `has_radar == 0`.
    pub radar_residual: f64,
    /// χ² of the radar residual. NaN when `has_radar == 0`.
    pub radar_chi2: f64,
    /// χ² survival probability of the radar residual. NaN when
    /// `has_radar == 0`.
    pub radar_probability: f64,
    /// Combined observed+predicted radar residual variance (s² for
    /// delay, Hz² for Doppler). NaN when unavailable or
    /// `has_radar == 0`.
    pub radar_variance: f64,
    /// Degrees of freedom of the radar residual (1 for radar). 0 when
    /// `has_radar == 0`.
    pub radar_dof: u32,
    /// 1 when this row is a radar observation and the `radar_*` fields
    /// are live. The optical RA/Dec residual fields are NaN on radar
    /// rows.
    pub has_radar: u8,
    /// `EMPYREAN_RADAR_KIND_DELAY` (0) or `EMPYREAN_RADAR_KIND_DOPPLER`
    /// (1). Only meaningful when `has_radar == 1`.
    pub radar_kind: u8,
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
/// Boolean fields are encoded as `u8` (0/1). Always populated on
/// [`EmpyreanODResult`]; on [`EmpyreanEvaluateResult`] the report is
/// filled with NaN/0 because evaluate does not produce a fitted orbit.
///
/// # NaN convention
///
/// **Every `f64` here is NaN when the quantity could not be computed**
/// (AT/CT ratio with no sky-motion rates, selected-arc spans when the fit
/// selected nothing, …) — NaN is the only "not computable" marker, never
/// `0.0`, because a threshold-comparison of `0.0` reads as a real
/// measurement that happens to be at the floor. The `_ok` booleans are
/// **always valid**: a gate whose value is NaN reports `0` (did not pass),
/// so a consumer can branch on the verdict without first testing the value
/// for NaN.
///
/// # Fit vs. extrapolation
///
/// `fit_acceptable` is the AND of the fit-quality gates (convergence,
/// positive-definite covariance, reduced χ², RMS, residual isotropy).
/// `extrapolation_acceptable` additionally requires the four selection /
/// coverage axes below — `selection_fraction_ok`,
/// `selected_arc_coverage_ok`, `trailing_gap_ok` and
/// `fractional_sigma_a_ok`. Those four are deliberately NOT part of
/// `fit_acceptable`: a heavily pruned fit can still describe its retained
/// subset well while being unsafe to propagate forward.
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
    /// FULL observation span at or above `arc_days_threshold`. Kept for
    /// callers that want the full-arc meaning; `extrapolation_acceptable`
    /// judges coverage on `selected_arc_coverage_ok` instead.
    pub arc_coverage_ok: u8,
    pub arc_days_value: f64,
    pub arc_days_threshold: f64,
    pub fractional_sigma_a_ok: u8,
    pub fractional_sigma_a_value: f64,
    pub fractional_sigma_a_threshold: f64,
    /// Fraction of observations retained (n_selected / n_obs) at or above
    /// the configured floor. `false` means the residual bars above
    /// describe a heavily pruned subset. Reproduce the fraction from
    /// `selection_fraction_value`, NOT from
    /// [`EmpyreanResidualSummary`] — the summary counts merged
    /// radar / occultation stub rows that were never candidates for
    /// outlier pruning, so its ratio is a different (smaller) number.
    pub selection_fraction_ok: u8,
    pub selection_fraction_value: f64,
    pub selection_fraction_threshold: f64,
    /// The SELECTED observations cover enough of the arc to extrapolate
    /// across it: the selected span clears the absolute
    /// `arc_days_threshold` floor AND spans at least
    /// `selected_arc_fraction_threshold` of the full observed span.
    pub selected_arc_coverage_ok: u8,
    /// Arc span (days) over the selected observations only. NaN when
    /// nothing is selected.
    pub selected_arc_days_value: f64,
    /// Selected-span / full-span ratio.
    pub selected_arc_fraction_value: f64,
    pub selected_arc_fraction_threshold: f64,
    /// The most-recent observations were NOT rejected. The absolute,
    /// asymmetric backstop the span-ratio axis cannot provide: it catches
    /// a short recent tail rejected off a long arc, where the ratio still
    /// passes but the discarded rows are the ones nearest a forward
    /// extrapolation target.
    pub trailing_gap_ok: u8,
    /// Days between the last selected and the last full-arc observation.
    /// `0.0` when the last kept observation is the last observation; NaN
    /// when nothing is selected.
    pub trailing_gap_days_value: f64,
    pub trailing_gap_threshold: f64,
    /// Radar astrometry joint-fit acceptability, as a tri-state:
    /// `1` = pass, `0` = fail, `-1` = not applicable (no radar
    /// contribution to this fit). Currently always `-1` — upstream
    /// reserves this for when optical and radar both constrain a fit.
    /// `-1` is distinct from `0` on purpose: "no radar" must never read
    /// as "radar failed".
    pub radar_fit_ok: i8,
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
    /// 1 when `dt_variance` carries a meaningful DT variance (the fitted
    /// posterior when DT was solved, else the carried-through prior); 0
    /// otherwise.
    pub has_dt_variance: u8,
    /// Prior/posterior variance on the non-grav time delay DT (days²).
    /// Only meaningful when `has_dt_variance = 1`.
    ///
    /// The DT posterior had no wire at all before 0.10.0: the fitted
    /// variance existed on the orbit and simply could not cross the ABI,
    /// so a solved-DT fit round-tripped with its DT column closed. Copy
    /// it onto `EmpyreanOrbit::non_grav_dt_variance` to re-open and
    /// prior that column in a follow-on refine.
    pub dt_variance: f64,
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
/// Per-axis parameter **dispositions** (mirrors scott's `SolveFor`).
/// Read only when [`EmpyreanODConfig::solve_for`] is
/// [`EMPYREAN_SOLVE_FOR_EXPLICIT`]; the three coarse
/// `EMPYREAN_SOLVE_FOR_*` codes cover the common shapes without it.
///
/// # A disposition, not a flag
///
/// Each axis says what the fit **does** with that parameter, and the
/// three answers are different operations with different mathematics:
///
/// - [`EMPYREAN_PARAM_FIXED`] — marginalized out of the prior in
///   covariance space. Contributes nothing; changes no number.
/// - [`EMPYREAN_PARAM_SOLVED`] — estimated from the data. Occupies a
///   solved slot and comes back with a posterior variance.
/// - [`EMPYREAN_PARAM_CONSIDERED`] — not estimated, but its prior
///   uncertainty reaches the posterior through its measurement partials
///   (Schmidt–Kalman consider analysis), so the reported σ accounts for
///   an error source the fit did not absorb.
///
/// A considered axis is **not** a safety margin. Under an uncorrelated
/// prior the correction strictly widens the posterior, but when the
/// orbit supplies cross terms between the considered axis and the solved
/// ones the cross-dependent terms are sign-indefinite and the posterior
/// can come back **tighter**.
///
/// Solving or considering an axis still requires its own precondition —
/// a declared prior on the orbit — enforced by scott.
///
/// # Zero-init and the version handshake
///
/// `0` is `FIXED` and `1` is `SOLVED`, which is exactly what the `0` /
/// `1` of the retired boolean flags meant, so a `memset(0)` config and
/// every value an older caller could have written are unchanged.
///
/// The widening is only safe in that direction. A caller writing `2`
/// for CONSIDERED against a pre-0.10.0 library would hit a bare
/// non-zero test and get the axis **silently solved** — a wider solved
/// set, a different fitted answer, and no error anywhere. Two things
/// prevent it: this boundary refuses any value outside `0 | 1 | 2` by
/// name and value, so a future fourth value fails loudly here rather
/// than degrading silently; and the tri-state rides the
/// [`EMPYREAN_ABI_VERSION`] break at 0.10.0, so
/// [`empyrean_abi_version`] is what makes the mismatch legible to a
/// caller compiled against 0.10.0's ABI and dynamically loaded
/// against an earlier library.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct EmpyreanSolveFor {
    /// Disposition of the Marsden A1/A2/A3 block (3 columns when
    /// solved). Solving or considering it requires the orbit to carry a
    /// non-grav covariance.
    pub marsden: u8,
    /// Disposition of the non-grav time delay DT (1 column when solved).
    /// Solving it requires `marsden` solved plus a DT value and prior
    /// variance on the orbit.
    pub dt: u8,
    /// Disposition of the SRP AMRAT (1 column when solved). Requires an
    /// SRP slot carrying an AMRAT prior variance.
    pub amrat: u8,
    /// Disposition of each declared thrust Δv segment, **positional with
    /// the orbit's `correction_covariances`** — entry `i` governs
    /// declared segment `i`.
    ///
    /// Positional rather than a count, because a considered or fixed
    /// segment sits *between* solved ones as readily as after them: a
    /// three-segment orbit with only the middle burn solved is not
    /// expressible as a count. Entries beyond the orbit's declared
    /// segment count must be [`EMPYREAN_PARAM_FIXED`].
    ///
    /// The solved count is derivable from this array, which is why the
    /// former `thrust_segments` count is gone rather than kept beside
    /// it — two spellings of one fact are two facts that can disagree.
    pub thrust_dispositions: [u8; EMPYREAN_MAX_THRUST_SEGMENTS],
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
    /// Magnitudes excluded from the fit because their photometric band
    /// has no adopted V-band conversion (unknown/unspecified band
    /// codes, comet total/nuclear magnitudes). Never silent: each
    /// exclusion is counted here and the distinct offending band codes
    /// are listed in `dropped_bands`. The observations' astrometry is
    /// unaffected.
    pub n_mags_dropped_unconvertible: usize,
    /// Distinct band codes that were dropped (owned array of owned C
    /// strings, sorted). Freed by `empyrean_od_result_free`. Null when
    /// `num_dropped_bands == 0`.
    pub dropped_bands: *mut *mut c_char,
    /// Number of entries in `dropped_bands` (0 when null).
    pub num_dropped_bands: usize,
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
    ///
    /// This is the coarse code, and it names a **solved** set: a fit
    /// that considers an axis reports `EXPLICIT` rather than the code
    /// its solved set alone would suggest, because a considered axis
    /// contributes to the delivered σ. Read
    /// [`dispositions`](Self::dispositions) for what the fit did with
    /// each axis.
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
    /// Number of **declared** thrust Δv segments (0..=3); 0 = the orbit
    /// declared no thrust.
    ///
    /// **Deprecated, and identical to
    /// [`n_thrust_segments`](Self::n_thrust_segments)** — read that one.
    /// Kept populated for one deprecation window, exactly as
    /// `covariance_9x9` is, so a consumer that reads the array bound
    /// off the field beside the array still compiles and still reads
    /// the right bound.
    ///
    /// Two names for one number is the defect that removed
    /// `EmpyreanSolveFor::thrust_segments` in this same release; this
    /// one survives only because it is a *published* field whose
    /// meaning changed rather than a new one, and deleting it in the
    /// same bump that re-indexed its array would leave a consumer no
    /// compiling intermediate state.
    pub thrust_delta_count: u32,
    /// Per-segment fitted Δv in m/s, expressed in
    /// [`dv_frame`](EmpyreanODResult::dv_frame), **indexed by declared
    /// segment**. Entries `0..thrust_delta_count` meaningful.
    ///
    /// A segment this fit did not solve has no correction and its entry
    /// is **NaN-filled**, exactly as its posterior covariance is. Read
    /// `dispositions.thrust_dispositions[i]` before the value.
    ///
    /// The index space changed in the 0.10.0 ABI, from solved order to
    /// declared order, so that this array, `thrust_correction_covariances`
    /// and `dispositions.thrust_dispositions` share one index. Under the
    /// old pairing a fit with a considered burn between two solved ones
    /// returned a Δv attributed to the wrong burn's covariance.
    pub thrust_delta_m_per_s: [[f64; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
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
    /// Event-aware trust verdict on the delivered covariance
    /// (`EMPYREAN_COVARIANCE_TRUST_*`). `NOT_EVALUATED` (0) means the
    /// call path ran no gate — absence of a verdict is not trust.
    pub covariance_trust: i32,
    /// Intervening-event kind (`EMPYREAN_TRUST_EVENT_*`); `NONE` unless
    /// `covariance_trust == ENCOUNTER_INTERVENES`.
    pub trust_event_kind: i32,
    /// Event epoch (MJD TDB). NaN when no event.
    pub trust_event_epoch_mjd_tdb: f64,
    /// Close-approach distance (AU). NaN unless
    /// `trust_event_kind == CLOSE_APPROACH`.
    pub trust_event_distance_au: f64,
    /// Nonlinearity ratio at the crossing. NaN unless
    /// `trust_event_kind == HIGH_NONLINEARITY`.
    pub trust_event_nonlinearity: f64,
    /// Threshold the nonlinearity exceeded. NaN unless
    /// `trust_event_kind == HIGH_NONLINEARITY`.
    pub trust_event_threshold: f64,
    /// Name of the approached body (owned C string; freed by
    /// `empyrean_od_result_free`). Null unless
    /// `trust_event_kind == CLOSE_APPROACH`.
    pub trust_event_body: *mut c_char,
    /// Solved-for width N of the fit the verdict refers to. 0 when the
    /// verdict carries no width (`TRUSTED` / `NOT_EVALUATED`).
    pub trust_solved_width: u32,
    /// 1 when a second-order (state-only) correction can recover the
    /// encounter (solved width 6); 0 otherwise. Meaningful only for
    /// `ENCOUNTER_INTERVENES`.
    pub trust_second_order_recoverable: u8,

    // ── Joint posterior + partition (0.10.0) ──────────────────────
    //
    // The fitted joint's CROSS terms live on `orbit.orbit_cov`, not
    // here. One home: a propagated state and a fitted orbit carry the
    // same `EmpyreanOrbitCovariance` under the same name, so leg
    // chaining reads the same field whatever produced the state, and
    // there is no second copy of the border that could disagree with
    // the first.
    /// What this fit did with each parameter axis the orbit declared —
    /// solved, considered or fixed, in the same encoding
    /// [`EmpyreanODConfig::solve_for_flags`] uses.
    ///
    /// Without it a covariance is ambiguous. An axis the fit
    /// **considered** already has its uncertainty inside the delivered
    /// 6×6, so re-attaching a prior to it double-counts; an axis the fit
    /// held **fixed** contributed nothing, so attaching a prior to it is
    /// conservative and correct. Same covariance, opposite conclusions.
    ///
    /// Reports the request as resolved against the orbit, so under
    /// [`EMPYREAN_SOLVE_FOR_AUTO`] it names the width the fit actually
    /// ran at rather than the width that was requested.
    pub dispositions: EmpyreanSolveFor,
    /// Number of **declared** thrust Δv segments — the length of the
    /// meaningful prefix of
    /// [`thrust_delta_m_per_s`](Self::thrust_delta_m_per_s),
    /// [`thrust_correction_covariances`](Self::thrust_correction_covariances)
    /// and `dispositions.thrust_dispositions`, which all share one index
    /// space.
    ///
    /// This is the count the orbit's own `correction_covariances`
    /// declares, **not** the number of segments solved — the two differ
    /// exactly when a segment is considered or fixed. Read
    /// `dispositions.thrust_dispositions[i]` to learn which.
    pub n_thrust_segments: u32,
    /// Per-segment fitted Δv correction covariances (AU/day)²,
    /// row-major, **indexed by declared segment**. Entries
    /// `0..n_thrust_segments` meaningful.
    ///
    /// A segment this fit did not solve carries no posterior and its 3×3
    /// is **NaN-filled** rather than echoing the prior: republishing a
    /// prior block under a posterior's name is the two-provenance defect
    /// this whole surface exists to remove. Read the disposition before
    /// the block.
    ///
    /// Re-feed by copying into `EmpyreanOrbit::correction_covariances`,
    /// which is caller-owned and borrowed — copy, do not alias.
    pub thrust_correction_covariances: [[[f64; 3]; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
    /// Non-fatal conditions the fit reports about itself — chiefly
    /// supplied covariance it deliberately did not use.
    ///
    /// Heap array of `num_warnings` NUL-terminated UTF-8 strings; null
    /// when `num_warnings == 0`, which is the common case. One list per
    /// fit, not per observation. Display-serialized so the ABI stays
    /// stable as the engine's warning taxonomy grows.
    ///
    /// These are delivered scientific payload, not log lines: a supplied
    /// prior cross term that had to be dropped changes how the σ for
    /// that slot should be read. Owned by the result and freed with it.
    pub warnings: *mut *mut c_char,
    /// Number of warning strings. 0 when the fit used everything it was
    /// given.
    pub num_warnings: usize,
}

// ── Per-object failure codes (EmpyreanODObjectResult::error_code) ──
/// The object delivered a fit; `error` is null.
pub const EMPYREAN_OD_FAILURE_NONE: i32 = 0;
/// Observation conversion failed (UTC → TDB, malformed record, …).
pub const EMPYREAN_OD_FAILURE_OBSERVATION_CONVERSION: i32 = 1;
/// Observer position could not be constructed (unknown observatory
/// code, epoch outside the loaded kernels, unsupported spacecraft).
pub const EMPYREAN_OD_FAILURE_OBSERVER_CONSTRUCTION: i32 = 2;
/// A roving-observer record used a coordinate system the engine does
/// not support.
pub const EMPYREAN_OD_FAILURE_UNSUPPORTED_COORDINATE_SYSTEM: i32 = 3;
/// The loaded Earth-orientation (BPC) coverage does not span the
/// observations. An **engine-configuration** failure, not a property of
/// the data — the three-Earth-kernel set must be present.
pub const EMPYREAN_OD_FAILURE_EARTH_ORIENTATION_COVERAGE: i32 = 4;
/// Initial orbit determination failed to produce a seed.
pub const EMPYREAN_OD_FAILURE_IOD: i32 = 5;
/// The N-body differential correction failed.
pub const EMPYREAN_OD_FAILURE_OD: i32 = 6;
/// Two observations of this object carried the same observation ID.
pub const EMPYREAN_OD_FAILURE_DUPLICATE_OBS_IDS: i32 = 7;
/// Radar observations were supplied with no optical astrometry. Radar
/// leaves the two plane-of-sky angular degrees of freedom
/// unconstrained, so the fit is under-determined.
pub const EMPYREAN_OD_FAILURE_RADAR_ONLY: i32 = 8;
/// An explicit non-gravitational solve could not recover A1/A2/A3.
/// Surfaced rather than silently degrading to a state-only fit.
pub const EMPYREAN_OD_FAILURE_NON_GRAV_NOT_RECOVERED: i32 = 9;

/// One object's slot in a batch [`empyrean_determine`] result.
///
/// Exactly one of the two payloads is live, selected by `delivered`:
///
/// - `delivered == 1` — `result` is a fully populated
///   [`EmpyreanODResult`], `error` is null and `error_code` is
///   [`EMPYREAN_OD_FAILURE_NONE`].
/// - `delivered == 0` — this object's fit failed. `error` carries the
///   engine's message and `error_code` classifies it
///   (`EMPYREAN_OD_FAILURE_*`). **`result` is NaN-poisoned**: every
///   `f64` is NaN, every pointer null, every count 0, and every
///   enumerated `i32` is `-1` (never a valid code), so a caller that
///   forgets to check `delivered` gets an obviously invalid record
///   rather than a plausible all-zero fit.
///
/// A failed object never aborts the batch — the other objects are still
/// fitted and delivered.
///
/// `object_id` and `error` are owned by the parent table and freed by
/// [`empyrean_determine_results_free`]. Do NOT free them manually.
#[repr(C)]
pub struct EmpyreanODObjectResult {
    /// ADES object identifier (permID / provID / trkSub) this slot's
    /// observations were grouped under. `"unknown"` when the group's
    /// records carried no identifier at all. Never null.
    pub object_id: *mut c_char,
    /// 1 when `result` carries a delivered fit; 0 when the fit failed.
    pub delivered: u8,
    /// The fit. Meaningful only when `delivered == 1`; NaN-poisoned
    /// otherwise (see the type-level note).
    pub result: EmpyreanODResult,
    /// Failure message. Null when `delivered == 1`.
    pub error: *mut c_char,
    /// `EMPYREAN_OD_FAILURE_*` classification of `error`.
    /// [`EMPYREAN_OD_FAILURE_NONE`] when `delivered == 1`.
    pub error_code: i32,
}

/// Result table of a batch [`empyrean_determine`] — one
/// [`EmpyreanODObjectResult`] per ADES object found in the
/// observations, in ascending `object_id` order.
///
/// Ordering is by identifier rather than by input row order so the same
/// observation set produces the same table regardless of how the rows
/// were interleaved, and so a caller can bisect for an object.
///
/// Release with [`empyrean_determine_results_free`] — including when
/// `empyrean_determine` returned
/// [`EMPYREAN_DETERMINE_NONE_DELIVERED`], which still populates the
/// table.
#[repr(C)]
pub struct EmpyreanDetermineResults {
    /// Owned array of per-object slots. Null only when
    /// `num_objects == 0`.
    pub objects: *mut EmpyreanODObjectResult,
    pub num_objects: usize,
    /// Initial-orbit keys that matched no observation group. A seed the
    /// caller supplied and the engine could not attach to any object is
    /// reported here rather than dropped: it means the seed's identity
    /// does not match any ADES identifier in the observations. Owned
    /// array of owned C strings.
    pub unmatched_orbit_ids: *mut *mut c_char,
    pub num_unmatched_orbit_ids: usize,
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
    /// Observation weighting pipeline configuration. Zero-init =
    /// `enabled = 0` = weighting DISABLED (uniform 1″); the
    /// production default (VFCC2017 + nightly de-weighting at floor-σ
    /// policy) must be requested explicitly. See
    /// [`EmpyreanWeightingConfig`].
    pub weighting: EmpyreanWeightingConfig,
    /// Catalog-bias-correction configuration. Zero-init =
    /// `enabled = 0` = debiasing DISABLED; the production default
    /// (EFCC2020 standard resolution, loaded from the DataManager
    /// default path) must be requested explicitly. See
    /// [`EmpyreanDebiasingConfig`].
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
    /// Allow the outward-expansion pipeline to truncate a sub-arc it
    /// cannot fit as one piece. Tri-state: `-1` (or any negative) =
    /// engine default (allowed), `1` = allowed, `0` = **forbidden**.
    ///
    /// Forbidding truncation makes an arc that spans a dynamical
    /// discontinuity FAIL loudly instead of delivering a fit of the
    /// reconcilable sub-arc with the rest tagged
    /// `EMPYREAN_REJECTION_OUTSIDE_ARC`. Two interactions matter before
    /// relying on `0`: per-observation rejection is orthogonal and still
    /// runs (set `rejection.enabled = 0` as well to fit the whole arc or
    /// fail), and under `EMPYREAN_ORIGIN_POLICY_AUTO` the refusal is a
    /// cascade trigger rather than a final answer — pin the origin with
    /// `EMPYREAN_ORIGIN_POLICY_EXPLICIT` to get a pure loud failure.
    pub allow_arc_truncation: i8,
    /// Master switch for the co-orbital IOD lane. Tri-state: `-1` (or any
    /// negative) = engine default (enabled), `1` = enabled, `0` = forced
    /// off (the historical cascade).
    ///
    /// Enabling it does not route ordinary objects through the lane: it
    /// still fires only when every co-orbitality gate passes. The lane's
    /// detection parameters are not exposed here — reach for the
    /// empyrean-core Rust API to tune them.
    pub coorbital_enabled: i8,
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
/// Test-only alias so sibling modules can build owned C strings for
/// fixture rows without duplicating the allocator.
#[cfg(test)]
pub(crate) fn alloc_cstring_for_test(s: &str) -> *mut c_char {
    alloc_cstring(s)
}

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

/// Map a per-observation rejection reason onto its stable C code.
///
/// Takes a reference: [`RejectionReason::ObserverConstructionFailed`]
/// carries a `String`, so the enum is no longer `Copy`.
fn rejection_reason_to_c(reason: &RejectionReason) -> i32 {
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
        RejectionReason::SpacecraftKernelMissing => EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING,
        RejectionReason::PerObservationSiteRequired => {
            EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED
        }
        RejectionReason::ObserverConstructionFailed(_) => {
            EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED
        }
        RejectionReason::NeverAbsorbed => EMPYREAN_REJECTION_NEVER_ABSORBED,
        RejectionReason::RejectionNotRun => EMPYREAN_REJECTION_NOT_EVALUATED,
        // Forward references: these two variants ship with the scott
        // observation-guard branch. The codes are already reserved and
        // published, so the mapping is written now and compiles the
        // moment the engine grows them — deleting it would silently
        // re-open the "48 of 48 used" hole they exist to close.
        RejectionReason::NonFiniteChi2 => EMPYREAN_REJECTION_NON_FINITE_CHI2,
        RejectionReason::MissingJacobian => EMPYREAN_REJECTION_MISSING_JACOBIAN,
    }
}

/// Map scott's per-obs result vector into a heap-allocated C array.
///
/// Each entry's `obs_id` and `ast_cat` strings are heap-allocated and
/// owned by the array. The caller frees them with the matching
/// [`empyrean_od_result_free`] / [`empyrean_evaluate_result_free`].
/// Marshal scott's per-observation records into the owned C array.
///
/// `object_id` is the ADES grouping key every row belongs to — `Some` on
/// the batch determine path (so a flattened multi-object residual table
/// stays attributable), `None` on the single-object evaluate / refine
/// paths, where it is written as a null pointer.
pub(crate) fn observation_results_to_c(
    observations: &[ObservationResult],
    object_id: Option<&str>,
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
                rejection_reason_to_c(&d.reason),
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
        let (cooks, lev, frac_info, info_loss) = match &obs.influence {
            Some(inf) => (
                inf.cooks_distance,
                inf.leverage,
                inf.fractional_information,
                inf.information_loss,
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };

        // Along/cross-track decomposition (None when no sky-motion rates).
        let (at, ct, at_err, ct_err, pa, at_ct_cov) = match &obs.along_cross_track {
            Some(act) => (
                act.along_track,
                act.cross_track,
                act.along_track_error,
                act.cross_track_error,
                act.position_angle,
                act.covariance.map(|c| c[0][1]).unwrap_or(f64::NAN),
            ),
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };

        // Radar residual (delay/Doppler) — None for optical rows.
        let (has_radar, radar_kind, radar_residual, radar_chi2, radar_dof, radar_prob, radar_var) =
            match &obs.radar {
                Some(r) => (
                    1u8,
                    match r {
                        RadarResidual::Delay(_) => EMPYREAN_RADAR_KIND_DELAY,
                        RadarResidual::Doppler(_) => EMPYREAN_RADAR_KIND_DOPPLER,
                    },
                    r.value(),
                    r.chi2(),
                    r.dof() as u32,
                    r.probability(),
                    r.variance().unwrap_or(f64::NAN),
                ),
                None => (0u8, 0u8, f64::NAN, f64::NAN, 0u32, f64::NAN, f64::NAN),
            };

        let entry = EmpyreanObservationResult {
            obs_id: alloc_cstring(&obs.obs_id),
            object_id: match object_id {
                Some(id) => alloc_cstring(id),
                None => std::ptr::null_mut(),
            },
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
            influence_information_loss: info_loss,
            along_cross_covariance_arcsec2: at_ct_cov,
            radar_residual,
            radar_chi2,
            radar_probability: radar_prob,
            radar_variance: radar_var,
            radar_dof,
            has_radar,
            radar_kind,
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
            free_cstring(entry.object_id);
            free_cstring(entry.ast_cat);
        }
        entry.obs_id = std::ptr::null_mut();
        entry.object_id = std::ptr::null_mut();
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

/// The `EmpyreanODResult` written into a failed object's slot.
///
/// Not a zeroed struct: an all-zero fit is a *plausible* fit (converged
/// at the origin with a singular covariance), and a caller who forgot to
/// test `delivered` would carry it forward silently. Every `f64` is NaN,
/// every enumerated `i32` is `-1` (outside every code's value set), every
/// count and flag is 0, and every owned pointer is null so the free path
/// is a no-op for this slot.
fn poisoned_od_result() -> EmpyreanODResult {
    EmpyreanODResult {
        orbit: EmpyreanPropagatedState {
            epoch_mjd_tdb: f64::NAN,
            x: f64::NAN,
            y: f64::NAN,
            z: f64::NAN,
            vx: f64::NAN,
            vy: f64::NAN,
            vz: f64::NAN,
            origin: -1,
            frame: -1,
            covariance: [[f64::NAN; 6]; 6],
            has_covariance: 0,
            stm: [[f64::NAN; 6]; 6],
            has_stm: 0,
            stt: [[[f64::NAN; 6]; 6]; 6],
            has_stt: 0,
            resolved_kind: 0,
            // Filled by `write_orbit_covariance` on the delivered
            // path; absent here so a poisoned slot frees nothing.
            orbit_cov: crate::joint::empty_orbit_covariance(),
        },
        observations: std::ptr::null_mut(),
        num_observations: 0,
        summary: EmpyreanResidualSummary {
            num_obs: 0,
            num_selected: 0,
            num_rejected: 0,
            chi2: f64::NAN,
            dof: 0,
            reduced_chi2: f64::NAN,
            rms_ra_arcsec: f64::NAN,
            rms_dec_arcsec: f64::NAN,
            rms_combined_arcsec: f64::NAN,
            weighted_rms_ra_arcsec: f64::NAN,
            weighted_rms_dec_arcsec: f64::NAN,
            weighted_rms_combined_arcsec: f64::NAN,
            mean_ra_arcsec: f64::NAN,
            mean_dec_arcsec: f64::NAN,
            std_ra_arcsec: f64::NAN,
            std_dec_arcsec: f64::NAN,
            rms_along_track_arcsec: f64::NAN,
            rms_cross_track_arcsec: f64::NAN,
        },
        iterations: 0,
        update_norm: f64::NAN,
        converged: 0,
        covariance: [[f64::NAN; 6]; 6],
        covariance_representation: -1,
        has_covariance_9x9: 0,
        covariance_9x9: [[f64::NAN; 9]; 9],
        has_non_grav_delta: 0,
        non_grav_delta: [f64::NAN; 3],
        has_non_grav: 0,
        non_grav: EmpyreanNonGravParams {
            a1: f64::NAN,
            a2: f64::NAN,
            a3: f64::NAN,
            ng_alpha: f64::NAN,
            ng_r0: f64::NAN,
            ng_m: f64::NAN,
            ng_n: f64::NAN,
            ng_k: f64::NAN,
            has_dt: 0,
            non_grav_dt: f64::NAN,
            has_covariance: 0,
            covariance: [[f64::NAN; 3]; 3],
            has_dt_variance: 0,
            dt_variance: f64::NAN,
        },
        rejection_passes: 0,
        num_oppositions_fit: 0,
        force_model_used: -1,
        solve_for_used: -1,
        acceptability: poisoned_acceptability_report(),
        station_biases: std::ptr::null_mut(),
        num_station_biases: 0,
        has_solved_covariance: 0,
        solved_covariance: EmpyreanSolvedCovariance {
            matrix: [[f64::NAN; EMPYREAN_SOLVE_WIDTH]; EMPYREAN_SOLVE_WIDTH],
            width: 0,
            marsden_slot: EMPYREAN_SLOT_NONE,
            dt_slot: EMPYREAN_SLOT_NONE,
            amrat_slot: EMPYREAN_SLOT_NONE,
            thrust_slots: [[EMPYREAN_SLOT_NONE; 3]; 3],
            thrust_count: 0,
        },
        has_dt_delta: 0,
        dt_delta: f64::NAN,
        has_amrat_delta: 0,
        amrat_delta: f64::NAN,
        thrust_delta_count: 0,
        thrust_delta_m_per_s: [[f64::NAN; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
        dv_frame: -1,
        has_photometry: 0,
        // Zeroed (not NaN-filled) because the free path walks its owned
        // arrays: the pointers must be null and the counts 0.
        photometry: zeroed_photometry_result(),
        has_srp: 0,
        srp: EmpyreanSRPParams {
            amrat: f64::NAN,
            cr: f64::NAN,
            has_amrat_variance: 0,
            amrat_variance: f64::NAN,
        },
        covariance_trust: -1,
        trust_event_kind: -1,
        trust_event_epoch_mjd_tdb: f64::NAN,
        trust_event_distance_au: f64::NAN,
        trust_event_nonlinearity: f64::NAN,
        trust_event_threshold: f64::NAN,
        trust_event_body: std::ptr::null_mut(),
        trust_solved_width: 0,
        trust_second_order_recoverable: 0,
        // Pointers null and counts 0 (not NaN-filled) because the free
        // path walks the owned carrier arrays, exactly as for
        // `photometry` above. The border and the thrust blocks ARE
        // NaN-filled: they are inline values, and a zeroed cross block
        // is a plausible "no correlation" claim a caller who skipped
        // the `delivered` test would carry forward.
        // Every axis FIXED: the always-valid reading, matching how the
        // acceptability block reports "no gate passed" rather than an
        // invented verdict.
        dispositions: EmpyreanSolveFor {
            marsden: EMPYREAN_PARAM_FIXED,
            dt: EMPYREAN_PARAM_FIXED,
            amrat: EMPYREAN_PARAM_FIXED,
            thrust_dispositions: [EMPYREAN_PARAM_FIXED; EMPYREAN_MAX_THRUST_SEGMENTS],
        },
        n_thrust_segments: 0,
        thrust_correction_covariances: [[[f64::NAN; 3]; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
        warnings: std::ptr::null_mut(),
        num_warnings: 0,
    }
}

/// The acceptability block of a failed object's slot: no gate passed
/// (every `_ok` is 0, which is the always-valid reading) and no
/// measurement exists (every value is NaN).
fn poisoned_acceptability_report() -> EmpyreanAcceptabilityReport {
    EmpyreanAcceptabilityReport {
        fit_acceptable: 0,
        extrapolation_acceptable: 0,
        converged_ok: 0,
        reduced_chi2_ok: 0,
        reduced_chi2_value: f64::NAN,
        reduced_chi2_threshold: f64::NAN,
        rms_ok: 0,
        rms_value_arcsec: f64::NAN,
        rms_threshold_arcsec: f64::NAN,
        residual_isotropy_ok: 0,
        at_ct_ratio_value: f64::NAN,
        at_ct_ratio_threshold: f64::NAN,
        covariance_ok: 0,
        arc_coverage_ok: 0,
        arc_days_value: f64::NAN,
        arc_days_threshold: f64::NAN,
        fractional_sigma_a_ok: 0,
        fractional_sigma_a_value: f64::NAN,
        fractional_sigma_a_threshold: f64::NAN,
        selection_fraction_ok: 0,
        selection_fraction_value: f64::NAN,
        selection_fraction_threshold: f64::NAN,
        selected_arc_coverage_ok: 0,
        selected_arc_days_value: f64::NAN,
        selected_arc_fraction_value: f64::NAN,
        selected_arc_fraction_threshold: f64::NAN,
        trailing_gap_ok: 0,
        trailing_gap_days_value: f64::NAN,
        trailing_gap_threshold: f64::NAN,
        radar_fit_ok: -1,
    }
}

/// Classify a per-object [`DetermineError`] into its stable
/// `EMPYREAN_OD_FAILURE_*` code.
///
/// Exhaustive on purpose: a new upstream variant must fail this match at
/// compile time rather than fall into a catch-all that reports the wrong
/// cause.
fn determine_error_code(e: &DetermineError) -> i32 {
    match e {
        DetermineError::ObservationConversion(_) => EMPYREAN_OD_FAILURE_OBSERVATION_CONVERSION,
        DetermineError::ObserverConstruction { .. } => EMPYREAN_OD_FAILURE_OBSERVER_CONSTRUCTION,
        DetermineError::UnsupportedCoordinateSystem { .. } => {
            EMPYREAN_OD_FAILURE_UNSUPPORTED_COORDINATE_SYSTEM
        }
        DetermineError::EarthOrientationCoverageIncomplete { .. } => {
            EMPYREAN_OD_FAILURE_EARTH_ORIENTATION_COVERAGE
        }
        DetermineError::IOD(_) => EMPYREAN_OD_FAILURE_IOD,
        DetermineError::OD(_) => EMPYREAN_OD_FAILURE_OD,
        DetermineError::DuplicateObsIds(_) => EMPYREAN_OD_FAILURE_DUPLICATE_OBS_IDS,
        DetermineError::RadarOnly { .. } => EMPYREAN_OD_FAILURE_RADAR_ONLY,
        DetermineError::NonGravNotRecovered { .. } => EMPYREAN_OD_FAILURE_NON_GRAV_NOT_RECOVERED,
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

/// Marshal a string slice into an owned C array of owned C strings.
/// Returns `(null, 0)` for an empty slice. Free with
/// [`free_string_array`].
fn string_vec_to_c(strings: &[String]) -> (*mut *mut c_char, usize) {
    if strings.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let n = strings.len();
    let layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut c_char;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, s) in strings.iter().enumerate() {
        unsafe { ptr.add(i).write(alloc_cstring(s)) };
    }
    (ptr, n)
}

/// Free a string array previously produced by [`string_vec_to_c`].
unsafe fn free_string_array(ptr: *mut *mut c_char, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    for i in 0..n {
        unsafe { free_cstring(*ptr.add(i)) };
    }
    let layout = std::alloc::Layout::array::<*mut c_char>(n).unwrap();
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
}

fn photometry_result_to_c(p: &PhotometryResult) -> EmpyreanODPhotometryResult {
    let (has_covariance, covariance) = match p.params.covariance {
        Some(c) => (1u8, c),
        None => (0u8, [[0.0; 3]; 3]),
    };
    let (per_band, num_per_band) = band_stats_to_c(&p.per_band);
    let (gates, num_gates) = gate_records_to_c(&p.gates);
    let (dropped_bands, num_dropped_bands) = string_vec_to_c(&p.dropped_bands);
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
        n_mags_dropped_unconvertible: p.n_mags_dropped_unconvertible,
        dropped_bands,
        num_dropped_bands,
    }
}

pub(crate) fn zeroed_photometry_result() -> EmpyreanODPhotometryResult {
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
        n_mags_dropped_unconvertible: 0,
        dropped_bands: std::ptr::null_mut(),
        num_dropped_bands: 0,
    }
}

/// Write the covariance-trust verdict fields on a result out-pointer.
/// `None` maps to `NOT_EVALUATED` — produced by call paths that run no
/// trust gate; absence of a verdict is not trust.
pub(crate) unsafe fn write_covariance_trust(
    result_out: *mut EmpyreanODResult,
    trust: &Option<CovarianceTrust>,
) {
    let (verdict, kind, epoch, dist, nonlin, thr, body, width, recoverable) = match trust {
        None => (
            EMPYREAN_COVARIANCE_TRUST_NOT_EVALUATED,
            EMPYREAN_TRUST_EVENT_NONE,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            std::ptr::null_mut(),
            0u32,
            0u8,
        ),
        Some(CovarianceTrust::Trusted) => (
            EMPYREAN_COVARIANCE_TRUST_TRUSTED,
            EMPYREAN_TRUST_EVENT_NONE,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            std::ptr::null_mut(),
            0u32,
            0u8,
        ),
        Some(CovarianceTrust::EncounterIntervenes {
            event,
            solved_width,
            second_order_recoverable,
        }) => {
            let (kind, epoch, dist, nonlin, thr, body) = match event {
                TrustGateEvent::CloseApproach {
                    body,
                    epoch_mjd_tdb,
                    distance_au,
                } => (
                    EMPYREAN_TRUST_EVENT_CLOSE_APPROACH,
                    *epoch_mjd_tdb,
                    *distance_au,
                    f64::NAN,
                    f64::NAN,
                    alloc_cstring(body),
                ),
                TrustGateEvent::HighNonlinearity {
                    epoch_mjd_tdb,
                    nonlinearity,
                    threshold,
                } => (
                    EMPYREAN_TRUST_EVENT_HIGH_NONLINEARITY,
                    *epoch_mjd_tdb,
                    f64::NAN,
                    *nonlinearity,
                    *threshold,
                    std::ptr::null_mut(),
                ),
            };
            (
                EMPYREAN_COVARIANCE_TRUST_ENCOUNTER_INTERVENES,
                kind,
                epoch,
                dist,
                nonlin,
                thr,
                body,
                *solved_width as u32,
                u8::from(*second_order_recoverable),
            )
        }
        Some(CovarianceTrust::WeaklyDeterminedHighN { solved_width }) => (
            EMPYREAN_COVARIANCE_TRUST_WEAKLY_DETERMINED_HIGH_N,
            EMPYREAN_TRUST_EVENT_NONE,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            std::ptr::null_mut(),
            *solved_width as u32,
            0u8,
        ),
    };
    unsafe {
        (*result_out).covariance_trust = verdict;
        (*result_out).trust_event_kind = kind;
        (*result_out).trust_event_epoch_mjd_tdb = epoch;
        (*result_out).trust_event_distance_au = dist;
        (*result_out).trust_event_nonlinearity = nonlin;
        (*result_out).trust_event_threshold = thr;
        (*result_out).trust_event_body = body;
        (*result_out).trust_solved_width = width;
        (*result_out).trust_second_order_recoverable = recoverable;
    }
}

/// Write every field of an [`ODResult`] into the C out-struct **except**
/// `orbit`, which the caller supplies because the entry paths build the
/// propagated state differently.
///
/// This is the single source of truth for the OD output surface. Both the
/// one-shot entry points and the session path route through it, so a field
/// added to [`ODResult`] can never again reach one surface while defaulting
/// to zero on the other — the failure mode that had the session path
/// returning an all-zero covariance, `NaN` non-gravitational parameters and
/// a `solve_for_used` that disagreed with the fit that actually ran.
pub(crate) unsafe fn write_od_result_fields(
    result_out: *mut EmpyreanODResult,
    od: &ODResult,
    object_id: Option<&str>,
) -> Result<(), String> {
    let (obs_ptr, obs_n) = observation_results_to_c(&od.observations, object_id);
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
        (*result_out).observations = obs_ptr;
        (*result_out).num_observations = obs_n;
        (*result_out).summary = summary;
        (*result_out).iterations = od.iterations as u32;
        (*result_out).update_norm = od.update_norm;
        (*result_out).converged = u8::from(od.acceptability.converged_ok);
        (*result_out).covariance = od.covariance;
        (*result_out).covariance_representation = coord_rep_to_int(od.covariance_representation);
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
        populate_wide_fitting_fields(result_out, od)?;
    }
    Ok(())
}

/// Populate the v0.9.0 wide-fitting fields on a result out-pointer from
/// scott's `ODResult`. ALWAYS writes every field (zeros / sentinels when
/// an axis was not solved) — no defaulted covariance presented as real,
/// per the full-population contract.
unsafe fn populate_wide_fitting_fields(
    result_out: *mut EmpyreanODResult,
    od: &ODResult,
) -> Result<(), String> {
    unsafe {
        write_covariance_trust(result_out, &od.covariance_trust);
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
        write_thrust_posterior_fields(result_out, od);
        (*result_out).dv_frame = od.dv_frame.map(frame_to_int).unwrap_or(0);
        (*result_out).dispositions = solve_for_to_c(&od.dispositions);
        write_orbit_covariance(result_out, od)?;
        let (warn_ptr, num_warnings) = od_warnings_to_c(&od.warnings);
        (*result_out).warnings = warn_ptr;
        (*result_out).num_warnings = num_warnings;
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
    Ok(())
}

/// Write the per-segment thrust surface — the Δv corrections, their
/// posterior covariances, and the declared count they share.
///
/// # One index space, deliberately
///
/// All three per-segment arrays on the result — Δv, covariance,
/// disposition — are indexed by **declared** segment. Before 0.10.0 the Δv
/// array was indexed by *solved* segment while nothing else was, which
/// was invisible while declared ≡ solved was an enforced invariant. It
/// no longer is: a considered or fixed burn is declared-but-unsolved,
/// and the two orders diverge the moment one appears. A consumer writing
/// the obvious `thrust_delta_m_per_s[i]` beside
/// `thrust_correction_covariances[i]` would then pair a Δv with another
/// burn's covariance — silently, having followed the header.
///
/// So the Δv array is re-indexed here and an unsolved segment's Δv is
/// NaN-filled exactly as its covariance is. That is a semantic change to
/// a shipped field, which is acceptable only because it rides an ABI
/// major bump and because the alternative is freezing two incompatible
/// index spaces into one struct forever.
unsafe fn write_thrust_posterior_fields(result_out: *mut EmpyreanODResult, od: &ODResult) {
    let (declared, dv, cov) = thrust_posterior_arrays(
        od.thrust_delta_declared().as_deref(),
        od.thrust_correction_covariances.as_deref(),
    );
    unsafe {
        (*result_out).n_thrust_segments = declared;
        (*result_out).thrust_delta_count = declared;
        (*result_out).thrust_delta_m_per_s = dv;
        (*result_out).thrust_correction_covariances = cov;
    }
}

/// The declared-indexed thrust arrays, as pure data.
///
/// Both inputs are already in **declared** segment order, so this
/// scatters nothing — it converts the Δv to m/s, NaN-fills every
/// unsolved slot in both arrays, and reports the declared count they
/// share. Split out of the writer so the index space and the NaN-fill
/// are testable without assembling a whole [`ODResult`].
#[allow(clippy::type_complexity)]
fn thrust_posterior_arrays(
    dv_declared: Option<&[Option<[f64; 3]>]>,
    cov_declared: Option<&[Option<[[f64; 3]; 3]>]>,
) -> (
    u32,
    [[f64; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
    [[[f64; 3]; 3]; EMPYREAN_MAX_THRUST_SEGMENTS],
) {
    use empyrean_core::constants::{M_PER_AU, S_PER_DAY};
    const AU_PER_DAY_TO_M_PER_S: f64 = M_PER_AU / S_PER_DAY;

    let mut dv = [[f64::NAN; 3]; EMPYREAN_MAX_THRUST_SEGMENTS];
    let mut cov = [[[f64::NAN; 3]; 3]; EMPYREAN_MAX_THRUST_SEGMENTS];

    // The declared count comes from the covariance array, whose length
    // IS the orbit's declared segment count — the space `wide_layout`
    // derives its thrust block in. Not `SolvedCovariance::thrust_count`,
    // which counts only the solved segments and diverges the moment one
    // is considered or fixed.
    let declared = cov_declared
        .map_or(0, |c| c.len())
        .min(EMPYREAN_MAX_THRUST_SEGMENTS);

    if let Some(dvs) = dv_declared {
        for (i, entry) in dvs.iter().take(EMPYREAN_MAX_THRUST_SEGMENTS).enumerate() {
            if let Some(v) = entry {
                dv[i] = [
                    v[0] * AU_PER_DAY_TO_M_PER_S,
                    v[1] * AU_PER_DAY_TO_M_PER_S,
                    v[2] * AU_PER_DAY_TO_M_PER_S,
                ];
            }
        }
    }
    if let Some(blocks) = cov_declared {
        for (i, entry) in blocks.iter().take(EMPYREAN_MAX_THRUST_SEGMENTS).enumerate() {
            // `None` stays NaN-filled: a considered or fixed segment has
            // no posterior, and echoing its prior here would republish a
            // prior under a posterior's name.
            if let Some(block) = entry {
                cov[i] = *block;
            }
        }
    }
    (declared as u32, dv, cov)
}

/// Read the fitted orbit's border and wide carrier out onto the result.
///
/// Both are sourced from the same `ODResult::orbit` every other
/// posterior block comes from, in that orbit's own output
/// representation — so the crosses and the diagonals they are
/// conditioned on describe one matrix rather than two provenances.
///
/// Fails on a failed allocation rather than publishing an absent
/// carrier: "this fit produced no cross terms" and "we could not
/// allocate them" are different claims, and a caller who read the first
/// when the second was true would re-feed a block-diagonal joint — the
/// exact error this surface exists to prevent.
unsafe fn write_orbit_covariance(
    result_out: *mut EmpyreanODResult,
    od: &ODResult,
) -> Result<(), String> {
    // ── Basis: whatever the 6×6 beside it is ──────────────────────
    //
    // Read the joint off the STORED coordinate, un-rescaled, because
    // that is exactly where `ODResult::covariance` comes from: the
    // estimator reads its 6×6 straight off this coordinate, and
    // `EmpyreanODResult::covariance` publishes it verbatim. Taking the
    // border and carrier in the coordinate's angular unit instead would
    // put degree-scaled crosses beside a radian 6×6 whenever the fit's
    // output representation is Keplerian / Cometary / Spherical — a
    // factor of 180/π on the angular rows, within one struct whose own
    // documented contract is that the two share a basis AND units.
    //
    // The one-shot entry points never reach a non-Cartesian result
    // (`od_orbit_to_propagated` refuses one), where the distinction is a
    // no-op because Cartesian has no angular rows. The SESSION path does
    // deliver Keplerian / Cometary / Spherical results, so the
    // distinction is live there and only there.
    let Some(coord) = od.orbit.coordinates().first() else {
        // No row 0 at all: leave the joint absent rather than
        // fabricating one. Absent is the truth here, not a fallback.
        return Ok(());
    };
    let joint = crate::joint::joint_to_c(
        coord.extended_covariance(),
        od.orbit.wide_cross(0),
        "fitted orbit",
    )?;
    unsafe {
        (*result_out).orbit.orbit_cov = joint;
    }
    Ok(())
}

/// Marshal the fit's warning list into the C string-array channel.
///
/// Display-serialized rather than mirrored as an enum, because the
/// engine's warning taxonomy is explicitly open-ended: mirroring it
/// would freeze a set that is meant to grow, and every consumer would
/// break on the next variant.
fn od_warnings_to_c(warnings: &[ODWarning]) -> (*mut *mut c_char, usize) {
    if warnings.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let n = warnings.len();
    let Ok(layout) = std::alloc::Layout::array::<*mut c_char>(n) else {
        set_last_error("allocation failed for the OD warnings array");
        return (std::ptr::null_mut(), 0);
    };
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut *mut c_char;
    if ptr.is_null() {
        set_last_error("allocation failed for the OD warnings array");
        return (std::ptr::null_mut(), 0);
    }
    for (i, w) in warnings.iter().enumerate() {
        unsafe { ptr.add(i).write(alloc_cstring(&w.to_string())) };
    }
    (ptr, n)
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
        selection_fraction_ok: u8::from(r.selection_fraction_ok),
        selection_fraction_value: r.selection_fraction_value,
        selection_fraction_threshold: r.selection_fraction_threshold,
        selected_arc_coverage_ok: u8::from(r.selected_arc_coverage_ok),
        selected_arc_days_value: r.selected_arc_days_value,
        selected_arc_fraction_value: r.selected_arc_fraction_value,
        selected_arc_fraction_threshold: r.selected_arc_fraction_threshold,
        trailing_gap_ok: u8::from(r.trailing_gap_ok),
        trailing_gap_days_value: r.trailing_gap_days_value,
        trailing_gap_threshold: r.trailing_gap_threshold,
        // Tri-state: -1 = no radar contribution, so "no radar" never
        // reads as "radar failed".
        radar_fit_ok: match r.radar_fit_ok {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
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
            // The two coarse codes name a SOLVED set, so a considered
            // axis disqualifies them: a fit that considers AMRAT is not
            // the state-only fit `STATE_ONLY` names, and reporting it as
            // one would hide a σ contribution the caller asked for. Such
            // a fit reports EXPLICIT, and the disposition echo carries
            // the detail.
            let quiet = |d: ParamDisposition| d == ParamDisposition::Fixed;
            let thrust_quiet = sf.thrust.iter().copied().all(quiet);
            let state_only = quiet(sf.marsden) && quiet(sf.dt) && quiet(sf.amrat) && thrust_quiet;
            let non_grav_only = sf.marsden == ParamDisposition::Solved
                && quiet(sf.dt)
                && quiet(sf.amrat)
                && thrust_quiet;
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
/// The non-grav half of [`od_result_non_grav_to_c`], driven directly by
/// the tests.
///
/// `od_result_non_grav_to_c` takes a whole `ODResult` — 30-odd fields,
/// none defaulted — but reads exactly one thing from it: the fitted
/// orbit's non-grav block. Splitting the read out lets the marshal
/// itself be tested without assembling a result, so the coverage is of
/// the shipped code path rather than of a fixture.
#[cfg(test)]
pub(crate) fn od_result_non_grav_to_c_for_test(orbit: &Orbits<AU>) -> (u8, EmpyreanNonGravParams) {
    non_grav_block_to_c(orbit)
}

fn od_result_non_grav_to_c(od: &ODResult) -> (u8, EmpyreanNonGravParams) {
    non_grav_block_to_c(&od.orbit)
}

/// Read an orbit's non-grav block into the flat C shape.
fn non_grav_block_to_c(orbit: &Orbits<AU>) -> (u8, EmpyreanNonGravParams) {
    match orbit.non_grav_params(0) {
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
            // re-feedable orbit keeps its non-grav prior for a
            // StateAndNonGrav refine.
            //
            // Sourced from the fitted ORBIT rather than from
            // `od.covariance_9x9`, which is where it used to come from.
            // The 9×9 is populated only for the width-9 Marsden fit, and
            // every fit carrying a wide joint is wider than 9 by
            // construction — a carrier needs a DT, AMRAT or thrust
            // column, and each of those pushes the width past 9. The two
            // sets are disjoint, so the old source reported "no
            // covariance" and a zero 3×3 for 100% of the fits this
            // surface exists to serve, while the orbit held the
            // posterior. Re-feeding that yielded either a border with no
            // parameter block (refused) or, worse, a caller clearing the
            // border and silently substituting their prior.
            //
            // The estimator writes every posterior block onto this one
            // orbit in a single pass and then builds the border and the
            // carrier by reading those blocks back, so sourcing them all
            // from here is what keeps the diagonals and the crosses
            // conditioned on them from coming from two provenances.
            let (has_covariance, covariance) = match ng.covariance {
                Some(c) => (1u8, c),
                None => (0u8, [[0.0_f64; 3]; 3]),
            };
            let (has_dt_variance, dt_variance) = match ng.dt_variance {
                Some(v) => (1u8, v),
                None => (0u8, f64::NAN),
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
                    has_dt_variance,
                    dt_variance,
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
        // Absent at construction: `write_orbit_covariance` fills it
        // from the fitted orbit once the result struct exists.
        orbit_cov: crate::joint::empty_orbit_covariance(),
    })
}

/// Translate one C disposition byte into the engine's
/// [`ParamDisposition`].
///
/// **Strict**: `0`, `1` and `2` are accepted and everything else is
/// refused, naming the field, the value and the legal set. A bare
/// non-zero test would be the same silent widening this tri-state
/// exists to remove, one layer down — and it would swallow a *future*
/// fourth disposition (a conditioned axis, say) as "solved" rather than
/// failing on a library that cannot perform it.
fn param_disposition_from_c(v: u8, field: &str) -> Result<ParamDisposition, String> {
    match v {
        EMPYREAN_PARAM_FIXED => Ok(ParamDisposition::Fixed),
        EMPYREAN_PARAM_SOLVED => Ok(ParamDisposition::Solved),
        EMPYREAN_PARAM_CONSIDERED => Ok(ParamDisposition::Considered),
        other => Err(format!(
            "solve_for_flags.{field} = {other} is not a parameter disposition; the legal \
             values are {EMPYREAN_PARAM_FIXED} (fixed), {EMPYREAN_PARAM_SOLVED} (solved) \
             and {EMPYREAN_PARAM_CONSIDERED} (considered)"
        )),
    }
}

/// Translate the engine's [`ParamDisposition`] into its C byte.
///
/// Total by construction — every variant maps, so a disposition the
/// engine grows later is a compile error here rather than a value that
/// crosses the ABI wearing another's meaning.
fn param_disposition_to_c(d: ParamDisposition) -> u8 {
    match d {
        ParamDisposition::Fixed => EMPYREAN_PARAM_FIXED,
        ParamDisposition::Solved => EMPYREAN_PARAM_SOLVED,
        ParamDisposition::Considered => EMPYREAN_PARAM_CONSIDERED,
    }
}

/// Read the per-axis disposition struct into scott's [`SolveFor`].
///
/// Every byte is validated: an out-of-range value on any axis, or on any
/// entry of the thrust array, refuses the call by name and value rather
/// than resolving to something the caller did not ask for.
fn solve_for_from_c(f: &EmpyreanSolveFor) -> Result<SolveFor, String> {
    let mut thrust = [ParamDisposition::Fixed; EMPYREAN_MAX_THRUST_SEGMENTS];
    for (i, d) in f.thrust_dispositions.iter().enumerate() {
        thrust[i] = param_disposition_from_c(*d, &format!("thrust_dispositions[{i}]"))?;
    }
    Ok(SolveFor {
        marsden: param_disposition_from_c(f.marsden, "marsden")?,
        dt: param_disposition_from_c(f.dt, "dt")?,
        amrat: param_disposition_from_c(f.amrat, "amrat")?,
        thrust,
    })
}

/// Flatten scott's [`SolveFor`] into the C disposition struct, for the
/// result path's echo of the partition the fit actually ran.
fn solve_for_to_c(s: &SolveFor) -> EmpyreanSolveFor {
    let mut thrust_dispositions = [EMPYREAN_PARAM_FIXED; EMPYREAN_MAX_THRUST_SEGMENTS];
    for (i, d) in s.thrust.iter().enumerate() {
        thrust_dispositions[i] = param_disposition_to_c(*d);
    }
    EmpyreanSolveFor {
        marsden: param_disposition_to_c(s.marsden),
        dt: param_disposition_to_c(s.dt),
        amrat: param_disposition_to_c(s.amrat),
        thrust_dispositions,
    }
}

fn int_to_solve_for(v: i32) -> Result<SolveForParams, String> {
    match v {
        EMPYREAN_SOLVE_FOR_STATE_ONLY => Ok(SolveForParams::state_only()),
        EMPYREAN_SOLVE_FOR_STATE_AND_NONGRAV => Ok(SolveForParams::state_and_non_grav()),
        EMPYREAN_SOLVE_FOR_AUTO => Ok(SolveForParams::Auto),
        EMPYREAN_SOLVE_FOR_EXPLICIT => Err(
            "solve_for = EXPLICIT (3) requires the per-axis dispositions \
             (marsden / dt / amrat / thrust_dispositions); pass them via the \
             EmpyreanSolveFor struct on EmpyreanODConfig"
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
    use empyrean_core::determination::{
        FloorsTable, NightlyDeweighting, SigmaPolicy, WeightingConfig, WeightingLayer,
    };
    use empyrean_core::time::Epoch;

    if c.enabled == 0 {
        return Ok(None);
    }

    // `preset = NONE` means exactly what it says: no preset rules, the
    // caller's `default_sigma_arcsec` applies uniformly (DefaultOnly
    // policy unless `sigma_policy` overrides it). There is NO silent
    // substitution of the production preset — a caller who wants VFCC2017
    // must request it. (A zero-initialized struct never reaches this
    // code: `enabled = 0` returns above, i.e. zero-init = weighting
    // disabled, not the production default.)
    // `default_sigma_arcsec` splits into a real "unset" state and a caller
    // bug. Exactly 0.0 is the documented zero-init sentinel and resolves to
    // 1 arcsec; anything negative or non-finite is not a sentinel, it is a
    // nonsense sigma, and silently reading it as 1 arcsec would fit the arc
    // against a weight the caller never asked for.
    if c.default_sigma_arcsec < 0.0 || c.default_sigma_arcsec.is_nan() {
        return Err(format!(
            "weighting default_sigma_arcsec must be finite and >= 0 \
             (0.0 means \"unset\", resolving to 1.0 arcsec); got {}",
            c.default_sigma_arcsec
        ));
    }
    if c.default_sigma_arcsec.is_infinite() {
        return Err(
            "weighting default_sigma_arcsec must be finite (0.0 means \"unset\", \
             resolving to 1.0 arcsec); got inf"
                .to_string(),
        );
    }
    let mut wcfg = match c.preset {
        EMPYREAN_WEIGHTING_PRESET_NONE => WeightingConfig {
            default_sigma_arcsec: if c.default_sigma_arcsec > 0.0 {
                c.default_sigma_arcsec
            } else {
                1.0
            },
            layers: Vec::new(),
            sigma_policy: SigmaPolicy::default(),
            // `NONE` builds the layer chain by hand, so it has no
            // published station-floors table behind it — the identity is
            // recorded as `FloorsTable::Custom`. The identity is *recorded*,
            // never inferred, so it stays `Custom` even once user layers
            // are inserted below: a chain that merely resembles a
            // published table must not be reported as one. The preset
            // arms below carry whatever identity their own constructor
            // set.
            //
            // Named directly now that `FloorsTable` is re-exported from
            // `empyrean_core::determination` (it is part of the public
            // shape of the re-exported `WeightingConfig`). This is the same
            // value the engine's own `Default` yields, but spelling the
            // variant makes the provenance visible at the call site and
            // fails to compile if the empty-chain identity ever changes;
            // `weighting_identity_tests` pins it with a direct
            // variant-equality assertion.
            //
            // This arm is the only site that sets the field: it builds an
            // EMPTY layer chain, so "no published table identity" is the
            // true provenance. The preset arms below keep whatever
            // identity their own constructor recorded.
            floors_table: FloorsTable::Custom,
        },
        EMPYREAN_WEIGHTING_PRESET_VFCC2017 => {
            WeightingConfig::veres_farnocchia_chesley_chamberlin_2017()
        }
        EMPYREAN_WEIGHTING_PRESET_NEODYS => WeightingConfig::neodys()
            .map_err(|e| format!("failed to load NEODyS weighting preset: {e}"))?,
        other => {
            return Err(format!(
                "unsupported weighting.preset = {other} (expected NONE = {} / VFCC2017 = {} / NEODYS = {})",
                EMPYREAN_WEIGHTING_PRESET_NONE,
                EMPYREAN_WEIGHTING_PRESET_VFCC2017,
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
        let mut user_layers: Vec<WeightingLayer> = Vec::with_capacity(slice.len());
        for (idx, layer) in slice.iter().enumerate() {
            let parsed = match layer.kind {
                EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE => {
                    // Strict obs_code decode. Station matching is exact
                    // and case-sensitive, so a malformed code silently
                    // matches nothing (or, after lossy repair/trim, the
                    // WRONG station) — reject instead of repairing.
                    let nul = layer
                        .obs_code
                        .iter()
                        .position(|&b| b == 0)
                        .unwrap_or(layer.obs_code.len());
                    if layer.obs_code[nul..].iter().any(|&b| b != 0) {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: obs_code has non-zero bytes \
                             after the NUL terminator ({:?}); pack the code left-aligned and \
                             NUL-pad",
                            layer.obs_code
                        ));
                    }
                    let code = match std::str::from_utf8(&layer.obs_code[..nul]) {
                        Ok(s) => s,
                        Err(_) => {
                            return Err(format!(
                                "weighting.additional_layers[{idx}]: obs_code bytes {:?} are \
                                 not valid UTF-8",
                                &layer.obs_code[..nul]
                            ));
                        }
                    };
                    if code.is_empty() {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: ObservatoryRule has empty \
                             obs_code"
                        ));
                    }
                    if !code.chars().all(|ch| ch.is_ascii_graphic()) {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: obs_code {code:?} must be \
                             printable ASCII with no whitespace — MPC station matching is \
                             exact and case-sensitive"
                        ));
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
                    // A non-positive or non-finite scale was previously
                    // clamped to 1.0 silently (and +inf sailed through,
                    // yielding NaN/-0.0 chi2). Weight scaling is a
                    // physical claim about the observations — reject
                    // instead of substituting.
                    if !layer.scale.is_finite() || layer.scale <= 0.0 {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: ObservatoryRule scale must be \
                             finite and > 0, got {}; use 1.0 for no scaling",
                            layer.scale
                        ));
                    }
                    WeightingLayer::ObservatoryRule {
                        obs_code: code.to_string(),
                        sigma: [layer.sigma_ra_arcsec, layer.sigma_dec_arcsec],
                        start_epoch,
                        end_epoch,
                        scale: layer.scale,
                    }
                }
                EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING => {
                    // NightlyDeweighting reads ONLY max_gap_days. The
                    // ObservatoryRule fields have no effect on this
                    // kind — rather than accept-and-ignore them (the
                    // caller believes they scoped the layer; the
                    // scoping silently never happens), reject loudly.
                    if layer.obs_code.iter().any(|&b| b != 0) {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: NIGHTLY_DEWEIGHTING does not \
                             support obs_code scoping (nightly de-weighting groups per station \
                             internally and always applies to every station); leave obs_code \
                             zeroed, or use an OBSERVATORY_RULE layer for per-station sigmas"
                        ));
                    }
                    if layer.sigma_ra_arcsec != 0.0 || layer.sigma_dec_arcsec != 0.0 {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: NIGHTLY_DEWEIGHTING does not \
                             read sigma_ra_arcsec/sigma_dec_arcsec; set them to 0.0, or use an \
                             OBSERVATORY_RULE layer to assign sigmas"
                        ));
                    }
                    let epoch_set = |v: f64| v.is_finite() && v != 0.0;
                    if epoch_set(layer.start_epoch_mjd_tdb) || epoch_set(layer.end_epoch_mjd_tdb) {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: NIGHTLY_DEWEIGHTING does not \
                             support a time range (start/end_epoch_mjd_tdb are not read); set \
                             them to NaN or 0.0"
                        ));
                    }
                    if layer.scale != 0.0 {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: NIGHTLY_DEWEIGHTING does not \
                             read scale; set it to 0.0"
                        ));
                    }
                    // Previously 0.0 (and any non-positive/NaN value)
                    // silently became the 0.5-day default. Same-night
                    // grouping is a physical claim — reject instead of
                    // substituting.
                    if !layer.max_gap_days.is_finite() || layer.max_gap_days <= 0.0 {
                        return Err(format!(
                            "weighting.additional_layers[{idx}]: NIGHTLY_DEWEIGHTING \
                             max_gap_days must be finite and > 0 (days), got {}; the \
                             production default is 0.5",
                            layer.max_gap_days
                        ));
                    }
                    WeightingLayer::NightlyDeweighting {
                        max_gap_days: layer.max_gap_days,
                        // The engine admits two published de-weighting
                        // laws; this ABI exposes only the current one
                        // (Vereš et al. 2017 §3, σ_eff = σ√(N/4) above
                        // N = 4). That is a *subset*, not a dropped
                        // request: `EmpyreanWeightingLayer` carries no
                        // scheme field, so there is no caller choice to
                        // discard. The pre-2017 √N law is retained
                        // upstream as a historical baseline only,
                        // reachable from the Rust engine API.
                        //
                        // Named directly now that `NightlyDeweighting` is
                        // re-exported from `empyrean_core::determination`.
                        // This is the same value the engine's own
                        // `Default` yields, but spelling the variant makes
                        // the selected law visible at the call site and
                        // fails to compile if the default ever changes;
                        // `weighting_identity_tests` pins it with a direct
                        // variant-equality assertion.
                        scheme: NightlyDeweighting::VFCC2017,
                    }
                }
                other => {
                    return Err(format!(
                        "unsupported weighting layer kind = {other} (expected OBSERVATORY_RULE = {} / NIGHTLY_DEWEIGHTING = {})",
                        EMPYREAN_WEIGHTING_LAYER_OBSERVATORY_RULE,
                        EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING,
                    ));
                }
            };
            user_layers.push(parsed);
        }
        // User layers must be able to override preset rules. scott's
        // sigma resolution is first-match-wins over the layer chain and
        // the preset rules are time-unbounded, so a user rule placed
        // AFTER the preset could never win its station. Insert the user
        // layers ahead of the preset chain (preserving their relative
        // order): user rules take their stations, the preset remains
        // the fallback. Only sigma-rule precedence changes — weight
        // *scale* factors and NightlyDeweighting are multiplicative
        // passes applied from every matching layer regardless of
        // position in the chain.
        user_layers.append(&mut wcfg.layers);
        wcfg.layers = user_layers;
    }

    // Duplicate nightly layers compound: each pass multiplies the
    // weights by another 1/sqrt(N) per night, which no production
    // scheme intends. Reject rather than silently over-de-weight.
    let nightly_count = wcfg
        .layers
        .iter()
        .filter(|l| matches!(l, WeightingLayer::NightlyDeweighting { .. }))
        .count();
    if nightly_count > 1 {
        return Err(format!(
            "weighting layer chain contains {nightly_count} NIGHTLY_DEWEIGHTING layers; each \
             additional pass compounds the per-night 1/sqrt(N) de-weighting multiplicatively — \
             include exactly one"
        ));
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

/// Build a scott [`ODConfig`] from the C request struct.
///
/// **The single OD-config parser in the C ABI.** Every entry point that
/// accepts an [`EmpyreanODConfig`] — the one-shot `determine` /
/// `evaluate` / `refine` surfaces here and
/// [`empyrean_session_new`](crate::session::empyrean_session_new) — goes
/// through this function, so the weighting chain
/// ([`build_weighting_from_c`]), the debiasing decision
/// ([`build_debiasing_from_c`]) and every other field resolve
/// identically on all of them. A second, partial parser would let a
/// change to the weighting contract land on one surface and not the
/// other, and would silently drop whatever it forgot to read — do not
/// add one.
pub(crate) fn build_od_config_from_c(c: &EmpyreanODConfig) -> Result<ODConfig, String> {
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
    // Tri-state: negative keeps the engine default, so a zero-initialized
    // struct is NOT read as "forbid truncation" / "disable the co-orbital
    // lane". Both knobs pick a documented behaviour, never a silent one.
    if c.allow_arc_truncation >= 0 {
        cfg.allow_arc_truncation = c.allow_arc_truncation != 0;
    }
    if c.coorbital_enabled >= 0 {
        cfg.coorbital.enabled = c.coorbital_enabled != 0;
    }
    // `radar_annealing` and `linear_cache` are solver policy, not fit
    // definition: they change how the escalation anneals radar σ and
    // whether trial evaluations are memoized, never what is delivered.
    // They stay at their engine defaults with no C-side knob — reach for
    // the empyrean-core Rust API to tune them.

    cfg.solve_for = if c.solve_for == EMPYREAN_SOLVE_FOR_EXPLICIT {
        // Explicit multi-axis request — the coarse code can't name it, so
        // read the per-axis disposition struct.
        SolveForParams::Explicit(solve_for_from_c(&c.solve_for_flags)?)
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
    // Attaches the 6×6, the state↔Marsden border and the wide carrier
    // together, converting all three from the caller's degrees in one
    // step. The OD paths consume the joint as their prior, so a carrier
    // dropped here would silently substitute a block-diagonal prior for
    // the correlated one the caller supplied.
    crate::joint::push_orbit_with_joint(&mut out, id.to_string(), coords, orbit)
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

/// The whole batch ran but **no** object produced a fit.
///
/// `results_out` is still fully populated — every slot carries its
/// per-object `error` / `error_code` — and MUST be released with
/// [`empyrean_determine_results_free`]. This is a distinct code from the
/// batch-level abort (`-3`), which writes nothing.
pub const EMPYREAN_DETERMINE_NONE_DELIVERED: i32 = -4;

/// Run the full orbit determination pipeline over every object in
/// `observations`.
///
/// The observations are grouped by ADES object identifier (permID /
/// provID / trkSub) and each group is fitted independently, so one call
/// determines a whole batch. `results_out` receives one
/// [`EmpyreanODObjectResult`] per group, ordered by `object_id`.
///
/// When `num_initial_orbits > 0`, the supplied orbits are used as DC
/// seeds (one per ADES object_id encountered in `observations`,
/// matched by orbit index). Pass `null, 0` to let the IOD pipeline
/// produce its own seeds. A seed that matches no group is reported in
/// [`EmpyreanDetermineResults::unmatched_orbit_ids`], never dropped.
///
/// # Return codes
///
/// - `0` — the batch ran and **at least one** object delivered a fit.
///   Individual failures do not abort the batch; check each slot's
///   `delivered` flag. `results_out` is populated.
/// - [`EMPYREAN_DETERMINE_NONE_DELIVERED`] (`-4`) — the batch ran but
///   every object failed. `results_out` IS populated with the per-object
///   errors and must still be freed.
/// - `-1` — null pointer or malformed input; nothing is written.
/// - `-3` — a batch-level failure (an unparseable weighting config, an
///   observation row with no identifier at all) aborted the run before
///   any object was fitted; nothing is written.
///
/// A single-object input is not a special case: it produces a
/// one-row table.
///
/// # Ownership
///
/// On `0` and `-4`, release `results_out` with
/// [`empyrean_determine_results_free`]. On `-1` / `-3` there is nothing
/// to free.
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
    results_out: *mut EmpyreanDetermineResults,
) -> i32 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if ctx.is_null() || observations.is_null() || config.is_null() || results_out.is_null() {
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

        // Batch-level failure (a row with no permID/provID/trkSub, an invalid
        // shared weighting config) aborts the whole call rather than producing
        // per-object results — surface it with its own message instead of
        // reporting the generic "produced no results".
        let determine_results = determine(
            ctx_ref,
            Observations::new(obs_vec, radar_vec),
            initial_map.as_ref(),
            &cfg,
            None,
        );

        // Sort by object_id so the table is a deterministic function of the
        // observation set, not of the order its rows happened to arrive in.
        let mut order: Vec<usize> = (0..determine_results.len()).collect();
        let ids = determine_results.orbit_ids();
        order.sort_by(|&a, &b| ids[a].cmp(&ids[b]));

        let mut slots: Vec<EmpyreanODObjectResult> = Vec::with_capacity(order.len());
        let mut delivered_count = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for i in order {
            let object_id = &ids[i];
            let slot = match &determine_results.results()[i] {
                Ok(det) => {
                    // A fitted orbit whose state cannot be expressed as a
                    // Cartesian record is a delivery failure for THIS object,
                    // not for the batch.
                    match od_orbit_to_propagated(&det.od.orbit, &det.od.covariance) {
                        Ok(prop_state) => {
                            let mut result = poisoned_od_result();
                            result.orbit = prop_state;
                            // Marshaling is fallible (the joint's carrier
                            // arrays are heap-allocated), and a result
                            // that cannot carry its joint is a delivery
                            // failure for THIS object — reported like any
                            // other, never delivered a block short.
                            match unsafe {
                                write_od_result_fields(&mut result, &det.od, Some(object_id))
                            } {
                                Ok(()) => {
                                    delivered_count += 1;
                                    EmpyreanODObjectResult {
                                        object_id: alloc_cstring(object_id),
                                        delivered: 1,
                                        result,
                                        error: std::ptr::null_mut(),
                                        error_code: EMPYREAN_OD_FAILURE_NONE,
                                    }
                                }
                                Err(e) => {
                                    // Release whatever the partial write
                                    // did allocate before discarding it.
                                    unsafe { free_od_result_fields(&mut result) };
                                    failures.push(format!("{object_id}: {e}"));
                                    EmpyreanODObjectResult {
                                        object_id: alloc_cstring(object_id),
                                        delivered: 0,
                                        result: poisoned_od_result(),
                                        error: alloc_cstring(&e),
                                        error_code: EMPYREAN_OD_FAILURE_OD,
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            failures.push(format!("{object_id}: {e}"));
                            EmpyreanODObjectResult {
                                object_id: alloc_cstring(object_id),
                                delivered: 0,
                                result: poisoned_od_result(),
                                error: alloc_cstring(&e),
                                error_code: EMPYREAN_OD_FAILURE_OD,
                            }
                        }
                    }
                }
                Err(e) => {
                    let message = e.to_string();
                    failures.push(format!("{object_id}: {message}"));
                    EmpyreanODObjectResult {
                        object_id: alloc_cstring(object_id),
                        delivered: 0,
                        result: poisoned_od_result(),
                        error: alloc_cstring(&message),
                        error_code: determine_error_code(e),
                    }
                }
            };
            slots.push(slot);
        }

        let (objects_ptr, num_objects) = object_results_to_c(slots);
        let (unmatched_ptr, num_unmatched) =
            string_vec_to_c(determine_results.unmatched_orbit_keys());
        unsafe {
            (*results_out).objects = objects_ptr;
            (*results_out).num_objects = num_objects;
            (*results_out).unmatched_orbit_ids = unmatched_ptr;
            (*results_out).num_unmatched_orbit_ids = num_unmatched;
        }

        if delivered_count == 0 {
            // Zero delivered is its own overall error, but the populated
            // table is the diagnosis — the caller still frees it.
            set_last_error(&format!(
                "orbit determination delivered no orbits ({} object(s) attempted): {}",
                num_objects,
                if failures.is_empty() {
                    "no observations to group".to_string()
                } else {
                    failures.join("; ")
                }
            ));
            return EMPYREAN_DETERMINE_NONE_DELIVERED;
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

/// Move per-object slots into an owned C array.
fn object_results_to_c(slots: Vec<EmpyreanODObjectResult>) -> (*mut EmpyreanODObjectResult, usize) {
    let n = slots.len();
    if n == 0 {
        return (std::ptr::null_mut(), 0);
    }
    let layout = std::alloc::Layout::array::<EmpyreanODObjectResult>(n)
        .unwrap_or(std::alloc::Layout::new::<EmpyreanODObjectResult>());
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanODObjectResult;
    if ptr.is_null() {
        return (std::ptr::null_mut(), 0);
    }
    for (i, slot) in slots.into_iter().enumerate() {
        unsafe { ptr.add(i).write(slot) };
    }
    (ptr, n)
}

/// Release the owned allocations hanging off one [`EmpyreanODResult`].
///
/// Shared by [`empyrean_od_result_free`] and
/// [`empyrean_determine_results_free`] so a slot inside a batch table is
/// freed exactly the way a standalone result is.
pub(crate) unsafe fn free_od_result_fields(result: *mut EmpyreanODResult) {
    let res = unsafe { &*result };
    let n = res.num_observations;
    let sb_n = res.num_station_biases;
    unsafe {
        free_observation_results(res.observations, n);
        free_station_biases(res.station_biases, sb_n);
        // Photometry owned arrays (null / 0 when no photometry ran).
        free_band_stats(res.photometry.per_band, res.photometry.num_per_band);
        free_gate_records(res.photometry.gates, res.photometry.num_gates);
        free_string_array(
            res.photometry.dropped_bands,
            res.photometry.num_dropped_bands,
        );
        free_cstring(res.trust_event_body);
        free_string_array(res.warnings, res.num_warnings);
        // The joint posterior's carrier arrays are library-owned on the
        // result (the mirror-image of the same fields on an input
        // orbit), so they are released here with everything else.
        crate::joint::free_orbit_covariance(&mut (*result).orbit.orbit_cov);
        (*result).observations = std::ptr::null_mut();
        (*result).num_observations = 0;
        (*result).station_biases = std::ptr::null_mut();
        (*result).num_station_biases = 0;
        (*result).photometry.per_band = std::ptr::null_mut();
        (*result).photometry.num_per_band = 0;
        (*result).photometry.gates = std::ptr::null_mut();
        (*result).photometry.num_gates = 0;
        (*result).photometry.dropped_bands = std::ptr::null_mut();
        (*result).photometry.num_dropped_bands = 0;
        (*result).trust_event_body = std::ptr::null_mut();
        (*result).warnings = std::ptr::null_mut();
        (*result).num_warnings = 0;
    }
}

/// Free a batch result table previously written by
/// `empyrean_determine()`.
///
/// Releases every per-object slot (including the fits inside the
/// delivered ones), the per-object identifier / error strings, and the
/// unmatched-seed list. Safe to call on a table returned with
/// [`EMPYREAN_DETERMINE_NONE_DELIVERED`], and idempotent — the table is
/// left empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_determine_results_free(results: *mut EmpyreanDetermineResults) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if results.is_null() {
            return;
        }
        let res = unsafe { &*results };
        let n = res.num_objects;
        if !res.objects.is_null() && n > 0 {
            for i in 0..n {
                let slot = unsafe { &mut *res.objects.add(i) };
                unsafe {
                    free_od_result_fields(&mut slot.result);
                    free_cstring(slot.object_id);
                    free_cstring(slot.error);
                }
                slot.object_id = std::ptr::null_mut();
                slot.error = std::ptr::null_mut();
            }
            let layout = std::alloc::Layout::array::<EmpyreanODObjectResult>(n)
                .unwrap_or(std::alloc::Layout::new::<EmpyreanODObjectResult>());
            unsafe {
                std::alloc::dealloc(res.objects as *mut u8, layout);
            }
        }
        unsafe {
            free_string_array(res.unmatched_orbit_ids, res.num_unmatched_orbit_ids);
            (*results).objects = std::ptr::null_mut();
            (*results).num_objects = 0;
            (*results).unmatched_orbit_ids = std::ptr::null_mut();
            (*results).num_unmatched_orbit_ids = 0;
        }
    }));
}

/// Free an OD result previously returned by `empyrean_refine()`.
///
/// Batch `empyrean_determine()` results are released with
/// [`empyrean_determine_results_free`] instead.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn empyrean_od_result_free(result: *mut EmpyreanODResult) {
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        if result.is_null() {
            return;
        }
        unsafe { free_od_result_fields(result) };
    }));
}

// ── empyrean_evaluate ───────────────────────────────────────

/// Evaluate residuals for a single orbit against observations.
///
/// # A supplied joint covariance changes nothing here
///
/// Evaluation measures how well a FIXED orbit predicts observations; it
/// forms no prior and performs no estimation, so an orbit carrying a
/// state↔Marsden border or a wide carrier scores exactly as the same
/// orbit without one. Nothing is dropped — there is simply nothing for
/// the joint to affect, and this result type carries no orbit to echo
/// one back on. The nine other orbit-reading entry points consume it.
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

        // Single-orbit evaluate: no ADES grouping key, so the rows carry a
        // null object_id (see EmpyreanObservationResult::object_id).
        let (obs_ptr, obs_n) = observation_results_to_c(&eval_result.observations, None);
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

        unsafe {
            (*result_out).orbit = prop_state;
            // Single-orbit refine: the caller supplied the orbit, so the
            // rows carry a null object_id.
            if let Err(e) = write_od_result_fields(result_out, &od_result, None) {
                free_od_result_fields(result_out);
                set_last_error(&e);
                return -3;
            }
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
pub(crate) mod tests {
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
            rejection_reason_to_c(&RejectionReason::CMC2003),
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
                has_non_grav_cross: 0,
                non_grav_cross: [[0.0; 3]; 6],
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
            state_param_cross: std::ptr::null(),
            n_state_param_cross: 0,
            param_pair_cross: std::ptr::null(),
            n_param_pair_cross: 0,
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
    pub(super) fn try_context() -> Option<EmpyreanContext> {
        empyrean_core::Context::from_data_dir(None).ok()
    }

    /// Parse the bundled Eros ADES fixture into a freshly-allocated C
    /// `EmpyreanObservation` array via the real `empyrean_read_ades` ABI entry
    /// point (the same path a C caller uses). Returns the pointer + count;
    /// the caller frees with `empyrean_observations_free`.
    pub(crate) fn read_eros_observations() -> (*mut EmpyreanObservation, usize) {
        let psv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/433_eros.psv");
        let content = std::fs::read_to_string(psv).expect("read bundled Eros fixture");
        read_eros_observations_from(&content)
    }

    /// The same parse, over caller-supplied ADES content — used by the
    /// join-back test, which composes a two-object batch out of the same
    /// bundled arc.
    pub(super) fn read_eros_observations_from(content: &str) -> (*mut EmpyreanObservation, usize) {
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
    pub(crate) fn standard_od_config() -> EmpyreanODConfig {
        let mut cfg: EmpyreanODConfig = unsafe { std::mem::zeroed() };
        cfg.force_model = 2; // Standard tier
        cfg
    }

    /// Reconstruct a re-feedable `EmpyreanOrbit` from a fitted
    /// `EmpyreanPropagatedState` (the determine/refine output orbit). Mirrors
    /// the output→input re-feed a real caller performs: flatten the propagated
    /// Cartesian state + 6×6 covariance back into an input orbit.
    pub(super) fn refeed_orbit(p: &EmpyreanPropagatedState) -> EmpyreanOrbit {
        EmpyreanOrbit {
            state: crate::CoordinateState {
                epoch_mjd_tdb: p.epoch_mjd_tdb,
                elements: [p.x, p.y, p.z, p.vx, p.vy, p.vz],
                covariance: p.covariance,
                has_covariance: p.has_covariance,
                representation: EMPYREAN_REPRESENTATION_CARTESIAN,
                frame: p.frame,
                origin: p.origin,
                has_non_grav_cross: 0,
                non_grav_cross: [[0.0; 3]; 6],
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
            state_param_cross: std::ptr::null(),
            n_state_param_cross: 0,
            param_pair_cross: std::ptr::null(),
            n_param_pair_cross: 0,
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
        // determine is batch-first: one slot per ADES object. The Eros
        // arc is one object, so the table has exactly one row.
        let mut det_results: EmpyreanDetermineResults = unsafe { std::mem::zeroed() };
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
                &mut det_results,
            )
        };
        assert_eq!(
            det_code,
            0,
            "empyrean_determine must fit Eros (code {det_code}, last_error: {})",
            unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy()
        );
        assert_eq!(
            det_results.num_objects, 1,
            "the single-object Eros arc must produce a one-row table"
        );
        assert_eq!(
            det_results.num_unmatched_orbit_ids, 0,
            "no seeds were supplied, so nothing can be unmatched"
        );
        let slot = unsafe { &*det_results.objects };
        assert_eq!(
            slot.delivered,
            1,
            "Eros must deliver (error: {})",
            unsafe { CStr::from_ptr(slot.error) }.to_string_lossy()
        );
        assert_eq!(slot.error_code, EMPYREAN_OD_FAILURE_NONE);
        assert!(slot.error.is_null(), "a delivered slot carries no error");
        let object_id = unsafe { CStr::from_ptr(slot.object_id) }.to_string_lossy();
        assert!(
            !object_id.is_empty(),
            "every slot is keyed by its ADES identifier"
        );
        let od_result = &slot.result;
        assert!(
            od_result.num_observations > 0,
            "determine must report fitted observations"
        );
        assert_eq!(od_result.converged, 1, "Eros DC must converge");
        assert_eq!(
            od_result.orbit.has_covariance, 1,
            "fitted orbit must carry covariance for refine's prior"
        );
        // Every residual row is attributable to the object it was fit against.
        let first_row = unsafe { &*od_result.observations };
        assert_eq!(
            unsafe { CStr::from_ptr(first_row.object_id) }.to_string_lossy(),
            object_id,
            "each residual row carries its own object_id"
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
            empyrean_determine_results_free(&mut det_results);
            empyrean_observations_free(obs_ptr, obs_n);
        }
    }
}

/// The weighting-config identity fields scott grew (`floors_table` on the
/// config, `scheme` on a nightly layer) and what this ABI resolves them to.
///
/// `build_weighting_from_c` names both directly (`FloorsTable::Custom`,
/// `NightlyDeweighting::VFCC2017`) now that both are re-exported from
/// `empyrean_core::determination`. These pin the resolved identities: the
/// identity a delivered fit reports must be `Custom` for a hand-assembled
/// chain, and a `NIGHTLY_DEWEIGHTING` layer must apply the current published
/// law, not the pre-2017 one. Asserted by direct variant equality against
/// the named types — a compile-time failure if a field's type changes to
/// another enum with a same-named variant, which `Debug`-string pinning
/// could not catch.
#[cfg(test)]
mod weighting_identity_tests {
    use super::*;
    use empyrean_core::determination::{FloorsTable, NightlyDeweighting, WeightingLayer};

    fn enabled_config(preset: u8) -> EmpyreanWeightingConfig {
        let mut c: EmpyreanWeightingConfig = unsafe { std::mem::zeroed() };
        c.enabled = 1;
        c.preset = preset;
        c.sigma_policy = -1;
        c
    }

    /// A `preset = NONE` chain is caller-assembled, so it carries no
    /// published floors-table identity. Reporting one would let a config
    /// that merely resembles a published table be delivered as being it.
    #[test]
    fn a_hand_assembled_chain_reports_a_custom_floors_table() {
        let cfg = build_weighting_from_c(&enabled_config(EMPYREAN_WEIGHTING_PRESET_NONE))
            .expect("preset NONE converts")
            .expect("weighting is enabled");
        assert_eq!(
            cfg.floors_table,
            FloorsTable::Custom,
            "a hand-assembled layer chain must not claim a published table"
        );
    }

    /// The VFC17 preset's own constructor records its identity; routing
    /// through this converter must not overwrite it with `Custom`.
    #[test]
    fn the_vfc17_preset_keeps_its_recorded_identity() {
        let cfg = build_weighting_from_c(&enabled_config(EMPYREAN_WEIGHTING_PRESET_VFCC2017))
            .expect("preset VFC17 converts")
            .expect("weighting is enabled");
        assert_eq!(
            cfg.floors_table,
            FloorsTable::VFCC2017,
            "the preset's recorded identity must survive the conversion"
        );
    }

    /// A `NIGHTLY_DEWEIGHTING` layer applies Vereš et al. (2017) §3
    /// (σ unchanged to N = 4, then σ√(N/4)). The engine still carries the
    /// pre-2017 σ√N law as a historical baseline, and it shipped as a
    /// library default once — this pins that the ABI does not select it.
    #[test]
    fn a_nightly_layer_applies_the_current_published_law() {
        let layers = [EmpyreanWeightingLayer {
            kind: EMPYREAN_WEIGHTING_LAYER_NIGHTLY_DEWEIGHTING,
            obs_code: [0; 4],
            sigma_ra_arcsec: 0.0,
            sigma_dec_arcsec: 0.0,
            start_epoch_mjd_tdb: f64::NAN,
            end_epoch_mjd_tdb: f64::NAN,
            scale: 0.0,
            max_gap_days: 0.5,
        }];
        let mut c = enabled_config(EMPYREAN_WEIGHTING_PRESET_NONE);
        c.num_additional_layers = layers.len();
        c.additional_layers = layers.as_ptr();

        let cfg = build_weighting_from_c(&c)
            .expect("a nightly layer converts")
            .expect("weighting is enabled");
        let (scheme, max_gap_days) = cfg
            .layers
            .iter()
            .find_map(|l| match l {
                WeightingLayer::NightlyDeweighting {
                    scheme,
                    max_gap_days,
                } => Some((*scheme, *max_gap_days)),
                _ => None,
            })
            .expect("the nightly layer reaches the engine config");
        assert_eq!(
            scheme,
            NightlyDeweighting::VFCC2017,
            "a nightly layer must apply the current published law, not the \
             pre-2017 sqrt(N) baseline"
        );
        assert_eq!(max_gap_days, 0.5, "max_gap_days must survive");
    }
}

#[cfg(test)]
mod batch_determine_tests {
    use super::*;

    /// Build the C-side per-observation record with the fields the free
    /// path walks. Everything else is zeroed — this fixture exists to
    /// exercise ownership, not numerics.
    fn observation_row(obs_id: &str, object_id: Option<&str>) -> EmpyreanObservationResult {
        let mut row: EmpyreanObservationResult = unsafe { std::mem::zeroed() };
        row.obs_id = alloc_cstring(obs_id);
        row.object_id = match object_id {
            Some(id) => alloc_cstring(id),
            None => std::ptr::null_mut(),
        };
        row.ast_cat = alloc_cstring("Gaia3");
        row
    }

    fn read_cstring(p: *mut c_char) -> Option<String> {
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    /// A failed object's slot must not look like a converged fit at the
    /// origin. Every f64 is NaN and every enumerated code is outside its
    /// own value set, so a caller that skips `delivered` cannot mistake
    /// the record for a result.
    #[test]
    fn poisoned_result_is_nan_not_zero() {
        let r = poisoned_od_result();
        assert!(r.update_norm.is_nan(), "update_norm must be NaN, not 0.0");
        assert!(r.orbit.x.is_nan() && r.orbit.vz.is_nan());
        assert!(r.orbit.epoch_mjd_tdb.is_nan());
        assert!(r.covariance.iter().flatten().all(|v| v.is_nan()));
        assert!(r.summary.reduced_chi2.is_nan() && r.summary.rms_ra_arcsec.is_nan());
        assert!(r.non_grav.a1.is_nan() && r.srp.amrat.is_nan());
        assert!(r.dt_delta.is_nan() && r.amrat_delta.is_nan());
        // Enumerated codes: -1 is not a member of any EMPYREAN_* set.
        assert_eq!(r.force_model_used, -1);
        assert_eq!(r.solve_for_used, -1);
        assert_eq!(r.covariance_representation, -1);
        assert_eq!(r.dv_frame, -1);
        assert_eq!(r.covariance_trust, -1);
        assert_eq!(r.trust_event_kind, -1);
        assert_eq!(r.orbit.origin, -1);
        assert_eq!(r.orbit.frame, -1);
        // Owned pointers null / counts zero so the free path is a no-op.
        assert!(r.observations.is_null() && r.num_observations == 0);
        assert!(r.station_biases.is_null() && r.num_station_biases == 0);
        assert!(r.photometry.per_band.is_null() && r.photometry.num_per_band == 0);
        assert!(r.photometry.gates.is_null() && r.photometry.num_gates == 0);
        assert!(r.photometry.dropped_bands.is_null());
        assert!(r.trust_event_body.is_null());
        // `converged` reads false, and no acceptability gate passes.
        assert_eq!(r.converged, 0);
        assert_eq!(r.acceptability.fit_acceptable, 0);
        assert_eq!(r.acceptability.extrapolation_acceptable, 0);
    }

    /// The `_ok` booleans stay readable on a poisoned slot even though
    /// every measurement is NaN — that is the documented ABI contract.
    #[test]
    fn poisoned_acceptability_keeps_ok_flags_valid() {
        let a = poisoned_acceptability_report();
        for ok in [
            a.converged_ok,
            a.reduced_chi2_ok,
            a.rms_ok,
            a.residual_isotropy_ok,
            a.covariance_ok,
            a.arc_coverage_ok,
            a.fractional_sigma_a_ok,
            a.selection_fraction_ok,
            a.selected_arc_coverage_ok,
            a.trailing_gap_ok,
        ] {
            assert!(ok == 0 || ok == 1, "an _ok flag must stay a valid bool");
        }
        for v in [
            a.reduced_chi2_value,
            a.rms_value_arcsec,
            a.at_ct_ratio_value,
            a.arc_days_value,
            a.fractional_sigma_a_value,
            a.selection_fraction_value,
            a.selected_arc_days_value,
            a.selected_arc_fraction_value,
            a.trailing_gap_days_value,
        ] {
            assert!(v.is_nan(), "a non-computable value must be NaN, not 0.0");
        }
        // "no radar" must never read as "radar failed".
        assert_eq!(a.radar_fit_ok, -1);
    }

    /// Every `DetermineError` variant gets its own code, so a consumer
    /// can branch on the cause without string-matching the message.
    #[test]
    fn failure_codes_are_distinct() {
        let codes = [
            EMPYREAN_OD_FAILURE_OBSERVATION_CONVERSION,
            EMPYREAN_OD_FAILURE_OBSERVER_CONSTRUCTION,
            EMPYREAN_OD_FAILURE_UNSUPPORTED_COORDINATE_SYSTEM,
            EMPYREAN_OD_FAILURE_EARTH_ORIENTATION_COVERAGE,
            EMPYREAN_OD_FAILURE_IOD,
            EMPYREAN_OD_FAILURE_OD,
            EMPYREAN_OD_FAILURE_DUPLICATE_OBS_IDS,
            EMPYREAN_OD_FAILURE_RADAR_ONLY,
            EMPYREAN_OD_FAILURE_NON_GRAV_NOT_RECOVERED,
        ];
        let mut seen: Vec<i32> = codes.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), codes.len(), "failure codes must be distinct");
        assert!(
            !codes.contains(&EMPYREAN_OD_FAILURE_NONE),
            "no failure may share the delivered code"
        );

        // Spot-check the classifier on the variants that are cheap to build.
        assert_eq!(
            determine_error_code(&DetermineError::RadarOnly { n_radar: 3 }),
            EMPYREAN_OD_FAILURE_RADAR_ONLY
        );
        assert_eq!(
            determine_error_code(&DetermineError::DuplicateObsIds(vec!["a".into()])),
            EMPYREAN_OD_FAILURE_DUPLICATE_OBS_IDS
        );
        assert_eq!(
            determine_error_code(&DetermineError::UnsupportedCoordinateSystem {
                obs_index: 0,
                sys: "WGS84".into(),
            }),
            EMPYREAN_OD_FAILURE_UNSUPPORTED_COORDINATE_SYSTEM
        );
        assert_eq!(
            determine_error_code(&DetermineError::EarthOrientationCoverageIncomplete {
                loaded: "none".into(),
                required: "historical+predict".into(),
            }),
            EMPYREAN_OD_FAILURE_EARTH_ORIENTATION_COVERAGE
        );
    }

    /// Freeing a table releases the per-object strings, the per-slot
    /// fits, and the unmatched-seed list — and leaves the table empty so
    /// a second free is a no-op rather than a double free.
    #[test]
    fn determine_results_free_releases_everything_and_is_idempotent() {
        // One delivered slot owning an observation array, one failed slot.
        let rows = vec![
            observation_row("obs-1", Some("2024 YR4")),
            observation_row("obs-2", Some("2024 YR4")),
        ];
        let n_rows = rows.len();
        let layout = std::alloc::Layout::array::<EmpyreanObservationResult>(n_rows).unwrap();
        let rows_ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservationResult;
        for (i, row) in rows.into_iter().enumerate() {
            unsafe { rows_ptr.add(i).write(row) };
        }

        let mut delivered = poisoned_od_result();
        delivered.observations = rows_ptr;
        delivered.num_observations = n_rows;
        delivered.trust_event_body = alloc_cstring("Earth");

        let slots = vec![
            EmpyreanODObjectResult {
                object_id: alloc_cstring("2024 YR4"),
                delivered: 1,
                result: delivered,
                error: std::ptr::null_mut(),
                error_code: EMPYREAN_OD_FAILURE_NONE,
            },
            EmpyreanODObjectResult {
                object_id: alloc_cstring("K25A00B"),
                delivered: 0,
                result: poisoned_od_result(),
                error: alloc_cstring("IOD failed: no viable seed"),
                error_code: EMPYREAN_OD_FAILURE_IOD,
            },
        ];
        let (objects, num_objects) = object_results_to_c(slots);
        let (unmatched, num_unmatched) =
            string_vec_to_c(&["seed-that-matched-nothing".to_string()]);
        let mut table = EmpyreanDetermineResults {
            objects,
            num_objects,
            unmatched_orbit_ids: unmatched,
            num_unmatched_orbit_ids: num_unmatched,
        };

        assert_eq!(table.num_objects, 2);
        let first = unsafe { &*table.objects };
        assert_eq!(read_cstring(first.object_id).as_deref(), Some("2024 YR4"));
        assert_eq!(first.delivered, 1);
        assert!(first.error.is_null());
        let second = unsafe { &*table.objects.add(1) };
        assert_eq!(second.delivered, 0);
        assert_eq!(second.error_code, EMPYREAN_OD_FAILURE_IOD);
        assert!(read_cstring(second.error).unwrap().contains("IOD failed"));
        // The per-row grouping key survives into the observation array.
        let row0 = unsafe { &*first.result.observations };
        assert_eq!(read_cstring(row0.object_id).as_deref(), Some("2024 YR4"));

        unsafe { empyrean_determine_results_free(&mut table) };
        assert!(table.objects.is_null());
        assert_eq!(table.num_objects, 0);
        assert!(table.unmatched_orbit_ids.is_null());
        assert_eq!(table.num_unmatched_orbit_ids, 0);

        // Idempotent: the emptied table frees again without touching
        // anything that was already released.
        unsafe { empyrean_determine_results_free(&mut table) };
        assert!(table.objects.is_null());

        // Null is accepted.
        unsafe { empyrean_determine_results_free(std::ptr::null_mut()) };
    }

    /// A row from the single-object paths carries a null `object_id`;
    /// freeing must handle both shapes.
    #[test]
    fn observation_rows_free_with_and_without_object_id() {
        for object_id in [Some("2024 YR4"), None] {
            let n = 1usize;
            let layout = std::alloc::Layout::array::<EmpyreanObservationResult>(n).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) } as *mut EmpyreanObservationResult;
            unsafe { ptr.write(observation_row("obs-1", object_id)) };
            assert_eq!(
                read_cstring(unsafe { &*ptr }.object_id).as_deref(),
                object_id
            );
            unsafe { free_observation_results(ptr, n) };
        }
    }

    /// An empty batch produces an empty table, not a null-pointer walk.
    #[test]
    fn empty_batch_frees_cleanly() {
        let (objects, num_objects) = object_results_to_c(Vec::new());
        assert!(objects.is_null() && num_objects == 0);
        let mut table = EmpyreanDetermineResults {
            objects,
            num_objects,
            unmatched_orbit_ids: std::ptr::null_mut(),
            num_unmatched_orbit_ids: 0,
        };
        unsafe { empyrean_determine_results_free(&mut table) };
        assert_eq!(table.num_objects, 0);
    }

    /// The zero-delivered return code is distinct from both success and
    /// the batch-level abort, because it is the one case where the
    /// caller must still free a populated table.
    #[test]
    fn none_delivered_code_is_its_own_signal() {
        assert_ne!(EMPYREAN_DETERMINE_NONE_DELIVERED, 0);
        assert_ne!(EMPYREAN_DETERMINE_NONE_DELIVERED, -1);
        assert_ne!(EMPYREAN_DETERMINE_NONE_DELIVERED, -3);
    }
}

#[cfg(test)]
mod residual_join_back_tests {
    use super::tests::{read_eros_observations_from, standard_od_config, try_context};
    use super::*;

    /// The bundled single-object arc, plus a copy of it filed under a
    /// second designation.
    ///
    /// Two objects out of one real arc: no new fixture, and the fits are
    /// necessarily identical, which makes "the rows did not get mixed up
    /// between objects" checkable rather than merely plausible. The
    /// `obsID` column is rewritten on the copy so the join key stays
    /// unique across the batch, exactly as it would be for two genuinely
    /// different objects.
    fn two_object_eros_psv() -> String {
        let psv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/433_eros.psv");
        let content = std::fs::read_to_string(psv).expect("read bundled Eros fixture");
        let mut lines = content.lines();
        let version = lines.next().expect("version line");
        let header = lines.next().expect("header line");
        let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();

        let mut out = format!("{version}\n{header}\n");
        for row in &rows {
            out.push_str(row);
            out.push('\n');
        }
        for row in &rows {
            // permID 433 -> 4330, and obsID gets a suffix so the join key
            // is unique batch-wide.
            let mut fields: Vec<String> = row.split('|').map(|s| s.to_string()).collect();
            fields[0] = "4330".to_string();
            fields[3] = format!("{}X", fields[3]);
            out.push_str(&fields.join("|"));
            out.push('\n');
        }
        out
    }

    /// Residuals written by a multi-object fit join back to the
    /// observations they came from, on `(object_id, obs_id)`, with the
    /// rejection attribution intact.
    ///
    /// This is the property a five-column residual file could not
    /// support: without `obs_id` there is no join key at all, and
    /// without `object_id` a batch's rows are unattributable.
    #[test]
    fn residuals_join_back_to_their_observations() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping: no engine data directory available");
            return;
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;

        let (obs_ptr, obs_n) = read_eros_observations_from(&two_object_eros_psv());
        let cfg = standard_od_config();

        let mut results: EmpyreanDetermineResults = unsafe { std::mem::zeroed() };
        let code = unsafe {
            empyrean_determine(
                ctx_ptr,
                obs_ptr,
                obs_n,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &cfg,
                &mut results,
            )
        };
        assert_eq!(
            code,
            0,
            "the two-object batch must deliver; last_error: {}",
            unsafe { CStr::from_ptr(crate::empyrean_last_error()) }.to_string_lossy()
        );
        assert_eq!(results.num_objects, 2, "one slot per designation");

        // The input join keys, per object.
        let input: std::collections::HashSet<String> = (0..obs_n)
            .map(|i| {
                let o = unsafe { &*obs_ptr.add(i) };
                let id = |p: *mut c_char| -> String {
                    if p.is_null() {
                        String::new()
                    } else {
                        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
                    }
                };
                let object = {
                    let perm = id(o.perm_id);
                    if perm.is_empty() {
                        let prov = id(o.prov_id);
                        if prov.is_empty() { id(o.trk_sub) } else { prov }
                    } else {
                        perm
                    }
                };
                format!("{object}\u{1}{}", id(o.obs_id))
            })
            .collect();

        // Flatten every delivered object's rows into one table, which is
        // exactly what the CLI writes.
        let mut flat: Vec<EmpyreanObservationResult> = Vec::new();
        let slots = unsafe { std::slice::from_raw_parts(results.objects, results.num_objects) };
        let mut delivered = 0usize;
        for slot in slots {
            if slot.delivered == 0 {
                continue;
            }
            delivered += 1;
            let rows = unsafe {
                std::slice::from_raw_parts(slot.result.observations, slot.result.num_observations)
            };
            for r in rows {
                // Shallow copy: the writer only reads, and the originals
                // stay owned by the batch table.
                flat.push(unsafe { std::ptr::read(r) });
            }
        }
        assert_eq!(delivered, 2, "both designations must deliver");
        assert!(!flat.is_empty(), "a delivered fit reports its residuals");

        let dir = std::env::temp_dir().join(format!("empyrean-join-back-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("residuals.csv");
        let c_path = CString::new(path.display().to_string()).unwrap();
        let wcode = unsafe {
            crate::io::empyrean_residuals_write_csv(c_path.as_ptr(), flat.as_ptr(), flat.len())
        };
        assert_eq!(wcode, 0, "residual CSV write must succeed");

        // Read back and join.
        let text = std::fs::read_to_string(&path).expect("read residuals.csv");
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().expect("header").split(',').collect();
        let col = |name: &str| header.iter().position(|h| *h == name).expect(name);
        let (i_object, i_obs, i_reason) =
            (col("object_id"), col("obs_id"), col("rejection_reason"));

        let known_reasons = [
            "accepted",
            "chi_squared",
            "sigma_clip",
            "cooks_distance",
            "adaptive",
            "unsupported_observatory",
            "cmc2003",
            "radar_observations_unsupported",
            "occultation_observations_unsupported",
            "outside_arc",
            "non_finite_chi2",
            "missing_jacobian",
            "spacecraft_kernel_missing",
            "observer_construction_failed",
            "never_absorbed",
            "not_evaluated",
        ];

        let mut seen_objects: std::collections::HashSet<String> = Default::default();
        let mut n_rows = 0usize;
        for line in lines.filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            let object = f[i_object];
            let obs = f[i_obs];
            let reason = f[i_reason];
            n_rows += 1;
            seen_objects.insert(object.to_string());
            assert!(
                input.contains(&format!("{object}\u{1}{obs}")),
                "residual row ({object}, {obs}) joins back to no input observation"
            );
            assert!(
                known_reasons.contains(&reason),
                "rejection attribution must survive the trip, got {reason:?}"
            );
        }
        assert_eq!(n_rows, flat.len(), "every residual row reached the file");
        assert_eq!(
            seen_objects.len(),
            2,
            "both designations appear in the flat table: {seen_objects:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
        unsafe {
            empyrean_determine_results_free(&mut results);
            empyrean_observations_free(obs_ptr, obs_n);
        }
    }
}

/// The parameter partition at the boundary: strict tri-state
/// validation, the disposition echo, and the per-declared-segment thrust
/// index space.
#[cfg(test)]
mod parameter_partition_tests {
    use super::*;

    fn flags(
        marsden: u8,
        dt: u8,
        amrat: u8,
        thrust: [u8; EMPYREAN_MAX_THRUST_SEGMENTS],
    ) -> EmpyreanSolveFor {
        EmpyreanSolveFor {
            marsden,
            dt,
            amrat,
            thrust_dispositions: thrust,
        }
    }

    /// `memset(0)` is every axis FIXED — exactly what a `false` flag
    /// always meant. This is what makes the semantic widening safe in
    /// the old-caller direction.
    #[test]
    fn a_zero_init_solve_for_is_every_axis_fixed() {
        let f = EmpyreanSolveFor::default();
        let s = solve_for_from_c(&f).expect("a zero-init struct is well-formed");
        assert_eq!(s.marsden, ParamDisposition::Fixed);
        assert_eq!(s.dt, ParamDisposition::Fixed);
        assert_eq!(s.amrat, ParamDisposition::Fixed);
        assert!(s.thrust.iter().all(|d| *d == ParamDisposition::Fixed));
        assert_eq!(s.solved_thrust_segments(), 0);
    }

    /// `1` still means solved, on every axis. An older caller's values
    /// are unchanged by the widening.
    #[test]
    fn one_still_means_solved_on_every_axis() {
        let s = solve_for_from_c(&flags(1, 1, 1, [1, 1, 1])).expect("all-solved is legal");
        assert_eq!(s.marsden, ParamDisposition::Solved);
        assert_eq!(s.dt, ParamDisposition::Solved);
        assert_eq!(s.amrat, ParamDisposition::Solved);
        assert_eq!(s.solved_thrust_segments(), 3);
    }

    /// `2` is considered, and it is a distinct third state rather than a
    /// synonym for solved. A library that read the byte as a bare
    /// non-zero test would silently SOLVE a considered axis — a wider
    /// solved set, a different fitted answer, and no error anywhere.
    #[test]
    fn two_is_considered_and_is_not_solved() {
        let s = solve_for_from_c(&flags(2, 0, 2, [0, 2, 0])).expect("considered is legal");
        assert_eq!(s.marsden, ParamDisposition::Considered);
        assert_eq!(s.amrat, ParamDisposition::Considered);
        assert!(!s.marsden.is_solved(), "considered must not read as solved");
        assert_eq!(s.solved_thrust_segments(), 0);
        assert_eq!(s.considered_thrust_segments(), 1);
    }

    /// Any value outside `0 | 1 | 2` is refused by name and value. This
    /// is what makes a FUTURE fourth disposition fail loudly against
    /// this release instead of degrading into one of the three it knows.
    #[test]
    fn an_unknown_disposition_is_refused_by_name_and_value() {
        for (f, field) in [
            (flags(3, 0, 0, [0; 3]), "marsden"),
            (flags(0, 7, 0, [0; 3]), "dt"),
            (flags(0, 0, 255, [0; 3]), "amrat"),
            (flags(0, 0, 0, [0, 4, 0]), "thrust_dispositions[1]"),
        ] {
            let err = solve_for_from_c(&f).expect_err("an unknown disposition is refused");
            assert!(err.contains(field), "the error names the field: {err}");
            assert!(
                err.contains("fixed") && err.contains("solved") && err.contains("considered"),
                "the error lists the legal set: {err}"
            );
        }
    }

    /// The disposition round trip: what a caller writes is what the
    /// result echoes back, including WHICH thrust segment carries which
    /// disposition. A count could not express the middle-segment case
    /// this asserts.
    #[test]
    fn the_disposition_echo_names_which_segment_is_which() {
        // Three declared burns; only the MIDDLE one solved, the first
        // considered, the last fixed.
        let supplied = flags(1, 0, 2, [2, 1, 0]);
        let engine = solve_for_from_c(&supplied).expect("legal");
        assert_eq!(engine.thrust[0], ParamDisposition::Considered);
        assert_eq!(engine.thrust[1], ParamDisposition::Solved);
        assert_eq!(engine.thrust[2], ParamDisposition::Fixed);
        assert_eq!(engine.solved_thrust_segments(), 1);

        let echoed = solve_for_to_c(&engine);
        assert_eq!(echoed.marsden, supplied.marsden);
        assert_eq!(echoed.dt, supplied.dt);
        assert_eq!(echoed.amrat, supplied.amrat);
        assert_eq!(
            echoed.thrust_dispositions, supplied.thrust_dispositions,
            "the echo must preserve per-segment identity, not just the counts"
        );
    }

    /// The coarse `EMPYREAN_SOLVE_FOR_*` code names a SOLVED set, so a
    /// fit that CONSIDERS an axis must not be reported as the coarse
    /// code its solved set alone suggests. A considered axis contributes
    /// to the delivered σ, and reporting such a fit as `STATE_ONLY`
    /// would hide that from every consumer that branches on the code.
    #[test]
    fn a_considered_axis_disqualifies_the_coarse_codes() {
        let state_only =
            SolveForParams::Explicit(solve_for_from_c(&flags(0, 0, 0, [0; 3])).unwrap());
        assert_eq!(solve_for_to_int(&state_only), EMPYREAN_SOLVE_FOR_STATE_ONLY);

        let marsden_only =
            SolveForParams::Explicit(solve_for_from_c(&flags(1, 0, 0, [0; 3])).unwrap());
        assert_eq!(
            solve_for_to_int(&marsden_only),
            EMPYREAN_SOLVE_FOR_STATE_AND_NONGRAV
        );

        // The discriminating pair: identical SOLVED sets, one with a
        // considered axis. They must not report the same code.
        let considered_amrat =
            SolveForParams::Explicit(solve_for_from_c(&flags(0, 0, 2, [0; 3])).unwrap());
        assert_eq!(
            solve_for_to_int(&considered_amrat),
            EMPYREAN_SOLVE_FOR_EXPLICIT,
            "a fit considering AMRAT solves nothing extra but reports a wider sigma; \
             it is not the state-only fit STATE_ONLY names"
        );

        let considered_thrust =
            SolveForParams::Explicit(solve_for_from_c(&flags(1, 0, 0, [0, 2, 0])).unwrap());
        assert_eq!(
            solve_for_to_int(&considered_thrust),
            EMPYREAN_SOLVE_FOR_EXPLICIT,
            "a considered burn disqualifies STATE_AND_NONGRAV for the same reason"
        );
    }

    /// EXPLICIT without the disposition struct is refused, and the
    /// message names the field to set rather than the field that used to
    /// exist.
    #[test]
    fn explicit_without_the_disposition_struct_is_refused() {
        let err = int_to_solve_for(EMPYREAN_SOLVE_FOR_EXPLICIT)
            .expect_err("EXPLICIT needs the per-axis struct");
        assert!(err.contains("thrust_dispositions"), "{err}");
        assert!(
            !err.contains("thrust_segments)"),
            "the message must not name the removed count field: {err}"
        );
    }

    /// Every rejection reason maps to a distinct code, including the one
    /// added in this release. A collision would merge two causes in the
    /// attribution census — which is the hole the reserved-code
    /// discipline exists to keep closed.
    #[test]
    fn every_rejection_code_is_distinct_and_contiguous() {
        let codes = [
            EMPYREAN_REJECTION_ACCEPTED,
            EMPYREAN_REJECTION_CHI_SQUARED,
            EMPYREAN_REJECTION_SIGMA_CLIP,
            EMPYREAN_REJECTION_COOKS_DISTANCE,
            EMPYREAN_REJECTION_ADAPTIVE,
            EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY,
            EMPYREAN_REJECTION_CMC2003,
            EMPYREAN_REJECTION_RADAR_UNSUPPORTED,
            EMPYREAN_REJECTION_OCCULTATION_UNSUPPORTED,
            EMPYREAN_REJECTION_OUTSIDE_ARC,
            EMPYREAN_REJECTION_NON_FINITE_CHI2,
            EMPYREAN_REJECTION_MISSING_JACOBIAN,
            EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING,
            EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED,
            EMPYREAN_REJECTION_NEVER_ABSORBED,
            EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED,
        ];
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "no two reasons share a code");
        assert_eq!(
            sorted,
            (0..codes.len() as i32).collect::<Vec<_>>(),
            "the codes stay contiguous from 0, so the next free code is unambiguous"
        );
        assert!(
            !codes.contains(&EMPYREAN_REJECTION_NOT_EVALUATED),
            "the not-evaluated sentinel is outside the reason set"
        );
    }

    /// The worked case: three declared burns with only the MIDDLE
    /// one solved. Both per-segment arrays are indexed by DECLARED
    /// segment, so the solved burn's Δv and its posterior covariance
    /// land on the same index — 1 — and the two unsolved neighbours are
    /// NaN in both.
    ///
    /// Under the retired pairing the Δv array was in SOLVED order, so the
    /// single solved burn's Δv sat at index 0 while its covariance sat
    /// at index 1: a consumer reading `thrust_delta_m_per_s[i]` beside
    /// `thrust_correction_covariances[i]` would have attributed the
    /// middle burn's Δv to the FIRST burn's covariance, silently, having
    /// followed the header.
    #[test]
    fn the_thrust_arrays_share_one_declared_index_space() {
        let dv = [None, Some([1.0e-6, 2.0e-6, 3.0e-6]), None];
        let cov = [
            None,
            Some([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]),
            None,
        ];

        let (declared, dv_out, cov_out) = thrust_posterior_arrays(Some(&dv), Some(&cov));

        assert_eq!(
            declared, 3,
            "the count is DECLARED, not the one solved burn"
        );

        // The solved burn is at declared index 1 in BOTH arrays.
        assert!(
            dv_out[1].iter().all(|v| v.is_finite()),
            "the solved burn's Δv lands at its declared index"
        );
        assert_eq!(cov_out[1][1][1], 5.0, "and so does its covariance");

        // Its neighbours carry no posterior, in both arrays.
        for i in [0usize, 2] {
            assert!(
                dv_out[i].iter().all(|v| v.is_nan()),
                "segment {i} was not solved: its Δv must be NaN, never 0"
            );
            assert!(
                cov_out[i].iter().flatten().all(|v| v.is_nan()),
                "segment {i} was not solved: its 3x3 must be NaN-filled rather than \
                 echoing the prior under a posterior's name"
            );
        }

        // The Δv is converted to m/s, not passed through in AU/day.
        assert!(
            dv_out[1][0] > 1.0,
            "1e-6 AU/day is ~1.7 m/s; a passthrough would leave it at 1e-6"
        );
    }

    /// An orbit that declared no thrust reports a zero count and an
    /// all-NaN pair of arrays — never zeros, which would read as a
    /// fitted Δv of exactly zero with a singular covariance.
    #[test]
    fn no_declared_thrust_reports_nan_rather_than_zero() {
        let (declared, dv, cov) = thrust_posterior_arrays(None, None);
        assert_eq!(declared, 0);
        assert!(dv.iter().flatten().all(|v| v.is_nan()));
        assert!(cov.iter().flatten().flatten().all(|v| v.is_nan()));
    }

    /// Declared segments whose covariance is absent still count toward
    /// the declared width: the count describes what the ORBIT declared,
    /// not how many posteriors came back. A count derived from the
    /// solved entries would shrink the array bound and hide the
    /// trailing declared burns from a consumer entirely.
    #[test]
    fn the_declared_count_is_the_orbits_width_not_the_solved_one() {
        let cov = [Some([[1.0; 3]; 3]), None, None];
        let (declared, _, _) = thrust_posterior_arrays(None, Some(&cov));
        assert_eq!(
            declared, 3,
            "three burns were declared even though one posterior came back"
        );
    }

    /// The engine's per-observation-site reason maps to code 15 rather
    /// than being folded into one of the two codes it deliberately is
    /// not: the observatory IS known (so not UNSUPPORTED_OBSERVATORY),
    /// and no kernel is missing (so not SPACECRAFT_KERNEL_MISSING).
    #[test]
    fn the_per_observation_site_reason_maps_to_its_own_code() {
        assert_eq!(
            rejection_reason_to_c(&RejectionReason::PerObservationSiteRequired),
            EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED
        );
        assert_ne!(
            EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED,
            EMPYREAN_REJECTION_UNSUPPORTED_OBSERVATORY
        );
        assert_ne!(
            EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED,
            EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING
        );
    }
}

/// The OD **output** half of the joint: the posterior blocks the result
/// re-sources from the fitted orbit, the joint on `orbit.orbit_cov`, the
/// disposition echo, the warning channel, and the re-feed round trip.
///
/// The input half is covered in `joint.rs`; this module exists because
/// the output half shipped untested — every field here is one a caller
/// re-feeds, so a silently absent or wrongly-sourced block is a wrong
/// prior on the next fit rather than a visible failure.
#[cfg(test)]
mod od_output_joint_tests {
    use super::tests::{read_eros_observations, refeed_orbit, standard_od_config};
    use super::*;

    fn last_err_text() -> String {
        unsafe { CStr::from_ptr(crate::empyrean_last_error()) }
            .to_string_lossy()
            .into_owned()
    }

    /// Fit the bundled Eros arc with the Marsden block solved, which is
    /// the narrowest fit that produces a state↔Marsden border — the
    /// output block this surface exists to carry.
    ///
    /// Returns the delivered result table; the caller frees it.
    fn determine_eros_with_marsden(
        ctx: *const EmpyreanContext,
    ) -> Option<EmpyreanDetermineResults> {
        let (obs_ptr, obs_n) = read_eros_observations();
        let mut cfg = standard_od_config();
        cfg.solve_for = EMPYREAN_SOLVE_FOR_EXPLICIT;
        cfg.solve_for_flags.marsden = EMPYREAN_PARAM_SOLVED;

        let mut out: EmpyreanDetermineResults = unsafe { std::mem::zeroed() };
        let code = unsafe {
            empyrean_determine(
                ctx,
                obs_ptr,
                obs_n,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                &cfg,
                &mut out,
            )
        };
        unsafe { crate::od::empyrean_observations_free(obs_ptr, obs_n) };
        if code != 0 {
            // A Marsden solve can legitimately fail to converge on a
            // thin local kernel set, so a developer without the full
            // data still gets a skip. But this is the SECOND skip axis
            // in this helper and it is the dangerous one: a convergence
            // regression on a kernel-bearing runner would otherwise turn
            // both headline tests green while they assert nothing.
            // `EMPYREAN_REQUIRE_DATA` — which CI sets — makes it a
            // failure, exactly as it does for a missing data directory.
            let msg = format!(
                "Marsden determine did not deliver (code {code}): {}",
                last_err_text()
            );
            if code == EMPYREAN_DETERMINE_NONE_DELIVERED {
                unsafe { empyrean_determine_results_free(&mut out) };
            }
            if std::env::var("EMPYREAN_REQUIRE_DATA").is_ok_and(|v| v != "0") {
                panic!(
                    "{msg} — EMPYREAN_REQUIRE_DATA is set, so this fit is required to deliver and the test must not skip past it."
                );
            }
            eprintln!("skipping: {msg}");
            return None;
        }
        Some(out)
    }

    /// The fitted joint reaches the caller, and every block of it is the
    /// engine's own posterior rather than a re-derivation.
    #[test]
    fn a_marsden_fit_publishes_its_posterior_joint() {
        let Some(ctx) =
            crate::testing::context_or_skip("a_marsden_fit_publishes_its_posterior_joint")
        else {
            return;
        };
        let Some(mut results) = determine_eros_with_marsden(&ctx) else {
            return;
        };
        assert_eq!(results.num_objects, 1);
        let slot = unsafe { &*results.objects };
        assert_eq!(
            slot.delivered,
            1,
            "the Eros arc must deliver: {}",
            last_err_text()
        );
        let res = &slot.result;

        // The border is present and finite, and it is NOT all zeros —
        // an all-zero border is what the engine reads as "absent", so a
        // zero-filled one here would mean the marshal ran but carried
        // nothing.
        assert_eq!(
            res.orbit.orbit_cov.has_non_grav_cross, 1,
            "a Marsden fit's state↔A cross block must reach the caller"
        );
        let border = res.orbit.orbit_cov.non_grav_cross;
        assert!(
            border.iter().flatten().all(|v| v.is_finite()),
            "every border entry must be finite"
        );
        assert!(
            border.iter().flatten().any(|v| *v != 0.0),
            "the border must carry real correlations, not a zero block"
        );

        // The Marsden 3×3 is present too — the border's other half. The
        // two ship together or the engine refuses the re-feed.
        assert_eq!(
            res.non_grav.has_covariance, 1,
            "the posterior 3×3 must accompany the border it conditions"
        );
        assert!(
            res.non_grav
                .covariance
                .iter()
                .flatten()
                .all(|v| v.is_finite()),
            "the posterior 3×3 must be finite"
        );

        // Sourced from the fitted ORBIT, not from covariance_9x9. On a
        // width-9 Marsden fit both exist and must agree; the point of
        // the re-sourcing is that the orbit is right for EVERY width,
        // and this pins that it did not change the width-9 answer.
        // Asserted rather than guarded: this fit solves Marsden and
        // nothing else, so it IS width 9 and the legacy field must be
        // populated. A conditional would let the comparison silently
        // stop running the day the field or the width moved, which is
        // the only thing it exists to catch.
        assert_eq!(
            res.has_covariance_9x9, 1,
            "a width-9 Marsden fit must populate the legacy 9x9 — without it this \
             comparison cannot run at all"
        );
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (res.non_grav.covariance[i][j] - res.covariance_9x9[6 + i][6 + j]).abs()
                        < 1e-30,
                    "the orbit-sourced 3×3 must equal the 9×9's block on a width-9 fit \
                     at [{i}][{j}]"
                );
            }
        }

        // The disposition echo names what the fit actually did.
        assert_eq!(
            res.dispositions.marsden, EMPYREAN_PARAM_SOLVED,
            "the echo must report Marsden solved"
        );
        assert_eq!(res.dispositions.dt, EMPYREAN_PARAM_FIXED);
        assert_eq!(res.dispositions.amrat, EMPYREAN_PARAM_FIXED);
        assert!(
            res.dispositions
                .thrust_dispositions
                .iter()
                .all(|d| *d == EMPYREAN_PARAM_FIXED),
            "no thrust was declared, so every segment is fixed"
        );

        // No thrust declared: the per-segment arrays report zero
        // declared segments and stay NaN, never zeros.
        assert_eq!(res.n_thrust_segments, 0);
        assert_eq!(res.thrust_delta_count, 0);
        assert!(
            res.thrust_correction_covariances
                .iter()
                .flatten()
                .flatten()
                .all(|v| v.is_nan()),
            "an orbit with no declared burn reports NaN, not a zero covariance"
        );

        // The warnings channel is well-formed. Empty is the common case
        // and is what this fit should produce; what must never happen is
        // a non-zero count with a null array.
        assert!(
            res.num_warnings == 0 || !res.warnings.is_null(),
            "a non-zero warning count must come with an array"
        );
        for k in 0..res.num_warnings {
            let p = unsafe { *res.warnings.add(k) };
            assert!(!p.is_null(), "warning {k} must not be null");
            let text = unsafe { CStr::from_ptr(p) }.to_string_lossy();
            assert!(!text.trim().is_empty(), "warning {k} must carry text");
        }

        unsafe { empyrean_determine_results_free(&mut results) };
    }

    /// The round trip the whole surface exists for: fit, copy the joint
    /// onto an input orbit, refine again. The re-fed orbit must be
    /// ACCEPTED — a border without its 3×3, or a carrier naming an
    /// undeclared parameter, is refused by the engine, so acceptance is
    /// the assertion that the marshaled pair is coherent.
    #[test]
    fn a_fitted_joint_re_feeds_into_a_refine() {
        let Some(ctx) = crate::testing::context_or_skip("a_fitted_joint_re_feeds_into_a_refine")
        else {
            return;
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;
        let Some(mut results) = determine_eros_with_marsden(&ctx) else {
            return;
        };
        let slot = unsafe { &*results.objects };
        assert_eq!(slot.delivered, 1);
        let res = &slot.result;

        // The re-feed, field for field — the copy a C caller writes.
        let mut refed = refeed_orbit(&res.orbit);
        refed.state.has_non_grav_cross = res.orbit.orbit_cov.has_non_grav_cross;
        refed.state.non_grav_cross = res.orbit.orbit_cov.non_grav_cross;
        refed.state_param_cross = res.orbit.orbit_cov.state_param_cross;
        refed.n_state_param_cross = res.orbit.orbit_cov.n_state_param_cross;
        refed.param_pair_cross = res.orbit.orbit_cov.param_pair_cross;
        refed.n_param_pair_cross = res.orbit.orbit_cov.n_param_pair_cross;
        // The diagonal blocks the crosses are conditioned on.
        refed.a1 = res.non_grav.a1;
        refed.a2 = res.non_grav.a2;
        refed.a3 = res.non_grav.a3;
        refed.has_non_grav_covariance = res.non_grav.has_covariance;
        refed.non_grav_covariance = res.non_grav.covariance;
        refed.non_grav_dt_variance = if res.non_grav.has_dt_variance == 1 {
            res.non_grav.dt_variance
        } else {
            f64::NAN
        };

        let (obs_ptr, obs_n) = read_eros_observations();
        let mut cfg = standard_od_config();
        cfg.solve_for = EMPYREAN_SOLVE_FOR_EXPLICIT;
        cfg.solve_for_flags.marsden = EMPYREAN_PARAM_SOLVED;

        let mut refined: EmpyreanODResult = unsafe { std::mem::zeroed() };
        let code = unsafe { empyrean_refine(ctx_ptr, &refed, obs_ptr, obs_n, &cfg, &mut refined) };
        unsafe { crate::od::empyrean_observations_free(obs_ptr, obs_n) };

        assert_eq!(
            code,
            0,
            "a refine given the fit's own joint must be accepted, not refused. \
             A refusal here means the marshaled border and its 3×3 disagree, or the \
             carrier names a parameter the orbit does not declare. last_error: {}",
            last_err_text()
        );
        assert_eq!(refined.orbit.has_covariance, 1);
        assert!(
            refined.covariance.iter().flatten().all(|v| v.is_finite()),
            "the re-fed fit must deliver a finite covariance"
        );

        unsafe { empyrean_od_result_free(&mut refined) };
        unsafe { empyrean_determine_results_free(&mut results) };
    }

    /// The DT variance wire, both directions: it had none before 0.10.0, so
    /// a solved-DT fit round-tripped with its DT column closed. Pinned
    /// on the marshal contract rather than on a DT fit, which needs a
    /// comet arc this crate does not bundle.
    #[test]
    fn the_dt_variance_wire_carries_both_directions() {
        // ── Out ──
        //
        // Drive the real output marshal on both arms. Asserting on
        // `EmpyreanNonGravParams::default()` instead would be
        // tautological — the struct derives Default, so it proves only
        // that `derive` works, not that `od_result_non_grav_to_c`
        // publishes the switch it is supposed to.
        //
        // The contract: a carried variance publishes with the switch
        // SET; an absent one publishes NaN with the switch CLEAR. Not
        // 0.0 — that is a legal (if degenerate) variance, so a consumer
        // could not tell it from a real one.
        let mut orbit: empyrean_core::orbits::Orbits<AU> = empyrean_core::orbits::Orbits::empty();
        let coord = empyrean_core::convert::coordinate_state_to_coordinates(
            &empyrean_core::convert::CoordinateState {
                epoch_mjd_tdb: 59000.0,
                elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
                covariance: [[0.0; 6]; 6],
                has_covariance: 0,
                representation: EMPYREAN_REPRESENTATION_CARTESIAN,
                frame: 0,
                origin: 10,
            },
        )
        .expect("a well-formed Cartesian state");
        orbit
            .push("dt-probe".to_string(), coord.into_radians())
            .expect("push");

        let with_variance = empyrean_core::nongrav::NonGravParams {
            a1: 1.0e-10,
            a2: 0.0,
            a3: 0.0,
            model: NonGravModel::MarsdenSekanina(
                empyrean_core::nongrav::GFunction::inverse_square(),
            ),
            covariance: None,
            dt: Some(45.7),
            dt_variance: Some(4.0),
        };
        orbit.set_non_grav_params(0, Some(with_variance.clone()));
        let (has, flat) = od_result_non_grav_to_c_for_test(&orbit);
        assert_eq!(has, 1, "the orbit carries a non-grav block");
        assert_eq!(
            flat.has_dt_variance, 1,
            "a carried DT variance sets its switch"
        );
        assert_eq!(flat.dt_variance, 4.0, "and publishes the value verbatim");
        assert_eq!(flat.has_dt, 1, "the DT value rides with it");
        assert_eq!(flat.non_grav_dt, 45.7);

        let mut without = with_variance;
        without.dt_variance = None;
        orbit.set_non_grav_params(0, Some(without));
        let (_, flat) = od_result_non_grav_to_c_for_test(&orbit);
        assert_eq!(
            flat.has_dt_variance, 0,
            "an absent DT variance clears its switch"
        );
        assert!(
            flat.dt_variance.is_nan(),
            "and publishes NaN, never 0.0 — a consumer could not tell a zero-width \
             prior from an absent one"
        );

        // In: the input side reads the variance only when finite and
        // positive, which is the trigger that opens the DT column.
        let mut o: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        o.non_grav_dt = 45.7;
        o.non_grav_dt_variance = 4.0;
        let params = crate::propagate::empyrean_orbit_non_grav_params(&o)
            .expect("a DT value alone declares a non-grav block");
        assert_eq!(params.dt, Some(45.7), "the DT VALUE must survive");
        assert_eq!(
            params.dt_variance,
            Some(4.0),
            "and so must its prior variance — the pair opens and priors the DT column"
        );

        // A zero or negative variance is "no prior", not a prior of zero.
        for bad in [0.0, -1.0, f64::NAN] {
            let mut o: EmpyreanOrbit = unsafe { std::mem::zeroed() };
            o.non_grav_dt = 45.7;
            o.non_grav_dt_variance = bad;
            let params = crate::propagate::empyrean_orbit_non_grav_params(&o).expect("dt present");
            assert_eq!(
                params.dt_variance, None,
                "dt_variance = {bad} must read as absent, never as a zero-width prior"
            );
        }
    }

    /// A malformed carrier fails the OD call with the ARGUMENT code
    /// (-1), not the engine code (-2). The split is the only signal a
    /// caller has for "my struct is wrong" versus "my covariance is
    /// wrong", and it is decided before the engine is ever entered.
    #[test]
    fn a_malformed_carrier_fails_the_od_call_with_the_argument_code() {
        let Some(ctx) = crate::testing::context_or_skip(
            "a_malformed_carrier_fails_the_od_call_with_the_argument_code",
        ) else {
            return;
        };
        let ctx_ptr: *const EmpyreanContext = &ctx;
        let (obs_ptr, obs_n) = read_eros_observations();
        let cfg = standard_od_config();

        let mut o: EmpyreanOrbit = unsafe { std::mem::zeroed() };
        o.state = crate::CoordinateState {
            epoch_mjd_tdb: 59000.0,
            elements: [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            covariance: [[0.0; 6]; 6],
            has_covariance: 0,
            representation: EMPYREAN_REPRESENTATION_CARTESIAN,
            frame: 0,
            origin: 10,
            has_non_grav_cross: 0,
            non_grav_cross: [[0.0; 3]; 6],
        };
        o.non_grav_dt = f64::NAN;
        o.non_grav_dt_variance = f64::NAN;
        o.phot_system = -1;
        o.h_mag = f64::NAN;
        o.srp_amrat_variance = f64::NAN;

        // An unknown column kind: malformed C, not malformed physics.
        let bad = [crate::joint::EmpyreanStateParamCross {
            column: crate::joint::EmpyreanParamColumn {
                kind: 99,
                index: 0,
                segment: 0,
                component: 0,
            },
            values: [1.0; 6],
        }];
        o.state_param_cross = bad.as_ptr();
        o.n_state_param_cross = bad.len();

        let mut out: EmpyreanODResult = unsafe { std::mem::zeroed() };
        let code = unsafe { empyrean_refine(ctx_ptr, &o, obs_ptr, obs_n, &cfg, &mut out) };
        unsafe { crate::od::empyrean_observations_free(obs_ptr, obs_n) };

        assert_eq!(
            code,
            -1,
            "a malformed parameter-column tag is an ARGUMENT error, not an engine \
             refusal; last_error: {}",
            last_err_text()
        );
        let msg = last_err_text();
        assert!(
            msg.contains("99") && msg.contains("kind"),
            "the message must name the offending tag: {msg}"
        );
    }
}
