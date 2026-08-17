//! Propagation result types: per-epoch states, detected events, and
//! the bundle returned by [`Context::propagate`](super::Context::propagate).

// Fixed-size covariance/state matrices are filled by explicit diagonal index
// loops, which read more clearly than iterator adapters here.
#![allow(clippy::needless_range_loop)]

use crate::JointCovariance;
use std::ffi::CStr;

use crate::coordinate::{Frame, Origin};
use crate::error::{Error, Result};

/// How a covariance was derived — the resolved kind at an output epoch.
///
/// The Monte-Carlo run seed is carried separately on
/// [`TaggedCovariance::mc_seed`], not in this tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceKind {
    /// Linear STM mapping Φ Σ₀ Φᵀ.
    Linear,
    /// Park–Scheeres second-order (Jet2 STT) correction.
    SecondOrder,
    /// Third-order (Jet3 STT3) extension.
    ThirdOrder,
    /// Adaptive Gaussian Mixture. This *tag's* matrix is the mixture's
    /// moment collapse \\(\Sigma = \sum_k w_k (\Sigma_k + d_k d_k^\top)\\)
    /// — a single second moment, which is all a covariance readback can
    /// carry. The mixture itself is **not** collapsed by the engine: the
    /// retained per-component weights, means and covariances are on
    /// [`PropagationResult::mixtures`] and
    /// [`PropagationResult::mixture_at`].
    Mixture,
    /// Monte Carlo sample covariance.
    MonteCarlo,
    /// Sigma-point sample covariance: the second moment of the propagated
    /// canonical 2N+1 sigma-point set. Deterministic and parameter-free.
    SigmaPoint,
}

impl CovarianceKind {
    // The C tag is a `u8` field; bindgen renders the `#define`d
    // constants as `u32`, so compare in `u32`.
    pub(crate) fn from_u8(tag: u8) -> Result<Self> {
        Ok(match tag as u32 {
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_LINEAR => Self::Linear,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_SECOND_ORDER => Self::SecondOrder,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_THIRD_ORDER => Self::ThirdOrder,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_MIXTURE => Self::Mixture,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_MONTE_CARLO => Self::MonteCarlo,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT => Self::SigmaPoint,
            other => {
                return Err(Error::invalid_input(format!(
                    "C ABI returned unknown covariance kind tag: {other}"
                )));
            }
        })
    }

    /// The C ABI `u8` tag for this kind — inverse of [`from_u8`](Self::from_u8).
    pub(crate) fn to_u8(self) -> u8 {
        let tag = match self {
            Self::Linear => empyrean_sys::EMPYREAN_COVARIANCE_KIND_LINEAR,
            Self::SecondOrder => empyrean_sys::EMPYREAN_COVARIANCE_KIND_SECOND_ORDER,
            Self::ThirdOrder => empyrean_sys::EMPYREAN_COVARIANCE_KIND_THIRD_ORDER,
            Self::Mixture => empyrean_sys::EMPYREAN_COVARIANCE_KIND_MIXTURE,
            Self::MonteCarlo => empyrean_sys::EMPYREAN_COVARIANCE_KIND_MONTE_CARLO,
            Self::SigmaPoint => empyrean_sys::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT,
        };
        tag as u8
    }
}

/// Definiteness of a covariance matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovarianceQuality {
    /// All eigenvalues positive within round-off.
    PositiveDefinite,
    /// At least one meaningfully negative eigenvalue (`min_eig`).
    Indefinite {
        /// The most-negative eigenvalue found.
        min_eig: f64,
    },
    /// Explicitly repaired to PSD; `min_eig` is the value *before* repair.
    Repaired {
        /// The most-negative eigenvalue before the PSD repair.
        min_eig: f64,
    },
    /// Definite, but the second-order expansion that produced it is not
    /// clearly valid at this epoch: the quadratic term is not small
    /// against the linear one.
    ///
    /// Deliberately **not** folded into
    /// [`PositiveDefinite`](Self::PositiveDefinite) — the matrix is
    /// definite, so a consumer checking only for definiteness would
    /// receive a degraded covariance with a clean bill of health.
    ExpansionSuspect {
        /// κ_state, the block-wise quadratic/linear ratio that produced
        /// this classification. `f64::INFINITY` when a zero-spread block
        /// carried a nonzero second-order correction. **Read-only
        /// provenance** — `is_nan`/`is_infinite`-guard before any
        /// arithmetic; never feed it to a clamp.
        kappa_state: f64,
    },
}

/// The functional a covariance's second moment describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFunctional {
    /// Generic Cartesian-state second moment.
    CartesianState,
    /// Tied to the close-approach miss-distance functional — not a
    /// generic state σ.
    CloseApproachMissDistance,
}

/// Provenance-tagged, resolved-kind covariance readback at one
/// `(orbit, epoch)` — the honest covariance, distinct from the bare
/// linear [`PropagatedState::covariance`].
///
/// The corrected mean is
/// `state + mean_shift_prop.unwrap_or([0;6]) + mean_shift_input.unwrap_or([0;6])`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaggedCovariance {
    /// Epoch of this covariance.
    pub epoch: crate::Epoch,
    /// Co-located propagated nominal state `[x, y, z, vx, vy, vz]` (AU, AU/day).
    pub state: [f64; 6],
    /// The 6×6 covariance (AU², AU²/day, AU²/day² blocks).
    pub matrix: [[f64; 6]; 6],
    /// How the covariance was derived.
    pub kind: CovarianceKind,
    /// Monte-Carlo run seed (`Some` only when `kind == MonteCarlo`).
    pub mc_seed: Option<u64>,
    /// Second-order propagation mean shift δμ_prop (zero at t₀).
    pub mean_shift_prop: Option<[f64; 6]>,
    /// OD-estimator mean shift δμ₀ (nonzero at t₀).
    pub mean_shift_input: Option<[f64; 6]>,
    /// Definiteness of `matrix`.
    pub quality: CovarianceQuality,
    /// Origin body of the basis.
    pub origin: Origin,
    /// Reference frame of the basis.
    pub frame: Frame,
    /// [A1, A2, A3] non-grav solved flags. The matrix is the
    /// *marginalized* state block of a possibly-wider fit.
    pub non_grav: [bool; 3],
    /// Thrust Δv segments solved for.
    pub thrust_segments: u32,
    /// Solved width (6 / 9 / 12 / …) — the conservative-vs-optimistic IP axis.
    pub solved_width: u32,
    /// The functional this second moment describes.
    pub target_functional: TargetFunctional,
}

impl TaggedCovariance {
    pub(crate) fn from_ffi(s: &empyrean_sys::EmpyreanTaggedCovariance) -> Result<Self> {
        let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for tagged-covariance origin: {}",
                s.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(s.frame)?;
        let quality = match s.quality as u32 {
            empyrean_sys::EMPYREAN_COVARIANCE_QUALITY_POSITIVE_DEFINITE => {
                CovarianceQuality::PositiveDefinite
            }
            empyrean_sys::EMPYREAN_COVARIANCE_QUALITY_INDEFINITE => CovarianceQuality::Indefinite {
                min_eig: s.quality_min_eig,
            },
            empyrean_sys::EMPYREAN_COVARIANCE_QUALITY_REPAIRED => CovarianceQuality::Repaired {
                min_eig: s.quality_min_eig,
            },
            empyrean_sys::EMPYREAN_COVARIANCE_QUALITY_EXPANSION_SUSPECT => {
                CovarianceQuality::ExpansionSuspect {
                    kappa_state: s.quality_kappa_state,
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "C ABI returned unknown covariance quality tag: {other}"
                )));
            }
        };
        let target_functional = match s.target_functional as u32 {
            empyrean_sys::EMPYREAN_TARGET_FUNCTIONAL_CARTESIAN_STATE => {
                TargetFunctional::CartesianState
            }
            empyrean_sys::EMPYREAN_TARGET_FUNCTIONAL_CLOSE_APPROACH_MISS_DISTANCE => {
                TargetFunctional::CloseApproachMissDistance
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "C ABI returned unknown target functional tag: {other}"
                )));
            }
        };
        Ok(Self {
            epoch: crate::Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
            state: s.state,
            matrix: s.matrix,
            kind: CovarianceKind::from_u8(s.kind)?,
            mc_seed: (s.has_mc_seed != 0).then_some(s.mc_seed),
            mean_shift_prop: (s.has_mean_shift_prop != 0).then_some(s.mean_shift_prop),
            mean_shift_input: (s.has_mean_shift_input != 0).then_some(s.mean_shift_input),
            quality,
            origin,
            frame,
            non_grav: [s.non_grav[0] != 0, s.non_grav[1] != 0, s.non_grav[2] != 0],
            thrust_segments: s.thrust_segments,
            solved_width: s.solved_width,
            target_functional,
        })
    }
}

/// A propagated state at one epoch.
///
/// **No longer `Copy` as of the joint-covariance release.** The type now
/// owns heap storage (the wide carrier on [`joint`](Self::joint)), so
/// `Copy` is not available to it. `Clone` is, and copying a struct that
/// already carried a 1728-byte STT was never cheap; call sites that
/// relied on implicit copies need an explicit `.clone()`.
#[derive(Debug, Clone, PartialEq)]
pub struct PropagatedState {
    /// Epoch.
    pub epoch: crate::Epoch,
    /// Cartesian position (AU).
    pub position: [f64; 3],
    /// Cartesian velocity (AU/day).
    pub velocity: [f64; 3],
    /// Origin body of the state vector.
    pub origin: Origin,
    /// Reference frame.
    pub frame: Frame,
    /// 6×6 Cartesian covariance (AU, AU/day). `None` if absent. This is
    /// always the linear Φ Σ₀ Φᵀ mapping; for the resolved-kind
    /// covariance at a close approach use
    /// [`PropagationResult::covariance_series_cartesian`].
    pub covariance: Option<[[f64; 6]; 6]>,
    /// State Transition Matrix Φ(t, t₀). `None` unless the propagation
    /// traced it — which happens when first- or second-order uncertainty
    /// propagation produced it from an input covariance, **or** when
    /// [`PropagationConfig::compute_stm`](super::PropagationConfig::compute_stm)
    /// requested the trace outright. `compute_stm` does not need an input
    /// covariance: it forces the hyperdual integration on its own, so an
    /// orbit with no covariance still comes back with an STM.
    pub stm: Option<[[f64; 6]; 6]>,
    /// State Transition Tensor Ψ(t, t₀):
    /// `stt[k][a][b] = ∂²x_k / ∂x₀_a ∂x₀_b`. `None` unless
    /// [`UncertaintyMethod::SecondOrder`](super::UncertaintyMethod::SecondOrder)
    /// was used.
    pub stt: Option<[[[f64; 6]; 6]; 6]>,
    /// Resolved covariance kind at this epoch — the cheap per-state hint
    /// (Linear outside `Auto` CA windows). The full provenance is on
    /// [`PropagationResult::covariance_series_cartesian`].
    pub resolved_kind: CovarianceKind,
    /// The propagated joint's cross terms at this epoch, in the
    /// Cartesian basis of [`covariance`](Self::covariance).
    ///
    /// [`covariance`](Self::covariance) is only the state block. Feeding
    /// a second leg that block alone hands the engine a block-diagonal
    /// covariance, while the joint it computed has non-zero
    /// state↔parameter columns **even when the input was
    /// block-diagonal** — propagation generates that correlation. So a
    /// chained propagation built on the 6×6 alone reports a tighter
    /// uncertainty than the first leg supports.
    ///
    /// Copy it onto the next leg's [`Orbit`](crate::Orbit), together
    /// with the parameter blocks it is conditioned on, which propagation
    /// carries through from the input orbit unchanged.
    pub joint: crate::JointCovariance,
}

impl PropagatedState {
    pub(crate) fn from_ffi(s: &empyrean_sys::EmpyreanPropagatedState) -> Result<Self> {
        let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for origin: {}",
                s.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(s.frame)?;
        Ok(Self {
            epoch: crate::Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
            position: [s.x, s.y, s.z],
            velocity: [s.vx, s.vy, s.vz],
            origin,
            frame,
            covariance: (s.has_covariance != 0).then_some(s.covariance),
            stm: (s.has_stm != 0).then_some(s.stm),
            stt: (s.has_stt != 0).then_some(s.stt),
            resolved_kind: CovarianceKind::from_u8(s.resolved_kind)?,
            joint: unsafe { crate::JointCovariance::from_ffi(&s.orbit_cov) }?,
        })
    }
}

/// A detected dynamical event from propagation.
///
/// Event types fall into three groups:
///
/// - **Encounter events**: `close_approach_start`, `close_approach_end`,
///   `periapsis`, `apoapsis`, `soi_entry`, `soi_exit`,
///   `capture_start`, `capture_end`, `shadow_entry`, `shadow_exit`,
///   `atmospheric_entry`, `atmospheric_exit`, `impact`,
///   `possible_impact`.
/// - **Diagnostic events** (emitted only when the corresponding
///   `*_threshold` is set on
///   [`DiagnosticsConfig`](super::config::DiagnosticsConfig)):
///   `high_sensitivity`, `chaotic_region`, `high_nonlinearity`.
///
/// Type-specific fields are NaN when not applicable to the event kind.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Event type (e.g. `"periapsis"`, `"close_approach_start"`).
    pub event_type: String,
    /// Orbit identifier — the caller's `Orbit.orbit_id`, or a positional
    /// `"orbit_{i}"` if the caller didn't tag the orbit.
    pub orbit_id: String,
    /// Object identifier joined from the input batch via `orbit_id`.
    /// Empty string when the input had no `object_id`.
    pub object_id: String,
    /// Body the event involves (e.g. [`Origin::EARTH`], [`Origin::MOON`]).
    /// `None` for non-body events.
    pub body: Option<Origin>,
    /// Event epoch.
    pub epoch: crate::Epoch,
    /// Distance to body (AU). NaN if not applicable.
    pub distance_au: f64,
    /// Distance to body (km). NaN if not applicable.
    pub distance_km: f64,
    /// Relative velocity (AU/day). NaN if not applicable.
    pub relative_velocity_au_day: f64,
    /// `capture_start` / `capture_end`: two-body energy w.r.t. the
    /// capturing body (AU²/day²). NaN for other events.
    pub two_body_energy: f64,
    /// `capture_*`: CR3BP Jacobi constant. NaN when unavailable.
    pub jacobi_constant: f64,
    /// `capture_*`: 1σ uncertainty on the Jacobi constant. NaN when unavailable.
    pub jacobi_constant_sigma: f64,
    /// `capture_*`: Jacobi constant at the L1 gateway. NaN when unavailable.
    pub jacobi_constant_l1: f64,
    /// `capture_*`: Jacobi constant at the L2 gateway. NaN when unavailable.
    pub jacobi_constant_l2: f64,
    /// `capture_end`: number of periapsis passages during the temporary
    /// capture (0 = flyby/TCF, ≥1 = orbiter/TCO). `None` for other events.
    pub n_periapses: Option<u32>,
    /// `impact`: planetodetic latitude of the surface intercept (degrees).
    /// NaN for other events or when unresolved.
    pub impact_latitude_deg: f64,
    /// `impact`: planetodetic longitude of the surface intercept (degrees).
    /// NaN for other events or when unresolved.
    pub impact_longitude_deg: f64,
    /// `impact`: altitude of the surface intercept above the reference
    /// ellipsoid (km). NaN for other events or when unresolved.
    pub impact_altitude_km: f64,
    /// `shadow_entry` / `shadow_exit`: fraction of the Sun's disk occulted
    /// by the body (0 = none, 1 = full umbra). NaN for other events.
    pub shadow_fraction: f64,
    /// `shadow_entry` / `shadow_exit`: fraction of incident sunlight
    /// reaching the particle (1 = full sun, 0 = total eclipse). NaN for
    /// other events.
    pub illumination: f64,
    /// `periapsis`: relative position x w.r.t. the approached body (AU).
    /// NaN for other events.
    pub relative_x: f64,
    /// `periapsis`: relative position y w.r.t. the approached body (AU).
    pub relative_y: f64,
    /// `periapsis`: relative position z w.r.t. the approached body (AU).
    pub relative_z: f64,
    /// `periapsis`: relative velocity x w.r.t. the approached body (AU/day).
    /// NaN for other events.
    pub relative_vx: f64,
    /// `periapsis`: relative velocity y w.r.t. the approached body (AU/day).
    pub relative_vy: f64,
    /// `periapsis`: relative velocity z w.r.t. the approached body (AU/day).
    pub relative_vz: f64,
    /// `possible_impact`: effective capture radius with gravitational
    /// focusing (AU). NaN for other events.
    pub effective_radius_au: f64,
    /// `possible_impact`: effective capture radius with gravitational
    /// focusing (km). NaN for other events.
    pub effective_radius_km: f64,
    /// `possible_impact`: 1σ uncertainty along the miss direction (AU).
    /// NaN for other events.
    pub sigma_distance_au: f64,
    /// `possible_impact`: linear impact probability. NaN for other events.
    pub ip_linear: f64,
    /// `possible_impact`: second-order (Edgeworth) impact probability.
    /// NaN for other events or first-order runs.
    pub ip_second_order: f64,
    /// `possible_impact`: local nonlinearity κ. NaN when unavailable.
    pub nonlinearity: f64,
    /// `possible_impact`: adaptive-Gaussian-mixture impact probability.
    /// NaN when not an AGM run.
    pub ip_agm: f64,
    /// `possible_impact`: Monte-Carlo impact probability. NaN when not a
    /// Monte-Carlo run.
    pub ip_mc: f64,
    /// `covariance_regime_change`: resolved covariance kind *before* the
    /// transition. `None` for other events.
    pub previous_kind: Option<CovarianceKind>,
    /// `covariance_regime_change`: resolved covariance kind *after* the
    /// transition. `None` for other events.
    pub regime_resolved_kind: Option<CovarianceKind>,
    /// `covariance_regime_change`: local nonlinearity κ recorded at the
    /// CA. NaN for other events.
    pub kappa: f64,
    /// `covariance_regime_change`: lower κ value recorded in this audit
    /// payload. NaN for other events.
    pub threshold_below: f64,
    /// `covariance_regime_change`: upper κ value recorded in this audit
    /// payload. NaN for other events.
    pub threshold_above: f64,
}

impl Event {
    pub(crate) fn from_ffi(e: &empyrean_sys::EmpyreanEvent) -> Self {
        fn cstr_to_string(ptr: *const std::ffi::c_char) -> String {
            if ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
            }
        }
        // The C ABI uses -1 for non-body events; map that to `None`.
        // Any positive code that doesn't resolve to a known body also
        // becomes `None` rather than failing the whole batch.
        let body = if e.body_naif_id < 0 {
            None
        } else {
            Origin::from_naif_id(e.body_naif_id)
        };
        // `0xFF` is the C ABI's "not a regime event" sentinel; any other
        // tag that fails to resolve also degrades to `None` rather than
        // failing the whole batch.
        let kind_opt = |tag: u8| {
            if tag == 0xFF {
                None
            } else {
                CovarianceKind::from_u8(tag).ok()
            }
        };
        Self {
            event_type: cstr_to_string(e.event_type),
            orbit_id: cstr_to_string(e.orbit_id),
            object_id: cstr_to_string(e.object_id),
            body,
            epoch: crate::Epoch::from_mjd_tdb(e.epoch_mjd_tdb),
            distance_au: e.distance_au,
            distance_km: e.distance_km,
            relative_velocity_au_day: e.relative_velocity_au_day,
            two_body_energy: e.two_body_energy,
            jacobi_constant: e.jacobi_constant,
            jacobi_constant_sigma: e.jacobi_constant_sigma,
            jacobi_constant_l1: e.jacobi_constant_l1,
            jacobi_constant_l2: e.jacobi_constant_l2,
            n_periapses: (e.n_periapses >= 0).then_some(e.n_periapses as u32),
            impact_latitude_deg: e.impact_latitude_deg,
            impact_longitude_deg: e.impact_longitude_deg,
            impact_altitude_km: e.impact_altitude_km,
            shadow_fraction: e.shadow_fraction,
            illumination: e.illumination,
            relative_x: e.relative_x,
            relative_y: e.relative_y,
            relative_z: e.relative_z,
            relative_vx: e.relative_vx,
            relative_vy: e.relative_vy,
            relative_vz: e.relative_vz,
            effective_radius_au: e.effective_radius_au,
            effective_radius_km: e.effective_radius_km,
            sigma_distance_au: e.sigma_distance_au,
            ip_linear: e.ip_linear,
            ip_second_order: e.ip_second_order,
            nonlinearity: e.nonlinearity,
            ip_agm: e.ip_agm,
            ip_mc: e.ip_mc,
            previous_kind: kind_opt(e.previous_kind),
            regime_resolved_kind: kind_opt(e.resolved_kind),
            kappa: e.kappa,
            threshold_below: e.threshold_below,
            threshold_above: e.threshold_above,
        }
    }
}

/// One Gaussian sub-component of an adaptive-Gaussian-mixture (AGM)
/// decomposition, retained at a close-approach epoch.
///
/// The mean is the *propagated* sub-Gaussian centroid at the CA epoch;
/// the covariance is the linearly-mapped
/// \\(\Phi \\, \Sigma_k \\, \Phi^\top\\) over the same segment (the
/// second-order mean correction is omitted by design). A consumer can
/// evaluate
/// \\(\sum_k w_k \\, \mathcal{N}(x \mid \mu_k, \Sigma_k)\\)
/// directly at that epoch — no further propagation is needed.
///
/// Each component is basis-tagged, so the mixture is self-describing
/// rather than relying on positional alignment with the propagated
/// states.
///
/// Named to match `empyrean_core::propagation::MixtureComponent`
/// exactly. It is deliberately **not** re-exported at the crate root,
/// where [`crate::MixtureComponent`] is the unrelated
/// [`split_gaussian`](crate::split_gaussian) primitive at \\(t_0\\);
/// reach this one by its module path,
/// `empyrean::propagate::MixtureComponent`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureComponent {
    /// Prior split weight from the Gaussian splitting library — never
    /// likelihood-reweighted.
    pub weight: f64,
    /// Propagated sub-Gaussian centroid `[x, y, z, vx, vy, vz]` at the
    /// CA epoch (AU, AU/day), in the basis given by
    /// [`frame`](Self::frame) / [`origin`](Self::origin).
    pub mean: [f64; 6],
    /// Linearly-mapped component covariance \\(\Phi \Sigma_k \Phi^\top\\)
    /// in the same basis as [`mean`](Self::mean).
    pub covariance: [[f64; 6]; 6],
    /// Reference frame `mean` / `covariance` are expressed in — the
    /// integration frame of the run.
    pub frame: Frame,
    /// Origin (center body) `mean` is expressed relative to. Matches the
    /// propagation origin at the split's close-approach epoch, so it can
    /// differ between CA epochs of the same chain when origin switching
    /// occurred.
    pub origin: Origin,
}

/// One orbit's retained AGM mixture decomposition — the components
/// kept at each close approach where the splitter actually fired.
///
/// Named to match `empyrean_core::propagation::MixtureChain` exactly;
/// see [`MixtureComponent`] for why neither type is re-exported at the
/// crate root.
///
/// # Scope of what is retained
///
/// Four limits apply, and each one is a real property of the engine's
/// retention rather than a marshaling shortfall:
///
/// - **Depth-0 only.** Only the initial split is retained; recursive
///   AGM calls (depth > 0) are not captured.
/// - **Only CA epochs where AGM fired.** An orbit that never triggered
///   a split gets an empty chain, not a one-component chain.
/// - **Component covariance is the linear map.** Each
///   [`MixtureComponent::covariance`] is \\(\Phi \Sigma_k \Phi^\top\\);
///   the second-order mean correction is intentionally omitted.
/// - **Retained weights may sum to less than 1.** A sub-Gaussian whose
///   own sub-propagation missed the close approach (or failed to
///   integrate) contributes no component, and the deficit is not
///   recorded anywhere. Do not assume \\(\sum_k w_k = 1\\); sum
///   [`MixtureComponent::weight`] and check.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureChain {
    /// Orbit identifier — the caller's `Orbit.orbit_id`, or a positional
    /// `"orbit_{i}"` if the caller didn't tag the orbit.
    ///
    /// Empty on a row padded in for a batch the engine returned no
    /// chains for at all (see [`PropagationResult::mixtures`]): there
    /// was no chain to carry a name. The row's position in `mixtures` is
    /// the join key in every case.
    pub orbit_id: String,
    /// Close-approach epochs (MJD TDB) at which components were
    /// retained. Same length as [`components`](Self::components).
    pub ca_epochs_mjd_tdb: Vec<f64>,
    /// Retained components, one inner vector per entry of
    /// [`ca_epochs_mjd_tdb`](Self::ca_epochs_mjd_tdb).
    pub components: Vec<Vec<MixtureComponent>>,
}

impl MixtureComponent {
    fn from_ffi(c: &empyrean_sys::EmpyreanMixtureComponent) -> Result<Self> {
        let origin = Origin::from_naif_id(c.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for mixture component origin: {}",
                c.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(c.frame)?;
        Ok(Self {
            weight: c.weight,
            mean: c.mean,
            covariance: c.covariance,
            frame,
            origin,
        })
    }
}

impl MixtureChain {
    /// A chain that retained nothing — no CA epochs, no components, no
    /// identifier.
    ///
    /// This is how "this orbit split nothing" is spelled, and it is what
    /// `marshal_propagation_result` pads with when the engine returns no
    /// chains at all for a batch
    /// whose orbits carry no covariance, so
    /// [`PropagationResult::mixtures`] stays positional with the input
    /// batch for every method.
    pub(crate) fn empty() -> Self {
        Self {
            orbit_id: String::new(),
            ca_epochs_mjd_tdb: Vec::new(),
            components: Vec::new(),
        }
    }

    /// Un-flatten one C-ABI chain into the nested Rust shape.
    ///
    /// The four pointers on `EmpyreanMixtureChain` are independently
    /// nullable (an orbit that produced no mixture gets a row with
    /// `num_ca_epochs == 0` and all-null pointers), and
    /// `slice::from_raw_parts` requires a non-null pointer even for
    /// length 0 — so each is guarded on its own rather than inferred
    /// from a sibling's count.
    ///
    /// `components_offset` / `components_per_epoch` cross an FFI
    /// boundary and are therefore validated, not trusted: a slice that
    /// would run past `num_components_total` is an error, never an
    /// out-of-bounds read.
    ///
    /// # Safety
    ///
    /// `c` must be a chain populated by `empyrean_propagate`, with its
    /// pointers either null or valid for the lengths it declares.
    pub(crate) unsafe fn from_ffi(
        c: &empyrean_sys::EmpyreanMixtureChain,
        chain_index: usize,
    ) -> Result<Self> {
        let orbit_id = if c.orbit_id.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(c.orbit_id) }
                .to_string_lossy()
                .into_owned()
        };
        let n_epochs = c.num_ca_epochs;
        if n_epochs == 0 {
            return Ok(Self {
                orbit_id,
                ca_epochs_mjd_tdb: Vec::new(),
                components: Vec::new(),
            });
        }
        if c.ca_epochs_mjd_tdb.is_null() {
            return Err(Error::invalid_input(format!(
                "mixture chain {chain_index}: null ca_epochs_mjd_tdb with num_ca_epochs = {n_epochs}"
            )));
        }
        if c.components_per_epoch.is_null() {
            return Err(Error::invalid_input(format!(
                "mixture chain {chain_index}: null components_per_epoch with num_ca_epochs = {n_epochs}"
            )));
        }
        if c.components_offset.is_null() {
            return Err(Error::invalid_input(format!(
                "mixture chain {chain_index}: null components_offset with num_ca_epochs = {n_epochs}"
            )));
        }
        let epochs = unsafe { std::slice::from_raw_parts(c.ca_epochs_mjd_tdb, n_epochs) };
        let counts = unsafe { std::slice::from_raw_parts(c.components_per_epoch, n_epochs) };
        let offsets = unsafe { std::slice::from_raw_parts(c.components_offset, n_epochs) };
        let total = c.num_components_total;
        let flat: &[empyrean_sys::EmpyreanMixtureComponent] = if total == 0 {
            &[]
        } else if c.components.is_null() {
            return Err(Error::invalid_input(format!(
                "mixture chain {chain_index}: null components with num_components_total = {total}"
            )));
        } else {
            unsafe { std::slice::from_raw_parts(c.components, total) }
        };

        let mut components: Vec<Vec<MixtureComponent>> = Vec::with_capacity(n_epochs);
        for k in 0..n_epochs {
            let (start, count) = (offsets[k], counts[k]);
            if start > total {
                return Err(Error::invalid_input(format!(
                    "mixture chain {chain_index}: components_offset[{k}] = {start} exceeds \
                     num_components_total = {total}"
                )));
            }
            let end = start.checked_add(count).ok_or_else(|| {
                Error::invalid_input(format!(
                    "mixture chain {chain_index}: components_offset[{k}] = {start} + \
                     components_per_epoch[{k}] = {count} overflows"
                ))
            })?;
            if end > total {
                return Err(Error::invalid_input(format!(
                    "mixture chain {chain_index}: components_offset[{k}] = {start} + \
                     components_per_epoch[{k}] = {count} exceeds num_components_total = {total}"
                )));
            }
            components.push(
                flat[start..end]
                    .iter()
                    .map(MixtureComponent::from_ffi)
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Ok(Self {
            orbit_id,
            ca_epochs_mjd_tdb: epochs.to_vec(),
            components,
        })
    }
}

/// Result of propagating one or more orbits.
///
/// `states` is organized as `num_orbits × num_epochs` flat entries in
/// orbit-major order.
///
/// Holds the underlying C-ABI result so the on-demand tagged-covariance
/// accessors ([`covariance_series_cartesian`](Self::covariance_series_cartesian)
/// / [`covariance_at_cartesian`](Self::covariance_at_cartesian)) can
/// recompute the resolved-kind readback; it is freed on drop.
#[derive(Debug)]
pub struct PropagationResult {
    /// Propagated states (flat, orbit-major order). Within each orbit,
    /// rows are in **ascending epoch order, always** — positional pairing
    /// against an ascending, duplicate-free request grid is exact; for
    /// any other request shape, join on each state's `epoch`.
    pub states: Vec<PropagatedState>,
    /// Object identifiers aligned with the orbits input.
    pub object_ids: Vec<String>,
    /// Detected events (one list across all orbits — disambiguate via
    /// `Event::orbit_id`).
    pub events: Vec<Event>,
    /// Retained AGM mixture decompositions — **one row per input
    /// orbit**, positional with the input batch, not per state. An orbit
    /// whose splitter never fired carries an empty
    /// [`MixtureChain::components`] rather than being absent, so the
    /// positional join holds for every method and every batch.
    ///
    /// The engine returns no chains at all for a batch in which no orbit
    /// produced sensitivity tensors — a batch whose orbits carry no
    /// covariance, under any method. Those rows are padded here to empty
    /// chains, which is the same claim ("nothing split") the engine
    /// makes per-orbit; a padded row has an empty
    /// [`orbit_id`](MixtureChain::orbit_id), because there was no chain
    /// to name it, so join positionally rather than on the id.
    ///
    /// Read [`MixtureChain`]'s scope notes before consuming these:
    /// depth-0 only, CA epochs only, linear component covariance, and
    /// retained weights that may sum to less than 1.
    pub mixtures: Vec<MixtureChain>,
    /// Retained C-ABI result, freed on drop. The owned `states` /
    /// `object_ids` / `events` / `mixtures` above are independent
    /// copies; this is kept solely to back the lazy tagged-covariance
    /// accessors.
    ffi: Box<empyrean_sys::EmpyreanPropagationResult>,
}

// SAFETY: the retained `EmpyreanPropagationResult` (and the rich result
// behind its `lazy_handle`) is uniquely owned by this `PropagationResult`
// — there is no shared mutable aliasing. The lazy accessors take `&self`
// and only read the retained result, and drop frees it exactly once on
// the owning thread, so the value is sound to move between threads.
unsafe impl Send for PropagationResult {}

impl Drop for PropagationResult {
    fn drop(&mut self) {
        unsafe { empyrean_sys::empyrean_propagation_result_free(&mut *self.ffi) };
    }
}

impl PropagationResult {
    pub(crate) fn new(
        states: Vec<PropagatedState>,
        object_ids: Vec<String>,
        events: Vec<Event>,
        mixtures: Vec<MixtureChain>,
        ffi: empyrean_sys::EmpyreanPropagationResult,
    ) -> Self {
        Self {
            states,
            object_ids,
            events,
            mixtures,
            ffi: Box::new(ffi),
        }
    }

    /// Retained mixture components for one orbit at a close-approach
    /// epoch, or `None` when that orbit retained no components within
    /// `tolerance_days` of `epoch_mjd_tdb`.
    ///
    /// Mirrors
    /// `empyrean_core::propagation::PropagationResult::mixture_at` by
    /// name, argument names and semantics: the nearest retained CA epoch
    /// within the tolerance wins, and an orbit index past the end is
    /// `None` rather than a panic.
    ///
    /// The components are the mixture *at that CA epoch*. Away from a
    /// retained CA there is nothing to return — the engine keeps no
    /// off-CA mixture — so this is not an interpolator.
    pub fn mixture_at(
        &self,
        orbit_index: usize,
        epoch_mjd_tdb: f64,
        tolerance_days: f64,
    ) -> Option<&[MixtureComponent]> {
        let chain = self.mixtures.get(orbit_index)?;
        let mut best: Option<(usize, f64)> = None;
        for (k, t) in chain.ca_epochs_mjd_tdb.iter().enumerate() {
            let d = (t - epoch_mjd_tdb).abs();
            if d <= tolerance_days && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        let (k, _) = best?;
        chain.components.get(k).map(|v| v.as_slice())
    }

    /// Resolved-kind tagged covariance at every output epoch for one
    /// orbit, in the Cartesian basis — the honest readback that
    /// distinguishes a second-order close-approach ellipsoid from the
    /// bare linear [`PropagatedState::covariance`].
    ///
    /// Entry `k` corresponds to the orbit's `k`-th output epoch, aligned
    /// with `states[orbit_index * num_epochs + k]`.
    pub fn covariance_series_cartesian(&self, orbit_index: usize) -> Result<Vec<TaggedCovariance>> {
        let mut out_series: *mut empyrean_sys::EmpyreanTaggedCovarianceSeries =
            std::ptr::null_mut();
        let code = unsafe {
            empyrean_sys::empyrean_propagation_covariance_series_cartesian(
                &*self.ffi,
                orbit_index,
                &mut out_series,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // `out_series` is non-null on success; marshal then free.
        let result = {
            let series = unsafe { &*out_series };
            unsafe { std::slice::from_raw_parts(series.entries, series.num_entries) }
                .iter()
                .map(TaggedCovariance::from_ffi)
                .collect::<Result<Vec<_>>>()
        };
        unsafe { empyrean_sys::empyrean_tagged_covariance_series_free(out_series) };
        result
    }

    /// Resolved-kind tagged covariance at a single `(orbit_index,
    /// epoch_index)`, Cartesian basis — the point query.
    pub fn covariance_at_cartesian(
        &self,
        orbit_index: usize,
        epoch_index: usize,
    ) -> Result<TaggedCovariance> {
        let mut out = std::mem::MaybeUninit::<empyrean_sys::EmpyreanTaggedCovariance>::uninit();
        let code = unsafe {
            empyrean_sys::empyrean_propagation_covariance_at_cartesian(
                &*self.ffi,
                orbit_index,
                epoch_index,
                out.as_mut_ptr(),
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        let init = unsafe { out.assume_init() };
        TaggedCovariance::from_ffi(&init)
    }

    /// The propagated joint's cross terms at one
    /// `(orbit_index, epoch_index)` — the state↔Marsden border and the
    /// wide carrier whose state block is
    /// [`TaggedCovariance::matrix`](crate::TaggedCovariance).
    ///
    /// # Why it is a separate call from the covariance
    ///
    /// The C ABI keeps its tagged-covariance struct free of owned
    /// storage, so a C caller who wants only the \\(6 \\times 6\\) frees
    /// nothing. Asking for the joint is an explicit acquisition there,
    /// and this mirrors that shape rather than folding the two together
    /// — the parity rule is by name and semantics, and hiding an
    /// allocation behind the covariance accessor would give the two
    /// channels different ownership stories for the same call.
    ///
    /// Rust callers get the allocation released for them: the FFI arrays
    /// are copied into owned Rust values and freed before this returns.
    ///
    /// An empty [`JointCovariance`] means the engine produced no cross
    /// terms at this row — the orbit declared no solved-parameter block
    /// — not that the terms were zero.
    pub fn joint_at(&self, orbit_index: usize, epoch_index: usize) -> Result<JointCovariance> {
        let mut out = empyrean_sys::EmpyreanOrbitCovariance::default();
        let code = unsafe {
            empyrean_sys::empyrean_propagation_joint_at(
                &*self.ffi,
                orbit_index,
                epoch_index,
                &mut out,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }
        // Copy out, then release the engine's arrays before returning:
        // the owned Rust value must not alias storage the caller has no
        // handle to free.
        let joint = unsafe { JointCovariance::from_ffi(&out) };
        unsafe { empyrean_sys::empyrean_orbit_covariance_free(&mut out) };
        joint
    }
}

#[cfg(test)]
mod order_lock_tests {
    use super::*;
    use crate::coordinate::CoordinateState;
    use crate::{Context, Epoch, Orbit, PropagationConfig};
    use std::path::PathBuf;

    /// Resolve a usable data dir: `EMPYREAN_DATA_DIR` (CI) else
    /// `~/.empyrean/data` (local). Returns `None` to skip when neither
    /// yields a working Context.
    fn try_context() -> Option<Context> {
        let candidates = [
            std::env::var("EMPYREAN_DATA_DIR").ok().map(PathBuf::from),
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".empyrean/data")),
        ];
        for dir in candidates.into_iter().flatten() {
            if let Ok(ctx) = Context::from_data_dir(Some(&dir)) {
                return Some(ctx);
            }
        }
        None
    }

    /// Locks the index-ordering invariant the
    /// tagged-covariance accessors rely on: entry `k` of
    /// `covariance_series_cartesian` is aligned epoch-for-epoch with the
    /// orbit's `states[k]`, and its resolved kind matches the per-state
    /// `resolved_kind` hint. If a future change ever reorders the
    /// covariance series relative to the state grid, this fails in CI
    /// rather than only at a consumer's runtime.
    #[test]
    fn covariance_series_is_index_ordered_with_states() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping covariance_series_is_index_ordered_with_states: no data dir");
            return;
        };

        let t0_mjd = 60000.0;
        let t0 = Epoch::from_mjd_tdb(t0_mjd);
        // Near-circular heliocentric state at ~2 AU.
        let mut cov = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            cov[i][i] = 1e-12;
        }
        for i in 3..6 {
            cov[i][i] = 1e-16;
        }
        let state = CoordinateState::cartesian(
            t0,
            [2.0, 0.0, 0.0, 0.0, 0.012_17, 0.0],
            Frame::EclipticJ2000,
            Origin::Sun,
        )
        .with_covariance(cov);
        let orbit = Orbit::new(state).with_orbit_id("order-lock");

        let offsets = [0.0, 10.0, 30.0, 60.0];
        let epochs: Vec<Epoch> = offsets
            .iter()
            .map(|d| Epoch::from_mjd_tdb(t0_mjd + d))
            .collect();

        let result = ctx
            .propagate(&[orbit], &epochs, &PropagationConfig::default())
            .expect("propagation should succeed");

        let series = result
            .covariance_series_cartesian(0)
            .expect("covariance series should be produced for a covariance-bearing orbit");

        let n = epochs.len();
        assert_eq!(series.len(), n, "one tagged covariance per output epoch");
        assert_eq!(result.states.len(), n, "one orbit × n epochs");

        for (k, tagged) in series.iter().enumerate() {
            // orbit-major: orbit 0's k-th epoch is states[k].
            let st = &result.states[k];

            // ── the order lock ──
            let s_epoch = tagged.epoch.mjd_tdb().unwrap();
            let st_epoch = st.epoch.mjd_tdb().unwrap();
            assert!(
                (s_epoch - st_epoch).abs() < 1e-9,
                "series[{k}] epoch {s_epoch} != state epoch {st_epoch}"
            );

            // co-located nominal state matches the propagated state.
            let st_state = [
                st.position[0],
                st.position[1],
                st.position[2],
                st.velocity[0],
                st.velocity[1],
                st.velocity[2],
            ];
            assert_eq!(
                tagged.state, st_state,
                "series[{k}] co-located state mismatch"
            );

            // resolved-kind alignment, and FirstOrder + no CA ⟹ Linear.
            assert_eq!(
                tagged.kind, st.resolved_kind,
                "series[{k}] kind != per-state resolved_kind"
            );
            assert_eq!(tagged.kind, CovarianceKind::Linear);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CovarianceKind;

    /// Every covariance kind round-trips through its C-ABI tag, and the
    /// new sample-based kinds map to the tags the engine emits.
    #[test]
    fn covariance_kind_round_trips_c_tags() {
        for kind in [
            CovarianceKind::Linear,
            CovarianceKind::SecondOrder,
            CovarianceKind::ThirdOrder,
            CovarianceKind::Mixture,
            CovarianceKind::MonteCarlo,
            CovarianceKind::SigmaPoint,
        ] {
            assert_eq!(CovarianceKind::from_u8(kind.to_u8()).unwrap(), kind);
        }
        assert_eq!(
            CovarianceKind::from_u8(5).unwrap(),
            CovarianceKind::SigmaPoint
        );
        assert!(CovarianceKind::from_u8(6).is_err(), "unknown tags reject");
    }
}

#[cfg(test)]
mod mixture_marshal_tests {
    use super::{MixtureChain, MixtureComponent};
    use crate::coordinate::{Frame, Origin};

    /// Build a component with a recognizable weight so slices can be
    /// identified by value.
    fn comp(weight: f64) -> empyrean_sys::EmpyreanMixtureComponent {
        empyrean_sys::EmpyreanMixtureComponent {
            weight,
            mean: [weight; 6],
            covariance: [[weight; 6]; 6],
            // 0 = ICRF, 399 = Earth — the encodings the C ABI documents.
            frame: 0,
            origin: 399,
        }
    }

    /// A chain over borrowed storage. The caller keeps the backing
    /// vectors alive for the duration of the `from_ffi` call, which is
    /// the same contract the real C result provides.
    fn chain(
        epochs: &mut [f64],
        counts: &mut [usize],
        offsets: &mut [usize],
        comps: &mut [empyrean_sys::EmpyreanMixtureComponent],
        total: usize,
    ) -> empyrean_sys::EmpyreanMixtureChain {
        empyrean_sys::EmpyreanMixtureChain {
            orbit_id: std::ptr::null_mut(),
            ca_epochs_mjd_tdb: epochs.as_mut_ptr(),
            num_ca_epochs: epochs.len(),
            components_per_epoch: counts.as_mut_ptr(),
            components_offset: offsets.as_mut_ptr(),
            components: comps.as_mut_ptr(),
            num_components_total: total,
        }
    }

    /// The un-flattening reproduces the exact per-epoch slices the
    /// prefix-sum offsets describe, and decodes the basis tags rather
    /// than defaulting them.
    #[test]
    fn unflattens_components_by_offset_and_count() {
        let mut epochs = [60000.0_f64, 60100.0, 60200.0];
        let mut counts = [2_usize, 0, 3];
        let mut offsets = [0_usize, 2, 2];
        let mut comps = [comp(1.0), comp(2.0), comp(3.0), comp(4.0), comp(5.0)];
        let c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 5);

        let out = unsafe { MixtureChain::from_ffi(&c, 0) }.expect("well-formed chain marshals");
        assert_eq!(out.ca_epochs_mjd_tdb, vec![60000.0, 60100.0, 60200.0]);
        assert_eq!(out.components.len(), 3);
        let weights: Vec<Vec<f64>> = out
            .components
            .iter()
            .map(|g| g.iter().map(|c| c.weight).collect())
            .collect();
        assert_eq!(weights, vec![vec![1.0, 2.0], vec![], vec![3.0, 4.0, 5.0]]);
        // An epoch with zero components is an empty group, not a
        // dropped epoch — the epoch list and the group list stay
        // index-aligned.
        assert!(out.components[1].is_empty());
        assert_eq!(out.components[0][0].frame, Frame::ICRF);
        assert_eq!(out.components[0][0].origin, Origin::EARTH);
        assert_eq!(out.components[2][2].mean, [5.0; 6]);
    }

    /// An offset+count that would run past the flat array is a typed
    /// error, not an out-of-bounds read. The offsets come from across an
    /// FFI boundary and are not a local invariant.
    #[test]
    fn malformed_offset_errors_rather_than_reading_out_of_bounds() {
        let mut epochs = [60000.0_f64];
        let mut counts = [4_usize];
        let mut offsets = [0_usize];
        let mut comps = [comp(1.0), comp(2.0)];
        let c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 2);

        let err = unsafe { MixtureChain::from_ffi(&c, 7) }.expect_err("must reject");
        assert!(
            err.message.contains("mixture chain 7") && err.message.contains("exceeds"),
            "expected a bounds error naming the chain, got: {}",
            err.message
        );
    }

    /// The offset axis and the extent axis are separate failures: an
    /// offset already past the end is reported as such even when its
    /// count is zero (so offset + count would not overflow).
    #[test]
    fn offset_past_end_errors_on_its_own_axis() {
        let mut epochs = [60000.0_f64];
        let mut counts = [0_usize];
        let mut offsets = [9_usize];
        let mut comps = [comp(1.0), comp(2.0)];
        let c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 2);

        let err = unsafe { MixtureChain::from_ffi(&c, 3) }.expect_err("must reject");
        assert!(
            err.message.contains("components_offset[0] = 9"),
            "expected the offset axis to be named, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("components_per_epoch[0]"),
            "the extent axis must not be blamed for an offset fault: {}",
            err.message
        );
    }

    /// A non-mixture orbit's row — `num_ca_epochs == 0` with all-null
    /// pointers — marshals to an empty chain. `slice::from_raw_parts`
    /// requires a non-null pointer even at length 0, so this must never
    /// reach it.
    #[test]
    fn all_null_chain_marshals_to_empty() {
        let c = empyrean_sys::EmpyreanMixtureChain {
            orbit_id: std::ptr::null_mut(),
            ca_epochs_mjd_tdb: std::ptr::null_mut(),
            num_ca_epochs: 0,
            components_per_epoch: std::ptr::null_mut(),
            components_offset: std::ptr::null_mut(),
            components: std::ptr::null_mut(),
            num_components_total: 0,
        };
        let out = unsafe { MixtureChain::from_ffi(&c, 0) }.expect("empty chain marshals");
        assert!(out.ca_epochs_mjd_tdb.is_empty());
        assert!(out.components.is_empty());
        assert_eq!(out.orbit_id, "");
    }

    /// A declared epoch count with a null parallel array is a loud
    /// error, one per pointer — the four are guarded independently
    /// rather than inferred from a sibling's count.
    #[test]
    fn null_parallel_array_with_nonzero_count_errors() {
        let mut epochs = [60000.0_f64];
        let mut counts = [1_usize];
        let mut offsets = [0_usize];
        let mut comps = [comp(1.0)];

        let mut c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 1);
        c.ca_epochs_mjd_tdb = std::ptr::null_mut();
        let err = unsafe { MixtureChain::from_ffi(&c, 0) }.expect_err("must reject");
        assert!(
            err.message.contains("null ca_epochs_mjd_tdb"),
            "{}",
            err.message
        );

        let mut c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 1);
        c.components_per_epoch = std::ptr::null_mut();
        let err = unsafe { MixtureChain::from_ffi(&c, 0) }.expect_err("must reject");
        assert!(
            err.message.contains("null components_per_epoch"),
            "{}",
            err.message
        );

        let mut c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 1);
        c.components_offset = std::ptr::null_mut();
        let err = unsafe { MixtureChain::from_ffi(&c, 0) }.expect_err("must reject");
        assert!(
            err.message.contains("null components_offset"),
            "{}",
            err.message
        );

        let mut c = chain(&mut epochs, &mut counts, &mut offsets, &mut comps, 1);
        c.components = std::ptr::null_mut();
        let err = unsafe { MixtureChain::from_ffi(&c, 0) }.expect_err("must reject");
        assert!(err.message.contains("null components"), "{}", err.message);
    }

    /// An unknown basis tag is refused rather than defaulted to
    /// ICRF / SSB — the same loud path `PropagatedState::from_ffi` takes.
    #[test]
    fn unknown_basis_tags_error_rather_than_default() {
        let mut epochs = [60000.0_f64];
        let mut counts = [1_usize];
        let mut offsets = [0_usize];

        let mut bad_frame = [comp(1.0)];
        bad_frame[0].frame = 77;
        let c = chain(&mut epochs, &mut counts, &mut offsets, &mut bad_frame, 1);
        assert!(unsafe { MixtureChain::from_ffi(&c, 0) }.is_err());

        let mut bad_origin = [comp(1.0)];
        bad_origin[0].origin = -12345;
        let c = chain(&mut epochs, &mut counts, &mut offsets, &mut bad_origin, 1);
        let err = unsafe { MixtureChain::from_ffi(&c, 0) }.expect_err("must reject");
        assert!(
            err.message.contains("mixture component origin"),
            "{}",
            err.message
        );
    }

    /// `mixture_at` picks the nearest retained CA within the tolerance
    /// and returns `None` outside it — it is a lookup, not an
    /// interpolator.
    #[test]
    fn mixture_at_selects_nearest_within_tolerance() {
        fn component(weight: f64) -> MixtureComponent {
            MixtureComponent {
                weight,
                mean: [0.0; 6],
                covariance: [[0.0; 6]; 6],
                frame: Frame::ICRF,
                origin: Origin::EARTH,
            }
        }
        let result = super::PropagationResult::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![MixtureChain {
                orbit_id: "a".into(),
                ca_epochs_mjd_tdb: vec![60000.0, 60000.5],
                components: vec![vec![component(0.25)], vec![component(0.75)]],
            }],
            empyrean_sys::EmpyreanPropagationResult {
                states: std::ptr::null_mut(),
                num_states: 0,
                object_ids: std::ptr::null_mut(),
                events: std::ptr::null_mut(),
                num_events: 0,
                mixtures: std::ptr::null_mut(),
                num_mixtures: 0,
                lazy_handle: std::ptr::null_mut(),
            },
        );

        let near = result
            .mixture_at(0, 60000.4, 1.0)
            .expect("within tolerance");
        assert_eq!(near[0].weight, 0.75, "nearest CA wins, not the first");
        assert!(
            result.mixture_at(0, 60050.0, 1.0).is_none(),
            "outside tolerance"
        );
        assert!(
            result.mixture_at(1, 60000.0, 1.0).is_none(),
            "orbit out of range"
        );
    }
}
