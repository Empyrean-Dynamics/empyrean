//! Propagation result types: per-epoch states, detected events, and
//! the bundle returned by [`Context::propagate`](super::Context::propagate).

// Fixed-size covariance/state matrices are filled by explicit diagonal index
// loops, which read more clearly than iterator adapters here.
#![allow(clippy::needless_range_loop)]

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
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_MIXTURE => Self::Mixture,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_MONTE_CARLO => Self::MonteCarlo,
            empyrean_sys::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT => Self::SigmaPoint,
            // Tag 2 was third-order, retired with the engine; the unknown
            // tag (254) is refused by every decoder. Both land here rather
            // than being defaulted.
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
            Self::Mixture => empyrean_sys::EMPYREAN_COVARIANCE_KIND_MIXTURE,
            Self::MonteCarlo => empyrean_sys::EMPYREAN_COVARIANCE_KIND_MONTE_CARLO,
            Self::SigmaPoint => empyrean_sys::EMPYREAN_COVARIANCE_KIND_SIGMA_POINT,
        };
        tag as u8
    }
}

/// Provenance-tagged, resolved-kind covariance readback at one
/// `(orbit, epoch)` — the honest covariance, distinct from the bare
/// linear [`PropagatedState::covariance`].
///
/// The whole \\((6+P)\\)-wide matrix rides one packed [`joint`](Self::joint);
/// its tags (kind, quality, functional), the solved width, and the layout all
/// read off that joint. The corrected mean is
/// `state + mean_shift_prop.unwrap_or([0;6])`.
#[derive(Debug, Clone, PartialEq)]
pub struct TaggedCovariance {
    /// Epoch of this covariance.
    pub epoch: crate::Epoch,
    /// Co-located propagated nominal state `[x, y, z, vx, vy, vz]` (AU, AU/day).
    pub state: [f64; 6],
    /// The covariance as one packed joint — the single covariance home. Its
    /// `kind` / `quality` / `functional` name what it describes, and its
    /// `layout` / `width` name which parameters were solved.
    pub joint: crate::PackedJoint,
    /// Monte-Carlo run seed. `Some` only when the joint's kind is
    /// [`MonteCarlo`](CovarianceKind::MonteCarlo), but not on every such row:
    /// an ensemble seeded from system entropy is not reproducible by
    /// construction and carries `None`.
    pub mc_seed: Option<u64>,
    /// Second-order propagation mean shift δμ_prop (zero at t₀).
    pub mean_shift_prop: Option<[f64; 6]>,
    /// Origin body of the basis.
    pub origin: Origin,
    /// Reference frame of the basis.
    pub frame: Frame,
}

impl TaggedCovariance {
    /// The 6×6 state covariance (AU², AU²/day, AU²/day² blocks) — a derived
    /// read-only view of the joint's leading state block.
    pub fn matrix(&self) -> [[f64; 6]; 6] {
        self.joint.state_block()
    }

    /// How the covariance was derived — the joint's kind.
    pub fn kind(&self) -> CovarianceKind {
        self.joint.kind
    }

    /// The definiteness of the matrix — the joint's quality.
    pub fn quality(&self) -> crate::JointQuality {
        self.joint.quality
    }

    /// The solved width (6 / 9 / 12 / …) — the conservative-vs-optimistic IP
    /// axis. Reads the joint's width.
    pub fn solved_width(&self) -> u8 {
        self.joint.width
    }

    pub(crate) fn from_ffi(s: &empyrean_sys::EmpyreanTaggedCovariance) -> Result<Self> {
        let origin = Origin::from_naif_id(s.origin).ok_or_else(|| {
            Error::invalid_input(format!(
                "C ABI returned unknown NAIF id for tagged-covariance origin: {}",
                s.origin
            ))
        })?;
        let frame = crate::coordinate::int_to_frame(s.frame)?;
        let joint = unsafe { crate::PackedJoint::from_ffi(&s.joint) }?.ok_or_else(|| {
            Error::invalid_input(
                "C ABI returned a width-0 joint for a tagged covariance, which must carry a \
                 matrix",
            )
        })?;
        Ok(Self {
            epoch: crate::Epoch::from_mjd_tdb(s.epoch_mjd_tdb),
            state: s.state,
            joint,
            mc_seed: (s.has_mc_seed != 0).then_some(s.mc_seed),
            mean_shift_prop: (s.has_mean_shift_prop != 0).then_some(s.mean_shift_prop),
            origin,
            frame,
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
    ///
    /// **Boxed.** The tensor is 1728 bytes and inlining it
    /// made every `PropagatedState` carry that weight whether or not the
    /// request asked for second-order — a first-order raster reading
    /// nothing but [`position`](Self::position) still paid it, on every
    /// state of every orbit. Behind a `Box` the absent case costs one
    /// pointer, and the present case pays the same 1728 bytes it always
    /// did, on the heap. Read it exactly as before: `Option` deref makes
    /// `stt[k][a][b]` unchanged at the use site; only a `let stt:
    /// Option<[[[f64; 6]; 6]; 6]> = state.stt` style binding needs a
    /// `.map(|t| *t)`.
    pub stt: Option<Box<[[[f64; 6]; 6]; 6]>>,
    /// Resolved covariance kind at this epoch — the cheap per-state hint
    /// (Linear outside `Auto` CA windows). The full provenance is on
    /// [`PropagationResult::covariance_series_cartesian`].
    pub resolved_kind: CovarianceKind,
    /// The propagated joint at this epoch — the whole \\((6+P)\\)-wide
    /// covariance as one packed object, in the Cartesian basis of
    /// [`covariance`](Self::covariance). `None` when the state carries no
    /// covariance.
    ///
    /// [`covariance`](Self::covariance) is a derived read-only view of this
    /// joint's leading \\(6 \times 6\\) state block. Feeding a second leg that
    /// block alone hands the engine a block-diagonal covariance, while the
    /// joint it computed has non-zero state↔parameter columns **even when the
    /// input was block-diagonal** — propagation generates that correlation. So
    /// a chained propagation built on the 6×6 alone reports a tighter
    /// uncertainty than the first leg supports.
    ///
    /// Copy it onto the next leg's [`Orbit`](crate::Orbit) via
    /// [`Orbit::with_covariance`](crate::Orbit::with_covariance) to carry the
    /// whole matrix the first leg computed.
    pub joint: Option<crate::PackedJoint>,
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
            stt: (s.has_stt != 0).then(|| Box::new(s.stt)),
            resolved_kind: CovarianceKind::from_u8(s.resolved_kind)?,
            joint: unsafe { crate::PackedJoint::from_ffi(&s.orbit_cov) }?,
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
    pub(crate) fn from_ffi(e: &empyrean_sys::EmpyreanEvent) -> Result<Self> {
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
        // `0xFF` is the C ABI's "not a regime event" sentinel and the
        // ONLY tag that legitimately means "no kind here".
        //
        // Every other unresolvable tag is an error, not a `None`. The C
        // ABI grew `EMPYREAN_COVARIANCE_KIND_UNKNOWN` (0xFE) precisely so
        // that a kind this ABI has no tag for is refused rather than
        // folded onto a concrete one — and swallowing it here would undo
        // that at the last hop, reporting "this event has no covariance
        // kind" for an event that has one this build cannot name. The two
        // facts are not the same and must not share a representation.
        let kind_opt = |tag: u8| -> Result<Option<CovarianceKind>> {
            if tag == 0xFF {
                Ok(None)
            } else {
                CovarianceKind::from_u8(tag).map(Some)
            }
        };
        Ok(Self {
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
            previous_kind: kind_opt(e.previous_kind)?,
            regime_resolved_kind: kind_opt(e.resolved_kind)?,
            kappa: e.kappa,
            threshold_below: e.threshold_below,
            threshold_above: e.threshold_above,
        })
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
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureComponent {
    /// Prior split weight from the Gaussian splitting library — never
    /// likelihood-reweighted.
    pub weight: f64,
    /// Propagated sub-Gaussian centroid `[x, y, z, vx, vy, vz]` at the
    /// CA epoch (AU, AU/day), in the basis given by
    /// [`frame`](Self::frame) / [`origin`](Self::origin).
    pub mean: [f64; 6],
    /// The component covariance as one packed joint — the single covariance
    /// home. Its leading \\(6 \times 6\\) is \\(\Phi \Sigma_k \Phi^\top\\) in
    /// the same basis as [`mean`](Self::mean).
    pub joint: crate::PackedJoint,
    /// How this component resolved at the close approach.
    pub status: ComponentStatus,
    /// Reference frame `mean` / `joint` are expressed in — the integration
    /// frame of the run.
    pub frame: Frame,
    /// Origin (center body) `mean` is expressed relative to. Matches the
    /// propagation origin at the split's close-approach epoch, so it can
    /// differ between CA epochs of the same chain when origin switching
    /// occurred.
    pub origin: Origin,
}

/// How one adaptive-Gaussian-mixture component resolved at a close approach.
///
/// The four states are distinct outcomes, surfaced rather than collapsed: a
/// component that could not be resolved is not the same as one that resolved
/// to a flat curvature, and neither is a failure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentStatus {
    /// The component resolved to a usable second moment.
    Resolved,
    /// The curvature test refused the second-order correction; `rho` is the
    /// measured quadratic/linear ratio that drove the refusal.
    CurvatureRefused {
        /// The measured curvature ρ.
        rho: f64,
    },
    /// The component did not resolve at this epoch.
    Unresolved,
    /// The component's sub-propagation failed.
    Failed,
    /// The sampled on-sky ensemble overflowed the flat-sky linearization
    /// bound; `extent` is the measured on-sky angular spread in RADIANS that
    /// drove the refusal — a different quantity from `CurvatureRefused`'s
    /// dimensionless `rho`, carried through the same C ABI side-value slot.
    SkyLinearizationRefused {
        /// The measured on-sky angular extent, radians.
        extent: f64,
    },
}

impl ComponentStatus {
    pub(crate) fn from_ffi(status: u8, rho: f64) -> Result<Self> {
        Ok(match status as u32 {
            empyrean_sys::EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED => Self::Resolved,
            empyrean_sys::EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED => {
                Self::CurvatureRefused { rho }
            }
            empyrean_sys::EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED => Self::Unresolved,
            empyrean_sys::EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED => Self::Failed,
            empyrean_sys::EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED => {
                // The C ABI's single side-value slot (`rho`) carries the
                // on-sky angular extent (radians) for this status, not a
                // curvature — decoded here by name so a consumer never reads
                // a radian spread as a ρ.
                Self::SkyLinearizationRefused { extent: rho }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "C ABI returned unknown mixture component status tag: {other}"
                )));
            }
        })
    }
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
///   the leading \\(6 \times 6\\) of [`MixtureComponent::joint`] is
///   \\(\Phi \Sigma_k \Phi^\top\\);
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
        let joint = unsafe { crate::PackedJoint::from_ffi(&c.joint) }?.ok_or_else(|| {
            Error::invalid_input(
                "C ABI returned a width-0 joint for a mixture component, which must carry a \
                 covariance",
            )
        })?;
        Ok(Self {
            weight: c.weight,
            mean: c.mean,
            joint,
            status: ComponentStatus::from_ffi(c.status, c.rho)?,
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

/// The delivery outcome of one input orbit in a batch
/// [`Context::propagate`](crate::Context::propagate), one entry per input orbit
/// in input order ([`PropagationResult::outcomes`]).
///
/// A per-orbit failure no longer ends the whole batch: the failing orbit's slot
/// is [`Failed`](OrbitOutcome::Failed) while every other orbit delivers, and the
/// call still succeeds. Delivery is this value alone — never a row count, which
/// is zero for a [`Delivered`](OrbitOutcome::Delivered) orbit whose marker stop
/// fired before its first requested epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrbitOutcome {
    /// The orbit was propagated; its rows are
    /// `states[first_row .. first_row + num_rows]` (which may be empty).
    Delivered {
        /// Offset of this orbit's first delivered row in
        /// [`PropagationResult::states`].
        first_row: usize,
        /// Delivered row count for this orbit.
        num_rows: usize,
    },
    /// The orbit failed; no rows were delivered for it.
    Failed {
        /// The `EMPYREAN_PROPAGATE_FAILURE_*` classification of the failure.
        code: i32,
        /// The engine's full failure text.
        message: String,
    },
}

impl OrbitOutcome {
    /// Whether this orbit was delivered.
    pub fn is_delivered(&self) -> bool {
        matches!(self, OrbitOutcome::Delivered { .. })
    }
}

/// Result of propagating one or more orbits.
///
/// `states` is flat and orbit-major: each input orbit's delivered rows form
/// a contiguous block, in input order. It is `num_orbits × num_epochs` only
/// when every orbit delivered every epoch — a failed or short orbit
/// contributes fewer rows (or none), so map rows to orbits with
/// [`rows_for`](Self::rows_for) / [`outcomes`](Self::outcomes), never a
/// uniform `orbit_index × num_epochs` grid.
///
/// Holds the underlying C-ABI result so the on-demand tagged-covariance
/// accessors ([`covariance_series_cartesian`](Self::covariance_series_cartesian)
/// / [`covariance_at_cartesian`](Self::covariance_at_cartesian)) can
/// recompute the resolved-kind readback; it is freed on drop.
///
/// That retained result is the larger half of this value — about 60 kB
/// per (orbit, epoch) against roughly 4 kB for the states, measured — so
/// a caller that reads nothing but [`states`](Self::states) is holding
/// fifteen sixteenths of the weight for nothing until the value goes out
/// of scope. [`into_states`](Self::into_states) is the way to hand it
/// back at the point the states are taken, and carries the measurement.
#[derive(Debug)]
pub struct PropagationResult {
    /// Propagated states (flat, orbit-major order). Within each orbit,
    /// rows are in **ascending epoch order, always** — positional pairing
    /// against an ascending, duplicate-free request grid is exact; for
    /// any other request shape, join on each state's `epoch`.
    pub states: Vec<PropagatedState>,
    /// Object identifiers, one per delivered state row and positionally
    /// parallel to [`states`](Self::states) (not one per input orbit — a
    /// failed orbit contributes no row here). Map rows to input orbits with
    /// [`rows_for`](Self::rows_for) / [`outcomes`](Self::outcomes).
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
    /// Per-input-orbit delivery outcome, one entry per input orbit in input
    /// order. A failed orbit is carried here rather than ending the batch;
    /// read delivery from this, never from a row count. Map rows to orbits
    /// with [`rows_for`](Self::rows_for).
    pub outcomes: Vec<OrbitOutcome>,
    /// Retained C-ABI result, freed on drop. The owned `states` /
    /// `object_ids` / `events` / `mixtures` above are independent
    /// copies; this is kept solely to back the lazy tagged-covariance
    /// accessors.
    ffi: RetainedFfiResult,
}

/// The retained C-ABI result, owned so that dropping it frees the
/// engine-side rich result behind `lazy_handle`.
///
/// The `Drop` lives here rather than on [`PropagationResult`] for one
/// reason: a type with a `Drop` impl cannot be destructured, and
/// [`PropagationResult::into_states`] is exactly a destructuring — it
/// hands back the owned states and releases everything else. With the
/// guard one level down, that method is ordinary safe code instead of a
/// `ManuallyDrop` / `ptr::read` dance.
#[derive(Debug)]
struct RetainedFfiResult(Box<empyrean_sys::EmpyreanPropagationResult>);

#[cfg(test)]
thread_local! {
    /// How many retained engine results this thread has released.
    ///
    /// Test-only, and the only way a test can watch a release that is
    /// otherwise invisible: the free happens across the FFI boundary and
    /// leaves nothing behind for the caller to inspect. Counting here is
    /// what lets [`PropagationResult::into_states`] be shown to release
    /// **early** rather than merely eventually — the distinction the
    /// method exists for.
    ///
    /// Per-thread rather than process-wide because the test harness runs
    /// tests concurrently: a global counter would be moved by whatever
    /// unrelated test happened to drop a result at the same moment, and
    /// a delta assertion on it would be a coin flip.
    pub(crate) static RETAINED_RESULT_FREES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

impl Drop for RetainedFfiResult {
    fn drop(&mut self) {
        unsafe { empyrean_sys::empyrean_propagation_result_free(&mut *self.0) };
        #[cfg(test)]
        RETAINED_RESULT_FREES.with(|n| n.set(n.get() + 1));
    }
}

impl std::ops::Deref for RetainedFfiResult {
    type Target = empyrean_sys::EmpyreanPropagationResult;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// SAFETY: the retained `EmpyreanPropagationResult` (and the rich result
// behind its `lazy_handle`) is uniquely owned by this `PropagationResult`
// — there is no shared mutable aliasing. The lazy accessors take `&self`
// and only read the retained result, and drop frees it exactly once on
// the owning thread, so the value is sound to move between threads.
unsafe impl Send for PropagationResult {}

impl PropagationResult {
    pub(crate) fn new(
        states: Vec<PropagatedState>,
        object_ids: Vec<String>,
        events: Vec<Event>,
        mixtures: Vec<MixtureChain>,
        outcomes: Vec<OrbitOutcome>,
        ffi: empyrean_sys::EmpyreanPropagationResult,
    ) -> Self {
        Self {
            states,
            object_ids,
            events,
            mixtures,
            outcomes,
            ffi: RetainedFfiResult(Box::new(ffi)),
        }
    }

    /// Take the propagated states and release the retained engine result
    /// **now**, instead of at the end of this value's scope.
    ///
    /// A `PropagationResult` holds two independent things: the owned
    /// copies a caller reads directly ([`states`](Self::states),
    /// [`object_ids`](Self::object_ids), [`events`](Self::events),
    /// [`mixtures`](Self::mixtures)) and the engine-side rich result
    /// that backs the lazy accessors —
    /// [`covariance_series_cartesian`](Self::covariance_series_cartesian),
    /// [`covariance_at_cartesian`](Self::covariance_at_cartesian),
    /// [`joint_at`](Self::joint_at), [`mixture_at`](Self::mixture_at).
    /// The second is the larger of the two and it is retained for the
    /// whole scope, so a caller that reads only the states still holds
    /// the engine's per-(orbit, epoch) sensitivity storage until the
    /// value goes out of scope. Chunking a large raster, that is what
    /// sets the peak.
    ///
    /// This is the release valve, and it is the type system that makes
    /// it safe: the lazy accessors take `&self`, so consuming `self`
    /// proves none of them can be called afterwards. There is no config
    /// flag and no "no lazy readback" mode to get wrong — a caller that
    /// wants the tagged covariance simply does not call this.
    ///
    /// # What it releases
    ///
    /// Measured on a 64-orbit × 2-epoch call, accumulating 24 such calls
    /// and reading resident-set growth (macOS aarch64, one thread; the
    /// harness is `empyrean/examples/retained_bytes.rs` in the
    /// repository, which this crate's package does not ship):
    ///
    /// | held | per (orbit, epoch) |
    /// |---|---|
    /// | whole `PropagationResult` | ≈ 64 kB |
    /// | states alone, after `into_states` | ≈ 4 kB |
    ///
    /// So roughly **60 kB per (orbit, epoch)** — about fifteen sixteenths
    /// of what a result weighs — is the engine-side storage behind the
    /// lazy accessors, and this method is what gives it back. Both
    /// figures held to within a few percent across FirstOrder and
    /// SecondOrder requests: the engine retains its sensitivity storage
    /// either way, so the request shape moves the number far less than
    /// the cell count does.
    ///
    /// The ≈ 4 kB is **not** `size_of::<PropagatedState>()`, which the
    /// sibling test pins at 888 bytes. It is resident set: the boxed
    /// tensor when one is present, the [`joint`](PropagatedState::joint)
    /// carrier's heap allocation, the per-orbit id strings, `Vec` growth
    /// slack, and allocator rounding — none of which a `size_of` sees.
    /// Both columns are measured the same way, so the ratio between them
    /// is the number to trust.
    ///
    /// Those are the numbers to size a chunk from. A caller holding the
    /// results of a 4096-cell chunk is holding ≈ 260 MB it may have no
    /// use for; the same chunk through this method is ≈ 16 MB. Both are
    /// resident-set figures on one machine and one allocator — read them
    /// as the ratio and the order of magnitude, not as a guarantee.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), empyrean::Error> {
    /// # let ctx = empyrean::Context::from_data_dir(None)?;
    /// # let orbits: Vec<empyrean::Orbit> = Vec::new();
    /// # let epochs: Vec<empyrean::Epoch> = Vec::new();
    /// # let config = empyrean::PropagationConfig::default();
    /// let states = ctx.propagate(&orbits, &epochs, &config)?.into_states();
    /// // the engine-side result is already freed here
    /// # let _ = states;
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_states(self) -> Vec<PropagatedState> {
        let Self {
            states,
            object_ids,
            events,
            mixtures,
            outcomes,
            ffi,
        } = self;
        // Free the engine result first, so the peak this method exists
        // to lower is actually lowered before the owned side arrays are
        // released.
        drop(ffi);
        drop(object_ids);
        drop(events);
        drop(mixtures);
        drop(outcomes);
        states
    }

    /// The INPUT orbit count — one [`OrbitOutcome`] per input orbit, whatever
    /// each delivered.
    pub fn num_orbits(&self) -> usize {
        self.outcomes.len()
    }

    /// Whether every input orbit was delivered.
    pub fn all_delivered(&self) -> bool {
        self.outcomes.iter().all(OrbitOutcome::is_delivered)
    }

    /// The delivered rows of input orbit `orbit_index`, or `None` if the index
    /// is out of range or the orbit failed. A delivered orbit with zero rows
    /// (a marker stop before its first requested epoch) returns an empty slice,
    /// not `None` — delivery is the outcome, never the row count.
    pub fn rows_for(&self, orbit_index: usize) -> Option<&[PropagatedState]> {
        match self.outcomes.get(orbit_index)? {
            OrbitOutcome::Delivered {
                first_row,
                num_rows,
            } => self.states.get(*first_row..*first_row + *num_rows),
            OrbitOutcome::Failed { .. } => None,
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
    /// Entry `k` corresponds to the orbit's `k`-th delivered output epoch,
    /// aligned with the orbit's own block of
    /// [`states`](Self::states) — [`rows_for`](Self::rows_for)`(orbit_index)[k]`
    /// — not a uniform `orbit_index * num_epochs + k` grid, which a failed or
    /// short earlier orbit shifts.
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

    /// The propagated joint at one `(orbit_index, epoch_index)` — the whole
    /// \\((6+P)\\)-wide covariance as one packed object, whose leading
    /// \\(6 \times 6\\) state block is the same matrix
    /// [`TaggedCovariance::matrix`](crate::TaggedCovariance::matrix) returns.
    ///
    /// # Why it is a separate call from the covariance
    ///
    /// The C ABI owns the joint's triangle (`tri`), so asking for the joint is
    /// an explicit acquisition there, freed with the dedicated free call. This
    /// mirrors that shape rather than folding the two together — the parity
    /// rule is by name and semantics.
    ///
    /// Rust callers get the allocation released for them: the triangle is
    /// copied into an owned Rust value and freed before this returns.
    ///
    /// `None` means the engine produced a width-0 joint at this row — the
    /// orbit declared no covariance — not that the terms were zero.
    pub fn joint_at(
        &self,
        orbit_index: usize,
        epoch_index: usize,
    ) -> Result<Option<crate::PackedJoint>> {
        let mut out = empyrean_sys::EmpyreanPackedJoint::default();
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
        // Copy out, then release the engine's triangle before returning: the
        // owned Rust value must not alias storage the caller has no handle to
        // free.
        let joint = unsafe { crate::PackedJoint::from_ffi(&out) };
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
                tagged.kind(),
                st.resolved_kind,
                "series[{k}] kind != per-state resolved_kind"
            );
            assert_eq!(tagged.kind(), CovarianceKind::Linear);
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
        // Tag 2 (retired third-order) and 6 (unassigned) are both refused.
        assert!(CovarianceKind::from_u8(2).is_err(), "retired tags reject");
        assert!(CovarianceKind::from_u8(6).is_err(), "unknown tags reject");
    }

    /// Every mixture-component status decodes from its C-ABI `(status, rho)`
    /// pair, including `SkyLinearizationRefused` (code 4, new in villeneuve
    /// 1.26.0): its side value is the on-sky angular extent in radians,
    /// carried through the same slot `CurvatureRefused`'s ρ uses but decoded
    /// to a distinct variant by name. Collapsing the code-4 arm makes the C
    /// ABI return an "unknown status tag" error, failing this round-trip.
    #[test]
    fn component_status_round_trips_c_tags() {
        use super::ComponentStatus;
        use empyrean_sys::{
            EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED,
            EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED, EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED,
            EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED,
            EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED,
        };
        let f = |code: u32, rho: f64| ComponentStatus::from_ffi(code as u8, rho).unwrap();
        assert_eq!(
            f(EMPYREAN_MIXTURE_COMPONENT_STATUS_RESOLVED, f64::NAN),
            ComponentStatus::Resolved
        );
        assert_eq!(
            f(EMPYREAN_MIXTURE_COMPONENT_STATUS_CURVATURE_REFUSED, 2.5),
            ComponentStatus::CurvatureRefused { rho: 2.5 }
        );
        assert_eq!(
            f(EMPYREAN_MIXTURE_COMPONENT_STATUS_UNRESOLVED, f64::NAN),
            ComponentStatus::Unresolved
        );
        assert_eq!(
            f(EMPYREAN_MIXTURE_COMPONENT_STATUS_FAILED, f64::NAN),
            ComponentStatus::Failed
        );
        assert_eq!(
            f(
                EMPYREAN_MIXTURE_COMPONENT_STATUS_SKY_LINEARIZATION_REFUSED,
                0.5
            ),
            ComponentStatus::SkyLinearizationRefused { extent: 0.5 }
        );
        // An unknown tag is refused loudly, never silently defaulted.
        assert!(ComponentStatus::from_ffi(7, f64::NAN).is_err());
    }
}

#[cfg(test)]
mod event_kind_decode_tests {
    use super::Event;

    /// A zeroed `EmpyreanEvent` — every pointer null (the string readers
    /// handle null) and every payload 0 — so a test can set exactly the
    /// two tag fields it is about. `#[repr(C)]` POD with raw pointers,
    /// so an all-zero bit pattern is a valid value.
    fn blank_event() -> empyrean_sys::EmpyreanEvent {
        unsafe { std::mem::zeroed() }
    }

    /// `0xFF` is the only tag that means "this event has no covariance
    /// kind", and it must keep decoding to `None`.
    #[test]
    fn the_not_applicable_sentinel_decodes_to_none() {
        let mut e = blank_event();
        e.previous_kind = 0xFF;
        e.resolved_kind = 0xFF;
        let ev = Event::from_ffi(&e).expect("0xFF is not an error — it is 'no kind here'");
        assert!(ev.previous_kind.is_none());
        assert!(ev.regime_resolved_kind.is_none());
    }

    /// `EMPYREAN_COVARIANCE_KIND_UNKNOWN` (0xFE) must ERROR, not decode
    /// to `None`.
    ///
    /// The two facts are different and must not share a representation:
    /// `None` says "this event carries no covariance kind", while 0xFE
    /// says "it carries one this build has no name for". Swallowing the
    /// second into the first would undo, at the last hop, the entire
    /// reason the UNKNOWN tag exists — and the C ABI's own documentation
    /// promises the wrappers above it stop on an unrecognised tag.
    #[test]
    fn the_unknown_kind_tag_errors_rather_than_reading_as_absent() {
        let mut e = blank_event();
        e.previous_kind = 0xFF;
        e.resolved_kind = 0xFE;
        let err = Event::from_ffi(&e)
            .expect_err("an UNKNOWN covariance kind must not decode as 'no kind'");
        let msg = err.to_string();
        assert!(
            msg.contains("254") || msg.to_lowercase().contains("unknown"),
            "the error must name the tag it could not resolve, got {msg:?}"
        );

        // ...and on the other tag field too — both are decoded, so both
        // must refuse.
        let mut e = blank_event();
        e.previous_kind = 0xFE;
        e.resolved_kind = 0xFF;
        assert!(
            Event::from_ffi(&e).is_err(),
            "previous_kind must refuse an UNKNOWN tag as well"
        );
    }
}

#[cfg(test)]
mod mixture_marshal_tests {
    use super::{MixtureChain, MixtureComponent};
    use crate::coordinate::{Frame, Origin};

    /// A width-6 state triangle backing every synthetic component's joint.
    /// `static` so its pointer is valid for the whole `from_ffi` call the
    /// borrowed C structs are read through.
    static COMP_TRI: [f64; 21] = [1.0; 21];

    /// Build a component with a recognizable weight so slices can be
    /// identified by value.
    fn comp(weight: f64) -> empyrean_sys::EmpyreanMixtureComponent {
        empyrean_sys::EmpyreanMixtureComponent {
            weight,
            mean: [weight; 6],
            joint: empyrean_sys::EmpyreanPackedJoint {
                layout_present: 0b111_111,
                layout_considered: 0,
                layout_marginalized: 0,
                width: 6,
                tri: COMP_TRI.as_ptr(),
                tri_len: COMP_TRI.len(),
                kind: 0,
                quality: 0,
                cov_rank: 0,
                cov_null_axes: 0,
                cov_min_eig_corr: f64::NAN,
                functional: 0,
                provenance: 0,
                asserted_source: empyrean_sys::EMPYREAN_JOINT_ASSERTED_SOURCE_NONE as u8,
            },
            status: 0,
            rho: f64::NAN,
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
                joint: crate::PackedJoint {
                    layout: crate::AxisLayout {
                        present: 0b111_111,
                        considered: 0,
                        marginalized: 0,
                    },
                    width: 6,
                    tri: vec![0.0; 21],
                    kind: crate::CovarianceKind::Linear,
                    quality: crate::JointQuality::PositiveDefinite,
                    functional: crate::JointFunctional::State,
                    provenance: crate::JointProvenance::Derived,
                    asserted_source: None,
                },
                status: super::ComponentStatus::Resolved,
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
            Vec::new(),
            empyrean_sys::EmpyreanPropagationResult {
                states: std::ptr::null_mut(),
                num_states: 0,
                object_ids: std::ptr::null_mut(),
                events: std::ptr::null_mut(),
                num_events: 0,
                mixtures: std::ptr::null_mut(),
                num_mixtures: 0,
                lazy_handle: std::ptr::null_mut(),
                num_orbits: 0,
                outcomes: std::ptr::null_mut(),
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

#[cfg(test)]
mod result_weight_tests {
    use super::PropagatedState;

    /// The pre-boxing layout, so the win is measured rather than
    /// asserted from memory: identical to [`PropagatedState`] except
    /// that the STT is inline. Kept in step with the real type by the
    /// field-for-field assertions below — if a field is added to one and
    /// not the other, `size_without_stt` stops agreeing and this test
    /// fails rather than silently comparing two unrelated numbers.
    #[allow(dead_code)]
    struct InlineSttState {
        epoch: crate::Epoch,
        position: [f64; 3],
        velocity: [f64; 3],
        origin: crate::Origin,
        frame: crate::Frame,
        covariance: Option<[[f64; 6]; 6]>,
        stm: Option<[[f64; 6]; 6]>,
        stt: Option<[[[f64; 6]; 6]; 6]>,
        resolved_kind: super::CovarianceKind,
        joint: Option<crate::PackedJoint>,
    }

    /// Boxing the STT is the whole of F4's per-state win, and it is
    /// large: 1728 bytes of tensor that a request which never asked for
    /// second order was carrying on every state of every orbit.
    ///
    /// The numbers are pinned rather than merely compared so a future
    /// field addition has to come here and say so.
    #[test]
    fn boxing_the_stt_is_worth_1728_bytes_per_state() {
        let inline = std::mem::size_of::<InlineSttState>();
        let boxed = std::mem::size_of::<PropagatedState>();
        assert_eq!(
            std::mem::size_of::<Option<[[[f64; 6]; 6]; 6]>>(),
            1736,
            "the inline STT slot is the 1728-byte tensor plus its discriminant word"
        );
        assert_eq!(
            std::mem::size_of::<Option<Box<[[[f64; 6]; 6]; 6]>>>(),
            8,
            "a boxed STT slot is one niche-optimized pointer — no discriminant word"
        );
        assert_eq!(inline, 2472, "PropagatedState with the tensor inline");
        assert_eq!(boxed, 744, "PropagatedState with the tensor boxed");
        assert_eq!(
            inline - boxed,
            1728,
            "the saving is exactly the tensor: 6 * 6 * 6 * 8 bytes"
        );
    }

    /// The positive control for the assertions above: the two layouts
    /// really are the same struct apart from the STT slot, so the
    /// difference measured there is the STT and nothing else.
    #[test]
    fn the_two_layouts_differ_only_in_the_stt_slot() {
        let inline_without_stt = std::mem::size_of::<InlineSttState>()
            - std::mem::size_of::<Option<[[[f64; 6]; 6]; 6]>>();
        let boxed_without_stt = std::mem::size_of::<PropagatedState>()
            - std::mem::size_of::<Option<Box<[[[f64; 6]; 6]; 6]>>>();
        assert_eq!(
            inline_without_stt, boxed_without_stt,
            "everything other than the STT slot must weigh the same in both layouts"
        );
    }

    /// The C-ABI row is untouched. The boundary hands back a flat
    /// `EmpyreanPropagatedState` with the tensor inline, exactly as it
    /// did in 0.10.0 — the boxing is a wrapper-side layout choice made
    /// while copying off that row, not an ABI change. A C consumer that
    /// hand-mirrored this struct keeps its layout.
    #[test]
    fn the_c_abi_row_still_carries_the_tensor_inline() {
        assert_eq!(
            std::mem::size_of::<empyrean_sys::EmpyreanPropagatedState>(),
            2448,
            "the C row must not move: F4 is a wrapper-side change"
        );
    }
}

#[cfg(test)]
mod into_states_tests {
    use super::{PropagationResult, RETAINED_RESULT_FREES};

    /// Releases counted on this thread so far.
    fn frees() -> usize {
        RETAINED_RESULT_FREES.with(|n| n.get())
    }

    /// A result with nothing in it. `empyrean_propagation_result_free`
    /// guards every pointer on null and every count on zero, so a zeroed
    /// C struct is a safe thing to hand it — which is what lets these
    /// tests exercise the release path without a data directory.
    fn empty_result() -> PropagationResult {
        PropagationResult::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            empyrean_sys::EmpyreanPropagationResult::default(),
        )
    }

    /// The point of the method: the engine-side result is gone by the
    /// time the states are handed back, not at the end of the caller's
    /// scope.
    #[test]
    fn into_states_releases_the_handle_before_it_returns() {
        let before = frees();
        let states = empty_result().into_states();
        assert_eq!(
            frees(),
            before + 1,
            "into_states must free the retained result, not defer it"
        );
        assert!(states.is_empty());
    }

    /// The positive control, and the reason the assertion above means
    /// anything: an ordinary result that is merely *held* has not been
    /// released, and is released exactly once when it finally drops. A
    /// counter that incremented on construction, or that never
    /// incremented at all, would pass the first test and fail this one.
    #[test]
    fn a_held_result_is_released_only_when_it_drops() {
        let before = frees();
        let held = empty_result();
        assert_eq!(
            frees(),
            before,
            "constructing a result must not release anything"
        );
        drop(held);
        assert_eq!(
            frees(),
            before + 1,
            "dropping a result releases the retained result exactly once"
        );
    }
}

#[cfg(test)]
mod outcome_channel_tests {
    use super::*;
    use crate::coordinate::CoordinateState;
    use crate::{Context, Epoch, Frame, Orbit, Origin, PropagationConfig};
    use std::path::PathBuf;

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

    /// A valid heliocentric Cartesian orbit (delivers at every epoch).
    fn delivered(id: &str, px: f64) -> Orbit {
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [px, 0.0, 0.0, 0.0, 0.012_17, 0.0],
            Frame::EclipticJ2000,
            Origin::Sun,
        );
        Orbit::new(state).with_orbit_id(id)
    }

    /// An orbit the engine refuses to integrate: a finite state whose epoch is
    /// far outside the ephemeris span, so the perturber lookup fails on the
    /// first integration step — a carried per-orbit `Failed`, not a call-level
    /// error.
    fn refused(id: &str) -> Orbit {
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(400_000.0),
            [1.0, 0.0, 0.0, 0.0, 0.012_17, 0.0],
            Frame::EclipticJ2000,
            Origin::Sun,
        );
        Orbit::new(state).with_orbit_id(id)
    }

    /// The same mixed batch the C contract test runs, through the safe wrapper:
    /// outcomes read by variant, `all_delivered()` false, and `rows_for`
    /// selecting one orbit's rows out of the shared `states`.
    #[test]
    fn a_mixed_batch_exposes_outcomes_and_rows_for() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping a_mixed_batch_exposes_outcomes_and_rows_for: no data dir");
            return;
        };
        let orbits = [delivered("a", 1.0), refused("b"), delivered("c", 1.2)];
        let epochs: Vec<Epoch> = [59000.0, 59005.0, 59012.0]
            .iter()
            .map(|&t| Epoch::from_mjd_tdb(t))
            .collect();
        let result = ctx
            .propagate(&orbits, &epochs, &PropagationConfig::default())
            .expect("a per-orbit failure does not end the batch");

        assert_eq!(result.num_orbits(), 3, "one outcome per input orbit");
        assert!(!result.all_delivered(), "the middle orbit failed");

        match &result.outcomes[0] {
            OrbitOutcome::Delivered {
                first_row,
                num_rows,
            } => {
                assert_eq!(*first_row, 0);
                assert_eq!(*num_rows, 3);
            }
            other => panic!("orbit 0 should deliver, got {other:?}"),
        }
        match &result.outcomes[1] {
            OrbitOutcome::Failed { code, message } => {
                assert_ne!(*code, 0, "a failed orbit carries a classified code");
                assert!(!message.is_empty(), "a failed orbit carries its message");
            }
            other => panic!("orbit 1 should fail, got {other:?}"),
        }

        // rows_for selects one orbit's block; the failed orbit has none.
        assert!(result.rows_for(1).is_none(), "a failed orbit has no rows");
        let rows2 = result.rows_for(2).expect("orbit 2 delivered rows");
        assert_eq!(rows2.len(), 3, "orbit 2 delivers one row per epoch");

        // rows_for returns a slice of the SAME states (not a copy): the first
        // row of orbit 2's block is exactly states[first_row].
        let OrbitOutcome::Delivered { first_row, .. } = &result.outcomes[2] else {
            panic!("orbit 2 should deliver");
        };
        assert_eq!(
            rows2[0].epoch.mjd_tdb().unwrap(),
            result.states[*first_row].epoch.mjd_tdb().unwrap(),
            "rows_for(2) is the slice of states starting at outcome 2's first_row"
        );
        // The failed middle orbit occupies NO rows, so orbit 2 begins right
        // after orbit 0's three rows.
        assert_eq!(*first_row, 3, "orbit 2's rows follow orbit 0's immediately");
        assert_eq!(
            result.states.len(),
            6,
            "num_states is delivered rows only (orbits 0 and 2)"
        );
    }

    /// Regression pin for the distribution-side out-of-bounds read: a mixed
    /// batch (delivered, failed, delivered) resolved to a SINGLE target epoch
    /// gives `num_states` (2) < `num_orbits` (3). The marshal once sized the
    /// object-id read by `num_orbits`, reading one `*const c_char` past the
    /// C-allocated array — a release segfault (debug survived by heap luck).
    /// Reaching these assertions at all proves the read stayed in bounds; the
    /// object-id column is parallel to `states` (one per delivered row), never
    /// to the input orbits. Reverting the read length to `num_orbits` makes
    /// this test crash or fail.
    #[test]
    fn a_single_epoch_mixed_batch_reads_object_ids_safely() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping a_single_epoch_mixed_batch_reads_object_ids_safely: no data dir");
            return;
        };
        let orbits = [delivered("a", 1.0), refused("b"), delivered("c", 1.2)];
        let epochs = [Epoch::from_mjd_tdb(59012.0)];
        let result = ctx
            .propagate(&orbits, &epochs, &PropagationConfig::default())
            .expect("a per-orbit failure does not end the batch");

        assert_eq!(result.num_orbits(), 3, "one outcome per input orbit");
        assert!(!result.all_delivered(), "the middle orbit failed");
        assert_eq!(
            result.states.len(),
            2,
            "two delivered orbits, one epoch each; num_states < num_orbits"
        );
        // The object-id column — the array the OOB read ran past — is sized by
        // num_states (parallel to states), not num_orbits.
        assert_eq!(
            result.object_ids.len(),
            result.states.len(),
            "object_ids is one per delivered state row"
        );
        assert!(!result.outcomes[1].is_delivered(), "orbit 1 failed");
        assert_eq!(result.rows_for(0).map(<[_]>::len), Some(1));
        assert!(result.rows_for(1).is_none(), "a failed orbit has no rows");
        assert_eq!(result.rows_for(2).map(<[_]>::len), Some(1));
    }

    /// A covariance-bearing orbit at the standard epoch — needed for a tagged
    /// covariance series to exist.
    fn delivered_with_cov(id: &str, px: f64) -> Orbit {
        let mut cov = [[0.0_f64; 6]; 6];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = if i < 3 { 1.0e-16 } else { 1.0e-20 };
        }
        let state = CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [px, 0.0, 0.0, 0.0, 0.012_17, 0.0],
            Frame::EclipticJ2000,
            Origin::Sun,
        )
        .with_covariance(cov);
        Orbit::new(state).with_orbit_id(id)
    }

    /// Regression pin for the lazy covariance-series accessor's row index: a
    /// covariance-bearing mixed batch (delivered, FAILED, delivered) over
    /// several epochs. The accessor co-locates each series entry's nominal
    /// state through the engine's per-orbit row ranges, so querying orbit 2 —
    /// whose block is shifted earlier because the failed orbit 1 delivered no
    /// rows — returns orbit 2's own states, bit-identical to the delivered
    /// state table's slice. The old `orbit_index * n_times` grid indexed past
    /// orbit 2's block and refused the whole series (STATE_MISSING); restoring
    /// it makes this fail.
    #[test]
    fn covariance_series_reindexes_through_orbit_blocks_for_a_mixed_batch() {
        let Some(ctx) = try_context() else {
            eprintln!(
                "skipping covariance_series_reindexes_through_orbit_blocks_for_a_mixed_batch: \
                 no data dir"
            );
            return;
        };
        let orbits = [
            delivered_with_cov("a", 1.0),
            refused("b"),
            delivered_with_cov("c", 1.2),
        ];
        let epochs: Vec<Epoch> = [59000.0, 59005.0, 59012.0]
            .iter()
            .map(|&t| Epoch::from_mjd_tdb(t))
            .collect();
        let result = ctx
            .propagate(&orbits, &epochs, &PropagationConfig::default())
            .expect("a per-orbit failure does not end the batch");

        let series = result
            .covariance_series_cartesian(2)
            .expect("orbit 2 carries a covariance series");
        let rows2 = result.rows_for(2).expect("orbit 2 delivered rows");
        assert_eq!(
            series.len(),
            rows2.len(),
            "one series entry per delivered epoch"
        );
        for (entry, state) in series.iter().zip(rows2.iter()) {
            let expected = [
                state.position[0],
                state.position[1],
                state.position[2],
                state.velocity[0],
                state.velocity[1],
                state.velocity[2],
            ];
            assert_eq!(
                entry.state, expected,
                "the series co-locates orbit 2's own nominal state (rows_for(2)), bit-identical"
            );
        }
    }
}
