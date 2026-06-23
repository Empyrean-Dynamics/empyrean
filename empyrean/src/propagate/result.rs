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
    /// Third-order (Jet3 STT3) extension.
    ThirdOrder,
    /// Adaptive Gaussian Mixture (moment-collapsed to a single second moment).
    Mixture,
    /// Monte Carlo sample covariance.
    MonteCarlo,
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// State Transition Matrix Φ(t, t₀). `None` unless first- or
    /// second-order uncertainty propagation produced it.
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
        fn cstr_to_string(ptr: *const i8) -> String {
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
    /// Propagated states (flat, orbit-major order).
    pub states: Vec<PropagatedState>,
    /// Object identifiers aligned with the orbits input.
    pub object_ids: Vec<String>,
    /// Detected events (one list across all orbits — disambiguate via
    /// `Event::orbit_id`).
    pub events: Vec<Event>,
    /// Retained C-ABI result, freed on drop. The owned `states` /
    /// `object_ids` / `events` above are independent copies; this is
    /// kept solely to back the lazy tagged-covariance accessors.
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
        ffi: empyrean_sys::EmpyreanPropagationResult,
    ) -> Self {
        Self {
            states,
            object_ids,
            events,
            ffi: Box::new(ffi),
        }
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
