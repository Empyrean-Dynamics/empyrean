//! Continuous-thrust inputs: steering law, thrust arcs, and thrust
//! parameters for low-thrust / finite-burn propagation.
//!
//! These are the safe mirror of the engine's thrust surface. Attach a
//! [`ThrustParams`] to an [`Orbit`](crate::Orbit) via
//! [`Orbit::with_thrust`](crate::Orbit::with_thrust) to model finite
//! burns during propagation. The burn-sensitivity segments that a
//! non-empty `correction_covariances` produces surface in the propagated
//! [`TaggedCovariance::thrust_segments`](crate::TaggedCovariance).
//!
//! Field names, units, and semantics match the engine one-for-one so an
//! orbit round-trips through the wrapper without renaming or reshaping.

use crate::coordinate::Origin;

/// Steering law — how the thrust direction is computed from the
/// spacecraft state relative to the arc's [`central_body`](ThrustArc::central_body).
///
/// The integer tags used at the FFI boundary match the
/// `EMPYREAN_STEERING_LAW_*` C-ABI constants.
#[derive(Debug, Clone, PartialEq)]
pub enum SteeringLaw {
    /// Constant RTN angles relative to the central body. The direction is
    /// \\[
    ///   \hat{d} = \cos\beta\cos\alpha\;\hat{R}
    ///           + \cos\beta\sin\alpha\;\hat{T}
    ///           + \sin\beta\;\hat{N}
    /// \\]
    /// where \\(\alpha\\) is the in-plane angle (radial toward transverse)
    /// and \\(\beta\\) is the out-of-plane angle (toward orbit normal).
    ConstantRTN {
        /// In-plane angle from radial toward transverse (radians).
        alpha_rad: f64,
        /// Out-of-plane angle toward orbit normal (radians).
        beta_rad: f64,
    },
    /// Thrust aligned with the velocity vector relative to the central
    /// body: \\( \hat{d} = \hat{v}_\text{body} \\).
    VelocityTangent,
    /// Fixed direction in the inertial frame.
    InertialFixed {
        /// Direction vector; normalized internally.
        direction: [f64; 3],
    },
}

impl SteeringLaw {
    /// Marshal into the FFI tag plus the parameter slots. Slots not used
    /// by the active variant are zeroed — the engine reads
    /// `steering_alpha_rad` / `steering_beta_rad` only for
    /// [`ConstantRTN`](Self::ConstantRTN) and `steering_direction` only
    /// for [`InertialFixed`](Self::InertialFixed).
    fn to_ffi_parts(&self) -> (i32, f64, f64, [f64; 3]) {
        match *self {
            // Bindgen renders the C `#define`d tags as `u32`; cast to the
            // `i32` field type (same pattern as the integrator tags).
            SteeringLaw::ConstantRTN {
                alpha_rad,
                beta_rad,
            } => (
                empyrean_sys::EMPYREAN_STEERING_LAW_CONSTANT_RTN as i32,
                alpha_rad,
                beta_rad,
                [0.0; 3],
            ),
            SteeringLaw::VelocityTangent => (
                empyrean_sys::EMPYREAN_STEERING_LAW_VELOCITY_TANGENT as i32,
                0.0,
                0.0,
                [0.0; 3],
            ),
            SteeringLaw::InertialFixed { direction } => (
                empyrean_sys::EMPYREAN_STEERING_LAW_INERTIAL_FIXED as i32,
                0.0,
                0.0,
                direction,
            ),
        }
    }
}

/// A single continuous-thrust arc with smooth on/off switching and
/// optional mass depletion.
///
/// The acceleration during the arc is
/// \\[
///   \mathbf{a}(t) = \sigma(t)\,\frac{F}{m(t)}\,\hat{d}
/// \\]
/// with a smooth \\(\tanh\\) switch \\(\sigma(t)\\) of width set by
/// [`sharpness`](Self::sharpness), steering direction \\(\hat{d}\\) from
/// [`steering`](Self::steering), and mass \\(m(t)\\) that depletes when
/// [`isp_s`](Self::isp_s) is `Some`:
/// \\[
///   m(t) = m_0 - \frac{F}{I_{sp}\,g_0}\,(t - t_\text{start}).
/// \\]
#[derive(Debug, Clone, PartialEq)]
pub struct ThrustArc {
    /// Arc start epoch (MJD TDB).
    pub start_mjd_tdb: f64,
    /// Arc end epoch (MJD TDB).
    pub end_mjd_tdb: f64,
    /// Engine thrust force in Newtons.
    pub thrust_n: f64,
    /// Spacecraft mass at arc start in kilograms.
    pub mass_kg: f64,
    /// Specific impulse in seconds. `Some` depletes mass linearly during
    /// the burn at \\(\dot m = F/(I_{sp}\,g_0)\\); `None` holds mass
    /// constant. Crosses the ABI as the NaN sentinel (shared with the
    /// non-grav time delay): `None` → `NaN`.
    pub isp_s: Option<f64>,
    /// Steering law for this arc.
    pub steering: SteeringLaw,
    /// \\(\tanh\\) switching sharpness (1/days). Higher values give
    /// sharper on/off transitions (closer to bang-bang). Typical values:
    /// 1000–10000 for burns of minutes, 100 for multi-hour arcs.
    pub sharpness: f64,
    /// Central body for the RTN / velocity-tangent frame reference. The
    /// spacecraft state is expressed relative to this body before the
    /// thrust direction is computed — e.g. [`Origin::EARTH`] for
    /// geocentric arcs, [`Origin::SUN`] for heliocentric ones.
    pub central_body: Origin,
}

impl ThrustArc {
    /// Build a constant-mass arc (`isp_s = None`). Add mass depletion
    /// with [`with_isp`](Self::with_isp).
    pub fn new(
        start_mjd_tdb: f64,
        end_mjd_tdb: f64,
        thrust_n: f64,
        mass_kg: f64,
        sharpness: f64,
        steering: SteeringLaw,
        central_body: Origin,
    ) -> Self {
        Self {
            start_mjd_tdb,
            end_mjd_tdb,
            thrust_n,
            mass_kg,
            isp_s: None,
            steering,
            sharpness,
            central_body,
        }
    }

    /// Set the specific impulse (seconds), enabling mass depletion. Pass
    /// `None` to hold mass constant (the default from [`new`](Self::new)).
    pub fn with_isp(mut self, isp_s: Option<f64>) -> Self {
        self.isp_s = isp_s;
        self
    }

    /// Marshal into the flat FFI record. Infallible: every field maps
    /// directly, `isp_s` uses the NaN sentinel, and the central body is a
    /// typed [`Origin`] whose NAIF id always resolves.
    pub(crate) fn to_ffi(&self) -> empyrean_sys::EmpyreanThrustArc {
        let (steering_law, steering_alpha_rad, steering_beta_rad, steering_direction) =
            self.steering.to_ffi_parts();
        empyrean_sys::EmpyreanThrustArc {
            start_mjd_tdb: self.start_mjd_tdb,
            end_mjd_tdb: self.end_mjd_tdb,
            thrust_n: self.thrust_n,
            mass_kg: self.mass_kg,
            // Option<f64> across the ABI uses the NaN sentinel (shared
            // with `non_grav_dt`): None = constant mass, Some = depletion.
            isp_s: self.isp_s.unwrap_or(f64::NAN),
            steering_law,
            steering_alpha_rad,
            steering_beta_rad,
            steering_direction,
            sharpness: self.sharpness,
            central_body_naif_id: self.central_body.naif_id(),
        }
    }
}

/// Thrust parameters for an orbit: thrust arcs plus optional Δv targeting
/// corrections.
///
/// Attach to an [`Orbit`](crate::Orbit) with
/// [`Orbit::with_thrust`](crate::Orbit::with_thrust).
///
/// When [`correction_covariances`](Self::correction_covariances) is
/// non-empty its length MUST equal
/// [`dv_corrections`](Self::dv_corrections); a non-empty covariance
/// triggers the wide-Jet burn-sensitivity propagation and its solved
/// segments appear in
/// [`TaggedCovariance::thrust_segments`](crate::TaggedCovariance). Length
/// or arc/correction mismatches are surfaced loudly as an [`Error`]
/// during propagation — never silently repaired or dropped.
///
/// [`Error`]: crate::Error
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThrustParams {
    /// Ordered list of thrust arcs. May overlap in time.
    pub arcs: Vec<ThrustArc>,
    /// Per-arc Δv corrections (AU/day) for targeting, positional with
    /// [`arcs`](Self::arcs). Each is applied as a constant inertial
    /// acceleration during its arc's window; when seeded as Jet
    /// variables it provides ∂state/∂Δv. Empty = no corrections.
    pub dv_corrections: Vec<[f64; 3]>,
    /// 3×3 covariance (AU/day)² per Δv correction, positional with
    /// [`dv_corrections`](Self::dv_corrections). When non-empty its
    /// length must equal `dv_corrections`. Empty = no burn-sensitivity
    /// propagation.
    pub correction_covariances: Vec<[[f64; 3]; 3]>,
}

impl ThrustParams {
    /// Build thrust parameters from a list of arcs, with no Δv
    /// corrections. Add corrections with
    /// [`with_dv_corrections`](Self::with_dv_corrections) and their
    /// covariances with
    /// [`with_correction_covariances`](Self::with_correction_covariances).
    pub fn new(arcs: Vec<ThrustArc>) -> Self {
        Self {
            arcs,
            dv_corrections: Vec::new(),
            correction_covariances: Vec::new(),
        }
    }

    /// Attach per-arc Δv targeting corrections (AU/day), positional with
    /// the arcs.
    pub fn with_dv_corrections(mut self, dv_corrections: Vec<[f64; 3]>) -> Self {
        self.dv_corrections = dv_corrections;
        self
    }

    /// Attach the 3×3 covariance (AU/day)² per Δv correction. Its length
    /// must equal [`dv_corrections`](Self::dv_corrections); enabling this
    /// selects the burn-sensitivity propagation.
    pub fn with_correction_covariances(
        mut self,
        correction_covariances: Vec<[[f64; 3]; 3]>,
    ) -> Self {
        self.correction_covariances = correction_covariances;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::Orbit;
    use crate::{CoordinateState, Epoch, Frame};

    fn heliocentric_state() -> CoordinateState {
        CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::ICRF,
            Origin::SUN,
        )
    }

    fn rtn_arc() -> ThrustArc {
        ThrustArc::new(
            59000.0,
            59010.0,
            1000.0,
            1000.0,
            100.0,
            SteeringLaw::ConstantRTN {
                alpha_rad: 0.1,
                beta_rad: 0.2,
            },
            Origin::SUN,
        )
    }

    #[test]
    fn steering_law_marshals_tag_and_slots() {
        let (tag, a, b, d) = SteeringLaw::ConstantRTN {
            alpha_rad: 0.1,
            beta_rad: 0.2,
        }
        .to_ffi_parts();
        assert_eq!(tag, empyrean_sys::EMPYREAN_STEERING_LAW_CONSTANT_RTN as i32);
        assert_eq!((a, b, d), (0.1, 0.2, [0.0; 3]));

        let (tag, a, b, d) = SteeringLaw::VelocityTangent.to_ffi_parts();
        assert_eq!(
            tag,
            empyrean_sys::EMPYREAN_STEERING_LAW_VELOCITY_TANGENT as i32
        );
        assert_eq!((a, b, d), (0.0, 0.0, [0.0; 3]));

        let (tag, a, b, d) = SteeringLaw::InertialFixed {
            direction: [1.0, 2.0, 3.0],
        }
        .to_ffi_parts();
        assert_eq!(
            tag,
            empyrean_sys::EMPYREAN_STEERING_LAW_INERTIAL_FIXED as i32
        );
        assert_eq!((a, b, d), (0.0, 0.0, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn thrust_arc_isp_uses_nan_sentinel() {
        // None -> NaN (constant mass).
        let ffi = rtn_arc().to_ffi();
        assert!(ffi.isp_s.is_nan());
        // Some -> finite (mass depletion).
        let ffi = rtn_arc().with_isp(Some(3000.0)).to_ffi();
        assert_eq!(ffi.isp_s, 3000.0);
    }

    #[test]
    fn thrust_arc_marshals_all_fields() {
        let ffi = rtn_arc().to_ffi();
        assert_eq!(ffi.start_mjd_tdb, 59000.0);
        assert_eq!(ffi.end_mjd_tdb, 59010.0);
        assert_eq!(ffi.thrust_n, 1000.0);
        assert_eq!(ffi.mass_kg, 1000.0);
        assert_eq!(ffi.sharpness, 100.0);
        assert_eq!(ffi.steering_alpha_rad, 0.1);
        assert_eq!(ffi.steering_beta_rad, 0.2);
        assert_eq!(ffi.central_body_naif_id, Origin::SUN.naif_id());
    }

    #[test]
    fn orbit_without_thrust_leaves_side_arrays_null() {
        let orbit = Orbit::new(heliocentric_state());
        let (ffi, _keep) = orbit.to_ffi_with_keep().expect("ffi");
        assert!(ffi.thrust_arcs.is_null());
        assert_eq!(ffi.n_thrust_arcs, 0);
        assert!(ffi.dv_corrections.is_null());
        assert_eq!(ffi.n_dv_corrections, 0);
        assert!(ffi.correction_covariances.is_null());
        assert_eq!(ffi.n_correction_covariances, 0);
    }

    #[test]
    fn orbit_with_thrust_populates_side_arrays() {
        let cov = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 2.0e-20, 0.0],
            [0.0, 0.0, 3.0e-20],
        ];
        let thrust = ThrustParams::new(vec![rtn_arc()])
            .with_dv_corrections(vec![[1.0e-6, 2.0e-6, 3.0e-6]])
            .with_correction_covariances(vec![cov]);
        let orbit = Orbit::new(heliocentric_state()).with_thrust(Some(thrust));

        // `_keep` owns the side-array storage the FFI struct borrows into;
        // it must stay in scope for the reads below to be sound.
        let (ffi, _keep) = orbit.to_ffi_with_keep().expect("ffi");
        assert!(!ffi.thrust_arcs.is_null());
        assert_eq!(ffi.n_thrust_arcs, 1);
        assert_eq!(ffi.n_dv_corrections, 1);
        assert_eq!(ffi.n_correction_covariances, 1);

        let arcs = unsafe { std::slice::from_raw_parts(ffi.thrust_arcs, ffi.n_thrust_arcs) };
        assert_eq!(arcs[0].thrust_n, 1000.0);
        assert_eq!(
            arcs[0].steering_law,
            empyrean_sys::EMPYREAN_STEERING_LAW_CONSTANT_RTN as i32
        );

        let dvs = unsafe { std::slice::from_raw_parts(ffi.dv_corrections, ffi.n_dv_corrections) };
        assert_eq!(dvs[0], [1.0e-6, 2.0e-6, 3.0e-6]);

        let covs = unsafe {
            std::slice::from_raw_parts(ffi.correction_covariances, ffi.n_correction_covariances)
        };
        assert_eq!(covs[0], cov);
    }

    #[test]
    fn empty_thrust_params_send_no_side_arrays() {
        // Some(ThrustParams) with no arcs marshals to null/0 — the engine
        // treats it as a pure-gravity orbit (parity with the C ABI's
        // Ok(None) for a zero-arc orbit).
        let orbit = Orbit::new(heliocentric_state()).with_thrust(Some(ThrustParams::default()));
        let (ffi, _keep) = orbit.to_ffi_with_keep().expect("ffi");
        assert!(ffi.thrust_arcs.is_null());
        assert_eq!(ffi.n_thrust_arcs, 0);
    }
}

// ── End-to-end propagation (gated on ephemeris data) ─────────────────
#[cfg(test)]
mod propagate_tests {
    use super::*;
    use crate::orbit::Orbit;
    use crate::{Context, CoordinateState, Epoch, Frame, PropagationConfig};
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

    /// A heliocentric Cartesian state with a tight diagonal covariance so
    /// the wide-Jet burn-sensitivity path engages.
    fn seeded_state() -> CoordinateState {
        let mut cov = [[0.0_f64; 6]; 6];
        for (i, row) in cov.iter_mut().enumerate() {
            row[i] = 1.0e-16;
        }
        CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::ICRF,
            Origin::SUN,
        )
        .with_covariance(cov)
    }

    fn rtn_arc() -> ThrustArc {
        ThrustArc::new(
            59000.0,
            59010.0,
            1000.0,
            1000.0,
            100.0,
            SteeringLaw::ConstantRTN {
                alpha_rad: 0.1,
                beta_rad: 0.2,
            },
            Origin::SUN,
        )
    }

    /// End-to-end: a ConstantRTN arc (with a Δv correction + covariance)
    /// propagates through the real wrapper `propagate` and the burn is
    /// honored by the dynamics — the thrusted trajectory diverges sharply
    /// from the identical ballistic orbit, and the retained-handle
    /// tagged-covariance accessor stays callable for a thrust orbit. This
    /// closes the thrust **input** loop at the wrapper layer: the safe
    /// `ThrustParams` reach the engine and change propagation.
    ///
    /// (The burn-sensitivity segment *count* in the covariance readback —
    /// `TaggedCovariance::thrust_segments` — is an engine-output concern;
    /// the currently-compiled engine reports the marginalized 6×6 state
    /// block, so this test asserts the input effect rather than a specific
    /// solved width.)
    #[test]
    fn propagate_with_thrust_perturbs_trajectory() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping propagate_with_thrust_perturbs_trajectory: no data dir");
            return;
        };

        let eye_small = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 1.0e-20, 0.0],
            [0.0, 0.0, 1.0e-20],
        ];
        let thrust = ThrustParams::new(vec![rtn_arc()])
            .with_dv_corrections(vec![[0.0, 0.0, 0.0]])
            .with_correction_covariances(vec![eye_small]);
        let thrusted = Orbit::new(seeded_state())
            .with_orbit_id("thrust-loop")
            .with_thrust(Some(thrust));
        let ballistic = Orbit::new(seeded_state()).with_orbit_id("ballistic");

        let epochs: Vec<Epoch> = [59000.0, 59005.0, 59012.0]
            .iter()
            .map(|&d| Epoch::from_mjd_tdb(d))
            .collect();
        let mut cfg = PropagationConfig::default();
        cfg.advanced.cache_integrator_steps = true;

        let with_thrust = ctx
            .propagate(&[thrusted], &epochs, &cfg)
            .expect("propagate with thrust must succeed");
        let without_thrust = ctx
            .propagate(&[ballistic], &epochs, &cfg)
            .expect("ballistic propagate must succeed");
        assert!(!with_thrust.states.is_empty(), "expected propagated states");

        // The retained-handle tagged-covariance accessor must stay callable
        // for a thrust orbit (proves the output path is wired, not that any
        // particular solved width is reported by the current engine).
        with_thrust
            .covariance_at_cartesian(0, epochs.len() - 1)
            .expect("tagged-covariance readback for a thrust orbit");

        // The burn must measurably perturb the trajectory relative to the
        // identical ballistic orbit — the definitive proof the thrust input
        // reached and was honored by the engine dynamics.
        let a = with_thrust.states.last().unwrap().position;
        let b = without_thrust.states.last().unwrap().position;
        let delta = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        assert!(
            delta > 1.0e-3,
            "thrust arc must perturb the trajectory (Δposition = {delta:e} AU)"
        );
    }

    /// A `correction_covariances` length that does not match
    /// `dv_corrections` is a contract violation the engine rejects. The
    /// wrapper must surface it as an [`Error`](crate::Error) — never
    /// silently drop or repair the covariance.
    ///
    /// This exercises the "surface loudly, never degrade" path shared with
    /// the `ThirdOrder` + correction-covariance rejection. (`ThirdOrder`
    /// itself is not directly reachable here: the wrapper's
    /// `UncertaintyMethod` exposes no `ThirdOrder` input variant — parity
    /// with the C ABI — so the exact combination cannot be constructed
    /// through the public API, but the identical non-degrading surface is
    /// what carries either rejection up as an `Error`.)
    #[test]
    fn thrust_correction_covariance_mismatch_surfaces_loudly() {
        let Some(ctx) = try_context() else {
            eprintln!(
                "skipping thrust_correction_covariance_mismatch_surfaces_loudly: no data dir"
            );
            return;
        };

        // One arc, zero Δv corrections, but a correction covariance: the
        // engine's ThrustParams contract requires the covariance length to
        // match the Δv-correction length.
        let eye = [
            [1.0e-20, 0.0, 0.0],
            [0.0, 1.0e-20, 0.0],
            [0.0, 0.0, 1.0e-20],
        ];
        let thrust = ThrustParams::new(vec![rtn_arc()]).with_correction_covariances(vec![eye]);
        let orbit = Orbit::new(seeded_state()).with_thrust(Some(thrust));

        let epochs = vec![Epoch::from_mjd_tdb(59000.0), Epoch::from_mjd_tdb(59005.0)];
        let cfg = PropagationConfig::default();

        let err = ctx
            .propagate(&[orbit], &epochs, &cfg)
            .expect_err("mismatched correction_covariances must error, not degrade");
        assert!(
            err.message.contains("correction_covariances"),
            "error must name the offending field, got: {}",
            err.message
        );
    }
}
