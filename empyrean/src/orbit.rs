//! Orbit input type (coordinate state + non-grav parameters).

use crate::coordinate::CoordinateState;
use crate::thrust::{ThrustArc, ThrustParams};

/// Phase-function model for HG-family photometry.
///
/// Mirrors `villeneuve::photometry::PhaseFunction`. The integer codes
/// match the corresponding `EMPYREAN_PHASE_FUNCTION_*` C-ABI constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PhaseFunction {
    /// HG (two-parameter) — Bowell et al. 1989. Default for asteroids.
    HG = 0,
    /// HG1G2 (three-parameter) — Muinonen et al. 2010.
    HG1G2 = 1,
    /// HG12 (two-parameter, single-slope) — Muinonen et al. 2010.
    HG12 = 2,
}

/// Solar-radiation-pressure force slot: an area-to-mass ratio (the fittable
/// AMRAT), a fixed radiation-pressure coefficient Cr, and an optional prior
/// variance that makes AMRAT a fittable parameter. An additive force slot —
/// combinable with the Marsden non-grav on the same orbit.
///
/// Only the product \\(C_r \cdot \text{AMRAT}\\) enters the dynamics, so a
/// fitted AMRAT absorbs Cr (the JPL AMR convention); Cr is fixed, never fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrpParams {
    /// Area-to-mass ratio AMRAT (m²/kg) — the SRP-effective, fittable
    /// parameter. Must be finite and > 0.
    pub amrat: f64,
    /// Radiation-pressure coefficient Cr (typically 1.0–2.0). Fixed, never
    /// fitted. Must be finite and > 0.
    pub cr: f64,
    /// Prior variance on AMRAT ((m²/kg)²). `Some(v)` with `v > 0` opens and
    /// priors the AMRAT column in a `StateAndAMRAT` / `StateAndNonGravAndAMRAT`
    /// refine; `None` applies SRP as a fixed force.
    pub amrat_variance: Option<f64>,
}

impl SrpParams {
    /// Reconstruct an SRP slot from a C-ABI orbit's SRP fields (`has_srp` +
    /// `srp_amrat` / `srp_cr` / `srp_amrat_variance`). `has_srp == 0` ⇒ `None`;
    /// a finite positive variance ⇒ AMRAT carries a fittable prior, else the
    /// SRP is a fixed force. Shared by every C→`Orbit` reverse-marshal path.
    pub(crate) fn from_ffi(amrat: f64, cr: f64, has_srp: u8, amrat_variance: f64) -> Option<Self> {
        if has_srp == 0 {
            return None;
        }
        Some(Self {
            amrat,
            cr,
            amrat_variance: if amrat_variance.is_finite() && amrat_variance > 0.0 {
                Some(amrat_variance)
            } else {
                None
            },
        })
    }
}

/// Orbit to propagate: coordinate state plus optional Marsden non-grav
/// coefficients (A1, A2, A3), a configurable g(r) distance scaling, and
/// optional photometric parameters.
///
/// The default g(r) (when [`Orbit::with_nongrav`] is used without an
/// explicit g(r) selector) is **inverse-square**, the standard
/// Yarkovsky / SRP case for asteroids. Comets with SBDB-supplied
/// water-ice or custom parameters should use [`Orbit::with_g_function`].
///
/// Photometry: when [`Orbit::with_photometry`] is used, ephemeris
/// generation produces apparent magnitude using the (`H`, slope1, slope2)
/// triple per the chosen phase function.
///
/// Identification: `orbit_id` and `object_id` thread through every
/// downstream output (propagated states, events, ephemeris, B-planes,
/// impact probabilities). `None` falls back to a synthetic positional
/// `"orbit_{i}"` tag at the C ABI layer. Set them via
/// [`Orbit::with_orbit_id`] / [`Orbit::with_object_id`] so that
/// downstream tooling can join results back to their input rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Orbit {
    /// Caller-supplied orbit identifier — primary key for joining
    /// outputs back to inputs. When `None`, the C ABI fabricates a
    /// positional `"orbit_{i}"` tag.
    pub orbit_id: Option<String>,
    /// Caller-supplied object identifier — typically the SBDB
    /// designation or full provisional name. Distinct from `orbit_id`
    /// so that multiple orbit hypotheses for the same object can share
    /// an `object_id`.
    pub object_id: Option<String>,
    /// Initial coordinate state.
    pub state: CoordinateState,
    /// Radial non-grav coefficient (AU/day²). Zero if unused.
    pub a1: f64,
    /// Transverse non-grav coefficient (AU/day²). Zero if unused.
    pub a2: f64,
    /// Normal non-grav coefficient (AU/day²). Zero if unused.
    pub a3: f64,
    /// g(r) function parameter α (normalizing constant).
    /// All-zeros across (alpha, r0, m, n, k) selects the inverse-square
    /// default at the FFI layer; otherwise the explicit values are used
    /// to build a Marsden g(r).
    pub ng_alpha: f64,
    /// g(r) reference distance r₀ (AU).
    pub ng_r0: f64,
    /// g(r) inner power-law exponent m.
    pub ng_m: f64,
    /// g(r) outer power-law exponent n.
    pub ng_n: f64,
    /// g(r) outer damping exponent k.
    pub ng_k: f64,
    /// SBDB-fit time delay (days) applied to the g(r) evaluation. The
    /// distance-dependent scaling \\(g(r)\\) is evaluated at the
    /// Keplerian-back-propagated position \\(r(t - \Delta T)\\) instead
    /// of \\(r(t)\\). Models thermal-inertia outgassing lag at perihelion.
    /// `None` is the asteroid default (no time delay); SBDB populates
    /// this for some Jupiter-family comets and 2I/Borisov.
    pub non_grav_dt: Option<f64>,
    /// Prior variance on the non-grav time delay DT (days²). `None` = no
    /// prior (the DT column stays closed); `Some(v)` with `v > 0` opens and
    /// priors the DT column in a `StateAndNonGravAndDT` refine.
    pub non_grav_dt_variance: Option<f64>,
    /// Fitted non-grav 3×3 covariance for (A1, A2, A3), row-major. Set by
    /// the OD output path (a fitted orbit) so the orbit re-feeds into a
    /// `StateAndNonGrav` refine without losing its non-grav prior.
    /// `None` for hand-built / SBDB / propagate inputs.
    pub ng_covariance: Option<[[f64; 3]; 3]>,
    /// Phase function. `None` disables magnitude computation in ephemeris
    /// generation; the corresponding row gets `mag = NaN`.
    pub phot_system: Option<PhaseFunction>,
    /// Absolute magnitude H. Only honored when `phot_system` is `Some`.
    pub h_mag: f64,
    /// Slope parameter slot 1 — G (HG), G₁ (HG1G2), or G₁₂ (HG12).
    pub slope1: f64,
    /// Slope parameter slot 2 — G₂ (HG1G2 only); 0 for HG / HG12.
    pub slope2: f64,
    /// Photometric 3×3 covariance over (H, slope1, slope2), row-major —
    /// the same parameter order as [`h_mag`](Self::h_mag) /
    /// [`slope1`](Self::slope1) / [`slope2`](Self::slope2), so which
    /// slope a row names follows [`phot_system`](Self::phot_system).
    ///
    /// Set it from a photometry fit
    /// ([`PhotometryResult::covariance`](crate::PhotometryResult)), an
    /// SBDB query, or an orbit file that carried one, and ephemeris
    /// generation adds the \\(\sigma_{\text{photo}}\\) term to
    /// `mag_sigma`. `None` (the default for a hand-built orbit) leaves
    /// `mag_sigma` carrying the state contribution alone.
    ///
    /// Rows and columns of parameters the producing fit held fixed are
    /// zero, which is what an H-only fit emits.
    ///
    /// Must be **symmetric**; checked at the C-ABI boundary (see
    /// [`with_photometry_covariance`](Self::with_photometry_covariance)).
    /// The orbit file formats carry only the six lower-triangle cells
    /// and mirror them on read, so an asymmetric matrix has no file
    /// representation — which is why it is refused rather than written
    /// short.
    pub phot_covariance: Option<[[f64; 3]; 3]>,
    /// Continuous-thrust parameters (arcs + optional Δv corrections).
    /// `None` is a pure gravity + non-grav orbit. When
    /// [`ThrustParams::correction_covariances`] is non-empty the
    /// resulting burn-sensitivity segments surface in the propagated
    /// [`TaggedCovariance::thrust_segments`](crate::TaggedCovariance).
    pub thrust: Option<ThrustParams>,
    /// Solar-radiation-pressure slot. `None` = no SRP. Additive with the
    /// Marsden non-grav above; a `Some` with `amrat_variance` set opens the
    /// AMRAT column in a `StateAndAMRAT` / `StateAndNonGravAndAMRAT` refine.
    pub srp: Option<SrpParams>,
    /// Cross-covariance terms beyond the state+Marsden \\(9 \\times 9\\) —
    /// state↔DT, state↔AMRAT, state↔\\(\Delta v\\), and every mixed
    /// parameter pair.
    ///
    /// `None` is an orbit with no such terms, which is what every
    /// hand-built and SBDB orbit is. Set it from a fit's own
    /// [`JointCovariance`](crate::JointCovariance) to hand the next leg
    /// the joint that fit computed rather than its diagonal blocks.
    ///
    /// The state↔Marsden border is NOT here — it rides
    /// [`CoordinateState::non_grav_cross`](crate::CoordinateState::non_grav_cross),
    /// beside the 6×6 it borders, so the two halves of one matrix travel
    /// together through a transform.
    pub wide_cross: Option<crate::WideCross>,
}

impl Orbit {
    /// Build an orbit with no non-grav terms, no photometry, and no
    /// orbit_id / object_id tags.
    pub fn new(state: CoordinateState) -> Self {
        Self {
            orbit_id: None,
            object_id: None,
            state,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
            ng_alpha: 0.0,
            ng_r0: 0.0,
            ng_m: 0.0,
            ng_n: 0.0,
            ng_k: 0.0,
            non_grav_dt: None,
            non_grav_dt_variance: None,
            ng_covariance: None,
            phot_system: None,
            h_mag: f64::NAN,
            slope1: 0.0,
            slope2: 0.0,
            phot_covariance: None,
            thrust: None,
            srp: None,
            wide_cross: None,
        }
    }

    /// Attach an orbit identifier — the primary key by which downstream
    /// outputs (states, events, ephemeris, B-planes, IP) are joined
    /// back to the input row.
    pub fn with_orbit_id(mut self, id: impl Into<String>) -> Self {
        self.orbit_id = Some(id.into());
        self
    }

    /// Attach an object identifier — typically the SBDB designation.
    /// Carried through every downstream output alongside `orbit_id`.
    pub fn with_object_id(mut self, id: impl Into<String>) -> Self {
        self.object_id = Some(id.into());
        self
    }

    /// Attach Marsden non-grav coefficients. Defaults to the
    /// inverse-square g(r) (asteroid Yarkovsky / SRP); pair with
    /// [`Orbit::with_g_function`] when SBDB provides comet-specific
    /// values.
    pub fn with_nongrav(mut self, a1: f64, a2: f64, a3: f64) -> Self {
        self.a1 = a1;
        self.a2 = a2;
        self.a3 = a3;
        self
    }

    /// Attach an explicit Marsden g(r) parameter set
    /// \\((\alpha, r_0, m, n, k)\\). Common SBDB values:
    /// inverse-square (asteroid default — leave unset / `(1, 1, 2, 0, 0)`),
    /// water-ice (Marsden comets — `(0.1113, 2.808, 2.15, 5.093, 4.6142)`).
    pub fn with_g_function(mut self, alpha: f64, r0: f64, m: f64, n: f64, k: f64) -> Self {
        self.ng_alpha = alpha;
        self.ng_r0 = r0;
        self.ng_m = m;
        self.ng_n = n;
        self.ng_k = k;
        self
    }

    /// Set the SBDB non-grav time delay (days). Pass `None` to disable
    /// (the default; appropriate for asteroids and short-period comets
    /// SBDB doesn't fit a delay for). Pass `Some(dt)` for objects where
    /// SBDB's `model_pars[]` exposes a `DT` field — Jupiter-family
    /// comets like 67P (DT≈+46d), 46P/Wirtanen (−14d), 103P/Hartley 2
    /// (+12d), and 2I/Borisov (−65d) are the common cases.
    pub fn with_non_grav_dt(mut self, dt: Option<f64>) -> Self {
        self.non_grav_dt = dt;
        self
    }

    /// Set the prior variance on the non-grav time delay DT (days²). Pass
    /// `None` (the default) to leave the DT column closed; pass `Some(v)`
    /// with `v > 0` to open and prior the DT column in a
    /// `StateAndNonGravAndDT` refine.
    pub fn with_non_grav_dt_variance(mut self, v: Option<f64>) -> Self {
        self.non_grav_dt_variance = v;
        self
    }

    /// Attach the fitted non-grav 3×3 covariance for (A1, A2, A3). Set by
    /// the OD output path so a fitted orbit re-feeds into a `StateAndNonGrav`
    /// refine without losing its non-grav prior.
    pub fn with_nongrav_covariance(mut self, covariance: Option<[[f64; 3]; 3]>) -> Self {
        self.ng_covariance = covariance;
        self
    }

    /// Attach photometric parameters. The slot mapping depends on the
    /// model — see [`PhaseFunction`].
    ///
    /// `h` must be a real absolute magnitude: `0.0` is the C ABI's
    /// "no absolute magnitude supplied" value (what `memset(0)` leaves
    /// on `EmpyreanOrbit::h_mag`), so an orbit carrying it is refused by
    /// name when it crosses the boundary rather than losing its
    /// photometry there.
    pub fn with_photometry(
        mut self,
        phot_system: PhaseFunction,
        h: f64,
        slope1: f64,
        slope2: f64,
    ) -> Self {
        self.phot_system = Some(phot_system);
        self.h_mag = h;
        self.slope1 = slope1;
        self.slope2 = slope2;
        self
    }

    /// Attach the photometric 3×3 covariance over (H, slope1, slope2).
    /// Pass `None` (the default) for an orbit with no photometric
    /// uncertainty.
    ///
    /// The companion to [`Orbit::with_photometry`], in the same shape as
    /// [`Orbit::with_nongrav_covariance`] /
    /// [`Orbit::with_non_grav_dt_variance`] /
    /// [`Orbit::with_srp_amrat_variance`]: the parameters and their
    /// covariance are set separately because most orbits carry the
    /// former without the latter.
    ///
    /// With it attached, ephemeris generation reports
    /// \\[
    ///   \sigma_V = \sqrt{\sigma^2_{\text{photo}} + \sigma^2_{\text{state}}}
    /// \\]
    /// rather than the state term alone, with
    /// \\(\sigma^2_{\text{photo}} = J \Sigma_p J^\top\\) contracting the
    /// full \\(3 \times 3\\) against
    /// \\(J = [\partial V/\partial H, \partial V/\partial \text{slope}_1,
    /// \partial V/\partial \text{slope}_2]\\). Because
    /// \\(V = H + 5\log_{10}(r\Delta) + \phi(\alpha)\\) gives
    /// \\(\partial V/\partial H \equiv 1\\), an orbit with no state
    /// covariance and a covariance of the H-only shape
    /// \\(\mathrm{diag}(\sigma_H^2, 0, 0)\\) reports
    /// \\(\sigma_V = \sigma_H\\) exactly; a covariance carrying slope
    /// variances or \\(H\\)–slope covariances reports
    /// \\(\sigma_V > \sigma_H\\), because those terms contract against
    /// \\(\partial V/\partial\text{slope}\\), which vanishes only at
    /// zero phase angle.
    ///
    /// The matrix must be **symmetric** — it is a covariance, and the
    /// orbit file formats store only its lower triangle, so an
    /// asymmetric matrix could not round-trip. Symmetry is checked when
    /// the orbit crosses the C ABI (propagate / ephemeris / impact / OD
    /// / file write), which fails loudly naming the offending pair
    /// rather than silently symmetrizing or truncating.
    ///
    /// ```no_run
    /// # use empyrean::{Orbit, PhaseFunction};
    /// # fn f(orbit: Orbit) -> Orbit {
    /// orbit
    ///     .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
    ///     .with_photometry_covariance(Some([
    ///         [0.09, 0.0, 0.0],
    ///         [0.0, 0.0025, 0.0],
    ///         [0.0, 0.0, 0.0],
    ///     ]))
    /// # }
    /// ```
    pub fn with_photometry_covariance(mut self, covariance: Option<[[f64; 3]; 3]>) -> Self {
        self.phot_covariance = covariance;
        self
    }

    /// Convenience: attach HG photometry (the asteroid default).
    pub fn with_hg(self, h: f64, g: f64) -> Self {
        self.with_photometry(PhaseFunction::HG, h, g, 0.0)
    }

    /// Attach continuous-thrust parameters (arcs + optional Δv
    /// corrections). Pass `None` for a pure gravity + non-grav orbit.
    /// A non-empty [`ThrustParams::correction_covariances`] engages the
    /// burn-sensitivity propagation whose solved segments appear in the
    /// output [`TaggedCovariance::thrust_segments`](crate::TaggedCovariance).
    pub fn with_thrust(mut self, thrust: Option<ThrustParams>) -> Self {
        self.thrust = thrust;
        self
    }

    /// Attach a solar-radiation-pressure slot (a fixed force) with the given
    /// area-to-mass ratio AMRAT (m²/kg) and radiation coefficient Cr. Combine
    /// with a Marsden non-grav on the same orbit via [`Orbit::with_nongrav`].
    /// Make AMRAT fittable with [`Orbit::with_srp_amrat_variance`].
    pub fn with_srp(mut self, amrat: f64, cr: f64) -> Self {
        let amrat_variance = self.srp.and_then(|s| s.amrat_variance);
        self.srp = Some(SrpParams {
            amrat,
            cr,
            amrat_variance,
        });
        self
    }

    /// Make the SRP AMRAT a fittable parameter by attaching a prior variance
    /// ((m²/kg)²) — the trigger + Bayesian prior that opens the AMRAT column in
    /// a `StateAndAMRAT` / `StateAndNonGravAndAMRAT` refine. Requires an SRP
    /// slot first ([`Orbit::with_srp`]); a no-op if none is set.
    pub fn with_srp_amrat_variance(mut self, v: Option<f64>) -> Self {
        if let Some(srp) = self.srp.as_mut() {
            srp.amrat_variance = v;
        }
        self
    }

    /// Convert to an FFI struct, returning the C struct alongside a
    /// keepalive bag that owns the heap-allocated identifier strings.
    ///
    /// The FFI struct holds raw `*const c_char` pointers into the
    /// keepalive's `CString` storage; the keepalive must outlive every
    /// use of the returned `EmpyreanOrbit`.
    pub(crate) fn to_ffi_with_keep(
        &self,
    ) -> crate::error::Result<(empyrean_sys::EmpyreanOrbit, OrbitFfiKeep)> {
        use std::ffi::CString;
        if let Some(cov) = self.phot_covariance {
            validate_photometry_covariance_symmetry(&cov)?;
        }
        let (phase_int, h, s1, s2) = match self.phot_system {
            Some(pf) => {
                // The C ABI reads `h_mag == 0.0` as "no photometry
                // supplied" — it is what `memset(0)` leaves there, and
                // `EMPYREAN_PHASE_FUNCTION_HG` is 0 too, so a
                // zero-initialized orbit would otherwise claim HG at
                // H = 0. A wrapper `Orbit` states presence in its own
                // `Option`, so the two disagree here, and the honest
                // move is to say so rather than let the photometry
                // vanish at the boundary.
                if self.h_mag == 0.0 {
                    return Err(crate::error::Error::invalid_input(
                        "photometry was attached with h_mag = 0.0, which the C ABI reads as \
                         'no absolute magnitude supplied' (it is the memset(0) value). No \
                         solar-system minor body has H = 0 — supply the real absolute \
                         magnitude, or attach no photometry at all.",
                    ));
                }
                (pf as i32, self.h_mag, self.slope1, self.slope2)
            }
            None => (-1, f64::NAN, 0.0, 0.0),
        };
        // Empty CString for absent ids — the C side checks the pointer's
        // first byte for an explicit "id absent" sentinel without having
        // to handle null. Using `CString::default()` avoids fallible
        // construction.
        let orbit_id_cstr =
            CString::new(self.orbit_id.as_deref().unwrap_or("")).unwrap_or_default();
        let object_id_cstr =
            CString::new(self.object_id.as_deref().unwrap_or("")).unwrap_or_default();
        let orbit_id_ptr = orbit_id_cstr.as_ptr();
        let object_id_ptr = object_id_cstr.as_ptr();
        // Marshal the optional thrust params into caller-owned side arrays
        // held by the keepalive. The `EmpyreanOrbit` borrows raw pointers
        // into these Vecs — a Vec's heap buffer address is stable across
        // the move into `OrbitFfiKeep` and the outer keep vector, so the
        // pointers stay valid for the FFI call (same contract as the id
        // CStrings above). Absent thrust → null pointers + zero lengths.
        let thrust_arcs: Vec<empyrean_sys::EmpyreanThrustArc> = self
            .thrust
            .as_ref()
            .map(|tp| tp.arcs.iter().map(ThrustArc::to_ffi).collect())
            .unwrap_or_default();
        let dv_corrections: Vec<[f64; 3]> = self
            .thrust
            .as_ref()
            .map(|tp| tp.dv_corrections.clone())
            .unwrap_or_default();
        let correction_covariances: Vec<[[f64; 3]; 3]> = self
            .thrust
            .as_ref()
            .map(|tp| tp.correction_covariances.clone())
            .unwrap_or_default();
        // `slice::as_ptr` on an empty Vec is non-null (dangling); the C
        // ABI keys off the length, but keep the null/0 pairing explicit so
        // an empty side array reads as "absent", matching every other
        // C-ABI path.
        let thrust_arcs_ptr = if thrust_arcs.is_empty() {
            std::ptr::null()
        } else {
            thrust_arcs.as_ptr()
        };
        let dv_corrections_ptr = if dv_corrections.is_empty() {
            std::ptr::null()
        } else {
            dv_corrections.as_ptr()
        };
        let correction_covariances_ptr = if correction_covariances.is_empty() {
            std::ptr::null()
        } else {
            correction_covariances.as_ptr()
        };
        // The wide carrier's two side arrays follow the same
        // caller-owned / borrowed-for-the-call contract as the thrust
        // arrays above, and the keepalive owns them for the same reason.
        let (state_param_cross, param_pair_cross) = match self.wide_cross.as_ref() {
            Some(w) => w.to_ffi_arrays(),
            None => (Vec::new(), Vec::new()),
        };
        let state_param_cross_ptr = if state_param_cross.is_empty() {
            std::ptr::null()
        } else {
            state_param_cross.as_ptr()
        };
        let param_pair_cross_ptr = if param_pair_cross.is_empty() {
            std::ptr::null()
        } else {
            param_pair_cross.as_ptr()
        };
        let ffi = empyrean_sys::EmpyreanOrbit {
            state: self.state.to_ffi()?,
            orbit_id: orbit_id_ptr,
            object_id: object_id_ptr,
            a1: self.a1,
            a2: self.a2,
            a3: self.a3,
            ng_alpha: self.ng_alpha,
            ng_r0: self.ng_r0,
            ng_m: self.ng_m,
            ng_n: self.ng_n,
            ng_k: self.ng_k,
            // C ABI uses NaN as the "no time delay" sentinel — the FFI
            // struct can't carry an Option directly. The C side checks
            // is_finite() to decide whether to populate NonGravParams.dt.
            non_grav_dt: self.non_grav_dt.unwrap_or(f64::NAN),
            // C ABI uses NaN as the "no DT prior" sentinel; the C side checks
            // is_finite() && > 0 to decide whether to open + prior the DT
            // column in a StateAndNonGravAndDT fit.
            non_grav_dt_variance: self.non_grav_dt_variance.unwrap_or(f64::NAN),
            // Carry the fitted non-grav prior covariance into the FFI so a
            // fitted orbit re-feeds into a StateAndNonGrav refine.
            has_non_grav_covariance: u8::from(self.ng_covariance.is_some()),
            non_grav_covariance: self.ng_covariance.unwrap_or([[0.0; 3]; 3]),
            phot_system: phase_int,
            h_mag: h,
            slope1: s1,
            slope2: s2,
            thrust_arcs: thrust_arcs_ptr,
            n_thrust_arcs: thrust_arcs.len(),
            dv_corrections: dv_corrections_ptr,
            n_dv_corrections: dv_corrections.len(),
            correction_covariances: correction_covariances_ptr,
            n_correction_covariances: correction_covariances.len(),
            // SRP slot. `has_srp` is the explicit switch (SRP is never
            // value-inferred); the C side reads the amrat/cr/variance only
            // when it is 1. NaN variance = fixed-force SRP (AMRAT column
            // closed); finite > 0 opens + priors the AMRAT column.
            has_srp: u8::from(self.srp.is_some()),
            srp_amrat: self.srp.map(|s| s.amrat).unwrap_or(0.0),
            srp_cr: self.srp.map(|s| s.cr).unwrap_or(0.0),
            srp_amrat_variance: self.srp.and_then(|s| s.amrat_variance).unwrap_or(f64::NAN),
            state_param_cross: state_param_cross_ptr,
            n_state_param_cross: state_param_cross.len(),
            param_pair_cross: param_pair_cross_ptr,
            n_param_pair_cross: param_pair_cross.len(),
            // Photometric parameter covariance. Same explicit-switch
            // shape as `has_non_grav_covariance` above; the C side reads
            // the 3×3 only when the switch is 1.
            has_phot_covariance: u8::from(self.phot_covariance.is_some()),
            phot_covariance: self.phot_covariance.unwrap_or([[0.0; 3]; 3]),
        };
        let keep = OrbitFfiKeep {
            _orbit_id: orbit_id_cstr,
            _object_id: object_id_cstr,
            _thrust_arcs: thrust_arcs,
            _dv_corrections: dv_corrections,
            _correction_covariances: correction_covariances,
            _state_param_cross: state_param_cross,
            _param_pair_cross: param_pair_cross,
        };
        Ok((ffi, keep))
    }
}

/// Relative tolerance on the photometric covariance's symmetry.
///
/// A covariance assembled from a fit is symmetric to round-off, not to
/// the bit — the tolerance admits that and nothing wider. It is
/// relative to the larger of the two entries, so a genuinely transposed
/// or half-filled matrix (one side zero) fails at any magnitude.
const PHOT_COV_SYMMETRY_RTOL: f64 = 1e-12;

/// Reject an asymmetric photometric 3×3 by name.
///
/// A covariance is symmetric by definition, and the orbit file formats
/// store only its six lower-triangle cells — so an asymmetric matrix has
/// no file representation and would come back silently truncated (the
/// upper entry dropped, not mirrored), with `mag_sigma` changing across
/// a round trip. Refusing here, at the one boundary every C-ABI path
/// crosses, is what keeps that from being a silent edit of the caller's
/// uncertainty.
fn validate_photometry_covariance_symmetry(cov: &[[f64; 3]; 3]) -> crate::error::Result<()> {
    const NAME: [&str; 3] = ["H", "slope1", "slope2"];
    for (i, row) in cov.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(crate::error::Error::invalid_input(format!(
                    "photometric covariance entry [{i}][{j}] ({}–{}) is {v}, not a finite \
                     number. A non-finite entry contracts to a NaN σ_V, which reads \
                     downstream as 'no photometric uncertainty was supplied' — supply the \
                     real variance, or leave the covariance absent.",
                    NAME[i], NAME[j],
                )));
            }
        }
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            let (a, b) = (cov[i][j], cov[j][i]);
            let scale = a.abs().max(b.abs());
            if (a - b).abs() > PHOT_COV_SYMMETRY_RTOL * scale {
                return Err(crate::error::Error::invalid_input(format!(
                    "photometric covariance is not symmetric: [{i}][{j}] = {a} ({}–{}) but \
                     [{j}][{i}] = {b}. It is a covariance over (H, slope1, slope2), and the \
                     orbit file formats store only its lower triangle, so an asymmetric \
                     matrix cannot round-trip — mirror the term you meant onto both sides.",
                    NAME[i], NAME[j],
                )));
            }
        }
    }
    Ok(())
}

/// Keepalive owner for the heap-allocated identifier strings carried by
/// [`Orbit::to_ffi_with_keep`]. Must outlive every use of the returned
/// [`empyrean_sys::EmpyreanOrbit`].
pub(crate) struct OrbitFfiKeep {
    _orbit_id: std::ffi::CString,
    _object_id: std::ffi::CString,
    /// Marshaled thrust-arc side array the FFI `EmpyreanOrbit` borrows.
    _thrust_arcs: Vec<empyrean_sys::EmpyreanThrustArc>,
    /// Δv-correction side array the FFI `EmpyreanOrbit` borrows.
    _dv_corrections: Vec<[f64; 3]>,
    /// Correction-covariance side array the FFI `EmpyreanOrbit` borrows.
    _correction_covariances: Vec<[[f64; 3]; 3]>,
    /// State↔parameter cross columns the FFI `EmpyreanOrbit` borrows.
    _state_param_cross: Vec<empyrean_sys::EmpyreanStateParamCross>,
    /// Parameter↔parameter cross terms the FFI `EmpyreanOrbit` borrows.
    _param_pair_cross: Vec<empyrean_sys::EmpyreanParamPairCross>,
}

/// Marshal a batch of orbits into their FFI representation plus the
/// keepalive that owns the heap storage each `EmpyreanOrbit` borrows.
///
/// Single conversion path shared by every forward-model entry point — the
/// one-shot [`Context::propagate`](crate::Context::propagate) /
/// [`Context::generate_ephemeris`](crate::Context::generate_ephemeris) and
/// the pre-built [`BuiltSystem`](crate::BuiltSystem) — so all of them feed
/// the engine byte-identical orbit rows. The returned keepalive `Vec` must
/// outlive every use of the returned `EmpyreanOrbit` slice.
pub(crate) fn orbits_to_ffi(
    orbits: &[Orbit],
) -> crate::error::Result<(Vec<empyrean_sys::EmpyreanOrbit>, Vec<OrbitFfiKeep>)> {
    let mut keep: Vec<OrbitFfiKeep> = Vec::with_capacity(orbits.len());
    let ffi = orbits
        .iter()
        .map(|o| {
            let (f, k) = o.to_ffi_with_keep()?;
            keep.push(k);
            Ok(f)
        })
        .collect::<crate::error::Result<Vec<_>>>()?;
    Ok((ffi, keep))
}

#[cfg(test)]
mod srp_tests {
    use super::*;
    use crate::Epoch;
    use crate::coordinate::{CoordinateState, Frame, Origin};

    fn cartesian_orbit() -> Orbit {
        Orbit::new(CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        ))
    }

    #[test]
    fn no_srp_marshals_absent() {
        let (ffi, _keep) = cartesian_orbit().to_ffi_with_keep().unwrap();
        assert_eq!(ffi.has_srp, 0);
        assert!(
            SrpParams::from_ffi(
                ffi.srp_amrat,
                ffi.srp_cr,
                ffi.has_srp,
                ffi.srp_amrat_variance
            )
            .is_none()
        );
    }

    #[test]
    fn fixed_force_srp_round_trips() {
        let o = cartesian_orbit().with_srp(3.0e-3, 1.2);
        let (ffi, _keep) = o.to_ffi_with_keep().unwrap();
        assert_eq!(ffi.has_srp, 1);
        assert_eq!(ffi.srp_amrat, 3.0e-3);
        assert_eq!(ffi.srp_cr, 1.2);
        assert!(ffi.srp_amrat_variance.is_nan()); // no prior ⇒ fixed force
        let back = SrpParams::from_ffi(
            ffi.srp_amrat,
            ffi.srp_cr,
            ffi.has_srp,
            ffi.srp_amrat_variance,
        )
        .unwrap();
        assert_eq!(
            back,
            SrpParams {
                amrat: 3.0e-3,
                cr: 1.2,
                amrat_variance: None
            }
        );
    }

    #[test]
    fn fittable_srp_round_trips() {
        let o = cartesian_orbit()
            .with_srp(3.0e-3, 1.0)
            .with_srp_amrat_variance(Some(1.0e-8));
        let (ffi, _keep) = o.to_ffi_with_keep().unwrap();
        assert_eq!(ffi.srp_amrat_variance, 1.0e-8);
        let back = SrpParams::from_ffi(
            ffi.srp_amrat,
            ffi.srp_cr,
            ffi.has_srp,
            ffi.srp_amrat_variance,
        )
        .unwrap();
        assert_eq!(back.amrat_variance, Some(1.0e-8));
    }

    #[test]
    fn amrat_variance_without_srp_is_noop() {
        // The variance builder requires an SRP slot; without one it's a no-op
        // (never fabricates a slot).
        let o = cartesian_orbit().with_srp_amrat_variance(Some(1.0e-8));
        assert!(o.srp.is_none());
    }
}

#[cfg(test)]
mod photometry_marshal_tests {
    use super::*;
    use crate::Epoch;
    use crate::coordinate::{CoordinateState, Frame, Origin};

    fn cartesian_orbit() -> Orbit {
        Orbit::new(CoordinateState::cartesian(
            Epoch::from_mjd_tdb(59000.0),
            [1.0, 0.1, 0.05, -0.005, 0.015, 0.001],
            Frame::EclipticJ2000,
            Origin::Sun,
        ))
    }

    /// A symmetric covariance crosses unchanged — the guard is about
    /// asymmetry, not about covariances in general.
    #[test]
    fn a_symmetric_covariance_marshals_unchanged() {
        let cov = [
            [0.09, -0.0042, 0.0],
            [-0.0042, 0.0025, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let o = cartesian_orbit()
            .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
            .with_photometry_covariance(Some(cov));
        let (ffi, _keep) = o.to_ffi_with_keep().expect("symmetric must marshal");
        assert_eq!(ffi.has_phot_covariance, 1);
        assert_eq!(ffi.phot_covariance, cov);
    }

    /// Round-off-level asymmetry is what a real fit produces, and must
    /// not be refused.
    #[test]
    fn round_off_asymmetry_is_tolerated() {
        let mut cov = [
            [0.09, -0.0042, 0.0],
            [-0.0042, 0.0025, 0.0],
            [0.0, 0.0, 0.0],
        ];
        cov[1][0] = cov[0][1] * (1.0 + 1.0e-15);
        let o = cartesian_orbit()
            .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
            .with_photometry_covariance(Some(cov));
        assert!(o.to_ffi_with_keep().is_ok());
    }

    /// An off-diagonal filled on ONE side only — the transposed /
    /// upper-triangular convention nothing in the API forbade — is
    /// refused by name.
    ///
    /// FAILS (marshals silently) without the guard, and the file
    /// formats then store only the lower triangle: the term comes back
    /// dropped rather than mirrored, and σ_V changes across a round
    /// trip with no warning.
    #[test]
    fn an_asymmetric_covariance_is_refused_by_name() {
        let cov = [[0.09, -0.02, 0.0], [0.0, 0.0025, 0.0], [0.0, 0.0, 0.0]];
        let o = cartesian_orbit()
            .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
            .with_photometry_covariance(Some(cov));
        let err = match o.to_ffi_with_keep() {
            Ok(_) => panic!("an asymmetric photometric covariance must not marshal"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("not symmetric") && err.contains("[0][1]"),
            "the error must name the offending pair, got: {err}"
        );
        assert!(
            err.contains("H") && err.contains("slope1"),
            "and name the parameters, got: {err}"
        );
    }

    /// A non-finite entry contracts to a NaN σ_V, which reads as "no
    /// photometric uncertainty was supplied" — a different failure axis
    /// with its own message.
    #[test]
    fn a_non_finite_covariance_entry_is_refused_on_its_own_axis() {
        let cov = [[0.09, 0.0, 0.0], [0.0, f64::NAN, 0.0], [0.0, 0.0, 0.0]];
        let o = cartesian_orbit()
            .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
            .with_photometry_covariance(Some(cov));
        let err = match o.to_ffi_with_keep() {
            Ok(_) => panic!("a NaN covariance entry must not marshal"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("[1][1]") && err.contains("finite"),
            "the finiteness axis must be named, not blamed on symmetry: {err}"
        );
    }

    /// `h_mag = 0.0` is the C ABI's "no absolute magnitude supplied"
    /// value. A wrapper `Orbit` states presence in its own `Option`, so
    /// the two disagree — and the photometry must not simply vanish at
    /// the boundary.
    #[test]
    fn photometry_with_a_zero_magnitude_is_refused_rather_than_dropped() {
        let o = cartesian_orbit().with_photometry(PhaseFunction::HG, 0.0, 0.15, 0.0);
        let err = match o.to_ffi_with_keep() {
            Ok(_) => panic!("H = 0.0 must be refused, not silently dropped"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("h_mag = 0.0"),
            "the error must name the value: {err}"
        );
    }

    /// An orbit with no photometry at all still marshals, and says so.
    #[test]
    fn no_photometry_marshals_as_absent() {
        let (ffi, _keep) = cartesian_orbit().to_ffi_with_keep().unwrap();
        assert_eq!(ffi.phot_system, -1);
        assert!(ffi.h_mag.is_nan());
        assert_eq!(ffi.has_phot_covariance, 0);
    }
}
