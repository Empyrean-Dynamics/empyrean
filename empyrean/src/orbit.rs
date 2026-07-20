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
    /// Continuous-thrust parameters (arcs + optional Δv corrections).
    /// `None` is a pure gravity + non-grav orbit. When
    /// [`ThrustParams::correction_covariances`] is non-empty the
    /// resulting burn-sensitivity segments surface in the propagated
    /// [`TaggedCovariance::thrust_segments`](crate::TaggedCovariance).
    pub thrust: Option<ThrustParams>,
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
            thrust: None,
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
        let (phase_int, h, s1, s2) = match self.phot_system {
            Some(pf) => (pf as i32, self.h_mag, self.slope1, self.slope2),
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
        };
        let keep = OrbitFfiKeep {
            _orbit_id: orbit_id_cstr,
            _object_id: object_id_cstr,
            _thrust_arcs: thrust_arcs,
            _dv_corrections: dv_corrections,
            _correction_covariances: correction_covariances,
        };
        Ok((ffi, keep))
    }
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
