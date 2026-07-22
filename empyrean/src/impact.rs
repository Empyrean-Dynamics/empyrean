//! Close-approach impact probability and B-plane geometry.
//!
//! Two entry points on [`Context`] live here:
//! [`Context::compute_impact_probabilities`] and
//! [`Context::compute_b_planes`]. Both run one full propagation per
//! supplied [`UncertaintyMethod`] and return tagged result rows so a
//! consumer can compare linear, second-order, and Monte-Carlo
//! breakdowns on the same encounter.
//!
//! # Example: Apophis 2029 with three methods
//!
//! ```no_run
//! use empyrean::{Context, Epoch, Origin, UncertaintyMethod};
//!
//! let ctx = Context::from_data_dir(None)?;
//! let batch = empyrean::query_sbdb(&["99942"], None)?;
//!
//! // 2031-07 — well past the 2029-04 deep flyby.
//! let end_epoch = Epoch::from_mjd_tdb(63000.0);
//! let methods = vec![
//!     UncertaintyMethod::FirstOrder,
//!     UncertaintyMethod::SecondOrder,
//!     UncertaintyMethod::monte_carlo(100_000),
//! ];
//! let body_filter = vec![Origin::EARTH, Origin::MOON];
//!
//! let ips = ctx.compute_impact_probabilities(
//!     &batch.orbits, end_epoch, &methods, &body_filter,
//! )?;
//! for r in &ips {
//!     println!(
//!         "{:?} on {:?}: linear={:.3e}, 2nd-order={:.3e}, MC={:.3e}",
//!         r.method, r.body, r.ip_linear, r.ip_second_order, r.ip_mc,
//!     );
//! }
//! # Ok::<(), empyrean::Error>(())
//! ```
//!
//! # Sanity-checking the IP estimates
//!
//! On a screened-in object, compare the per-method values:
//!
//! - All three within an order of magnitude → the linear gate is
//!   trustworthy.
//! - `ip_second_order` ≪ `ip_linear` → the linear approximation
//!   over-estimates; the second-order correction shrinks the encounter
//!   ellipse.
//! - Monte-Carlo diverges from both → tail probabilities matter;
//!   report the MC value.
//!
//! For close-encounter geometry interpretation (Öpik / Valsecchi
//! axes, gravitational focusing, the impact criterion
//! \\(|B| < R_\\mathrm{eff}\\)), see [`BPlane`].

use crate::context::Context;
use crate::coordinate::Origin;
use crate::error::{Error, Result};
use crate::orbit::Orbit;
use crate::propagate::UncertaintyMethod;
use std::ffi::CStr;

/// One impact-probability record produced by a single
/// [`UncertaintyMethod`] propagation.
///
/// All distances are denominated in both AU and km as the field
/// names indicate. Method-specific fields (`ip_second_order`,
/// `ip_agm`, `ip_mc`, `mc_n_samples`, `mc_n_impacts`) carry
/// `f64::NAN` / 0 sentinels for rows where the producing method
/// didn't compute them — e.g. Monte-Carlo sample counts are 0 on a
/// linear-method row.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpactProbability {
    /// Method that produced this record.
    pub method: UncertaintyMethod,
    /// Orbit-hypothesis identifier (matches the input `Orbit.orbit_id`).
    pub orbit_id: String,
    /// Physical-object identifier (empty when the input had no `object_id`).
    pub object_id: String,
    /// Body the impact probability was computed against.
    pub body: Origin,
    /// Epoch of closest approach.
    pub epoch: crate::Epoch,
    /// Closest-approach geocentric (or body-centric) distance at the
    /// nominal trajectory (AU).
    pub miss_distance_au: f64,
    /// Closest-approach distance (km).
    pub miss_distance_km: f64,
    /// Body radius inflated for atmospheric capture — what IP is
    /// computed against (AU).
    pub effective_radius_au: f64,
    /// Effective capture radius (km).
    pub effective_radius_km: f64,
    /// 1σ uncertainty along the miss-distance direction, linearised (AU).
    pub sigma_distance_au: f64,
    /// 1σ uncertainty along the miss-distance direction, linearised (km).
    pub sigma_distance_km: f64,
    /// Linear (Φ Σ Φᵀ-mapped) impact probability.
    pub ip_linear: f64,
    /// Encounter relative velocity (AU/day).
    pub relative_velocity_au_day: f64,
    /// Park-Scheeres second-order Gaussian IP. NaN when the producing
    /// method was first-order only.
    pub ip_second_order: f64,
    /// Local nonlinearity diagnostic at the close-approach epoch.
    /// Included for completeness; do not use for method selection.
    pub nonlinearity: f64,
    /// Adaptive Gaussian Mixture impact probability. NaN when the
    /// producing method was not AGM.
    pub ip_agm: f64,
    /// Monte-Carlo impact-probability fraction
    /// (`mc_n_impacts / mc_n_samples`). NaN when the producing method
    /// was not Monte Carlo.
    pub ip_mc: f64,
    /// Total Monte-Carlo samples drawn (0 on non-MC rows).
    pub mc_n_samples: u64,
    /// Of those samples, how many impacted (0 on non-MC rows).
    pub mc_n_impacts: u64,
    /// Geodetic latitude of the closest-approach surface point on the
    /// body's reference ellipsoid (degrees, north positive). NaN when no
    /// surface projection is available for this encounter (no
    /// body-orientation coverage, unmatched close approach, or the body
    /// has no registered ellipsoid).
    pub impact_latitude_deg: f64,
    /// Geodetic longitude of the closest-approach surface point
    /// (degrees, east positive, \\([-180, 180]\\)). NaN when unavailable.
    pub impact_longitude_deg: f64,
    /// Altitude of the closest-approach point above the reference
    /// ellipsoid (km). NaN when unavailable.
    pub impact_altitude_km: f64,
    /// Half-width of the 95% binomial (normal-approximation) confidence
    /// interval on `ip_mc`: \\(1.96\sqrt{p(1-p)/n}\\). The interval is
    /// `ip_mc ± mc_confidence_interval`. NaN on non-MC rows.
    pub mc_confidence_interval: f64,
    /// Second-order corrected mean miss distance (AU),
    /// \\(\mu_d = d_0 + \tfrac{1}{2}\mathrm{Tr}(H_d \Sigma_0)\\). On
    /// Monte-Carlo rows carries the sample-mean miss distance instead.
    /// NaN when the producing method computed neither.
    pub mean_distance_second_order_au: f64,
    /// Second-order corrected 1σ miss-distance uncertainty (AU). NaN
    /// when the producing method carried no second-order derivatives.
    pub sigma_distance_second_order_au: f64,
    /// Skewness \\(\gamma_1\\) of the miss-distance distribution under
    /// the second-order expansion (dimensionless). NaN when not
    /// computed.
    pub skewness: f64,
    /// Gradient \\(\partial d / \partial x_0\\) of the closest-approach
    /// distance with respect to the initial Cartesian state (position
    /// components dimensionless, velocity components in days). All-zero
    /// on Monte-Carlo rows and on degenerate zero-miss encounters.
    pub gradient: [f64; 6],
    /// Second derivatives \\(\partial^2 d / \partial x_{0i} \partial
    /// x_{0j}\\) of the closest-approach distance (6×6 symmetric, same
    /// initial-state units as `gradient`). Every entry NaN when the
    /// producing method carried no second-order derivatives.
    pub distance_hessian: [[f64; 6]; 6],
    /// Number of mixture components used by the adaptive
    /// Gaussian-mixture IP refinement. 0 when the refinement did not run
    /// (matches `ip_agm` = NaN).
    pub agm_components: u64,
}

/// One B-plane breakdown for a single close approach.
///
/// The B-plane is the encounter plane perpendicular to the hyperbolic
/// excess-velocity vector. Coordinates are reported in the canonical
/// Öpik / Valsecchi frame: the **T axis** points along the projection
/// of the planet's heliocentric velocity onto the B-plane, and the
/// **R axis** completes a right-handed frame with the inbound
/// asymptote of the encounter. `B·T` controls the along-track
/// encounter geometry (and the resonant-return / keyhole structure);
/// `B·R` controls the cross-track miss component.
///
/// # Impact criterion and gravitational focusing
///
/// An encounter impacts the body when
/// \\(|B| < R_\\mathrm{eff}\\), where the effective radius is the
/// geometric body radius inflated by the body's own gravity:
///
/// \\[
///   R_\\mathrm{eff}^2 = R^2 \\left(1 + \\frac{v_\\mathrm{esc}^2}{v_\\infty^2}\\right)
/// \\]
///
/// with \\(v_\\mathrm{esc}\\) the body's surface escape velocity
/// (Earth: 11.2 km/s) and \\(v_\\infty\\) the encounter's hyperbolic
/// excess velocity ([`v_inf_km_s`](Self::v_inf_km_s)). For a typical
/// NEA encounter at \\(v_\\infty \\sim 10\\) km/s the effective
/// radius is ≈ 1.4× the body radius; for slow encounters at
/// \\(v_\\infty \\sim 5\\) km/s it grows to ≈ 2× the body radius.
///
/// The `effective_radius_km` field carries the focused radius the
/// engine actually used; `body_radius_km` is the geometric radius for
/// reference.
///
/// # Reading the 3σ ellipse
///
/// The encounter uncertainty projects onto the B-plane as an ellipse;
/// [`semi_major_3sig_km`](Self::semi_major_3sig_km),
/// [`semi_minor_3sig_km`](Self::semi_minor_3sig_km), and
/// [`ellipse_angle_rad`](Self::ellipse_angle_rad) describe its 3σ
/// boundary (semi-axes in km, rotation in radians from +T).
/// Compare the ellipse semi-axes to `effective_radius_km`: when the
/// ellipse is small relative to \\(R_\\mathrm{eff}\\), the linear
/// impact probability is meaningful; when it spans many \\(R_\\mathrm{eff}\\),
/// reach for the second-order or sample-based methods.
#[derive(Debug, Clone, PartialEq)]
pub struct BPlane {
    /// Method that produced this record.
    pub method: UncertaintyMethod,
    /// Body the encounter is relative to.
    pub body: Origin,
    /// Epoch of closest approach.
    pub epoch: crate::Epoch,
    /// `B · T` — along-track encounter coordinate (km).
    pub b_dot_t_km: f64,
    /// `B · R` — cross-track encounter coordinate (km).
    pub b_dot_r_km: f64,
    /// Magnitude of the B vector (km).
    pub b_mag_km: f64,
    /// Hyperbolic-excess velocity v∞ at the encounter (km/s).
    pub v_inf_km_s: f64,
    /// Body capture radius inflated for gravitational focusing (km) —
    /// `R_eff² = R² (1 + v_esc² / v_inf²)`.
    pub effective_radius_km: f64,
    /// Geometric body radius (km).
    pub body_radius_km: f64,
    /// Projected B-plane covariance `[σ_TT, σ_TR, σ_RR]` (km²);
    /// `[NaN, NaN, NaN]` when no projected covariance is available.
    pub cov_b_plane: [f64; 3],
    /// 3σ encounter-ellipse semi-major axis on the B-plane (km).
    pub semi_major_3sig_km: f64,
    /// 3σ encounter-ellipse semi-minor axis on the B-plane (km).
    pub semi_minor_3sig_km: f64,
    /// 3σ ellipse rotation angle (radians from the +T axis).
    pub ellipse_angle_rad: f64,
    /// Linear (Φ Σ Φᵀ-mapped) impact probability for this encounter.
    pub ip_linear: f64,
}

fn method_from_tag(tag: u8) -> UncertaintyMethod {
    match tag {
        0 => UncertaintyMethod::FirstOrder,
        1 => UncertaintyMethod::SecondOrder,
        2 => UncertaintyMethod::sigma_point(),
        3 => UncertaintyMethod::monte_carlo(1000),
        // Tag 4 = Auto (adaptive per-CA-window regime selection). Without
        // this arm, Auto IP / B-plane results were silently relabelled
        // FirstOrder on readback — the IP value was correct but the
        // reported method was wrong.
        4 => UncertaintyMethod::auto(),
        _ => UncertaintyMethod::FirstOrder,
    }
}

unsafe fn cstr_to_string(p: *mut std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl Context {
    /// Compute impact probabilities for a batch of orbits over a
    /// propagation window, one full propagation per supplied
    /// [`UncertaintyMethod`]. The flat output array carries the
    /// method tag on each row so a downstream consumer can group by
    /// `(method, body, orbit_id)`.
    ///
    /// # Body filter
    ///
    /// `body_filter` selects which bodies to test for close approach
    /// and impact. Pass an empty slice to monitor every solid body the
    /// engine ships ephemerides for: Mercury, Venus, Earth, Moon,
    /// Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto (gas-giant
    /// barycenters as a proxy for the planet, since DE440 ships only
    /// the barycenter). The Sun is intentionally excluded — IP against
    /// the Sun is not meaningful at the wrapper API level.
    ///
    /// For planetary-defense work, the typical filter is
    /// `&[Origin::EARTH, Origin::MOON]`; for spacecraft mission
    /// analysis, target the body of interest plus its primary.
    pub fn compute_impact_probabilities(
        &self,
        orbits: &[Orbit],
        end_epoch: crate::Epoch,
        methods: &[UncertaintyMethod],
        body_filter: &[Origin],
    ) -> Result<Vec<ImpactProbability>> {
        let end_mjd_tdb = end_epoch.mjd_tdb()?;
        let mut _orbit_keep: Vec<crate::orbit::OrbitFfiKeep> = Vec::with_capacity(orbits.len());
        let ffi_orbits: Vec<_> = orbits
            .iter()
            .map(|o| {
                let (ffi, keep) = o.to_ffi_with_keep()?;
                _orbit_keep.push(keep);
                Ok(ffi)
            })
            .collect::<Result<Vec<_>>>()?;
        let ffi_methods: Vec<_> = methods.iter().map(UncertaintyMethod::to_ffi).collect();
        let body_filter_naif: Vec<i32> = body_filter.iter().map(|o| o.naif_id()).collect();
        let mut ffi_result = empyrean_sys::EmpyreanImpactProbabilitiesResult {
            records: std::ptr::null_mut(),
            num_records: 0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_compute_impact_probabilities(
                self.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                end_mjd_tdb,
                ffi_methods.as_ptr(),
                ffi_methods.len(),
                if body_filter_naif.is_empty() {
                    std::ptr::null()
                } else {
                    body_filter_naif.as_ptr()
                },
                body_filter_naif.len(),
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }

        let n = ffi_result.num_records;
        let mut out = Vec::with_capacity(n);
        if !ffi_result.records.is_null() && n > 0 {
            for i in 0..n {
                let rec = unsafe { &*ffi_result.records.add(i) };
                let body = Origin::from_naif_id(rec.body_naif_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "C ABI returned unknown NAIF id for body: {}",
                        rec.body_naif_id
                    ))
                })?;
                out.push(ImpactProbability {
                    method: method_from_tag(rec.method_tag),
                    orbit_id: unsafe { cstr_to_string(rec.orbit_id) },
                    object_id: unsafe { cstr_to_string(rec.object_id) },
                    body,
                    epoch: crate::Epoch::from_mjd_tdb(rec.epoch_mjd_tdb),
                    miss_distance_au: rec.miss_distance_au,
                    miss_distance_km: rec.miss_distance_km,
                    effective_radius_au: rec.effective_radius_au,
                    effective_radius_km: rec.effective_radius_km,
                    sigma_distance_au: rec.sigma_distance_au,
                    sigma_distance_km: rec.sigma_distance_km,
                    ip_linear: rec.ip_linear,
                    relative_velocity_au_day: rec.relative_velocity_au_day,
                    ip_second_order: rec.ip_second_order,
                    nonlinearity: rec.nonlinearity,
                    ip_agm: rec.ip_agm,
                    ip_mc: rec.ip_mc,
                    mc_n_samples: rec.mc_n_samples,
                    mc_n_impacts: rec.mc_n_impacts,
                    impact_latitude_deg: rec.impact_latitude_deg,
                    impact_longitude_deg: rec.impact_longitude_deg,
                    impact_altitude_km: rec.impact_altitude_km,
                    mc_confidence_interval: rec.mc_confidence_interval,
                    mean_distance_second_order_au: rec.mean_distance_second_order_au,
                    sigma_distance_second_order_au: rec.sigma_distance_second_order_au,
                    skewness: rec.skewness,
                    gradient: rec.gradient,
                    distance_hessian: rec.distance_hessian,
                    agm_components: rec.agm_components,
                });
            }
        }
        unsafe { empyrean_sys::empyrean_compute_impact_probabilities_result_free(&mut ffi_result) };
        Ok(out)
    }

    /// Compute the B-plane geometry breakdown for every close
    /// approach detected during one propagation per supplied
    /// [`UncertaintyMethod`].
    ///
    /// `body_filter` follows the same convention as
    /// [`Self::compute_impact_probabilities`] — pass an empty slice to
    /// monitor every solid body, or a specific subset (typically
    /// `&[Origin::EARTH]` for planetary-defense work).
    pub fn compute_b_planes(
        &self,
        orbits: &[Orbit],
        end_epoch: crate::Epoch,
        methods: &[UncertaintyMethod],
        body_filter: &[Origin],
    ) -> Result<Vec<BPlane>> {
        let end_mjd_tdb = end_epoch.mjd_tdb()?;
        let mut _orbit_keep: Vec<crate::orbit::OrbitFfiKeep> = Vec::with_capacity(orbits.len());
        let ffi_orbits: Vec<_> = orbits
            .iter()
            .map(|o| {
                let (ffi, keep) = o.to_ffi_with_keep()?;
                _orbit_keep.push(keep);
                Ok(ffi)
            })
            .collect::<Result<Vec<_>>>()?;
        let ffi_methods: Vec<_> = methods.iter().map(UncertaintyMethod::to_ffi).collect();
        let body_filter_naif: Vec<i32> = body_filter.iter().map(|o| o.naif_id()).collect();
        let mut ffi_result = empyrean_sys::EmpyreanBPlanesResult {
            records: std::ptr::null_mut(),
            num_records: 0,
        };
        let code = unsafe {
            empyrean_sys::empyrean_compute_b_planes(
                self.as_raw(),
                ffi_orbits.as_ptr(),
                ffi_orbits.len(),
                end_mjd_tdb,
                ffi_methods.as_ptr(),
                ffi_methods.len(),
                if body_filter_naif.is_empty() {
                    std::ptr::null()
                } else {
                    body_filter_naif.as_ptr()
                },
                body_filter_naif.len(),
                &mut ffi_result,
            )
        };
        if code != 0 {
            return Err(Error::capture(code));
        }

        let n = ffi_result.num_records;
        let mut out = Vec::with_capacity(n);
        if !ffi_result.records.is_null() && n > 0 {
            for i in 0..n {
                let rec = unsafe { &*ffi_result.records.add(i) };
                let body_str = unsafe { cstr_to_string(rec.body) };
                let body: Origin = body_str.parse().map_err(|_| {
                    Error::invalid_input(format!(
                        "C ABI returned unrecognized body name: {body_str:?}"
                    ))
                })?;
                out.push(BPlane {
                    method: method_from_tag(rec.method_tag),
                    body,
                    epoch: crate::Epoch::from_mjd_tdb(rec.epoch_mjd_tdb),
                    b_dot_t_km: rec.b_dot_t_km,
                    b_dot_r_km: rec.b_dot_r_km,
                    b_mag_km: rec.b_mag_km,
                    v_inf_km_s: rec.v_inf_km_s,
                    effective_radius_km: rec.effective_radius_km,
                    body_radius_km: rec.body_radius_km,
                    cov_b_plane: rec.cov_b_plane,
                    semi_major_3sig_km: rec.semi_major_3sig_km,
                    semi_minor_3sig_km: rec.semi_minor_3sig_km,
                    ellipse_angle_rad: rec.ellipse_angle_rad,
                    ip_linear: rec.ip_linear,
                });
            }
        }
        unsafe { empyrean_sys::empyrean_compute_b_planes_result_free(&mut ffi_result) };
        Ok(out)
    }
}
