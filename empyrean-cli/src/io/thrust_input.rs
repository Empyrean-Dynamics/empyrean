//! Continuous-thrust input for the `propagate` command.
//!
//! Parses a JSON thrust file into the wrapper's [`empyrean::ThrustParams`]
//! so finite-burn / low-thrust arcs can be attached to the propagated
//! orbits. Field names, units, and semantics are 1:1 with the engine
//! surface (`start_mjd_tdb`, `end_mjd_tdb`, `thrust_n`, `mass_kg`,
//! `isp_s`, `steering`, `sharpness`, `central_body`, plus
//! `dv_corrections` / `correction_covariances`) so the same fields carry
//! through the CLI, the wrapper, and the C ABI without renaming.
//!
//! JSON schema (one file describes one [`empyrean::ThrustParams`], applied
//! to every orbit in the batch):
//!
//! ```json
//! {
//!   "arcs": [
//!     {
//!       "start_mjd_tdb": 59000.0,
//!       "end_mjd_tdb":   59010.0,
//!       "thrust_n":      1000.0,
//!       "mass_kg":       1000.0,
//!       "isp_s":         3000.0,
//!       "steering":      { "type": "constant_rtn", "alpha_rad": 0.1, "beta_rad": 0.2 },
//!       "sharpness":     100.0,
//!       "central_body":  10
//!     }
//!   ],
//!   "dv_corrections":         [[0.0, 0.0, 0.0]],
//!   "correction_covariances": [[[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]]
//! }
//! ```
//!
//! - `isp_s` is optional (omit or `null` = constant mass; `Some` depletes
//!   mass at \\( \dot m = F/(I_{sp}\,g_0) \\)).
//! - `central_body` is a NAIF integer body code (10 = Sun, 399 = Earth,
//!   301 = Moon, …); it is the RTN / velocity-tangent frame reference.
//! - `steering.type` is one of `constant_rtn` (with `alpha_rad`,
//!   `beta_rad`), `velocity_tangent`, or `inertial_fixed` (with
//!   `direction`).
//! - `dv_corrections` is positional with `arcs`. When
//!   `correction_covariances` is non-empty its length MUST equal
//!   `dv_corrections`; this selects the burn-sensitivity propagation. A
//!   mismatch is rejected loudly by the engine at propagation time — never
//!   silently repaired.

use std::path::Path;

use anyhow::{Context, Result, bail};
use empyrean::{Origin, SteeringLaw, ThrustArc, ThrustParams};
use serde::Deserialize;

/// One steering law, tagged by `type`. Field names match the engine
/// (`alpha_rad`, `beta_rad`, `direction`).
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SteeringLawJson {
    /// Constant RTN angles relative to the arc's central body.
    ConstantRtn { alpha_rad: f64, beta_rad: f64 },
    /// Thrust aligned with the velocity vector relative to the central body.
    VelocityTangent,
    /// Fixed direction in the inertial frame (normalized by the engine).
    InertialFixed { direction: [f64; 3] },
}

impl From<SteeringLawJson> for SteeringLaw {
    fn from(s: SteeringLawJson) -> Self {
        match s {
            SteeringLawJson::ConstantRtn {
                alpha_rad,
                beta_rad,
            } => SteeringLaw::ConstantRTN {
                alpha_rad,
                beta_rad,
            },
            SteeringLawJson::VelocityTangent => SteeringLaw::VelocityTangent,
            SteeringLawJson::InertialFixed { direction } => {
                SteeringLaw::InertialFixed { direction }
            }
        }
    }
}

/// One continuous-thrust arc. `isp_s` is optional (constant mass when
/// absent); every other field is required, matching the engine, so a
/// missing value surfaces loudly rather than defaulting silently.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThrustArcJson {
    start_mjd_tdb: f64,
    end_mjd_tdb: f64,
    thrust_n: f64,
    mass_kg: f64,
    /// Specific impulse (seconds). Omit or `null` = constant mass.
    #[serde(default)]
    isp_s: Option<f64>,
    steering: SteeringLawJson,
    sharpness: f64,
    /// NAIF integer body code for the RTN / velocity-tangent reference.
    central_body: i32,
}

impl ThrustArcJson {
    fn into_arc(self) -> Result<ThrustArc> {
        let central_body = Origin::from_naif_id(self.central_body).ok_or_else(|| {
            anyhow::anyhow!(
                "thrust arc central_body = {} is not a supported NAIF body code \
                 (expected e.g. 10 = Sun, 399 = Earth, 301 = Moon)",
                self.central_body
            )
        })?;
        Ok(ThrustArc::new(
            self.start_mjd_tdb,
            self.end_mjd_tdb,
            self.thrust_n,
            self.mass_kg,
            self.sharpness,
            self.steering.into(),
            central_body,
        )
        .with_isp(self.isp_s))
    }
}

/// Top-level thrust file: arcs plus optional Δv targeting corrections and
/// their covariances.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThrustParamsJson {
    arcs: Vec<ThrustArcJson>,
    /// Per-arc Δv corrections (AU/day), positional with `arcs`.
    #[serde(default)]
    dv_corrections: Vec<[f64; 3]>,
    /// 3×3 covariance (AU/day)² per Δv correction. When non-empty its
    /// length must equal `dv_corrections` (checked by the engine).
    #[serde(default)]
    correction_covariances: Vec<[[f64; 3]; 3]>,
}

/// Load a thrust file and convert it into the wrapper's
/// [`empyrean::ThrustParams`].
///
/// The length-consistency contract between `dv_corrections` and
/// `correction_covariances` (and the `ThirdOrder` + covariance
/// rejection) is enforced by the engine at propagation time and surfaces
/// as a propagation error — never repaired here.
pub fn load_thrust_params(path: &Path) -> Result<ThrustParams> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read thrust file {}", path.display()))?;
    let parsed: ThrustParamsJson = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse thrust JSON {}", path.display()))?;

    if parsed.arcs.is_empty() {
        bail!(
            "thrust file {} contains no arcs; remove --thrust-arcs for a \
             pure gravity + non-grav propagation",
            path.display()
        );
    }

    let arcs = parsed
        .arcs
        .into_iter()
        .map(ThrustArcJson::into_arc)
        .collect::<Result<Vec<_>>>()?;

    Ok(ThrustParams::new(arcs)
        .with_dv_corrections(parsed.dv_corrections)
        .with_correction_covariances(parsed.correction_covariances))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<ThrustParams> {
        use std::sync::atomic::{AtomicU64, Ordering};
        // Unique per call so parallel tests never share a temp file.
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "empyrean_cli_thrust_test_{}_{}.json",
            std::process::id(),
            n
        ));
        std::fs::write(&path, json).unwrap();
        let out = load_thrust_params(&path);
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn parses_full_constant_rtn_arc_with_corrections() {
        let json = r#"{
            "arcs": [{
                "start_mjd_tdb": 59000.0,
                "end_mjd_tdb": 59010.0,
                "thrust_n": 1000.0,
                "mass_kg": 1000.0,
                "isp_s": 3000.0,
                "steering": {"type": "constant_rtn", "alpha_rad": 0.1, "beta_rad": 0.2},
                "sharpness": 100.0,
                "central_body": 10
            }],
            "dv_corrections": [[1e-6, 2e-6, 3e-6]],
            "correction_covariances": [[[1e-20, 0, 0], [0, 2e-20, 0], [0, 0, 3e-20]]]
        }"#;
        let tp = parse(json).expect("parse");
        assert_eq!(tp.arcs.len(), 1);
        let arc = &tp.arcs[0];
        assert_eq!(arc.start_mjd_tdb, 59000.0);
        assert_eq!(arc.end_mjd_tdb, 59010.0);
        assert_eq!(arc.thrust_n, 1000.0);
        assert_eq!(arc.mass_kg, 1000.0);
        assert_eq!(arc.isp_s, Some(3000.0));
        assert_eq!(arc.sharpness, 100.0);
        assert_eq!(arc.central_body, Origin::SUN);
        assert_eq!(
            arc.steering,
            SteeringLaw::ConstantRTN {
                alpha_rad: 0.1,
                beta_rad: 0.2
            }
        );
        assert_eq!(tp.dv_corrections, vec![[1e-6, 2e-6, 3e-6]]);
        assert_eq!(tp.correction_covariances.len(), 1);
    }

    #[test]
    fn isp_omitted_is_constant_mass() {
        let json = r#"{
            "arcs": [{
                "start_mjd_tdb": 59000.0,
                "end_mjd_tdb": 59010.0,
                "thrust_n": 1.0,
                "mass_kg": 500.0,
                "steering": {"type": "velocity_tangent"},
                "sharpness": 50.0,
                "central_body": 399
            }]
        }"#;
        let tp = parse(json).expect("parse");
        assert_eq!(tp.arcs[0].isp_s, None);
        assert_eq!(tp.arcs[0].steering, SteeringLaw::VelocityTangent);
        assert_eq!(tp.arcs[0].central_body, Origin::EARTH);
        // Missing correction arrays default to empty.
        assert!(tp.dv_corrections.is_empty());
        assert!(tp.correction_covariances.is_empty());
    }

    #[test]
    fn parses_inertial_fixed_direction() {
        let json = r#"{
            "arcs": [{
                "start_mjd_tdb": 59000.0,
                "end_mjd_tdb": 59010.0,
                "thrust_n": 1.0,
                "mass_kg": 500.0,
                "steering": {"type": "inertial_fixed", "direction": [1.0, 0.0, 0.0]},
                "sharpness": 50.0,
                "central_body": 10
            }]
        }"#;
        let tp = parse(json).expect("parse");
        assert_eq!(
            tp.arcs[0].steering,
            SteeringLaw::InertialFixed {
                direction: [1.0, 0.0, 0.0]
            }
        );
    }

    #[test]
    fn rejects_unknown_naif_central_body() {
        // Planet body-center 599 (Jupiter) is not a supported Origin —
        // must surface loudly, not silently fall back to a default body.
        let json = r#"{
            "arcs": [{
                "start_mjd_tdb": 59000.0,
                "end_mjd_tdb": 59010.0,
                "thrust_n": 1.0,
                "mass_kg": 500.0,
                "steering": {"type": "velocity_tangent"},
                "sharpness": 50.0,
                "central_body": 599
            }]
        }"#;
        let err = parse(json).expect_err("unsupported NAIF id must error");
        assert!(
            err.to_string().contains("599"),
            "error must echo the bad NAIF id: {err}"
        );
    }

    #[test]
    fn rejects_empty_arcs() {
        let json = r#"{ "arcs": [] }"#;
        let err = parse(json).expect_err("empty arcs must error");
        assert!(
            err.to_string().contains("no arcs"),
            "error must explain the empty-arcs rejection: {err}"
        );
    }

    #[test]
    fn rejects_unknown_field() {
        // A typo'd field name must surface loudly (deny_unknown_fields),
        // never be silently ignored.
        let json = r#"{
            "arcs": [{
                "start_mjd_tdb": 59000.0,
                "end_mjd_tdb": 59010.0,
                "thrust_n": 1.0,
                "mass_kg": 500.0,
                "steering": {"type": "velocity_tangent"},
                "sharpness": 50.0,
                "central_body": 10,
                "thrust_newtons": 999.0
            }]
        }"#;
        let err = parse(json).expect_err("unknown field must error");
        assert!(
            err.to_string().contains("parse")
                || err.to_string().contains("thrust_newtons")
                || err.to_string().contains("unknown"),
            "error should surface the unknown field: {err}"
        );
    }
}
