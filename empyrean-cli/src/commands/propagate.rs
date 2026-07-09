use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use empyrean::{OrbitBatch, PropagationConfig, PropagationResult, UncertaintyMethod};

use crate::io::output::OutputFormat;
use crate::io::{orbit_input, output};
use crate::{ForceModel, UncertaintyMethodArg};

#[derive(clap::Args)]
pub struct PropagateArgs {
    /// Object names to query from JPL SBDB.
    #[arg(long = "object-id", num_args = 1..)]
    pub object_ids: Option<Vec<String>>,

    /// Path to an orbits file (.parquet, .json, or .csv).
    #[arg(long, conflicts_with = "object_ids")]
    pub input: Option<PathBuf>,

    /// Target epoch (MJD TDB).
    #[arg(long)]
    pub epoch: f64,

    /// Force model tier.
    #[arg(long, default_value = "standard")]
    pub force_model: ForceModel,

    /// Uncertainty propagation method.
    #[arg(long, value_enum, default_value_t = UncertaintyMethodArg::FirstOrder)]
    pub uncertainty_method: UncertaintyMethodArg,

    /// Output directory.
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format for states + events.
    #[arg(long, value_enum, default_value_t = OutputFormat::Parquet)]
    pub format: OutputFormat,

    /// Print the resolved-kind tagged covariance series for each orbit
    /// (position 1σ, resolved kind, definiteness, solved width). Runs
    /// in-process — the daemon fast path is skipped when this is set.
    #[arg(long)]
    pub tagged_covariance: bool,

    /// Path to a JSON thrust file describing continuous-thrust arcs
    /// (finite burns / low-thrust). Applied to every orbit in the batch.
    /// Schema: `{ "arcs": [{ "start_mjd_tdb", "end_mjd_tdb", "thrust_n",
    /// "mass_kg", "isp_s"?, "steering", "sharpness", "central_body" }],
    /// "dv_corrections"?, "correction_covariances"? }`. `steering` is
    /// `{ "type": "constant_rtn", "alpha_rad", "beta_rad" }`,
    /// `{ "type": "velocity_tangent" }`, or `{ "type": "inertial_fixed",
    /// "direction": [x, y, z] }`; `central_body` is a NAIF body code
    /// (10 = Sun, 399 = Earth, 301 = Moon); `isp_s` omitted = constant
    /// mass. Runs in-process — the daemon fast path is skipped when this
    /// is set so the thrust is never silently dropped.
    #[arg(long)]
    pub thrust_arcs: Option<PathBuf>,
}

/// Astronomical unit in km (IAU 2012, exact). The tagged-covariance
/// matrices are AU-based; position σ is reported in km.
const AU_KM: f64 = 149_597_870.7;

pub fn run(data_dir: Option<PathBuf>, args: PropagateArgs) -> Result<()> {
    // Try daemon first — but only when neither the tagged-covariance
    // readback nor a thrust file is requested. The daemon protocol
    // returns a summary string and cannot stream the per-epoch series
    // (`--tagged-covariance`), and its wire request carries no thrust
    // fields, so `--thrust-arcs` must also fall through to the in-process
    // path below — sending it to the daemon would silently drop the burn.
    if !args.tagged_covariance && args.thrust_arcs.is_none() {
        let request = crate::daemon::protocol::Request::Propagate {
            object_ids: args.object_ids.clone(),
            input_path: args.input.as_ref().map(|p| p.display().to_string()),
            epoch: args.epoch,
            force_model: args.force_model.as_str().to_string(),
            uncertainty_method: args.uncertainty_method.as_str().to_string(),
            out_dir: args.out_dir.display().to_string(),
            format: format_to_str(args.format).into(),
        };
        if let Some(resp) = crate::daemon::client::try_request(&request) {
            if resp.success {
                eprintln!("{}", resp.message);
                return Ok(());
            } else {
                anyhow::bail!("daemon error: {}", resp.error.unwrap_or_default());
            }
        }
    }

    // In-process fallback.
    let t0 = Instant::now();
    let ctx =
        empyrean::Context::from_data_dir(data_dir.as_deref()).context("failed to load context")?;
    eprintln!("Loaded context ({:.1}s)", t0.elapsed().as_secs_f64());

    let mut batch = orbit_input::load_orbits(&args.object_ids, &args.input)?;

    // Attach continuous-thrust parameters, if any. One thrust file
    // describes one `ThrustParams`, applied to every orbit in the batch.
    // The engine enforces the dv_corrections / correction_covariances
    // length contract (and rejects ThirdOrder + correction covariances)
    // at propagation time; any violation surfaces below as a loud
    // `propagation failed` error rather than being repaired here.
    if let Some(thrust_path) = &args.thrust_arcs {
        let thrust = crate::io::thrust_input::load_thrust_params(thrust_path)?;
        eprintln!(
            "Attaching {} thrust arc(s) to {} orbit(s)",
            thrust.arcs.len(),
            batch.len()
        );
        for orbit in batch.orbits.iter_mut() {
            orbit.thrust = Some(thrust.clone());
        }
    }

    let config = PropagationConfig {
        force_model: args.force_model.to_empyrean(),
        uncertainty_method: args.uncertainty_method.to_empyrean(),
        ..PropagationConfig::default()
    };

    eprintln!(
        "Propagating {} orbit(s) to MJD {:.1}...",
        batch.len(),
        args.epoch
    );
    let t1 = Instant::now();
    let result = ctx
        .propagate(
            &batch.orbits,
            &[empyrean::Epoch::from_mjd_tdb(args.epoch)],
            &config,
        )
        .context("propagation failed")?;
    eprintln!("Propagation complete ({:.1}s)", t1.elapsed().as_secs_f64());

    print_event_summary(&result);

    if args.tagged_covariance {
        print_tagged_covariance_series(&batch, &result);
    }

    eprintln!("\n  Output: {}/", args.out_dir.display());
    let propagated = propagated_to_batch(&batch, &result);
    output::write_orbits(&args.out_dir, "states", &propagated, args.format)?;
    output::write_events(&args.out_dir, "events", &result.events, args.format)?;

    Ok(())
}

fn print_event_summary(result: &PropagationResult) {
    if result.events.is_empty() {
        eprintln!("\n  No events detected.");
    } else {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for ev in &result.events {
            *counts.entry(ev.event_type.as_str()).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        eprintln!("\n  Events:");
        for (name, count) in &sorted {
            eprintln!("    {:<25} {}", name, count);
        }
    }
}

/// Print the resolved-kind tagged covariance series for every orbit.
///
/// One labelled block per orbit (by `orbit_id`); within a block one row
/// per output epoch with the epoch (MJD TDB), resolved covariance kind,
/// position 1σ in km, definiteness, and solved width. If an orbit's
/// series cannot be produced (e.g. the input orbit carried no
/// covariance), a one-line note is printed for that orbit and the
/// remaining orbits are still rendered.
fn print_tagged_covariance_series(input: &OrbitBatch, result: &PropagationResult) {
    use empyrean::{CovarianceKind, CovarianceQuality};

    fn kind_str(kind: CovarianceKind) -> &'static str {
        match kind {
            CovarianceKind::Linear => "Linear",
            CovarianceKind::SecondOrder => "SecondOrder",
            CovarianceKind::ThirdOrder => "ThirdOrder",
            CovarianceKind::Mixture => "Mixture",
            CovarianceKind::MonteCarlo => "MonteCarlo",
            CovarianceKind::SigmaPoint => "SigmaPoint",
        }
    }

    fn quality_str(quality: CovarianceQuality) -> String {
        match quality {
            CovarianceQuality::PositiveDefinite => "pos-def".to_string(),
            CovarianceQuality::Indefinite { min_eig } => {
                format!("indefinite(min_eig={min_eig:.2e})")
            }
            CovarianceQuality::Repaired { min_eig } => {
                format!("repaired(min_eig={min_eig:.2e})")
            }
        }
    }

    eprintln!("\n  Tagged covariance (resolved kind):");
    for orbit_index in 0..input.len() {
        let label = input
            .orbit_ids
            .get(orbit_index)
            .cloned()
            .unwrap_or_else(|| format!("orbit_{orbit_index}"));
        eprintln!("\n  {label}");
        match result.covariance_series_cartesian(orbit_index) {
            Ok(series) => {
                eprintln!(
                    "    {:>14} {:>12} {:>12} {:>12} {:>12} {:>5} Quality",
                    "MJD_TDB", "Kind", "σx_km", "σy_km", "σz_km", "Width"
                );
                eprintln!("    {}", "-".repeat(78));
                for tagged in &series {
                    // Per-axis position 1σ: sqrt of the upper-left 3×3
                    // diagonal (AU² → AU) scaled to km.
                    let sigma_km = |i: usize| tagged.matrix[i][i].max(0.0).sqrt() * AU_KM;
                    let mjd = tagged.epoch.mjd_tdb().unwrap_or(f64::NAN);
                    eprintln!(
                        "    {:>14.5} {:>12} {:>12.3} {:>12.3} {:>12.3} {:>5} {}",
                        mjd,
                        kind_str(tagged.kind),
                        sigma_km(0),
                        sigma_km(1),
                        sigma_km(2),
                        tagged.solved_width,
                        quality_str(tagged.quality),
                    );
                }
            }
            Err(e) => {
                eprintln!("    no tagged covariance available: {e}");
            }
        }
    }
}

/// Build an [`OrbitBatch`] of propagated states using the input batch's
/// orbit/object IDs (carried positionally) and the propagation result's
/// state vectors at the target epoch(s).
fn propagated_to_batch(input: &OrbitBatch, result: &PropagationResult) -> OrbitBatch {
    use empyrean::{CoordinateState, Frame, Orbit, Representation};
    let mut orbits = Vec::with_capacity(result.states.len());
    let n_in = input.len();
    let n_times = if n_in > 0 {
        result.states.len() / n_in
    } else {
        1
    };
    let mut orbit_ids = Vec::with_capacity(result.states.len());
    let mut object_ids = Vec::with_capacity(result.states.len());
    for (i, state) in result.states.iter().enumerate() {
        let orbit_idx = if n_times > 0 { i / n_times } else { 0 };
        let id = input
            .orbit_ids
            .get(orbit_idx)
            .cloned()
            .unwrap_or_else(|| format!("orbit_{orbit_idx}"));
        let obj = input.object_ids.get(orbit_idx).cloned().flatten();
        let mut cs = CoordinateState::cartesian(
            state.epoch,
            [
                state.position[0],
                state.position[1],
                state.position[2],
                state.velocity[0],
                state.velocity[1],
                state.velocity[2],
            ],
            state.frame,
            state.origin,
        );
        if let Some(c) = state.covariance {
            cs = cs.with_covariance(c);
        }
        let _ = (Frame::ICRF, Representation::Cartesian);
        let template = input
            .orbits
            .get(orbit_idx)
            .cloned()
            .unwrap_or_else(|| Orbit::new(cs));
        // Carry the input orbit's non-grav + photometry forward; only
        // the state is replaced with the parsed `cs`. `..template`
        // filled non-state fields when this code was written; once
        // `Orbit` grew `phot_system` / `h_mag` / `slope1` / `slope2`
        // for ephemeris-mag support, the explicit struct literal here
        // stopped covering the full field set. Use functional update
        // syntax so future additions to `Orbit` don't break this site.
        let orbit = Orbit {
            state: cs,
            ..template
        };
        orbits.push(orbit);
        orbit_ids.push(id);
        object_ids.push(obj);
    }
    OrbitBatch {
        orbits,
        orbit_ids,
        object_ids,
    }
}

/// Parse a daemon-wire uncertainty-method string. CLI-side input is
/// already validated by clap via [`UncertaintyMethodArg`]; this exists
/// for the daemon server, which receives the value as a JSON string
/// over the Unix socket and must reject anything outside the supported
/// set rather than silently coercing to FirstOrder.
pub(crate) fn parse_uncertainty_method(s: &str) -> Result<UncertaintyMethod, String> {
    match s {
        "first-order" | "first" => Ok(UncertaintyMethod::FirstOrder),
        "second-order" | "second" => Ok(UncertaintyMethod::SecondOrder),
        "sigma-point" => Ok(UncertaintyMethod::sigma_point()),
        "monte-carlo" => Ok(UncertaintyMethod::monte_carlo(1000)),
        _ => Err(format!(
            "unknown uncertainty method '{s}' (expected one of: first-order, second-order, sigma-point, monte-carlo)"
        )),
    }
}

pub(crate) fn format_to_str(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Parquet => "parquet",
        OutputFormat::Json => "json",
        OutputFormat::Csv => "csv",
    }
}

pub(crate) fn parse_format(s: &str) -> OutputFormat {
    match s {
        "json" => OutputFormat::Json,
        "csv" => OutputFormat::Csv,
        _ => OutputFormat::Parquet,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_uncertainty_method_accepts_all_four() {
        assert!(matches!(
            parse_uncertainty_method("first-order"),
            Ok(UncertaintyMethod::FirstOrder)
        ));
        assert!(matches!(
            parse_uncertainty_method("first"),
            Ok(UncertaintyMethod::FirstOrder)
        ));
        assert!(matches!(
            parse_uncertainty_method("second-order"),
            Ok(UncertaintyMethod::SecondOrder)
        ));
        assert!(matches!(
            parse_uncertainty_method("second"),
            Ok(UncertaintyMethod::SecondOrder)
        ));
        assert!(matches!(
            parse_uncertainty_method("sigma-point"),
            Ok(UncertaintyMethod::SigmaPoint { .. })
        ));
        assert!(matches!(
            parse_uncertainty_method("monte-carlo"),
            Ok(UncertaintyMethod::MonteCarlo { .. })
        ));
    }

    #[test]
    fn parse_uncertainty_method_rejects_unknown() {
        // Hidden-fallback regression: must error, not coerce to FirstOrder.
        let err = parse_uncertainty_method("agm").unwrap_err();
        assert!(err.contains("agm"), "error must echo bad input: {err}");
        let err = parse_uncertainty_method("").unwrap_err();
        assert!(
            err.contains("first-order"),
            "error must list valid set: {err}"
        );
    }

    #[test]
    fn arg_string_roundtrip() {
        // Every UncertaintyMethodArg's wire string must round-trip
        // through the daemon-side parser without silent coercion.
        for arg in [
            UncertaintyMethodArg::FirstOrder,
            UncertaintyMethodArg::SecondOrder,
            UncertaintyMethodArg::SigmaPoint,
            UncertaintyMethodArg::MonteCarlo,
        ] {
            parse_uncertainty_method(arg.as_str())
                .unwrap_or_else(|e| panic!("round-trip failed for {}: {e}", arg.as_str()));
        }
    }
}
