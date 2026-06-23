//! Unix socket daemon server.
//!
//! Holds a loaded [`empyrean::Context`] and serves requests from CLI clients.
//! Each connection handles exactly one request-response cycle.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use empyrean::{OrbitBatch, PropagationConfig, PropagationResult};

use crate::commands::propagate::{parse_format, parse_uncertainty_method};
use crate::daemon::protocol::{self, Request, Response};
use crate::io::orbit_input;
use crate::io::output::{self, OutputFormat};

/// Start the daemon server. Blocks forever (until Shutdown or signal).
pub fn serve(
    data_dir: Option<&Path>,
    socket_path: &Path,
    _num_threads: Option<usize>,
) -> Result<()> {
    if socket_path.exists() {
        std::fs::remove_file(socket_path).ok();
    }
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind {}", socket_path.display()))?;

    let pid_path = protocol::default_pid_path();
    std::fs::write(&pid_path, std::process::id().to_string())?;

    eprintln!(
        "Daemon starting on {} (PID {})",
        socket_path.display(),
        std::process::id(),
    );

    eprintln!("Loading context...");
    let t0 = std::time::Instant::now();
    let ctx = empyrean::Context::from_data_dir(data_dir).context("failed to load context")?;
    eprintln!("Ready ({:.1}s)", t0.elapsed().as_secs_f64());

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Accept error: {e}");
                continue;
            }
        };

        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            continue;
        }

        let request: Request = match serde_json::from_str(line.trim()) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::err("invalid request", e.to_string());
                write_response(&stream, &resp);
                continue;
            }
        };

        match request {
            Request::Ping => {
                write_response(&stream, &Response::ok("pong"));
            }
            Request::Shutdown => {
                write_response(&stream, &Response::ok("shutting down"));
                break;
            }
            Request::Propagate {
                object_ids,
                input_path,
                epoch,
                force_model,
                uncertainty_method,
                out_dir,
                format,
            } => {
                let resp = handle_propagate(
                    &ctx,
                    &object_ids,
                    &input_path,
                    epoch,
                    &force_model,
                    &uncertainty_method,
                    &out_dir,
                    parse_format(&format),
                );
                write_response(&stream, &resp);
            }
            Request::Ephemeris {
                object_ids,
                input_path,
                observers,
                epoch,
                force_model,
                out_dir,
                format,
            } => {
                let resp = handle_ephemeris(
                    &ctx,
                    &object_ids,
                    &input_path,
                    &observers,
                    epoch,
                    &force_model,
                    &out_dir,
                    parse_format(&format),
                );
                write_response(&stream, &resp);
            }
            Request::Determine {
                ades_path,
                force_model,
                max_iterations,
                out_dir,
                format,
            } => {
                let resp = handle_determine(
                    &ctx,
                    &ades_path,
                    &force_model,
                    max_iterations,
                    &out_dir,
                    parse_format(&format),
                );
                write_response(&stream, &resp);
            }
        }
    }

    std::fs::remove_file(socket_path).ok();
    std::fs::remove_file(&pid_path).ok();
    eprintln!("Daemon stopped.");
    Ok(())
}

fn write_response(mut stream: &std::os::unix::net::UnixStream, resp: &Response) {
    if let Ok(json) = serde_json::to_string(resp) {
        let _ = writeln!(stream, "{json}");
    }
}

fn parse_force_model(s: &str) -> Result<empyrean::ForceModelTier, String> {
    match s {
        "approximate" => Ok(empyrean::ForceModelTier::Approximate),
        "basic" => Ok(empyrean::ForceModelTier::Basic),
        "standard" => Ok(empyrean::ForceModelTier::Standard),
        _ => Err(format!(
            "unknown force model '{s}' (expected one of: approximate, basic, standard)"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_propagate(
    ctx: &empyrean::Context,
    object_ids: &Option<Vec<String>>,
    input_path: &Option<String>,
    epoch: f64,
    force_model: &str,
    uncertainty_method: &str,
    out_dir: &str,
    format: OutputFormat,
) -> Response {
    let input_pb = input_path.as_ref().map(PathBuf::from);
    let batch = match orbit_input::load_orbits(object_ids, &input_pb) {
        Ok(b) => b,
        Err(e) => return Response::err("failed to load orbits", e.to_string()),
    };

    let force_model_tier = match parse_force_model(force_model) {
        Ok(f) => f,
        Err(e) => return Response::err("invalid force_model", e),
    };
    let uncertainty = match parse_uncertainty_method(uncertainty_method) {
        Ok(u) => u,
        Err(e) => return Response::err("invalid uncertainty_method", e),
    };
    let config = PropagationConfig {
        force_model: force_model_tier,
        uncertainty_method: uncertainty,
        ..PropagationConfig::default()
    };

    let result = match ctx.propagate(
        &batch.orbits,
        &[empyrean::Epoch::from_mjd_tdb(epoch)],
        &config,
    ) {
        Ok(r) => r,
        Err(e) => return Response::err("propagation failed", e.to_string()),
    };

    let dir = Path::new(out_dir);
    let propagated = propagated_to_batch(&batch, &result);
    if let Err(e) = output::write_orbits(dir, "states", &propagated, format) {
        return Response::err("failed to write states", e.to_string());
    }
    if let Err(e) = output::write_events(dir, "events", &result.events, format) {
        return Response::err("failed to write events", e.to_string());
    }

    Response::ok(format!(
        "Propagated {} orbit(s), {} events, {} states",
        batch.len(),
        result.events.len(),
        result.states.len()
    ))
}

#[allow(clippy::too_many_arguments)]
fn handle_ephemeris(
    ctx: &empyrean::Context,
    object_ids: &Option<Vec<String>>,
    input_path: &Option<String>,
    observers: &[String],
    epoch: f64,
    force_model: &str,
    out_dir: &str,
    format: OutputFormat,
) -> Response {
    let input_pb = input_path.as_ref().map(PathBuf::from);
    let batch = match orbit_input::load_orbits(object_ids, &input_pb) {
        Ok(b) => b,
        Err(e) => return Response::err("failed to load orbits", e.to_string()),
    };

    let obs_refs: Vec<&str> = observers.iter().map(|s| s.as_str()).collect();
    let obs_states = match ctx.get_observers(&obs_refs, &[empyrean::Epoch::from_mjd_tdb(epoch)]) {
        Ok(o) => o,
        Err(e) => return Response::err("failed to get observers", e.to_string()),
    };

    let force_model_tier = match parse_force_model(force_model) {
        Ok(f) => f,
        Err(e) => return Response::err("invalid force_model", e),
    };
    let cfg = empyrean::EphemerisConfig::with_force_model(force_model_tier);
    let entries = match ctx.generate_ephemeris(&batch.orbits, &obs_states, &cfg) {
        Ok(e) => e.entries,
        Err(e) => return Response::err("ephemeris generation failed", e.to_string()),
    };

    let n = entries.len();
    let dir = Path::new(out_dir);
    if let Err(e) = output::write_ephemeris(dir, "ephemeris", &entries, format) {
        return Response::err("failed to write ephemeris", e.to_string());
    }

    Response::ok(format!(
        "Generated ephemeris: {} row(s) for {} orbit(s), {} observer(s)",
        n,
        batch.len(),
        observers.len()
    ))
}

fn handle_determine(
    ctx: &empyrean::Context,
    ades_path: &str,
    force_model: &str,
    max_iterations: u32,
    out_dir: &str,
    format: OutputFormat,
) -> Response {
    let observations = match ctx.read_ades(ades_path) {
        Ok(o) => o,
        Err(e) => return Response::err("failed to read ADES", e.to_string()),
    };

    let force_model_tier = match parse_force_model(force_model) {
        Ok(f) => f,
        Err(e) => return Response::err("invalid force_model", e),
    };
    let cfg = empyrean::ODConfig {
        force_model: force_model_tier,
        max_iterations,
        ..empyrean::ODConfig::default()
    };
    let result = match ctx.determine(&observations, None, &cfg) {
        Ok(r) => r,
        Err(e) => return Response::err("orbit determination failed", e.to_string()),
    };

    let dir = Path::new(out_dir);
    if let Err(e) = std::fs::create_dir_all(dir) {
        return Response::err("failed to create output directory", e.to_string());
    }

    // `result.orbit` is already a re-feedable `Orbit` (state + covariance +
    // non-grav); write it straight out as a single-entry batch.
    let fitted_batch = OrbitBatch {
        orbits: vec![result.orbit.clone()],
        orbit_ids: vec!["fitted".to_string()],
        object_ids: vec![None],
    };
    if let Err(e) = output::write_orbits(dir, "fitted_orbit", &fitted_batch, format) {
        return Response::err("failed to write fitted orbit", e.to_string());
    }
    let resid_path = dir.join(match format {
        OutputFormat::Parquet => "residuals.parquet",
        OutputFormat::Json => "residuals.json",
        OutputFormat::Csv => "residuals.csv",
    });
    if let Err(e) = output::write_residuals(&resid_path, &result.residuals, format) {
        return Response::err("failed to write residuals", e.to_string());
    }

    let s = &result.summary;
    Response::ok(format!(
        "OD complete: converged={}, iter={}, RMS_RA={:.2}\", RMS_Dec={:.2}\", obs={}",
        result.converged, result.iterations, s.rms_ra_arcsec, s.rms_dec_arcsec, s.num_obs,
    ))
}

fn propagated_to_batch(input: &OrbitBatch, result: &PropagationResult) -> OrbitBatch {
    use empyrean::{CoordinateState, Orbit};
    let n_in = input.len();
    let n_times = if n_in > 0 {
        result.states.len() / n_in
    } else {
        1
    };
    let mut orbits = Vec::with_capacity(result.states.len());
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
        // Carry the input orbit's non-grav + photometry forward; only
        // the state is replaced with the parsed `cs`. Functional
        // update syntax (`..template`) so future additions to
        // `Orbit` don't break this site.
        let orbit = match input.orbits.get(orbit_idx).cloned() {
            Some(template) => Orbit {
                state: cs,
                ..template
            },
            None => Orbit::new(cs),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ForceModel;

    #[test]
    fn parse_force_model_accepts_all_three() {
        assert!(matches!(
            parse_force_model("approximate"),
            Ok(empyrean::ForceModelTier::Approximate)
        ));
        assert!(matches!(
            parse_force_model("basic"),
            Ok(empyrean::ForceModelTier::Basic)
        ));
        assert!(matches!(
            parse_force_model("standard"),
            Ok(empyrean::ForceModelTier::Standard)
        ));
    }

    #[test]
    fn parse_force_model_rejects_unknown() {
        // Hidden-fallback regression: unknown values must error, not
        // silently coerce to Standard.
        let err = parse_force_model("full").unwrap_err();
        assert!(err.contains("full"), "error must echo bad input: {err}");
        let err = parse_force_model("").unwrap_err();
        assert!(
            err.contains("approximate"),
            "error must list valid set: {err}"
        );
    }

    #[test]
    fn force_model_arg_string_roundtrip() {
        // Every ForceModel arg's wire string must round-trip through
        // the daemon-side parser without silent coercion.
        for arg in [
            ForceModel::Approximate,
            ForceModel::Basic,
            ForceModel::Standard,
        ] {
            parse_force_model(arg.as_str())
                .unwrap_or_else(|e| panic!("round-trip failed for {}: {e}", arg.as_str()));
        }
    }
}
