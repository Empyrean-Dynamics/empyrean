use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use empyrean::OrbitBatch;

use crate::ForceModel;
use crate::io::output::{self, OutputFormat};

/// Which parameters differential correction solves for. `dt` / `amrat` /
/// `non-grav-amrat` (and any `--thrust-segments`) map to the wide
/// `Explicit` solve; the rest to the coarse solve-for set.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SolveForArg {
    /// Escalate state-only → non-grav automatically on a poor fit.
    #[default]
    Auto,
    /// Solve the 6-element state only.
    StateOnly,
    /// State + Marsden A1/A2/A3 non-grav coefficients.
    NonGrav,
    /// State + Marsden + the non-grav time delay DT.
    Dt,
    /// State + SRP AMRAT.
    Amrat,
    /// State + Marsden + SRP AMRAT.
    NonGravAmrat,
}

#[derive(clap::Args)]
pub struct DetermineArgs {
    /// Path to ADES PSV observation file.
    pub ades_file: PathBuf,

    /// Force model tier.
    #[arg(long, default_value = "standard")]
    pub force_model: ForceModel,

    /// Maximum differential correction iterations.
    #[arg(long, default_value = "20")]
    pub max_iterations: u32,

    /// Parameters to solve for.
    #[arg(long, value_enum, default_value_t = SolveForArg::Auto)]
    pub solve_for: SolveForArg,

    /// Number of thrust Δv segments to solve (0 = none). Requires the
    /// burn windows to be bracketed by observations.
    #[arg(long, default_value = "0")]
    pub thrust_segments: u32,

    /// Run a post-OD photometric H/G fit over the arc's magnitudes.
    #[arg(long)]
    pub photometry: bool,

    /// Output directory.
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format for fitted orbit + residuals.
    #[arg(long, value_enum, default_value_t = OutputFormat::Parquet)]
    pub format: OutputFormat,
}

/// Build the wrapper's `SolveForParams` from the CLI selection. Any axis
/// the coarse variants can't name (DT / AMRAT / thrust) becomes an
/// `Explicit` solve, at parity with empyrean-core.
fn build_solve_for(mode: SolveForArg, thrust_segments: u32) -> empyrean::SolveForParams {
    use empyrean::{SolveFor, SolveForParams};
    if mode == SolveForArg::Auto && thrust_segments == 0 {
        return SolveForParams::Auto;
    }
    let (marsden, dt, amrat) = match mode {
        SolveForArg::Auto | SolveForArg::StateOnly => (false, false, false),
        SolveForArg::NonGrav => (true, false, false),
        SolveForArg::Dt => (true, true, false),
        SolveForArg::Amrat => (false, false, true),
        SolveForArg::NonGravAmrat => (true, false, true),
    };
    if !dt && !amrat && thrust_segments == 0 {
        return if marsden {
            SolveForParams::StateAndNonGrav
        } else {
            SolveForParams::StateOnly
        };
    }
    SolveForParams::Explicit(SolveFor {
        marsden,
        dt,
        amrat,
        thrust_segments,
    })
}

pub fn run(data_dir: Option<PathBuf>, args: DetermineArgs) -> Result<()> {
    // The daemon protocol only carries force_model + max_iterations, so a
    // fitting request (non-grav / DT / AMRAT / thrust / photometry) must
    // run in-process — the daemon can't express it yet.
    let uses_fitting = args.solve_for != SolveForArg::Auto
        || args.thrust_segments > 0
        || args.photometry;
    if !uses_fitting {
        let request = crate::daemon::protocol::Request::Determine {
            ades_path: args.ades_file.display().to_string(),
            force_model: args.force_model.as_str().to_string(),
            max_iterations: args.max_iterations,
            out_dir: args.out_dir.display().to_string(),
            format: super::propagate::format_to_str(args.format).into(),
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

    let path_str = args.ades_file.display().to_string();
    let observations = ctx
        .read_ades(&path_str)
        .context("failed to read ADES file")?;
    eprintln!(
        "Read {} observation(s) from {}",
        observations.len(),
        args.ades_file.display()
    );

    eprintln!("Running orbit determination...");
    let t1 = Instant::now();
    let config = empyrean::ODConfig {
        force_model: args.force_model.to_empyrean(),
        max_iterations: args.max_iterations,
        solve_for: build_solve_for(args.solve_for, args.thrust_segments),
        photometry: args
            .photometry
            .then(empyrean::PhotometryConfig::default),
        ..empyrean::ODConfig::default()
    };
    let result = ctx
        .determine(&observations, None, &config)
        .context("orbit determination failed")?;
    eprintln!("OD complete ({:.1}s)", t1.elapsed().as_secs_f64());

    let s = &result.summary;
    eprintln!(
        "\n  {:<9} {:>5} {:>8} {:>8} {:>5}",
        "Converged", "Iter", "RMS_RA\"", "RMS_Dec\"", "Obs"
    );
    eprintln!("  {}", "-".repeat(40));
    eprintln!(
        "  {:<9} {:>5} {:>8.2} {:>8.2} {:>5}",
        if result.converged { "yes" } else { "no" },
        result.iterations,
        s.rms_ra_arcsec,
        s.rms_dec_arcsec,
        s.num_obs,
    );

    // Wide-parameter fitting readback (v0.9.0). Each line appears only
    // when that axis was actually solved — a missing line reads as "not
    // recovered", never a zero.
    if let Some(sc) = &result.solved_covariance {
        eprintln!("  Solved covariance width: {}", sc.width);
    }
    if let Some(dt) = result.dt_delta {
        eprintln!("  Non-grav time delay  ΔDT = {dt:.4} d");
    }
    if let Some(a) = result.amrat_delta {
        eprintln!("  SRP AMRAT correction     = {a:.4e} m^2/kg");
    }
    for (i, dv) in result.thrust_delta_m_per_s.iter().enumerate() {
        eprintln!(
            "  Thrust dv[{i}] = [{:.3}, {:.3}, {:.3}] m/s",
            dv[0], dv[1], dv[2]
        );
    }
    if let Some(ph) = &result.photometry {
        eprintln!(
            "  Photometry: H = {:.3}  G1 = {:.3}  (model {:?}, chi2_r {:.2})",
            ph.h, ph.slope1, ph.model_used, ph.reduced_chi2
        );
    }

    std::fs::create_dir_all(&args.out_dir).context("failed to create output directory")?;

    // Write the fitted orbit as a single-entry batch. `result.orbit` is
    // already a re-feedable `Orbit` carrying state + covariance + non-grav.
    let fitted_batch = OrbitBatch {
        orbits: vec![result.orbit.clone()],
        orbit_ids: vec!["fitted".to_string()],
        object_ids: vec![None],
    };
    output::write_orbits(&args.out_dir, "fitted_orbit", &fitted_batch, args.format)?;

    // Write residuals.
    let resid_path = args
        .out_dir
        .join(format!("residuals.{}", format_extension(args.format)));
    output::write_residuals(&resid_path, &result.residuals, args.format)?;
    eprintln!(
        "  {} ({} rows)",
        resid_path.display(),
        result.residuals.len()
    );

    eprintln!("\n  Output: {}/", args.out_dir.display());
    Ok(())
}

fn format_extension(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Parquet => "parquet",
        OutputFormat::Json => "json",
        OutputFormat::Csv => "csv",
    }
}
