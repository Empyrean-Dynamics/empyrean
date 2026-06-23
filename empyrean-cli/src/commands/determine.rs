use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use empyrean::OrbitBatch;

use crate::ForceModel;
use crate::io::output::{self, OutputFormat};

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

    /// Output directory.
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format for fitted orbit + residuals.
    #[arg(long, value_enum, default_value_t = OutputFormat::Parquet)]
    pub format: OutputFormat,
}

pub fn run(data_dir: Option<PathBuf>, args: DetermineArgs) -> Result<()> {
    // Try daemon first.
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
