use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::ForceModel;
use crate::io::output::OutputFormat;
use crate::io::{orbit_input, output};

#[derive(clap::Args)]
pub struct EphemerisArgs {
    /// Object names to query from JPL SBDB.
    #[arg(long = "object-id", num_args = 1..)]
    pub object_ids: Option<Vec<String>>,

    /// Path to an orbits file (.parquet, .json, .csv).
    #[arg(long, conflicts_with = "object_ids")]
    pub input: Option<PathBuf>,

    /// MPC observatory codes (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub observers: Vec<String>,

    /// Observation epoch (MJD TDB).
    #[arg(long)]
    pub epoch: f64,

    /// Force model tier.
    #[arg(long, default_value = "standard")]
    pub force_model: ForceModel,

    /// Output directory.
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Parquet)]
    pub format: OutputFormat,
}

pub fn run(data_dir: Option<PathBuf>, args: EphemerisArgs) -> Result<()> {
    // Try daemon first.
    let request = crate::daemon::protocol::Request::Ephemeris {
        object_ids: args.object_ids.clone(),
        input_path: args.input.as_ref().map(|p| p.display().to_string()),
        observers: args.observers.clone(),
        epoch: args.epoch,
        force_model: args.force_model.as_str().to_string(),
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

    let batch = orbit_input::load_orbits(&args.object_ids, &args.input)?;

    let obs_refs: Vec<&str> = args.observers.iter().map(|s| s.as_str()).collect();
    let observers = ctx
        .get_observers(&obs_refs, &[empyrean::Epoch::from_mjd_tdb(args.epoch)])
        .context("failed to get observer states")?;
    eprintln!(
        "Observers: {} code(s) x 1 epoch(s) = {} state(s)",
        args.observers.len(),
        observers.len()
    );

    eprintln!("Generating ephemeris for {} orbit(s)...", batch.len());
    let t1 = Instant::now();
    let config = empyrean::EphemerisConfig::with_force_model(args.force_model.to_empyrean());
    let entries = ctx
        .generate_ephemeris(&batch.orbits, &observers, &config)
        .context("ephemeris generation failed")?
        .entries;
    eprintln!("Ephemeris complete ({:.1}s)", t1.elapsed().as_secs_f64());

    eprintln!("\n  Output: {}/", args.out_dir.display());
    output::write_ephemeris(&args.out_dir, "ephemeris", &entries, args.format)?;

    eprintln!(
        "\n  Summary: {} orbit(s), {} observer(s), {} row(s)",
        batch.len(),
        args.observers.len(),
        entries.len()
    );

    Ok(())
}
