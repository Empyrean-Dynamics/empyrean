use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(clap::Subcommand)]
pub enum QueryCommand {
    /// Fetch an SSB-centered ICRF Cartesian state vector from JPL Horizons.
    HorizonsVectors(HorizonsVectorsArgs),
}

#[derive(clap::Args)]
pub struct HorizonsVectorsArgs {
    /// Horizons COMMAND string (e.g. "99942;", "DES=C/2019 Q4;").
    #[arg(long)]
    pub command: String,

    /// Epoch in MJD, interpreted in the scale given by `--scale`.
    #[arg(long)]
    pub epoch: f64,

    /// Time scale of `--epoch` (`utc` or `tdb`). Required: an MJD carries no
    /// scale, so it is stated rather than assumed.
    #[arg(long, value_enum)]
    pub scale: crate::TimeScaleArg,

    /// Cache directory for Horizons JSON responses. Omit to skip caching.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

pub fn run(cmd: QueryCommand) -> Result<()> {
    match cmd {
        QueryCommand::HorizonsVectors(args) => {
            // Pass the scale-carrying `Epoch` the CLI already parses from
            // `--epoch` + `--scale` straight through; the wrapper converts it
            // to MJD TDB once at the FFI boundary.
            let epoch = args.scale.epoch(args.epoch);
            let (pos, vel) =
                empyrean::query_horizons_vectors(&args.command, epoch, args.cache_dir.as_deref())
                    .context("failed to query JPL Horizons vectors")?;

            eprintln!(
                "Horizons vectors for COMMAND '{}' at epoch {} MJD {} (SSB-centered, ICRF):",
                args.command,
                args.epoch,
                args.scale.as_str().to_uppercase()
            );
            println!(
                "position_au:     {:+.16e} {:+.16e} {:+.16e}",
                pos[0], pos[1], pos[2]
            );
            println!(
                "velocity_au_day: {:+.16e} {:+.16e} {:+.16e}",
                vel[0], vel[1], vel[2]
            );
            Ok(())
        }
    }
}
