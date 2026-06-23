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

    /// Epoch in MJD TDB.
    #[arg(long)]
    pub epoch_mjd_tdb: f64,

    /// Cache directory for Horizons JSON responses. Omit to skip caching.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}

pub fn run(cmd: QueryCommand) -> Result<()> {
    match cmd {
        QueryCommand::HorizonsVectors(args) => {
            let (pos, vel) = empyrean::query_horizons_vectors(
                &args.command,
                args.epoch_mjd_tdb,
                args.cache_dir.as_deref(),
            )
            .context("failed to query JPL Horizons vectors")?;

            eprintln!(
                "Horizons vectors for COMMAND '{}' at epoch {} MJD TDB (SSB-centered, ICRF):",
                args.command, args.epoch_mjd_tdb
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
