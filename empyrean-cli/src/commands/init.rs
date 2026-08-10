use anyhow::{Context, Result};

use super::DataOptions;

#[derive(clap::Args)]
pub struct InitArgs {
    /// Start a background daemon after initialization.
    /// Holds the loaded context in memory for faster subsequent commands.
    #[arg(long)]
    pub serve: bool,

    /// Maximum number of threads the daemon uses for computation.
    /// 0 = all available cores. Only used with --serve.
    #[arg(long)]
    pub num_threads: Option<usize>,
}

pub fn run(data: &DataOptions, args: InitArgs) -> Result<()> {
    // Context construction triggers any required kernel downloads for the
    // Standard tier and verifies the data directory loads cleanly. The
    // context itself is discarded — `init` is a one-shot bootstrap. When
    // `--serve` is passed, the just-downloaded files are immediately
    // re-used by the daemon.
    //
    // Under `--no-refresh` — or `EMPYREAN_OFFLINE=1`, which floors it to
    // the same thing — init is a *verifier*, not a fetcher: the download
    // step is skipped outright (it is the one call here whose whole
    // purpose is to reach the network, and it does not consult the floor
    // itself) and the strict-offline load reports exactly which files the
    // directory is missing.
    let resolved_dir = if data.effective_refresh() {
        eprintln!("Checking kernel files...");
        empyrean::download_data(data.dir()).context("failed to resolve data directory")?
    } else {
        eprintln!("Verifying kernel files (strict offline: nothing will be downloaded)...");
        match data.dir() {
            Some(dir) => dir.to_path_buf(),
            None => empyrean::default_data_dir().context("failed to resolve data directory")?,
        }
    };
    let _ctx = data.build_context().context(if data.effective_refresh() {
        "failed to download/load kernels"
    } else {
        "failed to load kernels offline"
    })?;
    eprintln!("Data directory: {}", resolved_dir.display());
    eprintln!("All kernel files ready.");

    if args.serve {
        let socket_path = crate::daemon::protocol::default_socket_path();
        let serve_data = DataOptions {
            data_dir: Some(resolved_dir),
            refresh: data.refresh,
        };
        crate::daemon::server::serve(&serve_data, &socket_path, args.num_threads)?;
    }

    Ok(())
}
