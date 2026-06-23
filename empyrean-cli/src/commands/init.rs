use std::path::PathBuf;

use anyhow::{Context, Result};

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

pub fn run(data_dir: Option<PathBuf>, args: InitArgs) -> Result<()> {
    eprintln!("Checking kernel files...");
    // `Context::from_data_dir` triggers any required kernel downloads
    // for the Standard tier and verifies the data directory loads
    // cleanly. The context itself is discarded — `init` is a one-shot
    // bootstrap. When `--serve` is passed, the just-downloaded files
    // are immediately re-used by the daemon.
    let resolved_dir =
        empyrean::download_data(data_dir.as_deref()).context("failed to resolve data directory")?;
    let _ctx = empyrean::Context::from_data_dir(data_dir.as_deref())
        .context("failed to download/load kernels")?;
    eprintln!("Data directory: {}", resolved_dir.display());
    eprintln!("All kernel files ready.");

    if args.serve {
        let socket_path = crate::daemon::protocol::default_socket_path();
        crate::daemon::server::serve(Some(&resolved_dir), &socket_path, args.num_threads)?;
    }

    Ok(())
}
