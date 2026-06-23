mod commands;
mod daemon;
mod io;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "empyrean",
    version,
    about = "High-fidelity orbital propagation, orbit determination, and ephemeris generation."
)]
struct Cli {
    /// Path to SPICE kernel data directory.
    /// Overrides EMPYREAN_DATA_DIR. Default: ~/.empyrean/data/
    #[arg(long, global = true, env = "EMPYREAN_DATA_DIR")]
    data_dir: Option<std::path::PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download SPICE kernels and initialize data directory.
    Init(commands::init::InitArgs),

    /// Propagate orbits to a target epoch.
    Propagate(commands::propagate::PropagateArgs),

    /// Generate predicted ephemeris (RA/Dec) for observers.
    Ephemeris(commands::ephemeris::EphemerisArgs),

    /// Determine orbits from ADES observations.
    Determine(commands::determine::DetermineArgs),

    /// Query external JPL data services.
    #[command(subcommand)]
    Query(commands::query::QueryCommand),

    /// Manage the API response cache (~/.empyrean/cache/).
    #[command(subcommand)]
    Cache(commands::cache::CacheCommand),

    /// Start the daemon (loads context once, serves requests via Unix socket).
    Serve {
        /// Number of threads for parallel compute (0 = all cores).
        #[arg(long)]
        num_threads: Option<usize>,
    },

    /// Stop a running daemon.
    Stop,

    /// Print version information for empyrean and its dependencies.
    Version,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ForceModel {
    Approximate,
    Basic,
    Standard,
}

impl ForceModel {
    pub fn to_empyrean(self) -> empyrean::ForceModelTier {
        match self {
            Self::Approximate => empyrean::ForceModelTier::Approximate,
            Self::Basic => empyrean::ForceModelTier::Basic,
            Self::Standard => empyrean::ForceModelTier::Standard,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Approximate => "approximate",
            Self::Basic => "basic",
            Self::Standard => "standard",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum UncertaintyMethodArg {
    FirstOrder,
    #[value(alias = "second")]
    SecondOrder,
    SigmaPoint,
    MonteCarlo,
    /// Adaptive — may use Jet1, Second, or AGM, adapting the
    /// uncertainty method automatically through close approaches. Cost amortizes on
    /// heterogeneous batches.
    Auto,
}

impl UncertaintyMethodArg {
    pub fn to_empyrean(self) -> empyrean::UncertaintyMethod {
        match self {
            Self::FirstOrder => empyrean::UncertaintyMethod::FirstOrder,
            Self::SecondOrder => empyrean::UncertaintyMethod::SecondOrder,
            Self::SigmaPoint => empyrean::UncertaintyMethod::sigma_point(),
            Self::MonteCarlo => empyrean::UncertaintyMethod::monte_carlo(1000),
            Self::Auto => empyrean::UncertaintyMethod::auto(),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstOrder => "first-order",
            Self::SecondOrder => "second-order",
            Self::SigmaPoint => "sigma-point",
            Self::MonteCarlo => "monte-carlo",
            Self::Auto => "auto",
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init(args) => commands::init::run(cli.data_dir, args),
        Command::Propagate(args) => commands::propagate::run(cli.data_dir, args),
        Command::Ephemeris(args) => commands::ephemeris::run(cli.data_dir, args),
        Command::Determine(args) => commands::determine::run(cli.data_dir, args),
        Command::Query(cmd) => commands::query::run(cmd),
        Command::Cache(cmd) => commands::cache::run(cmd),
        Command::Serve { num_threads } => {
            let socket = daemon::protocol::default_socket_path();
            daemon::server::serve(cli.data_dir.as_deref(), &socket, num_threads)
        }
        Command::Version => {
            // Print the CLI's own version, then the empyrean stack
            // (empyrean-core + villeneuve + scott + nolan, with their
            // git-populated `<tag>+<sha>` strings) so the reader can
            // tell which build of the cdylib this CLI is talking to.
            println!("empyrean-cli {}", env!("CARGO_PKG_VERSION"));
            match empyrean::version_string() {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("warning: empyrean::version_string failed: {e}");
                }
            }
            Ok(())
        }
        Command::Stop => {
            use daemon::protocol::Request;
            match daemon::client::try_request(&Request::Shutdown) {
                Some(resp) => {
                    eprintln!("{}", resp.message);
                    Ok(())
                }
                None => {
                    eprintln!("No daemon running.");
                    Ok(())
                }
            }
        }
    }
}
