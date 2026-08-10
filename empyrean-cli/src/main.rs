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
    /// Overrides EMPYREAN_DATA_DIR. Default: the platform data dir
    /// (~/.local/share/empyrean/data on Linux).
    #[arg(long, global = true, env = "EMPYREAN_DATA_DIR")]
    data_dir: Option<std::path::PathBuf>,

    /// Never reach the network: resolve every kernel from --data-dir
    /// alone and fail, naming the absent files, if any is missing.
    /// No partial load and no lower-tier fallback. Commands that would
    /// otherwise hand the work to a running daemon run in-process
    /// instead, since the daemon's context was built under its own
    /// policy and cannot honour this one retroactively. Setting
    /// EMPYREAN_OFFLINE=1 in the environment has the same effect and
    /// announces itself; it is a floor, so it can only ever turn network
    /// access off, never back on.
    #[arg(long, global = true)]
    no_refresh: bool,

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

    /// Browse output tables — page through a file, or list a directory.
    ///
    /// Reads the Parquet / CSV / JSON the pipeline commands write,
    /// streaming a page at a time so a multi-million-row residuals table
    /// opens instantly. Piped, it writes the whole table as aligned text
    /// instead of paging.
    ///
    /// In the pager: space/enter next page, b previous, ←/→ slide the
    /// column window, / filter rows, Esc clear the filter, q quit.
    ///
    /// This subcommand only reads files. It needs no SPICE kernels and
    /// never loads the engine, so the global --data-dir / --no-refresh
    /// options have no effect on it.
    Show(commands::show::ShowArgs),

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
    let data = commands::DataOptions {
        data_dir: cli.data_dir,
        refresh: !cli.no_refresh,
    };

    match cli.command {
        Command::Init(args) => commands::init::run(&data, args),
        Command::Propagate(args) => commands::propagate::run(&data, args),
        Command::Ephemeris(args) => commands::ephemeris::run(&data, args),
        Command::Determine(args) => commands::determine::run(&data, args),
        // `show` reads files the pipeline already wrote. It needs no
        // kernels and never loads the engine, so it takes no DataOptions.
        Command::Show(args) => commands::show::run(args),
        Command::Query(cmd) => commands::query::run(cmd),
        Command::Cache(cmd) => commands::cache::run(cmd),
        Command::Serve { num_threads } => {
            let socket = daemon::protocol::default_socket_path();
            daemon::server::serve(&data, &socket, num_threads)
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

#[cfg(test)]
mod cli_tests {
    use super::*;

    /// The clap surface itself: `--no-refresh` defaults off and is
    /// `global`, so it parses in either position. A non-global flag would
    /// be rejected after the subcommand and every documented invocation
    /// (`empyrean propagate --no-refresh …`) would fail at parse time.
    #[test]
    fn no_refresh_is_global_and_defaults_off() {
        let default = Cli::try_parse_from(["empyrean", "version"]).expect("bare parse");
        assert!(!default.no_refresh, "must default to refreshing");

        let before = Cli::try_parse_from(["empyrean", "--no-refresh", "version"])
            .expect("before subcommand");
        assert!(before.no_refresh);

        let after =
            Cli::try_parse_from(["empyrean", "version", "--no-refresh"]).expect("after subcommand");
        assert!(after.no_refresh, "--no-refresh must be a global arg");
    }

    /// The flag is inverted exactly once, at the single place that builds
    /// `DataOptions`. Pin the polarity end-to-end so a future refactor
    /// cannot turn `--no-refresh` into a no-op.
    #[test]
    fn no_refresh_maps_to_refresh_false() {
        let cli = Cli::try_parse_from(["empyrean", "--no-refresh", "version"]).unwrap();
        let data = commands::DataOptions {
            data_dir: cli.data_dir,
            refresh: !cli.no_refresh,
        };
        assert!(!data.refresh);

        let cli = Cli::try_parse_from(["empyrean", "version"]).unwrap();
        let data = commands::DataOptions {
            data_dir: cli.data_dir,
            refresh: !cli.no_refresh,
        };
        assert!(data.refresh);
    }

    /// `--compute-stm` is deliberately **not** on `empyrean ephemeris`:
    /// the command has no output channel for the observation-sensitivity
    /// rows the flag produces, so accepting it would be accept-and-drop.
    /// If someone adds the flag, this test fails and points at the
    /// rationale next to `EphemerisArgs`.
    #[test]
    fn ephemeris_does_not_accept_compute_stm() {
        let parsed = Cli::try_parse_from([
            "empyrean",
            "ephemeris",
            "--epoch",
            "60000.0",
            "--compute-stm",
        ]);
        assert!(
            parsed.is_err(),
            "empyrean ephemeris must reject --compute-stm until it can \
             write the sensitivity rows the flag produces (see the \
             EphemerisArgs doc comment)"
        );
    }
}
