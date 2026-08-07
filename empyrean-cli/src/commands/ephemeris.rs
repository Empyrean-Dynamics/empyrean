use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use super::{DataOptions, load_context};
use crate::ForceModel;
use crate::io::output::OutputFormat;
use crate::io::{orbit_input, output};

/// Why there is no `--compute-stm` here.
///
/// `compute_stm` now reaches the engine on the ephemeris path (the C-ABI
/// converter routes the ephemeris config through the shared propagation
/// converter), and its observable product is the observation-sensitivity
/// rows on `EphemerisResult::sensitivity`. This command has **no channel
/// for those rows**: `write_ephemeris` serializes `entries` only, and
/// nothing here reads `sensitivity`. Exposing the flag would accept the
/// request, pay for the hyperdual integration, and then discard every
/// partial it produced — accept-and-drop, the exact defect class the ABI
/// work removed.
///
/// It is also outside this command's established surface: `ephemeris`
/// exposes exactly one propagation knob (`--force-model`) and no sibling
/// propagation knobs at all, so `--compute-stm` would be new surface
/// rather than parity with an existing pattern. Subsetting is allowed;
/// inventing a knob whose output has nowhere to go is not.
///
/// Adding it is a two-part change: an output channel for the sensitivity
/// rows (writer + `--out-dir` file) *and* the flag, landed together.
///
/// # Known gap: SB441-N16 bodies
///
/// `ephemeris_overlap_policy` is subsetted out for the same reason, and
/// unlike `--compute-stm` its absence costs a capability rather than a
/// detail: an SB441-N16 body (1 Ceres, 2 Pallas, 4 Vesta, …) at Standard
/// tier is both the target and one of its own perturbers, and generating
/// its ephemeris needs either `EXCLUDE_AND_INTEGRATE` or an explicit
/// exclusion — neither of which this command can express. So
/// `empyrean ephemeris --object-id 1` fails, with no in-CLI remedy; use
/// the Python, Rust, or C channel for those sixteen bodies. Closing it
/// means growing this command's propagation surface deliberately, not
/// bolting on one flag.
#[derive(clap::Args)]
pub struct EphemerisArgs {
    /// Object names to query from JPL SBDB.
    #[arg(long = "object-id", num_args = 1..)]
    pub object_ids: Option<Vec<String>>,

    /// Path to an orbits file (.parquet, .json, .csv).
    #[arg(long, conflicts_with = "object_ids")]
    pub input: Option<PathBuf>,

    /// MPC observatory codes (comma-separated). States are computed in
    /// the ICRF / SSB construction basis, which is what ephemeris
    /// generation requires; the basis is not selectable here.
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

pub fn run(data: &DataOptions, args: EphemerisArgs) -> Result<()> {
    // Try daemon first — unless a strict-offline context was requested,
    // which a running daemon's already-built context cannot honour.
    if data.daemon_eligible() {
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
    }

    // In-process fallback.
    let ctx = load_context(data)?;

    let batch = orbit_input::load_orbits(&args.object_ids, &args.input)?;

    let obs_refs: Vec<&str> = args.observers.iter().map(|s| s.as_str()).collect();
    let observers = ctx
        .get_observers(
            &obs_refs,
            &[empyrean::Epoch::from_mjd_tdb(args.epoch)],
            // The construction basis: these observers go straight into
            // ephemeris generation, which requires it.
            empyrean::Frame::ICRF,
            empyrean::Origin::SSB,
        )
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
