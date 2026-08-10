//! `empyrean determine` — fit orbits to ADES astrometry.
//!
//! Batch-first: the ADES file is grouped by object identifier and every
//! object is fitted. The command writes one row per delivered object to
//! `fitted_orbits`, one row per **input** object to `fit_summary`
//! (delivered or not), and the residuals of every delivered fit tagged
//! with the object they belong to. Nothing is discarded on the way out.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use empyrean::{DetermineResults, FitSummaryRow, OrbitBatch};

use super::{DataOptions, load_context};
use crate::ForceModel;
use crate::io::output::{self, OutputFormat};

/// Every object delivered a fit.
pub const EXIT_ALL_DELIVERED: i32 = 0;
/// Some objects delivered and some failed. The outputs are written; the
/// failures are named on stderr and carried in `fit_summary`.
pub const EXIT_PARTIAL: i32 = 3;
/// The batch ran and no object delivered a fit. `fit_summary` still
/// records every attempt and why it failed.
pub const EXIT_NONE_DELIVERED: i32 = 4;

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

    /// SRP area-to-mass ratio AMRAT (m²/kg) — the fittable SRP parameter.
    /// Priors the SRP slot on the seed orbit. Required with
    /// `--solve-for amrat` or `--solve-for non-grav-amrat`.
    #[arg(long)]
    pub amrat: Option<f64>,

    /// SRP radiation coefficient Cr for the AMRAT prior (default 1.0 when
    /// `--amrat` is given). Used with `--solve-for amrat` or
    /// `--solve-for non-grav-amrat`.
    #[arg(long)]
    pub cr: Option<f64>,

    /// Prior variance on AMRAT ((m²/kg)²) — opens the AMRAT column in the
    /// refine. Required with `--solve-for amrat` or
    /// `--solve-for non-grav-amrat`.
    #[arg(long)]
    pub amrat_variance: Option<f64>,

    /// Non-grav time delay DT (days) — the fittable delay parameter. Priors
    /// the DT value on the seed orbit; omit to keep the seed's fitted value.
    /// Used with `--solve-for dt`.
    #[arg(long)]
    pub dt: Option<f64>,

    /// Prior variance on the non-grav time delay DT (days²) — opens the DT
    /// column in the refine. Required with `--solve-for dt`.
    #[arg(long)]
    pub dt_variance: Option<f64>,

    /// Number of thrust Δv segments to solve (0 = none). Requires the
    /// burn windows to be bracketed by observations.
    #[arg(long, default_value = "0")]
    pub thrust_segments: u32,

    /// Run a post-OD photometric H/G fit over the arc's magnitudes.
    #[arg(long)]
    pub photometry: bool,

    /// Output directory. Receives `fitted_orbits.<format>` (one row per
    /// delivered object), `fit_summary.parquet` AND `fit_summary.csv`
    /// (one row per input object, delivered or failed), and
    /// `residuals.<format>` (every delivered fit's residuals, tagged
    /// with their object_id).
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format for the fitted orbits + residuals. The
    /// per-object `fit_summary` is always written as both parquet and
    /// CSV regardless of this setting, so it can be read at a terminal
    /// without a parquet tool.
    #[arg(long, value_enum, default_value_t = OutputFormat::Parquet)]
    pub format: OutputFormat,
}

/// The `(marsden, dt, amrat)` fit axes a coarse [`SolveForArg`] expands to.
/// Shared by [`build_solve_for`] and the refine-path priming so the wide
/// solve and the priors that open its columns never drift apart.
fn solve_for_axes(mode: SolveForArg) -> (bool, bool, bool) {
    match mode {
        SolveForArg::Auto | SolveForArg::StateOnly => (false, false, false),
        SolveForArg::NonGrav => (true, false, false),
        SolveForArg::Dt => (true, true, false),
        SolveForArg::Amrat => (false, false, true),
        SolveForArg::NonGravAmrat => (true, false, true),
    }
}

/// Build the wrapper's `SolveForParams` from the CLI selection. Any axis
/// the coarse variants can't name (DT / AMRAT / thrust) becomes an
/// `Explicit` solve, at parity with empyrean-core.
fn build_solve_for(mode: SolveForArg, thrust_segments: u32) -> empyrean::SolveForParams {
    use empyrean::{SolveFor, SolveForParams};
    if mode == SolveForArg::Auto && thrust_segments == 0 {
        return SolveForParams::Auto;
    }
    let (marsden, dt, amrat) = solve_for_axes(mode);
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

/// Reject an invocation whose prior flags don't match the requested
/// `--solve-for` axis. Two failure modes, both loud:
///   * a prior was set for an axis that isn't being solved (it would be
///     silently ignored), or
///   * a refine-path axis (DT / AMRAT) is requested without the prior
///     variance that opens its column.
///
/// Runs before any context load so a misconfigured call fails fast.
fn validate_prior_flags(args: &DetermineArgs) -> Result<()> {
    let (_marsden, dt_axis, amrat_axis) = solve_for_axes(args.solve_for);

    // Priors set without a matching axis: fail rather than silently drop a
    // value the user deliberately provided.
    if !amrat_axis {
        if args.amrat.is_some() {
            anyhow::bail!(
                "--amrat is only used with --solve-for amrat or --solve-for non-grav-amrat"
            );
        }
        if args.cr.is_some() {
            anyhow::bail!("--cr is only used with --solve-for amrat or --solve-for non-grav-amrat");
        }
        if args.amrat_variance.is_some() {
            anyhow::bail!(
                "--amrat-variance is only used with --solve-for amrat or --solve-for non-grav-amrat"
            );
        }
    }
    if !dt_axis {
        if args.dt.is_some() {
            anyhow::bail!("--dt is only used with --solve-for dt");
        }
        if args.dt_variance.is_some() {
            anyhow::bail!("--dt-variance is only used with --solve-for dt");
        }
    }

    // A requested refine-path axis needs its prior to open the column.
    if amrat_axis {
        if args.amrat.is_none() {
            anyhow::bail!(
                "an AMRAT solve requires --amrat <m^2/kg> (the SRP area-to-mass ratio to prior)"
            );
        }
        if args.amrat_variance.is_none() {
            anyhow::bail!(
                "an AMRAT solve requires --amrat-variance <(m^2/kg)^2> to open the AMRAT column"
            );
        }
    }
    if dt_axis && args.dt_variance.is_none() {
        anyhow::bail!("--solve-for dt requires --dt-variance <days^2> to open the DT column");
    }
    Ok(())
}

/// Observations belonging to one ADES object.
///
/// The engine groups a batch by `permID` → `provID` → `trkSub`; this
/// partitions the same way (via [`empyrean::Observation::object_id`]) so
/// the per-object slice the refine path fits is exactly the slice the
/// determine pass fitted.
fn observations_for_object(
    all: &empyrean::Observations,
    object_id: &str,
) -> Result<empyrean::Observations> {
    let optical: Vec<empyrean::Observation> = all
        .iter()
        .filter(|o| o.object_id() == Some(object_id))
        .collect();
    let radar: Vec<empyrean::RadarObservation> = all
        .radar()
        .into_iter()
        .filter(|r| r.object_id() == Some(object_id))
        .collect();
    empyrean::Observations::from_arrays(&optical, &radar)
        .with_context(|| format!("failed to slice observations for object {object_id}"))
}

/// Run the determine → prime → refine two-pass used when a refine-path axis
/// (DT or AMRAT) is requested. Pass 1 is the coarse batch solve *without*
/// that axis; the priors from the flags are then attached to each delivered
/// object's re-feedable `result.orbit`, and pass 2 is the wide Bayesian
/// refine of that object against **its own** observations.
///
/// Assumes [`validate_prior_flags`] has already accepted `args`, so the
/// required prior flags for the requested axes are present.
fn run_refine_path(
    ctx: &empyrean::Context,
    observations: &empyrean::Observations,
    args: &DetermineArgs,
) -> Result<DetermineResults> {
    use empyrean::{DetermineEntry, DetermineFailure, DetermineFailureKind};

    let (marsden, dt_axis, amrat_axis) = solve_for_axes(args.solve_for);

    // Pass 1: the coarse solve WITHOUT the refine-path axis — state, plus the
    // Marsden non-grav when the wide solve needs it. Its `result.orbit`
    // re-feeds as a Cartesian orbit carrying state + covariance (and the
    // fitted non-grav 3×3 covariance when Marsden was solved), which is what
    // primes the non-grav column for a DT refine.
    let base_solve = if marsden {
        empyrean::SolveForParams::StateAndNonGrav
    } else {
        empyrean::SolveForParams::StateOnly
    };
    let base_config = empyrean::ODConfig {
        force_model: args.force_model.to_empyrean(),
        max_iterations: args.max_iterations,
        solve_for: base_solve,
        photometry: args.photometry.then(empyrean::PhotometryConfig::default),
        ..empyrean::ODConfig::default()
    };
    eprintln!(
        "  Pass 1 (seed): {}",
        if marsden {
            "state + non-grav"
        } else {
            "state-only"
        }
    );
    let seeds = ctx
        .determine(observations, None, &base_config)
        .context("refine-path pass 1 (seed solve) failed")?;

    let wide_config = empyrean::ODConfig {
        force_model: args.force_model.to_empyrean(),
        max_iterations: args.max_iterations,
        solve_for: build_solve_for(args.solve_for, args.thrust_segments),
        photometry: args.photometry.then(empyrean::PhotometryConfig::default),
        ..empyrean::ODConfig::default()
    };
    eprintln!(
        "  Pass 2 (wide refine): opening the requested column(s) for {} object(s)",
        seeds.delivered_count()
    );

    // Pass 2 runs per object. A seed that failed in pass 1 keeps its own
    // failure — it is never replaced by a refine error it never reached.
    let mut entries: Vec<DetermineEntry> = Vec::with_capacity(seeds.len());
    for entry in seeds.iter() {
        let object_id = entry.object_id.clone();
        let seed = match &entry.outcome {
            Ok(fit) => fit,
            Err(failure) => {
                entries.push(DetermineEntry {
                    object_id,
                    outcome: Err(failure.clone()),
                });
                continue;
            }
        };

        // Prime: attach the requested priors to the seed orbit. The prior
        // variance is the trigger that opens each wide column in the refine.
        let mut primed = seed.orbit.clone();
        if amrat_axis {
            // Present by construction — validate_prior_flags required both.
            let amrat = args
                .amrat
                .expect("validate_prior_flags requires --amrat for an AMRAT solve");
            let amrat_variance = args
                .amrat_variance
                .expect("validate_prior_flags requires --amrat-variance for an AMRAT solve");
            primed = primed
                .with_srp(amrat, args.cr.unwrap_or(1.0))
                .with_srp_amrat_variance(Some(amrat_variance));
        }
        if dt_axis {
            let dt_variance = args
                .dt_variance
                .expect("validate_prior_flags requires --dt-variance for a DT solve");
            // Use the supplied DT as the value prior when given; otherwise keep
            // whatever the seed carries (None from a StateAndNonGrav pass 1).
            if let Some(dt) = args.dt {
                primed = primed.with_non_grav_dt(Some(dt));
            }
            primed = primed.with_non_grav_dt_variance(Some(dt_variance));
        }

        let object_obs = observations_for_object(observations, &object_id)?;
        let outcome =
            ctx.refine(&primed, &object_obs, &wide_config)
                .map_err(|e| DetermineFailure {
                    object_id: object_id.clone(),
                    message: format!("wide refine failed: {}", e.message),
                    kind: DetermineFailureKind::OD,
                });
        entries.push(DetermineEntry { object_id, outcome });
    }

    Ok(DetermineResults::from_entries(
        entries,
        seeds.unmatched_orbit_ids().to_vec(),
    ))
}

/// Print the per-object outcome table, then list the failures.
///
/// The table is the command's primary human output: one line per object
/// the batch attempted, so a partially successful run reads as such
/// rather than as a success with a shorter orbit file.
fn report_batch(results: &DetermineResults) {
    eprintln!(
        "\n  {:<16} {:>9} {:>5} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7}",
        "Object", "Converged", "Iter", "RMS_RA\"", "RMS_Dec\"", "Obs", "Sel", "Fit", "Extrap"
    );
    eprintln!("  {}", "-".repeat(84));
    for entry in results.iter() {
        match &entry.outcome {
            Ok(fit) => {
                let s = &fit.summary;
                eprintln!(
                    "  {:<16} {:>9} {:>5} {:>8.2} {:>8.2} {:>6} {:>6} {:>7} {:>7}",
                    entry.object_id,
                    if fit.converged { "yes" } else { "no" },
                    fit.iterations,
                    s.rms_ra_arcsec,
                    s.rms_dec_arcsec,
                    s.num_obs,
                    s.num_selected,
                    yes_no(fit.acceptability.fit_acceptable),
                    yes_no(fit.acceptability.extrapolation_acceptable),
                );
            }
            Err(_) => {
                eprintln!(
                    "  {:<16} {:>9} {:>5} {:>8} {:>8} {:>6} {:>6} {:>7} {:>7}",
                    entry.object_id, "FAILED", "-", "-", "-", "-", "-", "-", "-"
                );
            }
        }
    }

    // Failures last and in full, so they are the last thing on screen.
    let failures: Vec<_> = results.failures().collect();
    if !failures.is_empty() {
        eprintln!(
            "\n  {} of {} object(s) produced no orbit:",
            failures.len(),
            results.len()
        );
        for f in failures {
            eprintln!("    {}: {}", f.object_id, f.message);
        }
    }
    if !results.unmatched_orbit_ids().is_empty() {
        eprintln!(
            "\n  {} seed orbit(s) matched no observations and constrained nothing: {}",
            results.unmatched_orbit_ids().len(),
            results.unmatched_orbit_ids().join(", ")
        );
    }
}

fn yes_no(v: bool) -> &'static str {
    if v { "yes" } else { "no" }
}

/// Per-object wide-fitting readback for the axes that were solved. Each
/// line appears only when that axis was actually solved — a missing line
/// reads as "not recovered", never a zero.
fn report_wide_axes(results: &DetermineResults) {
    for (object_id, fit) in results.delivered() {
        let mut printed_header = false;
        let header = |printed: &mut bool| {
            if !*printed {
                eprintln!("\n  {object_id}:");
                *printed = true;
            }
        };
        if let Some(sc) = &fit.solved_covariance {
            header(&mut printed_header);
            eprintln!("    Solved covariance width: {}", sc.width);
        }
        if let Some(dt) = fit.dt_delta {
            header(&mut printed_header);
            eprintln!("    Non-grav time delay  ΔDT = {dt:.4} d");
        }
        if let Some(a) = fit.amrat_delta {
            header(&mut printed_header);
            eprintln!("    SRP AMRAT correction     = {a:.4e} m^2/kg");
        }
        for (i, dv) in fit.thrust_delta_m_per_s.iter().enumerate() {
            header(&mut printed_header);
            eprintln!(
                "    Thrust dv[{i}] = [{:.3}, {:.3}, {:.3}] m/s",
                dv[0], dv[1], dv[2]
            );
        }
        if let Some(ph) = &fit.photometry {
            header(&mut printed_header);
            // Honest 1σ on H from the fit's parameter covariance (H is slot 0).
            let h_sigma = ph.covariance.map(|c| c[0][0].sqrt());
            match h_sigma {
                Some(sigma) => eprintln!(
                    "    Photometry: H = {:.3} ± {:.3}  G1 = {:.3}  (model {:?}, chi2_r {:.2})",
                    ph.h, sigma, ph.slope1, ph.model_used, ph.reduced_chi2
                ),
                None => eprintln!(
                    "    Photometry: H = {:.3}  G1 = {:.3}  (model {:?}, chi2_r {:.2})",
                    ph.h, ph.slope1, ph.model_used, ph.reduced_chi2
                ),
            }
        }
    }
}

/// Write the three batch artifacts.
///
/// Shared with the daemon so both paths produce byte-identical output
/// sets. Rows are emitted in the batch's `object_id` order, which the
/// engine already sorted, so the files are a deterministic function of
/// the input rather than of its row order.
pub fn write_batch_outputs(
    out_dir: &Path,
    results: &DetermineResults,
    format: OutputFormat,
) -> Result<()> {
    std::fs::create_dir_all(out_dir).context("failed to create output directory")?;

    let fitted_batch = batch_orbit_rows(results);
    output::write_orbits(out_dir, "fitted_orbits", &fitted_batch, format)?;

    // Fit summary — one row per INPUT object, delivered or not. Always
    // written in BOTH formats: parquet for downstream tooling, CSV so the
    // run can be read at a terminal without a parquet reader.
    let summary_rows = FitSummaryRow::from_results(results);
    let summary_parquet = out_dir.join("fit_summary.parquet");
    empyrean::write_fit_summary_parquet(&summary_parquet, &summary_rows)
        .with_context(|| format!("failed to write {}", summary_parquet.display()))?;
    let summary_csv = out_dir.join("fit_summary.csv");
    empyrean::write_fit_summary_csv(&summary_csv, &summary_rows)
        .with_context(|| format!("failed to write {}", summary_csv.display()))?;
    eprintln!(
        "  {} + {} ({} rows)",
        summary_parquet.display(),
        summary_csv
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        summary_rows.len()
    );

    let residuals = batch_residual_rows(results);
    let resid_path = out_dir.join(format!("residuals.{}", format_extension(format)));
    output::write_residuals(&resid_path, &residuals, format)?;
    eprintln!("  {} ({} rows)", resid_path.display(), residuals.len());

    Ok(())
}

/// The fitted-orbit rows: one per DELIVERED object, identified by its
/// ADES designation. `fit.orbit` is already a re-feedable `Orbit`
/// carrying state + covariance + non-grav, so it is written as-is.
///
/// A failed object contributes no row — the orbit file is orbits. Which
/// objects are missing, and why, is `fit_summary`'s job.
fn batch_orbit_rows(results: &DetermineResults) -> OrbitBatch {
    let mut orbits = Vec::new();
    let mut orbit_ids = Vec::new();
    let mut object_ids = Vec::new();
    for (object_id, fit) in results.delivered() {
        orbits.push(fit.orbit.clone());
        orbit_ids.push(object_id.to_string());
        object_ids.push(Some(object_id.to_string()));
    }
    OrbitBatch {
        orbits,
        orbit_ids,
        object_ids,
    }
}

/// Every delivered fit's residual rows, concatenated in table order.
/// Each row carries the `object_id` of the fit it came from, so the flat
/// file stays attributable.
fn batch_residual_rows(results: &DetermineResults) -> Vec<empyrean::ObservationResidual> {
    results
        .delivered()
        .flat_map(|(_, fit)| fit.residuals.iter().cloned())
        .collect()
}

/// The process exit code for a finished batch: success only when every
/// object delivered.
pub fn exit_code_for(results: &DetermineResults) -> i32 {
    if results.delivered_count() == results.len() && !results.is_empty() {
        EXIT_ALL_DELIVERED
    } else if results.delivered_count() == 0 {
        EXIT_NONE_DELIVERED
    } else {
        EXIT_PARTIAL
    }
}

pub fn run(data: &DataOptions, args: DetermineArgs) -> Result<()> {
    // Reject a mismatched prior/axis combination before any expensive work.
    validate_prior_flags(&args)?;

    // A refine-path axis (DT / AMRAT) needs a prior on the seed orbit, so it
    // runs as a determine → prime → refine two-pass instead of a single solve.
    let (_marsden, dt_axis, amrat_axis) = solve_for_axes(args.solve_for);
    let needs_refine_path = dt_axis || amrat_axis;

    // The daemon protocol only carries force_model + max_iterations, so a
    // fitting request (non-grav / DT / AMRAT / thrust / photometry) must
    // run in-process — the daemon can't express it yet.
    //
    // `--no-refresh` also runs in-process: a running daemon's context was
    // built under its own policy and cannot honour a strict-offline
    // request retroactively, so serving it would quietly ignore the flag.
    let uses_fitting =
        args.solve_for != SolveForArg::Auto || args.thrust_segments > 0 || args.photometry;
    if !uses_fitting && data.daemon_eligible() {
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
    let ctx = load_context(data)?;

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
    let results = if needs_refine_path {
        // DT / AMRAT: coarse seed solve, prime the requested priors, then
        // the wide Bayesian refine. See `run_refine_path`.
        run_refine_path(&ctx, &observations, &args)?
    } else {
        let config = empyrean::ODConfig {
            force_model: args.force_model.to_empyrean(),
            max_iterations: args.max_iterations,
            solve_for: build_solve_for(args.solve_for, args.thrust_segments),
            photometry: args.photometry.then(empyrean::PhotometryConfig::default),
            ..empyrean::ODConfig::default()
        };
        ctx.determine(&observations, None, &config)
            .context("orbit determination failed")?
    };
    eprintln!(
        "OD complete ({:.1}s): {} of {} object(s) delivered",
        t1.elapsed().as_secs_f64(),
        results.delivered_count(),
        results.len()
    );

    report_batch(&results);
    report_wide_axes(&results);

    write_batch_outputs(&args.out_dir, &results, args.format)?;
    eprintln!("\n  Output: {}/", args.out_dir.display());

    let code = exit_code_for(&results);
    if code != EXIT_ALL_DELIVERED {
        std::process::exit(code);
    }
    Ok(())
}

fn format_extension(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Parquet => "parquet",
        OutputFormat::Json => "json",
        OutputFormat::Csv => "csv",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use empyrean::{SolveFor, SolveForParams};

    /// A minimal `DetermineArgs` with every prior flag unset, `solve_for`
    /// caller-chosen. Only the fields the prior-flag logic reads matter.
    fn args_with(solve_for: SolveForArg) -> DetermineArgs {
        DetermineArgs {
            ades_file: PathBuf::from("obs.psv"),
            force_model: ForceModel::Standard,
            max_iterations: 20,
            solve_for,
            amrat: None,
            cr: None,
            amrat_variance: None,
            dt: None,
            dt_variance: None,
            thrust_segments: 0,
            photometry: false,
            out_dir: PathBuf::from("."),
            format: OutputFormat::Parquet,
        }
    }

    #[test]
    fn axes_match_coarse_variants() {
        assert_eq!(
            solve_for_axes(SolveForArg::StateOnly),
            (false, false, false)
        );
        assert_eq!(solve_for_axes(SolveForArg::NonGrav), (true, false, false));
        assert_eq!(solve_for_axes(SolveForArg::Dt), (true, true, false));
        assert_eq!(solve_for_axes(SolveForArg::Amrat), (false, false, true));
        assert_eq!(
            solve_for_axes(SolveForArg::NonGravAmrat),
            (true, false, true)
        );
    }

    #[test]
    fn wide_solve_for_opens_the_expected_columns() {
        // DT: Marsden + DT columns open.
        assert!(matches!(
            build_solve_for(SolveForArg::Dt, 0),
            SolveForParams::Explicit(SolveFor {
                marsden: true,
                dt: true,
                amrat: false,
                thrust_segments: 0,
            })
        ));
        // AMRAT alone: only the AMRAT column.
        assert!(matches!(
            build_solve_for(SolveForArg::Amrat, 0),
            SolveForParams::Explicit(SolveFor {
                marsden: false,
                dt: false,
                amrat: true,
                thrust_segments: 0,
            })
        ));
        // Non-grav + AMRAT: Marsden + AMRAT.
        assert!(matches!(
            build_solve_for(SolveForArg::NonGravAmrat, 0),
            SolveForParams::Explicit(SolveFor {
                marsden: true,
                dt: false,
                amrat: true,
                thrust_segments: 0,
            })
        ));
    }

    #[test]
    fn amrat_solve_requires_amrat_and_variance() {
        // No priors at all.
        let err = validate_prior_flags(&args_with(SolveForArg::Amrat)).unwrap_err();
        assert!(err.to_string().contains("--amrat"), "{err}");

        // AMRAT value but no variance.
        let mut a = args_with(SolveForArg::Amrat);
        a.amrat = Some(3.0e-3);
        let err = validate_prior_flags(&a).unwrap_err();
        assert!(err.to_string().contains("--amrat-variance"), "{err}");

        // Both present: accepted (Cr defaults later).
        a.amrat_variance = Some(1.0e-8);
        assert!(validate_prior_flags(&a).is_ok());
    }

    #[test]
    fn dt_solve_requires_variance() {
        // DT axis with no variance.
        let err = validate_prior_flags(&args_with(SolveForArg::Dt)).unwrap_err();
        assert!(err.to_string().contains("--dt-variance"), "{err}");

        // Variance present: accepted even without an explicit --dt value.
        let mut a = args_with(SolveForArg::Dt);
        a.dt_variance = Some(1.0e-2);
        assert!(validate_prior_flags(&a).is_ok());
    }

    #[test]
    fn non_grav_amrat_solve_requires_amrat_priors() {
        let mut a = args_with(SolveForArg::NonGravAmrat);
        assert!(validate_prior_flags(&a).is_err());
        a.amrat = Some(3.0e-3);
        a.amrat_variance = Some(1.0e-8);
        assert!(validate_prior_flags(&a).is_ok());
    }

    #[test]
    fn prior_without_matching_axis_is_rejected() {
        // AMRAT prior set but state-only solve.
        let mut a = args_with(SolveForArg::StateOnly);
        a.amrat = Some(3.0e-3);
        let err = validate_prior_flags(&a).unwrap_err();
        assert!(err.to_string().contains("--amrat"), "{err}");

        // Cr set but no AMRAT axis.
        let mut a = args_with(SolveForArg::NonGrav);
        a.cr = Some(1.2);
        let err = validate_prior_flags(&a).unwrap_err();
        assert!(err.to_string().contains("--cr"), "{err}");

        // DT prior set but AMRAT (not DT) solve.
        let mut a = args_with(SolveForArg::Amrat);
        a.amrat = Some(3.0e-3);
        a.amrat_variance = Some(1.0e-8);
        a.dt_variance = Some(1.0e-2);
        let err = validate_prior_flags(&a).unwrap_err();
        assert!(err.to_string().contains("--dt-variance"), "{err}");
    }

    /// The exit code is the batch's verdict: 0 only when every object
    /// the input named produced an orbit. A partially successful run
    /// must not look like a success, because its orbit file is shorter
    /// than its input.
    #[test]
    fn exit_code_reports_partial_and_total_failure_distinctly() {
        use empyrean::{DetermineEntry, DetermineFailure, DetermineFailureKind, DetermineResults};

        fn failed(id: &str) -> DetermineEntry {
            DetermineEntry {
                object_id: id.to_string(),
                outcome: Err(DetermineFailure {
                    object_id: id.to_string(),
                    message: "no viable IOD seed".to_string(),
                    kind: DetermineFailureKind::IOD,
                }),
            }
        }

        // Every object failed.
        let none = DetermineResults::from_entries(vec![failed("A"), failed("B")], Vec::new());
        assert_eq!(exit_code_for(&none), EXIT_NONE_DELIVERED);

        // An empty batch delivered nothing either.
        let empty = DetermineResults::from_entries(Vec::new(), Vec::new());
        assert_eq!(exit_code_for(&empty), EXIT_NONE_DELIVERED);

        // The three codes are distinct so a script can tell them apart.
        assert_ne!(EXIT_ALL_DELIVERED, EXIT_PARTIAL);
        assert_ne!(EXIT_ALL_DELIVERED, EXIT_NONE_DELIVERED);
        assert_ne!(EXIT_PARTIAL, EXIT_NONE_DELIVERED);
        // ...and none of them collides with anyhow's exit code for a
        // hard error (1), which means "the run did not happen".
        for code in [EXIT_PARTIAL, EXIT_NONE_DELIVERED] {
            assert_ne!(code, 1);
        }
    }

    /// A failed object contributes no orbit row and no residual rows,
    /// but the batch's other objects are unaffected.
    #[test]
    fn failed_objects_contribute_no_orbit_or_residual_rows() {
        use empyrean::{DetermineEntry, DetermineFailure, DetermineFailureKind, DetermineResults};

        let entries = vec![DetermineEntry {
            object_id: "K25A00B".to_string(),
            outcome: Err(DetermineFailure {
                object_id: "K25A00B".to_string(),
                message: "no viable IOD seed".to_string(),
                kind: DetermineFailureKind::IOD,
            }),
        }];
        let results = DetermineResults::from_entries(entries, Vec::new());

        assert_eq!(batch_orbit_rows(&results).len(), 0);
        assert_eq!(batch_residual_rows(&results).len(), 0);
        // The object still exists in the batch — it is only absent from
        // the orbit file.
        assert_eq!(results.len(), 1);
        assert_eq!(FitSummaryRow::from_results(&results).len(), 1);
    }

    #[test]
    fn no_priors_needed_for_coarse_solves() {
        for mode in [
            SolveForArg::Auto,
            SolveForArg::StateOnly,
            SolveForArg::NonGrav,
        ] {
            assert!(validate_prior_flags(&args_with(mode)).is_ok());
        }
    }
}
