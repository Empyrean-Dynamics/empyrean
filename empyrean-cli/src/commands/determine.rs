use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use empyrean::OrbitBatch;

use crate::ForceModel;
use crate::io::output::{self, OutputFormat};

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

    /// Output directory.
    #[arg(long, default_value = ".")]
    pub out_dir: PathBuf,

    /// Output file format for fitted orbit + residuals.
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

/// Run the determine → prime → refine two-pass used when a refine-path axis
/// (DT or AMRAT) is requested. Pass 1 is the coarse solve *without* that
/// axis; the priors from the flags are then attached to its re-feedable
/// `result.orbit`, and pass 2 is the wide Bayesian refine.
///
/// Assumes [`validate_prior_flags`] has already accepted `args`, so the
/// required prior flags for the requested axes are present.
fn run_refine_path(
    ctx: &empyrean::Context,
    observations: &empyrean::Observations,
    args: &DetermineArgs,
) -> Result<empyrean::DetermineResult> {
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
    let seed = ctx
        .determine(observations, None, &base_config)
        .context("refine-path pass 1 (seed solve) failed")?;

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

    // Pass 2: the wide Bayesian refine with the full Explicit solve-for.
    let wide_config = empyrean::ODConfig {
        force_model: args.force_model.to_empyrean(),
        max_iterations: args.max_iterations,
        solve_for: build_solve_for(args.solve_for, args.thrust_segments),
        photometry: args.photometry.then(empyrean::PhotometryConfig::default),
        ..empyrean::ODConfig::default()
    };
    eprintln!("  Pass 2 (wide refine): opening the requested column(s)");
    ctx.refine(&primed, observations, &wide_config)
        .context("refine-path pass 2 (wide refine) failed")
}

pub fn run(data_dir: Option<PathBuf>, args: DetermineArgs) -> Result<()> {
    // Reject a mismatched prior/axis combination before any expensive work.
    validate_prior_flags(&args)?;

    // A refine-path axis (DT / AMRAT) needs a prior on the seed orbit, so it
    // runs as a determine → prime → refine two-pass instead of a single solve.
    let (_marsden, dt_axis, amrat_axis) = solve_for_axes(args.solve_for);
    let needs_refine_path = dt_axis || amrat_axis;

    // The daemon protocol only carries force_model + max_iterations, so a
    // fitting request (non-grav / DT / AMRAT / thrust / photometry) must
    // run in-process — the daemon can't express it yet.
    let uses_fitting =
        args.solve_for != SolveForArg::Auto || args.thrust_segments > 0 || args.photometry;
    if !uses_fitting {
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
    let result = if needs_refine_path {
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

    // Wide-parameter fitting readback (v0.9.0). Each line appears only
    // when that axis was actually solved — a missing line reads as "not
    // recovered", never a zero.
    if let Some(sc) = &result.solved_covariance {
        eprintln!("  Solved covariance width: {}", sc.width);
    }
    if let Some(dt) = result.dt_delta {
        eprintln!("  Non-grav time delay  ΔDT = {dt:.4} d");
    }
    if let Some(a) = result.amrat_delta {
        eprintln!("  SRP AMRAT correction     = {a:.4e} m^2/kg");
    }
    for (i, dv) in result.thrust_delta_m_per_s.iter().enumerate() {
        eprintln!(
            "  Thrust dv[{i}] = [{:.3}, {:.3}, {:.3}] m/s",
            dv[0], dv[1], dv[2]
        );
    }
    if let Some(ph) = &result.photometry {
        // Honest 1σ on H from the fit's parameter covariance (H is slot 0).
        let h_sigma = ph.covariance.map(|c| c[0][0].sqrt());
        match h_sigma {
            Some(s) => eprintln!(
                "  Photometry: H = {:.3} ± {:.3}  G1 = {:.3}  (model {:?}, chi2_r {:.2})",
                ph.h, s, ph.slope1, ph.model_used, ph.reduced_chi2
            ),
            None => eprintln!(
                "  Photometry: H = {:.3}  G1 = {:.3}  (model {:?}, chi2_r {:.2})",
                ph.h, ph.slope1, ph.model_used, ph.reduced_chi2
            ),
        }
    }

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
