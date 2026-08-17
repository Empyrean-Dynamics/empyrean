<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean">

# empyrean
Safe Rust wrapper over libempyrean — uncertainty-first orbit propagation, ephemeris, orbit determination, and event detection for asteroids and comets, powered by automatic differentiation

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean"><img src="https://img.shields.io/crates/v/empyrean.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
<a href="https://docs.rs/empyrean"><img src="https://img.shields.io/docsrs/empyrean?style=flat-square&label=docs.rs" alt="docs.rs"></a>
<br>
<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://doi.org/10.5281/zenodo.21318471"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21318471-blue?style=flat-square" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

The idiomatic Rust API over the `libempyrean` C ABI. Every C function
exposed in the cdylib has a typed, `Result<_, Error>`-returning wrapper
here. RAII handles the underlying allocations so callers never juggle
raw FFI pointers.

```toml
[dependencies]
empyrean = "0.10.0-rc.0"
```

## What it does

- **Propagation** — N-body (Sun, planets, Moon, Pluto) with EIH general relativity, Sun J2 and Earth J2–J4 zonal harmonics, 16 asteroid perturbers, and the Marsden non-gravitational model — selectable across Approximate / Basic / Standard force-model tiers (Standard is the default). GR15 and DOP853 integrators. Optional finite-burn thrust arcs — constant-RTN, velocity-tangent, or inertial-fixed steering, with per-arc Δv targeting corrections — layer on as a continuous-thrust force input.
- **Uncertainty** — First-order (Jet1) state transition matrices; second-order (Jet2) state transition tensors; unscented sigma-point and Monte Carlo sampling; an adaptive Auto mode that escalates the method automatically through close approaches and relaxes it elsewhere. Optional per-epoch tagged-covariance readback. A fit over the state and *P* parameters produces one (6+*P*)×(6+*P*) covariance, and the **off-diagonal** blocks travel with it — onto the fitted orbit, through propagation, and into impact probability, B-planes and ephemeris — so a chained calculation is conditioned on the covariance the fit actually computed rather than on its diagonal blocks.
- **Ephemeris** — RA/Dec, rates, photometry (H–G, H–G₁G₂, H–G₁₂), light time, phase angle, solar elongation, local horizon. Each row carries the 6×6 sky-plane covariance over (ρ, RA, Dec) and their rates, and the aberrated barycentric ICRF state at the photon-emission epoch with its own 6×6 covariance — both present when the input orbit carries a state covariance.
- **Orbit determination** — Gauss, Herget, and systematic-ranging (admissible region + Manifold of Variations) IOD → N-body differential correction over optical and radar (delay / Doppler) observations fitted jointly, with span-grouped Jacobian reuse and outlier rejection. One call fits **every** object in an ADES set and returns per-object results keyed by designation. Solves beyond the six-element state for the Marsden A1/A2/A3 non-gravitational block, the cometary outgassing time delay DT, the SRP area-to-mass ratio AMRAT, and thrust Δv-correction segments — each partial supplied analytically by the hyperdual integrator, and each axis carrying a *disposition* (solved / considered / fixed) rather than a flag — and returns a tagged solved covariance that names every fitted parameter, a re-feedable orbit carrying that covariance's off-diagonal blocks, and an event-aware trust verdict on the delivered covariance. Optional post-fit photometry recovers H and the phase-function slope. Validated against `find_orb` and JPL SBDB.
- **Events** — Close approach (start/end), periapsis, gravitational capture (start/end), shadow entry/exit, atmospheric entry/exit, impact, and possible impact.

## Quick start

```rust,no_run
use empyrean::{Context, Epoch, PropagationConfig};

let ctx = Context::from_data_dir(None)?;

// Query SBDB for Apophis and propagate through its 2029 Earth flyby.
let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let result = ctx.propagate(&orbits, &epochs, &PropagationConfig::default())?;

println!("{} states, {} events", result.states.len(), result.events.len());
# Ok::<(), empyrean::Error>(())
```

## Orbit determination

`determine` runs a full IOD (Gauss / Herget / systematic ranging) → N-body
differential correction over **every object** the observations group into;
`refine` is a Bayesian update against a prior orbit; `evaluate` returns
residuals without fitting. The fitted `result.orbit` is a re-feedable
[`Orbit`] carrying state, covariance, and any fitted non-gravitational
parameters — pass it straight back into `propagate`, `generate_ephemeris`,
or `compute_impact_probabilities`.

```rust,no_run
# use empyrean::{Context, ODConfig};
# let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("observations.psv")?;   // optical + radar
// `determine` fits EVERY object in the arc and returns the batch;
// `into_single` unwraps the one-object case and refuses to pick if
// the file turned out to hold more.
let result = ctx.determine(&obs, None, &ODConfig::default())?.into_single()?;

println!(
    "converged={}, RMS = {:.2}\" RA / {:.2}\" Dec",
    result.converged,
    result.summary.rms_ra_arcsec,
    result.summary.rms_dec_arcsec,
);
# Ok::<(), empyrean::Error>(())
```

Every residual row carries per-observation diagnostics: χ² with its
survival probability, along/cross-track residuals with the full
symmetric 2×2 covariance, and influence measures including the
D-optimality information loss on removal (+∞ marks an observation whose
removal makes the normal matrix singular). Radar rows carry a typed
delay / Doppler block — observed − predicted in seconds / hertz, with
χ², survival probability, and the combined observed+predicted variance.
No observation is deselected anonymously: every row carries a typed
`RejectionReason` next to the criterion value and the threshold it was
tested against, including `NonFiniteChi2` (the residual χ² was not
finite, so the row could not enter any fit statistic) and
`MissingJacobian` (no Jacobian survived at that epoch, so the row never
contributed to the normal equations).
`result.covariance_trust` is an event-aware verdict on the delivered
covariance: `Trusted`, `EncounterIntervenes` (naming the intervening
close approach or high-nonlinearity crossing, and whether a
second-order state-only correction can recover it), or
`WeaklyDeterminedHighN` for wider-than-state fits. `None` means no
trust gate ran — absence of a verdict is not trust.

`ODConfig::default()` is the production hot path: the **VFCC2017**
weighting preset — Vereš, Farnocchia, Chesley & Chamberlin (2017)
per-station σ floors, with 1/√N same-night de-weighting chained on top —
over EFCC2020 catalog debiasing. `WeightingPreset::Neodys` and
`WeightingPreset::None` are the alternatives, and
`WeightingConfig::additional_layers` overrides the preset for named
stations. Optical and radar astrometry are fitted jointly, which is what
carries the hard objects; the co-orbital IOD lane (`coorbital_enabled`,
on by default) is what recovers Earth co-orbitals of the 2010 TK7 /
2020 XL5 class; and long comet arcs deliver as full-arc fits — set
`allow_arc_truncation: false` to make an arc that genuinely cannot be
fitted as one piece fail loudly instead of delivering the reconcilable
sub-arc with the remainder tagged `RejectionReason::OutsideArc`.

## Fitting a whole ADES set

[`DetermineResults`] is the table `determine` returns: one
[`DetermineEntry`] per ADES object identifier (permID / provID /
trkSub), in `object_id` order, each holding either the fit or a typed
[`DetermineFailure`]. One object failing never aborts the batch and
never removes the others — a failure is an entry carrying its reason,
not a gap — so `len()` always equals the number of objects the
observations grouped into. Iterate the table, look one object up with
`get(object_id)`, take only the fits with `delivered()`, or only the
reasons with `failures()`. `all_failed()` reports the batch that ran and
delivered nothing, and seed orbits that matched no observation group
come back on `unmatched_orbit_ids()` rather than being dropped.

```rust,no_run
use empyrean::{Context, ODConfig};

let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("nightly_batch.psv")?;
let fits = ctx.determine(&obs, None, &ODConfig::default())?;

for entry in fits.iter() {
    match &entry.outcome {
        Ok(fit) => println!(
            "{}: reduced χ² = {:.2}, extrapolable = {}",
            entry.object_id,
            fit.summary.reduced_chi2,
            fit.acceptability.extrapolation_acceptable,
        ),
        // Typed: branch on `kind`, never on the message text.
        Err(f) => println!("{}: FAILED ({:?}) — {}", entry.object_id, f.kind, f.message),
    }
}
println!("{}/{} delivered", fits.delivered_count(), fits.len());
# Ok::<(), empyrean::Error>(())
```

Each fit carries an `AcceptabilityReport`. `fit_acceptable` is the AND
of the fit-quality gates — convergence, positive-definite covariance,
reduced χ², RMS, AT/CT residual isotropy. `extrapolation_acceptable` is
that AND the selection / coverage gates: the fraction of observations
the fit retained, the span the *selected* observations still cover,
whether the most recent observations were rejected, and fractional σₐ.
Every gate reports its measured value beside the threshold it was tested
against, so a fit that did not clear one says which and by how much. Use
the first verdict to gate publication and the second to gate forward
propagation, ephemeris generation, or impact-risk assessment; tighten
either through `AcceptabilityThresholds`.

## Wide-parameter fitting

Beyond the six-element state, `determine` and `refine` can solve for the
Marsden A1/A2/A3 non-gravitational block, the cometary outgassing time delay
DT, the SRP area-to-mass ratio AMRAT, and thrust Δv-correction segments — every
partial derivative supplied analytically by the hyperdual integrator rather
than finite differences. Choose the axes with `SolveForParams`: `StateOnly`,
`StateAndNonGrav`, `Auto` (starts state-only and escalates the non-grav block
automatically on a poor fit), or `Explicit(SolveFor { .. })` for the wider
axes the coarse variants can't name.

Each axis on `SolveFor` carries a `ParamDisposition`, not a flag:
`Solved` estimates it, `Considered` does not estimate it but still lets its
prior uncertainty reach the posterior through its measurement partials
(Schmidt–Kalman consider analysis; Tapley, Byron D., Schutz, Bob E., and
Born, George H., *Statistical Orbit Determination*, Elsevier Academic Press,
2004, ch. 6), and `Fixed` marginalizes it out. Both of the last two produce a
well-formed covariance, so `false` could not say which was meant — there is
deliberately no `From<bool>` and no `Default` on the enum itself.
`DetermineResult::dispositions` reports the partition the fit actually ran,
which is what tells you whether re-attaching a prior to an axis would
double-count it: a considered axis already has its uncertainty inside the
delivered 6×6, a fixed one does not. Same covariance, opposite conclusions.

**Consider analysis is not a conservatism knob.** Under an uncorrelated
prior the consider correction strictly widens the posterior, but the fits
that need it are the ones with cross terms between the considered axis and
the solved ones — and there the correction is sign-indefinite. A considered
axis can come back *tighter*. Report it as what it is (an unestimated error
source folded through its partials), never as a safety margin.

`SolveFor::thrust` is `[ParamDisposition; MAX_THRUST_SEGMENTS]`, positional
with the orbit's declared Δv-correction segments rather than a count — a
considered or fixed burn sits between two solved ones as readily as after
them, and a count cannot say which burn is which. `with_leading_thrust(n)`
opens the leading *n* burns and **refuses** an over-budget request rather
than saturating; `solved_thrust_segments()` / `considered_thrust_segments()`
count them back.

DT, AMRAT, and thrust are refine-path solves: the input orbit must carry a
prior — the variance that *opens* the parameter. Request an axis without its
prior and the fit errors loudly; it never hands back a zeroed or defaulted
column. Covariance the fit was handed and deliberately did **not** use comes
back on `DetermineResult::warnings` — delivered payload rather than a log
line, because a dropped prior cross term changes how the σ for that slot
should be read. It is empty on a fit that used everything it was given.

Per-segment thrust results are indexed by **declared** segment and are
`Option`-valued: `thrust_delta_m_per_s[i]` and
`thrust_correction_covariances[i]` are `None` where segment *i* was not
solved, because a zero there would read as a fitted Δv of exactly zero, and
echoing a considered burn's prior would republish it under a posterior's
name. Read `dispositions.thrust[i]` before the value.

Every wide fit reports a `SolvedCovariance` whose fitted-parameter identities
travel with the matrix. Read a parameter's variance by its slot (`marsden_slot`,
`dt_slot`, `amrat_slot`, `thrust_slots`) rather than by guessing column order —
`width` alone is ambiguous (a 9×9 is Marsden-only *or* one thrust segment).

```rust,no_run
use empyrean::{Context, ODConfig, ParamDisposition, SolveFor, SolveForParams};

let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("comet_67p.psv")?;

// First solve state + Marsden A1/A2/A3.
let fit = ctx.determine(&obs, None, &ODConfig {
    solve_for: SolveForParams::StateAndNonGrav,
    ..Default::default()
})?.into_single()?;

// Refine, additionally solving the outgassing time delay DT. Opening DT
// requires a prior on it — its variance (days²) — carried on the orbit.
// Ask for DT without the prior and refine errors, never a zeroed column.
let prior = fit.orbit
    .with_non_grav_dt(Some(30.0))
    .with_non_grav_dt_variance(Some(100.0));
let refined = ctx.refine(&prior, &obs, &ODConfig {
    solve_for: SolveForParams::Explicit(SolveFor {
        marsden: ParamDisposition::Solved,
        dt: ParamDisposition::Solved,
        ..Default::default()
    }),
    ..Default::default()
})?;

// The solved covariance names its columns — read σ(DT) by slot.
if let Some(cov) = &refined.solved_covariance {
    if let Some(k) = cov.dt_slot {
        println!(
            "ΔDT = {:?} d,  σ(DT) = {:.3} d",
            refined.dt_delta,
            cov.matrix[k][k].sqrt(),
        );
    }
}
# Ok::<(), empyrean::Error>(())
```

## The joint covariance

A wide fit produces one (6+*P*)×(6+*P*) matrix. Its diagonal blocks — the 6×6
state covariance, the Marsden 3×3, a DT variance, an AMRAT variance, a
per-segment thrust 3×3 — have always crossed the boundary. The
**off-diagonal** blocks are what `JointCovariance` and `WideCross` carry, and
they ride the fitted orbit, propagation (in both directions), impact
probability, B-planes and ephemeris.

Dropping them is not a conservative simplification. A block-diagonal
covariance asserts that the data which produced the state and the data which
produced A2 were independent, when they are the same observations through the
same fit. Worse, the *propagated* joint has non-zero state↔parameter columns
**even from a block-diagonal input**, because propagation itself generates the
correlation — so a second leg handed only the 6×6 reports a *tighter*
uncertainty than the first leg supports.

### The four homes

One covariance entry belongs to exactly one place, and supplying it in two is
refused rather than merged:

| block | home |
|---|---|
| state ↔ state | `CoordinateState::covariance` |
| Aᵢ ↔ Aⱼ | `Orbit::ng_covariance` |
| state ↔ Aᵢ | `CoordinateState::non_grav_cross` (6×3) |
| Δvᵢ ↔ Δvⱼ, same segment | that segment's own 3×3 on `ThrustParams` |
| everything else | `Orbit::wide_cross` (a `WideCross`) |

The state↔Marsden border sits on the **coordinate**, beside the 6×6 it
borders, so a coordinate transform moves both halves of one matrix together;
`transform_coordinates` rotates the border with the state rather than leaving
it in the old basis. Everything else — state↔DT, state↔AMRAT, state↔Δv, and
the mixed parameter pairs — sits on the orbit.

Entries in a `WideCross` are keyed by `ParamColumn` (`Marsden(i)`, `Dt`,
`Amrat`, `Thrust { segment, component }`), never by column index, because
which column a parameter occupies depends on what *else* the orbit declares —
adding an SRP AMRAT shifts the thrust columns by one. An index recorded
against one orbit is wrong against the next, and the failure is silent: every
number finite, every gate passed, one parameter's correlations attached to
another. `ParamColumn::as_tag` / `from_tag` render and parse the canonical
strings (`"A1"`, `"DT"`, `"AMRAT"`, `"thrust[0].x"`) that the file formats and
the other language channels carry.

A `Thrust` segment index is the **declared** segment, not the solved one — the
same index space as `SolveFor::thrust` and the orbit's own correction
covariances.

### Absence is not zero

`WideCross::is_empty` reports entry *count*, not values: an entry whose six
numbers are all zero is a supplied zero correlation and makes it return
`false`. Omitting the entry is the only way to say "absent". The distinction
is load-bearing in both directions — the engine's definiteness gate engages on
a supplied zero — so a producer never emits a zero block to stand in for a
missing one, and `JointCovariance::non_grav_cross` is `Option`-typed for the
same reason.

### Reading it, and handing it on

```rust,no_run
use empyrean::{
    Context, Epoch, ODConfig, ParamColumn, PropagationConfig, SolveForParams,
};

let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("comet_67p.psv")?;
let fit = ctx.determine(&obs, None, &ODConfig {
    solve_for: SolveForParams::StateAndNonGrav,
    ..Default::default()
})?.into_single()?;

// The fitted orbit carries the fit's own off-diagonal blocks, in the two
// homes above. No reconstruction: `fit.orbit` is already the input type.
let border = fit.orbit.state.non_grav_cross;          // Option<[[f64; 3]; 6]>
if let Some(wide) = &fit.orbit.wide_cross {
    if let Some(col) = wide.state_cross(ParamColumn::Amrat) {
        println!("cov(x, AMRAT) = {:e}", col[0]);
    }
    for (a, b, value) in wide.param_crosses() {
        println!("cov({}, {}) = {value:e}", a.as_tag(), b.as_tag());
    }
}

// Propagate it. The joint goes in with the orbit and comes back per epoch.
let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let leg1 = ctx.propagate(&[fit.orbit.clone()], &epochs, &PropagationConfig::default())?;
let end = &leg1.states[0];
println!("joint present at the end of leg 1: {}", !end.joint.is_empty());
# let _ = border;
# Ok::<(), empyrean::Error>(())
```

Chaining a second leg by hand is three field copies onto the next orbit —
`state.covariance`, `state.non_grav_cross`, `wide_cross` — plus the parameter
blocks the cross terms are *conditioned* on:

```rust,no_run
# use empyrean::{Context, CoordinateState, Epoch, PropagationConfig};
# let ctx = Context::from_data_dir(None)?;
# let seed = empyrean::query_sbdb(&["Apophis"], None)?.orbits.remove(0);
# let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
# let leg1 = ctx.propagate(&[seed.clone()], &epochs, &PropagationConfig::default())?;
let end = &leg1.states[0];

let mut next = seed.clone();   // carries the parameter blocks unchanged
next.state = CoordinateState::cartesian(
    end.epoch,
    [
        end.position[0], end.position[1], end.position[2],
        end.velocity[0], end.velocity[1], end.velocity[2],
    ],
    end.frame,
    end.origin,
);
next.state.covariance = end.covariance;
next.state.non_grav_cross = end.joint.non_grav_cross;
next.wide_cross = end.joint.wide_cross.clone();

let leg2 = ctx.propagate(&[next], &epochs, &PropagationConfig::default())?;
# let _ = leg2;
# Ok::<(), empyrean::Error>(())
```

The parameter blocks come from the orbit that *started* the chain, not from
the propagated row: propagation passes the non-grav 3×3, the DT variance and
the AMRAT variance through unchanged rather than restating them on every
output epoch. A border supplied without the parameter block it conditions is
refused by the engine, not quietly ignored — a cross term with no diagonal
block to sit against is half a matrix.

`PropagationResult::joint_at(orbit_index, epoch_index)` reads the same cross
terms alongside the tagged-covariance accessors, for callers working from
indices rather than iterating `states`. It is a separate call rather than a
field on the tagged covariance so that the C ABI's equivalent struct stays
free of owned storage; Rust callers get the engine's arrays copied into owned
values and released before it returns.

`PropagatedState` is **no longer `Copy`** as a consequence — it now owns heap
storage — though it is still `Clone`, and copying a struct that already
carried a 1728-byte state-transition tensor was never cheap. Call sites that
relied on an implicit copy need an explicit `.clone()`.

### Downstream consumers

`compute_impact_probabilities`, `compute_b_planes` and `generate_ephemeris`
all read the joint off the orbits they are given. An impact probability
computed against a block-diagonal covariance materially understates the tails,
for the same reason chaining does: it asserts an independence the fit never
found. Pass the fitted orbit through whole and the question does not arise.

## Post-fit photometry

Attach a `PhotometryConfig` to `ODConfig::photometry` and the pipeline recovers
absolute magnitude H and the phase-function slope from the observation
magnitudes after the orbit is solved. The photometric fit has no astrometric
partials, so it never touches the state. In `Auto` it climbs a model ladder —
H-only → HG12 → HG1G2 (Muinonen et al. 2010) — admitting the richest model the
arc's phase-angle coverage supports and reporting the one it actually fit on
`model_used` (never `Auto`). H carries an honest 1σ through the fitted
`covariance`; the per-model gate decisions come back in `gates`. Magnitudes
whose band has no adopted V-band conversion are excluded and counted —
`n_mags_dropped_unconvertible`, with the distinct offending band codes in
`dropped_bands` — and the observations' astrometry is unaffected.

```rust,no_run
use empyrean::{Context, ODConfig, PhotometryConfig};

let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("observations.psv")?;

// Fit the orbit, then fit H/G from the magnitudes (Auto ladder:
// H-only -> HG12 -> HG1G2).
let fit = ctx.determine(&obs, None, &ODConfig {
    photometry: Some(PhotometryConfig::default()),
    ..Default::default()
})?.into_single()?;

if let Some(phot) = &fit.photometry {
    let sigma_h = phot.covariance.map(|c| c[0][0].sqrt());
    println!(
        "H = {:.2} ± {:.2}  ({:?}, {} mags, α span {:.1}°)",
        phot.h,
        sigma_h.unwrap_or(f64::NAN),
        phot.model_used,
        phot.n_mags_used,
        phot.alpha_span_deg,
    );
}
# Ok::<(), empyrean::Error>(())
```

Hand that covariance back to an orbit with `Orbit::with_photometry_covariance`
and ephemeris generation reports the H uncertainty in `mag_sigma`:

```rust,no_run
# use empyrean::{Orbit, PhaseFunction};
# fn f(orbit: Orbit, fitted: [[f64; 3]; 3]) -> Orbit {
orbit
    .with_photometry(PhaseFunction::HG, 19.7, 0.15, 0.0)
    .with_photometry_covariance(Some(fitted))
# }
```

The two contributions are summed in quadrature,
σ_V = sqrt(σ²_photo + σ²_state), where σ²_photo = J Σ_p Jᵀ contracts the full
3×3 against J = [∂V/∂H, ∂V/∂slope₁, ∂V/∂slope₂]. Because
V = H + 5·log₁₀(rΔ) + φ(α) gives ∂V/∂H ≡ 1 exactly, an orbit carrying no state
covariance and a covariance of the H-only shape diag(σ_H², 0, 0) reports
σ_V = σ_H. Slope variances and H–slope covariances do not drop out — they
contract against ∂V/∂slope, which vanishes only at zero phase angle — so any
covariance carrying them reports σ_V > σ_H. SBDB's published
diag(σ_H², σ_G², 0) is the common case. They are combined as independent: a fitted σ_H
is conditional on the fitted state (the photometric fit holds the geometry
exact) and no joint state↔photometry covariance is computed anywhere in the
stack, so there is no cross term to add — the resulting σ_V is mildly
conservative, which is the safe direction.

## Writing results

Orbits, residuals, per-object fit summaries, ephemerides, and events all
write to parquet, JSON, and CSV; orbits read back from all three as well
(the propagator and the OD pipeline are the canonical producers of the
rest). The three formats carry the same columns and differ only in how a
non-computable number is spelled — CSV a literal `NaN`, JSON `null`,
since JSON has no NaN literal. CSV is not the lossy choice for the ordinary
column set: `write_orbits_csv` emits the same columns as
`write_orbits_parquet`, covariance included.

The **joint** is where the three formats genuinely differ. Parquet carries the
state↔Marsden border and the wide carrier in a tagged tail, so a fitted orbit
round-trips through a parquet file holding the covariance the fit computed
rather than its diagonal blocks. It is the only orbit format here that can,
and the other two refuse such a batch **by name** rather than writing it
short: the JSON orbit format is a flat row shape carrying the 6×6 and nothing
beyond it, and CSV cannot express the difference between an absent cross and a
supplied zero cross — it renders both as an empty cell, and that difference is
load-bearing. A carrier holding thrust Δv terms is refused wherever it is
offered, because no orbit-file format can serialize the thrust arcs those
terms hang on. Refusing at the writer is the point: a silently dropped carrier
produces a file that reads back as a block-diagonal joint, which is a
different and tighter claim than the one you held, with nothing in the round
trip to signal it happened.

Residual files carry the **whole** `ObservationResidual` surface — all
36 fields — not a projection of it: the `obs_id` / `object_id` join
keys, the observatory code, catalog and epoch, the effective residual
covariances, the complete rejection attribution (reason, criterion,
threshold, effective threshold, information loss), the influence
diagnostics, the along/cross-track decomposition, and the radar block.
Because `object_id` travels with the row, residuals from a whole batch
concatenate into one table and stay attributable.

The fit summary is the artifact that makes a partially successful batch
readable: one row per **input** object, whether or not it produced an
orbit, carrying its convergence, RMS, both acceptability verdicts with
each gate's value and threshold, the solved width, and — on a failed
object — the reason. The orbit file holds only the objects that
delivered; the summary holds all of them.

```rust,no_run
use empyrean::{
    Context, FitSummaryRow, ODConfig, OrbitBatch, write_fit_summary_csv,
    write_fit_summary_parquet, write_orbits_csv, write_residuals_parquet,
};

let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("nightly_batch.psv")?;
let fits = ctx.determine(&obs, None, &ODConfig::default())?;

// One row per input object — delivered and failed alike.
let summary = FitSummaryRow::from_results(&fits);
write_fit_summary_parquet("fit_summary.parquet", &summary)?;
write_fit_summary_csv("fit_summary.csv", &summary)?;

// The delivered fits: orbits keyed by the designation they were fitted
// under, and one flat residual table across the batch.
let mut orbits = Vec::new();
let mut orbit_ids = Vec::new();
let mut object_ids = Vec::new();
let mut residuals = Vec::new();
for (object_id, fit) in fits.delivered() {
    orbits.push(fit.orbit.clone());
    orbit_ids.push(object_id.to_string());
    object_ids.push(Some(object_id.to_string()));
    residuals.extend(fit.residuals.iter().cloned());
}

// CSV carries the covariance too — same columns as parquet.
write_orbits_csv("fitted_orbits.csv", &OrbitBatch::new(orbits, orbit_ids, object_ids)?)?;
write_residuals_parquet("residuals.parquet", &residuals)?;
# Ok::<(), empyrean::Error>(())
```

## Ephemeris

```rust,no_run
# use empyrean::{Context, EphemerisConfig, Epoch, Frame, Origin};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
// ICRF / solar-system barycenter is the construction basis — the one
// ephemeris generation requires, and the one that takes no transform.
let observers = ctx.get_observers(&["W84", "F51"], &epochs, Frame::ICRF, Origin::SSB)?;
let eph = ctx.generate_ephemeris(&orbits, &observers, &EphemerisConfig::default())?;

for entry in &eph.entries {
    println!("RA {:.4}°  Dec {:.4}°  V {:.2}", entry.ra_deg, entry.dec_deg, entry.mag);
}
# Ok::<(), empyrean::Error>(())
```

Beyond the printed astrometry, each `EphemerisEntry` carries the 6×6
sky-plane covariance over (ρ, RA, Dec) and their rates (AU / degree
units), and the aberrated — light-time corrected — barycentric ICRF
Cartesian state at the photon-emission epoch with its own 6×6
covariance; both covariances are `None` when the input orbit carried no
state covariance. Non-fatal generation warnings (an Earth-orientation
kernel coverage gap handled by the analytic IAU 2006 fallback, a row
whose observation-sensitivity chain was skipped) come back on
`EphemerisResult::warnings` — empty when the run had nothing to report.

A target that is itself one of the loaded perturbers (1 Ceres, 2 Pallas,
4 Vesta, …) needs `EphemerisOverlapPolicy::ExcludeAndIntegrate` on the
inner propagation config. The default, `SubstituteSpk`, returns that
body's own SPK states — exact for the body, but no trajectory is
produced, and ephemeris generation has no light-time chain to read.

```rust,no_run
use empyrean::{EphemerisConfig, EphemerisOverlapPolicy, PropagationConfig};

let cfg = EphemerisConfig {
    propagation: PropagationConfig {
        ephemeris_overlap_policy: EphemerisOverlapPolicy::ExcludeAndIntegrate,
        ..Default::default()
    },
    ..Default::default()
};
# let _ = cfg;
```

## Uncertainty

First-order (the default) propagates the covariance with the state-transition
matrix — accurate when the orbit is approximately linear over the uncertainty
region. Second-order adds the state-transition tensor for the curvature that
linear covariance misses near a close approach.

```rust,no_run
# use empyrean::{Context, Epoch, PropagationConfig, UncertaintyMethod};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
# let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let config = PropagationConfig {
    uncertainty_method: UncertaintyMethod::SecondOrder,
    ..Default::default()
};
let result = ctx.propagate(&orbits, &epochs, &config)?;
# Ok::<(), empyrean::Error>(())
```

### Reading back the mixture

Under `UncertaintyMethod::Mixture` — and inside `Auto`'s close-approach
windows — the engine splits the input Gaussian and retains the resulting
components at every close approach where the splitter actually fired.
`PropagationResult::mixtures` carries one `MixtureChain` per input orbit
(empty for an orbit that never split), and `mixture_at` is the per-epoch
lookup. They are the mixture itself, not its moment collapse: a consumer can
evaluate Σ_k w_k · N(x | μ_k, Σ_k) directly at the CA epoch.

```rust,no_run
# use empyrean::{Context, Epoch, PropagationConfig, UncertaintyMethod};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
# let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let config = PropagationConfig {
    uncertainty_method: UncertaintyMethod::gaussian_mixture(),
    ..Default::default()
};
let result = ctx.propagate(&orbits, &epochs, &config)?;

for chain in &result.mixtures {
    for (k, group) in chain.components.iter().enumerate() {
        let retained: f64 = group.iter().map(|c| c.weight).sum();
        println!(
            "{} @ MJD {:.3}: {} components, retained weight {retained:.4}",
            chain.orbit_id, chain.ca_epochs_mjd_tdb[k], group.len(),
        );
    }
}
# Ok::<(), empyrean::Error>(())
```

Four limits on what is retained, each a real property of the engine's
retention rather than a marshaling shortfall: depth-0 splits only; only CA
epochs where AGM actually fired; the component covariance is the linear
Φ Σ_k Φᵀ map with the second-order mean correction omitted by design; and the
retained weights may sum to **less than 1**, because a sub-Gaussian whose own
sub-propagation missed the close approach contributes no component and the
deficit is not recorded anywhere. Do not assume the weights normalize.

Note the module path: `empyrean::propagate::MixtureComponent` is the
basis-tagged read-back component, a different type from the crate-root
`empyrean::MixtureComponent`, which is the `split_gaussian` primitive at t₀.

## Continuous thrust

Model finite burns / low-thrust arcs by attaching a `ThrustParams` to an
orbit before propagation. Each `ThrustArc` carries its own thrust, mass,
specific impulse, steering law (constant-RTN, velocity-tangent, or
inertial-fixed), and central body; the burn perturbs the trajectory
through the same differentiated dynamics as gravity and the
non-gravitational forces.

```rust,no_run
use empyrean::{Context, Epoch, Origin, PropagationConfig, SteeringLaw, ThrustArc, ThrustParams};

let ctx = Context::from_data_dir(None)?;
let orbit = empyrean::query_sbdb(&["Apophis"], None)?.orbits.remove(0);

// One finite burn: 1 N over MJD 65000–65010 on a 500 kg spacecraft,
// mass depleting at Isp = 3000 s, steered at constant RTN angles
// relative to the Sun. `sharpness` sets the tanh on/off transition.
let arc = ThrustArc::new(
    65000.0,                                                   // start_mjd_tdb
    65010.0,                                                   // end_mjd_tdb
    1.0,                                                       // thrust_n (N)
    500.0,                                                     // mass_kg
    100.0,                                                     // sharpness (1/day)
    SteeringLaw::ConstantRTN { alpha_rad: 0.0, beta_rad: 0.0 },
    Origin::SUN,                                               // RTN frame reference
)
.with_isp(Some(3000.0));

// Attach to the orbit and propagate. Add per-arc Δv targeting
// corrections with `ThrustParams::new(arcs).with_dv_corrections(..)`.
let orbit = orbit.with_thrust(Some(ThrustParams::new(vec![arc])));
let epochs = vec![Epoch::from_mjd_tdb(65020.0)];
let result = ctx.propagate(&[orbit], &epochs, &PropagationConfig::default())?;
println!("{} states", result.states.len());
# Ok::<(), empyrean::Error>(())
```

## System handles

Assembling the force model (planets, Moon, asteroid perturbers,
harmonics, relativistic corrections) has a fixed per-call cost. A
[`BuiltSystem`] assembles it once for a frozen `{force model, frame,
encounter-timescale divisor}` key and reuses it across many
propagations — the build-once, propagate-many pattern for
short-arc campaigns. It is `Send + Sync`, so `&handle` can be shared
across threads. A call whose config disagrees with the frozen key, or
that pairs the handle with a different data instance, is rejected
loudly by axis — never silently rebuilt against the wrong dynamics.

```rust,no_run
# use empyrean::{Context, ForceModelTier, Frame, PropagationConfig, Epoch};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
// Build once; freeze the divisor at the engine default (0.0).
let handle = ctx.built_system(ForceModelTier::Standard, Frame::EclipticJ2000, 0.0)?;

let epochs = vec![Epoch::from_mjd_tdb(65020.0)];
let result = handle.propagate(&ctx, &orbits, &epochs, &PropagationConfig::default())?;
println!("{} states", result.states.len());

// describe() reports the reproducibility record: the force-model menu
// plus the identity (SHA-256) of every loaded kernel.
let desc = handle.describe()?;
println!("{} perturbers, {} kernels", desc.perturber_origins.len(), desc.kernels.len());
# Ok::<(), empyrean::Error>(())
```

## Impact probability and B-plane geometry

For each detected close approach you can ask for an impact-probability
assessment or a full B-plane breakdown, and run several uncertainty methods
side-by-side on the same encounter. Each returns one record per
(method × orbit × body), tagged with its method and closest-approach epoch.
Each record also carries the geodetic impact point on the body's reference
ellipsoid (latitude / longitude / altitude — NaN when no surface projection
is available for the encounter), the 95% binomial confidence half-width on
the Monte-Carlo fraction, the second-order corrected mean miss distance
with its 1σ uncertainty and skewness, the closest-approach distance
gradient and 6×6 Hessian with respect to the initial state, and the
adaptive Gaussian-mixture component count — fields a given method didn't
compute carry NaN / 0 sentinels.

`UncertaintyMethod::auto()` is accepted here alongside the fixed
methods, and a hand-tuned `Auto { .. }` is carried through with the
thresholds you set rather than the defaults. Every record is tagged with
the method that produced it: an `Auto` row reads back tagged `Auto`,
never relabelled as one of the fixed methods.

```rust,no_run
# use empyrean::{Context, Epoch, UncertaintyMethod, Origin};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let end = Epoch::from_mjd_tdb(65000.0);

let ips = ctx.compute_impact_probabilities(
    &orbits,
    end,
    &[
        UncertaintyMethod::FirstOrder,
        UncertaintyMethod::SecondOrder,
        UncertaintyMethod::auto(),
    ],
    &[Origin::EARTH, Origin::MOON],
)?;
for ip in &ips {
    println!("{:?}: miss {:.0} km", ip.body, ip.miss_distance_km);
}

let bps = ctx.compute_b_planes(&orbits, end, &[UncertaintyMethod::SecondOrder], &[Origin::EARTH])?;
for bp in &bps {
    println!("B·T {:.1} km, B·R {:.1} km", bp.b_dot_t_km, bp.b_dot_r_km);
}
# Ok::<(), empyrean::Error>(())
```

## Observation planning

Given an orbit that already carries a covariance, `Context::evaluate_plan`
ranks candidate follow-up observations by how much each would tighten it.
Optical candidates contribute sky-plane information; radar candidates
contribute the line-of-sight range and range-rate that angles-only
astrometry cannot supply, with a measurement σ set by the Cramér-Rao
bound over the waveform bandwidth and the effective SNR — supplied, or
derived from a link budget whose assumptions come back on the candidate.

The orbit must be referenced to the Solar System barycenter; the frame is
free. An origin shift is a pure translation, so the covariance and every
metric derived from it are unchanged by the conversion.

Candidates come back in ascending epoch order, and each one's marginal
gain is measured against the covariance that already contains every
earlier candidate — the gains are conditional on that sequence, not
standalone scores. Every submitted candidate is folded, including one
reported unobservable, so `posterior` prices the plan as submitted.
`observable` is a real engine verdict on an optical row and always `true`
on a radar row, where no feasibility test runs. The non-gravitational
(σ(A2)) planning variant, the visibility survey, batch evaluation, and
the encounter B-plane are not exposed here; an orbit carrying non-grav
parameters is evaluated state-only, with the non-grav acceleration still
acting in the dynamics.

`PlanningConfig::observatories` takes `ObservatoryConfig` values — the MPC
code, the assumed 1σ (RA·cosδ, Dec), a limiting apparent magnitude, a minimum
solar elongation, plus the two visibility limits: `min_elevation_deg`
(geometric elevation above the site's local horizon, ignoring refraction;
`0.0` is the geometric horizon and the engine's default — the
least-opinionated statement the geometry can make, not an observing
recommendation, since airmass there is about 38 and real programs cut between
20° and 30°) and `max_sun_altitude_deg` (an `Option<f64>`; `None` takes the
engine's default of −18°, astronomical twilight, with civil at −6° and
nautical at −12°, and above +90° disabling the gate — an `Option` because
`0.0` is a legal solar altitude, the Sun's centre on the geometric horizon, so
a defaulted zero would quietly plan a campaign in daylight). The struct's
fields carry no defaults, so a config cannot be half-specified without saying
so.

`evaluate_plan` **does not consult** any of it: the field refuses a non-empty
list, each optical candidate's σ comes from its own `PlannedObservation`, and
the observability filters on that entry point are engine-set rather than
caller-configurable. The type is documented here because it is part of the
shared planning configuration and becomes live the day a surface that reads it
is exposed — not because this release reads it.

```rust,no_run
# use empyrean::{
#     Context, Epoch, Frame, Origin, PlannedObservation, PlanningConfig, RadarMode,
#     RadarPlanSpec, RadarStation, Representation,
# };
# let ctx = Context::from_data_dir(None)?;
# let mut orbit = empyrean::query_sbdb(&["Apophis"], None)?.orbits[0].clone();
orbit.state = ctx.transform_coordinates_single(
    &orbit.state,
    Representation::Cartesian,
    Frame::EclipticJ2000,
    Origin::SSB,
)?;
let t0 = orbit.state.epoch.mjd_tdb()?;

let planned = vec![
    PlannedObservation::optical("F51", [0.2, 0.2], Epoch::from_mjd_tdb(t0 + 30.0)),
    PlannedObservation::radar(
        RadarPlanSpec::given(
            RadarStation::GoldstoneDSS14,
            RadarStation::GoldstoneDSS14,
            RadarMode::Both,
            1.0e5,
            0.1,
            50.0,
        ),
        Epoch::from_mjd_tdb(t0 + 45.0),
    ),
];

let plan = ctx.evaluate_plan(&orbit, Some("apophis"), &planned, &PlanningConfig::default())?;
println!(
    "position σ {:.1} km → {:.1} km",
    plan.prior.position_sigma_km, plan.posterior.position_sigma_km,
);
for c in &plan.candidates {
    println!("{}: {:.1}%", c.obs_code, 100.0 * c.marginal_position_improvement);
}
# Ok::<(), empyrean::Error>(())
```

## Data directory and offline operation

`Context::from_data_dir` loads the Standard-tier kernel set, acquiring
whatever the tier needs and the directory does not have.
`Context::from_data_dir_with` is its superset —
`from_data_dir_with(dir, DataDirOptions::default())` is exactly
`from_data_dir(dir)` — and `refresh: false` is why it exists. Strict
offline resolves the tier's kernels from the directory alone (no HTTP
HEAD, no download, no staleness check) and fails naming **every** file
the tier needs and the directory lacks, as a list on
`Error::missing_data_files()` rather than as prose a caller would have
to split. Nothing is degraded to make an incomplete directory work: no
lower-tier fallback, no download-just-this-one, no partially loaded
context.

`download_data` provisions without loading — it downloads and caches the
kernel set and stops there, so a provisioning step never pays for a
context assembly it would immediately discard. It is idempotent: files
already present are kept.

```rust,no_run
use empyrean::{Context, DataDirOptions, DataTier};

// Provision once, with the network.
let dir = empyrean::download_data(None)?;

// From here on, never reach for it. Any absent file fails the
// construction and is named in the error.
let ctx = Context::from_data_dir_with(
    Some(&dir),
    DataDirOptions { refresh: false, tier: DataTier::Standard },
)
.inspect_err(|e| {
    for missing in e.missing_data_files() {
        eprintln!("absent: {missing}");
    }
})?;
# let _ = ctx;
# Ok::<(), empyrean::Error>(())
```

`EMPYREAN_OFFLINE=1` is a floor, never an override: it downgrades a
requested `refresh: true` to `false` and says so on stderr, and it can
never turn a `false` into a `true`. Only the exact value `1` asserts it.
`offline_floor_is_active()` reports whether it is in force, for a caller
that does network work of its own before building a context.
`Context::from_data_dir` does not consult it — it predates the variable,
and quietly reinterpreting it would change the meaning of code written
before the variable existed — so reach for `from_data_dir_with` when the
variable should apply.

`DataTier` selects which kernel set has to be on disk — Approximate,
Basic, or Standard (the default) — and lines up with the
`ForceModelTier` a propagation then runs under.

## Runtime requirement

This crate (via empyrean-sys) loads `libempyrean.{dylib,so}` at
run time, which is distributed separately as a binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. The path is resolved from the
`EMPYREAN_LIB` environment variable if set, else a `libempyrean.*`
sitting next to the loaded module, else a build-time location — an
`EMPYREAN_LIB_DIR` override, a sibling `../target/release` build, or
a checksum-pinned prebuilt downloaded from the GitHub release (in
that order); no system library path setup is required.

Whichever path resolves, the library must be the one built for **this
crate's version**. empyrean-sys calls `empyrean_abi_version()` the moment
it opens `libempyrean` and compares it against the `EMPYREAN_ABI_VERSION`
it compiled with; any mismatch fails there and then, naming both numbers
and the resolved path, rather than reading your arguments through a
layout that moved. The number encodes the release, so pairing across
releases is exactly what it rejects. It encodes the **base** version
only, though: a version and its pre-releases share one number, so the
check cannot separate `0.10.0-rc.1` from `0.10.0`, and a boundary change
inside a pre-release cycle needs both sides rebuilt together rather than
caught here. The bundled and downloaded artifacts satisfy the pairing by
construction — only a hand-set `EMPYREAN_LIB` can pair the wrong two.

Prebuilt engine binaries are currently published for four targets —
macOS arm64 (`macos-aarch64`), macOS x86_64 (`macos-x86_64`), Linux
x86_64 (`linux-x86_64`), and Linux aarch64 (`linux-aarch64`); on other
targets the build stops with an error unless `EMPYREAN_LIB_DIR` points
at an engine build.

The full distribution surface (Python wheel, CLI binary, C SDK, this
Rust crate) lives at the
[main repository](https://github.com/Empyrean-Dynamics/empyrean) —
see its README for installation paths and the cross-channel quickstart.

## Accuracy

Validated against JPL Horizons, ASSIST, and `find_orb` on
43 objects across 13 dynamical populations (NEOs, MBAs, Trojans, TNOs,
comets, and more). Sub-meter propagation accuracy on bounded timescales;
see the [validation notes](https://github.com/Empyrean-Dynamics/empyrean#validation)
in the main repository for the comparison setup.

## No guarantee of accuracy

empyrean performs numerical computations used in planetary-science and
mission-planning contexts. Outputs should not be used as the sole basis
for any decision — including but not limited to impact monitoring,
mission planning, collision avoidance, or navigation — without
independent verification. See the LICENSE file for the full terms.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime it loads at runtime is governed by a separate proprietary
binary license; see the main repository for the dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.

## Links

- Website: https://www.empyrean-dynamics.com
- Repository: https://github.com/Empyrean-Dynamics/empyrean
- Issues: https://github.com/Empyrean-Dynamics/empyrean/issues
