<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean">

# empyrean
Uncertainty-first orbit propagation, ephemeris, orbit determination, and event detection for asteroids and comets, powered by automatic differentiation

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://pypi.org/project/empyrean/"><img src="https://img.shields.io/pypi/v/empyrean.svg?style=flat-square&label=PyPI" alt="PyPI"></a>
<a href="https://pypi.org/project/empyrean/"><img src="https://img.shields.io/pypi/pyversions/empyrean.svg?style=flat-square&label=python" alt="Python versions"></a>
<br>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://doi.org/10.5281/zenodo.21318471"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21318471-blue?style=flat-square" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

```bash
pip install --pre empyrean==0.10.0rc0
```

Current release: **0.10.0rc0** (release candidate) — `--pre` is required
until 0.10.0 is final.

A plain install pulls empyrean
together with the B612 Foundation's
pre-packaged SPICE kernels (~740 MB — see the table below). After
installation, the first call to `empyrean.initialize()` downloads a
small remainder (the
`moon_pa` Moon-orientation kernel and the `bias.dat` star-catalog
debiasing table — about 50 MB) that isn't available on PyPI.

Wheels are published for CPython >= 3.10 as a single abi3 stable-ABI
wheel per architecture — one wheel covers CPython 3.10 and every newer
version — across four platforms: macOS arm64, macOS x86_64,
manylinux_2_28 x86_64, and manylinux_2_28 aarch64. There is no source
distribution, so the install will not resolve on other platforms — use
the
[other distribution channels](https://github.com/Empyrean-Dynamics/empyrean#install)
in the meantime.

Each wheel bundles the `libempyrean` engine built for its own release,
and the binding checks that pairing when the library opens: it reads the
engine's ABI version and compares it against the one the wheel was built
against, failing immediately — naming both numbers — if they differ. The
version is per release, so an engine from a different release is rejected
rather than used through a layout that may have moved. Nothing to
configure; the bundled engine always matches. It matters only if you
point the loader at an engine of your own.

## What it does

- **Propagation** — N-body (Sun, planets, Moon, Pluto) with EIH general relativity, Sun J2 and Earth J2–J4 zonal harmonics, 16 asteroid perturbers, and the Marsden non-gravitational model — selectable across Approximate / Basic / Standard force-model tiers (Standard is the default). GR15 and DOP853 integrators. Optional finite-burn thrust arcs — constant-RTN, velocity-tangent, or inertial-fixed steering, with per-arc Δv targeting corrections — layer on as a continuous-thrust force input.
- **Uncertainty** — First-order (Jet1) state transition matrices; second-order (Jet2) state transition tensors; unscented sigma-point and Monte Carlo sampling; an adaptive Auto mode that escalates the method automatically through close approaches and relaxes it elsewhere. Optional per-epoch tagged-covariance readback.
- **Ephemeris** — RA/Dec, rates, photometry (H–G, H–G₁G₂, H–G₁₂), light time, phase angle, solar elongation, local horizon, and the aberrated (light-time corrected) barycentric state per row — with sky-plane and aberrated-state covariances when the input orbit carries one.
- **Orbit determination** — batch-first: one `determine()` call fits every object in an ADES set and returns per-object results. Gauss, Herget, and systematic-ranging (admissible region + Manifold of Variations) IOD, plus a co-orbital lane that recovers Earth co-orbitals of the 2010 TK7 / 2020 XL5 class → N-body differential correction over optical and radar (delay / Doppler) observations solved jointly rather than in sequence, with STM caching and outlier rejection. Long comet arcs deliver as full-arc fits. Solves the state — escalating to the Marsden A1/A2/A3 non-gravitational coefficients on a poor fit — plus, on the refine path, the cometary outgassing time delay DT, SRP area-to-mass, and continuous-thrust Δv corrections, all differentiated analytically, returned in a tagged solved covariance. Every deselected observation carries a typed reason. Optional post-OD H–G photometry fit recovers absolute magnitude H with an honest σ. Validated against `find_orb` and JPL SBDB.
- **Events** — Close approach (start/end), periapsis, gravitational capture (start/end), shadow entry/exit, atmospheric entry/exit, impact, and possible impact.

## Quick start

```python
import empyrean
from empyrean import Epochs

empyrean.download_data()   # SPICE kernels, first run only
empyrean.initialize()

# Query SBDB for Apophis and propagate through its 2029 Earth flyby
orbits = empyrean.query_sbdb(["Apophis"])
epochs = Epochs.from_mjd([65000.0], scale="tdb")
result = empyrean.propagate(orbits, epochs)

# Event timeline
for i in range(len(result.events.summary)):
    ev = result.events.summary
    print(f"{ev.event_type.to_pylist()[i]:25s} "
          f"{ev.body.to_pylist()[i]:8s} "
          f"MJD {ev.epoch.to_numpy()[i]:.2f}")
```

Every time you hand to empyrean is an `Epochs`, and every `Epochs`
states its scale — apart from the carve-outs whose scale is fixed by
definition (columns named `mjd_tdb`, arguments named `epoch_mjd_tdb`,
and the coordinate tables' `epoch` column, which is MJD TDB). A bare
list or array is refused: `61000.5` read as UTC
and `61000.5` read as TDB are about 69 seconds apart — easily enough to
move an encounter geometry — so which one you mean is stated rather than
defaulted. Build them with `Epochs.from_mjd(values, scale="tdb")` /
`scale="utc"`, `Epochs.from_jd`, or `Epochs.from_iso([...])` for ISO-8601
UTC timestamps.

## Orbit determination

`determine` is batch-first. It groups the observations by ADES object
identifier (`permID`, else `provID`, else `trkSub`) and fits **every**
object, so a multi-object file is one call. Radar is optional; when
supplied it is grouped by the same identifier and fitted jointly with the
optical astrometry.

```python
obs, radar = empyrean.read_ades("observations.psv")   # (optical, radar)
fits = empyrean.determine(obs, radar=radar)            # every object in the file

print(len(fits), "object(s),", len(fits.delivered), "delivered")
print(fits.orbits)      # one row per DELIVERED object, carrying object_id
print(fits.summary)     # one row per INPUT object, delivered or failed
print(fits.residuals)   # every delivered fit's rows, tagged with object_id

for object_id, failure in fits.failures.items():
    print(f"{object_id}: {failure.kind}: {failure.message}")

result = fits["2024 YR4"]                              # DetermineResult
print(
    f"converged={result.converged}, "
    f"RMS={result.summary.rms_ra_arcsec:.2f}\" RA / "
    f"{result.summary.rms_dec_arcsec:.2f}\" Dec"
)
```

A failed object never aborts the batch and never disappears: it gets a
`.summary` row with `status="failed"`, NaN measurements (never `0.0`,
which would read as a value at the floor), the reason in `error`, and a
typed `DetermineFailure` in `.failures` — branch on its `kind`
(`"iod"` / `"od"` / `"radar_only"` / `"observer_construction"` /
`"earth_orientation_coverage"` / …), not on the message text. Seed orbits
passed as `initial_orbits` that matched no observation group come back in
`.unmatched_orbit_ids` rather than being dropped.

Fitting one object is the one-entry case of the same call, not a
different call: `fits.single()` unwraps it and refuses — naming the
objects — rather than choosing among several.

```python
result = empyrean.determine(obs).single()              # one object in, one fit out
```

`fits.orbits` is a `CartesianOrbits` table carrying `object_id`,
covariance, and the fitted non-grav / SRP slots — feed it straight back
into `propagate`, `generate_ephemeris`, or `compute_impact_probabilities`.

Underneath, the fit is the engine's: optical and radar are solved jointly
rather than in sequence, so a hard object's delay and Doppler tighten the
same covariance the astrometry does; the co-orbital IOD lane recovers
Earth co-orbitals of the 2010 TK7 / 2020 XL5 class; and long comet arcs
deliver as full-arc fits. Two `ODConfig` switches turn those lanes off
when you want the historical behaviour: `coorbital_enabled=False`
(leaving it on does not route ordinary objects through the lane), and
`allow_arc_truncation=False`, which makes an arc that genuinely cannot be
fitted as one piece **fail** rather than deliver its reconcilable part
with the remainder tagged `outside_arc`. Per-observation rejection is
orthogonal and still runs.

### Fit summary

`fits.summary` is a `FitSummary` quivr table with one row per **input**
object, so a partially successful batch is readable rather than silently
shorter than its input. The column names match the `fit_summary.parquet`
/ `fit_summary.csv` files the CLI writes, so a table read back off disk
and this one describe a fit identically.

Both acceptability verdicts are columns on it, and every extrapolation
gate sits beside the value it measured and the threshold it was compared
against — so a `False` says which axis failed and by how much:

```python
s = fits.summary

print(s.object_id.to_pylist(), s.status.to_pylist())   # "delivered" / "failed"
print(s.fit_acceptable.to_pylist())
print(s.extrapolation_acceptable.to_pylist())

def col(name):     # value columns are nullable — NaN where not computable
    return s.column(name).to_numpy(zero_copy_only=False)

print(col("selection_fraction"),    col("selection_fraction_threshold"))
print(col("selected_arc_fraction"), col("selected_arc_fraction_threshold"))
print(col("trailing_gap_days"),     col("trailing_gap_threshold_days"))
print(col("fractional_sigma_a"),    col("fractional_sigma_a_threshold"))
```

The same verdict in full structured form is `result.acceptability`, an
`AcceptabilityReport` whose every `*_ok` flag is paired with its
`*_value` and `*_threshold` — plus `radar_fit_ok`, a tri-state that is
`None` when no radar contributed, which is never the same as `False`.
Tune the bounds with `AcceptabilityThresholds` on `ODConfig`.

### Per-observation residuals

`result.observations` (and, batch-wide, `fits.residuals`) is an
`ObservationResults` table carrying the whole 35-column residual surface,
not a projection of it: the `obs_id` / `object_id` join keys, observatory
code, catalog and epoch; the RA/Dec residuals with their effective
combined covariance (`residual_cov_ra` / `residual_cov_dec` /
`residual_cov_corr`); the complete rejection block; the influence
diagnostics; the along/cross-track decomposition with its full 2×2
covariance; and the radar block. Radar rows carry the delay / Doppler
residual (observed − predicted in seconds / hertz) with its χ², dof,
survival probability, and combined variance instead of RA/Dec.

Nothing is deselected silently — `rejection_reason` names the layer that
dropped each row (`chi_squared`, `sigma_clip`, `cooks_distance`,
`adaptive`, `cmc2003`, `unsupported_observatory`, `outside_arc`,
`non_finite_chi2`, `missing_jacobian`, …) alongside the criterion value,
the static threshold, and the effective threshold it was tested against.
`influence_information_loss` is the D-optimality information loss on
removal (+inf marks an indispensable observation).

```python
print(result.observations.rejected_only().rejection_reason.to_pylist())
print(result.observations.worst_chi2(5).obs_id.to_pylist())
print(fits.residuals.select_station("F51").rms_combined_arcsec)
```

`result.covariance_trust` is an
event-aware verdict on the delivered covariance: `trusted`,
`encounter_intervenes` (naming the intervening close-approach or
high-nonlinearity event and whether a second-order state-only
correction can recover it), or `weakly_determined_high_n`. It is
`None` when no trust gate ran — absence of a verdict is not trust.

### Weighting

`ODConfig` ships production defaults: the `VFCC2017` weighting preset
(Vereš, Farnocchia, Chesley & Chamberlin 2017 station floors) plus
per-night 1/√N de-weighting, and EFCC2020 catalog debiasing (Eggl,
Farnocchia, Chamberlin & Chesley 2020). `WeightingConfig` replaces or
extends the layer chain — first-match-wins, so an entry in
`additional_layers` overrides the preset for its stations and the preset
is the fallback.

```python
from empyrean import (
    ODConfig, WeightingConfig, WeightingLayer, WeightingLayerKind, WeightingPreset,
)

config = ODConfig(
    weighting=WeightingConfig(
        preset=WeightingPreset.VFCC2017,
        additional_layers=[
            WeightingLayer(
                kind=WeightingLayerKind.OBSERVATORY_RULE,
                obs_code="F51",
                sigma=(0.1, 0.1),
            ),
            # additional_layers REPLACES the default list — re-include the
            # nightly layer to keep production behavior.
            WeightingLayer(kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING),
        ],
    )
)
```

### Wide-parameter fitting

A fit solves the 6-element state by default, escalating to the Marsden
A1/A2/A3 non-gravitational coefficients on a poor fit. `SolveFor` on
`ODConfig.solve_for_flags` requests an explicit wider solve: beyond
state + Marsden, `determine` and `refine` can also solve for the
cometary outgassing time delay `dt`, the solar-radiation-pressure
area-to-mass ratio `amrat`, and per-segment thrust Δv corrections
(`thrust`) — each differentiated analytically by the same hyperdual
integrator that drives the dynamics.

Each axis takes a **disposition**, not a flag, because "not solved" is
two different answers:

| disposition | what the fit does |
|---|---|
| `"solved"` | estimated from the data; comes back with a posterior variance |
| `"considered"` | not estimated, but its prior uncertainty still reaches the posterior through its measurement partials |
| `"fixed"` | marginalized out; contributes nothing and changes no number |

A considered axis is not a safety margin. Under an uncorrelated prior the
consider correction strictly widens the posterior, but the fits that need
it are the ones with cross terms between the considered axis and the
solved ones — and there the correction is sign-indefinite, so the
posterior can come back *tighter*. Report it as an unestimated error
source folded through its measurement partials, never as conservatism
(Schmidt–Kalman consider analysis; Tapley, Byron D., Schutz, Bob E., and
Born, George H., *Statistical Orbit Determination*, Elsevier Academic
Press, 2004, ch. 6).

`False` cannot say which of the last two was meant, so a bool is refused
by name rather than coerced. `result.dispositions` reports the partition
the fit actually ran — the partition resolved against the orbit, not the
one requested, so an `AUTO` escalation is readable after the fact. It is
also the only place a considered axis appears, since a solved
covariance's slot tags record what occupied a column and a considered
axis occupies none. That is what tells you whether re-attaching a prior
to an axis would double-count it: a considered axis already has its
uncertainty inside the delivered 6×6, a fixed one does not. Same
covariance, opposite conclusions.

`result.warnings` is the other half of that honesty: covariance the fit
was handed and deliberately did **not** use, delivered as payload rather
than written to a log, because a dropped prior cross term changes how the
σ for that slot should be read. It is empty on a fit that used everything
it was given, which is the common case.

`dt`, `amrat`, and thrust are refine-path solves: the seed orbit must
carry the prior that opens each axis, so run them through `refine`. The
DT prior is `NonGravParams.dt_variance` (days²) on the orbit's non-grav
block; Marsden needs a non-grav covariance; AMRAT needs an SRP AMRAT
prior. Requesting an axis whose prior is absent is rejected loudly —
the fit never returns a zeroed or defaulted column.

```python
from empyrean import ODConfig, ParamDisposition, SolveFor

# Solve state + Marsden A1/A2/A3 + the outgassing time delay DT. The
# seed orbit carries a non-grav covariance (opens Marsden) and a DT
# prior variance (opens DT), e.g. its non-grav block was built with
#   NonGravParams.from_kwargs(..., dt=[<days>], dt_variance=[<days**2>])
config = ODConfig(solve_for_flags=SolveFor(marsden="solved", dt="solved"))
result = empyrean.refine(orbit, obs, config=config)

print(result.dt_delta)      # fitted ΔDT (days); None if DT was not solved
print(result.amrat_delta)   # fitted ΔAMRAT (m²/kg); None if not solved
print(result.dispositions)  # what the fit did with each axis

# The enum is equivalent to the string form, and is what `dispositions`
# reports back:
ParamDisposition.parse("considered") is ParamDisposition.CONSIDERED

print(result.warnings)      # covariance the fit declined to use; often []

# Per-segment thrust dispositions are positional with the orbit's
# declared correction covariances — a considered or fixed burn sits
# between solved ones as readily as after them, so a count could not
# say which burn is which. At most three entries (the engine's
# MAX_THRUST_SEGMENTS, on empyrean.od.result); a longer list raises
# rather than being truncated to a shorter fit that drops a burn
# silently. A bool raises too, by name — it cannot say which of
# "considered" and "fixed" was meant.
SolveFor(thrust=["solved", "fixed", "solved"])
```

### Tagged solved covariance

A wide fit returns a `SolvedCovariance` on `result.solved_covariance`
whose fitted-parameter identities travel with the matrix. Read a
parameter's variance by its slot — never by guessing column order:

```python
sc = result.solved_covariance          # None for a state-only fit
if sc is not None and sc.dt_slot is not None:
    dt_var = sc.matrix[sc.dt_slot, sc.dt_slot]   # DT variance (days²)
    print(f"σ(DT) = {dt_var ** 0.5:.4f} days")
# sc.marsden_slot / sc.amrat_slot / sc.thrust_slots locate the rest;
# canonical layout is [state 6 | Marsden 3 | DT 1 | AMRAT 1 | thrust 3×k].
```

### Carrying the joint onward

The fitted orbit carries the off-diagonal blocks of that same matrix, so
it can be fed straight back into propagation without falling back to the
diagonal. They ride in two places: the 6×3 state↔Marsden border on
`orbit.non_grav.non_grav_cross`, and everything else — state↔DT,
state↔AMRAT, state↔Δv and the mixed parameter pairs — on
`orbit.wide_cross`. Entries are keyed by parameter name (`"AMRAT"`,
`"thrust[0].x"`), never by column index, because which column a
parameter occupies depends on what else the orbit declares:

```python
cross = fit.orbit.wide_cross.state_cross(0)   # {tag: 6-vector}, by tag
sigma_amrat_x = cross["AMRAT"][0]             # cov(x, AMRAT)
```

Propagated states carry the same two columns, holding the joint at each
output epoch. This is what makes a chained propagation match the
single-leg answer: the propagated state↔parameter columns are non-zero
even when the input was block-diagonal, because propagation itself
generates the correlation, so a second leg handed only the 6×6 reports a
*tighter* uncertainty than the first leg supports. A row with no cross
terms is null rather than zero — an absent correlation and a measured
zero correlation are different claims, and only one of them is yours to
make.

When you chain legs by hand, carry the parameter blocks the cross terms
are conditioned on (the non-grav 3×3, the DT and AMRAT prior variances)
from the orbit that started the chain: propagation passes those through
unchanged rather than restating them on every output row. A border
supplied without its parameter block is refused, not quietly ignored.

The joint is read by every downstream entry point that takes orbits, not
just `propagate`: `compute_impact_probabilities`, `compute_b_planes`,
`generate_ephemeris`, and the `determine` / `evaluate` / `refine` seed
path all condition on it when the orbit carries it. An impact probability
computed against a block-diagonal covariance materially understates the
tails, for the same reason chaining does — it asserts an independence the
fit never found. Feeding a fitted orbit through whole is what avoids the
question; nothing has to be reassembled by hand.

`TaggedCovariances` carries the same two columns alongside the 21
synthesized `cov_*` scalars those columns are the state block of, so the
per-epoch readback and an orbit table describe one joint the same way.
It is populated on every uncertainty method that produces a joint,
including the sampled ones, which recover the state↔parameter columns
from the propagated cloud. `WideCross` is a nullable sub-table column
rather than an optional attribute, so it is **never** `None` on a parent
table — quivr returns a table of parent length regardless. Absence is
per-row nulls: test `row_is_empty(i)` or the per-row accessors, never
`orbits.wide_cross is None`.

```python
result = empyrean.propagate(orbits, epochs, tagged_covariance=True)

tc = result.tagged_covariance_series(0)[-1]   # last epoch of orbit 0
tc.non_grav_cross              # (6, 3) ndarray, or None
tc.state_cross["AMRAT"]        # 6-vector, keyed by parameter tag
tc.param_cross[("AMRAT", "DT")]

# The same thing off the flat table, per row:
wc = result.tagged_covariance.wide_cross
wc.row_is_empty(0)
wc.state_cross(0)              # {tag: 6-vector}
```

### Post-OD photometry

Attach a `PhotometryConfig` to recover the absolute magnitude *H* and a
phase-function slope from the observation magnitudes after the orbit is
solved — the fit has no astrometric partials, so it never touches the
state. In `AUTO` it climbs a model ladder — H-only → HG12 → HG1G2
(Muinonen et al. 2010) — admitting the richest model the arc's
phase-angle coverage supports, and reports the model it actually fitted
on `model_used`. *H* comes back with an honest 1σ from the fit
covariance. Magnitudes whose band has no adopted V-band conversion are
never silently used: the report counts them
(`n_mags_dropped_unconvertible`) and lists the distinct offending band
codes (`dropped_bands`) — the observations' astrometry is unaffected.

```python
from empyrean import ODConfig, PhotometryConfig

config = ODConfig(photometry=PhotometryConfig())   # AUTO ladder
result = empyrean.determine(obs, config=config).single()

phot = result.photometry               # None if photometry was not requested
if phot is not None and phot.covariance is not None:
    sigma_h = phot.covariance[0, 0] ** 0.5
    print(f"H = {phot.h:.2f} ± {sigma_h:.2f} mag  (model {phot.model_used.value})")
```

## Ephemeris

```python
from empyrean import Frame, Origin

observers = empyrean.get_observer_states(["W84", "F51"], epochs)
eph = empyrean.generate_ephemeris(orbits, observers)

# Observer states default to the ICRF / SSB construction basis, which is
# what ephemeris generation and orbit determination require and which is
# returned untransformed. Pass frame= / origin= when you want the sites
# somewhere else — e.g. heliocentric ecliptic, for geometry plots:
sites = empyrean.get_observer_states(
    ["W84", "F51"], epochs, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN
)

print(eph.ephemeris.coordinates.lon.to_numpy())   # RA (degrees)
print(eph.ephemeris.coordinates.lat.to_numpy())   # Dec (degrees)
print(eph.ephemeris.mag.to_numpy(zero_copy_only=False))  # apparent V (nullable)

# Orbits carrying a covariance also get, per row, the 6×6 sky-plane
# covariance over (rho, RA, Dec + rates) in AU / degree units, and the
# aberrated (light-time corrected) barycentric ICRF state at the
# photon-emission epoch with its own 6×6 covariance:
print(eph.ephemeris.coordinates.covariance.to_matrix().shape)      # (N, 6, 6)
print(eph.ephemeris.aberrated_state.covariance.to_matrix().shape)  # (N, 6, 6)
```

`eph.warnings` lists non-fatal generation warnings — e.g. an
Earth-orientation kernel coverage gap handled by the analytic IAU 2006
fallback, or rows whose sensitivity chain was skipped — naming the
affected orbit / observatory / epoch. Empty when the run had nothing
to report.

### Self-perturbing targets

All sixteen SB441-N16 bodies (1 Ceres, 2 Pallas, 4 Vesta, 7 Iris, …) are
simultaneously members of the Standard force model and legitimate objects
to propagate. `ephemeris_overlap_policy` says what the engine does when a
target coincides with its own perturber: `SUBSTITUTE_SPK` (the default)
returns the body's authoritative SPK states and integrates nothing — so
there is no dense trajectory, no STM, and no sensitivity chain, and
ephemeris generation for such a body fails outright.
`EXCLUDE_AND_INTEGRATE` drops the overlapped perturber, integrates the
caller's own initial conditions, and reports the overlap. It is what
generating an ephemeris for an SB441-N16 body at Standard tier requires.

```python
from empyrean import EphemerisConfig, EphemerisOverlapPolicy, PropagationConfig

config = EphemerisConfig(
    propagation=PropagationConfig(
        ephemeris_overlap_policy=EphemerisOverlapPolicy.EXCLUDE_AND_INTEGRATE,
    )
)
ceres = empyrean.query_sbdb(["Ceres"])
eph = empyrean.generate_ephemeris(ceres, observers, config=config)
```

Naming the body in `excluded_perturbers` is the other escape. Use one or
the other: overlap detection is what decides whether the policy is
consulted at all, and it is suppressed whenever an explicit exclusion
list is present.

## Uncertainty

```python
from empyrean import UncertaintyMethod

# Second-order: populates STM (6x6) and STT (6x6x6)
result = empyrean.propagate(
    orbits, epochs,
    uncertainty_method=UncertaintyMethod.SECOND_ORDER,
)
print(result.sensitivity.stms_array().shape)   # (N, 6, 6)
print(result.sensitivity.stts_array().shape)   # (N, 6, 6, 6)
```

## Continuous thrust

Model finite burns / low-thrust arcs by passing one `ThrustParams` per
orbit through `propagate`'s `thrust_arcs` keyword (`None` for the
ballistic orbits). Each `ThrustArc` carries its own thrust, mass,
specific impulse, steering law (constant-RTN, velocity-tangent, or
inertial-fixed), and central body — the burn perturbs the trajectory
through the same differentiated dynamics as gravity and the
non-gravitational forces.

```python
import empyrean
from empyrean import Origin
from empyrean.orbits.thrust import ConstantRTN, ThrustArc, ThrustParams

# One finite burn: 1 N over MJD 65000-65010 on a 500 kg spacecraft,
# mass depleting at Isp = 3000 s, steered at constant RTN angles
# relative to the Sun. `sharpness` sets the tanh on/off transition.
arc = ThrustArc(
    start_mjd_tdb=65000.0,
    end_mjd_tdb=65010.0,
    thrust_n=1.0,
    mass_kg=500.0,
    steering=ConstantRTN(alpha_rad=0.0, beta_rad=0.0),
    sharpness=100.0,
    central_body=Origin.SUN,
    isp_s=3000.0,
)

# One entry per orbit, positionally aligned with `orbits`. Add per-arc Δv
# targeting corrections with ThrustParams(arcs=[arc], dv_corrections=[...]).
result = empyrean.propagate(orbits, epochs, thrust_arcs=[ThrustParams(arcs=[arc])])
```

## System handles

Assembling the force model has a fixed per-call cost. `build_system`
assembles it once for a frozen `{force model, frame, encounter-timescale
divisor}` key and returns a `BuiltSystem` you reuse across many
propagations — the build-once, propagate-many pattern for short-arc
campaigns. Its `propagate` / `generate_ephemeris` release the GIL, so
the handle can be shared across threads. A call that disagrees with the
frozen key is rejected loudly, never silently rebuilt; rebuild the
handle after any `initialize()` / data reload.

```python
import empyrean
from empyrean import ForceModelTier, Frame

# Build once for the Standard model in the ecliptic frame. force_model and
# frame accept the enums or their string / int forms.
system = empyrean.build_system(ForceModelTier.STANDARD, Frame.ECLIPTICJ2000)

result = system.propagate(orbits, epochs)

# describe() is the reproducibility record: the force-model menu plus the
# identity of every loaded kernel (SHA-256 for file-backed kernels; the
# model name for built-in fields).
desc = system.describe()
print(len(desc.perturber_origins), "perturbers,", len(desc.kernels), "kernels")
```

## Impact probability and B-plane geometry

For each detected close approach, you can ask the propagator for an
impact-probability assessment or a full B-plane breakdown — and run
several uncertainty methods side-by-side on the same encounter:

```python
import pyarrow.compute as pc

from empyrean import UncertaintyMethod

end_epoch = empyrean.Epochs.from_mjd([63000.0], scale="tdb")

ips = empyrean.compute_impact_probabilities(
    orbits,
    end_epoch=end_epoch,
    methods=[UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER],
)
ips.epochs.scale                    # "tdb"
second = ips.where(pc.field("method") == "second_order")
second.ip_second_order.to_numpy(zero_copy_only=False)   # nullable
ips.ip_linear.to_numpy()            # always populated

bps = empyrean.compute_b_planes(orbits, end_epoch, [UncertaintyMethod.SECOND_ORDER])
print(bps.b_dot_t_km.to_numpy())    # B·T (km)
print(bps.b_dot_r_km.to_numpy())    # B·R (km)
print(bps.semi_major_3sig_km.to_numpy(zero_copy_only=False))  # 3σ semi-major
```

Returns typed `ImpactProbabilities` and `BPlanes` quivr tables — one
row per (method × orbit × body) encounter, with the closest-approach
time as an embedded `Epochs` sub-table so `.to_utc()` / `.to_tdb()`
just works.

Each entry carries its own parameters — methods in one call never share
them. `Auto` is the adaptive method: it escalates the covariance
treatment over each close-approach window and relaxes it elsewhere,
choosing among first-order, second-order, and the adaptive Gaussian
mixture. Pass the `Auto` dataclass instead of the enum to tune its κ band
edges and mixture knobs, and those are the values the engine runs with:

```python
from empyrean import Auto, MonteCarlo

ips = empyrean.compute_impact_probabilities(
    orbits,
    end_epoch=empyrean.Epochs.from_mjd([63000.0], scale="tdb"),
    methods=[
        Auto(threshold_first=0.05, threshold_mixture=5.0, gmm_max_depth=4),
        MonteCarlo(n_samples=100_000, seed=7),
    ],
)
# ip_agm is populated when the mixture refinement fired; a null means it
# was a no-op, so use ip_agm when finite and ip_linear otherwise. Never
# back-fill it — the null is what distinguishes "ran" from "no-op".
print(ips.ip_agm.to_numpy(zero_copy_only=False))
print(ips.agm_components.to_numpy(zero_copy_only=False))
```

`UncertaintyMethod.AUTO` (or `"auto"`) selects the same method with the
engine-default thresholds; a default-constructed `Auto()` is identical
to it.

Each `ImpactProbabilities` row also carries the geodetic impact point
(latitude / longitude / altitude on the body's reference ellipsoid;
null when no surface projection is available), the 95% binomial
confidence half-width on `ip_mc`, the second-order corrected mean miss
distance, 1σ miss-distance uncertainty, and skewness, the
closest-approach distance gradient and 6×6 Hessian with respect to the
initial state, and the adaptive Gaussian-mixture component count.

## Observation planning

Given an orbit that already carries a covariance, `evaluate_plan` ranks
candidate follow-up observations by how much each one would tighten it —
before you spend the telescope time:

```python
import empyrean
from empyrean import (
    CartesianCoordinates,
    PlannedObservation,
    transform_coordinates,
)

# The planner consumes a Cartesian, barycentric covariance and converts
# neither, so both are required. One call does both; the origin half is a
# pure translation, so the covariance and its metrics come across unchanged.
coords = transform_coordinates(fit.orbit.coordinates, CartesianCoordinates, origin="SSB")
orbit = fit.orbit.set_column("coordinates", coords)
t0 = float(coords.epoch.to_numpy()[0])

plan = empyrean.evaluate_plan(
    orbit,
    [
        PlannedObservation.optical(t0 + 30.0, "F51", (0.2, 0.2)),
        PlannedObservation.optical(t0 + 31.0, "568", (0.3, 0.3)),
        PlannedObservation.radar(
            t0 + 45.0,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_snr=50.0,
        ),
    ],
)

plan.metrics.to_dataframe()                 # two rows: "prior", "posterior"
plan.metrics.prior().position_sigma_km[0].as_py()      # before any candidate
plan.metrics.posterior().position_sigma_km[0].as_py()  # after all of them
plan.candidates.best_by_information_gain(3).obs_code.to_pylist()
```

`plan.metrics` is a two-row `PlanMetrics` table with a `stage` column
(`"prior"` / `"posterior"`) plus the five covariance summary metrics —
RSS position and velocity σ, the 1σ position-ellipsoid semi-axes, and
`log_det`. Keeping it a table means plans concatenate and join like any
other output; `prior()` / `posterior()` are the one-row views.

`plan.candidates` is a `PlanCandidates` quivr table — one row per
candidate, carrying `marginal_volume_reduction` (the per-dimension
generalized-variance ratio `(det Σ_post / det Σ_prior)^(1/6)` over the
6×6 state covariance — a D-optimality score normalized to one dimension,
so it reads as a linear scale factor; the 1σ ellipsoid volume ratio is
that value cubed), the fractional position-σ improvement, and the running
covariance metrics after that candidate and every one folded before it.
`plan.ephemeris` is the predicted sky position at each optical
candidate's epoch, with the epoch as an embedded `Epochs` sub-table; an
optical row's `index` is its row there. A radar row has no epoch of its
own — its `index` is its rank among the radar candidates, ordered by
epoch.

Rows come back in ascending epoch order, and each candidate's marginal
gain is measured against the covariance that already contains every
earlier one. The gains are therefore **conditional**: two identical
observations do not score identically, and
`best_by_information_gain` ranks contributions within one campaign rather
than standalone candidate value. To compare candidates head to head,
evaluate a one-candidate plan for each.

`observable` means different things per kind. On an optical row it is a
real engine verdict — today a solar-elongation test and nothing else,
since the target's absolute magnitude does not reach the planner, so the
limiting-magnitude filter cannot fire. On a radar row it is always
`True`: no radar feasibility test runs here, not even the antenna
elevation limit, so `True` means "not assessed". Either way the filters
are engine-set and not caller-configurable, and `observable` **does not
gate the fold** — an unobservable candidate still contributes to
`posterior` and to every later `cumulative_*` row. What does gate it is
whether the engine could compute the candidate's observation partials; a
candidate for which it could not reports a `marginal_volume_reduction` of
exactly 1 and leaves the covariance untouched. `observable_only()` filters
rows, not information; to price the observable subset, drop those
candidates from the plan and evaluate again.

Not exposed in this release, recorded so the omissions are not mistaken
for oversights: the non-gravitational planning variant that solves over
state ⊕ (A1, A2, A3) and reports the σ(A2) tightening a radar campaign
buys, the visibility survey, batch evaluation across many orbits, and the
encounter B-plane. An orbit carrying non-gravitational parameters is
accepted and evaluated state-only — the acceleration still acts in the
dynamics, but the solve-for set stays 6×6 and no σ(A2) is reported.

Optical and radar candidates fill different blocks and the cross-block is
null, never zero: sky-plane geometry (`along_track_sigma_arcsec`,
`position_angle_deg`, …) is populated only on optical rows, and
`radar_mode` / `radar_snr` / `radar_range_km` only on radar rows. Radar
adds the line-of-sight range and range-rate that angles-only astrometry
cannot supply; its measurement σ is the Cramér-Rao bound set by the
waveform bandwidth and the effective SNR.

Leave `radar_snr` at `None` and the SNR is derived from a link budget over
the target's physical properties instead. That path never substitutes a
value it was not given — a missing property it needs is a loud refusal,
and anything it *derived or adjusted* comes back on `radar_provenance`.
Supplying an SNR selects the other request shape, so the link-budget
inputs are refused alongside it rather than dropped:

```python
plan = empyrean.evaluate_plan(
    orbit,
    [
        PlannedObservation.radar(
            t0 + 45.0,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_target_h_mag=19.7,
            radar_target_visual_albedo=0.23,
            radar_target_radar_albedo=0.15,
            radar_integration_s=600.0,
        )
    ],
)
plan.candidates.radar_provenance.to_pylist()
# [['diameter derived from H + p_V',
#   'spin period unknown — coherent integration uncapped']]
```

## Reading and writing files

`empyrean.io` writes orbits, ephemerides, events, and residuals to
parquet / JSON / CSV, and reads orbits back.

```python
empyrean.io.write_orbits_parquet("orbits.parquet", fits.orbits)
empyrean.io.write_orbits_csv("orbits.csv", fits.orbits)
empyrean.io.write_residuals_parquet("residuals.parquet", fits.residuals)

orbits = empyrean.io.read_orbits_parquet("orbits.parquet")
```

Neither writer projects. The residual writers marshal every column of
`ObservationResults` across the boundary — join keys, rejection
attribution, influence diagnostics, sky-motion decomposition and radar
block included — for a 36-column file (the table's 35 plus a `has_radar`
flag), identical across all three formats; `rejection_reason` and
`radar_kind` are written as names rather than integer codes, and a
non-computable number is a literal `NaN` in CSV and `null` in JSON. The
orbit CSV path goes through the same
engine writer parquet uses, so the two emit an identical column set
(state, covariance, non-grav including `dt` and its variance, photometry,
SRP).

Parquet additionally carries the **wide cross-covariance** — the
state↔parameter and parameter↔parameter terms beyond the state+Marsden
9×9 — in a tagged tail, so a fitted orbit round-trips through a parquet
file carrying the joint the fit actually computed rather than its
diagonal blocks. It is the only orbit format here that can, and the
other two refuse such a batch by name rather than writing it short —
both pointing at parquet. The JSON orbit format is a flat row shape
carrying the 6×6 and nothing beyond it; the reason CSV cannot: this schema makes the difference between an absent
cross and a supplied zero cross load-bearing, and CSV renders both as an
empty cell. A carrier holding thrust Δv terms is refused wherever it is
offered, because no orbit-file format can serialize the thrust arcs
those terms describe.

## Data files

empyrean needs a set of SPICE kernels. Most arrive via PyPI as
installation dependencies; the remainder download on first use.

**From pip (installed automatically with `empyrean`)**

| Package | File | Size |
|---------|------|------|
| `naif-de440` | `de440.bsp` | 114 MB |
| `jpl-small-bodies-de441-n16` | `sb441-n16.bsp` | 616 MB |
| `naif-eop-high-prec` | `earth_latest_high_prec.bpc` | 5 MB |
| `naif-eop-historical` | `earth_620120_*.bpc` | 5 MB |
| `naif-eop-predict` | `earth_*_predict.bpc` | 1 MB |
| `mpc-obscodes` | `obscodes_extended.json` | 266 KB |

empyrean bundles `gm_de440.tpc` (12 KB) in the wheel itself. On
`initialize()`, empyrean stages symlinks to these files in the
platform data directory (`~/.local/share/empyrean/data/` on Linux,
`~/Library/Application Support/empyrean/data/` on macOS; honors
`EMPYREAN_DATA_DIR`) under the filenames the engine expects.

**Downloaded by the engine when needed**

| File | Size | When | Source |
|------|------|------|--------|
| `moon_pa_de440_200625.bpc` | 12 MB | first `initialize()` | NAIF — Moon orientation |
| `bias.dat` | 35 MB | first `initialize()` | Star-catalog debiasing table (Eggl, Farnocchia, Chamberlin & Chesley 2020) |
| `jwst_rec.bsp` | 121 MB | on demand, for JWST observers | NAIF — JWST ephemeris |

Any of these can be relocated by pointing `EMPYREAN_DATA_DIR` at a
directory holding them.

**Provisioning up front**

`empyrean.download_data()` provisions a usable data directory and does
nothing else — it does not build a context or load anything. It is
idempotent: files already present are kept, only what is missing is
fetched, and whatever the installed B612 wheels supply is staged from
disk with zero network access. It returns the directory it provisioned.

```python
path = empyrean.download_data()            # or download_data(data_dir=...)
```

**Running offline**

`empyrean.initialize(refresh=False)` resolves the kernel set from the
data directory alone — no downloads, no staleness checks. If anything is
missing it raises `FileNotFoundError` naming *every* absent file, both in
the message and as a `missing_data_files` list on the exception, so one
run tells you exactly what to stage. There is no partial load and no
quiet fall back to a smaller force model.

```python
try:
    empyrean.initialize(refresh=False)
except FileNotFoundError as e:
    print("stage these first:", e.missing_data_files)
```

Setting `EMPYREAN_OFFLINE=1` in the environment applies the same policy
to every `initialize()` in the process and announces itself on stderr. It
is a floor, not a switch: it can turn network access off, never back on.

## Accuracy

Validated against JPL Horizons, ASSIST, and `find_orb` on
43 objects across 13 dynamical populations (NEOs, MBAs, Trojans, TNOs,
comets, etc.). Sub-meter propagation accuracy on bounded timescales.
See [the validation notes](https://github.com/Empyrean-Dynamics/empyrean#validation).

## No guarantee of accuracy

empyrean performs numerical computations used in planetary-science
and mission-planning contexts. Outputs should not be used as the sole
basis for any decision — including but not limited to impact
monitoring, mission planning, collision avoidance, or navigation —
without independent verification. See the LICENSE file shipped with
this package for the full terms.

## License

empyrean is **dual-licensed**:

- **Wrapper / binding source code** — the Rust API surface, C-ABI
  bindings, and Python wrapper sources in the
  [main repository](https://github.com/Empyrean-Dynamics/empyrean) —
  is licensed under the
  [BSD 3-Clause License](https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD).
- **This Python wheel** (and any other pre-compiled binary
  distribution of empyrean) is licensed under the proprietary
  [Empyrean Binary License](https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY).
  The wheel is free to install and use (including commercial use)
  but **may not be redistributed, modified, reverse-engineered,
  decompiled, or disassembled**.

The BSD-3 grant covers **only the binding / integration layers**
in the public repository. The propagation engine, orbit-
determination engine, and automatic-differentiation library are
proprietary closed-source components distributed only inside the
compiled wheel — the wrapper sources call into them through
stable internal APIs but do not contain their implementations.
Cloning the repository will not let you build a working empyrean
from source; install the published wheel.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.

## Links

- Website: https://www.empyrean-dynamics.com
- Repository: https://github.com/Empyrean-Dynamics/empyrean
- Issues: https://github.com/Empyrean-Dynamics/empyrean/issues
