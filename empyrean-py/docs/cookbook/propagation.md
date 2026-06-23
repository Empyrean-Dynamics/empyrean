# Propagation

Forward-integrate one or more orbits to a list of target epochs.

## Inputs and outputs

{func}`empyrean.propagate` takes any of the four orbit flavors
({class}`~empyrean.CartesianOrbits` / {class}`~empyrean.KeplerianOrbits`
/ {class}`~empyrean.CometaryOrbits` / {class}`~empyrean.SphericalOrbits`)
plus a target time grid and returns a {class}`~empyrean.PropagationResult`:

```python
result = empyrean.propagate(orbits, epochs, config=cfg)
result.states        # CartesianOrbits with optional covariance
result.events        # Events: summary + 12 typed event sub-tables (CAs, impacts, …)
result.sensitivity   # StateSensitivities | None — STMs (and STTs if SECOND_ORDER)
```

## Force-model tiers

| Tier          | What's included                                                                  |
|---------------|----------------------------------------------------------------------------------|
| `Approximate` | Point-mass planets + Moon + Pluto                                                |
| `Basic`       | + EIH (Einstein-Infeld-Hoffmann) general relativity + Sun $J_2$              |
| `Standard`    | + 16 SB441 small-body perturbers (paired with DE440 planets) + Earth $J_2$–$J_4$ + Marsden non-grav |

`Standard` is the default and matches ASSIST's standard NEO force
model. SB441 is the small-body kernel family JPL ships paired with
DE441; using it alongside DE440 is standard practice and the
difference is below the 16-perturber accuracy floor.

## Uncertainty methods

| Method              | Output                                                                                                          | Cost                              |
|---------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `FirstOrder`        | STM $\Phi$ via `Jet1`                                                                                           | ~10× scalar                       |
| `SecondOrder`       | STM + STT $\Psi$ via `Jet2` (Park-Scheeres second-order Gaussian; second-order impact-probability correction)   | ~50× scalar                       |
| `GaussianMixture(…)`| Adaptive splitting along the most non-linear direction; one mixture component per resulting sub-orbit            | depth-bounded                     |
| `SigmaPoint(…)`     | Output covariance from unscented sigma points                                                                   | $N$ × scalar                      |
| `MonteCarlo(…)`     | Output covariance from random samples                                                                           | $N$ × scalar                      |

Pass an `UncertaintyMethod` enum (default parameters) or one of the
parameterized dataclasses ({class}`~empyrean.GaussianMixture`,
{class}`~empyrean.SigmaPoint`, {class}`~empyrean.MonteCarlo`).

`FirstOrder` is the default and is adequate for the bulk of NEO
work — STM-mapped covariance agrees with sample-based estimates to
better than ~1% on linear arcs. Reach for `SecondOrder` when working
close to a planetary close approach (where the dynamics get
non-linear over the uncertainty volume) or whenever you need the
second-order impact-probability correction. `GaussianMixture` adapts
on its own — it splits along the most non-linear directions where a
single second-order map isn't enough, and otherwise behaves like
`SecondOrder`, which is what you want for a deep flyby with a long
lead-up of linear dynamics.
`SigmaPoint` and `MonteCarlo` are sample-based alternatives when you
need tail probabilities or want to exercise the full distribution.

## Multi-orbit batch propagation

`propagate` accepts an Orbits table of any length and integrates each
row in parallel under the configured thread count. The output
`states` table is orbit-major: rows 0..N-1 belong to orbit 0 at the
N requested epochs, rows N..2N-1 belong to orbit 1, and so on.
Filter to one orbit's chain with the standard quivr `select`:

```python
import empyrean
from empyrean import PropagationConfig

empyrean.initialize()

# The four canonical close-approach scenarios — same set the empyrean 3D
# viewer ships as built-in fixtures. Each one stresses a different
# corner of the propagation pipeline.
scenarios = empyrean.query_sbdb([
    "99942",                # Apophis — 2029-04-13 deep Earth flyby (MJD 62239)
    "2024 YR4",             # 2032-12 close approach with non-zero impact corridor (MJD 63587)
    "2020 CD3",             # mini-moon temporary capture (MJD 57561)
    "2008 TC3",             # first asteroid predicted to impact Earth — Almahata Sitta airburst (MJD 54746.12, 2008-10-07 02:46 UTC)
])

cfg = PropagationConfig(num_threads=8)             # 1 core per orbit, up to 8
result = empyrean.propagate(
    scenarios,
    empyrean.Epochs.linspace(54000.0, 64000.0, 41, scale="tdb"),  # ~27 yr span
    config=cfg,
)

print(f"{len(result.states)} state rows = "
      f"{len(scenarios)} orbits × {len(result.states) // len(scenarios)} epochs")

# Pull just one scenario's chain:
apophis = result.states.select("orbit_id", "(99942) Apophis")
```

The `orbit_id` column on every output table (`states`, the 12 event
sub-tables, `sensitivity`) preserves the input identifier — that's
your join key for risk-list operations. The four scenarios above are
the same fixtures bundled with the empyrean 3D viewer, so any state
you produce here can be cross-referenced against the rendered
trajectory there.

## Non-gravitational forces (Yarkovsky / outgassing)

For asteroids with a measurable Yarkovsky drift or comets with
outgassing, attach Marsden $(A_1, A_2, A_3)$ coefficients via
{class}`~empyrean.NonGravParams` on the input orbit:

```python
from empyrean import NonGravParams

# Apophis Yarkovsky: A2 ≈ -2.9e-14 AU/day², radial / normal terms
# negligible. Values from JPL SBDB.
yarkovsky = NonGravParams.from_kwargs(
    model=["inverse_square"],   # g(r) ∝ 1/r² — the standard Yarkovsky form (required)
    a1=[0.0],
    a2=[-2.9e-14],
    a3=[0.0],
)

apophis_with_yark = empyrean.query_sbdb(["99942"]).set_column(
    "non_grav", yarkovsky,
)

epochs = empyrean.Epochs.from_mjd([62239.0], scale="tdb")   # Apophis 2029 Earth CA
result = empyrean.propagate(apophis_with_yark, epochs)
```

SBDB's `query_sbdb` already attaches non-grav parameters when JPL has
fitted them, so the manual construction above is only needed when
overriding or for objects without a SBDB non-grav fit. Standard force
model honours non-grav by default; Approximate / Basic ignore it.

### Cometary outgassing — `model`, `g(r)`, and `dt`

For asteroids the inverse-square Marsden law is the right default
(``model="inverse_square"``). Comets with measurable water-ice
sublimation use ``model="marsden_water"``, which evaluates the
canonical Marsden $g(r)$ function

$$
g(r) = \alpha\,(r/r_0)^{-m}\,\left[1 + (r/r_0)^n\right]^{-k}
$$

with the standard $H_2O$ sublimation parameters
$(\alpha, r_0, m, n, k) = (0.1113,\ 2.808,\ 2.15,\ 5.093,\ 4.6142)$.
For non-water-ice volatiles or custom fits, use ``model="marsden"``
and set ``alpha``/``r0``/``m``/``n``/``k`` explicitly.

Some Jupiter-family comets and 2I/Borisov also fit a peak-outgassing
**time delay** $\Delta t$ relative to perihelion — SBDB exposes this
as `model_pars[]`'s `DT` field. Set it via the ``dt`` column on
``NonGravParams``:

| Object              | $\Delta t$ (days) |
|---------------------|-------------------|
| 67P/Churyumov-Gerasimenko | +46         |
| 46P/Wirtanen        | −14               |
| 103P/Hartley 2      | +12               |
| 2I/Borisov          | −65               |

Asteroids and short-period comets that SBDB doesn't fit a delay for
should leave ``dt`` unset (`None` / null) — the default.

### Fitting non-grav parameters

For OD that *fits* the non-grav parameters from observations, see
{class}`~empyrean.SolveForParams.STATE_AND_NONGRAV` on
{class}`~empyrean.ODConfig`.

## Events

Set toggles on {class}`~empyrean.EventConfig`:

```python
from empyrean import EventConfig, PropagationConfig

cfg = PropagationConfig(
    events=EventConfig(
        close_approaches=True,
        possible_impacts=True,
        body_filter=["Earth"],
        dense_output=True,
        dense_output_cadence_days=5.0 / 1440.0,  # 5 min
    ),
)
```
