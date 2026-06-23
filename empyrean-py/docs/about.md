# About empyrean

empyrean is a Python toolkit for high-precision Solar System dynamics
— trajectory propagation, ephemeris generation, orbit determination,
and event analysis (close approaches, gravitational capture, eclipses,
atmospheric entry) on real Solar System bodies. It pairs a
friendly, columnar Python surface (PyArrow / quivr tables, dataclass
configs, type-hinted everything) with a Rust engine that delivers
production-grade numerics at millisecond-class single-orbit
performance.

The full pipeline runs in one process: query JPL SBDB or MPC for an
orbit, propagate it forward or backward across decades, predict
apparent positions for any observatory in the MPC catalog, fit an
orbit from astrometric observations, and detect close approaches,
gravitational capture, and eclipses along the trajectory — all without leaving
Python and without round-tripping to a server.

## How it's built

empyrean composes three specialized Rust libraries into one Python
package, each owning one layer of the stack:

| Library | What it owns |
|---|---|
| **[nolan](https://github.com/Empyrean-Dynamics/nolan)** | Forward-mode automatic differentiation. The `Jet1` / `Jet2` number types that ride through every position, velocity, and force computation — so state-transition matrices (STMs) and state-transition tensors (STTs) come out as a by-product of integration, no finite differences. This is what makes uncertainty-first propagation possible at production speed. |
| **[villeneuve](https://github.com/Empyrean-Dynamics/villeneuve)** | Numerical propagation. Force models from approximate (point-mass planets, Moon, and Pluto) through standard (planets, sixteen asteroid perturbers, EIH general relativity, Earth zonal harmonics, Marsden non-gravitationals). High-order adaptive integration. Ephemeris generation against the MPC observatory catalog. Event detection — close approaches, eclipses (shadow entry / exit), gravitational capture, atmospheric entry / exit. |
| **[scott](https://github.com/Empyrean-Dynamics/scott)** | Orbit determination. Initial-orbit determination (Gauss, Herget, systematic ranging) and differential correction with adaptive multi-layer outlier rejection, per-station bias fitting, and catalog astrometric debiasing (EFCC2020). Returns the fitted orbit + covariance + per-observation residuals + an acceptability verdict you can gate on.|

Each library ships as a proprietary compiled binary; empyrean is the
Python integration layer that exposes their behaviour through a single
coherent API.

The same engine is the source of truth for the **C** and **CLI**
distributions of empyrean — every numerical result is bit-identical
across `empyrean` (Python), the C ABI (`libempyrean.dylib` /
`libempyrean.so`), and the standalone `empyrean` command-line binary.
The Python wheel is just the most ergonomic of the three for
notebook-driven and script-driven work.

## See it in action

Each scenario below is a real interactive page on
[empyrean-dynamics.com](https://empyrean-dynamics.com/scenarios)
running the same propagation, OD, and impact-probability machinery
this Python package exposes. Open one to scrub through the
trajectory, inspect the covariance, and see the close-approach or
impact geometry in 3D.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Apophis 2029
:link: https://empyrean-dynamics.com/scenarios/apophis-2029
A textbook nonlinear-dynamics encounter — Earth's gravity field
bending the trajectory and amplifying initial-condition uncertainty
across a 15-year propagation.
:::

:::{grid-item-card} 2024 YR4
:link: https://empyrean-dynamics.com/scenarios/2024-yr4
A recently-discovered NEO with a broad a-priori covariance —
sigma-point propagation showing how a virtual-asteroid sample
disperses across a multi-year forecast.
:::

:::{grid-item-card} 2008 TC3
:link: https://empyrean-dynamics.com/scenarios/2008-tc3
Recovering an orbit from a 19-hour discovery arc — the original
October 2008 observations, fit and propagated to atmospheric entry
over Sudan.
:::

:::{grid-item-card} 2024 PT5 (mini-moon)
:link: https://empyrean-dynamics.com/scenarios/2024-pt5
A temporary Earth co-orbital captured into a near-Earth lasso in
2024 — chaotic dynamics where small initial-condition differences
amplify dramatically.
:::

:::{grid-item-card} 67P / Churyumov-Gerasimenko
:link: https://empyrean-dynamics.com/scenarios/67p
Rosetta's target — a Jupiter-family comet with non-zero outgassing
forces and a measurable Marsden time-of-perihelion delay.
:::

:::{grid-item-card} 1I / 'Oumuamua
:link: https://empyrean-dynamics.com/scenarios/1i-oumuamua
The first interstellar object — hyperbolic orbit with custom
non-gravitational acceleration model.
:::

::::

The same engine powers each scenario; what changes is the orbit, the
force model, and the time horizon.

## How the pieces fit together

A typical end-to-end workflow exercises every layer:

```python
import empyrean

empyrean.initialize()

# 1. Pull an orbit from JPL SBDB → CometaryOrbits with covariance.
orbits = empyrean.query_sbdb(["99942"])             # nolan + villeneuve I/O

# 2. Pull historical observations from MPC → ADESObservations.
obs = empyrean.query_observations(["99942"])

# 3. Re-fit the orbit from the observations → ODResult.
fit = empyrean.refine(orbits, obs)                  # scott + villeneuve

# 4. Propagate the fitted orbit forward 10 years → states + events.
#    The fitted epoch is a time-scale-aware Epochs; do the +10 yr offset
#    in the TDB numeric domain and rebuild a typed grid.
epoch_0 = fit.orbit.coordinates.epoch
targets = empyrean.Epochs.from_mjd(epoch_0.mjd_tdb() + 10 * 365.25, scale="tdb")
result  = empyrean.propagate(fit.orbit, targets)
result.events.close_approach_starts  # one row per CA per body
result.events.capture_starts         # temporary gravitational capture
result.events.shadow_entries         # umbral / penumbral entry (eclipse)

# 5. Predict apparent positions at Mauna Kea (obs code 568) → Ephemeris.
#    generate_ephemeris reads its prediction times from the Observers.
observers = empyrean.Observers.from_code("568", targets)
eph = empyrean.generate_ephemeris(fit.orbit, observers)
```

For close-encounter analysis (impact probability, B-plane geometry,
keyhole structure), see the
[Close-approach cookbook](cookbook/impact-probability.md).

You see one Python API, but each call dispatches into the right
combination of nolan (autodiff), villeneuve (numerics), and scott
(estimation) underneath. From your perspective they're a single
coherent toolkit; from the engine's perspective they're three
specialized libraries doing what each does best.
