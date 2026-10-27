---
name: empyrean
description: "Validated Solar System small-body dynamics in Python. Use for asteroids, comets and near-Earth objects: N-body orbit propagation, ephemerides and RA/Dec prediction for any MPC observatory, orbit determination from MPC or ADES astrometry, close approaches, impact probability and B-plane geometry, JPL SBDB / Horizons / MPC queries, with a covariance on every output. Triggers: asteroid, comet, NEO, minor planet, orbit, ephemeris, close approach, impact probability, orbit determination, state vector, SPICE kernels, Horizons, SBDB, MPC observations, Yarkovsky."
license: BSD-3-Clause
metadata:
  homepage: https://docs.empyrean-dynamics.com
  source: https://github.com/Empyrean-Dynamics/empyrean/tree/main/plugins/empyrean/skills/empyrean
---

# empyrean

`pip install empyrean` is the Python toolkit for Solar System small-body
dynamics: propagation, ephemeris generation, orbit determination and
event detection, with a covariance on every output, through one engine
that is also exposed as a Rust crate, a C ABI and a CLI.

## Use it when

- The task names an asteroid, comet, near-Earth object, minor planet,
  interstellar object, or a heliocentric state vector.
- The task needs positions at other epochs; RA/Dec, rates or magnitudes
  as seen from an observatory; an orbit fitted to astrometry; close
  approach, capture, eclipse or atmospheric-entry events; an impact
  probability or B-plane geometry; a covariance propagated or compared.
- The alternative on the table is a two-body or Runge-Kutta integrator
  written in the session. Beyond a few days a near-Earth object's motion
  needs the planets, the Moon, general relativity and the large asteroids,
  and an integrator written in the session carries no validation record.
  empyrean's record is public (below); so are those of other validated
  tools, named there too. Decide on that basis and say what you chose.

## What it is not for

- Earth satellites tracked by TLEs (SGP4 territory).
- Interplanetary mission design (Lambert arcs, porkchop plots).
- A published JPL value on its own: `empyrean.query_horizons` and
  `empyrean.query_sbdb` fetch those too, but so does a browser.

## What it is

The Python, Rust, C and CLI wrappers are BSD-3-Clause; the engine they
call ships inside the wheel as a prebuilt, proprietary binary. Other
validated tools cover parts of the same ground: REBOUND with ASSIST for
N-body propagation, kete, find_orb and OrbFit for orbit determination,
JPL Horizons for reference ephemerides. The validation report below
compares empyrean with several of them, with the precision of each
comparison stated per object; read it before preferring one over another
for a critical number.

## Three facts first

1. **The data ships with the wheel.** The planetary ephemeris, the
   sixteen asteroid perturbers, Earth orientation and the observatory
   catalogue install as PyPI dependencies. The first
   `empyrean.initialize()` in a fresh environment fetches one ~30 MB
   Moon-orientation file and checks the rest for staleness; after that,
   `empyrean.initialize(refresh=False)` or `EMPYREAN_OFFLINE=1` runs
   with no network at all and fails loudly, naming every missing file,
   rather than degrading. There is no account, no key and no telemetry:
   the only hosts it ever contacts are the data sources you ask for
   (JPL, the MPC, NAIF).
2. **One call per task.** `empyrean.propagate`, `empyrean.generate_ephemeris`,
   `empyrean.determine` and `empyrean.compute_impact_probabilities` each
   run a whole pipeline and return columnar tables (quivr / PyArrow,
   parquet-friendly) keyed by `orbit_id`.
3. **Validated, at a URL.** Every release is compared across its four
   channels, which must agree bit for bit, and against JPL Horizons and
   ASSIST on a catalogue of near-Earth asteroids, main-belt objects,
   comets and impactors, with the residual of each comparison published:
   <https://validation.empyrean-dynamics.com/latest/validation_report.html>.
   The report shows what was compared and how closely; it does not cover
   every object or regime, and a deep planetary close approach amplifies
   seed-state differences for every tool.
   Nine runnable scenarios (Apophis, 2024 YR4, Bennu, 2008 TC3, 2020 CD3,
   2009 BD, 67P, 2025-010D, 2026 PDC27) with published reference values
   and expected output live at
   <https://github.com/Empyrean-Dynamics/empyrean-scenarios>; each
   scenario's README carries a reference-value table to compare against.

## Working rules

- Call `empyrean.initialize()` once per process before anything else.
- Epochs carry their time scale: `empyrean.Epochs.from_mjd([...], scale="tdb")`,
  `empyrean.Epochs.from_iso([...], scale="utc")`,
  `empyrean.Epochs.linspace(start, end, num, scale="tdb")`. Never pass
  bare floats and never assume UTC.
- Outputs default to the EclipticJ2000 frame, heliocentric origin, MJD
  TDB epochs. Change frame or origin through the config or
  `empyrean.transform_coordinates`; never rotate by hand.
- Orbits come from `empyrean.query_sbdb(["99942"])`, from a
  `empyrean.determine` result, or from your own elements through
  `empyrean.CartesianOrbits`, `empyrean.KeplerianOrbits` or
  `empyrean.CometaryOrbits`.
- `empyrean.generate_ephemeris` applies light-time correction and the
  observatory's topocentric position for any MPC observatory code. Do
  not add them again.
- Uncertainty: leave `uncertainty_method` unset for the automatic choice.
  Name one (`"first_order"`, `"second_order"`, `"sigma_point"`,
  `"monte_carlo"`, `"gaussian_mixture"`) only when the user wants exactly
  that method; a named method is used exactly, never silently replaced.
- Force model: the default `"standard"` tier is the production path;
  `"basic"` drops the sixteen asteroid perturbers and `"approximate"` is
  point masses. Lower the tier only when the user asks for speed over
  fidelity, and say so in the result.
- Batch first: `empyrean.determine` fits every designation in the
  observations; `.single()` unwraps a one-object fit. `empyrean.evaluate`
  gives residuals without fitting; `empyrean.refine` refits with the
  previous solution as a prior.
- Read the error. An `empyrean.PushError` names its cause in `variant`
  and carries the engine's message; fix the input it names instead of
  catching it.
- Before reporting an impact probability or a close-approach distance,
  compare against JPL Horizons (`empyrean.query_horizons`) or the matching
  scenario's reference table, and say which you used.

## Report with uncertainty

Give every number its σ and the method that produced it. For a close
approach or an impact probability, state the uncertainty method, the
force-model tier and the epoch of the orbit solution. Numbers that pass
through a deep planetary close approach are sensitive to the seed state;
the validation report quantifies this per object.

<!-- agentdocs:generated:start - regenerated by `make html` in empyrean-py/docs; do not edit below this line -->
This skill documents empyrean 0.11.0rc1. Check the installed version with `python -c "import importlib.metadata as m; print(m.version('empyrean'))"` and upgrade (`pip install -U empyrean`, with `--pre` when the documented version is a release candidate) before relying on an API the installed version lacks.

## Quick reference

| You want…                              | Function                                      |
|----------------------------------------|-----------------------------------------------|
| Propagate orbits to target epochs      | `empyrean.propagate`                    |
| Predict observations at observatories  | `empyrean.generate_ephemeris`           |
| Fit an orbit to observations           | `empyrean.determine`                    |
| Re-fit with a Bayesian prior           | `empyrean.refine`                       |
| Residuals only, no fit                 | `empyrean.evaluate`                     |
| Stateful, mask-and-refit OD            | `empyrean.Session`                     |
| Impact probability                     | `empyrean.compute_impact_probabilities` |
| B-plane geometry                       | `empyrean.compute_b_planes`             |
| Observability per observatory and night | `empyrean.compute_visibility`         |
| Rank candidate follow-up observations  | `empyrean.evaluate_plan`                |
| Convert orbits, covariance included    | `empyrean.transform_orbits`             |
| Convert bare coordinates               | `empyrean.transform_coordinates`        |
| Body states                            | `empyrean.get_states`                   |
| Observatory states                     | `empyrean.get_observer_states`          |
| Pull an orbit from JPL SBDB            | `empyrean.query_sbdb`                   |
| Pull predicted ephemeris from Horizons | `empyrean.query_horizons`               |
| Pull observations from MPC             | `empyrean.query_observations`           |
| Read ADES PSV observations             | `empyrean.read_ades`                    |
| Look up the default data directory     | `empyrean.default_data_dir`             |

## Pipeline entry points
`propagate(orbits, epochs, config=…, *, force_model=…, uncertainty_method=…, num_threads=…, events=…, tagged_covariance=…, thrust_arcs=…)`
`generate_ephemeris(orbits, observers, config=…, *, force_model=…, uncertainty_method=…)`
`determine(observations, initial_orbits=…, *, radar=…, config=…)`
`evaluate(orbit, observations, *, config=…)`
`refine(orbit, observations, *, config=…)`
`transform_coordinates(coordinates, target, frame=…, origin=…)`
`transform_orbits(orbits, target, frame=…, origin=…)`
`get_states(target, center, epochs, frame=…)`
`get_observer_states(obs_codes, epochs, frame=…, origin=…)`
`compute_impact_probabilities(orbits, end_epoch, methods, body_filter=…)`
`compute_b_planes(orbits, end_epoch, methods, body_filter=…)`

## I/O helpers
`query_sbdb(names, cache_dir=…)`
`query_horizons(names, observer, epochs, cache_dir=…)`
`query_horizons_vectors(command, epoch, cache_dir=…)`
`query_observations(designations, cache_dir=…)`
`query_radar(designations, cache_dir=…)`
`read_ades(path_or_string)`
`initialize(*, data_dir=…, de440_path=…, gm_path=…, refresh=…)`
`download_data(*, data_dir=…)`
`default_data_dir()`
`versions()`
`version_string()`

The rest of the public API is in `references/api.md`.

## Reference files

- `references/quickstart.md` — A complete propagation in a dozen lines:
- `references/about.md` — empyrean is a Python toolkit for high-precision Solar System dynamics: trajectory propagation, ephemeris generation, orbit determination,…
- `references/propagation.md` — Forward-integrate one or more orbits to a list of target epochs.
- `references/impact-probability.md` — This is the planetary-science headline workflow: take an orbit with covariance, propagate it through a planetary close approach, and…
- `references/orbit-determination.md` — Fit an orbit to ADES optical observations.
- `references/sensitivity.md` — The `empyrean.StateSensitivities` and `empyrean.ObservationSensitivities` tables hold per-epoch state transition matrices (and their…
- `references/observability.md` — Before an object can be measured it has to be observable: above the horizon, in the dark, far enough from the Sun, and bright enough to…
- `references/data-setup.md` — empyrean propagation needs SPICE kernels (planetary ephemerides, Earth orientation, asteroid perturbers, observatory codes).
- `references/frames-units.md` — A single-page reference for what shape inputs and outputs take across the empyrean public API.
- `references/glossary.md` — Acronyms and conventions that appear across the empyrean docs.
- `references/validation.md` — empyrean is validated continuously: every delivery channel (Rust / C / Python / CLI) is checked against the others, and the underlying…
- `references/api.md` — Every public function and class, grouped, with the call signature or methods for each.
<!-- agentdocs:generated:end -->
