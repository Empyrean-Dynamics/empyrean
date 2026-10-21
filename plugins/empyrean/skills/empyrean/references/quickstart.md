<!-- generated from empyrean-py/docs/index.md by agentdocs.py; do not edit -->

# Quickstart

A complete propagation in a dozen lines:

```python
import empyrean

empyrean.initialize()                              # one-time, loads kernels

orbits = empyrean.query_sbdb(["99942"])            # CometaryOrbits — Apophis

# A time-scale-aware epoch grid: the scale (TDB) is carried by the
# type, not asserted in a comment. Spans 2024-07, 2027-04, 2029-12.
epochs = empyrean.Epochs.from_mjd([60500.0, 61500.0, 62500.0], scale="tdb")
result = empyrean.propagate(orbits, epochs)

print(f"{len(result.states)} states, "
      f"{len(result.events.summary)} events, "
      f"{len(result.sensitivity)} STMs")
```

## Output structure

`empyrean.PropagationResult` bundles three things:

```python
result.states        # CartesianOrbits (one row per epoch × orbit)
result.events        # Events (summary + 12 typed event sub-tables)
result.sensitivity   # StateSensitivities | None — STMs / STTs per epoch
```

All three are [quivr](https://github.com/B612-Asteroid-Institute/quivr)
tables — columnar, PyArrow-backed, parquet-friendly.

**Output frame and origin** default to EclipticJ2000, heliocentric
(Sun-centered), MJD TDB epochs. To get ICRF or SSB-centered output,
set `empyrean.PropagationConfig.frame` or transform via
`empyrean.transform_coordinates`.

## Inline configuration

Use the keyword shortcuts when you don't need a full config object:

```python
result = empyrean.propagate(
    orbits, epochs,
    force_model="basic",                # skip the 16 asteroid perturbers
    uncertainty_method="second_order",  # populate STTs
    num_threads=8,
)
```

For the full surface, build a `empyrean.PropagationConfig`.

## Quick reference

| You want…                              | Function                                      |
|----------------------------------------|-----------------------------------------------|
| Propagate orbits to target epochs      | `empyrean.propagate`                    |
| Predict observations at observatories  | `empyrean.generate_ephemeris`           |
| Fit an orbit to observations           | `empyrean.determine`                    |
| Re-fit with a Bayesian prior           | `empyrean.refine`                       |
| Residuals only — no fit                | `empyrean.evaluate`                     |
| Stateful, mask-and-refit OD            | `empyrean.Session`                     |
| Impact probability                     | `empyrean.compute_impact_probabilities` |
| B-plane geometry                       | `empyrean.compute_b_planes`             |
| Convert between coordinate types       | `empyrean.transform_coordinates`        |
| Body states                            | `empyrean.get_states`                   |
| Observatory states                     | `empyrean.get_observer_states`          |
| Pull an orbit from JPL SBDB            | `empyrean.query_sbdb`                   |
| Pull predicted ephemeris from Horizons | `empyrean.query_horizons`               |
| Pull observations from MPC             | `empyrean.query_observations`           |
| Read ADES PSV observations             | `empyrean.read_ades`                    |
| Look up the default data directory     | `empyrean.default_data_dir`             |
