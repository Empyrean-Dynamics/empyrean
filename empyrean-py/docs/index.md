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

{class}`~empyrean.PropagationResult` bundles three things:

```python
result.states        # CartesianOrbits (one row per epoch × orbit)
result.events        # Events (summary + 12 typed event sub-tables)
result.sensitivity   # StateSensitivities | None — STMs / STTs per epoch
```

All three are [quivr](https://github.com/B612-Asteroid-Institute/quivr)
tables — columnar, PyArrow-backed, parquet-friendly.

**Output frame and origin** default to EclipticJ2000, heliocentric
(Sun-centered), MJD TDB epochs. To get ICRF or SSB-centered output,
set {attr}`~empyrean.PropagationConfig.frame` or transform via
{func}`~empyrean.transform_coordinates`.

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

For the full surface, build a {class}`~empyrean.PropagationConfig`.

## Quick reference

| You want…                              | Function                                      |
|----------------------------------------|-----------------------------------------------|
| Propagate orbits to target epochs      | {func}`empyrean.propagate`                    |
| Predict observations at observatories  | {func}`empyrean.generate_ephemeris`           |
| Fit an orbit to observations           | {func}`empyrean.determine`                    |
| Re-fit with a Bayesian prior           | {func}`empyrean.refine`                       |
| Residuals only — no fit                | {func}`empyrean.evaluate`                     |
| Stateful, mask-and-refit OD            | {class}`empyrean.Session`                     |
| Impact probability                     | {func}`empyrean.compute_impact_probabilities` |
| B-plane geometry                       | {func}`empyrean.compute_b_planes`             |
| Convert between coordinate types       | {func}`empyrean.transform_coordinates`        |
| Body states                            | {func}`empyrean.get_states`                   |
| Observatory states                     | {func}`empyrean.get_observer_states`          |
| Pull an orbit from JPL SBDB            | {func}`empyrean.query_sbdb`                   |
| Pull predicted ephemeris from Horizons | {func}`empyrean.query_horizons`               |
| Pull observations from MPC             | {func}`empyrean.query_observations`           |
| Read ADES PSV observations             | {func}`empyrean.read_ades`                    |
| Look up the default data directory     | {func}`empyrean.default_data_dir`             |

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

about
cookbook/propagation
cookbook/impact-probability
cookbook/orbit-determination
cookbook/sensitivity
cookbook/data-setup
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

reference/frames-units
reference/glossary
reference/validation
api
```
