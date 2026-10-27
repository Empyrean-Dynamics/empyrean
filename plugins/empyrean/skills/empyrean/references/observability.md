<!-- generated from empyrean-py/docs/cookbook/observability.md by agentdocs.py; do not edit -->

# Observability

Before an object can be measured it has to be observable: above the
horizon, in the dark, far enough from the Sun, and bright enough to
detect. `empyrean.compute_visibility` surveys that question for
every orbit and every observatory across a window of nights, reporting
both the per-epoch geometry and the structured night-by-night product a
scheduler needs.

```python
vis = empyrean.compute_visibility(orbits, start, end, step_days, config)
```

It steps across the window on a `step_days` grid and, for each orbit and
each site, computes:

- the per-epoch observability: four gates plus the Moon's position;
- the contiguous observable windows;
- for a ground site, the structure of each local night: the twilight
  epochs, the local solar midnight, the per-night observable intervals,
  and anchor epochs with their circumstances.

Each twilight, midnight, transit and window-boundary epoch is found by
bisection off the grid and refined to one second. The result tables come
back in the window's time scale, so build the window with
`Epochs.from_mjd(value, scale="utc")` (or `scale="tdb"`); a bare float is
refused because it carries no time scale.

## The result tables

`empyrean.compute_visibility` returns a
`empyrean.VisibilityResult`, five quivr tables that join on
`orbit_id` and `obs_code` (and `night_index` for a night's windows and
anchors):

| Attribute | Table | One row per |
|---|---|---|
| `points` | `empyrean.VisibilityPoints` | orbit × observatory × grid epoch |
| `windows` | `empyrean.VisibilityWindows` | contiguous observable interval |
| `nights` | `empyrean.Nights` | orbit × observatory × local night |
| `anchors` | `empyrean.VisibilityAnchors` | night × anchor policy |
| `observatories` | `empyrean.ObservatorySummary` | orbit × observatory |

**Points.** Each `empyrean.VisibilityPoints` row carries the four
observability gates, `elongation_ok`, `magnitude_ok`, `elevation_ok` and
`darkness_ok`, whose conjunction is the site-aware verdict `observable`.
A row also reports the solar elongation (`solar_elongation_deg`), the
target elevation (`elevation_deg`), the Sun's altitude
(`sun_altitude_deg`), the predicted magnitude (`mag`), and the Moon's
altitude, azimuth and illuminated fraction (`moon_altitude_deg`,
`moon_azimuth_deg`, `moon_illuminated_fraction`). A null magnitude fails
`magnitude_ok` rather than passing vacuously; elevation and Moon columns
are null for a horizonless, space-based observer, where that gate passes.

**Windows.** Each `empyrean.VisibilityWindows` row is a
contiguous observable interval, `start_epochs` to `end_epochs`, with
`hours_observable`, the peak elongation (`peak_elongation_deg`) and
brightest magnitude (`brightest_mag`) over the window, the maximum target
altitude and its epoch (`max_altitude_deg`, `max_altitude_epochs`), and
the `closing_gate` that ended it (`"elongation"`, `"magnitude"`,
`"elevation"`, `"darkness"`, `"end_of_night"`, or null). A `night_index`
of -1 is a whole-survey window; a non-negative index joins to that night
in `nights`.

**Nights.** Each `empyrean.Nights` row is one local night at one
site, running from one local solar noon to the next (`noon_start_epochs`,
`noon_end_epochs`). It carries the twilight epochs at each convention,
`sunset_epochs`, `civil_dusk_epochs`, `nautical_dusk_epochs`,
`astronomical_dusk_epochs` and their dawn counterparts, plus the
`local_solar_midnight_epochs`. A twilight epoch is null for a polar night
or a span that never reached the crossing, with the reason in `note`.

**Anchors.** Each `empyrean.VisibilityAnchors` row is one anchor
epoch for one night under one `policy`:
`"nearest_observable_to_midnight"`, `"midnight"`, `"transit"` or
`"explicit_utc"`. It reports the circumstances at that epoch: the
target's altitude and azimuth (`altitude_deg`, `azimuth_deg`), the Moon's
altitude and illuminated fraction (`moon_altitude_deg`,
`moon_illuminated_fraction`), the predicted magnitude with its
uncertainty (`mag`, `sigma_mag`), and the sky covariance, tagged
`sky_covariance_kind` and carried as the `joint` packed-joint column the
ephemeris rows also carry (null when the orbit has no covariance). An
anchor with no epoch this night carries a null `epochs` and null
circumstances.

**Observatories.** `empyrean.ObservatorySummary` carries the
per-site survey provenance: the resolved uncertainty method
(`resolved_method`) and the Earth-orientation summary, plus a `note` that
says why `nights` is empty for a structural reason, such as a horizonless
or space-based observer.

## Configuring observatories and limits

A survey needs at least one `empyrean.ObservatoryConfig` in
`PlanningConfig.observatories`. Each observatory carries its astrometric
assumption and its observability gates:

- `obs_code`, the MPC observatory code;
- `sigma_arcsec`, the assumed 1σ (RA·cosδ, Dec) astrometric uncertainty;
- `max_apparent_mag`, the limiting magnitude for the `magnitude_ok` gate;
- `min_elongation_deg`, the minimum solar elongation for the
  `elongation_ok` gate;
- `min_elevation_deg`, the minimum geometric elevation for the
  `elevation_ok` gate (default 0.0, the geometric horizon);
- `max_sun_altitude_deg`, the solar altitude at or below which the site
  counts as dark for the `darkness_ok` gate (default -18°, astronomical
  twilight).

The night product reads four twilight thresholds,
`sunset_sun_altitude_deg` (default -0.833°),
`civil_twilight_sun_altitude_deg` (-6°),
`nautical_twilight_sun_altitude_deg` (-12°) and
`astronomical_twilight_sun_altitude_deg` (-18°). Supply all four or none;
a partial set is refused.

Anchors come under four policies. Three are site-derived: the nearest
observable epoch to local solar midnight, local solar midnight itself,
and meridian transit. The fourth, `"explicit_utc"`, appears only when
`explicit_anchor_utc` is set; pass a single-row, scale-carrying
`empyrean.Epochs`, and every night whose noon-to-noon span
contains it also carries an anchor there.

`empyrean.PlanningConfig` also sets the force-model tier
(`force_model`, default `STANDARD`), the integrator tolerance
(`epsilon`), and the uncertainty method for the covariance fold
(`uncertainty_method`, default `FIRST_ORDER`, the same vocabulary
`empyrean.propagate` uses).

## Apophis from one observatory over one week

```python
import empyrean
from empyrean import ObservatoryConfig, PlanningConfig

empyrean.initialize()

# Apophis, with covariance, from JPL SBDB.
orbits = empyrean.query_sbdb(["99942"])

# One week in UTC. The scale rides on the Epochs, not a comment.
start = empyrean.Epochs.from_mjd([59280.0], scale="utc")   # 2021-03-07
end = empyrean.Epochs.from_mjd([59287.0], scale="utc")     # 2021-03-14

# One ground site, Mauna Kea (MPC code 568), with its astrometric
# assumption and observability gates. The darkness gate and the twilight
# thresholds fall back to the engine defaults.
config = PlanningConfig(
    observatories=[
        ObservatoryConfig(
            obs_code="568",
            sigma_arcsec=(0.3, 0.3),
            max_apparent_mag=22.0,
            min_elongation_deg=60.0,
            min_elevation_deg=20.0,
        ),
    ],
)

vis = empyrean.compute_visibility(orbits, start, end, step_days=0.02, config=config)

# Observable windows: a whole-survey window (night_index -1) and one per
# night, each with the gate that closed it.
w = vis.windows
print(f"{len(w)} window rows")
for i in range(min(6, len(w))):
    print(
        f"  night {w.night_index[i].as_py():>3}  "
        f"{w.start_epochs.to_iso()[i]} -> {w.end_epochs.to_iso()[i]}  "
        f"{w.hours_observable[i].as_py():5.2f} h  "
        f"closing={w.closing_gate[i].as_py()}"
    )

# Night anchors: one row per night per policy, with the object's
# circumstances and, when the orbit carries covariance, the sky covariance.
a = vis.anchors
print(f"{len(a)} anchor rows")
for i in range(min(6, len(a))):
    alt = a.altitude_deg[i].as_py()
    mag = a.mag[i].as_py()
    print(
        f"  night {a.night_index[i].as_py():>3}  {a.policy[i].as_py():<32}  "
        f"observable={a.observable[i].as_py()!s:<5}  "
        f"alt={'n/a' if alt is None else format(alt, '5.1f')}  "
        f"mag={'n/a' if mag is None else format(mag, '5.2f')}  "
        f"cov={a.sky_covariance_kind[i].as_py()}"
    )
```

## Ranking follow-up observations

Once an object is observable, `empyrean.evaluate_plan` ranks
candidate follow-up observations by how much each would tighten the
orbit, taking one Cartesian, covariance-bearing orbit and a list of
candidate observations and reporting what each would buy.
