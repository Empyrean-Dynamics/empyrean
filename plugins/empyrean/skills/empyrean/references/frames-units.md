<!-- generated from empyrean-py/docs/reference/frames-units.md by agentdocs.py; do not edit -->

# Frames, units, and time scales

A single-page reference for what shape inputs and outputs take across
the empyrean public API. Defaults are chosen to match the integration
frame and the convention most NEO-work pipelines expect; override on
`empyrean.PropagationConfig` /
`empyrean.transform_coordinates` when you need something else.

## Time scales

The `empyrean.TimeScale` enum carries the two scales empyrean
exposes at the API boundary:

| Scale  | Use                                                                 |
|--------|---------------------------------------------------------------------|
| `TDB`  | Barycentric Dynamical Time: the integration time scale, and what everything converts to internally. |
| `UTC`  | Coordinated Universal Time, leap-second-aware. Use for human-readable epochs (ISO strings) and observation timestamps. |

**Every time-bearing column is an `empyrean.Epochs` sub-table**
(an `mjd` array plus a table-level `scale`), and every time you hand to
empyrean is an `Epochs` table. A table's primary time column is
`epochs`; any other time column is `<role>_epochs` (`start_epochs`,
`ca_epochs`, `perihelion_epochs`, and so on). One table carries one
scale, and a mismatch is refused by name. There is no bare-MJD entry
point and no default scale to fall back on: a plain `61000.5` does not
say which clock it was read from, and read as UTC rather than TDB it
names an instant about 69 seconds earlier, a difference that grows with
every leap second and is easily large enough to move a close-approach
geometry. Passing a bare list, array or float where a time is expected
raises `TypeError` naming the fix.

```python
epochs = empyrean.Epochs.from_mjd([61000.5, 61010.5], scale="tdb")
epochs = empyrean.Epochs.from_mjd([61000.5, 61010.5], scale="utc")   # not the same instants
epochs = empyrean.Epochs.from_iso(["2026-01-01T12:00:00.000Z"])      # ISO is UTC by format
```

`Epochs` carries the scale as a `StringAttribute` and exposes
`.to_tdb()` / `.to_utc()` / `.to_scale(...)` for conversion, plus
`.mjd_tdb()` / `.mjd_utc()` and `.to_iso()` to read the values back out
in a named scale. UTC↔TDB conversion (leap seconds plus the TDB−TT
periodic terms) is built into the engine, so no separate leap-second
kernel is installed.

Results come back in the scale the caller supplied: requested instants
are echoed as the caller's own values, and instants the engine finds
(close approaches, impacts, events, rise and set, twilight) are
converted once. To turn a coordinate table's epochs into a time input,
read its `epochs` sub-table directly or let
`Epochs.from_orbits(orbits, offsets_in_days)` build the target grid.
File and stdout writers are the one exception: they still write TDB
`epoch_mjd_tdb` columns.

Earth-rotation kernels are stored on the TT scale internally; the
TT ↔ TDB difference is small (a quasi-periodic term with peak
amplitude ~1.7 ms, dominated by an annual component) and is applied
automatically.

## Reference frames

The `empyrean.Frame` enum (use the all-caps Python member
name; the string slug shown in the second column also works):

| Member            | String slug      | Notes                                                                  |
|-------------------|------------------|------------------------------------------------------------------------|
| `ECLIPTICJ2000`   | `"eclipticj2000"`| Mean ecliptic and equinox of J2000.0. **Integration frame; default propagation output.** Convert to ICRF via `empyrean.transform_coordinates(..., frame="icrf")` when downstream tools (Horizons, most observatory pipelines) expect equatorial. |
| `ICRF`            | `"icrf"`         | International Celestial Reference Frame (≈ J2000 equatorial).         |
| `ITRF93`          | `"itrf93"`       | Earth-fixed (rotating), for ground-station vectors.                   |

Conversion is via `empyrean.transform_coordinates`, which
accepts a target frame as a kwarg and propagates covariance through
the rotation.

## Origins

The `empyrean.Origin` enum is keyed by NAIF body code; only
the bodies whose ephemeris is actually loaded by `ForceModelTier`
≥ APPROXIMATE are exposed:

| Member               | String slug          | Use                                                              |
|----------------------|----------------------|------------------------------------------------------------------|
| `SSB`                | `"SSB"`              | Solar System Barycenter (NAIF 0), the propagation origin.       |
| `SUN`                | `"Sun"`              | NAIF 10. Heliocentric; what SBDB returns for small-body orbits. |
| `EARTH`              | `"Earth"`            | NAIF 399. Geocentric, for Earth-observer ephemeris.             |
| `MOON`               | `"Moon"`             | NAIF 301. Selenocentric.                                         |
| `MERCURY`            | `"Mercury"`          | NAIF 199.                                                        |
| `VENUS`              | `"Venus"`            | NAIF 299.                                                        |
| `MARS_BARYCENTER`    | `"Mars Barycenter"`  | NAIF 4. Mars *body-center* not exposed (DE440 ships the barycenter only). |
| `JUPITER_BARYCENTER` | `"Jupiter Barycenter"` | NAIF 5.                                                        |
| `SATURN_BARYCENTER`  | `"Saturn Barycenter"`  | NAIF 6.                                                        |
| `URANUS_BARYCENTER`  | `"Uranus Barycenter"`  | NAIF 7.                                                        |
| `NEPTUNE_BARYCENTER` | `"Neptune Barycenter"` | NAIF 8.                                                        |
| `PLUTO_BARYCENTER`   | `"Pluto Barycenter"`   | NAIF 9. Pluto body-center is not exposed; same DE440 reason as Mars. |
| `Origin.asteroid(n)` | `"asteroid_<n>"`     | Numbered-asteroid origin, e.g. `Origin.asteroid(433)` for Eros. Useful when fitting an asteroid that the force model would otherwise include as a perturber: pass it via `excluded_perturbers` so it does not self-attract. |

### Observer states

`empyrean.Observers.from_code` /
`empyrean.Observers.from_codes` and
`empyrean.get_observer_states` return ICRF / SSB by default,
the **construction basis**, and the one
`empyrean.generate_ephemeris` and orbit determination require.
Requesting it takes no transform at all: the states come back exactly as
constructed, bit for bit.

Pass `frame=` / `origin=` when you want the geometry in some other
basis, plotting heliocentric ecliptic site positions, say:

```python
sites = empyrean.Observers.from_codes(
    ["500", "568"], epochs, frame="eclipticj2000", origin="Sun"
)
```

Every returned row carries the basis it is expressed in, read off the
state rather than echoed from the request, so a table is never ambiguous
about which basis produced it.

For unfamiliar bodies, see
[NAIF's body-ID page](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html);
to add a new body, the underlying ephemeris kernel set has to carry
its segment.

## Units

| Quantity              | Unit                       | Notes                                  |
|-----------------------|----------------------------|----------------------------------------|
| Position              | astronomical units (AU)    | At the public API boundary.            |
| Velocity              | AU / day                   | Same.                                  |
| Time                  | days                       | Propagation step sizes, durations, etc. |
| Epoch                 | `Epochs` (MJD + its scale) | Every time column is an `Epochs` sub-table carrying its own scale; results come back in the scale you supplied. File and stdout writers alone use TDB `epoch_mjd_tdb` columns. See above. |
| Angle                 | degrees                    | At the boundary; converted to radians internally. |
| Astrometric residuals | arcseconds                 | RA·cos(δ), Dec, AT/CT decompositions.   |
| Track position angle  | degrees (East of North)    | On `empyrean.ObservationResults`. |
| Astrometric magnitudes | V-band                    | H, V, mag are V-equivalent unless documented otherwise. |
| Impact-probability    | dimensionless probability  | `[0.0, 1.0]`.                          |
| B-plane coordinates   | km                         | `b_dot_t_km`, `b_dot_r_km`, etc.       |
| B-plane covariance    | km²                        | `cov_tt_km2`, `cov_tr_km2`, `cov_rr_km2`. |
| Hyperbolic excess velocity | km / s                | `v_inf_km_s`.                          |
| Earth-rotation kernels | TT internally             | Conversion is automatic; users see TDB. |

When a function takes a `Frame` / `Origin` parameter, you can pass
either the enum value (`Frame.ICRF`) or the string slug
(`"icrf"` / `"eclipticj2000"`, case-insensitive). Same for `Origin`
which also accepts a raw NAIF integer.

## Coordinate representations

The `empyrean.CartesianCoordinates` /
`empyrean.KeplerianCoordinates` /
`empyrean.CometaryCoordinates` /
`empyrean.SphericalCoordinates` family supports four
representations on input and output:

| Representation | Elements                              | Notes                                    |
|----------------|---------------------------------------|------------------------------------------|
| Cartesian      | `(x, y, z, vx, vy, vz)` in AU, AU/day | The integration representation.          |
| Keplerian      | `(a, e, i, raan, ap, ma)`             | Mean anomaly. Angles in degrees.         |
| Cometary       | `(q, e, i, raan, ap, perihelion_epochs)` | Perihelion-time element is an `Epochs` sub-table. **What SBDB returns for small bodies.** |
| Spherical      | `(r, lon, lat, vr, vlon, vlat)`       | Topocentric or geocentric astrometry.    |

The propagator accepts any of the four. One packed joint carries the
state and every solved parameter (non-grav, SRP, thrust) alongside it,
and `empyrean.transform_orbits` transports that joint across
representation, frame and origin through the analytic Jacobians, so it
is the entry point for moving a covariance between bases.
`empyrean.transform_coordinates` transforms bare states and
refuses a covariance-bearing input by name, pointing you to
`transform_orbits`.
