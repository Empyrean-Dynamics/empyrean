# Frames, units, and time scales

A single-page reference for what shape inputs and outputs take across
the empyrean public API. Defaults are chosen to match the integration
frame and the convention most NEO-work pipelines expect — override on
:class:`~empyrean.PropagationConfig` /
:class:`~empyrean.transform_coordinates` when you need something else.

## Time scales

The :class:`~empyrean.TimeScale` enum carries the two scales empyrean
exposes at the API boundary:

| Scale  | Use                                                                 |
|--------|---------------------------------------------------------------------|
| `TDB`  | Barycentric Dynamical Time — the integration time scale. **Default everywhere in empyrean.** All raw MJD floats at the API boundary are MJD TDB unless documented otherwise. |
| `UTC`  | Coordinated Universal Time — leap-second-aware. Use for human-readable epochs (ISO strings) and observation timestamps. |

:class:`~empyrean.Epochs` carries the scale as a `StringAttribute` and
exposes `.to_tdb()` / `.to_utc()` / `.to_scale(...)` for conversion.
UTC↔TDB conversion (leap seconds plus the TDB−TT periodic terms) is
built into the engine — no separate leap-second kernel is installed.

Earth-rotation kernels are stored on the TT scale internally; the
TT ↔ TDB difference is small — a quasi-periodic term with peak
amplitude ~1.7 ms (dominated by an annual component) — and is applied
automatically.

## Reference frames

The {class}`~empyrean.Frame` enum (use the all-caps Python member
name; the string slug shown in the second column also works):

| Member            | String slug      | Notes                                                                  |
|-------------------|------------------|------------------------------------------------------------------------|
| `ECLIPTICJ2000`   | `"eclipticj2000"`| Mean ecliptic and equinox of J2000.0. **Integration frame; default propagation output.** Convert to ICRF via {func}`~empyrean.transform_coordinates(..., frame="icrf")` when downstream tools (Horizons, most observatory pipelines) expect equatorial. |
| `ICRF`            | `"icrf"`         | International Celestial Reference Frame (≈ J2000 equatorial).         |
| `ITRF93`          | `"itrf93"`       | Earth-fixed (rotating) — for ground-station vectors.                   |

Conversion is via {func}`~empyrean.transform_coordinates`, which
accepts a target frame as a kwarg and propagates covariance through
the rotation.

## Origins

The {class}`~empyrean.Origin` enum is keyed by NAIF body code; only
the bodies whose ephemeris is actually loaded by `ForceModelTier`
≥ Approximate are exposed:

| Member               | String slug          | Use                                                              |
|----------------------|----------------------|------------------------------------------------------------------|
| `SSB`                | `"SSB"`              | Solar System Barycenter (NAIF 0) — the propagation origin.       |
| `SUN`                | `"Sun"`              | NAIF 10. Heliocentric — what SBDB returns for small-body orbits. |
| `EARTH`              | `"Earth"`            | NAIF 399. Geocentric — for Earth-observer ephemeris.             |
| `MOON`               | `"Moon"`             | NAIF 301. Selenocentric.                                         |
| `MERCURY`            | `"Mercury"`          | NAIF 199.                                                        |
| `VENUS`              | `"Venus"`            | NAIF 299.                                                        |
| `MARS_BARYCENTER`    | `"Mars Barycenter"`  | NAIF 4. Mars *body-center* not exposed (DE440 ships the barycentre only). |
| `JUPITER_BARYCENTER` | `"Jupiter Barycenter"` | NAIF 5.                                                        |
| `SATURN_BARYCENTER`  | `"Saturn Barycenter"`  | NAIF 6.                                                        |
| `URANUS_BARYCENTER`  | `"Uranus Barycenter"`  | NAIF 7.                                                        |
| `NEPTUNE_BARYCENTER` | `"Neptune Barycenter"` | NAIF 8.                                                        |
| `PLUTO_BARYCENTER`   | `"Pluto Barycenter"`   | NAIF 9. Pluto body-center is not exposed; same DE440 reason as Mars. |
| `ICRF`               | `"icrf"`             | A frame, not a body, but accepted in the same parameter slot for API consistency. |
| `Origin.asteroid(n)` | `"asteroid_<n>"`     | Numbered-asteroid origin — e.g. `Origin.asteroid(433)` for Eros. Useful when fitting an asteroid that the force model would otherwise include as a perturber: pass it via `excluded_perturbers` so it does not self-attract. |

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
| Epoch                 | MJD (TDB by default)       | See above on time scales.              |
| Angle                 | degrees                    | At the boundary; converted to radians internally. |
| Astrometric residuals | arcseconds                 | RA·cos(δ), Dec, AT/CT decompositions.   |
| Track position angle  | degrees (East of North)    | On {class}`~empyrean.ObservationResults`. |
| Astrometric magnitudes | V-band                    | H, V, mag are V-equivalent unless documented otherwise. |
| Impact-probability    | dimensionless probability  | `[0.0, 1.0]`.                          |
| B-plane coordinates   | km                         | `b_dot_t_km`, `b_dot_r_km`, etc.       |
| B-plane covariance    | km²                        | `cov_tt_km2`, `cov_tr_km2`, `cov_rr_km2`. |
| Hyperbolic excess velocity | km / s                | `v_inf_km_s`.                          |
| Earth-rotation kernels | TT internally             | Conversion is automatic; users see TDB. |

When a function takes a `Frame` / `Origin` parameter, you can pass
either the enum value (`Frame.ICRF`) or the string slug
(`"icrf"` / `"eclipticj2000"` — case-insensitive). Same for `Origin`
which also accepts a raw NAIF integer.

## Coordinate representations

The {class}`~empyrean.CartesianCoordinates` /
{class}`~empyrean.KeplerianCoordinates` /
{class}`~empyrean.CometaryCoordinates` /
{class}`~empyrean.SphericalCoordinates` family supports four
representations on input and output:

| Representation | Elements                              | Notes                                    |
|----------------|---------------------------------------|------------------------------------------|
| Cartesian      | `(x, y, z, vx, vy, vz)` in AU, AU/day | The integration representation.          |
| Keplerian      | `(a, e, i, raan, ap, ma)`             | Mean anomaly. Angles in degrees.         |
| Cometary       | `(q, e, i, raan, ap, tp)`             | Time-of-perihelion. **What SBDB returns for small bodies.** |
| Spherical      | `(r, lon, lat, vr, vlon, vlat)`       | Topocentric or geocentric astrometry.    |

The propagator accepts any of the four; covariance gets transformed
along with the elements via the analytic Jacobians.
