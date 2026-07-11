<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean">

# empyrean
Uncertainty-first orbit propagation, ephemeris, orbit determination, and event detection for asteroids and comets, powered by automatic differentiation

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://pypi.org/project/empyrean/"><img src="https://img.shields.io/pypi/v/empyrean.svg?style=flat-square&label=PyPI" alt="PyPI"></a>
<a href="https://pypi.org/project/empyrean/"><img src="https://img.shields.io/pypi/pyversions/empyrean.svg?style=flat-square&label=python" alt="Python versions"></a>

<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://zenodo.org/badge/latestdoi/1278090652"><img src="https://zenodo.org/badge/1278090652.svg" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

```bash
pip install empyrean
```

A plain install pulls empyrean together with the B612 Foundation's
pre-packaged SPICE kernels (~740 MB — see the table below). After
installation, the first call to `empyrean.initialize()` downloads a
small remainder (the
`moon_pa` Moon-orientation kernel and the `bias.dat` star-catalog
debiasing table — about 50 MB) that isn't available on PyPI.

Wheels are published for CPython >= 3.10 as a single abi3 stable-ABI
wheel per architecture — one wheel covers CPython 3.10 and every newer
version — across four platforms: macOS arm64, macOS x86_64,
manylinux_2_28 x86_64, and manylinux_2_28 aarch64. There is no source
distribution, so `pip install empyrean` on other platforms will not
resolve — use the
[other distribution channels](https://github.com/Empyrean-Dynamics/empyrean#install)
in the meantime.

## What it does

- **Propagation** — N-body (Sun, planets, Moon, Pluto) with EIH general relativity, Sun J2 and Earth J2–J4 zonal harmonics, 16 asteroid perturbers, and the Marsden non-gravitational model — selectable across Approximate / Basic / Standard force-model tiers (Standard is the default). GR15 and DOP853 integrators. Optional finite-burn thrust arcs — constant-RTN, velocity-tangent, or inertial-fixed steering, with per-arc Δv targeting corrections — layer on as a continuous-thrust force input.
- **Uncertainty** — First-order (Jet1) state transition matrices; second-order (Jet2) state transition tensors; unscented sigma-point and Monte Carlo sampling; an adaptive Auto mode that escalates the method automatically through close approaches and relaxes it elsewhere. Optional per-epoch tagged-covariance readback.
- **Ephemeris** — RA/Dec, rates, photometry (H–G, H–G₁G₂, H–G₁₂), light time, phase angle, solar elongation, local horizon.
- **Orbit determination** — Gauss, Herget, and systematic-ranging (admissible region + Manifold of Variations) IOD → N-body differential correction over optical and radar (delay / Doppler) observations, with STM caching and outlier rejection. Validated against `find_orb` and JPL SBDB.
- **Events** — Close approach (start/end), periapsis, gravitational capture (start/end), shadow entry/exit, atmospheric entry/exit, impact, and possible impact.

## Quick start

```python
import empyrean
from empyrean import Epochs, TimeScale

empyrean.download_data()   # SPICE kernels, first run only
empyrean.initialize()

# Query SBDB for Apophis and propagate through its 2029 Earth flyby
orbits = empyrean.query_sbdb(["Apophis"])
epochs = Epochs.from_kwargs(mjd=[65000.0], scale=TimeScale.TDB)
result = empyrean.propagate(orbits, epochs)

# Event timeline
for i in range(len(result.events.summary)):
    ev = result.events.summary
    print(f"{ev.event_type.to_pylist()[i]:25s} "
          f"{ev.body.to_pylist()[i]:8s} "
          f"MJD {ev.epoch.to_numpy()[i]:.2f}")
```

## Orbit determination

```python
obs, radar = empyrean.read_ades("observations.psv")   # (optical, radar)
result = empyrean.determine(obs)                       # one fit per call
print(
    f"converged={result.converged}, "
    f"RMS={result.summary.rms_ra_arcsec:.2f}\" RA / "
    f"{result.summary.rms_dec_arcsec:.2f}\" Dec"
)
```

## Ephemeris

```python
observers = empyrean.get_observer_states(["W84", "F51"], epochs)
eph = empyrean.generate_ephemeris(orbits, observers)

print(eph.ephemeris.coordinates.lon.to_numpy())   # RA (degrees)
print(eph.ephemeris.coordinates.lat.to_numpy())   # Dec (degrees)
print(eph.ephemeris.mag.to_numpy())               # apparent V magnitude
```

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
# identity (SHA-256) of every loaded kernel.
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

ips = empyrean.compute_impact_probabilities(
    orbits,
    end_epoch=63000.0,
    methods=[UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER],
)
ips.epochs.scale                    # "tdb"
ips.where(pc.field("method") == "second_order").ip_second_order.to_numpy()
ips.ip_linear.to_numpy()            # always populated

bps = empyrean.compute_b_planes(orbits, 63000.0, [UncertaintyMethod.SECOND_ORDER])
print(bps.b_dot_t_km.to_numpy())    # B·T (km)
print(bps.b_dot_r_km.to_numpy())    # B·R (km)
print(bps.semi_major_3sig_km.to_numpy())  # 3σ ellipse semi-major
```

Returns typed `ImpactProbabilities` and `BPlanes` quivr tables — one
row per (method × orbit × body) encounter, with the closest-approach
time as an embedded `Epochs` sub-table so `.to_utc()` / `.to_tdb()`
just works.

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
| `bias.dat` | 35 MB | first `initialize()` | Star-catalog debiasing table (Eggl et al. 2020) |
| `jwst_rec.bsp` | 121 MB | on demand, for JWST observers | NAIF — JWST ephemeris |

Any of these can be relocated with `EMPYREAN_DATA_DIR`, and individual
files can be preset with `VILLENEUVE_*_PATH` environment variables.

## Accuracy

Validated against JPL Horizons, ASSIST (reboundx), and `find_orb` on
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
