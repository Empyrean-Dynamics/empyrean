<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean">

# empyrean
Safe Rust wrapper over libempyrean — uncertainty-first orbit propagation, ephemeris, orbit determination, and event detection for asteroids and comets, powered by automatic differentiation

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean"><img src="https://img.shields.io/crates/v/empyrean.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
<a href="https://docs.rs/empyrean"><img src="https://img.shields.io/docsrs/empyrean?style=flat-square&label=docs.rs" alt="docs.rs"></a>
<br>
<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://doi.org/10.5281/zenodo.21318471"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21318471-blue?style=flat-square" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

The idiomatic Rust API over the `libempyrean` C ABI. Every C function
exposed in the cdylib has a typed, `Result<_, Error>`-returning wrapper
here. RAII handles the underlying allocations so callers never juggle
raw FFI pointers.

```toml
[dependencies]
empyrean = "0.8"
```

## What it does

- **Propagation** — N-body (Sun, planets, Moon, Pluto) with EIH general relativity, Sun J2 and Earth J2–J4 zonal harmonics, 16 asteroid perturbers, and the Marsden non-gravitational model — selectable across Approximate / Basic / Standard force-model tiers (Standard is the default). GR15 and DOP853 integrators. Optional finite-burn thrust arcs — constant-RTN, velocity-tangent, or inertial-fixed steering, with per-arc Δv targeting corrections — layer on as a continuous-thrust force input.
- **Uncertainty** — First-order (Jet1) state transition matrices; second-order (Jet2) state transition tensors; unscented sigma-point and Monte Carlo sampling; an adaptive Auto mode that escalates the method automatically through close approaches and relaxes it elsewhere. Optional per-epoch tagged-covariance readback.
- **Ephemeris** — RA/Dec, rates, photometry (H–G, H–G₁G₂, H–G₁₂), light time, phase angle, solar elongation, local horizon.
- **Orbit determination** — Gauss, Herget, and systematic-ranging (admissible region + Manifold of Variations) IOD → N-body differential correction over optical and radar (delay / Doppler) observations, with STM caching and outlier rejection. Validated against `find_orb` and JPL SBDB.
- **Events** — Close approach (start/end), periapsis, gravitational capture (start/end), shadow entry/exit, atmospheric entry/exit, impact, and possible impact.

## Quick start

```rust,no_run
use empyrean::{Context, Epoch, PropagationConfig};

let ctx = Context::from_data_dir(None)?;

// Query SBDB for Apophis and propagate through its 2029 Earth flyby.
let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let result = ctx.propagate(&orbits, &epochs, &PropagationConfig::default())?;

println!("{} states, {} events", result.states.len(), result.events.len());
# Ok::<(), empyrean::Error>(())
```

## Orbit determination

`determine` runs a full IOD (Gauss / Herget / systematic ranging) → N-body differential correction. The
fitted `result.orbit` is a re-feedable [`Orbit`] carrying state, covariance,
and any fitted non-gravitational parameters — pass it straight back into
`propagate`, `generate_ephemeris`, or `compute_impact_probabilities`.

```rust,no_run
# use empyrean::{Context, ODConfig};
# let ctx = Context::from_data_dir(None)?;
let obs = ctx.read_ades("observations.psv")?;   // optical + radar
let result = ctx.determine(&obs, None, &ODConfig::default())?;

println!(
    "converged={}, RMS = {:.2}\" RA / {:.2}\" Dec",
    result.converged,
    result.summary.rms_ra_arcsec,
    result.summary.rms_dec_arcsec,
);
# Ok::<(), empyrean::Error>(())
```

## Ephemeris

```rust,no_run
# use empyrean::{Context, EphemerisConfig, Epoch};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let observers = ctx.get_observers(&["W84", "F51"], &epochs)?;
let eph = ctx.generate_ephemeris(&orbits, &observers, &EphemerisConfig::default())?;

for entry in &eph.entries {
    println!("RA {:.4}°  Dec {:.4}°  V {:.2}", entry.ra_deg, entry.dec_deg, entry.mag);
}
# Ok::<(), empyrean::Error>(())
```

## Uncertainty

First-order (the default) propagates the covariance with the state-transition
matrix — accurate when the orbit is approximately linear over the uncertainty
region. Second-order adds the state-transition tensor for the curvature that
linear covariance misses near a close approach.

```rust,no_run
# use empyrean::{Context, Epoch, PropagationConfig, UncertaintyMethod};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
# let epochs = vec![Epoch::from_mjd_tdb(65000.0)];
let config = PropagationConfig {
    uncertainty_method: UncertaintyMethod::SecondOrder,
    ..Default::default()
};
let result = ctx.propagate(&orbits, &epochs, &config)?;
# Ok::<(), empyrean::Error>(())
```

## Continuous thrust

Model finite burns / low-thrust arcs by attaching a `ThrustParams` to an
orbit before propagation. Each `ThrustArc` carries its own thrust, mass,
specific impulse, steering law (constant-RTN, velocity-tangent, or
inertial-fixed), and central body; the burn perturbs the trajectory
through the same differentiated dynamics as gravity and the
non-gravitational forces.

```rust,no_run
use empyrean::{Context, Epoch, Origin, PropagationConfig, SteeringLaw, ThrustArc, ThrustParams};

let ctx = Context::from_data_dir(None)?;
let orbit = empyrean::query_sbdb(&["Apophis"], None)?.orbits.remove(0);

// One finite burn: 1 N over MJD 65000–65010 on a 500 kg spacecraft,
// mass depleting at Isp = 3000 s, steered at constant RTN angles
// relative to the Sun. `sharpness` sets the tanh on/off transition.
let arc = ThrustArc::new(
    65000.0,                                                   // start_mjd_tdb
    65010.0,                                                   // end_mjd_tdb
    1.0,                                                       // thrust_n (N)
    500.0,                                                     // mass_kg
    100.0,                                                     // sharpness (1/day)
    SteeringLaw::ConstantRTN { alpha_rad: 0.0, beta_rad: 0.0 },
    Origin::SUN,                                               // RTN frame reference
)
.with_isp(Some(3000.0));

// Attach to the orbit and propagate. Add per-arc Δv targeting
// corrections with `ThrustParams::new(arcs).with_dv_corrections(..)`.
let orbit = orbit.with_thrust(Some(ThrustParams::new(vec![arc])));
let epochs = vec![Epoch::from_mjd_tdb(65020.0)];
let result = ctx.propagate(&[orbit], &epochs, &PropagationConfig::default())?;
println!("{} states", result.states.len());
# Ok::<(), empyrean::Error>(())
```

## System handles

Assembling the force model (planets, Moon, asteroid perturbers,
harmonics, relativistic corrections) has a fixed per-call cost. A
[`BuiltSystem`] assembles it once for a frozen `{force model, frame,
encounter-timescale divisor}` key and reuses it across many
propagations — the build-once, propagate-many pattern for
short-arc campaigns. It is `Send + Sync`, so `&handle` can be shared
across threads. A call whose config disagrees with the frozen key, or
that pairs the handle with a different data instance, is rejected
loudly by axis — never silently rebuilt against the wrong dynamics.

```rust,no_run
# use empyrean::{Context, ForceModelTier, Frame, PropagationConfig, Epoch};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
// Build once; freeze the divisor at the engine default (0.0).
let handle = ctx.built_system(ForceModelTier::Standard, Frame::EclipticJ2000, 0.0)?;

let epochs = vec![Epoch::from_mjd_tdb(65020.0)];
let result = handle.propagate(&ctx, &orbits, &epochs, &PropagationConfig::default())?;
println!("{} states", result.states.len());

// describe() reports the reproducibility record: the force-model menu
// plus the identity (SHA-256) of every loaded kernel.
let desc = handle.describe()?;
println!("{} perturbers, {} kernels", desc.perturber_origins.len(), desc.kernels.len());
# Ok::<(), empyrean::Error>(())
```

## Impact probability and B-plane geometry

For each detected close approach you can ask for an impact-probability
assessment or a full B-plane breakdown, and run several uncertainty methods
side-by-side on the same encounter. Each returns one record per
(method × orbit × body), tagged with its method and closest-approach epoch.

```rust,no_run
# use empyrean::{Context, Epoch, UncertaintyMethod, Origin};
# let ctx = Context::from_data_dir(None)?;
# let orbits = empyrean::query_sbdb(&["Apophis"], None)?.orbits;
let end = Epoch::from_mjd_tdb(65000.0);

let ips = ctx.compute_impact_probabilities(
    &orbits,
    end,
    &[UncertaintyMethod::FirstOrder, UncertaintyMethod::SecondOrder],
    &[Origin::EARTH, Origin::MOON],
)?;
for ip in &ips {
    println!("{:?}: miss {:.0} km", ip.body, ip.miss_distance_km);
}

let bps = ctx.compute_b_planes(&orbits, end, &[UncertaintyMethod::SecondOrder], &[Origin::EARTH])?;
for bp in &bps {
    println!("B·T {:.1} km, B·R {:.1} km", bp.b_dot_t_km, bp.b_dot_r_km);
}
# Ok::<(), empyrean::Error>(())
```

## Runtime requirement

This crate (via empyrean-sys) loads `libempyrean.{dylib,so}` at
run time, which is distributed separately as a binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. The path is resolved from the
`EMPYREAN_LIB` environment variable if set, else a `libempyrean.*`
sitting next to the loaded module, else a build-time location — an
`EMPYREAN_LIB_DIR` override, a sibling `../target/release` build, or
a checksum-pinned prebuilt downloaded from the GitHub release (in
that order); no system library path setup is required.

Prebuilt engine binaries are currently published for four targets —
macOS arm64 (`macos-aarch64`), macOS x86_64 (`macos-x86_64`), Linux
x86_64 (`linux-x86_64`), and Linux aarch64 (`linux-aarch64`); on other
targets the build stops with an error unless `EMPYREAN_LIB_DIR` points
at an engine build.

The full distribution surface (Python wheel, CLI binary, C SDK, this
Rust crate) lives at the
[main repository](https://github.com/Empyrean-Dynamics/empyrean) —
see its README for installation paths and the cross-channel quickstart.

## Accuracy

Validated against JPL Horizons, ASSIST (reboundx), and `find_orb` on
43 objects across 13 dynamical populations (NEOs, MBAs, Trojans, TNOs,
comets, and more). Sub-meter propagation accuracy on bounded timescales;
see the [validation notes](https://github.com/Empyrean-Dynamics/empyrean#validation)
in the main repository for the comparison setup.

## No guarantee of accuracy

empyrean performs numerical computations used in planetary-science and
mission-planning contexts. Outputs should not be used as the sole basis
for any decision — including but not limited to impact monitoring,
mission planning, collision avoidance, or navigation — without
independent verification. See the LICENSE file for the full terms.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime it loads at runtime is governed by a separate proprietary
binary license; see the main repository for the dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.

## Links

- Website: https://www.empyrean-dynamics.com
- Repository: https://github.com/Empyrean-Dynamics/empyrean
- Issues: https://github.com/Empyrean-Dynamics/empyrean/issues
