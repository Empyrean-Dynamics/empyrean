<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean">

# empyrean
Safe Rust wrapper over libempyrean — uncertainty-first orbit propagation, ephemeris, orbit determination, and event detection for asteroids and comets, powered by automatic differentiation

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean"><img src="https://img.shields.io/crates/v/empyrean.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
<a href="https://docs.rs/empyrean"><img src="https://img.shields.io/docsrs/empyrean?style=flat-square&label=docs.rs" alt="docs.rs"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
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
empyrean = "0.7"
```

## What it does

- **Propagation** — N-body with general relativity, Earth J2–J4, 16 asteroid perturbers, Marsden non-gravitational model. Adaptive 15th-order Gauss-Radau integrator.
- **Uncertainty** — First-order (Jet1) state transition matrices; second-order (Jet2) state transition tensors.
- **Ephemeris** — RA/Dec, rates, photometry (H–G, H–G₁G₂, H–G₁₂), light time, phase angle, solar elongation, local horizon.
- **Orbit determination** — Herget IOD → N-body differential correction with STM caching and outlier rejection. Validated against `find_orb` and JPL SBDB.
- **Events** — Close approach, periapsis, SOI entry/exit, shadow, atmospheric entry, possible impact.

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

`determine` runs a full Herget IOD → N-body differential correction. The
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

This crate links against `libempyrean.{dylib,so,dll}`, which is
distributed separately as a binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. `cargo install empyrean-cli`
will build the binary against the cdylib and expects to find it on
the system library path.

The full distribution surface (Python wheel, CLI binary, C SDK, this
Rust crate) lives at the
[main repository](https://github.com/Empyrean-Dynamics/empyrean) —
see its README for installation paths and the cross-channel quickstart.

## Accuracy

Validated against JPL Horizons, ASSIST (reboundx), and `find_orb` on
43 objects across 13 dynamical populations (NEOs, MBAs, Trojans, TNOs,
comets, and more). Sub-meter propagation accuracy on bounded timescales.

## No guarantee of accuracy

empyrean performs numerical computations used in planetary-science and
mission-planning contexts. Outputs should not be used as the sole basis
for any decision — including but not limited to impact monitoring,
mission planning, collision avoidance, or navigation — without
independent verification. See the LICENSE file for the full terms.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime it links against is governed by a separate proprietary
binary license; see the main repository for the dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.

## Links

- Website: https://www.empyrean-dynamics.com
- Repository: https://github.com/Empyrean-Dynamics/empyrean
- Issues: https://github.com/Empyrean-Dynamics/empyrean/issues
