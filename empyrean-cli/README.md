<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean-cli">

# empyrean-cli
Command-line interface for empyrean — orbit propagation, ephemeris generation, orbit determination, and event detection

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean-cli"><img src="https://img.shields.io/crates/v/empyrean-cli.svg?style=flat-square&label=crates.io" alt="crates.io"></a>

<a href="Cargo.toml"><img src="https://img.shields.io/badge/rustc-1.90%2B-orange?style=flat-square&logo=rust" alt="MSRV 1.90"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BSD"><img src="https://img.shields.io/badge/source-BSD--3--Clause-blue.svg?style=flat-square" alt="Source license"></a>
<a href="https://github.com/Empyrean-Dynamics/empyrean/blob/main/LICENSE-BINARY"><img src="https://img.shields.io/badge/binary-proprietary-lightgrey.svg?style=flat-square" alt="Binary license"></a>
<a href="https://zenodo.org/badge/latestdoi/1278090652"><img src="https://zenodo.org/badge/1278090652.svg" alt="DOI"></a>
<br>
<a href="https://claude.ai"><img src="https://img.shields.io/badge/Built%20with-Claude%20Code-D97757?logo=anthropic&logoColor=white&style=flat-square" alt="Built with Claude Code"></a>
<a href="https://www.empyrean-dynamics.com"><img src="https://img.shields.io/badge/Website-empyrean--dynamics.com-1a1a2e?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4MT0iMiIgeTE9IjEyIiB4Mj0iMjIiIHkyPSIxMiIvPjxwYXRoIGQ9Ik0xMiAyYTE1LjMgMTUuMyAwIDAgMSA0IDEwIDE1LjMgMTUuMyAwIDAgMS00IDEwIDE1LjMgMTUuMyAwIDAgMS00LTEwIDE1LjMgMTUuMyAwIDAgMSA0LTEweiIvPjwvc3ZnPg==&logoColor=white&style=flat-square" alt="Website"></a>
<a href="https://github.com/Empyrean-Dynamics"><img src="https://img.shields.io/badge/GitHub-Empyrean--Dynamics-1a1a2e?logo=github&logoColor=white&style=flat-square" alt="GitHub"></a>

---

empyrean-cli is the command-line interface to empyrean. It publishes
one binary — `empyrean` — that runs every headline pipeline (orbit
propagation, ephemeris generation, orbit determination, and event
detection) and emits Parquet output you can join in pandas / Polars /
DuckDB.

## Install

```sh
cargo install empyrean-cli
```

`cargo install` fetches the closed-source `libempyrean` engine
automatically (a checksum-pinned download at build time). Prebuilt
engine binaries exist for four targets — macOS arm64 (`macos-aarch64`),
macOS x86_64 (`macos-x86_64`), Linux x86_64 (`linux-x86_64`), and Linux
aarch64 (`linux-aarch64`); other targets are not yet supported.

Alternatively, grab a pre-built binary for your platform from
[GitHub Releases](https://github.com/Empyrean-Dynamics/empyrean/releases).
The installed binary is named `empyrean`. The release tarball
(`empyrean-<target>.tar.gz`) contains the binary + LICENSE only — also
download the matching `libempyrean-<target>.tar.gz` and either place
the shared library next to the binary or point `EMPYREAN_LIB` at it:

```sh
tar xzf empyrean-macos-aarch64.tar.gz
tar xzf libempyrean-macos-aarch64.tar.gz
export EMPYREAN_LIB=$PWD/libempyrean.dylib   # or place it next to `empyrean`
./empyrean version
```

## Quickstart

```sh
# One-time: download SPICE kernels into the platform data directory
# (~/.local/share/empyrean/data/ on Linux, ~/Library/Application Support/empyrean/data/
# on macOS; honors EMPYREAN_DATA_DIR).
empyrean init

# Propagate Apophis 10 years past its SBDB epoch.
empyrean propagate --object-id 99942 --epoch 64922.0 --out-dir ./out

# Generate an ephemeris from the same orbit at observatory 568.
empyrean ephemeris --object-id 99942 --observers 568 --epoch 64922.0 --out-dir ./out

# Fit an orbit from an ADES PSV.
empyrean determine apophis.psv --out-dir ./out

# Confirm the build provenance — every binary carries the `<tag>+<sha>`
# strings of the villeneuve / scott / nolan commits it was built against.
empyrean version
```

All commands emit Parquet tables under `--out-dir` by default
(`--format json` / `--format csv` are also available). The schemas
match the Python and Rust API outputs exactly — same `orbit_id` /
`object_id` join keys, same time scales, same physical units — so you
can mix-and-match channels for the same workflow.

Beyond the headline pipelines: `propagate` takes `--uncertainty-method`
(`first-order` / `second-order` / `sigma-point` / `monte-carlo` /
`auto`) and `--tagged-covariance`; `empyrean query horizons-vectors`
fetches JPL Horizons state vectors; `empyrean cache info` / `cache clear`
manage the API response cache; and `empyrean serve` / `empyrean stop`
run a daemon that keeps the loaded kernels in memory for faster
subsequent commands. See `empyrean <command> --help` for the full
flag surface.

## Continuous thrust

`propagate` accepts `--thrust-arcs <FILE>`, a JSON file describing
finite-burn / low-thrust arcs. One file describes one set of thrust
parameters, applied to every orbit in the batch. Supplying it runs the
propagation in-process (the daemon fast path is skipped) so the thrust
is never silently dropped.

```json
{
  "arcs": [
    {
      "start_mjd_tdb": 65000.0,
      "end_mjd_tdb":    65010.0,
      "thrust_n":       1.0,
      "mass_kg":        500.0,
      "isp_s":          3000.0,
      "steering":       { "type": "constant_rtn", "alpha_rad": 0.0, "beta_rad": 0.0 },
      "sharpness":      100.0,
      "central_body":   10
    }
  ],
  "dv_corrections":         [[0.0, 0.0, 0.0]],
  "correction_covariances": [[[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]]
}
```

- `isp_s` is optional — omit or `null` for constant mass; otherwise mass depletes over the burn.
- `steering.type` is `constant_rtn` (with `alpha_rad`, `beta_rad`), `velocity_tangent`, or `inertial_fixed` (with `direction`).
- `central_body` is a NAIF body code (10 = Sun, 399 = Earth, 301 = Moon) — the reference for the RTN / velocity-tangent frame.
- `dv_corrections` is positional with `arcs`; `correction_covariances`, when present, must match its length. A mismatch is rejected at propagation time, never silently repaired.

```sh
empyrean propagate --object-id 99942 --epoch 64922.0 --thrust-arcs burn.json --out-dir ./out
```

## Runtime requirement

The `empyrean` binary loads `libempyrean.{dylib,so}` at run time,
which is distributed separately as a binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. The path is resolved from the
`EMPYREAN_LIB` environment variable if set, else a `libempyrean.*`
sitting next to the binary, else a build-time location — an
`EMPYREAN_LIB_DIR` override, a sibling `../target/release` build, or
a checksum-pinned prebuilt downloaded from the GitHub release (in
that order); no system library path setup is needed.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime the binary loads at run time is governed by a separate
proprietary binary license; see the main repository for the
dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.
