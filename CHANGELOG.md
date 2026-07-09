# Changelog

Notable changes to the empyrean distribution — the `empyrean`, `empyrean-sys`,
`empyrean-c`, and `empyrean-cli` crates and the `empyrean` Python package. This
project adheres to [Semantic Versioning](https://semver.org).

## [0.8.0] — 2026-07-09

Continuous-thrust inputs, a reusable force-model handle, one abi3 wheel per
architecture across four platforms — and two covariance-accuracy fixes found
by validating the release candidate, neither of which ever shipped in a
stable release with wrong behavior.

### Added

- **Continuous-thrust inputs** across every channel: finite-burn arcs with
  constant-RTN, velocity-tangent, and inertial-fixed steering laws, plus
  Δv-targeting corrections whose covariances flow into the tagged
  per-epoch covariance. Reaches the dynamics through propagation, ephemeris
  generation, observation planning, impact analysis, and I/O.
- **A reusable system handle**: `build_system` assembles the force model
  once and reuses it across propagation and ephemeris calls — thread-safe,
  with a `describe()` provenance record carrying the force-model
  configuration and the SHA-256 identity of every loaded kernel. Every call
  is validated against the handle's data and frozen key, erroring by axis
  on any mismatch rather than silently rebuilding.
- **Sigma-point provenance**: covariances produced by the sigma-point
  method are now tagged `sigma_point` in the per-epoch tagged-covariance
  readback (previously they read back as `linear`).
- **Explicit kernel-load error categories**: I/O versus parse failures no
  longer collapse into one variant.

### Fixed

- **B-plane uncertainty for element-space orbits.** `compute_b_planes`
  projected the input covariance in its native representation through the
  Cartesian state-transition matrix, skipping the element→Cartesian
  Jacobian — for Cometary/Keplerian/Spherical inputs (the SBDB-query
  common case) the projected 3σ ellipse inflated by orders of magnitude.
  The projection now uses the propagated Cartesian covariance at each
  close-approach epoch. Cartesian inputs, impact probabilities, and
  propagation covariances were never affected.
- **Sigma-point covariance normalization.** The sigma-point method now
  uses the canonical 2N+1 unscented construction; the previous sampling
  under-estimated recovered covariances by a factor of ~6. Degenerate
  input covariances and non-default legacy sampling parameters now fail
  loudly instead of silently degrading.
- **Observatory-code validation.** A 4-character observatory code is now a
  loud error at every input boundary instead of being silently truncated
  to a 3-character prefix that names a different observatory. (4-character
  MPC codes are not yet supported end-to-end.)

### Changed

- **Wheels are abi3.** One `cp310-abi3` wheel per architecture, installing
  on CPython 3.10+.
- **Four platforms.** Prebuilt engine, wheels, and CLI for macOS arm64,
  macOS x86_64, Linux x86_64, and Linux aarch64; the macOS x86_64
  artifacts are cross-compiled on arm64 runners.
- **Documented ordering guarantees.** Propagation states are epoch-ordered
  (forward ascending, then backward descending) — join on `epoch_mjd_tdb`;
  ephemeris entries are orbit-major with within-orbit observer-input
  order. Both are now stated in the API docs at every layer, along with
  `mag_sigma` population conditions and the observation Jacobian's
  light-time caveat.

## [0.7.0] — 2026-07-03

First stable release of the empyrean distribution: uncertainty-first orbit
propagation, ephemeris generation, orbit determination, and close-approach /
impact analysis for asteroids and comets, powered by automatic
differentiation. Distributed as a Rust crate (`empyrean`), a C ABI
(libempyrean), a Python package (`empyrean` on PyPI), and a command-line tool
over a consistent API. Includes all fixes from the 0.7.0 release candidates
below.

### Added

- **Propagation & events.** N-body propagation with non-gravitational forces,
  GR15 and DOP853 integrators, and event detection: close approaches, B-plane
  geometry, and impact-probability estimation across multiple uncertainty
  methods.
- **Uncertainty on every published quantity.** Linear (first-order),
  second-order, and adaptive uncertainty mapping via automatic
  differentiation, with per-epoch tagged covariances.
- **Orbit determination** via `determine` / `evaluate` / `refine`: initial
  orbit determination through N-body differential correction with outlier
  rejection, optical and radar astrometry, and non-gravitational parameter
  recovery. Fitted orbits carry state, covariance, and non-gravitational
  parameters for direct re-use in propagation and further fitting.
- **Ephemeris generation** for ground-based observers with sky-plane
  uncertainties.
- **Data provisioning.** `download_data` fetches the complete kernel set into
  a local data directory (idempotent — only missing files are downloaded); in
  Python, installed B612 Foundation data packages are staged from the wheels
  with no network access and only the remainder is fetched.

## [0.7.0-rc.4] — 2026-06-25

### Fixed

- **Concurrent context construction no longer races.** Native context
  construction (`empyrean_context_from_data_dir` / `_new_minimal` / `_with_spk`)
  is now serialized by a process-global lock **inside libempyrean (the C ABI)**,
  so construction is thread-safe for every consumer — the Rust wrapper, the
  Python package, the CLI, and direct C SDK users. The engine's first-init
  kernel provisioning does writable-cache I/O that raced when several contexts
  were built at once, surfacing as a path-less `I/O error: … (os error 2)`.
  Concurrent *use* of a built context (propagation, ephemeris, OD) is unaffected
  and stays unserialized — no hot-path or single-threaded regression.

- **`download_data` actually provisions the data directory.** It was a no-op that
  returned a path without fetching anything. It now downloads the complete
  Standard-tier kernel set into the target (or default) directory — idempotent
  (files already present are kept; only missing files are downloaded) — and
  returns the resolved directory, so a subsequent `Context::from_data_dir` loads
  with no further downloads. In Python, installed B612 Foundation data packages
  (`naif-de440`, `jpl-small-bodies-de441-n16`, `naif-eop-*`, `mpc-obscodes`) are
  staged from the wheels with no network access and only the remainder is fetched.

- **Actionable missing-data errors.** A failed `from_data_dir` now names the
  missing kernel and the data directory and hints the remedy (run `download_data`
  or set `EMPYREAN_DATA_DIR`), instead of a path-less message. The doubled
  `I/O error: I/O error:` wrapping is collapsed to a single prefix.

Earlier release candidates (rc.0–rc.3) are documented in their tagged GitHub
releases.

[0.7.0]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.7.0
[0.7.0-rc.4]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.7.0-rc.4
