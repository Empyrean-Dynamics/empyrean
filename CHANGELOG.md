# Changelog

Notable changes to the empyrean distribution — the `empyrean`, `empyrean-sys`,
`empyrean-c`, and `empyrean-cli` crates and the `empyrean` Python package. This
project adheres to [Semantic Versioning](https://semver.org).

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
