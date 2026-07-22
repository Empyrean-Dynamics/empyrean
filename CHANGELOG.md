# Changelog

Notable changes to the empyrean distribution — the `empyrean`, `empyrean-sys`,
`empyrean-c`, and `empyrean-cli` crates and the `empyrean` Python package. This
project adheres to [Semantic Versioning](https://semver.org).

## [0.9.0] — 2026-07-21

The complete output surface: every value the engine computes now crosses
the C ABI (empyrean-core v0.9.2, villeneuve v1.20.2, scott v1.15.0).
C ABI version 2 — struct shapes grew by appending fields, so ABI-1
consumers must recompile against the version-2 header.

### Added

- **Ephemeris uncertainty outputs.** Each ephemeris row carries the 6×6
  sky-plane covariance over (ρ, RA, Dec) and their rates (AU / degree
  units), and the aberrated (light-time corrected) barycentric ICRF
  Cartesian state at the photon-emission epoch with its own 6×6
  covariance — populated whenever the input orbit carries a state
  covariance, never zero-filled. A generate call also returns run-level
  non-fatal **generation warnings** (e.g. an Earth-orientation kernel
  coverage gap handled by the analytic IAU 2006 fallback, or a row
  whose sensitivity chain was skipped), on both the one-shot and
  handle-based paths.
- **Per-observation OD diagnostics.** Residual rows carry radar
  delay / Doppler residuals (observed − predicted in seconds / hertz,
  with χ², degrees of freedom, survival probability, and combined
  variance), the D-optimality information loss on removal (+∞ marks an
  indispensable observation), and the along/cross-track covariance
  off-diagonal completing the 2×2.
- **Covariance trust verdict.** `determine` results carry an
  event-aware verdict on the delivered covariance: trusted;
  encounter-intervenes (naming the intervening close approach or
  high-nonlinearity crossing, and whether a second-order state-only
  correction can recover it); or weakly-determined for wider-than-state
  fits. Absence of a verdict is not trust — it means no gate ran.
- **Photometry drop report.** Magnitudes whose band has no adopted
  V-band conversion are excluded from the photometric fit, counted
  (`n_mags_dropped_unconvertible`), and their distinct band codes
  listed (`dropped_bands`); the observations' astrometry is unaffected.
  The band→V table itself gained the modern two-character ADES codes
  (Johnson-Cousins, Sloan, Pan-STARRS, LSST, ATLAS).
- **Impact-probability detail.** Each record carries the geodetic
  impact point (latitude / longitude / altitude on the body's reference
  ellipsoid, when a surface projection is available), the Monte-Carlo
  95% binomial confidence half-width, the second-order corrected mean
  miss distance with its 1σ uncertainty and skewness, the
  closest-approach distance gradient and 6×6 Hessian with respect to
  the initial state, and the adaptive Gaussian-mixture component count.
- **Plan evaluation outputs (C ABI).** Radar candidates report the
  effective SNR (linear power ratio), one-way range (km), measurement
  mode, and link-budget provenance notes; plan results carry the
  predicted optical ephemeris (epoch MJD TDB, RA, Dec per optical
  candidate).
- **Basis-tagged mixture components (C ABI).** Each Gaussian-mixture
  component carries the reference frame and center-body origin its mean
  and covariance are expressed in.
- **Output-integrity tests.** New finiteness contracts assert that
  analytic uncertainty outputs are populated with finite values — never
  silently all-NaN — alongside the expanded no-silent-drops guards.

### Changed

- `EMPYREAN_ABI_VERSION` is now **2**. Fields are only ever appended,
  never reordered or removed; recompile against the version-2 header.
- The aberrated-state rows in the Python ephemeris table are stamped at
  the photon-emission epoch (observation epoch − light time), matching
  the state they carry.
- Covariance sub-tables with mixed per-row presence now carry genuinely
  null rows where no covariance exists, rather than NaN-valued rows.

### Fixed

- `generate_ephemeris` shipped an all-NaN sky covariance in
  v0.9.0-rc.0 even when the input orbit carried a finite covariance —
  the marshaling dropped it at the C boundary. Fixed end-to-end across
  every channel.
- Session OD paths now explicitly zero every owned-pointer surface they
  do not populate, so freeing a session result never touches
  uninitialized caller memory.
- A clean ephemeris re-save into a reused directory removes a stale
  `warnings.json` instead of attributing old warnings to new data.

## [0.9.0-rc.0] — 2026-07-20

Wide-parameter orbit determination and post-OD photometry, at API parity
across every channel (empyrean-core v0.9.1, villeneuve v1.20.1, scott
v1.14.1).

### Added

- **Wide-parameter OD fitting.** `determine` / `refine` can solve a wider
  parameter vector than the state plus Marsden non-grav: the cometary
  outgassing time delay **DT**, SRP **area-to-mass**, and impulsive
  **thrust Δv** segments, each differentiated analytically by the
  hyperdual integrator. Requested through the wide solve-for surface on
  every channel. DT / AMRAT / thrust are refine-path solves — the input
  orbit carries the prior that opens the parameter, and a requested
  parameter with no prior errors loudly rather than returning a zeroed
  column.
- **SRP area-to-mass (AMRAT) force slot on the input orbit.** A first-class
  solar-radiation-pressure slot — additive with the Marsden non-grav — is
  now carried on every input path (the C `EmpyreanOrbit`'s `has_srp` /
  `srp_amrat` / `srp_cr` / `srp_amrat_variance`, the Rust wrapper's
  `Orbit.srp`, the Python `orbits.srp` `SRPParams` table, and the CLI's
  `--amrat` / `--cr` / `--amrat-variance`). SRP is never value-inferred —
  an explicit switch enables it — and a finite `amrat_variance` both opens
  and priors the AMRAT column in a `StateAndAMRAT` /
  `StateAndNonGravAndAMRAT` refine. A fitted orbit carries its absolute
  AMRAT (and fitted posterior variance) back out for a lossless re-feed.
- **SBDB queries carry SRP.** `query_sbdb` now populates `orbits.srp.amrat`
  from JPL's fitted area-to-mass, so an SBDB-sourced orbit round-trips its
  area-to-mass into propagation and re-feed. Previously the fitted SRP force
  was dropped (`srp.amrat` came back null) for the objects JPL fits an
  area-to-mass for — e.g. 101955 Bennu (2.636e-6 ± 1.908e-7 m²/kg).
- **Tagged solved covariance.** OD results carry a solved covariance whose
  parameter identities travel with the matrix, so a caller reads a fitted
  parameter's variance by its slot (DT, AMRAT, each thrust component)
  instead of guessing column order. Populated identically across the Rust,
  C, Python, and CLI channels.
- **Post-OD photometry.** An optional photometric fit recovers the
  absolute magnitude H and phase-function slope from the observation
  magnitudes once the orbit is solved, climbing a model ladder (H-only →
  HG12 → HG1G2) to the richest model the arc's phase-angle coverage
  supports, with an honest 1σ on H from its parameter covariance.

### Changed

- **Python `model='srp'` is rejected loudly.** The SRP force now lives on
  its own `orbits.srp` `SRPParams` table (area-to-mass + `Cr` + prior
  variance); `NonGravParams` is Marsden-only. Passing `model='srp'` (or a
  non-null `cr`) on `NonGravParams` now raises with a migration pointer
  rather than being silently reinterpreted as an inverse-square radial
  force — any prior `model='srp'` results were computed as Marsden-radial
  and are invalid.
- **C ABI grew (recompile required).** `EmpyreanOrbit` and `EmpyreanODResult`
  gained the SRP input / re-feed fields, so their `sizeof` changed; C
  consumers and `empyrean-sys` callers must recompile against the
  v0.9.0-rc.0 header (ABI version 1). `empyrean_abi_version()` reports 1.
- **Engine.** Binds empyrean-core v0.9.1 (villeneuve v1.20.1, scott
  v1.14.1), which shares one force-model system across every batch OD call.

## [0.8.2] — 2026-07-11

Engine release (empyrean-core v0.8.3, villeneuve v1.18.2, scott
v1.13.4). No API changes in any channel — every fix below arrives
through the same functions with the same signatures.

### Fixed

- **Backward propagation arcs from encounter epochs.** Propagating
  backward from an epoch inside a close encounter (the natural epoch
  for an impactor fit — e.g. 2008 TC3, determined hours before entry)
  produced a pre-epoch arc displaced by the encounter body's position
  (~1 au). Forward/backward legs and their seed accelerations are now
  frame-consistent throughout.
- **Captured objects no longer report per-revolution close
  approaches.** A temporarily captured object (a minimoon such as
  2020 CD3) emitted a "close approach" — and a meaningless linear
  impact probability — for every perigee of its bound orbit. Perigees
  during a capture are now reported as structure nested inside the
  capture event; close-approach and impact-probability records cover
  genuine flybys only.
- **Impact ground tracks end at the entry point.** The ground-track
  summary for an impacting trajectory previously reported the
  sub-surface minimum of a straight-line extrapolation (hundreds of
  kilometers underground and off-site); it now reports the impact's
  own surface coordinates.
- **Stop conditions truncate output at the trigger.** An opted-in stop
  (e.g. stop-at-impact) no longer emits states past the trigger epoch
  in either time direction.
- **Ephemeris validation gate restored.** The ephemeris-vs-reference
  acceptance test compares in consistent units again and is back in
  the release gate (the engine output itself was always correct).

### Added

- **Citable releases.** Every GitHub release is archived on Zenodo
  with a version DOI; `CITATION.cff` and the DOI badge ship with this
  release.

## [0.8.1] — 2026-07-10

### Fixed

- **Fitted non-grav covariance reaches every Python forward model.** The
  Python bindings silently dropped the non-grav 3×3 covariance from
  orbit-determination fits when marshaling into `propagate`,
  `generate_ephemeris`, `compute_impact_probabilities`, and
  `compute_b_planes` — understating propagated σ for non-grav-solved
  orbits (~2,800 km in quadrature at Apophis's 2029 close approach).
  The Rust channel always forwarded it; the two channels now agree.
- **Observing nights for western observatories.** MPC east-of-Greenwich
  longitudes are wrapped to signed values before the local-noon fold, so
  Chilean (and all western) nights are stamped with the local observing
  night instead of the UTC date (via villeneuve v1.18.1).
- **Observation sensitivities without an input covariance.** Requesting
  STM tracing now populates the observation Jacobians whether or not the
  orbit carries a covariance; only the projected sky covariance still
  requires one (via villeneuve v1.18.1).
- **macOS C-ABI tarball is linkable as shipped.** The released
  `libempyrean.dylib` now carries an `@rpath` install name instead of the
  build machine's absolute path; C consumers link with `-Wl,-rpath`.
  `dlopen`-based consumers (the Rust crate and the wheels) were never
  affected.

### Changed

- **Propagation output is in ascending epoch order, always** (villeneuve
  v1.18.0): within each orbit, rows come back chronologically regardless
  of request order, so positional pairing against an ascending,
  duplicate-free request grid is exact. Previously rows were emitted
  forward-then-backward with non-chronological blocks possible around
  encounters. Ephemeris entries keep their (deliberately different)
  observer-input order — now also an engine-tested guarantee.
- **Input-marshal drop-proofing.** All Python-extension orbit builders
  route through a single exhaustive construction site, so future orbit
  fields cannot be silently dropped at the FFI boundary, and the
  no-silent-drops contract suite now exercises the non-grav covariance
  input channel end to end.
- **Engine.** Binds empyrean-core v0.8.2 (villeneuve v1.18.1, scott
  v1.13.3).

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

[0.9.0]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.9.0
[0.9.0-rc.0]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.9.0-rc.0
[0.8.2]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.8.2
[0.8.1]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.8.1
[0.8.0]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.8.0
[0.7.0]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.7.0
[0.7.0-rc.4]: https://github.com/Empyrean-Dynamics/empyrean/releases/tag/v0.7.0-rc.4
