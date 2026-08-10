<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean-sys">

# empyrean-sys
Low-level FFI bindings to the libempyrean astrodynamics C ABI

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean-sys"><img src="https://img.shields.io/crates/v/empyrean-sys.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
<a href="https://docs.rs/empyrean-sys"><img src="https://img.shields.io/docsrs/empyrean-sys?style=flat-square&label=docs.rs" alt="docs.rs"></a>
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

empyrean-sys exposes the C ABI of `libempyrean` to Rust as raw,
`unsafe`, bindgen-generated declarations. It does not attempt to wrap,
type-check, or RAII-manage the underlying handles.

```toml
[dependencies]
empyrean-sys = "0.10.0-rc.0"
```

```rust
use empyrean_sys::*;

// All entry points are unsafe; pointer ownership and lifetime are the
// caller's responsibility. See include/empyrean.h at the repository
// root for the authoritative C ABI documentation.
unsafe {
    // Null data_dir = the platform default data directory; downloads
    // any missing kernels. Returns null on error (see empyrean_last_error).
    let ctx: *mut EmpyreanContext = empyrean_context_from_data_dir(std::ptr::null());
    assert!(!ctx.is_null());
    empyrean_context_free(ctx);
}
```

**Most users want the safe wrapper instead** — see the
[`empyrean`](https://crates.io/crates/empyrean) crate, which builds on
empyrean-sys to provide typed handles, `Result`-returning entry points,
and Rust-native lifetime management.

## What the bindings cover

The declarations track the full C ABI at `EMPYREAN_ABI_VERSION`
(currently `3`), including the v0.9.0 wide-parameter fitting surface
and the ABI-2 output surface below. ABI 2 grew existing struct shapes by
appending fields only; ABI 3 is the first version to change function
shapes and struct interiors, so consumers must recompile against the
matching header.

**Function shapes.** `empyrean_determine` keeps its name and arity, but
its final out-parameter is now `EmpyreanDetermineResults *` — the batch
table, one slot per ADES object — rather than a single
`EmpyreanODResult *`, and it is released with the new
`empyrean_determine_results_free` (`empyrean_od_result_free` still
releases `empyrean_refine`'s single result).
`empyrean_transform_coordinates` becomes the batched (array in / array
out) entry point, with the one-state form renamed
`empyrean_transform_coordinates_single`, and `empyrean_get_observers`
gains `frame` / `origin` parameters. New entry points:
`empyrean_context_from_data_dir_with` with `empyrean_missing_data_files`
/ `empyrean_missing_data_files_free`, `empyrean_download_data`, and
`empyrean_fit_summary_write_parquet` / `_csv` / `_json`. The weighting
preset constant is now `EMPYREAN_WEIGHTING_PRESET_VFCC2017` (Vereš,
Farnocchia, Chesley & Chamberlin 2017), replacing the `..._VFC17`
spelling.

**Struct shapes** (64-bit sizes, every one asserted at compile time in
the generated bindings): `EmpyreanPropagationConfig` 288 → 296
(`ephemeris_overlap_policy`), `EmpyreanObserver` 72 → 80 (`frame` /
`origin`), `EmpyreanTaggedCovariance` 520 → 528
(`quality_kappa_state`), and `EmpyreanAcceptabilityReport` 120 → 208
(the new gates), carrying `EmpyreanODResult` 7600 → 7688 with it. Two
are interior changes rather than appends:
`EmpyreanObservationResult` 264 → 272 inserts `object_id` right after
`obs_id`, and `EmpyreanODConfig` stays exactly 432 bytes while
replacing `use_stm_cache` with `allow_arc_truncation` and
`coorbital_enabled` at offset 208 — same size, different meaning, which
no size check would catch. New types: `EmpyreanDetermineResults` (32),
`EmpyreanODObjectResult` (7720), `EmpyreanFitSummary` (184),
`EmpyreanDataDirOptions` (8), `EmpyreanMissingDataFiles` (16).

The version handshake is enforced here, not merely documented: the
loader calls `empyrean_abi_version()` the moment it opens `libempyrean`
and panics — naming both versions and the resolved path — if the engine
disagrees with `EMPYREAN_ABI_VERSION`. `dlsym` matches on symbol name
alone, so a stale engine picked up from `EMPYREAN_LIB` or a leftover
`target/release` would do worse than return wrong numbers: an ABI-2
`empyrean_get_observers` reads the caller's `frame` integer as its
out-pointer, and an ABI-2 `empyrean_determine` writes a 7600-byte
`EmpyreanODResult` through a pointer the caller sized for a 32-byte
`EmpyreanDetermineResults`.

Each type below maps 1:1 onto a C struct; consult `include/empyrean.h`
at the repository root for field-level semantics.

- **Batch orbit determination.** `empyrean_determine` groups the
  observations by ADES object identifier (permID / provID / trkSub),
  fits each group, and returns one `EmpyreanODObjectResult` per object
  in ascending `object_id` order inside `EmpyreanDetermineResults`. Each
  slot's `delivered` flag selects the live payload: a fully populated
  `EmpyreanODResult`, or a typed failure carrying the engine's `error`
  message and an `EMPYREAN_OD_FAILURE_*` `error_code` (`IOD`, `OD`,
  `RADAR_ONLY`, `DUPLICATE_OBS_IDS`, `OBSERVATION_CONVERSION`,
  `OBSERVER_CONSTRUCTION`, `EARTH_ORIENTATION_COVERAGE`,
  `NON_GRAV_NOT_RECOVERED`, `UNSUPPORTED_COORDINATE_SYSTEM`) — with
  `result` NaN-poisoned so an unchecked read is obviously invalid rather
  than a plausible all-zero fit. One object's failure never aborts the
  batch; every object failing returns
  `EMPYREAN_DETERMINE_NONE_DELIVERED` (`-4`), which still populates the
  table and still requires freeing it. Seed orbits matching no group
  come back in `unmatched_orbit_ids` rather than being dropped, and each
  `EmpyreanObservationResult` in the table carries the `object_id` it
  was fitted under, so a caller may concatenate every object's rows
  into one flat table and still know which fit each row belongs to.
  Radar observations ride the same call as the optical set and are
  fitted jointly with it.
- **Acceptability verdicts and typed rejections.**
  `EmpyreanAcceptabilityReport` carries both `fit_acceptable` and
  `extrapolation_acceptable` alongside a per-gate `_ok` flag and, where
  the gate is numeric, the `_value` / `_threshold` pair behind it —
  convergence, reduced χ², RMS, residual isotropy, covariance, arc
  coverage, fractional σ_a, selection fraction, selected-arc coverage,
  trailing gap, and radar fit — so a verdict serializes together with
  the number that produced it. `EmpyreanFitSummary` is the flat
  per-object row of that verdict (identity, status, counts, RMS,
  reduced χ², solved width, and the gates), written by
  `empyrean_fit_summary_write_parquet` / `_csv` / `_json`. Rejection
  reasons gained
  `EMPYREAN_REJECTION_NON_FINITE_CHI2`,
  `EMPYREAN_REJECTION_MISSING_JACOBIAN`,
  `EMPYREAN_REJECTION_OBSERVER_CONSTRUCTION_FAILED`,
  `EMPYREAN_REJECTION_SPACECRAFT_KERNEL_MISSING`, and
  `EMPYREAN_REJECTION_NEVER_ABSORBED`, so a deselected observation
  always carries a reason rather than a bare flag.
- **Hard-object switches.** `EmpyreanODConfig::allow_arc_truncation`
  and `coorbital_enabled` are tri-state (`-1` engine default, `1` on,
  `0` off). Forbidding truncation makes an arc spanning a dynamical
  discontinuity fail loudly instead of delivering a fit of the
  reconcilable sub-arc with the rest tagged
  `EMPYREAN_REJECTION_OUTSIDE_ARC`; the co-orbital IOD lane is what
  recovers 2010 TK7 / 2020 XL5-class Earth co-orbitals.
- **Wide-parameter OD.** `empyrean_determine` / `empyrean_refine` solve
  beyond the 6-parameter state and the Marsden A1/A2/A3 non-gravitational
  block for the cometary outgassing time delay DT, the SRP area-to-mass
  ratio AMRAT, and per-segment thrust Δv corrections on continuous-thrust
  arcs — each partial produced analytically by the hyperdual integrator. The per-axis `EmpyreanSolveFor`
  (`marsden` / `dt` / `amrat` / `thrust_segments`) is read under
  `EMPYREAN_SOLVE_FOR_EXPLICIT`. DT / AMRAT / thrust are refine-path solves:
  the input orbit must carry the corresponding prior (its declared variance)
  to open the axis, and requesting an axis without its prior errors out
  rather than returning a zeroed or defaulted column.
- **Tagged solved covariance.** `EmpyreanSolvedCovariance` carries the
  fitted-parameter identities alongside the matrix: `marsden_slot`,
  `dt_slot`, `amrat_slot`, and `thrust_slots` locate each parameter's
  row/column, with `EMPYREAN_SLOT_NONE` marking an axis that was not solved.
  Read a parameter's variance by its slot — the `width` alone is ambiguous.
- **Post-OD photometry.** With `EmpyreanODConfig::has_photometry` set,
  `EmpyreanPhotometryConfig` requests a fit recovering absolute magnitude
  `H` and the phase-function slopes from the observation magnitudes,
  climbing the H-only → HG12 → HG1G2 ladder (Muinonen et al. 2010) to the
  richest model the arc's phase coverage supports. `EmpyreanODPhotometryResult`
  reports the fitted `h` / `slope1` / `slope2`, the `model_used`, a 3×3
  `covariance` giving an honest 1σ on `h`, plus
  `n_mags_dropped_unconvertible` and the distinct offending band codes
  in `dropped_bands` when magnitudes could not be converted to V — the
  observations' astrometry is unaffected.
- **Ephemeris covariances and warnings.** Each `EmpyreanEphemerisEntry`
  carries a 6×6 sky-plane covariance over (rho, RA, Dec) and their rates
  in (AU, deg) units, and the aberrated (light-time corrected)
  barycentric ICRF Cartesian state at the photon-emission epoch with its
  own 6×6 covariance — `has_covariance` / `has_aberrated_covariance`
  gate each block (absent unless the input orbit carried a state
  covariance). `EmpyreanEphemerisResult` returns run-level non-fatal
  `warnings`, e.g. Earth-orientation kernel coverage gaps served by the
  analytic IAU 2006 fallback, or rows whose sensitivity chain was
  skipped.
- **Per-observation diagnostics.** `EmpyreanObservationResult` carries
  radar delay/Doppler residuals (observed − predicted, seconds / hertz,
  with `radar_chi2`, `radar_dof`, `radar_probability`, and the combined
  `radar_variance`), the D-optimality `influence_information_loss` on
  removal (+∞ marks an indispensable observation), and
  `along_cross_covariance_arcsec2` completing the 2×2 along/cross-track
  covariance.
- **Covariance trust verdict.** `EmpyreanODResult::covariance_trust`
  reports an event-aware verdict on the delivered covariance
  (`EMPYREAN_COVARIANCE_TRUST_*`): `TRUSTED`, `ENCOUNTER_INTERVENES`
  (with the intervening close-approach or high-nonlinearity event and
  whether a second-order state-only correction can recover it), or
  `WEAKLY_DETERMINED_HIGH_N`. `NOT_EVALUATED` means no gate ran —
  absence of a verdict is not trust.
- **Impact-probability detail.** Each `EmpyreanImpactProbability` row
  carries the geodetic impact point (latitude / longitude / altitude on
  the body's reference ellipsoid, when the surface projection is
  available), the Monte-Carlo 95% binomial confidence half-width on
  `ip_mc`, the second-order corrected mean miss distance with its 1σ
  uncertainty and skewness, the closest-approach distance `gradient`
  (6-vector) and `distance_hessian` (6×6) with respect to the initial
  state, and the adaptive Gaussian-mixture component count.
- **Plan evaluation.** Radar candidates in `EmpyreanPlanCandidate`
  carry the effective SNR (linear power ratio), one-way range (km),
  link-budget provenance notes, and the measurement mode;
  `EmpyreanPlanResult` carries the predicted optical ephemeris (epoch
  MJD TDB, RA, Dec per optical candidate).
- **Basis-tagged mixture components.** Each `EmpyreanMixtureComponent`
  is tagged with the reference `frame` and center-body `origin` (NAIF
  id) its mean and covariance are expressed in.
- **Data provisioning and strict offline.**
  `empyrean_context_from_data_dir_with` takes `EmpyreanDataDirOptions`
  (`refresh`, `tier`); a null or zeroed struct reproduces
  `empyrean_context_from_data_dir` exactly.
  `refresh = EMPYREAN_DATA_REFRESH_OFF` resolves the tier's kernels
  from `data_dir` alone and fails naming **every** absent file — no
  lower-tier fallback, no partially loaded context — with the list
  retrievable through `empyrean_missing_data_files` as structured
  entries rather than a string to split (file names may contain the
  separator); release it with `empyrean_missing_data_files_free`.
  `empyrean_download_data` provisions a data directory without building
  a context at all. These entry points read no environment variable:
  the C ABI honours exactly what the caller passed.

## Runtime requirement

empyrean-sys opens `libempyrean.{dylib,so}` at run time via
`libloading` (dlopen). The library is distributed separately as a
binary release on
[GitHub](https://github.com/Empyrean-Dynamics/empyrean/releases) and
inside the published Python wheel. The path is resolved from the
`EMPYREAN_LIB` environment variable if set, else a `libempyrean.*`
sitting next to the loaded module, else a build-time location — an
`EMPYREAN_LIB_DIR` override, a sibling `../target/release` build, or
a checksum-pinned download from the GitHub release (in that order).
The FFI bindings are pre-generated and committed, so no C header,
libclang, or bindgen is needed to build.

Prebuilt engine binaries are currently published for four targets:
macOS arm64 (`macos-aarch64`), macOS x86_64 (`macos-x86_64`), Linux
x86_64 (`linux-x86_64`), and Linux aarch64 (`linux-aarch64`). On other
targets the build stops with an error unless `EMPYREAN_LIB_DIR` points
at an engine build.

## License

Source code in this crate is licensed under the
[BSD 3-Clause License](LICENSE). The closed-source `libempyrean`
runtime it loads at runtime is governed by a separate proprietary binary
license; see the main repository for the dual-license breakdown.

Copyright © 2024–2026 Joachim Moeyens. All rights reserved.
