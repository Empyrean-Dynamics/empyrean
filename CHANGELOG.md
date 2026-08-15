# Changelog

Notable changes to the empyrean distribution — the `empyrean`, `empyrean-sys`,
`empyrean-c`, and `empyrean-cli` crates and the `empyrean` Python package. This
project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added

- **The joint solved-parameter covariance crosses the boundary, in both
  directions.** A fit over the state and its parameters produces one
  \((6+P) \times (6+P)\) matrix, and until now only its diagonal blocks
  could cross the C ABI — the 6×6, the Marsden 3×3, a DT variance, an
  AMRAT variance, a per-segment thrust 3×3. A caller who fitted an orbit
  and re-propagated it therefore handed the engine a block-diagonal
  covariance, which is not a conservative approximation of the real one
  but a different claim: that the data producing the state and the data
  producing \(A_2\) were independent, when they are the same
  observations through the same fit.

  `CoordinateState` now carries the state↔Marsden border
  (`has_non_grav_cross` / `non_grav_cross`), placed beside the 6×6 it
  borders so a coordinate transform moves the pair together.
  `EmpyreanOrbit` carries the rest as two identity-tagged side arrays,
  `state_param_cross` and `param_pair_cross`. Entries name the
  *parameter* — `EMPYREAN_PARAM_COLUMN_MARSDEN` / `_DT` / `_AMRAT` /
  `_THRUST` — never a column index, because which column a parameter
  occupies depends on which *other* parameters the orbit declares:
  adding an SRP AMRAT shifts the thrust columns by one, and an
  index-keyed input would attach one parameter's correlations to another
  with every number finite and every gate passed. Entry order is
  therefore not contract, and a repeated entry (including a pair in its
  swapped form) is refused rather than resolved last-one-wins.

  Nine of the ten orbit-reading entry points consume it; the tenth,
  `empyrean_evaluate`, forms no prior — it measures residuals against a
  fixed orbit — so a supplied joint affects no output there and is
  returned unchanged rather than dropped. The file writers receive it.

- **Propagation reports the joint it computed, so legs chain.** Every
  `EmpyreanPropagatedState` gains `orbit_cov` — the propagated
  state↔Marsden border and wide carrier at that epoch — and
  `EmpyreanODResult::orbit` carries the same struct under the same name.
  `covariance` alone is only the state block; the joint the engine
  computes has non-zero state↔parameter columns **even when the input
  was block-diagonal**, because propagation itself generates that
  correlation. Chaining legs on the 6×6 alone therefore quoted a tighter
  uncertainty than the propagation supports, and there was no way to
  read the rest out. Now leg 2's input is leg 1's output, field for
  field, with no slot arithmetic: `propagate → propagate` and
  `determine → propagate` are the same copy. The posterior blocks are
  read off the object the engine already made coherent — the fitted
  orbit, or the propagated row — never re-derived from a slot-tagged
  matrix, whose *solved* layout disagrees with a re-fed orbit's
  *declared* one the moment the fit did not solve every declared axis.

- **Parameter dispositions replace solve-for flags.** Each axis on
  `EmpyreanSolveFor` now says what the fit *does* with that parameter:
  `EMPYREAN_PARAM_FIXED` (marginalized out, changes no number),
  `EMPYREAN_PARAM_SOLVED` (estimated), or `EMPYREAN_PARAM_CONSIDERED`
  (not estimated, but its prior uncertainty reaches the posterior
  through its measurement partials). A considered axis is not a safety
  margin: with a correlated prior the cross-dependent terms are
  sign-indefinite and the posterior can come back *tighter*.
  `EmpyreanODResult::dispositions` echoes the partition the fit actually
  ran, which is what tells a caller whether re-attaching a prior to an
  axis double-counts it.

- **A structured warning channel on OD results.**
  `EmpyreanODResult::warnings` reports supplied covariance a fit
  deliberately did not use — delivered payload rather than a log line,
  since a dropped prior cross term changes how the σ for that slot
  should be read.

- **Parquet orbit files carry the wide cross-covariance.** It gains a
  tagged tail for the state↔parameter and parameter↔parameter terms, and
  it is the only orbit format here that can carry them. The other two
  refuse by name rather than writing a row short:

  CSV refuses because this schema makes the difference between an absent
  cross and a supplied zero cross load-bearing, and CSV renders both as
  an empty cell. The JSON orbit format — this crate's own flat row
  shape, not the engine's schema — refuses because it holds the 6×6 and
  nothing beyond it; both refusals name parquet. A carrier holding
  thrust terms is refused everywhere, since no orbit format can
  serialize the thrust arcs those terms describe.

  Silence was the alternative worth avoiding: a file written short reads
  back as a block-diagonal joint, which is a tighter claim than the
  caller held, with nothing in the round trip to signal it.

- **`EmpyreanObservatoryConfig` matches the engine's observatory
  config.** It gains `min_elevation_deg` and `max_sun_altitude_deg`
  (behind `has_max_sun_altitude_deg`, since `0.0` is a legal solar
  altitude and a zero-init struct must not quietly plan a campaign in
  daylight — the switch resolves it to the engine's astronomical-twilight
  default).

  **No entry point exported by this ABI applies them.** Both values are
  marshaled across in full and reach the engine's own observatory config
  — but the gates that read them belong to the engine's visibility
  survey, which has no C entry point. `empyrean_evaluate_plan`, the one
  exported function that consumes this struct, consults the
  site-invariant pair alone, so setting either limit changes no number it
  returns and a candidate reported `observable` may still sit under that
  site's horizon or in its daylight. The fields ride
  the struct now so that exposing the survey later needs no further ABI
  break — stated plainly because a config field that looks live and is
  inert is worse than an absent one.

- **`EMPYREAN_REJECTION_PER_OBSERVATION_SITE_REQUIRED` (15).** The
  roving-observer codes `247` / `270` and the occultation code `275`
  name sites whose position travels with each observation. Distinct from
  `UNSUPPORTED_OBSERVATORY` (the code is well known) and from
  `SPACECRAFT_KERNEL_MISSING` (nothing needs loading): the fix is to
  supply the coordinates the ADES record already carries per
  observation.

- **Two new entry points read the propagated joint:
  `empyrean_propagation_joint_at` and `empyrean_orbit_covariance_free`.**
  The first returns the state↔Marsden border and the wide carrier at one
  `(orbit_index, epoch_index)` — the cross terms that
  `EmpyreanTaggedCovariance::matrix` is the state block of; the second
  releases the two arrays it allocates.

  They are a separate call rather than fields on the tagged-covariance
  struct because that struct is plain old data: a caller declares one on
  the stack, fills it through `empyrean_propagation_covariance_at_cartesian`,
  and frees nothing. Hanging owned arrays off it would have made every
  such caller — code that is correct today and recompiles without a
  murmur — leak two allocations per call. Nothing would fail; memory
  would simply grow. So the joint is opt-in, and the tagged covariance
  keeps its contract. `has_non_grav_cross = 0` with null carrier
  pointers means the engine produced no cross terms at that row, never
  that they were zero.

  These are the only two symbols this release's ABI adds, taking the
  exported surface from 92 to 94; every other change in it is a struct, a
  struct field, or a constant.

- **The joint reaches Python, in both directions.** Every orbit table
  gains a `wide_cross` sub-table and `NonGravParams` gains
  `non_grav_cross`, the 6×3 state↔Marsden border. Both are populated
  from the engine — on a fit's returned orbit, on every propagated
  state, and on the provenance-tagged covariance readback — and both are
  consumed on the way back in by `propagate`, `determine`, `evaluate`
  and `refine`, so a Python chain carries the joint the engine computed
  instead of its diagonal blocks. A chained propagation now reproduces
  the single-leg answer; on the same input, dropping the cross terms
  understates the propagated position variance.

  How much it understates by is fixture-specific, so here is one
  measured case rather than a general claim. On the wide-layout orbit in
  `empyrean-py/tests/test_wide_cross.py` — propagated, re-linked so the
  cross terms are the engine's own, then carried 3000 days to its
  close approach — dropping them understates the close-approach σ by
  about 19% and the B-plane covariance by about 34%. Both figures are
  pinned by bands in that file, so they turn red rather than drift
  silently if an engine pin moves them.

  Entries are keyed by parameter name (`"AMRAT"`, `"thrust[0].x"`)
  through `WideCross.state_cross` / `.param_cross`, which refuse to
  reshape a row whose tags and payload disagree rather than attaching
  one parameter's covariances to another. Absence is a null row, never a
  row of zeros: an absent correlation and a measured zero correlation
  are different claims, and only the second is one the engine's
  definiteness gate should see. `SolveFor` axes take
  `ParamDisposition` tags in place of booleans, and a bool is refused by
  name — `False` cannot say whether an axis was considered or fixed.

- **Observation planning reaches Rust and Python.** `empyrean_evaluate_plan`
  has been carried by the C ABI since v0.7.0-rc.4 with no consumer in any
  other channel; both now exist. Rust gains `Context::evaluate_plan` with typed
  inputs (`PlannedObservation::optical` / `::radar`, `RadarPlanSpec::given` /
  `::link_budget`, `PlanningConfig`) and owned results. Python gains
  `empyrean.evaluate_plan`, taking one covariance-bearing orbit plus
  `PlannedObservation` candidates and returning quivr tables: `PlanMetrics`
  (two rows, `stage` = `"prior"` / `"posterior"`), `PlanCandidates`
  (per-candidate marginal information gain; sky-plane geometry null on radar
  rows, the radar block null on optical rows; link-budget provenance notes
  carried verbatim, never summarized to a code), and `PlanEphemeris`
  (predicted sky positions with epochs as an `Epochs` sub-table). Candidates
  return in ascending epoch order regardless of the order supplied; an
  optical row joins its epoch and sky position through `index` into
  `PlanEphemeris`, and a radar row's `index` is its rank among the radar
  candidates in that same order.

  The planner requires a barycentric prior and refuses anything else by
  naming the fix — a heliocentric `determine` fit converts with
  `transform_coordinates(..., origin="SSB")`, a pure translation that leaves
  the covariance and every metric unchanged. Sky-plane fields the engine
  reports as structural zeros on radar rows surface as nulls, never as
  measured-looking numbers. The CLI does not gain a `plan` command in this
  release — a recorded subsetting decision, not an omission.

- **A capability-parity gate.** `empyrean-sys` now carries a committed
  manifest of the C ABI's 94 exported functions and five tests over it: the
  manifest must match the **compiled engine's actual exported symbols**
  (`nm` on the shipped library — an anchor no generated file can fake) and
  the generated shim surface; every export must have a call-form consumer in
  the Rust wrapper or an entry in a committed allow-list of recorded
  decisions, each with its reason; the allow-list is reverse-enforced — an
  entry whose symbol gains a consumer fails as stale until removed, and
  duplicate entries fail by line number; and the consumption scanner's own
  discrimination (calls count; comments, strings, imports, and test-gated
  code do not) is pinned by a fixture test. The gate exists because
  `empyrean_evaluate_plan` shipped in every published release of this
  distribution, from 0.7.0-rc.4 through 0.9.0, with no consumer in any
  channel, and nothing noticed. The allow-list opens with a
  single entry: the load-time ABI handshake, consumed by `empyrean-sys`
  itself.

  **The gate measures the Rust wrapper channel only.** Check 3 scans
  `empyrean/src` for call-form consumers, and `not_yet_wrapped.txt` records
  what that channel deliberately leaves unconsumed. Python and CLI
  consumption are not measured by any test here — the CLI's decision not to
  gain a `plan` command in this release is prose above, not a gated entry —
  so a symbol wired in one channel and forgotten in another is still a
  manual check. Widening the scanner to the other source roots is follow-up
  work.

### Fixed

- **The impact and B-plane paths dropped the caller's non-grav DT.**
  Both entry points — `empyrean_compute_impact_probabilities` and
  `empyrean_compute_b_planes` — share one orbit-marshaling routine, and
  it carried the DT *prior variance* while hardcoding the DT *value*
  absent. A DT comet's \(g(r)\) was therefore evaluated at zero delay on
  the two entry points where an understated joint does the most damage.

  That same routine also dropped the Marsden 3×3, as did the second
  inline copy on the ephemeris side, which serves
  `empyrean_generate_ephemeris` and
  `empyrean_builtsystem_generate_ephemeris`. So: two inline copies, four
  exported entry points, with the DT drop confined to the first copy and
  the Marsden drop common to both. Both now route through the shared
  marshaling helper that the propagation and OD paths already used.

  One behaviour change rides that routing and is called out rather than
  folded in: those three paths and the orbit-file writer now share the
  propagation path's non-grav **presence rule**, under which an orbit
  declaring a non-grav covariance or a DT prior carries a non-grav model
  even with all-zero A coefficients. That opens Marsden columns these
  paths previously left closed.

- **The orbit-file read path reported no non-grav covariance.** Orbit
  files have carried a `cov_dim = 9` state+Marsden joint since long
  before this release, and `empyrean_orbits_read_*` discarded it. The
  batch now carries it, along with the border and the carrier.

- **A solved DT posterior had no wire.** `EmpyreanNonGravParams` gains
  `has_dt_variance` / `dt_variance`; a solved-DT fit used to round-trip
  with its DT column closed.

- **Posterior blocks are sourced from the fitted orbit.** The non-grav
  3×3, the DT variance and the AMRAT variance previously came from
  `covariance_9x9`, which is populated only for the width-9 Marsden fit
  — while every fit carrying a wide joint is wider than 9 by
  construction. The two sets are disjoint, so the old source reported an
  absent covariance and a zero 3×3 for every such fit while the fitted
  orbit held the posterior. `covariance_9x9` remains populated for its
  deprecation window and is no longer a source.

### Changed

- **Python time inputs are `Epochs`, and every `Epochs` states its
  scale.** *Breaking.* Every public entry point in the `empyrean` Python
  package that takes a time now takes an `Epochs` table and nothing
  else. A bare list, array, float or numpy scalar is refused at the
  entry point with a `TypeError` that names the fix — not a deprecation
  warning, and never a silent assumption of TDB.

  The scale-pinned carve-outs are unchanged and stay plain floats,
  because their scale is fixed by definition rather than by the caller:
  columns named `mjd_tdb`, arguments named `epoch_mjd_tdb`, and the
  `epoch` column on the four coordinate tables, which is MJD TDB. That
  column is one you read rather than a time you hand in; converting it
  for reuse is `Epochs.from_mjd(..., scale="tdb")`, or
  `Epochs.from_orbits(orbits, offsets)` which offsets and tags in one
  step.

  A Modified Julian Date is a clock reading, not an instant. `61000.5`
  read as UTC and `61000.5` read as TDB name two moments about 69
  seconds apart, and that gap has grown with every leap second since
  1972. Which one a caller meant is a modelling statement, and the old
  signatures — `epochs: Epochs | np.ndarray | Sequence[float]`, with
  arrays taken as TDB — let a call site inherit that statement instead
  of making it. Sixty-nine seconds is not a rounding error. Mislabelling
  one scale as the other displaces a heliocentric state along track by
  \(|v| \times 69.18\,\text{s}\) — about 1460 km for a body at 2 AU
  (21.1 km/s) and about 2060 km at 1 AU (29.8 km/s) — before any
  encounter amplifies it; an object crossing the Earth's sphere of
  influence covers roughly 350 to 1040 km in that time, at a geocentric
  5 to 15 km/s.

  Thirteen signatures change: `propagate` and `BuiltSystem.propagate`,
  `get_states`, `get_observer_states`, `Observers.from_code` /
  `from_codes`, `query_horizons`, `compute_impact_probabilities` and
  `compute_b_planes` (whose `end_epoch` is now a length-1 `Epochs`
  rather than a float), and `index_at` / `up_to` on both
  `StateSensitivities` and `ObservationSensitivities`. To migrate, wrap
  the values you already had and say what they were — they were TDB:

  ```python
  # before
  empyrean.propagate(orbits, [61000.5, 61010.5])
  # after
  empyrean.propagate(orbits, Epochs.from_mjd([61000.5, 61010.5], scale="tdb"))
  ```

  `Epochs.from_mjd` and `Epochs.from_jd` accordingly **require**
  `scale`; it no longer defaults to TDB. `Epochs.linspace` and
  `Epochs.arange` require it as a keyword. `Epochs.now` keeps its
  `"utc"` default, because there the operation names its own clock, and
  `Epochs.from_iso` keeps its output-scale default, because an ISO
  timestamp's trailing `Z` already states the input's scale.

  Columns named `mjd_tdb`, and the arguments named `epoch_mjd_tdb` that
  mirror the **C ABI's** flattened fields (`PlannedObservation.optical` /
  `.radar`, `query_horizons_vectors`), are unchanged: there the name pins
  the scale, so nothing is left unstated. The Rust forms of those same
  constructors take a typed `Epoch` instead — the flattening is the C
  boundary's, not a divergence. Output tables are unaffected.

  The Rust channel keeps `f64` times, and its time parameters name their
  scale explicitly — every one is either a typed `Epoch` that carries
  its own `TimeScale` or a parameter named `*_mjd_tdb`, and the three
  `empyrean-cli` flags that take a time (`propagate --epoch`,
  `ephemeris --epoch`, `query horizons-vectors --epoch-mjd-tdb`) all name
  MJD TDB in their help text. An audit of
  every public function in `empyrean/src` found no ambiguous time
  parameter, so no signature changed there.

- **The time-scale conversions are cross-validated against astropy.**
  `Epochs.to_tdb` / `to_utc` are now checked against astropy's ERFA path
  across four regimes — the modern era, every leap-second boundary,
  the pre-1972 "rubber second" era, and epochs past the last tabulated
  leap second. Off leap-second days the two agree **bit for bit**, so
  the tests assert exact equality rather than a tolerance, and the
  leap-second tables are compared entry for entry (27 days, identical)
  so that either side falling behind IERS fails loudly instead of
  drifting quietly. astropy is a test-only dependency; no package code
  imports it outside the existing optional `Epochs.from_astropy` /
  `to_astropy` interop.

  The comparison also corrects the documentation: the TT↔TDB conversion
  carries the **full periodic** Fairhead & Bretagnon (1990) series, not
  the secular-only truncation the docstrings claimed. The measured
  TDB−UTC offset varies by 3.3148 ms peak-to-peak over the six-year grid
  the test samples (MJD 60300–62500), matching
  astropy's to the microsecond.

  The comparison surfaced one genuine defect, which this release
  **documents rather than fixes**: on the 27 UTC days that contain a
  leap second (and on pre-1972 rubber-second days), a *fractional* MJD
  converts as if the day were 86400 s long rather than 86401, so the
  error ramps linearly from zero at 00:00 to a full second at 24:00 —
  900 ms at 21:36, on every affected day. Midnight boundaries and every
  other day are exact, and no other regime is affected. It is pinned by
  a strict-`xfail` test that names its magnitude and turns red the
  moment the behavior changes, so it cannot be resolved silently. The
  fix is engine-side and is not in this release.

- **The engine pin advances to empyrean-core v0.10.1.** The release
  that carries everything above: the wide cross-covariance wire format
  and its units channel, the correlated joint prior with full
  Schmidt–Kalman consider analysis, the tri-state solve partition, and
  the per-site elevation and darkness visibility gates. Cross-free
  fits are bit-identical to the previous engine generation over the
  full validation catalog (61/61 orbit-determination rows, 8958/8958
  propagation and ephemeris rows), so existing results do not move.

- **Struct sizes and the two shrinks.** Nine frozen structs change size
  — seven grow, two shrink. The seven: `CoordinateState` by the border
  (360 → 512), `EmpyreanOrbit` by the carrier arrays (648 → 832),
  `EmpyreanPropagatedState` by `orbit_cov` (2392 → 2576),
  `EmpyreanNonGravParams` by the DT variance pair (160 → 176),
  `EmpyreanObservatoryConfig` by the visibility fields (40 → 64), and
  `EmpyreanODResult` by the joint, dispositions, thrust posteriors and
  warnings (7688 → 8128) — which also grows the
  `EmpyreanODObjectResult` that embeds it (7720 → 8160). The two:
  `EmpyreanSolveFor` **shrinks** 8 → 6, and the `EmpyreanODConfig` that
  embeds it shrinks 432 → 424 with it.

  Those two are the exception to a rule every previous version of this
  ABI could state without qualification. Replacing `thrust_segments` (a
  `u32`) with `thrust_dispositions[3]` (three `u8`s) also drops
  `EmpyreanSolveFor`'s alignment from 4 to 1, so every field after
  `solve_for_flags` inside `EmpyreanODConfig` shifts —
  `allow_unbracketed_maneuvers` 392 → 390, `has_photometry` 393 → 391,
  `photometry` 400 → 392 — which is what takes that config from 432 to
  424. **They are the first structs on this ABI to get smaller.** A
  consumer with a hand-mirrored `EmpyreanODConfig` must re-derive its
  whole layout rather than appending to it: keeping the old prefix
  writes `photometry` eight bytes past where the library reads it,
  corrupting it and the two bytes before it with no diagnostic. The
  other seven grow at the tail.

- **The C ABI is now versioned by the distribution release.**
  `EMPYREAN_ABI_VERSION` encodes the distribution's own version as
  `major * 10000 + minor * 100 + patch` and advances with every release
  whether or not any boundary type changed. The scheme **begins with
  0.10.0**, which reports `1000`; every release before it reported the
  retired independent counter instead, whose last published value is 2
  (v0.9.0). Values below 1000 are therefore counter-era, not release
  numbers, and no library has ever reported anything between the two.

  Only the **base** version is encoded — the pre-release suffix is not,
  so `0.10.0-rc.1` and `0.10.0` both report `1000`. The number separates
  one version from another and never a version from its own
  pre-releases, so a boundary type that moves inside a pre-release cycle
  is not caught by the handshake and both sides must be rebuilt
  together. Within a version's pre-releases the installed artifact or
  tag is what identifies the exact build.

  The consequence for consumers is that the number no longer says
  anything about layout: the only reading it supports is equality. A
  difference means a different release — rebuild against that release's
  header, or repoint at the matching engine — and a value that did not
  move is no longer a promise that no struct did. The distribution's own
  release string is not exported; a build is identified by the artifact
  or tag it was installed from, while `empyrean_version_string()` reports
  the engine crates' build provenance instead.

  This release's break is a single batched one; two
  changes are semantic rather than additive and are called out here
  rather than left to be discovered:

  `EmpyreanSolveFor`'s `marsden` / `dt` / `amrat` become the tri-state
  above. `0` and `1` keep their exact former meaning, so `memset(0)` and
  every value an older caller could write are unchanged — but a *new*
  caller writing `2` against a pre-0.10.0 library would hit a bare
  non-zero test and get the axis silently solved, so the boundary
  validates `0 | 1 | 2` strictly and the version handshake is what makes
  a dynamically-loaded mismatch fail at the check rather than in the
  physics. `EmpyreanSolveFor::thrust_segments` (a count) is replaced by
  `thrust_dispositions[3]` (one per declared segment): two counts cannot
  say *which* burn is considered, and a three-segment orbit with only
  the middle burn solved is now a routine case.

  `EmpyreanODResult::thrust_delta_m_per_s` is re-indexed from solved to
  declared segment order, and `thrust_delta_count` becomes the declared
  count. Every per-segment array on the result — Δv, covariance,
  disposition — now shares one index space, and an unsolved segment's Δv
  is NaN-filled exactly as its covariance is. Under the old pairing a
  consumer reading `thrust_delta_m_per_s[i]` beside
  `thrust_correction_covariances[i]` would have attributed a Δv to the
  wrong burn's covariance the moment any segment was considered or
  fixed.

## [0.10.0-rc.0] — 2026-08-09

C ABI version **3**. Three exported functions change signature, nine are
added, and seven struct sizes change (five grow directly;
`EmpyreanEphemerisConfig` grows by embedding the propagation config, and
`EmpyreanODResult` by embedding the widened acceptability report), so
this release is **source-breaking for C consumers** — recompile against
the version-3 header. Growth is appended at the tail in four of the five
directly-grown structs; the exceptions, spelled out so no binding trusts
a stale layout: `EmpyreanObservationResult` **inserts** `object_id` at
field index 1, shifting every following field by 8 bytes; the two
structs that grow by embedding shift every field after the embedded
member; and the dead `use_stm_cache` byte in `EmpyreanODConfig` gives
its slot to `allow_arc_truncation` (its own entry below). Recompiling
against the new header handles all of it — hand-mirrored layouts must
not assume any prefix survived. The sizes are enumerated under the
`EMPYREAN_ABI_VERSION` entry below.

### Added

- **`empyrean show` — a streaming pager over output artifacts.** Point it
  at a results directory to list the artifacts and pick one, or at a file
  directly; page like `more`, slide a horizontal column window across wide
  tables (a covariance block alone is 45 columns), filter rows, and pipe
  it — non-interactive output is a plain aligned stream that composes
  with `grep` and `head`. Streams parquet by record batch, CSV and JSON
  by row: constant memory at any file size. `--limit`, `--columns`,
  `--no-header`, `--full-precision`.

- **Strict-offline context construction.** A context can be built with
  the network switched off: it resolves the tier's kernel set from the
  data directory alone and fails, **naming every absent file**, if any is
  missing. There is no try-the-network-and-tolerate path and no
  degrade-to-a-lower-tier path — an offline context either has the full
  requested tier on disk or it does not get built. Available on every
  channel: `empyrean_context_from_data_dir_with(dir, options)` with the
  new `EmpyreanDataDirOptions { refresh, tier }` in C (a `NULL` options
  pointer is exactly the old constructor), `Context::from_data_dir_with`
  with `DataDirOptions` in Rust, `initialize(..., refresh=False)` in
  Python, and the global `--no-refresh` flag on the CLI.
- **The absent files come back as a list, not as prose.** A strict-offline
  failure carries every missing filename as structured data rather than
  one long message a caller has to split on a separator a filename may
  itself contain: `empyrean_missing_data_files()` /
  `empyrean_missing_data_files_free()` in C, `Error::missing_data_files()`
  in Rust, and a `missing_data_files` attribute on the `FileNotFoundError`
  Python raises. The list is complete — every file the tier needs and the
  directory lacks, not just the first one hit.
- **`EMPYREAN_OFFLINE=1` as a floor.** Set in the environment, it
  downgrades a requested `refresh` to off and says so on stderr. It is a
  floor, never an override: it can only ever *remove* network access, so
  a machine that must not reach the network cannot have that decision
  reversed by a library call. Only the exact value `1` asserts it.
  Applied at the Rust wrapper layer; the C ABI itself reads no
  environment variable. `empyrean::offline_floor_is_active()` reports
  whether it is in force, for callers that do network work of their own
  before building a context — the CLI uses it so `empyrean init` skips
  its kernel download under the floor instead of covering only the load.

  **Where it binds, exactly.** `Context::from_data_dir` does **not**
  consult it: reinterpreting the older, options-less constructor would
  change the meaning of code written before the variable existed, so a
  Rust service calling it reaches the network regardless. Every other
  channel does — `Context::from_data_dir_with`, Python's `initialize()`
  (which now routes through that constructor, a behaviour change on the
  default path), and every CLI command including `init`'s pre-context
  download. Reach for `from_data_dir_with` when the variable should
  apply.

  The floor downgrades **every** requested `refresh: true`, including one
  written by hand: `refresh` is a plain `bool` mirroring the engine's own
  options bag, so "defaulted" and "explicit" are not distinguishable
  states to branch on. A process that genuinely must reach the network on
  such a machine unsets the variable for itself. Every downgrade is
  announced on stderr.
- **`empyrean_download_data` — provisioning without a context.**
  `download_data` used to build a full Standard-tier context and discard
  it just to trigger the kernel fetch. The C ABI now exports the
  engine's provision-only path directly; the Rust wrapper routes through
  it with an unchanged public signature, and Python inherits the same
  chain. On a warm, complete directory it issues only the staleness
  checks and downloads nothing; on failure the structured missing-file
  list is available through `empyrean_missing_data_files`.
- **The C ABI writes fit summaries.**
  `empyrean_fit_summary_write_parquet` / `_csv` / `_json` take an
  `EmpyreanFitSummary` array and emit the same per-object fit-summary
  artifact the CLI writes, so a C consumer is not the one channel that
  cannot produce it.
- **Ephemeris overlap policy.** `EmpyreanPropagationConfig` gains
  `ephemeris_overlap_policy`, selecting what the engine does when a target
  coincides with one of its own perturbers:
  `EMPYREAN_EPHEMERIS_OVERLAP_POLICY_SUBSTITUTE_SPK` (0, the historical
  behaviour, and what a `memset(0)` config still means) or
  `EMPYREAN_EPHEMERIS_OVERLAP_POLICY_EXCLUDE_AND_INTEGRATE` (1). The
  second is what generating an ephemeris for an SB441-N16 body (1 Ceres,
  2 Pallas, 4 Vesta, …) at Standard tier requires — without it the engine
  substitutes the body's own SPK states, produces no dense trajectory, and
  the call fails. An unrecognized value is refused by value rather than
  defaulted. Reachable on every channel that carries a propagation config:
  `PropagationConfig::ephemeris_overlap_policy` / `EphemerisOverlapPolicy`
  in Rust, and `PropagationConfig(ephemeris_overlap_policy=...)` taking
  `"substitute_spk"` / `"exclude_and_integrate"` in Python. It is **not**
  on the CLI, which exposes no sibling propagation knobs on the commands
  that would need it; `empyrean ephemeris` for an SB441-N16 body has no
  in-CLI remedy and must go through one of the other three channels.

  Overlap *detection* is what decides whether the policy is consulted, it
  is on by default, and the engine suppresses it entirely whenever the
  exclusion list is non-empty — so setting `EXCLUDE_AND_INTEGRATE`
  alongside an explicit `excluded_perturbers` entry does nothing. Use one
  or the other. The detection toggle itself is not exposed at any
  distribution layer.
- **The sampling uncertainty methods, wired through.** `SIGMA_POINT` and
  `MONTE_CARLO` are honored on `propagate` — previously they were
  accepted and silently ignored — and `generate_ephemeris` now rejects
  them **by name** with a typed error rather than ignoring them. The
  adaptive Gaussian mixture joins the method set as
  `EMPYREAN_UNCERTAINTY_MIXTURE` (`5`, a new constant with no
  struct-layout change), and the impact-probability tables carry the
  `ip_agm` / `ip_mc` columns those methods produce.
- **The engine capabilities arriving with the empyrean-core v0.10.0
  pin.** Joint optical + radar orbit determination — radar delay and
  Doppler fold into escalated fits through a deterministic continuation,
  delivered only at true weight. The co-orbital initial-orbit-
  determination lane for Earth-resonant objects, on by default with its
  own config switch. Full-arc comet fits that escalate to the
  non-gravitational model instead of structurally truncating
  apparitions. Five new per-observation rejection codes —
  `EMPYREAN_REJECTION_NON_FINITE_CHI2` (`10`), `_MISSING_JACOBIAN`
  (`11`), `_SPACECRAFT_KERNEL_MISSING` (`12`),
  `_OBSERVER_CONSTRUCTION_FAILED` (`13`), `_NEVER_ABSORBED` (`14`).
  Nightly de-weighting follows Vereš, Farnocchia, Chesley & Chamberlin
  (2017) under the VFCC2017 preset, with the Carpino, Milani & Chesley
  (2003) rejection defaults at the papers' values. And the engine now
  refuses to run without all three Earth-orientation kernels —
  historical, high-precision, and long-term predict — with epochs before
  the historical floor falling back to the analytic IAU model behind a
  loud accuracy statement.
- **Observer states in a caller-chosen basis.** Observatory-code lookups
  can be returned in any supported `(frame, origin)` rather than only the
  ICRF / SSB construction basis — `frame` / `origin` arguments on
  `empyrean_get_observers`, `Context::get_observers`,
  `Observers.from_code` / `from_codes`, and `get_observer_states`. ICRF /
  SSB remains the default and is returned untransformed, bit for bit;
  it is still what ephemeris generation and orbit determination require.
  Each returned row carries the basis it is actually expressed in, read
  off the state rather than echoed from the request.
- **Covariance-quality detail.** `EmpyreanTaggedCovariance` gains
  `quality_kappa_state` and the `EMPYREAN_COVARIANCE_QUALITY_EXPANSION_SUSPECT`
  verdict, so a covariance that is definite but whose expansion is not
  trustworthy is reported as such instead of passing as clean.
  Chain/orbit count and sample-row epoch mismatches get their own error
  codes (`EMPYREAN_TAGGED_COV_CHAIN_ORBIT_COUNT_MISMATCH`,
  `EMPYREAN_TAGGED_COV_SAMPLE_ROW_EPOCH_MISMATCH`) rather than being
  collapsed onto a shared one — the two have different remedies.

### Changed

- **BREAKING — the weighting configuration is honored as written.**
  Preset `NONE` means none; user-supplied layers insert ahead of the
  preset chain instead of behind it; malformed values meet typed errors
  instead of silent clamps or `obs_code` repairs. A zero-initialized
  `EmpyreanWeightingConfig` / `EmpyreanDebiasingConfig` now means
  **disabled** — the production defaults (VFCC2017 weighting, EFCC2020
  debiasing) must be requested explicitly rather than being what
  `memset(0)` silently selected.

- **A tuned `Auto` uncertainty spec is accepted on the impact path.** The
  same dataclass `propagate` takes now lowers through
  `compute_impact_probabilities` / `compute_b_planes` with its κ
  thresholds honored, instead of raising `unsupported method spec`. The
  remaining unsupported-spec error names what the path accepts.

- **BREAKING — orbit determination is batch-first at every layer.**
  `determine` groups its observations by ADES object identifier and fits
  **every** object, returning one result per object instead of one result
  per call. Previously it ran the same batch internally and then returned
  a single fit — "the first acceptable one, else the first" — so an
  N-object arc silently discarded N−1 fits and the caller had no way to
  know it. Fitting one object is now the one-entry case of the batch
  call, not a different call.

  The batch keeps the main name at each layer:
  `empyrean_determine` writes an `EmpyreanDetermineResults` table of
  `EmpyreanODObjectResult` slots (each carrying its `object_id` and
  either the full `EmpyreanODResult` or a typed failure), released with
  the new `empyrean_determine_results_free`; `Context::determine` returns
  `DetermineResults` with iteration, `len`, by-object lookup, and an
  `into_single()` that unwraps the one-object case and **refuses** —
  naming the objects — rather than choosing among several; Python's
  `determine` returns `.orbits` / `.summary` / `.residuals` and indexes
  by object identifier, with `single()` as the same loud one-object
  convenience.

  A failed object never aborts the batch and never disappears: it is a
  slot with `delivered = 0`, an error string, and an
  `EMPYREAN_OD_FAILURE_*` classification. Its embedded result is
  NaN-poisoned — every float NaN, every enumerated code `-1`, every
  pointer null — so a caller that skips the `delivered` check gets an
  obviously invalid record rather than a plausible all-zero fit. A batch
  that delivers nothing at all returns the distinct
  `EMPYREAN_DETERMINE_NONE_DELIVERED` (`-4`) and **still populates** the
  table, which the caller must free. Seed orbits that matched no
  observation group come back in `unmatched_orbit_ids` rather than being
  dropped.
- **BREAKING — the CLI's `determine` writes per-object outputs.**
  `fitted_orbit.<ext>` (always exactly one row, `orbit_id = "fitted"`,
  no object identity) becomes `fitted_orbits.<ext>` — one row per
  delivered object, keyed by its ADES designation. Two artifacts join it:
  `fit_summary.parquet` **and** `fit_summary.csv`, always both, with one
  row per *input* object whether or not it produced an orbit (its
  convergence, RMS, acceptability verdicts, the four extrapolation gate
  axes, and on failure the reason); and `residuals.<ext>`, which gains an
  `object_id` column so a flat table across a batch stays attributable.
  stderr becomes a per-object table with the failures listed in full at
  the end. The exit code is the batch's verdict: `0` only when every
  object delivered, `3` when some did, `4` when none did.
- **BREAKING — the residual writers emit the whole residual surface.**
  `residuals.{parquet,csv,json}` carried five columns — RA / Dec
  residual, χ², probability, selected — out of the thirty-six the
  in-memory record holds. Everything else was computed and then dropped
  at the file boundary: `obs_id` (so the file had no join key at all),
  `object_id`, the observatory code, catalog and epoch, the entire
  rejection attribution (reason, criterion, threshold, effective
  threshold, information loss), the influence diagnostics, the
  along/cross-track decomposition, and the radar block. All thirty-six
  now reach disk in all three formats.

  The three formats are emitted from one column table, so they cannot
  disagree about what a residual file contains. Values are encoded per
  format where the format forces it: a non-computable number is a
  literal `NaN` in CSV and `null` in JSON, which has no NaN literal.
  `rejection_reason` and `radar_kind` are written as names rather than
  integer codes, matching the Python `ObservationResults` columns.
- **BREAKING — orbit CSV carries the full column set.** `--format csv`
  wrote 20 columns and **no covariance**, against parquet's 82 — a
  silently lossy format choice. The CSV path now goes through the same
  engine writer the parquet path uses, so the two emit an identical
  82-column schema (state, covariance, non-grav including `dt` and its
  variance, photometry, SRP), and a batch carrying a wide
  cross-covariance the row schema cannot express is refused rather than
  written short. The reader follows: it keeps the Marsden A1/A2/A3 block
  that the previous CSV reader dropped.
- **Python residual writers marshal the whole table.** `write_residuals_*`
  passed five columns across the boundary and let the rest default, so a
  residual file written from Python would have had the new columns and
  no content in them. Every column of `ObservationResults` now crosses,
  and `rejection_reason` round-trips by name.
- **The acceptability report carries the four extrapolation gate axes.**
  `EmpyreanAcceptabilityReport` (and its Rust / Python mirrors) grew
  `selection_fraction_*`, `selected_arc_coverage_ok` /
  `selected_arc_days_value` / `selected_arc_fraction_*`,
  `trailing_gap_*`, and the reserved `radar_fit_ok` tri-state — always
  `-1` / not-evaluated today, since radar-fit acceptability is not yet
  assessed — alongside the existing `fractional_sigma_a_*`. The four
  gate axes are what `extrapolation_acceptable` ANDs together on top of
  `fit_acceptable`, so a caller can now report
  *why* a fit is not safe to propagate — a heavily pruned arc, a
  selected span that no longer covers the requested one, or a rejected
  recent tail — rather than only that it is not. Every `f64` in the
  report is NaN when the quantity could not be computed; the `_ok`
  booleans are always valid.
- **BREAKING — `empyrean_transform_coordinates` is now the batched form.**
  It takes an array of states and an output array
  (`states`, `num_states`, `states_out`, all caller-owned) and transforms
  the whole batch in one call; the previous single-state shape is
  unchanged but renamed `empyrean_transform_coordinates_single`. The
  batched form carries the main name because a whole table going to one
  target basis is the common call, and one call is one error instead of
  `N`. Element `i` of a batch is **bit-identical** to the single-state
  call on `states[i]`, so this is a call-shape choice and never a
  numerical one — see the Python entry below for what it does and does
  not buy in wall-clock. Failures are fail-fast and index-attributed —
  the message names the offending element and `states_out` is left
  untouched, so there is never a partially-written output array. The two
  return codes are split by *what went wrong with the call*, not by where
  the failure happened: `-1` means the arguments are wrong (a null
  pointer, an unresolvable target basis), `-2` means element `i` failed,
  whether it was malformed on the way in, refused by the engine, or
  unmarshalable on the way out. Mirrored
  as `Context::transform_coordinates` (slice) and
  `Context::transform_coordinates_single` in Rust. C callers of the old
  scalar function change the name; Rust callers of `Context::transform`
  change to `transform_coordinates_single`.
- **BREAKING — `empyrean_get_observers` gained `frame` and `origin`
  parameters**, widened in place rather than given a sibling entry point,
  and `EmpyreanObserver` grew `frame` / `origin` tail fields (72 → 80
  bytes; every prior offset unchanged). Passing `0` / `0` is the
  construction basis and exactly the old behaviour, so a `memset(0)`
  request and an untouched consumer are unaffected beyond the recompile.
  `Context::get_observers` takes the two arguments in Rust; `Observer`
  gains the matching fields.
- **BREAKING — `EmpyreanODConfig` loses `use_stm_cache`, gains two
  axes.** The STM-cache knob is dead engine-side, so the field was a
  control that did nothing; it is removed rather than left as a lie.
  In its place the config exposes `allow_arc_truncation` (forbid the
  outward-expansion pipeline from truncating a sub-arc it cannot fit, so
  such an arc fails loudly instead of delivering the reconcilable part)
  and `coorbital_enabled` (master switch for the co-orbital IOD lane).
  Both are tri-state at the C ABI — negative means "engine default" — so
  a zero-initialized struct is never read as a deliberate "off". The
  engine's radar-annealing and linear-cache policies are solver policy
  rather than fit definition and stay at their defaults with no C knob.
- **BREAKING — `EMPYREAN_ABI_VERSION` is now 3.** Bumped once for the
  whole batch of changes above. Seven struct sizes change:
  `EmpyreanPropagationConfig` grew by `ephemeris_overlap_policy`
  (288 → 296 bytes), `EmpyreanEphemerisConfig` by embedding it
  (312 → 320) — easy to miss, since it gained no field of its own —
  `EmpyreanObserver` by the basis fields (72 → 80),
  `EmpyreanTaggedCovariance` by `quality_kappa_state` (520 → 528),
  `EmpyreanObservationResult` by `object_id` (264 → 272), and
  `EmpyreanAcceptabilityReport` by the four gate axes (120 → 208), which
  also grows the `EmpyreanODResult` that embeds it (7600 → 7688). Growth
  is at the tail except where called out: `object_id` is inserted at
  `EmpyreanObservationResult` index 1 (every following field shifts by
  8 bytes), the embedding growers shift everything after the embedded
  member, and the `use_stm_cache` slot is repurposed per its own entry.
  The constant space also grows: `EMPYREAN_UNCERTAINTY_MIXTURE` (`5`)
  and the five rejection codes `10`–`14` listed in their entries above.

  `empyrean-sys` now **enforces** the handshake the header has always
  documented: it calls `empyrean_abi_version()` the moment it opens
  `libempyrean` and panics, naming both versions and the resolved path, if
  the loaded engine disagrees. `dlsym` matches on symbol name alone, and
  ABI 3 is the first version to change the parameter list of an existing
  exported function, so a stale engine no longer merely returns wrong
  values — it reads a caller's integer as an out-pointer. C consumers
  should make the same check at load.
- **Python `transform_coordinates` crosses the FFI once per table.** The
  binding used to walk the table and call the single-state entry point
  once per row; it now marshals the whole table and makes one batched
  call. The Python surface is unchanged — still table-in, table-out — and
  the results are bit-identical row for row. Measured against the old
  shape it is ~17% faster from a thousand rows up and indistinguishable
  at one row: the engine's gravitational-parameter and origin-shift memos
  are scoped to the context, so they already amortized across successive
  single-state calls, and what the batch saves is the per-call boundary
  crossing. Failures are now attributed to the offending row.
- **CLI: a global `--no-refresh`.** Accepted alongside `--data-dir` on
  every context-building command. A command that would otherwise hand its
  work to a running daemon runs in-process instead when the flag is set:
  the daemon's context was built once, under its own options, and cannot
  honour a strict-offline request retroactively — serving it anyway would
  ignore the flag without saying so. On `init` the flag turns the command
  into a verifier: it downloads nothing and reports exactly which files
  the data directory is missing. `EMPYREAN_OFFLINE=1` reaches all of the
  same places, `init`'s download included.

### Fixed

- **The six observation-sensitivity rows are named constants at every
  layer** rather than an undocumented ordering a consumer had to know.
- **Photometry rides the fitted orbit.** A `determine` result carries
  the arc's fitted photometry on the orbit itself, so a downstream
  ephemeris predicts apparent magnitudes instead of returning none.
- **A seeded fit delivers under the seed's identity.** Supplying
  `initial_orbits={"my-name": orbit}` labels the fitted orbit
  `"my-name"` again, as documented — the batch rework had returned the
  ADES designation instead. Batch indexing is unchanged; only the fitted
  orbit's own identity is relabelled.
- **`read_ades` resolves file paths on the Rust wrapper.** The
  documented path-or-content detection handed the path itself to the
  content parser, so `empyrean determine <file>` failed parsing its own
  argument as MPC80.

- **`Epochs.to_iso` no longer relabels time scales.** The `scale`
  keyword was documented as cross-scale formatting but implemented as a
  source reinterpretation, so `to_iso(scale="utc")` on TDB epochs
  printed the TDB clock under a UTC label — 69 seconds wrong, silently.
  The method always emits UTC wall-clock for the stored instant; a
  mismatched `scale` argument now raises with directions to convert
  first.
- **Absent impact probabilities are Arrow nulls.** In the
  possible-impacts events table they marshaled as NaN while the
  dedicated IP tables used nulls, so an `is not None` gate never fired.
  The typed `Impacts` table also gains the body-relative
  `relative_velocity` column the flat surface already carried.
- **`compute_stm` reaches the engine on the ephemeris path.** The C-ABI
  ephemeris-config converter hand-rolled its narrow config instead of
  routing through the shared propagation converter, so every
  propagation-level knob it did not happen to copy was silently
  discarded — `compute_stm` above all, on **both** the one-shot and
  handle-based entry points. A caller that asked for observation
  sensitivities on an orbit with no covariance got a clean success code,
  no warnings, and no partials. The converter now routes through the
  shared one and narrows field by field with no defaulting tail, so a
  knob added upstream breaks the build rather than starting a fresh
  silent drop. `excluded_perturbers_naif`, `num_threads`, the
  `ephemeris_overlap_policy` above, and the whole `advanced` block were
  dropped by
  the same defect and are fixed with it. The two blocks ephemeris
  generation genuinely cannot honour — `events` and `diagnostics` — are
  now **refused by name** rather than accepted and ignored.
- **A non-default observer basis returned states instead of a caught
  panic.** The first cut of the widened `empyrean_get_observers`
  marshaled through accessors that assert ICRF / SSB, so every
  non-default request unwound into the FFI boundary and surfaced as
  error code `-99`.
- **Documented contracts that described the bug.** The Python
  `EphemerisResult.sensitivity` docs, the sensitivity-table error
  messages, and the Rust `PropagatedState::stm` docs all said partials
  require an input covariance. They require a *traced STM*, which an
  input covariance produces and which `compute_stm` also produces on its
  own. Corrected across the wrapper and the Python package.
- **Python observation-sensitivity rows are keyed on the caller's ids.**
  The sensitivity table carried the C ABI's synthetic `"orbit_0"` while
  the ephemeris table beside it carried the real `orbit_id`, so the
  documented per-chain filter — `sens.select("orbit_id", oid)` — returned
  an empty table for every real orbit id, and `object_id` was always
  null. Both are now recovered from the caller's input the same way the
  ephemeris rows already were.
- **Coordinate-state conversion failures surface.** The internal
  coordinate-state converter used on the OD-session and orbit-write paths
  is now fallible and its failures are propagated, rather than a sentinel
  value being substituted for a state that could not be converted.
- **A short batch-transform result can no longer be read back as data.**
  The C batch wrote its output by zipping the engine's result against the
  caller's array, so a result shorter than `num_states` would have stopped
  early and still returned success — leaving the tail of `states_out`
  whatever it was, which for a zeroed array is a perfectly valid
  Cartesian / ICRF / SSB state at MJD 0. The row count is now checked
  against the input before anything is written.
- **Python `generate_ephemeris` no longer substitutes observer geometry.**
  A failed observatory-code lookup — an epoch outside the loaded BPC's
  coverage, an unknown code — fell back to the caller's own
  position/velocity columns with `observing_night = -1`, returning
  astrometry computed from stale geometry with a clean success, no
  warning, and the nightly grouping the OD weighting depends on silently
  discarded. The lookup's error now propagates. The observer table's
  state columns are no longer sent across the boundary at all: every
  observer is recomputed from its (code, epoch), so there is one source
  for those numbers rather than two.
- **`ExpansionSuspect` covariances reach the Rust and Python layers.**
  The wrapper's tagged-covariance reader had no arm for the new quality
  tag and rejected the whole record with "unknown covariance quality
  tag: 3"; `quality_kappa_state` was never read. Both are now carried
  through, as `CovarianceQuality::ExpansionSuspect { kappa_state }` in
  Rust and as a `quality_kappa_state` column plus an `expansion_suspect`
  member of `CovarianceQuality` in Python.
- **The ephemeris config's Python docs no longer describe the bug they
  were written to fix.** `EphemerisConfig` still claimed that "every
  propagation-side knob" is set on the embedded `PropagationConfig`, a
  sentence false in both directions: `events` and `diagnostics` are
  hard-refused, and neither `excluded_perturbers` nor
  `ephemeris_overlap_policy` — the two knobs that decide whether an
  SB441-N16 ephemeris works at all — was mentioned. Rewritten to the
  explicit list.
- **The events / diagnostics refusal is a `ValueError` in Python.** It
  reached the caller as a `RuntimeError` from the FFI marshaling step,
  while the unsupported-`uncertainty_method` rejection on the same call is
  documented as raising `ValueError`. Both are caller mistakes and both
  now raise the same class.
- **`empyrean-sys`'s generated shims were double-encoded.** A
  regeneration wrote the file through a Latin-1 round trip, so every
  em-dash, ellipsis and Greek letter in its doc text — the docs.rs surface
  of the whole v3 API — rendered as mojibake. Nothing that compiles the
  crate reads doc text, so no gate saw it; a test now checks both
  generated files for the byte pattern. `shims.rs` is also generated
  mechanically from `bindings.rs` now, so the two cannot drift in doc text
  or signature, and the exact `bindgen` invocation is recorded at the top
  of `bindings.rs`.

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
