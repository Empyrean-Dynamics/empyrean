<img src="https://raw.githubusercontent.com/Empyrean-Dynamics/empyrean/main/docs/empyrean-dynamics-icon.png" width="140" alt="empyrean-cli">

# empyrean-cli
Command-line interface for empyrean — orbit propagation, ephemeris generation, orbit determination, and event detection

<a href="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml"><img src="https://github.com/Empyrean-Dynamics/empyrean/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://crates.io/crates/empyrean-cli"><img src="https://img.shields.io/crates/v/empyrean-cli.svg?style=flat-square&label=crates.io" alt="crates.io"></a>
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

empyrean-cli is the command-line interface to empyrean. It publishes
one binary — `empyrean` — that runs every headline pipeline (orbit
propagation, ephemeris generation, orbit determination, and event
detection), emits Parquet output you can join in pandas / Polars /
DuckDB, and pages through what it wrote without leaving the terminal.

## Install

Current release: **0.10.0-rc.0** (release candidate).

```sh
cargo install empyrean-cli --version 0.10.0-rc.0
```

A release candidate is not selected by a bare `cargo install`, so the
`--version` is required until the final release.

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

# Fit every object in an ADES PSV — one fit per designation, one command.
empyrean determine observations.psv --out-dir ./out

# Page through what any of them wrote.
empyrean show --out-dir ./out

# Confirm the build provenance — every binary carries the `<tag>+<sha>`
# strings of the villeneuve / scott / nolan commits it was built against.
empyrean version
```

The pipeline commands (`propagate` / `ephemeris` / `determine`) emit
Parquet tables under `--out-dir` by default; `--format json` and
`--format csv` are also available, and CSV is no longer a lossy choice —
it carries the same 82-column orbit schema Parquet does, covariance
included. The schemas match the Python and Rust API outputs exactly —
same `orbit_id` / `object_id` join keys, same time scales, same physical
units — so you can mix-and-match channels for the same workflow.

`empyrean show` browses what they wrote — see [Browsing output](#browsing-output).

Beyond the headline pipelines: `propagate` takes `--uncertainty-method`
(`first-order` / `second-order` / `sigma-point` / `monte-carlo` /
`auto`) and `--tagged-covariance`; `empyrean query horizons-vectors`
fetches JPL Horizons state vectors; `empyrean cache info` / `cache clear`
manage the API response cache; and `empyrean serve` / `empyrean stop`
run a daemon that keeps the loaded kernels in memory for faster
subsequent commands. See `empyrean <command> --help` for the full
flag surface.

## Running offline

`--no-refresh` is global — it goes on any command, before or after the
subcommand. It builds the context from `--data-dir` alone and never
reaches the network: no partial load, no fallback to a smaller kernel
set. If a required file is absent, the command fails and **names every
missing file**, which is the remedy rather than a hint at it.

```sh
# Verify a data directory without downloading anything.
empyrean --no-refresh init --data-dir /mnt/kernels

# Fit on an air-gapped machine.
empyrean determine observations.psv --no-refresh --out-dir ./out
```

`EMPYREAN_OFFLINE=1` in the environment does the same thing and
announces itself. It is a floor, not a switch: it can only ever turn
network access off, never back on. On `init` it turns the command into
a verifier — the kernel download is skipped outright rather than run
and then discarded.

Under either form, a command that would normally hand its work to a
running daemon runs in-process instead. The daemon's context was built
once, under its own policy, and cannot honour a strict-offline request
after the fact; serving it anyway would ignore the flag without saying
so.

## Browsing output

`empyrean show` pages through the tables the pipeline commands write —
Parquet, CSV, or JSON — without leaving the terminal and without a
notebook.

```sh
# List an output directory and pick a file to open.
empyrean show --out-dir ./out

# Straight into one file.
empyrean show ./out/residuals.parquet
```

The listing gives each artifact's row count, size, and what it holds,
then asks which one to open:

```text
$ empyrean show --out-dir ./out
./out

    FILE               ROWS  SIZE      DESCRIPTION
 1. ephemeris.parquet  96    12.4 KiB  Predicted ephemeris — RA/Dec per observer and epoch
 2. events.parquet     3     9.8 KiB   Detected events — close approaches, impacts, and their geometry
 3. states.parquet     1     20.1 KiB  Propagated states — position, velocity, covariance at the target epoch

Open [1-3] (q to quit): 1
```

Parquet row counts come from the file footer, so they are exact and cost
no decoding; a text file past 64 MiB is sampled instead and its count is
marked `~`. A file `show` cannot read is skipped rather than failing the
listing — a stray `README.md` next to the tables does not stop you
browsing them. Naming an unreadable file *directly* is still a hard
error, by name, saying what was tried.

In the pager: **space** / **enter** for the next page, **b** for the
previous one, **←** / **→** to slide the column window (an orbits table
is 82 columns — most of them a covariance block), **g** / **Home** to
jump back to the first row, **/** to filter rows by substring, **Esc** to
drop the filter, **q** to quit. The status line says where you are, and
declines to guess a total it has not reached yet:

```text
rows 1-21  ·  cols 1-7 of 21  —  space/b page  ←/→ cols  / filter  q quit
```

It streams. The first page of a multi-million-row residuals table appears
immediately and memory does not grow with the file, because rows are
pulled a record batch at a time rather than loaded. Paging *backward*
re-reads from the start of the file to reach the target row, so `b` gets
slower the deeper into a very large file you are; paging forward does not.

Piped, `show` drops the interactivity and writes the whole table as
aligned text — nothing truncated, since a clipped number on its way into
`awk` is a wrong number — so it composes:

```sh
empyrean show ./out/residuals.parquet --columns obs_id,object_id,chi2 | grep -v NaN
empyrean show ./out/states.parquet --limit 5 --no-header
empyrean show ./out/fit_summary.csv --columns object_id,status,rms_ra_arcsec,fit_acceptable
```

```text
object_id  status     rms_ra_arcsec  fit_acceptable
99942      delivered  0.32           true
2010 TK7   delivered  0.41           true
K25X99Z    failed     NaN            false
```

Flags: `--limit N` caps the rows, `--columns a,b,c` selects and reorders
them (a name the file does not have is an error listing the names it
does), `--filter TEXT` keeps rows containing that text case-insensitively,
`--no-header` drops the header line, and `--full-precision` prints every
digit of every float instead of the default six significant figures (the
default is readable; the flag is exact and round-trips). `--out-dir DIR`
is the same as passing the directory positionally.

The filter matches the row *as displayed*, so `--filter NaN` finds
exactly the uncomputable cells: an absent value renders as an empty cell,
never the text `null`, and cannot be swept up by accident. That
distinction is kept everywhere — in a residuals table a null `ast_cat`
means no star catalog was recorded while a NaN `chi2` means a χ² that
could not be evaluated, and neither must ever read as the other.

`show` reads files and nothing else — it needs no SPICE kernels and never
loads the `libempyrean` runtime, so it works on a machine that has only
the CLI and on outputs copied off a cluster.

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

## Orbit determination

`determine` is batch-first: it groups the ADES PSV by object identifier
(`permID` → `provID` → `trkSub`) and fits **every** object in the file.
A file with one object is the one-row case, not a different command.
Optical and radar (delay / Doppler) rows in the same file are fitted
jointly, which is what pulls a hard object's orbit down.

```sh
# 6-parameter fit per object. The default `--solve-for auto` starts
# state-only and escalates to non-grav automatically on a poor fit.
empyrean determine observations.psv --out-dir ./out
```

```text
Loaded context (2.4s)
Read 883 observation(s) from observations.psv
Running orbit determination...
OD complete (41.7s): 2 of 3 object(s) delivered

  Object           Converged  Iter  RMS_RA" RMS_Dec"    Obs    Sel     Fit  Extrap
  ------------------------------------------------------------------------------------
  99942                  yes    11     0.32     0.28    128    126     yes     yes
  2010 TK7               yes     9     0.41     0.37    318    311     yes      no
  K25X99Z             FAILED     -        -        -      -      -       -       -

  1 of 3 object(s) produced no orbit:
    K25X99Z: no viable IOD seed

  ./out/fitted_orbits.parquet (2 rows)
  ./out/fit_summary.parquet + fit_summary.csv (3 rows)
  ./out/residuals.parquet (437 rows)

  Output: ./out/
```

The table is the command's primary human output: one line per object the
batch *attempted*, so a partially successful run reads as such rather
than as a success with a shorter orbit file. `Fit` and `Extrap` are the
two acceptability verdicts — is the fit good, and is it safe to
extrapolate forward — reported separately because they fail for
different reasons.

Three artifacts land under `--out-dir`:

- `fitted_orbits.<ext>` — one row per object that produced an orbit,
  keyed by its ADES designation. Fully re-feedable: state, covariance,
  and non-gravitational model carry straight into a follow-on
  `empyrean propagate` / `empyrean ephemeris` with no reconstruction.
  A wide fit's orbit additionally carries the **joint** — the
  off-diagonal blocks of the fitted covariance, beyond the state+Marsden
  9×9. Only parquet (the `--format` default) can hold them; `--format
  csv` and `--format json` refuse such a batch **by name** rather than
  writing it short, because a file written short reads back as a
  block-diagonal joint, which is a tighter claim than the fit made and
  leaves nothing in the round trip to signal the loss. Keep the default
  format for wide fits; a fit whose orbit carries no cross terms writes
  in whichever format you ask for.
- `fit_summary.parquet` **and** `fit_summary.csv` — one row per *input*
  object, delivered or not: `status`, convergence, iterations, observation
  and selected counts, residual RMS, reduced χ², both acceptability
  verdicts with a value and a threshold column for each gate
  (`selection_fraction`, `selected_arc_*`, `trailing_gap_*`,
  `fractional_sigma_a`), the solved width, and on failure the reason.
  Always written in both formats so a run is readable at a terminal
  without a parquet tool. A quantity that does not exist is `NaN`, never
  `0.0` — a failed object has no RMS, and a floor reading is not the same
  as no reading.
- `residuals.<ext>` — every delivered fit's per-observation rows, each
  tagged with the `object_id` it belongs to. The file carries the whole
  36-column residual surface: the `obs_id` / `object_id` join keys,
  observatory code, epoch and star catalog, the effective residual
  covariance, the full rejection attribution (`rejection_reason`,
  criterion, threshold, effective threshold, information loss), the
  influence diagnostics (`cooks_distance`, `leverage`,
  `fractional_information`), the along/cross-track decomposition, and the
  radar block. All three formats emit the same columns; a non-computable
  number is a literal `NaN` in CSV and `null` in JSON.

An object whose fit fails never removes the others and never disappears:
it gets a typed failure and a `fit_summary` row saying so. The exit code
is the batch's verdict — `0` when every object delivered, `3` when some
did, `4` when none did — so a script can tell a partial run from a clean
one without parsing stderr.

```sh
empyrean determine observations.psv --out-dir ./out
echo $?    # 3 — some objects delivered, some did not
ls out/    # fit_summary.csv  fit_summary.parquet  fitted_orbits.parquet  residuals.parquet
```

Every deselected observation carries a typed reason rather than
vanishing: `rejection_reason` is written as a name — `chi_squared`,
`sigma_clip`, `cooks_distance`, `adaptive`, `outside_arc`,
`unsupported_observatory`, `non_finite_chi2`, `missing_jacobian`, and the
rest — alongside the criterion and threshold that produced it, so a
thinned arc can be audited rather than guessed at. Observations are
weighted with the VFCC2017 preset (Vereš, Farnocchia, Chesley &
Chamberlin 2017) station floors plus nightly de-weighting, which is the
default and needs no flag.

Hard objects are the point of the batch path. A comet arc a narrower
solve cannot reconcile escalates instead of being cut short, so the
delivered fit spans the whole arc; a co-orbital object of the
2010 TK7 / 2020 XL5 class is recovered by an IOD lane that fires when the
co-orbitality gates pass, rather than falling out of the ordinary
cascade; and radar delay / Doppler rows tighten the same solve the
optical rows constrain.

### Solving for more than the state

`--solve-for` chooses which parameters differential correction recovers,
beyond the 6-element state:

- `state-only` — the 6-element Cartesian state.
- `non-grav` — state + Marsden A1/A2/A3 radial/transverse/normal coefficients.
- `dt` — state + Marsden + the cometary outgassing **time delay DT** (days).
- `amrat` — state + **SRP area-to-mass ratio AMRAT** (m²/kg).
- `non-grav-amrat` — state + Marsden + AMRAT.
- `auto` (default) — state-only, escalating to non-grav automatically on a poor fit.

`--thrust-segments <N>` additionally solves the leading `N` impulsive
**thrust Δv segments** (0 = none), up to the engine's maximum of 3. Each
solved segment is a 3-vector in the integration frame; its burn window must be
bracketed by observations, or empyrean refuses the fit rather than letting the
state quietly absorb the maneuver. An over-budget `N` is **refused**, not
clamped — silently fitting three of four requested burns would marginalize the
fourth without a word.

Underneath, each axis carries a *disposition* rather than a flag: solved,
fixed, or **considered** — not estimated, but its prior uncertainty still
reaching the posterior through its measurement partials. This CLI opens an
axis or leaves it out, so it only ever produces solved or fixed; there is no
flag for considered, because it is a distinct modelling statement rather than
a shade of "off" and wants its own surface rather than an overloading of
`--solve-for`. An axis you did not request is marginalized out and changes no
number.

DT and AMRAT are refine-path axes: each is opened only when a prior
variance is supplied for it (`--dt-variance` / `--amrat-variance`).
`determine` runs a seed solve, attaches the prior to the fitted orbit,
then refines — so `--solve-for dt` / `amrat` require their prior flags,
and a requested axis with no prior errors loudly rather than handing back
a zeroed column. Thrust segments (`--thrust-segments`) are opened instead
by bracketing the burn window with observations. Every solved axis is
differentiated analytically by the hyperdual integrator, so the partials
are exact rather than finite-differenced.

```sh
# State + Marsden + the cometary outgassing time delay. --dt-variance (days²)
# opens + priors the DT column; --dt sets the value, else the seed's is kept.
empyrean determine comet.psv --solve-for dt --dt-variance 400 --out-dir ./out

# State + SRP area-to-mass ratio. --amrat (m²/kg) seeds the SRP slot,
# --amrat-variance ((m²/kg)²) opens + priors the AMRAT column; --cr defaults to 1.0.
empyrean determine object.psv --solve-for amrat --amrat 3.0e-3 --amrat-variance 1e-8 --out-dir ./out

# State + Marsden + two solved thrust Δv-correction segments.
empyrean determine maneuvering.psv --solve-for non-grav --thrust-segments 2 --out-dir ./out
```

After the per-object table, each delivered object gets a readback of
exactly the wide axes it recovered, under its own designation. A line for
a scalar axis appears only when that axis was actually solved, so a missing
line reads as "not recovered", never a zero. Thrust is the exception, and
deliberately so: the segments are indexed by **declared** burn, so every
declared segment gets a line and an unsolved one says `not solved` rather
than printing zeros — a zero there would read as a fitted correction of
exactly zero, which is a result the fit never produced:

```text
  Object           Converged  Iter  RMS_RA" RMS_Dec"    Obs    Sel     Fit  Extrap
  ------------------------------------------------------------------------------------
  1P                     yes    14     0.38     0.35    642    629     yes     yes

  1P:
    Solved covariance width: 10
    Non-grav time delay  ΔDT = 0.0142 d
```

With three burns declared on the orbit and `--thrust-segments 2` opening the
leading two, the readback is positional across all three:

```text
  2024 ABC:
    Solved covariance width: 15
    Thrust dv[0] = [0.412, -0.089, 0.003] m/s
    Thrust dv[1] = [0.118, 0.201, -0.044] m/s
    Thrust dv[2] = not solved
```

### Tagged solved covariance

A wide fit carries a **tagged solved covariance**: the fitted-parameter
identities travel with the matrix, so you read a parameter's variance by
its slot — DT, AMRAT, or a thrust component — rather than guessing at
column order. The canonical layout is
`[state 6 | Marsden 3 | DT 1 | AMRAT 1 | thrust 3×k]`, but the width alone
is ambiguous (a width-9 solve is Marsden-only *or* one thrust segment), so
each solved axis is located by its tag and reported by name in the
readback above. Its width is also carried per object in `fit_summary`'s
`solve_for_width` column, and the fitted state covariance rides along in
`fitted_orbits.<ext>`.

### Post-OD photometry

`--photometry` runs an optional photometric fit after the orbit is solved,
recovering the absolute magnitude **H** and phase-function slope from the
observation magnitudes. Photometry has no astrometric partials, so it
never perturbs the fitted state.

The fit climbs a model ladder — **H-only → HG12 → HG1G2** (Muinonen et al.
2010) — admitting the richest model the arc's phase-angle coverage
supports, and reports the model it actually fit alongside an honest 1σ on
H:

```sh
empyrean determine apophis.psv --photometry --out-dir ./out
```

```text
  99942:
    Photometry: H = 19.234 ± 0.041  G1 = 0.150  (model HG12, chi2_r 1.02)
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
