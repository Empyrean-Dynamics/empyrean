# Data setup

empyrean propagation needs SPICE kernels (planetary ephemerides, Earth
orientation, asteroid perturbers, observatory codes). A fresh
`pip install empyrean` pulls all of them as PyPI dependencies — no
network fetch on first call.

## What ships via PyPI

| Package                        | File                              | Size    |
|--------------------------------|-----------------------------------|---------|
| `naif-de440`                   | `de440.bsp`                       | 114 MB  |
| `jpl-small-bodies-de441-n16`   | `sb441-n16.bsp`                   | 616 MB  |
| `naif-eop-high-prec`           | `earth_latest_high_prec.bpc`      | 5 MB    |
| `naif-eop-historical`          | `earth_620120_*.bpc`              | 8 MB    |
| `naif-eop-predict`             | `earth_*_predict.bpc`             | 9 MB    |
| `mpc-obscodes`                 | `obscodes_extended.json`          | <1 MB   |

All are maintained by the
[B612 Asteroid Institute](https://b612.ai).

```{note}
**SB441 vs DE441 vs DE440.** SB441 is JPL's asteroid mass-perturber
kernel family, released paired with DE441 (the long-arc planetary
ephemeris). empyrean ships SB441 alongside **DE440** (the
shorter-span DE441 sibling, identical in dynamics within its support
window). Using SB441 with DE440 is standard practice in NEO work and
the difference is below the 16-perturber accuracy floor; the
``jpl-small-bodies-de441-n16`` package name reflects SB441's
*release pairing*, not a constraint on which planetary kernel you
load it with.
```

## Discovery and caching

{func}`empyrean.initialize` checks whether each B612 package is
installed, then symlinks the files into a B612-cache directory under
the OS-appropriate XDG-compliant data root:

| Platform | Cache directory                                            |
|----------|------------------------------------------------------------|
| Linux    | `$XDG_DATA_HOME/empyrean/b612-cache/`                      |
| macOS    | `~/Library/Application Support/empyrean/b612-cache/`       |
| Windows  | `%APPDATA%\empyrean\b612-cache\`                           |

Override the data root with the `EMPYREAN_DATA_DIR` environment
variable:

```bash
export EMPYREAN_DATA_DIR=/scratch/shared/empyrean
```

## Bundled assets

`gm_de440.tpc` (gravitational parameters) ships inside the empyrean
wheel itself — it isn't on PyPI separately.

## Lazy-fetched extras

A handful of smaller kernels not packaged on PyPI are downloaded on
demand and cached under the same XDG data dir:

| File                              | When fetched                                          | Source                                                    |
|-----------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `moon_pa_de440_200625.bpc`        | First {func}`empyrean.initialize` (~30 MB)            | `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/` |
| Spacecraft SPK (JWST, Gaia, HST)  | Only when an observation cites that observatory code  | `https://naif.jpl.nasa.gov/pub/naif/…`                    |

For air-gapped or offline ops, mirror these into your
``EMPYREAN_DATA_DIR`` ahead of time and ``initialize()`` will skip
the network fetch.

## Time scales

empyrean does not use a SPICE leap-second kernel. UTC↔TDB conversion —
leap seconds plus the TDB−TT periodic terms — is built into the engine,
so {meth}`Epochs.to_tdb` and the other scale conversions work with no
separate kernel install.

## Bypassing PyPI

If you have your own kernel set:

```python
empyrean.initialize(data_dir="/path/to/my/kernels")
```

The directory must contain files under empyrean's expected
filenames. {func}`empyrean.default_data_dir` returns the path where
empyrean would put them by default.

## Network-query caches

The {func}`~empyrean.query_sbdb`,
{func}`~empyrean.query_horizons`, and
{func}`~empyrean.query_observations` helpers all cache JSON / PSV
responses on disk so repeat calls don't hit the upstream service. By
default, responses go under ``$EMPYREAN_CACHE_DIR/<service>`` (or
``~/.empyrean/cache/<service>`` if ``EMPYREAN_CACHE_DIR`` is unset).

```python
# Default — cache under EMPYREAN_CACHE_DIR.
orbits = empyrean.query_sbdb(["99942"])

# Force a fresh fetch (no cache read or write):
orbits = empyrean.query_sbdb(["99942"], cache_dir=False)

# Pin to a specific cache directory:
orbits = empyrean.query_sbdb(["99942"], cache_dir="/scratch/sbdb-cache")
```

Cache directories are safe to delete to force a refresh; SBDB and
the MPC apply rate limits that the cache helps avoid.
