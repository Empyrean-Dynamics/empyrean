# Orbit determination

Fit an orbit to ADES optical observations.

## End-to-end

```python
import empyrean

empyrean.initialize()

obs, radar = empyrean.read_ades("apophis_2004_2021.psv")  # optical + radar tables
fit = empyrean.determine(obs)                             # IOD + DC (optical only)

print(f"converged={fit.converged}, "
      f"χ²_red={fit.summary.reduced_chi2:.2f}, "
      f"acceptable={fit.acceptability.fit_acceptable}")
```

{func}`~empyrean.determine` runs initial orbit determination on the
observation arc — using Gauss + Herget refinement or systematic
ranging — to seed differential
correction, then iterates DC with adaptive outlier rejection.

## Re-fitting with a prior

When you already have an orbit (e.g. from SBDB) and want to fold in
new observations:

```python
prior  = empyrean.query_sbdb(["99942"])         # CometaryOrbits with covariance
new    = empyrean.query_observations(["99942"]) # MPC ADES
result = empyrean.refine(prior, new)            # ODResult
```

The prior covariance gates the Bayesian update via
{class}`~empyrean.ODConfig`'s `use_prior` toggle.

## Evaluating without fitting

To compute residuals only — no parameter update:

```python
result = empyrean.evaluate(orbits, observations)
s = result.summary

# RA·cos(δ) and Dec residuals (arcsec). Note the upstream RMS values
# already apply the cos(δ) factor — they're sky-plane residuals, not
# raw RA differences, so they stay sensible near the poles.
print(f"sky RMS:  RA·cos(δ) {s.rms_ra_arcsec:.3f}'', Dec {s.rms_dec_arcsec:.3f}''")

# Along-track / cross-track decomposition is the more diagnostic view:
# AT >> CT means a timing-like residual (gravity model, light-time);
# CT >> AT means a geometry-like residual (frame, parallax). Available
# when the propagation populated sky-motion rates.
print(f"AT/CT RMS: AT {s.rms_along_track_arcsec:.3f}'', CT {s.rms_cross_track_arcsec:.3f}''")

# Per-observation residuals + diagnostics live on result.observations
# (an ObservationResults quivr table) — filter with `.where()` to
# inspect outliers, leverage, Cook's distance, etc.
```

## Stateful sessions

For interactive workflows where you want to mask observations and
compare fits, use {class}`~empyrean.Session`:

```python
sess = empyrean.Session("apophis.psv")
fit0 = sess.refine()

# Mask by row index in the obs-time-sorted arc (i.e. position in
# `sess.observations.sort_by("epoch_mjd_tdb")`, not file order). To
# mask a specific physical observation, look it up first via
# obs_id / epoch / station and convert to the sorted index.
sess.mask(7)
fit1 = sess.refine()

diff = sess.diff(0)               # compare to initial fit
print(f"Δχ²_red = {diff.reduced_chi2_delta:+.3f}")
```

Per-observation rejection diagnostics — Cook's distance, leverage,
fractional information loss — are emitted on
{class}`~empyrean.ObservationResults` for every fit; use those to
pick which observation to mask rather than guessing by index.

## Station biases

Short-arc fits and impact monitoring benefit from solving for
per-station nuisance biases (RA / Dec offsets, optionally a station
timing bias) alongside the orbit. Enable on
{class}`~empyrean.ODConfig`:

```python
from empyrean import ODConfig, StationRaDecConfig

cfg = ODConfig(
    fit_station_biases=True,
    station_radec=StationRaDecConfig(sigma_prior_arcsec=0.5),
    # Per-observation weighting follows Vereš et al. 2017 by default.
)

fit = empyrean.determine(observations, config=cfg)

# Per-station fitted biases come back as a quivr table on the result.
biases = fit.station_biases
for row in zip(
    biases.obs_code.to_pylist(),
    biases.bias_ra_arcsec.to_pylist(),
    biases.bias_dec_arcsec.to_pylist(),
    biases.significance.to_pylist(),
):
    code, b_ra, b_dec, sig = row
    print(f"{code}  ΔRA={b_ra:+.3f}'' ΔDec={b_dec:+.3f}'' (σ={sig:.1f})")
```

Significance ≥ 3 indicates a real systematic worth keeping fitted.
The Schur-coupled marginalisation means the reported σ already
includes uncertainty inherited through the orbit fit, so you can
report `bias_ra ± sigma_ra` directly without further inflation.

## Weighting and catalog debiasing

Per-observation weights and pre-Gaia astrometric catalog debiasing are
on by default — {func}`~empyrean.determine` ships with the production
preset (Vereš, Farnocchia, Chesley et al. 2017 station floors plus
nightly de-weighting at the floor-σ policy, and EFCC2020 catalog-bias
correction at standard healpix resolution). For most workflows you
don't need to touch these.

To disable either pipeline, or to tune them:

```python
from empyrean import (
    ODConfig, WeightingConfig, WeightingLayer, WeightingLayerKind,
    WeightingPreset, SigmaPolicy, DebiasingConfig, DebiasingResolution,
)

# Uniform 1″ weighting + no catalog debiasing — the unweighted baseline
cfg_unweighted = ODConfig(
    weighting=WeightingConfig(enabled=False),
    debiasing=DebiasingConfig(enabled=False),
)

# VFC17 stations plus a per-survey override for one observatory
cfg_custom = ODConfig(
    weighting=WeightingConfig(
        preset=WeightingPreset.VFC17,
        sigma_policy=SigmaPolicy.FLOOR,
        additional_layers=[
            WeightingLayer(
                kind=WeightingLayerKind.OBSERVATORY_RULE,
                obs_code="F51",
                sigma=(0.15, 0.15),         # 1σ RA·cos(δ), Dec in arcsec
                scale=1.0,
            ),
            WeightingLayer(
                kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING,
                max_gap_days=0.5,
            ),
        ],
    ),
    debiasing=DebiasingConfig(
        enabled=True,
        resolution=DebiasingResolution.HIRES,    # ~567 MB, NSIDE=256
    ),
)
```

The default ``ODConfig().weighting`` already includes a
``NightlyDeweighting`` layer; override ``additional_layers`` only
when you want to replace that pipeline rather than extend it.

## Convergence tolerance

The default DC step-tolerance ($\Delta\mathbf{x}^\top \mathcal{N}\,\Delta\mathbf{x}$
on the parameter update) is ``1e-5`` — sigma-quality fits suitable for
impact-risk and close-approach assessment. Loosen to ``1e-3`` for
survey-grade speed:

```python
cfg = ODConfig(convergence_tol=1e-3)
```

## Fit acceptability

{class}`~empyrean.AcceptabilityReport` is the post-fit verdict:
`fit_acceptable` requires convergence + positive-definite covariance +
reduced $\chi^2$ + RMS + residual isotropy thresholds;
`extrapolation_acceptable` adds arc-coverage and
$\sigma_a / |a|$ gates for trustworthy forward propagation.
