"""Weighting config-resolution contract suite (Wave A).

Locks the five config-resolution contracts fixed after the July-2026
external weighting report, each demonstrated failing against the
pre-fix build (failure signatures recorded per test):

* **preset NONE is honored** — ``preset=NONE`` + empty
  ``additional_layers`` means uniform weighting at the user's
  ``default_sigma_arcsec``, never a silent VFCC2017 substitution.
* **user layers beat preset rules** — an additional observatory rule
  for a preset-covered station wins that station (first-match-wins;
  the preset is the fallback).
* **strict per-kind layer validation** — a NIGHTLY_DEWEIGHTING layer
  carrying observatory scoping fields raises instead of silently
  ignoring them; duplicate nightly layers raise; malformed
  scale / obs_code values raise naming the layer.
* **defense-in-depth sigma validation** — non-finite / non-positive
  layer sigmas raise at the Python layer (engine-side value
  validation ships separately with the next scott release).
* **session / one-shot parity** — a :class:`Session` resolves the
  weighting and debiasing configuration through the same builder as
  ``determine`` / ``evaluate`` / ``refine``, so the same observations
  and the same configuration produce the same fit on both surfaces,
  and every layer-validation error above is raised by the session
  constructor too.

The fixture is hermetic: a hardcoded heliocentric Cartesian state
(the Apophis state also used by ``test_no_silent_drops``) and
synthetic F51 observations generated from the orbit's own predicted
ephemeris — no network, no fixture files. The observations carry **no
reported sigmas**, so every chi-squared below is set entirely by the
weighting rules, and chi2 ratios between configs are exact:
chi2 scales as 1/sigma^2.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    Epochs,
    Origin,
    Session,
    determine,
    evaluate,
    generate_ephemeris,
)
from empyrean.observers.observers import Observers
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.result import (
    DebiasingConfig,
    ODConfig,
    WeightingConfig,
    WeightingLayer,
    WeightingLayerKind,
    WeightingPreset,
)
from empyrean.propagation.config import ForceModelTier

# Hardcoded Apophis heliocentric state (ecliptic J2000, AU / AU day⁻¹)
# — same hermetic fixture as test_no_silent_drops.
_STATE = {
    "epoch": 61000.0,
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}

# VFCC2017 assigns F51 (Pan-STARRS 1) a 0.2″ station floor; with no
# reported sigmas the preset chi2 is (1/0.2)² = 25× uniform.
_VFCC2017_F51_RATIO = 25.0

_MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


def _mjd_to_iso(mjd: float) -> str:
    dt = _MJD0 + timedelta(days=mjd)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@pytest.fixture(scope="module")
def probe_orbit() -> CartesianOrbits:
    coords = CartesianCoordinates.from_kwargs(
        epoch=[_STATE["epoch"]],
        x=[_STATE["x"]],
        y=[_STATE["y"]],
        z=[_STATE["z"]],
        vx=[_STATE["vx"]],
        vy=[_STATE["vy"]],
        vz=[_STATE["vz"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(orbit_id=["weighting_probe"], coordinates=coords)


@pytest.fixture(scope="module")
def f51_observations(probe_orbit) -> ADESObservations:
    """Six synthetic F51 observations, one per night, built from the
    orbit's own predicted RA/Dec plus a fixed 1″ Dec offset.

    The offset guarantees a non-zero residual identical across every
    weighting config; no reported rms columns, so layer/default sigmas
    fully determine the weights. One observation per night keeps
    nightly de-weighting a no-op where it appears.
    """
    epochs = [61001.1 + i for i in range(6)]
    observers = Observers.from_code("F51", Epochs.from_mjd(epochs, scale="tdb"))
    eph = generate_ephemeris(probe_orbit, observers, force_model=ForceModelTier.APPROXIMATE)
    ra = eph.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    dec = eph.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    n = len(ra)
    assert n == len(epochs)
    return ADESObservations.from_kwargs(
        stn=["F51"] * n,
        obs_time=[_mjd_to_iso(e) for e in epochs],
        ra=ra.tolist(),
        dec=(dec + 1.0 / 3600.0).tolist(),
        rms_ra=[None] * n,
        rms_dec=[None] * n,
    )


@pytest.fixture(scope="module")
def fittable_f51_observations(probe_orbit) -> ADESObservations:
    """The same synthetic F51 arc, long enough to fit from scratch.

    Section (e) compares *fits*, not evaluations, so the arc has to
    carry enough curvature for IOD to converge: 15 observations across
    30 days instead of the 6-night evaluation arc above. Everything
    else is identical — the orbit's own predicted RA/Dec plus a fixed
    1″ Dec offset, and no reported rms columns, so the weighting rules
    alone set the weights.

    ``ast_cat`` is populated deliberately: catalog debiasing is a no-op
    for rows without a star catalog, so a debiasing-parity test built on
    a catalog-less fixture passes no matter what the session does with
    the debiasing config. With UCAC4 the correction is live and the test
    can actually fail on the axis it names.
    """
    epochs = [61001.1 + 30.0 * i / 14 for i in range(15)]
    observers = Observers.from_code("F51", Epochs.from_mjd(epochs, scale="tdb"))
    eph = generate_ephemeris(probe_orbit, observers, force_model=ForceModelTier.APPROXIMATE)
    ra = eph.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    dec = eph.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    n = len(ra)
    assert n == len(epochs)
    return ADESObservations.from_kwargs(
        stn=["F51"] * n,
        obs_time=[_mjd_to_iso(e) for e in epochs],
        ra=ra.tolist(),
        dec=(dec + 1.0 / 3600.0).tolist(),
        rms_ra=[None] * n,
        rms_dec=[None] * n,
        trk_sub=["probe"] * n,
        ast_cat=["UCAC4"] * n,
    )


def _chi2(orbit, obs, weighting: WeightingConfig) -> float:
    cfg = ODConfig(
        force_model=ForceModelTier.APPROXIMATE,
        weighting=weighting,
        debiasing=DebiasingConfig(enabled=False),
    )
    result = evaluate(orbit, obs, config=cfg)
    chi2 = result.summary.chi2
    assert np.isfinite(chi2)
    return chi2


class _RawWeightingODConfig(ODConfig):
    """ODConfig whose weighting wire dict is injected raw.

    Bypasses the dataclass-level validation so the tests can prove the
    deeper defense layers (PyO3 binding, Rust wrapper, C ABI parser)
    reject malformed layers independently.
    """

    def __init__(self, raw_weighting, **kwargs):
        super().__init__(**kwargs)
        self._raw_weighting = raw_weighting

    def _to_wire_dict(self):
        wire = super()._to_wire_dict()
        wire["weighting"] = self._raw_weighting
        return wire


def _eval_raw(orbit, obs, raw_weighting):
    cfg = _RawWeightingODConfig(
        raw_weighting,
        force_model=ForceModelTier.APPROXIMATE,
        debiasing=DebiasingConfig(enabled=False),
    )
    return evaluate(orbit, obs, config=cfg)


# ── (a) preset NONE is honored ───────────────────────────────────────


def test_preset_none_empty_layers_equals_uniform_weighting(probe_orbit, f51_observations):
    """preset=NONE + additional_layers=[] at default_sigma=1″ is
    bit-identical to weighting disabled (uniform 1″).

    Pre-fix failure signature: the conversion silently substituted the
    VFCC2017 preset, whose F51 floor is 0.2″ — measured chi2 ratio to
    uniform was 25.000000 (should be 1.0).
    """
    c_uniform = _chi2(probe_orbit, f51_observations, WeightingConfig(enabled=False))
    c_none = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(
            preset=WeightingPreset.NONE,
            additional_layers=[],
            default_sigma_arcsec=1.0,
        ),
    )
    assert c_uniform > 0.0
    np.testing.assert_allclose(c_none, c_uniform, rtol=1e-12)


def test_preset_none_uses_user_default_sigma(probe_orbit, f51_observations):
    """preset=NONE + additional_layers=[] honors the user's
    default_sigma_arcsec: doubling sigma quarters chi2 exactly.

    Pre-fix failure signature: default_sigma_arcsec was ignored
    entirely (VFCC2017 substituted) — chi2 ratio to uniform was
    25.000000 for BOTH sigma=1 and sigma=2 (should be 1.0 and 0.25).
    """
    c_uniform = _chi2(probe_orbit, f51_observations, WeightingConfig(enabled=False))
    c_none2 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(
            preset=WeightingPreset.NONE,
            additional_layers=[],
            default_sigma_arcsec=2.0,
        ),
    )
    np.testing.assert_allclose(c_none2, c_uniform / 4.0, rtol=1e-12)


# ── (b) user layers beat preset rules ────────────────────────────────


def test_vfcc2017_preset_station_floor_applies(probe_orbit, f51_observations):
    """Sanity anchor: the plain VFCC2017 preset assigns F51 its 0.2″
    floor, so chi2 is exactly 25× uniform on unreported-sigma obs.
    (This was true pre-fix too — it pins the baseline the override
    test below must beat.)"""
    c_uniform = _chi2(probe_orbit, f51_observations, WeightingConfig(enabled=False))
    c_vfcc2017 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFCC2017, additional_layers=[]),
    )
    np.testing.assert_allclose(c_vfcc2017, c_uniform * _VFCC2017_F51_RATIO, rtol=1e-12)


def test_user_layer_beats_preset_rule_for_station(probe_orbit, f51_observations):
    """VFCC2017 + one additional OBSERVATORY_RULE for F51 (sigma=10″):
    the user rule wins its station — sigma resolution is
    first-match-wins and user layers go ahead of the preset chain.

    Pre-fix failure signature: the user layer was appended AFTER the
    (time-unbounded) preset rules, so it could never match — chi2 was
    identical to the plain VFCC2017 preset, ratio 25.000000 to uniform
    (should be 0.010000 = (1/10)²).
    """
    c_uniform = _chi2(probe_orbit, f51_observations, WeightingConfig(enabled=False))
    user_rule = WeightingLayer(
        kind=WeightingLayerKind.OBSERVATORY_RULE,
        obs_code="F51",
        sigma=(10.0, 10.0),
    )
    c_override = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFCC2017, additional_layers=[user_rule]),
    )
    np.testing.assert_allclose(c_override, c_uniform / 100.0, rtol=1e-12)
    # And it is decisively NOT the preset floor.
    c_vfcc2017 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFCC2017, additional_layers=[]),
    )
    assert c_override < c_vfcc2017 / 100.0


def test_nightly_layer_position_does_not_change_station_sigmas(probe_orbit, f51_observations):
    """Regression guard for the reorder: with one observation per
    night the nightly layer is a no-op, so the production default
    (VFCC2017 + nightly, now prepended) must match the bare VFCC2017 preset
    exactly. Nightly de-weighting and scale factors are position-
    independent — only sigma-rule precedence changed."""
    c_default = _chi2(probe_orbit, f51_observations, WeightingConfig())
    c_vfcc2017 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFCC2017, additional_layers=[]),
    )
    np.testing.assert_allclose(c_default, c_vfcc2017, rtol=1e-12)


# ── (c) strict per-kind layer validation ─────────────────────────────


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("obs_code", "F51"),
        ("sigma", (0.5, 0.5)),
        ("start_epoch_mjd_tdb", 61000.0),
        ("end_epoch_mjd_tdb", 61010.0),
        ("scale", 2.0),
    ],
)
def test_nightly_layer_rejects_observatory_scoping_fields(field, value):
    """A NIGHTLY_DEWEIGHTING layer carrying any ObservatoryRule field
    raises at construction — nightly de-weighting cannot be scoped by
    station or time range.

    Pre-fix failure signature: every one of these layers was accepted
    and the field silently ignored (measured: NIGHTLY with
    obs_code="F51" produced chi2 identical to an unscoped nightly
    layer — ratio 1.000000 to uniform on a one-obs-per-night arc).
    """
    with pytest.raises(ValueError, match="NIGHTLY_DEWEIGHTING"):
        WeightingLayer(kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING, **{field: value})


def test_two_nightly_layers_rejected(probe_orbit, f51_observations):
    """More than one NIGHTLY_DEWEIGHTING layer raises — duplicates
    compound the per-night 1/sqrt(N) de-weighting multiplicatively.

    Pre-fix failure signature: two nightly layers were accepted
    silently.
    """
    wcfg = WeightingConfig(
        preset=WeightingPreset.VFCC2017,
        additional_layers=[
            WeightingLayer(kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING),
            WeightingLayer(kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING, max_gap_days=0.3),
        ],
    )
    with pytest.raises(ValueError, match="NIGHTLY_DEWEIGHTING"):
        _chi2(probe_orbit, f51_observations, wcfg)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_observatory_rule_scale_values_rejected(bad_scale):
    """scale must be finite and > 0; the error names the layer's
    station.

    Pre-fix failure signatures: scale=0 and scale=-1 were silently
    clamped to 1.0 (chi2 ratio 1.000000 — the "scaled" rule behaved
    as unscaled); scale=inf sailed through to infinite weights and a
    chi2 of -0.000000.
    """
    with pytest.raises(ValueError, match="scale must be finite and > 0"):
        WeightingLayer(
            kind=WeightingLayerKind.OBSERVATORY_RULE,
            obs_code="F51",
            sigma=(1.0, 1.0),
            scale=bad_scale,
        )


@pytest.mark.parametrize(
    "bad_code",
    [
        "F51 junk",  # overlong AND whitespace
        "PS1X5",  # overlong, would truncate to a DIFFERENT valid-looking code
        " F51",  # leading whitespace
        "F51 ",  # trailing whitespace
        "É51",  # non-ASCII
        "",  # empty
    ],
)
def test_malformed_obs_codes_rejected(bad_code):
    """obs_code must fit the 4-byte MPC field: printable ASCII, no
    whitespace, non-empty. Matching is exact and case-sensitive.

    Pre-fix failure signature: "F51 junk" was silently truncated to 4
    bytes ("F51 ") by the wrapper and trimmed to "F51" by the C
    parser — the junk rule MATCHED F51 (measured chi2 ratio 0.010000
    from a sigma=10″ rule the caller believed was inert).
    """
    with pytest.raises(ValueError, match="obs_code"):
        WeightingLayer(
            kind=WeightingLayerKind.OBSERVATORY_RULE,
            obs_code=bad_code,
            sigma=(10.0, 10.0),
        )


# ── (d) defense-in-depth sigma validation at the Python layer ────────


@pytest.mark.parametrize("bad_sigma", [(-1.0, 1.0), (float("inf"), 1.0), (0.0, 1.0)])
def test_sigma_values_rejected_at_python_layer(bad_sigma):
    """Non-finite / non-positive layer sigmas raise at the dataclass —
    the wheel must never hand the FFI a NaN/infinite weight, even
    before the engine-side layer-sigma validation ships.

    Pre-fix failure signature: sigma=(-1, 1) was accepted; the sign
    vanished in squaring, so the rule behaved exactly like (1, 1)
    (chi2 ratio 1.000000) — a silent sign-error absorber.
    """
    with pytest.raises(ValueError, match="sigma must be finite and > 0"):
        WeightingLayer(
            kind=WeightingLayerKind.OBSERVATORY_RULE,
            obs_code="F51",
            sigma=bad_sigma,
        )


# ── Defense layers beneath the dataclass (raw wire dicts) ────────────
#
# The dataclass raises first on the public path; these prove the PyO3
# binding, Rust wrapper, and C parser reject the same malformed layers
# independently when the dataclass is bypassed.


def test_raw_wire_nightly_scoping_rejected_at_binding(probe_orbit, f51_observations):
    """The PyO3 binding rejects a nightly layer dict carrying a sigma
    key (strict per-kind validation, layer index in the message)."""
    raw = {
        "enabled": True,
        "preset": "none",
        "additional_layers": [{"kind": "nightly_deweighting", "sigma": [1.0, 1.0]}],
    }
    with pytest.raises(ValueError, match=r"layer 0 \(nightly_deweighting\)"):
        _eval_raw(probe_orbit, f51_observations, raw)


def test_raw_wire_overlong_obs_code_rejected_at_wrapper(probe_orbit, f51_observations):
    """The Rust wrapper rejects an obs_code longer than the 4-byte
    C-ABI field instead of truncating (pre-fix: .take(4))."""
    raw = {
        "enabled": True,
        "preset": "none",
        "additional_layers": [
            {"kind": "observatory_rule", "obs_code": "PS1X5", "sigma": [1.0, 1.0]}
        ],
    }
    with pytest.raises(RuntimeError, match="longer than the 4-byte"):
        _eval_raw(probe_orbit, f51_observations, raw)


def test_raw_wire_duplicate_nightly_rejected_at_engine_boundary(probe_orbit, f51_observations):
    """The C parser rejects a chain with two nightly layers even when
    the Python-level count check is bypassed."""
    raw = {
        "enabled": True,
        "preset": "vfcc2017",
        "additional_layers": [
            {"kind": "nightly_deweighting", "max_gap_days": 0.5},
            {"kind": "nightly_deweighting", "max_gap_days": 0.3},
        ],
    }
    with pytest.raises(RuntimeError, match="NIGHTLY_DEWEIGHTING"):
        _eval_raw(probe_orbit, f51_observations, raw)


def test_raw_wire_scale_zero_rejected_at_engine_boundary(probe_orbit, f51_observations):
    """The C parser rejects scale=0 (pre-fix: silently clamped to
    1.0) when the dataclass is bypassed."""
    raw = {
        "enabled": True,
        "preset": "none",
        "additional_layers": [
            {
                "kind": "observatory_rule",
                "obs_code": "F51",
                "sigma": [1.0, 1.0],
                "scale": 0.0,
            }
        ],
    }
    with pytest.raises(RuntimeError, match="scale must be finite and > 0"):
        _eval_raw(probe_orbit, f51_observations, raw)


def test_raw_wire_bad_sigma_rejected_at_binding(probe_orbit, f51_observations):
    """The PyO3 binding re-checks sigma values for raw-dict callers."""
    raw = {
        "enabled": True,
        "preset": "none",
        "additional_layers": [
            {"kind": "observatory_rule", "obs_code": "F51", "sigma": [float("inf"), 1.0]}
        ],
    }
    with pytest.raises(ValueError, match="sigma must be finite and > 0"):
        _eval_raw(probe_orbit, f51_observations, raw)


# ── (e) session / one-shot parity ────────────────────────────────────
#
# Everything above runs through the one-shot surface. The session
# constructor used to parse the OD config with its own partial parser
# that read the force model and four convergence knobs and nothing
# else, so a session caller's weighting and debiasing configuration was
# discarded wholesale — no diagnostic, and none of the contracts above
# applied. Both surfaces now share a single OD-config builder; these
# tests pin that they cannot drift apart again.


def _od_config(weighting: WeightingConfig, *, debiasing_enabled: bool = False) -> ODConfig:
    return ODConfig(
        force_model=ForceModelTier.APPROXIMATE,
        weighting=weighting,
        debiasing=DebiasingConfig(enabled=debiasing_enabled),
    )


def _uniform(sigma: float) -> WeightingConfig:
    return WeightingConfig(
        preset=WeightingPreset.NONE,
        additional_layers=[],
        default_sigma_arcsec=sigma,
    )


def _state(result) -> np.ndarray:
    c = result.orbit.coordinates
    return np.array(
        [
            c.x.to_numpy(zero_copy_only=False)[0],
            c.y.to_numpy(zero_copy_only=False)[0],
            c.z.to_numpy(zero_copy_only=False)[0],
            c.vx.to_numpy(zero_copy_only=False)[0],
            c.vy.to_numpy(zero_copy_only=False)[0],
            c.vz.to_numpy(zero_copy_only=False)[0],
        ]
    )


def _assert_same_fit(session_fit, one_shot_fit) -> None:
    """Both surfaces ran the same fit: same chi2, same residual
    statistics, same fitted state."""
    assert np.isfinite(one_shot_fit.summary.chi2)
    assert one_shot_fit.summary.chi2 > 0.0
    np.testing.assert_allclose(session_fit.summary.chi2, one_shot_fit.summary.chi2, rtol=1e-12)
    np.testing.assert_allclose(
        session_fit.summary.reduced_chi2, one_shot_fit.summary.reduced_chi2, rtol=1e-12
    )
    np.testing.assert_allclose(
        session_fit.summary.rms_combined_arcsec,
        one_shot_fit.summary.rms_combined_arcsec,
        rtol=1e-12,
    )
    assert session_fit.summary.num_obs == one_shot_fit.summary.num_obs
    np.testing.assert_allclose(_state(session_fit), _state(one_shot_fit), rtol=1e-12)

    # The OUTPUT surface, not just the fit. The session path used to
    # populate a hand-written subset of the result struct, so it returned
    # an all-zero covariance (which reads as infinite precision, the
    # worst possible way to be wrong), NaN non-gravitational parameters,
    # a solve_for_used that disagreed with the fit that had just run, and
    # `converged` read off a zeroed acceptability block. Both surfaces
    # now go through one writer.
    one_shot_cov = np.asarray(one_shot_fit.covariance, dtype=float)
    session_cov = np.asarray(session_fit.covariance, dtype=float)
    assert np.any(one_shot_cov != 0.0), "one-shot covariance is degenerate; fixture is not fitting"
    assert np.all(np.isfinite(session_cov))
    assert np.any(session_cov != 0.0), (
        "session covariance is all zeros — output surface not written"
    )
    np.testing.assert_allclose(session_cov, one_shot_cov, rtol=1e-12)
    assert session_fit.converged == one_shot_fit.converged
    assert session_fit.solve_for_used == one_shot_fit.solve_for_used


@pytest.mark.parametrize(
    ("label", "weighting"),
    [
        ("preset_none_uniform_1as", _uniform(1.0)),
        ("preset_vfcc2017", WeightingConfig(preset=WeightingPreset.VFCC2017, additional_layers=[])),
        (
            "vfcc2017_plus_user_f51_rule",
            WeightingConfig(
                preset=WeightingPreset.VFCC2017,
                additional_layers=[
                    WeightingLayer(
                        kind=WeightingLayerKind.OBSERVATORY_RULE,
                        obs_code="F51",
                        sigma=(10.0, 10.0),
                    )
                ],
            ),
        ),
    ],
)
def test_session_fit_matches_one_shot_fit(label, weighting, fittable_f51_observations):
    """The same observations and the same weighting configuration give
    the same fit through ``Session.refine`` as through ``determine``.

    Pre-fix failure signature (``preset_none_uniform_1as``): the session
    constructor never read the weighting config, so the session ran the
    engine default (VFCC2017, whose F51 floor is 0.2″) while the one-shot
    ran the requested uniform 1″ — one-shot chi2 = 0.00974934879161377,
    session chi2 = 0.24373371932879223, ratio 25.000000 (should be
    1.000000). ``vfcc2017_plus_user_f51_rule`` failed the same way with
    one-shot chi2 = 9.91897809191094e-05 vs the same pinned session
    value (the user layer was discarded along with everything else);
    ``preset_vfcc2017`` passed pre-fix only because it happens to be the
    engine default the session silently substituted.
    """
    cfg = _od_config(weighting)
    one_shot = determine(fittable_f51_observations, config=cfg).single()
    session_fit = Session(fittable_f51_observations, config=cfg).refine()
    _assert_same_fit(session_fit, one_shot)


def test_session_weighting_change_changes_the_session_fit(fittable_f51_observations):
    """A weighting change actually moves the session result.

    Guards against a routing that compiles but still discards the
    config: with uniform weighting and no reported sigmas, chi2 scales
    as 1/sigma², so quadrupling the default sigma must drop chi2 by
    ~16× (not exactly, because the fitted state moves slightly with the
    weights).

    Pre-fix failure signature: sigma=1″ and sigma=4″ both produced
    chi2 = 0.24373371932879223 — bit-identical, ratio 1.000000, the
    VFCC2017 default the session substituted for both.
    """
    obs = fittable_f51_observations
    c1 = Session(obs, config=_od_config(_uniform(1.0))).refine().summary.chi2
    c4 = Session(obs, config=_od_config(_uniform(4.0))).refine().summary.chi2
    assert c1 > 0.0
    np.testing.assert_allclose(c4 / c1, 1.0 / 16.0, rtol=0.02)


def test_session_honors_debiasing_config(fittable_f51_observations):
    """The debiasing decision rides the same builder: disabling
    debiasing on a session gives the one-shot disabled-debiasing fit.

    Pre-fix failure signature: the session ignored ``debiasing`` along
    with ``weighting`` — session chi2 was 0.24373371932879223 whether
    debiasing was enabled or disabled, against a one-shot value of
    0.00974934879161377 for both.
    """
    obs = fittable_f51_observations
    for enabled in (False, True):
        cfg = _od_config(_uniform(1.0), debiasing_enabled=enabled)
        _assert_same_fit(Session(obs, config=cfg).refine(), determine(obs, config=cfg).single())


@pytest.mark.parametrize(
    ("case", "raw", "message"),
    [
        (
            "scale_zero",
            {
                "enabled": True,
                "preset": "none",
                "additional_layers": [
                    {
                        "kind": "observatory_rule",
                        "obs_code": "F51",
                        "sigma": [1.0, 1.0],
                        "scale": 0.0,
                    }
                ],
            },
            "scale must be finite and > 0",
        ),
        (
            "duplicate_nightly",
            {
                "enabled": True,
                "preset": "vfcc2017",
                "additional_layers": [
                    {"kind": "nightly_deweighting", "max_gap_days": 0.5},
                    {"kind": "nightly_deweighting", "max_gap_days": 0.3},
                ],
            },
            "NIGHTLY_DEWEIGHTING",
        ),
    ],
)
def test_session_constructor_rejects_invalid_layer(case, raw, message, fittable_f51_observations):
    """An invalid weighting layer is rejected by the **session
    constructor**, as a typed error carrying the parser's reason —
    not swallowed, and not deferred to the first refine.

    Pre-fix failure signature: both constructors returned a live
    session with no error at all, and the subsequent refine returned
    the substituted-default chi2 = 0.24373371932879223.
    """
    cfg = _RawWeightingODConfig(
        raw,
        force_model=ForceModelTier.APPROXIMATE,
        debiasing=DebiasingConfig(enabled=False),
    )
    with pytest.raises(RuntimeError, match=message):
        Session(fittable_f51_observations, config=cfg)
