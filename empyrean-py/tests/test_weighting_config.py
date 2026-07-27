"""Weighting config-resolution contract suite (Wave A).

Locks the four config-resolution contracts fixed after the July-2026
external weighting report, each demonstrated failing against the
pre-fix build (failure signatures recorded per test):

* **preset NONE is honored** — ``preset=NONE`` + empty
  ``additional_layers`` means uniform weighting at the user's
  ``default_sigma_arcsec``, never a silent VFC17 substitution.
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
    Origin,
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

# VFC17 assigns F51 (Pan-STARRS 1) a 0.2″ station floor; with no
# reported sigmas the preset chi2 is (1/0.2)² = 25× uniform.
_VFC17_F51_RATIO = 25.0

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
    observers = Observers.from_code("F51", epochs)
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
    VFC17 preset, whose F51 floor is 0.2″ — measured chi2 ratio to
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
    entirely (VFC17 substituted) — chi2 ratio to uniform was
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


def test_vfc17_preset_station_floor_applies(probe_orbit, f51_observations):
    """Sanity anchor: the plain VFC17 preset assigns F51 its 0.2″
    floor, so chi2 is exactly 25× uniform on unreported-sigma obs.
    (This was true pre-fix too — it pins the baseline the override
    test below must beat.)"""
    c_uniform = _chi2(probe_orbit, f51_observations, WeightingConfig(enabled=False))
    c_vfc17 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFC17, additional_layers=[]),
    )
    np.testing.assert_allclose(c_vfc17, c_uniform * _VFC17_F51_RATIO, rtol=1e-12)


def test_user_layer_beats_preset_rule_for_station(probe_orbit, f51_observations):
    """VFC17 + one additional OBSERVATORY_RULE for F51 (sigma=10″):
    the user rule wins its station — sigma resolution is
    first-match-wins and user layers go ahead of the preset chain.

    Pre-fix failure signature: the user layer was appended AFTER the
    (time-unbounded) preset rules, so it could never match — chi2 was
    identical to the plain VFC17 preset, ratio 25.000000 to uniform
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
        WeightingConfig(preset=WeightingPreset.VFC17, additional_layers=[user_rule]),
    )
    np.testing.assert_allclose(c_override, c_uniform / 100.0, rtol=1e-12)
    # And it is decisively NOT the preset floor.
    c_vfc17 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFC17, additional_layers=[]),
    )
    assert c_override < c_vfc17 / 100.0


def test_nightly_layer_position_does_not_change_station_sigmas(probe_orbit, f51_observations):
    """Regression guard for the reorder: with one observation per
    night the nightly layer is a no-op, so the production default
    (VFC17 + nightly, now prepended) must match the bare VFC17 preset
    exactly. Nightly de-weighting and scale factors are position-
    independent — only sigma-rule precedence changed."""
    c_default = _chi2(probe_orbit, f51_observations, WeightingConfig())
    c_vfc17 = _chi2(
        probe_orbit,
        f51_observations,
        WeightingConfig(preset=WeightingPreset.VFC17, additional_layers=[]),
    )
    np.testing.assert_allclose(c_default, c_vfc17, rtol=1e-12)


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
        preset=WeightingPreset.VFC17,
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
        "preset": "vfc17",
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
