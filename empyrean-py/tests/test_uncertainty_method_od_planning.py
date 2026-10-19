"""`uncertainty_method` on the OD and planning seams (0.11.0).

Closes the API-parity gap that pinned both seams to first order internally:
a Python caller can now request a method on ``ODConfig`` / ``PlanningConfig``
exactly as on ``PropagationConfig``, and the result reports what it resolved
to. The OD fit is first-order by construction — ``FIRST_ORDER`` delivered,
``AUTO`` resolved to it (and said so on ``warnings``), every richer method
refused by name. The planning fold honours each method and reports the
resolved covariance kind.

The wire-lowering tests run without an engine; the round-trip tests need
kernels (the autouse ``initialize_empyrean`` fixture skips them otherwise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from empyrean import (
    PlannedObservation,
    PlanningConfig,
    determine,
    evaluate_plan,
    read_ades,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.od.result import ODConfig
from empyrean.orbits.joint import CovarianceKind
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import Auto, SigmaPoint, UncertaintyMethod

DATA_DIR = Path(__file__).parent / "fixtures"
APOPHIS_MULTIAPP = DATA_DIR / "99942_apophis_multiapp.psv"

# Apophis at MJD 61000 TDB, barycentric ecliptic — the same hermetic state
# test_planning.py uses, so the planning round-trip needs no fetch.
_EPOCH_MJD_TDB = 61000.0
_APOPHIS_SSB_ECLIPTIC = {
    "x": -8.18903154079353157e-02,
    "y": -8.25280676309557037e-01,
    "z": 4.20312157909258355e-02,
    "vx": 1.98823903215481629e-02,
    "vy": 1.32186530066730917e-03,
    "vz": 3.99358827504333659e-04,
}


# ── Wire lowering (no engine) ────────────────────────────────────────


def test_od_config_default_lowers_to_first_order():
    """A default ODConfig carries the first-order tag — byte-identical to a
    pre-0.11.0 caller, who had no knob at all."""
    sub = ODConfig()._to_wire_dict()["uncertainty_method"]
    assert sub["tag"] == 0, sub


def test_od_config_explicit_first_order_matches_the_default():
    """Setting FIRST_ORDER explicitly lowers to the same sub-dict as the
    default — the unset path and the explicit path are one."""
    assert (
        ODConfig(uncertainty_method=UncertaintyMethod.FIRST_ORDER)._to_wire_dict()[
            "uncertainty_method"
        ]
        == ODConfig()._to_wire_dict()["uncertainty_method"]
    )


def test_od_config_lowers_each_method_to_its_tag():
    for method, tag in [
        (UncertaintyMethod.SECOND_ORDER, 1),
        (UncertaintyMethod.AUTO, 4),
        (SigmaPoint(), 2),
        (Auto(), 4),
    ]:
        sub = ODConfig(uncertainty_method=method)._to_wire_dict()["uncertainty_method"]
        assert sub["tag"] == tag, f"{method!r} → tag {sub['tag']}, expected {tag}"


def test_planning_config_default_lowers_to_first_order():
    sub = PlanningConfig()._to_wire_dict()["uncertainty_method"]
    assert sub["tag"] == 0, sub


def test_planning_config_lowers_sigma_point_params():
    cfg = PlanningConfig(uncertainty_method=SigmaPoint(n_sigma=1.0, samples_per_plane=8))
    sub = cfg._to_wire_dict()["uncertainty_method"]
    assert sub["tag"] == 2
    assert sub["sigma_n_sigma"] == 1.0
    assert sub["sigma_samples_per_plane"] == 8


# ── OD round-trip (needs kernels + fixture) ──────────────────────────


@pytest.fixture(scope="module")
def apophis_observations():
    if not APOPHIS_MULTIAPP.exists():
        pytest.skip(f"missing fixture: {APOPHIS_MULTIAPP}")
    optical, _radar = read_ades(APOPHIS_MULTIAPP)
    return optical


def test_od_resolved_method_rides_the_covariance_kind_vocabulary():
    """The OD result's ``resolved_method`` is a :class:`CovarianceKind`, decoded
    on the covariance-kind wire codes — the one vocabulary every covariance-
    bearing result is tagged in — not the uncertainty-method codes. The two
    vocabularies disagree at ``SIGMA_POINT``: it is code 5 here but code 2 in the
    method table (whose code 5 is ``GAUSSIAN_MIXTURE``). So a synthetic code-5
    result round-trips to ``SIGMA_POINT``; a revert of the OD decode to the
    method table reads code 5 as the wrong kind and breaks this pin."""
    from empyrean.orbits.joint import _KIND_BY_CODE, _KIND_TO_CODE
    from empyrean.propagation.config import _uncertainty_method_from_int

    assert _KIND_TO_CODE[CovarianceKind.SIGMA_POINT] == 5
    assert _KIND_BY_CODE[5] == CovarianceKind.SIGMA_POINT
    assert _KIND_BY_CODE[0] == CovarianceKind.LINEAR
    # The uncertainty-method table decodes the same code 5 differently —
    # the distinguishing round trip the two vocabularies disagree on.
    assert _uncertainty_method_from_int(5) != CovarianceKind.SIGMA_POINT


def test_default_od_fit_reports_first_order(apophis_observations):
    """A delivered fit names the covariance kind it resolved to: linear, the
    only kind the OD seam delivers."""
    fit = determine(apophis_observations).single()
    assert fit.resolved_method == CovarianceKind.LINEAR


def test_auto_od_fit_resolves_to_first_order_and_says_so(apophis_observations):
    """`AUTO` is delivered (not refused) and resolves to the linear covariance
    kind, reported on both ``resolved_method`` and, by name, on ``warnings``."""
    fit = determine(
        apophis_observations,
        config=ODConfig(uncertainty_method=UncertaintyMethod.AUTO),
    ).single()
    assert fit.resolved_method == CovarianceKind.LINEAR
    assert any("Auto" in w and "first-order" in w for w in fit.warnings), (
        f"the Auto→first-order resolution must be named on warnings, got {fit.warnings!r}"
    )


def test_second_order_od_is_refused_by_name(apophis_observations):
    """`SECOND_ORDER` has no honest OD implementation, so it is refused by
    name — the method and the entry point in the message — never silently
    downgraded to a first-order fit."""
    results = determine(
        apophis_observations,
        config=ODConfig(uncertainty_method=UncertaintyMethod.SECOND_ORDER),
    )
    assert results.delivered == [], "a refused method must deliver no fit"
    assert results.failures, "the refusal must surface as a per-object failure"
    messages = " | ".join(f.message for f in results.failures.values())
    assert "SecondOrder" in messages, messages
    assert "determine" in messages, messages


# ── Planning round-trip (needs kernels; hermetic orbit) ──────────────


def _plan_orbit() -> CartesianOrbits:
    cov = np.zeros((1, 6, 6))
    for k, d in enumerate([1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 1e-18]):
        cov[0, k, k] = d
    return CartesianOrbits.from_kwargs(
        orbit_id=["apophis_plan"],
        object_id=["99942"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_EPOCH_MJD_TDB]),
            x=[_APOPHIS_SSB_ECLIPTIC["x"]],
            y=[_APOPHIS_SSB_ECLIPTIC["y"]],
            z=[_APOPHIS_SSB_ECLIPTIC["z"]],
            vx=[_APOPHIS_SSB_ECLIPTIC["vx"]],
            vy=[_APOPHIS_SSB_ECLIPTIC["vy"]],
            vz=[_APOPHIS_SSB_ECLIPTIC["vz"]],
            frame="ecliptic_j2000",
            origin=["SSB"],
            covariance=CartesianCovariance.from_matrix(cov),
        ),
    )


def _plan() -> list[PlannedObservation]:
    return [
        PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2)),
        PlannedObservation.optical(_EPOCH_MJD_TDB + 12.0, "568", (0.3, 0.3)),
    ]


@pytest.mark.validation
def test_default_plan_resolves_to_linear():
    plan = evaluate_plan(_plan_orbit(), _plan())
    assert plan.resolved_method == CovarianceKind.LINEAR


@pytest.mark.validation
def test_second_order_plan_resolves_to_second_order():
    """The second-order fold reports its resolved kind, and its transported
    metrics withhold the epoch-variant log-det under the named reason."""
    plan = evaluate_plan(
        _plan_orbit(),
        _plan(),
        PlanningConfig(uncertainty_method=UncertaintyMethod.SECOND_ORDER),
    )
    assert plan.resolved_method == CovarianceKind.SECOND_ORDER
    assert plan.metrics.posterior().log_det_withheld[0].as_py() == "second_order_moment"


@pytest.mark.validation
def test_sigma_point_plan_resolves_to_sigma_point():
    plan = evaluate_plan(
        _plan_orbit(),
        _plan(),
        PlanningConfig(uncertainty_method=SigmaPoint()),
    )
    assert plan.resolved_method == CovarianceKind.SIGMA_POINT
