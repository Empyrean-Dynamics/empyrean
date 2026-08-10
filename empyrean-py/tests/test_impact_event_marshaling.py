"""Impact / possible-impact event marshaling — bd empyrean-h7t1g + empyrean-snzy0.

Two distribution-output-contract fixes on the ``_build_events`` marshaler,
driven by a synthetic flat event dict (no propagation, no data):

* h7t1g: ``PossibleImpacts`` absent second-order / AGM / MC probabilities must
  marshal to Arrow **null**, matching ``compute_impact_probabilities``' own
  encoding — not NaN, which made ``row.ip_agm is not None`` a false positive on
  every event row.
* snzy0: ``Impacts`` must carry ``relative_velocity_au_day`` (the impact speed
  the flat C-ABI surface already ships), NaN -> null where unresolved.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from empyrean.propagation.propagate import _build_events


def _events(**cols: object) -> dict[str, object]:
    """Wrap per-row event columns into the flat result dict ``_build_events`` reads."""
    return {"events": cols}


def test_possible_impact_absent_probabilities_are_null_not_nan() -> None:
    """PROVING (pre-fix passed NaN straight through into the nullable columns).

    ``ip_second_order`` / ``nonlinearity`` / ``ip_agm`` / ``ip_mc`` are NaN on
    the flat row when the matching method did not run; they must come back as
    null so ``is None`` is a true test. ``ip_linear`` / ``effective_radius`` /
    ``sigma_distance`` are always populated and stay dense.
    """
    ev = _events(
        orbit_ids=["a"],
        object_ids=[""],
        event_types=["possible_impact"],
        bodies=["Earth"],
        epochs=[61000.0],
        distance_au=[1.0e-3],
        distance_km=[1.5e5],
        relative_velocity_au_day=[0.01],
        ip_linear=[0.5],
        effective_radius_au=[4.3e-5],
        effective_radius_km=[6371.0],
        sigma_distance_au=[1.0e-4],
        # Absent (method did not run) -> NaN on the flat row.
        ip_second_order=[np.nan],
        nonlinearity=[np.nan],
        ip_agm=[np.nan],
        ip_mc=[np.nan],
    )
    pi = _build_events(ev).possible_impacts
    assert len(pi) == 1
    # The four nullable probability columns must be null, not NaN.
    assert pi.ip_second_order.to_pylist() == [None]
    assert pi.nonlinearity.to_pylist() == [None]
    assert pi.ip_agm.to_pylist() == [None]
    assert pi.ip_mc.to_pylist() == [None]
    # The always-populated columns are untouched.
    assert pi.ip_linear.to_pylist() == [0.5]
    assert pi.sigma_distance_au.to_pylist() == [pytest.approx(1.0e-4)]


def test_possible_impact_present_probabilities_survive() -> None:
    """REGRESSION GUARD: a finite probability must not be nulled."""
    ev = _events(
        orbit_ids=["a"],
        object_ids=[""],
        event_types=["possible_impact"],
        bodies=["Earth"],
        epochs=[61000.0],
        distance_au=[1.0e-3],
        distance_km=[1.5e5],
        relative_velocity_au_day=[0.01],
        ip_linear=[0.5],
        effective_radius_au=[4.3e-5],
        effective_radius_km=[6371.0],
        sigma_distance_au=[1.0e-4],
        ip_second_order=[0.42],
        nonlinearity=[0.1],
        ip_agm=[0.44],
        ip_mc=[np.nan],  # MC not run -> still null
    )
    pi = _build_events(ev).possible_impacts
    assert pi.ip_second_order.to_pylist() == [pytest.approx(0.42)]
    assert pi.ip_agm.to_pylist() == [pytest.approx(0.44)]
    assert pi.ip_mc.to_pylist() == [None]


def test_impacts_carry_relative_velocity_null_where_unresolved() -> None:
    """PROVING (pre-fix ``Impacts`` had no velocity column at all).

    The impact speed rides the flat schema's ``relative_velocity_au_day``
    (same column periapses / atmospheric entries read); it must appear on the
    typed ``Impacts`` table, NaN -> null where the ABI did not resolve one.
    """
    ev = _events(
        orbit_ids=["hit", "graze"],
        object_ids=["", ""],
        event_types=["impact", "impact"],
        bodies=["Moon", "Earth"],
        epochs=[61000.0, 61001.0],
        distance_au=[0.0, 0.0],
        distance_km=[0.0, 0.0],
        # First impact resolved a speed (~2.43 km/s ≈ 0.0014 AU/day); second did not.
        relative_velocity_au_day=[0.0014, np.nan],
    )
    imp = _build_events(ev).impacts
    assert len(imp) == 2
    rv = imp.relative_velocity_au_day.to_pylist()
    assert rv[0] == pytest.approx(0.0014)
    assert rv[1] is None
    # Sanity: not silently converted to a NaN sentinel.
    assert not any(v is not None and math.isnan(v) for v in rv)
