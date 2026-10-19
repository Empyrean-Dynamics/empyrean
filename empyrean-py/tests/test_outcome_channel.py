"""Per-orbit delivery status channel (propagate + generate_ephemeris).

The engine carries a per-input-orbit delivery verdict: one orbit's failure
does not end the batch — the failing orbit is recorded and every other orbit
delivers its rows, and the call still succeeds (nothing raises). These tests
pin that the Python layer surfaces the verdict as a first-class ``status``
table on both result types, with ``all_delivered`` and a per-orbit selector,
and that the table survives a Parquet round trip.

The failing orbit is deterministic and offline: a state stamped at MJD
400000.0, far outside the ephemeris span, so its first perturber query fails
— the same middle-of-a-three-orbit-batch fixture the C and wrapper suites
use. No network, no catalog object.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import empyrean
import numpy as np
from empyrean import (
    CartesianOrbits,
    Epochs,
    OrbitOutcome,
    Origin,
    PropagationResult,
    generate_ephemeris,
    propagate,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.ephemeris.result import EphemerisResult
from empyrean.observers.observers import Observers

# A valid heliocentric ecliptic state (Apophis at MJD 61000 TDB). Orbits 0
# and 2 use it at a normal epoch; orbit 1 reuses the state vector but at MJD
# 400000.0, which is outside the ephemeris span — so it fails per-orbit while
# the others deliver.
_STATE = {
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}
_GOOD_EPOCH = 61000.0
_BAD_EPOCH = 400000.0  # outside the ephemeris span → deterministic failure


def _mixed_batch() -> CartesianOrbits:
    """Three orbits whose MIDDLE one is stamped outside the ephemeris span."""
    epochs = [_GOOD_EPOCH, _BAD_EPOCH, _GOOD_EPOCH]
    coords = CartesianCoordinates.from_kwargs(
        epoch=epochs,
        x=[_STATE["x"]] * 3,
        y=[_STATE["y"]] * 3,
        z=[_STATE["z"]] * 3,
        vx=[_STATE["vx"]] * 3,
        vy=[_STATE["vy"]] * 3,
        vz=[_STATE["vz"]] * 3,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)] * 3,
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["good_a", "bad_mid", "good_b"],
        object_id=["obj_a", "obj_mid", "obj_b"],
        coordinates=coords,
    )


def _propagate_mixed() -> PropagationResult:
    orbits = _mixed_batch()
    target = Epochs.from_mjd(
        np.array([_GOOD_EPOCH, _GOOD_EPOCH + 10.0, _GOOD_EPOCH + 20.0]), scale="tdb"
    )
    return propagate(orbits, target)


def _ephemeris_mixed() -> EphemerisResult:
    orbits = _mixed_batch()
    observers = Observers.from_code("500", Epochs.from_mjd([_GOOD_EPOCH + 0.5], scale="tdb"))
    return generate_ephemeris(orbits, observers)


# ── propagate ────────────────────────────────────────────────


def test_propagate_status_records_every_orbit_by_value() -> None:
    """One status row per INPUT orbit, in input order; the middle orbit is
    FAILED with a code + message and no rows, the others DELIVERED. Nothing
    raised for the partial batch."""
    result = _propagate_mixed()
    status = result.status

    assert len(status) == 3, "one status row per input orbit"
    assert status.orbit_id.to_pylist() == ["good_a", "bad_mid", "good_b"]
    assert status.orbit_index.to_pylist() == [0, 1, 2]
    assert status.outcome.to_pylist() == [
        OrbitOutcome.DELIVERED.value,
        OrbitOutcome.FAILED.value,
        OrbitOutcome.DELIVERED.value,
    ]

    # The failed middle orbit carries a classified code + a non-empty engine
    # message and zero rows; it is NOT dropped from the table.
    codes = status.error_code.to_pylist()
    messages = status.error_message.to_pylist()
    assert codes[0] is None and codes[2] is None, "delivered orbits carry no error code"
    assert isinstance(codes[1], int), "the failed orbit carries a classified error code"
    assert messages[0] is None and messages[2] is None
    assert messages[1], "the failed orbit carries the engine's message"

    first_row = status.first_row.to_pylist()
    num_rows = status.num_rows.to_pylist()
    assert first_row[1] is None and num_rows[1] == 0, "a failed orbit occupies no row block"
    # Delivered orbits map onto their contiguous blocks of `states`.
    assert first_row[0] == 0 and num_rows[0] == 3
    assert first_row[2] == 3 and num_rows[2] == 3
    assert len(result.states) == 6, "only the two delivered orbits contribute rows"


def test_propagate_all_delivered_false_and_selector_slices() -> None:
    """``all_delivered`` is False for the partial batch; the selector returns
    the delivered orbit's exact slice of ``states`` and ``None`` for the
    failed orbit."""
    result = _propagate_mixed()
    assert result.all_delivered is False

    assert result.rows_for(1) is None, "a failed orbit has no rows"
    rows2 = result.rows_for(2)
    assert rows2 is not None
    assert len(rows2) == 3
    # The selector is the delivered table's slice, value for value.
    expected = result.states[3:6]
    assert rows2.orbit_id.to_pylist() == expected.orbit_id.to_pylist()
    assert rows2.coordinates.x.to_pylist() == expected.coordinates.x.to_pylist()
    # Out-of-range is None, not a raise.
    assert result.rows_for(9) is None


def test_propagate_status_survives_parquet_round_trip() -> None:
    """to_dir / from_dir carries the status table — the failed orbit is still
    on it after a Parquet round trip."""
    result = _propagate_mixed()
    with tempfile.TemporaryDirectory() as d:
        result.to_dir(d)
        assert (Path(d) / "status.parquet").exists(), "the status table is persisted"
        back = PropagationResult.from_dir(d)

    assert back.status.outcome.to_pylist() == result.status.outcome.to_pylist()
    assert back.status.orbit_id.to_pylist() == result.status.orbit_id.to_pylist()
    assert back.all_delivered is False
    assert back.status.error_message.to_pylist()[1], "the failure message round-trips"


# ── generate_ephemeris ───────────────────────────────────────


def test_ephemeris_status_records_every_orbit_by_value() -> None:
    """The same three-orbit batch through generate_ephemeris: one status row
    per input orbit, the middle FAILED, nothing raised."""
    result = _ephemeris_mixed()
    status = result.status

    assert len(status) == 3
    assert status.orbit_id.to_pylist() == ["good_a", "bad_mid", "good_b"]
    assert status.outcome.to_pylist() == [
        OrbitOutcome.DELIVERED.value,
        OrbitOutcome.FAILED.value,
        OrbitOutcome.DELIVERED.value,
    ]
    assert status.error_message.to_pylist()[1], "the failed orbit carries its message"
    assert status.num_rows.to_pylist()[1] == 0


def test_ephemeris_all_delivered_false_and_selector_slices() -> None:
    """``all_delivered`` False; the selector returns the delivered orbit's
    ephemeris slice and ``None`` for the failed orbit."""
    result = _ephemeris_mixed()
    assert result.all_delivered is False
    assert result.rows_for(1) is None

    rows2 = result.rows_for(2)
    assert rows2 is not None
    first = result.status.first_row.to_pylist()[2]
    num = result.status.num_rows.to_pylist()[2]
    expected = result.ephemeris[first : first + num]
    assert rows2.orbit_id.to_pylist() == expected.orbit_id.to_pylist()


def test_ephemeris_status_survives_parquet_round_trip() -> None:
    result = _ephemeris_mixed()
    with tempfile.TemporaryDirectory() as d:
        result.to_dir(d)
        assert (Path(d) / "status.parquet").exists()
        back = EphemerisResult.from_dir(d)
    assert back.status.outcome.to_pylist() == result.status.outcome.to_pylist()
    assert back.all_delivered is False


def _mixed_cov_batch(bad_middle: bool) -> CartesianOrbits:
    """Three covariance-bearing orbits. The middle is at the bad epoch when
    ``bad_middle`` (fails), else a good epoch (the all-delivered reference).
    Orbits 0 and 2 are identical across both batches, so their tagged
    covariance is directly comparable."""
    from empyrean.coordinates.covariance import CartesianCovariance

    mid = _BAD_EPOCH if bad_middle else _GOOD_EPOCH
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])
    covariance = CartesianCovariance.from_matrix(np.repeat(cov[None, :, :], 3, axis=0))
    coords = CartesianCoordinates.from_kwargs(
        epoch=[_GOOD_EPOCH, mid, _GOOD_EPOCH],
        x=[_STATE["x"]] * 3,
        y=[_STATE["y"]] * 3,
        z=[_STATE["z"]] * 3,
        vx=[_STATE["vx"]] * 3,
        vy=[_STATE["vy"]] * 3,
        vz=[_STATE["vz"]] * 3,
        covariance=covariance,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)] * 3,
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["good_a", "mid", "good_c"],
        object_id=["obj_a", "obj_mid", "obj_c"],
        coordinates=coords,
    )


def test_tagged_covariance_survives_a_failed_orbit_before_a_delivered_one() -> None:
    """A delivered orbit that follows a FAILED one in a mixed batch keeps its
    opt-in tagged covariance — the extension writes each orbit's tagged rows at
    its own delivered block, not a uniform ``orbit_index * n_times`` grid that
    would place them past the (shorter) array and silently drop them.

    good_c (orbit 2) sits after the failed middle orbit; its tagged rows must
    be present and bit-identical to a same-shape all-delivered run's good_c.
    """
    target = Epochs.from_mjd(
        np.array([_GOOD_EPOCH, _GOOD_EPOCH + 10.0, _GOOD_EPOCH + 20.0]), scale="tdb"
    )
    r_mixed = propagate(_mixed_cov_batch(bad_middle=True), target, tagged_covariance=True)
    r_all = propagate(_mixed_cov_batch(bad_middle=False), target, tagged_covariance=True)
    assert r_mixed.all_delivered is False
    assert r_all.all_delivered is True

    assert r_mixed.tagged_covariance is not None and r_all.tagged_covariance is not None
    cg_mixed = r_mixed.tagged_covariance.select("orbit_id", "good_c")
    cg_all = r_all.tagged_covariance.select("orbit_id", "good_c")
    assert len(cg_mixed) == len(cg_all) == 3, "good_c keeps its three delivered epochs"
    # The orbit after the failed one must actually carry its tagged covariance.
    assert all(cg_mixed.has_tagged.to_pylist()), (
        "the delivered orbit after a failed one lost its tagged covariance"
    )
    # ...and bit-identical to the all-delivered reference's good_c.
    for tm, ta in zip(cg_mixed.to_series(), cg_all.to_series(), strict=True):
        assert np.array_equal(tm.matrix, ta.matrix)
        assert np.array_equal(np.asarray(tm.state), np.asarray(ta.state))


def test_initialize_once() -> None:
    """Guard: the batch fixtures need an initialized context."""
    empyrean.initialize()
