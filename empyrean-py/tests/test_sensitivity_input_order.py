"""The observation-sensitivity rows arrive in observer-input order.

The engine groups its sensitivity chains by ``(orbit_id, obs_code)`` through a
BTreeMap, so iterating them yields rows grouped by code rather than in the
observer-input order the flat ephemeris ``entries`` use. The distribution
flatten re-interleaves the rows back into observer-input order, so within an
orbit block sensitivity row ``k`` pairs position-for-position with ephemeris
row ``k`` — same observatory, same epoch. A repeated ``(code, epoch)``
exposure is precisely the case a keyed join cannot disambiguate, so the
positional contract is the only thing that can carry it.

The fixture interleaves two offline observatories (``500`` geocenter, ``689``
Flagstaff) and repeats one ``(code, epoch)`` exposure. ``"500" < "689"``, so
the BTreeMap would hand back every ``500`` row ahead of every ``689`` row — an
order that disagrees with the request at the interleaved positions. Dropping
the flatten's sort makes that the delivered order and reddens
:func:`test_sensitivity_rows_follow_observer_input_order`.
"""

from __future__ import annotations

import numpy as np
import quivr as qv
from empyrean import (
    CartesianOrbits,
    EphemerisConfig,
    Epochs,
    Observers,
    Origin,
    PropagationConfig,
    generate_ephemeris,
)
from empyrean.coordinates.coordinates import CartesianCoordinates

# A valid heliocentric ecliptic state (Apophis at MJD 61000 TDB), the same
# state the outcome-channel suite uses — far from a close approach and
# resolvable fully offline.
_STATE = {
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}
_IC_EPOCH = 61000.0
# Three observation epochs. The point is the marshalled row order, not a long
# integration, so they sit a few days apart.
_E0, _E1, _E2 = 61000.5, 61010.5, 61020.5


def _orbit() -> CartesianOrbits:
    coords = CartesianCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=[_IC_EPOCH], scale="tdb"),
        x=[_STATE["x"]],
        y=[_STATE["y"]],
        z=[_STATE["z"]],
        vx=[_STATE["vx"]],
        vy=[_STATE["vy"]],
        vz=[_STATE["vz"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(orbit_id=["a"], object_id=["obj_a"], coordinates=coords)


def _obs(code: str, mjd_tdb: float) -> Observers:
    return Observers.from_code(code, Epochs.from_mjd([mjd_tdb], scale="tdb"))


def _stm_config() -> EphemerisConfig:
    """``compute_stm`` with no input covariance traces ``Jet1<6>`` and fills
    the observation-sensitivity Jacobians — the rows whose order this pins."""
    return EphemerisConfig(propagation=PropagationConfig(compute_stm=True))


def _jacobian_multiset(sens) -> list[bytes]:
    """The Jacobian payloads as an order-independent multiset of bytes.

    Read straight off the flat ``jacobian`` column — ``jacobians_array`` is a
    per-chain accessor that refuses a table spanning two observatory codes,
    which is exactly the interleaved table this test holds.
    """
    rows = sens.jacobian.to_pylist()
    assert all(r is not None for r in rows), (
        "compute_stm was set, so every row must carry a Jacobian"
    )
    return sorted(np.asarray(r, dtype=np.float64).tobytes() for r in rows)


def test_sensitivity_rows_follow_observer_input_order() -> None:
    orbit = _orbit()
    # Request: 500@e0, 689@e1, 500@e0 (duplicate of row 0), 689@e1, 500@e2.
    observers = qv.concatenate(
        [
            _obs("500", _E0),
            _obs("689", _E1),
            _obs("500", _E0),
            _obs("689", _E1),
            _obs("500", _E2),
        ]
    )

    result = generate_ephemeris(orbit, observers, _stm_config())
    sens = result.sensitivity
    eph = result.ephemeris
    assert sens is not None
    assert len(sens) == len(eph) == len(observers) == 5, (
        "one entry and one sensitivity row per observer, the duplicate included"
    )

    # Position-for-position with the ephemeris entries — grouped-by-code order
    # fails this at the first interleaved position, which is the mutation.
    assert sens.obs_code.to_pylist() == eph.obs_code.to_pylist(), (
        "sensitivity rows are not in the ephemeris entries' order"
    )
    np.testing.assert_allclose(
        sens.epochs.mjd_tdb(),
        eph.coordinates.epochs.mjd_tdb(),
        atol=1e-9,
        err_msg="sensitivity epochs disagree with the entry epochs row for row",
    )

    # ...and with the request's own observer list — the pairing a caller makes
    # against its observer array.
    assert sens.obs_code.to_pylist() == observers.obs_code.to_pylist()
    np.testing.assert_allclose(
        sens.epochs.mjd_tdb(),
        np.array([_E0, _E1, _E0, _E1, _E2]),
        atol=1e-9,
    )

    # The reorder is a pure permutation of row contents: the multiset of
    # Jacobian payloads is byte-identical to the same exposures requested in
    # code-grouped order (what the unsorted flatten would have delivered).
    # Permutation-invariant, so this does NOT catch the dropped sort — the
    # positional checks above do; this proves no payload was altered, dropped,
    # or duplicated by the reordering.
    grouped = qv.concatenate(
        [
            _obs("500", _E0),
            _obs("500", _E0),
            _obs("500", _E2),
            _obs("689", _E1),
            _obs("689", _E1),
        ]
    )
    grouped_sens = generate_ephemeris(orbit, grouped, _stm_config()).sensitivity
    assert grouped_sens is not None
    assert _jacobian_multiset(sens) == _jacobian_multiset(grouped_sens), (
        "reordering changed a Jacobian payload — it must only permute rows"
    )
