"""A failed batch names the orbit it failed on, at the Python layer.

`empyrean.propagate` takes a whole batch and fails as a whole. The
exception used to carry a message and nothing positional, so the
only route from "the call failed" to "orbit 2 failed" was to re-run the
batch one orbit at a time. The exception now carries `orbit_index`,
`orbit_id` and `epoch_mjd_tdb`, mirroring the Rust wrapper's `Error` and
the C ABI's `empyrean_error_location`.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    Epochs,
    Origin,
    PropagationConfig,
    propagate,
)

# Apophis at MJD 61000 TDB, heliocentric ecliptic J2000.
APOPHIS = (
    -7.85264914906904643e-02,
    -8.19748051902064567e-01,
    4.18939515323390882e-02,
    1.98751024968884596e-02,
    1.32208844536140196e-03,
    3.99496044422352188e-04,
)
T0 = 61000.0

IDS = ["good-0", "good-1", "the-bad-one", "good-3"]
BAD_ROW = 2


def _batch(*, break_row: int | None) -> CartesianOrbits:
    """Four copies of the Apophis state, optionally with one row's
    z-velocity set to NaN."""
    n = len(IDS)
    vz = [APOPHIS[5]] * n
    if break_row is not None:
        vz[break_row] = float("nan")
    coords = CartesianCoordinates.from_kwargs(
        epoch=[T0] * n,
        x=[APOPHIS[0]] * n,
        y=[APOPHIS[1]] * n,
        z=[APOPHIS[2]] * n,
        vx=[APOPHIS[3]] * n,
        vy=[APOPHIS[4]] * n,
        vz=vz,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)] * n,
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=IDS,
        object_id=[f"obj-{i}" for i in range(n)],
        coordinates=coords,
    )


def _epochs() -> Epochs:
    return Epochs.from_mjd(np.array([T0 + 30.0]), scale="tdb")


def test_a_bad_batch_member_is_named_on_the_exception() -> None:
    """The headline: the exception says which row was bad."""
    with pytest.raises(Exception) as excinfo:
        propagate(_batch(break_row=BAD_ROW), _epochs())

    err = excinfo.value
    assert err.orbit_index == BAD_ROW, f"got {err.orbit_index} from: {err}"
    assert err.orbit_id == IDS[BAD_ROW]
    assert err.epoch_mjd_tdb == pytest.approx(T0)


def test_the_same_batch_without_the_nan_succeeds() -> None:
    """The positive control. Everything about this batch except the NaN
    is fine, so the failure above is the NaN and not the fixture."""
    result = propagate(_batch(break_row=None), _epochs())
    assert len(result.states) == len(IDS)


def test_a_failure_with_no_offending_orbit_reports_none() -> None:
    """The other positive control. Attributes that are always present but
    always `None` would pass the first test and mean nothing; a failure
    belonging to no single row must report no row rather than row 0."""
    with pytest.raises(Exception) as excinfo:
        propagate(_batch(break_row=None), Epochs.from_mjd(np.array([]), scale="tdb"))

    err = excinfo.value
    assert err.orbit_index is None, f"no row is at fault: {err}"
    assert err.orbit_id is None
    assert err.epoch_mjd_tdb is None


def test_the_attributes_exist_on_every_empyrean_failure() -> None:
    """They are set unconditionally, `None` included, so a caller reads
    them directly instead of guarding every access with `getattr`."""
    with pytest.raises(Exception) as excinfo:
        propagate(_batch(break_row=BAD_ROW), _epochs())

    for attr in ("orbit_index", "orbit_id", "epoch_mjd_tdb"):
        assert hasattr(excinfo.value, attr), f"missing {attr}"


def test_the_attributes_exist_on_a_builtsystem_guard_failure() -> None:
    """ "Every wrapper failure" has to include the ones that raise their
    own exception type. The BuiltSystem identity guard raises a
    ``ValueError`` of its own rather than going through the ordinary
    mapping, and it used to arrive without the three attributes -- so
    ``except Exception as e: e.orbit_index`` raised ``AttributeError``
    exactly where a caller is most likely to be catching broadly."""
    system = empyrean.BuiltSystem(force_model="standard", frame="icrf")
    mismatched = PropagationConfig(force_model="basic", frame="icrf")

    with pytest.raises(ValueError) as excinfo:
        system.propagate(_batch(break_row=None), _epochs(), mismatched)

    assert "identity guard" in str(excinfo.value), str(excinfo.value)
    for attr in ("orbit_index", "orbit_id", "epoch_mjd_tdb"):
        assert hasattr(excinfo.value, attr), f"guard failure is missing {attr}"
