"""``Observers`` are computed in a caller-chosen ``(frame, origin)`` basis.

``Observers.from_code`` / ``from_codes`` (and the
:func:`empyrean.get_observer_states` shortcut) used to hard-code
ICRF / SSB into the table they built, regardless of what the engine
returned. They now take ``frame`` / ``origin``, forward them to the
widened engine entry point, and stamp the returned table with the basis
**read off the returned states** rather than echoed from the request.

The contracts under test:

* ICRF / SSB stays the default and the construction basis, returned
  untransformed.
* A non-default request genuinely moves the numbers, and lands exactly
  where the engine's own :func:`empyrean.transform_coordinates` puts them.
* The table reports the basis it is actually in, so a row is never
  mislabelled.
"""

import empyrean
import numpy as np
import pytest
from empyrean import CartesianCoordinates, Frame, Observers, Origin

CODES = ["500", "W84"]
EPOCHS = [60000.0, 60001.0]


def _xyz(observers: Observers) -> np.ndarray:
    c = observers.coordinates
    return np.column_stack(
        [
            c.x.to_numpy(zero_copy_only=False),
            c.y.to_numpy(zero_copy_only=False),
            c.z.to_numpy(zero_copy_only=False),
        ]
    )


def _vxyz(observers: Observers) -> np.ndarray:
    c = observers.coordinates
    return np.column_stack(
        [
            c.vx.to_numpy(zero_copy_only=False),
            c.vy.to_numpy(zero_copy_only=False),
            c.vz.to_numpy(zero_copy_only=False),
        ]
    )


# ── The default is the construction basis ─────────────────────────────


def test_default_basis_is_icrf_ssb():
    """Omitting the basis must be exactly asking for ICRF / SSB."""
    default = Observers.from_codes(CODES, EPOCHS)
    explicit = Observers.from_codes(CODES, EPOCHS, frame=Frame.ICRF, origin=Origin.SSB)

    assert default.coordinates.frame == Frame.ICRF.value
    assert default.coordinates.origin.to_pylist() == [str(Origin.SSB)] * len(default)
    np.testing.assert_array_equal(_xyz(default), _xyz(explicit))
    np.testing.assert_array_equal(_vxyz(default), _vxyz(explicit))
    assert default.obs_code.to_pylist() == explicit.obs_code.to_pylist()
    assert default.observing_night.to_pylist() == explicit.observing_night.to_pylist()


def test_from_code_forwards_the_basis():
    """The single-code convenience must not drop the basis on the floor."""
    one = Observers.from_code("500", EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN)
    many = Observers.from_codes(["500"], EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN)
    assert one.coordinates.frame == Frame.ECLIPTICJ2000.value
    assert one.coordinates.origin.to_pylist() == [str(Origin.SUN)] * len(one)
    np.testing.assert_array_equal(_xyz(one), _xyz(many))


def test_get_observer_states_forwards_the_basis():
    """The top-level shortcut is at parity with the classmethod."""
    shortcut = empyrean.get_observer_states(
        CODES, EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN
    )
    method = Observers.from_codes(CODES, EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN)
    assert shortcut.coordinates.frame == method.coordinates.frame
    np.testing.assert_array_equal(_xyz(shortcut), _xyz(method))


# ── A non-default request is honoured ─────────────────────────────────


@pytest.mark.parametrize(
    ("frame", "origin"),
    [
        (Frame.ECLIPTICJ2000, Origin.SSB),  # rotation only
        (Frame.ICRF, Origin.SUN),  # translation only
        (Frame.ECLIPTICJ2000, Origin.SUN),  # both
    ],
)
def test_requested_basis_moves_the_state(frame, origin):
    """A non-default basis must change the numbers, not just the label.

    A binding that accepted ``frame=`` / ``origin=`` and then relabelled
    the untouched ICRF / SSB numbers would be the worst possible failure —
    correct-looking tags on the wrong vectors. This catches it.
    """
    icrf_ssb = Observers.from_codes(CODES, EPOCHS)
    moved = Observers.from_codes(CODES, EPOCHS, frame=frame, origin=origin)

    assert moved.coordinates.frame == frame.value
    assert moved.coordinates.origin.to_pylist() == [str(origin)] * len(moved)
    assert not np.allclose(_xyz(icrf_ssb), _xyz(moved)), (
        f"({frame}, {origin}) returned the ICRF/SSB vectors unchanged — "
        f"the request was labelled but not applied"
    )


@pytest.mark.parametrize(
    ("frame", "origin"),
    [
        (Frame.ECLIPTICJ2000, Origin.SSB),
        (Frame.ICRF, Origin.SUN),
        (Frame.ECLIPTICJ2000, Origin.SUN),
    ],
)
def test_requested_basis_agrees_with_the_engine_transform(frame, origin):
    """The basis change is the engine's own transform, bit for bit.

    Cross-checks the widened observer path against
    :func:`empyrean.transform_coordinates` — two independent routes to the
    same numbers, so a bespoke rotation sneaking into either one shows up.
    """
    icrf_ssb = Observers.from_codes(CODES, EPOCHS)
    direct = Observers.from_codes(CODES, EPOCHS, frame=frame, origin=origin)
    via_transform = empyrean.transform_coordinates(
        icrf_ssb.coordinates, CartesianCoordinates, frame=frame, origin=origin
    )

    np.testing.assert_array_equal(
        _xyz(direct),
        np.column_stack(
            [
                via_transform.x.to_numpy(zero_copy_only=False),
                via_transform.y.to_numpy(zero_copy_only=False),
                via_transform.z.to_numpy(zero_copy_only=False),
            ]
        ),
        err_msg=f"({frame}, {origin}): observer basis change diverged from transform_coordinates",
    )
    np.testing.assert_array_equal(
        _vxyz(direct),
        np.column_stack(
            [
                via_transform.vx.to_numpy(zero_copy_only=False),
                via_transform.vy.to_numpy(zero_copy_only=False),
                via_transform.vz.to_numpy(zero_copy_only=False),
            ]
        ),
        err_msg=f"({frame}, {origin}): observer velocity diverged from transform_coordinates",
    )


def test_non_default_basis_is_not_a_panic():
    """Regression: the widened C entry point marshaled observer states
    through accessors that *assert* ICRF / SSB, so every non-default
    request unwound into the FFI boundary and came back as a caught
    panic. Any resurrection of that shape fails here."""
    moved = Observers.from_codes(CODES, EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN)
    assert len(moved) == len(CODES) * len(EPOCHS)
    assert np.all(np.isfinite(_xyz(moved)))
    assert np.all(np.isfinite(_vxyz(moved)))


# ── Accepted spellings ────────────────────────────────────────────────


def test_basis_accepts_strings_and_ints():
    """``frame`` / ``origin`` route through the same normalizers the rest
    of the API uses, so the string and int spellings are equivalent."""
    typed = Observers.from_codes(CODES, EPOCHS, frame=Frame.ECLIPTICJ2000, origin=Origin.SUN)
    stringy = Observers.from_codes(CODES, EPOCHS, frame="eclipticj2000", origin="Sun")
    np.testing.assert_array_equal(_xyz(typed), _xyz(stringy))

    inty = Observers.from_codes(CODES, EPOCHS, frame=1, origin=10)
    np.testing.assert_array_equal(_xyz(typed), _xyz(inty))
