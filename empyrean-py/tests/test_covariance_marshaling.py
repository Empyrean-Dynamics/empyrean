"""Malformed covariance surfaces instead of being zeroed — bd empyrean-ekqe.

``coordinates_to_arrays`` and ``orbits_to_orbit_batch_dict`` each wrapped the
covariance conversion in a bare ``except Exception`` and, on any failure,
substituted a zero 6x6 with ``has_covariance = False``. Anything malformed —
an unconvertible sub-table, a partially-populated row, an infinite entry —
became a confidently-wrong covariance-free run instead of an error.

Both sites now share one marshaling helper that raises. The rule it enforces
is deliberately narrow: a row is rejected only when a **diagonal** entry is
null/NaN while the row is not fully absent. A null off-diagonal is a
zero-correlation fill (the standard reading of an omitted correlation, and the
natural result of diagonal-only construction); a null diagonal would have to be
fabricated as a zero variance, which is a delta function the caller never
supplied.

Each test is marked DISCRIMINATING (fails on the pre-fix code, which zeroed
and continued) or REGRESSION GUARD (passes both before and after; it pins the
behaviour the narrowing must not disturb).
"""

from __future__ import annotations

import numpy as np
import pytest
from empyrean._convert import coordinates_to_arrays, orbits_to_orbit_batch_dict
from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from empyrean.coordinates.covariance import (
    CartesianCovariance,
    CometaryCovariance,
    KeplerianCovariance,
    SphericalCovariance,
)
from empyrean.coordinates.enums import Origin
from empyrean.orbits.orbits import CartesianOrbits

# (coordinate class, covariance class, element kwargs for one row)
_REPRESENTATIONS = [
    (
        CartesianCoordinates,
        CartesianCovariance,
        {"x": [1.0], "y": [0.0], "z": [0.0], "vx": [0.0], "vy": [0.017], "vz": [0.0]},
    ),
    (
        KeplerianCoordinates,
        KeplerianCovariance,
        {"a": [1.0], "e": [0.1], "i": [5.0], "raan": [10.0], "ap": [20.0], "ma": [30.0]},
    ),
    (
        CometaryCoordinates,
        CometaryCovariance,
        {"q": [0.9], "e": [0.1], "i": [5.0], "raan": [10.0], "ap": [20.0], "tp": [60000.0]},
    ),
    (
        SphericalCoordinates,
        SphericalCovariance,
        {
            "rho": [1.0],
            "lon": [10.0],
            "lat": [5.0],
            "vrho": [0.0],
            "vlon": [0.01],
            "vlat": [0.0],
        },
    ),
]

_GOOD_DIAGONAL = [1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16]


def _coords(coord_cls, cov_cls, elements, cov_matrix=None, n=1):
    """Build one coordinate table, optionally carrying a covariance."""
    kwargs = {k: v * n for k, v in elements.items()}
    kwargs["epoch"] = [60000.0] * n
    kwargs["frame"] = "ecliptic_j2000"
    kwargs["origin"] = [str(Origin.SUN)] * n
    if cov_matrix is not None:
        kwargs["covariance"] = cov_cls.from_matrix(cov_matrix)
    return coord_cls.from_kwargs(**kwargs)


def _cartesian_orbits(cov_matrix=None, orbit_ids=None) -> CartesianOrbits:
    coord_cls, cov_cls, elements = _REPRESENTATIONS[0]
    n = 1 if cov_matrix is None else cov_matrix.shape[0]
    return CartesianOrbits.from_kwargs(
        orbit_id=orbit_ids if orbit_ids is not None else [f"o{i}" for i in range(n)],
        coordinates=_coords(coord_cls, cov_cls, elements, cov_matrix, n=n),
    )


def _good_matrix(n: int = 1) -> np.ndarray:
    return np.tile(np.diag(_GOOD_DIAGONAL), (n, 1, 1))


# ══════════════════════════════════════════════════════════════════
#  Discriminating: malformed input raises
# ══════════════════════════════════════════════════════════════════


def test_null_diagonal_row_raises_at_coordinates_entry_point() -> None:
    """DISCRIMINATING.

    A row with a NaN variance on one state variable but real entries elsewhere
    is not "absent" — it is incomplete. The pre-fix code took
    ``has_covariance = ~isnan(cov[:, 0, 0])`` and ``nan_to_num(..., nan=0.0)``,
    so this row marshalled through as a covariance asserting *zero* variance on
    ``vy``. The error must name the entry point, the row, and the offending
    state variable.
    """
    cov = _good_matrix()
    cov[0, 4, 4] = np.nan
    coords = _coords(*_REPRESENTATIONS[0][:2], _REPRESENTATIONS[0][2], cov)

    with pytest.raises(ValueError) as excinfo:
        coordinates_to_arrays(coords)
    message = str(excinfo.value)
    assert "coordinates_to_arrays" in message
    assert "row 0" in message
    assert "vy" in message


def test_null_diagonal_row_raises_at_orbits_entry_point() -> None:
    """DISCRIMINATING.

    Same defect through the orbit-batch entry point. Both entry points see the
    same coordinate classes, so the error must name the entry point (not just
    the class) and, here, the ``orbit_id`` — otherwise a user cannot tell which
    call failed or which record to fix.
    """
    cov = _good_matrix()
    cov[0, 0, 0] = np.nan
    orbits = _cartesian_orbits(cov, orbit_ids=["BAD-ORBIT"])

    with pytest.raises(ValueError) as excinfo:
        orbits_to_orbit_batch_dict(orbits)
    message = str(excinfo.value)
    assert "orbits_to_orbit_batch_dict" in message
    assert "BAD-ORBIT" in message
    assert "'x'" in message


def test_non_finite_entry_raises() -> None:
    """DISCRIMINATING.

    ``np.nan_to_num(x, nan=0.0)`` leaves ``posinf`` at its default, so the
    pre-fix path converted ``+inf`` into 1.797e308 — a finite-looking variance
    the engine would happily use. An infinity must be rejected, not rescaled.
    """
    cov = _good_matrix()
    cov[0, 2, 2] = np.inf
    coords = _coords(*_REPRESENTATIONS[0][:2], _REPRESENTATIONS[0][2], cov)

    with pytest.raises(ValueError, match="non-finite"):
        coordinates_to_arrays(coords)


def test_negative_infinite_off_diagonal_raises() -> None:
    """DISCRIMINATING. ``-inf`` anywhere in the matrix, not just on the
    diagonal, is corruption rather than an omitted correlation."""
    cov = _good_matrix()
    cov[0, 1, 3] = -np.inf
    cov[0, 3, 1] = -np.inf
    coords = _coords(*_REPRESENTATIONS[0][:2], _REPRESENTATIONS[0][2], cov)

    with pytest.raises(ValueError, match="non-finite"):
        coordinates_to_arrays(coords)


@pytest.mark.parametrize(
    ("entry_point", "invoke"),
    [
        (
            "coordinates_to_arrays",
            lambda: coordinates_to_arrays(
                _coords(*_REPRESENTATIONS[0][:2], _REPRESENTATIONS[0][2], _good_matrix())
            ),
        ),
        (
            "orbits_to_orbit_batch_dict",
            lambda: orbits_to_orbit_batch_dict(_cartesian_orbits(_good_matrix())),
        ),
    ],
)
def test_conversion_failure_is_not_swallowed(monkeypatch, entry_point, invoke) -> None:
    """DISCRIMINATING, at BOTH sites.

    Forces the conversion itself to fail. The pre-fix code caught it and
    returned zeros with ``has_covariance = False``; the caller had no way to
    know. It must now raise a chained ``ValueError`` naming the entry point,
    with the original exception preserved as ``__cause__``.
    """
    import empyrean._convert as convert

    def _boom(_covariance):
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(convert, "_covariance_to_matrix", _boom)

    with pytest.raises(ValueError) as excinfo:
        invoke()
    assert entry_point in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "synthetic conversion failure" in str(excinfo.value.__cause__)


def test_wrong_shape_raises(monkeypatch) -> None:
    """DISCRIMINATING. A covariance that is not ``(n, 6, 6)`` is rejected
    rather than silently reshaped or replaced with zeros."""
    import empyrean._convert as convert

    monkeypatch.setattr(convert, "_covariance_to_matrix", lambda _c: np.zeros((1, 3, 3)))

    with pytest.raises(ValueError, match=r"shaped \(1, 6, 6\)"):
        coordinates_to_arrays(
            _coords(*_REPRESENTATIONS[0][:2], _REPRESENTATIONS[0][2], _good_matrix())
        )


def test_propagate_surfaces_malformed_covariance() -> None:
    """DISCRIMINATING.

    The user-facing consequence: ``propagate`` used to run to completion on a
    fabricated zero covariance and hand back uncertainties that were never
    asked for. It must fail instead.
    """
    import empyrean

    empyrean.initialize()
    cov = _good_matrix()
    cov[0, 5, 5] = np.nan

    with pytest.raises(ValueError, match="coordinates_to_arrays"):
        empyrean.propagate(_cartesian_orbits(cov), np.array([60001.0]))


def test_write_orbits_parquet_surfaces_malformed_covariance(tmp_path) -> None:
    """DISCRIMINATING.

    Writing is the worse case: a silently zeroed covariance becomes a
    persisted file that looks authoritative. ``write_orbits_parquet`` must
    raise rather than write the fabrication to disk.
    """
    from empyrean.io.orbits import write_orbits_parquet

    cov = _good_matrix()
    cov[0, 3, 3] = np.nan
    target = tmp_path / "bad.parquet"

    with pytest.raises(ValueError, match="orbits_to_orbit_batch_dict"):
        write_orbits_parquet(str(target), _cartesian_orbits(cov))
    assert not target.exists(), "a malformed covariance reached disk"


# ══════════════════════════════════════════════════════════════════
#  Regression guards: everything legitimate still works
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("coord_cls", "cov_cls", "elements"),
    _REPRESENTATIONS,
    ids=["cartesian", "keplerian", "cometary", "spherical"],
)
def test_absent_covariance_marshals_as_has_covariance_false(coord_cls, cov_cls, elements) -> None:
    """REGRESSION GUARD.

    No covariance supplied at all: quivr materialises an all-null sub-table
    (never ``None``), which reads back as an all-NaN 6x6. That is genuinely
    absent, not malformed — it must still marshal as zeros with
    ``has_covariance = False``, on every representation.
    """
    coords = _coords(coord_cls, cov_cls, elements)
    _, _, cov_matrices, has_cov, _, _, _ = coordinates_to_arrays(coords)
    assert has_cov.tolist() == [False]
    assert cov_matrices.shape == (1, 6, 6)
    assert np.all(cov_matrices == 0.0)


@pytest.mark.parametrize(
    ("coord_cls", "cov_cls", "elements"),
    _REPRESENTATIONS,
    ids=["cartesian", "keplerian", "cometary", "spherical"],
)
def test_mixed_presence_batch(coord_cls, cov_cls, elements) -> None:
    """REGRESSION GUARD.

    A batch where some rows carry a covariance and others are fully absent
    (all-NaN, the ``from_matrix`` absent sentinel). Per-row presence must be
    honoured — present rows keep their values, absent rows zero out — and the
    absent rows must not be mistaken for malformed partial rows.
    """
    cov = np.full((3, 6, 6), np.nan)
    cov[0] = np.diag(_GOOD_DIAGONAL)
    cov[2] = np.diag(_GOOD_DIAGONAL)
    coords = _coords(coord_cls, cov_cls, elements, cov, n=3)

    _, _, cov_matrices, has_cov, _, _, _ = coordinates_to_arrays(coords)
    assert has_cov.tolist() == [True, False, True]
    assert np.all(cov_matrices[1] == 0.0)
    np.testing.assert_allclose(np.diagonal(cov_matrices[0]), _GOOD_DIAGONAL)
    np.testing.assert_allclose(np.diagonal(cov_matrices[2]), _GOOD_DIAGONAL)


@pytest.mark.parametrize(
    ("coord_cls", "cov_cls", "elements"),
    _REPRESENTATIONS,
    ids=["cartesian", "keplerian", "cometary", "spherical"],
)
def test_diagonal_only_construction_still_works(coord_cls, cov_cls, elements) -> None:
    """REGRESSION GUARD — the reason the rule is narrow, not all-or-nothing.

    Building a covariance from only the six ``cov_<a>_<a>`` columns leaves all
    15 off-diagonals null. That is a legitimate, natural user pattern: it says
    the state variables are uncorrelated. An all-or-nothing null rule would
    have rejected it. The variances must survive and the off-diagonals must
    read back as zero.
    """
    labels = cov_cls._state_labels
    covariance = cov_cls.from_kwargs(
        **{f"cov_{label}_{label}": [_GOOD_DIAGONAL[k]] for k, label in enumerate(labels)}
    )
    kwargs = dict(elements)
    kwargs["epoch"] = [60000.0]
    kwargs["frame"] = "ecliptic_j2000"
    kwargs["origin"] = [str(Origin.SUN)]
    kwargs["covariance"] = covariance
    coords = coord_cls.from_kwargs(**kwargs)

    _, _, cov_matrices, has_cov, _, _, _ = coordinates_to_arrays(coords)
    assert has_cov.tolist() == [True]
    np.testing.assert_allclose(np.diagonal(cov_matrices[0]), _GOOD_DIAGONAL)
    off_diagonal = cov_matrices[0][~np.eye(6, dtype=bool)]
    assert np.all(off_diagonal == 0.0), "off-diagonals must fill as uncorrelated zeros"


@pytest.mark.parametrize(
    ("coord_cls", "cov_cls", "elements"),
    _REPRESENTATIONS,
    ids=["cartesian", "keplerian", "cometary", "spherical"],
)
def test_empty_table(coord_cls, cov_cls, elements) -> None:
    """REGRESSION GUARD. A zero-row table marshals to empty arrays without
    raising — the validation must not treat "no rows" as "no diagonal"."""
    kwargs = {k: [] for k in elements}
    kwargs["epoch"] = []
    kwargs["frame"] = "ecliptic_j2000"
    kwargs["origin"] = []
    coords = coord_cls.from_kwargs(**kwargs)

    epochs, _, cov_matrices, has_cov, _, _, _ = coordinates_to_arrays(coords)
    assert len(epochs) == 0
    assert cov_matrices.shape == (0, 6, 6)
    assert has_cov.shape == (0,)


def test_partial_off_diagonal_is_accepted() -> None:
    """REGRESSION GUARD for the narrowing itself.

    A complete diagonal plus *some* correlations: the supplied correlations
    survive and the omitted ones fill with zero. This is what distinguishes the
    narrow rule from an all-or-nothing one.
    """
    labels = CartesianCovariance._state_labels
    kwargs_cov = {f"cov_{label}_{label}": [_GOOD_DIAGONAL[k]] for k, label in enumerate(labels)}
    # Exactly one correlation supplied; the other 14 stay null.
    kwargs_cov["cov_x_y"] = [3e-13]
    covariance = CartesianCovariance.from_kwargs(**kwargs_cov)

    coord_cls, _, elements = _REPRESENTATIONS[0]
    kwargs = dict(elements)
    kwargs["epoch"] = [60000.0]
    kwargs["frame"] = "ecliptic_j2000"
    kwargs["origin"] = [str(Origin.SUN)]
    kwargs["covariance"] = covariance
    coords = coord_cls.from_kwargs(**kwargs)

    _, _, cov_matrices, has_cov, _, _, _ = coordinates_to_arrays(coords)
    assert has_cov.tolist() == [True]
    np.testing.assert_allclose(np.diagonal(cov_matrices[0]), _GOOD_DIAGONAL)
    assert cov_matrices[0][0, 1] == pytest.approx(3e-13)
    assert cov_matrices[0][0, 2] == 0.0


# ── Degenerate-geometry transforms ────────────────────────────
#
# The one reachable behaviour change. `transform_coordinates` to a
# Keplerian/cometary basis at an exactly-degenerate geometry emits a
# covariance whose angular-element variance is NaN, because the element
# has no definition there. Zero-filling it claimed zero variance on an
# undefined element — the fabricated delta function this fix exists to
# stop — so the marshal now raises. The blast radius is measure-zero
# geometry only, which the control test below pins.


def _cartesian_at(x, y, z, vx, vy, vz):
    return CartesianCoordinates.from_kwargs(
        epoch=np.array([60000.0]),
        x=[x],
        y=[y],
        z=[z],
        vx=[vx],
        vy=[vy],
        vz=[vz],
        frame="ecliptic",
        origin=[str(Origin.SUN)],
        covariance=CartesianCovariance.from_matrix(
            np.diag([1e-10, 1e-10, 1e-10, 1e-14, 1e-14, 1e-14])[None, :, :]
        ),
    )


@pytest.mark.parametrize("target", ["keplerian", "cometary"])
def test_degenerate_transform_surfaces_undefined_element(target) -> None:
    """DISCRIMINATING. An exactly-circular-plane / at-periapsis geometry
    transforms to a covariance with a NaN angular-element variance. Before
    the fix that was zero-filled and marshalled as a *present* covariance
    asserting zero variance on an undefined element; now it raises, and the
    message points at the transform rather than at the caller's input."""
    import empyrean

    empyrean.initialize()
    mu = 0.00029591220828559104
    a, e = 2.5, 0.2
    r = a * (1 - e)
    v = np.sqrt(mu * (2 / r - 1 / a))
    # Position on +x with velocity purely transverse: exactly at periapsis
    # and exactly in-plane, so an angular element is undefined.
    coords = _cartesian_at(r, 0.0, 0.0, 0.0, v, 0.0)
    out = empyrean.transform_coordinates(coords, target)

    matrix = out.covariance.to_matrix()[0]
    assert np.isnan(np.diagonal(matrix)).any(), "expected a NaN angular-element variance"

    with pytest.raises(ValueError, match="null or NaN variance"):
        coordinates_to_arrays(out)
    with pytest.raises(ValueError, match="transform_coordinates"):
        coordinates_to_arrays(out)


@pytest.mark.parametrize("target", ["keplerian", "cometary", "cartesian"])
def test_ordinary_orbits_still_transform_and_marshal(target) -> None:
    """REGRESSION GUARD — the important half. The narrowed rule must not
    reject ordinary science input. Generic (non-degenerate) states and a real
    catalogue orbit round-trip through every representation untouched;
    measured 0/120 rejections across randomised generic states."""
    import empyrean

    empyrean.initialize()
    mu = 0.00029591220828559104
    rng = np.random.default_rng(7)
    checked = 0
    for _ in range(12):
        pos = rng.normal(size=3) * 1.5 + np.array([2.0, 0.3, 0.1])
        vel = rng.normal(size=3) * 0.005 + np.array([0.001, 0.010, 0.002])
        # Keep only bound draws. An unbound state has no elliptical
        # Keplerian form and is not what this guard is about; screening on
        # the orbital energy up front keeps the loop free of exception
        # handling, so a genuine marshaling failure can never be mistaken
        # for an unbound draw and skipped.
        if 0.5 * float(vel @ vel) - mu / float(np.linalg.norm(pos)) >= 0.0:
            continue
        out = empyrean.transform_coordinates(_cartesian_at(*pos, *vel), target)
        assert np.isfinite(np.diagonal(out.covariance.to_matrix()[0])).all()
        coordinates_to_arrays(out)
        checked += 1
    # Without this the screen could reject every draw and the guard would
    # pass by testing nothing — the exact failure mode this file exists for.
    assert checked > 0, "no bound draws survived the energy screen"
