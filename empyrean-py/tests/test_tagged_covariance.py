"""Tests for the provenance-tagged covariance readback.

``propagate(..., tagged_covariance=True)`` surfaces villeneuve's
resolved-kind, provenance-tagged covariance at every output epoch —
the honest covariance distinct from the bare linear ``Φ Σ₀ Φᵀ`` mapping
on the propagated states. These tests pin:

- the opt-in flag is off by default (no tagged table, zero cost),
- the tagged table is aligned 1:1 with the propagated states
  (orbit-major), kind Linear for a FirstOrder propagation with no close
  approach, the t₀ matrix equals the input covariance, and the matrix
  round-trips as a contiguous symmetric 6×6,
- an orbit carrying no input covariance yields ``has_tagged=False``
  zero-filled rows that stay aligned,
- ``to_dir`` / ``from_dir`` persist the tagged table,
- multi-orbit batches keep per-orbit series separable.
"""

from __future__ import annotations

import tempfile

import empyrean
import numpy as np
from empyrean import (
    CartesianCoordinates,
    CartesianCovariance,
    CartesianOrbits,
    CovarianceKind,
    PropagationResult,
    TaggedCovariances,
)

# Near-circular heliocentric state at ~2 AU (matches the Rust order-lock
# test in empyrean/src/propagate/result.rs).
_T0 = 60000.0
_OFFSETS = [0.0, 10.0, 30.0, 60.0]


def _diag_cov() -> np.ndarray:
    cov = np.zeros((6, 6))
    for i in range(3):
        cov[i, i] = 1e-12
    for i in range(3, 6):
        cov[i, i] = 1e-16
    return cov


def _orbit_with_cov(orbit_id: str = "order-lock") -> CartesianOrbits:
    cov = _diag_cov()
    coords = CartesianCoordinates.from_kwargs(
        epoch=[_T0],
        x=[2.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.01217],
        vz=[0.0],
        frame="eclipticj2000",
        origin=["Sun"],
        covariance=CartesianCovariance.from_matrix(cov[np.newaxis, :, :]),
    )
    return CartesianOrbits.from_kwargs(orbit_id=[orbit_id], coordinates=coords)


def _orbit_no_cov(orbit_id: str = "no-cov") -> CartesianOrbits:
    coords = CartesianCoordinates.from_kwargs(
        epoch=[_T0],
        x=[2.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.01217],
        vz=[0.0],
        frame="eclipticj2000",
        origin=["Sun"],
    )
    return CartesianOrbits.from_kwargs(orbit_id=[orbit_id], coordinates=coords)


def _epochs() -> np.ndarray:
    return np.array([_T0 + d for d in _OFFSETS], dtype=np.float64)


def test_tagged_covariance_off_by_default() -> None:
    result = empyrean.propagate(_orbit_with_cov(), _epochs())
    assert result.tagged_covariance is None


def test_tagged_covariance_aligned_and_linear() -> None:
    epochs = _epochs()
    n = len(epochs)
    result = empyrean.propagate(_orbit_with_cov(), epochs, tagged_covariance=True)

    assert isinstance(result.tagged_covariance, TaggedCovariances)
    assert len(result.tagged_covariance) == n
    assert len(result.states) == n

    series = result.tagged_covariance_series(0)
    assert len(series) == n

    states = result.states.coordinates
    state_epochs = states.epoch.to_numpy(zero_copy_only=False)
    sx = states.x.to_numpy(zero_copy_only=False)
    sy = states.y.to_numpy(zero_copy_only=False)
    sz = states.z.to_numpy(zero_copy_only=False)
    svx = states.vx.to_numpy(zero_copy_only=False)
    svy = states.vy.to_numpy(zero_copy_only=False)
    svz = states.vz.to_numpy(zero_copy_only=False)

    for k, tagged in enumerate(series):
        # Order lock: tagged epoch == state epoch.
        assert abs(tagged.epoch_mjd_tdb - state_epochs[k]) < 1e-9
        # FirstOrder + no close approach ⟹ Linear.
        assert tagged.kind is CovarianceKind.LINEAR
        # Co-located nominal state matches the propagated state.
        st = np.array([sx[k], sy[k], sz[k], svx[k], svy[k], svz[k]])
        assert np.allclose(tagged.state, st)
        # Finite, contiguous, symmetric 6×6.
        m = tagged.matrix
        assert m.shape == (6, 6)
        assert m.flags["C_CONTIGUOUS"]
        assert np.isfinite(m).all()
        assert np.allclose(m, m.T)

    assert all(result.tagged_covariance.has_tagged.to_pylist())

    # The t₀ matrix is the linear map's starting point Σ₀.
    assert np.allclose(series[0].matrix, _diag_cov(), atol=1e-18)


def test_tagged_covariance_no_input_covariance() -> None:
    epochs = _epochs()
    n = len(epochs)
    result = empyrean.propagate(_orbit_no_cov(), epochs, tagged_covariance=True)

    tc = result.tagged_covariance
    assert tc is not None
    # Still aligned 1:1, but flagged false and zero-filled.
    assert len(tc) == n
    assert not any(tc.has_tagged.to_pylist())

    series = result.tagged_covariance_series(0)
    assert len(series) == n
    assert np.allclose(series[0].matrix, 0.0)


def test_tagged_covariance_round_trip() -> None:
    result = empyrean.propagate(_orbit_with_cov(), _epochs(), tagged_covariance=True)
    m0 = result.tagged_covariance_series(0)[0].matrix

    with tempfile.TemporaryDirectory() as d:
        result.to_dir(d)
        loaded = PropagationResult.from_dir(d)
        assert loaded.tagged_covariance is not None
        rt = loaded.tagged_covariance_series(0)
        assert len(rt) == len(result.tagged_covariance)
        assert np.allclose(rt[0].matrix, m0)
        assert rt[0].kind is CovarianceKind.LINEAR


def test_tagged_covariance_multi_orbit_separable() -> None:
    a = _orbit_with_cov("orbit-a")
    b = _orbit_with_cov("orbit-b")
    batch = CartesianOrbits.from_kwargs(
        orbit_id=["orbit-a", "orbit-b"],
        coordinates=CartesianCoordinates.from_kwargs(
            **{
                col: np.concatenate(
                    [
                        getattr(a.coordinates, col).to_numpy(zero_copy_only=False),
                        getattr(b.coordinates, col).to_numpy(zero_copy_only=False),
                    ]
                )
                for col in ("epoch", "x", "y", "z", "vx", "vy", "vz")
            },
            frame="eclipticj2000",
            origin=["Sun", "Sun"],
            covariance=CartesianCovariance.from_matrix(np.stack([_diag_cov(), _diag_cov()])),
        ),
    )

    epochs = _epochs()
    n = len(epochs)
    result = empyrean.propagate(batch, epochs, tagged_covariance=True)

    assert len(result.tagged_covariance) == 2 * n
    assert result.tagged_covariance.orbit_ids_unique() == ["orbit-a", "orbit-b"]

    series_a = result.tagged_covariance_series(0)
    series_b = result.tagged_covariance_series(1)
    assert len(series_a) == n
    assert len(series_b) == n
    # Same input ⟹ same t₀ covariance for both orbits.
    assert np.allclose(series_a[0].matrix, series_b[0].matrix)
