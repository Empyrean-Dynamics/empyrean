"""Output-integrity contracts: analytic uncertainty outputs must be
POPULATED WITH FINITE VALUES, never silently all-NaN.

This guards a whole class of bug where the machinery runs (partials, STM,
propagation all compute) but a final covariance/field ships as NaN — the
caller then silently gets garbage error bars. It is the generalization of
"no hidden fallbacks": an output that claims to be a covariance must be a
covariance, not a NaN placeholder.

v0.9.0-rc.0 shipped an all-NaN sky covariance from
``generate_ephemeris(..., uncertainty_method=...)`` even though the input
covariance and the observation jacobian were both finite. No test asserted
the output covariance was finite — so it slipped through. This module
closes that gap.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import UncertaintyMethod
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.orbits.orbits import CartesianOrbits

# A self-contained, network-free heliocentric orbit with a finite state
# covariance — enough to exercise the covariance-propagation path without
# querying SBDB/MPC.
_EPOCH_MJD_TDB = 61000.0


def _orbit_with_covariance() -> CartesianOrbits:
    cov = np.zeros((1, 6, 6))
    for k, d in enumerate([1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 1e-18]):
        cov[0, k, k] = d
    return CartesianOrbits.from_kwargs(
        orbit_id=["contract"],
        object_id=["contract"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_EPOCH_MJD_TDB]),
            x=[1.6],
            y=[0.1],
            z=[0.02],
            vx=[-0.002],
            vy=[0.011],
            vz=[0.001],
            frame="ecliptic_j2000",
            origin=["Sun"],
            covariance=CartesianCovariance.from_matrix(cov),
        ),
    )


@pytest.fixture(scope="module")
def orbit() -> CartesianOrbits:
    empyrean.initialize()
    return _orbit_with_covariance()


def test_ephemeris_uncertainty_covariance_is_finite(orbit: CartesianOrbits) -> None:
    """A covariance-bearing orbit must yield a FINITE sky covariance on
    every ephemeris row — never all-NaN. This is the exact regression
    v0.9.0-rc.0 shipped."""
    t0 = float(orbit.coordinates.epoch.to_numpy()[0])
    observers = empyrean.get_observer_states(["500"], np.array([t0, t0 + 30.0, t0 + 365.0]))
    for method in (UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER):
        eph = empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)
        cov = eph.ephemeris.coordinates.covariance
        assert cov is not None, f"{method.value}: covariance column missing"
        m = cov.to_matrix()
        assert np.isfinite(m).all(), (
            f"{method.value}: ephemeris sky covariance is not finite (diag[0] = {np.diag(m[0])})"
        )
        # Sanity: a real covariance has non-negative variances.
        assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all(), (
            f"{method.value}: negative variance on the diagonal"
        )


def test_propagate_covariance_is_finite(orbit: CartesianOrbits) -> None:
    """Sibling contract (currently GREEN): the propagated state covariance
    stays finite. Kept alongside the ephemeris check so the two analytic
    covariance surfaces are guarded together — if a future change breaks
    the shared machinery, both fail, localizing the regression."""
    t0 = float(orbit.coordinates.epoch.to_numpy()[0])
    result = empyrean.propagate(
        orbit,
        np.array([t0, t0 + 30.0, t0 + 365.0]),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
    )
    m = result.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), f"propagated state covariance not finite: {np.diag(m[-1])}"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all()
