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
from empyrean import Epochs, UncertaintyMethod
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
    observers = empyrean.get_observer_states(
        ["500"], Epochs.from_mjd(np.array([t0, t0 + 30.0, t0 + 365.0]), scale="tdb")
    )
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
        Epochs.from_mjd(np.array([t0, t0 + 30.0, t0 + 365.0]), scale="tdb"),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
    )
    m = result.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), f"propagated state covariance not finite: {np.diag(m[-1])}"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all()


def _barycentric_orbit() -> CartesianOrbits:
    """The same covariance-bearing orbit, in the barycentric basis the
    planner evaluates in. The origin shift is a pure translation, so the
    covariance — and every metric derived from it — is unchanged."""
    orbit = _orbit_with_covariance()
    coords = empyrean.transform_coordinates(orbit.coordinates, CartesianCoordinates, origin="SSB")
    return CartesianOrbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=coords,
    )


def test_plan_prior_and_posterior_metrics_are_finite() -> None:
    """A plan's covariance metrics must be real numbers.

    Same class of bug as the sky covariance above: the machinery runs
    (propagation, sensitivity chains, the information fold), but a
    summary metric ships as NaN and the caller silently reads a garbage
    σ. σ and the ellipsoid axes must also be non-negative, and folding
    observations can only add information — so the posterior can never be
    looser than the prior.
    """
    plan = empyrean.evaluate_plan(
        _barycentric_orbit(),
        [
            empyrean.PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2)),
            empyrean.PlannedObservation.radar(
                _EPOCH_MJD_TDB + 15.0,
                radar_bandwidth_hz=1.0e5,
                radar_freq_resolution_hz=0.1,
                radar_snr=50.0,
            ),
        ],
    )

    assert plan.metrics.stage.to_pylist() == ["prior", "posterior"], (
        f"metrics must carry one row per campaign stage, got {plan.metrics.stage.to_pylist()}"
    )

    for label, metrics in (
        ("prior", plan.metrics.prior()),
        ("posterior", plan.metrics.posterior()),
    ):
        assert len(metrics) == 1, f"{label}: expected exactly one row, got {len(metrics)}"
        for name in (
            "position_sigma_km",
            "velocity_sigma_m_s",
            "semi_major_km",
            "semi_minor_km",
            "log_det",
        ):
            value = metrics.column(name).to_numpy(zero_copy_only=False)
            assert np.isfinite(value).all(), f"{label}.{name} is not finite: {value}"
        for name in (
            "position_sigma_km",
            "velocity_sigma_m_s",
            "semi_major_km",
            "semi_minor_km",
        ):
            value = metrics.column(name).to_numpy(zero_copy_only=False)
            assert (value >= 0.0).all(), f"{label}.{name} is negative: {value}"
        assert metrics.semi_major_km[0].as_py() >= metrics.semi_minor_km[0].as_py(), (
            f"{label}: semi-major {metrics.semi_major_km[0].as_py()} < "
            f"semi-minor {metrics.semi_minor_km[0].as_py()}"
        )

    prior = plan.metrics.prior()
    posterior = plan.metrics.posterior()
    assert posterior.position_sigma_km[0].as_py() <= prior.position_sigma_km[0].as_py(), (
        f"folding observations cannot loosen the orbit: prior "
        f"{prior.position_sigma_km[0].as_py()} km → posterior "
        f"{posterior.position_sigma_km[0].as_py()} km"
    )
    # Unit pin. log_det is dimensional and ships in AU / AU·day⁻¹ while
    # its four siblings are km and m/s; the docs state that gap with a
    # specific ≈199.13 offset. A value near -22 instead of near -221 means
    # the engine switched to km / m·s⁻¹, and both doc sites plus that
    # constant have to be updated.
    prior_log_det = prior.log_det[0].as_py()
    assert -400.0 < prior_log_det < -50.0, (
        f"prior log_det {prior_log_det} is outside the AU-convention band; if the "
        f"engine now reports km / m·s⁻¹ the ≈199.13 offset in the PlanMetrics.log_det "
        f"and CovarianceMetrics::log_det docs is stale"
    )

    assert posterior.log_det[0].as_py() <= prior.log_det[0].as_py(), (
        f"posterior log_det {posterior.log_det[0].as_py()} exceeds prior {prior.log_det[0].as_py()}"
    )


def test_plan_candidate_metrics_are_finite() -> None:
    """Every per-candidate metric the plan reports on every row must be a
    real number — a NaN there would silently rank a candidate."""
    plan = empyrean.evaluate_plan(
        _barycentric_orbit(),
        [
            empyrean.PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2)),
            empyrean.PlannedObservation.optical(_EPOCH_MJD_TDB + 11.0, "568", (0.3, 0.3)),
        ],
    )
    for column in (
        "marginal_volume_reduction",
        "marginal_position_improvement",
        "cumulative_position_sigma_km",
        "cumulative_velocity_sigma_m_s",
        "cumulative_semi_major_km",
        "cumulative_semi_minor_km",
        "cumulative_log_det",
    ):
        values = plan.candidates.column(column).to_numpy(zero_copy_only=False)
        assert np.isfinite(values).all(), f"PlanCandidates.{column} is not finite: {values}"
