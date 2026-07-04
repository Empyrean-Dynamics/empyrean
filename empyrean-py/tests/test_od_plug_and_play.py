"""Plug-and-play OD-output acceptance test (the headline contract).

The ``determine`` / ``evaluate`` / ``refine`` /
``compute_impact_probabilities`` surfaces are designed so a fitted orbit
flows straight back in as input with **zero manual reconstruction** — no
hand-built :class:`CartesianCoordinates`, no re-wrapped :class:`Orbit`,
and no silently-dropped force model. This test exercises the entire hop
chain on a real Apophis arc and locks three sub-contracts that shipped
without acceptance coverage:

* ``833t`` — ``evaluate(fit.orbit, obs)`` returns residuals; it must not
  raise ``NoValidObservations`` because the fitted orbit was somehow
  unusable as evaluate input.
* ``qr13`` — ``compute_impact_probabilities`` accepts ``fit.orbit``
  directly, with no manual ``CoordinateState`` / ``Orbit`` rebuild.
* ``o7fk`` — the fitted **absolute** non-grav (A1/A2/A3 + g(r) + dt) is
  carried on ``fit.orbit.non_grav`` and survives the output→input re-feed,
  so ``propagate(fit.orbit)`` is measurably different from propagating the
  same orbit with its force model stripped.

The arc is the multi-apparition Apophis astrometry (2004–2021). With an
explicit ``StateAndNonGrav`` fit it recovers a Yarkovsky A2 of order
−3e-14 AU/d² — sign-correct (retrograde rotator, secular semi-major-axis
*decrease*) and consistent with the JPL SBDB solution (≈ −2.9e-14
AU/d², Pérez-Hernández & Benet 2022; Farnocchia et al. 2013). The famous
2029-04-13 Earth encounter (geocentric miss ≈ 38,000 km) falls out of the
propagation, which both checks the IP surface and anchors the trajectory
against a published value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from empyrean import (
    Origin,
    compute_impact_probabilities,
    determine,
    evaluate,
    propagate,
    read_ades,
    refine,
)
from empyrean.od.result import ODConfig, SolveForParams
from empyrean.orbits.nongrav import NonGravParams
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import UncertaintyMethod

DATA_DIR = Path(__file__).parent / "fixtures"
APOPHIS_MULTIAPP = DATA_DIR / "99942_apophis_multiapp.psv"

# Apophis 2029 close approach — published geocentric miss distance is
# ≈ 38,000 km (≈ 31,600 km from Earth's *surface*). We assert the IP
# detector lands in a generous band around the center-to-center value so
# the test is anchored to reality without being brittle to the fit's
# covariance.
APOPHIS_2029_MJD_TDB = 62239.9  # 2029-04-13 (expected encounter epoch)
# The close-approach detector is a step-boundary sampler: the propagation
# window must run *past* the encounter for the approach to be bracketed,
# so the end epoch is set a few weeks beyond the 2029 CA.
APOPHIS_IP_END_MJD_TDB = 62260.0  # ~2029-05-04
APOPHIS_2029_MISS_KM_LO = 30_000.0
APOPHIS_2029_MISS_KM_HI = 45_000.0


@pytest.fixture(scope="module")
def apophis_observations():
    """Read the bundled multi-apparition Apophis ADES PSV arc.

    ``read_ades`` returns ``(optical, radar)``; this arc is optical-only,
    so the radar table comes back empty and is discarded.
    """
    if not APOPHIS_MULTIAPP.exists():
        pytest.skip(f"missing fixture: {APOPHIS_MULTIAPP}")
    optical, _radar = read_ades(APOPHIS_MULTIAPP)
    assert len(optical) > 1000, "expected the full multi-apparition arc"
    return optical


@pytest.fixture(scope="module")
def apophis_fit(apophis_observations):
    """Fit Apophis with an explicit non-grav solve.

    ``SolveForParams.STATE_AND_NONGRAV`` forces the 9-parameter fit so the
    returned orbit carries a fitted Yarkovsky A2 — the object of the
    ``o7fk`` non-grav round-trip assertion. (Default ``AUTO`` correctly
    declines to escalate on this arc and returns a state-only fit, which
    is exercised separately below.)
    """
    config = ODConfig(solve_for=SolveForParams.STATE_AND_NONGRAV)
    fit = determine(apophis_observations, config=config)
    assert fit.converged, "Apophis non-grav fit did not converge"
    return fit


def test_fit_orbit_is_a_refeedable_cartesian_orbit(apophis_fit):
    """The fitted orbit is a fully-formed, re-feedable ``CartesianOrbits``.

    No caller should have to reach into the result and reconstruct a
    coordinate state or orbit table by hand. The orbit must be a
    single-row ``CartesianOrbits`` with epoch, state, covariance, and the
    fitted non-grav already attached.
    """
    orbit = apophis_fit.orbit
    assert isinstance(orbit, CartesianOrbits)
    assert len(orbit) == 1
    # Covariance present — required for refine's Bayesian prior.
    assert orbit.coordinates.covariance is not None
    # Fitted absolute non-grav rode along (StateAndNonGrav).
    assert orbit.non_grav is not None


def test_fitted_yarkovsky_a2_is_recovered(apophis_fit):
    """o7fk: the fitted absolute A2 is non-null, finite, and physical.

    The multi-apparition arc constrains Apophis's Yarkovsky drift. The
    recovered A2 must be:

    * non-null (the force model survived output marshaling, not silently
      dropped to ``None``);
    * negative — Apophis is a retrograde rotator, so the transverse
      Yarkovsky acceleration drives a secular *decrease* in semi-major
      axis (Farnocchia et al. 2013; Pérez-Hernández & Benet 2022);
    * order 1e-14 AU/d² — consistent with the JPL SBDB value
      (≈ −2.9e-14 AU/d²), not the 1e-9 overfit noise a too-short arc
      would produce.
    """
    ng = apophis_fit.orbit.non_grav
    a2 = ng.a2.to_pylist()[0]
    assert a2 is not None, "fitted A2 must not be None — non-grav was dropped on output"
    assert np.isfinite(a2)
    assert a2 < 0.0, f"Apophis A2 must be negative (retrograde Yarkovsky); got {a2:.3e}"
    assert 1e-15 < abs(a2) < 1e-13, (
        f"A2 magnitude {abs(a2):.3e} outside the physical Yarkovsky band; "
        f"JPL SBDB is ≈ 2.9e-14 AU/d²"
    )


def test_evaluate_accepts_fit_orbit_without_reconstruction(apophis_fit, apophis_observations):
    """833t: ``evaluate(fit.orbit, obs)`` returns residuals, never raises.

    The fitted orbit is handed straight to ``evaluate`` with no rebuild.
    A regression that broke the orbit's re-feedability would surface here
    as a raised ``NoValidObservations`` (or any exception) or as an empty
    residual table.
    """
    result = evaluate(apophis_fit.orbit, apophis_observations)
    # One residual row per input observation — nothing dropped, nothing raised.
    assert result.summary.num_obs == len(apophis_observations)
    assert len(result.observations) == len(apophis_observations)
    # A real residual distribution, not a degenerate all-zero / NaN table.
    assert np.isfinite(result.summary.rms_combined_arcsec)
    assert result.summary.rms_combined_arcsec > 0.0


def test_refine_accepts_fit_orbit_and_returns_refeedable_orbit(apophis_fit, apophis_observations):
    """``refine(fit.orbit, obs)`` runs the Bayesian update and returns a
    re-feedable orbit.

    The seed orbit's covariance is the prior; ``refine`` must accept
    ``fit.orbit`` directly (no manual covariance re-attach) and return a
    ``DetermineResult`` whose orbit is itself a valid re-feed input —
    single-row Cartesian, with state and covariance.
    """
    refined = refine(apophis_fit.orbit, apophis_observations)
    assert refined.converged
    ro = refined.orbit
    assert isinstance(ro, CartesianOrbits)
    assert len(ro) == 1
    assert ro.coordinates.covariance is not None
    # The refined orbit is itself acceptable to evaluate with no rebuild —
    # closes the loop on re-feedability.
    ev = evaluate(ro, apophis_observations)
    assert ev.summary.num_obs == len(apophis_observations)


def test_non_grav_survives_propagation_refeed(apophis_fit):
    """o7fk: the fitted force model measurably changes the trajectory.

    ``propagate(fit.orbit)`` includes the recovered Yarkovsky A2; the
    same orbit with its ``non_grav`` stripped is gravity-only. Over the
    ~17-year span to the 2029 encounter the two diverge by far more than
    any integrator noise floor. If the absolute non-grav were silently
    dropped on the output orbit, the two would coincide.
    """
    orbit = apophis_fit.orbit
    epoch0 = orbit.coordinates.epoch[0].as_py()
    targets = np.array([APOPHIS_2029_MJD_TDB])

    # Strip the force model: same state and covariance, but A1/A2/A3 all
    # zero under the inverse-square asteroid g(r) — i.e. gravity-only.
    n = len(orbit)
    grav_only_ng = NonGravParams.from_kwargs(
        a1=[0.0] * n,
        a2=[0.0] * n,
        a3=[0.0] * n,
        model=["inverse_square"] * n,
    )
    orbit_gravity_only = CartesianOrbits.from_kwargs(
        orbit_id=orbit.orbit_id.to_pylist(),
        object_id=orbit.object_id.to_pylist() if orbit.object_id is not None else None,
        coordinates=orbit.coordinates,
        non_grav=grav_only_ng,
    )
    assert orbit_gravity_only.non_grav.a2.to_pylist()[0] == 0.0

    state_ng = propagate(orbit, targets).states.coordinates
    state_grav = propagate(orbit_gravity_only, targets).states.coordinates

    pos_ng = np.array([state_ng.x[0].as_py(), state_ng.y[0].as_py(), state_ng.z[0].as_py()])
    pos_grav = np.array([state_grav.x[0].as_py(), state_grav.y[0].as_py(), state_grav.z[0].as_py()])
    delta_au = float(np.linalg.norm(pos_ng - pos_grav))

    span_years = (APOPHIS_2029_MJD_TDB - epoch0) / 365.25
    assert span_years > 15.0, "expected a multi-decade propagation span"
    # A2 ≈ 3e-14 AU/d² integrated over ~17 yr produces a position
    # difference of ~10⁻⁶ AU (hundreds of km) — orders of magnitude above
    # the ~10⁻¹² AU integrator noise floor. A coincidence here means the
    # fitted force model was dropped on the round-trip.
    assert delta_au > 1e-9, (
        f"non-grav round-trip broken: with-force-model and gravity-only "
        f"trajectories agree to {delta_au:.3e} AU over {span_years:.1f} yr "
        f"(expected > 1e-9 AU). The fitted absolute non-grav was likely "
        f"dropped on the output orbit."
    )


def test_compute_impact_probabilities_accepts_fit_orbit(apophis_fit):
    """qr13: ``compute_impact_probabilities`` accepts ``fit.orbit`` with no
    reconstruction and recovers the 2029 Apophis Earth encounter.

    The fitted orbit (state + covariance + non-grav) goes straight in.
    The detector must surface the published 2029-04-13 close approach at a
    geocentric miss distance of ≈ 38,000 km, with a vanishing linear
    impact probability (Apophis does not strike Earth in 2029).
    """
    ips = compute_impact_probabilities(
        apophis_fit.orbit,
        APOPHIS_IP_END_MJD_TDB,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    assert len(ips) >= 1, "expected the 2029 Earth close approach to be detected"

    bodies = ips.body.to_pylist()
    assert all(b == "Earth" for b in bodies), f"body_filter not honored: {bodies}"

    epochs = ips.epochs.mjd.to_pylist()
    miss_km = ips.miss_distance_km.to_pylist()
    ip_linear = ips.ip_linear.to_pylist()

    # Pick the row nearest the published 2029 encounter epoch.
    idx = int(np.argmin([abs(e - APOPHIS_2029_MJD_TDB) for e in epochs]))
    assert abs(epochs[idx] - APOPHIS_2029_MJD_TDB) < 5.0, (
        f"detected encounter epoch {epochs[idx]:.2f} far from the published "
        f"2029-04-13 value {APOPHIS_2029_MJD_TDB}"
    )
    assert APOPHIS_2029_MISS_KM_LO < miss_km[idx] < APOPHIS_2029_MISS_KM_HI, (
        f"2029 miss distance {miss_km[idx]:.0f} km outside the published ~38,000 km geocentric band"
    )
    # Apophis is not an impactor in 2029 — linear IP is effectively zero.
    assert ip_linear[idx] < 1e-6, f"unexpected non-trivial 2029 impact probability {ip_linear[idx]}"


def test_default_auto_fit_round_trips_non_grav_as_none(apophis_observations):
    """The plug-and-play chain works for a gravity-only fit too.

    Default ``AUTO`` declines to escalate to non-grav on this arc — the
    honest result, since a ~400-obs-per-apparition arc does not strongly
    constrain Yarkovsky on its own. The contract is that the chain still
    runs clean and that non-grav round-trips faithfully as ``None`` rather
    than being fabricated.
    """
    fit = determine(apophis_observations, config=ODConfig(solve_for=SolveForParams.AUTO))
    assert fit.converged
    assert fit.solve_for_used == SolveForParams.STATE_ONLY
    # NonGravParams table is present but every coefficient is null — the
    # state-only fit attaches no force model.
    ng = fit.orbit.non_grav
    assert ng is None or ng.a2.to_pylist()[0] is None

    # The state-only orbit is still re-feedable into evaluate.
    ev = evaluate(fit.orbit, apophis_observations)
    assert ev.summary.num_obs == len(apophis_observations)


def test_fit_orbit_carries_non_grav_covariance(apophis_fit):
    """A StateAndNonGrav fit carries its non-grav covariance on
    ``fit.orbit.non_grav``, so the orbit re-feeds into a non-grav refine
    without losing its prior."""
    cov = apophis_fit.orbit.non_grav.covariance.to_pylist()[0]
    assert cov is not None, "fitted orbit dropped its non-grav covariance"
    assert len(cov) == 9
    mat = np.asarray(cov, dtype=float).reshape(3, 3)
    assert np.all(np.isfinite(mat))
    assert np.all(np.diag(mat) > 0.0), "non-grav variances must be positive"

    # It must be the fitted POSTERIOR (the [6:9, 6:9] block of the 9x9), not
    # the escalation seed prior (sigma=1e-7 -> 1e-14 diagonal). The fix is
    # sourcing the authoritative covariance_9x9 block.
    posterior_block = np.asarray(apophis_fit.covariance_9x9)[6:9, 6:9]
    np.testing.assert_allclose(mat, posterior_block, rtol=1e-9, atol=0)
    sigma_a2 = np.sqrt(mat[1, 1])
    a2 = abs(apophis_fit.orbit.non_grav.a2.to_pylist()[0])
    # The posterior sigma_A2 is a sensible fraction of |A2| (a real fit), NOT
    # the 1e-7 prior floor (which would be ~3e6 x |A2|).
    assert sigma_a2 < a2, f"sigma_A2 {sigma_a2:.2e} not tighter than |A2| {a2:.2e} (prior leaked?)"


def test_refine_state_and_non_grav_round_trips(apophis_fit, apophis_observations):
    """Re-feeding a StateAndNonGrav fit into a StateAndNonGrav refine
    converges — the 9x9 prior is carried on ``fit.orbit`` (no opaque
    'normal matrix is singular')."""
    refined = refine(
        apophis_fit.orbit,
        apophis_observations,
        config=ODConfig(solve_for=SolveForParams.STATE_AND_NONGRAV),
    )
    assert refined.converged
    # Chained re-feed: the refined orbit again carries its non-grav covariance.
    assert refined.orbit.non_grav.covariance.to_pylist()[0] is not None


def test_evaluate_uses_seed_non_grav_not_gravity_only(apophis_fit, apophis_observations):
    """The OD input path threads the seed orbit's non-grav: evaluating the
    fitted (A2-bearing) orbit yields very different residuals than the same
    state stripped to gravity-only. Previously the seed force model was
    silently dropped at the PyO3 ``build_orbit_from_dict`` boundary."""
    ev_ng = evaluate(apophis_fit.orbit, apophis_observations)
    grav = CartesianOrbits.from_kwargs(
        orbit_id=apophis_fit.orbit.orbit_id.to_pylist(),
        object_id=apophis_fit.orbit.object_id.to_pylist(),
        coordinates=apophis_fit.orbit.coordinates,
    )
    ev_grav = evaluate(grav, apophis_observations)
    assert abs(ev_ng.summary.rms_combined_arcsec - ev_grav.summary.rms_combined_arcsec) > 1e-3, (
        "seed non-grav was not threaded into evaluate"
    )
