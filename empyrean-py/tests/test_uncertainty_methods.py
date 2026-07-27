"""Behavioral contracts for the sampling uncertainty methods
(``SIGMA_POINT`` / ``MONTE_CARLO``) in :func:`empyrean.propagate` and
:func:`empyrean.generate_ephemeris` — bd empyrean-02j4.

These functions previously mishandled the two sampling methods, and the
two failed *differently*:

* ``propagate`` eager-rejected ``SIGMA_POINT`` / ``MONTE_CARLO`` with a
  raw ``RuntimeError`` leaking enum ints, and discarded the sampling
  params (``sigma_n_sigma`` / ``sigma_samples_per_plane`` /
  ``mc_n_samples`` / ``mc_seed``) it received.
* ``generate_ephemeris`` *silently ignored* the requested method for all
  five values, always running the default (first-order) covariance
  transport — a hidden quality degradation.

No test exercised either path, which is exactly why the bug shipped. This
module is the differential coverage that pins the fixed behavior:

* ``propagate(SIGMA_POINT)`` reconstructs a genuine sample-based state
  covariance, tagged ``sigma_point`` and numerically distinct from the
  linear first-order one.
* ``propagate(MONTE_CARLO)`` runs and reports the Monte-Carlo impact
  probability (seed-reproducibly); it produces no per-epoch state
  covariance (that is the engine contract, not a bug), so covariance-
  bearing rows come back with the standard "absent covariance"
  representation.
* ``generate_ephemeris`` rejects the sampling methods with a typed,
  descriptive ``ValueError`` instead of silently downgrading to first
  order (the sky-plane covariance is a first-order STM projection that
  cannot consume a sampled ensemble).
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import MonteCarlo, SigmaPoint, UncertaintyMethod
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.events import EventConfig

_EPOCH_MJD_TDB = 61000.0


def _orbit_with_covariance() -> CartesianOrbits:
    """A self-contained, network-free heliocentric orbit with a finite
    state covariance — enough to exercise the covariance-propagation
    path without querying SBDB/MPC."""
    cov = np.zeros((1, 6, 6))
    for k, d in enumerate([1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16]):
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


@pytest.fixture(scope="module")
def times() -> np.ndarray:
    # A multi-year arc so the sigma-point sample covariance has room to
    # diverge from the linear one even in this weakly-nonlinear regime.
    return np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 365.0, _EPOCH_MJD_TDB + 730.0])


# ══════════════════════════════════════════════════════════════════
#  propagate — SIGMA_POINT
# ══════════════════════════════════════════════════════════════════


def test_propagate_sigma_point_covariance_differs_from_first_order(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """``SIGMA_POINT`` must produce a genuine, provenance-tagged sample
    covariance that differs from the linear first-order one — not a
    silently-substituted copy of it (the silent-ignore failure mode).
    """
    res_fo = empyrean.propagate(
        orbit, times, uncertainty_method=UncertaintyMethod.FIRST_ORDER, tagged_covariance=True
    )
    res_sp = empyrean.propagate(
        orbit, times, uncertainty_method=UncertaintyMethod.SIGMA_POINT, tagged_covariance=True
    )

    m_fo = res_fo.states.coordinates.covariance.to_matrix()
    m_sp = res_sp.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_fo).all(), "first-order state covariance not finite"
    assert np.isfinite(m_sp).all(), "sigma-point state covariance not finite"
    # A real covariance has non-negative variances on the diagonal.
    assert (np.diagonal(m_sp, axis1=1, axis2=2) >= 0).all()

    # Provenance: the sigma-point rows must be tagged as a sigma-point
    # sample covariance (the unambiguous proof the sampling path ran),
    # while first-order rows stay linear.
    assert set(res_fo.tagged_covariance.kind.to_pylist()) == {"linear"}
    assert set(res_sp.tagged_covariance.kind.to_pylist()) == {"sigma_point"}

    # The reconstructed covariance is numerically distinct from the linear
    # mapping (a "does it run" test would pass even under the silent-ignore
    # bug, so assert an actual numerical difference).
    assert not np.array_equal(m_sp, m_fo), (
        "sigma-point covariance is bit-identical to first-order — the method was ignored"
    )


def test_propagate_sigma_point_honors_dataclass(orbit: CartesianOrbits, times: np.ndarray) -> None:
    """The ``SigmaPoint`` dataclass (default params) reaches the engine
    and produces a sigma-point covariance, not a downgraded first-order
    one — the wire-dict serialization of the dataclass is lossy, so this
    guards the flat-arg authority."""
    res = empyrean.propagate(orbit, times, uncertainty_method=SigmaPoint(), tagged_covariance=True)
    assert set(res.tagged_covariance.kind.to_pylist()) == {"sigma_point"}


def test_propagate_sigma_point_non_default_params_raise(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """Non-default sigma-point params are passed through unchanged and
    rejected LOUDLY by the engine (the canonical 2N+1 construction is
    parameter-free) — the wrapper must not silently clamp them to the
    defaults, which would fabricate a covariance the caller did not ask
    for."""
    with pytest.raises(RuntimeError, match=r"[Ss]igma-point construction"):
        empyrean.propagate(orbit, times, uncertainty_method=SigmaPoint(n_sigma=2.0))


# ══════════════════════════════════════════════════════════════════
#  propagate — MONTE_CARLO
# ══════════════════════════════════════════════════════════════════


def test_propagate_monte_carlo_omits_state_covariance(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """``MONTE_CARLO`` runs but produces NO per-epoch state covariance —
    its deliverable is the Monte-Carlo impact probability, not a sampled
    state covariance. The covariance therefore comes back as the standard
    "absent" (all-NaN) representation, measurably different from the
    finite first-order covariance on the same input.
    """
    res_fo = empyrean.propagate(orbit, times, uncertainty_method=UncertaintyMethod.FIRST_ORDER)
    res_mc = empyrean.propagate(orbit, times, uncertainty_method=MonteCarlo(n_samples=64, seed=7))

    m_fo = res_fo.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_fo).all(), "first-order state covariance not finite"

    # Monte-Carlo attaches no state covariance: quivr represents the
    # absent covariance as all-NaN (identical to a first-order propagate
    # of an orbit that carries no input covariance). This is the honest
    # "no covariance here" signal, not a NaN masquerading as a real one.
    m_mc = res_mc.states.coordinates.covariance.to_matrix()
    assert np.isnan(m_mc).all(), (
        "MONTE_CARLO produced a (partly) finite state covariance — the engine does not "
        "reconstruct one on this path; a finite value would be a spurious readback"
    )


def _earth_approaching_orbit() -> CartesianOrbits:
    """A network-free Earth-approaching orbit that reliably produces a
    close approach (and, with a broad covariance, a partial Monte-Carlo
    impact probability). Built by co-moving with Earth's geocenter state
    at ``_EPOCH_MJD_TDB`` plus a small offset + closing velocity.
    """
    empyrean.initialize()
    dt = 0.02
    r0 = np.asarray(
        empyrean.get_observer_states(["500"], np.array([_EPOCH_MJD_TDB])).coordinates.r[0]
    )
    r1 = np.asarray(
        empyrean.get_observer_states(["500"], np.array([_EPOCH_MJD_TDB + dt])).coordinates.r[0]
    )
    v = (r1 - r0) / dt  # finite-difference Earth velocity (ICRF, SSB origin)
    return CartesianOrbits.from_kwargs(
        orbit_id=["imp"],
        object_id=["imp"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_EPOCH_MJD_TDB]),
            x=[r0[0] + 0.0015],
            y=[r0[1]],
            z=[r0[2]],
            vx=[v[0]],
            vy=[v[1] - 3.0e-4],
            vz=[v[2]],
            frame="icrf",
            origin=["SSB"],
            covariance=CartesianCovariance.from_matrix(
                np.diag([5e-7, 5e-7, 5e-7, 5e-9, 5e-9, 5e-9])[None, :, :]
            ),
        ),
    )


def _possible_impact_ip_mc(result) -> np.ndarray | None:
    """Extract the ``ip_mc`` column of the possible-impact events, or
    ``None`` if the scenario produced none."""
    pi = getattr(result.events, "possible_impacts", None)
    if pi is None or len(pi) == 0:
        return None
    return np.asarray(pi.ip_mc.to_numpy(zero_copy_only=False))


@pytest.fixture(scope="module")
def mc_close_approach():
    """Run FIRST_ORDER + Monte-Carlo (two identical seeds + one different
    seed) once over an Earth close approach, sharing the (expensive)
    sample propagations across the seed / differential tests."""
    orbit = _earth_approaching_orbit()
    times = np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 20.0])
    events = EventConfig(close_approaches=True, possible_impacts=True, body_filter=["Earth"])

    def run(method):
        return empyrean.propagate(orbit, times, uncertainty_method=method, events=events)

    return {
        "fo": _possible_impact_ip_mc(run(UncertaintyMethod.FIRST_ORDER)),
        "mc_seed7_a": _possible_impact_ip_mc(run(MonteCarlo(n_samples=256, seed=7))),
        "mc_seed7_b": _possible_impact_ip_mc(run(MonteCarlo(n_samples=256, seed=7))),
        "mc_seed13": _possible_impact_ip_mc(run(MonteCarlo(n_samples=256, seed=13))),
    }


def test_propagate_monte_carlo_populates_ip_mc_vs_first_order(mc_close_approach) -> None:
    """Differential: over a close approach, ``MONTE_CARLO`` populates a
    finite Monte-Carlo impact probability (``ip_mc``) while
    ``FIRST_ORDER`` leaves it NaN — the measurable, method-driven
    difference for Monte-Carlo in ``propagate``."""
    ip_mc = mc_close_approach["mc_seed7_a"]
    ip_fo = mc_close_approach["fo"]
    if ip_mc is None:
        pytest.skip("synthetic close-approach produced no possible-impact event")

    assert np.isfinite(ip_mc).any(), "MONTE_CARLO left ip_mc all-NaN over a close approach"
    finite = ip_mc[np.isfinite(ip_mc)]
    assert ((finite >= 0.0) & (finite <= 1.0)).all(), "ip_mc outside [0, 1]"
    # FIRST_ORDER does not compute a Monte-Carlo IP.
    if ip_fo is not None:
        assert np.isnan(ip_fo).all(), "FIRST_ORDER unexpectedly populated ip_mc"


def test_propagate_monte_carlo_seed_reproducible(mc_close_approach) -> None:
    """Same ``seed`` → bit-identical Monte-Carlo impact probability;
    a different ``seed`` changes it (the sampling is seeded and the seed
    genuinely threads through, not ignored)."""
    a = mc_close_approach["mc_seed7_a"]
    b = mc_close_approach["mc_seed7_b"]
    c = mc_close_approach["mc_seed13"]
    if a is None or b is None or c is None:
        pytest.skip("synthetic close-approach produced no possible-impact event")

    # Reproducibility: identical seed → identical result.
    np.testing.assert_array_equal(a, b)

    # Seed sensitivity: a different seed perturbs the estimate — unless the
    # scenario is degenerate (ip_mc pinned at 0 or 1, where sampling noise
    # cannot show), in which case the reproducibility assertion above still
    # carries the test.
    a_fin = a[np.isfinite(a)]
    if a_fin.size and not np.all((a_fin == 0.0) | (a_fin == 1.0)):
        assert not np.array_equal(a, c), (
            "different seeds produced identical ip_mc — the seed appears to be ignored"
        )


# ══════════════════════════════════════════════════════════════════
#  propagate — invalid method
# ══════════════════════════════════════════════════════════════════


def test_propagate_unknown_uncertainty_method_raises(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """An out-of-range integer tag is a typed ``ValueError`` naming the
    supported set — never a raw ``RuntimeError`` leaking enum ints."""
    with pytest.raises(ValueError, match="unsupported uncertainty_method"):
        empyrean.propagate(orbit, times, uncertainty_method=7)


@pytest.mark.parametrize("method", [UncertaintyMethod.SECOND_ORDER, UncertaintyMethod.AUTO])
def test_propagate_analytic_methods_attach_finite_covariance(
    orbit: CartesianOrbits, times: np.ndarray, method
) -> None:
    """Non-regression: making the flat ``uncertainty_method`` arg
    authoritative over the wire dict must not have disturbed the analytic
    methods — SECOND_ORDER and AUTO still attach a finite state covariance
    (FIRST_ORDER is already covered by the sigma-point differential)."""
    res = empyrean.propagate(orbit, times, uncertainty_method=method)
    m = res.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), f"{method.value}: state covariance not finite"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all()


# ══════════════════════════════════════════════════════════════════
#  generate_ephemeris — sampling methods rejected
# ══════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def observers():
    empyrean.initialize()
    return empyrean.get_observer_states(["500"], np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 30.0]))


@pytest.mark.parametrize(
    "method",
    [
        UncertaintyMethod.SIGMA_POINT,
        SigmaPoint(),
        UncertaintyMethod.MONTE_CARLO,
        MonteCarlo(n_samples=64, seed=7),
    ],
)
def test_generate_ephemeris_rejects_sampling_methods(
    orbit: CartesianOrbits, observers, method
) -> None:
    """The core ephemeris fix: a sampling method is rejected with a
    typed, descriptive ``ValueError`` — never silently downgraded to the
    first-order sky covariance (the old hidden-fallback behavior)."""
    with pytest.raises(ValueError, match="sampling uncertainty methods"):
        empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)


def test_generate_ephemeris_unknown_uncertainty_method_raises(
    orbit: CartesianOrbits, observers
) -> None:
    """An out-of-range integer tag is a typed ``ValueError``."""
    with pytest.raises(ValueError, match="unsupported uncertainty_method"):
        empyrean.generate_ephemeris(orbit, observers, uncertainty_method=7)


@pytest.mark.parametrize(
    "method",
    [UncertaintyMethod.FIRST_ORDER, UncertaintyMethod.SECOND_ORDER, UncertaintyMethod.AUTO],
)
def test_generate_ephemeris_analytic_methods_still_work(
    orbit: CartesianOrbits, observers, method
) -> None:
    """Regression guard: the supported analytic methods still produce a
    finite sky-plane covariance after the fix (the flat ``uncertainty_method``
    arg is now authoritative and must not have broken them)."""
    eph = empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)
    cov = eph.ephemeris.coordinates.covariance
    assert cov is not None, f"{method.value}: sky covariance column missing"
    m = cov.to_matrix()
    assert np.isfinite(m).all(), f"{method.value}: sky covariance not finite"
