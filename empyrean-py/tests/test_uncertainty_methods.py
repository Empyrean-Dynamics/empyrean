"""Behavioral contracts for the sampling uncertainty methods
(``SIGMA_POINT`` / ``MONTE_CARLO``) in :func:`empyrean.propagate` and
:func:`empyrean.generate_ephemeris`.

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
  probability (seed-reproducibly), and since villeneuve 1.25.0
  (bd empyrean-z28n8) also publishes the per-epoch ensemble covariance —
  the centered sample second moment about the ensemble mean, tagged
  ``monte_carlo`` with the run's seed. Two absences stay honest: an
  ensemble below the 7-draw rank floor publishes nothing (n draws give
  rank ≤ n−1, and n ∈ {0, 1} would publish NaN / all-zero matrices), and
  a run seeded from system entropy publishes a covariance with no seed
  beside it, because that ensemble is not reproducible by construction.
* ``generate_ephemeris`` HONORS the sampling methods since villeneuve
  1.25.0 (bd empyrean-848gm.1): the member set flies through the
  generation pipeline and the row covariance is the ensemble's sky
  moment — numerically distinct from the first-order projection, seed-
  reproducible for Monte Carlo — and what the engine does not deliver
  (a non-canonical sigma knob, fewer than 8 draws) comes back as the
  engine's own typed refusal naming the method, never as a silent
  downgrade. The wrapper adds no gate of its own on this seam, so it is
  exactly as strict as the C ABI and the Rust wrapper beneath it.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import Epochs, MonteCarlo, SigmaPoint, UncertaintyMethod
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
def times() -> Epochs:
    # A multi-year arc so the sigma-point sample covariance has room to
    # diverge from the linear one even in this weakly-nonlinear regime.
    return Epochs.from_mjd(
        np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 365.0, _EPOCH_MJD_TDB + 730.0]),
        scale="tdb",
    )


# ══════════════════════════════════════════════════════════════════
#  propagate — SIGMA_POINT
# ══════════════════════════════════════════════════════════════════


def test_propagate_sigma_point_covariance_differs_from_first_order(
    orbit: CartesianOrbits, times: Epochs
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


def test_propagate_sigma_point_honors_dataclass(orbit: CartesianOrbits, times: Epochs) -> None:
    """The ``SigmaPoint`` dataclass (default params) reaches the engine
    and produces a sigma-point covariance, not a downgraded first-order
    one — the wire-dict serialization of the dataclass is lossy, so this
    guards the flat-arg authority."""
    res = empyrean.propagate(orbit, times, uncertainty_method=SigmaPoint(), tagged_covariance=True)
    assert set(res.tagged_covariance.kind.to_pylist()) == {"sigma_point"}


def test_propagate_sigma_point_non_default_params_raise(
    orbit: CartesianOrbits, times: Epochs
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


def test_propagate_monte_carlo_publishes_the_ensemble_covariance(
    orbit: CartesianOrbits, times: Epochs
) -> None:
    """``MONTE_CARLO`` publishes a per-epoch state covariance: the
    centered sample second moment of the propagated ensemble, tagged
    ``monte_carlo`` and carrying the seed that produced it.

    It must agree with the first-order covariance to sampling noise on
    this mildly nonlinear arc — the cross-check that separates a real
    ensemble moment from a mislabelled linear readback. The relative
    standard error of a sampled sigma at n = 256 is ~4.4%; the band below
    is villeneuve's own, wide enough to absorb that plus genuine
    nonlinearity over the arc.

    Mirrors villeneuve ``tests/test_uncertainty_methods.rs::
    test_monte_carlo_populates_the_single_epoch_covariance``
    (bd empyrean-z28n8).
    """
    res_fo = empyrean.propagate(orbit, times, uncertainty_method=UncertaintyMethod.FIRST_ORDER)
    res_mc = empyrean.propagate(
        orbit, times, uncertainty_method=MonteCarlo(n_samples=256, seed=3), tagged_covariance=True
    )

    m_fo = res_fo.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_fo).all(), "first-order state covariance not finite"

    m_mc = res_mc.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_mc).all(), (
        "MONTE_CARLO published no state covariance — the ensemble moment is the "
        "deliverable of this path since villeneuve 1.25.0"
    )
    # A covariance is a second moment: its diagonal cannot be negative.
    assert (np.diagonal(m_mc, axis1=1, axis2=2) > 0.0).all()

    # The rows say which construction produced them, and name the seed —
    # without both, a sampled covariance is indistinguishable from a
    # linear one and cannot be reproduced.
    assert set(res_mc.tagged_covariance.kind.to_pylist()) == {"monte_carlo"}
    assert set(res_mc.tagged_covariance.mc_seed.to_pylist()) == {3}

    # Sampling-noise cross-check against the linear transport.
    pos_sigma = lambda m: np.sqrt(m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])  # noqa: E731
    ratio = pos_sigma(m_mc) / pos_sigma(m_fo)
    assert np.all((ratio > 0.8) & (ratio < 1.25)), (
        f"MC ensemble sigma vs first-order — ratios {ratio} outside the sampling-noise band"
    )

    # Entrywise, under the correlation normalization |a−b| / √(b_ii·b_jj).
    # The scalar position-sigma ratio above is blind to exactly the
    # defects this docstring claims to catch: a wrong basis, a dropped
    # velocity block, or a truncation slip moves OFF-DIAGONAL structure
    # while leaving the position trace almost untouched. Single-entry
    # sampling noise is ~1/√n ≈ 6.3% at n = 256, but the worst of 36
    # correlated entries runs 3–4× that — measured 0.229 here on the
    # frozen seed, and villeneuve measures 0.223 on its own. The 0.4 bar
    # is villeneuve's, sitting above the extreme-value band and many
    # orders below what a basis error produces.
    denom = np.sqrt(
        m_fo.diagonal(axis1=1, axis2=2)[:, :, None] * m_fo.diagonal(axis1=1, axis2=2)[:, None, :]
    )
    worst = float(np.abs((m_mc - m_fo) / denom).max())
    assert worst < 0.4, (
        f"MC vs first-order entrywise gap {worst:.3f} past the n = 256 sampling budget — "
        "a scalar sigma ratio would not have caught this"
    )


def test_propagate_monte_carlo_below_the_rank_floor_publishes_nothing(
    orbit: CartesianOrbits, times: Epochs
) -> None:
    """Below the 7-draw rank floor no covariance is published at all.

    ``n`` draws give a sample moment of rank ≤ ``n−1``, so a 6×6 built
    from fewer than 7 is rank-deficient by construction; publishing one
    would hand back a NaN or all-zero matrix wearing a covariance's name.
    The absence is the honest answer, and it appears exactly at the
    floor.

    Mirrors villeneuve ``tests/test_uncertainty_methods.rs::
    test_monte_carlo_moments_floor``. NOTE: villeneuve also names the
    reason once per run via ``PropagationWarning::
    MonteCarloEnsembleBelowFloor``; that warning has no channel on the
    propagate path at any layer of this distribution (the C ABI marshals
    a warnings list for ephemeris generation only), so Python can see the
    absence but not its reason. Tracked as the parity gap it is rather
    than asserted here.
    """
    for n in (1, 6):
        res = empyrean.propagate(orbit, times, uncertainty_method=MonteCarlo(n_samples=n, seed=3))
        m = res.states.coordinates.covariance.to_matrix()
        assert np.isnan(m).all(), (
            f"n = {n} is below the rank floor — a published covariance here would be "
            "rank-deficient by construction"
        )

    at_floor = empyrean.propagate(
        orbit, times, uncertainty_method=MonteCarlo(n_samples=7, seed=3), tagged_covariance=True
    )
    m_floor = at_floor.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_floor).all(), "at the floor the ensemble covariance must publish"
    assert set(at_floor.tagged_covariance.kind.to_pylist()) == {"monte_carlo"}


def test_propagate_monte_carlo_without_a_seed_publishes_no_seed(
    orbit: CartesianOrbits, times: Epochs
) -> None:
    """An entropy-seeded run still publishes its ensemble covariance, but
    reports NO seed beside it.

    ``MonteCarlo(seed=None)`` draws from system entropy, so the ensemble
    is not reproducible by construction and there is no seed to report.
    The seed column is null rather than ``0`` — zero is a real,
    reproducible seed, and reporting it would offer a reproducibility the
    run cannot deliver. This is the Python end of the C ABI's
    ``has_mc_seed`` presence flag: ``kind == monte_carlo`` does not imply
    a seed, so read the flag (here, the null), not the kind.
    """
    res = empyrean.propagate(
        orbit,
        times,
        uncertainty_method=MonteCarlo(n_samples=64, seed=None),
        tagged_covariance=True,
    )
    m = res.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), "an entropy-seeded run still publishes its ensemble covariance"
    assert set(res.tagged_covariance.kind.to_pylist()) == {"monte_carlo"}
    assert set(res.tagged_covariance.mc_seed.to_pylist()) == {None}, (
        "an entropy-seeded ensemble has no seed to report — null, never 0"
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
        empyrean.get_observer_states(
            ["500"], Epochs.from_mjd(np.array([_EPOCH_MJD_TDB]), scale="tdb")
        ).coordinates.r[0]
    )
    r1 = np.asarray(
        empyrean.get_observer_states(
            ["500"], Epochs.from_mjd(np.array([_EPOCH_MJD_TDB + dt]), scale="tdb")
        ).coordinates.r[0]
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
    times = Epochs.from_mjd(np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 20.0]), scale="tdb")
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


def test_propagate_unknown_uncertainty_method_raises(orbit: CartesianOrbits, times: Epochs) -> None:
    """An out-of-range integer tag is a typed ``ValueError`` naming the
    supported set — never a raw ``RuntimeError`` leaking enum ints."""
    with pytest.raises(ValueError, match="unsupported uncertainty_method"):
        empyrean.propagate(orbit, times, uncertainty_method=7)


@pytest.mark.parametrize("method", [UncertaintyMethod.SECOND_ORDER, UncertaintyMethod.AUTO])
def test_propagate_analytic_methods_attach_finite_covariance(
    orbit: CartesianOrbits, times: Epochs, method
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
    return empyrean.get_observer_states(
        ["500"],
        Epochs.from_mjd(np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 30.0]), scale="tdb"),
    )


def _sky_cov(eph) -> np.ndarray:
    cov = eph.ephemeris.coordinates.covariance
    assert cov is not None, "sky covariance column missing"
    return cov.to_matrix()


@pytest.mark.parametrize("method", [UncertaintyMethod.SIGMA_POINT, SigmaPoint()])
def test_generate_ephemeris_sigma_point_delivers_a_sampled_sky_covariance(
    orbit: CartesianOrbits, observers, method
) -> None:
    """Mirrors villeneuve ``sigma_sky_matches_linear_on_a_tight_prior``:
    on this tight (1e-6 AU) prior the unscented sky moment agrees with
    the first-order projection to better than a percent on the diagonal
    — and is NOT the same array, which is the delivery witness (the old
    hidden fallback returned the first-order matrix under the sigma-point
    name; the wrapper then refused the method outright)."""
    sp = _sky_cov(empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method))
    fo = _sky_cov(
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=UncertaintyMethod.FIRST_ORDER
        )
    )
    assert sp.shape == fo.shape
    assert np.isfinite(sp).all(), "sigma-point sky covariance not finite"
    assert (np.diagonal(sp, axis1=1, axis2=2) >= 0).all()
    assert not np.array_equal(sp, fo), (
        "sigma-point sky covariance is bit-identical to first order — the sampled "
        "delivery did not run"
    )
    d_sp = np.diagonal(sp, axis1=1, axis2=2)
    d_fo = np.diagonal(fo, axis1=1, axis2=2)
    rel = np.abs(d_sp - d_fo) / np.abs(d_fo)
    assert (rel < 1e-2).all(), f"sigma vs linear diag rel {rel.max():.2e} on a tight prior"


def test_generate_ephemeris_monte_carlo_is_seed_reproducible(
    orbit: CartesianOrbits, observers
) -> None:
    """Mirrors villeneuve ``mc_sky_is_seed_reproducible_and_floor_gated``:
    the same seed reproduces the delivered rows bit for bit, a different
    seed moves the sampled moments, and the Monte-Carlo sky is not the
    first-order projection."""
    a = _sky_cov(
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=MonteCarlo(n_samples=64, seed=7)
        )
    )
    b = _sky_cov(
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=MonteCarlo(n_samples=64, seed=7)
        )
    )
    c = _sky_cov(
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=MonteCarlo(n_samples=64, seed=8)
        )
    )
    fo = _sky_cov(
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=UncertaintyMethod.FIRST_ORDER
        )
    )
    assert np.isfinite(a).all()
    assert np.array_equal(a, b), "same seed must reproduce the sky moments bit-exactly"
    assert not np.array_equal(a, c), "a different seed must move the sampled moments"
    assert not np.array_equal(a, fo), "Monte-Carlo sky is not the first-order projection"


@pytest.mark.parametrize(
    ("method", "name"),
    [
        (SigmaPoint(n_sigma=2.0), "SigmaPoint"),
        (SigmaPoint(samples_per_plane=4), "SigmaPoint"),
        (MonteCarlo(n_samples=4, seed=7), "MonteCarlo"),
    ],
)
def test_generate_ephemeris_engine_refuses_non_canonical_sampling_by_name(
    orbit: CartesianOrbits, observers, method, name
) -> None:
    """What the engine does not deliver it refuses BY NAME, from the
    engine (``RuntimeError``), not from a wrapper gate: the unscented set
    is parameter-free, and an ensemble under 8 draws has no full-rank sky
    moment."""
    with pytest.raises(RuntimeError, match=name):
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
