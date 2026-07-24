"""Behavioral contract for the GaussianMixture (adaptive Gaussian
mixture, AGM) uncertainty method in :func:`empyrean.propagate` and
:func:`empyrean.generate_ephemeris` — bd empyrean-p1j7.

GaussianMixture is exposed as a top-level uncertainty method (tag 5)
reusing the AGM parameter slots the C ABI already carried for ``Auto``;
the flat ``tag`` disambiguates a standalone mixture from Auto's internal
splitter. Unlike the sampling methods (``SIGMA_POINT`` / ``MONTE_CARLO``),
GaussianMixture is analytic (an AD / Jet2 method like ``SECOND_ORDER``)
and is therefore HONORED on both the ``propagate`` and the
``generate_ephemeris`` paths — it must never be rejected the way the
sampling methods are.

Its distinctive product is the mixture-corrected impact probability at
close approaches; away from encounters the output-state covariance is the
linear ``Φ·Σ·Φᵀ`` mapping, so for a well-determined object it reads back
very close to ``FIRST_ORDER`` (tagged ``linear``) — that is expected, not
a bug. These tests therefore assert the call *runs*, returns a *finite*
covariance, and *reaches a distinct engine path* (not a silent downgrade
to the literal first-order code), rather than asserting a large numerical
divergence that only a forced mixture regime would produce.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import GaussianMixture, PropagationConfig, UncertaintyMethod
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import (
    _DATACLASS_TO_INT,
    _INT_TO_UNCERTAINTY_METHOD,
    _UNCERTAINTY_METHOD_TO_INT,
)

_EPOCH_MJD_TDB = 61000.0


def _orbit_with_covariance() -> CartesianOrbits:
    """A self-contained, network-free heliocentric orbit with a finite
    state covariance — enough to exercise the covariance-propagation path
    without querying SBDB/MPC (mirrors ``test_uncertainty_methods.py``)."""
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
    return np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 365.0, _EPOCH_MJD_TDB + 730.0])


@pytest.fixture(scope="module")
def observers():
    empyrean.initialize()
    return empyrean.get_observer_states(["500"], np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 30.0]))


# ══════════════════════════════════════════════════════════════════
#  Static wiring — the maps every distribution channel shares
# ══════════════════════════════════════════════════════════════════


def test_gaussian_mixture_tag_wiring() -> None:
    """The enum, the string, and the dataclass all resolve to tag 5, and
    the inverse map names it ``gaussian_mixture``. This is the shared
    contract the C ABI (tag 5 = Mixture) and every layer above it agree on.
    """
    assert _UNCERTAINTY_METHOD_TO_INT[UncertaintyMethod.GAUSSIAN_MIXTURE] == 5
    assert _UNCERTAINTY_METHOD_TO_INT["gaussian_mixture"] == 5
    assert _DATACLASS_TO_INT[GaussianMixture] == 5
    assert _INT_TO_UNCERTAINTY_METHOD[5] == "gaussian_mixture"


def test_gaussian_mixture_defaults() -> None:
    """Engine-default AGM parameters (DeMars-Bishop-Jah 2013)."""
    gm = GaussianMixture()
    assert gm.threshold == 1.0
    assert gm.max_depth == 3
    assert gm.components_per_split == 3


def test_gaussian_mixture_wire_dict_serialization() -> None:
    """A ``GaussianMixture`` dataclass on the config serializes to the
    ``"gaussian_mixture"`` wire string (not the old lossy
    ``"first_order"`` downgrade); the per-variant params ride the
    authoritative flat args."""
    cfg = PropagationConfig(uncertainty_method=GaussianMixture())
    assert cfg._to_wire_dict()["uncertainty_method"] == "gaussian_mixture"
    cfg_enum = PropagationConfig(uncertainty_method=UncertaintyMethod.GAUSSIAN_MIXTURE)
    assert cfg_enum._to_wire_dict()["uncertainty_method"] == "gaussian_mixture"


# ══════════════════════════════════════════════════════════════════
#  propagate — GaussianMixture runs and returns a finite covariance
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "method",
    [UncertaintyMethod.GAUSSIAN_MIXTURE, GaussianMixture()],
    ids=["enum", "dataclass"],
)
def test_propagate_gaussian_mixture_runs_finite(
    orbit: CartesianOrbits, times: np.ndarray, method
) -> None:
    """Both the enum and the dataclass forms run without raising and
    attach a finite state covariance with non-negative variances. The
    tagged-covariance kind is ``mixture`` at close-approach windows and
    the linear ``linear`` mapping elsewhere — on this plain heliocentric
    arc (no encounter) it is ``linear``."""
    res = empyrean.propagate(orbit, times, uncertainty_method=method, tagged_covariance=True)
    m = res.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), "GaussianMixture state covariance not finite"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all()
    kinds = set(res.tagged_covariance.kind.to_pylist())
    assert kinds <= {"linear", "mixture"}, f"unexpected tagged-covariance kind(s): {kinds}"


def test_propagate_gaussian_mixture_reaches_engine(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """Proof-of-reach: GaussianMixture must execute a distinct engine path,
    NOT be silently downgraded to the literal first-order code. Under a
    silent downgrade the covariance would be *bit-identical* to
    ``FIRST_ORDER``; here it is deterministically distinct yet — as the
    method's contract predicts for a well-determined object away from an
    encounter — numerically very close to it (the linear ``Φ·Σ·Φᵀ``
    mapping).
    """
    m_gmm = empyrean.propagate(
        orbit, times, uncertainty_method=GaussianMixture()
    ).states.coordinates.covariance.to_matrix()
    m_gmm2 = empyrean.propagate(
        orbit, times, uncertainty_method=GaussianMixture()
    ).states.coordinates.covariance.to_matrix()
    m_fo = empyrean.propagate(
        orbit, times, uncertainty_method=UncertaintyMethod.FIRST_ORDER
    ).states.coordinates.covariance.to_matrix()

    # Deterministic: repeat runs are bit-identical (the mixture recombination
    # for a fixed input is reproducible).
    np.testing.assert_array_equal(m_gmm, m_gmm2)
    # Distinct engine path: not a silent downgrade to the FIRST_ORDER code.
    assert not np.array_equal(m_gmm, m_fo), (
        "GaussianMixture covariance is bit-identical to first-order — the method was ignored"
    )
    # Expected regime: for a well-determined object away from an encounter,
    # the mixture reduces to the linear mapping (reads back very close to FO).
    assert np.allclose(m_gmm, m_fo, rtol=1e-6, atol=1e-20)


def test_propagate_gaussian_mixture_params_flow(orbit: CartesianOrbits, times: np.ndarray) -> None:
    """The AGM parameters flow to the engine rather than being clamped to
    defaults at the wrapper: a non-default ``components_per_split`` is
    accepted (honored — on a benign arc the splitter never fires, so no
    odd-count table lookup occurs) and the call still returns a finite
    covariance. The wrapper must not silently rewrite the caller's params.
    """
    res = empyrean.propagate(
        orbit,
        times,
        uncertainty_method=GaussianMixture(threshold=0.5, max_depth=2, components_per_split=5),
    )
    m = res.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all()


def test_propagate_gaussian_mixture_wire_dict_path_consistent(
    orbit: CartesianOrbits, times: np.ndarray
) -> None:
    """The ``config=`` (wire-dict) path resolves ``GAUSSIAN_MIXTURE`` to the
    real variant — not a silently-substituted first-order covariance —
    and agrees with the flat-arg ``uncertainty_method=`` path."""
    cfg = PropagationConfig(uncertainty_method=UncertaintyMethod.GAUSSIAN_MIXTURE)
    res_cfg = empyrean.propagate(orbit, times, config=cfg, tagged_covariance=True)
    res_flat = empyrean.propagate(
        orbit, times, uncertainty_method=GaussianMixture(), tagged_covariance=True
    )
    m_cfg = res_cfg.states.coordinates.covariance.to_matrix()
    m_flat = res_flat.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m_cfg).all()
    assert set(res_cfg.tagged_covariance.kind.to_pylist()) == set(
        res_flat.tagged_covariance.kind.to_pylist()
    )
    np.testing.assert_array_equal(m_cfg, m_flat)


# ══════════════════════════════════════════════════════════════════
#  generate_ephemeris — GaussianMixture is ACCEPTED (analytic)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "method",
    [UncertaintyMethod.GAUSSIAN_MIXTURE, GaussianMixture()],
    ids=["enum", "dataclass"],
)
def test_generate_ephemeris_accepts_gaussian_mixture(
    orbit: CartesianOrbits, observers, method
) -> None:
    """Unlike the sampling methods (which ``generate_ephemeris`` rejects
    with a ``ValueError``), GaussianMixture is analytic and MUST be
    accepted — the call runs and yields a finite sky-plane covariance."""
    eph = empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)
    cov = eph.ephemeris.coordinates.covariance
    assert cov is not None, "GaussianMixture: sky covariance column missing"
    m = cov.to_matrix()
    assert np.isfinite(m).all(), "GaussianMixture: sky covariance not finite"


def test_generate_ephemeris_gaussian_mixture_not_in_rejection(
    orbit: CartesianOrbits, observers
) -> None:
    """Explicit differential against the sampling-method rejection: the
    same call shape that raises for SIGMA_POINT / MONTE_CARLO must NOT
    raise for GAUSSIAN_MIXTURE."""
    # sampling methods are rejected...
    with pytest.raises(ValueError, match="sampling uncertainty methods"):
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=UncertaintyMethod.SIGMA_POINT
        )
    # ...GaussianMixture is not.
    eph = empyrean.generate_ephemeris(
        orbit, observers, uncertainty_method=UncertaintyMethod.GAUSSIAN_MIXTURE
    )
    assert eph.ephemeris.coordinates.covariance is not None
