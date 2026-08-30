"""Behavioral contract for the GaussianMixture (adaptive Gaussian
mixture, AGM) uncertainty method in :func:`empyrean.propagate` and
:func:`empyrean.generate_ephemeris`.

GaussianMixture is exposed as a top-level uncertainty method (tag 5)
reusing the AGM parameter slots the C ABI already carried for ``Auto``;
the flat ``tag`` disambiguates a standalone mixture from Auto's internal
splitter.

It is HONORED on ``propagate``, where the mixture is born and retained.
It is REFUSED on ``generate_ephemeris``: since villeneuve 1.25.0 (the
uniform uncertainty-method gate, bd empyrean-848gm) the optical seams
refuse any method they have no delivery for, by name, rather than
silently composing a first-order sky covariance under the requested
method's label. Mixture components are born during propagation and
retained on-grid only, so no per-observation sky projection exists yet —
that delivery is bd empyrean-oihdg, and the on-grid moment-matched
mixture covariance is reachable through ``covariance_series`` meanwhile.

So both GaussianMixture and the sampling methods are refused on the
ephemeris path, but by different mechanisms and for different reasons —
a distinction these tests pin, because collapsing them would hide which
layer is doing the refusing.

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
from empyrean import (
    Epochs,
    GaussianMixture,
    MixtureChains,
    PropagationConfig,
    UncertaintyMethod,
)
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
def times() -> Epochs:
    return Epochs.from_mjd(
        np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 365.0, _EPOCH_MJD_TDB + 730.0]),
        scale="tdb",
    )


@pytest.fixture(scope="module")
def observers():
    empyrean.initialize()
    return empyrean.get_observer_states(
        ["500"],
        Epochs.from_mjd(np.array([_EPOCH_MJD_TDB, _EPOCH_MJD_TDB + 30.0]), scale="tdb"),
    )


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
    orbit: CartesianOrbits, times: Epochs, method
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


def test_propagate_gaussian_mixture_reaches_engine(orbit: CartesianOrbits, times: Epochs) -> None:
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


def test_propagate_gaussian_mixture_params_flow(orbit: CartesianOrbits, times: Epochs) -> None:
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
    orbit: CartesianOrbits, times: Epochs
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
#  generate_ephemeris — GaussianMixture is REFUSED BY NAME (no delivery)
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "method",
    [UncertaintyMethod.GAUSSIAN_MIXTURE, GaussianMixture()],
    ids=["enum", "dataclass"],
)
def test_generate_ephemeris_refuses_gaussian_mixture_by_name(
    orbit: CartesianOrbits, observers, method
) -> None:
    """The optical seam has no mixture sky delivery, so it REFUSES —
    naming both the method and the seam that refused it — rather than
    composing a first-order sky covariance and labelling it
    GaussianMixture (bd empyrean-848gm; the delivery itself is
    bd empyrean-oihdg).

    A silent downgrade here would be undetectable downstream: the caller
    asked for a mixture, would receive a linear projection, and nothing
    in the result would say so. Both the enum and the dataclass request
    forms must refuse identically — a knob that refuses one spelling and
    honors the other is worse than either.

    Mirrors villeneuve ``tests/test_ephemeris_marginal_covariance.rs::
    sampled_methods_are_refused_by_name_at_the_optical_entries``, which
    pins the same refusal at the engine seam.
    """
    with pytest.raises(RuntimeError) as excinfo:
        empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)

    msg = str(excinfo.value)
    assert "GaussianMixture" in msg, f"the refusal must name the method, got {msg!r}"
    assert "generate_optical_with_system" in msg, (
        f"the refusal must name the seam that refused it, got {msg!r}"
    )
    assert "silently downgraded" in msg, (
        "the refusal must say what it declined to do instead — that is the "
        f"whole reason it is an error and not a quiet first-order row: {msg!r}"
    )


def test_generate_ephemeris_refuses_the_mixture_by_name_while_sampling_is_delivered(
    orbit: CartesianOrbits, observers
) -> None:
    """The wrapper adds no gate of its own on the ephemeris path: the
    engine decides. ``SIGMA_POINT`` is delivered (the sampled sky moment,
    villeneuve 1.25.0, bd empyrean-848gm.1 — pinned in
    ``test_uncertainty_methods.py``), while ``GAUSSIAN_MIXTURE`` is
    refused by the engine as a missing delivery (``RuntimeError`` naming
    the method), because a mixture IS analytic and the gap is a
    per-observation sky projection that has not been built yet
    (bd empyrean-oihdg), not a category error.

    Collapsing the two would tell a caller to change their method when
    the honest answer is "this one is coming".
    """
    eph = empyrean.generate_ephemeris(
        orbit, observers, uncertainty_method=UncertaintyMethod.SIGMA_POINT
    )
    assert eph.ephemeris.coordinates.covariance is not None

    with pytest.raises(RuntimeError, match="GaussianMixture"):
        empyrean.generate_ephemeris(
            orbit, observers, uncertainty_method=UncertaintyMethod.GAUSSIAN_MIXTURE
        )


# ══════════════════════════════════════════════════════════════════
#  Mixture component readback — the surface the wrapper used to drop
# ══════════════════════════════════════════════════════════════════
#
# The safe Rust wrapper's marshal copied states / object_ids / events off
# the FFI result and never touched `mixtures`, so every retained
# component died one layer above the C ABI and Python had no mixture
# surface at all. These tests pin it from this end.


APOPHIS_STATE = [
    -7.85264914906904643e-02,
    -8.19748051902064567e-01,
    4.18939515323390882e-02,
    1.98751024968884596e-02,
    1.32208844536140196e-03,
    3.99496044422352188e-04,
]
_APOPHIS_EPOCH_MJD_TDB = 61000.0


def _apophis_loose() -> CartesianOrbits:
    """Apophis with a covariance loose enough that the mapping through the
    2029 Earth encounter is genuinely nonlinear — the regime AGM exists
    for, and the only regime that produces retained components."""
    cov = np.zeros((1, 6, 6))
    for k in range(3):
        cov[0, k, k] = 1e-10
        cov[0, k + 3, k + 3] = 1e-16
    s = APOPHIS_STATE
    return CartesianOrbits.from_kwargs(
        orbit_id=["apophis-mixture"],
        object_id=["99942"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_APOPHIS_EPOCH_MJD_TDB]),
            x=[s[0]],
            y=[s[1]],
            z=[s[2]],
            vx=[s[3]],
            vy=[s[4]],
            vz=[s[5]],
            frame="ecliptic_j2000",
            origin=["Sun"],
            covariance=CartesianCovariance.from_matrix(cov),
        ),
    )


def _encounter_epochs() -> Epochs:
    """A grid straddling the 2029-04-13 Earth close approach."""
    return Epochs.from_mjd(
        np.arange(_APOPHIS_EPOCH_MJD_TDB, _APOPHIS_EPOCH_MJD_TDB + 1401.0, 100.0),
        scale="tdb",
    )


@pytest.fixture(scope="module")
def encounter_result():
    empyrean.initialize()
    return empyrean.propagate(
        _apophis_loose(),
        _encounter_epochs(),
        uncertainty_method=GaussianMixture(),
    )


def test_mixture_components_reach_python(encounter_result) -> None:
    """The components arrive, every emitted column has the same length,
    and the values are honest: finite positive weights, finite means and
    covariances, decoded basis names.

    This is the test that would have caught the wrapper drop — under it,
    the table is empty and the first assertion fails.
    """
    table = encounter_result.mixtures
    assert len(table) > 0, (
        "the 2029 Apophis encounter with a loose covariance must fire the splitter; "
        "an empty table here means the components were dropped between the C ABI "
        "and Python, or the fixture stopped being nonlinear"
    )

    n = len(table)
    for name in (
        "orbit_id",
        "orbit_index",
        "ca_epoch_mjd_tdb",
        "component_index",
        "weight",
        "mean_x",
        "covariance",
        "origin",
        "frame",
    ):
        assert len(table.column(name)) == n, f"column {name} is not length {n}"

    weights = table.weight.to_numpy(zero_copy_only=False)
    assert np.isfinite(weights).all(), "mixture weights must be finite"
    assert (weights > 0.0).all(), "mixture weights must be positive"

    assert set(table.orbit_id.to_pylist()) == {"apophis-mixture"}
    assert set(table.orbit_index.to_numpy(zero_copy_only=False).tolist()) == {0}
    assert all(o for o in table.origin.to_pylist()), "origins must decode to names"
    assert all(f for f in table.frame.to_pylist()), "frames must decode to names"

    covs = np.asarray(table.covariance.to_pylist(), dtype=np.float64)
    assert covs.shape == (n, 36)
    assert np.isfinite(covs).all(), "component covariances must be finite"


def test_mixture_weights_never_exceed_one(encounter_result) -> None:
    """Retained weights may sum to LESS than 1 — a sub-Gaussian that
    missed the close approach contributes nothing and the deficit is not
    recorded anywhere — but they must never sum to more."""
    table = encounter_result.mixtures
    epochs = table.ca_epoch_mjd_tdb.to_numpy(zero_copy_only=False)
    weights = table.weight.to_numpy(zero_copy_only=False)
    for t in np.unique(epochs):
        total = float(weights[epochs == t].sum())
        assert total <= 1.0 + 1e-9, f"weights at CA {t} sum to {total}"


def test_mixture_chains_accessor_groups_by_ca_epoch(encounter_result) -> None:
    """``to_chains`` regroups the flat table into one list per retained CA
    epoch, with the matrices re-materialized as contiguous (6, 6)."""
    chains = encounter_result.mixture_chains(0)
    assert len(chains) == len(encounter_result.mixtures.ca_epochs(0))
    assert any(len(group) > 0 for group in chains)
    flat = [c for group in chains for c in group]
    assert len(flat) == len(encounter_result.mixtures)
    for c in flat:
        assert c.mean.shape == (6,)
        assert c.covariance.shape == (6, 6)
        assert np.isfinite(c.covariance).all()
        assert c.frame and c.origin


def test_mixture_chains_round_trip_through_a_directory(encounter_result, tmp_path) -> None:
    """``to_dir`` / ``from_dir`` round-trip the table bit-identically."""
    table = encounter_result.mixtures
    table.to_dir(str(tmp_path))
    back = MixtureChains.from_dir(str(tmp_path))
    assert back.table.equals(table.table), "the mixture table must round-trip bit-identically"


def test_first_order_yields_an_empty_mixture_table() -> None:
    """A FIRST_ORDER propagation retains nothing, and "nothing" is ZERO
    ROWS — never zero-filled placeholder rows, which would read as
    one-component mixtures of weight 0."""
    empyrean.initialize()
    result = empyrean.propagate(
        _apophis_loose(),
        _encounter_epochs(),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
    )
    assert len(result.mixtures) == 0
    assert result.mixture_chains(0) == []


def test_propagation_result_dir_round_trip_carries_mixtures(encounter_result, tmp_path) -> None:
    """The whole result persists and reloads with its mixtures intact."""
    encounter_result.to_dir(str(tmp_path))
    back = empyrean.PropagationResult.from_dir(str(tmp_path))
    assert len(back.mixtures) == len(encounter_result.mixtures)
    assert back.mixtures.table.equals(encounter_result.mixtures.table)


def test_a_missing_mixtures_key_raises_rather_than_reading_as_nothing_split() -> None:
    """The extension sets ``"mixtures"`` unconditionally, so a missing key
    can only mean a compiled extension older than this Python package.

    Returning an empty table there spelled that skew exactly like a
    genuine "the splitter never fired" — a GAUSSIAN_MIXTURE run that did
    split would report no components, with no warning, and ``to_dir``
    would persist the wrong claim.
    """
    from empyrean.propagation.mixtures import build_mixture_chains

    with pytest.raises(RuntimeError, match="older than this Python package"):
        build_mixture_chains({"states": []})


def test_a_present_but_empty_mixtures_key_is_nothing_split() -> None:
    """Present-and-empty is the genuine "nothing split" shape and must
    stay a quiet empty table — the two cases are distinguished, not
    merged."""
    from empyrean.propagation.mixtures import build_mixture_chains

    empty = {
        "mixture_orbit_index": np.zeros(0, dtype=np.uint32),
        "mixture_orbit_id": [],
        "mixture_ca_epoch_mjd_tdb": np.zeros(0, dtype=np.float64),
        "mixture_component_index": np.zeros(0, dtype=np.uint32),
        "mixture_weight": np.zeros(0, dtype=np.float64),
        "mixture_mean": np.zeros((0, 6), dtype=np.float64),
        "mixture_covariance": np.zeros((0, 6, 6), dtype=np.float64),
        "mixture_frame": np.zeros(0, dtype=np.int64),
        "mixture_origin": np.zeros(0, dtype=np.int64),
    }
    assert len(build_mixture_chains({"mixtures": empty})) == 0
