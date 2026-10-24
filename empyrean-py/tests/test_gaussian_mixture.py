"""Behavioral contract for the GaussianMixture (adaptive Gaussian
mixture, AGM) uncertainty method in :func:`empyrean.propagate` and
:func:`empyrean.generate_ephemeris`.

GaussianMixture is exposed as a top-level uncertainty method (tag 5)
reusing the AGM parameter slots the C ABI already carried for ``Auto``;
the flat ``tag`` disambiguates a standalone mixture from Auto's internal
splitter.

It is HONORED on ``propagate``, where the mixture is born and retained.
It is also DELIVERED on ``generate_ephemeris``: villeneuve 1.26.0's
mixture generation seams fly the mixture through the optical generation
pipeline (light time and all) and deliver its per-observation sky moment
as the row covariance — ephemeris rows and a per-orbit ``DELIVERED``
outcome, never a silent first-order sky covariance wearing the mixture's
label. The ephemeris seam exposes the delivery, not the component
breakdown: the per-component table lives on ``propagate``'s ``mixtures``;
the ephemeris row carries the moment-matched sky covariance alone.

So GaussianMixture and the sampling methods alike now deliver on the
ephemeris path. Away from a close approach the mixture reduces to the
linear ``Φ·Σ·Φᵀ`` mapping, so for a well-determined object the delivered
sky covariance reads back very close to ``FIRST_ORDER`` while remaining a
distinct engine path (not a bit-identical downgrade) — a distinction
these tests pin, because collapsing the two would hide a silent downgrade.

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
from empyrean.ephemeris.result import OrbitOutcome
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import (
    _DATACLASS_TO_INT,
    _INT_TO_UNCERTAINTY_METHOD,
    _UNCERTAINTY_METHOD_TO_INT,
)

_EPOCH_MJD_TDB = 61000.0


def _mixture_tables_equal(a, b) -> bool:
    """Faithful (bit-identical) round-trip equality for a mixture table.

    The packed-joint covariance carries ``cov_min_eig_corr == NaN`` for a
    positive-definite component, and pyarrow's ``Table.equals`` treats
    ``NaN != NaN``, so it cannot express the bit-identity of a NaN-bearing
    column. This compares the whole table (nested joint sub-table included)
    NaN-aware — a NaN matches a NaN — so it is strictly the faithful
    round-trip check, not a looser one.
    """
    import math

    def eq(x: object, y: object) -> bool:
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(eq(u, v) for u, v in zip(x, y, strict=True))
        if isinstance(x, float) and isinstance(y, float):
            return x == y or (math.isnan(x) and math.isnan(y))
        return bool(x == y)

    return a.schema.equals(b.schema) and eq(a.to_pylist(), b.to_pylist())


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
            epochs=Epochs.from_kwargs(mjd=np.array([_EPOCH_MJD_TDB]), scale="tdb"),
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


def test_component_status_sky_linearization_refused_decodes_by_name() -> None:
    """The ``SkyLinearizationRefused`` component status (wire code 4, new in
    villeneuve 1.26.0) decodes by name at the Python side of the wire, exactly
    as the other four do. The extension emits the status as a string; dropping
    the enum member would make this ``ComponentStatus("SkyLinearizationRefused")``
    raise ``ValueError`` instead of resolving to the member.
    """
    from empyrean.propagation.mixtures import ComponentStatus

    assert ComponentStatus("SkyLinearizationRefused") is ComponentStatus.SKY_LINEARIZATION_REFUSED
    # The side value ``rho`` carries the on-sky angular extent (radians) for
    # this status — a distinct quantity from CurvatureRefused's ρ — so the two
    # refusal members stay distinct, never collapsed onto one tag.
    assert ComponentStatus.SKY_LINEARIZATION_REFUSED != ComponentStatus.CURVATURE_REFUSED
    assert ComponentStatus.SKY_LINEARIZATION_REFUSED.value == "SkyLinearizationRefused"


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
    tagged-covariance kind is ``mixture`` at close-approach windows, the
    second-order ``second_order`` fold where the 0.11.0 engine carries it,
    and the linear ``linear`` mapping elsewhere."""
    res = empyrean.propagate(orbit, times, uncertainty_method=method, tagged_covariance=True)
    m = res.states.coordinates.covariance.to_matrix()
    assert np.isfinite(m).all(), "GaussianMixture state covariance not finite"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all()
    kinds = set(res.tagged_covariance.kind.to_pylist())
    assert kinds <= {"linear", "second_order", "mixture"}, (
        f"unexpected tagged-covariance kind(s): {kinds}"
    )


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
#  generate_ephemeris — GaussianMixture DELIVERS the mixture sky moment
# ══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "method",
    [UncertaintyMethod.GAUSSIAN_MIXTURE, GaussianMixture()],
    ids=["enum", "dataclass"],
)
def test_generate_ephemeris_delivers_gaussian_mixture(
    orbit: CartesianOrbits, observers, method
) -> None:
    """The optical seam now DELIVERS a GaussianMixture ephemeris.

    One cause: villeneuve 1.26.0's mixture generation seams fly the
    mixture through the optical generation pipeline and deliver its
    per-observation sky moment as the row covariance — so the seam no
    longer refuses the method by name. Delivery is pinned here as the
    engine reports it: one row per observer, a per-orbit ``DELIVERED``
    outcome, a finite sky covariance on every row, and no warning (this
    well-determined prior disperses nowhere near the sky-chart bound).
    Both the enum and the dataclass request forms must deliver
    identically — a knob that delivers one spelling and refuses the other
    would be worse than either.

    What reaches Python at this seam is the delivery, not the component
    breakdown: the per-component table is a ``propagate`` surface
    (``PropagationResult.mixtures``); the ephemeris row carries only the
    moment-matched sky covariance, so this test pins rows + outcome +
    finite covariance, not a component count.
    """
    eph = empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)

    assert len(eph.ephemeris) == len(observers.coordinates.epochs.mjd_tdb()), (
        "GaussianMixture now delivers one ephemeris row per observer, not a refusal"
    )
    assert eph.all_delivered, (
        f"the per-orbit outcome must be DELIVERED, not FAILED: {eph.status.outcome.to_pylist()}"
    )
    assert eph.status.outcome[0].as_py() == OrbitOutcome.DELIVERED.value
    cov = eph.ephemeris.coordinates.covariance
    assert cov is not None, "the delivered mixture ephemeris must carry a sky covariance"
    m = cov.to_matrix()
    assert np.isfinite(m).all(), f"the delivered mixture sky covariance must be finite: {m}"
    assert (np.diagonal(m, axis1=1, axis2=2) >= 0).all(), "negative variance on the diagonal"
    assert eph.warnings == [], (
        f"a well-determined prior delivers with no warning, got {eph.warnings!r}"
    )


def test_generate_ephemeris_delivers_both_the_mixture_and_sampling(
    orbit: CartesianOrbits, observers
) -> None:
    """The wrapper adds no gate of its own on the ephemeris path: the
    engine decides, and it now delivers both sampled and mixture sky
    moments. One cause: with villeneuve 1.26.0 the mixture generation
    seams exist, so ``GAUSSIAN_MIXTURE`` joins ``SIGMA_POINT`` as a
    delivered method rather than a by-name refusal.

    The mixture is a distinct engine path, not a silent downgrade to the
    first-order code: its delivered sky covariance is not bit-identical to
    ``FIRST_ORDER``, yet — for this well-determined object away from an
    encounter — reduces to the linear ``Φ·Σ·Φᵀ`` mapping, so it reads back
    very close to it. A bit-identical covariance here would mean the
    method was ignored.
    """
    for method in (UncertaintyMethod.SIGMA_POINT, UncertaintyMethod.GAUSSIAN_MIXTURE):
        eph = empyrean.generate_ephemeris(orbit, observers, uncertainty_method=method)
        assert eph.all_delivered, (
            f"{method}: outcome must be DELIVERED: {eph.status.outcome.to_pylist()}"
        )
        cov = eph.ephemeris.coordinates.covariance
        assert cov is not None and np.isfinite(cov.to_matrix()).all(), (
            f"{method}: must deliver a finite sky covariance"
        )

    m_gmm = empyrean.generate_ephemeris(
        orbit, observers, uncertainty_method=UncertaintyMethod.GAUSSIAN_MIXTURE
    ).ephemeris.coordinates.covariance.to_matrix()
    m_fo = empyrean.generate_ephemeris(
        orbit, observers, uncertainty_method=UncertaintyMethod.FIRST_ORDER
    ).ephemeris.coordinates.covariance.to_matrix()
    assert not np.array_equal(m_gmm, m_fo), (
        "the mixture sky covariance is bit-identical to first-order — the method was ignored"
    )
    assert np.allclose(m_gmm, m_fo, rtol=1e-6, atol=1e-20), (
        "away from an encounter the mixture must reduce to the linear mapping"
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
            epochs=Epochs.from_kwargs(mjd=np.array([_APOPHIS_EPOCH_MJD_TDB]), scale="tdb"),
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
        "ca_epochs",
        "component_index",
        "weight",
        "mean_x",
        "joint",
        "status",
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

    # The component covariance is the one packed joint; its flat 6×6 is the
    # derived state block (no separate covariance column any more).
    covs = np.asarray(
        [table.joint.joint(i).state_block().ravel() for i in range(n)], dtype=np.float64
    )
    assert covs.shape == (n, 36)
    assert np.isfinite(covs).all(), "component covariances must be finite"


def test_mixture_weights_never_exceed_one(encounter_result) -> None:
    """Retained weights may sum to LESS than 1 — a sub-Gaussian that
    missed the close approach contributes nothing and the deficit is not
    recorded anywhere — but they must never sum to more."""
    table = encounter_result.mixtures
    epochs = table.ca_epochs.mjd_tdb()
    weights = table.weight.to_numpy(zero_copy_only=False)
    for t in np.unique(epochs):
        total = float(weights[epochs == t].sum())
        assert total <= 1.0 + 1e-9, f"weights at CA {t} sum to {total}"


def test_mixture_chains_accessor_groups_by_ca_epoch(encounter_result) -> None:
    """``to_chains`` regroups the flat table into one list per retained CA
    epoch, with the matrices re-materialized as contiguous (6, 6)."""
    chains = encounter_result.mixture_chains(0)
    assert len(chains) == len(encounter_result.mixtures.ca_epoch_values(0))
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
    assert _mixture_tables_equal(back.table, table.table), (
        "the mixture table must round-trip bit-identically"
    )


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
    assert _mixture_tables_equal(back.mixtures.table, encounter_result.mixtures.table)


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
        build_mixture_chains({"states": []}, "tdb")


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
        # The component covariance rides the one packed joint under
        # `mixture_covariance`; nothing split means no joints, so the key is
        # absent (as the extension's writer leaves it), and status / rho are
        # empty.
        "mixture_status": [],
        "mixture_rho": [],
        "mixture_frame": np.zeros(0, dtype=np.int64),
        "mixture_origin": np.zeros(0, dtype=np.int64),
    }
    assert len(build_mixture_chains({"mixtures": empty}, "tdb")) == 0
