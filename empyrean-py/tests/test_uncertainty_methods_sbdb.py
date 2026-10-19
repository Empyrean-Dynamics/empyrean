"""SBDB integration repro for the sampling uncertainty methods: query a
real object from JPL SBDB and propagate it over a two-epoch grid under
each uncertainty method.

These tests hit the network (SBDB) and are skipped when it is
unavailable, so they document/verify the end-to-end fix without gating
the offline unit suite in ``test_uncertainty_methods.py``.

The object comes back in the **cometary** representation (SBDB
convention), which used to surface a distinct, deeper engine limitation:
the sigma-point path silently skipped non-Cartesian input orbits and left
a first-order (linear) covariance in place. That is fixed engine-side —
sigma points are constructed in native element space — so the cometary
path is now asserted directly rather than xfailed. The Cartesian orbit derived from the same SBDB
record is kept as a second, independent check of the same property.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    Epochs,
    GaussianMixture,
    MonteCarlo,
    Origin,
    UncertaintyMethod,
    compute_impact_probabilities,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.ephemeris.result import OrbitOutcome
from empyrean.orbits.orbits import CartesianOrbits

_TARGET = "Apophis"


@pytest.fixture(scope="module")
def sbdb_orbit():
    """Query SBDB for a real NEO with covariance; skip if the network /
    SBDB is unavailable."""
    empyrean.initialize()
    try:
        orb = empyrean.query_sbdb([_TARGET])
    except Exception as e:  # noqa: BLE001 — any network/parse failure → skip
        pytest.skip(f"SBDB query for {_TARGET} unavailable: {type(e).__name__}: {e}")
    if orb.coordinates.covariance is None:
        pytest.skip(f"SBDB record for {_TARGET} carries no covariance")
    return orb


@pytest.fixture(scope="module")
def sbdb_grid(sbdb_orbit) -> np.ndarray:
    t0 = float(sbdb_orbit.coordinates.epoch.to_numpy(zero_copy_only=False)[0])
    return np.array([t0, t0 + 365.0])


@pytest.fixture(scope="module")
def sbdb_orbit_cartesian(sbdb_orbit, sbdb_grid) -> CartesianOrbits:
    """The same SBDB record as a Cartesian orbit (state + covariance at
    the epoch), obtained by propagating to its own epoch. Originally the
    workaround for the non-Cartesian sigma-point engine limitation;
    retained after the fix as a second representation exercising the
    same property."""
    t0 = sbdb_grid[0]
    r0 = empyrean.propagate(
        sbdb_orbit,
        Epochs.from_mjd(np.array([t0]), scale="tdb"),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
    )
    c = r0.states.coordinates
    cov0 = c.covariance.to_matrix()[0]

    def col(name):
        return float(getattr(c, name).to_numpy(zero_copy_only=False)[0])

    frame = c.frame if isinstance(c.frame, str) else c.frame.to_pylist()[0]
    origin = c.origin.to_pylist()
    return CartesianOrbits.from_kwargs(
        orbit_id=["apo"],
        object_id=["apo"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([t0]),
            x=[col("x")],
            y=[col("y")],
            z=[col("z")],
            vx=[col("vx")],
            vy=[col("vy")],
            vz=[col("vz")],
            frame=frame,
            origin=origin,
            covariance=CartesianCovariance.from_matrix(cov0[None, :, :]),
        ),
    )


def test_sbdb_propagate_sampling_methods_do_not_raise(sbdb_orbit, sbdb_grid) -> None:
    """The original bug: ``propagate`` eager-rejected SIGMA_POINT /
    MONTE_CARLO. After the fix they are accepted and run end-to-end on a
    real (cometary) SBDB orbit."""
    for method in (
        UncertaintyMethod.SIGMA_POINT,
        MonteCarlo(n_samples=64, seed=7),
    ):
        res = empyrean.propagate(
            sbdb_orbit, Epochs.from_mjd(sbdb_grid, scale="tdb"), uncertainty_method=method
        )
        assert len(res.states) == len(sbdb_grid)


def _assert_sigma_point_is_not_linear(res_fo, res_sp) -> None:
    """Pin that a ``sigma_point``-tagged covariance actually carries
    resampled values rather than the linear ones under a new label.

    Asserting the tag alone cannot distinguish a working sigma-point path
    from a mislabelled first-order one — the covariance has to be shown
    to differ. The correction is genuine but small on a well-determined
    orbit over a short arc, so this checks the covariance moved at all
    and stayed finite, without pinning a magnitude that would re-baseline
    on every engine change.
    """
    cov_fo = res_fo.states.coordinates.covariance.to_matrix()
    cov_sp = res_sp.states.coordinates.covariance.to_matrix()
    assert cov_fo.shape == cov_sp.shape
    for i in range(cov_fo.shape[0]):
        a, b = cov_fo[i], cov_sp[i]
        assert np.all(np.isfinite(b)), f"sigma-point covariance not finite at epoch {i}"
        assert np.all(np.diag(b) > 0.0), (
            f"sigma-point covariance has a non-positive variance at epoch {i}"
        )
        assert not np.array_equal(a, b), (
            f"sigma-point covariance is bit-identical to the first-order one at "
            f"epoch {i} — tagged 'sigma_point' but never resampled"
        )


def test_sbdb_cartesian_sigma_point_is_genuine(sbdb_orbit_cartesian, sbdb_grid) -> None:
    """On a Cartesian orbit derived from the SBDB record, SIGMA_POINT
    produces a genuine, provenance-tagged sample covariance distinct from
    the linear first-order one — the fix's core value."""
    res_fo = empyrean.propagate(
        sbdb_orbit_cartesian,
        Epochs.from_mjd(sbdb_grid, scale="tdb"),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
        tagged_covariance=True,
    )
    res_sp = empyrean.propagate(
        sbdb_orbit_cartesian,
        Epochs.from_mjd(sbdb_grid, scale="tdb"),
        uncertainty_method=UncertaintyMethod.SIGMA_POINT,
        tagged_covariance=True,
    )
    assert set(res_sp.tagged_covariance.kind.to_pylist()) == {"sigma_point"}
    assert set(res_fo.tagged_covariance.kind.to_pylist()) == {"linear"}
    _assert_sigma_point_is_not_linear(res_fo, res_sp)


def test_agm_fires_and_labels_on_a_synthetic_earth_encounter() -> None:
    """GaussianMixture (AGM) in ``compute_impact_probabilities`` fires and
    labels correctly on a SYNTHETIC Earth-encounter orbit — the firing case
    the heliocentric offline fixtures in ``test_impact`` cannot reach (no
    encounter → empty table), reached here with no network query and no
    bug-report object.

    The orbit is the validation suite's synthetic ``impactor`` (a
    constructed Earth-crossing Cartesian state from
    ``fixtures/parity_manifest.json``), its covariance inflated ×1e6 so the
    encounter's nonlinearity clears the AGM firing threshold. Measured
    against the pinned engine (villeneuve 1.26.0): the splitter does not
    fire below κ ≈ 1 (κ = 0.95 at ×1e4, mixture absent) and fires above it
    (κ ≈ 9.5 at ×1e6, an 3-component mixture, ``ip_agm`` ≈ 3e-3). The tag-5
    row must:

    1. be labelled ``"gaussian_mixture"`` — NOT silently relabelled
       ``"first_order"`` as it was before the ``method_from_tag`` readback
       fix (``empyrean/src/impact.rs``); the paired ``first_order`` row with
       a distinct label proves the tag is not collapsed in the round-trip;
       and
    2. carry a non-null ``agm_components`` count and a finite,
       mixture-corrected ``ip_agm`` (the splitter constructed a mixture).

    Offline: needs the local SPICE kernels (``EMPYREAN_DATA_DIR``), no SBDB.
    """
    empyrean.initialize()
    # Synthetic Earth-impactor from the validation parity manifest — a
    # constructed fixture, never a real or bug-report object.
    epoch = 54746.0
    state = [
        0.9703740077176525,
        0.24140202303297825,
        0.00010161047612020427,
        -0.008415158408739472,
        0.01715090747796052,
        -0.0008295684050048718,
    ]
    base_diag = np.array([1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16])
    # ×1e6 puts the encounter's nonlinearity at κ ≈ 9.5 — well above the
    # ~κ=1 firing threshold measured above, so the firing is robust to the
    # ~1e-10 platform spread in the SPICE-backed propagation.
    cov = np.diag(base_diag * 1e6)
    impactor = CartesianOrbits.from_kwargs(
        orbit_id=["impactor"],
        object_id=["impactor"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([epoch]),
            x=[state[0]],
            y=[state[1]],
            z=[state[2]],
            vx=[state[3]],
            vy=[state[4]],
            vz=[state[5]],
            frame="ecliptic_j2000",
            origin=[str(Origin.SUN)],
            covariance=CartesianCovariance.from_matrix(cov[None, :, :]),
        ),
    )
    # The encounter is within a day of the epoch; end just past it.
    ips = compute_impact_probabilities(
        impactor,
        end_epoch=Epochs.from_mjd([54748.0], scale="tdb"),
        methods=[UncertaintyMethod.FIRST_ORDER, GaussianMixture()],
        body_filter=[Origin.EARTH],
    )
    df = ips.to_dataframe()
    labels = set(df["method"].tolist())
    # Distinct, correct labels — the tag is not collapsed in the round-trip.
    assert "gaussian_mixture" in labels, f"tag-5 row mislabelled; got {labels}"
    assert "first_order" in labels, f"first_order row missing; got {labels}"

    gm = df[df["method"] == "gaussian_mixture"].iloc[0]
    # The splitter fired: a mixture was constructed (non-null component
    # count) and the mixture-corrected impact probability is finite.
    assert np.isfinite(gm["agm_components"]) and gm["agm_components"] > 0, (
        f"AGM did not construct a mixture; agm_components={gm['agm_components']}"
    )
    assert np.isfinite(gm["ip_agm"]), "AGM fired but ip_agm is not finite"
    assert np.isfinite(gm["ip_linear"])


def test_sbdb_cometary_sigma_point_is_genuine(sbdb_orbit, sbdb_grid) -> None:
    """The SBDB (cometary) sigma-point path, which is the workflow a user
    actually runs — ``query_sbdb`` returns cometary orbits.

    This was a strict-xfail guard for the case where the engine silently
    skipped non-Cartesian input and left a first-order covariance in
    place under a SIGMA_POINT request. That is fixed engine-side (sigma
    points are now constructed in native element space), so the guard is
    retired and the path is asserted directly.

    The tag alone is not the assertion: a covariance mislabelled
    ``sigma_point`` while still holding the linear values is exactly the
    failure r2dq was, so genuineness is pinned by comparing against the
    FIRST_ORDER covariance of the same propagation.
    """
    res_fo = empyrean.propagate(
        sbdb_orbit,
        Epochs.from_mjd(sbdb_grid, scale="tdb"),
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
        tagged_covariance=True,
    )
    res_sp = empyrean.propagate(
        sbdb_orbit,
        Epochs.from_mjd(sbdb_grid, scale="tdb"),
        uncertainty_method=UncertaintyMethod.SIGMA_POINT,
        tagged_covariance=True,
    )
    assert set(res_sp.tagged_covariance.kind.to_pylist()) == {"sigma_point"}
    assert set(res_fo.tagged_covariance.kind.to_pylist()) == {"linear"}
    _assert_sigma_point_is_not_linear(res_fo, res_sp)


def test_sbdb_generate_ephemeris_sampling_is_the_engines_call(
    sbdb_orbit, sbdb_orbit_cartesian, sbdb_grid
) -> None:
    """generate_ephemeris adds no wrapper gate on the sampling methods: the
    engine decides, and on a real SBDB record both of its answers are pinned
    here, as the engine delivers them.

    The record as queried carries a Cometary primary coordinate. The one
    cause this test pins: villeneuve #264 samples a non-Cartesian primary
    (Cometary, Keplerian, Spherical) in its OWN representation, so the
    sampled sky path now DELIVERS for it — ephemeris rows and a per-orbit
    ``DELIVERED`` outcome — instead of returning an empty ephemeris (the old
    uncovered-Cometary residue) or refusing by name. Apophis's SBDB
    element-space covariance is wide enough that the sampled sky moment over
    a year is not positive-definite, so the per-row sky covariance is
    *withheld* and the reason rides ``EphemerisResult.warnings`` while the
    ephemeris rows themselves are still delivered — delivery with a named,
    withheld covariance, never a silent empty table. A future engine that
    returns to an empty ephemeris or a by-name refusal for a Cometary
    primary is caught here. The SAME record as a Cartesian orbit, whose
    propagated-to-epoch covariance is well conditioned, delivers a finite
    sky covariance that is not the first-order projection.
    """
    observers = empyrean.get_observer_states(["500"], Epochs.from_mjd(sbdb_grid, scale="tdb"))

    # Cometary primary under a sampling method: the engine samples it in its
    # own representation and DELIVERS. Rows + a `delivered` outcome come
    # through; the wide element-space cloud's sky moment is not
    # positive-definite, so the covariance is withheld with a named warning.
    for method in (UncertaintyMethod.SIGMA_POINT, MonteCarlo(n_samples=64, seed=7)):
        eph = empyrean.generate_ephemeris(sbdb_orbit, observers, uncertainty_method=method)
        assert len(eph.ephemeris) == len(sbdb_grid), (
            f"{method}: a non-Cartesian primary now DELIVERS sampled ephemeris rows "
            "(villeneuve #264 samples it in its own representation), not an empty table"
        )
        assert eph.all_delivered, (
            f"{method}: the per-orbit outcome must be DELIVERED, not FAILED: "
            f"{eph.status.outcome.to_pylist()}"
        )
        assert eph.status.outcome[0].as_py() == OrbitOutcome.DELIVERED.value
        assert any("CovarianceWithheld" in w for w in eph.warnings), (
            f"{method}: a withheld sampled covariance must be named in warnings; "
            f"got {eph.warnings!r}"
        )
        assert any("not positive-definite" in w for w in eph.warnings), (
            f"{method}: the withheld-covariance warning must name the reason "
            f"(sky moment not positive-definite); got {eph.warnings!r}"
        )

    # Cartesian primary: delivered — a finite sampled sky covariance distinct
    # from the first-order projection.
    def sky(method):
        cov = empyrean.generate_ephemeris(
            sbdb_orbit_cartesian, observers, uncertainty_method=method
        ).ephemeris.coordinates.covariance
        assert cov is not None, f"{method}: sky covariance column missing"
        return cov.to_matrix()

    fo = sky(UncertaintyMethod.FIRST_ORDER)
    for method in (UncertaintyMethod.SIGMA_POINT, MonteCarlo(n_samples=64, seed=7)):
        m = sky(method)
        assert m.shape[0] > 0 and np.isfinite(m).all(), (
            f"{method}: sampled sky covariance not finite"
        )
        assert not np.array_equal(m, fo), f"{method}: sky covariance is the first-order one"
