"""SBDB integration repro for the sampling uncertainty methods
(bd empyrean-02j4). Mirrors the self-contained reproduction attached to
the issue: query a real object from JPL SBDB and propagate it over a
two-epoch grid under each uncertainty method.

These tests hit the network (SBDB) and are skipped when it is
unavailable, so they document/verify the end-to-end fix without gating
the offline unit suite in ``test_uncertainty_methods.py``.

The object comes back in the **cometary** representation (SBDB
convention), which surfaces a distinct, deeper engine limitation
(bd empyrean-r2dq): villeneuve's sigma-point path silently skips
non-Cartesian input orbits and leaves a first-order (linear) covariance
in place. The marshaling fix in this task is correct regardless — it
threads SIGMA_POINT / MONTE_CARLO through and stops the ephemeris
silent-ignore — and the genuine sigma-point covariance is verified here
on a Cartesian orbit derived from the same SBDB record.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    GaussianMixture,
    MonteCarlo,
    Origin,
    UncertaintyMethod,
    compute_impact_probabilities,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
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
    the epoch), obtained by propagating to its own epoch. Sidesteps the
    non-Cartesian sigma-point engine limitation (empyrean-r2dq)."""
    t0 = sbdb_grid[0]
    r0 = empyrean.propagate(
        sbdb_orbit, np.array([t0]), uncertainty_method=UncertaintyMethod.FIRST_ORDER
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
        res = empyrean.propagate(sbdb_orbit, sbdb_grid, uncertainty_method=method)
        assert len(res.states) == len(sbdb_grid)


def test_sbdb_cartesian_sigma_point_is_genuine(sbdb_orbit_cartesian, sbdb_grid) -> None:
    """On a Cartesian orbit derived from the SBDB record, SIGMA_POINT
    produces a genuine, provenance-tagged sample covariance distinct from
    the linear first-order one — the fix's core value."""
    res_fo = empyrean.propagate(
        sbdb_orbit_cartesian,
        sbdb_grid,
        uncertainty_method=UncertaintyMethod.FIRST_ORDER,
        tagged_covariance=True,
    )
    res_sp = empyrean.propagate(
        sbdb_orbit_cartesian,
        sbdb_grid,
        uncertainty_method=UncertaintyMethod.SIGMA_POINT,
        tagged_covariance=True,
    )
    assert set(res_sp.tagged_covariance.kind.to_pylist()) == {"sigma_point"}
    assert set(res_fo.tagged_covariance.kind.to_pylist()) == {"linear"}


def test_agm_impact_probability_fires_and_labels_correctly() -> None:
    """GaussianMixture (AGM) in ``compute_impact_probabilities`` on REAL
    data — the firing case the offline acceptance test in ``test_impact``
    cannot reach (its heliocentric fixture has no encounter → empty
    table). A short-arc (covariance-inflated) 2024 YR4 through its
    2032-12-22 Earth encounter produces a tag-5 impact row that:

    1. is labelled ``"gaussian_mixture"`` — NOT silently relabelled
       ``"first_order"`` as it was before the ``method_from_tag`` readback
       fix (``empyrean/src/impact.rs``); this is the exact assertion that
       would have caught the pre-fix bug; and
    2. carries a finite, mixture-corrected ``ip_agm`` (the AGM splitter
       fires above its nonlinearity threshold at this inflation).

    The paired ``first_order`` row with a distinct label proves the tag is
    not collapsed anywhere in the round-trip. Network-gated (SBDB); slow
    (AGM splitting through a multi-year propagation).
    """
    empyrean.initialize()
    try:
        orb = empyrean.query_sbdb(["2024 YR4"])
    except Exception as e:  # noqa: BLE001 — any network/parse failure → skip
        pytest.skip(f"SBDB query for 2024 YR4 unavailable: {type(e).__name__}: {e}")
    if orb.coordinates.covariance is None:
        pytest.skip("SBDB record for 2024 YR4 carries no covariance")

    t0 = float(orb.coordinates.epoch.to_numpy(zero_copy_only=False)[0])
    r0 = empyrean.propagate(orb, np.array([t0]), uncertainty_method=UncertaintyMethod.FIRST_ORDER)
    c = r0.states.coordinates
    cov0 = c.covariance.to_matrix()[0]

    def col(name):
        return float(getattr(c, name).to_numpy(zero_copy_only=False)[0])

    frame = c.frame if isinstance(c.frame, str) else c.frame.to_pylist()[0]
    # Inflate the covariance ×1e6 (σ×1000) to simulate an early short-arc
    # solution — the large-covariance regime where the AGM splitter fires.
    yr4 = CartesianOrbits.from_kwargs(
        orbit_id=["yr4"],
        object_id=["yr4"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([t0]),
            x=[col("x")],
            y=[col("y")],
            z=[col("z")],
            vx=[col("vx")],
            vy=[col("vy")],
            vz=[col("vz")],
            frame=frame,
            origin=c.origin.to_pylist(),
            covariance=CartesianCovariance.from_matrix((cov0 * 1e6)[None, :, :]),
        ),
    )
    # 2032-12-22 encounter is ≈ MJD 63588.5; end a few days past it.
    ips = compute_impact_probabilities(
        yr4,
        end_epoch=63594.0,
        methods=[UncertaintyMethod.FIRST_ORDER, GaussianMixture()],
        body_filter=[Origin.EARTH],
    )
    df = ips.to_dataframe()
    labels = set(df["method"].tolist())
    # Distinct, correct labels — the tag is not collapsed in the round-trip.
    assert "gaussian_mixture" in labels, f"tag-5 row mislabelled; got {labels}"
    assert "first_order" in labels, f"first_order row missing; got {labels}"

    gm = df[df["method"] == "gaussian_mixture"].iloc[0]
    # The AGM fired: the mixture-corrected IP is finite (a genuine payoff,
    # not a NaN no-op), alongside the always-present linear IP.
    assert np.isfinite(gm["ip_agm"]), "AGM did not fire (ip_agm NaN) at cov×1e6"
    assert np.isfinite(gm["ip_linear"])


@pytest.mark.xfail(
    reason="empyrean-r2dq: villeneuve sigma-point silently skips non-Cartesian "
    "(cometary) input orbits and leaves a linear covariance in place",
    strict=True,
)
def test_sbdb_cometary_sigma_point_is_genuine(sbdb_orbit, sbdb_grid) -> None:
    """Forward-looking guard for the SBDB (cometary) sigma-point path.

    This is the workflow a user actually runs (``query_sbdb`` returns
    cometary orbits). It currently **xfails**: the engine leaves a linear
    covariance in place. When empyrean-r2dq is fixed engine-side, this
    flips to xpass and flags that the limitation — and the workaround in
    :func:`test_sbdb_cartesian_sigma_point_is_genuine` — can be retired.
    """
    res_sp = empyrean.propagate(
        sbdb_orbit,
        sbdb_grid,
        uncertainty_method=UncertaintyMethod.SIGMA_POINT,
        tagged_covariance=True,
    )
    assert set(res_sp.tagged_covariance.kind.to_pylist()) == {"sigma_point"}


def test_sbdb_generate_ephemeris_rejects_sampling(sbdb_orbit, sbdb_grid) -> None:
    """generate_ephemeris rejects the sampling methods with a typed error
    (the silent-ignore fix), on a real SBDB orbit + observer."""
    observers = empyrean.get_observer_states(["500"], sbdb_grid)
    for method in (UncertaintyMethod.SIGMA_POINT, UncertaintyMethod.MONTE_CARLO):
        with pytest.raises(ValueError, match="sampling uncertainty methods"):
            empyrean.generate_ephemeris(sbdb_orbit, observers, uncertainty_method=method)
