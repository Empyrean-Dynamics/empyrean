"""SBDB non-grav time delay (DT) plumbing tests.

Verifies that the `non_grav.dt` column on a `NonGravParams` quivr table
threads through the full empyrean-py → empyrean-c → empyrean →
empyrean-core → villeneuve stack and produces a measurably different
trajectory for objects where SBDB fits a non-zero DT (e.g. 67P at
+45.7 days).

Without this plumbing the non-grav force is evaluated at r(t) instead
of r(t − DT), which causes ~10²–10⁶× error blow-up on long-arc
propagation of Jupiter-family comets and 2I/Borisov.
"""

from __future__ import annotations

import empyrean
import numpy as np
from empyrean import Origin
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.orbits.nongrav import NonGravParams
from empyrean.orbits.orbits import CartesianOrbits


def _make_comet_orbit(dt: float | None) -> CartesianOrbits:
    """Build a single-orbit batch with non-grav A1 and the given DT.

    Heliocentric Cartesian state at MJD 60000 in the inner solar
    system (1 AU, near-circular). Non-grav: A1=1e-9 (radial outgassing),
    water-ice g(r). DT comes from the caller.
    """
    coords = CartesianCoordinates.from_kwargs(
        epoch=[60000.0],
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017],
        vz=[0.0],
        frame="icrf",
        origin=[str(Origin.SUN)],
    )
    non_grav = NonGravParams.from_kwargs(
        a1=[1.0e-9],
        a2=[0.0],
        a3=[0.0],
        model=["marsden_water"],
        dt=[dt] if dt is not None else [None],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["comet_dt_test"],
        coordinates=coords,
        non_grav=non_grav,
    )


def test_non_grav_dt_changes_propagated_state() -> None:
    """Same orbit propagated 365 days: with vs. without DT must differ.

    If the DT plumbing is broken (the regression that triggered this
    test), both runs use g(r(t)) and produce identical states. With DT
    correctly threaded, the +45.7-day delay shifts where g(r) is
    sampled and yields a measurably different position.
    """
    orbits_no_dt = _make_comet_orbit(dt=None)
    orbits_with_dt = _make_comet_orbit(dt=45.689)  # 67P value

    target_epochs = np.array([60365.0])
    result_no_dt = empyrean.propagate(orbits_no_dt, target_epochs)
    result_with_dt = empyrean.propagate(orbits_with_dt, target_epochs)

    cs_no = result_no_dt.states.coordinates
    cs_with = result_with_dt.states.coordinates

    pos_no = np.array([cs_no.x[0].as_py(), cs_no.y[0].as_py(), cs_no.z[0].as_py()])
    pos_with = np.array([cs_with.x[0].as_py(), cs_with.y[0].as_py(), cs_with.z[0].as_py()])

    # Δ position over 1 year with A1=1e-9 AU/d² and a 46-day DT shift
    # is ≳ 100 km — comfortably above any numerical noise floor. If
    # both states match to <1e-12 AU (~150 m), DT was silently dropped.
    delta_au = float(np.linalg.norm(pos_with - pos_no))
    assert delta_au > 1e-9, (
        f"DT plumbing broken: with-DT and no-DT trajectories match to "
        f"{delta_au:.3e} AU (expected > 1e-9 AU). The runner likely "
        f"dropped non_grav.dt before reaching villeneuve."
    )


def test_non_grav_dt_zero_delay_matches_no_dt() -> None:
    """DT=0 must equal no-DT — the time-delay code path with a zero
    delay reduces to evaluating g(r) at the present epoch, the same
    thing the no-DT path does. Catches accidentally-active code paths
    that diverge on an edge case."""
    orbits_no_dt = _make_comet_orbit(dt=None)
    orbits_zero_dt = _make_comet_orbit(dt=0.0)

    target_epochs = np.array([60100.0])
    r1 = empyrean.propagate(orbits_no_dt, target_epochs).states.coordinates
    r2 = empyrean.propagate(orbits_zero_dt, target_epochs).states.coordinates

    delta_au = float(
        np.linalg.norm(
            np.array([r1.x[0].as_py(), r1.y[0].as_py(), r1.z[0].as_py()])
            - np.array([r2.x[0].as_py(), r2.y[0].as_py(), r2.z[0].as_py()])
        )
    )
    assert delta_au < 1e-12, f"DT=0 should match no-DT exactly; got Δ = {delta_au:.3e} AU"
