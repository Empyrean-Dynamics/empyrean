"""Structured continuous-thrust input plumbing tests.

Verifies that the ``thrust_arcs`` keyword of :func:`empyrean.propagate`
threads structured :class:`~empyrean.ThrustParams` /
:class:`~empyrean.ThrustArc` / steering-law dataclasses through the full
empyrean-py -> empyrean-c -> empyrean -> empyrean-core -> villeneuve
stack and measurably perturbs the propagated trajectory relative to an
identical ballistic orbit.

The burn INPUT is the deliverable here: a finite-burn arc reaches the
engine dynamics and changes propagation. The burn-sensitivity segment
count is no longer a tagged-covariance column — it rides the packed
joint's thrust-axis layout — so the readback test pins that the output
path is wired (via ``solved_width``) rather than a specific count: the
currently linked engine reports the marginalized 6x6 state block
(``solved_width == 6``).
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianCovariance,
    CartesianOrbits,
    ConstantRTN,
    Epochs,
    InertialFixed,
    Origin,
    PropagationConfig,
    ThrustArc,
    ThrustParams,
    VelocityTangent,
)

_T0 = 59000.0
_EPOCHS = Epochs.from_mjd(np.array([59000.0, 59005.0, 59012.0]), scale="tdb")


def _diag_cov() -> np.ndarray:
    cov = np.zeros((6, 6))
    for i in range(6):
        cov[i, i] = 1.0e-16
    return cov


def _orbit(orbit_id: str, with_cov: bool = True) -> CartesianOrbits:
    """A heliocentric Cartesian state with a tight diagonal covariance so
    the burn-sensitivity path can engage."""
    kwargs = {
        "epochs": Epochs.from_kwargs(mjd=[_T0], scale="tdb"),
        "x": [1.0],
        "y": [0.1],
        "z": [0.05],
        "vx": [-0.005],
        "vy": [0.015],
        "vz": [0.001],
        "frame": "icrf",
        "origin": ["Sun"],
    }
    if with_cov:
        kwargs["covariance"] = CartesianCovariance.from_matrix(_diag_cov()[np.newaxis, :, :])
    coords = CartesianCoordinates.from_kwargs(**kwargs)
    return CartesianOrbits.from_kwargs(orbit_id=[orbit_id], coordinates=coords)


def _rtn_arc(central_body: object = Origin.SUN) -> ThrustArc:
    return ThrustArc(
        start_mjd=59000.0,
        end_mjd=59010.0,
        scale="tdb",
        thrust_n=1000.0,
        mass_kg=1000.0,
        steering=ConstantRTN(alpha_rad=0.1, beta_rad=0.2),
        sharpness=100.0,
        central_body=central_body,
    )


def _cfg() -> PropagationConfig:
    cfg = PropagationConfig()
    cfg.advanced.cache_integrator_steps = True
    return cfg


def _last_position(result: empyrean.PropagationResult) -> np.ndarray:
    c = result.states.coordinates
    return np.array([c.x.to_numpy()[-1], c.y.to_numpy()[-1], c.z.to_numpy()[-1]])


# ── Public type surface (no engine / kernels needed) ──────────


def test_public_thrust_types_expose_wrapper_field_names() -> None:
    """The Python dataclasses mirror the engine surface one-for-one."""
    rtn = ConstantRTN(alpha_rad=0.1, beta_rad=0.2)
    assert (rtn.alpha_rad, rtn.beta_rad) == (0.1, 0.2)

    fixed = InertialFixed(direction=(1.0, 2.0, 3.0))
    assert fixed.direction == (1.0, 2.0, 3.0)

    # VelocityTangent carries no parameters.
    VelocityTangent()

    arc = ThrustArc(
        start_mjd=59000.0,
        end_mjd=59010.0,
        scale="tdb",
        thrust_n=1000.0,
        mass_kg=1000.0,
        steering=rtn,
        sharpness=100.0,
        central_body=Origin.SUN,
    )
    # Field names match the wrapper's ThrustArc exactly.
    assert arc.isp_s is None  # constant mass by default
    assert arc.central_body == Origin.SUN
    assert arc.steering is rtn

    params = ThrustParams(arcs=[arc])
    assert params.arcs == [arc]
    assert params.dv_corrections == []


def test_thrust_arcs_length_must_match_orbit_count() -> None:
    """A per-orbit thrust list of the wrong length is rejected up front,
    before the engine is ever called."""
    orbits = _orbit("solo")
    with pytest.raises(ValueError, match="one entry per orbit"):
        empyrean.propagate(
            orbits,
            _EPOCHS,
            config=_cfg(),
            thrust_arcs=[ThrustParams(arcs=[_rtn_arc()]), None],
        )


# ── End-to-end (requires kernels; conftest skips the session
#    when they are unavailable) ─────────────────────────────────


def test_thrust_perturbs_trajectory() -> None:
    """A ConstantRTN burn must measurably move the final state relative to
    the identical ballistic orbit — the definitive proof the structured
    thrust input reached and was honored by the engine dynamics."""
    thrust = ThrustParams(arcs=[_rtn_arc()])

    with_thrust = empyrean.propagate(_orbit("thrust"), _EPOCHS, config=_cfg(), thrust_arcs=[thrust])
    ballistic = empyrean.propagate(_orbit("ballistic"), _EPOCHS, config=_cfg())

    delta = np.linalg.norm(_last_position(with_thrust) - _last_position(ballistic))
    assert delta > 1.0e-3, f"thrust arc must perturb the trajectory (delta = {delta:e} AU)"


def test_velocity_tangent_steering_perturbs_trajectory() -> None:
    """VelocityTangent steering marshals and is honored by the engine."""
    arc = ThrustArc(
        start_mjd=59000.0,
        end_mjd=59010.0,
        scale="tdb",
        thrust_n=1000.0,
        mass_kg=1000.0,
        steering=VelocityTangent(),
        sharpness=100.0,
        central_body=Origin.SUN,
    )
    with_thrust = empyrean.propagate(
        _orbit("vt"), _EPOCHS, config=_cfg(), thrust_arcs=[ThrustParams(arcs=[arc])]
    )
    ballistic = empyrean.propagate(_orbit("vt-ballistic"), _EPOCHS, config=_cfg())
    delta = np.linalg.norm(_last_position(with_thrust) - _last_position(ballistic))
    assert delta > 1.0e-3


def test_central_body_accepts_origin_naif_and_string() -> None:
    """`central_body` accepts a typed Origin, a bare NAIF id, or a
    canonical string; all three resolve to the same body and produce the
    identical propagated state."""
    results = []
    for central in (Origin.SUN, 10, "Sun"):
        r = empyrean.propagate(
            _orbit("cb"),
            _EPOCHS,
            config=_cfg(),
            thrust_arcs=[ThrustParams(arcs=[_rtn_arc(central_body=central)])],
        )
        results.append(_last_position(r))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=0)
    np.testing.assert_allclose(results[0], results[2], rtol=0, atol=0)


def test_thrust_tagged_covariance_readback_is_wired() -> None:
    """With a Δv correction the tagged-covariance readback stays populated
    and the ``solved_width`` column is readable for a thrust orbit (the
    output path is wired).

    The burn-sensitivity segment count is no longer a tagged-covariance
    column — it is encoded in the packed joint's thrust-axis layout — so
    the remaining per-row wiring check is ``solved_width``: the currently
    linked engine reports the marginalized 6x6 block (``solved_width ==
    6``), so this asserts the plumbing rather than a specific solved width.
    """
    # The burn-prior covariance now rides the orbit's packed joint (its
    # thrust axes), not a `correction_covariances` sibling; this test pins
    # that the tagged-covariance readback path is wired for a thrust orbit.
    thrust = ThrustParams(
        arcs=[_rtn_arc()],
        dv_corrections=[(0.0, 0.0, 0.0)],
    )
    result = empyrean.propagate(
        _orbit("tagged"),
        _EPOCHS,
        config=_cfg(),
        thrust_arcs=[thrust],
        tagged_covariance=True,
    )
    tc = result.tagged_covariance
    assert tc is not None
    assert len(tc) == len(_EPOCHS)
    widths = tc.column("solved_width").to_numpy(zero_copy_only=False)
    # The column is present, aligned 1:1 with states, and state-inclusive —
    # the readback path is wired end to end.
    assert len(widths) == len(_EPOCHS)
    assert (widths >= 6).all()
