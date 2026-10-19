"""A solved axis the observations cannot constrain is refused by name.

The engine's ``SolvedAxisUnconstrained`` refusal (OD error code 53) must reach
the Python layer by name: a fit that marks an axis solved but whose
observations carry no information on it raises, naming the axis and stating
that its posterior σ did not fall below the prior — never a converged result
publishing the prior back as a posterior.

This is the complement of the zero-seeded-non-grav fix: that fix keeps the
Marsden force present at a zero seed so a well-determined arc estimates A2;
this guard catches the arc that genuinely cannot. A single-apparition (1999)
Bennu arc determines the state but leaves the Yarkovsky A2 unmeasurable, so a
Marsden solve on it is refused rather than returning the prior echo.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    CovarianceKind,
    JointFunctional,
    JointProvenance,
    JointQuality,
    NonGravParams,
    ODConfig,
    PackedJoint,
    PackedJoints,
    SolveForParams,
)

# The Marsden axes are priored far tighter than any real optical arc can
# improve on, so their posterior σ cannot fall below the prior — a prior
# echo the guard must refuse rather than publish as an estimate.
_SIGMA_MARSDEN = 1e-16


def _prior_joint(state6: np.ndarray) -> PackedJoint:
    width = 9
    full = np.zeros((width, width), dtype=np.float64)
    full[:6, :6] = np.asarray(state6, dtype=np.float64)
    full[6, 6] = _SIGMA_MARSDEN**2
    full[7, 7] = _SIGMA_MARSDEN**2
    full[8, 8] = _SIGMA_MARSDEN**2
    tri = [full[i, j] for i in range(width) for j in range(i + 1)]
    return PackedJoint(
        layout_present=0b111,
        layout_considered=0,
        layout_marginalized=0,
        width=width,
        tri=np.asarray(tri, dtype=np.float64),
        kind=CovarianceKind.LINEAR,
        quality=JointQuality.POSITIVE_DEFINITE,
        cov_rank=0,
        cov_null_axes=0,
        cov_min_eig_corr=float("nan"),
        functional=JointFunctional.STATE,
        provenance=JointProvenance.DERIVED,
        asserted_source=None,
    )


def _early_arc(obs):
    """The 1999–2000 discovery arc — the same short optical arc the
    zero-seeded-non-grav regression determines and refines on."""
    years = [t[:4] for t in obs.table.column("obs_time").to_pylist()]
    idx = np.nonzero(np.array([y in ("1999", "2000") for y in years]))[0]
    return obs.take(idx)


def test_unconstrained_marsden_axis_is_refused_by_name() -> None:
    obs = empyrean.query_observations(["101955"])
    short = _early_arc(obs)
    if len(short) < 6:
        pytest.skip("no cached early Bennu arc to leave the Marsden block unconstrained")

    anchor = empyrean.determine(
        short, config=ODConfig(solve_for=SolveForParams.STATE_ONLY)
    ).single()
    state6 = anchor.orbit.covariance.joint(0).state_block()

    ng = NonGravParams.from_kwargs(a1=[0.0], a2=[0.0], a3=[0.0], model=["inverse_square"])
    primed = anchor.orbit.set_column("non_grav", ng)
    primed = primed.set_column("covariance", PackedJoints.from_joints([_prior_joint(state6)]))

    # The Marsden block is priored far tighter than this optical arc can
    # improve on, so the solve would return the prior back as a posterior —
    # the fit must refuse by name instead.
    with pytest.raises(Exception) as excinfo:
        empyrean.refine(primed, short, config=ODConfig(solve_for=SolveForParams.STATE_AND_NONGRAV))
    message = str(excinfo.value)
    assert "no information" in message.lower(), (
        f"the refusal must state the observations carry no information: {message!r}"
    )
    assert "A1" in message or "A2" in message or "A3" in message or "Marsden" in message, (
        f"the refusal must name the axis: {message!r}"
    )
