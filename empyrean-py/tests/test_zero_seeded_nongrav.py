"""Regression: a zero-seeded solved non-grav axis is estimated, not silently dropped.

A ``refine`` that solves the Marsden block with the A2 coefficient seeded at
exactly 0.0 (A1 = A3 = 0 too) used to come back ``converged`` with ``A2 = NaN``,
a posterior sigma equal to the prior, and no warning. The distribution's orbit
converter inferred "no non-grav model" from the all-zero coefficient vector, so
the engine received a gravity-only orbit while the supplied packed joint still
declared the Marsden prior block — a declared-but-unforced axis the observations
could not constrain.

A declared Marsden model with zero coefficients is a model with zero
coefficients, never absent. These end-to-end refines (through the public Python
API, on the 101955 Bennu arc) assert the fit now delivers a finite A2 with a
posterior sigma below the prior, and that the zero seed agrees with a nonzero
seed within tolerance (they minimize to the same point).
"""

from __future__ import annotations

import empyrean
import numpy as np
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

# The prior 1-sigma on A2 (sqrt of its variance); A1/A3 are pinned tight.
_SIGMA_A2 = 1e-12
_SIGMA_TIGHT = 1e-15


def _prior_joint(state6: np.ndarray) -> PackedJoint:
    """A width-9 state+Marsden prior: the fitted state 6x6 with a diagonal
    Marsden 3x3 (A1/A3 tight, A2 at _SIGMA_A2), zero state-Marsden cross."""
    width = 9
    full = np.zeros((width, width), dtype=np.float64)
    full[:6, :6] = np.asarray(state6, dtype=np.float64)
    full[6, 6] = _SIGMA_TIGHT**2
    full[7, 7] = _SIGMA_A2**2
    full[8, 8] = _SIGMA_TIGHT**2
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


def _refine_with_seed(anchor_orbit, state6, obs, seed: float):
    """Prime the anchor with an inverse-square Marsden model (A2 = seed) and a
    packed Marsden prior, then refine the Marsden block on the full arc."""
    ng = NonGravParams.from_kwargs(a1=[0.0], a2=[seed], a3=[0.0], model=["inverse_square"])
    primed = anchor_orbit.set_column("non_grav", ng)
    primed = primed.set_column("covariance", PackedJoints.from_joints([_prior_joint(state6)]))
    result = empyrean.refine(
        primed, obs, config=ODConfig(solve_for=SolveForParams.STATE_AND_NONGRAV)
    )
    a2 = float(result.orbit.non_grav.a2.to_numpy(zero_copy_only=False)[0])
    sc = result.solved_covariance
    row = int(sc.marsden_slot) + 1  # marsden_slot is A1; +1 -> A2
    sigma_a2 = float(sc.matrix[row, row]) ** 0.5
    return result, a2, sigma_a2


def test_zero_seeded_marsden_axis_is_estimated_not_dropped() -> None:
    obs = empyrean.query_observations(["101955"])
    radar = empyrean.query_radar(["101955"])
    years = [t[:4] for t in obs.table.column("obs_time").to_pylist()]
    short = obs.take(np.nonzero(np.array([y in ("1999", "2000") for y in years]))[0])
    r_years = [t[:4] for t in radar.table.column("obs_time").to_pylist()]
    radar_short = radar.take(np.nonzero(np.array([y in ("1999", "2000") for y in r_years]))[0])

    anchor = empyrean.determine(
        short,
        radar=radar_short if len(radar_short) else None,
        config=ODConfig(solve_for=SolveForParams.STATE_ONLY),
    ).single()
    state6 = anchor.orbit.covariance.joint(0).state_block()

    r0, a2_zero, sigma_zero = _refine_with_seed(anchor.orbit, state6, obs, 0.0)
    _rn, a2_nonzero, sigma_nonzero = _refine_with_seed(anchor.orbit, state6, obs, -1e-14)

    # The zero-seeded fit must deliver, not silently no-solve.
    assert r0.converged, "the zero-seeded Marsden fit must converge"
    assert np.isfinite(a2_zero), f"A2 from a zero seed must be finite, got {a2_zero}"
    assert sigma_zero < _SIGMA_A2, (
        f"posterior sigma(A2) {sigma_zero:.3e} must be below the prior {_SIGMA_A2:.1e} "
        "— equality means the observations carried no information on the axis"
    )

    # The zero seed and a nonzero seed minimize to the same point.
    assert np.isfinite(a2_nonzero) and sigma_nonzero < _SIGMA_A2
    assert abs(a2_zero - a2_nonzero) <= 1e-4 * abs(a2_nonzero), (
        f"zero-seed A2 {a2_zero:.6e} disagrees with the nonzero-seed fit "
        f"{a2_nonzero:.6e} beyond tolerance"
    )
