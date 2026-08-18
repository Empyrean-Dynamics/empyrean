"""``compute_stm`` reaches the engine on the ephemeris path.

The C-ABI ephemeris-config converter used to hand-roll its narrow config
instead of routing through the shared propagation converter, so
``compute_stm`` — along with several other propagation-level knobs — was
accepted at the Python surface and silently discarded before the engine
saw it. A caller asking for observation partials on an orbit with no
covariance got a clean success, no warnings, and no partials, and the
published docs described that behaviour as the contract.

The converter now routes through the shared one, so the flag is live end
to end. These tests assert the *observable* consequence rather than
re-reading the converter: with ``compute_stm`` set and **no input
covariance at all**, every ephemeris epoch carries an observation
Jacobian. Sabotaging the converter routing drives that to zero.

They also pin the corrected docstrings: the sensitivity table is keyed on
the caller's own ``orbit_id`` and ``obs_code``, so the per-chain filter
the docs prescribe actually selects rows.
"""

import empyrean
import numpy as np
import pytest
from empyrean import (
    SENSITIVITY_ROW_DEC,
    SENSITIVITY_ROW_RA,
    CartesianCoordinates,
    CartesianOrbits,
    Epochs,
    Observers,
    Origin,
    UncertaintyMethod,
)
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.ephemeris.result import EphemerisConfig
from empyrean.propagation.config import PropagationConfig

ORBIT_ID = "compute-stm-fixture"
OBS_CODE = "500"
OBS_EPOCHS = Epochs.from_mjd([61000.5, 61010.5, 61020.5], scale="tdb")

# A bound heliocentric state in the ephemeris integration frame.
_STATE = {
    "epoch": 61000.0,
    "x": 0.9412,
    "y": 0.3311,
    "z": -0.0142,
    "vx": -0.006_12,
    "vy": 0.016_10,
    "vz": 0.000_31,
}


def _orbit(*, with_covariance: bool, delta: np.ndarray | None = None) -> CartesianOrbits:
    state = np.array(
        [_STATE["x"], _STATE["y"], _STATE["z"], _STATE["vx"], _STATE["vy"], _STATE["vz"]],
        dtype=float,
    )
    if delta is not None:
        state = state + delta
    kwargs = {
        "epoch": [_STATE["epoch"]],
        "x": [state[0]],
        "y": [state[1]],
        "z": [state[2]],
        "vx": [state[3]],
        "vy": [state[4]],
        "vz": [state[5]],
        "frame": "ecliptic_j2000",
        "origin": [str(Origin.SUN)],
    }
    if with_covariance:
        kwargs["covariance"] = CartesianCovariance.from_matrix(
            np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
        )
    return CartesianOrbits.from_kwargs(
        orbit_id=[ORBIT_ID],
        coordinates=CartesianCoordinates.from_kwargs(**kwargs),
    )


def _generate(*, with_covariance: bool, compute_stm: bool, delta: np.ndarray | None = None):
    config = EphemerisConfig(
        propagation=PropagationConfig(
            compute_stm=compute_stm,
            uncertainty_method=UncertaintyMethod.FIRST_ORDER,
        )
    )
    return empyrean.generate_ephemeris(
        _orbit(with_covariance=with_covariance, delta=delta),
        Observers.from_code(OBS_CODE, OBS_EPOCHS),
        config,
    )


def _jacobian_rows(result) -> int:
    """Rows whose Jacobian is actually populated."""
    sens = result.sensitivity
    if sens is None or len(sens) == 0:
        return 0
    return len(sens) - sens.column("jacobian").null_count


# ── The flag reaches the engine ───────────────────────────────────────


def test_compute_stm_produces_partials_without_any_covariance():
    """The headline contract: no input covariance, ``compute_stm=True``,
    and every ephemeris epoch comes back with an observation Jacobian.

    This is what makes the covariance-attach shim unnecessary. If the
    converter stops forwarding the flag, this goes to zero.
    """
    off = _generate(with_covariance=False, compute_stm=False)
    on = _generate(with_covariance=False, compute_stm=True)

    assert _jacobian_rows(off) == 0, (
        "without a covariance and without compute_stm there is no STM to "
        "compose, so no epoch should carry a Jacobian"
    )
    assert _jacobian_rows(on) == len(OBS_EPOCHS), (
        "compute_stm=True must reach the engine and force the hyperdual "
        "integration that fills the observation Jacobians — zero populated "
        "rows means the flag was dropped between the Python config and "
        "EphemerisPropagationConfig"
    )


def test_partials_are_finite_and_correctly_shaped():
    """Populated is not enough: the partials have to be real numbers of
    the shape the table advertises."""
    on = _generate(with_covariance=False, compute_stm=True)
    sens = on.sensitivity
    jac = sens.jacobians_array()
    assert jac is not None
    n_params = int(sens.column("n_params")[0].as_py())
    assert jac.shape == (len(OBS_EPOCHS), 6, n_params)
    assert np.all(np.isfinite(jac)), "observation Jacobians must not be NaN-filled"
    assert np.any(jac != 0.0), "an all-zero Jacobian is a dropped partial wearing a shape"


def test_an_input_covariance_still_produces_partials():
    """The pre-existing route is untouched: a covariance-bearing orbit
    gets partials with or without the flag."""
    assert _jacobian_rows(_generate(with_covariance=True, compute_stm=False)) == len(OBS_EPOCHS)
    assert _jacobian_rows(_generate(with_covariance=True, compute_stm=True)) == len(OBS_EPOCHS)


# ── The corrected docstrings are true ─────────────────────────────────


def test_sensitivity_rows_carry_the_callers_orbit_id():
    """The per-chain filter the docs prescribe must select rows.

    The sensitivity table used to carry the C ABI's synthetic
    ``"orbit_{i}"`` id while the ephemeris table beside it carried the
    caller's real one, so ``sens.select("orbit_id", oid)`` — the
    documented way to reach a single chain — silently returned an empty
    table for every real orbit id.
    """
    result = _generate(with_covariance=False, compute_stm=True)
    sens = result.sensitivity

    assert result.ephemeris.orbit_id.to_pylist() == [ORBIT_ID] * len(OBS_EPOCHS)
    assert sens.orbit_id.to_pylist() == [ORBIT_ID] * len(OBS_EPOCHS), (
        "the sensitivity table must be keyed on the same orbit_id as the "
        "ephemeris table it accompanies"
    )
    assert sens.chain_keys() == [(ORBIT_ID, OBS_CODE)]

    chain = sens.select("orbit_id", ORBIT_ID).select("obs_code", OBS_CODE)
    assert len(chain) == len(OBS_EPOCHS), (
        "the documented per-chain filter selected nothing — the table is "
        "keyed on something other than the caller's ids"
    )
    assert chain.jacobians_array() is not None


def test_missing_partials_error_names_compute_stm():
    """The corrected error text has to name the remedy that now works.

    Before the fix the message blamed a missing input covariance, which
    was the *symptom*; the remedy is either a covariance or
    ``compute_stm``, and a caller who reads the message must be able to
    act on it.
    """
    result = _generate(with_covariance=False, compute_stm=False)
    sens = result.sensitivity
    assert sens is not None and len(sens) > 0, (
        "the run should still emit sensitivity rows, just without partials"
    )
    chain = sens.select("orbit_id", ORBIT_ID).select("obs_code", OBS_CODE)
    with pytest.raises(ValueError, match="compute_stm"):
        chain.propagate_covariance(np.eye(6) * 1e-16)


# ── The partials are the numbers the shim produced ────────────────────


def test_no_covariance_partials_equal_the_shim_partials():
    """The shim-removal warrant.

    Attaching a dummy covariance purely to switch the partials on was the
    0.9.0-era workaround. The partials produced with ``compute_stm`` and
    **no** covariance are not merely present — they are the same numbers,
    element for element, so the shim can be deleted without a
    re-validation pass.

    Exact equality is deliberate. Both runs take the same hyperdual arm
    and Σ₀ feeds only the covariance composition, never the integration.
    A red here does not mean the comparison is too strict; it means a
    seeded Σ₀ has started influencing step selection and the two paths no
    longer integrate the same trajectory. Do not replace this with a
    tolerance.
    """
    without = _generate(with_covariance=False, compute_stm=True)
    shim = _generate(with_covariance=True, compute_stm=False)

    a = without.sensitivity.jacobians_array()
    b = shim.sensitivity.jacobians_array()
    assert a is not None and b is not None
    assert a.shape == b.shape == (len(OBS_EPOCHS), 6, 6)
    np.testing.assert_array_equal(
        a,
        b,
        err_msg=(
            "the no-covariance partials must be bit-identical to the ones the "
            "dummy-covariance shim produced"
        ),
    )


# ── The partials are correct, not merely present ──────────────────────


def _sky(delta: np.ndarray | None) -> np.ndarray:
    """(n_epochs, 2) array of predicted (RA, Dec) in degrees."""
    result = _generate(with_covariance=False, compute_stm=False, delta=delta)
    coords = result.ephemeris.coordinates
    return np.column_stack([np.asarray(coords.lon.to_numpy()), np.asarray(coords.lat.to_numpy())])


def _wrap_degrees(d: np.ndarray) -> np.ndarray:
    """Fold an angle difference into (-180, 180] so an RA seam is not a
    360-degree jump."""
    return (d + 180.0) % 360.0 - 180.0


def _central_differences(steps: np.ndarray) -> np.ndarray:
    """``(n_epochs, 2, 6)`` central-difference d(RA, Dec)/dx₀, in deg/AU
    and deg/(AU/day)."""
    columns = []
    for k in range(6):
        delta = np.zeros(6)
        delta[k] = steps[k]
        columns.append(_wrap_degrees(_sky(delta) - _sky(-delta)) / (2.0 * steps[k]))
    return np.stack(columns, axis=-1)


# Position in AU, velocity in AU/day. Both are far above f64 round-off on
# the state and far below the arc's curvature scale; the residual
# disagreement below is systematic, not step-dependent (it is unchanged
# across 1e-7 … 1e-4 AU).
_FD_STEPS = np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])


def test_partials_match_a_central_difference():
    """ "Finite and not all zero" is not "correct".

    Central-difference each of the six input state components and compare
    the resulting dRA/dx₀ and dDec/dx₀ against the returned Jacobian rows,
    indexed through ``SENSITIVITY_ROW_RA`` / ``SENSITIVITY_ROW_DEC``
    rather than by literal — so a row misread is caught in the same
    assertion as a wrong number. A wrong row or a wrong unit is off by
    orders of magnitude and fails here immediately.

    The Jacobian is reported against the input state in the input frame,
    translated to the barycentre — a translation a Jacobian is invariant
    under — so perturbing the ecliptic-J2000 state this fixture supplies
    is the matching axis.

    Tolerances
    ----------
    The engine documents that the observation Jacobian composes
    ∂(obs)/∂(state at t_obs)·Φ(t_obs, t₀) and *omits the light-time
    terms*: the STM is sampled at t_obs rather than at emission
    t_obs − τ. That approximation, not the finite difference, sets the
    floor here — the disagreement is unchanged from a 1e-7 to a 1e-4 AU
    step. The position columns land within 1.2e-4 relative and the
    velocity columns within 7.8e-3 at the shortest baseline, which is the
    documented τ/Δt scale (τ ≈ 0.004 d over a 0.5 d arc). The tolerances
    below sit an order of magnitude above the measured values so an
    ordinary numerical drift is caught, and
    :func:`test_the_light_time_omission_shrinks_with_baseline` pins the
    approximation's own signature rather than letting it hide in slack.
    """
    analytic = _generate(with_covariance=False, compute_stm=True).sensitivity.jacobians_array()
    assert analytic is not None
    finite_difference = _central_differences(_FD_STEPS)

    for k in range(6):
        # Columns 0..3 are the position partials; 3..6 the velocity
        # partials, which carry the omitted light-time term.
        rel_tol = 1e-3 if k < 3 else 2e-2
        for epoch_index in range(len(OBS_EPOCHS)):
            for name, row, column in (
                ("RA", SENSITIVITY_ROW_RA, 0),
                ("Dec", SENSITIVITY_ROW_DEC, 1),
            ):
                fd = finite_difference[epoch_index, column, k]
                ad = analytic[epoch_index, row, k]
                tol = rel_tol * max(abs(fd), 1.0)
                assert abs(ad - fd) <= tol, (
                    f"d{name}/dx0[{k}] at epoch {epoch_index}: analytic {ad:.9e} vs "
                    f"central difference {fd:.9e} (tolerance {tol:.3e})"
                )


def test_the_light_time_omission_shrinks_with_baseline():
    """The velocity-column gap is the documented light-time omission, not
    noise — so it must scale like τ/Δt.

    ``OBS_EPOCHS`` runs 0.5, 10.5 and 20.5 days past the orbit epoch. If
    the velocity-column disagreement is the omitted −v·∂τ/∂x term, the
    relative gap at the 20-day baseline is far smaller than at the
    half-day one. A gap that did *not* shrink would mean something other
    than light time is wrong, and this test is what tells the two apart.
    """
    analytic = _generate(with_covariance=False, compute_stm=True).sensitivity.jacobians_array()
    assert analytic is not None
    finite_difference = _central_differences(_FD_STEPS)

    def worst_velocity_gap(epoch_index: int) -> float:
        gaps = []
        for k in range(3, 6):
            for row, column in ((SENSITIVITY_ROW_RA, 0), (SENSITIVITY_ROW_DEC, 1)):
                fd = finite_difference[epoch_index, column, k]
                ad = analytic[epoch_index, row, k]
                gaps.append(abs(ad - fd) / max(abs(fd), 1.0))
        return max(gaps)

    shortest = worst_velocity_gap(0)
    longest = worst_velocity_gap(len(OBS_EPOCHS) - 1)
    assert shortest > longest * 5, (
        "the velocity-column disagreement must shrink with the baseline, as the "
        f"omitted light-time term does (got {shortest:.3e} at 0.5 d vs "
        f"{longest:.3e} at 20.5 d)"
    )
