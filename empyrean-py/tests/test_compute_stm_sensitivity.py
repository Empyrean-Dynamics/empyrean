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


def _orbit(*, with_covariance: bool) -> CartesianOrbits:
    kwargs = {
        "epoch": [_STATE["epoch"]],
        "x": [_STATE["x"]],
        "y": [_STATE["y"]],
        "z": [_STATE["z"]],
        "vx": [_STATE["vx"]],
        "vy": [_STATE["vy"]],
        "vz": [_STATE["vz"]],
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


def _generate(*, with_covariance: bool, compute_stm: bool):
    config = EphemerisConfig(
        propagation=PropagationConfig(
            compute_stm=compute_stm,
            uncertainty_method=UncertaintyMethod.FIRST_ORDER,
        )
    )
    return empyrean.generate_ephemeris(
        _orbit(with_covariance=with_covariance),
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
