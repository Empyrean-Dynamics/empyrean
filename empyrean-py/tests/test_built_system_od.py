"""Orbit determination through the reusable pre-built force-model handle.

``BuiltSystem.determine`` / ``.evaluate`` / ``.refine`` mirror the
module-level :func:`empyrean.determine` / :func:`empyrean.evaluate` /
:func:`empyrean.refine` argument for argument. Two things have to hold and
are pinned here:

1. A fit through a matching handle is *identical* to the one-shot — the
   handle changes only when the force model is assembled.
2. A mismatched handle **refuses** the fit rather than quietly assembling
   a per-solve force model. The engine's own adoption filter degrades
   silently, which is correct there (it costs only speed); at a named
   reuse API a caller who thinks they are amortizing and is not is the
   hidden fallback this package will not ship.
"""

from __future__ import annotations

from pathlib import Path

import empyrean
import numpy as np
import pytest
from empyrean import ForceModelTier, Frame, build_system, od_system, read_ades, refine
from empyrean.od.result import ODConfig

DATA_DIR = Path(__file__).parent / "fixtures"
APOPHIS_MULTIAPP = DATA_DIR / "99942_apophis_multiapp.psv"

# A short slice of the bundled multi-apparition arc: enough to fit, small
# enough that a twenty-call refine loop is seconds rather than minutes.
_ARC_ROWS = 120
# The loop length the reuse claim is actually about — a measure-and-extend
# consumer refines the same object over and over.
_LOOP = 20


@pytest.fixture(scope="module")
def short_arc():
    if not APOPHIS_MULTIAPP.exists():
        pytest.skip(f"missing fixture: {APOPHIS_MULTIAPP}")
    optical, _radar = read_ades(APOPHIS_MULTIAPP)
    return optical[:_ARC_ROWS]


@pytest.fixture(scope="module")
def prior_orbit(short_arc):
    """A covariance-bearing orbit for ``refine`` / ``evaluate`` to consume."""
    fit = empyrean.determine(short_arc).single()
    assert fit.converged, "the short-arc seed fit did not converge"
    return fit.orbit


def _state(orbit) -> np.ndarray:
    c = orbit.coordinates
    return np.array(
        [
            c.x.to_numpy(zero_copy_only=False)[0],
            c.y.to_numpy(zero_copy_only=False)[0],
            c.z.to_numpy(zero_copy_only=False)[0],
            c.vx.to_numpy(zero_copy_only=False)[0],
            c.vy.to_numpy(zero_copy_only=False)[0],
            c.vz.to_numpy(zero_copy_only=False)[0],
        ]
    )


# ── The OD-keyed constructor ──────────────────────────────────


def test_od_system_freezes_the_recipe_a_fit_runs_under() -> None:
    """The whole reason ``od_system`` exists: the frame and the divisor an
    OD fit uses are not the caller's to choose, and the frame is the easy
    one to get wrong."""
    system = od_system()
    assert system.force_model is ForceModelTier.STANDARD
    assert system.frame is Frame.ECLIPTICJ2000, "OD does not integrate in ICRF"
    assert system.encounter_timescale_divisor == 1000.0

    described = system.describe()
    assert described.frame is Frame.ECLIPTICJ2000
    assert described.encounter_timescale_divisor == 1000.0
    assert described.force_model is ForceModelTier.STANDARD


def test_od_system_accepts_the_same_tier_spellings_as_build_system() -> None:
    for spelling in (ForceModelTier.BASIC, "basic", 1):
        assert od_system(spelling).force_model is ForceModelTier.BASIC


# ── Parity with the one-shots ─────────────────────────────────


def test_refine_matches_the_one_shot(short_arc, prior_orbit) -> None:
    system = od_system()
    one_shot = refine(prior_orbit, short_arc)
    via_handle = system.refine(prior_orbit, short_arc)
    np.testing.assert_array_equal(
        _state(via_handle.orbit),
        _state(one_shot.orbit),
        err_msg="refine through the handle must be identical to the one-shot",
    )
    assert via_handle.converged == one_shot.converged


def test_evaluate_matches_the_one_shot(short_arc, prior_orbit) -> None:
    system = od_system()
    one_shot = empyrean.evaluate(prior_orbit, short_arc)
    via_handle = system.evaluate(prior_orbit, short_arc)
    np.testing.assert_array_equal(
        via_handle.observations.ra_residual.to_numpy(zero_copy_only=False),
        one_shot.observations.ra_residual.to_numpy(zero_copy_only=False),
    )
    np.testing.assert_array_equal(
        via_handle.observations.dec_residual.to_numpy(zero_copy_only=False),
        one_shot.observations.dec_residual.to_numpy(zero_copy_only=False),
    )


def test_determine_matches_the_one_shot(short_arc) -> None:
    system = od_system()
    one_shot = empyrean.determine(short_arc).single()
    via_handle = system.determine(short_arc).single()
    np.testing.assert_array_equal(_state(via_handle.orbit), _state(one_shot.orbit))


def test_a_refine_loop_is_identical_to_the_same_loop_without_a_handle(
    short_arc, prior_orbit
) -> None:
    """The measure-and-extend shape, twenty deep.

    One handle serves every call in the loop, and every result matches the
    handle-less loop element for element — so the amortization is free of
    numerical consequence, which is the only way it is worth having.
    """
    system = od_system()
    with_handle = [_state(system.refine(prior_orbit, short_arc).orbit) for _ in range(_LOOP)]
    without = [_state(refine(prior_orbit, short_arc).orbit) for _ in range(_LOOP)]
    assert len(with_handle) == _LOOP
    for i, (a, b) in enumerate(zip(with_handle, without, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"refine call {i}")


# ── The guard refuses; it does not degrade ────────────────────


def test_an_icrf_handle_is_refused_by_axis(short_arc, prior_orbit) -> None:
    """The documented footgun: every propagation example freezes ICRF, and
    OD integrates in EclipticJ2000. This must be a loud refusal, not a
    silent per-solve rebuild."""
    icrf = build_system(force_model="standard", frame="icrf")
    with pytest.raises(ValueError, match="frame mismatch"):
        icrf.refine(prior_orbit, short_arc)
    with pytest.raises(ValueError, match="identity guard"):
        icrf.evaluate(prior_orbit, short_arc)


def test_a_wrong_tier_handle_is_refused_by_axis(short_arc, prior_orbit) -> None:
    basic = od_system(ForceModelTier.BASIC)
    standard = ODConfig()  # force_model defaults to Standard
    with pytest.raises(ValueError, match="force-model mismatch"):
        basic.refine(prior_orbit, short_arc, config=standard)


def test_a_non_default_divisor_handle_is_refused_by_axis(short_arc, prior_orbit) -> None:
    odd = build_system(
        force_model="standard",
        frame="ecliptic_j2000",
        encounter_timescale_divisor=500.0,
    )
    assert odd.encounter_timescale_divisor == 500.0
    with pytest.raises(ValueError, match="divisor mismatch"):
        odd.refine(prior_orbit, short_arc)


def test_auto_force_model_is_refused_rather_than_silently_unamortized(
    short_arc, prior_orbit
) -> None:
    """``auto_force_model`` lets the fit re-pick its own tier part-way
    through, and no frozen handle can follow it — the engine would drop the
    handle mid-fit and reassemble per solve while the call still returned a
    result. Nothing about the handle is wrong, so the refusal is its own
    guard axis, and it has to name the field."""
    system = od_system()
    auto = ODConfig(auto_force_model=True)
    for call in (
        lambda: system.refine(prior_orbit, short_arc, config=auto),
        lambda: system.evaluate(prior_orbit, short_arc, config=auto),
        lambda: system.determine(short_arc, config=auto),
    ):
        with pytest.raises(ValueError, match="auto_force_model"):
            call()
    # …and the same handle with the same config minus that one field is
    # accepted, so the refusal is specific rather than blanket.
    assert system.refine(prior_orbit, short_arc, config=ODConfig()).converged


def test_the_refusal_names_the_remedy(short_arc, prior_orbit) -> None:
    """An error a caller cannot act on is its own failure: the message has
    to say the handle must be rebuilt, which is what makes it visibly not
    a silent fallback."""
    icrf = build_system(force_model="standard", frame="icrf")
    with pytest.raises(ValueError, match="rebuild the handle"):
        icrf.determine(short_arc)
