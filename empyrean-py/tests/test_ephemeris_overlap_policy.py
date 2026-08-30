"""The ephemeris-overlap policy at the Python channel.

An SB441-N16 body (1 Ceres, 2 Pallas, 4 Vesta, …) is simultaneously a
member of the Standard force model and a legitimate object to propagate,
so its orbit sits on top of the ephemeris the force model is reading.
Under the default ``SUBSTITUTE_SPK`` the engine returns the body's own
SPK states and skips integration; since villeneuve 1.25.0 (bd
empyrean-c67i6) it SERVES the ephemeris off that SPK-backed trajectory
rather than failing for want of a dense one, and announces the
substitution on the warnings channel. ``EXCLUDE_AND_INTEGRATE`` drops the
body from its own force model and integrates the caller's state instead,
and says so on the same channel.

Both policies therefore produce rows, and the warning is what tells them
apart — the substituted rows are SPK samples of the body, not an
integration of the caller's initial condition, and they carry no
covariance. These pin that the distinction survives to Python, because
Python is where most callers are: the knob is a field on
``PropagationConfig``, it reaches the engine, and the run reports which
of the two answers it gave.

Also pinned here: the two sub-configs ephemeris generation cannot honour
are rejected as ``ValueError``, the same class the sibling
unsupported-``uncertainty_method`` rejection on this call uses and the
class ``generate_ephemeris``'s own docs promise.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    DiagnosticsConfig,
    EphemerisConfig,
    EphemerisOverlapPolicy,
    Epochs,
    EventConfig,
    Observers,
    PropagationConfig,
)

CERES_EPOCHS = Epochs.from_mjd([61000.5, 61010.5], scale="tdb")


@pytest.fixture(scope="module")
def ceres():
    empyrean.initialize()
    return empyrean.query_sbdb(["1"])


@pytest.fixture(scope="module")
def observers():
    return Observers.from_code("500", CERES_EPOCHS)


# ── The knob exists and is spelled the way the engine spells it ──


def test_the_policy_is_a_propagation_config_field_at_its_core_name() -> None:
    """API parity: the field carries the engine's own name, not a
    shortened one, so a reader of the core docs finds it here."""
    cfg = PropagationConfig()
    assert cfg.ephemeris_overlap_policy is EphemerisOverlapPolicy.SUBSTITUTE_SPK
    assert cfg._to_wire_dict()["ephemeris_overlap_policy"] == "substitute_spk"

    cfg.ephemeris_overlap_policy = EphemerisOverlapPolicy.EXCLUDE_AND_INTEGRATE
    assert cfg._to_wire_dict()["ephemeris_overlap_policy"] == "exclude_and_integrate"


def test_an_unknown_policy_is_refused_by_value(ceres, observers) -> None:
    cfg = PropagationConfig()
    cfg.ephemeris_overlap_policy = "sideways"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown ephemeris_overlap_policy"):
        empyrean.generate_ephemeris(ceres, observers, config=EphemerisConfig(propagation=cfg))


def _substituted_body(warning: str) -> str:
    """The body-name clause of an overlap-substitution warning.

    The engine formats it as ``orbit {orbit_id}: rows are SPK samples of
    {body_name}: ...``. Both names appear in one string, so a test that
    greps the whole message cannot tell which one it matched — and the
    body name is the load-bearing half here.
    """
    _head, _, rest = warning.partition("rows are SPK samples of ")
    assert rest, f"unrecognized substitution warning shape: {warning!r}"
    body, sep, _ = rest.partition(":")
    assert sep, f"substitution warning has no body-name clause: {warning!r}"
    return body


def _excluded_body(warning: str) -> str:
    """The body-name clause of an overlap-exclusion warning.

    Formatted as ``orbit {orbit_id}: rows integrate the supplied state
    with {body_name} removed from the force model (...)``. Same reason as
    :func:`_substituted_body` for isolating it.
    """
    _, _, rest = warning.partition("the supplied state with ")
    assert rest, f"unrecognized exclusion warning shape: {warning!r}"
    body, sep, _ = rest.partition(" removed from the force model")
    assert sep, f"exclusion warning has no body-name clause: {warning!r}"
    return body


# ── The case the knob exists for ──


def test_the_default_policy_serves_an_n16_ephemeris_and_says_it_substituted(
    ceres, observers
) -> None:
    """The default ``SUBSTITUTE_SPK`` produces rows — off the body's own
    SPK, the best available trajectory for an N16 body — and explains
    itself rather than substituting silently.

    This is deliberate 1.25.0 behavior (bd empyrean-c67i6), not a hidden
    fallback: the run names the body it sampled and says the caller's
    declared covariance did not survive the substitution, so rows built
    this way can never be mistaken for an integration of the caller's own
    initial condition.

    Mirrors villeneuve ``tests/test_c67i6_ephemeris_overlap_policy.rs::
    iris_ephemeris_works_under_exclude_and_integrate_and_explains_itself_under_substitute``.
    """
    result = empyrean.generate_ephemeris(ceres, observers)

    assert len(result.ephemeris) == len(CERES_EPOCHS)
    lon = result.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat = result.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))

    # A substitution transports no covariance: the rows carry the honest
    # all-NaN "absent" representation. A finite value here would be a sky
    # covariance the substituted trajectory never propagated.
    m = result.ephemeris.coordinates.covariance.to_matrix()
    assert np.isnan(m).all(), "an SPK substitution must carry no sky covariance"

    # The warning is the whole point — without it these rows are
    # indistinguishable from an integration of the supplied state.
    substitutions = [w for w in result.warnings if "SPK samples" in w]
    assert len(substitutions) == 1, (
        f"expected exactly one substitution warning, got {result.warnings}"
    )
    # Match the BODY-NAME clause specifically, not the whole message: the
    # message also carries the orbit id, and an orbit a caller happened to
    # name "Ceres" would satisfy a bare substring check even if the body
    # name went missing entirely.
    assert "Ceres" in _substituted_body(substitutions[0]), (
        f"the warning must name the body it sampled, got {substitutions[0]!r}"
    )
    # Ceres comes from SBDB carrying a covariance, so the drop is flagged.
    assert "covariance was NOT transported" in substitutions[0], (
        "the warning must say the declared covariance did not survive"
    )


def test_exclude_and_integrate_generates_an_n16_ephemeris(ceres, observers) -> None:
    cfg = EphemerisConfig(
        propagation=PropagationConfig(
            ephemeris_overlap_policy=EphemerisOverlapPolicy.EXCLUDE_AND_INTEGRATE
        )
    )
    result = empyrean.generate_ephemeris(ceres, observers, config=cfg)

    assert len(result.ephemeris) == len(CERES_EPOCHS)
    lon = result.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat = result.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
    # Two epochs ten days apart must not land on the same sky position;
    # a policy that silently did nothing would give identical rows or no
    # rows at all.
    assert lon[0] != lon[1]

    # This arm integrates, so the covariance rides through — the
    # measurable difference from the substituting arm above.
    m = result.ephemeris.coordinates.covariance.to_matrix()
    assert np.isfinite(m).any(), "the integrated arm must transport a sky covariance"

    # And it too reports which answer it gave: this force model ran one
    # perturber short, and a caller reading these σ needs to know that.
    exclusions = [w for w in result.warnings if "removed from the force model" in w]
    assert len(exclusions) == 1, f"expected exactly one exclusion warning, got {result.warnings}"
    assert "Ceres" in _excluded_body(exclusions[0]), (
        f"the warning must name the excluded perturber, got {exclusions[0]!r}"
    )


def test_the_two_policies_agree_on_the_sky_to_sub_arcsecond(ceres, observers) -> None:
    """Both answers are sane answers to different questions, and neither
    is allowed to wander: the SPK-sampled and integrated trajectories
    must land on the same sky to well inside an arcsecond.

    Mirrors the closing assertion of the villeneuve test above, which
    bounds the same gap at 5 arcsec for 7 Iris over a 30-day arc.
    """
    sub = empyrean.generate_ephemeris(ceres, observers)
    exc = empyrean.generate_ephemeris(
        ceres,
        observers,
        config=EphemerisConfig(
            propagation=PropagationConfig(
                ephemeris_overlap_policy=EphemerisOverlapPolicy.EXCLUDE_AND_INTEGRATE
            )
        ),
    )
    # Guard against a vacuous pass: an empty comparison satisfies every
    # bound below.
    assert len(sub.ephemeris) == len(CERES_EPOCHS)
    assert len(exc.ephemeris) == len(CERES_EPOCHS)

    lon_s = sub.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat_s = sub.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)
    lon_e = exc.ephemeris.coordinates.lon.to_numpy(zero_copy_only=False)
    lat_e = exc.ephemeris.coordinates.lat.to_numpy(zero_copy_only=False)

    dlon = np.abs(lon_s - lon_e) * 3600.0 * np.cos(np.radians(lat_e))
    dlat = np.abs(lat_s - lat_e) * 3600.0
    assert np.all(dlon < 1.0) and np.all(dlat < 1.0), (
        f'the two policies\' sky rows disagree by ({dlon}", {dlat}") — '
        "one of the trajectories is wrong"
    )


def test_the_exclusion_list_is_the_other_escape(ceres, observers) -> None:
    """Documented alongside the policy, so it is pinned alongside it.
    Ceres is NAIF 2000001."""
    cfg = EphemerisConfig(
        propagation=PropagationConfig(excluded_perturbers=[empyrean.Origin.asteroid(1)])
    )
    result = empyrean.generate_ephemeris(ceres, observers, config=cfg)
    assert len(result.ephemeris) == len(CERES_EPOCHS)


# ── Unsupported sub-configs are caller errors, not engine faults ──


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("events", EventConfig(dense_output=True)),
        ("events", EventConfig(impacts=False)),
        ("diagnostics", DiagnosticsConfig(lyapunov=True)),
    ],
)
def test_unsupported_sub_configs_raise_value_error(ceres, observers, field, value) -> None:
    """`events` / `diagnostics` have no home on the ephemeris path and no
    output channel on the result, so they are refused by name. They used
    to arrive as a RuntimeError from the FFI marshaling step — the class
    this codebase uses for engine faults — while the sibling
    unsupported-`uncertainty_method` rejection on the same call is
    documented as a ValueError."""
    cfg = EphemerisConfig(propagation=PropagationConfig(**{field: value}))
    with pytest.raises(ValueError) as excinfo:
        empyrean.generate_ephemeris(ceres, observers, config=cfg)
    assert field in str(excinfo.value)
