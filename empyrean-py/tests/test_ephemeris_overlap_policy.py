"""The ephemeris-overlap policy at the Python channel.

An SB441-N16 body (1 Ceres, 2 Pallas, 4 Vesta, …) is simultaneously a
member of the Standard force model and a legitimate object to propagate,
so its orbit sits on top of the ephemeris the force model is reading.
Under the default ``SUBSTITUTE_SPK`` the engine returns the body's own
SPK states and skips integration, which leaves no dense trajectory for
ephemeris generation to read — the call fails.

The escape exists at the C ABI and in the Rust wrapper. These pin that it
also exists here, because Python is where most callers are: the knob is a
field on ``PropagationConfig``, it reaches the engine, and it turns a
failing Ceres ephemeris into a succeeding one.

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


# ── The case the knob exists for ──


def test_the_default_policy_cannot_generate_an_n16_ephemeris(ceres, observers) -> None:
    """The failure is loud and names the overlap — it is not a silent
    substitution of SPK samples for the caller's own orbit."""
    with pytest.raises(RuntimeError) as excinfo:
        empyrean.generate_ephemeris(ceres, observers)
    assert "overlap" in str(excinfo.value).lower()


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
