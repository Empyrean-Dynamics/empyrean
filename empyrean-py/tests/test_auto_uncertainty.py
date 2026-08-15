"""AUTO reachable and tunable from empyrean-py.

A frozen :class:`Auto` dataclass exposes AUTO's caller-tunable κ band edges
+ AGM knobs, mirrors the wrapper / C-ABI struct field-for-field (post
``threshold_second`` removal), maps to tag 4, and — the point of these
tests — its thresholds actually reach the engine instead of silently
collapsing to ``auto()`` defaults.
"""

from __future__ import annotations

import dataclasses

import empyrean
import numpy as np
import pytest
from empyrean import Auto, Epochs
from empyrean.coordinates.enums import Origin
from empyrean.propagation.config import (
    _DATACLASS_TO_INT,
    _UNCERTAINTY_PARAM_DEFAULTS,
    PropagationConfig,
    UncertaintyMethod,
    _uncertainty_method_params,
)
from empyrean.propagation.events import EventConfig

# The exact post-removal field set the current wrapper / C-ABI struct marshals
# (empyrean::UncertaintyMethod::Auto): threshold_first, threshold_mixture,
# threshold_ip_skip, gmm_max_depth, gmm_components_per_split. No threshold_second.
_WRAPPER_AUTO_DEFAULTS = {
    "threshold_first": 0.1,
    "threshold_mixture": 10.0,
    "threshold_ip_skip": 1e-12,
    "gmm_max_depth": 3,
    "gmm_components_per_split": 3,
}


# ── Pure-Python surface ──────────────────────────────────────────────


def test_auto_is_a_frozen_dataclass_mirroring_the_wrapper() -> None:
    """PROVING (``Auto`` did not exist pre-fix).

    Field names + defaults match the wrapper struct exactly, and no invented
    field (notably no ``threshold_second``, which the engine removed) sneaks in.
    """
    fields = {f.name: f.default for f in dataclasses.fields(Auto)}
    assert fields == _WRAPPER_AUTO_DEFAULTS
    a = Auto()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.threshold_first = 0.5  # type: ignore[misc]


def test_auto_maps_to_tag_four() -> None:
    """PROVING: the int mapping places Auto at the engine's AUTO tag (4)."""
    assert _DATACLASS_TO_INT[Auto] == 4


def test_default_auto_lowers_to_engine_defaults() -> None:
    """REGRESSION GUARD: ``Auto()`` is a no-op — same flat slots as ``"auto"``."""
    assert _uncertainty_method_params(Auto()) == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params(UncertaintyMethod.AUTO) == _UNCERTAINTY_PARAM_DEFAULTS
    assert _uncertainty_method_params("auto") == _UNCERTAINTY_PARAM_DEFAULTS


def test_parameterized_auto_lowers_each_knob() -> None:
    """PROVING: a tuned ``Auto`` lowers every knob (pre-fix had no auto slots)."""
    p = _uncertainty_method_params(
        Auto(
            threshold_first=0.5,
            threshold_mixture=7.0,
            threshold_ip_skip=1e-9,
            gmm_max_depth=4,
            gmm_components_per_split=5,
        )
    )
    assert p.auto_threshold_first == 0.5
    assert p.auto_threshold_mixture == 7.0
    assert p.auto_threshold_ip_skip == 1e-9
    assert p.auto_gmm_max_depth == 4
    assert p.auto_gmm_components_per_split == 5


def test_auto_serializes_to_the_auto_wire_string() -> None:
    """The wire dict names the method ``auto``; the knobs ride the flat args."""
    cfg = PropagationConfig(uncertainty_method=Auto(threshold_first=0.3))
    assert cfg._to_wire_dict()["uncertainty_method"] == "auto"


# ── Behavioural: the thresholds reach the engine ─────────────────────


@pytest.fixture(scope="module")
def _apophis():
    """Apophis + the loose covariance that drives AUTO to escalate at the 2029
    flyby (the villeneuve reference fixture, reused from ``test_no_silent_drops``)."""
    empyrean.initialize()
    from test_no_silent_drops import _auto_escalation_orbit

    return _auto_escalation_orbit()


def test_auto_threshold_first_reaches_the_engine(_apophis) -> None:
    """PROVING (pre-fix every ``Auto`` collapsed to ``auto()`` defaults).

    Default ``Auto()`` (threshold_first = 0.1) escalates Linear -> SecondOrder
    over the 2029 flyby and emits covariance-regime-change rows. Pinning
    ``threshold_first`` above the flyby κ keeps every window First, so no
    regime change fires — observable only if the tuned threshold actually
    reached the engine rather than being replaced by the 0.1 default.
    """
    t_ca = 62240.0  # ~2029-04-13 Earth flyby
    epochs = Epochs.from_mjd(
        np.array([t_ca - 30.0, t_ca - 5.0, t_ca, t_ca + 5.0, t_ca + 30.0]), scale="tdb"
    )
    events = EventConfig(body_filter=[Origin.EARTH])

    default = empyrean.propagate(_apophis, epochs, uncertainty_method=Auto(), events=events)
    assert len(default.events.covariance_regime_changes) > 0, (
        "default Auto did not escalate — fixture no longer drives regime changes"
    )

    # A κ band pinned far above the flyby nonlinearity (valid band:
    # threshold_first < threshold_mixture) keeps every window First.
    pinned = empyrean.propagate(
        _apophis,
        epochs,
        uncertainty_method=Auto(threshold_first=1.0e6, threshold_mixture=1.0e7),
        events=events,
    )
    assert len(pinned.events.covariance_regime_changes) == 0, (
        "threshold_first was ignored: a band pinned above the flyby κ still "
        "escalated, so the tuned Auto threshold did not reach the engine"
    )
