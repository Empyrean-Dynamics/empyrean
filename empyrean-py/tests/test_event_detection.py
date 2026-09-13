"""Switching event detection off is honoured at the Python layer.

`EventConfig`'s five per-type flags filter what is *emitted*. They do not
stop the detectors running: the C ABI used to drop
``detection_enabled`` entirely, so a caller who turned every detector off
still paid per-substep detection on every accepted integrator step and
had no way to say otherwise.

Pinned here: the switch reaches the engine (no events come back), it
moves no propagated number, and the two enum fields that were dropped
alongside it reach the engine too — including refusing a value they do
not recognise instead of falling back to the default.
"""

from __future__ import annotations

import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    Epochs,
    EventConfig,
    Origin,
    PropagationConfig,
    propagate,
)

# Apophis at MJD 61000 TDB, heliocentric ecliptic J2000. Just over one
# orbital period of arc, so the detectors have something to find.
APOPHIS = (
    -7.85264914906904643e-02,
    -8.19748051902064567e-01,
    4.18939515323390882e-02,
    1.98751024968884596e-02,
    1.32208844536140196e-03,
    3.99496044422352188e-04,
)
T0 = 61000.0
T1 = 61400.0
N_ORBITS = 8


def _orbits() -> CartesianOrbits:
    """A covariance-free batch — the shape a caller uses when it wants
    states and nothing else."""
    xs = [APOPHIS[0] + i * 1.0e-7 for i in range(N_ORBITS)]
    coords = CartesianCoordinates.from_kwargs(
        epoch=[T0] * N_ORBITS,
        x=xs,
        y=[APOPHIS[1]] * N_ORBITS,
        z=[APOPHIS[2]] * N_ORBITS,
        vx=[APOPHIS[3]] * N_ORBITS,
        vy=[APOPHIS[4]] * N_ORBITS,
        vz=[APOPHIS[5]] * N_ORBITS,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)] * N_ORBITS,
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=[f"o{i}" for i in range(N_ORBITS)],
        object_id=[f"obj-{i}" for i in range(N_ORBITS)],
        coordinates=coords,
    )


def _epochs() -> Epochs:
    return Epochs.from_mjd(np.array([T1 - 30.0, T1]), scale="tdb")


def _config(detection_enabled: bool) -> PropagationConfig:
    return PropagationConfig(
        events=EventConfig(detection_enabled=detection_enabled),
        num_threads=1,
    )


def test_detection_off_yields_no_events_and_on_yields_some() -> None:
    """The headline, with its own positive control: detection on must
    produce events, or the off-case proves only that the batch is dull."""
    on = propagate(_orbits(), _epochs(), config=_config(True))
    assert len(on.events.summary) > 0, "the fixture must produce events with detection on"

    off = propagate(_orbits(), _epochs(), config=_config(False))
    assert len(off.events.summary) == 0, (
        f"detection off must yield no events, got {off.events.count_by_type()}"
    )


def test_detection_off_changes_no_propagated_number() -> None:
    """Detection is observation, not dynamics — switching it off is a
    free performance knob, not a fidelity trade."""
    on = propagate(_orbits(), _epochs(), config=_config(True))
    off = propagate(_orbits(), _epochs(), config=_config(False))

    for axis in ("x", "y", "z", "vx", "vy", "vz"):
        np.testing.assert_array_equal(
            getattr(on.states.coordinates, axis).to_numpy(zero_copy_only=False),
            getattr(off.states.coordinates, axis).to_numpy(zero_copy_only=False),
            err_msg=f"{axis} moved when detection was switched off",
        )


def test_the_other_two_dropped_fields_reach_the_engine() -> None:
    """`dense_origin` and `capture_criterion` were dropped by the same
    marshalling gap. Every documented value must be accepted."""
    for origin in ("bodycentric", "barycentric"):
        for criterion in ("population", "individual", "energy_only"):
            config = PropagationConfig(
                events=EventConfig(dense_origin=origin, capture_criterion=criterion),
                num_threads=1,
            )
            result = propagate(_orbits(), _epochs(), config=config)
            assert len(result.states) == N_ORBITS * 2


@pytest.mark.parametrize(
    ("field", "value"),
    [("dense_origin", "heliocentric"), ("capture_criterion", "granvik")],
)
def test_an_unrecognised_value_is_refused(field: str, value: str) -> None:
    """The positive control for the test above. A config string that is
    silently ignored is worse than one that is rejected: the caller reads
    results produced under a setting it did not ask for."""
    config = PropagationConfig(events=EventConfig(**{field: value}), num_threads=1)
    with pytest.raises(ValueError, match=value):
        propagate(_orbits(), _epochs(), config=config)


@pytest.mark.parametrize(
    ("method", "named"),
    [("auto", "Auto"), ("gaussian_mixture", "GaussianMixture")],
)
def test_a_detection_driven_method_under_detection_off_is_refused(method: str, named: str) -> None:
    """Auto resolves its refinement windows from detected close approaches
    and the mixture splits at them, so with detection off neither fails --
    they quietly return a linear covariance and no impact probabilities.
    Refused by name instead."""
    config = PropagationConfig(
        uncertainty_method=method,
        events=EventConfig(detection_enabled=False),
        num_threads=1,
    )
    with pytest.raises(Exception) as excinfo:
        propagate(_orbits(), _epochs(), config=config)

    message = str(excinfo.value)
    assert "detection_enabled = false" in message, message
    assert f"uncertainty_method = {named}" in message, message
    assert "close approaches" in message, message


def test_every_other_pairing_is_served() -> None:
    """The positive control. A blanket refusal would pass the test above
    and fail this one: detection off is legal for every method that does
    not read detector output, and both refused methods are legal with
    detection on."""
    for method in ("first_order", "second_order"):
        config = PropagationConfig(
            uncertainty_method=method,
            events=EventConfig(detection_enabled=False),
            num_threads=1,
        )
        result = propagate(_orbits(), _epochs(), config=config)
        assert len(result.states) == N_ORBITS * 2

    for method in ("auto", "gaussian_mixture"):
        config = PropagationConfig(uncertainty_method=method, num_threads=1)
        result = propagate(_orbits(), _epochs(), config=config)
        assert len(result.states) == N_ORBITS * 2
