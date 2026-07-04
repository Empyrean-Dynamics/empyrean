"""Tests for compute_impact_probabilities / compute_b_planes input handling."""

import empyrean
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    NonGravParams,
    Origin,
    UncertaintyMethod,
    compute_impact_probabilities,
)


def _make_test_orbit(*, with_non_grav: bool) -> CartesianOrbits:
    """Build a single-orbit batch with or without non-gravitational params.

    Heliocentric, near-1 AU, near-circular — won't trigger Earth events
    in the short propagation window the tests use, but still exercises
    the input-marshalling code path.
    """
    coords = CartesianCoordinates.from_kwargs(
        epoch=[60000.0],
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017],
        vz=[0.0],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    kwargs = {"orbit_id": ["test"], "coordinates": coords}
    if with_non_grav:
        kwargs["non_grav"] = NonGravParams.from_kwargs(
            a1=[1.0e-9],
            a2=[0.0],
            a3=[0.0],
            model=["inverse_square"],
        )
    return CartesianOrbits.from_kwargs(**kwargs)


def test_compute_impact_probabilities_without_non_grav():
    """Regression: all-null non-grav sub-table must not break IP computation.

    `from_kwargs(orbit_id=[...], coordinates=...)` produces an
    `orbits.non_grav` value that is not Python `None` but holds an
    all-null sub-table. The earlier `if orbits.non_grav is not None`
    guard followed by `.to_numpy()` (default `zero_copy_only=True`)
    raised `pyarrow.lib.ArrowInvalid: Needed to copy 1 chunks with 1
    nulls, but zero_copy_only was True`. The fixed helper reads with
    `zero_copy_only=False` and lets `nan_to_num` normalize.
    """
    empyrean.initialize()
    orbits = _make_test_orbit(with_non_grav=False)
    # Plain call — no body filter, no methods of substance — just
    # exercise `_common_orbit_args` end-to-end. Should not raise.
    ips = compute_impact_probabilities(
        orbits,
        end_epoch=60010.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    # Heliocentric near-1-AU orbit with a 10-day window won't have
    # any Earth events; what matters is that the call succeeded.
    assert isinstance(ips, type(ips))


def test_compute_impact_probabilities_with_non_grav_present():
    """Same call shape, but with non_grav populated — confirms the
    happy path still works after the fix."""
    empyrean.initialize()
    orbits = _make_test_orbit(with_non_grav=True)
    ips = compute_impact_probabilities(
        orbits,
        end_epoch=60010.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    assert isinstance(ips, type(ips))
