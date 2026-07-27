"""Regression tests for orbit_id / object_id plumbing.

The empyrean-py wrapper accepts user-supplied `orbit_id` and `object_id`
strings on the input batch and is expected to thread them through
*every* output channel — propagated states, events, ephemerides, and
OD residuals. Historically the C ABI (`empyrean-c/src/propagate.rs:512`)
fabricates a synthetic `format!("orbit_{i}")` because the underlying
`EmpyreanOrbit` struct has no orbit_id field, and the Rust binding
must reverse-engineer the original IDs from those positional indices.

These tests pin that recovery path so a regression — e.g. event
`orbit_id` defaulting back to the synthetic "orbit_0", or `object_id`
silently going `NaN` — fails loudly.

Caught by user 2026-05-01 (Apophis SBDB → propagate → events showed
`orbit_id="orbit_0"` and `object_id=NaN` instead of the user's
"99942 Apophis (2004 MN4)#220" / "99942 Apophis (2004 MN4)").
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    Origin,
    UncertaintyMethod,
    compute_b_planes,
    compute_impact_probabilities,
    generate_ephemeris,
)
from empyrean.observers.observers import Observers

# ── Helpers ──────────────────────────────────────────────────────────


# Apophis state from SBDB at MJD 61000 TDB (heliocentric ecliptic J2000),
# pulled from `tests/data/cartesian_sun_ecliptic.csv`. Deterministically
# triggers Earth close approaches on year-scale propagation (notably the
# well-known 2029 flyby at MJD ~62239), so the events table is non-empty
# and the ID-plumbing assertions actually run.
APOPHIS_STATE = {
    "epoch": 61000.0,
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}
# Bennu state at the same epoch (also ECLIPJ2000 heliocentric). Bennu's
# 2135 Earth approach is far past our test window, but the state-vector
# tagging path (multi-orbit batch → events table) runs on Apophis alone;
# Bennu rides along to confirm orbit_id ↔ object_id pairing isn't swapped.
BENNU_STATE = {
    "epoch": 61000.0,
    "x": -1.06185,
    "y": 0.36085,
    "z": 0.04553,
    "vx": -0.00578,
    "vy": -0.01413,
    "vz": -0.00103,
}


def _earth_crosser_orbit(orbit_id: str, object_id: str) -> CartesianOrbits:
    """Build an Apophis-state orbit that fires Earth events on multi-year
    propagation. Tagged with the user-supplied orbit_id / object_id."""
    s = APOPHIS_STATE
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[object_id],
        coordinates=coords,
    )


# ── Tests ────────────────────────────────────────────────────────────


def test_states_carry_user_orbit_id_and_object_id() -> None:
    """Propagated states must echo the user's orbit_id and object_id.

    This already worked before the events fix — guarding so a future
    refactor of the index-based recovery in `_propagate` doesn't silently
    drop it.
    """
    orbits = _earth_crosser_orbit(orbit_id="MY_ORBIT_TAG", object_id="my_object_42")
    target_epochs = np.array([61000.0, 61010.0, 61020.0])
    result = empyrean.propagate(orbits, target_epochs)

    states_orbit_ids = result.states.orbit_id.to_pylist()
    states_object_ids = result.states.object_id.to_pylist()

    assert all(oid == "MY_ORBIT_TAG" for oid in states_orbit_ids), (
        f"states orbit_ids should all be 'MY_ORBIT_TAG'; got {states_orbit_ids}"
    )
    assert all(oid == "my_object_42" for oid in states_object_ids), (
        f"states object_ids should all be 'my_object_42'; got {states_object_ids}"
    )


def test_events_carry_user_orbit_id_and_object_id() -> None:
    """Events must carry the user's orbit_id and object_id, not the
    fabricated 'orbit_0' from the C ABI.

    This is the regression the user caught with the Apophis SBDB query:
    `result.events.summary.to_dataframe()` showed `orbit_id="orbit_0"`
    and `object_id=NaN` instead of the strings from the input orbits.
    """
    orbits = _earth_crosser_orbit(orbit_id="MY_ORBIT_TAG", object_id="my_object_42")
    # Multi-year window covers Apophis's 2029 Earth flyby, guaranteeing
    # at least one event in the result.
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(orbits, target_epochs)

    summary_df = result.events.summary.to_dataframe()
    if summary_df.empty:
        pytest.skip(
            "No events fired for synthetic orbit; ID-plumbing path not exercised. "
            "Adjust _earth_crosser_orbit to ensure events."
        )

    bad_orbit_ids = summary_df["orbit_id"][summary_df["orbit_id"] != "MY_ORBIT_TAG"]
    assert bad_orbit_ids.empty, (
        f"events orbit_id must be 'MY_ORBIT_TAG' for every row; "
        f"got {bad_orbit_ids.tolist()}. The C ABI's fabricated 'orbit_N' is "
        f"leaking through — check parse_fabricated_orbit_index in lib.rs."
    )

    # Pandas turns empty Python strings into NaN through the arrow→pandas
    # bridge — check both forms to defend against that.
    bad_object_ids = summary_df["object_id"][
        (summary_df["object_id"] != "my_object_42") & summary_df["object_id"].notna()
    ]
    nan_object_ids = summary_df["object_id"][summary_df["object_id"].isna()]
    assert bad_object_ids.empty and nan_object_ids.empty, (
        f"events object_id must be 'my_object_42' for every row; "
        f"got {len(nan_object_ids)} NaN and {bad_object_ids.tolist()} mismatches."
    )


def test_multiple_orbits_each_keep_their_own_id() -> None:
    """With multiple orbits in one batch, each event must carry the
    correct orbit_id / object_id for the orbit that produced it — not
    a shared default and not a swap with another orbit's ID.
    """
    a, b = APOPHIS_STATE, BENNU_STATE
    coords = CartesianCoordinates.from_kwargs(
        epoch=[a["epoch"], b["epoch"]],
        x=[a["x"], b["x"]],
        y=[a["y"], b["y"]],
        z=[a["z"], b["z"]],
        vx=[a["vx"], b["vx"]],
        vy=[a["vy"], b["vy"]],
        vz=[a["vz"], b["vz"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN), str(Origin.SUN)],
    )
    orbits = CartesianOrbits.from_kwargs(
        orbit_id=["FIRST", "SECOND"],
        object_id=["first_obj", "second_obj"],
        coordinates=coords,
    )
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(orbits, target_epochs)

    df = result.events.summary.to_dataframe()
    if df.empty:
        pytest.skip(
            "No events fired for synthetic two-orbit batch; ID-plumbing path not exercised."
        )

    # Every orbit_id must be one of the user's two strings.
    valid_orbit_ids = {"FIRST", "SECOND"}
    invalid = df["orbit_id"][~df["orbit_id"].isin(valid_orbit_ids)]
    assert invalid.empty, (
        f"events orbit_id must be 'FIRST' or 'SECOND'; got synthetic IDs: "
        f"{invalid.unique().tolist()}"
    )

    # And the orbit_id ↔ object_id pairing must be intact (no swap).
    for orbit_id, expected_object_id in [
        ("FIRST", "first_obj"),
        ("SECOND", "second_obj"),
    ]:
        rows = df[df["orbit_id"] == orbit_id]
        if rows.empty:
            continue
        bad = rows["object_id"][rows["object_id"] != expected_object_id]
        assert bad.empty, (
            f"orbit_id '{orbit_id}' rows must all have object_id "
            f"'{expected_object_id}'; got {bad.unique().tolist()}"
        )


def test_ephemeris_carries_user_orbit_id_and_object_id() -> None:
    """Generated ephemerides (RA/Dec) must carry the user's orbit_id and
    object_id — not the C ABI's fabricated 'orbit_N' and not empty
    object_id (which the binding previously hardcoded to ``""``).
    """
    orbits = _earth_crosser_orbit(orbit_id="EPH_TAG", object_id="eph_obj")
    # Geocentric observer at the orbit's epoch + a few days.
    observers = Observers.from_code("500", [61000.5, 61010.5, 61020.5])
    result = generate_ephemeris(orbits, observers)
    eph = result.ephemeris
    assert len(eph) > 0, "ephemeris result is empty"

    orbit_ids = eph.orbit_id.to_pylist()
    object_ids = eph.object_id.to_pylist()
    assert all(oid == "EPH_TAG" for oid in orbit_ids), (
        f"ephemeris orbit_id must be 'EPH_TAG' for every row; got "
        f"{set(orbit_ids)}. The C ABI's fabricated 'orbit_N' is leaking through."
    )
    assert all(oid == "eph_obj" for oid in object_ids), (
        f"ephemeris object_id must be 'eph_obj' for every row; got {set(object_ids)}."
    )


def _orbit_with_covariance(orbit_id: str, object_id: str) -> CartesianOrbits:
    """Apophis-state orbit with a small diagonal covariance — required
    for the impact / B-plane channels which need uncertainty to compute
    a probability."""
    from empyrean.coordinates.covariance import CartesianCovariance

    s = APOPHIS_STATE
    # Diagonal covariance: ~1e-8 AU pos sigma, 1e-10 AU/d vel sigma.
    cov_matrix = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
    covariance = CartesianCovariance.from_matrix(cov_matrix)
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        covariance=covariance,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=[orbit_id],
        object_id=[object_id],
        coordinates=coords,
    )


def test_impact_probabilities_carry_user_orbit_id_and_object_id() -> None:
    """`compute_impact_probabilities` output must carry the user's
    orbit_id and object_id on every row.
    """
    orbits = _orbit_with_covariance(orbit_id="IP_TAG", object_id="ip_obj")
    ips = compute_impact_probabilities(
        orbits,
        end_epoch=63000.0,  # past Apophis 2029 flyby
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    df = ips.to_dataframe()
    if df.empty:
        pytest.skip("compute_impact_probabilities returned no rows")

    bad = df["orbit_id"][df["orbit_id"] != "IP_TAG"]
    assert bad.empty, f"impact_probabilities orbit_id must be 'IP_TAG'; got {bad.unique().tolist()}"
    bad_obj = df["object_id"][(df["object_id"] != "ip_obj") & df["object_id"].notna()]
    nan_obj = df["object_id"][df["object_id"].isna()]
    assert bad_obj.empty and nan_obj.empty, (
        f"impact_probabilities object_id must be 'ip_obj'; got "
        f"{len(nan_obj)} NaN and {bad_obj.unique().tolist()} mismatches."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BPlanes schema has no orbit_id / object_id columns yet, and the "
        "underlying EmpyreanBPlane C struct (empyrean-c/src/impact.rs:75-94) "
        "doesn't carry them either. A known gap — when those "
        "fields are wired through, this test should start passing and the "
        "xfail marker can be removed."
    ),
)
def test_b_planes_carry_user_orbit_id_and_object_id() -> None:
    """`compute_b_planes` output must carry the user's orbit_id and
    object_id on every row. Currently the channel literally does not
    expose these — the C ABI struct lacks the fields. The test is xfail
    until the fix lands; remove the marker when it does.
    """
    orbits = _orbit_with_covariance(orbit_id="BP_TAG", object_id="bp_obj")
    bps = compute_b_planes(
        orbits,
        end_epoch=63000.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    # The schema check is what currently fails — accessing a column that
    # doesn't exist in the table.
    orbit_ids = bps.orbit_id.to_pylist()  # AttributeError until fix
    object_ids = bps.object_id.to_pylist()
    assert all(oid == "BP_TAG" for oid in orbit_ids)
    assert all(oid == "bp_obj" for oid in object_ids)


def test_orbit_io_roundtrip_preserves_ids() -> None:
    """Writing orbits to disk (parquet/JSON/CSV) and reading them back
    must preserve orbit_id and object_id strings exactly. Catches any
    encoding round-trip that silently rewrites the IDs.
    """
    orbits = _earth_crosser_orbit(
        orbit_id="IO_TAG_99942 (special chars: !@#)",
        object_id="io_obj_(2004 MN4)",
    )

    with tempfile.TemporaryDirectory() as tmp:
        for ext, write_fn, read_fn in [
            (
                "parquet",
                empyrean.io.write_orbits_parquet,
                empyrean.io.read_orbits_parquet,
            ),
            ("json", empyrean.io.write_orbits_json, empyrean.io.read_orbits_json),
            ("csv", empyrean.io.write_orbits_csv, empyrean.io.read_orbits_csv),
        ]:
            path = Path(tmp) / f"orbits.{ext}"
            write_fn(str(path), orbits)
            roundtrip = read_fn(str(path))

            ids_in = orbits.orbit_id.to_pylist()
            ids_out = roundtrip.orbit_id.to_pylist()
            assert ids_in == ids_out, (
                f"{ext}: orbit_id round-trip mismatch: in={ids_in}, out={ids_out}"
            )

            obj_in = orbits.object_id.to_pylist()
            obj_out = roundtrip.object_id.to_pylist()
            assert obj_in == obj_out, (
                f"{ext}: object_id round-trip mismatch: in={obj_in}, out={obj_out}"
            )
