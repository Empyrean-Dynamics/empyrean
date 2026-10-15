"""Solar-radiation-pressure (SRP) force-slot plumbing tests.

SRP is a first-class, additive force slot carried on ``orbits.srp``
(:class:`empyrean.SRPParams`) — an area-to-mass ratio (the fittable AMRAT)
and a fixed radiation-pressure coefficient Cr. The AMRAT prior rides the
orbit's packed joint (its AMRAT axis), not an ``amrat_variance`` column.
It is NOT a NonGravParams model: the old in-band ``model="srp"``
tag silently applied the AMRAT as a radial *Marsden* acceleration, so any
orbit fit or propagated under it was wrong. These tests verify:

- the ``orbits.srp`` / ``SRPParams`` public surface and its marshal
  round-trip (orbits -> dict -> orbits and orbits -> parquet -> orbits),
- the loud ``model="srp"`` rejection at every orbit-marshaling entry point,
- per-row SRP field validation (amrat / cr),
- that a propagate accepts an orbit carrying an SRP slot.
"""

from __future__ import annotations

import os
import tempfile

import empyrean
import numpy as np
import pytest
from empyrean import (
    AssertedSource,
    CartesianOrbits,
    CovarianceKind,
    Epochs,
    JointFunctional,
    JointProvenance,
    JointQuality,
    NonGravParams,
    Origin,
    PackedJoint,
    PackedJoints,
    SRPParams,
    UncertaintyMethod,
    compute_b_planes,
    compute_impact_probabilities,
    generate_ephemeris,
)
from empyrean._convert import (
    extract_srp,
    orbit_batch_dict_to_orbits,
    orbits_to_orbit_batch_dict,
    validate_non_grav_marsden_only,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.io.orbits import read_orbits_parquet, write_orbits_parquet
from empyrean.observers.observers import Observers

# A benign heliocentric Cartesian state; SRP magnitude is irrelevant to the
# marshal-identity tests, only that the values round-trip.
_STATE = {"epoch": 60000.0, "x": 1.2, "y": -0.3, "z": 0.05, "vx": 0.004, "vy": 0.014, "vz": -0.001}


def _coords(n: int = 1) -> CartesianCoordinates:
    return CartesianCoordinates.from_kwargs(
        epoch=[_STATE["epoch"]] * n,
        x=[_STATE["x"] + 0.01 * i for i in range(n)],
        y=[_STATE["y"]] * n,
        z=[_STATE["z"]] * n,
        vx=[_STATE["vx"]] * n,
        vy=[_STATE["vy"]] * n,
        vz=[_STATE["vz"]] * n,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)] * n,
    )


def _orbit_with_srp(amrat: float = 3.0e-3, cr: float = 1.2) -> CartesianOrbits:
    return CartesianOrbits.from_kwargs(
        orbit_id=["srp_test"],
        object_id=["srp_obj"],
        coordinates=_coords(1),
        srp=SRPParams.from_kwargs(amrat=[amrat], cr=[cr]),
    )


def _orbit_with_srp_model() -> CartesianOrbits:
    """An orbit carrying the now-rejected in-band ``model='srp'`` non-grav."""
    return CartesianOrbits.from_kwargs(
        orbit_id=["bad_srp_model"],
        coordinates=_coords(1),
        non_grav=NonGravParams.from_kwargs(a1=[3.0e-3], a2=[0.0], a3=[0.0], model=["srp"]),
    )


# ── Public surface ───────────────────────────────────────────────────


def test_srp_params_public_api() -> None:
    """SRPParams is exported and carries amrat / cr. The AMRAT prior rides
    the orbit's packed joint (its AMRAT axis), not an amrat_variance column."""
    srp = SRPParams.from_kwargs(amrat=[3.0e-3], cr=[1.2])
    assert srp.amrat.to_pylist() == [3.0e-3]
    assert srp.cr.to_pylist() == [1.2]
    assert "amrat_variance" not in srp.table.column_names


def test_orbits_srp_attach_and_extract() -> None:
    """orbits.srp attaches, and extract_srp gates on the explicit switch."""
    orbits = _orbit_with_srp(amrat=2.5e-3, cr=1.0)
    has_srp, amrat, cr = extract_srp(orbits)
    assert has_srp.tolist() == [1]
    assert amrat.tolist() == [2.5e-3]
    assert cr.tolist() == [1.0]


def test_extract_srp_absent_is_switch_off() -> None:
    """An orbit with no srp sub-table marshals has_srp = 0 (no value-infer)."""
    orbits = CartesianOrbits.from_kwargs(orbit_id=["no_srp"], coordinates=_coords(1))
    has_srp, amrat, _cr = extract_srp(orbits)
    assert has_srp.tolist() == [0]
    assert amrat.tolist() == [0.0]


def test_non_grav_params_has_no_cr_column() -> None:
    """Cr moved to SRPParams — NonGravParams must not carry a cr column."""
    ng = NonGravParams.from_kwargs(a1=[0.0], a2=[0.0], a3=[0.0], model=["inverse_square"])
    assert "cr" not in ng.table.column_names


# ── Marshal round-trips (identity) ───────────────────────────────────


def test_srp_convert_roundtrip_fixed_force() -> None:
    """orbits.srp -> batch dict -> orbits is identity (fixed-force SRP)."""
    orbits = _orbit_with_srp(amrat=3.0e-3, cr=1.2)
    back = orbit_batch_dict_to_orbits(orbits_to_orbit_batch_dict(orbits))
    assert back.srp is not None
    assert back.srp.amrat.to_pylist() == [3.0e-3]
    assert back.srp.cr.to_pylist() == [1.2]


def test_srp_amrat_prior_rides_the_joint_amrat_axis() -> None:
    """The SRP AMRAT prior is carried on the orbit's packed joint — its
    AMRAT axis (bit 4 of the canonical axis table) — not an
    ``amrat_variance`` column, and it round-trips through the batch dict
    with the joint.

    The joint is a synthetic width-7 (6 state + AMRAT) Cartesian prior, so
    a bit-level round-trip assertion is admissible: no SPICE-backed value
    rides it and the batch dict carries ``tri`` verbatim.
    """
    # Block-diagonal 7x7: a tight state block + the AMRAT prior variance on
    # the single parameter axis (matrix index 6).
    width = 7
    tri: list[float] = []
    for i in range(width):
        for j in range(i + 1):
            if i == 6 and j == 6:
                tri.append(9.0e-6)  # AMRAT prior variance
            elif i == j:
                tri.append(1.0e-10)
            else:
                tri.append(0.0)
    joint = PackedJoint(
        layout_present=1 << 4,  # AMRAT axis present
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
        provenance=JointProvenance.ASSERTED,
        asserted_source=AssertedSource.EXTERNAL_FILE,  # caller-supplied prior
    )
    orbits = CartesianOrbits.from_kwargs(
        orbit_id=["srp_test"],
        object_id=["srp_obj"],
        coordinates=_coords(1),
        srp=SRPParams.from_kwargs(amrat=[1.5e-3], cr=[1.5]),
        covariance=PackedJoints.from_joints([joint]),
    )
    back = orbit_batch_dict_to_orbits(orbits_to_orbit_batch_dict(orbits))
    assert back.srp is not None
    assert back.srp.amrat.to_pylist() == [1.5e-3]
    j = back.covariance.joint(0)
    assert j is not None
    assert j.width == 7
    assert j.layout_present == (1 << 4)
    # The AMRAT prior variance is the last triangle entry (axis 6 diagonal).
    assert np.array_equal(
        np.asarray(j.tri, dtype=np.float64).view(np.uint64),
        np.asarray(tri, dtype=np.float64).view(np.uint64),
    )
    assert j.tri[-1] == pytest.approx(9.0e-6)


def test_srp_convert_roundtrip_mixed_batch() -> None:
    """A batch where only some rows carry an SRP slot round-trips: present
    rows keep their slot, absent rows stay absent (NaN-amrat sentinel)."""
    coords = _coords(2)
    srp = SRPParams.from_kwargs(amrat=[3.0e-3, np.nan], cr=[1.2, np.nan])
    orbits = CartesianOrbits.from_kwargs(orbit_id=["a", "b"], coordinates=coords, srp=srp)
    back = orbit_batch_dict_to_orbits(orbits_to_orbit_batch_dict(orbits))
    has_srp, amrat, cr = extract_srp(back)
    assert has_srp.tolist() == [1, 0]
    assert amrat[0] == pytest.approx(3.0e-3)
    assert cr[0] == pytest.approx(1.2)


def test_srp_parquet_roundtrip() -> None:
    """orbits.srp survives a parquet write/read round-trip through the FFI."""
    orbits = _orbit_with_srp(amrat=3.0e-3, cr=1.2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "orbits.parquet")
        write_orbits_parquet(path, orbits)
        back = read_orbits_parquet(path)
    assert back.srp is not None
    assert back.srp.amrat.to_pylist() == pytest.approx([3.0e-3])
    assert back.srp.cr.to_pylist() == pytest.approx([1.2])


def test_srp_parquet_roundtrip_absent_stays_absent() -> None:
    """An orbit with no SRP slot round-trips through parquet without one."""
    orbits = CartesianOrbits.from_kwargs(orbit_id=["no_srp"], coordinates=_coords(1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "orbits.parquet")
        write_orbits_parquet(path, orbits)
        back = read_orbits_parquet(path)
    # No SRP slot present -> extract reads has_srp = 0.
    has_srp, _, _ = extract_srp(back)
    assert has_srp.tolist() == [0]


# ── Field validation (loud) ──────────────────────────────────────────


@pytest.mark.parametrize(
    "amrat, cr, needle",
    [
        (0.0, 1.2, "amrat"),
        (-1.0e-3, 1.2, "amrat"),
        (np.inf, 1.2, "amrat"),
        (3.0e-3, 0.0, "cr"),
        (3.0e-3, -1.0, "cr"),
    ],
)
def test_srp_field_validation(amrat, cr, needle) -> None:
    orbits = _orbit_with_srp(amrat=amrat, cr=cr)
    with pytest.raises(ValueError, match=needle):
        extract_srp(orbits)


# ── model='srp' loud rejection at every marshaling entry point ────────


def test_model_srp_rejected_by_validator() -> None:
    with pytest.raises(ValueError, match="model='srp' is no longer supported"):
        validate_non_grav_marsden_only(_orbit_with_srp_model())


def test_unknown_non_grav_model_rejected() -> None:
    orbits = CartesianOrbits.from_kwargs(
        orbit_id=["bad_model"],
        coordinates=_coords(1),
        non_grav=NonGravParams.from_kwargs(a1=[0.0], a2=[0.0], a3=[0.0], model=["yarkovsky"]),
    )
    with pytest.raises(ValueError, match="unknown NonGravParams model"):
        validate_non_grav_marsden_only(orbits)


def test_model_srp_rejected_by_propagate() -> None:
    with pytest.raises(ValueError, match=r"orbits\.srp"):
        empyrean.propagate(
            _orbit_with_srp_model(), Epochs.from_mjd(np.array([60000.0, 60010.0]), scale="tdb")
        )


def test_model_srp_rejected_by_generate_ephemeris() -> None:
    observers = Observers.from_code("500", Epochs.from_mjd([60000.5], scale="tdb"))
    with pytest.raises(ValueError, match=r"orbits\.srp"):
        generate_ephemeris(_orbit_with_srp_model(), observers)


def test_model_srp_rejected_by_compute_impact_probabilities() -> None:
    with pytest.raises(ValueError, match=r"orbits\.srp"):
        compute_impact_probabilities(
            _orbit_with_srp_model(),
            end_epoch=Epochs.from_mjd([60100.0], scale="tdb"),
            methods=[UncertaintyMethod.FIRST_ORDER],
            body_filter=[Origin.EARTH],
        )


def test_model_srp_rejected_by_compute_b_planes() -> None:
    with pytest.raises(ValueError, match=r"orbits\.srp"):
        compute_b_planes(
            _orbit_with_srp_model(),
            end_epoch=Epochs.from_mjd([60100.0], scale="tdb"),
            methods=[UncertaintyMethod.FIRST_ORDER],
            body_filter=[Origin.EARTH],
        )


def test_model_srp_rejected_by_od_marshal() -> None:
    from empyrean.od.determine import _orbits_to_dict

    with pytest.raises(ValueError, match=r"orbits\.srp"):
        _orbits_to_dict(_orbit_with_srp_model())


def test_model_srp_rejected_by_io_write() -> None:
    with pytest.raises(ValueError, match=r"orbits\.srp"), tempfile.TemporaryDirectory() as d:
        write_orbits_parquet(os.path.join(d, "o.parquet"), _orbit_with_srp_model())


# ── Fitted-orbit re-feed (OD result -> orbits.srp) ───────────────────


def test_fitted_srp_reconstructed_from_result() -> None:
    """A solved-AMRAT OD result dict reconstructs orbits.srp on the fitted
    orbit (independent of the Marsden non-grav)."""
    from empyrean.od.determine import _build_cartesian_orbits_single

    # Minimal flat state snapshot (the `orbit_` prefix the Rust OD result
    # emits) plus the fitted SRP re-feed fields.
    result = {
        "orbit_x": _STATE["x"],
        "orbit_y": _STATE["y"],
        "orbit_z": _STATE["z"],
        "orbit_vx": _STATE["vx"],
        "orbit_vy": _STATE["vy"],
        "orbit_vz": _STATE["vz"],
        "orbit_epoch": _STATE["epoch"],
        "orbit_frame": 1,  # ecliptic_j2000
        "orbit_origin": 10,  # Sun NAIF id
        "orbit_orbit_id": "fitted",
        "orbit_srp_amrat": 2.7e-3,
        "orbit_srp_cr": 1.3,
    }
    orbits = _build_cartesian_orbits_single(result, prefix="orbit_")
    assert orbits.srp is not None
    assert orbits.srp.amrat.to_pylist() == pytest.approx([2.7e-3])
    assert orbits.srp.cr.to_pylist() == pytest.approx([1.3])


# ── Propagate accepts an SRP-bearing orbit ───────────────────────────


def test_propagate_accepts_srp_orbit() -> None:
    """A fixed-force SRP orbit propagates without error (SRP is additive)."""
    orbits = _orbit_with_srp(amrat=3.0e-3, cr=1.2)
    result = empyrean.propagate(orbits, Epochs.from_mjd(np.array([60000.0, 60030.0]), scale="tdb"))
    assert len(result.states) == 2
