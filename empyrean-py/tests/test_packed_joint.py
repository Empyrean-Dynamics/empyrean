"""The packed joint covariance — the one covariance home.

Mirrors the three ruling tests the C and wrapper layers pin, plus the
packed-joint surface this package exposes: the one-home authority, its
derived flat 6x6, the retired third-order discriminant refused by name,
the removed four-homes fields, and the ``transform_orbits`` covariance
transport (with the state-level transform refusing a covariance-bearing
input).
"""

import empyrean
import numpy as np
import pytest
from empyrean import (
    AssertedSource,
    CartesianOrbits,
    CovarianceKind,
    JointFunctional,
    JointProvenance,
    JointQuality,
    PackedJoint,
    PackedJoints,
)
from empyrean.coordinates.coordinates import CartesianCoordinates, KeplerianCoordinates
from empyrean.coordinates.transform import transform_coordinates, transform_orbits
from empyrean.errors import PushError, PushErrorVariant


def _spd_tri(width: int) -> np.ndarray:
    """Packed lower triangle of an SPD ``width x width`` matrix.

    A diagonally-dominant matrix built from a fixed seed, so the triangle
    is admissible and reproducible. Synthetic — not SPICE-backed — so a
    bit-identical comparison is a valid platform-independent assertion.
    """
    rng = np.random.default_rng(1234)
    a = rng.standard_normal((width, width))
    mat = a @ a.T + width * np.eye(width)
    tri = []
    for i in range(width):
        for j in range(i + 1):
            tri.append(mat[i, j])
    return np.asarray(tri, dtype=np.float64)


def _joint(width: int, kind: CovarianceKind = CovarianceKind.LINEAR) -> PackedJoint:
    # width = 6 + popcount(layout_present): the bitset masks the parameter
    # axes beyond the six state axes (see the C ABI's EmpyreanPackedJoint).
    present = (1 << (width - 6)) - 1
    return PackedJoint(
        layout_present=present,
        layout_considered=0,
        layout_marginalized=0,
        width=width,
        tri=_spd_tri(width),
        kind=kind,
        quality=JointQuality.POSITIVE_DEFINITE,
        cov_rank=0,
        cov_null_axes=0,
        cov_min_eig_corr=float("nan"),
        functional=JointFunctional.STATE,
        provenance=JointProvenance.DERIVED,
        asserted_source=None,
    )


def test_width9_marsden_joint_round_trips_bit_identical() -> None:
    """Ruling (c): a width-9 Marsden joint round-trips bit-identical push →
    read-back.

    The joint is synthetic Cartesian (6 state + 3 Marsden), so a
    ``tri.view(np.uint64)`` bit-equality is admissible — no SPICE-backed
    value rides it, and the dict wire the extension speaks is a verbatim
    carry of ``float64``. Exercises the dict wire
    (:meth:`PackedJoint.to_wire` / :meth:`PackedJoint.from_wire`) and the
    quivr column (:meth:`PackedJoints.from_joints` / :meth:`PackedJoints.joint`).
    """
    j = _joint(9)

    # Dict wire: to_wire → from_wire.
    back = PackedJoint.from_wire(j.to_wire())
    assert np.array_equal(
        np.asarray(j.tri, dtype=np.float64).view(np.uint64),
        np.asarray(back.tri, dtype=np.float64).view(np.uint64),
    )
    assert back.width == 9
    assert back.kind is CovarianceKind.LINEAR

    # Quivr column: from_joints → joint, a None row stays None.
    col = PackedJoints.from_joints([j, None])
    assert col.row_is_empty(1)
    rt = col.joint(0)
    assert rt is not None
    assert np.array_equal(
        np.asarray(j.tri, dtype=np.float64).view(np.uint64),
        np.asarray(rt.tri, dtype=np.float64).view(np.uint64),
    )


def test_asserted_joint_carries_its_source_round_trip() -> None:
    """An asserted (SBDB-sourced) joint round-trips with BOTH its provenance
    and its asserted source, and a provenance/source mismatch is refused.

    The joint carries its asserted source at every layer: an SBDB orbit
    re-fed to the engine keeps ``provenance=ASSERTED`` AND
    ``asserted_source=SBDB`` rather than being silently re-tagged or
    refused. The mismatches (an asserted joint with no source, a source on
    a derived joint, an unknown source code) are refused by name, never
    defaulted.
    """
    j = _joint(6)
    j.provenance = JointProvenance.ASSERTED
    j.asserted_source = AssertedSource.SBDB

    # Dict wire: the source code rides the wire, not the none sentinel.
    wire = j.to_wire()
    assert wire["asserted_source"] == 0  # SBDB
    back = PackedJoint.from_wire(wire)
    assert back.provenance is JointProvenance.ASSERTED
    assert back.asserted_source is AssertedSource.SBDB

    # Quivr column carries the source too.
    col = PackedJoints.from_joints([j])
    rt = col.joint(0)
    assert rt is not None
    assert rt.asserted_source is AssertedSource.SBDB

    # Mismatch (a): an asserted joint with no source is refused.
    bad = _joint(6)
    bad.provenance = JointProvenance.ASSERTED
    bad.asserted_source = None
    with pytest.raises(ValueError, match="asserted"):
        bad.to_wire()

    # Mismatch (b): a source on a derived joint is refused.
    bad = _joint(6)
    bad.asserted_source = AssertedSource.SBDB
    with pytest.raises(ValueError, match="derived"):
        bad.to_wire()

    # Mismatch (c): an unknown source code on an asserted joint is refused by
    # name as PushError(UnknownAssertedSource).
    wire = j.to_wire()
    wire["asserted_source"] = 99
    with pytest.raises(PushError) as exc:
        PackedJoint.from_wire(wire)
    assert exc.value.variant is PushErrorVariant.UNKNOWN_ASSERTED_SOURCE


def test_flat_6x6_property_equals_state_block() -> None:
    """The state's flat 6x6 is the derived leading block of the width-9 joint.

    One authority (the joint), one derived view (its state block); the
    6x6 is symmetric, read-only, and matches the triangle's first 21
    entries.
    """
    j = _joint(9)
    sb = j.state_block()
    assert sb.shape == (6, 6)
    assert not sb.flags.writeable
    assert np.allclose(sb, sb.T)
    # The first 21 triangle entries ARE the packed lower triangle of the
    # state block.
    idx = 0
    for i in range(6):
        for k in range(i + 1):
            assert sb[i, k] == j.tri[idx]
            idx += 1


def test_retired_third_order_kind_is_refused_by_name() -> None:
    """Ruling (b): the retired third-order kind (code 2) is refused by name,
    never coerced to a neighbouring kind.

    In the Python layer the refusal lands at the dict-wire decode
    (:meth:`PackedJoint.from_wire`), before the joint can reach the
    extension — and it raises the same :class:`PushError` the extension's
    push boundary would raise, tagged
    :attr:`PushErrorVariant.RETIRED_DISCRIMINANT`, never a bare
    ``ValueError`` from an enum lookup. The contract the ruling pins — tag
    2 is refused, not silently remapped — holds, and a caller classifies
    it the same way at every layer.
    """
    wire = _joint(6).to_wire()
    wire["kind"] = 2
    with pytest.raises(PushError, match="third-order") as exc:
        PackedJoint.from_wire(wire)
    assert exc.value.variant is PushErrorVariant.RETIRED_DISCRIMINANT


def test_removed_four_homes_fields_and_kwargs_raise() -> None:
    """Every removed four-homes kwarg / field raises — no silent acceptance.

    The state ``6x6``, the Marsden ``3x3`` and its border, the DT / AMRAT
    prior variances and the wide-cross carrier are gone: one packed joint
    now carries them all.
    """
    from empyrean.orbits.nongrav import NonGravParams
    from empyrean.orbits.srp import SRPParams
    from empyrean.orbits.thrust import ThrustParams

    base = {"a1": [0.0], "a2": [0.0], "a3": [0.0], "model": ["inverse_square"]}
    with pytest.raises(TypeError):
        NonGravParams.from_kwargs(**base, dt_variance=[1.0])
    with pytest.raises(TypeError):
        NonGravParams.from_kwargs(**base, covariance=[[0.0] * 9])
    with pytest.raises(TypeError):
        NonGravParams.from_kwargs(**base, non_grav_cross=[[0.0] * 18])
    with pytest.raises(TypeError):
        SRPParams.from_kwargs(amrat=[1.0], cr=[1.0], amrat_variance=[1.0])
    with pytest.raises(TypeError):
        ThrustParams(arcs=[], correction_covariances=[[(0.0, 0.0, 0.0)]])

    coords = CartesianCoordinates.from_kwargs(
        epoch=[59000.0],
        x=[1.0],
        y=[0.0],
        z=[0.0],
        vx=[0.0],
        vy=[0.017],
        vz=[0.0],
        frame="icrf",
        origin=["Sun"],
    )
    with pytest.raises(TypeError):
        CartesianOrbits.from_kwargs(orbit_id=["a"], coordinates=coords, wide_cross=None)
    orb = CartesianOrbits.from_kwargs(orbit_id=["a"], coordinates=coords)
    with pytest.raises(AttributeError):
        _ = orb.wide_cross
    # The one authority is present.
    assert isinstance(orb.covariance, PackedJoints)


def _cartesian_orbit_with_joint() -> CartesianOrbits:
    """A one-row Cartesian orbit carrying a width-6 state joint."""
    coords = CartesianCoordinates.from_kwargs(
        epoch=[59000.0],
        x=[1.2],
        y=[0.3],
        z=[0.1],
        vx=[-0.004],
        vy=[0.013],
        vz=[0.002],
        frame="ecliptic_j2000",
        origin=["Sun"],
    )
    joint = PackedJoints.from_joints([_joint(6)])
    return CartesianOrbits.from_kwargs(orbit_id=["obj"], coordinates=coords, covariance=joint)


def test_transform_orbits_round_trips_state_and_joint() -> None:
    """``transform_orbits`` moves a covariance-bearing orbit Cartesian →
    Keplerian → Cartesian, transporting the packed joint through the
    6x6 Jacobian both ways.

    The returned state and the joint's state block round-trip to a tight
    band (the C transform test's tolerance).
    """
    orb = _cartesian_orbit_with_joint()
    kep = transform_orbits(orb, KeplerianCoordinates)
    assert not kep.covariance.row_is_empty(0)
    back = transform_orbits(kep, CartesianOrbits)

    assert back.covariance.joint(0) is not None
    # State round-trips.
    for name in ("x", "y", "z", "vx", "vy", "vz"):
        np.testing.assert_allclose(
            getattr(back.coordinates, name).to_numpy(zero_copy_only=False),
            getattr(orb.coordinates, name).to_numpy(zero_copy_only=False),
            rtol=1e-8,
            atol=1e-10,
        )
    # Joint state block round-trips to the transform band.
    np.testing.assert_allclose(
        back.covariance.joint(0).state_block(),
        orb.covariance.joint(0).state_block(),
        rtol=1e-6,
        atol=1e-12,
    )


def test_state_level_transform_refuses_a_covariance_and_names_transform_orbits() -> None:
    """The −50 refusal: the state-level :func:`transform_coordinates` refuses a
    covariance-bearing input loudly as a translated :class:`PushError`, tagged
    :attr:`PushErrorVariant.TRANSFORM_COVARIANCE_UNSUPPORTED`, whose message
    names ``transform_orbits`` as the entry point that transports a covariance
    between bases.

    The refusal routes through the one translation point
    (:func:`~empyrean.errors.translate_push_errors`), so it is the same
    exception type every packed-joint refusal takes — never a bare
    ``RuntimeError``.
    """
    cov = empyrean.CartesianCovariance.from_matrix(np.eye(6)[np.newaxis] * 1e-6)
    coords = CartesianCoordinates.from_kwargs(
        epoch=[59000.0],
        x=[1.2],
        y=[0.3],
        z=[0.1],
        vx=[-0.004],
        vy=[0.013],
        vz=[0.002],
        frame="ecliptic_j2000",
        origin=["Sun"],
        covariance=cov,
    )
    with pytest.raises(PushError, match="transform_orbits") as exc:
        transform_coordinates(coords, KeplerianCoordinates)
    assert exc.value.variant is PushErrorVariant.TRANSFORM_COVARIANCE_UNSUPPORTED
