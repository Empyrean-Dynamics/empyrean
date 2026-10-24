"""Cross-layer API parity for the packed-joint / transform / push-code
families (the API-parity rule: identical names and argument order across
C / wrapper / Python / CLI, subsetting only).

Two guards:

* **NAME parity** walks the C header's joint / transform / push-code
  families and fails if any has no counterpart in the Python public
  surface (a rename or a dropped symbol). The C ↔ wrapper (Rust) pair is
  covered by ``empyrean-sys/tests/capability_parity.rs`` (every exported
  symbol is consumed or allow-listed) and the wrapper's own
  ``error.rs::every_push_code_round_trips_to_its_own_variant``; this adds
  the C-header ↔ Python pair. The CLI is a strict subset (it has no
  ``transform`` command; it surfaces covariance through
  ``propagate --tagged-covariance`` and the ``determine`` prior flags),
  which is the parity rule's "subsetting only".

* **VALUE parity** pushes the same synthetic width-9 Marsden orbit through
  the Python C-extension wire and asserts the packed triangle and layout
  bits come back bit-identical. The C and wrapper layers pin the same
  width-9 bit-identity in their own suites (the C contract test and the
  wrapper joint round-trip), so the one synthetic joint is bit-stable at
  every layer.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    OrbitOutcome,
    PackedJoint,
    PackedJoints,
    transform_coordinates,
    transform_orbits,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.epoch import Epochs
from empyrean.errors import PushErrorVariant

_HEADER = Path(__file__).parents[2] / "include" / "empyrean.h"


def _header_text() -> str:
    if not _HEADER.exists():
        pytest.skip(f"C header not found at {_HEADER}")
    return _HEADER.read_text()


def test_name_parity_push_code_family_c_header_to_python() -> None:
    """Every C push-code constant has exactly one Python ``PushErrorVariant``
    and vice versa — a rename or a dropped code on either side fails here.

    The header names each refusal ``EMPYREAN_PUSH_<AXIS>`` (plus the
    transform refusal ``EMPYREAN_TRANSFORM_COVARIANCE_UNSUPPORTED``); the
    Python enum member is the same ``<AXIS>`` token. The comparison is on
    the token set, so neither a new C code without a Python variant nor a
    Python variant without a C code can slip through.
    """
    text = _header_text()
    header_tokens = {m.group(1) for m in re.finditer(r"#define\s+EMPYREAN_PUSH_([A-Z_]+)\b", text)}
    # The state-level transform refusal rides the same family but is named
    # without the PUSH infix in the header.
    if re.search(r"#define\s+EMPYREAN_TRANSFORM_COVARIANCE_UNSUPPORTED\b", text):
        header_tokens.add("TRANSFORM_COVARIANCE_UNSUPPORTED")

    python_tokens = {v.name for v in PushErrorVariant}

    assert header_tokens, "no EMPYREAN_PUSH_* constants found in the header"
    assert header_tokens == python_tokens, (
        "push-code family diverged between the C header and Python "
        f"PushErrorVariant.\n  C only: {sorted(header_tokens - python_tokens)}"
        f"\n  Python only: {sorted(python_tokens - header_tokens)}"
    )


def test_name_parity_outcome_family_c_header_to_python() -> None:
    """Every C ``EMPYREAN_ORBIT_OUTCOME_*`` constant has exactly one Python
    ``OrbitOutcome`` member and vice versa — a rename or a dropped status on
    either side fails here.

    The header names each delivery status ``EMPYREAN_ORBIT_OUTCOME_<NAME>``;
    the Python enum member is the same ``<NAME>`` token. The C ↔ sys-binding
    pair (plus the two per-command failure families) is pinned by
    ``empyrean-sys/tests/capability_parity.rs::the_outcome_const_families_match_between_header_and_bindings``;
    this adds the C-header ↔ Python pair. The CLI surfaces the same channel
    as its exit codes (0 / 3 / 4) and the per-orbit status table rather than
    a named enum, which is the parity rule's "subsetting only".
    """
    text = _header_text()
    header_tokens = {
        m.group(1) for m in re.finditer(r"#define\s+EMPYREAN_ORBIT_OUTCOME_([A-Z_]+)\b", text)
    }
    python_tokens = {v.name for v in OrbitOutcome}

    assert header_tokens, "no EMPYREAN_ORBIT_OUTCOME_* constants found in the header"
    assert header_tokens == python_tokens, (
        "outcome status family diverged between the C header and Python "
        f"OrbitOutcome.\n  C only: {sorted(header_tokens - python_tokens)}"
        f"\n  Python only: {sorted(python_tokens - header_tokens)}"
    )

    # The two per-command failure families (the classified codes the Python
    # status table carries on OrbitStatuses.error_code) are present in the
    # header, each with its non-exhaustive _OTHER catch-all.
    assert re.search(r"#define\s+EMPYREAN_PROPAGATE_FAILURE_OTHER\b", text), (
        "the propagate failure family (with its _OTHER catch-all) is missing from the header"
    )
    assert re.search(r"#define\s+EMPYREAN_EPHEMERIS_FAILURE_OTHER\b", text), (
        "the ephemeris failure family (with its _OTHER catch-all) is missing from the header"
    )


def test_name_parity_transform_and_joint_families_present_in_python() -> None:
    """The transform and joint families named in the C header are all present
    in the Python public surface by name.

    The header declares ``empyrean_transform_orbits`` (orbit-level) beside
    ``empyrean_transform_coordinates`` (state-level); Python exports
    ``transform_orbits`` and ``transform_coordinates`` with the same verbs.
    The joint tag families (kind / quality / functional / provenance /
    asserted-source) each have a Python enum. A dropped or renamed family
    fails here.
    """
    text = _header_text()

    # Transform family: same verb at the orbit and state levels.
    assert "empyrean_transform_orbits" in text
    assert "empyrean_transform_coordinates" in text
    assert callable(transform_orbits)
    assert callable(transform_coordinates)
    # Argument order (subsetting only): both take the orbits/coords first,
    # then the target coordinate class — pin the leading parameter names.
    import inspect

    assert list(inspect.signature(transform_orbits).parameters)[:2] == [
        "orbits",
        "target",
    ]
    assert list(inspect.signature(transform_coordinates).parameters)[:2] == [
        "coordinates",
        "target",
    ]

    # Joint tag families — present as Python enums / classes by name.
    for cls in (
        PackedJoint,
        PackedJoints,
        CovarianceKind,
        JointQuality,
        JointFunctional,
        JointProvenance,
        AssertedSource,
    ):
        assert cls is not None

    # The retired third-order kind stays absent at the Python layer (its
    # number is unassigned), matching the C header's retired tag.
    assert not hasattr(CovarianceKind, "THIRD_ORDER")


def _marsden_width9_tri() -> np.ndarray:
    """Packed lower triangle of a synthetic width-9 Marsden joint.

    Deterministic, exact-in-``float64`` decimals (no RNG, no SPICE), so a
    bit-for-bit comparison of the triangle is a valid platform-independent
    parity assertion — the dict wire the extension speaks is a verbatim
    ``float64`` carry.
    """
    n = 9 * (9 + 1) // 2  # 45 lower-triangle entries
    return np.array([(k + 1) * 0.5e-9 for k in range(n)], dtype=np.float64)


def _marsden_width9_joint() -> PackedJoint:
    # width 9 = 6 state + 3 Marsden axes (bits 0, 1, 2 → present = 0b111).
    return PackedJoint(
        layout_present=0b111,
        layout_considered=0,
        layout_marginalized=0,
        width=9,
        tri=_marsden_width9_tri(),
        kind=CovarianceKind.LINEAR,
        quality=JointQuality.POSITIVE_DEFINITE,
        cov_rank=0,
        cov_null_axes=0,
        cov_min_eig_corr=float("nan"),
        functional=JointFunctional.STATE,
        provenance=JointProvenance.DERIVED,
        asserted_source=None,
    )


def _bits(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def test_value_parity_width9_marsden_tri_bit_identical_through_the_wire() -> None:
    """The same synthetic width-9 Marsden joint is bit-identical through the
    Python C-extension wire and the quivr column — the Python-layer value
    parity.

    ``to_bits`` is admissible: the triangle is synthetic Cartesian (no
    angular degree↔radian congruence, no SPICE-backed value), and the wire
    is a verbatim ``float64`` carry, so the round trip is exact rather than
    a platform-dependent numeric result.
    """
    j = _marsden_width9_joint()
    want = _bits(_marsden_width9_tri())
    assert np.array_equal(_bits(j.tri), want), "constructor altered the triangle"

    # Dict wire (the extension's push boundary): to_wire → from_wire.
    back = PackedJoint.from_wire(j.to_wire())
    assert back.width == 9
    assert back.layout_present == 0b111
    assert back.kind is CovarianceKind.LINEAR
    assert np.array_equal(_bits(back.tri), want), "dict wire changed the triangle bits"

    # Quivr column (the package's columnar carrier): from_joints → joint.
    col = PackedJoints.from_joints([j, None])
    assert col.row_is_empty(1)
    rt = col.joint(0)
    assert rt is not None
    assert rt.layout_present == 0b111
    assert np.array_equal(_bits(rt.tri), want), "quivr column changed the triangle bits"


def test_value_parity_width9_marsden_orbit_survives_a_parquet_round_trip() -> None:
    """The width-9 Marsden joint, carried on an orbit, survives a parquet
    round trip bit-identically — the persisted file is the one the CLI and
    Python share.
    """
    empyrean.initialize()
    coords = CartesianCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=[59000.0], scale="tdb"),
        x=[1.2],
        y=[0.3],
        z=[0.1],
        vx=[-0.004],
        vy=[0.013],
        vz=[0.002],
        frame="ecliptic_j2000",
        origin=["Sun"],
    )
    orb = CartesianOrbits.from_kwargs(
        orbit_id=["m9"],
        coordinates=coords,
        covariance=PackedJoints.from_joints([_marsden_width9_joint()]),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m9.parquet"
        orb.to_parquet(str(path))
        back = CartesianOrbits.from_parquet(str(path))
    j = back.covariance.joint(0)
    assert j is not None
    assert j.width == 9
    assert j.layout_present == 0b111
    assert np.array_equal(_bits(j.tri), _bits(_marsden_width9_tri()))
