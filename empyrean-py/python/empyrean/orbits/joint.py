"""The packed joint covariance — the one covariance home.

A fit over the state and *P* parameters produces one
``(6+P) x (6+P)`` symmetric matrix. The whole matrix rides here as a
single packed lower triangle, tagged with how it was derived, its
definiteness, the functional it describes, and its provenance. The flat
state ``6x6`` is the leading block of that matrix, derived on demand
(:meth:`PackedJoint.state_block`) rather than stored a second time.

This replaces the earlier four-homes layout (a state ``6x6`` on the
coordinates, a Marsden ``3x3`` and its border on the non-grav params, and
every remaining cross term on a separate wide-cross table). One matrix,
one home: the state and the parameters came from the same observations
through the same fit, so their correlations belong together.

The axis layout
---------------

Each axis of the joint is one of three dispositions, recorded as a bitset
over the ``(6+P)`` axes:

- ``layout_present`` — axes carried in ``tri``.
- ``layout_considered`` — consider parameters (held, not solved).
- ``layout_marginalized`` — axes marginalized out of a wider fit.

The three bitsets and ``width`` mirror the C ABI's ``EmpyreanPackedJoint``
field for field, so the matrix round-trips through the engine without
reshaping.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import quivr as qv

from empyrean.errors import PushError, PushErrorVariant

__all__ = [
    "AssertedSource",
    "CovarianceKind",
    "JointFunctional",
    "JointProvenance",
    "JointQuality",
    "PackedJoint",
    "PackedJoints",
]


class CovarianceKind(str, enum.Enum):
    """How a covariance was derived.

    Subclasses ``str`` so values serialize directly into Arrow string
    columns. The integer codes match the C ABI ``EMPYREAN_COVARIANCE_KIND_*``
    order. Tag ``2`` (third-order) was retired with the engine and is
    refused by name on decode — never coerced to a neighbour.
    """

    LINEAR = "linear"
    """Linear STM mapping ``Φ Σ₀ Φᵀ`` (code 0)."""
    SECOND_ORDER = "second_order"
    """Park–Scheeres second-order (Jet2 STT) correction (code 1)."""
    MIXTURE = "mixture"
    """Adaptive Gaussian Mixture, moment-collapsed (code 3)."""
    MONTE_CARLO = "monte_carlo"
    """Monte Carlo sample covariance (code 4)."""
    SIGMA_POINT = "sigma_point"
    """Sigma-point sample covariance — the second moment of the
    propagated canonical 2N+1 sigma-point set (code 5)."""


class JointQuality(str, enum.Enum):
    """Definiteness of a packed joint matrix.

    The ``(rank, null_axes, min_eig_corr)`` payload rides alongside in the
    joint's own fields, each meaningful only for the tag that names it:
    ``rank`` / ``null_axes`` for :attr:`POSITIVE_SEMIDEFINITE`,
    ``min_eig_corr`` for :attr:`INDEFINITE` and :attr:`REPAIRED`.
    """

    POSITIVE_DEFINITE = "positive_definite"
    """All eigenvalues positive within round-off (code 0)."""
    POSITIVE_SEMIDEFINITE = "positive_semidefinite"
    """Rank-deficient but with no negative eigenvalue; ``rank`` and
    ``null_axes`` describe the null space (code 1)."""
    INDEFINITE = "indefinite"
    """At least one meaningfully negative eigenvalue; ``min_eig_corr``
    carries the most-negative correlation-scaled eigenvalue (code 2)."""
    REPAIRED = "repaired"
    """Explicitly repaired to PSD; ``min_eig_corr`` is the value before
    repair (code 3)."""


class JointFunctional(str, enum.Enum):
    """The functional a packed joint's second moment describes."""

    STATE = "state"
    """Generic Cartesian-state second moment (code 0)."""
    SKY = "sky"
    """Sky-plane (right-ascension / declination) functional (code 1)."""
    DELAY_DOPPLER = "delay_doppler"
    """Radar delay–Doppler functional (code 2)."""


class JointProvenance(str, enum.Enum):
    """Where a packed joint came from."""

    DERIVED = "derived"
    """Computed by the engine from observations or propagation (code 0)."""
    ASSERTED = "asserted"
    """Supplied directly by the caller (code 1)."""


class AssertedSource(str, enum.Enum):
    """Which external source asserted a joint's second moment.

    Meaningful only for an :attr:`JointProvenance.ASSERTED` joint; a derived
    joint carries ``None``.
    """

    SBDB = "sbdb"
    """The JPL Small-Body Database (code 0)."""
    EXTERNAL_FILE = "external_file"
    """Another external orbit file or tool (code 1)."""


# Integer-code ↔ enum maps. The codes are the C-ABI wire values the
# extension emits and consumes; an unknown code is refused by name rather
# than defaulted, so the distribution never silently coerces a tag.
_KIND_TO_CODE = {
    CovarianceKind.LINEAR: 0,
    CovarianceKind.SECOND_ORDER: 1,
    CovarianceKind.MIXTURE: 3,
    CovarianceKind.MONTE_CARLO: 4,
    CovarianceKind.SIGMA_POINT: 5,
}
_KIND_BY_CODE = {v: k for k, v in _KIND_TO_CODE.items()}

_QUALITY_TO_CODE = {
    JointQuality.POSITIVE_DEFINITE: 0,
    JointQuality.POSITIVE_SEMIDEFINITE: 1,
    JointQuality.INDEFINITE: 2,
    JointQuality.REPAIRED: 3,
}
_QUALITY_BY_CODE = {v: k for k, v in _QUALITY_TO_CODE.items()}

_FUNCTIONAL_TO_CODE = {
    JointFunctional.STATE: 0,
    JointFunctional.SKY: 1,
    JointFunctional.DELAY_DOPPLER: 2,
}
_FUNCTIONAL_BY_CODE = {v: k for k, v in _FUNCTIONAL_TO_CODE.items()}

_PROVENANCE_TO_CODE = {
    JointProvenance.DERIVED: 0,
    JointProvenance.ASSERTED: 1,
}
_PROVENANCE_BY_CODE = {v: k for k, v in _PROVENANCE_TO_CODE.items()}

_ASSERTED_SOURCE_TO_CODE = {
    AssertedSource.SBDB: 0,
    AssertedSource.EXTERNAL_FILE: 1,
}
_ASSERTED_SOURCE_BY_CODE = {v: k for k, v in _ASSERTED_SOURCE_TO_CODE.items()}
# The ABI's not-available sentinel for ``asserted_source`` (mirrors
# ``EMPYREAN_JOINT_ASSERTED_SOURCE_NONE``), carried by any non-asserted joint.
_ASSERTED_SOURCE_NONE = 255


def _kind_from_code(code: int) -> CovarianceKind:
    try:
        return _KIND_BY_CODE[int(code)]
    except KeyError:
        # A retired (code 2, third-order) or otherwise unknown kind is refused
        # by name as the same PushError the extension's push boundary raises —
        # never a bare ValueError from an enum lookup, so a caller classifies a
        # bad discriminant the same way at the decode and at the push.
        raise PushError(
            PushErrorVariant.RETIRED_DISCRIMINANT,
            f"unknown covariance kind code {code}; 2 (third-order) is retired and refused by name",
        ) from None


def _quality_from_code(code: int) -> JointQuality:
    try:
        return _QUALITY_BY_CODE[int(code)]
    except KeyError:
        raise ValueError(
            f"unknown joint quality code {code}; expected 0 (positive-definite), "
            "1 (positive-semidefinite), 2 (indefinite) or 3 (repaired)"
        ) from None


def _functional_from_code(code: int) -> JointFunctional:
    try:
        return _FUNCTIONAL_BY_CODE[int(code)]
    except KeyError:
        # The extension names an unknown-but-live functional code
        # `UnknownFunctional`; the Python decode raises the same variant.
        raise PushError(
            PushErrorVariant.UNKNOWN_FUNCTIONAL,
            f"unknown joint functional code {code}; expected 0 (state), "
            "1 (sky) or 2 (delay-Doppler)",
        ) from None


def _provenance_from_code(code: int) -> JointProvenance:
    try:
        return _PROVENANCE_BY_CODE[int(code)]
    except KeyError:
        # The extension names an unknown-but-live provenance code
        # `UnknownProvenance`; the Python decode raises the same variant.
        raise PushError(
            PushErrorVariant.UNKNOWN_PROVENANCE,
            f"unknown joint provenance code {code}; expected 0 (derived) or 1 (asserted)",
        ) from None


def _asserted_source_to_code(provenance: JointProvenance, source: AssertedSource | None) -> int:
    """Encode the asserted source against its provenance, enforcing the pairing.

    An asserted joint must carry a source; a derived joint must not. A
    mismatch is refused loudly rather than written as a defaulted byte.
    """
    if provenance is JointProvenance.ASSERTED:
        if source is None:
            raise ValueError("an asserted joint must carry its asserted source, but none was set")
        return _ASSERTED_SOURCE_TO_CODE[source]
    if source is not None:
        raise ValueError("a derived joint must not carry an asserted source")
    return _ASSERTED_SOURCE_NONE


def _asserted_source_from_code(code: int, provenance: JointProvenance) -> AssertedSource | None:
    """Decode the asserted source against its provenance.

    The source is present iff the joint is asserted. An asserted joint with
    the none sentinel or an unknown code is refused by name as the same
    ``PushError`` the extension's push boundary raises; a derived joint
    carrying a real source is a malformed row and refused loudly.
    """
    if provenance is JointProvenance.ASSERTED:
        try:
            return _ASSERTED_SOURCE_BY_CODE[int(code)]
        except KeyError:
            raise PushError(
                PushErrorVariant.UNKNOWN_ASSERTED_SOURCE,
                f"unknown joint asserted-source code {code}; expected 0 (SBDB) "
                "or 1 (external file)",
            ) from None
    if int(code) != _ASSERTED_SOURCE_NONE:
        raise ValueError(
            f"a derived joint carried asserted-source code {code}, but a derived "
            "joint has no source"
        )
    return None


def _tri_len(width: int) -> int:
    """Packed-lower-triangle length for a ``width x width`` matrix."""
    return width * (width + 1) // 2


@dataclass
class PackedJoint:
    """One packed joint covariance at a single ``(orbit, epoch)``.

    The ergonomic per-row view mirroring the C ABI's
    ``EmpyreanPackedJoint`` field for field. ``tri`` is the packed lower
    triangle of the full ``(6+P) x (6+P)`` matrix in row-major order
    (entry for ``(i, j)``, ``i >= j``, at index ``i(i+1)/2 + j``).

    Attributes
    ----------
    layout_present : int
        Bitset of axes carried in ``tri``.
    layout_considered : int
        Bitset of consider axes (held, not solved).
    layout_marginalized : int
        Bitset of axes marginalized out of a wider fit.
    width : int
        Side length of the joint matrix (``6 + P``).
    tri : np.ndarray
        Packed lower triangle, ``float64``, length ``width(width+1)/2``.
    kind : CovarianceKind
        How the matrix was derived.
    quality : JointQuality
        Definiteness of the matrix.
    cov_rank : int
        Numerical rank; meaningful for a positive-semidefinite joint.
    cov_null_axes : int
        Null-space dimension; meaningful for a positive-semidefinite joint.
    cov_min_eig_corr : float
        Most-negative correlation-scaled eigenvalue; meaningful for an
        indefinite or repaired joint, ``NaN`` otherwise.
    functional : JointFunctional
        The functional the second moment describes.
    provenance : JointProvenance
        Whether the joint was derived or asserted.
    asserted_source : AssertedSource | None
        Which external source asserted the joint — ``Some`` iff
        ``provenance`` is :attr:`JointProvenance.ASSERTED`, ``None`` for a
        derived joint. A mismatch is refused, never defaulted.
    """

    layout_present: int
    layout_considered: int
    layout_marginalized: int
    width: int
    tri: npt.NDArray[np.float64]
    kind: CovarianceKind
    quality: JointQuality
    cov_rank: int
    cov_null_axes: int
    cov_min_eig_corr: float
    functional: JointFunctional
    provenance: JointProvenance
    asserted_source: AssertedSource | None

    def state_block(self) -> npt.NDArray[np.float64]:
        """The leading state ``6x6`` of the joint, as a read-only array.

        Derived from the first 21 entries of ``tri`` (the packed lower
        triangle of the top-left ``6x6`` block), symmetrized. The joint is
        the authority; this is a view of its state block, never a second
        stored copy.
        """
        tri = np.asarray(self.tri, dtype=np.float64)
        if self.width < 6 or tri.size < 21:
            raise ValueError(
                f"packed joint has width {self.width} and {tri.size} triangle "
                "entries; a state block needs at least a 6x6 (21 entries)"
            )
        mat = np.zeros((6, 6), dtype=np.float64)
        idx = 0
        for i in range(6):
            for j in range(i + 1):
                mat[i, j] = tri[idx]
                mat[j, i] = tri[idx]
                idx += 1
        mat.setflags(write=False)
        return mat

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the flat dict the extension consumes.

        The enum tags encode to their C-ABI integer codes; ``tri`` becomes
        a contiguous ``float64`` array. The inverse of :meth:`from_wire`.
        """
        return {
            "layout_present": int(self.layout_present),
            "layout_considered": int(self.layout_considered),
            "layout_marginalized": int(self.layout_marginalized),
            "width": int(self.width),
            "tri": np.ascontiguousarray(self.tri, dtype=np.float64),
            "kind": _KIND_TO_CODE[self.kind],
            "quality": _QUALITY_TO_CODE[self.quality],
            "cov_rank": int(self.cov_rank),
            "cov_null_axes": int(self.cov_null_axes),
            "cov_min_eig_corr": float(self.cov_min_eig_corr),
            "functional": _FUNCTIONAL_TO_CODE[self.functional],
            "provenance": _PROVENANCE_TO_CODE[self.provenance],
            "asserted_source": _asserted_source_to_code(self.provenance, self.asserted_source),
        }

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> PackedJoint:
        """Rebuild from the flat dict the extension produces.

        Every discriminant is decoded by name; an unknown code is refused,
        never defaulted. The inverse of :meth:`to_wire`.
        """
        provenance = _provenance_from_code(wire["provenance"])
        return cls(
            layout_present=int(wire["layout_present"]),
            layout_considered=int(wire["layout_considered"]),
            layout_marginalized=int(wire["layout_marginalized"]),
            width=int(wire["width"]),
            tri=np.asarray(wire["tri"], dtype=np.float64),
            kind=_kind_from_code(wire["kind"]),
            quality=_quality_from_code(wire["quality"]),
            cov_rank=int(wire["cov_rank"]),
            cov_null_axes=int(wire["cov_null_axes"]),
            cov_min_eig_corr=float(wire["cov_min_eig_corr"]),
            functional=_functional_from_code(wire["functional"]),
            provenance=provenance,
            asserted_source=_asserted_source_from_code(wire["asserted_source"], provenance),
        )


class PackedJoints(qv.Table):
    """Row-aligned packed joint covariances — the orbit covariance column.

    Attached to each orbit table as a nullable sub-table column: an orbit
    with no joint carries per-row nulls (a gravity-only or SBDB orbit), so
    every existing table and round trip is unchanged. Test
    :meth:`row_is_empty` or the per-row :meth:`joint` accessor, never
    ``orbits.covariance is None`` — quivr returns a table of parent length
    regardless.

    The enum tags ride as string columns (their ``.value``); ``tri`` rides
    as a flat ``LargeListColumn`` whose per-row length is
    ``width(width+1)/2``.
    """

    layout_present = qv.UInt32Column(nullable=True)
    layout_considered = qv.UInt32Column(nullable=True)
    layout_marginalized = qv.UInt32Column(nullable=True)
    width = qv.UInt32Column(nullable=True)
    tri = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    kind = qv.LargeStringColumn(nullable=True)
    quality = qv.LargeStringColumn(nullable=True)
    cov_rank = qv.UInt32Column(nullable=True)
    cov_null_axes = qv.UInt32Column(nullable=True)
    cov_min_eig_corr = qv.Float64Column(nullable=True)
    functional = qv.LargeStringColumn(nullable=True)
    provenance = qv.LargeStringColumn(nullable=True)
    asserted_source = qv.LargeStringColumn(nullable=True)

    def row_is_empty(self, i: int) -> bool:
        """Whether row ``i`` carries no joint at all.

        The per-row absence test. ``PackedJoints`` itself is never ``None``
        on a parent table, so this is what "this orbit has no covariance"
        looks like.
        """
        return self.width[i].as_py() is None

    def joint(self, i: int) -> PackedJoint | None:
        """Row ``i`` as a :class:`PackedJoint`, or ``None`` when empty."""
        if self.row_is_empty(i):
            return None
        return PackedJoint(
            layout_present=int(self.layout_present[i].as_py()),
            layout_considered=int(self.layout_considered[i].as_py()),
            layout_marginalized=int(self.layout_marginalized[i].as_py()),
            width=int(self.width[i].as_py()),
            tri=np.asarray(self.tri[i].as_py(), dtype=np.float64),
            kind=CovarianceKind(self.kind[i].as_py()),
            quality=JointQuality(self.quality[i].as_py()),
            cov_rank=int(self.cov_rank[i].as_py()),
            cov_null_axes=int(self.cov_null_axes[i].as_py()),
            cov_min_eig_corr=float(self.cov_min_eig_corr[i].as_py()),
            functional=JointFunctional(self.functional[i].as_py()),
            provenance=JointProvenance(self.provenance[i].as_py()),
            asserted_source=(
                AssertedSource(self.asserted_source[i].as_py())
                if self.asserted_source[i].as_py() is not None
                else None
            ),
        )

    @classmethod
    def from_joints(cls, joints: list[PackedJoint | None]) -> PackedJoints:
        """Build a table from per-row joints, nulling the empty rows.

        A ``None`` row is an orbit with no covariance, written as per-row
        nulls rather than as a zero-filled joint (which would read as a
        supplied zero covariance).
        """
        n = len(joints)
        present: list[int | None] = [None] * n
        considered: list[int | None] = [None] * n
        marginalized: list[int | None] = [None] * n
        width: list[int | None] = [None] * n
        tri: list[list[float] | None] = [None] * n
        kind: list[str | None] = [None] * n
        quality: list[str | None] = [None] * n
        rank: list[int | None] = [None] * n
        null_axes: list[int | None] = [None] * n
        min_eig: list[float | None] = [None] * n
        functional: list[str | None] = [None] * n
        provenance: list[str | None] = [None] * n
        asserted_source: list[str | None] = [None] * n
        for i, j in enumerate(joints):
            if j is None:
                continue
            present[i] = int(j.layout_present)
            considered[i] = int(j.layout_considered)
            marginalized[i] = int(j.layout_marginalized)
            width[i] = int(j.width)
            tri[i] = [float(v) for v in np.asarray(j.tri, dtype=np.float64).ravel()]
            kind[i] = j.kind.value
            quality[i] = j.quality.value
            rank[i] = int(j.cov_rank)
            null_axes[i] = int(j.cov_null_axes)
            min_eig[i] = float(j.cov_min_eig_corr)
            functional[i] = j.functional.value
            provenance[i] = j.provenance.value
            asserted_source[i] = j.asserted_source.value if j.asserted_source is not None else None
        return cls.from_kwargs(
            layout_present=present,
            layout_considered=considered,
            layout_marginalized=marginalized,
            width=width,
            tri=tri,
            kind=kind,
            quality=quality,
            cov_rank=rank,
            cov_null_axes=null_axes,
            cov_min_eig_corr=min_eig,
            functional=functional,
            provenance=provenance,
            asserted_source=asserted_source,
        )

    def to_wire_list(self) -> list[dict[str, Any] | None]:
        """The per-row packed-joint list the extension consumes.

        One entry per row — a :meth:`PackedJoint.to_wire` dict, or ``None``
        for an empty row.
        """
        out: list[dict[str, Any] | None] = []
        for i in range(len(self)):
            j = self.joint(i)
            out.append(j.to_wire() if j is not None else None)
        return out

    @classmethod
    def from_wire_list(cls, rows: list[dict[str, Any] | None] | None) -> PackedJoints | None:
        """Build a table from the extension's per-row packed-joint list.

        ``None`` (no key on the wire) means the batch carried no joint at
        all and returns ``None``; a ``None`` row is an orbit with no
        covariance.
        """
        if rows is None:
            return None
        return cls.from_joints([PackedJoint.from_wire(r) if r is not None else None for r in rows])
