"""Provenance-tagged, resolved-kind covariance readback.

The honest per-``(orbit, epoch)`` covariance distinct from the bare
linear ``Φ Σ₀ Φᵀ`` mapping carried on the propagated states. When a
propagation crosses an ``Auto`` close-approach window the resolved
covariance there can be a second-order Park–Scheeres ellipsoid rather
than the linear one; this table records the packed joint produced at each
epoch — the single covariance authority — along with its co-located
nominal state, any second-order mean shift, and the basis
(origin / frame). The joint's :class:`~empyrean.orbits.joint.CovarianceKind`,
:class:`~empyrean.orbits.joint.JointQuality`, state ``6×6`` and solved
width ride beside it as derived convenience columns.

Two shapes, mirroring the rest of the package:

- :class:`TaggedCovariances` — a flat quivr Table, one row per
  ``(orbit, epoch)``, rows grouped contiguously by ``orbit_id`` to match
  propagation's orbit-major output. The joint rides as the
  :class:`~empyrean.orbits.joint.PackedJoints` column; its leading state
  ``6×6`` is mirrored as 21 lower-triangular ``cov_*`` columns.
- :class:`TaggedCovariance` — a small per-epoch dataclass carrying the
  :class:`~empyrean.orbits.joint.PackedJoint` with the matrix, kind,
  quality and solved width derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import quivr as qv

from empyrean.coordinates.covariance import (
    _cov_column_names,
    _lower_tri_indices,
)
from empyrean.coordinates.epoch import Epochs
from empyrean.orbits.joint import (
    _KIND_BY_CODE,
    _QUALITY_BY_CODE,
    CovarianceKind,
    JointQuality,
    PackedJoint,
    PackedJoints,
)

__all__ = [
    "CovarianceKind",
    "JointQuality",
    "TaggedCovariance",
    "TaggedCovariances",
    "build_tagged_covariances",
]

# 6×6 Cartesian state labels — the matrix basis is always Cartesian
# [x, y, z, vx, vy, vz] for the tagged readback.
_STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz"]
_COV_NAMES = _cov_column_names(_STATE_LABELS)
_LOWER_TRI = _lower_tri_indices(6)


@dataclass
class TaggedCovariance:
    """Provenance-tagged covariance at a single ``(orbit, epoch)``.

    The ergonomic per-epoch view yielded by
    :meth:`TaggedCovariances.to_series`. The packed
    :attr:`joint` is the authority; :attr:`matrix`, :attr:`kind`,
    :attr:`quality` and :attr:`solved_width` are derived from it.

    Attributes
    ----------
    epoch : Epochs
        Epoch of this covariance as a single-row :class:`~empyrean.Epochs`,
        in the request scale (carrying its scale as data).
    state : np.ndarray
        Co-located propagated nominal state ``[x, y, z, vx, vy, vz]``
        (AU, AU/day), shape ``(6,)``.
    joint : PackedJoint
        The packed joint covariance — the single covariance home.
    mc_seed : int, optional
        Monte-Carlo run seed. Set only when :attr:`kind` is
        :attr:`CovarianceKind.MONTE_CARLO`, and not on every such row: an
        ensemble seeded from system entropy carries ``None``.
    mean_shift_prop : np.ndarray, optional
        Second-order propagation mean shift ``δμ_prop`` (zero at t₀),
        shape ``(6,)`` or ``None``.
    origin : str
        Canonical origin (center body) name of the basis.
    frame : str
        Reference frame of the basis (canonical name, e.g. ``"icrf"``).
    """

    epoch: Epochs
    state: np.ndarray
    joint: PackedJoint
    mc_seed: int | None
    mean_shift_prop: np.ndarray | None
    origin: str
    frame: str

    @property
    def matrix(self) -> np.ndarray:
        """The state ``6×6``, derived from the joint's leading block."""
        return self.joint.state_block()

    @property
    def kind(self) -> CovarianceKind:
        """How the covariance was derived (derived from the joint)."""
        return self.joint.kind

    @property
    def quality(self) -> JointQuality:
        """Definiteness of the matrix (derived from the joint)."""
        return self.joint.quality

    @property
    def solved_width(self) -> int:
        """Solved width ``6 + P`` (derived from the joint)."""
        return self.joint.width

    @property
    def corrected_mean(self) -> np.ndarray:
        """The corrected mean: ``state + δμ_prop``.

        The mean shift defaults to zero when absent, so this always
        returns a ``(6,)`` array.
        """
        out = np.asarray(self.state, dtype=np.float64).copy()
        if self.mean_shift_prop is not None:
            out = out + self.mean_shift_prop
        return np.ascontiguousarray(out, dtype=np.float64)


class TaggedCovariances(qv.Table):
    """Per-``(orbit, epoch)`` provenance-tagged covariance readback.

    One row per output epoch; rows are grouped contiguously by
    ``orbit_id`` (matching propagation's orbit-major output). Filter to
    one chain with quivr's standard ``select`` before calling the
    per-chain accessor::

        chain = tagged.select("orbit_id", "2024 YR4")
        series = chain.to_series()

    Notes
    -----
    The packed joint rides as the
    :class:`~empyrean.orbits.joint.PackedJoints` column — the covariance
    authority. Its leading state ``6×6`` is mirrored as 21
    lower-triangular ``cov_*`` columns, and its kind / quality / solved
    width ride as derived convenience columns.

    ``has_tagged`` is ``False`` on rows where the underlying orbit carried
    no covariance — those rows are zero-filled placeholders that keep the
    table aligned 1:1 with the propagated states.
    """

    orbit_id = qv.LargeStringColumn()
    """Orbit primary key (matches the input ``Orbits.orbit_id``)."""
    object_id = qv.LargeStringColumn(nullable=True)
    """Object metadata label, if carried on the input orbit."""
    epochs = Epochs.as_column()
    """Output epochs as an :class:`~empyrean.Epochs` sub-table, in the request scale."""

    # Co-located propagated nominal state [x, y, z, vx, vy, vz].
    state_x = qv.Float64Column()
    state_y = qv.Float64Column()
    state_z = qv.Float64Column()
    state_vx = qv.Float64Column()
    state_vy = qv.Float64Column()
    state_vz = qv.Float64Column()

    kind = qv.LargeStringColumn()
    """Resolved covariance kind (``CovarianceKind`` value), derived from
    the joint."""
    quality = qv.LargeStringColumn()
    """Definiteness (``JointQuality`` value), derived from the joint."""
    quality_rank = qv.UInt32Column(nullable=True)
    """Numerical rank for a positive-semidefinite joint."""
    quality_null_axes = qv.UInt32Column(nullable=True)
    """Null-space dimension for a positive-semidefinite joint."""
    quality_min_eig_corr = qv.Float64Column(nullable=True)
    """Most-negative correlation-scaled eigenvalue for an indefinite /
    repaired joint; NaN otherwise."""
    solved_width = qv.UInt32Column()
    """Solved width ``6 + P``, derived from the joint."""

    mc_seed = qv.UInt64Column(nullable=True)
    """Monte-Carlo run seed; null unless ``kind`` is ``monte_carlo``, and
    null on a ``monte_carlo`` row whose ensemble was seeded from system
    entropy (not reproducible by construction)."""

    # Second-order propagation mean shift δμ_prop (zero at t₀).
    mean_shift_prop_x = qv.Float64Column(nullable=True)
    mean_shift_prop_y = qv.Float64Column(nullable=True)
    mean_shift_prop_z = qv.Float64Column(nullable=True)
    mean_shift_prop_vx = qv.Float64Column(nullable=True)
    mean_shift_prop_vy = qv.Float64Column(nullable=True)
    mean_shift_prop_vz = qv.Float64Column(nullable=True)
    has_mean_shift_prop = qv.BooleanColumn()
    """Whether ``mean_shift_prop_*`` carries a value on this row."""

    origin = qv.LargeStringColumn()
    """Canonical origin (center body) name of the basis."""
    frame = qv.LargeStringColumn()
    """Reference frame of the basis (canonical name)."""

    # 6×6 state block as 21 lower-triangular columns (derived from the
    # joint's leading block).
    cov_x_x = qv.Float64Column(nullable=True)
    cov_x_y = qv.Float64Column(nullable=True)
    cov_y_y = qv.Float64Column(nullable=True)
    cov_x_z = qv.Float64Column(nullable=True)
    cov_y_z = qv.Float64Column(nullable=True)
    cov_z_z = qv.Float64Column(nullable=True)
    cov_x_vx = qv.Float64Column(nullable=True)
    cov_y_vx = qv.Float64Column(nullable=True)
    cov_z_vx = qv.Float64Column(nullable=True)
    cov_vx_vx = qv.Float64Column(nullable=True)
    cov_x_vy = qv.Float64Column(nullable=True)
    cov_y_vy = qv.Float64Column(nullable=True)
    cov_z_vy = qv.Float64Column(nullable=True)
    cov_vx_vy = qv.Float64Column(nullable=True)
    cov_vy_vy = qv.Float64Column(nullable=True)
    cov_x_vz = qv.Float64Column(nullable=True)
    cov_y_vz = qv.Float64Column(nullable=True)
    cov_z_vz = qv.Float64Column(nullable=True)
    cov_vx_vz = qv.Float64Column(nullable=True)
    cov_vy_vz = qv.Float64Column(nullable=True)
    cov_vz_vz = qv.Float64Column(nullable=True)

    has_tagged = qv.BooleanColumn()
    """``False`` on zero-filled placeholder rows where the underlying
    orbit carried no covariance."""

    covariance = PackedJoints.as_column(nullable=True)
    """The packed joint per ``(orbit, epoch)`` — the covariance authority.
    Null on a row whose orbit carried no covariance; see
    :meth:`PackedJoints.row_is_empty`."""

    # ── Introspection ─────────────────────────────────────────

    def orbit_ids_unique(self) -> list[str]:
        """Unique ``orbit_id`` values, in first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for v in self.orbit_id.to_pylist():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    # ── Matrix reshaping ──────────────────────────────────────

    def matrices(self) -> np.ndarray:
        """Reshape the lower-tri ``cov_*`` columns to ``(n, 6, 6)``.

        Rows with ``has_tagged=False`` come back zero-filled. Works on the
        full table or a filtered single chain.
        """
        n = len(self)
        mat = np.zeros((n, 6, 6), dtype=np.float64)
        for name, (i, j) in zip(_COV_NAMES, _LOWER_TRI, strict=False):
            vals = self.column(name).to_numpy(zero_copy_only=False)
            vals = np.nan_to_num(vals, nan=0.0)
            mat[:, i, j] = vals
            if i != j:
                mat[:, j, i] = vals
        return mat

    # ── Per-epoch series ──────────────────────────────────────

    def to_series(self) -> list[TaggedCovariance]:
        """Materialize this table as a list of :class:`TaggedCovariance`.

        One entry per row, in table order. Filter to a single chain via
        ``select("orbit_id", oid)`` first to get one orbit's series.
        """
        n = len(self)
        origins = self.column("origin").to_pylist()
        frames = self.column("frame").to_pylist()
        mc_seeds = self.column("mc_seed").to_pylist()
        has_prop = self.column("has_mean_shift_prop").to_numpy(zero_copy_only=False)

        state = np.column_stack(
            [self.column(f"state_{lab}").to_numpy(zero_copy_only=False) for lab in _STATE_LABELS]
        )
        prop = np.column_stack(
            [
                self.column(f"mean_shift_prop_{lab}").to_numpy(zero_copy_only=False)
                for lab in _STATE_LABELS
            ]
        )

        out: list[TaggedCovariance] = []
        for i in range(n):
            pj = self.covariance.joint(i)
            if pj is None:
                continue
            out.append(
                TaggedCovariance(
                    # Single-row slice of the request-scale epochs sub-table:
                    # the per-row epoch carries its scale as data.
                    epoch=self.epochs[i : i + 1],
                    state=np.ascontiguousarray(state[i], dtype=np.float64),
                    joint=pj,
                    mc_seed=int(mc_seeds[i]) if mc_seeds[i] is not None else None,
                    mean_shift_prop=(
                        np.ascontiguousarray(prop[i], dtype=np.float64)
                        if bool(has_prop[i])
                        else None
                    ),
                    origin=str(origins[i]),
                    frame=str(frames[i]),
                )
            )
        return out


def build_tagged_covariances(
    result: dict[str, Any],
    orbit_ids: list[str],
    object_ids: list[str | None],
    epochs_mjd_tdb: np.ndarray,
    request_scale: str,
) -> TaggedCovariances | None:
    """Build a :class:`TaggedCovariances` table from the Rust result.

    ``result`` is the dict returned by the ``_propagate`` extension when
    ``with_tagged_covariance=True``; ``orbit_ids`` / ``object_ids`` /
    ``epochs_mjd_tdb`` are the already-flattened per-row arrays from the
    states (length ``n``, orbit-major). Returns ``None`` if the extension
    produced no tagged sub-dict.
    """
    from empyrean._convert import int_to_frame, joint_from_result, naif_to_origin

    tagged = result.get("tagged_covariance")
    if tagged is None:
        return None

    matrix = np.asarray(tagged["matrix"], dtype=np.float64)  # (n, 6, 6)
    state = np.asarray(tagged["state"], dtype=np.float64)  # (n, 6)
    kind_codes = np.asarray(tagged["kind"])
    mc_seed = np.asarray(tagged["mc_seed"], dtype=np.uint64)
    has_mc_seed = np.asarray(tagged["has_mc_seed"], dtype=bool)
    mean_shift_prop = np.asarray(tagged["mean_shift_prop"], dtype=np.float64)
    has_mean_shift_prop = np.asarray(tagged["has_mean_shift_prop"], dtype=bool)
    quality_codes = np.asarray(tagged["quality"])
    quality_rank = np.asarray(tagged["quality_rank"], dtype=np.uint32)
    quality_null_axes = np.asarray(tagged["quality_null_axes"], dtype=np.uint32)
    quality_min_eig_corr = np.asarray(tagged["quality_min_eig_corr"], dtype=np.float64)
    solved_width = np.asarray(tagged["solved_width"], dtype=np.uint32)
    origin_codes = np.asarray(tagged["origin"], dtype=np.int64)
    frame_codes = np.asarray(tagged["frame"], dtype=np.int64)
    has_tagged = np.asarray(tagged["has_tagged"], dtype=bool)

    n = len(orbit_ids)

    kind_strs = [_KIND_BY_CODE[int(c)].value for c in kind_codes]
    quality_strs = [_QUALITY_BY_CODE[int(c)].value for c in quality_codes]
    origin_strs = [naif_to_origin(int(o)) for o in origin_codes]
    frame_strs = [int_to_frame(int(f)).value for f in frame_codes]

    # mc_seed → null where absent.
    mc_seed_col = [int(mc_seed[i]) if has_mc_seed[i] else None for i in range(n)]

    kwargs: dict[str, Any] = {
        "orbit_id": orbit_ids,
        "object_id": object_ids,
        "epochs": Epochs.from_kwargs(
            mjd=np.asarray(epochs_mjd_tdb, dtype=np.float64), scale="tdb"
        ).to_scale(request_scale),
        "state_x": state[:, 0],
        "state_y": state[:, 1],
        "state_z": state[:, 2],
        "state_vx": state[:, 3],
        "state_vy": state[:, 4],
        "state_vz": state[:, 5],
        "kind": kind_strs,
        "quality": quality_strs,
        "quality_rank": quality_rank,
        "quality_null_axes": quality_null_axes,
        "quality_min_eig_corr": quality_min_eig_corr,
        "solved_width": solved_width,
        "mc_seed": mc_seed_col,
        "mean_shift_prop_x": mean_shift_prop[:, 0],
        "mean_shift_prop_y": mean_shift_prop[:, 1],
        "mean_shift_prop_z": mean_shift_prop[:, 2],
        "mean_shift_prop_vx": mean_shift_prop[:, 3],
        "mean_shift_prop_vy": mean_shift_prop[:, 4],
        "mean_shift_prop_vz": mean_shift_prop[:, 5],
        "has_mean_shift_prop": has_mean_shift_prop,
        "origin": origin_strs,
        "frame": frame_strs,
        "has_tagged": has_tagged,
    }
    # 6×6 state block → 21 lower-tri columns (derived convenience mirror
    # of the joint's leading block).
    for name, (i, j) in zip(_COV_NAMES, _LOWER_TRI, strict=False):
        kwargs[name] = matrix[:, i, j]

    # The packed joint per row — the covariance authority.
    joint = joint_from_result(tagged)
    if joint is not None:
        kwargs["covariance"] = joint

    return TaggedCovariances.from_kwargs(**kwargs)
