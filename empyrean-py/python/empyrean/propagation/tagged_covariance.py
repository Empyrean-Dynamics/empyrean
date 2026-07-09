"""Provenance-tagged, resolved-kind covariance readback.

The honest per-``(orbit, epoch)`` covariance distinct from the bare
linear ``Φ Σ₀ Φᵀ`` mapping carried on the propagated states. When a
propagation crosses an ``Auto`` close-approach window the resolved
covariance there can be a second-order Park–Scheeres ellipsoid rather
than the linear one; this table records *which* kind produced each
epoch's matrix, along with its definiteness, any mean shifts, the
solved-for width, and the basis (origin / frame).

Two shapes, mirroring the rest of the package:

- :class:`TaggedCovariances` — a flat quivr Table, one row per
  ``(orbit, epoch)``, rows grouped contiguously by ``orbit_id`` to
  match propagation's orbit-major output. The 6×6 matrix rides as 21
  lower-triangular ``cov_*`` columns, exactly like
  :class:`~empyrean.coordinates.covariance.CartesianCovariance`.
- :class:`TaggedCovariance` — a small per-epoch dataclass with the
  matrix re-materialized as a contiguous ``(6, 6)`` ``np.ndarray`` and
  the enums decoded, returned by
  :meth:`~empyrean.PropagationResult.tagged_covariance_series`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import quivr as qv

from empyrean.coordinates.covariance import (
    _cov_column_names,
    _lower_tri_indices,
)

# 6×6 Cartesian state labels — the matrix basis is always Cartesian
# [x, y, z, vx, vy, vz] for the tagged readback.
_STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz"]
_COV_NAMES = _cov_column_names(_STATE_LABELS)
_LOWER_TRI = _lower_tri_indices(6)


class CovarianceKind(str, enum.Enum):
    """How a covariance was derived at an output epoch.

    Subclasses ``str`` so values serialize directly into the
    ``TaggedCovariances.kind`` string column. The integer codes match
    the C ABI ``EMPYREAN_COVARIANCE_KIND_*`` order.
    """

    LINEAR = "linear"
    """Linear STM mapping ``Φ Σ₀ Φᵀ`` (code 0)."""
    SECOND_ORDER = "second_order"
    """Park–Scheeres second-order (Jet2 STT) correction (code 1)."""
    THIRD_ORDER = "third_order"
    """Third-order (Jet3 STT3) extension (code 2)."""
    MIXTURE = "mixture"
    """Adaptive Gaussian Mixture, moment-collapsed (code 3)."""
    MONTE_CARLO = "monte_carlo"
    """Monte Carlo sample covariance (code 4)."""
    SIGMA_POINT = "sigma_point"
    """Sigma-point sample covariance — the second moment of the
    propagated canonical 2N+1 sigma-point set (code 5)."""


class CovarianceQuality(str, enum.Enum):
    """Definiteness of a tagged covariance matrix.

    The associated ``min_eig`` (NaN for positive-definite) rides
    alongside in the ``quality_min_eig`` column / dataclass field.
    """

    POSITIVE_DEFINITE = "positive_definite"
    """All eigenvalues positive within round-off (code 0)."""
    INDEFINITE = "indefinite"
    """At least one meaningfully negative eigenvalue (code 1)."""
    REPAIRED = "repaired"
    """Explicitly repaired to PSD; ``min_eig`` is the value before
    repair (code 2)."""


class TargetFunctional(str, enum.Enum):
    """The functional a tagged covariance's second moment describes."""

    CARTESIAN_STATE = "cartesian_state"
    """Generic Cartesian-state second moment (code 0)."""
    CLOSE_APPROACH_MISS_DISTANCE = "close_approach_miss_distance"
    """Tied to the close-approach miss-distance functional, not a
    generic state σ (code 1)."""


# Integer-code → enum decoders. The codes are the wire values the
# Rust extension emits (matching the C ABI EMPYREAN_* constants).
_KIND_BY_CODE = {
    0: CovarianceKind.LINEAR,
    1: CovarianceKind.SECOND_ORDER,
    2: CovarianceKind.THIRD_ORDER,
    3: CovarianceKind.MIXTURE,
    4: CovarianceKind.MONTE_CARLO,
    5: CovarianceKind.SIGMA_POINT,
}
_QUALITY_BY_CODE = {
    0: CovarianceQuality.POSITIVE_DEFINITE,
    1: CovarianceQuality.INDEFINITE,
    2: CovarianceQuality.REPAIRED,
}
_TARGET_BY_CODE = {
    0: TargetFunctional.CARTESIAN_STATE,
    1: TargetFunctional.CLOSE_APPROACH_MISS_DISTANCE,
}


@dataclass
class TaggedCovariance:
    """Provenance-tagged covariance at a single ``(orbit, epoch)``.

    The ergonomic per-epoch view yielded by
    :meth:`~empyrean.PropagationResult.tagged_covariance_series`. The
    matrix is a contiguous ``(6, 6)`` array and the enums are decoded.

    Attributes
    ----------
    epoch_mjd_tdb : float
        Epoch of this covariance (MJD TDB).
    state : np.ndarray
        Co-located propagated nominal state ``[x, y, z, vx, vy, vz]``
        (AU, AU/day), shape ``(6,)``.
    matrix : np.ndarray
        The 6×6 covariance, contiguous, shape ``(6, 6)``.
    kind : CovarianceKind
        How the covariance was derived.
    quality : CovarianceQuality
        Definiteness of ``matrix``.
    quality_min_eig : float
        Minimum eigenvalue for indefinite / repaired matrices; NaN when
        positive-definite.
    mc_seed : int, optional
        Monte-Carlo run seed (set only when ``kind`` is
        :attr:`CovarianceKind.MONTE_CARLO`).
    mean_shift_prop : np.ndarray, optional
        Second-order propagation mean shift ``δμ_prop`` (zero at t₀),
        shape ``(6,)`` or ``None``.
    mean_shift_input : np.ndarray, optional
        OD-estimator mean shift ``δμ₀`` (nonzero at t₀), shape ``(6,)``
        or ``None``.
    non_grav : np.ndarray
        ``[A1, A2, A3]`` non-grav solved flags, shape ``(3,)`` bool.
    thrust_segments : int
        Thrust Δv segments solved for.
    solved_width : int
        Solved width (6 / 9 / 12 / …) — the conservative-vs-optimistic
        information-product axis.
    target_functional : TargetFunctional
        The functional this second moment describes.
    origin : str
        Canonical origin (center body) name of the basis.
    frame : str
        Reference frame of the basis (canonical name, e.g. ``"icrf"``).
    """

    epoch_mjd_tdb: float
    state: np.ndarray
    matrix: np.ndarray
    kind: CovarianceKind
    quality: CovarianceQuality
    quality_min_eig: float
    mc_seed: int | None
    mean_shift_prop: np.ndarray | None
    mean_shift_input: np.ndarray | None
    non_grav: np.ndarray
    thrust_segments: int
    solved_width: int
    target_functional: TargetFunctional
    origin: str
    frame: str

    @property
    def corrected_mean(self) -> np.ndarray:
        """The corrected mean: ``state + δμ_prop + δμ_input``.

        Mean shifts default to zero when absent, so this always returns
        a ``(6,)`` array.
        """
        out = np.asarray(self.state, dtype=np.float64).copy()
        if self.mean_shift_prop is not None:
            out = out + self.mean_shift_prop
        if self.mean_shift_input is not None:
            out = out + self.mean_shift_input
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
    The 6×6 matrix rides as 21 lower-triangular ``cov_{i}_{j}`` columns
    (same layout as
    :class:`~empyrean.coordinates.covariance.CartesianCovariance`). The
    co-located nominal state and the optional mean-shift vectors ride as
    six scalar columns each, paired with a presence flag for the
    optional vectors.

    ``has_tagged`` is ``False`` on rows where the underlying orbit
    carried no covariance — those rows are zero-filled placeholders that
    keep the table aligned 1:1 with the propagated states.
    """

    orbit_id = qv.LargeStringColumn()
    """Orbit primary key (matches the input ``Orbits.orbit_id``)."""
    object_id = qv.LargeStringColumn(nullable=True)
    """Object metadata label, if carried on the input orbit."""
    epoch_mjd_tdb = qv.Float64Column()
    """Output epoch (MJD TDB)."""

    # Co-located propagated nominal state [x, y, z, vx, vy, vz].
    state_x = qv.Float64Column()
    state_y = qv.Float64Column()
    state_z = qv.Float64Column()
    state_vx = qv.Float64Column()
    state_vy = qv.Float64Column()
    state_vz = qv.Float64Column()

    kind = qv.LargeStringColumn()
    """Resolved covariance kind (``CovarianceKind`` value)."""
    quality = qv.LargeStringColumn()
    """Definiteness (``CovarianceQuality`` value)."""
    quality_min_eig = qv.Float64Column(nullable=True)
    """Minimum eigenvalue for indefinite / repaired matrices; null
    (NaN) when positive-definite."""

    mc_seed = qv.UInt64Column(nullable=True)
    """Monte-Carlo run seed; null unless ``kind`` is ``monte_carlo``."""

    # Second-order propagation mean shift δμ_prop (zero at t₀).
    mean_shift_prop_x = qv.Float64Column(nullable=True)
    mean_shift_prop_y = qv.Float64Column(nullable=True)
    mean_shift_prop_z = qv.Float64Column(nullable=True)
    mean_shift_prop_vx = qv.Float64Column(nullable=True)
    mean_shift_prop_vy = qv.Float64Column(nullable=True)
    mean_shift_prop_vz = qv.Float64Column(nullable=True)
    has_mean_shift_prop = qv.BooleanColumn()
    """Whether ``mean_shift_prop_*`` carries a value on this row."""

    # OD-estimator mean shift δμ₀ (nonzero at t₀).
    mean_shift_input_x = qv.Float64Column(nullable=True)
    mean_shift_input_y = qv.Float64Column(nullable=True)
    mean_shift_input_z = qv.Float64Column(nullable=True)
    mean_shift_input_vx = qv.Float64Column(nullable=True)
    mean_shift_input_vy = qv.Float64Column(nullable=True)
    mean_shift_input_vz = qv.Float64Column(nullable=True)
    has_mean_shift_input = qv.BooleanColumn()
    """Whether ``mean_shift_input_*`` carries a value on this row."""

    # [A1, A2, A3] non-grav solved flags.
    non_grav_a1 = qv.BooleanColumn()
    non_grav_a2 = qv.BooleanColumn()
    non_grav_a3 = qv.BooleanColumn()

    thrust_segments = qv.UInt32Column()
    """Thrust Δv segments solved for."""
    solved_width = qv.UInt32Column()
    """Solved width (6 / 9 / 12 / …)."""
    target_functional = qv.LargeStringColumn()
    """The functional this second moment describes
    (``TargetFunctional`` value)."""

    origin = qv.LargeStringColumn()
    """Canonical origin (center body) name of the basis."""
    frame = qv.LargeStringColumn()
    """Reference frame of the basis (canonical name)."""

    # 6×6 covariance as 21 lower-triangular columns.
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

        Rows with ``has_tagged=False`` come back zero-filled. Works on
        the full table or a filtered single chain.
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
        mats = self.matrices()
        epochs = self.column("epoch_mjd_tdb").to_numpy(zero_copy_only=False)
        kinds = self.column("kind").to_pylist()
        qualities = self.column("quality").to_pylist()
        min_eigs = self.column("quality_min_eig").to_numpy(zero_copy_only=False)
        targets = self.column("target_functional").to_pylist()
        origins = self.column("origin").to_pylist()
        frames = self.column("frame").to_pylist()
        thrust = self.column("thrust_segments").to_numpy(zero_copy_only=False)
        widths = self.column("solved_width").to_numpy(zero_copy_only=False)
        mc_seeds = self.column("mc_seed").to_pylist()
        has_prop = self.column("has_mean_shift_prop").to_numpy(zero_copy_only=False)
        has_input = self.column("has_mean_shift_input").to_numpy(zero_copy_only=False)

        state = np.column_stack(
            [self.column(f"state_{lab}").to_numpy(zero_copy_only=False) for lab in _STATE_LABELS]
        )
        prop = np.column_stack(
            [
                self.column(f"mean_shift_prop_{lab}").to_numpy(zero_copy_only=False)
                for lab in _STATE_LABELS
            ]
        )
        inp = np.column_stack(
            [
                self.column(f"mean_shift_input_{lab}").to_numpy(zero_copy_only=False)
                for lab in _STATE_LABELS
            ]
        )
        ng = np.column_stack(
            [self.column(f"non_grav_a{k}").to_numpy(zero_copy_only=False) for k in (1, 2, 3)]
        )

        out: list[TaggedCovariance] = []
        for i in range(n):
            out.append(
                TaggedCovariance(
                    epoch_mjd_tdb=float(epochs[i]),
                    state=np.ascontiguousarray(state[i], dtype=np.float64),
                    matrix=np.ascontiguousarray(mats[i], dtype=np.float64),
                    kind=CovarianceKind(kinds[i]),
                    quality=CovarianceQuality(qualities[i]),
                    quality_min_eig=float(min_eigs[i]),
                    mc_seed=int(mc_seeds[i]) if mc_seeds[i] is not None else None,
                    mean_shift_prop=(
                        np.ascontiguousarray(prop[i], dtype=np.float64)
                        if bool(has_prop[i])
                        else None
                    ),
                    mean_shift_input=(
                        np.ascontiguousarray(inp[i], dtype=np.float64)
                        if bool(has_input[i])
                        else None
                    ),
                    non_grav=np.asarray(ng[i], dtype=bool),
                    thrust_segments=int(thrust[i]),
                    solved_width=int(widths[i]),
                    target_functional=TargetFunctional(targets[i]),
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
) -> TaggedCovariances | None:
    """Build a :class:`TaggedCovariances` table from the Rust result.

    ``result`` is the dict returned by the ``_propagate`` extension when
    ``with_tagged_covariance=True``; ``orbit_ids`` / ``object_ids`` /
    ``epochs_mjd_tdb`` are the already-flattened per-row arrays from the
    states (length ``n``, orbit-major). Returns ``None`` if the
    extension produced no tagged sub-dict.
    """
    from empyrean._convert import int_to_frame, naif_to_origin

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
    mean_shift_input = np.asarray(tagged["mean_shift_input"], dtype=np.float64)
    has_mean_shift_input = np.asarray(tagged["has_mean_shift_input"], dtype=bool)
    quality_codes = np.asarray(tagged["quality"])
    quality_min_eig = np.asarray(tagged["quality_min_eig"], dtype=np.float64)
    non_grav = np.asarray(tagged["non_grav"], dtype=bool)  # (n, 3)
    thrust_segments = np.asarray(tagged["thrust_segments"], dtype=np.uint32)
    solved_width = np.asarray(tagged["solved_width"], dtype=np.uint32)
    target_codes = np.asarray(tagged["target_functional"])
    origin_codes = np.asarray(tagged["origin"], dtype=np.int64)
    frame_codes = np.asarray(tagged["frame"], dtype=np.int64)
    has_tagged = np.asarray(tagged["has_tagged"], dtype=bool)

    n = len(orbit_ids)

    kind_strs = [_KIND_BY_CODE[int(c)].value for c in kind_codes]
    quality_strs = [_QUALITY_BY_CODE[int(c)].value for c in quality_codes]
    target_strs = [_TARGET_BY_CODE[int(c)].value for c in target_codes]
    origin_strs = [naif_to_origin(int(o)) for o in origin_codes]
    frame_strs = [int_to_frame(int(f)).value for f in frame_codes]

    # mc_seed → null where absent; min_eig stays NaN for PD (nullable).
    mc_seed_col = [int(mc_seed[i]) if has_mc_seed[i] else None for i in range(n)]

    kwargs: dict[str, Any] = {
        "orbit_id": orbit_ids,
        "object_id": object_ids,
        "epoch_mjd_tdb": np.asarray(epochs_mjd_tdb, dtype=np.float64),
        "state_x": state[:, 0],
        "state_y": state[:, 1],
        "state_z": state[:, 2],
        "state_vx": state[:, 3],
        "state_vy": state[:, 4],
        "state_vz": state[:, 5],
        "kind": kind_strs,
        "quality": quality_strs,
        "quality_min_eig": quality_min_eig,
        "mc_seed": mc_seed_col,
        "mean_shift_prop_x": mean_shift_prop[:, 0],
        "mean_shift_prop_y": mean_shift_prop[:, 1],
        "mean_shift_prop_z": mean_shift_prop[:, 2],
        "mean_shift_prop_vx": mean_shift_prop[:, 3],
        "mean_shift_prop_vy": mean_shift_prop[:, 4],
        "mean_shift_prop_vz": mean_shift_prop[:, 5],
        "has_mean_shift_prop": has_mean_shift_prop,
        "mean_shift_input_x": mean_shift_input[:, 0],
        "mean_shift_input_y": mean_shift_input[:, 1],
        "mean_shift_input_z": mean_shift_input[:, 2],
        "mean_shift_input_vx": mean_shift_input[:, 3],
        "mean_shift_input_vy": mean_shift_input[:, 4],
        "mean_shift_input_vz": mean_shift_input[:, 5],
        "has_mean_shift_input": has_mean_shift_input,
        "non_grav_a1": non_grav[:, 0],
        "non_grav_a2": non_grav[:, 1],
        "non_grav_a3": non_grav[:, 2],
        "thrust_segments": thrust_segments,
        "solved_width": solved_width,
        "target_functional": target_strs,
        "origin": origin_strs,
        "frame": frame_strs,
        "has_tagged": has_tagged,
    }
    # 6×6 matrix → 21 lower-tri columns.
    for name, (i, j) in zip(_COV_NAMES, _LOWER_TRI, strict=False):
        kwargs[name] = matrix[:, i, j]

    return TaggedCovariances.from_kwargs(**kwargs)
