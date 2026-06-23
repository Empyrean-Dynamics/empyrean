"""Event types from orbit propagation.

Each event type is a quivr Table with ``orbit_id`` (primary key) and
``object_id`` (optional metadata) columns. Tables are grouped inside
the :class:`Events` dataclass returned by propagation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from empyrean.coordinates.enums import Origin

if TYPE_CHECKING:
    from typing import Protocol

    class _IsIn(Protocol):
        """Static signature for ``pyarrow.compute.is_in``.

        pyarrow ships ``py.typed`` but generates every ``pyarrow.compute``
        function dynamically at import, so it declares no static signature for
        ``is_in``. Bind it to this precisely-typed callable so the call site
        type-checks without resorting to ``Any`` or ``# type: ignore``.
        """

        def __call__(
            self, values: pa.ChunkedArray, /, *, value_set: pa.Array
        ) -> pa.BooleanArray: ...

    _is_in: _IsIn
else:
    _is_in = pc.is_in

_T = TypeVar("_T", bound=qv.Table)


def _select_event_table_by_orbit(table: _T, orbit_ids: list[str]) -> _T:
    """Filter a per-event-type quivr Table to rows matching the given orbit_ids."""
    if len(table) == 0:
        return table
    mask = _is_in(table.column("orbit_id"), value_set=pa.array(orbit_ids))
    return table.apply_mask(mask)


# ── Summary ──────────────────────────────────────────────────


class EventSummary(qv.Table):
    """One-row-per-event summary across all event types."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    event_type = qv.LargeStringColumn()
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()  # MJD TDB


# ── Close approaches ────────────────────────────────────────


class CloseApproachStarts(qv.Table):
    """Close approach zone entry events."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    distance_km = qv.Float64Column()


class CloseApproachEnds(qv.Table):
    """Close approach zone exit events."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    distance_km = qv.Float64Column()


# ── Periapses ────────────────────────────────────────────────


class Periapses(qv.Table):
    """Periapsis (closest approach) events within a CA zone."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    distance_km = qv.Float64Column()
    relative_velocity_au_day = qv.Float64Column()
    relative_x = qv.Float64Column()
    relative_y = qv.Float64Column()
    relative_z = qv.Float64Column()
    relative_vx = qv.Float64Column()
    relative_vy = qv.Float64Column()
    relative_vz = qv.Float64Column()


# ── Impacts ──────────────────────────────────────────────────


class Impacts(qv.Table):
    """Nominal impact events (body surface intersection)."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    latitude_deg = qv.Float64Column(nullable=True)
    longitude_deg = qv.Float64Column(nullable=True)
    altitude_km = qv.Float64Column(nullable=True)


# ── Possible impacts ────────────────────────────────────────


class PossibleImpacts(qv.Table):
    """Probabilistic impact assessments from covariance analysis."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    miss_distance_au = qv.Float64Column()
    miss_distance_km = qv.Float64Column()
    effective_radius_au = qv.Float64Column()
    effective_radius_km = qv.Float64Column()
    sigma_distance_au = qv.Float64Column()
    ip_linear = qv.Float64Column()
    relative_velocity_au_day = qv.Float64Column()
    ip_second_order = qv.Float64Column(nullable=True)
    nonlinearity = qv.Float64Column(nullable=True)
    ip_agm = qv.Float64Column(nullable=True)
    ip_mc = qv.Float64Column(nullable=True)


# ── Atmospheric ──────────────────────────────────────────────


class AtmosphericEntries(qv.Table):
    """Atmospheric entry events (Karman line inbound crossing).

    ``distance_au`` is the body-CENTER crossing distance (the Karman
    radius). ``altitude_km`` is the true altitude above the reference
    ellipsoid from the planetodetic ground track, and
    ``latitude_deg`` / ``longitude_deg`` the sub-point — all three null
    when the ground track is unresolved (they are NOT
    ``distance_au`` relabelled). ``relative_velocity_au_day`` is the
    body-relative speed at entry.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    altitude_km = qv.Float64Column(nullable=True)
    relative_velocity_au_day = qv.Float64Column(nullable=True)
    latitude_deg = qv.Float64Column(nullable=True)
    longitude_deg = qv.Float64Column(nullable=True)


class AtmosphericExits(qv.Table):
    """Atmospheric exit events."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()


# ── Capture ──────────────────────────────────────────────────


class CaptureStarts(qv.Table):
    """Temporary gravitational capture start events."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    distance_km = qv.Float64Column()
    relative_velocity_au_day = qv.Float64Column()
    two_body_energy = qv.Float64Column()
    jacobi_constant = qv.Float64Column(nullable=True)
    jacobi_constant_sigma = qv.Float64Column(nullable=True)
    jacobi_constant_l1 = qv.Float64Column(nullable=True)
    jacobi_constant_l2 = qv.Float64Column(nullable=True)


class CaptureEnds(qv.Table):
    """Temporary gravitational capture end (escape) events."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    distance_au = qv.Float64Column()
    distance_km = qv.Float64Column()
    relative_velocity_au_day = qv.Float64Column()
    two_body_energy = qv.Float64Column()
    jacobi_constant = qv.Float64Column(nullable=True)
    jacobi_constant_sigma = qv.Float64Column(nullable=True)
    jacobi_constant_l1 = qv.Float64Column(nullable=True)
    jacobi_constant_l2 = qv.Float64Column(nullable=True)
    n_periapses = qv.Int32Column()


# ── Shadow ───────────────────────────────────────────────────


class ShadowEntries(qv.Table):
    """Shadow zone entry events (Sun occluded by a body)."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    shadow_fraction = qv.Float64Column()
    illumination = qv.Float64Column()


class ShadowExits(qv.Table):
    """Shadow zone exit events (Sun no longer occluded)."""

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn()
    epoch = qv.Float64Column()
    shadow_fraction = qv.Float64Column()
    illumination = qv.Float64Column()


# ── Covariance regime change ─────────────────────────────────


class CovarianceRegimeChanges(qv.Table):
    """Covariance-regime transitions from ``UncertaintyMethod.AUTO``.

    Each row records a close-approach-window boundary where the resolved
    covariance kind changed (e.g. ``linear`` -> ``second_order``) because
    the local nonlinearity ``kappa`` crossed a band edge. This is the
    audit trail behind Auto's per-window covariance-kind decisions; the
    ``previous_kind`` / ``resolved_kind`` strings are
    :class:`~empyrean.propagation.tagged_covariance.CovarianceKind` values.
    """

    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    body = qv.LargeStringColumn(nullable=True)
    epoch = qv.Float64Column()
    previous_kind = qv.LargeStringColumn(nullable=True)
    resolved_kind = qv.LargeStringColumn(nullable=True)
    kappa = qv.Float64Column(nullable=True)
    threshold_below = qv.Float64Column(nullable=True)
    threshold_above = qv.Float64Column(nullable=True)


# ── Container ────────────────────────────────────────────────


@dataclass
class Events:
    """All events detected during propagation, grouped by type."""

    summary: EventSummary
    close_approach_starts: CloseApproachStarts
    close_approach_ends: CloseApproachEnds
    periapses: Periapses
    impacts: Impacts
    possible_impacts: PossibleImpacts
    atmospheric_entries: AtmosphericEntries
    atmospheric_exits: AtmosphericExits
    capture_starts: CaptureStarts
    capture_ends: CaptureEnds
    shadow_entries: ShadowEntries
    shadow_exits: ShadowExits
    covariance_regime_changes: CovarianceRegimeChanges

    def count_by_type(self) -> dict[str, int]:
        """Per-event-type row count, including ``summary``.

        Useful as a quick triage view — `events.count_by_type()` answers
        "did the propagation produce any close approaches / impacts /
        captures?" without a multi-line group-by.
        """
        return {f.name: len(getattr(self, f.name)) for f in fields(self)}

    def select_orbit(self, orbit_ids: str | Iterable[str]) -> Events:
        """Return a new :class:`Events` containing only rows whose
        ``orbit_id`` matches one of the requested IDs.

        Filters every per-event-type table at once, including the
        ``summary``, so callers don't have to remember which 13 tables
        to thread the filter through.
        """
        if isinstance(orbit_ids, str):
            ids: list[str] = [orbit_ids]
        else:
            ids = list(orbit_ids)
        return Events(
            **{
                f.name: _select_event_table_by_orbit(getattr(self, f.name), ids)
                for f in fields(self)
            }
        )


# ── Configuration ────────────────────────────────────────────


@dataclass
class EventConfig:
    """Configuration for event detection during propagation.

    Parameters
    ----------
    close_approaches : bool
        Detect close approach periapses. Default True.
    impacts : bool
        Detect nominal impacts. Default True.
    atmospheric : bool
        Detect atmospheric entry/exit. Default True.
    possible_impacts : bool
        Compute impact probabilities. Default True.
    shadow_events : bool
        Detect shadow entry/exit. Default True.
    body_filter : list[Origin | str], optional
        Restrict monitoring to specific bodies. Pass a list of
        :class:`Origin` (e.g. ``[Origin.EARTH, Origin.MOON]``) or the
        canonical names (e.g. ``["Earth", "Moon"]``). ``None`` means
        all bodies.
    dense_output : bool
        Insert dense-state points around close approaches via the
        integrator's per-step interpolant. Auto-enables
        :attr:`AdvancedIntegratorConfig.cache_integrator_steps`.
        Default False.
    dense_output_cadence_days : float
        Cadence (days) of dense output points around close approaches.
        Default 5 minutes (= ``5.0 / 1440.0``).
    """

    close_approaches: bool = True
    impacts: bool = True
    atmospheric: bool = True
    possible_impacts: bool = True
    shadow_events: bool = True
    body_filter: list[Origin | str] | None = None
    dense_output: bool = False
    dense_output_cadence_days: float = 5.0 / 1440.0
