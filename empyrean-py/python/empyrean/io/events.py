"""Write events to parquet / JSON / CSV.

Flattens the per-subtype tables in :class:`~empyrean.propagation.events.Events`
into a single record capturing the common subset of fields:
orbit_id, event_type (variant tag), body, epoch, distance_au,
distance_km, relative_velocity_au_day. Subtype-specific fields
(latitude / longitude on impacts, jacobi constants on captures,
illumination on shadow events, etc.) do not round-trip through this
surface — they remain in the original :class:`Events` dataclass for
in-memory use.
"""

from typing import Any, Protocol

import numpy as np
import pyarrow as pa

from empyrean._convert import origin_to_naif
from empyrean.propagation.events import Events


class _CommonEventTable(Protocol):
    """Structural view of the columns shared by every event subtype table.

    Every per-event-type :class:`quivr.Table` in :class:`Events` exposes at
    least ``orbit_id``, ``body`` and ``epoch``; accessing such a column on a
    table *instance* yields the underlying :class:`pyarrow.Array`.
    """

    orbit_id: pa.Array
    body: pa.Array
    epoch: pa.Array

    def __len__(self) -> int: ...


def _optional_column_values(table: _CommonEventTable, name: str, n: int) -> list[float]:
    """Return [NaN, NaN, ...] when a column doesn't exist on this subtype."""
    col = getattr(table, name, None)
    if col is None:
        return [float("nan")] * n
    values: list[float] = col.to_pylist()
    return values


def _events_to_dict(
    events: Events,
) -> dict[
    str,
    list[str] | np.ndarray[Any, np.dtype[np.int32]] | np.ndarray[Any, np.dtype[np.float64]],
]:
    sources: list[tuple[str, _CommonEventTable]] = [
        ("close_approach_start", events.close_approach_starts),
        ("close_approach_end", events.close_approach_ends),
        ("periapsis", events.periapses),
        ("impact", events.impacts),
        ("possible_impact", events.possible_impacts),
        ("atmospheric_entry", events.atmospheric_entries),
        ("atmospheric_exit", events.atmospheric_exits),
        ("capture_start", events.capture_starts),
        ("capture_end", events.capture_ends),
        ("shadow_entry", events.shadow_entries),
        ("shadow_exit", events.shadow_exits),
        # Carries only the common orbit_id/body/epoch columns through this
        # flat writer (distance/velocity absent -> NaN); the kappa / kind
        # payload round-trips via PropagationResult.to_dir, not here.
        # Without this entry the whole sub-table vanished silently.
        ("covariance_regime_change", events.covariance_regime_changes),
    ]
    orbit_ids: list[str] = []
    event_types: list[str] = []
    bodies: list[str] = []
    epochs: list[float] = []
    distance_au: list[float] = []
    distance_km: list[float] = []
    rel_v: list[float] = []
    for name, table in sources:
        if table is None or len(table) == 0:
            continue
        m = len(table)
        orbit_ids.extend(table.orbit_id.to_pylist())
        event_types.extend([name] * m)
        bodies.extend(table.body.to_pylist())
        epochs.extend(table.epoch.to_pylist())
        distance_au.extend(_optional_column_values(table, "distance_au", m))
        distance_km.extend(_optional_column_values(table, "distance_km", m))
        rel_v.extend(_optional_column_values(table, "relative_velocity_au_day", m))

    # The on-disk writer wants per-row body identifiers as integers;
    # reconstruct them from the canonical body strings so the column
    # shape is unchanged. Empty body strings → -1 (non-body event).
    body_naif_ids = [origin_to_naif(b) if b else -1 for b in bodies]
    return {
        "orbit_ids": orbit_ids,
        "event_types": event_types,
        "bodies": bodies,
        "body_naif_ids": np.asarray(body_naif_ids, dtype=np.int32),
        "epochs": np.asarray(epochs, dtype=np.float64),
        "distance_au": np.asarray(distance_au, dtype=np.float64),
        "distance_km": np.asarray(distance_km, dtype=np.float64),
        "relative_velocity_au_day": np.asarray(rel_v, dtype=np.float64),
    }


def write_events_parquet(path: str, events: Events) -> None:
    """Write a flattened :class:`Events` to parquet."""
    from empyrean._empyrean_rs import _write_events_parquet

    _write_events_parquet(path, _events_to_dict(events))


def write_events_json(path: str, events: Events) -> None:
    """Write a flattened :class:`Events` to JSON."""
    from empyrean._empyrean_rs import _write_events_json

    _write_events_json(path, _events_to_dict(events))


def write_events_csv(path: str, events: Events) -> None:
    """Write a flattened :class:`Events` to CSV."""
    from empyrean._empyrean_rs import _write_events_csv

    _write_events_csv(path, _events_to_dict(events))
