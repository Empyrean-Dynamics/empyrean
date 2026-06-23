"""Propagation result type.

:class:`PropagationResult` bundles the propagated states, dynamical
events, and (when available) per-orbit sensitivity chains. The
:meth:`PropagationResult.to_dir` / :meth:`PropagationResult.from_dir`
helpers persist the whole result as a directory of Parquet files +
a ``sensitivity/`` subdir.
"""

import os
from dataclasses import dataclass
from typing import TypeVar

import quivr as qv

from empyrean.ephemeris.sensitivity import StateSensitivities
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.events import (
    AtmosphericEntries,
    AtmosphericExits,
    CaptureEnds,
    CaptureStarts,
    CloseApproachEnds,
    CloseApproachStarts,
    CovarianceRegimeChanges,
    Events,
    EventSummary,
    Impacts,
    Periapses,
    PossibleImpacts,
    ShadowEntries,
    ShadowExits,
)
from empyrean.propagation.tagged_covariance import (
    TaggedCovariance,
    TaggedCovariances,
)

# Table name → quivr class. Used by from_dir to map directory entries
# back into typed event tables.
_EVENT_TABLE_MAP = {
    "summary": EventSummary,
    "close_approach_starts": CloseApproachStarts,
    "close_approach_ends": CloseApproachEnds,
    "periapses": Periapses,
    "impacts": Impacts,
    "possible_impacts": PossibleImpacts,
    "atmospheric_entries": AtmosphericEntries,
    "atmospheric_exits": AtmosphericExits,
    "capture_starts": CaptureStarts,
    "capture_ends": CaptureEnds,
    "shadow_entries": ShadowEntries,
    "shadow_exits": ShadowExits,
    "covariance_regime_changes": CovarianceRegimeChanges,
}

_T = TypeVar("_T", bound=qv.Table)


def _load_event_table(table_cls: type[_T], path: str, name: str) -> _T:
    """Load one event table from ``{path}/{name}.parquet``, or empty.

    Preserves the precise table type of ``table_cls`` so callers receive
    a value typed as the concrete event-table subclass rather than the
    erased :class:`quivr.Table` base.
    """
    fpath = os.path.join(path, f"{name}.parquet")
    if os.path.exists(fpath):
        return table_cls.from_parquet(fpath)
    return table_cls.empty()


@dataclass
class PropagationResult:
    """Result of orbit propagation.

    Attributes
    ----------
    states : CartesianOrbits
        Propagated Cartesian states with optional covariance.
    events : Events
        All detected dynamical events, grouped by type.
    sensitivity : StateSensitivities, optional
        Flat per-``(orbit, epoch)`` sensitivity table — STMs and
        optional STTs. ``None`` when the propagation method did not
        produce sensitivities (Monte Carlo / SigmaPoint).
    tagged_covariance : TaggedCovariances, optional
        Flat per-``(orbit, epoch)`` provenance-tagged, resolved-kind
        covariance readback — the honest covariance that distinguishes
        a second-order close-approach ellipsoid from the bare linear
        ``Φ Σ₀ Φᵀ`` mapping on ``states``. ``None`` unless
        :func:`~empyrean.propagate` was called with
        ``tagged_covariance=True``. Use
        :meth:`tagged_covariance_series` for the per-epoch view of one
        orbit.
    """

    states: CartesianOrbits
    events: Events
    sensitivity: StateSensitivities | None = None
    tagged_covariance: TaggedCovariances | None = None

    def tagged_covariance_series(self, orbit_index: int) -> list[TaggedCovariance]:
        """Per-epoch tagged covariance for one orbit, as dataclasses.

        Yields one :class:`~empyrean.propagation.tagged_covariance.TaggedCovariance`
        per output epoch of the orbit at ``orbit_index`` (orbit-major,
        matching the input order), with each matrix re-materialized as a
        contiguous ``(6, 6)`` array and the provenance enums decoded.

        Parameters
        ----------
        orbit_index : int
            Zero-based index into the input orbits (orbit-major output
            order).

        Returns
        -------
        list[TaggedCovariance]
            The orbit's tagged covariance at every output epoch.

        Raises
        ------
        ValueError
            If the result carries no tagged covariance (propagate was
            not called with ``tagged_covariance=True``), or
            ``orbit_index`` is out of range.
        """
        if self.tagged_covariance is None:
            raise ValueError(
                "this result has no tagged covariance; call "
                "propagate(..., tagged_covariance=True) to populate it"
            )
        oids = self.tagged_covariance.orbit_ids_unique()
        if orbit_index < 0 or orbit_index >= len(oids):
            raise ValueError(
                f"orbit_index {orbit_index} out of range (result holds {len(oids)} orbits)"
            )
        chain = self.tagged_covariance.select("orbit_id", oids[orbit_index])
        return chain.to_series()

    def to_dir(self, path: str) -> None:
        """Write a propagation result to a directory of Parquet files."""
        os.makedirs(path, exist_ok=True)

        self.states.to_parquet(os.path.join(path, "states.parquet"))

        for name, table in [
            ("summary", self.events.summary),
            ("close_approach_starts", self.events.close_approach_starts),
            ("close_approach_ends", self.events.close_approach_ends),
            ("periapses", self.events.periapses),
            ("impacts", self.events.impacts),
            ("possible_impacts", self.events.possible_impacts),
            ("atmospheric_entries", self.events.atmospheric_entries),
            ("atmospheric_exits", self.events.atmospheric_exits),
            ("capture_starts", self.events.capture_starts),
            ("capture_ends", self.events.capture_ends),
            ("shadow_entries", self.events.shadow_entries),
            ("shadow_exits", self.events.shadow_exits),
            ("covariance_regime_changes", self.events.covariance_regime_changes),
        ]:
            if len(table) > 0:
                table.to_parquet(os.path.join(path, f"{name}.parquet"))

        if self.sensitivity is not None and len(self.sensitivity) > 0:
            self.sensitivity.to_parquet(os.path.join(path, "sensitivity.parquet"))

        if self.tagged_covariance is not None and len(self.tagged_covariance) > 0:
            self.tagged_covariance.to_parquet(os.path.join(path, "tagged_covariance.parquet"))

    @classmethod
    def from_dir(cls, path: str) -> "PropagationResult":
        """Load a propagation result written by :meth:`to_dir`."""
        states_path = os.path.join(path, "states.parquet")
        states = CartesianOrbits.from_parquet(states_path)

        events = Events(
            summary=_load_event_table(EventSummary, path, "summary"),
            close_approach_starts=_load_event_table(
                CloseApproachStarts, path, "close_approach_starts"
            ),
            close_approach_ends=_load_event_table(CloseApproachEnds, path, "close_approach_ends"),
            periapses=_load_event_table(Periapses, path, "periapses"),
            impacts=_load_event_table(Impacts, path, "impacts"),
            possible_impacts=_load_event_table(PossibleImpacts, path, "possible_impacts"),
            atmospheric_entries=_load_event_table(AtmosphericEntries, path, "atmospheric_entries"),
            atmospheric_exits=_load_event_table(AtmosphericExits, path, "atmospheric_exits"),
            capture_starts=_load_event_table(CaptureStarts, path, "capture_starts"),
            capture_ends=_load_event_table(CaptureEnds, path, "capture_ends"),
            shadow_entries=_load_event_table(ShadowEntries, path, "shadow_entries"),
            shadow_exits=_load_event_table(ShadowExits, path, "shadow_exits"),
            covariance_regime_changes=_load_event_table(
                CovarianceRegimeChanges, path, "covariance_regime_changes"
            ),
        )

        sens_path = os.path.join(path, "sensitivity.parquet")
        sensitivity = (
            StateSensitivities.from_parquet(sens_path) if os.path.exists(sens_path) else None
        )

        tagged_path = os.path.join(path, "tagged_covariance.parquet")
        tagged_covariance = (
            TaggedCovariances.from_parquet(tagged_path) if os.path.exists(tagged_path) else None
        )

        return cls(
            states=states,
            events=events,
            sensitivity=sensitivity,
            tagged_covariance=tagged_covariance,
        )
