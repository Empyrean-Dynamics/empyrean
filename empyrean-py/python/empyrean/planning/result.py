"""Observation-planning configuration and result types.

Mirrors the Rust wrapper's ``empyrean::PlanningConfig`` /
``empyrean::PlanResult`` field-for-field. ``PlanningConfig()`` defaults
match ``PlanningConfig::default()`` on the Rust side, so a
default-constructed config round-trips through the C ABI as a no-op.

A plan answers one question: for an orbit that already carries a
covariance, how much would each candidate follow-up observation tighten
it? :class:`PlanMetrics` brackets the campaign with a prior and a
posterior row, and :class:`PlanCandidates` carries the per-observation
marginal gain.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

from empyrean.coordinates.epoch import Epochs
from empyrean.propagation.config import ForceModelTier

# pyarrow's compute functions are generated at runtime into the
# ``pyarrow.compute`` module namespace, so the bundled type stubs do not
# declare them. Bind the ones we use to precisely-typed module-level
# aliases via the module ``__dict__`` (the alias is the same function
# object — runtime behavior is unchanged) so every call site gets a real
# signature.
_is_in: Callable[..., pa.BooleanArray] = pc.__dict__["is_in"]

# JSON-like value type for the wire dicts marshaled across the C ABI
# boundary (str / numeric / bool / None leaves, plus nested dicts and
# lists of the same).
WireValue: TypeAlias = str | int | float | bool | list["WireValue"] | dict[str, "WireValue"] | None


# ── Enums ────────────────────────────────────────────────────


class PlannedObservationKind(str, Enum):
    """What a :class:`PlannedObservation` would measure."""

    OPTICAL = "optical"
    """RA/Dec astrometry from a registered observatory."""
    RADAR = "radar"
    """Delay (range) and/or Doppler (range-rate) from a radar dish."""


class RadarMode(str, Enum):
    """Which radar observable(s) a candidate would measure."""

    DELAY = "delay"
    """Round-trip delay only — a range measurement."""
    DOPPLER = "doppler"
    """Doppler shift only — a range-rate measurement."""
    BOTH = "both"
    """Both delay and Doppler."""


class RadarStation(str, Enum):
    """A radar dish the planner can schedule against."""

    GOLDSTONE_DSS14 = "goldstone_dss14"
    """Goldstone DSS-14, the 70 m transmit/receive antenna."""
    GREEN_BANK = "green_bank"
    """The Green Bank Telescope, receive-only — pair it with a
    transmitting dish for a bistatic observation."""
    ARECIBO = "arecibo"
    """Arecibo. Collapsed in 2020; accepted only so a historical plan can
    be described, and rejected if you try to schedule it."""


# Keyed by both the enum member and its bare wire string so a raw string
# works everywhere an enum does.
_KIND_TO_INT: dict[PlannedObservationKind | str, int] = {
    PlannedObservationKind.OPTICAL: 0,
    PlannedObservationKind.RADAR: 1,
    "optical": 0,
    "radar": 1,
}

_RADAR_MODE_TO_INT: dict[RadarMode | str, int] = {
    RadarMode.DELAY: 0,
    RadarMode.DOPPLER: 1,
    RadarMode.BOTH: 2,
    "delay": 0,
    "doppler": 1,
    "both": 2,
}

_RADAR_STATION_TO_INT: dict[RadarStation | str, int] = {
    RadarStation.GOLDSTONE_DSS14: 0,
    RadarStation.GREEN_BANK: 1,
    RadarStation.ARECIBO: 2,
    "goldstone_dss14": 0,
    "green_bank": 1,
    "arecibo": 2,
}

# Inverse maps for the tags that come BACK from the engine. Without
# them, an unrecognized tag would have to be coerced to some default —
# silently relabelling the row rather than surfacing the mismatch.
_INT_TO_KIND: dict[int, str] = {0: "optical", 1: "radar"}

# -1 is the engine's "not a radar candidate" sentinel and maps to None,
# never to a mode.
_INT_TO_RADAR_MODE: dict[int, str] = {0: "delay", 1: "doppler", 2: "both"}

# ── Campaign-stage labels ────────────────────────────────────
#
# Stable text labels for the ``stage`` column of :class:`PlanMetrics`.
# They line up with the ``prior`` / ``posterior`` wording used
# throughout the planning surface, so a caller can filter with
# ``metrics.select("stage", STAGE_POSTERIOR)`` — or the
# :meth:`PlanMetrics.posterior` helper — without an extra mapping step.

STAGE_PRIOR = "prior"
STAGE_POSTERIOR = "posterior"


def _enum_value(v: Enum | str) -> str:
    """Accept either an Enum or a bare string; return a string."""
    return str(v.value) if isinstance(v, Enum) else str(v)


def _nan_to_null(arr: npt.NDArray[np.float64]) -> pa.Array:
    """Convert a float64 numpy array with NaN sentinels to a nullable
    pyarrow array — quivr nullable columns expect arrow nulls, not NaN,
    for downstream consumers (pandas, polars, joins, …)."""
    mask = np.isnan(arr)
    return pa.array(arr, mask=mask)


# ── Inputs ───────────────────────────────────────────────────


@dataclass
class PlannedObservation:
    """One candidate observation: when it would be taken and what it
    would measure.

    Tagged-union shape — the active fields depend on :attr:`kind`, and
    fields belonging to the *other* kind must be left at their defaults.
    Setting an inapplicable field raises ``ValueError`` at construction
    rather than being silently ignored. Prefer the
    :meth:`optical` / :meth:`radar` constructors, which set the
    discriminator for you.

    Mirrors ``empyrean::PlannedObservation``.
    """

    epoch_mjd_tdb: float
    """Planned epoch (MJD TDB) — the *receive* epoch for radar."""
    kind: PlannedObservationKind = PlannedObservationKind.OPTICAL

    # ── Optical fields ─────────────────────────────────────────
    optical_code: str = ""
    """Registered MPC observatory code (e.g. ``"F51"``)."""
    optical_sigma_arcsec: tuple[float, float] = (1.0, 1.0)
    """Assumed 1σ (RA·cosδ, Dec) astrometric uncertainty in arcsec."""

    # ── Radar fields ───────────────────────────────────────────
    radar_transmit_station: RadarStation = RadarStation.GOLDSTONE_DSS14
    """Transmitting dish."""
    radar_receive_station: RadarStation = RadarStation.GOLDSTONE_DSS14
    """Receiving dish. Equal to the transmit station = monostatic."""
    radar_mode: RadarMode = RadarMode.BOTH
    """Which observable(s) to measure."""
    radar_bandwidth_hz: float = 0.0
    """Waveform bandwidth (Hz) — sets the delay (range) σ. Must be
    positive for :attr:`RadarMode.DELAY` or :attr:`RadarMode.BOTH`."""
    radar_freq_resolution_hz: float = 0.0
    """Doppler frequency resolution (Hz) — sets the range-rate σ. Must be
    positive for :attr:`RadarMode.DOPPLER` or :attr:`RadarMode.BOTH`."""
    radar_snr: float | None = None
    """Effective SNR as a **linear power ratio, not dB**.

    ``None`` — the default — means *derive it from the link budget* over
    the ``radar_target_*`` properties and
    :attr:`radar_integration_s`. That is a different request, not a
    missing value: supply a number and the link budget is not consulted
    at all. Whatever the link budget had to assume comes back on
    :attr:`PlanCandidates.radar_provenance`, so a derived SNR is never
    silent."""
    radar_target_h_mag: float | None = None
    """Link-budget target absolute magnitude :math:`H` (mag). ``None`` =
    not known."""
    radar_target_visual_albedo: float | None = None
    """Link-budget target visual geometric albedo :math:`p_V`. ``None`` =
    not known."""
    radar_target_radar_albedo: float | None = None
    """Link-budget target radar (OC) albedo. ``None`` = not known."""
    radar_target_diameter_km: float | None = None
    """Link-budget target effective diameter (km). ``None`` = not
    known."""
    radar_target_spin_period_hours: float | None = None
    """Link-budget target rotation period (hours); caps the coherent
    integration time. ``None`` = not known."""
    radar_integration_s: float = 0.0
    """Coherent integration time (s) for the link budget. Read only when
    :attr:`radar_snr` is ``None``."""

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ``ValueError`` for any field this candidate's kind does
        not read, or any value the engine would reject.

        Called from :meth:`__post_init__` and again from
        :func:`evaluate_plan` on every candidate it was handed — this
        dataclass is mutable by design, so a field assigned after
        construction would otherwise skip the construction-time check and
        surface from a layer below as a different exception class (or,
        for an inapplicable field, not at all).
        """
        # An out-of-domain `kind` used to match neither branch below, so
        # the object was built with NO validation at all and failed much
        # later as a KeyError in the wire lowering. Coerce first, so the
        # arms are exhaustive and the refusal is the documented class.
        try:
            self.kind = PlannedObservationKind(_enum_value(self.kind))
        except ValueError:
            accepted = ", ".join(repr(k.value) for k in PlannedObservationKind)
            raise ValueError(
                f"PlannedObservation: kind must be one of {accepted} (or the "
                f"matching PlannedObservationKind member), got {self.kind!r}."
            ) from None
        if not math.isfinite(self.epoch_mjd_tdb):
            raise ValueError(
                f"PlannedObservation: epoch_mjd_tdb must be a finite MJD TDB, "
                f"got {self.epoch_mjd_tdb!r}."
            )
        if self.kind == PlannedObservationKind.OPTICAL:
            # An optical candidate reads none of the radar slots; a value
            # here would be silently inert engine-side.
            inert: list[str] = []
            if self.radar_transmit_station != RadarStation.GOLDSTONE_DSS14:
                inert.append(f"radar_transmit_station={_enum_value(self.radar_transmit_station)!r}")
            if self.radar_receive_station != RadarStation.GOLDSTONE_DSS14:
                inert.append(f"radar_receive_station={_enum_value(self.radar_receive_station)!r}")
            if self.radar_mode != RadarMode.BOTH:
                inert.append(f"radar_mode={_enum_value(self.radar_mode)!r}")
            if self.radar_bandwidth_hz != 0.0:
                inert.append(f"radar_bandwidth_hz={self.radar_bandwidth_hz!r}")
            if self.radar_freq_resolution_hz != 0.0:
                inert.append(f"radar_freq_resolution_hz={self.radar_freq_resolution_hz!r}")
            if self.radar_snr is not None:
                inert.append(f"radar_snr={self.radar_snr!r}")
            for name in (
                "radar_target_h_mag",
                "radar_target_visual_albedo",
                "radar_target_radar_albedo",
                "radar_target_diameter_km",
                "radar_target_spin_period_hours",
            ):
                value = getattr(self, name)
                if value is not None:
                    inert.append(f"{name}={value!r}")
            if self.radar_integration_s != 0.0:
                inert.append(f"radar_integration_s={self.radar_integration_s!r}")
            if inert:
                raise ValueError(
                    f"OPTICAL candidate reads only optical_code and "
                    f"optical_sigma_arcsec, but {', '.join(inert)} was set. "
                    f"Use PlannedObservation.radar(...) for a radar candidate."
                )
            if self.optical_code == "":
                raise ValueError(
                    "OPTICAL candidate: optical_code must be a non-empty MPC observatory code."
                )
            sigma = tuple(self.optical_sigma_arcsec)
            if len(sigma) != 2:
                raise ValueError(
                    f"OPTICAL candidate (optical_code={self.optical_code!r}): "
                    f"optical_sigma_arcsec must be a (ra, dec) pair, got {sigma!r}."
                )
            if not all(math.isfinite(s) and s > 0.0 for s in sigma):
                raise ValueError(
                    f"OPTICAL candidate (optical_code={self.optical_code!r}): "
                    f"optical_sigma_arcsec must be finite and > 0 arcsec, got {sigma!r}."
                )
        else:
            inert = []
            if self.optical_code != "":
                inert.append(f"optical_code={self.optical_code!r}")
            if tuple(self.optical_sigma_arcsec) != (1.0, 1.0):
                inert.append(f"optical_sigma_arcsec={tuple(self.optical_sigma_arcsec)!r}")
            if inert:
                raise ValueError(
                    f"RADAR candidate reads none of the optical fields, but "
                    f"{', '.join(inert)} was set. A radar candidate's receive "
                    f"station comes from radar_receive_station, and its "
                    f"measurement σ from the waveform and SNR. Use "
                    f"PlannedObservation.optical(...) for an optical candidate."
                )
            self.radar_transmit_station = RadarStation(_enum_value(self.radar_transmit_station))
            self.radar_receive_station = RadarStation(_enum_value(self.radar_receive_station))
            self.radar_mode = RadarMode(_enum_value(self.radar_mode))
            if self.radar_transmit_station == RadarStation.GREEN_BANK:
                raise ValueError(
                    "RADAR candidate: Green Bank is receive-only and cannot be "
                    "the transmit station; pair it with a transmitting dish "
                    "(radar_transmit_station=RadarStation.GOLDSTONE_DSS14) for "
                    "a bistatic observation."
                )
            if self.radar_snr is not None and (
                not math.isfinite(self.radar_snr) or self.radar_snr <= 0.0
            ):
                raise ValueError(
                    f"RADAR candidate: radar_snr must be a finite linear power "
                    f"ratio > 0, got {self.radar_snr!r}; pass None to derive it "
                    f"from the link budget."
                )
            # Supplying an SNR selects a different request shape, and the
            # link-budget inputs have nowhere to go in it — the engine's
            # spec is a sum type whose supplied-SNR arm carries no target
            # and no integration time. Refuse rather than drop them.
            if self.radar_snr is not None:
                unused = [
                    f"{name}={getattr(self, name)!r}"
                    for name in (
                        "radar_target_h_mag",
                        "radar_target_visual_albedo",
                        "radar_target_radar_albedo",
                        "radar_target_diameter_km",
                        "radar_target_spin_period_hours",
                    )
                    if getattr(self, name) is not None
                ]
                if self.radar_integration_s != 0.0:
                    unused.append(f"radar_integration_s={self.radar_integration_s!r}")
                if unused:
                    raise ValueError(
                        f"RADAR candidate: radar_snr={self.radar_snr!r} supplies "
                        f"the SNR directly, so the link budget never runs and "
                        f"{', '.join(unused)} would be dropped. Pass "
                        f"radar_snr=None to derive the SNR from those "
                        f"properties, or remove them."
                    )
            if not math.isfinite(self.radar_integration_s) or self.radar_integration_s < 0.0:
                raise ValueError(
                    f"RADAR candidate: radar_integration_s must be finite and "
                    f">= 0 seconds, got {self.radar_integration_s!r}."
                )
            # The waveform sets the Cramér-Rao measurement σ; a
            # non-positive value there is a non-finite or zero weight
            # three layers down. Only the modes that use it are checked.
            if self.radar_mode in (RadarMode.DELAY, RadarMode.BOTH) and not (
                math.isfinite(self.radar_bandwidth_hz) and self.radar_bandwidth_hz > 0.0
            ):
                raise ValueError(
                    f"RADAR candidate: radar_bandwidth_hz must be finite and > 0 "
                    f"for a {_enum_value(self.radar_mode)} measurement, got "
                    f"{self.radar_bandwidth_hz!r}; it sets the delay (range) σ."
                )
            if self.radar_mode in (RadarMode.DOPPLER, RadarMode.BOTH) and not (
                math.isfinite(self.radar_freq_resolution_hz) and self.radar_freq_resolution_hz > 0.0
            ):
                raise ValueError(
                    f"RADAR candidate: radar_freq_resolution_hz must be finite "
                    f"and > 0 for a {_enum_value(self.radar_mode)} measurement, "
                    f"got {self.radar_freq_resolution_hz!r}; it sets the "
                    f"range-rate σ."
                )
            for name in (
                "radar_target_h_mag",
                "radar_target_visual_albedo",
                "radar_target_radar_albedo",
                "radar_target_diameter_km",
                "radar_target_spin_period_hours",
            ):
                value = getattr(self, name)
                if value is not None and not math.isfinite(value):
                    raise ValueError(
                        f"RADAR candidate: {name} must be finite, got {value!r}; "
                        f"pass None to leave it unknown."
                    )

    @classmethod
    def optical(
        cls,
        epoch_mjd_tdb: float,
        optical_code: str,
        optical_sigma_arcsec: tuple[float, float] = (1.0, 1.0),
    ) -> PlannedObservation:
        """An optical candidate at a registered observatory.

        Mirrors ``empyrean::PlannedObservation::optical``.
        """
        return cls(
            epoch_mjd_tdb=epoch_mjd_tdb,
            kind=PlannedObservationKind.OPTICAL,
            optical_code=optical_code,
            optical_sigma_arcsec=optical_sigma_arcsec,
        )

    @classmethod
    def radar(
        cls,
        epoch_mjd_tdb: float,
        *,
        radar_transmit_station: RadarStation | str = RadarStation.GOLDSTONE_DSS14,
        radar_receive_station: RadarStation | str = RadarStation.GOLDSTONE_DSS14,
        radar_mode: RadarMode | str = RadarMode.BOTH,
        radar_bandwidth_hz: float = 0.0,
        radar_freq_resolution_hz: float = 0.0,
        radar_snr: float | None = None,
        radar_target_h_mag: float | None = None,
        radar_target_visual_albedo: float | None = None,
        radar_target_radar_albedo: float | None = None,
        radar_target_diameter_km: float | None = None,
        radar_target_spin_period_hours: float | None = None,
        radar_integration_s: float = 0.0,
    ) -> PlannedObservation:
        """A radar candidate at the given receive epoch.

        Mirrors ``empyrean::PlannedObservation::radar``. Leave
        ``radar_snr`` at ``None`` to have the effective SNR derived from
        the link budget over the ``radar_target_*`` properties and
        ``radar_integration_s``.
        """
        return cls(
            epoch_mjd_tdb=epoch_mjd_tdb,
            kind=PlannedObservationKind.RADAR,
            radar_transmit_station=RadarStation(_enum_value(radar_transmit_station)),
            radar_receive_station=RadarStation(_enum_value(radar_receive_station)),
            radar_mode=RadarMode(_enum_value(radar_mode)),
            radar_bandwidth_hz=radar_bandwidth_hz,
            radar_freq_resolution_hz=radar_freq_resolution_hz,
            radar_snr=radar_snr,
            radar_target_h_mag=radar_target_h_mag,
            radar_target_visual_albedo=radar_target_visual_albedo,
            radar_target_radar_albedo=radar_target_radar_albedo,
            radar_target_diameter_km=radar_target_diameter_km,
            radar_target_spin_period_hours=radar_target_spin_period_hours,
            radar_integration_s=radar_integration_s,
        )

    def _to_wire_dict(self) -> dict[str, WireValue]:
        """Serialize to the flat dict shape the binding consumes.

        Internal — called by :func:`empyrean.evaluate_plan` to marshal
        the candidate across the FFI boundary. Absent optional values
        lower to the NaN sentinel the C ABI reads; the caller never sees
        a NaN. For user-facing serialization, use
        :func:`dataclasses.asdict`.
        """
        return {
            "epoch_mjd_tdb": float(self.epoch_mjd_tdb),
            "kind": _KIND_TO_INT[self.kind],
            "optical_code": self.optical_code,
            "optical_sigma_ra_arcsec": float(self.optical_sigma_arcsec[0]),
            "optical_sigma_dec_arcsec": float(self.optical_sigma_arcsec[1]),
            "radar_transmit_station": _RADAR_STATION_TO_INT[self.radar_transmit_station],
            "radar_receive_station": _RADAR_STATION_TO_INT[self.radar_receive_station],
            "radar_mode": _RADAR_MODE_TO_INT[self.radar_mode],
            "radar_bandwidth_hz": float(self.radar_bandwidth_hz),
            "radar_freq_resolution_hz": float(self.radar_freq_resolution_hz),
            # NaN is the engine's "derive it from the link budget"
            # discriminant, not a missing value.
            "radar_snr": _to_nan(self.radar_snr),
            "radar_target_h_mag": _to_nan(self.radar_target_h_mag),
            "radar_target_visual_albedo": _to_nan(self.radar_target_visual_albedo),
            "radar_target_radar_albedo": _to_nan(self.radar_target_radar_albedo),
            "radar_target_diameter_km": _to_nan(self.radar_target_diameter_km),
            "radar_target_spin_period_hours": _to_nan(self.radar_target_spin_period_hours),
            "radar_integration_s": float(self.radar_integration_s),
        }


def _to_nan(v: float | None) -> float:
    """``None`` → the NaN sentinel the C ABI reads for "absent"."""
    return float("nan") if v is None else float(v)


@dataclass
class ObservatoryConfig:
    """Per-site astrometric assumptions and observability filters.

    **Not consulted by** :func:`empyrean.evaluate_plan`: the only field
    that takes these, :attr:`PlanningConfig.observatories`, refuses any
    non-empty list. Pass each candidate's σ to
    :meth:`PlannedObservation.optical` instead; the observability filters
    are engine-set on that entry point and are not caller-configurable.
    The class exists because it is part of the shared planning
    configuration, and becomes live if a surface that reads it is
    exposed.

    Mirrors ``empyrean::ObservatoryConfig``, whose fields carry no
    defaults — none are invented here either, so a config cannot be
    half-specified without saying so.
    """

    obs_code: str
    """MPC observatory code."""
    sigma_arcsec: tuple[float, float]
    """Assumed 1σ (RA·cosδ, Dec) uncertainty in arcsec."""
    max_apparent_mag: float
    """Limiting apparent magnitude."""
    min_elongation_deg: float
    """Minimum solar elongation (degrees)."""

    def _to_wire_dict(self) -> dict[str, WireValue]:
        """Serialize to the dict shape the binding consumes.

        Internal — called by :meth:`PlanningConfig._to_wire_dict`. For
        user-facing serialization, use :func:`dataclasses.asdict`.
        """
        return {
            "obs_code": self.obs_code,
            "sigma_ra_arcsec": float(self.sigma_arcsec[0]),
            "sigma_dec_arcsec": float(self.sigma_arcsec[1]),
            "max_apparent_mag": float(self.max_apparent_mag),
            "min_elongation_deg": float(self.min_elongation_deg),
        }


@dataclass
class PlanningConfig:
    """Configuration for :func:`empyrean.evaluate_plan`.

    Defaults match ``PlanningConfig::default()`` on the Rust side, so a
    default-constructed config is a no-op across the boundary.

    Mirrors ``empyrean::PlanningConfig``.
    """

    force_model: ForceModelTier = ForceModelTier.STANDARD
    """Force-model tier for the planning propagation. A bare wire string
    (``"approximate"`` / ``"basic"`` / ``"standard"``) is accepted and
    coerced."""
    epsilon: float = 1e-9
    """Adaptive integrator truncation-error tolerance."""
    observatories: list[ObservatoryConfig] = field(default_factory=list)
    """Per-site astrometric assumptions.

    **Not consulted by** :func:`empyrean.evaluate_plan`, which reads each
    optical candidate's σ from that candidate's own
    :class:`PlannedObservation` and applies engine-set observability
    filters that no field on this config reaches. Supplying a non-empty
    list therefore raises ``ValueError`` — at construction, and again
    inside :func:`empyrean.evaluate_plan` for a config mutated after
    construction — rather than being accepted and ignored. The field
    exists because it is part of the shared planning configuration; it
    will become live if a surface that reads it is exposed."""
    num_threads: int = 0
    """``0`` = use all available cores.

    **Not consulted by** :func:`empyrean.evaluate_plan`, which evaluates
    one orbit and does not shard the work. Any other value raises
    ``ValueError`` — at construction, and again inside
    :func:`empyrean.evaluate_plan` for a config mutated after
    construction — rather than being accepted and ignored."""

    def __post_init__(self) -> None:
        self._reject_unread_knobs()

    def _reject_unread_knobs(self) -> None:
        """Raise ``ValueError`` for any field :func:`evaluate_plan` never
        reads.

        A knob that accepts a value and silently does nothing is the
        failure mode this package refuses everywhere else (see
        :meth:`WeightingLayer.__post_init__`). Neither field below
        reaches the planner, so setting one is refused rather than
        quietly dropped somewhere across three layers.

        Called twice on purpose. :meth:`__post_init__` catches the
        common case at construction, and :func:`evaluate_plan` calls it
        again on the config it was handed — this dataclass is mutable by
        design, so ``cfg.num_threads = 4`` after construction would
        otherwise skip the construction-time check and surface as the
        Rust wrapper's ``RuntimeError`` backstop instead of the
        ``ValueError`` this class documents. Delete the matching arm —
        and the field's doc caveat — if a release ever wires one of them
        through.
        """
        try:
            self.force_model = ForceModelTier(_enum_value(self.force_model))
        except ValueError:
            accepted = ", ".join(repr(t.value) for t in ForceModelTier)
            raise ValueError(
                f"PlanningConfig: force_model must be one of {accepted} (or the "
                f"matching ForceModelTier member), got {self.force_model!r}."
            ) from None
        if self.observatories:
            raise ValueError(
                f"PlanningConfig.observatories is not consulted by "
                f"evaluate_plan (got {len(self.observatories)} entr"
                f"{'y' if len(self.observatories) == 1 else 'ies'}). Each "
                f"optical candidate's σ comes from its own "
                f"PlannedObservation — pass optical_sigma_arcsec to "
                f"PlannedObservation.optical(...) instead. Observability "
                f"filters are engine-set on this entry point and are not "
                f"caller-configurable."
            )
        if self.num_threads != 0:
            raise ValueError(
                f"PlanningConfig.num_threads is not consulted by "
                f"evaluate_plan (got {self.num_threads!r}); it evaluates a "
                f"single orbit and does not shard the work. Leave it at 0."
            )

    def _to_wire_dict(self) -> dict[str, WireValue]:
        """Serialize to the nested dict shape the binding consumes.

        Internal — called by :func:`empyrean.evaluate_plan` to marshal
        the config across the FFI boundary. For user-facing
        serialization (saving config to JSON, displaying it in a
        notebook, etc.), use :func:`dataclasses.asdict`.
        """
        return {
            "force_model": _enum_value(self.force_model),
            "epsilon": float(self.epsilon),
            "observatories": [o._to_wire_dict() for o in self.observatories],
            # The C ABI spells "every available core" as -1; the Python
            # surface spells it 0, matching ODConfig.num_threads.
            "num_threads": -1 if self.num_threads == 0 else int(self.num_threads),
        }


# ── Outputs ──────────────────────────────────────────────────


class PlanMetrics(qv.Table):
    """Covariance summary metrics bracketing the campaign — two rows.

    Mirrors ``empyrean::CovarianceMetrics``, once for the state before
    any candidate is folded and once for the state after every candidate
    the engine could fold — including any reported unobservable, since
    the fold does not consult :attr:`PlanCandidates.observable`.
    The :attr:`stage` column carries which is which, so the pair can be
    filtered, joined, or concatenated across plans without unpacking a
    scalar object; :meth:`prior` and :meth:`posterior` are the
    one-row views.

    The per-candidate running totals are the ``cumulative_*`` columns on
    :class:`PlanCandidates`, which carry the same five quantities after
    each observation is folded.

    The 1σ position ellipsoid has three axes but only the longest and
    shortest are carried. Recover the intermediate one from the identity
    that the three semi-axes are the square roots of the position block's
    eigenvalues, whose sum is :attr:`position_sigma_km` squared::

        b = math.sqrt(max(position_sigma_km**2 - semi_major_km**2 - semi_minor_km**2, 0.0))

    The clamp guards the rounding case where the three squares sum a
    hair past the total.
    """

    stage = qv.LargeStringColumn()
    """Which end of the campaign this row describes — ``"prior"``
    (before any candidate) or ``"posterior"`` (after all of them)."""
    position_sigma_km = qv.Float64Column()
    """RSS position 1σ (km)."""
    velocity_sigma_m_s = qv.Float64Column()
    """RSS velocity 1σ (m/s) at the orbit epoch."""
    semi_major_km = qv.Float64Column()
    """Semi-major axis of the 1σ position ellipsoid (km)."""
    semi_minor_km = qv.Float64Column()
    """Semi-minor axis of the 1σ position ellipsoid (km)."""
    log_det = qv.Float64Column()
    """:math:`\\ln \\det \\Sigma` over the 6×6 state covariance — the
    D-optimality criterion.

    **In AU and AU·day⁻¹**, unlike the four columns above, which are
    rescaled to km and m/s. A log-determinant is dimensional, so the
    absolute value depends on that choice: the same covariance expressed
    in km and m/s gives a value larger by :math:`6\\ln(\\mathrm{km/AU}) +
    6\\ln(\\mathrm{(m/s)/(AU/day)}) \\approx 199.13`. Differences between
    two ``log_det`` values on this surface are unit-invariant and can be
    compared directly."""

    # ── Selection helpers ─────────────────────────────────

    def prior(self) -> PlanMetrics:
        """The ``"prior"`` row — the covariance before any candidate."""
        return self.select("stage", STAGE_PRIOR)

    def posterior(self) -> PlanMetrics:
        """The ``"posterior"`` row — the covariance after every
        candidate has been folded."""
        return self.select("stage", STAGE_POSTERIOR)


class PlanCandidates(qv.Table):
    """Per-candidate information gain — one row per planned observation.

    Mirrors a vector of ``empyrean::PlanCandidate`` field-for-field.

    Rows are in the engine's evaluation order, which is **not**
    necessarily the order the candidates were supplied in, so a row does
    not carry its input epoch. For an optical row, :attr:`index` is the
    row in the companion :class:`PlanEphemeris` table, which does carry
    the epoch along with the predicted sky position; for a radar row it
    is that candidate's position among the radar candidates.

    Angular quantities are in **arcseconds**; :attr:`position_angle_deg`
    is in **degrees** (east of north).
    """

    # ── Identification ────────────────────────────────────
    index = qv.UInt64Column()
    """Row in :class:`PlanEphemeris` for an optical candidate; rank among
    the radar candidates, ordered by epoch, for a radar one.

    A radar row carries no epoch of its own, so this rank is the only key
    back to the input: sort the radar candidates you submitted by epoch
    and the *n*-th is the row with ``index == n``."""
    obs_code = qv.LargeStringColumn()
    """Observatory code (optical) or receive-station code (radar)."""
    kind = qv.LargeStringColumn()
    """``"optical"`` or ``"radar"``."""
    observable = qv.BooleanColumn()
    """Whether the candidate passes its observability filters — with a
    different meaning per :attr:`kind`, so branch on it before using
    this as a gate.

    On an **optical** row this is a real engine verdict, and today it is
    a solar-elongation test and nothing else: the limiting magnitude the
    engine would also apply cannot fire, because the target's absolute
    magnitude does not reach the planner. On a **radar** row it is
    always ``True`` — no radar feasibility test runs on this entry
    point, so ``True`` means "not assessed", not "checked and cleared".
    In particular no antenna-elevation or horizon test is applied, so a
    track below the horizon still reports ``True``.

    The filters are engine-set and not caller-configurable — no field on
    :class:`PlanningConfig` or :class:`PlannedObservation` reaches them.
    An unobservable candidate is reported rather than dropped, **and is
    still folded** into the ``cumulative_*`` columns and into
    :attr:`PlanResult.posterior`; see :meth:`observable_only`."""

    # ── Information gain (populated on every row) ─────────
    marginal_volume_reduction = qv.Float64Column()
    """Per-dimension generalized-variance ratio from this one
    observation, :math:`(\\det \\Sigma_\\mathrm{post} / \\det
    \\Sigma_\\mathrm{prior})^{1/6}` over the 6×6 state covariance
    (≤ 1) — a D-optimality score normalized to one dimension, so it
    reads as a linear scale factor and is comparable across plans.

    The 1σ ellipsoid *volume* ratio is this value **cubed**, and the raw
    determinant ratio is it to the **sixth** power. Conditional on the
    candidates folded before this one — see
    :meth:`best_by_information_gain`."""
    marginal_position_improvement = qv.Float64Column()
    """Fractional position-σ improvement from this one observation, in
    :math:`[0, 1]`. Conditional on the candidates folded before it, like
    :attr:`marginal_volume_reduction`."""
    active_width = qv.UInt64Column()
    """Width of the solve-for set this candidate folded into. Always 6
    (state-only) on this entry point — the non-gravitational solve is
    not exposed; see the Notes on :func:`empyrean.evaluate_plan`."""

    # ── Cumulative covariance metrics ─────────────────────
    cumulative_position_sigma_km = qv.Float64Column()
    """RSS position 1σ (km) after this observation and every one folded
    before it — including any reported unobservable, since the fold does
    not consult :attr:`observable`."""
    cumulative_velocity_sigma_m_s = qv.Float64Column()
    """RSS velocity 1σ (m/s) at the orbit epoch, after this observation
    and every one folded before it."""
    cumulative_semi_major_km = qv.Float64Column()
    """Semi-major axis of the cumulative 1σ position ellipsoid (km)."""
    cumulative_semi_minor_km = qv.Float64Column()
    """Semi-minor axis of the cumulative 1σ position ellipsoid (km)."""
    cumulative_log_det = qv.Float64Column()
    """Cumulative :math:`\\ln \\det \\Sigma`, in AU and AU·day⁻¹ like
    :attr:`PlanMetrics.log_det`."""

    # ── Optical sky-plane geometry (null on radar rows) ───
    along_track_sigma_arcsec = qv.Float64Column(nullable=True)
    """Prior along-track 1σ on the sky plane, in the frame of the
    predicted sky motion. Null on a radar row: radar measures
    line-of-sight range and range-rate, so there is no on-sky geometry to
    report.

    "Prior" is literal — the campaign prior mapped to this candidate's
    epoch with **no** candidate folded, not even this one. Its partner
    :attr:`post_along_track_sigma_arcsec` is cumulative, so the pair is
    not a single-observation bracket."""
    cross_track_sigma_arcsec = qv.Float64Column(nullable=True)
    """Prior cross-track 1σ on the sky plane, same "no candidate folded"
    basis as :attr:`along_track_sigma_arcsec`. Null on a radar row.

    Along- and cross-track are a projection onto the sky-motion frame,
    not the principal axes of the sky covariance, so cross-track may
    legitimately exceed along-track."""
    ra_sigma_arcsec = qv.Float64Column(nullable=True)
    """Prior RA·cosδ 1σ, no candidate folded. Null on a radar row."""
    dec_sigma_arcsec = qv.Float64Column(nullable=True)
    """Prior Dec 1σ, no candidate folded. Null on a radar row."""
    position_angle_deg = qv.Float64Column(nullable=True)
    """Position angle of the predicted **sky motion** (degrees, east of
    north) — the axis the along/cross-track σ above are projected onto.
    Null on a radar row.

    This is kinematic and does not depend on the covariance: it is not
    the orientation of the sky-plane uncertainty ellipse. The range is
    :math:`(-180, 180]`; add 360 for the conventional :math:`[0, 360)`
    position-angle convention."""
    post_along_track_sigma_arcsec = qv.Float64Column(nullable=True)
    """Along-track 1σ after folding this observation **and every one
    folded before it**. Null on a radar row.

    Cumulative, on the same basis as the ``cumulative_*`` columns — not
    the far end of a single-observation bracket against
    :attr:`along_track_sigma_arcsec`, which folds nothing."""
    post_cross_track_sigma_arcsec = qv.Float64Column(nullable=True)
    """Cross-track 1σ after folding this observation and every one folded
    before it. Cumulative, like
    :attr:`post_along_track_sigma_arcsec`. Null on a radar row."""

    # ── Radar block (null on optical rows) ────────────────
    radar_mode = qv.LargeStringColumn(nullable=True)
    """``"delay"`` / ``"doppler"`` / ``"both"``. Null on an optical
    row."""
    radar_snr = qv.Float64Column(nullable=True)
    """Effective SNR the measurement σ was derived from (a linear power
    ratio, not dB) — the supplied value, or the one the link budget
    produced. Null on an optical row."""
    radar_range_km = qv.Float64Column(nullable=True)
    """One-way topocentric range to the target at the receive epoch (km),
    from the predicted round-trip delay. Null on an optical row."""
    radar_provenance = qv.LargeListColumn(qv.LargeStringColumn())
    """Assumptions the link budget had to make to reach the SNR — for
    example a diameter derived from :math:`H` and :math:`p_V`, or
    coherent integration left uncapped because the spin period is
    unknown. Empty for an optical candidate, a caller-supplied SNR, or a
    fully specified link budget. Never summarized to a code: a note the
    engine adds later would be lost."""

    # ── Selection helpers ─────────────────────────────────

    def observable_only(self) -> PlanCandidates:
        """Rows with ``observable == True``.

        .. warning::

           This filters **rows, not information**. Every candidate was
           folded regardless of its verdict, so a surviving row's
           ``cumulative_*`` columns still contain the contributions of
           the rows this dropped, and :attr:`PlanResult.posterior` still
           prices the whole submitted plan. To price the observable
           subset, rebuild ``planned`` without the unobservable
           candidates and call :func:`empyrean.evaluate_plan` again.
        """
        return self.apply_mask(self.column("observable"))

    def select_station(self, obs_codes: str | Sequence[str]) -> PlanCandidates:
        """Rows from one or more observatory / receive-station codes."""
        codes = [obs_codes] if isinstance(obs_codes, str) else list(obs_codes)
        mask = _is_in(self.column("obs_code"), value_set=pa.array(codes))
        return self.apply_mask(mask)

    def best_by_information_gain(self, n: int = 10) -> PlanCandidates:
        """Top-``n`` rows by :attr:`marginal_volume_reduction`, **in rank
        order** — the best candidate first, not table order.

        The metric is a *reduction factor*, so smaller is better: a
        candidate that halves the generalized variance ranks above one
        that barely moves it. NaN rows sort last. Ties keep their
        relative table order.

        .. warning::

           The gains are **order-conditional**. The engine folds
           candidates in ascending epoch order and measures each against
           the covariance that already contains every earlier one, so a
           later candidate is scored against a tighter prior and reports
           a smaller gain. Two identical observations do not score
           identically. This ranks conditional contributions within one
           campaign; to compare candidates head to head, evaluate a
           separate one-candidate plan for each.
        """
        gain = self.marginal_volume_reduction.to_numpy(zero_copy_only=False)
        # argsort is stable, and NaN sorts to the end already; replace it
        # with +inf so a NaN never outranks a real reduction factor.
        order = np.argsort(np.nan_to_num(gain, nan=np.inf), kind="stable")
        # `take` preserves the order of the indices it is handed;
        # `apply_mask` would return the same rows in table order and
        # quietly break the ranking this method promises. The index array
        # is typed explicitly so an empty table takes an empty *int64*
        # array rather than a null-typed one arrow has no kernel for.
        return self.take(pa.array(order[:n], type=pa.int64()))


class PlanEphemeris(qv.Table):
    """Predicted sky position at each optical candidate's epoch.

    One row per optical candidate, in chronological order; an optical
    :class:`PlanCandidates` row's ``index`` is its row here. Empty for a
    radar-only plan (radar candidates carry no sky-plane prediction).

    Mirrors a vector of ``empyrean::PlanEphemerisPoint``. The epoch is
    carried as an :class:`Epochs` sub-table (always emitted in TDB)
    rather than a raw MJD float, so consumers can do
    ``eph.epochs.to_utc()`` and get back the same row alignment.
    """

    epochs = Epochs.as_column()
    """Prediction epoch."""
    ra_deg = qv.Float64Column()
    """Predicted topocentric right ascension (degrees, ICRF)."""
    dec_deg = qv.Float64Column()
    """Predicted topocentric declination (degrees, ICRF)."""


@dataclass
class PlanResult:
    """The result of evaluating an observation plan.

    Three tables describe the run, plus the two scalars that label it::

        plan = evaluate_plan(orbit, planned)
        plan.metrics.posterior().position_sigma_km[0].as_py()
        plan.candidates.best_by_information_gain(3).obs_code.to_pylist()

    Mirrors ``empyrean::PlanResult``.
    """

    orbit_id: str
    """Orbit identifier the plan was evaluated for."""
    metrics: PlanMetrics
    """Two rows — the covariance before any of the candidates and after
    **every** candidate submitted, observable or not."""
    candidates: PlanCandidates
    """Per-candidate information gain, in ascending epoch order rather
    than the order the candidates were supplied in."""
    ephemeris: PlanEphemeris
    """Predicted sky position for each optical candidate."""
    active_width: int
    """Width of the solve-for set. Always 6 (state-only) on this entry
    point; see the Notes on :func:`empyrean.evaluate_plan`."""
