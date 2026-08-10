"""Ephemeris quivr table, configuration, and result container.

Mirrors the shape of :class:`empyrean.propagation.result.PropagationResult`:
the observable table plus a per-``(orbit, observer)`` sensitivity container.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import quivr as qv

from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    SphericalCoordinates,
)
from empyrean.ephemeris.sensitivity import ObservationSensitivities
from empyrean.propagation.config import PropagationConfig

# ── Ephemeris quivr table ────────────────────────────────────


class Ephemeris(qv.Table):
    """Predicted astrometric ephemeris for observed objects.

    Each row is one (orbit, observer, epoch) combination with topocentric
    spherical coordinates (with covariance), aberrated Cartesian state,
    and ancillary data. All angles are in degrees.
    """

    # Identity
    orbit_id = qv.LargeStringColumn()
    object_id = qv.LargeStringColumn(nullable=True)
    obs_code = qv.LargeStringColumn()

    # Topocentric astrometry (covariance lives inside coordinates)
    coordinates = SphericalCoordinates.as_column()

    # Aberrated state at light-time corrected epoch
    aberrated_state = CartesianCoordinates.as_column(nullable=True)

    # Light time & geometry
    light_time = qv.Float64Column(nullable=True)  # one-way (days)
    phase_angle = qv.Float64Column(nullable=True)  # Sun-Object-Observer (deg)
    elongation = qv.Float64Column(nullable=True)  # Sun-Observer-Object (deg)
    heliocentric_distance = qv.Float64Column(nullable=True)  # AU

    # Photometry
    mag = qv.Float64Column(nullable=True)
    # 1-sigma magnitude uncertainty. Populated iff photometry is enabled
    # AND the input orbit carried a state covariance; null otherwise.
    # State contribution only — H-magnitude uncertainty is not yet an
    # input, so sigma_V is under-reported when H uncertainty matters.
    mag_sigma = qv.Float64Column(nullable=True)

    # Local horizon
    zenith_angle = qv.Float64Column(nullable=True)
    azimuth = qv.Float64Column(nullable=True)
    hour_angle = qv.Float64Column(nullable=True)

    # Lunar geometry
    lunar_elongation = qv.Float64Column(nullable=True)

    # Sky motion
    position_angle = qv.Float64Column(nullable=True)
    sky_rate = qv.Float64Column(nullable=True)


# ── Configuration ────────────────────────────────────────────


@dataclass
class EphemerisConfig:
    """Configuration for :func:`empyrean.generate_ephemeris`.

    Embeds a :class:`PropagationConfig`. The knobs the integrator
    consults while bringing each orbit to its observation epoch all
    apply: ``force_model``, ``excluded_perturbers``,
    ``uncertainty_method``, ``compute_stm``, ``frame``, ``num_threads``,
    ``ephemeris_overlap_policy``, and the whole ``advanced`` block.
    Ephemeris-specific fields (light-time iteration limits, diagnostic
    computation) live on this struct directly.

    Two sub-configs do **not** apply: ``propagation.events`` and
    ``propagation.diagnostics``. Ephemeris generation runs with event
    detection and timeseries diagnostics off, and
    :class:`EphemerisResult` carries neither, so modifying either one
    raises a :class:`ValueError` naming the offending fields rather than
    being silently dropped. Leave them at their defaults and use
    :func:`empyrean.propagate` when you need them.

    Generating an ephemeris for an SB441-N16 body (1 Ceres, 2 Pallas,
    4 Vesta, …) at Standard tier needs one of the two escapes from the
    self-perturbation case: ``propagation.ephemeris_overlap_policy =
    EphemerisOverlapPolicy.EXCLUDE_AND_INTEGRATE``, or naming the body
    in ``propagation.excluded_perturbers``. Without either, the engine
    substitutes the body's own SPK states, produces no dense trajectory,
    and the call fails.

    Parameters
    ----------
    propagation : PropagationConfig
        Inner propagation configuration. Default:
        :class:`PropagationConfig()` (Standard, FirstOrder, etc.).
        ``events`` and ``diagnostics`` must be left at their defaults.
    max_light_time_iterations : int
        Light-time convergence loop cap. Default 3.
    light_time_tolerance_days : float
        Light-time convergence tolerance in days. Default 1e-10.
    compute_diagnostics : bool
        Compute phase angle, elongation, heliocentric distance, and
        apparent magnitude. Skip during DC iterations for speed.
        Default True.
    """

    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    max_light_time_iterations: int = 3
    light_time_tolerance_days: float = 1e-10
    compute_diagnostics: bool = True

    def _to_wire_dict(self) -> dict[str, Any]:
        """Serialize to the nested dict shape the binding consumes.

        Internal — called by :func:`empyrean.generate_ephemeris` to
        marshal the config across the FFI boundary. For user-facing
        serialization, use :func:`dataclasses.asdict`.
        """
        return {
            "propagation": self.propagation._to_wire_dict(),
            "max_light_time_iterations": self.max_light_time_iterations,
            "light_time_tolerance_days": self.light_time_tolerance_days,
            "compute_diagnostics": self.compute_diagnostics,
        }


# ── Result container ─────────────────────────────────────────


@dataclass
class EphemerisResult:
    """Result of :func:`empyrean.generate_ephemeris`.

    Attributes
    ----------
    ephemeris : Ephemeris
        Predicted astrometry table (one row per orbit × observer ×
        epoch) with topocentric coordinates and observation covariance.
    sensitivity : ObservationSensitivities, optional
        Flat per-``(orbit_id, obs_code, epoch)`` sensitivity table —
        observation Jacobians + optional Hessians. Populated whenever
        the propagation traced the state-transition matrix: either the
        input orbit carried a covariance, or
        ``config.propagation.compute_stm=True`` requested the trace
        outright (which works with **no** input covariance — the flag
        reaches the engine on this path). ``None`` when neither
        happened.
    warnings : list[str]
        Non-fatal generation warnings, in engine emission order. Empty
        when the run had nothing to report. Messages name the affected
        orbit / observatory / epoch where applicable (e.g.
        Earth-orientation kernel coverage gaps handled by an analytic
        fallback, or rows whose sensitivity chain was skipped).
    """

    ephemeris: Ephemeris
    sensitivity: ObservationSensitivities | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dir(self, path: str) -> None:
        """Persist to ``<path>/ephemeris.parquet`` +
        ``<path>/sensitivity.parquet`` (+ ``<path>/warnings.json`` when
        the run produced warnings)."""
        os.makedirs(path, exist_ok=True)
        self.ephemeris.to_parquet(os.path.join(path, "ephemeris.parquet"))
        if self.sensitivity is not None and len(self.sensitivity) > 0:
            self.sensitivity.to_parquet(os.path.join(path, "sensitivity.parquet"))
        warn_path = os.path.join(path, "warnings.json")
        if self.warnings:
            with open(warn_path, "w") as f:
                json.dump(self.warnings, f)
        elif os.path.exists(warn_path):
            # A clean re-save into a reused directory must not leave a
            # stale warnings file attributed to the new data.
            os.remove(warn_path)

    @classmethod
    def from_dir(cls, path: str) -> EphemerisResult:
        ephemeris = Ephemeris.from_parquet(os.path.join(path, "ephemeris.parquet"))
        sens_path = os.path.join(path, "sensitivity.parquet")
        sensitivity = (
            ObservationSensitivities.from_parquet(sens_path) if os.path.exists(sens_path) else None
        )
        warn_path = os.path.join(path, "warnings.json")
        if os.path.exists(warn_path):
            with open(warn_path) as f:
                warnings = json.load(f)
        else:
            warnings = []
        return cls(ephemeris=ephemeris, sensitivity=sensitivity, warnings=warnings)
