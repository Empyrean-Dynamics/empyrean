"""Ephemeris quivr table, configuration, and result container.

Mirrors the shape of :class:`empyrean.propagation.result.PropagationResult`:
the observable table plus a per-``(orbit, observer)`` sensitivity container.
"""

from __future__ import annotations

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

    Embeds a :class:`PropagationConfig` — every propagation-side knob
    (force model, uncertainty method, integrator tolerance, thread
    count, non-grav, etc.) is set there. Ephemeris-specific fields
    (light-time iteration limits, diagnostic computation) live on this
    struct directly.

    Parameters
    ----------
    propagation : PropagationConfig
        Inner propagation configuration. Default:
        :class:`PropagationConfig()` (Standard, FirstOrder, etc.).
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
        observation Jacobians + optional Hessians. ``None`` when no
        input covariance was supplied.
    """

    ephemeris: Ephemeris
    sensitivity: ObservationSensitivities | None = None

    def to_dir(self, path: str) -> None:
        """Persist to ``<path>/ephemeris.parquet`` + ``<path>/sensitivity.parquet``."""
        os.makedirs(path, exist_ok=True)
        self.ephemeris.to_parquet(os.path.join(path, "ephemeris.parquet"))
        if self.sensitivity is not None and len(self.sensitivity) > 0:
            self.sensitivity.to_parquet(os.path.join(path, "sensitivity.parquet"))

    @classmethod
    def from_dir(cls, path: str) -> EphemerisResult:
        ephemeris = Ephemeris.from_parquet(os.path.join(path, "ephemeris.parquet"))
        sens_path = os.path.join(path, "sensitivity.parquet")
        sensitivity = (
            ObservationSensitivities.from_parquet(sens_path) if os.path.exists(sens_path) else None
        )
        return cls(ephemeris=ephemeris, sensitivity=sensitivity)
