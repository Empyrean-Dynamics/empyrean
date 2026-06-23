"""Coordinate types for each orbital representation.

All angles at the Python boundary are in **degrees**.
Use :class:`~empyrean.types.enums.Frame` enum values for frames.

Heliocentric μ used by the orbital-element helpers below comes from
:data:`empyrean.coordinates.constants.MU_SUN_AU3_PER_DAY2`. Methods
that need a different central body's GM (e.g., a geocentric orbit
around Earth) take an explicit ``mu`` argument so the call site is
unambiguous.
"""

from typing import Any

import numpy as np
import quivr as qv

from empyrean.coordinates.covariance import (
    CartesianCovariance as _CartesianCovariance,
)
from empyrean.coordinates.covariance import (
    CometaryCovariance as _CometaryCovariance,
)
from empyrean.coordinates.covariance import (
    KeplerianCovariance as _KeplerianCovariance,
)
from empyrean.coordinates.covariance import (
    SphericalCovariance as _SphericalCovariance,
)

# The covariance classes are built dynamically by
# ``_make_covariance_class`` (via ``type(...)``), so its return is typed
# as the bare ``type``. They are genuine ``qv.Table`` subclasses at
# runtime, which is the most precise type statically knowable for a
# dynamically-synthesized schema. Re-bind under the public names with
# ``type[qv.Table]`` so the ``.as_column`` classmethod resolves.
CartesianCovariance: type[qv.Table] = _CartesianCovariance
CometaryCovariance: type[qv.Table] = _CometaryCovariance
KeplerianCovariance: type[qv.Table] = _KeplerianCovariance
SphericalCovariance: type[qv.Table] = _SphericalCovariance

# 1-D / 2-D float64 numpy arrays returned at the Python boundary.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# Gravitational parameter of the Sun in AU³/day². IAU 2012 nominal
# value (1.32712440018e20 m³/s² → AU³/day²). Adequate for elements
# helpers; kept inline to avoid a circular import on
# `empyrean.constants`.
MU_SUN_AU3_PER_DAY2 = 2.9591220828559115e-04


def _stack(*cols: FloatArray) -> FloatArray:
    """Column-stack 1-D float arrays into a contiguous ``(N, k)`` array."""
    out: FloatArray = np.column_stack(cols).astype(np.float64, copy=False)
    return out


class CartesianCoordinates(qv.Table):
    """Cartesian state vectors."""

    epoch = qv.Float64Column()  # MJD TDB
    x = qv.Float64Column()  # AU
    y = qv.Float64Column()  # AU
    z = qv.Float64Column()  # AU
    vx = qv.Float64Column()  # AU/day
    vy = qv.Float64Column()  # AU/day
    vz = qv.Float64Column()  # AU/day
    covariance = CartesianCovariance.as_column(nullable=True)
    frame = qv.StringAttribute()
    origin = qv.LargeStringColumn()

    # ── Vector accessors ─────────────────────────────────────────
    # All vectors are returned in the table's `frame` attribute,
    # relative to each row's `origin` column. Mixed-origin tables
    # (e.g., a propagation that crossed Earth's SOI) will return
    # vectors in different reference points per row — interpret
    # `r_mag` etc. accordingly.

    @property
    def r(self) -> FloatArray:
        """``(N, 3)`` position vectors in AU."""
        return _stack(
            self.x.to_numpy(zero_copy_only=False),
            self.y.to_numpy(zero_copy_only=False),
            self.z.to_numpy(zero_copy_only=False),
        )

    @property
    def r_mag(self) -> FloatArray:
        """``(N,)`` heliocentric / origin-relative distance in AU."""
        mag: FloatArray = np.linalg.norm(self.r, axis=1)
        return mag

    @property
    def v(self) -> FloatArray:
        """``(N, 3)`` velocity vectors in AU/day."""
        return _stack(
            self.vx.to_numpy(zero_copy_only=False),
            self.vy.to_numpy(zero_copy_only=False),
            self.vz.to_numpy(zero_copy_only=False),
        )

    @property
    def v_mag(self) -> FloatArray:
        """``(N,)`` speed in AU/day."""
        mag: FloatArray = np.linalg.norm(self.v, axis=1)
        return mag

    # ── Two-body invariants ──────────────────────────────────────

    def specific_energy(self, mu: float = MU_SUN_AU3_PER_DAY2) -> FloatArray:
        """``(N,)`` specific orbital energy ε = v²/2 − μ/r [AU²/day²].

        Negative for bound orbits, zero for parabolic, positive for
        hyperbolic. Heliocentric μ by default; override for orbits
        relative to a non-Sun origin.
        """
        energy: FloatArray = 0.5 * self.v_mag**2 - mu / self.r_mag
        return energy

    def specific_angular_momentum(self) -> FloatArray:
        """``(N, 3)`` specific orbital angular momentum L = r × v
        [AU²/day]. Frame-independent magnitude: ``np.linalg.norm(L, axis=1)``.
        """
        cross: FloatArray = np.cross(self.r, self.v)
        return cross


class KeplerianCoordinates(qv.Table):
    """Keplerian orbital elements."""

    epoch = qv.Float64Column()  # MJD TDB
    a = qv.Float64Column()  # semi-major axis (AU)
    e = qv.Float64Column()  # eccentricity
    i = qv.Float64Column()  # inclination (deg)
    raan = qv.Float64Column()  # longitude of ascending node (deg)
    ap = qv.Float64Column()  # argument of perihelion (deg)
    ma = qv.Float64Column()  # mean anomaly (deg)
    covariance = KeplerianCovariance.as_column(nullable=True)
    frame = qv.StringAttribute()
    origin = qv.LargeStringColumn()

    @property
    def perihelion_au(self) -> FloatArray:
        """``(N,)`` perihelion distance q = a(1 − e) [AU]. Negative
        when ``a < 0`` (hyperbolic), so ``q = a(1 − e)`` stays positive."""
        a = self.a.to_numpy(zero_copy_only=False)
        e = self.e.to_numpy(zero_copy_only=False)
        peri: FloatArray = a * (1.0 - e)
        return peri

    @property
    def aphelion_au(self) -> FloatArray:
        """``(N,)`` aphelion distance Q = a(1 + e) [AU]. Returns NaN
        for hyperbolic / parabolic orbits (e ≥ 1) where aphelion is
        not defined.
        """
        a = self.a.to_numpy(zero_copy_only=False)
        e = self.e.to_numpy(zero_copy_only=False)
        out: FloatArray = a * (1.0 + e)
        out = np.where(e < 1.0, out, np.nan)
        return out

    def period_days(self, mu: float = MU_SUN_AU3_PER_DAY2) -> FloatArray:
        """``(N,)`` orbital period T = 2π√(a³/μ) [days]. Returns NaN
        for hyperbolic orbits (a < 0)."""
        a = self.a.to_numpy(zero_copy_only=False)
        with np.errstate(invalid="ignore"):
            T = 2.0 * np.pi * np.sqrt(a**3 / mu)
        period: FloatArray = np.where(a > 0.0, T, np.nan)
        return period


class CometaryCoordinates(qv.Table):
    """Cometary orbital elements."""

    epoch = qv.Float64Column()  # MJD TDB
    q = qv.Float64Column()  # perihelion distance (AU)
    e = qv.Float64Column()  # eccentricity
    i = qv.Float64Column()  # inclination (deg)
    raan = qv.Float64Column()  # longitude of ascending node (deg)
    ap = qv.Float64Column()  # argument of perihelion (deg)
    tp = qv.Float64Column()  # time of perihelion passage (MJD TDB)
    covariance = CometaryCovariance.as_column(nullable=True)
    frame = qv.StringAttribute()
    origin = qv.LargeStringColumn()

    @property
    def is_hyperbolic(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """``(N,)`` bool — true when e ≥ 1 (unbound orbit). Useful
        for masking before calling :meth:`aphelion_au` / :meth:`period_days`.
        """
        mask: np.ndarray[Any, np.dtype[np.bool_]] = self.e.to_numpy(zero_copy_only=False) >= 1.0
        return mask

    @property
    def aphelion_au(self) -> FloatArray:
        """``(N,)`` aphelion distance Q = q(1 + e)/(1 − e) [AU].
        Returns NaN for hyperbolic / parabolic orbits (e ≥ 1).
        """
        q = self.q.to_numpy(zero_copy_only=False)
        e = self.e.to_numpy(zero_copy_only=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            Q = q * (1.0 + e) / (1.0 - e)
        aph: FloatArray = np.where(e < 1.0, Q, np.nan)
        return aph

    def period_days(self, mu: float = MU_SUN_AU3_PER_DAY2) -> FloatArray:
        """``(N,)`` orbital period T = 2π√(a³/μ) [days] with a = q/(1−e).
        Returns NaN for hyperbolic / parabolic orbits (e ≥ 1)."""
        q = self.q.to_numpy(zero_copy_only=False)
        e = self.e.to_numpy(zero_copy_only=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = q / (1.0 - e)
            T = 2.0 * np.pi * np.sqrt(a**3 / mu)
        period: FloatArray = np.where(e < 1.0, T, np.nan)
        return period


class SphericalCoordinates(qv.Table):
    """Spherical coordinates (topocentric / observer-centric)."""

    epoch = qv.Float64Column()  # MJD TDB
    rho = qv.Float64Column()  # radial distance (AU)
    lon = qv.Float64Column()  # longitude / RA (deg)
    lat = qv.Float64Column()  # latitude / Dec (deg)
    vrho = qv.Float64Column()  # radial velocity (AU/day)
    vlon = qv.Float64Column()  # angular velocity in lon (deg/day)
    vlat = qv.Float64Column()  # angular velocity in lat (deg/day)
    covariance = SphericalCovariance.as_column(nullable=True)
    frame = qv.StringAttribute()
    origin = qv.LargeStringColumn()

    @property
    def unit_vector(self) -> FloatArray:
        """``(N, 3)`` topocentric line-of-sight unit vectors in the
        table's frame. Useful for ephemeris geometry / angular
        separations without going through Cartesian.
        """
        lon_rad = np.deg2rad(self.lon.to_numpy(zero_copy_only=False))
        lat_rad = np.deg2rad(self.lat.to_numpy(zero_copy_only=False))
        cos_lat = np.cos(lat_rad)
        return _stack(cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
