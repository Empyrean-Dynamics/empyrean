"""Continuous-thrust inputs: steering law, thrust arcs, and thrust
parameters for low-thrust / finite-burn propagation.

These dataclasses are the Python mirror of the engine's thrust surface.
Attach a :class:`ThrustParams` to an orbit at propagation time via the
``thrust_arcs`` keyword of :func:`empyrean.propagate` — one entry per
orbit, positionally aligned with the orbit batch (pass ``None`` for the
gravity / non-grav-only orbits). Field names, units, and semantics match
the engine one-for-one so a thrust arc round-trips without renaming or
reshaping.

The acceleration during an arc is

.. math::

    \\mathbf{a}(t) = \\sigma(t)\\,\\frac{F}{m(t)}\\,\\hat{d}

with a smooth :math:`\\tanh` switch :math:`\\sigma(t)` whose width is set
by :attr:`ThrustArc.sharpness`, steering direction :math:`\\hat{d}` from
:attr:`ThrustArc.steering`, and mass :math:`m(t)` that depletes when
:attr:`ThrustArc.isp_s` is set:

.. math::

    m(t) = m_0 - \\frac{F}{I_{sp}\\,g_0}\\,(t - t_\\text{start}).

Example
-------
>>> from empyrean import Origin
>>> from empyrean.orbits.thrust import (
...     ConstantRTN,
...     ThrustArc,
...     ThrustParams,
... )
>>> arc = ThrustArc(
...     start_mjd_tdb=59000.0,
...     end_mjd_tdb=59010.0,
...     thrust_n=1000.0,
...     mass_kg=1000.0,
...     steering=ConstantRTN(alpha_rad=0.1, beta_rad=0.2),
...     sharpness=100.0,
...     central_body=Origin.SUN,
... )
>>> params = ThrustParams(arcs=[arc])
"""

from __future__ import annotations

from dataclasses import dataclass, field

from empyrean.coordinates.enums import Origin

__all__ = [
    "ConstantRTN",
    "InertialFixed",
    "SteeringLaw",
    "ThrustArc",
    "ThrustParams",
    "VelocityTangent",
]


@dataclass(frozen=True)
class ConstantRTN:
    """Constant RTN steering angles relative to the arc's central body.

    The thrust direction is

    .. math::

        \\hat{d} = \\cos\\beta\\cos\\alpha\\;\\hat{R}
                 + \\cos\\beta\\sin\\alpha\\;\\hat{T}
                 + \\sin\\beta\\;\\hat{N}

    where :math:`\\alpha` is the in-plane angle (radial toward transverse)
    and :math:`\\beta` is the out-of-plane angle (toward orbit normal).
    """

    alpha_rad: float
    """In-plane angle from radial toward transverse (radians)."""
    beta_rad: float
    """Out-of-plane angle toward orbit normal (radians)."""


@dataclass(frozen=True)
class VelocityTangent:
    """Thrust aligned with the velocity vector relative to the central
    body: :math:`\\hat{d} = \\hat{v}_\\text{body}`."""


@dataclass(frozen=True)
class InertialFixed:
    """Fixed direction in the inertial frame (normalized internally)."""

    direction: tuple[float, float, float]
    """Direction vector; normalized by the engine."""


# The steering law is one of the three variants above. The Python class
# name of the variant is the discriminant the binding reads when it
# rebuilds the engine's `SteeringLaw` enum.
SteeringLaw = ConstantRTN | VelocityTangent | InertialFixed


@dataclass
class ThrustArc:
    """A single continuous-thrust arc with smooth on/off switching and
    optional mass depletion."""

    start_mjd_tdb: float
    """Arc start epoch (MJD TDB)."""
    end_mjd_tdb: float
    """Arc end epoch (MJD TDB)."""
    thrust_n: float
    """Engine thrust force in Newtons."""
    mass_kg: float
    """Spacecraft mass at arc start in kilograms."""
    steering: SteeringLaw
    """Steering law for this arc."""
    sharpness: float
    r""":math:`\tanh` switching sharpness (1/days). Higher values give
    sharper on/off transitions (closer to bang-bang). Typical values:
    1000-10000 for burns of minutes, 100 for multi-hour arcs."""
    central_body: Origin
    """Central body for the RTN / velocity-tangent frame reference. The
    spacecraft state is expressed relative to this body before the thrust
    direction is computed — e.g. ``Origin.EARTH`` for geocentric arcs,
    ``Origin.SUN`` for heliocentric ones."""
    isp_s: float | None = None
    r"""Specific impulse in seconds. When set, mass depletes linearly
    during the burn at :math:`\dot m = F/(I_{sp}\,g_0)`; ``None`` holds
    mass constant."""


@dataclass
class ThrustParams:
    """Thrust parameters for one orbit: thrust arcs plus optional
    :math:`\\Delta v` targeting corrections.

    When :attr:`correction_covariances` is non-empty its length MUST equal
    :attr:`dv_corrections`; a non-empty covariance triggers the wide-Jet
    burn-sensitivity propagation and its solved segments appear in the
    propagated
    :attr:`~empyrean.TaggedCovariance.thrust_segments`. Length or
    arc/correction mismatches surface loudly as an exception during
    propagation — never silently repaired or dropped.
    """

    arcs: list[ThrustArc]
    """Ordered list of thrust arcs. May overlap in time."""
    dv_corrections: list[tuple[float, float, float]] = field(default_factory=list)
    r"""Per-arc :math:`\Delta v` corrections (AU/day) for targeting,
    positional with :attr:`arcs`. Each is applied as a constant inertial
    acceleration during its arc's window; when seeded as Jet variables it
    provides :math:`\partial\text{state}/\partial\Delta v`. Empty = no
    corrections."""
    correction_covariances: list[list[tuple[float, float, float]]] = field(default_factory=list)
    r"""3x3 covariance (AU/day):math:`^2` per :math:`\Delta v` correction,
    positional with :attr:`dv_corrections`. When non-empty its length must
    equal ``dv_corrections``. Empty = no burn-sensitivity propagation."""
