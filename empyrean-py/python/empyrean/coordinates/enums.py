"""Reference frame and origin types."""

import enum
from dataclasses import dataclass
from typing import ClassVar


class Frame(str, enum.Enum):
    """Reference frame for coordinate states.

    Subclasses ``str`` so values serialize directly into Arrow / quivr
    string columns.
    """

    ICRF = "icrf"
    """International Celestial Reference Frame (inertial, equatorial)."""

    ECLIPTICJ2000 = "eclipticj2000"
    """J2000 ecliptic frame (inertial, ecliptic plane)."""

    ITRF93 = "itrf93"
    """Earth body-fixed rotating frame (ITRF93).

    Requires a high-precision Earth-orientation BPC kernel
    (``earth_latest_high_prec.bpc`` or equivalent) to be loaded.
    :func:`empyrean.initialize` stages this kernel automatically when
    the ``naif-eop-high-prec`` PyPI package is installed; lazy fetch
    otherwise.
    """


@dataclass(frozen=True, eq=True)
class Origin:
    """Origin (center body) for a coordinate state.

    Use the class-level constants for named bodies::

        Origin.EARTH
        Origin.SUN
        Origin.JUPITER_BARYCENTER

    For numbered asteroids, use the factory::

        Origin.asteroid(1)  # Ceres
        Origin.asteroid(99942)  # Apophis

    Origin instances are immutable, hashable, and serialize to a stable
    canonical string when stored in quivr columns (``"Earth"``,
    ``"asteroid_1"``). Compare with ``==``.

    Mercury, Venus, Earth, and the Moon resolve to body centres.
    Mars through Pluto are exposed as ``*_BARYCENTER`` aliases.

    Origins are accepted in **all** API arguments that take a body
    reference (``body_filter``, ``excluded_perturbers``, the
    ``origin`` field on coordinate types, etc.). String forms
    (``"Earth"``, ``"asteroid_1"``) are also accepted via
    :meth:`Origin.from_string`.
    """

    name: str
    """Canonical body name. ``"Earth"`` / ``"Sun"`` / ``"asteroid_1"``."""

    # Class-level constants for the named bodies. Declared here so the
    # type checker knows they exist; the instances are assigned below the
    # class body (see end of file) once the dataclass machinery is wired.
    SSB: ClassVar["Origin"]
    SUN: ClassVar["Origin"]
    MERCURY: ClassVar["Origin"]
    VENUS: ClassVar["Origin"]
    EARTH: ClassVar["Origin"]
    MOON: ClassVar["Origin"]
    MARS_BARYCENTER: ClassVar["Origin"]
    JUPITER_BARYCENTER: ClassVar["Origin"]
    SATURN_BARYCENTER: ClassVar["Origin"]
    URANUS_BARYCENTER: ClassVar["Origin"]
    NEPTUNE_BARYCENTER: ClassVar["Origin"]
    PLUTO_BARYCENTER: ClassVar["Origin"]

    @classmethod
    def asteroid(cls, number: int) -> "Origin":
        """Construct an origin for a numbered asteroid (IAU number).

        Examples::

            Origin.asteroid(1)  # Ceres
            Origin.asteroid(4)  # Vesta
            Origin.asteroid(99942)  # Apophis
        """
        if not isinstance(number, int) or number <= 0:
            raise ValueError(f"asteroid number must be a positive integer, got {number!r}")
        return cls(f"asteroid_{number}")

    @classmethod
    def from_string(cls, s: str) -> "Origin":
        """Parse a canonical name (case-sensitive class attributes,
        case-insensitive otherwise).

        Inverse of ``str(origin)``. Useful for re-hydrating origins
        from text columns or human-typed config.
        """
        if not isinstance(s, str):
            raise TypeError(f"Origin.from_string expects str, got {type(s).__name__}")
        # Asteroid encoding round-trips via the factory.
        if s.startswith("asteroid_"):
            try:
                n = int(s[9:])
            except ValueError as e:
                raise ValueError(f"unparseable asteroid name: {s!r}") from e
            return cls.asteroid(n)
        # Direct match on the class attribute names (canonical strings).
        for known in _NAMED_ORIGINS:
            if s == known.name:
                return known
        # Lower-case fallback so things like "earth" / "ssb" / "jupiter_barycenter"
        # still resolve. Stay strict on whitespace; only accept exactly the
        # known canonical or canonical-with-underscores form.
        lower = s.strip().lower().replace(" ", "_")
        for known in _NAMED_ORIGINS:
            if lower == known.name.lower().replace(" ", "_"):
                return known
        # A few common short aliases.
        aliases = {
            "ssb": cls("SSB"),
            "mars": cls("Mars Barycenter"),
            "jupiter": cls("Jupiter Barycenter"),
            "saturn": cls("Saturn Barycenter"),
            "uranus": cls("Uranus Barycenter"),
            "neptune": cls("Neptune Barycenter"),
            "pluto": cls("Pluto Barycenter"),
        }
        if lower in aliases:
            return aliases[lower]
        raise ValueError(f"unknown origin: {s!r}")

    def __str__(self) -> str:
        return self.name


# Class-level constants for the named bodies. Populated below the class
# body so the dataclass machinery is fully wired before we try to
# construct instances.
Origin.SSB = Origin("SSB")
"""Solar System Barycenter."""
Origin.SUN = Origin("Sun")
"""Sun center."""
Origin.MERCURY = Origin("Mercury")
"""Mercury center."""
Origin.VENUS = Origin("Venus")
"""Venus center."""
Origin.EARTH = Origin("Earth")
"""Earth center."""
Origin.MOON = Origin("Moon")
"""Moon center."""
Origin.MARS_BARYCENTER = Origin("Mars Barycenter")
"""Mars system barycenter."""
Origin.JUPITER_BARYCENTER = Origin("Jupiter Barycenter")
"""Jupiter system barycenter."""
Origin.SATURN_BARYCENTER = Origin("Saturn Barycenter")
"""Saturn system barycenter."""
Origin.URANUS_BARYCENTER = Origin("Uranus Barycenter")
"""Uranus system barycenter."""
Origin.NEPTUNE_BARYCENTER = Origin("Neptune Barycenter")
"""Neptune system barycenter."""
Origin.PLUTO_BARYCENTER = Origin("Pluto Barycenter")
"""Pluto system barycenter."""

# Internal — used by `Origin.from_string` to enumerate the named
# bodies. Tests can rely on this list being complete.
_NAMED_ORIGINS = (
    Origin.SSB,
    Origin.SUN,
    Origin.MERCURY,
    Origin.VENUS,
    Origin.EARTH,
    Origin.MOON,
    Origin.MARS_BARYCENTER,
    Origin.JUPITER_BARYCENTER,
    Origin.SATURN_BARYCENTER,
    Origin.URANUS_BARYCENTER,
    Origin.NEPTUNE_BARYCENTER,
    Origin.PLUTO_BARYCENTER,
)
