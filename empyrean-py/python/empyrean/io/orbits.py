"""Read + write orbits in parquet / JSON / CSV.

Covariance coverage differs by format. Parquet preserves the full packed
joint (the whole (6+P)x(6+P) matrix). CSV keeps only the state 6x6 (the
joint's leading block); a joint carrying solved-parameter axes is refused
by name (ParameterCovarianceNotWritable) -- write parquet for the full
joint. JSON carries no covariance for a state-only joint: its row shape
has no joint column, and a quivr Orbits table carries only the joint, not
a separate flat 6x6, so there is nothing for the JSON row to serialize; a
wider joint is refused by name.
"""

from empyrean._convert import (
    orbit_batch_dict_to_orbits,
    orbits_to_orbit_batch_dict,
)
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)

OrbitsTable = CartesianOrbits | KeplerianOrbits | CometaryOrbits | SphericalOrbits


def read_orbits_parquet(path: str) -> OrbitsTable:
    """Read an orbits parquet file into the matching quivr Orbits table."""
    from empyrean._empyrean_rs import _read_orbits_parquet

    return orbit_batch_dict_to_orbits(_read_orbits_parquet(path))


def read_orbits_json(path: str) -> OrbitsTable:
    """Read an orbits JSON file."""
    from empyrean._empyrean_rs import _read_orbits_json

    return orbit_batch_dict_to_orbits(_read_orbits_json(path))


def read_orbits_csv(path: str) -> OrbitsTable:
    """Read an orbits CSV file. CSV carries the state 6x6 only (the joint's
    leading block), never the wider parameter axes."""
    from empyrean._empyrean_rs import _read_orbits_csv

    return orbit_batch_dict_to_orbits(_read_orbits_csv(path))


def write_orbits_parquet(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to parquet (the full packed joint is
    preserved)."""
    from empyrean._empyrean_rs import _write_orbits_parquet

    _write_orbits_parquet(path, orbits_to_orbit_batch_dict(orbits))


def write_orbits_json(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to JSON. JSON carries no covariance (its
    row shape has no joint column, and the table carries no separate flat
    6x6)."""
    from empyrean._empyrean_rs import _write_orbits_json

    _write_orbits_json(path, orbits_to_orbit_batch_dict(orbits))


def write_orbits_csv(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to CSV. Only the state 6x6 is written (the
    joint's leading block); a joint carrying solved-parameter axes is
    refused by name -- write parquet for the full joint."""
    from empyrean._empyrean_rs import _write_orbits_csv

    _write_orbits_csv(path, orbits_to_orbit_batch_dict(orbits))
