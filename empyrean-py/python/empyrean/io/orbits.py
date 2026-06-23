"""Read + write orbits in parquet / JSON / CSV.

Round-trips covariance through parquet and JSON; CSV drops covariance.
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
    """Read an orbits CSV file. CSV does not carry covariance."""
    from empyrean._empyrean_rs import _read_orbits_csv

    return orbit_batch_dict_to_orbits(_read_orbits_csv(path))


def write_orbits_parquet(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to parquet (covariance preserved)."""
    from empyrean._empyrean_rs import _write_orbits_parquet

    _write_orbits_parquet(path, orbits_to_orbit_batch_dict(orbits))


def write_orbits_json(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to JSON."""
    from empyrean._empyrean_rs import _write_orbits_json

    _write_orbits_json(path, orbits_to_orbit_batch_dict(orbits))


def write_orbits_csv(path: str, orbits: OrbitsTable) -> None:
    """Write an orbits quivr table to CSV. Covariance is dropped."""
    from empyrean._empyrean_rs import _write_orbits_csv

    _write_orbits_csv(path, orbits_to_orbit_batch_dict(orbits))
