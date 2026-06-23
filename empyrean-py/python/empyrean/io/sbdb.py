"""Query JPL Small-Body Database for orbital elements."""

from pathlib import Path

from empyrean.io._cache import resolve_cache_dir
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)

OrbitsTable = CartesianOrbits | KeplerianOrbits | CometaryOrbits | SphericalOrbits


def query_sbdb(
    names: list[str],
    cache_dir: str | Path | bool | None = None,
) -> OrbitsTable:
    """Query JPL SBDB for orbital elements with covariance.

    Parameters
    ----------
    names : list[str]
        Object names or designations (e.g. ``["Apophis", "67P"]``).
    cache_dir : str | Path | bool, optional
        - ``None`` (default): cache JSON responses under
          ``$EMPYREAN_CACHE_DIR/sbdb`` (or ``~/.empyrean/cache/sbdb`` if
          unset).
        - ``False``: disable caching for this call.
        - explicit path: use this directory.

    Returns
    -------
    quivr Orbits table
        Cartesian / Keplerian / Cometary / Spherical orbits depending
        on what SBDB returned (cometary by convention for asteroid /
        comet records). Includes covariance and non-gravitational
        parameters when SBDB exposes them.
    """
    from empyrean._convert import orbit_batch_dict_to_orbits
    from empyrean._empyrean_rs import _query_sbdb

    result = _query_sbdb(names, resolve_cache_dir(cache_dir, "sbdb"))
    return orbit_batch_dict_to_orbits(result)
