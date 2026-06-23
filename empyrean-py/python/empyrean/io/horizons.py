"""Query JPL Horizons for ephemerides and state vectors."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs
from empyrean.ephemeris.result import Ephemeris
from empyrean.io._cache import resolve_cache_dir

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def query_horizons(
    names: Sequence[str],
    observer: str,
    epochs: Epochs | FloatArray | Sequence[float],
    cache_dir: str | Path | bool | None = None,
) -> Ephemeris:
    """Query JPL Horizons for observer-table ephemerides.

    Returns topocentric RA/Dec, range, rates, light-time, phase angle,
    elongation, heliocentric distance, and V-band magnitude pulled from
    the Horizons observer-table query.

    Parameters
    ----------
    names : list[str]
        Object names / designations / SPK IDs.
    observer : str
        MPC observatory code (e.g. ``"W84"``, ``"F51"``, ``"@399"``).
    epochs : Epochs | array-like
        Observation epochs. :class:`~empyrean.coordinates.epoch.Epochs`
        table (converted to TDB internally) or a 1-D array of MJD TDB.
    cache_dir : str | Path | bool, optional
        - ``None`` (default): cache JSON responses under
          ``$EMPYREAN_CACHE_DIR/horizons`` (or
          ``~/.empyrean/cache/horizons``).
        - ``False``: disable caching for this call.
        - explicit path: use this directory.

    Returns
    -------
    Ephemeris
        Predicted ephemeris from Horizons (all angles in degrees).
        Aberrated state, local-horizon angles, lunar elongation, sky
        rate, and position angle are not produced by Horizons; those
        columns come back as NaN.
    """
    from empyrean._empyrean_rs import _query_horizons
    from empyrean.coordinates.coordinates import (
        CartesianCoordinates,
        SphericalCoordinates,
    )

    if isinstance(epochs, Epochs):
        tdb = epochs.to_tdb()
        times_arr = np.asarray(tdb.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)
    else:
        times_arr = np.asarray(epochs, dtype=np.float64)

    result = _query_horizons(
        list(names), observer, times_arr, resolve_cache_dir(cache_dir, "horizons")
    )

    m = len(result["epoch"])
    epoch_arr = np.asarray(result["epoch"])
    nans = np.full(m, np.nan)

    coordinates = SphericalCoordinates.from_kwargs(
        epoch=epoch_arr,
        rho=np.asarray(result["rho"]),
        lon=np.asarray(result["ra"]),
        lat=np.asarray(result["dec"]),
        vrho=np.asarray(result["vrho"]),
        vlon=np.asarray(result["vra"]),
        vlat=np.asarray(result["vdec"]),
        frame=Frame.ICRF.value,
        origin=result["obs_code"],
    )

    # Horizons doesn't return aberrated states — fill with NaN but set
    # the frame attribute so the quivr nested validation passes.
    aberrated_state = CartesianCoordinates.from_kwargs(
        epoch=epoch_arr,
        x=nans.copy(),
        y=nans.copy(),
        z=nans.copy(),
        vx=nans.copy(),
        vy=nans.copy(),
        vz=nans.copy(),
        frame=Frame.ICRF.value,
        origin=[str(Origin.SSB)] * m,
    )

    def _nullable(key: str) -> FloatArray | pa.Array:
        arr = np.asarray(result[key])
        mask = np.isnan(arr)
        if mask.any():
            return pa.array(arr.tolist(), type=pa.float64(), mask=mask)
        return arr

    return Ephemeris.from_kwargs(
        orbit_id=result["orbit_id"],
        object_id=[s if s else None for s in result["object_id"]],
        obs_code=result["obs_code"],
        coordinates=coordinates,
        aberrated_state=aberrated_state,
        light_time=_nullable("light_time"),
        phase_angle=_nullable("phase_angle"),
        elongation=_nullable("elongation"),
        heliocentric_distance=_nullable("heliocentric_distance"),
        mag=_nullable("mag"),
        mag_sigma=_nullable("mag_sigma"),
        zenith_angle=nans,
        azimuth=nans,
        hour_angle=nans,
        lunar_elongation=nans,
        position_angle=nans,
        sky_rate=nans,
    )


def query_horizons_vectors(
    command: str,
    epoch_mjd_tdb: float,
    cache_dir: str | Path | bool | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Query JPL Horizons for a Cartesian state vector at a single epoch.

    Returns the solar-system-barycenter (SSB) centered, ICRF Cartesian
    state from the Horizons vector-table query.

    Parameters
    ----------
    command : str
        Horizons COMMAND string (e.g. ``"99942;"``, ``"DES=C/2019 Q4;"``).
    epoch_mjd_tdb : float
        Epoch in MJD TDB.
    cache_dir : str | Path | bool, optional
        - ``None`` (default): cache JSON responses under
          ``$EMPYREAN_CACHE_DIR/horizons`` (or
          ``~/.empyrean/cache/horizons``).
        - ``False``: disable caching for this call.
        - explicit path: use this directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(position, velocity)`` — length-3 arrays in AU and AU/day,
        SSB-centered, ICRF.
    """
    from empyrean._empyrean_rs import _query_horizons_vectors

    pos, vel = _query_horizons_vectors(
        command, epoch_mjd_tdb, resolve_cache_dir(cache_dir, "horizons")
    )
    return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64)
