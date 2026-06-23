"""Write ephemeris tables in parquet / JSON / CSV.

The C-ABI flat ephemeris schema covers RA / Dec / rho / rates /
light-time / phase / elongation / heliocentric distance / mag. The
quivr :class:`~empyrean.ephemeris.result.Ephemeris` table also
carries aberrated state, local-horizon angles, lunar elongation, sky
rate, and position angle; those columns do not round-trip through
this surface.
"""

from typing import Any

import numpy as np
import pyarrow as pa

from empyrean.ephemeris.result import Ephemeris

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _ephemeris_to_dict(eph: Ephemeris) -> dict[str, list[str] | FloatArray]:
    n = len(eph)
    coords = eph.coordinates
    return {
        "orbit_ids": eph.orbit_id.to_pylist(),
        "obs_codes": eph.obs_code.to_pylist(),
        "epoch_mjd_tdb": np.asarray(coords.epoch.to_numpy(zero_copy_only=False)),
        "ra_deg": np.asarray(coords.lon.to_numpy(zero_copy_only=False)),
        "dec_deg": np.asarray(coords.lat.to_numpy(zero_copy_only=False)),
        "rho_au": np.asarray(coords.rho.to_numpy(zero_copy_only=False)),
        "vrho_au_day": np.asarray(coords.vrho.to_numpy(zero_copy_only=False)),
        "vra_deg_day": np.asarray(coords.vlon.to_numpy(zero_copy_only=False)),
        "vdec_deg_day": np.asarray(coords.vlat.to_numpy(zero_copy_only=False)),
        "light_time_days": _column_to_float(eph.light_time, n),
        "phase_angle_deg": _column_to_float(eph.phase_angle, n),
        "elongation_deg": _column_to_float(eph.elongation, n),
        "heliocentric_distance_au": _column_to_float(eph.heliocentric_distance, n),
        "mag": _column_to_float(eph.mag, n),
        "mag_sigma": _column_to_float(eph.mag_sigma, n),
    }


def _column_to_float(col: pa.Array | None, n: int) -> FloatArray:
    if col is None:
        return np.full(n, np.nan)
    arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
    result: FloatArray = np.nan_to_num(arr, nan=np.nan)
    return result


def write_ephemeris_parquet(path: str, eph: Ephemeris) -> None:
    """Write an :class:`Ephemeris` table to parquet."""
    from empyrean._empyrean_rs import _write_ephemeris_parquet

    _write_ephemeris_parquet(path, _ephemeris_to_dict(eph))


def write_ephemeris_json(path: str, eph: Ephemeris) -> None:
    """Write an :class:`Ephemeris` table to JSON."""
    from empyrean._empyrean_rs import _write_ephemeris_json

    _write_ephemeris_json(path, _ephemeris_to_dict(eph))


def write_ephemeris_csv(path: str, eph: Ephemeris) -> None:
    """Write an :class:`Ephemeris` table to CSV."""
    from empyrean._empyrean_rs import _write_ephemeris_csv

    _write_ephemeris_csv(path, _ephemeris_to_dict(eph))
