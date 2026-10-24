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
    from empyrean.orbits.joint import KIND_TO_CODE, CovarianceKind

    n = len(eph)
    coords = eph.coordinates
    # The row's covariance kind rides as its wire code (the public
    # KIND_TO_CODE table), so the file writers carry the real kind rather than
    # a hardcoded placeholder. The packed joint is intentionally not carried —
    # this surface is a documented subset (see module docstring).
    cov_kind_codes = np.asarray(
        [KIND_TO_CODE[CovarianceKind(k)] for k in eph.cov_kind.to_pylist()],
        dtype=np.uint8,
    )
    return {
        "orbit_ids": eph.orbit_id.to_pylist(),
        "obs_codes": eph.obs_code.to_pylist(),
        "cov_kind": cov_kind_codes,
        "epoch_mjd_tdb": coords.epochs.mjd_tdb(),
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
        # The row's six pre-retention mixture tallies ride the file writers as
        # the same wire the C/Rust writers carry: a null count becomes the `-1`
        # "absent" sentinel and a null weight becomes NaN, never a fabricated 0.
        # Presence is recovered downstream from `mix_n_components_total >= 0`
        # (a real mixture always carries a non-negative component count, and
        # `-1` is the dedicated absent sentinel), so the dict needs no separate
        # has-flag column — the same single presence convention the file schema
        # itself uses, avoiding a second source of truth that could disagree.
        "mix_n_components_total": _int_column_to_wire(eph.mix_n_components_total, n),
        "mix_weight_delivered": _column_to_float(eph.mix_weight_delivered, n),
        "mix_n_failed": _int_column_to_wire(eph.mix_n_failed, n),
        "mix_n_unresolved": _int_column_to_wire(eph.mix_n_unresolved, n),
        "mix_n_curvature_refused": _int_column_to_wire(eph.mix_n_curvature_refused, n),
        "mix_n_sky_linearization_refused": _int_column_to_wire(
            eph.mix_n_sky_linearization_refused, n
        ),
    }


def _column_to_float(col: pa.Array | None, n: int) -> FloatArray:
    if col is None:
        return np.full(n, np.nan)
    arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)
    result: FloatArray = np.nan_to_num(arr, nan=np.nan)
    return result


IntArray = np.ndarray[Any, np.dtype[np.int64]]


def _int_column_to_wire(col: pa.Array | None, n: int) -> IntArray:
    """A nullable integer column to its i64 wire form: a null row becomes the
    ``-1`` "absent / not carried" sentinel the mixture tallies use, never a
    fabricated 0."""
    if col is None:
        return np.full(n, -1, dtype=np.int64)
    arr = np.asarray(col.fill_null(-1).to_numpy(zero_copy_only=False), dtype=np.int64)
    return arr


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
