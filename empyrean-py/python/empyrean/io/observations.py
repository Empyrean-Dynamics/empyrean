"""Query the MPC / JPL for ADES observation records."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from empyrean.io._cache import resolve_cache_dir
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.radar_observations import ADESRadarObservations


def query_observations(
    designations: Sequence[str],
    cache_dir: str | Path | bool | None = None,
) -> ADESObservations:
    """Fetch ADES observation records from the MPC for one or more
    designations.

    Returns the full ADES schema parsed into a quivr
    :class:`~empyrean.od.ades_observations.ADESObservations` table —
    every column the MPC API populates round-trips losslessly.

    Parameters
    ----------
    designations : list[str]
        MPC designations (e.g. ``"99942"``, ``"2024 YR4"``, ``"67P"``).
    cache_dir : str | Path | bool, optional
        - ``None`` (default): cache JSON responses under
          ``$EMPYREAN_CACHE_DIR/mpc`` (or ``~/.empyrean/cache/mpc``).
        - ``False``: disable caching for this call.
        - explicit path: use this directory.
    """
    from empyrean._empyrean_rs import _query_observations
    from empyrean.od.determine import _nullable_float

    result = _query_observations(list(designations), resolve_cache_dir(cache_dir, "mpc"))
    n = len(result["obs_time"])
    if n == 0:
        return ADESObservations.empty()

    def _str_list(key: str) -> list[str | None]:
        return [s if s else None for s in result[key]]

    n_stars_arr = np.asarray(result["n_stars"], dtype=np.int32)
    n_stars_list = [int(v) if v >= 0 else None for v in n_stars_arr]

    return ADESObservations.from_kwargs(
        # ── Identification ──
        perm_id=_str_list("perm_id"),
        prov_id=_str_list("prov_id"),
        trk_sub=_str_list("trk_sub"),
        obs_id=_str_list("obs_id"),
        obs_sub_id=_str_list("obs_sub_id"),
        trk_id=_str_list("trk_id"),
        # ── Observer ──
        stn=result["stn"],
        mode=_str_list("mode"),
        prog=_str_list("prog"),
        # ── Observer location ──
        sys=_str_list("sys"),
        ctr=_nullable_float(np.asarray(result["ctr"])),
        pos1=_nullable_float(np.asarray(result["pos1"])),
        pos2=_nullable_float(np.asarray(result["pos2"])),
        pos3=_nullable_float(np.asarray(result["pos3"])),
        # ── Astrometry ──
        obs_time=result["obs_time"],
        ra=np.asarray(result["ra"]),
        dec=np.asarray(result["dec"]),
        # ── Uncertainties ──
        rms_ra=_nullable_float(np.asarray(result["rms_ra"])),
        rms_dec=_nullable_float(np.asarray(result["rms_dec"])),
        rms_corr=_nullable_float(np.asarray(result["rms_corr"])),
        # ── Catalog ──
        ast_cat=_str_list("ast_cat"),
        # ── Photometry ──
        mag=_nullable_float(np.asarray(result["mag"])),
        rms_mag=_nullable_float(np.asarray(result["rms_mag"])),
        band=_str_list("band"),
        phot_cat=_str_list("phot_cat"),
        phot_ap=_nullable_float(np.asarray(result["phot_ap"])),
        # ── Supplementary diagnostics ──
        log_snr=_nullable_float(np.asarray(result["log_snr"])),
        seeing=_nullable_float(np.asarray(result["seeing"])),
        exp=_nullable_float(np.asarray(result["exp"])),
        rms_fit=_nullable_float(np.asarray(result["rms_fit"])),
        n_stars=n_stars_list,
        notes=_str_list("notes"),
        remarks=_str_list("remarks"),
    )


def query_radar(
    designations: Sequence[str],
    cache_dir: str | Path | bool | None = None,
) -> ADESRadarObservations:
    """Fetch radar (delay / Doppler) astrometry from JPL for one or more
    designations.

    Asteroid radar astrometry is a JPL SSD product served by the
    ``sb_radar`` API — it is **not** carried by the MPC observations API,
    so it ships as its own query parallel to :func:`query_observations`.
    Returns the records parsed into a quivr
    :class:`~empyrean.od.radar_observations.ADESRadarObservations` table in
    ADES-native units (delay value in seconds, its σ in microseconds,
    Doppler in Hz, frequency in MHz). An object with no radar astrometry
    yields an empty table. Fold the result into a fit by passing it as the
    ``radar=`` argument to :func:`empyrean.determine`.

    Parameters
    ----------
    designations : list[str]
        MPC designations (e.g. ``"99942"``, ``"2024 YR4"``).
    cache_dir : str | Path | bool, optional
        - ``None`` (default): cache JSON responses under
          ``$EMPYREAN_CACHE_DIR/sb_radar`` (or
          ``~/.empyrean/cache/sb_radar``).
        - ``False``: disable caching for this call.
        - explicit path: use this directory.
    """
    from empyrean._empyrean_rs import _query_radar
    from empyrean.od.determine import _nullable_float

    radar = _query_radar(list(designations), resolve_cache_dir(cache_dir, "sb_radar"))
    n = len(radar["obs_time"])
    if n == 0:
        return ADESRadarObservations.empty()

    def _str_list(key: str) -> list[str | None]:
        return [s if s else None for s in radar[key]]

    # com: i8 tri-state (-1 absent, 0 false, 1 true) -> Optional[bool].
    com_arr = np.asarray(radar["com"], dtype=np.int8)
    com_list = [None if v < 0 else bool(v) for v in com_arr]

    return ADESRadarObservations.from_kwargs(
        # ── Identification ──
        perm_id=_str_list("perm_id"),
        prov_id=_str_list("prov_id"),
        trk_sub=_str_list("trk_sub"),
        # ── Bistatic geometry ──
        trx=radar["trx"],
        rcv=radar["rcv"],
        # ── Core measurement ──
        obs_time=radar["obs_time"],
        observable=radar["observable"],
        delay=_nullable_float(np.asarray(radar["delay"])),
        rms_delay=_nullable_float(np.asarray(radar["rms_delay"])),
        doppler=_nullable_float(np.asarray(radar["doppler"])),
        rms_doppler=_nullable_float(np.asarray(radar["rms_doppler"])),
        # ── Reduction metadata ──
        frq=np.asarray(radar["frq"]),
        com=com_list,
        log_snr=_nullable_float(np.asarray(radar["log_snr"])),
        remarks=_str_list("remarks"),
    )
