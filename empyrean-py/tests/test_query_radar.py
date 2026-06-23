"""query_radar marshaling tests.

The live JPL ``sb_radar`` fetch is network-bound (and so, like the other
``query_*`` entry points, has no live test here). What we CAN pin offline is
the marshaling contract on the Python side of the binding: the flat dict the
``_query_radar`` pyo3 function returns must decode into an
:class:`~empyrean.od.radar_observations.ADESRadarObservations` table with the
no-silent-fallback semantics the C ABI promises —

* the ``observable`` discriminator selects the live measurement pair and the
  *other* pair is null (NaN -> null), never a silent 0;
* the ``com`` center-of-mass flag is a tri-state (``-1`` absent -> ``None``,
  ``0`` -> ``False``, ``1`` -> ``True``), never silently defaulted;
* absent optional strings decode to ``None``;
* ADES-native units (delay seconds, rms_delay microseconds, Doppler Hz,
  frequency MHz) are carried through verbatim.

A key/shape drift between the binding emit and this decode would silently drop
a field against live JPL; this test catches it without a network call by
feeding a synthetic dict through the real decode path.
"""

from __future__ import annotations

import empyrean
import numpy as np
from empyrean.od.radar_observations import ADESRadarObservations


def _synthetic_radar_dict() -> dict[str, object]:
    """One delay row + one Doppler row exercising every decode branch.

    Mirrors the 15-key flat dict shape ``_query_radar`` returns: absent
    optional strings as ``""`` (the binding's ``unwrap_or_default``), the
    inactive measurement pair as ``NaN``, ``com`` as an int8 tri-state.
    """
    return {
        # delay row carries perm_id + remarks + com=true; Doppler row omits them.
        "perm_id": ["99942", ""],
        "prov_id": ["", "2004 MN4"],
        "trk_sub": ["", ""],
        "trx": ["253", "253"],
        "rcv": ["253", "257"],
        "obs_time": ["2021-03-11T08:20:00Z", "2021-03-08T02:50:00Z"],
        "observable": ["delay", "doppler"],
        # ADES-native units: delay seconds, rms_delay MICROSECONDS.
        "delay": np.array([120.5, np.nan]),
        "rms_delay": np.array([0.25, np.nan]),
        # Doppler Hz (signed), rms_doppler Hz.
        "doppler": np.array([np.nan, -5000.0]),
        "rms_doppler": np.array([np.nan, 0.2]),
        "frq": np.array([8560.0, 2380.0]),  # MHz (Goldstone X / Arecibo S)
        "com": np.array([1, -1], dtype=np.int8),  # true ; absent
        "log_snr": np.array([2.5, np.nan]),
        "remarks": ["test", ""],
    }


def test_query_radar_decode_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "empyrean._empyrean_rs._query_radar",
        lambda _designations, _cache: _synthetic_radar_dict(),
    )
    r = empyrean.query_radar(["99942"])
    assert isinstance(r, ADESRadarObservations)
    assert len(r) == 2
    assert r.observable.to_pylist() == ["delay", "doppler"]

    delay = r.delay.to_pylist()
    rms_delay = r.rms_delay.to_pylist()
    doppler = r.doppler.to_pylist()
    rms_doppler = r.rms_doppler.to_pylist()

    # Delay row: live delay/rms_delay (ADES units), NULL Doppler pair — never 0.
    assert delay[0] == 120.5 and rms_delay[0] == 0.25
    assert doppler[0] is None and rms_doppler[0] is None
    # Doppler row: live Doppler pair (signed Hz), NULL delay pair.
    assert doppler[1] == -5000.0 and rms_doppler[1] == 0.2
    assert delay[1] is None and rms_delay[1] is None

    # com tri-state: 1 -> True, -1 -> None (NOT False).
    assert r.com.to_pylist() == [True, None]

    # Absent optional strings ("") decode to None; present ones survive.
    assert r.perm_id.to_pylist() == ["99942", None]
    assert r.prov_id.to_pylist() == [None, "2004 MN4"]
    assert r.trk_sub.to_pylist() == [None, None]
    assert r.remarks.to_pylist() == ["test", None]

    # log_snr: present -> value, absent -> None (NaN -> null).
    assert r.log_snr.to_pylist() == [2.5, None]

    # Carried-through units / geometry (frequency MHz, bistatic codes).
    assert r.frq.to_pylist() == [8560.0, 2380.0]
    assert r.trx.to_pylist() == ["253", "253"]
    assert r.rcv.to_pylist() == ["253", "257"]


def test_query_radar_empty(monkeypatch) -> None:
    """An object with no radar astrometry yields an empty table, not an error."""
    empty = {
        k: ([] if not isinstance(v, np.ndarray) else np.array([]))
        for k, v in _synthetic_radar_dict().items()
    }
    monkeypatch.setattr(
        "empyrean._empyrean_rs._query_radar",
        lambda _designations, _cache: empty,
    )
    r = empyrean.query_radar(["2024 YR4"])
    assert isinstance(r, ADESRadarObservations)
    assert len(r) == 0
