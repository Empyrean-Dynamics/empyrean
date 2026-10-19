"""Provisioning verifies the Earth-orientation kernel it links into a slot.

The incident (measured 2026-08-23): a pip data package shipped the standard
NAIF ``earth_200101_990827_predict.bpc`` (Earth-orientation coverage to
~2099) under the long-term predict slot, so every fresh install failed
``initialize()`` at the three-Earth-kernel mandate (horizon 2125) while every
warm data dir masked it. These pin the name-vs-content check at the one link
point, before any context is built: a file that cannot cover its slot is
named at provisioning (slot, linked file, coverage found, required), not
discovered at ``initialize()``.

No network: every kernel is read from an installed pip data package or the
warm ``EMPYREAN_DATA_DIR``. Tests skip **by name** — never silently pass —
when a file they need is not present.
"""

import os
from pathlib import Path

import empyrean
import pytest

# Engine kernel-role codes, mirroring EMPYREAN_EARTH_KERNEL_ROLE_* in the C
# ABI (and empyrean._EARTH_ORIENTATION_SLOT_ROLE).
_HISTORICAL, _HIGH_PRECISION, _PREDICT = 0, 1, 2

_PREDICT_SLOT = empyrean._B612_TO_VILLENEUVE_FILENAME["earth_predict"]
_HISTORICAL_SLOT = empyrean._B612_TO_VILLENEUVE_FILENAME["earth_historical"]


def _coverage(path: str, role: int):
    from empyrean._empyrean_rs import _earth_orientation_kernel_coverage

    return _earth_orientation_kernel_coverage(path, role)


def _warm_file(name: str) -> Path:
    """A real kernel from the warm data directory, or a named skip."""
    data_dir = os.environ.get("EMPYREAN_DATA_DIR")
    if not data_dir:
        try:
            data_dir = empyrean.default_data_dir()
        except Exception:  # noqa: BLE001 - no resolvable data dir at all
            pytest.skip("no resolvable data directory (set EMPYREAN_DATA_DIR)")
    f = Path(data_dir) / name
    if not f.exists():
        pytest.skip(f"warm data file {f} not present")
    return f


# ── The coverage verdict (the engine's own per-kernel mandate) ────────


def test_predict_role_rejects_short_coverage_and_accepts_sufficient():
    """The predict slot's mandate is end-of-coverage >= the 2125 horizon.

    A kernel that stops well short of 2125 — here the historical kernel,
    which ends ~2026 — cannot satisfy it; the actual predict kernel does.
    Both files are real; the verdict is the engine's own, read through the
    same BPC reader the mandate uses at load.
    """
    short = _coverage(str(_warm_file(_HISTORICAL_SLOT)), _PREDICT)
    satisfied, _start, end, _floor, horizon = short
    assert satisfied is False
    assert end < horizon  # ends before the mandate horizon

    good = _coverage(str(_warm_file(_PREDICT_SLOT)), _PREDICT)
    satisfied, _start, end, _floor, horizon = good
    assert satisfied is True
    assert end >= horizon


def test_a_non_bpc_file_is_a_loud_read_error():
    """A file that is not a BPC (here the DE440 SPK) is a read error, not a
    silent ``satisfied == False`` — the two failure axes stay distinct."""
    with pytest.raises(RuntimeError):
        _coverage(str(_warm_file("de440.bsp")), _PREDICT)


def test_an_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        _coverage(str(_warm_file(_PREDICT_SLOT)), 7)


# ── The link-point refusal (slot, linked file, coverage found, required) ──


def test_provisioning_names_a_short_predict_slot(tmp_path):
    """Linking a file that cannot cover the predict slot is refused at
    provisioning, naming all four facts — the mutation the contract exists
    for. A short file (the historical kernel) stands in for the shipped-wrong
    package so the test is deterministic regardless of the pip package."""
    source = _warm_file(_HISTORICAL_SLOT)
    link = tmp_path / _PREDICT_SLOT
    link.symlink_to(source)

    with pytest.raises(empyrean.KernelCoverageError) as excinfo:
        empyrean._verify_earth_orientation_slot("earth_predict", _PREDICT_SLOT, link, source)

    e = excinfo.value
    # (1) slot, (2) linked file, (3) coverage found, (4) required — all four,
    # as attributes and in the message.
    assert e.slot == _PREDICT_SLOT
    assert e.linked_file == str(source)
    assert e.coverage_found[1] < e.required[1]
    message = str(e)
    assert _PREDICT_SLOT in message
    assert source.name in message
    assert f"{e.required[1]:.0f}" in message


def test_provisioning_accepts_a_sufficient_predict_slot(tmp_path):
    """The converse: a kernel that does reach the mandate horizon provisions
    without a refusal (the check does not false-positive)."""
    source = _warm_file(_PREDICT_SLOT)
    link = tmp_path / _PREDICT_SLOT
    link.symlink_to(source)
    # Must not raise.
    empyrean._verify_earth_orientation_slot("earth_predict", _PREDICT_SLOT, link, source)


# ── Clean data dir, provisioned from the installed pip data packages ──


def test_clean_data_dir_provisioning_from_pip_packages(tmp_path, monkeypatch):
    """Provision an EMPTY data dir from the installed pip data packages, no
    network, and verify the link-time contract on exactly what pip ships.

    The shipped predict kernel decides the outcome: a package whose predict
    kernel reaches the 2125 mandate provisions cleanly; a package that ships
    a short kernel (the incident) is refused at the link point, naming the
    slot and the file. Either way the contract's verdict matches the file —
    this is the fresh-install path the warm data dir used to mask.
    """
    b612 = empyrean._discover_b612_data()
    needed = set(empyrean._B612_TO_VILLENEUVE_FILENAME)
    absent = needed - set(b612)
    if absent:
        pytest.skip(f"pip data packages not installed: {sorted(absent)}")

    monkeypatch.setenv("EMPYREAN_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("EMPYREAN_OFFLINE", raising=False)

    predict_src = b612["earth_predict"]
    predict_reaches_mandate = _coverage(predict_src, _PREDICT)[0]

    if predict_reaches_mandate:
        empyrean._stage_b612_cache(b612)
        assert (tmp_path / _PREDICT_SLOT).exists()
    else:
        with pytest.raises(empyrean.KernelCoverageError) as excinfo:
            empyrean._stage_b612_cache(b612)
        assert excinfo.value.slot == _PREDICT_SLOT
        assert excinfo.value.linked_file == predict_src
