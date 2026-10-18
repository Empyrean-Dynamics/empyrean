"""Band→V conversions on the post-OD photometric fit, through the Python layer.

Three things are proven end to end (Python dict → PyO3 → Rust wrapper → C ABI
→ engine):

* the config serializes ``band_conversions`` onto the wire only when supplied,
  so the default leaves the fit on the MPC Magnitude Bands table;
* with the default table, each converted band's ``offset_applied`` reports the
  MPC value (spot-checked against the pinned table) — the audit trail;
* a supplied entry replaces a band's offset and ``offset_applied`` reports the
  supplied value, so a supplied conversion enters the fit.

The LSST ``L*`` codes carry no MPC default and are refused by name until
supplied; that plumbing is pinned at the C-ABI config level
(``photometry_config_from_c_builds_the_band_table``) and in the engine
(villeneuve's band table). The Apophis fixture carries no LSST band, so the
"supplied entry enters the fit" property is proven here by replacing a band the
arc does have (``r``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from empyrean import determine, read_ades
from empyrean.od.result import ODConfig, PhotometryConfig

APOPHIS = Path(__file__).parent / "fixtures" / "99942_apophis_multiapp.psv"

# Spot-checks from the MPC Magnitude Bands table (2026-10-16 snapshot).
_MPC = {"r": 0.23, "R": 0.4, "V": 0.0, "G": 0.28, "w": 0.16, "Ao": 0.33, "Ac": -0.05}


@pytest.fixture(scope="module")
def apophis_observations():
    if not APOPHIS.exists():
        pytest.skip(f"missing fixture: {APOPHIS}")
    optical, _radar = read_ades(APOPHIS)
    return optical


def test_default_config_omits_band_conversions_from_the_wire() -> None:
    """``None`` band_conversions emits no key, so the engine runs the MPC
    table; a supplied mapping is emitted as ``[offset, sigma]`` pairs."""
    default = ODConfig(photometry=PhotometryConfig())._to_wire_dict()
    assert "band_conversions" not in default["photometry"]

    supplied = ODConfig(
        photometry=PhotometryConfig(band_conversions={"Lu": (-1.41, 0.30)})
    )._to_wire_dict()
    assert supplied["photometry"]["band_conversions"] == {"Lu": [-1.41, 0.30]}


def test_default_table_reports_mpc_offsets(apophis_observations) -> None:
    """With the default (MPC) table, each converted band's ``offset_applied``
    is the pinned MPC value."""
    fit = determine(apophis_observations, config=ODConfig(photometry=PhotometryConfig())).single()
    assert fit.photometry is not None, "photometric fit did not run"
    by_band = {b.band: b.offset_applied for b in fit.photometry.per_band}
    # At least one spot-checkable band must be present, and every one that is
    # present must carry its MPC offset.
    checked = [c for c in _MPC if c in by_band]
    assert checked, f"no spot-checkable band in the fit; bands = {sorted(by_band)}"
    for code in checked:
        assert by_band[code] == pytest.approx(_MPC[code], abs=1e-12), (
            f"band {code!r} offset_applied {by_band[code]} != MPC {_MPC[code]}"
        )


def test_supplied_entry_enters_the_fit_and_offset_applied_reports_it(
    apophis_observations,
) -> None:
    """A supplied conversion for a band the arc has replaces the MPC value: the
    per-band ``offset_applied`` reports the supplied offset, proving the entry
    rode the whole chain into the fit."""
    base = determine(apophis_observations, config=ODConfig(photometry=PhotometryConfig())).single()
    base_bands = {b.band: b.offset_applied for b in base.photometry.per_band}
    assert "r" in base_bands and base_bands["r"] == pytest.approx(0.23, abs=1e-12)

    supplied = determine(
        apophis_observations,
        config=ODConfig(photometry=PhotometryConfig(band_conversions={"r": (0.50, 0.05)})),
    ).single()
    supplied_bands = {b.band: b.offset_applied for b in supplied.photometry.per_band}
    assert supplied_bands["r"] == pytest.approx(0.50, abs=1e-12), (
        f"supplied r offset did not enter the fit: offset_applied = {supplied_bands['r']}"
    )
    # The other bands are untouched — a supplied entry replaces one code, not
    # the whole table.
    assert supplied_bands.get("V", 0.0) == pytest.approx(0.0, abs=1e-12)
