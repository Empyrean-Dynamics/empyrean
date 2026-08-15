"""Cross-validate our UTC↔TDB conversions against astropy.

astropy is an independent implementation of the same standards (it
converts through ERFA, the SOFA-derived C library), so it is a genuine
external reference rather than a second copy of our own arithmetic.
It is a **test-only** dependency: the package itself imports astropy
only inside the optional :meth:`Epochs.from_astropy` / :meth:`to_astropy`
interop, never on a conversion path.

What our conversion carries
---------------------------

``Epochs.to_tdb`` / ``to_utc`` route through the engine's
``_convert_epochs``, which applies the leap-second table for UTC↔TAI↔TT
and the Fairhead & Bretagnon (1990) TDB−TT series for TT↔TDB. That
series is the **full periodic** one, not a secular-only truncation —
which is why the tolerances below are as tight as they are. The
measured TDB−UTC offset varies by 3.3148 ms peak-to-peak over the
six-year grid sampled below (MJD 60300–62500), and astropy's varies by
the same 3.3148 ms; a secular-only implementation would show no such
variation and would sit up to ~1.7 ms from astropy. Off leap-second days
we agree with astropy **bit for bit** over the modern era, so
:data:`_EXACT` is exact equality rather than a tolerance — anything
looser would hide a regression.

A tight assertion here is deliberate. If the engine's leap-second table
falls behind IERS while astropy's does not, these tests go red, and
that failure is the point: it is a data-currency signal, not noise.

The one place we differ
-----------------------

On a UTC day that does **not** contain 86400 seconds — the 27 historical
leap-second days, and the pre-1972 "rubber second" days — a *fractional*
MJD is converted as if the day were exactly 86400 s long. The error
grows linearly across such a day, from 0 at 00:00 to the full 1 s leap
at 24:00. :func:`test_leap_second_day_fractional_mjd_matches_astropy`
pins that as a known defect with ``xfail(strict=True)``, so it stays
visible and the marker is forced off the moment it is fixed. Midnight on
those days, and every instant not on such a day, is exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from empyrean import Epochs

astropy_time = pytest.importorskip("astropy.time", reason="astropy is a test-only dependency")
Time = astropy_time.Time


# Off leap-second days our conversion is bit-identical to astropy's over
# the modern era (verified 1000/1000 samples, 0 ULP). Assert that, not a
# tolerance: a tolerance would silently absorb a real regression.
_EXACT = 0.0

# Sub-microsecond, for comparisons where float64 MJD resolution alone
# (~1.3 µs at MJD 61000) is the floor rather than the algorithm.
_FLOAT64_MJD_FLOOR_S = 2e-6


def _seconds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference between two MJD arrays, in seconds."""
    return np.abs(np.asarray(a) - np.asarray(b)) * 86400.0


def _leap_days_from(offsets: np.ndarray, days: np.ndarray) -> set[int]:
    """Day-boundary MJDs where TDB−UTC steps by one second."""
    step = np.diff(offsets)
    return set(days[:-1][np.abs(step - 1.0) < 0.01].astype(int).tolist())


def _tdb_minus_utc(days: np.ndarray) -> np.ndarray:
    return (Epochs.from_mjd(days, scale="utc").to_tdb().to_numpy() - days) * 86400.0


def _tdb_minus_utc_astropy(days: np.ndarray) -> np.ndarray:
    return (Time(days, format="mjd", scale="utc").tdb.mjd - days) * 86400.0


# Every day boundary from 1972-01-01 (when integer leap seconds began)
# to a little past the present. The leap days are *derived* from this
# rather than hard-coded: a hand-maintained table is one typo away from
# testing the wrong thing, and deriving it turns the table itself into
# something the astropy comparison can check.
#
# Derived from ASTROPY, deliberately, even though the engine could
# answer it too. This list is a parametrize argument, so it is built
# during collection — before conftest's autouse fixture has had a chance
# to turn a missing kernel set or an unloadable engine into a skip. An
# engine call here would raise at collection instead, taking the rest of
# the suite's collection report with it. astropy is already guarded by
# the importorskip above, so deriving from it is collection-safe, and
# our own table is checked against astropy's entry for entry by
# test_our_leap_second_table_matches_astropys_entry_for_entry below.
_LEAP_SCAN_DAYS = np.arange(41317.0, 61200.0)
_LEAP_SECOND_DAYS = sorted(
    _leap_days_from(_tdb_minus_utc_astropy(_LEAP_SCAN_DAYS), _LEAP_SCAN_DAYS)
)

# Whole days only: a fractional MJD on a leap day hits the known defect
# documented in the module docstring and pinned separately below.
_MODERN = np.linspace(60300.0, 62500.0, 500)
_PRE_1972 = np.array([37300.0, 38000.0, 39000.0, 40000.0, 41316.0])
_FUTURE = np.array([64328.0, 70000.0, 77000.0, 88000.0])


# ── Forward: UTC → TDB ────────────────────────────────────────────────


def test_utc_to_tdb_matches_astropy_modern_era() -> None:
    """The regime that matters for every current-epoch workflow."""
    ours = Epochs.from_mjd(_MODERN, scale="utc").to_tdb().to_numpy()
    reference = Time(_MODERN, format="mjd", scale="utc").tdb.mjd
    assert _seconds(ours, reference).max() <= _EXACT


def test_tdb_to_utc_matches_astropy_modern_era() -> None:
    """The reverse leg, independently — not just the round trip."""
    tdb = Time(_MODERN, format="mjd", scale="utc").tdb.mjd
    ours = Epochs.from_mjd(tdb, scale="tdb").to_utc().to_numpy()
    reference = Time(tdb, format="mjd", scale="tdb").utc.mjd
    assert _seconds(ours, reference).max() <= _EXACT


def test_the_tdb_tt_series_is_periodic_not_secular() -> None:
    """We carry the full Fairhead–Bretagnon series, and this proves it.

    A secular-only TDB−TT would make the TDB−UTC offset a smooth ramp
    between leap seconds. The real series adds an annual-dominated
    periodic term of a few milliseconds peak-to-peak. Our offset varies
    by the same amount as astropy's, to the microsecond — so the
    periodic terms are present, and the tight tolerances elsewhere in
    this file are justified rather than lucky.
    """
    ours = (Epochs.from_mjd(_MODERN, scale="utc").to_tdb().to_numpy() - _MODERN) * 86400.0
    reference = (Time(_MODERN, format="mjd", scale="utc").tdb.mjd - _MODERN) * 86400.0

    our_spread = ours.max() - ours.min()
    reference_spread = reference.max() - reference.min()

    assert our_spread > 2e-3, (
        f"TDB-UTC varies by only {our_spread * 1e3:.4f} ms across the grid; "
        f"a full periodic series should show a few ms. Secular-only?"
    )
    assert abs(our_spread - reference_spread) < 1e-6, (
        f"periodic amplitude disagrees with astropy: ours "
        f"{our_spread * 1e3:.4f} ms vs {reference_spread * 1e3:.4f} ms"
    )


# ── Leap seconds ──────────────────────────────────────────────────────


@pytest.mark.parametrize("day", _LEAP_SECOND_DAYS)
def test_leap_second_day_midnight_matches_astropy(day: int) -> None:
    """Midnight on a leap day is exact — the day-length defect is
    proportional to the fraction elapsed, so it vanishes at 00:00."""
    boundaries = np.array([float(day), float(day + 1)])
    ours = Epochs.from_mjd(boundaries, scale="utc").to_tdb().to_numpy()
    reference = Time(boundaries, format="mjd", scale="utc").tdb.mjd
    assert _seconds(ours, reference).max() <= _EXACT


def test_our_leap_second_table_matches_astropys_entry_for_entry() -> None:
    """The data-currency gate.

    Both sides are asked the same question — on which day boundaries
    does TDB−UTC step by a second? — and must name the same set. If the
    engine's leap table falls behind IERS while astropy's advances (or
    the reverse), this fails and names the offending days. That failure
    is the signal, not noise: it means one of the two is stale.
    """
    ours = _leap_days_from(_tdb_minus_utc(_LEAP_SCAN_DAYS), _LEAP_SCAN_DAYS)
    reference = _leap_days_from(_tdb_minus_utc_astropy(_LEAP_SCAN_DAYS), _LEAP_SCAN_DAYS)

    assert ours == reference, (
        f"leap tables disagree. only in ours: {sorted(ours - reference)}; "
        f"only in astropy: {sorted(reference - ours)}"
    )
    assert len(ours) == 27, (
        f"expected the 27 leap seconds inserted between 1972 and 2016, found "
        f"{len(ours)}. A 28th would mean IERS announced a new one — update "
        f"this count deliberately, with the engine's table refreshed."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT: a fractional MJD on a leap-second day is converted as "
        "if the day were 86400 s long, so the error ramps linearly from 0 at "
        "00:00 to the full 1 s at 24:00 (measured 900.000 ms at fraction 0.9, "
        "on all 27 historical leap days). Whole-day MJDs and every non-leap "
        "day are exact. When this is fixed, this test starts passing and "
        "strict xfail fails the suite until the marker is removed."
    ),
)
def test_leap_second_day_fractional_mjd_matches_astropy() -> None:
    """Pins the one place our conversion departs from astropy."""
    fractions = np.array([0.25, 0.5, 0.75, 0.9])
    worst = 0.0
    for day in _LEAP_SECOND_DAYS:
        grid = day + fractions
        ours = Epochs.from_mjd(grid, scale="utc").to_tdb().to_numpy()
        reference = Time(grid, format="mjd", scale="utc").tdb.mjd
        worst = max(worst, _seconds(ours, reference).max())
    assert worst <= _FLOAT64_MJD_FLOOR_S, f"max deviation {worst * 1e3:.6f} ms"


def test_the_leap_day_defect_has_the_magnitude_we_documented() -> None:
    """The defect is bounded and linear — assert its shape, not just that
    it exists, so a change in its character shows up as a failure here
    rather than as a silently different wrong answer."""
    day = 57753  # 2016-12-31
    for fraction in (0.25, 0.5, 0.75, 0.9):
        grid = np.array([day + fraction])
        ours = Epochs.from_mjd(grid, scale="utc").to_tdb().to_numpy()
        reference = Time(grid, format="mjd", scale="utc").tdb.mjd
        deviation = _seconds(ours, reference)[0]
        assert abs(deviation - fraction) < 1e-3, (
            f"fraction {fraction}: deviation {deviation:.6f} s is not the "
            f"expected linear {fraction} s day-stretch"
        )


# ── Regimes outside the modern era ────────────────────────────────────


def test_pre_1972_rubber_second_era_matches_astropy_at_day_boundaries() -> None:
    """Before 1972 UTC ran on "rubber seconds" — a rate offset plus
    step adjustments rather than integer leap seconds.

    We track astropy exactly at day boundaries, so the engine carries
    the pre-1972 UTC−TAI polynomials rather than clamping to the first
    leap entry. Fractional MJDs in this era hit the same day-length
    defect as leap days (those days are not 86400 s either), so this
    pins the boundaries.
    """
    ours = Epochs.from_mjd(_PRE_1972, scale="utc").to_tdb().to_numpy()
    reference = Time(_PRE_1972, format="mjd", scale="utc").tdb.mjd
    assert _seconds(ours, reference).max() <= _EXACT


def test_future_epochs_beyond_the_leap_table_match_astropy() -> None:
    """Past the last tabulated leap second both implementations hold the
    offset flat — no leap second is announced more than ~6 months out,
    so extrapolating one would be invention. We and astropy agree that
    TDB−UTC stays at its last known value."""
    ours = Epochs.from_mjd(_FUTURE, scale="utc").to_tdb().to_numpy()
    reference = Time(_FUTURE, format="mjd", scale="utc").tdb.mjd
    assert _seconds(ours, reference).max() <= _EXACT


def test_future_offset_is_held_flat_not_extrapolated() -> None:
    """The flat-hold is a deliberate behavior, so pin it directly."""
    offsets = (Epochs.from_mjd(_FUTURE, scale="utc").to_tdb().to_numpy() - _FUTURE) * 86400.0
    assert offsets.max() - offsets.min() < 5e-3, (
        f"TDB-UTC drifts by {(offsets.max() - offsets.min()) * 1e3:.3f} ms across "
        f"decades of future epochs; only the ~3 ms periodic term should vary"
    )
    assert 69.0 < offsets.mean() < 70.0, offsets.mean()


# ── Round-trip identity, our implementation alone ─────────────────────


@pytest.mark.parametrize(
    ("label", "grid"),
    [
        ("modern", _MODERN),
        ("pre-1972", _PRE_1972),
        ("future", _FUTURE),
        ("leap-day boundaries", np.array([float(d) for d in _LEAP_SECOND_DAYS])),
    ],
)
def test_utc_tdb_round_trip_is_the_identity(label: str, grid: np.ndarray) -> None:
    """``to_tdb().to_utc()`` returns the input to float64 precision."""
    back = Epochs.from_mjd(grid, scale="utc").to_tdb().to_utc().to_numpy()
    assert _seconds(back, grid).max() <= _FLOAT64_MJD_FLOOR_S, label


@pytest.mark.parametrize(
    ("label", "grid"),
    [("modern", _MODERN), ("pre-1972", _PRE_1972), ("future", _FUTURE)],
)
def test_tdb_utc_round_trip_is_the_identity(label: str, grid: np.ndarray) -> None:
    """And the other way around."""
    back = Epochs.from_mjd(grid, scale="tdb").to_utc().to_tdb().to_numpy()
    assert _seconds(back, grid).max() <= _FLOAT64_MJD_FLOOR_S, label


def test_to_scale_agrees_with_the_named_converters() -> None:
    """``to_scale`` is a dispatcher, not a third implementation."""
    epochs = Epochs.from_mjd(_MODERN, scale="utc")
    assert np.array_equal(epochs.to_scale("tdb").to_numpy(), epochs.to_tdb().to_numpy())
    assert np.array_equal(epochs.to_scale("utc").to_numpy(), epochs.to_utc().to_numpy())


def test_every_supported_scale_pair_is_covered() -> None:
    """If a third scale is ever added, this test fails and points here.

    The cross-validation above covers utc↔tdb in both directions, which
    is every ordered pair over the supported set. A new member of
    :class:`~empyrean.TimeScale` would need its own comparison rather
    than inheriting coverage silently.
    """
    from empyrean import TimeScale

    assert {s.value for s in TimeScale} == {"utc", "tdb"}, (
        "TimeScale gained a member; add its astropy cross-validation above "
        "before widening this assertion."
    )


# ── astropy interop round-trip ────────────────────────────────────────


@pytest.mark.parametrize("scale", ["utc", "tdb"])
def test_from_astropy_to_astropy_round_trip(scale: str) -> None:
    """The optional interop preserves both the values and the scale."""
    original = Time(_MODERN, format="mjd", scale=scale)
    epochs = Epochs.from_astropy(original)
    assert epochs.scale == scale
    assert _seconds(epochs.to_numpy(), original.mjd).max() <= _EXACT

    back = epochs.to_astropy()
    assert back.scale == scale
    assert _seconds(back.mjd, original.mjd).max() <= _EXACT
