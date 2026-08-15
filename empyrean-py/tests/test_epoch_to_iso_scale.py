"""``Epochs.to_iso(scale=...)`` must convert, not silently relabel.

``to_iso`` always emits the UTC wall-clock time of the stored instant,
interpreting the stored MJD in the table's own scale. Passing a *different*
``scale`` used to forward that scale to the formatter as a reinterpretation of
the stored MJD — a TDB reading stamped ``Z`` as if it were UTC, a silent ~69 s
error at 2026 epochs. Option A (the honest surface): a mismatched ``scale``
raises; ``scale == self.scale`` or ``None`` keeps the correct path.
"""

from __future__ import annotations

import pytest
from empyrean.coordinates.epoch import Epochs, TimeScale


def test_to_iso_cross_scale_raises() -> None:
    """PROVING (fails on the pre-fix code, which relabelled instead of raising).

    A TDB table asked to format as UTC must raise rather than emit the raw TDB
    clock reading stamped ``Z`` (a +69.18 s silent error).
    """
    tdb = Epochs.from_mjd([61000.0], scale="tdb")
    with pytest.raises(ValueError, match="to_iso"):
        tdb.to_iso(scale="utc")
    # Enum form of the mismatched scale raises identically.
    with pytest.raises(ValueError):
        tdb.to_iso(scale=TimeScale.UTC)
    # …and the symmetric case: a UTC table asked to format as TDB.
    utc = Epochs.from_mjd([61000.0], scale="utc")
    with pytest.raises(ValueError):
        utc.to_iso(scale="tdb")


def test_to_iso_same_scale_and_default_are_the_honest_path() -> None:
    """REGRESSION GUARD + PROVING the convert path is preserved.

    ``scale is None`` and ``scale == self.scale`` both format the stored
    instant honestly, and the honest cross-scale conversion
    (``to_utc().to_iso()``) round-trips to the same UTC wall-clock — the
    equality this pins (TDB−UTC = 69.1832 s).
    """
    tdb = Epochs.from_mjd([61000.0], scale="tdb")
    default = tdb.to_iso()
    # scale == self.scale is a no-op guard, not a mismatch.
    assert tdb.to_iso(scale="tdb") == default
    assert tdb.to_iso(scale=TimeScale.TDB) == default
    # The honest conversion path (convert first, then bare .to_iso) yields the
    # same UTC wall-clock of the stored instant.
    assert tdb.to_utc().to_iso() == default
