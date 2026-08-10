"""``generate_ephemeris`` must not substitute observer geometry.

The binding recomputes every observer from its ``(obs_code, epoch)``
rather than trusting the table's own state columns, so that
``observing_night`` and the engine's internal rounding match what the
Rust path sees. That recompute used to be wrapped in a catch-all that
fell back to the caller's position / velocity columns with
``observing_night = -1`` on *any* failure.

That is the shape of a silent degrade. An epoch outside the loaded BPC's
coverage window, or a code the registry does not know, produced
astrometry computed from stale geometry, returned with a clean success
and no warning, and threw away the nightly grouping the OD weighting
depends on — after the engine had gone to the trouble of splitting its
errors by remedy ("retry with different arguments" vs "fetch the
kernel").
"""

from __future__ import annotations

import empyrean
import pyarrow as pa
import pytest
from empyrean import Observers


@pytest.fixture(scope="module")
def orbits():
    empyrean.initialize()
    return empyrean.query_sbdb(["99942"])


def _observers_with_code(code: str) -> Observers:
    """A well-formed Observers table carrying real states from code 500
    but relabelled to `code` — so the state columns are usable and only
    the lookup fails. That is exactly the situation the old fallback was
    silently rescuing."""
    real = Observers.from_code("500", [61000.5, 61010.5])
    table = real.table
    idx = table.schema.get_field_index("obs_code")
    relabelled = table.set_column(
        idx,
        "obs_code",
        pa.array([code] * len(real), type=table.column("obs_code").type),
    )
    return Observers.from_pyarrow(relabelled)


def test_an_unknown_observatory_code_surfaces_instead_of_falling_back(orbits) -> None:
    bogus = _observers_with_code("ZZZ")

    with pytest.raises(RuntimeError) as excinfo:
        empyrean.generate_ephemeris(orbits, bogus)

    message = str(excinfo.value)
    # The message has to name the code, the epoch, and the engine's own
    # reason — a bare "lookup failed" leaves the caller no better off
    # than the fallback did.
    assert "ZZZ" in message
    assert "61000.5" in message
    assert "unknown observatory code" in message


def test_a_known_code_still_generates(orbits) -> None:
    """The guard rejects failures, not observers."""
    good = Observers.from_code("500", [61000.5, 61010.5])
    result = empyrean.generate_ephemeris(orbits, good)
    assert len(result.ephemeris) == 2
    # `observing_night` comes from the recompute, and a real one is
    # never the -1 the fallback used to stamp.
    nights = result.ephemeris.table.column("obs_code").to_pylist()
    assert nights == ["500", "500"]
