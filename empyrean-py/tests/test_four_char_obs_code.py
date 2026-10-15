"""Four-character MPC observatory codes must survive the FFI round trip.

The 2026 MPC rollout introduces four-character station codes (``W68a``-class).
The C-ABI observatory-code field is four bytes; a four-character code must
reach the engine and return from it with all four characters intact, never
clipped to a three-byte prefix that silently names a *different*
observatory (wrong topocentric geometry, no diagnostic).
"""

from __future__ import annotations

from pathlib import Path

import empyrean
import pyarrow as pa
import pytest
from empyrean import Epochs, determine, get_observer_states, read_ades
from empyrean.od.ades_observations import ADESObservations

FIXTURE = Path(__file__).parent / "fixtures" / "99942_apophis_multiapp.psv"


@pytest.fixture(scope="module")
def apophis_optical() -> ADESObservations:
    empyrean.initialize()
    if not FIXTURE.exists():
        pytest.skip(f"missing fixture: {FIXTURE}")
    optical, _radar = read_ades(FIXTURE)
    return optical


def _relabel_stn(obs: ADESObservations, code: str) -> ADESObservations:
    """Return a copy of ``obs`` with every station code set to ``code``."""
    table = obs.table
    idx = table.schema.get_field_index("stn")
    relabelled = table.set_column(
        idx, "stn", pa.array([code] * len(obs), type=table.column("stn").type)
    )
    return ADESObservations.from_pyarrow(relabelled)


def test_four_char_code_reaches_engine_observer_resolution() -> None:
    """A four-character code handed to observer resolution reaches the
    engine with all four characters. The engine's registry has no such
    code, so it refuses by name — and the error carries the *full* code.
    Clipped to ``W68`` it would resolve to a real, different observatory
    and never error.
    """
    empyrean.initialize()
    with pytest.raises(RuntimeError) as excinfo:
        get_observer_states(["W68a"], Epochs.from_mjd([61000.5], scale="tdb"))
    assert "W68a" in str(excinfo.value), (
        f"the engine error must name the full four-character code: {excinfo.value}"
    )


def test_four_char_code_round_trips_through_od_fit(apophis_optical) -> None:
    """The observatory code crosses the OD observation-input field into the
    engine and returns in the per-observation residual rows. With every
    station relabelled to a four-character code, the fitted result's codes
    must read back as those same four characters — never a three-byte
    prefix.

    This exercises both ``[u8; 4]`` carriers on the path (the observation
    input ``EmpyreanObservation.obs_code`` and the result
    ``EmpyreanObservationResult.obs_code``). Restoring any ``min(3)`` /
    ``[..3]`` clip on the path reddens this: the code reaches the engine as
    ``"W68"`` instead, so the result carries ``"W68"`` or the fit fails on
    the wrong geometry with a message that never names ``"W68a"``.
    """
    relabelled = _relabel_stn(apophis_optical, "W68a")
    try:
        result = determine(relabelled).single()
    except (RuntimeError, ValueError) as err:
        # If the fit raises, the engine must at least have seen the full
        # four-character code — a clipped ``"W68"`` would fail without ever
        # naming it.
        assert "W68a" in str(err), (
            f"a failing fit must still name the full four-character code: {err}"
        )
        return
    codes = set(result.observations.obs_code.to_pylist())
    assert codes == {"W68a"}, (
        f"four-character code must round-trip through the engine intact, got {codes}"
    )
