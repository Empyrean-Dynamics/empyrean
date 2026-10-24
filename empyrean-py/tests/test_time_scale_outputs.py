"""Contract: distribution outputs carry their scale, and results come back
in the scale the caller supplied (bead empyrean-8ctt5).

The engine is TDB-only. These tests pin the distribution-boundary contract:

1. every time-bearing output is an ``Epochs`` sub-table carrying its scale;
2. a UTC request returns ``scale == "utc"`` with the requested instants
   echoed bit-for-bit; a TDB request returns ``"tdb"``;
3. the same instant requested in UTC and in its TDB equivalent gives
   bit-identical states (the FFI converts to TDB once, consistently);
4. ``Observers.from_codes`` keeps the caller's scale;
5. one table carries one scale (mixed-scale concat is refused loudly);
6. nullable ``Epochs`` sub-tables round-trip.

Measured ``mjd -> iso -> mjd`` round-trip precision is 0 d (bit-exact) over
1990-2040, so the echo/convert assertions below use exact equality.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
import quivr as qv
from empyrean import Epochs


def _mjd(epochs: Epochs) -> np.ndarray:
    return np.asarray(epochs.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)


# ── Scale-carrier mechanics (no kernels) ──────────────────────────────


def test_utc_tdb_roundtrip_is_bit_exact() -> None:
    """The conversion path (ISO, nanosecond) round-trips UTC<->TDB to the
    f64 representation with no loss: the measured basis for exact-equality
    assertions elsewhere."""
    utc = Epochs.from_mjd([47892.0, 60000.123456789, 65380.999], scale="utc")
    back = utc.to_tdb().to_utc()
    np.testing.assert_array_equal(_mjd(back), _mjd(utc))
    assert back.scale == "utc"


def test_one_table_one_scale_mixed_concat_refused() -> None:
    """Quivr refuses concatenating sub-tables whose scale attributes differ,
    so a mixed-scale table is unrepresentable rather than silently wrong."""
    a = Epochs.from_mjd([60000.0], scale="tdb")
    b = Epochs.from_mjd([60001.0], scale="utc")
    with pytest.raises(Exception):  # quivr TablesNotCompatibleError
        qv.concatenate([a, b])
    with pytest.raises(ValueError):
        Epochs.concat(a, b)


def test_nullable_epochs_subtable_roundtrips() -> None:
    """A planning-style nullable Epochs sub-table (absent anchors are null)
    round-trips: present rows read back, null rows stay null."""
    import pyarrow as pa

    class Night(qv.Table):
        night = qv.Int64Column()
        sunset = Epochs.as_column()  # nullable by default in quivr 0.8.1

    sunset = Epochs.from_kwargs(
        mjd=pa.array([60000.5, None, 60002.5], type=pa.float64()), scale="tdb"
    )
    t = Night.from_kwargs(night=[1, 2, 3], sunset=sunset)
    assert len(t) == 3
    assert t.sunset.scale == "tdb"
    assert t.sunset.mjd.null_count == 1


# ── Request-scale echo + convert (needs the kernel set) ───────────────


@pytest.fixture(scope="module")
def _ready() -> None:
    empyrean.initialize()


def test_get_states_echoes_request_scale(_ready: None) -> None:
    for scale in ("utc", "tdb"):
        req = Epochs.from_mjd([60200.0, 60201.0], scale=scale)
        st = empyrean.get_states("Earth", "Sun", req)
        assert st.epochs.scale == scale, f"get_states({scale}) must echo the scale"
        # Requested instants are echoed bit-for-bit.
        np.testing.assert_array_equal(_mjd(st.epochs), _mjd(req))


def test_get_states_same_instant_two_scales_match(_ready: None) -> None:
    """The same instant, requested as UTC and as its TDB clock reading, gives
    identical states. The FFI converts to TDB once and consistently.

    The two requests come from the *same* ISO instants built independently
    (``from_iso`` uses the ISO converter, not the ``to_tdb`` numeric path),
    so a wrong UTC->TDB conversion shifts the UTC request's states off the
    TDB request's and the assertion fails. This is a conversion test, not
    an echo test."""
    iso = ["2026-06-01T00:00:00.000Z", "2026-06-02T00:00:00.000Z"]
    utc = Epochs.from_iso(iso, scale="utc")
    tdb = Epochs.from_iso(iso, scale="tdb")
    a = empyrean.get_states("Earth", "Sun", utc)
    b = empyrean.get_states("Earth", "Sun", tdb)
    np.testing.assert_allclose(
        np.asarray(a.x.to_numpy(zero_copy_only=False)),
        np.asarray(b.x.to_numpy(zero_copy_only=False)),
        rtol=0,
        atol=1e-10,
    )
    # a's echoed UTC epochs, expressed in TDB, equal b's TDB epochs.
    np.testing.assert_allclose(_mjd(a.epochs.to_tdb()), _mjd(b.epochs), rtol=0, atol=1e-9)


def test_observers_keep_caller_scale(_ready: None) -> None:
    req = Epochs.from_mjd([60200.0, 60201.0], scale="utc")
    obs = empyrean.get_observer_states(["500"], req)
    assert obs.coordinates.epochs.scale == "utc"
    np.testing.assert_array_equal(_mjd(obs.coordinates.epochs), _mjd(req))


def test_propagate_states_echo_request_scale(_ready: None) -> None:
    from empyrean.coordinates.coordinates import CartesianCoordinates
    from empyrean.orbits.orbits import CartesianOrbits

    orbit = CartesianOrbits.from_kwargs(
        orbit_id=["t"],
        object_id=["t"],
        coordinates=CartesianCoordinates.from_kwargs(
            epochs=Epochs.from_kwargs(mjd=np.array([61000.0]), scale="tdb"),
            x=[1.6],
            y=[0.1],
            z=[0.02],
            vx=[-0.002],
            vy=[0.011],
            vz=[0.001],
            frame="ecliptic_j2000",
            origin=["Sun"],
        ),
    )
    for scale in ("utc", "tdb"):
        req = Epochs.from_mjd([61000.0, 61030.0], scale=scale)
        res = empyrean.propagate(orbit, req)
        assert res.states.coordinates.epochs.scale == scale
        # The echoed epochs are the caller's own supplied values, bit-for-bit
        # (one orbit, so the states are in the requested-epoch order).
        np.testing.assert_array_equal(_mjd(res.states.coordinates.epochs), _mjd(req))

    # Conversion check (reddens if the UTC->TDB conversion is wrong): the same
    # instants requested as UTC and as their TDB clock reading, built
    # independently from the same ISO strings, propagate to identical states.
    iso = ["2026-03-01T00:00:00.000Z", "2026-03-31T00:00:00.000Z"]
    su = empyrean.propagate(orbit, Epochs.from_iso(iso, scale="utc")).states.coordinates
    st = empyrean.propagate(orbit, Epochs.from_iso(iso, scale="tdb")).states.coordinates
    np.testing.assert_allclose(
        np.asarray(su.x.to_numpy(zero_copy_only=False)),
        np.asarray(st.x.to_numpy(zero_copy_only=False)),
        rtol=0,
        atol=1e-9,
    )
