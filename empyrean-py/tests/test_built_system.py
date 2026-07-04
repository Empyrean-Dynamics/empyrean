"""Tests for the reusable pre-built force-model handle (BuiltSystem).

Covers construction, forward-model parity with the one-shot entry points
(the handle only changes *when* the force model is assembled, never the
numerics), the identity guard (a loud, distinct error on a key mismatch —
never a silent rebuild), full provenance from ``describe()``, and safe
cross-thread sharing of a single handle.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import empyrean
import numpy as np
import pytest
from empyrean import (
    BuiltSystem,
    ForceModelTier,
    Frame,
    KernelProvenance,
    PropagationConfig,
    SystemDescription,
    build_system,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.epoch import Epochs
from empyrean.observers.state import get_observer_states
from empyrean.orbits.orbits import CartesianOrbits


def _orbit() -> CartesianOrbits:
    """A gravity-only heliocentric ICRF Cartesian orbit (AU, AU/day)."""
    coords = CartesianCoordinates.from_kwargs(
        epoch=[59000.0],
        x=[1.0],
        y=[0.1],
        z=[0.05],
        vx=[-0.005],
        vy=[0.015],
        vz=[0.001],
        frame="icrf",
        origin=["Sun"],
    )
    return CartesianOrbits.from_kwargs(orbit_id=["a"], coordinates=coords)


def _epochs() -> np.ndarray:
    return np.array([59000.0, 59010.0, 59030.0])


# ── Construction / frozen key ─────────────────────────────────


def test_build_system_freezes_key() -> None:
    system = build_system(force_model="standard", frame="icrf")
    assert isinstance(system, BuiltSystem)
    assert system.force_model is ForceModelTier.STANDARD
    assert system.frame is Frame.ICRF
    # 0.0 sentinel resolves to the engine default divisor.
    assert system.encounter_timescale_divisor == 1000.0
    assert "standard" in repr(system) and "icrf" in repr(system)


def test_build_system_accepts_enums_and_ints() -> None:
    a = build_system(force_model=ForceModelTier.BASIC, frame=Frame.ECLIPTICJ2000)
    b = build_system(force_model=1, frame=1)
    assert a.force_model is b.force_model is ForceModelTier.BASIC
    assert a.frame is b.frame is Frame.ECLIPTICJ2000


# ── Forward-model parity with the one-shot ────────────────────


def test_propagate_matches_one_shot() -> None:
    orbits, epochs = _orbit(), _epochs()
    cfg = PropagationConfig(force_model=ForceModelTier.STANDARD, frame=Frame.ICRF)

    one_shot = empyrean.propagate(orbits, epochs, cfg)
    system = build_system(force_model="standard", frame="icrf")
    via = system.propagate(orbits, epochs, cfg)

    assert len(via.states) == len(one_shot.states) == len(epochs)
    osc, vhc = one_shot.states.coordinates, via.states.coordinates
    for name in ("x", "y", "z", "vx", "vy", "vz"):
        a = getattr(osc, name).to_numpy(zero_copy_only=False)
        b = getattr(vhc, name).to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(a, b, err_msg=f"{name} not bit-identical")


def test_propagate_default_config_uses_frozen_key() -> None:
    # No config passed → the handle builds one from its frozen key, so the
    # call matches by construction and runs without a guard trip.
    system = build_system(force_model="standard", frame="eclipticj2000")
    via = system.propagate(_orbit(), _epochs())
    one_shot = empyrean.propagate(
        _orbit(),
        _epochs(),
        PropagationConfig(force_model=ForceModelTier.STANDARD, frame=Frame.ECLIPTICJ2000),
    )
    assert len(via.states) == len(one_shot.states)
    np.testing.assert_array_equal(
        via.states.coordinates.x.to_numpy(zero_copy_only=False),
        one_shot.states.coordinates.x.to_numpy(zero_copy_only=False),
    )


def test_generate_ephemeris_matches_one_shot() -> None:
    orbits = _orbit()
    observers = get_observer_states(["500"], Epochs.from_kwargs(mjd=[59000.0], scale="tdb"))

    one_shot = empyrean.generate_ephemeris(orbits, observers)
    # The ephemeris pipeline integrates in EclipticJ2000.
    system = build_system(force_model="standard", frame="eclipticj2000")
    via = system.generate_ephemeris(orbits, observers)

    assert len(via.ephemeris) == len(one_shot.ephemeris)
    assert len(one_shot.ephemeris) > 0
    oe, ve = one_shot.ephemeris.coordinates, via.ephemeris.coordinates
    for name in ("lon", "lat", "rho"):
        a = getattr(oe, name).to_numpy(zero_copy_only=False)
        b = getattr(ve, name).to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(a, b, err_msg=f"ephemeris {name} not bit-identical")


# ── Provenance ────────────────────────────────────────────────


def test_describe_reports_full_provenance() -> None:
    system = build_system(force_model="standard", frame="icrf")
    desc = system.describe()

    assert isinstance(desc, SystemDescription)
    assert desc.force_model is ForceModelTier.STANDARD
    assert desc.frame is Frame.ICRF
    assert desc.encounter_timescale_divisor == 1000.0
    assert desc.relativistic is True
    assert desc.asteroids is True
    assert desc.has_bpc is True
    assert len(desc.perturber_origins) > 0
    assert all(isinstance(o, int) for o in desc.perturber_origins)
    assert len(desc.kernels) > 0


def test_describe_kernel_records_are_populated() -> None:
    desc = build_system(force_model="standard", frame="icrf").describe()

    file_records = [k for k in desc.kernels if k.provenance is KernelProvenance.FILE]
    assert file_records, "expected at least one FILE-provenance kernel"

    for rec in file_records:
        # A FILE record carries a well-formed hash + byte count matching disk.
        assert rec.path is not None
        assert rec.sha256 is not None and len(rec.sha256) == 64
        assert all(c in "0123456789abcdef" for c in rec.sha256)
        assert rec.bytes is not None and rec.bytes > 0
        assert os.path.getsize(rec.path) == rec.bytes
        # FILE records leave the BUILT_IN-only field unset.
        assert rec.name is None

    # Non-FILE records carry no path/hash — never a defaulted stand-in.
    for rec in desc.kernels:
        if rec.provenance is not KernelProvenance.FILE:
            assert rec.path is None and rec.sha256 is None and rec.bytes is None


# ── Identity guard (loud, distinct, never a silent rebuild) ───


def test_guard_fires_on_force_model_mismatch() -> None:
    system = build_system(force_model="standard", frame="icrf")
    bad = PropagationConfig(force_model=ForceModelTier.BASIC, frame=Frame.ICRF)
    with pytest.raises(ValueError, match="force-model mismatch"):
        system.propagate(_orbit(), _epochs(), bad)


def test_guard_fires_on_frame_mismatch() -> None:
    system = build_system(force_model="standard", frame="icrf")
    bad = PropagationConfig(force_model=ForceModelTier.STANDARD, frame=Frame.ECLIPTICJ2000)
    with pytest.raises(ValueError, match="frame mismatch"):
        system.propagate(_orbit(), _epochs(), bad)


def test_guard_fires_on_divisor_mismatch() -> None:
    # Freeze a non-default divisor; a default-divisor config trips the guard.
    system = build_system(force_model="standard", frame="icrf", encounter_timescale_divisor=500.0)
    assert system.encounter_timescale_divisor == 500.0
    cfg = PropagationConfig(force_model=ForceModelTier.STANDARD, frame=Frame.ICRF)  # divisor 1000
    with pytest.raises(ValueError, match="divisor mismatch"):
        system.propagate(_orbit(), _epochs(), cfg)


def test_guard_message_says_rebuild_not_silent() -> None:
    # The error must instruct rebuilding — proving no silent fallback.
    system = build_system(force_model="standard", frame="icrf")
    bad = PropagationConfig(force_model=ForceModelTier.BASIC, frame=Frame.ICRF)
    with pytest.raises(ValueError, match="identity guard"):
        system.propagate(_orbit(), _epochs(), bad)


# ── Cross-thread sharing (Send + Sync usability) ──────────────


def test_handle_shared_across_threads() -> None:
    # One handle, many worker threads calling &self concurrently. The
    # native handle is Send + Sync and each call releases the GIL, so this
    # must not deadlock, corrupt, or diverge from the one-shot result.
    orbits, epochs = _orbit(), _epochs()
    cfg = PropagationConfig(force_model=ForceModelTier.STANDARD, frame=Frame.ICRF)
    system = build_system(force_model="standard", frame="icrf")
    expected = system.propagate(orbits, epochs, cfg).states.coordinates.x.to_numpy(
        zero_copy_only=False
    )

    def work(_: int) -> np.ndarray:
        return system.propagate(orbits, epochs, cfg).states.coordinates.x.to_numpy(
            zero_copy_only=False
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(32)))

    for r in results:
        np.testing.assert_array_equal(r, expected)
