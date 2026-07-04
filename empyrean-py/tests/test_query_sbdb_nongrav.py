"""Lock the SBDB non-grav round-trip.

``query_sbdb`` -> ``orbit_batch_dict_to_orbits`` must carry the FULL
Marsden-Sekanina g(r) — the exponents (alpha/r0/m/n/k) AND the thermal-lag
``dt`` — not just A1/A2/A3. Dropping the exponents silently re-propagates a
comet as inverse-square with no lag (the real 67P divergence that motivated
this fix).

These are network-free: they feed ``orbit_batch_dict_to_orbits`` the exact
flat dict shape ``_query_sbdb`` emits (see ``orbit_batch_to_pydict`` in
``empyrean-py/src/lib.rs``), so they lock the Python output marshaling
without hitting SBDB. The C-ABI / wrapper layers that populate the dict are
covered by ``orbits_to_batch`` reading ``NonGravModel::MarsdenSekanina``.
"""

from typing import Any

import numpy as np
from empyrean._convert import orbit_batch_dict_to_orbits

# 67P/Churyumov-Gerasimenko SBDB Marsden fit (the canonical divergence case).
P67_ALPHA = 0.1113
P67_R0 = 2.808
P67_M = 2.15
P67_N = 5.093
P67_K = 4.6142
P67_DT = 45.6889
P67_A1 = 1.0e-10
P67_A2 = 2.0e-11
P67_A3 = -3.0e-12


def _sbdb_dict(**overrides: Any) -> dict[str, Any]:
    """A single-orbit flat dict matching ``orbit_batch_to_pydict``'s shape."""
    base: dict[str, Any] = {
        "orbit_ids": ["67P"],
        "object_ids": ["67P"],
        "representation": ["cartesian"],
        "epoch_mjd_tdb": np.array([59000.0]),
        "elements": np.array([[3.5, 0.0, 0.0, 0.0, 0.006, 0.0]]),
        "has_covariance": np.array([0], dtype=np.uint8),
        "covariance": np.zeros((1, 6, 6)),
        "frame": ["icrf"],
        "origin": np.array([10], dtype=np.int32),  # Sun
        "a1": np.array([P67_A1]),
        "a2": np.array([P67_A2]),
        "a3": np.array([P67_A3]),
        "ng_alpha": np.array([P67_ALPHA]),
        "ng_r0": np.array([P67_R0]),
        "ng_m": np.array([P67_M]),
        "ng_n": np.array([P67_N]),
        "ng_k": np.array([P67_K]),
        "non_grav_dt": np.array([P67_DT]),
        "phot_system": [None],
    }
    base.update(overrides)
    return base


def test_sbdb_marsden_gr_survives_round_trip() -> None:
    """Custom SBDB g(r) exponents + dt must reach the Orbits table intact."""
    orbits = orbit_batch_dict_to_orbits(_sbdb_dict())
    ng = orbits.non_grav
    assert ng is not None, "non_grav dropped entirely — comet would propagate gravity-only"
    assert ng.model[0].as_py() == "marsden", (
        "custom SBDB g(r) must map to model='marsden', not inverse_square"
    )
    # Absolute coefficients.
    np.testing.assert_allclose(ng.a1.to_numpy(zero_copy_only=False), [P67_A1], rtol=0, atol=0)
    np.testing.assert_allclose(ng.a2.to_numpy(zero_copy_only=False), [P67_A2], rtol=0, atol=0)
    np.testing.assert_allclose(ng.a3.to_numpy(zero_copy_only=False), [P67_A3], rtol=0, atol=0)
    # g(r) exponents — the bit that silently vanished and caused the 67P drift.
    np.testing.assert_allclose(ng.alpha.to_numpy(zero_copy_only=False), [P67_ALPHA])
    np.testing.assert_allclose(ng.r0.to_numpy(zero_copy_only=False), [P67_R0])
    np.testing.assert_allclose(ng.m.to_numpy(zero_copy_only=False), [P67_M])
    np.testing.assert_allclose(ng.n.to_numpy(zero_copy_only=False), [P67_N])
    np.testing.assert_allclose(ng.k.to_numpy(zero_copy_only=False), [P67_K])
    # Thermal-lag delay.
    np.testing.assert_allclose(ng.dt.to_numpy(zero_copy_only=False), [P67_DT])


def test_sbdb_inverse_square_sentinel_keeps_yarkovsky() -> None:
    """alpha==0 is the inverse-square sentinel; A2 (Yarkovsky) must survive."""
    d = _sbdb_dict(
        ng_alpha=np.array([0.0]),
        ng_r0=np.array([0.0]),
        ng_m=np.array([0.0]),
        ng_n=np.array([0.0]),
        ng_k=np.array([0.0]),
        non_grav_dt=np.array([np.nan]),
        a1=np.array([0.0]),
        a3=np.array([0.0]),
    )
    orbits = orbit_batch_dict_to_orbits(d)
    ng = orbits.non_grav
    assert ng is not None, "non_grav dropped — Yarkovsky A2 would vanish on re-propagation"
    assert ng.model[0].as_py() == "inverse_square"
    np.testing.assert_allclose(ng.a2.to_numpy(zero_copy_only=False), [P67_A2])


def test_no_nongrav_when_all_coefficients_zero() -> None:
    """A gravity-only orbit must NOT carry a spurious non_grav model.

    ``non_grav`` is a ``nullable`` sub-table column, so quivr fills it with a
    null-valued row rather than leaving the attribute ``None``. Either is an
    acceptable "no non-grav" signal; what must NOT happen is a real
    coefficient or a populated model leaking in.
    """
    d = _sbdb_dict(
        a1=np.array([0.0]),
        a2=np.array([0.0]),
        a3=np.array([0.0]),
        ng_alpha=np.array([0.0]),
        ng_r0=np.array([0.0]),
        ng_m=np.array([0.0]),
        ng_n=np.array([0.0]),
        ng_k=np.array([0.0]),
        non_grav_dt=np.array([np.nan]),
    )
    orbits = orbit_batch_dict_to_orbits(d)
    ng = orbits.non_grav
    if ng is not None and len(ng) > 0:
        a1 = np.nan_to_num(ng.a1.to_numpy(zero_copy_only=False), nan=0.0)
        a2 = np.nan_to_num(ng.a2.to_numpy(zero_copy_only=False), nan=0.0)
        a3 = np.nan_to_num(ng.a3.to_numpy(zero_copy_only=False), nan=0.0)
        assert not (a1.any() or a2.any() or a3.any()), "spurious non-grav coefficients"
