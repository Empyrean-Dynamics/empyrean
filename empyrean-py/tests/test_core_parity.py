"""Cross-channel parity: empyrean-core (oracle) vs the empyrean-py channel.

Companion to :mod:`test_no_silent_drops`. That test walks the channel's
output columns for *nulls* — it is structurally blind to a dropped field
the channel backfills with a non-null **sentinel** (``0.0``). This test
closes that gap by comparing the channel's values, field by field, against
**empyrean-core** — the only layer that still carries every field (it reads
villeneuve directly, upstream of the C-ABI chokepoint). Any field core
populates that the channel drops, NaNs, or hardcodes to ``0.0`` is a C-ABI
contract violation, caught here as a value mismatch with no
"expected-populated" list to maintain.

The two channels can't run in one process (empyrean-sys links a prebuilt
``libempyrean`` dylib with its own statically-linked engine, while
empyrean-core compiles a second copy — a dual-engine init hazard), so the
oracle is a **committed fixture**: ``tests/fixtures/core_parity_oracle.json``,
a flat ``"<scenario>/<op>/<table>/<index>/<field>" -> scalar`` projection
emitted by ``empyrean-core``'s ``parity-oracle`` binary over the same
hardcoded fixtures this test runs through empyrean-py. Regenerate it with::

    cd empyrean-core && cargo run --features validate --bin parity-oracle -- \\
      --output ../empyrean/empyrean-py/tests/fixtures/core_parity_oracle.json

Design (ADR cross-channel-parity, bd empyrean-na4h.2 / na4h.4):

- Both sides start from **byte-identical** Cartesian states + covariance and
  run identical physics (ForceModelTier.Standard, all event detectors on),
  so any value divergence is a marshaling fault, not a physics difference.
- Events are bucketed by sub-table and sorted by epoch so the per-table
  index aligns across channels; epoch is an **alignment check** (a large Δ
  means the event streams diverged, which fails hard rather than comparing
  mismatched pairs).
- Known C-ABI drops go in :data:`KNOWN_DROPS` keyed ``(table, field)`` with a
  tracking issue. The list is **reverse-enforced**: if every occurrence of a
  listed field starts matching core (the drop was fixed), the entry is stale
  and the test fails to force its removal.
"""

from __future__ import annotations

import json
from pathlib import Path

import empyrean
import numpy as np
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    NonGravParams,
    Origin,
    UncertaintyMethod,
    compute_b_planes,
    compute_impact_probabilities,
    generate_ephemeris,
)
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.observers.observers import Observers

_DATA = Path(__file__).parent / "fixtures"
ORACLE_PATH = _DATA / "core_parity_oracle.json"
MANIFEST_PATH = _DATA / "parity_manifest.json"

# ── Shared manifest — single source of truth across all channels ─────
#
# Scenarios (fixture inputs), known_drops (allow-list), and tolerances all
# live in parity_manifest.json and are consumed identically by the core
# oracle (parity-oracle.rs), this test, the wrapper test, and the cli test.
# Add a scenario / drop ONCE there and every channel picks it up.

_MANIFEST = json.loads(MANIFEST_PATH.read_text())

SCENARIOS: dict[str, dict] = {}
for _s in _MANIFEST["scenarios"]:
    _g = _s["grid"]
    SCENARIOS[_s["name"]] = dict(
        epoch=_s["epoch_mjd_tdb"],
        state=_s["state"],
        cov=_s["cov_diag"],
        grid=[_g["start"] + _g["step"] * i for i in range(_g["n"])],
        ip_end=_s["ip_end_mjd"],
        eph_obs_code=_s.get("ephemeris_obs_code"),
        eph_epochs=_s.get("ephemeris_epochs", []),
    )

# Per event sub-table, the value columns to compare (epoch handled
# separately, as an alignment check). Field names match the parity-oracle
# emission exactly.
TABLE_FIELDS: dict[str, list[str]] = {
    "close_approach_starts": ["distance_au", "distance_km"],
    "close_approach_ends": ["distance_au", "distance_km"],
    "periapses": [
        "distance_au",
        "distance_km",
        "relative_velocity_au_day",
        "relative_x",
        "relative_y",
        "relative_z",
        "relative_vx",
        "relative_vy",
        "relative_vz",
    ],
    "impacts": ["latitude_deg", "longitude_deg", "altitude_km"],
    "atmospheric_entries": ["distance_au"],
    "atmospheric_exits": ["distance_au"],
    "shadow_entries": ["shadow_fraction", "illumination"],
    "shadow_exits": ["shadow_fraction", "illumination"],
    "possible_impacts": [
        "miss_distance_au",
        "miss_distance_km",
        "effective_radius_au",
        "effective_radius_km",
        "sigma_distance_au",
        "ip_linear",
        "relative_velocity_au_day",
        "ip_second_order",
        "nonlinearity",
        "ip_agm",
        "ip_mc",
    ],
}

# Standalone-op field lists (op -> comparable columns). These ops are fully
# wired through the C ABI — positive controls where the channel must match
# core exactly. The standalone compute_impact_probabilities path carrying
# effective_radius / sigma_distance is the direct contrast to the events
# path zeroing them.
IP_FIELDS = [
    "miss_distance_au",
    "miss_distance_km",
    "effective_radius_au",
    "effective_radius_km",
    "sigma_distance_au",
    "sigma_distance_km",
    "ip_linear",
    "relative_velocity_au_day",
    "ip_second_order",
    "nonlinearity",
    "ip_agm",
    "ip_mc",
]
BPLANE_FIELDS = [
    "b_dot_t_km",
    "b_dot_r_km",
    "b_mag_km",
    "v_inf_km_s",
    "effective_radius_km",
    "body_radius_km",
    "cov_tt_km2",
    "cov_tr_km2",
    "cov_rr_km2",
    "semi_major_3sig_km",
    "semi_minor_3sig_km",
    "ellipse_angle_rad",
    "ip_linear",
]
# Ephemeris sky-position scalars. The aberrated-state / sky-covariance drops
# (empyrean-j2ue / -mgn9) are all-null and owned by test_no_silent_drops, so
# they are intentionally not projected here.
EPH_COORD_FIELDS = ["lon", "lat", "rho", "vrho", "vlon", "vlat"]
EPH_TOP_FIELDS = [
    "light_time",
    "phase_angle",
    "elongation",
    "heliocentric_distance",
    "mag",
    "mag_sigma",
    "zenith_angle",
    "azimuth",
    "hour_angle",
    "lunar_elongation",
    "position_angle",
    "sky_rate",
]

# Fields the C ABI drops, by (table, field) -> tracking issue. Each is a
# value core populates but the channel hardcodes to a sentinel
# (np.nan / 0.0). Remove an entry when its issue closes — the test
# reverse-enforces (fails if every occurrence starts matching core).
# Known C-ABI drops (table, field) -> issue, from the shared manifest.
# Reverse-enforced below: if every occurrence of a listed field starts
# matching core (the drop was fixed), the entry is stale and the test fails
# to force its removal from the manifest.
KNOWN_DROPS: dict[tuple[str, str], str] = {
    (d["table"], d["field"]): d["issue"] for d in _MANIFEST["known_drops"]
}

# Tolerances from the manifest. The default rtol is tight; a scenario may
# override it (the chaotic impactor needs a loose rtol — the oracle and the
# channel dylib are separately compiled, so ULP-level codegen differences
# amplify through its near-surface encounter).
RTOL = _MANIFEST["tolerances"]["rtol"]
ATOL = _MANIFEST["tolerances"]["atol"]
SCENARIO_RTOL = {s["name"]: s.get("rtol", RTOL) for s in _MANIFEST["scenarios"]}
# Epoch alignment: events paired by index must be the same event. A Δ
# beyond this means the streams diverged (pairing is meaningless).
EPOCH_TOL_DAYS = _MANIFEST["tolerances"]["epoch_tol_days"]


def _build_orbit(s: dict) -> CartesianOrbits:
    cov = CartesianCovariance.from_matrix(np.diag(s["cov"])[None, :, :])
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["state"][0]],
        y=[s["state"][1]],
        z=[s["state"][2]],
        vx=[s["state"][3]],
        vy=[s["state"][4]],
        vz=[s["state"][5]],
        covariance=cov,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["parity"],
        object_id=["parity"],
        coordinates=coords,
        non_grav=NonGravParams.from_kwargs(a1=[0.0], a2=[0.0], a3=[0.0], model=["inverse_square"]),
    )


def _scalar(x) -> float | None:
    """A single arrow/numpy cell -> finite float or None (NaN -> None, so
    the channel's NaN sentinels compare equal to core's null)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _emit_rows(
    fp: dict[str, float | None],
    name: str,
    op: str,
    table: str,
    epochs: np.ndarray,
    cols: dict[str, np.ndarray],
) -> None:
    """Sort rows by epoch (so the per-table index aligns with the oracle)
    and flatten to fingerprint entries."""
    order = np.argsort(epochs, kind="stable")
    for idx, row in enumerate(order):
        base = f"{name}/{op}/{table}/{idx}"
        fp[f"{base}/epoch_mjd_tdb"] = _scalar(epochs[row])
        for f, arr in cols.items():
            fp[f"{base}/{f}"] = _scalar(arr[row])


def _channel_fingerprint() -> dict[str, float | None]:
    """Project the empyrean-py channel into the oracle's flat vocabulary."""
    fp: dict[str, float | None] = {}
    for name, s in SCENARIOS.items():
        orbit = _build_orbit(s)

        # propagate -> events
        events = empyrean.propagate(orbit, np.array(s["grid"])).events
        for table, fields in TABLE_FIELDS.items():
            t = getattr(events, table)
            if len(t) == 0:
                continue
            _emit_rows(
                fp,
                name,
                "propagate",
                table,
                np.asarray(t.epoch, dtype=float),
                {f: np.asarray(getattr(t, f)) for f in fields},
            )

        # standalone compute_impact_probabilities (fully wired)
        ips = compute_impact_probabilities(
            orbit,
            end_epoch=s["ip_end"],
            methods=[UncertaintyMethod.FIRST_ORDER],
            body_filter=[Origin.EARTH],
        )
        if len(ips) > 0:
            _emit_rows(
                fp,
                name,
                "compute_impact_probabilities",
                "impact_probabilities",
                np.asarray(ips.epochs.mjd, dtype=float),
                {f: np.asarray(getattr(ips, f)) for f in IP_FIELDS},
            )

        # standalone compute_b_planes (fully wired)
        bps = compute_b_planes(
            orbit,
            end_epoch=s["ip_end"],
            methods=[UncertaintyMethod.FIRST_ORDER],
            body_filter=[Origin.EARTH],
        )
        if len(bps) > 0:
            _emit_rows(
                fp,
                name,
                "compute_b_planes",
                "b_planes",
                np.asarray(bps.epochs.mjd, dtype=float),
                {f: np.asarray(getattr(bps, f)) for f in BPLANE_FIELDS},
            )

        # generate_ephemeris sky-position scalars (per-manifest obs code/epochs)
        if s["eph_obs_code"] and s["eph_epochs"]:
            observers = Observers.from_code(s["eph_obs_code"], s["eph_epochs"])
            eph = generate_ephemeris(orbit, observers).ephemeris
            if len(eph) > 0:
                cols = {f: np.asarray(getattr(eph.coordinates, f)) for f in EPH_COORD_FIELDS}
                cols.update({f: np.asarray(getattr(eph, f)) for f in EPH_TOP_FIELDS})
                _emit_rows(
                    fp,
                    name,
                    "generate_ephemeris",
                    "ephemeris",
                    np.asarray(eph.coordinates.epoch, dtype=float),
                    cols,
                )
    return fp


def _close(a: float | None, b: float | None, rtol: float = RTOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= ATOL + rtol * max(1.0, abs(a))


def test_core_parity_no_silent_value_drops() -> None:
    assert ORACLE_PATH.exists(), (
        f"Missing oracle fixture {ORACLE_PATH}. Regenerate with:\n"
        "  cd empyrean-core && cargo run --features validate --bin parity-oracle -- "
        f"--output {ORACLE_PATH}"
    )
    oracle: dict[str, float | None] = json.loads(ORACLE_PATH.read_text())
    channel = _channel_fingerprint()

    def parse(key: str) -> tuple[str, str]:
        # "<scenario>/propagate/<table>/<index>/<field>"
        parts = key.split("/")
        return parts[2], parts[-1]  # (table, field)

    new_violations: list[str] = []
    misalignments: list[str] = []
    # (table, field) -> did EVERY occurrence match core? (for reverse-enforce)
    drop_all_match: dict[tuple[str, str], bool] = {}

    all_keys = set(oracle) | set(channel)
    for key in sorted(all_keys):
        table, field = parse(key)
        core = oracle.get(key)
        chan = channel.get(key, "__missing__")

        if field == "epoch_mjd_tdb":
            # Pure alignment guard — never a drop, never allow-listed.
            if chan == "__missing__" or core is None or chan is None:
                misalignments.append(f"{key}: core={core} chan={chan} (event stream divergence)")
            elif abs(core - chan) > EPOCH_TOL_DAYS:
                misalignments.append(f"{key}: |Δepoch|={abs(core - chan):.3e} d > {EPOCH_TOL_DAYS}")
            continue

        if chan == "__missing__":
            # Core emitted a field the channel has no column for.
            if (table, field) in KNOWN_DROPS:
                drop_all_match[(table, field)] = False
            else:
                new_violations.append(f"{key}: core={core} but channel has no such field")
            continue
        if key not in oracle:
            new_violations.append(f"{key}: channel emitted a field core does not")
            continue

        matched = _close(core, chan, SCENARIO_RTOL.get(key.split("/")[0], RTOL))
        if (table, field) in KNOWN_DROPS:
            # Track whether the drop is fully fixed (all occurrences match).
            prev = drop_all_match.get((table, field), True)
            drop_all_match[(table, field)] = prev and matched
        elif not matched:
            new_violations.append(f"{key}: core={core!r} channel={chan!r}")

    # Reverse-enforce: a known drop whose every occurrence now matches core
    # has been fixed — force the stale entry out.
    stale = [
        f"{tf} (was {KNOWN_DROPS[tf]})"
        for tf, all_match in drop_all_match.items()
        if all_match and tf in KNOWN_DROPS
    ]

    msgs: list[str] = []
    if misalignments:
        msgs.append(
            "Event-stream MISALIGNMENT (core vs channel paired different events — "
            "the rest of the diff is meaningless):\n  " + "\n  ".join(misalignments)
        )
    if new_violations:
        msgs.append(
            "NEW core-vs-channel value drops/divergences (a field core populates "
            "that the channel drops, NaNs, 0.0s, or computes differently — wire it "
            "through the C ABI or, if a by-design drop, add to KNOWN_DROPS with an "
            "issue):\n  " + "\n  ".join(new_violations)
        )
    if stale:
        msgs.append(
            "STALE KNOWN_DROPS (every occurrence now matches core — the drop was "
            "fixed; remove the entry):\n  " + "\n  ".join(stale)
        )
    assert not msgs, "\n\n".join(msgs)
