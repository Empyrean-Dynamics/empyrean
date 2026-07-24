"""Forcing-function: every output column must be populated, or be on a
tracking allow-list.

The empyrean stack has a recurring failure mode where empyrean_core
computes a field, but the C ABI (`EmpyreanOrbit`, `EmpyreanEvent`,
`EmpyreanBPlane`, `EmpyreanEphemerisEntry`, …) doesn't carry it, and
the empyrean-py schema declares the column anyway. The result is
columns that are silently always-NaN / always-empty across every row,
discovered one by one when users hit them.

This test exercises every major output channel (propagation states,
events, ephemeris, impact probabilities, B-planes) with a realistic
input that exercises the full feature surface — covariance, photometry,
non-grav, multi-year window with Earth close approaches — and asserts
that **every nullable column has at least one non-null value**.

When a column legitimately can be all-null in the test inputs (no
covariance for an OD-only field, no observer for a heliocentric-only
field, etc.) it goes in :data:`ALLOWED_ALL_NULL` with a reason. Each
"known drop" entry should be removed when the drop is fixed — and
the test enforces this in reverse: if an
allow-listed column has *any* non-null values, the test fails to
force the entry out.

Adding a new column without wiring it through the C ABI will fail this
test loudly. That's the point.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pyarrow as pa
import pytest
import quivr as qv
from empyrean import (
    CartesianCoordinates,
    CartesianOrbits,
    CometaryCoordinates,
    KeplerianCoordinates,
    NonGravParams,
    Origin,
    PhotometricParams,
    UncertaintyMethod,
    compute_b_planes,
    compute_impact_probabilities,
    generate_ephemeris,
    transform_coordinates,
)
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.coordinates.enums import Frame
from empyrean.observers.observers import Observers
from empyrean.propagation.config import (
    AdvancedIntegratorConfig,
    ForceModelTier,
    OriginSwitchingConfig,
    PropagationConfig,
)
from empyrean.propagation.events import EventConfig

# ── Allow-list ───────────────────────────────────────────────────────
#
# Keys are "<TableClassName>.<flat_column_path>" — the dotted path matches
# what `pyarrow.Table.flatten()` produces for a quivr table. Each entry
# pairs the column with a reason.
#
# Categories:
#   - "by-design": column legitimately can be null in this test setup
#     because the feature isn't requested or doesn't apply (e.g. impact
#     fields on non-impact close approaches).
#   - "known drop": column is silently dropped; fixing the drop should
#     re-populate it. When fixed, REMOVE the entry — the test will
#     enforce this by failing if the column starts having values.

ALLOWED_ALL_NULL: dict[str, str] = {
    # ── PropagatedStates: schema artifact ──
    # `result.states` is typed as `CartesianOrbits`, so it inherits the
    # input non_grav and photometric sub-tables that don't apply to a
    # propagation OUTPUT. Either change the schema (separate PropagatedStates
    # type) or accept these as schema-leftover.
    "PropagatedStates.non_grav": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.a1": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.a2": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.a3": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.model": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.alpha": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.r0": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.m": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.n": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.k": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.dt": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.dt_variance": "schema-artifact (input fields on output table)",
    "PropagatedStates.non_grav.covariance": "schema-artifact (input fields on output table)",
    # SRP is a first-class, additive input force slot (orbits.srp); like the
    # non_grav sub-table above it rides on the CartesianOrbits schema and is
    # not repopulated on a propagation OUTPUT.
    "PropagatedStates.srp": "schema-artifact (input fields on output table)",
    "PropagatedStates.srp.amrat": "schema-artifact (input fields on output table)",
    "PropagatedStates.srp.cr": "schema-artifact (input fields on output table)",
    "PropagatedStates.srp.amrat_variance": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.model": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.h": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.g": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.g1": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.g2": "schema-artifact (input fields on output table)",
    "PropagatedStates.photometric.g12": "schema-artifact (input fields on output table)",
    # ── Ephemeris aberrated state + both covariances: populated as of
    # v0.9.0 — the C ABI carries the sky-plane covariance,
    # the aberrated Cartesian state, and the aberrated covariance, so none
    # of them are allow-listed: this test re-fails if any regress to
    # all-null. Same for the six local-horizon / sky-motion angles
    # (zenith_angle, azimuth, hour_angle, lunar_elongation,
    # position_angle, sky_rate). ──
    # ── BPlanes ip_linear: known drop (still pending) ──
    # cov / ellipse fields below are now populated upstream (test
    # caught the stale entry) and have been removed; ip_linear stays
    # on the allow-list until the upstream fix lands.
    "BPlanes.ip_linear": "known drop (pending upstream fix)",
    # ── ObservationSensitivities: now wired through the C ABI
    # — orbit_id key + Jacobian populated (deliberately
    # NOT listed). The Hessian is Jet2-only (null for the first-order
    # fixture); object_id is null because villeneuve's sensitivity chain is
    # keyed by (orbit_id, obs_code) and doesn't carry the optional object_id
    # the way the Ephemeris table does — a villeneuve-level metadata gap, not
    # a distribution drop. ──
    "ObservationSensitivities.hessian": "by-design (Jet2 method only)",
    "ObservationSensitivities.object_id": "villeneuve chain not keyed by object_id",
    # ── Impact probabilities: by-design when method != MonteCarlo ──
    "ImpactProbabilities.mc_n_samples": "by-design (MC method only)",
    "ImpactProbabilities.mc_n_impacts": "by-design (MC method only)",
    "ImpactProbabilities.ip_mc": "by-design (MC method only)",
    "ImpactProbabilities.ip_second_order": "by-design (Jet2 method only)",
    "ImpactProbabilities.nonlinearity": "by-design (Jet2 method only)",
    "ImpactProbabilities.ip_agm": "by-design (AGM method only)",
    "ImpactProbabilities.mc_confidence_interval": "by-design (MC method only)",
    "ImpactProbabilities.mean_distance_second_order_au": "by-design (Jet2 or MC method only)",
    "ImpactProbabilities.sigma_distance_second_order_au": "by-design (Jet2 method only)",
    "ImpactProbabilities.skewness": "by-design (Jet2 method only)",
    "ImpactProbabilities.distance_hessian": "by-design (Jet2 method only)",
    "ImpactProbabilities.agm_components": "by-design (AGM refinement only)",
    # (impact_latitude_deg / impact_longitude_deg / impact_altitude_km are
    # populated by the enrichment pass on the fixture — deliberately NOT
    # allow-listed, so this test re-fails if they regress to all-null.)
    # (Periapses.relative_{x,y,z,vx,vy,vz} were once dropped here
    # but are now wired through the C ABI and populated — removed, so this
    # test re-fails if they ever regress to all-null.)
    # ── PossibleImpacts second-order / AGM / Monte-Carlo probabilities are
    # method-dependent: NaN/null for the first-order Apophis fixture (the
    # engine only computes them under Jet2 / AGM / MC). Now carried through
    # the C ABI — null here is by-design, not a drop. The
    # 0.0-sentinel fields (effective_radius/sigma/ip_linear) the parity test
    # caught are wired too and deliberately NOT listed. ──
    "PossibleImpacts.ip_second_order": "by-design (Jet2 method only)",
    "PossibleImpacts.nonlinearity": "by-design (Jet2 method only)",
    "PossibleImpacts.ip_agm": "by-design (AGM method only)",
    "PossibleImpacts.ip_mc": "by-design (MonteCarlo method only)",
}


# ── Inputs ───────────────────────────────────────────────────────────


# Apophis state from SBDB at MJD 61000 TDB. Fires Earth close approaches
# on multi-year propagation (well-known 2029 flyby), with covariance and
# photometry attached so every output channel has the inputs it needs.
APOPHIS_STATE = {
    "epoch": 61000.0,
    "x": -7.85264914906904643e-02,
    "y": -8.19748051902064567e-01,
    "z": 4.18939515323390882e-02,
    "vx": 1.98751024968884596e-02,
    "vy": 1.32208844536140196e-03,
    "vz": 3.99496044422352188e-04,
}


def _full_feature_orbit() -> CartesianOrbits:
    """Apophis state with covariance, photometry, and non-grav attached.

    This is the input that every output channel should be able to fully
    populate — anything that's still null on a result row points at a
    real wiring gap.
    """
    s = APOPHIS_STATE
    cov_matrix = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
    covariance = CartesianCovariance.from_matrix(cov_matrix)
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        covariance=covariance,
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    # Fitted non-grav 3x3 (A1, A2, A3) covariance, row-major flattened, so the
    # full-feature input exercises the ng_covariance marshal channel too (the
    # OD-output field the forward-model paths historically dropped). The A
    # coefficients are zero here, so this covariance is inert on the output
    # (it only inflates the propagated covariance when the non-grav force term
    # is active) — the load-bearing DIFFERENCE check lives in
    # test_ng_covariance_reaches_propagated_covariance with an active fixture.
    non_grav = NonGravParams.from_kwargs(
        a1=[0.0],
        a2=[0.0],
        a3=[0.0],
        model=["inverse_square"],
        covariance=[np.diag([1e-20, 1e-20, 1e-20]).reshape(9).tolist()],
    )
    photometric = PhotometricParams.from_kwargs(
        model=["hg"],
        h=[19.7],
        g=[0.15],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["FORCING_TEST"],
        object_id=["forcing_test_obj"],
        coordinates=coords,
        non_grav=non_grav,
        photometric=photometric,
    )


# 2008 TC3 — the first asteroid ever detected before it hit Earth
# (2008-Oct-07). JPL SBDB cometary elements at MJD 54746.0 TDB, the
# covariance epoch (lifted from villeneuve/tests/test_jet2.rs so the
# fixture stays hermetic — no SBDB/Horizons fetch). Angles in degrees;
# tp is MJD TDB (JD 2454790.8988481034 − 2400000.5).
TC3_COMETARY = {
    "epoch": 54746.0,
    "q": 0.8999568608039946,
    "e": 0.3120674404369712,
    "i": 2.542215283712214,
    "raan": 194.1011435928938,
    "ap": 234.44892519,
    "tp": 2_454_790.898_848_103_4 - 2_400_000.5,
}


def _impactor_orbit() -> CartesianOrbits:
    """2008 TC3 as a Cartesian orbit on its final approach to Earth.

    The Apophis fixture fires close approaches + periapses but never an
    impact, atmospheric entry, or shadow entry — a grazing flyby reaches
    none of those. An impactor does: propagated across its last day it
    enters the atmosphere and Earth's shadow and strikes the ground (so it
    fires the ``*_entry`` families but never the ``*_exit`` ones — it
    doesn't come back out). That makes this the fixture that catches silent
    drops on ``impacts`` / ``atmospheric_entries`` / ``shadow_entries``.

    Built from cometary elements via :func:`transform_coordinates`, with a
    small diagonal covariance attached so covariance-bearing event fields
    are exercised too.
    """
    c = TC3_COMETARY
    com = CometaryCoordinates.from_kwargs(
        epoch=[c["epoch"]],
        q=[c["q"]],
        e=[c["e"]],
        i=[c["i"]],
        raan=[c["raan"]],
        ap=[c["ap"]],
        tp=[c["tp"]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    cart = transform_coordinates(com, CartesianCoordinates)
    cov_matrix = np.diag([1e-12, 1e-12, 1e-12, 1e-16, 1e-16, 1e-16])[None, :, :]
    cart = cart.set_column("covariance", CartesianCovariance.from_matrix(cov_matrix))
    non_grav = NonGravParams.from_kwargs(
        a1=[0.0],
        a2=[0.0],
        a3=[0.0],
        model=["inverse_square"],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["IMPACTOR_TEST"],
        object_id=["2008 TC3"],
        coordinates=cart,
        non_grav=non_grav,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _walk_columns(table: qv.Table, prefix: str) -> list[tuple[str, int, int]]:
    """Flatten a quivr table to leaf columns and return
    ``(path, effective_null_count, total)`` for each. Path is dotted
    from the root, prefixed with the class name (e.g.
    ``"Ephemeris.coordinates.lon"``).

    "Effective null" counts pyarrow nulls AND float NaN sentinels — the
    Rust binding sometimes emits NaN where it should emit a null, so
    pure ``null_count`` would miss those. Recursively flattens struct
    columns to leaf level so nested quivr sub-tables are checked too.
    """
    import pandas as pd

    arrow_table = table.table
    # Recursively flatten until no struct columns remain.
    while any(pa.types.is_struct(f.type) for f in arrow_table.schema):
        arrow_table = arrow_table.flatten()

    rows: list[tuple[str, int, int]] = []
    for name in arrow_table.schema.names:
        col = arrow_table.column(name)
        path = f"{prefix}.{name}"
        total = len(col)
        # `to_pandas()` collapses both arrow nulls and float NaN into
        # `pd.NA` / `NaN`, both of which `pd.isna` flags. That's the
        # right "effectively missing" definition for this test.
        try:
            arr = col.to_pandas()
            null_count = int(pd.isna(arr).sum())
        except Exception:  # noqa: BLE001 — fall back to arrow null_count
            null_count = col.null_count
        rows.append((path, null_count, total))
    return rows


def _check_no_silent_drops(table: qv.Table, class_name: str) -> tuple[list[str], list[str]]:
    """Return two lists for one output table:

    - ``unexpected_all_null`` — column paths that are 100% null and NOT
      in :data:`ALLOWED_ALL_NULL`. These are the bugs.
    - ``unexpected_not_null`` — column paths that ARE in the allow-list
      but have at least one non-null value. The allow-list should be
      pruned (the underlying issue was fixed).
    """
    unexpected_all_null: list[str] = []
    unexpected_not_null: list[str] = []
    for path, null_count, total in _walk_columns(table, class_name):
        if total == 0:
            continue
        all_null = null_count == total
        in_allow = path in ALLOWED_ALL_NULL
        if all_null and not in_allow:
            unexpected_all_null.append(path)
        elif not all_null and in_allow:
            unexpected_not_null.append(path)
    return unexpected_all_null, unexpected_not_null


def _format_failures(
    class_name: str,
    unexpected_all_null: list[str],
    unexpected_not_null: list[str],
) -> str:
    parts: list[str] = []
    if unexpected_all_null:
        parts.append(
            f"\n{class_name}: columns are 100% null but not on the allow-list "
            f"(probably silently dropped at the C ABI):"
        )
        for c in unexpected_all_null:
            parts.append(f"  - {c}")
        parts.append(
            "  Fix: wire the field through the C ABI, OR add to "
            "ALLOWED_ALL_NULL with a reason if legitimately N/A."
        )
    if unexpected_not_null:
        parts.append(
            f"\n{class_name}: columns are on the allow-list but DO have "
            f"non-null values — the underlying issue was fixed:"
        )
        for c in unexpected_not_null:
            reason = ALLOWED_ALL_NULL.get(c, "?")
            parts.append(f"  - {c}  (allow-list reason: {reason!r})")
        parts.append(
            "  Fix: remove these entries from ALLOWED_ALL_NULL — they're "
            "now populated, so the entry is stale."
        )
    return "\n".join(parts)


# ── Tests ────────────────────────────────────────────────────────────


def test_propagation_states_no_silent_drops() -> None:
    orbits = _full_feature_orbit()
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(orbits, target_epochs)

    bad_null, bad_not_null = _check_no_silent_drops(result.states, "PropagatedStates")
    assert not (bad_null or bad_not_null), _format_failures(
        "PropagatedStates", bad_null, bad_not_null
    )


def test_propagation_events_summary_no_silent_drops() -> None:
    orbits = _full_feature_orbit()
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(orbits, target_epochs)

    if len(result.events.summary) == 0:
        pytest.skip("No events in test propagation; channel not exercised.")

    bad_null, bad_not_null = _check_no_silent_drops(result.events.summary, "EventSummary")
    assert not (bad_null or bad_not_null), _format_failures("EventSummary", bad_null, bad_not_null)


# (accessor, class name) for every typed event sub-table. The
# summary-only check above is blind to per-table drops (e.g. the
# historical periapsis relative_* drop) — this walks each one.
_EVENT_SUBTABLES: list[tuple[str, str]] = [
    ("close_approach_starts", "CloseApproachStarts"),
    ("close_approach_ends", "CloseApproachEnds"),
    ("periapses", "Periapses"),
    ("impacts", "Impacts"),
    ("possible_impacts", "PossibleImpacts"),
    ("atmospheric_entries", "AtmosphericEntries"),
    ("atmospheric_exits", "AtmosphericExits"),
    ("capture_starts", "CaptureStarts"),
    ("capture_ends", "CaptureEnds"),
    ("shadow_entries", "ShadowEntries"),
    ("shadow_exits", "ShadowExits"),
    ("covariance_regime_changes", "CovarianceRegimeChanges"),
]


def test_event_subtables_coverage_is_complete() -> None:
    """Forcing function: every sub-table declared on the ``Events`` type must
    be in :data:`_EVENT_SUBTABLES`, so the silent-drop walk can't miss one.

    The walk lists are hand-maintained — the failure mode this guards is a
    new event family being added upstream (a new field on ``Events``) without
    anyone extending the walk, leaving the new sub-table silently unchecked.
    Reflecting the live type turns "remember to add it" into a red test.
    Static (reads ``Events.__annotations__`` — no propagation/network).
    """
    from empyrean.propagation import events as _evmod

    def class_name(t: object) -> str:
        # `from __future__ import annotations` makes these strings; be robust.
        return t if isinstance(t, str) else getattr(t, "__name__", str(t))

    # All typed table fields on Events, minus the scalar `summary` rollup.
    declared = {name: class_name(t) for name, t in _evmod.Events.__annotations__.items()}
    declared.pop("summary", None)

    walked = {attr: cls for attr, cls in _EVENT_SUBTABLES}

    missing = set(declared) - set(walked)
    extra = set(walked) - set(declared)
    renamed = [
        (a, walked[a], declared[a]) for a in set(walked) & set(declared) if walked[a] != declared[a]
    ]

    problems = []
    if missing:
        problems.append(
            f"Events sub-table(s) NOT walked by test_no_silent_drops: {sorted(missing)} "
            "— add them to _EVENT_SUBTABLES so they're checked for silent drops."
        )
    if extra:
        problems.append(
            f"_EVENT_SUBTABLES lists table(s) not on the Events type: {sorted(extra)} "
            "— remove the stale entries (renamed/removed upstream)."
        )
    if renamed:
        problems.append(f"_EVENT_SUBTABLES class-name mismatch vs Events: {renamed}")
    assert not problems, "\n".join(problems)


def test_propagation_event_subtables_no_silent_drops() -> None:
    """Walk every typed event sub-table, not just ``events.summary``.

    An empty sub-table just means this fixture didn't fire that event
    family (impacts / shadows / captures have their own fixtures below),
    so it's skipped rather than failed. The Apophis
    fixture fires close approaches + periapses, so this is what catches
    the historical periapsis ``relative_*`` drop.
    """
    orbits = _full_feature_orbit()
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(orbits, target_epochs)
    events = result.events

    all_bad_null: list[str] = []
    all_bad_not_null: list[str] = []
    fired: list[str] = []
    for attr, class_name in _EVENT_SUBTABLES:
        table = getattr(events, attr)
        if len(table) == 0:
            continue  # scenario didn't fire this event family
        fired.append(attr)
        bad_null, bad_not_null = _check_no_silent_drops(table, class_name)
        all_bad_null += bad_null
        all_bad_not_null += bad_not_null

    if not fired:
        pytest.skip("No event sub-tables populated by the fixture.")

    assert not (all_bad_null or all_bad_not_null), (
        f"(fired sub-tables: {', '.join(fired)})\n"
        + _format_failures("Event sub-tables", all_bad_null, all_bad_not_null)
    )


def test_impactor_event_subtables_no_silent_drops() -> None:
    """Walk the impact / atmospheric-entry / shadow-entry sub-tables.

    Apophis grazes Earth but never hits it, so it never populates the
    ``impacts`` / ``atmospheric_entries`` / ``shadow_entries`` families —
    they'd stay invisibly empty in the Apophis walk above. 2008 TC3
    actually strikes the ground, firing all three. We *assert* they fired
    (an empty result here is a fixture regression, not a silent skip)
    and then walk every fired sub-table for silent drops.
    """
    orbit = _impactor_orbit()
    # Last day before impact (~MJD 54746.116), fine steps so the encounter
    # is bracketed.
    target_epochs = np.array([54746.0 + 0.04 * i for i in range(26)])
    result = empyrean.propagate(orbit, target_epochs)
    events = result.events

    # These three MUST fire — the whole point of the impactor fixture. If
    # they don't, the fixture has regressed (kernel/elements/window drift)
    # and the families below would be silently un-exercised.
    assert len(events.impacts) > 0, "impactor fixture fired no impact event"
    assert len(events.atmospheric_entries) > 0, "impactor fixture fired no atmospheric entry"
    assert len(events.shadow_entries) > 0, "impactor fixture fired no shadow entry"

    all_bad_null: list[str] = []
    all_bad_not_null: list[str] = []
    fired: list[str] = []
    for attr, class_name in _EVENT_SUBTABLES:
        table = getattr(events, attr)
        if len(table) == 0:
            continue
        fired.append(attr)
        bad_null, bad_not_null = _check_no_silent_drops(table, class_name)
        all_bad_null += bad_null
        all_bad_not_null += bad_not_null

    assert not (all_bad_null or all_bad_not_null), (
        f"(fired sub-tables: {', '.join(fired)})\n"
        + _format_failures("Impactor event sub-tables", all_bad_null, all_bad_not_null)
    )


# ── Grazing flyby: atmospheric_exits + shadow_exits ──
#
# The 2008 TC3 impactor strikes the ground, so it only ever fires the
# ``*_entry`` event families — it never comes back out. A grazing
# hyperbolic Earth flyby that dips into the atmosphere and Earth's shadow
# and *exits* without striking the ground is the only thing that fires the
# ``atmospheric_exits`` / ``shadow_exits`` sub-tables. The Earth radius /
# GM literals below must track villeneuve's monitored-body registry
# (villeneuve/src/events/detectors/default_registry.rs).
_R_EQ_AU = 4.2635e-5  # Earth equatorial radius (AU) = 6378.137 km
_MU_EARTH = 8.887692446e-10  # Earth GM (AU^3 / day^2)
_GRAZER_EPOCH = 60200.0  # MJD TDB (matches the villeneuve CA-detector tests)


def _grazing_flyby_orbit() -> tuple[CartesianOrbits, np.ndarray]:
    """Grazing hyperbolic Earth flyby with perigee ~43 km above the surface.

    Aimed down the anti-solar axis so the close approach crosses Earth's
    shadow. Enters and EXITS both the atmosphere (band: surface .. surface +
    100 km) and the shadow without ever reaching the ground, so it fires the
    ``*_exit`` families the impactor never reaches. Built analytically as a
    geocentric hyperbolic conic offset from Earth's ephemeris state, so there
    is no Python-side numerical integration. Hermetic: only ``get_states``
    (de440 kernel) + ``propagate``, no network.
    """
    epoch = _GRAZER_EPOCH

    # Earth and Sun heliocentric (SSB-centered) ecliptic states at the epoch.
    earth = empyrean.get_states(Origin.EARTH, Origin.SSB, [epoch], Frame.ECLIPTICJ2000)
    e_r = np.array([earth.x[0].as_py(), earth.y[0].as_py(), earth.z[0].as_py()])
    e_v = np.array([earth.vx[0].as_py(), earth.vy[0].as_py(), earth.vz[0].as_py()])
    sun = empyrean.get_states(Origin.SUN, Origin.SSB, [epoch], Frame.ECLIPTICJ2000)
    s_r = np.array([sun.x[0].as_py(), sun.y[0].as_py(), sun.z[0].as_py()])
    s_v = np.array([sun.vx[0].as_py(), sun.vy[0].as_py(), sun.vz[0].as_py()])

    # Shadow axis: from Earth, directly away from the Sun.
    antisolar = (e_r - s_r) / np.linalg.norm(e_r - s_r)

    # Geocentric hyperbolic conic with perigee on the anti-solar axis.
    rp = 1.007 * _R_EQ_AU  # perigee radius (~43 km altitude; atmo band is 0-100 km)
    e_orb = 1.6  # hyperbolic eccentricity
    r_p_hat = antisolar
    tang = np.cross([0.0, 0.0, 1.0], r_p_hat)
    tang /= np.linalg.norm(tang)  # perigee velocity direction (in-plane)

    p = rp * (1.0 + e_orb)  # semi-latus rectum
    h = np.sqrt(_MU_EARTH * p)  # specific angular momentum
    a = rp / (e_orb - 1.0)  # |semi-major axis|

    # Inbound start at ~5 R_eq (above the atmosphere, inside the 0.01 AU
    # close-approach zone): solve the conic for the inbound true anomaly.
    r_start = 5.0 * _R_EQ_AU
    cos_nu = np.clip((p / r_start - 1.0) / e_orb, -1.0, 1.0)
    nu = -np.arccos(cos_nu)  # negative => inbound branch

    # Perifocal position/velocity, then rotate into the ecliptic frame.
    r_pf = r_start * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = (_MU_EARTH / h) * np.array([-np.sin(nu), e_orb + np.cos(nu), 0.0])
    rot = np.column_stack([r_p_hat, tang, np.cross(r_p_hat, tang)])
    r_geo = rot @ r_pf
    v_geo = rot @ v_pf

    # Geocentric offset -> Sun-centered (heliocentric) ecliptic state.
    helio_x = e_r + r_geo - s_r
    helio_v = e_v + v_geo - s_v

    coords = CartesianCoordinates.from_kwargs(
        epoch=[epoch],
        x=[helio_x[0]],
        y=[helio_x[1]],
        z=[helio_x[2]],
        vx=[helio_v[0]],
        vy=[helio_v[1]],
        vz=[helio_v[2]],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    orbits = CartesianOrbits.from_kwargs(
        orbit_id=["GRAZER_TEST"],
        object_id=["grazing_flyby"],
        coordinates=coords,
    )

    # Time-to-perigee from the hyperbolic mean anomaly; the grid spans the
    # full encounter (inbound + perigee + outbound) with fine steps so both
    # the atmosphere and shadow crossings are bracketed.
    big_f = np.arcsinh(np.sin(nu) * np.sqrt(e_orb**2 - 1.0) / (1.0 + e_orb * np.cos(nu)))
    mean_anom = e_orb * np.sinh(big_f) - big_f
    t_to_peri = -mean_anom / np.sqrt(_MU_EARTH / a**3)
    target_epochs = epoch + np.linspace(0.0, 2.4 * t_to_peri, 200)
    return orbits, target_epochs


def test_grazer_event_subtables_no_silent_drops() -> None:
    """Walk the atmospheric-exit / shadow-exit sub-tables.

    We *assert* atmospheric_exits and shadow_exits fired (and impacts == 0 —
    an impact means it struck the ground and the ``*_exit`` families would be
    empty), then walk every fired sub-table for silent drops.
    """
    orbits, target_epochs = _grazing_flyby_orbit()
    cfg = EventConfig(
        close_approaches=True,
        impacts=True,
        atmospheric=True,
        shadow_events=True,
        possible_impacts=False,
        body_filter=[Origin.EARTH],
    )
    result = empyrean.propagate(orbits, target_epochs, events=cfg)
    events = result.events

    assert len(events.impacts) == 0, "grazer struck the ground — *_exit families won't fire"
    assert len(events.atmospheric_exits) > 0, "grazer fired no atmospheric exit"
    assert len(events.shadow_exits) > 0, "grazer fired no shadow exit"

    all_bad_null: list[str] = []
    all_bad_not_null: list[str] = []
    fired: list[str] = []
    for attr, class_name in _EVENT_SUBTABLES:
        table = getattr(events, attr)
        if len(table) == 0:
            continue
        fired.append(attr)
        bad_null, bad_not_null = _check_no_silent_drops(table, class_name)
        all_bad_null += bad_null
        all_bad_not_null += bad_not_null

    assert not (all_bad_null or all_bad_not_null), (
        f"(fired sub-tables: {', '.join(fired)})\n"
        + _format_failures("Grazer event sub-tables", all_bad_null, all_bad_not_null)
    )


# ── Temporary capture: capture_starts + capture_ends ──


def _cd3_capture_orbit() -> CartesianOrbits:
    """2020 CD3 on its real temporarily-captured (mini-moon) arc.

    Real 2020 CD3 Keplerian elements at MJD 61000 TDB (MPCORB) — the same
    state villeneuve's ground-truth Rust test
    ``src/propagation.rs::test_cd3_capture_encounter_trajectory`` uses.
    Converted to Cartesian so a small diagonal covariance can be attached
    (same idiom as ``_full_feature_orbit``): without it the capture
    sub-tables' ``jacobi_constant_sigma`` column is all-null (the C_J
    uncertainty is computed from the propagated 6x6 covariance), so the
    covariance exercises that column rather than masking it.
    """
    kep = KeplerianCoordinates.from_kwargs(
        epoch=[61000.0],
        a=[1.0290321],
        e=[0.0123739],
        i=[0.63395],
        raan=[82.23884],
        ap=[49.97029],
        ma=[204.06955],
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    cart = transform_coordinates(kep, CartesianCoordinates)
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
    cart = cart.set_column("covariance", CartesianCovariance.from_matrix(cov))
    return CartesianOrbits.from_kwargs(
        orbit_id=["2020 CD3"],
        object_id=["2020 CD3"],
        coordinates=cart,
    )


def test_capture_event_subtables_no_silent_drops() -> None:
    """2020 CD3 temporary Earth capture fires capture_starts + capture_ends.

    The object becomes gravitationally bound to Earth (capture_start), orbits
    through the 2017-2020 window, then escapes (capture_end). We assert both
    capture families fired (an empty result is a fixture regression, not a
    silent skip) and walk every fired sub-table for silent drops. Only the
    counts/columns are asserted — the exact capture epochs are chaotic and
    FP-sensitive, so they are deliberately not pinned.
    """
    orbit = _cd3_capture_orbit()
    config = PropagationConfig(
        force_model=ForceModelTier.APPROXIMATE,
        events=EventConfig(
            close_approaches=True,
            dense_output=True,
            dense_output_cadence_days=1.0 / 24.0,  # 1-hour cadence
        ),
        advanced=AdvancedIntegratorConfig(
            origin_switching=OriginSwitchingConfig(enabled=True, hysteresis=0.2),
        ),
    )
    # Coarse 50-day grid over the real 2017-2020 CD3 capture window. Capture
    # detection runs on the main integration grid, so a coarse target grid is
    # sufficient.
    target_epochs = np.array([57000.0 + 50.0 * i for i in range(41)])
    result = empyrean.propagate(orbit, target_epochs, config=config)
    events = result.events

    assert len(events.capture_starts) > 0, "CD3 fixture fired no capture_start"
    assert len(events.capture_ends) > 0, "CD3 fixture fired no capture_end"

    all_bad_null: list[str] = []
    all_bad_not_null: list[str] = []
    fired: list[str] = []
    for attr, class_name in _EVENT_SUBTABLES:
        table = getattr(events, attr)
        if len(table) == 0:
            continue
        fired.append(attr)
        bad_null, bad_not_null = _check_no_silent_drops(table, class_name)
        all_bad_null += bad_null
        all_bad_not_null += bad_not_null

    assert not (all_bad_null or all_bad_not_null), (
        f"(fired sub-tables: {', '.join(fired)})\n"
        + _format_failures("Capture event sub-tables", all_bad_null, all_bad_not_null)
    )


# ── Covariance regime change: UncertaintyMethod.AUTO ──
#
# covariance_regime_changes fire under UncertaintyMethod.AUTO, which records a
# Linear -> SecondOrder transition as a CovarianceRegimeChange. AUTO used to be
# unreachable from empyrean.propagate (silently coerced to first_order in
# PropagationConfig._to_wire_dict) — since fixed.
_KM_PER_AU = 149_597_870.7
_S_PER_DAY = 86_400.0


def _auto_escalation_orbit() -> CartesianOrbits:
    """Apophis with the loose covariance the villeneuve reference test uses to
    exercise AUTO escalation to second order at the 2029 flyby.

    sigma_pos ~5000 km, sigma_vel ~5 cm/s. The 2029 Earth flyby drives AUTO to
    escalate Linear -> SecondOrder at the CA window and emit the regime-change
    rows.
    """
    s = APOPHIS_STATE
    sigma_pos_au = 5000.0 / _KM_PER_AU
    sigma_vel_au_d = 0.05 * _S_PER_DAY / 1e3 / _KM_PER_AU  # 5 cm/s -> AU/day
    cov_matrix = np.diag(
        [
            sigma_pos_au**2,
            sigma_pos_au**2,
            sigma_pos_au**2,
            sigma_vel_au_d**2,
            sigma_vel_au_d**2,
            sigma_vel_au_d**2,
        ]
    )[None, :, :]
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        covariance=CartesianCovariance.from_matrix(cov_matrix),
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    return CartesianOrbits.from_kwargs(
        orbit_id=["AUTO_ESCALATION_TEST"],
        object_id=["99942 Apophis"],
        coordinates=coords,
    )


def test_covariance_regime_changes_fire_under_auto() -> None:
    """covariance_regime_changes fire under UncertaintyMethod.AUTO.

    The Apophis 2029 deep flyby with a 5000 km / 5 cm/s covariance drives AUTO
    to escalate to second order under a real Jet2 pass, so AUTO emits
    Linear -> SecondOrder (and back) regime-change rows at the CA window
    boundary. We assert the family fired (an empty result means
    AUTO regressed to a silent first_order downgrade — a previously fixed bug)
    and walk the CovarianceRegimeChanges columns for silent drops, since this
    is the only fixture that reaches this event family.
    """
    orbits = _auto_escalation_orbit()
    t_ca = 62240.0  # ~2029-04-13 Earth flyby
    target_epochs = np.array([t_ca - 30.0, t_ca - 5.0, t_ca, t_ca + 5.0, t_ca + 30.0])
    result = empyrean.propagate(
        orbits,
        target_epochs,
        uncertainty_method=UncertaintyMethod.AUTO,
        events=EventConfig(body_filter=[Origin.EARTH]),
    )
    regime_changes = result.events.covariance_regime_changes
    assert len(regime_changes) > 0, (
        "AUTO fired no covariance_regime_changes — likely a silent first_order downgrade regression"
    )

    bad_null, bad_not_null = _check_no_silent_drops(regime_changes, "CovarianceRegimeChanges")
    assert not (bad_null or bad_not_null), _format_failures(
        "CovarianceRegimeChanges", bad_null, bad_not_null
    )


def test_auto_method_label_round_trips_in_ip_and_bplane() -> None:
    """The IP / B-plane ``method`` column must report the method that ran.

    Auto IP/B-plane results were silently relabelled ``first_order`` on
    readback because the wrapper's ``method_from_tag`` and the Python
    ``_TAG_TO_METHOD`` both lacked the tag-4 (Auto) arm — the
    IP value was correct but the reported method was wrong. Pin every method's
    label so a future tag-map gap fails loudly rather than silently collapsing
    to first_order.
    """
    orbits = _auto_escalation_orbit()
    for method, expected in (
        (UncertaintyMethod.FIRST_ORDER, "first_order"),
        (UncertaintyMethod.SECOND_ORDER, "second_order"),
        (UncertaintyMethod.AUTO, "auto"),
    ):
        ips = compute_impact_probabilities(
            orbits, end_epoch=62300.0, methods=[method], body_filter=[Origin.EARTH]
        )
        if len(ips) > 0:
            assert set(ips.method.to_pylist()) == {expected}, (
                f"IP method label for {method.value}: got {set(ips.method.to_pylist())}"
            )
        bps = compute_b_planes(
            orbits, end_epoch=62300.0, methods=[method], body_filter=[Origin.EARTH]
        )
        if len(bps) > 0:
            assert set(bps.method.to_pylist()) == {expected}, (
                f"B-plane method label for {method.value}: got {set(bps.method.to_pylist())}"
            )


def test_propagation_state_sensitivities_no_silent_drops() -> None:
    """STM/STT arrays on ``result.sensitivity`` (StateSensitivities).

    Requested via a second-order uncertainty method. The inventory found
    this family clean across the C ABI — this locks that in, so a future
    regression that drops an STM/STT column fails loudly.
    """
    orbits = _full_feature_orbit()
    target_epochs = np.array([61000.0 + 60.0 * i for i in range(40)])
    result = empyrean.propagate(
        orbits, target_epochs, uncertainty_method=UncertaintyMethod.SECOND_ORDER
    )
    if result.sensitivity is None or len(result.sensitivity) == 0:
        pytest.skip("No state sensitivities produced.")

    bad_null, bad_not_null = _check_no_silent_drops(result.sensitivity, "StateSensitivities")
    assert not (bad_null or bad_not_null), _format_failures(
        "StateSensitivities", bad_null, bad_not_null
    )


def test_ephemeris_observation_sensitivities_no_silent_drops() -> None:
    """Observation Jacobian/Hessian chains on ``EphemerisResult.sensitivity``.

    Historically dropped at the C ABI — if the table comes back
    empty, this skips. Once sensitivities are wired through it activates
    and asserts the columns are populated.
    """
    orbits = _full_feature_orbit()
    observers = Observers.from_code("500", [61000.5, 61010.5, 61020.5])
    result = generate_ephemeris(orbits, observers, uncertainty_method=UncertaintyMethod.FIRST_ORDER)
    sens = result.sensitivity
    if sens is None or len(sens) == 0:
        pytest.skip("ObservationSensitivities empty (dropped at C ABI).")

    bad_null, bad_not_null = _check_no_silent_drops(sens, "ObservationSensitivities")
    assert not (bad_null or bad_not_null), _format_failures(
        "ObservationSensitivities", bad_null, bad_not_null
    )


def test_ephemeris_no_silent_drops() -> None:
    orbits = _full_feature_orbit()
    observers = Observers.from_code("500", [61000.5, 61010.5, 61020.5])
    result = generate_ephemeris(orbits, observers)

    if len(result.ephemeris) == 0:
        pytest.skip("No ephemeris rows generated.")

    bad_null, bad_not_null = _check_no_silent_drops(result.ephemeris, "Ephemeris")
    assert not (bad_null or bad_not_null), _format_failures("Ephemeris", bad_null, bad_not_null)


def test_impact_probabilities_no_silent_drops() -> None:
    orbits = _full_feature_orbit()
    ips = compute_impact_probabilities(
        orbits,
        end_epoch=63000.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    if len(ips) == 0:
        pytest.skip("No impact-probability rows produced.")

    bad_null, bad_not_null = _check_no_silent_drops(ips, "ImpactProbabilities")
    assert not (bad_null or bad_not_null), _format_failures(
        "ImpactProbabilities", bad_null, bad_not_null
    )


def test_b_planes_no_silent_drops() -> None:
    orbits = _full_feature_orbit()
    bps = compute_b_planes(
        orbits,
        end_epoch=63000.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    if len(bps) == 0:
        pytest.skip("No b-plane rows produced.")

    bad_null, bad_not_null = _check_no_silent_drops(bps, "BPlanes")
    assert not (bad_null or bad_not_null), _format_failures("BPlanes", bad_null, bad_not_null)


# ── Non-grav covariance reaches the engine (empyrean-3qoe) ────────────
#
# The forward-model marshals (propagate / generate_ephemeris / impact) used to
# silently drop the fitted non-grav 3x3 covariance (`ng_covariance`) that the
# OD path already threaded — only the OD determine→refine loop carried it. The
# fix routes every builder through one exhaustive `assemble_orbit` chokepoint
# and adds the (has_non_grav_cov, non_grav_cov) kwargs to each entry point. The
# load-bearing assertion below is a DIFFERENCE, not a magnitude: an orbit with
# a fitted non-grav covariance must propagate to a LARGER per-state covariance
# than the otherwise-identical orbit without one, because the non-grav
# parameter uncertainty maps into state uncertainty through the STM's non-grav
# partials. That only happens when the non-grav force term is active, so this
# fixture carries a real (transverse Yarkovsky) A2 — unlike the all-zero-coef
# `_full_feature_orbit`, where the covariance is inert.


def _non_grav_solved_orbit(with_cov: bool) -> CartesianOrbits:
    """Apophis with an ACTIVE transverse non-grav (Yarkovsky A2) and,
    when ``with_cov``, a fitted 3x3 non-grav covariance attached.

    The two variants are byte-identical except for the presence of the
    non-grav covariance, so any difference in the propagated per-state
    covariance is attributable solely to ``ng_covariance`` reaching the
    engine.
    """
    s = APOPHIS_STATE
    cov_matrix = np.diag([1e-16, 1e-16, 1e-16, 1e-20, 1e-20, 1e-20])[None, :, :]
    coords = CartesianCoordinates.from_kwargs(
        epoch=[s["epoch"]],
        x=[s["x"]],
        y=[s["y"]],
        z=[s["z"]],
        vx=[s["vx"]],
        vy=[s["vy"]],
        vz=[s["vz"]],
        covariance=CartesianCovariance.from_matrix(cov_matrix),
        frame="ecliptic_j2000",
        origin=[str(Origin.SUN)],
    )
    ng_kwargs: dict[str, object] = {
        "a1": [0.0],
        "a2": [1.0e-14],  # transverse Yarkovsky (AU/day^2) — activates the non-grav term
        "a3": [0.0],
        "model": ["inverse_square"],
    }
    if with_cov:
        # Fitted (A1, A2, A3) covariance, row-major flattened (9 values).
        ng_kwargs["covariance"] = [np.diag([1.0e-20, 1.0e-20, 1.0e-20]).reshape(9).tolist()]
    non_grav = NonGravParams.from_kwargs(**ng_kwargs)
    return CartesianOrbits.from_kwargs(
        orbit_id=["NG_COV_TEST"],
        object_id=["99942 Apophis"],
        coordinates=coords,
        non_grav=non_grav,
    )


def _position_sigma(states: CartesianOrbits) -> np.ndarray:
    """Per-state 1σ position magnitude (AU) from the propagated 6x6 covariance."""
    from empyrean._convert import _covariance_to_matrix

    m = _covariance_to_matrix(states.coordinates.covariance)  # (n, 6, 6)
    return np.sqrt(m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2])


def test_ng_covariance_reaches_propagated_covariance() -> None:
    """A fitted non-grav covariance must inflate the propagated per-state
    covariance — the direct empyrean-3qoe reproduction.

    Propagates two orbits identical except for the presence of ``ng_covariance``
    and asserts (1) the per-state position sigma is strictly larger with the
    covariance present (proving it reached the engine — the bug dropped it and
    the two runs were bit-identical), and (2) the no-covariance run is
    deterministic (control: re-running it reproduces the covariance exactly, so
    the difference in (1) is attributable to ``ng_covariance``, not noise).
    """
    target_epochs = np.array([61000.0 + 30.0 * i for i in range(6)])

    with_cov = empyrean.propagate(
        _non_grav_solved_orbit(with_cov=True),
        target_epochs,
        uncertainty_method=UncertaintyMethod.SECOND_ORDER,
    )
    without_cov = empyrean.propagate(
        _non_grav_solved_orbit(with_cov=False),
        target_epochs,
        uncertainty_method=UncertaintyMethod.SECOND_ORDER,
    )

    sig_with = _position_sigma(with_cov.states)
    sig_without = _position_sigma(without_cov.states)

    # Later states must be strictly inflated by the non-grav prior. (The first
    # state is at the epoch, where the covariance is the input 6x6 for both.)
    assert np.all(sig_with[1:] > sig_without[1:] * (1.0 + 1.0e-6)), (
        "ng_covariance did not inflate the propagated covariance — it was "
        f"dropped before reaching the engine.\n  with cov: {sig_with}\n"
        f"  without:  {sig_without}"
    )

    # Control: the no-cov run is deterministic, so the inflation above is the
    # covariance's doing, not run-to-run noise.
    without_cov_again = empyrean.propagate(
        _non_grav_solved_orbit(with_cov=False),
        target_epochs,
        uncertainty_method=UncertaintyMethod.SECOND_ORDER,
    )
    sig_without_again = _position_sigma(without_cov_again.states)
    np.testing.assert_array_equal(
        sig_without,
        sig_without_again,
        err_msg="no-covariance propagation is non-deterministic; control invalid",
    )


def test_ng_covariance_threads_through_ephemeris_and_impact() -> None:
    """The ephemeris / impact entry points accept and thread ``ng_covariance``.

    Both marshals now route through the same ``assemble_orbit`` chokepoint and
    accept the ``(has_non_grav_cov, non_grav_cov)`` kwargs, so a fitted orbit
    reaches these paths without a marshal-time drop or crash. This test drives
    both with a covariance-bearing orbit and asserts they produce valid output.

    A numeric DIFFERENCE assertion is deliberately NOT made here — it is not
    observable in this distribution:

    * generate_ephemeris — the sky (spherical) covariance is dropped at the C
      ABI (see the ``Ephemeris.coordinates.covariance.*`` allow-list entries
      above), so the non-grav contribution cannot be read back from Python.
    * compute_impact_probabilities / compute_b_planes — the engine's IP /
      B-plane uncertainty is built from the 6x6 state covariance only; it does
      not fold in the non-grav 3x3, so ``sigma_distance`` is unchanged by
      ``ng_covariance`` (verified: unchanged even with a 1e-12 covariance).

    The load-bearing DIFFERENCE gate is
    :func:`test_ng_covariance_reaches_propagated_covariance` (propagate path).
    """
    orbits = _non_grav_solved_orbit(with_cov=True)

    observers = Observers.from_code("500", [61000.5, 61010.5, 61020.5])
    eph = generate_ephemeris(orbits, observers)
    assert len(eph.ephemeris) > 0, "ephemeris path produced no rows with ng_covariance present"

    ips = compute_impact_probabilities(
        orbits,
        end_epoch=62300.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    bps = compute_b_planes(
        orbits,
        end_epoch=62300.0,
        methods=[UncertaintyMethod.FIRST_ORDER],
        body_filter=[Origin.EARTH],
    )
    # The Apophis 2029 flyby fires a close approach, so both channels return a
    # row; a finite sigma confirms the covariance-bearing orbit propagated
    # through without a marshal-time failure.
    assert len(ips) > 0 and np.all(
        np.isfinite(ips.sigma_distance_km.to_numpy(zero_copy_only=False))
    )
    assert len(bps) > 0
