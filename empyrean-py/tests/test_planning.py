"""Contracts for the observation-planning surface.

Two halves, split the way ``test_determine_batch.py`` splits: the
pure-Python assembly (wire lowering, tagged-union validation, enum maps,
table schema) runs always, and the engine round-trip is
``validation``-marked because it needs kernels and the compiled
extension.

The contracts locked here:

* Every ``PlannedObservation`` field reaches the wire, and an absent
  optional lowers to the NaN sentinel the C ABI reads — never to a
  substituted number. ``radar_snr=None`` is a *request* to derive the
  SNR from the link budget, not a missing value, and it lowers the same
  way.
* ``num_threads=0`` ("all available cores" on the Python surface) lowers
  to the C ABI's ``-1`` spelling of the same thing.
* A field belonging to the other kind of candidate is rejected at
  construction, with a message naming every offending field.
* Tag maps are bidirectional, and the ``-1`` radar-mode sentinel comes
  back as ``None`` rather than being coerced to a mode.
* The optical and radar blocks are each populated only on their own
  rows, the cross-block is null (not zero), and link-budget provenance
  appears only where the link budget actually ran.
"""

from __future__ import annotations

import dataclasses

import empyrean
import numpy as np
import pytest
from empyrean import (
    STAGE_POSTERIOR,
    STAGE_PRIOR,
    CartesianOrbits,
    ObservatoryConfig,
    PlanCandidates,
    PlanEphemeris,
    PlanMetrics,
    PlannedObservation,
    PlannedObservationKind,
    PlanningConfig,
    RadarMode,
    RadarStation,
    evaluate_plan,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.planning.result import (
    _INT_TO_KIND,
    _INT_TO_RADAR_MODE,
    _KIND_TO_INT,
    _RADAR_MODE_TO_INT,
    _RADAR_STATION_TO_INT,
)

# The PlanCandidates schema, in order. Every field the C ABI's
# EmpyreanPlanCandidate declares reaches one of these columns; a
# reordering or a silent removal has to show up here first.
EXPECTED_COLUMNS = [
    "index",
    "obs_code",
    "kind",
    "observable",
    "marginal_volume_reduction",
    "marginal_position_improvement",
    "active_width",
    "cumulative_position_sigma_km",
    "cumulative_velocity_sigma_m_s",
    "cumulative_semi_major_km",
    "cumulative_semi_minor_km",
    "cumulative_log_det",
    "along_track_sigma_arcsec",
    "cross_track_sigma_arcsec",
    "ra_sigma_arcsec",
    "dec_sigma_arcsec",
    "position_angle_deg",
    "post_along_track_sigma_arcsec",
    "post_cross_track_sigma_arcsec",
    "radar_mode",
    "radar_snr",
    "radar_range_km",
    "radar_provenance",
]

# The PlanMetrics schema, in order. `stage` labels which end of the
# campaign a row describes; the other five mirror
# EmpyreanCovarianceMetrics.
EXPECTED_METRIC_COLUMNS = [
    "stage",
    "position_sigma_km",
    "velocity_sigma_m_s",
    "semi_major_km",
    "semi_minor_km",
    "log_det",
]

# Apophis state at MJD 61000 TDB, shifted to the barycentric ecliptic
# basis the planner evaluates in. Hardcoded so the fixture is hermetic —
# no SBDB/Horizons fetch, no bundled file.
_EPOCH_MJD_TDB = 61000.0
_APOPHIS_SSB_ECLIPTIC = {
    "x": -8.18903154079353157e-02,
    "y": -8.25280676309557037e-01,
    "z": 4.20312157909258355e-02,
    "vx": 1.98823903215481629e-02,
    "vy": 1.32186530066730917e-03,
    "vz": 3.99358827504333659e-04,
}


def _plan_orbit() -> CartesianOrbits:
    """Apophis with a diagonal covariance, barycentric ecliptic.

    The covariance is the information prior the plan is evaluated
    against; a synthetic diagonal one exercises the whole marshal +
    metrics path without a fit.
    """
    cov = np.zeros((1, 6, 6))
    for k, d in enumerate([1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 1e-18]):
        cov[0, k, k] = d
    return CartesianOrbits.from_kwargs(
        orbit_id=["apophis_plan"],
        object_id=["99942"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_EPOCH_MJD_TDB]),
            x=[_APOPHIS_SSB_ECLIPTIC["x"]],
            y=[_APOPHIS_SSB_ECLIPTIC["y"]],
            z=[_APOPHIS_SSB_ECLIPTIC["z"]],
            vx=[_APOPHIS_SSB_ECLIPTIC["vx"]],
            vy=[_APOPHIS_SSB_ECLIPTIC["vy"]],
            vz=[_APOPHIS_SSB_ECLIPTIC["vz"]],
            frame="ecliptic_j2000",
            origin=["SSB"],
            covariance=CartesianCovariance.from_matrix(cov),
        ),
    )


def _full_plan() -> list[PlannedObservation]:
    """Two optical candidates at different sites, one radar candidate
    with a caller-supplied SNR, one that has to run the link budget.

    Every block of the output surface has an input that populates it:
    the optical rows fill the sky-plane geometry, the radar rows fill the
    radar block, and only the link-budget row can produce provenance
    notes.
    """
    return [
        PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2)),
        PlannedObservation.optical(_EPOCH_MJD_TDB + 12.0, "568", (0.3, 0.3)),
        PlannedObservation.radar(
            _EPOCH_MJD_TDB + 15.0,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_snr=50.0,
        ),
        PlannedObservation.radar(
            _EPOCH_MJD_TDB + 20.0,
            radar_mode=RadarMode.DELAY,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_target_h_mag=19.7,
            radar_target_visual_albedo=0.23,
            radar_target_radar_albedo=0.15,
            radar_integration_s=600.0,
        ),
    ]


# ── Wire lowering ────────────────────────────────────────────────────


def test_optical_candidate_lowers_every_field():
    obs = PlannedObservation.optical(61010.0, "F51", (0.2, 0.35))
    wire = obs._to_wire_dict()

    assert wire["kind"] == 0, f"optical must lower to kind 0, got {wire['kind']}"
    assert wire["epoch_mjd_tdb"] == 61010.0
    assert wire["optical_code"] == "F51"
    assert wire["optical_sigma_ra_arcsec"] == 0.2
    assert wire["optical_sigma_dec_arcsec"] == 0.35
    # Every radar slot still ships — the C struct is flat and reads the
    # radar half only when kind == 1, but a missing key would be a
    # KeyError at the binding, not a silent default.
    for key in ("radar_snr", "radar_target_h_mag", "radar_target_diameter_km"):
        assert np.isnan(wire[key]), f"{key} must lower to NaN on an optical candidate"


def test_radar_candidate_lowers_absent_properties_to_nan():
    obs = PlannedObservation.radar(
        61020.0,
        radar_transmit_station=RadarStation.GOLDSTONE_DSS14,
        radar_receive_station=RadarStation.GREEN_BANK,
        radar_mode=RadarMode.DELAY,
        radar_bandwidth_hz=1.0e5,
        radar_freq_resolution_hz=0.1,
        radar_target_diameter_km=0.34,
        radar_integration_s=600.0,
    )
    wire = obs._to_wire_dict()

    assert wire["kind"] == 1
    assert wire["radar_transmit_station"] == 0
    assert wire["radar_receive_station"] == 1
    assert wire["radar_mode"] == 0
    assert wire["radar_bandwidth_hz"] == 1.0e5
    assert wire["radar_freq_resolution_hz"] == 0.1
    assert wire["radar_integration_s"] == 600.0
    assert wire["radar_target_diameter_km"] == 0.34
    # snr=None is the "derive it from the link budget" request, and the
    # unknown properties are genuinely unknown. Both are NaN on the wire;
    # neither may become a number.
    assert np.isnan(wire["radar_snr"]), "radar_snr=None must lower to NaN"
    for key in (
        "radar_target_h_mag",
        "radar_target_visual_albedo",
        "radar_target_radar_albedo",
        "radar_target_spin_period_hours",
    ):
        assert np.isnan(wire[key]), f"absent {key} must lower to NaN, got {wire[key]}"


def test_supplied_snr_lowers_as_a_number():
    obs = PlannedObservation.radar(
        61020.0,
        radar_bandwidth_hz=1.0e5,
        radar_freq_resolution_hz=0.1,
        radar_snr=50.0,
    )
    assert obs._to_wire_dict()["radar_snr"] == 50.0


def test_planning_config_default_lowering_is_a_no_op():
    wire = PlanningConfig()._to_wire_dict()
    assert wire["force_model"] == "standard"
    assert wire["epsilon"] == 1e-9, "epsilon default must match the engine default"
    assert wire["observatories"] == []
    # The Python surface spells "all available cores" 0 (matching
    # ODConfig.num_threads); the C ABI spells it -1.
    assert wire["num_threads"] == -1, f"num_threads=0 must lower to -1, got {wire['num_threads']}"


def test_config_knobs_the_planner_never_reads_are_refused_at_construction():
    """``observatories`` and ``num_threads`` ride the shared planning
    config but are unread on this entry point.

    Accepting either silently would be a dead knob carried across three
    layers, so both are rejected with a message naming the field and the
    alternative. Delete the matching arm in
    ``PlanningConfig._reject_unread_knobs`` — and the field's doc caveat
    — if a release ever wires one of them through.
    """
    with pytest.raises(ValueError) as exc:
        PlanningConfig(observatories=[_observatory()])
    message = str(exc.value)
    assert "observatories" in message, message
    assert "PlannedObservation" in message, f"the refusal must name the alternative: {message}"

    with pytest.raises(ValueError, match="num_threads"):
        PlanningConfig(num_threads=4)


def test_unread_config_knobs_set_after_construction_are_still_refused_as_valueerror():
    """The refusal survives mutation, and stays a ``ValueError``.

    ``PlanningConfig`` is an unfrozen dataclass by house convention, so a
    field assigned after construction never re-runs ``__post_init__``.
    Without the second check inside ``evaluate_plan`` the caller would
    fall through to the Rust wrapper's guard and get a ``RuntimeError``
    from the engine layer — the wrong class for a caller mistake caught
    before the engine runs, and not what the field docstrings promise.
    """
    orbit = _plan_orbit()
    planned = [PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2))]

    mutated = PlanningConfig()
    mutated.num_threads = 4
    with pytest.raises(ValueError, match="num_threads"):
        evaluate_plan(orbit, planned, mutated)

    mutated = PlanningConfig()
    mutated.observatories = [_observatory()]
    with pytest.raises(ValueError, match="observatories"):
        evaluate_plan(orbit, planned, mutated)


def test_observatory_config_lowers_every_field():
    wire = ObservatoryConfig(
        obs_code="F51",
        sigma_arcsec=(0.2, 0.3),
        max_apparent_mag=23.5,
        min_elongation_deg=45.0,
    )._to_wire_dict()
    assert wire == {
        "obs_code": "F51",
        "sigma_ra_arcsec": 0.2,
        "sigma_dec_arcsec": 0.3,
        "max_apparent_mag": 23.5,
        "min_elongation_deg": 45.0,
    }


def test_config_lowering_covers_every_dataclass_field():
    """Forcing function: a new PlanningConfig field must reach the wire.

    The wire dict is hand-written, so a field added to the dataclass and
    forgotten in ``_to_wire_dict`` would be silently inert. Reflecting
    the live dataclass turns that into a red test.
    """
    declared = {f.name for f in dataclasses.fields(PlanningConfig)}
    lowered = set(PlanningConfig()._to_wire_dict())
    missing = declared - lowered
    assert not missing, (
        f"PlanningConfig field(s) never reach the wire: {sorted(missing)} — "
        f"add them to PlanningConfig._to_wire_dict and to the binding's "
        f"build_planning_config_from_dict."
    )


def test_planned_observation_lowering_covers_every_dataclass_field():
    declared = {f.name for f in dataclasses.fields(PlannedObservation)}
    # optical_sigma_arcsec is a pair that splits into two wire columns.
    declared.discard("optical_sigma_arcsec")
    lowered = set(PlannedObservation.optical(61000.0, "F51")._to_wire_dict())
    missing = declared - lowered
    assert not missing, (
        f"PlannedObservation field(s) never reach the wire: {sorted(missing)} — "
        f"add them to _to_wire_dict, to _PLANNED_COLUMNS, and to the "
        f"binding's build_planned_observations."
    )


# ── Tagged-union validation ──────────────────────────────────────────


def test_optical_candidate_rejects_radar_fields():
    with pytest.raises(ValueError) as exc:
        PlannedObservation(
            epoch_mjd_tdb=61000.0,
            kind=PlannedObservationKind.OPTICAL,
            optical_code="F51",
            radar_bandwidth_hz=1.0e5,
            radar_snr=50.0,
        )
    message = str(exc.value)
    assert "radar_bandwidth_hz" in message, message
    assert "radar_snr" in message, message


def test_radar_candidate_rejects_optical_fields():
    with pytest.raises(ValueError) as exc:
        PlannedObservation(
            epoch_mjd_tdb=61000.0,
            kind=PlannedObservationKind.RADAR,
            optical_code="F51",
            optical_sigma_arcsec=(0.2, 0.2),
            radar_bandwidth_hz=1.0e5,
        )
    message = str(exc.value)
    assert "optical_code" in message, message
    assert "optical_sigma_arcsec" in message, message


def test_optical_candidate_requires_a_station_code():
    with pytest.raises(ValueError, match="optical_code"):
        PlannedObservation(epoch_mjd_tdb=61000.0, kind=PlannedObservationKind.OPTICAL)


def test_optical_sigma_must_be_positive():
    with pytest.raises(ValueError, match="optical_sigma_arcsec"):
        PlannedObservation.optical(61000.0, "F51", (0.0, 0.2))


def test_non_positive_snr_is_rejected_rather_than_read_as_absent():
    """A zero or negative SNR is a mistake, not a request to run the link
    budget — only ``None`` means that."""
    with pytest.raises(ValueError, match="radar_snr"):
        PlannedObservation.radar(
            61000.0,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_snr=0.0,
        )


def test_non_finite_epoch_is_rejected():
    with pytest.raises(ValueError, match="epoch_mjd_tdb"):
        PlannedObservation.optical(float("nan"), "F51")


# ── Enum maps ────────────────────────────────────────────────────────


def test_tag_maps_accept_both_the_enum_and_its_bare_string():
    assert _KIND_TO_INT[PlannedObservationKind.RADAR] == _KIND_TO_INT["radar"]
    assert _RADAR_MODE_TO_INT[RadarMode.DOPPLER] == _RADAR_MODE_TO_INT["doppler"]
    assert _RADAR_STATION_TO_INT[RadarStation.GREEN_BANK] == _RADAR_STATION_TO_INT["green_bank"]


def test_tag_maps_are_bidirectional():
    for member in PlannedObservationKind:
        assert _INT_TO_KIND[_KIND_TO_INT[member]] == member.value
    for mode in RadarMode:
        assert _INT_TO_RADAR_MODE[_RADAR_MODE_TO_INT[mode]] == mode.value


def test_optical_rows_carry_no_radar_mode():
    """``-1`` is the engine's "not a radar candidate" sentinel; it must
    never resolve to a mode."""
    assert -1 not in _INT_TO_RADAR_MODE


def test_every_radar_station_has_a_tag():
    for station in RadarStation:
        assert station in _RADAR_STATION_TO_INT, f"{station} has no wire tag"


# ── Table schemas ────────────────────────────────────────────────────


def test_plan_candidates_schema_is_pinned():
    names = PlanCandidates.empty().table.schema.names
    assert names == EXPECTED_COLUMNS, (
        f"PlanCandidates schema drifted:\n  got      {names}\n  expected {EXPECTED_COLUMNS}"
    )


def test_plan_metrics_schema_is_pinned():
    names = PlanMetrics.empty().table.schema.names
    assert names == EXPECTED_METRIC_COLUMNS, (
        f"PlanMetrics schema drifted:\n  got      {names}\n  expected {EXPECTED_METRIC_COLUMNS}"
    )


def test_plan_metrics_stage_helpers_filter_by_label():
    metrics = PlanMetrics.from_kwargs(
        stage=[STAGE_PRIOR, STAGE_POSTERIOR],
        position_sigma_km=[25.0, 16.0],
        velocity_sigma_m_s=[3.0e-3, 2.9e-3],
        semi_major_km=[15.0, 14.9],
        semi_minor_km=[15.0, 0.7],
        log_det=[-221.0, -238.0],
    )
    assert metrics.prior().position_sigma_km[0].as_py() == 25.0
    assert metrics.posterior().position_sigma_km[0].as_py() == 16.0
    assert isinstance(metrics.prior(), PlanMetrics)


def test_plan_ephemeris_carries_epochs_as_a_subtable():
    """The epoch rides as an ``Epochs`` sub-table, not a raw MJD float, so
    ``eph.epochs.to_utc()`` keeps row alignment."""
    names = PlanEphemeris.empty().table.schema.names
    assert names == ["epochs", "ra_deg", "dec_deg"], names


def _candidates_with_gains(gains: list[float]) -> PlanCandidates:
    """A hand-built candidate table carrying just the ranking metric.

    Every other required column is filled with a constant, so a ranking
    assertion can only be about ``marginal_volume_reduction``.
    """
    n = len(gains)
    return PlanCandidates.from_kwargs(
        index=list(range(n)),
        obs_code=[f"S{i}" for i in range(n)],
        kind=["optical"] * n,
        observable=[True] * n,
        marginal_volume_reduction=gains,
        marginal_position_improvement=[0.0] * n,
        active_width=[6] * n,
        cumulative_position_sigma_km=[1.0] * n,
        cumulative_velocity_sigma_m_s=[1.0] * n,
        cumulative_semi_major_km=[1.0] * n,
        cumulative_semi_minor_km=[1.0] * n,
        cumulative_log_det=[0.0] * n,
        radar_provenance=[[] for _ in range(n)],
    )


def test_best_by_information_gain_returns_rank_order():
    """The rows come back best-first, not in table order.

    ``apply_mask`` returns the right SET in table order, which silently
    breaks the ranking the method promises — the reason this helper goes
    through ``take`` with the sorted index array.
    """
    # Best (smallest reduction factor) is the LAST row, so a table-order
    # result would be distinguishable from a ranked one.
    candidates = _candidates_with_gains([0.9, 0.5, 0.99, 0.1])

    best = candidates.best_by_information_gain(3)
    assert best.marginal_volume_reduction.to_pylist() == [0.1, 0.5, 0.9], (
        f"expected best-first ranking, got {best.marginal_volume_reduction.to_pylist()}"
    )
    assert best.obs_code.to_pylist() == ["S3", "S1", "S0"], best.obs_code.to_pylist()

    # n larger than the table returns every row, still ranked.
    assert candidates.best_by_information_gain(99).marginal_volume_reduction.to_pylist() == [
        0.1,
        0.5,
        0.9,
        0.99,
    ]


def test_best_by_information_gain_sorts_unranked_candidates_last():
    """A NaN reduction factor is an absent ranking, not the best one."""
    candidates = _candidates_with_gains([float("nan"), 0.4, 0.2])
    ranked = candidates.best_by_information_gain(3).marginal_volume_reduction.to_pylist()
    assert ranked[:2] == [0.2, 0.4], ranked
    assert np.isnan(ranked[2]), f"NaN must rank last, got {ranked}"


def test_selection_helpers_return_the_same_type():
    empty = PlanCandidates.empty()
    assert isinstance(empty.observable_only(), PlanCandidates)
    assert isinstance(empty.select_station("F51"), PlanCandidates)
    assert isinstance(empty.best_by_information_gain(3), PlanCandidates)


# ── Input guards on the entry point ──────────────────────────────────


def test_evaluate_plan_rejects_an_empty_plan():
    with pytest.raises(ValueError, match="at least one planned observation"):
        evaluate_plan(_plan_orbit(), [])


def test_evaluate_plan_rejects_an_orbit_without_covariance():
    orbit = _plan_orbit()
    bare = CartesianOrbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=orbit.coordinates.epoch,
            x=orbit.coordinates.x,
            y=orbit.coordinates.y,
            z=orbit.coordinates.z,
            vx=orbit.coordinates.vx,
            vy=orbit.coordinates.vy,
            vz=orbit.coordinates.vz,
            frame="ecliptic_j2000",
            origin=["SSB"],
        ),
    )
    with pytest.raises(ValueError, match="covariance"):
        evaluate_plan(bare, _full_plan())


def test_evaluate_plan_names_the_fix_for_a_non_barycentric_orbit():
    """A heliocentric fit is the common input; the refusal has to say how
    to convert it, not just that it is wrong."""
    orbit = _plan_orbit()
    coords = orbit.coordinates
    helio = CartesianOrbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=coords.epoch,
            x=coords.x,
            y=coords.y,
            z=coords.z,
            vx=coords.vx,
            vy=coords.vy,
            vz=coords.vz,
            frame="ecliptic_j2000",
            origin=["Sun"],
            covariance=coords.covariance,
        ),
    )
    with pytest.raises(ValueError) as exc:
        evaluate_plan(helio, _full_plan())
    message = str(exc.value)
    assert "barycenter" in message, message
    assert "transform_coordinates" in message, message


# ── Engine round-trip ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def plan():
    empyrean.initialize()
    return evaluate_plan(_plan_orbit(), _full_plan())


def _observatory() -> ObservatoryConfig:
    """A fully specified site — the dataclass invents no defaults, so a
    refusal test has to say what it is refusing."""
    return ObservatoryConfig(
        obs_code="F51",
        sigma_arcsec=(0.2, 0.2),
        max_apparent_mag=23.5,
        min_elongation_deg=45.0,
    )


@pytest.mark.validation
def test_plan_round_trip_populates_every_block(plan):
    """One plan, four candidates, both kinds — each block populated only
    on its own rows, and the campaign can only add information."""
    assert plan.orbit_id == "apophis_plan", plan.orbit_id
    assert plan.active_width == 6, plan.active_width
    assert len(plan.candidates) == 4, f"expected 4 candidates, got {len(plan.candidates)}"

    assert plan.metrics.stage.to_pylist() == ["prior", "posterior"], (
        f"metrics brackets the campaign with exactly those two rows, got "
        f"{plan.metrics.stage.to_pylist()}"
    )
    prior = plan.metrics.prior()
    posterior = plan.metrics.posterior()
    assert posterior.position_sigma_km[0].as_py() <= prior.position_sigma_km[0].as_py(), (
        f"folding observations cannot loosen the orbit: prior "
        f"{prior.position_sigma_km[0].as_py()} km → posterior "
        f"{posterior.position_sigma_km[0].as_py()} km"
    )
    assert posterior.log_det[0].as_py() <= prior.log_det[0].as_py(), (
        f"posterior log_det {posterior.log_det[0].as_py()} exceeds prior {prior.log_det[0].as_py()}"
    )


@pytest.mark.validation
def test_optical_and_radar_blocks_are_populated_on_their_own_rows(plan):
    kinds = plan.candidates.kind.to_pylist()
    assert sorted(kinds) == ["optical", "optical", "radar", "radar"], kinds

    optical = plan.candidates.apply_mask(np.array([k == "optical" for k in kinds]))
    radar = plan.candidates.apply_mask(np.array([k == "radar" for k in kinds]))

    # Sky-plane geometry: real on optical rows, null on radar rows. A
    # zero there would read as a measured zero, not as "not applicable".
    for column in (
        "along_track_sigma_arcsec",
        "cross_track_sigma_arcsec",
        "ra_sigma_arcsec",
        "dec_sigma_arcsec",
        "position_angle_deg",
        "post_along_track_sigma_arcsec",
        "post_cross_track_sigma_arcsec",
    ):
        assert optical.column(column).null_count == 0, f"{column} null on an optical row"
        assert radar.column(column).null_count == len(radar), (
            f"{column} must be null on every radar row, got {radar.column(column).to_pylist()}"
        )

    # Radar block: the mirror image.
    for column in ("radar_mode", "radar_snr", "radar_range_km"):
        assert radar.column(column).null_count == 0, f"{column} null on a radar row"
        assert optical.column(column).null_count == len(optical), (
            f"{column} must be null on every optical row, got {optical.column(column).to_pylist()}"
        )

    snrs = radar.radar_snr.to_pylist()
    assert all(s is not None and s > 0.0 for s in snrs), snrs
    ranges = radar.radar_range_km.to_pylist()
    assert all(r is not None and r > 0.0 for r in ranges), ranges
    assert set(radar.radar_mode.to_pylist()) <= {"delay", "doppler", "both"}


@pytest.mark.validation
def test_provenance_appears_only_where_the_link_budget_ran(plan):
    """A caller-supplied SNR needs no assumptions; a derived one records
    every assumption it made."""
    provenance = plan.candidates.radar_provenance.to_pylist()
    kinds = plan.candidates.kind.to_pylist()
    snrs = plan.candidates.radar_snr.to_pylist()

    for kind, notes in zip(kinds, provenance, strict=True):
        if kind == "optical":
            assert notes == [], f"optical row carries link-budget notes: {notes}"

    # Exactly one candidate ran the link budget: the one built without an
    # SNR. It is the only row that may carry notes, and it must, because
    # its diameter had to be derived from H and p_V.
    non_empty = [n for n in provenance if n]
    assert len(non_empty) == 1, f"expected notes on exactly one candidate, got {provenance}"
    assert all(isinstance(note, str) and note for note in non_empty[0]), non_empty[0]
    # The supplied-SNR candidate came back with exactly what was asked for.
    assert 50.0 in snrs, snrs


@pytest.mark.validation
def test_ephemeris_has_one_row_per_optical_candidate_and_cross_links(plan):
    kinds = plan.candidates.kind.to_pylist()
    n_optical = sum(1 for k in kinds if k == "optical")
    assert len(plan.ephemeris) == n_optical, (
        f"expected {n_optical} ephemeris rows (one per optical candidate), "
        f"got {len(plan.ephemeris)}"
    )

    # An optical candidate's `index` resolves into the ephemeris table.
    indices = plan.candidates.index.to_pylist()
    optical_indices = sorted(i for i, k in zip(indices, kinds, strict=True) if k == "optical")
    assert optical_indices == list(range(n_optical)), optical_indices

    ra = np.asarray(plan.ephemeris.ra_deg.to_numpy(zero_copy_only=False))
    dec = np.asarray(plan.ephemeris.dec_deg.to_numpy(zero_copy_only=False))
    assert np.isfinite(ra).all() and np.isfinite(dec).all(), (ra, dec)
    assert ((ra >= 0.0) & (ra <= 360.0)).all(), ra
    assert ((dec >= -90.0) & (dec <= 90.0)).all(), dec

    epochs = plan.ephemeris.epochs.to_tdb().mjd.to_numpy(zero_copy_only=False)
    assert (np.diff(epochs) > 0).all(), f"ephemeris must be chronological, got {epochs}"


@pytest.mark.validation
def test_cumulative_metrics_tighten_monotonically(plan):
    """Each fold adds information, so the running log-determinant can only
    fall and the last row must land on the posterior."""
    log_det = plan.candidates.cumulative_log_det.to_numpy(zero_copy_only=False)
    assert (np.diff(log_det) <= 1e-9).all(), log_det
    assert log_det[-1] == pytest.approx(plan.metrics.posterior().log_det[0].as_py(), rel=1e-12)

    reductions = plan.candidates.marginal_volume_reduction.to_numpy(zero_copy_only=False)
    assert ((reductions > 0.0) & (reductions <= 1.0 + 1e-12)).all(), reductions
    improvements = plan.candidates.marginal_position_improvement.to_numpy(zero_copy_only=False)
    assert ((improvements >= -1e-12) & (improvements <= 1.0)).all(), improvements


@pytest.mark.validation
def test_selection_helpers_run_against_a_real_plan(plan):
    best = plan.candidates.best_by_information_gain(2)
    assert len(best) == 2, len(best)
    gains = plan.candidates.marginal_volume_reduction.to_numpy(zero_copy_only=False)
    kept = best.marginal_volume_reduction.to_numpy(zero_copy_only=False)
    assert max(kept) <= sorted(gains)[1], (kept, gains)
    # Rank order, not table order — see test_best_by_information_gain_returns_rank_order.
    assert list(kept) == sorted(kept), f"best-first ranking not preserved: {kept}"

    codes = plan.candidates.obs_code.to_pylist()
    picked = plan.candidates.select_station(codes[0])
    assert len(picked) >= 1
    assert set(picked.obs_code.to_pylist()) == {codes[0]}

    assert len(plan.candidates.observable_only()) <= len(plan.candidates)


@pytest.mark.validation
def test_explicit_orbit_id_overrides_the_orbit_label():
    plan = evaluate_plan(_plan_orbit(), _full_plan(), orbit_id="campaign-2029")
    assert plan.orbit_id == "campaign-2029", plan.orbit_id


@pytest.mark.validation
def test_a_defunct_dish_is_refused_rather_than_scheduled():
    with pytest.raises(RuntimeError):
        evaluate_plan(
            _plan_orbit(),
            [
                PlannedObservation.radar(
                    _EPOCH_MJD_TDB + 15.0,
                    radar_transmit_station=RadarStation.ARECIBO,
                    radar_receive_station=RadarStation.ARECIBO,
                    radar_bandwidth_hz=1.0e5,
                    radar_freq_resolution_hz=0.1,
                    radar_snr=50.0,
                )
            ],
        )


@pytest.mark.validation
def test_an_underspecified_link_budget_is_refused_rather_than_defaulted():
    """The link budget names the property it is missing instead of
    substituting a plausible value."""
    with pytest.raises(RuntimeError) as exc:
        evaluate_plan(
            _plan_orbit(),
            [
                PlannedObservation.radar(
                    _EPOCH_MJD_TDB + 15.0,
                    radar_bandwidth_hz=1.0e5,
                    radar_freq_resolution_hz=0.1,
                    radar_integration_s=600.0,
                )
            ],
        )
    assert "albedo" in str(exc.value).lower(), str(exc.value)


# ── Value-level pins the review asked for ────────────────────────────


@pytest.mark.validation
def test_radar_rows_always_report_observable_true(plan):
    """The column means different things per kind, and the radar half is
    a structural constant.

    No radar feasibility test runs on this entry point — not even the
    antenna-elevation limit the station capability itself declares — so a
    radar row can never carry ``False``. Pinning it keeps the docstring
    honest: if the engine ever starts gating radar rows, this fails and
    the doc has to be rewritten.
    """
    kinds = plan.candidates.kind.to_pylist()
    observable = plan.candidates.observable.to_pylist()
    radar_flags = [o for k, o in zip(kinds, observable, strict=True) if k == "radar"]
    assert radar_flags, "fixture must contain radar candidates"
    assert all(radar_flags), f"radar rows must all report observable=True, got {radar_flags}"


@pytest.mark.validation
def test_unobservable_candidates_still_fold_into_the_posterior():
    """``observable`` does not gate the fold.

    The engine folds every submitted candidate; the flag is reported, not
    acted on. A caller who reads ``posterior`` as "what the observable
    subset buys" is reading it wrong, so pin the actual contract: adding
    an unobservable candidate to a plan still tightens the posterior.
    """
    orbit = _plan_orbit()
    base = [PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2))]

    baseline = evaluate_plan(orbit, base)
    # Solar conjunction: inside the engine's elongation floor, so this
    # candidate comes back unobservable.
    conjunction = PlannedObservation.optical(_EPOCH_MJD_TDB + 100.0, "F51", (0.2, 0.2))
    widened = evaluate_plan(orbit, [*base, conjunction])

    flags = widened.candidates.observable.to_pylist()
    assert not all(flags), (
        f"fixture must contain an unobservable candidate for this contract to "
        f"be testable, got {flags}"
    )
    assert len(widened.candidates) == 2, len(widened.candidates)

    base_log_det = baseline.metrics.posterior().log_det[0].as_py()
    widened_log_det = widened.metrics.posterior().log_det[0].as_py()
    assert widened_log_det < base_log_det, (
        f"the unobservable candidate was still folded, so the posterior must "
        f"tighten: {base_log_det} → {widened_log_det}"
    )


@pytest.mark.validation
def test_every_candidate_actually_informed_the_fit(plan):
    """Guard against a plan that evaluated nothing.

    The engine has a silent no-op fallback: a candidate whose Jacobian is
    missing is reported with a reduction factor of exactly 1.0, zero
    position improvement, and the prior σ copied into the post-fold
    fields — indistinguishable from a candidate that was evaluated and
    found worthless. Every band below excludes those values, so a fixture
    that stopped informing the fit fails here rather than passing
    vacuously.
    """
    reductions = plan.candidates.marginal_volume_reduction.to_numpy(zero_copy_only=False)
    improvements = plan.candidates.marginal_position_improvement.to_numpy(zero_copy_only=False)
    assert (reductions < 1.0).all(), f"a candidate contributed nothing: {reductions}"
    assert (improvements > 0.0).all(), f"a candidate contributed nothing: {improvements}"

    log_det = plan.candidates.cumulative_log_det.to_numpy(zero_copy_only=False)
    assert (np.diff(log_det) < 0.0).all(), f"cumulative log_det must strictly fall: {log_det}"
    prior = plan.metrics.prior().log_det[0].as_py()
    posterior = plan.metrics.posterior().log_det[0].as_py()
    assert posterior < prior, f"posterior must be strictly tighter: {prior} → {posterior}"

    # The sky-plane block is the other place a structural zero could pass
    # for a measurement: a missing on-sky covariance yields 0.0 arcsec.
    kinds = plan.candidates.kind.to_pylist()
    optical = plan.candidates.apply_mask(np.array([k == "optical" for k in kinds]))
    assert len(optical) > 0
    for column in (
        "along_track_sigma_arcsec",
        "cross_track_sigma_arcsec",
        "ra_sigma_arcsec",
        "dec_sigma_arcsec",
        "post_along_track_sigma_arcsec",
        "post_cross_track_sigma_arcsec",
    ):
        values = optical.column(column).to_numpy(zero_copy_only=False)
        assert np.isfinite(values).all(), f"{column} is not finite: {values}"
        assert (values > 0.0).all(), f"{column} carries a structural zero: {values}"

    # The seventh sky-plane column cannot carry a `> 0` band — it is an
    # atan2 result that is legitimately negative for westward sky motion —
    # so it gets a range assertion instead. Without it this is the one
    # column with no finiteness check at any layer, and it is the one
    # whose formula can produce NaN from a degenerate motion vector.
    position_angle = optical.position_angle_deg.to_numpy(zero_copy_only=False)
    assert np.isfinite(position_angle).all(), f"position_angle_deg: {position_angle}"
    assert ((position_angle > -180.0) & (position_angle <= 180.0)).all(), position_angle


@pytest.mark.validation
def test_radar_index_ranks_the_radar_candidates_by_epoch(plan):
    """The radar half of the ``index`` cross-link contract.

    A radar row carries no epoch, so ``index`` is the only key back to
    the input — and it numbers the radar candidates in epoch order, not
    submission order.
    """
    kinds = plan.candidates.kind.to_pylist()
    indices = plan.candidates.index.to_pylist()
    radar_indices = [i for i, k in zip(indices, kinds, strict=True) if k == "radar"]
    n_radar = len(radar_indices)
    assert n_radar >= 2, "fixture must contain at least two radar candidates"
    assert sorted(radar_indices) == list(range(n_radar)), radar_indices


@pytest.mark.validation
def test_a_radar_only_plan_returns_an_empty_ephemeris():
    """The documented radar-only shape, and the null-ephemeris marshal
    branch that no other fixture reaches."""
    plan = evaluate_plan(
        _plan_orbit(),
        [
            PlannedObservation.radar(
                _EPOCH_MJD_TDB + 15.0,
                radar_bandwidth_hz=1.0e5,
                radar_freq_resolution_hz=0.1,
                radar_snr=50.0,
            ),
            PlannedObservation.radar(
                _EPOCH_MJD_TDB + 18.0,
                radar_bandwidth_hz=1.0e5,
                radar_freq_resolution_hz=0.1,
                radar_snr=40.0,
            ),
        ],
    )
    assert len(plan.candidates) == 2, len(plan.candidates)
    assert len(plan.ephemeris) == 0, "a radar-only plan has no sky-plane prediction"
    for column in (
        "along_track_sigma_arcsec",
        "cross_track_sigma_arcsec",
        "ra_sigma_arcsec",
        "dec_sigma_arcsec",
        "position_angle_deg",
        "post_along_track_sigma_arcsec",
        "post_cross_track_sigma_arcsec",
    ):
        assert plan.candidates.column(column).null_count == 2, column


@pytest.mark.validation
def test_a_singular_prior_covariance_is_refused():
    """The documented ``RuntimeError`` for a prior that cannot be
    inverted — the guard that replaced a silent fall-back to the prior."""
    orbit = _plan_orbit()
    cov = np.zeros((1, 6, 6))
    # Rank-deficient: the last state direction carries no information, so
    # the information matrix does not exist.
    for k, d in enumerate([1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 0.0]):
        cov[0, k, k] = d
    singular = CartesianOrbits.from_kwargs(
        orbit_id=orbit.orbit_id,
        object_id=orbit.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=orbit.coordinates.epoch,
            x=orbit.coordinates.x,
            y=orbit.coordinates.y,
            z=orbit.coordinates.z,
            vx=orbit.coordinates.vx,
            vy=orbit.coordinates.vy,
            vz=orbit.coordinates.vz,
            frame="ecliptic_j2000",
            origin=["SSB"],
            covariance=CartesianCovariance.from_matrix(cov),
        ),
    )
    with pytest.raises(RuntimeError):
        evaluate_plan(singular, _full_plan())


def test_evaluate_plan_rejects_a_multi_row_orbit():
    """The message explains the covariance-as-prior semantics, so it is
    the one a batch-minded caller hits first."""
    import quivr as qv

    two = qv.concatenate([_plan_orbit(), _plan_orbit()])
    with pytest.raises(ValueError, match="exactly one orbit"):
        evaluate_plan(two, _full_plan())


# ── Candidate validation ─────────────────────────────────────────────


def test_link_budget_fields_are_refused_alongside_a_supplied_snr():
    """Supplying an SNR selects a different request shape; the
    link-budget inputs have nowhere to go in it."""
    with pytest.raises(ValueError) as exc:
        PlannedObservation.radar(
            61000.0,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_snr=50.0,
            radar_target_h_mag=19.7,
            radar_integration_s=600.0,
        )
    message = str(exc.value)
    assert "radar_target_h_mag" in message, message
    assert "radar_integration_s" in message, message
    assert "radar_snr=None" in message, f"the refusal must name the fix: {message}"


def test_a_waveform_the_mode_needs_must_be_positive():
    with pytest.raises(ValueError, match="radar_bandwidth_hz"):
        PlannedObservation.radar(61000.0, radar_mode=RadarMode.DELAY, radar_snr=50.0)
    with pytest.raises(ValueError, match="radar_freq_resolution_hz"):
        PlannedObservation.radar(
            61000.0, radar_mode=RadarMode.DOPPLER, radar_bandwidth_hz=1.0e5, radar_snr=50.0
        )
    # A delay-only candidate does not need a Doppler resolution.
    PlannedObservation.radar(
        61000.0, radar_mode=RadarMode.DELAY, radar_bandwidth_hz=1.0e5, radar_snr=50.0
    )


def test_green_bank_cannot_transmit():
    """Receive-only: zero transmit power. The link-budget path fails
    engine-side on the resulting SNR, but a supplied SNR skips the link
    budget entirely."""
    with pytest.raises(ValueError, match="receive-only"):
        PlannedObservation.radar(
            61000.0,
            radar_transmit_station=RadarStation.GREEN_BANK,
            radar_bandwidth_hz=1.0e5,
            radar_freq_resolution_hz=0.1,
            radar_snr=50.0,
        )


def test_an_out_of_domain_kind_is_a_valueerror_not_a_keyerror():
    """An unrecognized ``kind`` used to match neither validation arm, so
    the object was built unvalidated and failed much later in the wire
    lowering as a ``KeyError``."""
    with pytest.raises(ValueError, match="kind"):
        PlannedObservation(epoch_mjd_tdb=61000.0, kind="infrared")  # type: ignore[arg-type]


def test_candidate_fields_set_after_construction_are_still_refused():
    """``PlannedObservation`` is mutable like ``PlanningConfig``, so
    ``evaluate_plan`` re-runs its validation for the same reason."""
    mutated = PlannedObservation.optical(_EPOCH_MJD_TDB + 10.0, "F51", (0.2, 0.2))
    mutated.optical_sigma_arcsec = (0.0, 0.2)
    with pytest.raises(ValueError, match="optical_sigma_arcsec"):
        evaluate_plan(_plan_orbit(), [mutated])


def test_force_model_accepts_the_enum_and_a_bare_string_and_refuses_junk():
    from empyrean import ForceModelTier

    assert PlanningConfig(force_model="basic")._to_wire_dict()["force_model"] == "basic"
    assert (
        PlanningConfig(force_model=ForceModelTier.APPROXIMATE)._to_wire_dict()["force_model"]
        == "approximate"
    )
    with pytest.raises(ValueError, match="force_model"):
        PlanningConfig(force_model="full")


def test_planned_columns_map_covers_the_whole_wire_dict():
    """The transposition map is the layer where a new wire field would be
    dropped for the whole batch — the dataclass-side reflection test
    above cannot see it."""
    from empyrean.planning.plan import _PLANNED_COLUMNS

    wire = PlannedObservation.optical(61000.0, "F51")._to_wire_dict()
    assert set(wire) == set(_PLANNED_COLUMNS), (
        f"_PLANNED_COLUMNS and _to_wire_dict disagree:\n"
        f"  only in wire dict: {sorted(set(wire) - set(_PLANNED_COLUMNS))}\n"
        f"  only in the map:   {sorted(set(_PLANNED_COLUMNS) - set(wire))}"
    )


# ── Input basis ──────────────────────────────────────────────────────


def _reexpress(orbit, target_type, **kwargs):
    """Re-express the fixture orbit in another representation, keeping
    its covariance."""
    from empyrean import transform_coordinates

    return transform_coordinates(orbit.coordinates, target_type, **kwargs)


@pytest.mark.parametrize("target", ["cometary", "keplerian"])
def test_evaluate_plan_refuses_a_non_cartesian_orbit(target):
    """The covariance is consumed as a Cartesian state prior and is never
    converted.

    Nothing below this guard inspects the representation: the engine tags
    a covariance by representation and then reads indices 0-2 as AU
    positions and 3-5 as AU/day velocities regardless. A cometary
    covariance therefore produced plausible finite numbers that meant
    nothing — the worst failure shape, since no exception ever surfaced.
    """
    from empyrean import CometaryOrbits, KeplerianOrbits
    from empyrean.coordinates.coordinates import (
        CometaryCoordinates,
        KeplerianCoordinates,
    )

    cartesian = _plan_orbit()
    if target == "cometary":
        coords = _reexpress(cartesian, CometaryCoordinates)
        orbit = CometaryOrbits.from_kwargs(
            orbit_id=cartesian.orbit_id, object_id=cartesian.object_id, coordinates=coords
        )
    else:
        coords = _reexpress(cartesian, KeplerianCoordinates)
        orbit = KeplerianOrbits.from_kwargs(
            orbit_id=cartesian.orbit_id, object_id=cartesian.object_id, coordinates=coords
        )

    assert coords.covariance is not None, "fixture must keep its covariance"
    with pytest.raises(ValueError) as exc:
        evaluate_plan(orbit, _full_plan())
    message = str(exc.value)
    assert "Cartesian" in message, message
    assert "transform_coordinates" in message, f"the refusal must name the fix: {message}"


def test_both_basis_guards_name_the_same_one_call_conversion():
    """A caller whose orbit is wrong on both axes should not have to
    discover the fix twice."""
    from empyrean.planning.plan import _CONVERSION_HINT

    assert "CartesianCoordinates" in _CONVERSION_HINT
    assert "SSB" in _CONVERSION_HINT

    helio = _plan_orbit()
    coords = helio.coordinates
    helio = CartesianOrbits.from_kwargs(
        orbit_id=helio.orbit_id,
        object_id=helio.object_id,
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=coords.epoch,
            x=coords.x,
            y=coords.y,
            z=coords.z,
            vx=coords.vx,
            vy=coords.vy,
            vz=coords.vz,
            frame="ecliptic_j2000",
            origin=["Sun"],
            covariance=coords.covariance,
        ),
    )
    with pytest.raises(ValueError) as exc:
        evaluate_plan(helio, _full_plan())
    assert _CONVERSION_HINT in str(exc.value), str(exc.value)


@pytest.mark.validation
def test_a_spin_capped_link_budget_records_the_cap():
    """A *fully specified* link budget still emits a provenance note when
    a known spin period caps the requested integration — the case the
    docs used to describe as note-free."""
    plan = evaluate_plan(
        _plan_orbit(),
        [
            PlannedObservation.radar(
                _EPOCH_MJD_TDB + 15.0,
                radar_bandwidth_hz=1.0e5,
                radar_freq_resolution_hz=0.1,
                radar_target_h_mag=19.7,
                radar_target_visual_albedo=0.23,
                radar_target_radar_albedo=0.15,
                radar_target_diameter_km=0.34,
                # Every input given, and an integration far past the
                # speckle-decorrelation time of a 30.6 h rotator.
                radar_target_spin_period_hours=30.6,
                radar_integration_s=1.0e6,
            )
        ],
    )
    notes = plan.candidates.radar_provenance.to_pylist()[0]
    assert notes, "a capped integration must be recorded"
    assert any("spin" in n for n in notes), notes
