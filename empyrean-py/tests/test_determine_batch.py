"""``determine`` is batch-first: every input object is accounted for.

The assembly under test is pure Python — it turns the Rust batch dict
into the three tables — so these run without touching the engine. The
session-scoped ``initialize_empyrean`` fixture in ``conftest.py`` is
``autouse``, so a machine without kernels will *skip* rather than run
them; the logic itself needs neither kernels nor the extension module.

The multi-object end-to-end path (real astrometry through the engine) is
marked ``validation`` and runs at the validation gate.
"""

import math

import numpy as np
import pytest
from empyrean.od.determine import (
    _build_determine_results,
    _summary_row_delivered,
    _summary_row_failed,
)
from empyrean.od.residuals import FitSummary
from empyrean.od.result import DetermineFailure

# The fit_summary schema, in order. This is the same column set and order
# the CLI writes to fit_summary.parquet / fit_summary.csv and the C ABI's
# EmpyreanFitSummary declares — a table read back off disk and this one
# must describe a fit identically.
EXPECTED_COLUMNS = [
    "object_id",
    "status",
    "converged",
    "iterations",
    "n_obs",
    "n_selected",
    "rms_ra_arcsec",
    "rms_dec_arcsec",
    "reduced_chi2",
    "fit_acceptable",
    "extrapolation_acceptable",
    "selection_fraction_ok",
    "selection_fraction",
    "selection_fraction_threshold",
    "selected_arc_coverage_ok",
    "selected_arc_days",
    "selected_arc_fraction",
    "selected_arc_fraction_threshold",
    "trailing_gap_ok",
    "trailing_gap_days",
    "trailing_gap_threshold_days",
    "fractional_sigma_a_ok",
    "fractional_sigma_a",
    "fractional_sigma_a_threshold",
    "solve_for_width",
    "error",
]

# Columns whose absence must read as NaN, never as a measurement at zero.
NAN_ON_FAILURE = [
    "rms_ra_arcsec",
    "rms_dec_arcsec",
    "reduced_chi2",
    "selection_fraction",
    "selection_fraction_threshold",
    "selected_arc_days",
    "selected_arc_fraction",
    "selected_arc_fraction_threshold",
    "trailing_gap_days",
    "trailing_gap_threshold_days",
    "fractional_sigma_a",
    "fractional_sigma_a_threshold",
]


def failed_entry(object_id, message="IOD failed: no viable seed", kind="iod"):
    """A Rust batch entry for an object that produced no orbit."""
    return {
        "object_id": object_id,
        "delivered": False,
        "error": message,
        "error_kind": kind,
    }


def test_fit_summary_schema_matches_the_written_table():
    """The quivr schema is the documented one, in order."""
    columns = [f.name for f in FitSummary.empty().table.schema]
    assert columns == EXPECTED_COLUMNS


def test_fit_summary_carries_all_four_extrapolation_gate_axes():
    """Each gate crosses as its own ok/value/threshold triple, so a
    caller can report *why* a fit is not extrapolable, not merely that it
    is not."""
    columns = {f.name for f in FitSummary.empty().table.schema}
    for axis, value, threshold in [
        ("selection_fraction_ok", "selection_fraction", "selection_fraction_threshold"),
        (
            "selected_arc_coverage_ok",
            "selected_arc_fraction",
            "selected_arc_fraction_threshold",
        ),
        ("trailing_gap_ok", "trailing_gap_days", "trailing_gap_threshold_days"),
        (
            "fractional_sigma_a_ok",
            "fractional_sigma_a",
            "fractional_sigma_a_threshold",
        ),
    ]:
        assert {axis, value, threshold} <= columns, axis
    # The selected-arc span itself, alongside its ratio.
    assert "selected_arc_days" in columns


def test_a_failed_object_gets_a_row_with_nan_measurements():
    """A failed fit has no RMS and no gate values. Zero would read as a
    measurement at the floor, so every absent number is NaN."""
    row = _summary_row_failed("K25A00B", DetermineFailure("K25A00B", "boom", "iod"))

    assert row["object_id"] == "K25A00B"
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    assert row["converged"] is False
    assert row["iterations"] == 0
    assert row["n_obs"] == 0
    assert row["n_selected"] == 0
    assert row["solve_for_width"] == 0
    for name in NAN_ON_FAILURE:
        assert math.isnan(row[name]), f"{name} must be NaN on a failed object"
    # No gate can pass without a fit.
    for name in (
        "fit_acceptable",
        "extrapolation_acceptable",
        "selection_fraction_ok",
        "selected_arc_coverage_ok",
        "trailing_gap_ok",
        "fractional_sigma_a_ok",
    ):
        assert row[name] is False, name
    # Every schema column is present — a row is never partially built.
    assert set(row) == set(EXPECTED_COLUMNS)


def test_summary_row_covers_every_schema_column_for_a_delivered_fit():
    """The delivered row builder fills the same column set as the failed
    one, so the two never produce ragged tables."""

    class _Acceptability:
        fit_acceptable = True
        extrapolation_acceptable = False
        selection_fraction_ok = True
        selection_fraction_value = 0.95
        selection_fraction_threshold = 0.7
        selected_arc_coverage_ok = True
        selected_arc_days_value = 880.0
        selected_arc_fraction_value = 0.98
        selected_arc_fraction_threshold = 0.8
        trailing_gap_ok = False
        trailing_gap_days_value = 41.0
        trailing_gap_threshold_days = 30.0
        trailing_gap_threshold = 30.0
        fractional_sigma_a_ok = True
        fractional_sigma_a_value = 1.0e-6
        fractional_sigma_a_threshold = 1.0e-3

    class _Summary:
        num_obs = 42
        num_selected = 40
        rms_ra_arcsec = 0.31
        rms_dec_arcsec = 0.28
        reduced_chi2 = 1.12

    class _Fit:
        converged = True
        iterations = 5
        summary = _Summary()
        acceptability = _Acceptability()
        solved_covariance = None

    row = _summary_row_delivered("2024 YR4", _Fit())
    assert set(row) == set(EXPECTED_COLUMNS)
    assert row["status"] == "delivered"
    assert row["n_selected"] == 40
    # A state-only fit has no tagged solved covariance; its width is the
    # 6-element state, not 0.
    assert row["solve_for_width"] == 6
    # The gate that failed is reported as failed, with its measurement.
    assert row["trailing_gap_ok"] is False
    assert row["trailing_gap_days"] == 41.0
    assert row["extrapolation_acceptable"] is False


def test_every_input_object_appears_even_when_all_of_them_fail():
    """The whole point of the batch surface: N failures produce N rows,
    not an empty result."""
    batch = {
        "objects": [failed_entry("SYNTH01"), failed_entry("SYNTH02", kind="radar_only")],
        "unmatched_orbit_ids": [],
    }
    results = _build_determine_results(batch)

    assert len(results) == 2
    assert results.object_ids == ["SYNTH01", "SYNTH02"]
    assert results.delivered == []
    assert results.all_failed is True
    assert len(results.orbits) == 0
    assert len(results.residuals) == 0
    assert len(results.summary) == 2
    assert results.summary.status.to_pylist() == ["failed", "failed"]
    # The classified cause is available without parsing the message.
    assert results.failures["SYNTH02"].kind == "radar_only"


def test_indexing_a_failed_object_raises_with_the_reason():
    """A failed object must never return an empty or default result."""
    results = _build_determine_results(
        {"objects": [failed_entry("SYNTH01", "no viable seed")], "unmatched_orbit_ids": []}
    )

    assert "SYNTH01" in results
    with pytest.raises(ValueError, match="no viable seed"):
        results["SYNTH01"]
    with pytest.raises(KeyError, match="1997 XF11"):
        results["1997 XF11"]


def test_single_refuses_to_choose_among_several_objects():
    """`single()` exists for the one-object call; picking one of many is
    exactly the silent discard this surface replaced."""
    results = _build_determine_results(
        {
            "objects": [failed_entry("SYNTH01"), failed_entry("SYNTH02")],
            "unmatched_orbit_ids": [],
        }
    )
    with pytest.raises(ValueError) as excinfo:
        results.single()
    message = str(excinfo.value)
    assert "2 objects" in message
    # Both identities are named so the caller can index instead.
    assert "SYNTH01" in message and "SYNTH02" in message


def test_single_surfaces_the_failure_of_a_lone_failed_object():
    results = _build_determine_results(
        {"objects": [failed_entry("SYNTH01", "boom")], "unmatched_orbit_ids": []}
    )
    with pytest.raises(ValueError, match="boom"):
        results.single()


def test_an_empty_batch_reports_itself_as_empty():
    results = _build_determine_results({"objects": [], "unmatched_orbit_ids": []})
    assert len(results) == 0
    assert results.object_ids == []
    # Nothing attempted is not "everything failed".
    assert results.all_failed is False
    with pytest.raises(ValueError, match="no objects"):
        results.single()


def test_unmatched_seed_orbits_are_reported_not_dropped():
    """A seed whose identity matches no observation group constrained
    nothing; saying so is the difference between a no-op and a silent
    one."""
    results = _build_determine_results(
        {"objects": [failed_entry("SYNTH01")], "unmatched_orbit_ids": ["1997 XF11"]}
    )
    assert results.unmatched_orbit_ids == ["1997 XF11"]


def test_summary_table_types_survive_the_column_orientation():
    """The integer columns are Int32 in the schema; building the table
    from Python ints must not widen or narrow them."""
    results = _build_determine_results(
        {"objects": [failed_entry("SYNTH01")], "unmatched_orbit_ids": []}
    )
    summary = results.summary
    assert summary.iterations.to_pylist() == [0]
    assert summary.n_obs.to_pylist() == [0]
    assert summary.solve_for_width.to_pylist() == [0]
    # NaN survives as NaN, not as null-coerced zero.
    assert np.isnan(summary.reduced_chi2.to_numpy(zero_copy_only=False)[0])


@pytest.mark.validation
def test_multi_object_determine_end_to_end(tmp_path):
    """Two objects through the real engine: both appear, and the fits do
    not contaminate each other.

    Needs kernels and the compiled extension; runs at the validation
    gate. The fixture is deliberately short-arc so it exercises the
    per-object failure path without a catalog object.
    """
    import empyrean

    psv = (
        "# version=2022\n"
        "permID|provID|trkSub|mode|stn|obsTime|ra|dec|rmsRA|rmsDec|astCat\n"
        "|SYNTH01||CCD|703|2024-01-10T08:00:00.000Z|120.000000|15.000000|0.5|0.5|Gaia2\n"
        "|SYNTH01||CCD|703|2024-01-10T09:00:00.000Z|120.010000|15.004000|0.5|0.5|Gaia2\n"
        "|SYNTH02||CCD|703|2024-01-10T08:30:00.000Z|200.000000|-8.000000|0.5|0.5|Gaia2\n"
        "|SYNTH02||CCD|703|2024-01-10T09:30:00.000Z|200.020000|-8.006000|0.5|0.5|Gaia2\n"
    )
    optical, _radar = empyrean.read_ades(psv)
    results = empyrean.determine(optical)

    assert len(results) == 2, "both objects must be accounted for"
    assert sorted(results.object_ids) == ["SYNTH01", "SYNTH02"]
    assert len(results.summary) == 2
    # Whatever the engine decides, no object is silently absent.
    for object_id in results.object_ids:
        assert object_id in results
    # Residual rows, if any, are attributable to their object.
    if len(results.residuals) > 0:
        assert set(results.residuals.object_id.to_pylist()) <= set(results.object_ids)


def test_determine_refuses_an_id_less_row_by_name():
    """scott #126 at the Python surface: an observation row carrying no
    object identifier (``permID`` / ``provID`` / ``trkSub`` all null)
    refuses the WHOLE batch by name — a ``RuntimeError`` whose message names
    the offending row indices — rather than silently skipping the row and
    reporting a complete-looking batch (the no-hidden-fallbacks failure mode
    the old single-fit path had).

    The row indices reach the message so the caller can repair the exact
    rows. Mutation guard: dropping the offending row list from the refusal
    (the C ABI's ``set_last_error(&e.to_string())`` that surfaces the engine
    message) empties the ``optical rows: [3]`` match and this fails.
    """
    import empyrean
    from empyrean.od.ades_observations import ADESObservations

    # Three attributable rows for one object, then one row carrying NO
    # identifier at index 3. A single id-less row aborts the batch by name.
    ids = [
        ("", "SYNTH01", ""),
        ("", "SYNTH01", ""),
        ("", "SYNTH01", ""),
        (None, None, None),
    ]
    n = len(ids)
    obs = ADESObservations.from_kwargs(
        perm_id=[p for p, _, _ in ids],
        prov_id=[v for _, v, _ in ids],
        trk_sub=[t for _, _, t in ids],
        stn=["500"] * n,
        obs_time=["2024-01-01T00:00:00Z"] * n,
        ra=[0.0] * n,
        dec=[0.0] * n,
    )
    with pytest.raises(RuntimeError, match=r"no object identifier"):
        empyrean.determine(obs)
    # The index of the offending row is named, so the caller can repair it.
    with pytest.raises(RuntimeError, match=r"optical rows: \[3\]"):
        empyrean.determine(obs)


def test_unmatched_seed_orbit_reaches_python_from_the_engine():
    """The ``unmatched`` channel survives the C ABI → Python marshaling.

    ``test_unmatched_seed_orbits_are_reported_not_dropped`` above pins the
    dict-assembly reader; this pins the other half — that ``_determine``
    actually populates ``unmatched_orbit_ids`` from a real engine batch.
    Seeds pair positionally with observation groups in first-appearance
    order, so a seed beyond the last group is attributed to no object and
    reported here rather than dropped. One group + two seeds leaves the
    second seed unmatched.

    Mutation guard: dropping the ``unmatched_orbit_ids`` ``set_item`` in the
    PyO3 ``_determine`` empties the list and this fails.
    """
    import empyrean
    from empyrean.coordinates.coordinates import CartesianCoordinates
    from empyrean.orbits.orbits import CartesianOrbits

    psv = (
        "# version=2022\n"
        "permID|provID|trkSub|mode|stn|obsTime|ra|dec|rmsRA|rmsDec|astCat\n"
        "|SYNTH01||CCD|703|2024-01-10T08:00:00.000Z|120.000000|15.000000|0.5|0.5|Gaia2\n"
        "|SYNTH01||CCD|703|2024-01-10T09:00:00.000Z|120.010000|15.004000|0.5|0.5|Gaia2\n"
    )
    optical, _radar = empyrean.read_ades(psv)

    def seed(orbit_id: str) -> CartesianOrbits:
        coords = CartesianCoordinates.from_kwargs(
            epoch=[60000.0],
            x=[2.0],
            y=[0.0],
            z=[0.0],
            vx=[0.0],
            vy=[0.01217],
            vz=[0.0],
            frame="eclipticj2000",
            origin=["Sun"],
        )
        return CartesianOrbits.from_kwargs(orbit_id=[orbit_id], coordinates=coords)

    results = empyrean.determine(
        optical, initial_orbits={"REALSEED": seed("REALSEED"), "GHOST": seed("GHOST")}
    )
    # The second seed pairs with no group; the engine reports it by its
    # positional C-ABI orbit id ("orbit_1"), carried across as a Python list.
    assert isinstance(results.unmatched_orbit_ids, list)
    assert len(results.unmatched_orbit_ids) == 1, results.unmatched_orbit_ids
    assert results.unmatched_orbit_ids == ["orbit_1"], results.unmatched_orbit_ids


def test_residual_marshaling_carries_every_observation_column():
    """The Python residual writers must hand the Rust side the whole
    table.

    Passing a projection would produce a file with all the columns and
    content in only five of them — a worse silent drop than the short
    schema it replaced, because the header would promise the rest.
    """
    from empyrean.io.residuals import _residuals_to_dict
    from empyrean.od.residuals import ObservationResults

    table = ObservationResults.empty()
    wire = _residuals_to_dict(table)

    # Every column of the table reaches the boundary. The wire keys are
    # the plural forms the Rust side reads; map them back by stripping
    # the pluralization the boundary uses.
    covered = {
        "obs_id": "obs_ids",
        "object_id": "object_ids",
        "obs_code": "obs_codes",
        "ast_cat": "ast_cats",
        "epoch_mjd_tdb": "epochs",
        "ra_residual": "ra_residuals_arcsec",
        "dec_residual": "dec_residuals_arcsec",
        "chi2": "chi2",
        "dof": "dofs",
        "probability": "probability",
        "selected": "selected",
        "residual_cov_ra": "residual_cov_ras",
        "residual_cov_dec": "residual_cov_decs",
        "residual_cov_corr": "residual_cov_corrs",
        "rejection_reason": "rejection_reasons",
        "rejection_criterion": "rejection_criterions",
        "rejection_threshold": "rejection_thresholds",
        "rejection_effective_threshold": "rejection_effective_thresholds",
        "rejection_information_loss": "rejection_information_losses",
        "cooks_distance": "cooks_distances",
        "leverage": "leverages",
        "fractional_information": "fractional_informations",
        "along_track": "along_tracks",
        "cross_track": "cross_tracks",
        "along_track_error": "along_track_errors",
        "cross_track_error": "cross_track_errors",
        "track_position_angle_deg": "track_position_angles",
        "influence_information_loss": "influence_information_losses",
        "along_cross_covariance_arcsec2": "along_cross_covariances",
        "radar_kind": "radar_kinds",
        "radar_residual": "radar_residuals",
        "radar_chi2": "radar_chi2s",
        "radar_probability": "radar_probabilities",
        "radar_variance": "radar_variances",
        "radar_dof": "radar_dofs",
    }

    table_columns = {f.name for f in table.table.schema}
    missing = table_columns - set(covered)
    assert not missing, f"columns with no wire mapping: {sorted(missing)}"

    for column, key in covered.items():
        assert column in table_columns, f"{column} is not an ObservationResults column"
        assert key in wire, f"{column} does not reach the writer (wire key {key!r})"
