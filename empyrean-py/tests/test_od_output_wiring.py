"""OD-output wiring contract.

The fitted orbit that ``determine`` / ``refine`` return must carry two pieces
of ancillary data onto the orbit itself — not only in a side-channel — so the
downstream propagate / ephemeris surfaces keep working:

* **Identity** — the ``determine`` / ``refine`` C ABI hardcodes an empty
  ``orbit_id`` / ``object_id`` on the fit, so without a re-attach the fitted
  orbit carries no identity and it is lost for every downstream step.
* **Photometry** — the post-OD H/G fit is returned only in
  ``DetermineResult.photometry``; when it is not attached to the orbit's
  ``photometric`` column, ``generate_ephemeris`` from the fitted orbit silently
  yields ``mag=None``.

Both shipped without acceptance coverage: ``test_id_propagation`` never
exercised an OD *output*, and no test ran a real fit through
``generate_ephemeris`` and checked the magnitude was populated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from empyrean import Epochs, determine, generate_ephemeris, read_ades
from empyrean.observers.observers import Observers
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.determine import _seed_labels
from empyrean.od.result import ODConfig, PhotometryConfig

DATA_DIR = Path(__file__).parent / "fixtures"
APOPHIS_MULTIAPP = DATA_DIR / "99942_apophis_multiapp.psv"


@pytest.fixture(scope="module")
def apophis_observations():
    if not APOPHIS_MULTIAPP.exists():
        pytest.skip(f"missing fixture: {APOPHIS_MULTIAPP}")
    optical, _radar = read_ades(APOPHIS_MULTIAPP)
    return optical


def test_determine_fitted_orbit_carries_identity(apophis_observations):
    """An unseeded fit derives its object id from the observations."""
    fit = determine(apophis_observations).single()
    assert fit.converged
    object_id = fit.orbit.object_id.to_pylist()
    orbit_id = fit.orbit.orbit_id.to_pylist()
    assert object_id and object_id[0], f"fitted orbit object_id dropped: {object_id!r}"
    assert orbit_id and orbit_id[0], f"fitted orbit orbit_id dropped: {orbit_id!r}"


def test_determine_seeded_inherits_seed_identity(apophis_observations):
    """A seeded fit inherits the seed's object id (the initial_orbits key)."""
    seed = determine(apophis_observations).single().orbit
    fit = determine(apophis_observations, initial_orbits={"my-apophis": seed}).single()
    assert fit.orbit.object_id.to_pylist()[0] == "my-apophis"


def _ades_rows(ids: list[tuple[str | None, str | None, str | None]]) -> ADESObservations:
    """Minimal ADES table carrying only the three identifier columns."""
    n = len(ids)
    return ADESObservations.from_kwargs(
        perm_id=[p for p, _, _ in ids],
        prov_id=[v for _, v, _ in ids],
        trk_sub=[t for _, _, t in ids],
        stn=["500"] * n,
        obs_time=["2024-01-01T00:00:00Z"] * n,
        ra=[0.0] * n,
        dec=[0.0] * n,
    )


def test_seed_labels_pair_positionally_with_first_appearance_groups():
    """The seed-key pairing that carries a caller's id onto the fit.

    ``initial_orbits`` is keyed in Python but crosses the C ABI as a bare
    array, so ``empyrean_determine`` pairs the i-th seed with the i-th
    unique ADES object id in first-appearance order. ``_seed_labels``
    reconstructs that pairing; if it drifts from the engine's rule a
    seeded batch fit gets labelled with the *wrong* caller's id, which is
    worse than not relabelling at all. Pin the rule directly — the
    Apophis contract above only covers the one-object, one-seed case.
    """
    observations = _ades_rows(
        [
            ("99942", "2004 MN4", "4X4E25A"),  # permID wins
            ("", "2024 YR4", "K24Y04R"),  # empty permID falls through to provID
            ("99942", None, None),  # repeat: must not open a new group
            (None, None, "TRK-ONLY"),  # trkSub is the last resort
            (None, None, None),  # no identifier at all
        ]
    )
    groups = ["99942", "2024 YR4", "TRK-ONLY", "unknown"]
    all_labelled = dict(zip(groups, ["a", "b", "c", "d"], strict=True))

    # One seed labels the first group only; the rest keep their ADES id.
    assert _seed_labels(observations, ["my-apophis"]) == {"99942": "my-apophis"}

    # Seeds pair in order, across the precedence and the repeat.
    assert _seed_labels(observations, ["a", "b", "c", "d"]) == all_labelled

    # A seed beyond the last group has nothing to label (the engine reports
    # it in `unmatched_orbit_ids`); it must not shift the others.
    assert _seed_labels(observations, ["a", "b", "c", "d", "e"]) == all_labelled


def test_fitted_orbit_predicts_magnitudes(apophis_observations):
    """The post-OD photometric fit is attached to the fitted orbit, so an
    ephemeris generated from that orbit predicts real (non-null) magnitudes."""
    fit = determine(apophis_observations, config=ODConfig(photometry=PhotometryConfig())).single()
    assert fit.converged
    assert fit.photometry is not None, "photometric fit did not run"

    h = fit.orbit.photometric.h.to_pylist()
    assert h and h[0] is not None, "fitted H was not attached to the orbit"

    epoch0 = fit.orbit.coordinates.epoch.to_pylist()[0]
    epochs = Epochs.from_mjd([epoch0 + 1.0, epoch0 + 10.0], scale="tdb")
    eph = generate_ephemeris(fit.orbit, Observers.from_code("500", epochs)).ephemeris
    mags = eph.mag.to_pylist()
    assert all(m is not None for m in mags), f"fitted-orbit ephemeris mag was None: {mags}"
    assert all(0.0 < m < 40.0 for m in mags), f"unphysical magnitude: {mags}"


_SOLVER_STOPS = {
    "gradient_tolerance",
    "step_tolerance",
    "cost_tolerance",
    "max_iterations",
    "damping_exhausted",
    "inner_trials_exhausted",
    "stalled_delivered",
    "schur_step_tolerance",
    "unrecognized",
}


def test_fit_reports_how_its_solver_stopped(apophis_observations):
    """The solver-termination block survives the whole chain — engine → C ABI
    → Rust wrapper → pyo3 dict → dataclass — with its absent-value convention
    intact.

    Before it existed, ``update_norm`` was the only convergence-adjacent number
    a Python caller had, and reading it as one is wrong in both directions: it
    is the mu-DAMPED accepted step, while the tolerance is tested on the
    undamped Gauss-Newton step (``gn_step_qnorm``)."""
    fit = determine(apophis_observations).single()
    assert fit.converged

    assert fit.termination in _SOLVER_STOPS, f"unknown stop tag: {fit.termination!r}"
    assert fit.termination != "unrecognized", (
        "an unrecognized stop means the bindings' mapping has fallen behind the solver's own enum"
    )

    # Absent readings are None, never 0.0 / NaN standing in for absence.
    assert fit.gn_step_qnorm is None or fit.gn_step_qnorm == fit.gn_step_qnorm
    if fit.termination == "step_tolerance":
        assert fit.gn_step_qnorm is not None, (
            "a step_tolerance stop is DECIDED on the undamped step, so that step must reach Python"
        )
    # mu is absent exactly on the Schur path, which forms no comparable damping.
    assert (fit.mu_final is None) == (fit.termination == "schur_step_tolerance")

    assert fit.final_solve_iterations is not None
    assert fit.final_solve_iterations <= fit.iterations, (
        "the final solve's iterations cannot exceed the total across every solve the path ran"
    )

    # accepted_steps disambiguates update_norm's 0.0 sentinel: determine
    # converges, transfers to the mid-arc epoch and re-solves, so its final
    # solve can legitimately latch at its starting point without moving.
    if fit.accepted_steps == 0:
        assert fit.update_norm == 0.0
    else:
        assert fit.update_norm == fit.update_norm  # not NaN

    # This arc converges on a solver criterion, so there is no stall block.
    if fit.stall_delivery is None:
        assert fit.termination != "stalled_delivered"
    else:
        assert fit.termination == "stalled_delivered"
        assert fit.stall_delivery.underlying_stop in {
            "damping_exhausted",
            "inner_trials_exhausted",
        }
        assert fit.stall_delivery.step_sigmas >= 0.0
