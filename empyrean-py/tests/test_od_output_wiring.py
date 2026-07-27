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
from empyrean import determine, generate_ephemeris, read_ades
from empyrean.observers.observers import Observers
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
    fit = determine(apophis_observations)
    assert fit.converged
    object_id = fit.orbit.object_id.to_pylist()
    orbit_id = fit.orbit.orbit_id.to_pylist()
    assert object_id and object_id[0], f"fitted orbit object_id dropped: {object_id!r}"
    assert orbit_id and orbit_id[0], f"fitted orbit orbit_id dropped: {orbit_id!r}"


def test_determine_seeded_inherits_seed_identity(apophis_observations):
    """A seeded fit inherits the seed's object id (the initial_orbits key)."""
    seed = determine(apophis_observations).orbit
    fit = determine(apophis_observations, initial_orbits={"my-apophis": seed})
    assert fit.orbit.object_id.to_pylist()[0] == "my-apophis"


def test_fitted_orbit_predicts_magnitudes(apophis_observations):
    """The post-OD photometric fit is attached to the fitted orbit, so an
    ephemeris generated from that orbit predicts real (non-null) magnitudes."""
    fit = determine(apophis_observations, config=ODConfig(photometry=PhotometryConfig()))
    assert fit.converged
    assert fit.photometry is not None, "photometric fit did not run"

    h = fit.orbit.photometric.h.to_pylist()
    assert h and h[0] is not None, "fitted H was not attached to the orbit"

    epoch0 = fit.orbit.coordinates.epoch.to_pylist()[0]
    epochs = [epoch0 + 1.0, epoch0 + 10.0]
    eph = generate_ephemeris(fit.orbit, Observers.from_code("500", epochs)).ephemeris
    mags = eph.mag.to_pylist()
    assert all(m is not None for m in mags), f"fitted-orbit ephemeris mag was None: {mags}"
    assert all(0.0 < m < 40.0 for m in mags), f"unphysical magnitude: {mags}"
