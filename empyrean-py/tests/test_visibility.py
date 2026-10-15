"""`empyrean.compute_visibility` — the night product, anchors and Moon
circumstances through Python.

The engine runs under the session-autouse context fixture (skips without
kernels). These assert the table shapes, the four-gate ``observable``
wiring (the site-aware verdict plan evaluation cannot give), the Moon
columns, the anchor sky covariance, and the twilight-threshold validation.
"""

from __future__ import annotations

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianOrbits,
    ObservatoryConfig,
    PlanningConfig,
    compute_visibility,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import CartesianCovariance
from empyrean.orbits.photometry import PhotometricParams

_T0 = 61000.0
_APOPHIS_SSB_ECLIPTIC = {
    "x": -8.18903154079353157e-02,
    "y": -8.25280676309557037e-01,
    "z": 4.20312157909258355e-02,
    "vx": 1.98823903215481629e-02,
    "vy": 1.32186530066730917e-03,
    "vz": 3.99358827504333659e-04,
}


def _orbit() -> CartesianOrbits:
    cov = np.zeros((1, 6, 6))
    for k, d in enumerate([1e-14, 1e-14, 1e-14, 1e-18, 1e-18, 1e-18]):
        cov[0, k, k] = d
    return CartesianOrbits.from_kwargs(
        orbit_id=["apophis_vis"],
        object_id=["99942"],
        coordinates=CartesianCoordinates.from_kwargs(
            epoch=np.array([_T0]),
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
        # Apophis's absolute magnitude (HG, H = 19.7, G = 0.15), so the engine
        # predicts an apparent magnitude and the magnitude gate is live. Without
        # it mag is NaN and observable is structurally all-False.
        photometric=PhotometricParams.from_kwargs(model=["hg"], h=[19.7], g=[0.15]),
    )


def _config() -> PlanningConfig:
    return PlanningConfig(
        observatories=[
            ObservatoryConfig(
                obs_code="F51",
                sigma_arcsec=(0.2, 0.2),
                max_apparent_mag=24.0,
                min_elongation_deg=60.0,
                min_elevation_deg=20.0,
                explicit_anchor_utc_mjd_tdb=_T0 + 1.0,
            ),
            ObservatoryConfig(
                obs_code="500",
                sigma_arcsec=(0.2, 0.2),
                max_apparent_mag=24.0,
                min_elongation_deg=60.0,
            ),
        ]
    )


def _result() -> empyrean.VisibilityResult:
    return compute_visibility(_orbit(), _T0, _T0 + 2.0, 0.02, _config())


# ── Table columns pinned by name ─────────────────────────────────────


def test_tables_carry_their_pinned_columns():
    r = _result()
    for col in [
        "orbit_id",
        "obs_code",
        "epoch_mjd_tdb",
        "solar_elongation_deg",
        "elevation_deg",
        "sun_altitude_deg",
        "observable",
        "elongation_ok",
        "mag",
        "magnitude_ok",
        "elevation_ok",
        "darkness_ok",
        "moon_altitude_deg",
        "moon_azimuth_deg",
        "moon_illuminated_fraction",
        "lunar_elongation_deg",
    ]:
        assert hasattr(r.points, col), f"VisibilityPoints is missing {col}"
    for col in ["night_index", "closing_gate", "hours_observable", "max_altitude_deg"]:
        assert hasattr(r.windows, col)
    for col in ["sunset_mjd_tdb", "local_solar_midnight_mjd_tdb", "sunrise_mjd_tdb", "note"]:
        assert hasattr(r.nights, col)
    for col in ["policy", "altitude_deg", "moon_altitude_deg", "sky_covariance_kind", "joint"]:
        assert hasattr(r.anchors, col)
    for col in ["note", "resolved_method", "earth_orientation_present"]:
        assert hasattr(r.observatories, col)


# ── observable is the four-gate conjunction (the PlanCandidates fix) ──


# Window sweep: a coarse 1-day scan over T0..T0+180 from F51 put Apophis's
# solar elongation at ~12 deg (deep conjunction) at T0, rising past ~33 deg
# only near T0+90. T0+90..+98 is the first short window where the object also
# clears ~15 deg during astronomical darkness, so the site-aware gates each
# take both values; the magnitude threshold is tuned to the band below.
def _gate_window() -> empyrean.VisibilityResult:
    """F51-only survey over an 8-day window chosen so each gate flips. The
    geometry thresholds bisect the window's own geometry: min_elongation
    34 deg sits inside the 33.3-34.9 deg elongation band, and min_elevation
    10 deg is below the object's ~15 deg culmination during darkness — so a
    dark, high-enough, far-enough row exists alongside rows failing each gate.
    Both geometry margins are ~0.7 deg or more, far above any platform-level
    ephemeris difference.

    max_apparent_mag = 21.81 bisects the window's measured apparent-magnitude
    band. Over this window the predicted mag spans 21.746-21.832 (HG, H=19.7,
    G=0.15); the geometrically observable rows (elong & elev & dark) span
    21.792-21.824. A limit of 21.81 therefore leaves some rows passing the
    magnitude gate and some failing it, and lets at least one geometrically
    observable row also clear the magnitude gate — so magnitude_ok takes both
    values and observable has a True row. (The band bisector 21.79 would make
    magnitude_ok both-valued but leave observable all-False, since every
    observable-geometry row is fainter than 21.79.)"""
    config = PlanningConfig(
        observatories=[
            ObservatoryConfig(
                obs_code="F51",
                sigma_arcsec=(0.2, 0.2),
                max_apparent_mag=21.81,
                min_elongation_deg=34.0,
                min_elevation_deg=10.0,
            )
        ]
    )
    return compute_visibility(_orbit(), _T0 + 90.0, _T0 + 98.0, 0.02, config)


def test_observable_is_the_conjunction_of_all_four_gates():
    """``observable`` is the AND of the elongation, magnitude, elevation and
    darkness gates, not any subset. In the chosen window all four gates take
    both values and some row clears all four, so wiring ``observable`` from a
    subset — the review's finding, e.g. ``elevation_ok & darkness_ok`` only —
    breaks the equality on the rows where a dropped gate is the one that is
    False.

    The magnitude gate is now live end to end: the orbit carries photometry
    (HG, H=19.7), ``_orbits_to_dict`` emits it, ``build_orbit_from_dict`` reads
    it, and the C ABI carries it into the survey, so ``mag`` is finite and
    ``magnitude_ok`` reflects the configured limit rather than being a
    structural all-False. The threshold (21.81) bisects the window's measured
    magnitude band (see ``_gate_window``)."""
    p = _gate_window().points
    assert len(p) > 0
    obs = p.observable.to_numpy(zero_copy_only=False)
    elong = p.elongation_ok.to_numpy(zero_copy_only=False)
    mag = p.magnitude_ok.to_numpy(zero_copy_only=False)
    elev = p.elevation_ok.to_numpy(zero_copy_only=False)
    dark = p.darkness_ok.to_numpy(zero_copy_only=False)

    # The wiring under test: observable is exactly the four-gate AND. Wiring it
    # from any subset breaks this on the rows where a dropped gate is False.
    assert np.array_equal(obs, elong & mag & elev & dark)

    # Each gate — magnitude included, now that photometry reaches the engine —
    # genuinely varies across the window, so each is load-bearing in the AND.
    for name, gate in (
        ("elongation", elong),
        ("magnitude", mag),
        ("elevation", elev),
        ("darkness", dark),
    ):
        assert gate.any() and not gate.all(), f"{name}_ok has both True and False"

    # Some row clears every gate at once: the night product has a genuinely
    # observable epoch, not merely a geometrically observable one.
    assert obs.any(), "observable has a True row once the magnitude gate is live"


# ── Ground vs horizonless; Moon; anchors; sky covariance ─────────────


def test_ground_site_has_nights_moon_and_anchors_geocentre_does_not():
    r = _result()

    # Ground site: nights with the three site-derived anchors, plus the
    # in-window explicit anchor on the night it lands in.
    f51_nights = r.nights.apply_mask(r.nights.obs_code.to_numpy(zero_copy_only=False) == "F51")
    assert len(f51_nights) > 0, "F51 has local nights in a 2-day window"
    f51_anchors = r.anchors.apply_mask(r.anchors.obs_code.to_numpy(zero_copy_only=False) == "F51")
    policies = set(f51_anchors.policy.to_pylist())
    assert {"midnight", "transit", "nearest_observable_to_midnight"} <= policies
    assert "explicit_utc" in policies, "the in-window explicit anchor is present"

    # Moon circumstances on the F51 points.
    f51_points = r.points.apply_mask(r.points.obs_code.to_numpy(zero_copy_only=False) == "F51")
    moon_alt = f51_points.moon_altitude_deg.to_numpy(zero_copy_only=False)
    assert np.isfinite(moon_alt).any(), "Moon altitude populated for a ground site"

    # A covariance-bearing orbit yields a sky covariance (packed joint) at a
    # midnight anchor (which always has an epoch).
    midnights = f51_anchors.apply_mask(
        f51_anchors.policy.to_numpy(zero_copy_only=False) == "midnight"
    )
    assert len(midnights) > 0
    assert midnights.joint.width.to_numpy(zero_copy_only=False).max() > 0, (
        "the midnight anchor carries a packed-joint sky covariance"
    )

    # Horizonless observer: no nights; the note explains why; no elevation.
    geo_nights = r.nights.apply_mask(r.nights.obs_code.to_numpy(zero_copy_only=False) == "500")
    assert len(geo_nights) == 0, "the geocentre has no local night"
    geo_summary = r.observatories.apply_mask(
        r.observatories.obs_code.to_numpy(zero_copy_only=False) == "500"
    )
    assert geo_summary.note.to_pylist()[0] is not None
    geo_points = r.points.apply_mask(r.points.obs_code.to_numpy(zero_copy_only=False) == "500")
    assert geo_points.elevation_deg.null_count == len(geo_points)


# ── Twilight-threshold validation reaches the engine ─────────────────


def test_partial_twilight_thresholds_are_refused_before_the_engine():
    cfg = PlanningConfig(
        observatories=[
            ObservatoryConfig(
                obs_code="F51",
                sigma_arcsec=(0.2, 0.2),
                max_apparent_mag=24.0,
                min_elongation_deg=60.0,
                sunset_sun_altitude_deg=-0.833,  # only one of four
            )
        ]
    )
    with pytest.raises(ValueError, match="all-or-nothing"):
        compute_visibility(_orbit(), _T0, _T0 + 1.0, 0.05, cfg)


def test_non_descending_twilight_is_refused_by_the_observatory_code():
    cfg = PlanningConfig(
        observatories=[
            ObservatoryConfig(
                obs_code="F51",
                sigma_arcsec=(0.2, 0.2),
                max_apparent_mag=24.0,
                min_elongation_deg=60.0,
                sunset_sun_altitude_deg=-6.0,
                civil_twilight_sun_altitude_deg=-0.833,  # above sunset → invalid
                nautical_twilight_sun_altitude_deg=-12.0,
                astronomical_twilight_sun_altitude_deg=-18.0,
            )
        ]
    )
    with pytest.raises(RuntimeError, match="F51"):
        compute_visibility(_orbit(), _T0, _T0 + 1.0, 0.05, cfg)
