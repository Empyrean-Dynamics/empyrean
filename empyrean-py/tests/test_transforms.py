"""Validation tests for coordinate transformations against JPL Horizons reference data.

Mirrors villeneuve's test_jpl_validation_transforms.rs using the same
CSV reference files from data/validation/.
"""

import empyrean
import numpy as np
from empyrean import (
    CartesianCoordinates,
    CometaryCoordinates,
    Frame,
    KeplerianCoordinates,
    Origin,
)

# ── Tolerances (matching villeneuve's Rust tests) ────────────

FRAME_ROTATION_POS_TOL = 1e-14  # AU (~0.0015 mm)
FRAME_ROTATION_VEL_TOL = 1e-14  # AU/day
ORIGIN_TRANSLATION_POS_TOL = 1e-13  # AU (~0.015 mm)
ORIGIN_TRANSLATION_VEL_TOL = 1e-15  # AU/day
COMBINED_POS_TOL = 1e-13  # AU
COMBINED_VEL_TOL = 1e-15  # AU/day
ELEMENT_TOL = 1e-9
ANGLE_TOL_DEG = np.degrees(1e-9)  # convert radian tolerance to degrees


# ── Helpers ──────────────────────────────────────────────────


def cartesian_from_row(row, frame, origin):
    """Build CartesianCoordinates from a CSV row."""
    return CartesianCoordinates.from_kwargs(
        epoch=[row["epoch_mjd_tdb"]],
        x=[row["x_au"]],
        y=[row["y_au"]],
        z=[row["z_au"]],
        vx=[row["vx_au_d"]],
        vy=[row["vy_au_d"]],
        vz=[row["vz_au_d"]],
        frame=frame.value,
        origin=[str(origin)],
    )


def assert_cartesian_close(result, expected_row, pos_tol, vel_tol, label=""):
    """Assert a CartesianCoordinates row matches a reference CSV row."""
    x, y, z = result.x.to_numpy()[0], result.y.to_numpy()[0], result.z.to_numpy()[0]
    vx, vy, vz = (
        result.vx.to_numpy()[0],
        result.vy.to_numpy()[0],
        result.vz.to_numpy()[0],
    )

    dx = abs(x - expected_row["x_au"])
    dy = abs(y - expected_row["y_au"])
    dz = abs(z - expected_row["z_au"])
    dvx = abs(vx - expected_row["vx_au_d"])
    dvy = abs(vy - expected_row["vy_au_d"])
    dvz = abs(vz - expected_row["vz_au_d"])

    pos_err = np.sqrt(dx**2 + dy**2 + dz**2)
    vel_err = np.sqrt(dvx**2 + dvy**2 + dvz**2)

    name = expected_row.get("object_id", "unknown")
    assert pos_err < pos_tol, f"{label} {name}: pos error {pos_err:.2e} AU > {pos_tol:.2e}"
    assert vel_err < vel_tol, f"{label} {name}: vel error {vel_err:.2e} AU/d > {vel_tol:.2e}"


# ── Frame rotation tests ────────────────────────────────────


class TestFrameRotation:
    def test_icrf_to_ecliptic(self, cartesian_sun_icrf, cartesian_sun_ecliptic):
        """ICRF → Ecliptic frame rotation matches Horizons."""
        for _, icrf_row in cartesian_sun_icrf.iterrows():
            obj_id = icrf_row["object_id"]
            ecl_row = cartesian_sun_ecliptic[cartesian_sun_ecliptic["object_id"] == obj_id].iloc[0]

            coords = cartesian_from_row(icrf_row, Frame.ICRF, Origin.SUN)
            result = empyrean.transform_coordinates(
                coords,
                CartesianCoordinates,
                frame=Frame.ECLIPTICJ2000,
            )
            assert_cartesian_close(
                result,
                ecl_row,
                FRAME_ROTATION_POS_TOL,
                FRAME_ROTATION_VEL_TOL,
                "ICRF→Ecliptic",
            )

    def test_ecliptic_to_icrf(self, cartesian_sun_icrf, cartesian_sun_ecliptic):
        """Ecliptic → ICRF frame rotation matches Horizons."""
        for _, ecl_row in cartesian_sun_ecliptic.iterrows():
            obj_id = ecl_row["object_id"]
            icrf_row = cartesian_sun_icrf[cartesian_sun_icrf["object_id"] == obj_id].iloc[0]

            coords = cartesian_from_row(ecl_row, Frame.ECLIPTICJ2000, Origin.SUN)
            result = empyrean.transform_coordinates(
                coords,
                CartesianCoordinates,
                frame=Frame.ICRF,
            )
            assert_cartesian_close(
                result,
                icrf_row,
                FRAME_ROTATION_POS_TOL,
                FRAME_ROTATION_VEL_TOL,
                "Ecliptic→ICRF",
            )


# ── Origin translation tests ────────────────────────────────


class TestOriginTranslation:
    def test_sun_to_ssb(self, cartesian_sun_icrf, cartesian_ssb_icrf):
        """Sun → SSB origin translation matches Horizons."""
        for _, sun_row in cartesian_sun_icrf.iterrows():
            obj_id = sun_row["object_id"]
            ssb_match = cartesian_ssb_icrf[cartesian_ssb_icrf["object_id"] == obj_id]
            if ssb_match.empty:
                continue
            ssb_row = ssb_match.iloc[0]

            coords = cartesian_from_row(sun_row, Frame.ICRF, Origin.SUN)
            result = empyrean.transform_coordinates(
                coords,
                CartesianCoordinates,
                origin=Origin.SSB,
            )
            assert_cartesian_close(
                result,
                ssb_row,
                ORIGIN_TRANSLATION_POS_TOL,
                ORIGIN_TRANSLATION_VEL_TOL,
                "Sun→SSB",
            )

    def test_ssb_to_sun(self, cartesian_sun_icrf, cartesian_ssb_icrf):
        """SSB → Sun origin translation matches Horizons."""
        for _, ssb_row in cartesian_ssb_icrf.iterrows():
            obj_id = ssb_row["object_id"]
            sun_match = cartesian_sun_icrf[cartesian_sun_icrf["object_id"] == obj_id]
            if sun_match.empty:
                continue
            sun_row = sun_match.iloc[0]

            coords = cartesian_from_row(ssb_row, Frame.ICRF, Origin.SSB)
            result = empyrean.transform_coordinates(
                coords,
                CartesianCoordinates,
                origin=Origin.SUN,
            )
            assert_cartesian_close(
                result,
                sun_row,
                ORIGIN_TRANSLATION_POS_TOL,
                ORIGIN_TRANSLATION_VEL_TOL,
                "SSB→Sun",
            )


# ── Combined transform tests ────────────────────────────────


class TestCombinedTransforms:
    def test_sun_icrf_to_ssb_ecliptic(self, cartesian_sun_icrf, cartesian_ssb_ecliptic):
        """Sun/ICRF → SSB/Ecliptic (frame + origin) matches Horizons."""
        for _, sun_row in cartesian_sun_icrf.iterrows():
            obj_id = sun_row["object_id"]
            ssb_match = cartesian_ssb_ecliptic[cartesian_ssb_ecliptic["object_id"] == obj_id]
            if ssb_match.empty:
                continue
            ssb_row = ssb_match.iloc[0]

            coords = cartesian_from_row(sun_row, Frame.ICRF, Origin.SUN)
            result = empyrean.transform_coordinates(
                coords,
                CartesianCoordinates,
                frame=Frame.ECLIPTICJ2000,
                origin=Origin.SSB,
            )
            assert_cartesian_close(
                result,
                ssb_row,
                COMBINED_POS_TOL,
                COMBINED_VEL_TOL,
                "Sun/ICRF→SSB/Ecliptic",
            )


# ── Element conversion tests ────────────────────────────────


class TestElementConversions:
    def test_cartesian_to_cometary_roundtrip(self, cartesian_sun_ecliptic):
        """Cartesian → Cometary → Cartesian round-trip preserves state.

        Near-parabolic orbits (Halley, e~0.97) lose more digits in the
        universal variable solver. Tolerance matches villeneuve's Rust test.
        """
        for _, row in cartesian_sun_ecliptic.iterrows():
            coords = cartesian_from_row(row, Frame.ECLIPTICJ2000, Origin.SUN)

            cometary = empyrean.transform_coordinates(coords, CometaryCoordinates)
            back = empyrean.transform_coordinates(cometary, CartesianCoordinates)

            assert_cartesian_close(
                back,
                row,
                1e-11,
                1e-11,
                "Cart→Com→Cart",
            )

    def test_cartesian_to_keplerian_roundtrip(self, cartesian_sun_ecliptic):
        """Cartesian → Keplerian → Cartesian round-trip preserves state.

        Tolerance matches villeneuve's Rust test (1e-11 AU ≈ 1.5 mm).
        """
        for _, row in cartesian_sun_ecliptic.iterrows():
            coords = cartesian_from_row(row, Frame.ECLIPTICJ2000, Origin.SUN)

            keplerian = empyrean.transform_coordinates(coords, KeplerianCoordinates)
            back = empyrean.transform_coordinates(keplerian, CartesianCoordinates)

            assert_cartesian_close(
                back,
                row,
                1e-11,
                1e-11,
                "Cart→Kep→Cart",
            )
