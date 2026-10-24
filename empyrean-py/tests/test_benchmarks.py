"""Performance benchmarks for empyrean Python bindings.

Run with: pytest tests/test_benchmarks.py --benchmark-only
"""

import empyrean
import numpy as np
import pytest
from empyrean import (
    CartesianCoordinates,
    CometaryCoordinates,
    CometaryCovariance,
    Frame,
    KeplerianCoordinates,
    Origin,
)
from empyrean.coordinates.epoch import Epochs
from empyrean.coordinates.transform import transform_orbits
from empyrean.orbits.joint import (
    CovarianceKind,
    JointFunctional,
    JointProvenance,
    JointQuality,
    PackedJoint,
    PackedJoints,
)
from empyrean.orbits.orbits import CartesianOrbits, CometaryOrbits


def _state_joint(cov6: np.ndarray) -> PackedJoint:
    """A width-6 derived state joint whose state block is ``cov6`` (6x6)."""
    tri = [cov6[i, j] for i in range(6) for j in range(i + 1)]
    return PackedJoint(
        layout_present=0,
        layout_considered=0,
        layout_marginalized=0,
        width=6,
        tri=np.asarray(tri, dtype=np.float64),
        kind=CovarianceKind.LINEAR,
        quality=JointQuality.POSITIVE_DEFINITE,
        cov_rank=0,
        cov_null_axes=0,
        cov_min_eig_corr=float("nan"),
        functional=JointFunctional.STATE,
        provenance=JointProvenance.DERIVED,
        asserted_source=None,
    )


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def single_cometary():
    """A single cometary coordinate (Apophis-like)."""
    return CometaryCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=[60200.0], scale="tdb"),
        q=[0.7461],
        e=[0.1914],
        i=[3.339],
        raan=[204.446],
        ap=[126.687],
        perihelion_epochs=Epochs.from_kwargs(mjd=[60159.0], scale="tdb"),
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)],
    )


@pytest.fixture
def single_cartesian():
    """A single Cartesian coordinate."""
    return CartesianCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=[60200.0], scale="tdb"),
        x=[0.5],
        y=[-0.8],
        z=[-0.1],
        vx=[0.012],
        vy=[0.005],
        vz=[0.002],
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)],
    )


def make_batch_cometary(n):
    """Create a batch of N cometary coordinates with slight variations."""
    rng = np.random.default_rng(42)
    return CometaryCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=np.full(n, 60200.0), scale="tdb"),
        q=0.7 + rng.uniform(0, 0.5, n),
        e=0.1 + rng.uniform(0, 0.4, n),
        i=rng.uniform(0, 30, n),
        raan=rng.uniform(0, 360, n),
        ap=rng.uniform(0, 360, n),
        perihelion_epochs=Epochs.from_kwargs(mjd=60100.0 + rng.uniform(0, 200, n), scale="tdb"),
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)] * n,
    )


def make_batch_cometary_orbits_with_cov(n):
    """Create a batch of N cometary orbits each carrying a width-6 joint.

    A covariance-bearing transform rides the orbit's packed joint, so the
    benchmark measures the orbit-level ``transform_orbits`` call.
    """
    rng = np.random.default_rng(42)
    joints = [_state_joint(np.diag(rng.uniform(1e-10, 1e-6, 6))) for _ in range(n)]
    coords = CometaryCoordinates.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=np.full(n, 60200.0), scale="tdb"),
        q=0.7 + rng.uniform(0, 0.5, n),
        e=0.1 + rng.uniform(0, 0.4, n),
        i=rng.uniform(0, 30, n),
        raan=rng.uniform(0, 360, n),
        ap=rng.uniform(0, 360, n),
        perihelion_epochs=Epochs.from_kwargs(mjd=60100.0 + rng.uniform(0, 200, n), scale="tdb"),
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)] * n,
    )
    return CometaryOrbits.from_kwargs(
        orbit_id=[f"o{i}" for i in range(n)],
        coordinates=coords,
        covariance=PackedJoints.from_joints(joints),
    )


# ── Transform benchmarks ────────────────────────────────────


class TestTransformBenchmarks:
    def test_cometary_to_cartesian_single(self, benchmark, single_cometary):
        """Single cometary → Cartesian transform."""
        benchmark(
            empyrean.transform_coordinates,
            single_cometary,
            CartesianCoordinates,
        )

    def test_cartesian_to_keplerian_single(self, benchmark, single_cartesian):
        """Single Cartesian → Keplerian transform."""
        benchmark(
            empyrean.transform_coordinates,
            single_cartesian,
            KeplerianCoordinates,
        )

    def test_cometary_to_cartesian_batch_100(self, benchmark):
        """Batch of 100 cometary → Cartesian transforms."""
        coords = make_batch_cometary(100)
        benchmark(
            empyrean.transform_coordinates,
            coords,
            CartesianCoordinates,
        )

    def test_cometary_to_cartesian_batch_10k(self, benchmark):
        """Batch of 10,000 cometary → Cartesian transforms."""
        coords = make_batch_cometary(10_000)
        benchmark(
            empyrean.transform_coordinates,
            coords,
            CartesianCoordinates,
        )

    def test_cometary_to_cartesian_with_covariance(self, benchmark):
        """Batch of 100 cometary orbits → Cartesian with joint propagation."""
        orbits = make_batch_cometary_orbits_with_cov(100)
        benchmark(
            transform_orbits,
            orbits,
            CartesianOrbits,
        )

    def test_origin_translation_sun_to_ssb(self, benchmark, single_cometary):
        """Single cometary transform with Sun → SSB origin change."""
        benchmark(
            empyrean.transform_coordinates,
            single_cometary,
            CartesianCoordinates,
            origin=Origin.SSB,
        )

    def test_combined_frame_and_origin(self, benchmark, single_cometary):
        """Single transform: Ecliptic/Sun → ICRF/SSB."""
        benchmark(
            empyrean.transform_coordinates,
            single_cometary,
            CartesianCoordinates,
            frame=Frame.ICRF,
            origin=Origin.SSB,
        )


# ── Type construction benchmarks ─────────────────────────────


class TestTypeConstructionBenchmarks:
    def test_cometary_from_kwargs_100(self, benchmark):
        """Construct CometaryCoordinates with 100 rows."""
        benchmark(make_batch_cometary, 100)

    def test_cometary_from_kwargs_10k(self, benchmark):
        """Construct CometaryCoordinates with 10,000 rows."""
        benchmark(make_batch_cometary, 10_000)

    def test_covariance_from_matrix_100(self, benchmark):
        """Construct CometaryCovariance from 100 matrices."""
        matrices = np.eye(6).reshape(1, 6, 6) * np.ones((100, 1, 1)) * 1e-8
        benchmark(CometaryCovariance.from_matrix, matrices)
