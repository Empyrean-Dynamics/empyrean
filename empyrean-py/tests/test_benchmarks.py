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

# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def single_cometary():
    """A single cometary coordinate (Apophis-like)."""
    return CometaryCoordinates.from_kwargs(
        epoch=[60200.0],
        q=[0.7461],
        e=[0.1914],
        i=[3.339],
        raan=[204.446],
        ap=[126.687],
        tp=[60159.0],
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)],
    )


@pytest.fixture
def single_cartesian():
    """A single Cartesian coordinate."""
    return CartesianCoordinates.from_kwargs(
        epoch=[60200.0],
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
        epoch=np.full(n, 60200.0),
        q=0.7 + rng.uniform(0, 0.5, n),
        e=0.1 + rng.uniform(0, 0.4, n),
        i=rng.uniform(0, 30, n),
        raan=rng.uniform(0, 360, n),
        ap=rng.uniform(0, 360, n),
        tp=60100.0 + rng.uniform(0, 200, n),
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)] * n,
    )


def make_batch_cometary_with_cov(n):
    """Create a batch of N cometary coordinates with covariance."""
    rng = np.random.default_rng(42)
    cov = np.zeros((n, 6, 6))
    for i in range(n):
        diag = rng.uniform(1e-10, 1e-6, 6)
        cov[i] = np.diag(diag)
    return CometaryCoordinates.from_kwargs(
        epoch=np.full(n, 60200.0),
        q=0.7 + rng.uniform(0, 0.5, n),
        e=0.1 + rng.uniform(0, 0.4, n),
        i=rng.uniform(0, 30, n),
        raan=rng.uniform(0, 360, n),
        ap=rng.uniform(0, 360, n),
        tp=60100.0 + rng.uniform(0, 200, n),
        frame=Frame.ECLIPTICJ2000.value,
        origin=[str(Origin.SUN)] * n,
        covariance=CometaryCovariance.from_matrix(cov),
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
        """Batch of 100 cometary → Cartesian with covariance propagation."""
        coords = make_batch_cometary_with_cov(100)
        benchmark(
            empyrean.transform_coordinates,
            coords,
            CartesianCoordinates,
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
