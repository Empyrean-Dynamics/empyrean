"""Shared fixtures for empyrean tests."""

from pathlib import Path

import empyrean
import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent / "data"


def data_available(*filenames):
    """Check if all validation data files exist."""
    return all((DATA_DIR / f).exists() for f in filenames)


@pytest.fixture(scope="session", autouse=True)
def initialize_empyrean():
    """Initialize empyrean context for the test session.

    Uses EMPYREAN_DATA_DIR or ~/.empyrean/data/ for SPICE kernels.
    Skips all tests if kernels are not available.
    """
    try:
        empyrean.initialize()
    except Exception as e:
        pytest.skip(f"empyrean initialization failed (missing kernels?): {e}")


@pytest.fixture
def cartesian_sun_icrf():
    """Load Cartesian Sun-centered ICRF reference data from Horizons."""
    path = DATA_DIR / "cartesian_sun_icrf.csv"
    if not path.exists():
        pytest.skip("cartesian_sun_icrf.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def cartesian_sun_ecliptic():
    """Load Cartesian Sun-centered Ecliptic reference data from Horizons."""
    path = DATA_DIR / "cartesian_sun_ecliptic.csv"
    if not path.exists():
        pytest.skip("cartesian_sun_ecliptic.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def cartesian_ssb_icrf():
    """Load Cartesian SSB ICRF reference data from Horizons."""
    path = DATA_DIR / "cartesian_ssb_icrf.csv"
    if not path.exists():
        pytest.skip("cartesian_ssb_icrf.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def cartesian_ssb_ecliptic():
    """Load Cartesian SSB Ecliptic reference data from Horizons."""
    path = DATA_DIR / "cartesian_ssb_ecliptic.csv"
    if not path.exists():
        pytest.skip("cartesian_ssb_ecliptic.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def cometary_sbdb():
    """Load cometary elements from SBDB reference data."""
    path = DATA_DIR / "cometary_sbdb.csv"
    if not path.exists():
        pytest.skip("cometary_sbdb.csv not found")
    return pd.read_csv(path)


@pytest.fixture
def elements_sun_ecliptic():
    """Load Keplerian/Cometary elements Sun Ecliptic reference data."""
    path = DATA_DIR / "elements_sun_ecliptic.csv"
    if not path.exists():
        pytest.skip("elements_sun_ecliptic.csv not found")
    return pd.read_csv(path)
