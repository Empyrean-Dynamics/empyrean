"""Write OD per-observation residuals to parquet / JSON / CSV."""

from typing import Any

import numpy as np

from empyrean.od.residuals import ObservationResults

ResidualArray = np.ndarray[Any, np.dtype[np.float64]] | np.ndarray[Any, np.dtype[np.uint8]]


def _residuals_to_dict(residuals: ObservationResults) -> dict[str, ResidualArray]:
    return {
        "ra_residuals_arcsec": np.asarray(residuals.ra_residual.to_numpy(zero_copy_only=False)),
        "dec_residuals_arcsec": np.asarray(residuals.dec_residual.to_numpy(zero_copy_only=False)),
        "chi2": np.asarray(residuals.chi2.to_numpy(zero_copy_only=False)),
        "probability": np.asarray(residuals.probability.to_numpy(zero_copy_only=False)),
        "selected": np.asarray(residuals.selected.to_numpy(zero_copy_only=False), dtype=np.uint8),
    }


def write_residuals_parquet(path: str, residuals: ObservationResults) -> None:
    """Write an :class:`ObservationResults` table to parquet."""
    from empyrean._empyrean_rs import _write_residuals_parquet

    _write_residuals_parquet(path, _residuals_to_dict(residuals))


def write_residuals_json(path: str, residuals: ObservationResults) -> None:
    """Write residuals to JSON."""
    from empyrean._empyrean_rs import _write_residuals_json

    _write_residuals_json(path, _residuals_to_dict(residuals))


def write_residuals_csv(path: str, residuals: ObservationResults) -> None:
    """Write residuals to CSV."""
    from empyrean._empyrean_rs import _write_residuals_csv

    _write_residuals_csv(path, _residuals_to_dict(residuals))
