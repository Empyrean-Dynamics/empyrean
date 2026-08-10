"""Write OD per-observation residuals to parquet / JSON / CSV."""

from typing import Any

import numpy as np

from empyrean.od.residuals import ObservationResults

ResidualArray = np.ndarray[Any, np.dtype[np.float64]] | np.ndarray[Any, np.dtype[np.uint8]]


def _residuals_to_dict(residuals: ObservationResults) -> dict[str, Any]:
    """Flatten the whole :class:`ObservationResults` table for the Rust
    writer.

    Every column crosses. A projection here would write a residual file
    whose join keys, rejection attribution and influence diagnostics are
    all blank — the file would have the columns and none of the content.
    """

    def _f(name: str) -> ResidualArray:
        col = getattr(residuals, name)
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)

    def _s(name: str) -> list[str | None]:
        return list(getattr(residuals, name).to_pylist())

    def _u32(name: str) -> list[int]:
        return [int(v) if v is not None else 0 for v in getattr(residuals, name).to_pylist()]

    return {
        # ── Identification / join keys ──
        "obs_ids": _s("obs_id"),
        "object_ids": _s("object_id"),
        "obs_codes": _s("obs_code"),
        "ast_cats": _s("ast_cat"),
        "epochs": _f("epoch_mjd_tdb"),
        # ── Core residuals ──
        "ra_residuals_arcsec": _f("ra_residual"),
        "dec_residuals_arcsec": _f("dec_residual"),
        "chi2": _f("chi2"),
        "dofs": _u32("dof"),
        "probability": _f("probability"),
        "selected": np.asarray(residuals.selected.to_numpy(zero_copy_only=False), dtype=np.uint8),
        # ── Residual covariance ──
        "residual_cov_ras": _f("residual_cov_ra"),
        "residual_cov_decs": _f("residual_cov_dec"),
        "residual_cov_corrs": _f("residual_cov_corr"),
        # ── Rejection (the attribution taxonomy) ──
        "rejection_reasons": _s("rejection_reason"),
        "rejection_criterions": _f("rejection_criterion"),
        "rejection_thresholds": _f("rejection_threshold"),
        "rejection_effective_thresholds": _f("rejection_effective_threshold"),
        "rejection_information_losses": _f("rejection_information_loss"),
        # ── Influence ──
        "cooks_distances": _f("cooks_distance"),
        "leverages": _f("leverage"),
        "fractional_informations": _f("fractional_information"),
        "influence_information_losses": _f("influence_information_loss"),
        # ── Sky motion ──
        "along_tracks": _f("along_track"),
        "cross_tracks": _f("cross_track"),
        "along_track_errors": _f("along_track_error"),
        "cross_track_errors": _f("cross_track_error"),
        "track_position_angles": _f("track_position_angle_deg"),
        "along_cross_covariances": _f("along_cross_covariance_arcsec2"),
        # ── Radar block (blank on optical rows) ──
        "radar_kinds": _s("radar_kind"),
        "radar_residuals": _f("radar_residual"),
        "radar_chi2s": _f("radar_chi2"),
        "radar_probabilities": _f("radar_probability"),
        "radar_variances": _f("radar_variance"),
        "radar_dofs": _u32("radar_dof"),
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
