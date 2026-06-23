"""Residual types for orbit determination."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

# pyarrow's compute functions are generated at runtime into the
# ``pyarrow.compute`` module namespace (see ``_make_global_functions``),
# so the bundled type stubs do not declare them and mypy cannot resolve
# ``pc.invert`` / ``pc.is_in`` / ... as attributes. Bind the ones we use
# to precisely-typed module-level aliases via the module ``__dict__``
# (the alias is the exact same function object — runtime behavior is
# unchanged) so every call site gets a real signature.
_invert: Callable[[pa.Array], pa.BooleanArray] = pc.__dict__["invert"]
_is_in: Callable[..., pa.BooleanArray] = pc.__dict__["is_in"]
_and_kleene: Callable[[pa.Array, pa.Array], pa.BooleanArray] = pc.__dict__["and_kleene"]
_is_finite: Callable[[pa.Array], pa.BooleanArray] = pc.__dict__["is_finite"]
_greater_equal: Callable[[pa.Array, pa.Scalar], pa.BooleanArray] = pc.__dict__["greater_equal"]


class ObservationResults(qv.Table):
    """Per-observation OD results — full upstream surface.

    Mirrors ``scott::results::ObservationResult`` field-for-field.
    Use :attr:`obs_id` to cross-match a row back to its source ADES
    observation. Null values mark stats that weren't computed for the
    call type (e.g. evaluate doesn't run rejection or influence
    diagnostics, so those fields come back null /
    ``rejection_reason="not_evaluated"``).

    All angular quantities are in **arcseconds**;
    :attr:`track_position_angle_deg` is in **degrees** (East of North).
    """

    # ── Identification (cross-match) ─────────────────────
    obs_id = qv.LargeStringColumn()
    """ADES `obsID` (or scott auto-assigned) — cross-match key."""
    obs_code = qv.LargeStringColumn()
    """MPC observatory code."""
    ast_cat = qv.LargeStringColumn(nullable=True)
    """Star catalog used for astrometric reduction (ADES `astCat`)."""
    epoch_mjd_tdb = qv.Float64Column()
    """Observation epoch (MJD TDB)."""

    # ── Core residuals ────────────────────────────────────
    ra_residual = qv.Float64Column()
    """RA·cos(Dec) residual O−C (arcsec)."""
    dec_residual = qv.Float64Column()
    """Dec residual O−C (arcsec)."""
    chi2 = qv.Float64Column()
    """Mahalanobis χ². NaN if combined covariance unavailable."""
    dof = qv.Int32Column()
    """Degrees of freedom (number of non-NaN residual dimensions)."""
    probability = qv.Float64Column()
    """χ² survival probability."""
    selected = qv.BooleanColumn()
    """True = used in fit."""

    # ── Residual covariance ───────────────────────────────
    residual_cov_ra = qv.Float64Column(nullable=True)
    """Combined obs+predicted RA·cos(Dec) variance (arcsec²)."""
    residual_cov_dec = qv.Float64Column(nullable=True)
    """Combined obs+predicted Dec variance (arcsec²)."""
    residual_cov_corr = qv.Float64Column(nullable=True)
    """RA-Dec correlation coefficient (dimensionless, [-1, 1])."""

    # ── Rejection ─────────────────────────────────────────
    rejection_reason = qv.LargeStringColumn()
    """One of: ``accepted`` / ``chi_squared`` / ``sigma_clip`` /
    ``cooks_distance`` / ``adaptive`` / ``unsupported_observatory`` /
    ``cmc2003`` / ``radar_observations_unsupported`` /
    ``occultation_observations_unsupported`` / ``outside_arc`` /
    ``not_evaluated``. Mirrors ``scott::rejection::RejectionReason``
    snake-cased."""
    rejection_criterion = qv.Float64Column(nullable=True)
    """The criterion value (χ², Cook's D, ...) tested against the threshold."""
    rejection_threshold = qv.Float64Column(nullable=True)
    """Static threshold the criterion was compared against."""
    rejection_effective_threshold = qv.Float64Column(nullable=True)
    """Effective threshold for adaptive rejection (Layer 3)."""
    rejection_information_loss = qv.Float64Column(nullable=True)
    """D-optimality information loss from removing this observation."""

    # ── Influence diagnostics ─────────────────────────────
    cooks_distance = qv.Float64Column(nullable=True)
    """Cook's distance."""
    leverage = qv.Float64Column(nullable=True)
    """Scalar leverage h_ii ∈ [0, 2]."""
    fractional_information = qv.Float64Column(nullable=True)
    """Fractional Fisher-information contribution f_i = tr(N⁻¹ I_i)."""

    # ── Along/cross-track decomposition ───────────────────
    along_track = qv.Float64Column(nullable=True)
    """Along-track residual (arcsec). Null if no sky-motion rates."""
    cross_track = qv.Float64Column(nullable=True)
    """Cross-track residual (arcsec). Null if no sky-motion rates."""
    along_track_error = qv.Float64Column(nullable=True)
    """Along-track 1-σ uncertainty (arcsec)."""
    cross_track_error = qv.Float64Column(nullable=True)
    """Cross-track 1-σ uncertainty (arcsec)."""
    track_position_angle_deg = qv.Float64Column(nullable=True)
    """Position angle of sky motion (deg, East of North)."""

    # ── Selection helpers ─────────────────────────────────

    def selected_only(self) -> "ObservationResults":
        """Rows with ``selected == True`` (used in the fit)."""
        return self.apply_mask(self.column("selected"))

    def rejected_only(self) -> "ObservationResults":
        """Rows with ``selected == False`` (rejected — see
        :attr:`rejection_reason` for which layer dropped them)."""
        return self.apply_mask(_invert(self.column("selected")))

    def select_station(self, obs_codes: str | list[str]) -> "ObservationResults":
        """Rows from one or more MPC observatory codes."""
        codes = [obs_codes] if isinstance(obs_codes, str) else list(obs_codes)
        mask = _is_in(self.column("obs_code"), value_set=pa.array(codes))
        return self.apply_mask(mask)

    def worst_chi2(self, n: int = 10) -> "ObservationResults":
        """Top-``n`` rows by :attr:`chi2`, descending. NaN χ² rows
        sort last (they're left-overs of the not-evaluated path)."""
        chi2 = self.chi2.to_numpy(zero_copy_only=False)
        # argsort puts NaNs at the end with stable=True
        order = np.argsort(-np.nan_to_num(chi2, nan=-np.inf), kind="stable")
        keep = order[:n]
        mask = _is_in(
            pa.array(np.arange(len(self), dtype=np.int64)),
            value_set=pa.array(keep.astype(np.int64)),
        )
        return self.apply_mask(mask)

    @property
    def rms_combined_arcsec(self) -> float:
        """:math:`\\sqrt{\\mathrm{rms}_\\text{ra}^2 + \\mathrm{rms}_\\text{dec}^2}`
        — convenience for callers that want a single per-table RMS
        figure rather than the RA / Dec split."""
        ra = self.ra_residual.to_numpy(zero_copy_only=False)
        dec = self.dec_residual.to_numpy(zero_copy_only=False)
        if len(ra) == 0:
            return float("nan")
        return float(np.sqrt(np.nanmean(ra**2 + dec**2)))


@dataclass
class ResidualSummary:
    """Aggregate residual statistics over a set of observations.

    Mirrors ``scott::results::ObservationResidualSummary``. AT/CT RMS
    fields are NaN when no along/cross-track decomposition was
    computed (no sky-motion rates available).

    All angular quantities in **arcseconds**.
    """

    num_obs: int
    num_selected: int
    num_rejected: int
    chi2: float
    dof: int
    reduced_chi2: float
    rms_ra_arcsec: float
    rms_dec_arcsec: float
    rms_combined_arcsec: float
    """Combined RA·cos(δ) + Dec residual RMS. Matches the find_orb /
    OrbFit ``rms`` reporting convention — a single number directly
    comparable across tools."""
    weighted_rms_ra_arcsec: float
    weighted_rms_dec_arcsec: float
    weighted_rms_combined_arcsec: float
    """Combined weighted RA·cos(δ) + Dec residual RMS."""
    mean_ra_arcsec: float
    mean_dec_arcsec: float
    std_ra_arcsec: float
    std_dec_arcsec: float
    rms_along_track_arcsec: float
    rms_cross_track_arcsec: float


class StationBiases(qv.Table):
    """Per-station fitted nuisance biases.

    Mirrors a vector of ``scott::results::StationBias``. Returned in
    :attr:`DetermineResult.station_biases` when
    :attr:`ODConfig.fit_station_biases` is enabled. Stations whose
    ``min_obs_per_station`` threshold wasn't met are absent from the table.

    Marginalized over the orbit fit, so the σ values include orbit
    uncertainty inherited through the Schur coupling
    \\(N_{ob}\\,(N_{bb}+P_b)^{-1}\\).
    """

    obs_code = qv.LargeStringColumn()
    """MPC observatory code."""
    n_obs = qv.UInt64Column()
    """Pre-rejection observation count from this station."""
    bias_ra_arcsec = qv.Float64Column()
    """Fitted RA·cos(δ) offset (arcsec)."""
    sigma_ra_arcsec = qv.Float64Column()
    """1-σ uncertainty on the RA bias (arcsec)."""
    bias_dec_arcsec = qv.Float64Column()
    """Fitted Dec offset (arcsec)."""
    sigma_dec_arcsec = qv.Float64Column()
    """1-σ uncertainty on the Dec bias (arcsec)."""
    bias_timing_sec = qv.Float64Column(nullable=True)
    """Fitted timing offset (seconds), populated only when a
    ``BiasKind::StationTiming`` nuisance was active."""
    sigma_timing_sec = qv.Float64Column(nullable=True)
    """1-σ on the timing bias, matching ``bias_timing_sec``."""
    significance = qv.Float64Column()
    """Scalar significance: max of :math:`|b_i| / \\sigma_i` across
    populated components. :math:`\\geq 3` indicates a real systematic
    worth keeping fitted; NaN when no component has a usable
    :math:`\\sigma`."""

    def significant(self, n_sigma: float = 3.0) -> "StationBiases":
        """Rows whose :attr:`significance` clears ``n_sigma``.

        Default of 3σ matches the conventional "real systematic worth
        flagging" threshold used by the OD pipeline's bias-fitting
        diagnostics. Rows with NaN significance are excluded.
        """
        sig = self.column("significance")
        mask = _and_kleene(
            _is_finite(sig),
            _greater_equal(sig, pa.scalar(n_sigma, type=pa.float64())),
        )
        return self.apply_mask(mask)


@dataclass
class AcceptabilityReport:
    """Structured fit-quality verdict — mirrors
    ``scott::od::AcceptabilityReport``.

    Each ``*_ok`` flag is the verdict; ``*_value`` is the measured
    statistic; ``*_threshold`` is the bound it was compared against.
    Override the thresholds via :class:`AcceptabilityThresholds` on
    :class:`ODConfig` (e.g. tighten ``fractional_sigma_a``
    for Sentry-grade impact monitoring).
    """

    fit_acceptable: bool
    """Top-level pass: converged AND positive-definite covariance AND
    reduced :math:`\\chi^2` AND RMS AND residual-isotropy thresholds
    all met. Trustworthy state vector at the arc epoch."""
    extrapolation_acceptable: bool
    """:attr:`fit_acceptable` AND arc-coverage AND
    :math:`\\sigma_a / |a|` thresholds met. Gate that VA sampling /
    close-approach prediction / follow-up scheduling should check
    before relying on extrapolated state."""

    converged_ok: bool
    """DC iteration reached the configured update-norm tolerance
    within the iteration budget."""

    reduced_chi2_ok: bool
    """Reduced :math:`\\chi^2` at or below
    :attr:`AcceptabilityThresholds.reduced_chi2`."""
    reduced_chi2_value: float
    """Measured reduced :math:`\\chi^2` of the post-DC fit."""
    reduced_chi2_threshold: float
    """Threshold the value was compared against."""

    rms_ok: bool
    """Combined astrometric RMS at or below
    :attr:`AcceptabilityThresholds.rms_arcsec`."""
    rms_value_arcsec: float
    """Combined RA·cos(δ) and Dec residual RMS (arcsec)."""
    rms_threshold_arcsec: float
    """Threshold the value was compared against."""

    residual_isotropy_ok: bool
    """Residual cloud is roughly isotropic in the sky plane:
    :math:`\\max(AT/CT,\\; CT/AT)` at or below
    :attr:`AcceptabilityThresholds.at_ct_ratio`. NaN when no
    along/cross-track decomposition was computed (no sky-motion rates
    available)."""
    at_ct_ratio_value: float
    """Measured :math:`\\max(AT/CT,\\; CT/AT)` ratio."""
    at_ct_ratio_threshold: float
    """Threshold the value was compared against."""

    covariance_ok: bool
    """Final 6×6 state covariance is finite and positive-definite."""

    arc_coverage_ok: bool
    """Observation-arc length at or above
    :attr:`AcceptabilityThresholds.min_arc_days`."""
    arc_days_value: float
    """Length of the observation arc actually used in the fit (days)."""
    arc_days_threshold: float
    """Threshold the value was compared against."""

    fractional_sigma_a_ok: bool
    """Fractional uncertainty :math:`\\sigma_a / |a|` at or below
    :attr:`AcceptabilityThresholds.fractional_sigma_a`. The default
    is a loose general-purpose gate; tighten it for
    Sentry-grade impact monitoring."""
    fractional_sigma_a_value: float
    """Measured :math:`\\sigma_a / |a|`."""
    fractional_sigma_a_threshold: float
    """Threshold the value was compared against."""
