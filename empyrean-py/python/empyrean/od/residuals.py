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
    object_id = qv.LargeStringColumn(nullable=True)
    """ADES object identifier of the fit this row belongs to.

    Populated by :func:`~empyrean.od.determine.determine`, which fits per
    object — so the residuals of a whole batch live in one table and stay
    attributable. Null for :func:`~empyrean.od.determine.evaluate` /
    :func:`~empyrean.od.determine.refine`, where the caller supplied the
    one orbit and there is no grouping key."""
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
    ``non_finite_chi2`` / ``missing_jacobian`` / ``not_evaluated``.
    Mirrors ``scott::rejection::RejectionReason`` snake-cased."""
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
    influence_information_loss = qv.Float64Column(nullable=True)
    """D-optimality information loss from removing this observation:
    Δ_i = logdet(N) − logdet(N − I_i). Large values mean removal
    significantly degrades solution precision. +∞ when the observation
    is indispensable (N − I_i singular). Null when no influence pass
    was run (e.g. evaluate)."""

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
    along_cross_covariance_arcsec2 = qv.Float64Column(nullable=True)
    """Off-diagonal element of the 2×2 along/cross-track residual
    covariance (arcsec²). Together with the AT/CT 1σ fields this
    reconstructs the full symmetric 2×2. Null when no sky-motion-rate
    decomposition was available."""

    # ── Radar residual block (null on optical rows) ───────
    radar_kind = qv.LargeStringColumn(nullable=True)
    """``"delay"`` or ``"doppler"`` for radar observations; null for
    optical rows."""
    radar_residual = qv.Float64Column(nullable=True)
    """Radar residual, observed − predicted: round-trip delay in
    seconds (``radar_kind = "delay"``) or two-way Doppler in hertz
    (``radar_kind = "doppler"``). The optical RA/Dec residual columns
    are null on radar rows."""
    radar_chi2 = qv.Float64Column(nullable=True)
    """χ² of the radar residual."""
    radar_dof = qv.Int32Column(nullable=True)
    """Degrees of freedom of the radar residual (1 for radar)."""
    radar_probability = qv.Float64Column(nullable=True)
    """χ² survival probability of the radar residual."""
    radar_variance = qv.Float64Column(nullable=True)
    """Combined observed+predicted radar residual variance (s² for
    delay, Hz² for Doppler)."""

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

    selection_fraction_ok: bool
    """Did the fit retain enough of its input? ``False`` means the
    residual bars above describe a heavily pruned subset. Gates
    :attr:`extrapolation_acceptable`; deliberately not part of
    :attr:`fit_acceptable`.

    Reproduce the fraction from :attr:`selection_fraction_value`, not
    from :class:`ResidualSummary` — the summary counts merged radar /
    occultation stub rows that were never candidates for outlier
    pruning, so its ratio is a different (smaller) number."""
    selection_fraction_value: float
    """Fraction of observations retained (n_selected / n_obs)."""
    selection_fraction_threshold: float
    """Minimum retained fraction the gate required."""

    selected_arc_coverage_ok: bool
    """Do the **selected** observations still span enough of the arc to
    extrapolate across it? This is the coverage axis
    :attr:`extrapolation_acceptable` gates on — a strict tightening of
    the full-span :attr:`arc_coverage_ok`, which keeps its original
    meaning for callers that want it."""
    selected_arc_days_value: float
    """Arc span (days) over the selected observations only. NaN when
    nothing was selected."""
    selected_arc_fraction_value: float
    """Selected-span / full-span ratio."""
    selected_arc_fraction_threshold: float
    """Minimum span ratio the gate required."""

    trailing_gap_ok: bool
    """Were the most-recent observations kept? The absolute, asymmetric
    backstop the span ratio cannot provide: it catches a short recent
    tail rejected off a long arc, where the ratio still passes but the
    discarded rows are the ones nearest a forward extrapolation
    target."""
    trailing_gap_days_value: float
    """Days between the last selected and the last full-arc observation.
    ``0.0`` when the last kept observation is the last observation; NaN
    when nothing was selected."""
    trailing_gap_threshold: float
    """Largest trailing gap the gate allowed (days)."""

    radar_fit_ok: bool | None
    """Radar astrometry joint-fit acceptability. ``None`` when no radar
    contributed to the fit — which is never the same as ``False``."""


class FitSummary(qv.Table):
    """One row per **input** object of a batch orbit determination —
    delivered or not.

    :func:`~empyrean.od.determine.determine` fits every object in the
    observations. ``.orbits`` holds the objects that produced an orbit;
    this table holds *all* of them, so a partially successful batch is
    readable rather than silently shorter than its input. A failed
    object's measurement columns are NaN (never ``0.0``, which would read
    as a value at the floor) and :attr:`error` says why.

    The column names match the ``fit_summary`` files the CLI writes, so
    a table read back from ``fit_summary.parquet`` and this one describe
    a fit identically.
    """

    object_id = qv.LargeStringColumn()
    """ADES object identifier (permID / provID / trkSub)."""
    status = qv.LargeStringColumn()
    """``"delivered"`` or ``"failed"``."""

    converged = qv.BooleanColumn()
    """Did the differential correction reach its stopping criterion?"""
    iterations = qv.Int32Column()
    """DC iterations used. ``0`` on a failed object."""
    n_obs = qv.Int32Column()
    """Observations this object contributed."""
    n_selected = qv.Int32Column()
    """Observations the fit retained."""

    rms_ra_arcsec = qv.Float64Column(nullable=True)
    """RA·cos(Dec) residual RMS (arcsec)."""
    rms_dec_arcsec = qv.Float64Column(nullable=True)
    """Dec residual RMS (arcsec)."""
    reduced_chi2 = qv.Float64Column(nullable=True)
    r"""Reduced :math:`\chi^2` of the fit."""

    fit_acceptable = qv.BooleanColumn()
    """Aggregate fit-quality verdict."""
    extrapolation_acceptable = qv.BooleanColumn()
    """Aggregate verdict on forward extrapolation: :attr:`fit_acceptable`
    AND the four selection / coverage axes below."""

    selection_fraction_ok = qv.BooleanColumn()
    """Did the fit retain enough of its input?"""
    selection_fraction = qv.Float64Column(nullable=True)
    """Fraction of observations retained."""
    selection_fraction_threshold = qv.Float64Column(nullable=True)
    """Minimum retained fraction the gate required."""

    selected_arc_coverage_ok = qv.BooleanColumn()
    """Do the selected observations still span enough of the arc?"""
    selected_arc_days = qv.Float64Column(nullable=True)
    """Arc span over the selected observations only (days)."""
    selected_arc_fraction = qv.Float64Column(nullable=True)
    """Selected-span / full-span ratio."""
    selected_arc_fraction_threshold = qv.Float64Column(nullable=True)
    """Minimum span ratio the gate required."""

    trailing_gap_ok = qv.BooleanColumn()
    """Were the most-recent observations kept?"""
    trailing_gap_days = qv.Float64Column(nullable=True)
    """Days between the last selected and the last full-arc
    observation."""
    trailing_gap_threshold_days = qv.Float64Column(nullable=True)
    """Largest trailing gap the gate allowed (days)."""

    fractional_sigma_a_ok = qv.BooleanColumn()
    r"""Did :math:`\sigma_a / |a|` pass its threshold?"""
    fractional_sigma_a = qv.Float64Column(nullable=True)
    r"""Measured :math:`\sigma_a / |a|`."""
    fractional_sigma_a_threshold = qv.Float64Column(nullable=True)
    r"""Threshold for :math:`\sigma_a / |a|`."""

    solve_for_width = qv.Int32Column()
    """Width of the solved-parameter set (6 for a state-only fit). ``0``
    on a failed object."""
    error = qv.LargeStringColumn(nullable=True)
    """Failure message. Null on a delivered object."""
