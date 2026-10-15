"""Observation-plan evaluation.

:func:`evaluate_plan` wraps ``empyrean_core::planning::evaluate_plan_single``
(via the C ABI). Given a barycentric orbit that already carries a 6×6
Cartesian covariance and a list of candidate observations, it reports how
much each candidate would tighten the orbit.

Ordering, the conditional nature of the marginal gains, and the
deliberate subsetting are documented in :func:`evaluate_plan`'s own
Notes section — this module docstring is not rendered in the API docs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from empyrean._convert import AnyOrbits, naif_to_origin
from empyrean.coordinates.epoch import Epochs
from empyrean.od.determine import _orbits_to_dict
from empyrean.orbits.joint import _KIND_BY_CODE, CovarianceKind, PackedJoints
from empyrean.planning.result import (
    _INT_TO_KIND,
    _INT_TO_LOG_DET_WITHHELD,
    _INT_TO_RADAR_MODE,
    STAGE_POSTERIOR,
    STAGE_PRIOR,
    Nights,
    ObservatorySummary,
    PlanCandidates,
    PlanEphemeris,
    PlanMetrics,
    PlannedObservation,
    PlanningConfig,
    PlanResult,
    VisibilityAnchors,
    VisibilityPoints,
    VisibilityResult,
    VisibilityWindows,
    _nan_to_null,
)

ResultDict = dict[str, Any]

# NAIF id of the Solar System barycenter — the basis the planner folds
# every candidate's sensitivity chain against.
_SSB_NAIF_ID = 0

# `Representation::Cartesian`. The prior is consumed as a Cartesian state
# covariance and is never converted, so this is a hard requirement rather
# than a convenience.
_CARTESIAN_REPRESENTATION = 0

_REPRESENTATION_NAMES: dict[int, str] = {
    0: "cartesian",
    1: "keplerian",
    2: "cometary",
    3: "spherical",
}

# One call does both conversions the planner requires, so both guards
# name the same fix.
_CONVERSION_HINT = "transform_coordinates(orbit.coordinates, CartesianCoordinates, origin='SSB')"

# Every column the binding fills for one planned observation, in the
# order `PlannedObservation._to_wire_dict` emits them. Transposing
# through this list — rather than reading whatever keys the first row
# happened to carry — keeps a future field from being silently dropped
# for the whole batch.
_PLANNED_COLUMNS: dict[str, str] = {
    "epoch_mjd_tdb": "epochs",
    "kind": "kinds",
    "optical_code": "optical_codes",
    "optical_sigma_ra_arcsec": "optical_sigma_ra_arcsec",
    "optical_sigma_dec_arcsec": "optical_sigma_dec_arcsec",
    "radar_transmit_station": "radar_transmit_stations",
    "radar_receive_station": "radar_receive_stations",
    "radar_mode": "radar_modes",
    "radar_bandwidth_hz": "radar_bandwidth_hz",
    "radar_freq_resolution_hz": "radar_freq_resolution_hz",
    "radar_snr": "radar_snr",
    "radar_target_h_mag": "radar_target_h_mag",
    "radar_target_visual_albedo": "radar_target_visual_albedo",
    "radar_target_radar_albedo": "radar_target_radar_albedo",
    "radar_target_diameter_km": "radar_target_diameter_km",
    "radar_target_spin_period_hours": "radar_target_spin_period_hours",
    "radar_integration_s": "radar_integration_s",
}


def _planned_to_dict(planned: Sequence[PlannedObservation]) -> dict[str, list[Any]]:
    """Transpose a candidate list into the parallel columns the binding
    consumes. Every column is emitted at full length, so the binding's
    alignment guard can only fire on a genuine mismatch.
    """
    rows = [p._to_wire_dict() for p in planned]
    return {column: [row[field] for row in rows] for field, column in _PLANNED_COLUMNS.items()}


# Campaign stage → the binding's key prefix for that stage's block, in
# the row order the table carries them.
_METRIC_STAGES: tuple[tuple[str, str], ...] = (
    (STAGE_PRIOR, "prior_"),
    (STAGE_POSTERIOR, "posterior_"),
)


def _metrics_table(out: ResultDict) -> PlanMetrics:
    """Assemble the two-row prior/posterior metrics table from the
    binding's flat scalar entries.

    Each column is read for both stages through the same key suffix, so
    the two rows can only ever disagree about which block they came
    from — never about which metric.
    """

    def column(name: str) -> list[float]:
        return [float(out[f"{prefix}{name}"]) for _, prefix in _METRIC_STAGES]

    def withheld(prefix: str) -> str:
        code = int(out[f"{prefix}log_det_withheld"])
        try:
            return _INT_TO_LOG_DET_WITHHELD[code]
        except KeyError:
            raise RuntimeError(
                f"engine returned an unknown log-det withheld code: {code}"
            ) from None

    def withheld_method(prefix: str) -> str | None:
        # -1 is the "no sampled method applies" sentinel (every reason but
        # SAMPLED_MOMENT); it maps to a null, never to a kind.
        code = int(out[f"{prefix}log_det_withheld_method"])
        if code < 0:
            return None
        try:
            return _KIND_BY_CODE[code].value
        except KeyError:
            raise RuntimeError(
                f"engine returned an unknown log-det withheld method code: {code}"
            ) from None

    return PlanMetrics.from_kwargs(
        stage=[stage for stage, _ in _METRIC_STAGES],
        position_sigma_km=column("position_sigma_km"),
        velocity_sigma_m_s=column("velocity_sigma_m_s"),
        semi_major_km=column("semi_major_km"),
        semi_minor_km=column("semi_minor_km"),
        log_det=column("log_det"),
        log_det_withheld=[withheld(prefix) for _, prefix in _METRIC_STAGES],
        log_det_withheld_method=[withheld_method(prefix) for _, prefix in _METRIC_STAGES],
    )


def _floats(out: ResultDict, key: str) -> npt.NDArray[np.float64]:
    return np.asarray(out[key], dtype=np.float64)


def evaluate_plan(
    orbit: AnyOrbits,
    planned: Sequence[PlannedObservation],
    config: PlanningConfig | None = None,
    orbit_id: str | None = None,
) -> PlanResult:
    """Rank candidate follow-up observations by how much each would
    tighten an orbit.

    The orbit's covariance is the information prior; each candidate is
    folded into it in turn, and the result reports the marginal gain per
    candidate plus the prior / posterior covariance summary for the
    campaign as a whole.

    Optical candidates contribute sky-plane (RA/Dec) information; radar
    candidates contribute the line-of-sight range and range-rate that
    angles-only astrometry cannot supply. A radar candidate's
    measurement σ is the Cramér-Rao bound set by the waveform bandwidth
    and the effective SNR — supply the SNR, or leave
    :attr:`PlannedObservation.radar_snr` at ``None`` to have it derived
    from a link budget over the target's physical properties.

    Parameters
    ----------
    orbit : CartesianOrbits
        Single orbit (exactly one row) carrying a 6×6 **Cartesian**
        covariance, with its origin at the **Solar System barycenter**.
        The frame is free.

        Both requirements are hard: the covariance is consumed as a
        Cartesian state prior and is never converted, so elements in
        another representation would be silently reinterpreted. A fit in
        another basis — a :func:`~empyrean.od.determine.determine`
        result is heliocentric, and may be cometary or Keplerian —
        converts in one call with
        ``transform_coordinates(fit.orbit.coordinates,
        CartesianCoordinates, origin="SSB")``. The origin half of that is
        a pure translation, so the covariance and every metric below it
        are unchanged by it.
    planned : sequence of PlannedObservation
        Candidate observations. Build them with
        :meth:`PlannedObservation.optical` /
        :meth:`PlannedObservation.radar`. Must not be empty.
    config : PlanningConfig, optional
        Configuration. Defaults to :class:`PlanningConfig` defaults
        (Standard force model).
    orbit_id : str, optional
        Label carried through onto :attr:`PlanResult.orbit_id`. Defaults
        to the input orbit's own ``orbit_id``.

    Returns
    -------
    PlanResult
        The two-row :class:`PlanMetrics` table bracketing the campaign,
        the :class:`PlanCandidates` table of per-candidate gains, and the
        :class:`PlanEphemeris` table of predicted sky positions.

    Raises
    ------
    ValueError
        The orbit is not exactly one row, carries no covariance, is not
        Cartesian, is not barycentric, or the candidate list is empty.
    RuntimeError
        The engine rejected the plan — a singular prior covariance, an
        unregistered observatory code, a defunct radar dish, a
        non-positive bandwidth for a delay measurement, or a link budget
        missing a property it needs. The message names the cause.

    Notes
    -----
    Candidates are evaluated in **ascending epoch order** regardless of
    the order they were supplied in, and each is folded into the
    covariance that already contains every earlier one. The marginal
    gains are therefore conditional on that sequence: two identical
    observations do not score identically, and
    :meth:`PlanCandidates.best_by_information_gain` ranks contributions
    within one campaign rather than standalone candidate value. Evaluate
    a one-candidate plan per candidate to compare them head to head.

    **What this entry point does not expose**, recorded so the omissions
    are not mistaken for oversights: the engine also offers a
    non-gravitational planning variant that solves over
    state ⊕ (A1, A2, A3) and reports the σ(A2) tightening a radar
    campaign buys, a visibility survey over a time window, batch
    evaluation across many orbits, and an encounter-B-plane
    characterization. None of them is reachable from this package.

    An orbit carrying non-gravitational parameters — a Yarkovsky fit,
    say — is accepted here and evaluated **state-only**. The
    non-gravitational acceleration still acts in the dynamics, so the
    predicted trajectory and sky positions account for it, but the
    solve-for set stays 6×6 (:attr:`PlanResult.active_width` reports
    ``6``), the A1/A2/A3 columns are not folded, and no σ(A2) is
    reported. The plan prices what the observations buy for the *state*
    under that force model, not what they buy for the non-gravitational
    parameters themselves.

    This function is the **single-object** form, while ``evaluate_plan``
    is the batch name one layer down in the engine. That is deliberate:
    the C symbol ``empyrean_evaluate_plan`` has carried single-object
    semantics since v0.7.0 and is frozen, and this package mirrors the C
    ABI it wraps. If a batch form is exposed later it follows the
    migration precedent :func:`empyrean.transform_coordinates` set — the
    batch takes the plain name and the single-object form gains a
    ``_single`` suffix — rather than renaming this function out from
    under callers now.

    Examples
    --------
    Two optical nights and one Goldstone radar run against a fitted
    orbit, shifted to the barycentre first:

    >>> from empyrean import (  # doctest: +SKIP
    ...     CartesianCoordinates,
    ...     CartesianOrbits,
    ...     PlannedObservation,
    ...     evaluate_plan,
    ...     transform_coordinates,
    ... )
    >>> coords = transform_coordinates(  # doctest: +SKIP
    ...     fit.orbit.coordinates, CartesianCoordinates, origin="SSB"
    ... )
    >>> orbit = fit.orbit.set_column("coordinates", coords)  # doctest: +SKIP
    >>> t0 = float(coords.epoch.to_numpy()[0])  # doctest: +SKIP
    >>> plan = evaluate_plan(  # doctest: +SKIP
    ...     orbit,
    ...     [
    ...         PlannedObservation.optical(t0 + 30.0, "F51", (0.2, 0.2)),
    ...         PlannedObservation.optical(t0 + 31.0, "F51", (0.2, 0.2)),
    ...         PlannedObservation.radar(
    ...             t0 + 45.0,
    ...             radar_bandwidth_hz=1.0e5,
    ...             radar_freq_resolution_hz=0.1,
    ...             radar_snr=50.0,
    ...         ),
    ...     ],
    ... )
    >>> plan.metrics.stage.to_pylist()  # doctest: +SKIP
    ['prior', 'posterior']
    >>> before = plan.metrics.prior().position_sigma_km[0].as_py()  # doctest: +SKIP
    >>> after = plan.metrics.posterior().position_sigma_km[0].as_py()  # doctest: +SKIP
    >>> after <= before  # doctest: +SKIP
    True
    """
    from empyrean._empyrean_rs import _evaluate_plan

    if len(orbit) != 1:
        raise ValueError(
            f"evaluate_plan takes exactly one orbit, got {len(orbit)}. The "
            f"covariance of that orbit is the information prior the plan is "
            f"evaluated against."
        )
    if len(planned) == 0:
        raise ValueError(
            "evaluate_plan requires at least one planned observation; an "
            "empty plan has nothing to evaluate."
        )
    if config is None:
        config = PlanningConfig()
    # PlanningConfig is mutable by design, so a field set after
    # construction skips its __post_init__ check. Re-run it here and the
    # refusal stays a ValueError raised before the engine is touched,
    # rather than the wrapper's RuntimeError backstop.
    config._reject_unread_knobs()
    # Same reason, same hole: PlannedObservation is mutable too, so a
    # field assigned after construction skips its __post_init__ check.
    # Re-running it here keeps every refusal a ValueError raised before
    # the engine is touched — including the sigma positivity that no
    # layer below Python enforces.
    for candidate in planned:
        candidate._validate()

    orbit_dict = _orbits_to_dict(orbit)
    if not bool(np.asarray(orbit_dict["has_covariance"])[0]):
        raise ValueError(
            "evaluate_plan requires an orbit carrying a 6×6 covariance — it is "
            "the information prior each candidate is folded into. Pass a "
            "determine() result, or attach a covariance to the coordinates."
        )
    # The prior is consumed as a 6×6 CARTESIAN state covariance: the
    # engine inverts it directly into the Fisher accumulator that every
    # Cartesian sensitivity Jacobian folds into. A covariance over
    # cometary or Keplerian elements is a different matrix in different
    # units, and nothing below this line would notice — the engine tags
    # it by representation but never converts it, so the run returns
    # plausible finite numbers that mean nothing.
    representation = int(np.asarray(orbit_dict["representations"])[0])
    if representation != _CARTESIAN_REPRESENTATION:
        raise ValueError(
            f"evaluate_plan requires Cartesian orbits, got "
            f"{_REPRESENTATION_NAMES.get(representation, representation)}. The "
            f"covariance is consumed as a 6×6 Cartesian state prior and is "
            f"never converted, so elements in any other representation would "
            f"be reinterpreted rather than rejected. Convert with "
            f"{_CONVERSION_HINT}."
        )
    # The planner folds each candidate's sensitivity chain against the
    # prior in one shared basis, and that basis is barycentric. Catching
    # a non-barycentric orbit here turns a mid-propagation mismatch into
    # an actionable message naming the fix; the engine still guards the
    # same invariant behind it.
    origin_naif = int(np.asarray(orbit_dict["origins"])[0])
    if origin_naif != _SSB_NAIF_ID:
        raise ValueError(
            f"evaluate_plan requires an orbit with its origin at the Solar "
            f"System barycenter, got {naif_to_origin(origin_naif)}. Convert "
            f"with {_CONVERSION_HINT} — an origin shift is a pure translation, "
            f"so the covariance and its metrics are unchanged."
        )
    if orbit_id is None:
        ids = orbit.orbit_id.to_pylist()
        orbit_id = ids[0] if ids and ids[0] else None

    out: ResultDict = _evaluate_plan(
        orbit_dict,
        _planned_to_dict(planned),
        config._to_wire_dict(),
        orbit_id,
    )

    # An unrecognized tag means this package and the engine disagree
    # about the result shape. The Rust wrapper refuses it one layer up,
    # so these are defence in depth — but relabelling instead of raising
    # would put a row of unknown shape into a typed table, which is what
    # the wrapper already declined to do.
    kind_tags = np.asarray(out["candidate_kind"])
    kinds: list[str] = []
    for tag in kind_tags:
        try:
            kinds.append(_INT_TO_KIND[int(tag)])
        except KeyError:
            raise RuntimeError(
                f"engine returned an unknown plan-candidate kind: {int(tag)}"
            ) from None
    mode_tags = np.asarray(out["candidate_radar_mode"])
    modes: list[str | None] = []
    for tag in mode_tags:
        # -1 is the engine's "not a radar candidate" sentinel; it becomes
        # a null, never a mode.
        if int(tag) < 0:
            modes.append(None)
            continue
        try:
            modes.append(_INT_TO_RADAR_MODE[int(tag)])
        except KeyError:
            raise RuntimeError(f"engine returned an unknown radar mode: {int(tag)}") from None

    candidates = PlanCandidates.from_kwargs(
        index=np.asarray(out["candidate_index"], dtype=np.uint64),
        obs_code=out["candidate_obs_code"],
        kind=kinds,
        observable=out["candidate_observable"],
        marginal_volume_reduction=_floats(out, "candidate_marginal_volume_reduction"),
        marginal_position_improvement=_floats(out, "candidate_marginal_position_improvement"),
        active_width=np.asarray(out["candidate_active_width"], dtype=np.uint64),
        cumulative_position_sigma_km=_floats(out, "candidate_cumulative_position_sigma_km"),
        cumulative_velocity_sigma_m_s=_floats(out, "candidate_cumulative_velocity_sigma_m_s"),
        cumulative_semi_major_km=_floats(out, "candidate_cumulative_semi_major_km"),
        cumulative_semi_minor_km=_floats(out, "candidate_cumulative_semi_minor_km"),
        cumulative_log_det=_floats(out, "candidate_cumulative_log_det"),
        along_track_sigma_arcsec=_nan_to_null(_floats(out, "candidate_along_track_sigma_arcsec")),
        cross_track_sigma_arcsec=_nan_to_null(_floats(out, "candidate_cross_track_sigma_arcsec")),
        ra_sigma_arcsec=_nan_to_null(_floats(out, "candidate_ra_sigma_arcsec")),
        dec_sigma_arcsec=_nan_to_null(_floats(out, "candidate_dec_sigma_arcsec")),
        position_angle_deg=_nan_to_null(_floats(out, "candidate_position_angle_deg")),
        post_along_track_sigma_arcsec=_nan_to_null(
            _floats(out, "candidate_post_along_track_sigma_arcsec")
        ),
        post_cross_track_sigma_arcsec=_nan_to_null(
            _floats(out, "candidate_post_cross_track_sigma_arcsec")
        ),
        radar_mode=modes,
        radar_snr=_nan_to_null(_floats(out, "candidate_radar_snr")),
        radar_range_km=_nan_to_null(_floats(out, "candidate_radar_range_km")),
        radar_provenance=out["candidate_radar_provenance"],
    )

    ephemeris = PlanEphemeris.from_kwargs(
        epochs=Epochs.from_kwargs(mjd=_floats(out, "ephemeris_epoch_mjd_tdb"), scale="tdb"),
        ra_deg=_floats(out, "ephemeris_ra_deg"),
        dec_deg=_floats(out, "ephemeris_dec_deg"),
    )

    resolved_code = int(out["resolved_method"])
    try:
        resolved_method = _KIND_BY_CODE[resolved_code]
    except KeyError:
        raise RuntimeError(
            f"engine returned an unknown resolved-method covariance-kind code: {resolved_code}"
        ) from None

    return PlanResult(
        orbit_id=str(out["orbit_id"]),
        metrics=_metrics_table(out),
        candidates=candidates,
        ephemeris=ephemeris,
        active_width=int(out["active_width"]),
        log_det_gain=float(out["log_det_gain"]),
        resolved_method=resolved_method,
    )


def compute_visibility(
    orbits: AnyOrbits,
    start: float,
    end: float,
    step_days: float,
    config: PlanningConfig | None = None,
) -> VisibilityResult:
    """Survey when and whether each orbit is observable from a set of sites.

    For every orbit in ``orbits`` and every observatory in
    ``config.observatories``, step across ``[start, end]`` (MJD TDB) on a
    ``step_days`` grid and report:

    - the per-epoch :class:`~empyrean.VisibilityPoints` (the four
      observability gates and the Moon's altitude, azimuth and illuminated
      fraction);
    - the contiguous observable :class:`~empyrean.VisibilityWindows`;
    - for a ground site, the :class:`~empyrean.Nights` product — twilight
      epochs, local solar midnight, the per-night observable intervals, and
      the :class:`~empyrean.VisibilityAnchors` (a chosen epoch and its
      Moon-aware circumstances under each policy).

    Unlike :func:`evaluate_plan`, this reads the whole of each
    :class:`~empyrean.ObservatoryConfig`: the elevation and darkness gates
    and the night-product twilight thresholds, not just the site-invariant
    pair. A :class:`~empyrean.VisibilityPoints` row's ``observable`` is the
    conjunction of all four gates — the site-aware verdict plan evaluation
    cannot give.

    Parameters
    ----------
    orbits : Orbits
        One or more orbits to survey. Each is surveyed independently; the
        result tables carry its ``orbit_id``.
    start, end : float
        Survey window, MJD TDB.
    step_days : float
        Grid step in days; each twilight / midnight / transit epoch is
        refined by bisection off this grid to one second.
    config : PlanningConfig, optional
        Must carry at least one :attr:`PlanningConfig.observatories` entry.

    Returns
    -------
    VisibilityResult
        The :class:`~empyrean.VisibilityPoints`,
        :class:`~empyrean.VisibilityWindows`, :class:`~empyrean.Nights` and
        :class:`~empyrean.VisibilityAnchors` tables, plus the per-observatory
        :class:`~empyrean.ObservatorySummary`.

    Raises
    ------
    ValueError
        ``config`` carries no observatory, or a twilight-threshold set is
        partial.
    RuntimeError
        The engine refused a survey — an invalid window, a non-descending
        twilight set (named by the observatory), an unresolvable station
        code, or a propagation failure.
    """
    from empyrean._empyrean_rs import _compute_visibility

    if config is None:
        config = PlanningConfig()
    if not config.observatories:
        raise ValueError(
            "compute_visibility needs at least one observatory in "
            "PlanningConfig.observatories — each carries the site's gates and "
            "the night-product twilight thresholds."
        )
    config_wire = config._to_wire_dict()

    point_cols: dict[str, list[Any]] = {}
    window_cols: dict[str, list[Any]] = {}
    night_cols: dict[str, list[Any]] = {}
    anchor_cols: dict[str, list[Any]] = {}
    obs_cols: dict[str, list[Any]] = {}
    anchor_joint_wires: list[dict[str, Any] | None] = []

    def extend(store: dict[str, list[Any]], out: ResultDict, prefix: str, orbit_id: str) -> None:
        keys = [k for k in out if k.startswith(prefix)]
        n = len(out[keys[0]]) if keys else 0
        store.setdefault("orbit_id", []).extend([orbit_id] * n)
        for k in keys:
            store.setdefault(k[len(prefix) :], []).extend(out[k])

    for i in range(len(orbits)):
        single = orbits[i : i + 1]
        orbit_dict = _orbits_to_dict(single)
        out: ResultDict = _compute_visibility(
            orbit_dict,
            config_wire,
            float(start),
            float(end),
            float(step_days),
            None,
        )
        orbit_id = str(out["orbit_id"])
        extend(point_cols, out, "point_", orbit_id)
        extend(window_cols, out, "window_", orbit_id)
        extend(night_cols, out, "night_", orbit_id)
        extend(anchor_cols, out, "anchor_", orbit_id)
        # Per-observatory metadata: the un-prefixed obs_* keys.
        n_obs = len(out["obs_code"])
        obs_cols.setdefault("orbit_id", []).extend([orbit_id] * n_obs)
        obs_cols.setdefault("obs_code", []).extend(out["obs_code"])
        obs_cols.setdefault("note", []).extend(out["obs_note"])
        obs_cols.setdefault("resolved_method", []).extend(
            CovarianceKind(_KIND_BY_CODE[int(c)]).value for c in out["obs_resolved_method"]
        )
        obs_cols.setdefault("earth_orientation_present", []).extend(
            out["obs_earth_orientation_present"]
        )
        obs_cols.setdefault("earth_orientation_bpc_epochs", []).extend(
            out["obs_earth_orientation_bpc_epochs"]
        )
        obs_cols.setdefault("earth_orientation_analytic_iau_2006_epochs", []).extend(
            out["obs_earth_orientation_analytic_iau_2006_epochs"]
        )
        # The anchor sky-covariance joints ride a parallel list (one wire dict
        # or None per anchor), joined positionally to the anchor rows.
        anchor_joint_wires.extend(out["anchor_joints"])

    points = VisibilityPoints.from_kwargs(
        orbit_id=point_cols.get("orbit_id", []),
        obs_code=point_cols.get("obs_code", []),
        epoch_mjd_tdb=point_cols.get("epoch", []),
        solar_elongation_deg=point_cols.get("solar_elongation_deg", []),
        elevation_deg=_nan_to_null(
            np.asarray(point_cols.get("elevation_deg", []), dtype=np.float64)
        ),
        sun_altitude_deg=_nan_to_null(
            np.asarray(point_cols.get("sun_altitude_deg", []), dtype=np.float64)
        ),
        observable=point_cols.get("observable", []),
        elongation_ok=point_cols.get("elongation_ok", []),
        mag=_nan_to_null(np.asarray(point_cols.get("mag", []), dtype=np.float64)),
        magnitude_ok=point_cols.get("magnitude_ok", []),
        elevation_ok=point_cols.get("elevation_ok", []),
        darkness_ok=point_cols.get("darkness_ok", []),
        moon_altitude_deg=_nan_to_null(
            np.asarray(point_cols.get("moon_altitude_deg", []), dtype=np.float64)
        ),
        moon_azimuth_deg=_nan_to_null(
            np.asarray(point_cols.get("moon_azimuth_deg", []), dtype=np.float64)
        ),
        moon_illuminated_fraction=_nan_to_null(
            np.asarray(point_cols.get("moon_illuminated_fraction", []), dtype=np.float64)
        ),
        lunar_elongation_deg=point_cols.get("lunar_elongation_deg", []),
    )

    windows = VisibilityWindows.from_kwargs(
        orbit_id=window_cols.get("orbit_id", []),
        obs_code=window_cols.get("obs_code", []),
        night_index=np.asarray(window_cols.get("night_index", []), dtype=np.int64),
        start_epoch_mjd_tdb=window_cols.get("start_epoch", []),
        end_epoch_mjd_tdb=window_cols.get("end_epoch", []),
        peak_elongation_deg=window_cols.get("peak_elongation_deg", []),
        brightest_mag=_nan_to_null(
            np.asarray(window_cols.get("brightest_mag", []), dtype=np.float64)
        ),
        hours_observable=window_cols.get("hours_observable", []),
        max_altitude_deg=_nan_to_null(
            np.asarray(window_cols.get("max_altitude_deg", []), dtype=np.float64)
        ),
        max_altitude_epoch_mjd_tdb=_nan_to_null(
            np.asarray(window_cols.get("max_altitude_epoch", []), dtype=np.float64)
        ),
        closing_gate=window_cols.get("closing_gate", []),
    )

    nights = Nights.from_kwargs(
        orbit_id=night_cols.get("orbit_id", []),
        obs_code=night_cols.get("obs_code", []),
        night_index=np.asarray(night_cols.get("index", []), dtype=np.int64),
        local_date_year=np.asarray(night_cols.get("year", []), dtype=np.int64),
        local_date_month=np.asarray(night_cols.get("month", []), dtype=np.int64),
        local_date_day=np.asarray(night_cols.get("day", []), dtype=np.int64),
        noon_start_mjd_tdb=night_cols.get("noon_start", []),
        noon_end_mjd_tdb=night_cols.get("noon_end", []),
        sunset_mjd_tdb=_nan_to_null(np.asarray(night_cols.get("sunset", []), dtype=np.float64)),
        civil_dusk_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("civil_dusk", []), dtype=np.float64)
        ),
        nautical_dusk_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("nautical_dusk", []), dtype=np.float64)
        ),
        astronomical_dusk_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("astronomical_dusk", []), dtype=np.float64)
        ),
        local_solar_midnight_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("local_solar_midnight", []), dtype=np.float64)
        ),
        astronomical_dawn_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("astronomical_dawn", []), dtype=np.float64)
        ),
        nautical_dawn_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("nautical_dawn", []), dtype=np.float64)
        ),
        civil_dawn_mjd_tdb=_nan_to_null(
            np.asarray(night_cols.get("civil_dawn", []), dtype=np.float64)
        ),
        sunrise_mjd_tdb=_nan_to_null(np.asarray(night_cols.get("sunrise", []), dtype=np.float64)),
        note=night_cols.get("note", []),
    )

    anchors = VisibilityAnchors.from_kwargs(
        orbit_id=anchor_cols.get("orbit_id", []),
        obs_code=anchor_cols.get("obs_code", []),
        night_index=np.asarray(anchor_cols.get("night_index", []), dtype=np.int64),
        policy=anchor_cols.get("policy", []),
        epoch_mjd_tdb=_nan_to_null(np.asarray(anchor_cols.get("epoch", []), dtype=np.float64)),
        observable=anchor_cols.get("observable", []),
        altitude_deg=_nan_to_null(
            np.asarray(anchor_cols.get("altitude_deg", []), dtype=np.float64)
        ),
        azimuth_deg=_nan_to_null(np.asarray(anchor_cols.get("azimuth_deg", []), dtype=np.float64)),
        sun_altitude_deg=_nan_to_null(
            np.asarray(anchor_cols.get("sun_altitude_deg", []), dtype=np.float64)
        ),
        solar_elongation_deg=_nan_to_null(
            np.asarray(anchor_cols.get("solar_elongation_deg", []), dtype=np.float64)
        ),
        phase_angle_deg=_nan_to_null(
            np.asarray(anchor_cols.get("phase_angle_deg", []), dtype=np.float64)
        ),
        lunar_elongation_deg=_nan_to_null(
            np.asarray(anchor_cols.get("lunar_elongation_deg", []), dtype=np.float64)
        ),
        moon_altitude_deg=_nan_to_null(
            np.asarray(anchor_cols.get("moon_altitude_deg", []), dtype=np.float64)
        ),
        moon_illuminated_fraction=_nan_to_null(
            np.asarray(anchor_cols.get("moon_illuminated_fraction", []), dtype=np.float64)
        ),
        mag=_nan_to_null(np.asarray(anchor_cols.get("mag", []), dtype=np.float64)),
        sigma_mag=_nan_to_null(np.asarray(anchor_cols.get("sigma_mag", []), dtype=np.float64)),
        sky_rate=_nan_to_null(np.asarray(anchor_cols.get("sky_rate", []), dtype=np.float64)),
        position_angle_deg=_nan_to_null(
            np.asarray(anchor_cols.get("position_angle_deg", []), dtype=np.float64)
        ),
        hour_angle_deg=_nan_to_null(
            np.asarray(anchor_cols.get("hour_angle_deg", []), dtype=np.float64)
        ),
        sky_covariance_kind=[
            None if int(c) < 0 else CovarianceKind(_KIND_BY_CODE[int(c)]).value
            for c in anchor_cols.get("sky_covariance_kind", [])
        ],
        joint=PackedJoints.from_wire_list(anchor_joint_wires),
    )

    observatories = ObservatorySummary.from_kwargs(
        orbit_id=obs_cols.get("orbit_id", []),
        obs_code=obs_cols.get("obs_code", []),
        note=obs_cols.get("note", []),
        resolved_method=obs_cols.get("resolved_method", []),
        earth_orientation_present=obs_cols.get("earth_orientation_present", []),
        earth_orientation_bpc_epochs=np.asarray(
            obs_cols.get("earth_orientation_bpc_epochs", []), dtype=np.int64
        ),
        earth_orientation_analytic_iau_2006_epochs=np.asarray(
            obs_cols.get("earth_orientation_analytic_iau_2006_epochs", []), dtype=np.int64
        ),
    )

    return VisibilityResult(
        points=points,
        windows=windows,
        nights=nights,
        anchors=anchors,
        observatories=observatories,
    )
