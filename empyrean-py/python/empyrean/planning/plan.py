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
from empyrean.planning.result import (
    _INT_TO_KIND,
    _INT_TO_RADAR_MODE,
    STAGE_POSTERIOR,
    STAGE_PRIOR,
    PlanCandidates,
    PlanEphemeris,
    PlanMetrics,
    PlannedObservation,
    PlanningConfig,
    PlanResult,
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

    return PlanMetrics.from_kwargs(
        stage=[stage for stage, _ in _METRIC_STAGES],
        position_sigma_km=column("position_sigma_km"),
        velocity_sigma_m_s=column("velocity_sigma_m_s"),
        semi_major_km=column("semi_major_km"),
        semi_minor_km=column("semi_minor_km"),
        log_det=column("log_det"),
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

    return PlanResult(
        orbit_id=str(out["orbit_id"]),
        metrics=_metrics_table(out),
        candidates=candidates,
        ephemeris=ephemeris,
        active_width=int(out["active_width"]),
    )
