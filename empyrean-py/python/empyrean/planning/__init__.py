"""Observation planning: rank candidate follow-up observations by information gain.

:func:`evaluate_plan` takes one barycentric, Cartesian, covariance-bearing
orbit plus a list of :class:`PlannedObservation` candidates and reports
what each would buy. Its Notes section records what the engine offers
that this package does not expose — the non-gravitational σ(A2) planning
variant, the visibility survey, batch evaluation, and the encounter
B-plane — and why a non-gravitational orbit is evaluated state-only.
"""

from empyrean.planning.plan import evaluate_plan
from empyrean.planning.result import (
    STAGE_POSTERIOR,
    STAGE_PRIOR,
    ObservatoryConfig,
    PlanCandidates,
    PlanEphemeris,
    PlanMetrics,
    PlannedObservation,
    PlannedObservationKind,
    PlanningConfig,
    PlanResult,
    RadarMode,
    RadarStation,
)

__all__ = [
    "STAGE_POSTERIOR",
    "STAGE_PRIOR",
    "ObservatoryConfig",
    "PlanCandidates",
    "PlanEphemeris",
    "PlanMetrics",
    "PlanResult",
    "PlannedObservation",
    "PlannedObservationKind",
    "PlanningConfig",
    "RadarMode",
    "RadarStation",
    "evaluate_plan",
]
