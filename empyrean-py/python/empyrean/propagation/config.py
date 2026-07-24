"""Propagation configuration types.

Force-model tier, uncertainty method (and parameterized variants), and
the top-level :class:`PropagationConfig` consumed by
:func:`empyrean.propagate`. Shared scalar fields at the top, nested
sub-dataclasses for ``events`` / ``diagnostics`` / ``advanced``.
Sensible production defaults out of the box.
"""

import enum
from dataclasses import dataclass, field
from typing import Any

from empyrean.coordinates.enums import Frame, Origin
from empyrean.propagation.events import EventConfig


class ForceModelTier(str, enum.Enum):
    """Force model tier for propagation.

    Controls which physical effects are included in the force model.
    """

    APPROXIMATE = "approximate"
    """Point-mass planets + Moon + Pluto. Fast, for visualization."""

    BASIC = "basic"
    """Approximate + EIH GR + Sun J2. Good for distant objects."""

    STANDARD = "standard"
    """Basic + 16 asteroid perturbers + Earth J2-J4 + non-grav. Default."""


# Integer codes for the Rust boundary
_FORCE_MODEL_TO_INT = {
    ForceModelTier.APPROXIMATE: 0,
    ForceModelTier.BASIC: 1,
    ForceModelTier.STANDARD: 2,
    "approximate": 0,
    "basic": 1,
    "standard": 2,
}


class UncertaintyMethod(str, enum.Enum):
    """Uncertainty propagation method — shortcut identifiers for the
    zero-arg / default-params cases.

    Controls how the input covariance is mapped through the dynamics.
    For parameterized methods (``SIGMA_POINT``, ``MONTE_CARLO``,
    ``GAUSSIAN_MIXTURE``), passing the enum value selects the method with
    engine-default parameters. To customize parameters, pass the
    corresponding dataclass instance directly (:class:`SigmaPoint`,
    :class:`MonteCarlo`, :class:`GaussianMixture`) to
    :func:`~empyrean.propagate`.
    """

    FIRST_ORDER = "first_order"
    """First-order STM-based covariance propagation."""

    SECOND_ORDER = "second_order"
    """Second-order STT-based propagation (Hessians + second-order IP correction)."""

    SIGMA_POINT = "sigma_point"
    """Sigma-point (unscented) transform."""

    MONTE_CARLO = "monte_carlo"
    """Monte Carlo sampling from the input covariance."""

    AUTO = "auto"
    """Adaptive per-close-approach-window regime selection. Escalates the
    covariance method from first-order to a second-order (Jet2) covariance
    over close-approach windows, recording each transition as a
    ``CovarianceRegimeChange`` event."""

    GAUSSIAN_MIXTURE = "gaussian_mixture"
    """Adaptive Gaussian mixture (AGM). The engine recursively splits the
    input Gaussian into a mixture wherever the local nonlinearity exceeds
    a threshold, propagates each component, and recombines at the output.
    Its distinctive product is the mixture-corrected impact probability at
    close approaches; away from encounters the output-state covariance is
    the linear ``Φ·Σ·Φᵀ`` mapping (like ``SECOND_ORDER``), so for a
    well-determined object it reads back very close to ``FIRST_ORDER``.
    Reference: DeMars-Bishop-Jah (JGCD 2013)."""


@dataclass(frozen=True)
class SigmaPoint:
    """Sigma-point (unscented) transform.

    The engine uses the canonical Julier–Uhlmann 2N+1 unscented set,
    which is **parameter-free**: ``n_sigma`` and ``samples_per_plane`` are
    legacy knobs and only their default values are currently accepted. A
    non-default value is rejected loudly by the engine
    (``SigmaPointConstruction``) rather than silently reinterpreted — pass
    ``SigmaPoint()`` and leave both at their defaults.

    Parameters
    ----------
    n_sigma : float
        Legacy sigma-point spread. Must be ``1.0`` (the default); any
        other value raises.
    samples_per_plane : int
        Legacy points per coordinate-plane pair. Must be ``8`` (the
        default); any other value raises.
    """

    n_sigma: float = 1.0
    samples_per_plane: int = 8


@dataclass(frozen=True)
class MonteCarlo:
    """Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
        Number of samples drawn. Default 1000.
    seed : int, optional
        RNG seed. Default ``42`` (reproducible). Pass ``None`` for a
        non-deterministic source.
    """

    n_samples: int = 1000
    seed: int | None = 42


@dataclass(frozen=True)
class GaussianMixture:
    """Adaptive Gaussian mixture (AGM).

    The engine recursively splits the input Gaussian into a mixture of
    sub-Gaussians wherever the local nonlinearity exceeds ``threshold``,
    propagates each component through the dynamics, and recombines them at
    the output. Its distinctive product is the mixture-corrected impact
    probability at close approaches; away from encounters the output-state
    covariance is the linear ``Φ·Σ·Φᵀ`` mapping. Reference:
    DeMars-Bishop-Jah (JGCD 2013).

    Parameters
    ----------
    threshold : float
        Nonlinearity threshold above which the splitter fires. Default
        ``1.0``.
    max_depth : int
        Maximum recursion depth for nested splitting. Default ``3``.
    components_per_split : int
        Number of sub-Gaussians produced per split. The DeMars-Bishop-Jah
        splitting tables are tabulated only for odd counts (``3`` or
        ``5``). Default ``3``.
    """

    threshold: float = 1.0
    max_depth: int = 3
    components_per_split: int = 3


# Tag space matches the engine's UncertaintyMethod enum exactly.
_UNCERTAINTY_METHOD_TO_INT = {
    UncertaintyMethod.FIRST_ORDER: 0,
    UncertaintyMethod.SECOND_ORDER: 1,
    UncertaintyMethod.SIGMA_POINT: 2,
    UncertaintyMethod.MONTE_CARLO: 3,
    UncertaintyMethod.AUTO: 4,
    UncertaintyMethod.GAUSSIAN_MIXTURE: 5,
    "first_order": 0,
    "second_order": 1,
    "sigma_point": 2,
    "monte_carlo": 3,
    "auto": 4,
    "gaussian_mixture": 5,
}

# Inverse tag map for serializing a raw-int ``uncertainty_method`` back to its
# wire string. Without this, ``_to_wire_dict`` coerced every non-str/non-enum
# method (i.e. any legacy int) to "first_order" — the same silent-downgrade
# class of bug previously fixed for AUTO. Sigma-point / Monte-Carlo carry
# their per-variant params (n_sigma / samples_per_plane / n_samples / seed)
# via separate flat args on ``_propagate`` / ``_generate_ephemeris``, which
# are authoritative over this wire dict; the wire string here only names the
# method so a config-dict-only consumer resolves the real variant (with
# default params) rather than a silently substituted first-order covariance.
_INT_TO_UNCERTAINTY_METHOD = {
    0: "first_order",
    1: "second_order",
    2: "sigma_point",
    3: "monte_carlo",
    4: "auto",
    5: "gaussian_mixture",
}

_DATACLASS_TO_INT = {
    SigmaPoint: 2,
    MonteCarlo: 3,
    GaussianMixture: 5,
}


UncertaintyMethodLike = UncertaintyMethod | SigmaPoint | MonteCarlo | GaussianMixture | str
"""Type alias for inputs accepted by the ``uncertainty_method`` argument."""


@dataclass
class DiagnosticsConfig:
    """Per-trajectory diagnostic outputs (sensitivity, nonlinearity,
    Lyapunov, keyhole, bifurcation). All metrics off by default."""

    sensitivity: bool = False
    nonlinearity: bool = False
    lyapunov: bool = False
    keyholes: bool = False
    bifurcations: bool = False
    sample_stride: int = 0
    """Timeseries sampling stride: every Nth integration step.
    ``0`` → engine default (1)."""
    sensitivity_threshold: float | None = None
    """Emit a HighSensitivity event when the metric exceeds this."""
    lyapunov_threshold: float | None = None
    """Emit a ChaoticRegion event when the Lyapunov exponent exceeds this."""
    nonlinearity_threshold: float | None = None
    """Emit a HighNonlinearity event when the metric exceeds this
    (requires second-order uncertainty propagation)."""


class IntegratorChoice(str, enum.Enum):
    """Integrator backend selector.

    IAS15 is intentionally not available in this distribution —
    callers needing IAS15 must build a custom engine.
    """

    GR15 = "gr15"
    """Gauss-Radau 15. Default. Tightest accuracy."""

    DOP853 = "dop853"
    """Dormand-Prince 8(5,3). ~1.4× faster than GR15 with
    looser median Horizons error (~358 m vs GR15's ~35 m)."""


@dataclass
class OriginSwitchingConfig:
    """Trajectory splitting at body acceleration-dominance boundaries
    (Amato/Baù/Bombardelli 2017 §6). Default enabled at the empyrean
    wrapper layer for the planetary-encounter workflow.

    When ``enabled = True`` the integrator re-centers on the dominant
    body when its gravitational acceleration on the particle exceeds
    the integration origin's. This dramatically improves accuracy
    through deep planetary encounters by keeping the integrated radius
    vector small (body-relative) instead of the catastrophically-
    cancelling 1-AU-scale Sun-relative difference.
    """

    enabled: bool = True
    """Enable trajectory splitting. Default ``True`` (matches the Rust
    wrapper's brand default for the planetary-encounter workflow)."""
    hysteresis: float = 0.2
    """Hysteresis band around the acceleration-ratio crossover
    (``0.2`` = ±20 %)."""


@dataclass
class AdvancedIntegratorConfig:
    """Integrator-tuning knobs.

    Defaults are calibrated for production. Most callers don't touch
    this — :class:`PropagationConfig.advanced` exists to make the
    surface complete and to enable bespoke runs (custom step bounds
    for tight encounters, dense output for visualization, etc.).
    """

    integrator: IntegratorChoice = IntegratorChoice.GR15
    """Integrator backend. Default :attr:`IntegratorChoice.GR15`."""
    epsilon: float = 1e-9
    """Adaptive integrator truncation-error tolerance (relative b₆ for
    GR15, rtol for DOP853 paired with a fixed atol = 1e-14)."""
    dt_initial: float | None = None
    """Initial step size in days. ``None`` = auto from orbital timescale."""
    dt_min: float | None = None
    """Minimum allowed step size in days. ``None`` = auto."""
    encounter_timescale_divisor: float = 1000.0
    """Divisor K for encounter dynamical-timescale step floor."""
    max_steps: int = 10_000_000
    max_dense_steps: int = 100_000
    cache_integrator_steps: bool = False
    """Enable dense-state interpolation between integration steps —
    used for light-time iteration, off-step state queries, and event
    refinement around close approaches."""
    origin_switching: OriginSwitchingConfig = field(default_factory=OriginSwitchingConfig)
    """Origin-switching trajectory splitting. Default enabled."""


@dataclass
class PropagationConfig:
    """Configuration for orbit propagation.

    Sensible defaults out of the box; adjust fields when you need to
    deviate. Default output frame is :attr:`Frame.ECLIPTICJ2000`; set
    ``frame=Frame.ICRF`` for ICRF output.

    Parameters
    ----------
    force_model : ForceModelTier
        Force-model preset. See :class:`ForceModelTier` for the
        available tiers and what each includes.
    excluded_perturbers : list[Origin | str]
        Bodies to omit from the perturber set. Useful when propagating
        an asteroid that the force model would otherwise include as a
        perturber — exclude it from its own perturber set so it does
        not self-attract. Pass :class:`Origin` instances (e.g.
        ``[Origin.asteroid(1)]``) or canonical names.
    uncertainty_method : UncertaintyMethod | SigmaPoint | MonteCarlo | GaussianMixture | str
        Uncertainty propagation method. See :class:`UncertaintyMethod`
        and the parameterized variants
        (:class:`SigmaPoint`, :class:`MonteCarlo`, :class:`GaussianMixture`).
    compute_stm : bool
        Produce STMs even when the input has no covariance.
    frame : Frame
        Output reference frame.
    events : EventConfig
        Event-detection configuration.
    diagnostics : DiagnosticsConfig
        Per-trajectory diagnostic outputs.
    num_threads : int, optional
        Threads for multi-orbit propagation. ``None`` (default) and
        ``0`` both mean "use all available cores" (Rayon default);
        positive ``n`` pins exactly ``n`` threads. Each orbit is
        integrated on a single thread; parallelism is across orbits,
        not within a single trajectory.
    advanced : AdvancedIntegratorConfig
        Integrator-tuning knobs (rarely touched).
    """

    force_model: ForceModelTier = ForceModelTier.STANDARD
    excluded_perturbers: list[Origin | str] = field(default_factory=list)
    uncertainty_method: UncertaintyMethodLike = UncertaintyMethod.FIRST_ORDER
    compute_stm: bool = False
    frame: Frame = Frame.ECLIPTICJ2000
    events: EventConfig = field(default_factory=EventConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    num_threads: int | None = None
    advanced: AdvancedIntegratorConfig = field(default_factory=AdvancedIntegratorConfig)

    # ── Back-compat shim ─────────────────────────────────────
    @property
    def epsilon(self) -> float | None:
        """Back-compat alias for ``advanced.epsilon``. Returns ``None``
        if the integrator tolerance is at its default; otherwise
        returns the override.
        """
        eps = self.advanced.epsilon
        return None if eps == 1e-9 else eps

    @epsilon.setter
    def epsilon(self, value: float | None) -> None:
        if value is None:
            self.advanced.epsilon = 1e-9
        else:
            self.advanced.epsilon = value

    def _to_wire_dict(self) -> dict[str, Any]:
        """Serialize to the nested dict shape the binding consumes.

        Internal — called by :func:`empyrean.propagate` and
        :func:`empyrean.generate_ephemeris` to marshal config across
        the FFI boundary. For user-facing serialization (saving config
        to JSON, displaying it in a notebook, etc.), use
        :func:`dataclasses.asdict`.
        """
        from empyrean._convert import origin_to_naif

        events = self.events
        diag = self.diagnostics
        adv = self.advanced
        um_method = self.uncertainty_method
        if isinstance(um_method, enum.Enum):
            um: Any = um_method.value
        elif isinstance(um_method, SigmaPoint):
            # The wire string only names the method; the per-variant params
            # (n_sigma / samples_per_plane) ride on the authoritative flat
            # args, so serialize the method name rather than the old lossy
            # "first_order".
            um = "sigma_point"
        elif isinstance(um_method, MonteCarlo):
            um = "monte_carlo"
        elif isinstance(um_method, GaussianMixture):
            # The wire string only names the method; the per-variant params
            # (threshold / max_depth / components_per_split) ride on the
            # authoritative flat args, so serialize the method name.
            um = "gaussian_mixture"
        elif isinstance(um_method, bool):
            um = um_method  # not a valid method; falls through to "first_order"
        elif isinstance(um_method, int):
            # Legacy raw-int tag (0..=5). Map back to the wire string instead
            # of silently coercing to "first_order".
            um = _INT_TO_UNCERTAINTY_METHOD.get(um_method, um_method)
        else:
            um = um_method
        return {
            "force_model": _enum_str(self.force_model),
            "excluded_perturbers_naif": [origin_to_naif(o) for o in self.excluded_perturbers],
            "uncertainty_method": um if isinstance(um, str) else "first_order",
            "compute_stm": self.compute_stm,
            "frame": _enum_str(self.frame),
            "events": {
                "close_approaches": events.close_approaches,
                "impacts": events.impacts,
                "atmospheric": events.atmospheric,
                "possible_impacts": events.possible_impacts,
                "shadow_events": events.shadow_events,
                "body_filter_naif": [origin_to_naif(o) for o in (events.body_filter or [])],
                "dense_output": events.dense_output,
                "dense_output_cadence_days": events.dense_output_cadence_days,
            },
            "diagnostics": {
                "sensitivity": diag.sensitivity,
                "nonlinearity": diag.nonlinearity,
                "lyapunov": diag.lyapunov,
                "keyholes": diag.keyholes,
                "bifurcations": diag.bifurcations,
                "sample_stride": diag.sample_stride,
                "sensitivity_threshold": diag.sensitivity_threshold,
                "lyapunov_threshold": diag.lyapunov_threshold,
                "nonlinearity_threshold": diag.nonlinearity_threshold,
            },
            "num_threads": self.num_threads,
            "advanced": {
                "integrator": _enum_str(adv.integrator),
                "epsilon": adv.epsilon,
                "dt_initial": adv.dt_initial,
                "dt_min": adv.dt_min,
                "encounter_timescale_divisor": adv.encounter_timescale_divisor,
                "max_steps": adv.max_steps,
                "max_dense_steps": adv.max_dense_steps,
                "cache_integrator_steps": adv.cache_integrator_steps,
                "origin_switching": {
                    "enabled": adv.origin_switching.enabled,
                    "hysteresis": adv.origin_switching.hysteresis,
                },
            },
        }


def _enum_str(v: enum.Enum | str) -> str:
    """Coerce a string-Enum or bare string to a string."""
    return v.value if isinstance(v, enum.Enum) else str(v)
