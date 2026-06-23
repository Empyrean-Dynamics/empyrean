"""Orbit determination result types and configuration.

This module mirrors the Rust wrapper's ``empyrean::ODConfig`` and
``empyrean::DetermineResult`` field-for-field, with the same nested
structure (no flattening). ``ODConfig()`` defaults are identical to
``ODConfig::default()`` on the Rust side so no surprises round-
tripping through the C ABI.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np

from empyrean.coordinates.enums import Frame, Origin
from empyrean.od.residuals import (
    AcceptabilityReport,
    ObservationResults,
    ResidualSummary,
    StationBiases,
)
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)

# JSON-like value type for the nested wire dicts marshaled across the
# C ABI boundary (str / numeric / bool / None leaves, plus nested dicts
# and lists of the same).
WireValue: TypeAlias = str | int | float | bool | None | list["WireValue"] | dict[str, "WireValue"]


# ── Enums ────────────────────────────────────────────────────


class ForceModelTier(str, Enum):
    """Force-model tier for OD propagation."""

    APPROXIMATE = "approximate"
    BASIC = "basic"
    STANDARD = "standard"


class SolveForParams(str, Enum):
    """Parameters to solve for in differential correction.

    Mirrors ``scott::od::SolveForParams``.
    """

    STATE_ONLY = "state_only"
    """Solve only for the 6-element state vector."""
    STATE_AND_NONGRAV = "state_and_nongrav"
    """Solve for state + (A1, A2, A3) non-grav coefficients (9 params)."""
    AUTO = "auto"
    """Start with state-only, escalate to 9-param on poor fit. Tuned
    via :class:`AutoEscalationPolicy`."""


class CovarianceRepresentation(str, Enum):
    """Coordinate basis the OD output covariance is reported in."""

    CARTESIAN = "cartesian"
    KEPLERIAN = "keplerian"
    COMETARY = "cometary"
    SPHERICAL = "spherical"


# ── Output epoch (tagged-union dataclass) ────────────────────


class OutputEpochMode(str, Enum):
    """How :class:`OutputEpoch` selects the fitted-orbit epoch.

    Mirrors the discriminant on ``scott::od::OutputEpoch``.
    """

    MID_ARC = "mid_arc"
    """Midpoint of the observation arc (default). Resolved against
    the active observation set (not the full input arc) so multi-year
    arcs whose mid-arc target lies in a chaotic interval keep the
    integrator anchor inside the IOD opposition window."""
    LAST_OBSERVATION = "last_observation"
    """Epoch of the last observation, resolved against the active set."""
    IOD_EPOCH = "iod_epoch"
    """Anchor at the IOD-derived epoch — the state stays where the
    initial-orbit determination produced it. Matches OrbFit's
    ``epoch.eq0`` and find_orb's "anchor at most recent good fit"
    pattern."""
    EXPLICIT = "explicit"
    """:attr:`OutputEpoch.mjd_tdb` is honored."""


@dataclass
class OutputEpoch:
    """Output epoch for the fitted orbit. Mirrors
    ``scott::od::OutputEpoch``.
    """

    mode: OutputEpochMode = OutputEpochMode.MID_ARC
    mjd_tdb: float | None = None
    """Required when ``mode == OutputEpochMode.EXPLICIT``."""


# ── Origin policy (tagged-union dataclass) ───────────────────


class OriginPolicyMode(str, Enum):
    """How :class:`OriginPolicy` selects the central body for IOD + DC.

    Mirrors the discriminant on ``scott::od::OriginPolicy``.
    """

    AUTO = "auto"
    """Selects the central body (heliocentric vs Earth-centric)
    automatically. Default."""
    EXPLICIT = "explicit"
    """Pin IOD + DC to a specific central body — set
    :attr:`OriginPolicy.origin` to the desired :class:`Origin`. Skips
    the cascade."""


@dataclass
class OriginPolicy:
    """Origin-policy selector for the OD pipeline. Mirrors
    ``scott::od::OriginPolicy``.

    Auto handles TCOs / minimoons / geocentric impactors / chaotic-
    capture interiors without per-object regime classification by the
    caller. Explicit is required for cataloged satellites where
    heliocentric Gauss is unphysical, and recommended for pipelines
    that already know the regime.
    """

    mode: OriginPolicyMode = OriginPolicyMode.AUTO
    origin: "Origin | str | None" = None
    """The central body to pin to. Required when
    ``mode == OriginPolicyMode.EXPLICIT``. Pass an :class:`Origin`
    instance or a canonical name string."""


# ── Nested config bundles ────────────────────────────────────


@dataclass
class IODConfig:
    """IOD ranging tuning. Mirrors the IOD section of
    ``scott::od::ODConfig``. Defaults match ``ODConfig::default()``."""

    max_triplet_attempts: int = 10
    max_triplet_span_days: float = 30.0
    opposition_gap_days: float = 90.0
    """Set to a negative value to disable opposition splitting."""
    max_iod_arc_days: float = 30.0
    """Maximum arc length (days) used for IOD."""
    curvature_snr_threshold: float = 3.0
    max_iod_fractional_sigma_a: float = 1.0


@dataclass
class AutoEscalationPolicy:
    """Trigger thresholds for :attr:`SolveForParams.AUTO` escalation.
    Mirrors ``scott::od::AutoEscalationPolicy``."""

    reduced_chi2: float = 10.0
    at_ct_ratio: float = 3.0
    min_arc_days: float = 30.0
    min_n_obs: int = 50


@dataclass
class AcceptabilityThresholds:
    """Thresholds for the post-DC fit-acceptability sub-checks. Mirrors
    ``scott::od::AcceptabilityThresholds``.

    Defaults are tuned for production NEO survey work; tighten for
    Sentry-grade impact-monitoring orbits (e.g. ``fractional_sigma_a =
    1e-4``), loosen for short-arc discovery fits.
    """

    reduced_chi2: float = 3.0
    rms_arcsec: float = 1.0
    at_ct_ratio: float = 3.0
    min_arc_days: float = 7.0
    fractional_sigma_a: float = 0.1


# ── Weighting (mirrors empyrean::WeightingConfig) ─────────────


class WeightingPreset(str, Enum):
    """Preset selector for :class:`WeightingConfig`.

    Picking a preset seeds the layer chain with scott's curated
    layers; entries in :attr:`WeightingConfig.additional_layers`
    are appended in order.
    """

    NONE = "none"
    """No preset — only ``additional_layers`` apply."""
    VFC17 = "vfc17"
    """Vereš, Farnocchia, Chesley et al. 2017 station floors +
    nightly de-weighting at floor-σ policy. Production default."""
    NEODYS = "neodys"
    """NEODyS production preset."""


class SigmaPolicy(str, Enum):
    """How a weighting layer's σ combines with the per-observation
    reported σ. Mirrors ``scott::weighting::SigmaPolicy``."""

    DEFAULT_ONLY = "default_only"
    """``σ = reported`` if present, else ``σ = rule``. Default."""
    FLOOR = "floor"
    """``σ = max(reported, rule)``. VFC17 / NEODyS production policy."""


class WeightingLayerKind(str, Enum):
    """Discriminator for :class:`WeightingLayer` variants."""

    OBSERVATORY_RULE = "observatory_rule"
    NIGHTLY_DEWEIGHTING = "nightly_deweighting"


@dataclass
class WeightingLayer:
    """One element of the weighting pipeline. Tagged-union shape —
    the active fields depend on :attr:`kind`.

    Mirrors ``scott::weighting::WeightingLayer``.
    """

    kind: WeightingLayerKind = WeightingLayerKind.OBSERVATORY_RULE

    # ── ObservatoryRule fields ─────────────────────────────────
    obs_code: str = ""
    """MPC observatory code (e.g. ``"F51"``)."""
    sigma: tuple[float, float] = (1.0, 1.0)
    """1σ (RA·cos(δ), Dec) in arcseconds."""
    start_epoch_mjd_tdb: float | None = None
    """Start of applicable time range (MJD TDB). ``None`` =
    unbounded."""
    end_epoch_mjd_tdb: float | None = None
    """End of applicable time range (MJD TDB). ``None`` =
    unbounded."""
    scale: float = 1.0
    """Scale factor on the final weight."""

    # ── NightlyDeweighting fields ──────────────────────────────
    max_gap_days: float = 0.5
    """Max gap (days) between observations to count as the same
    night (NightlyDeweighting only)."""


def _default_weighting_layers() -> list["WeightingLayer"]:
    # Mirrors `scott::od::ODConfig::default()` — the production preset is
    # VFC17 station floors WITH NightlyDeweighting (1/√N within 0.5 days)
    # appended. Without the nightly layer, OD on objects with clustered
    # same-station-same-night observations diverges from `validate-core`'s
    # direct-scott path because the rejection layer treats high-weight
    # cluster residuals as outliers.
    return [
        WeightingLayer(
            kind=WeightingLayerKind.NIGHTLY_DEWEIGHTING,
            max_gap_days=0.5,
        )
    ]


@dataclass
class WeightingConfig:
    """Observation weighting pipeline. Mirrors
    ``empyrean::WeightingConfig``.

    Default = enabled with the VFC17 preset + a NightlyDeweighting layer
    (production hot path; matches ``scott::od::ODConfig::default()``).
    Set ``enabled=False`` for uniform 1″ weighting; pick a different
    preset or replace ``additional_layers`` for custom pipelines.
    """

    enabled: bool = True
    preset: WeightingPreset = WeightingPreset.VFC17
    default_sigma_arcsec: float = 1.0
    """Default 1σ when no rule applies (arcsec). Used only when
    ``preset = NONE``."""
    sigma_policy: SigmaPolicy | None = None
    """Sigma combination policy override. ``None`` = use the
    preset's policy."""
    additional_layers: list[WeightingLayer] = field(default_factory=_default_weighting_layers)
    """Layers appended to the preset's chain."""


# ── Debiasing (mirrors empyrean::DebiasingConfig) ─────────────


class DebiasingResolution(str, Enum):
    """Healpix resolution of a debiasing table."""

    STANDARD = "standard"
    """NSIDE = 64, ~35 MB. Production default."""
    HIRES = "hires"
    """NSIDE = 256, ~567 MB."""


@dataclass
class DebiasingConfig:
    """Catalog-bias-correction configuration. Mirrors scott's
    ``Option<Arc<DebiasingTable>>`` field on ``ODConfig``.

    Default = enabled at standard resolution with no explicit path
    (uses the DataManager default lookup at
    ``~/.empyrean/data/bias.dat``). Set ``enabled=False`` to disable
    catalog debiasing entirely.
    """

    enabled: bool = True
    resolution: DebiasingResolution = DebiasingResolution.STANDARD
    bias_dat_path: str | None = None


class RejectionKind(str, Enum):
    """Outlier-rejection strategy selector.

    Pick the variant that matches the reference pipeline you're
    interoperating with — `ADAPTIVE` is the production default
    (information-loss-weighted, Layer 3); `CMC2003` matches the
    OrbFit / NEODyS χ²-with-hysteresis scheme of Carpino, Milani
    & Chesley (2003).
    """

    ADAPTIVE = "adaptive"
    CMC2003 = "cmc2003"


@dataclass
class RejectionConfig:
    """Outlier-rejection configuration. The active fields are
    determined by :attr:`kind`.

    Mirrors ``scott::rejection::RejectionStrategy`` plus the upstream
    ``max_rejection_passes`` knob. Set ``enabled=False`` to disable
    the rejection pass entirely.
    """

    enabled: bool = True
    kind: RejectionKind = RejectionKind.ADAPTIVE
    """Strategy selector. Default :attr:`RejectionKind.ADAPTIVE`."""

    # ── Adaptive (kind = ADAPTIVE) ──────────────────────────
    chi2_base: float = 9.21
    """χ²(2 dof, p = 0.01) — Carpino, Milani & Chesley 2003.
    Adaptive rejection only."""
    lambda_: float = 1.0
    """Adaptation strength. ``0`` reduces to standard χ² rejection;
    higher values protect informative observations more.
    Adaptive rejection only."""
    max_threshold: float = 100.0
    """Effective-threshold cap for adaptive rejection."""

    # ── CMC2003 (kind = CMC2003) ────────────────────────────
    chi2_rej: float = 8.0
    """χ²-with-hysteresis upper threshold — reject when χ² > chi2_rej.
    CMC2003 only. Default 8.0 (≈ 98.2% confidence at 2 DOF)."""
    chi2_rec: float = 7.0
    """χ²-with-hysteresis lower threshold — recover a previously-
    rejected observation when χ² < chi2_rec. CMC2003 only. Must
    satisfy ``chi2_rec < chi2_rej`` for hysteresis to break cycles.
    Default 7.0 (≈ 96.9% confidence at 2 DOF)."""

    # ── Both ────────────────────────────────────────────────
    max_passes: int = 4


@dataclass
class StationRaDecConfig:
    """Per-station RA/Dec bias-fit configuration.

    Schur-eliminated nuisance parameters that absorb per-station
    pointing offsets, fit alongside the orbit. Default thresholds
    target modern survey arcs.

    Attributes
    ----------
    sigma_prior_arcsec : float
        1-σ Gaussian prior on the per-station offset, in arcseconds.
        Default 0.3.
    min_obs_per_station : int
        Minimum observations per station required to allocate a
        bias parameter for that station. Default 5.
    """

    sigma_prior_arcsec: float = 0.3
    min_obs_per_station: int = 5


# ── Top-level config ─────────────────────────────────────────


@dataclass
class ODConfig:
    """Unified orbit-determination configuration.

    Sensible production defaults out of the box:

      - VFC17 station weighting + nightly de-weighting
        (:attr:`WeightingConfig.preset`)
      - EFCC2020 catalog debiasing enabled
        (:attr:`DebiasingConfig.enabled`)
      - :attr:`SolveForParams.AUTO` (escalates 6→9 parameters on poor fit)
      - Adaptive outlier rejection enabled, ``max_passes = 4``
    """

    # ── Shared (all OD entry points) ────────────────────────
    force_model: ForceModelTier = ForceModelTier.STANDARD
    epsilon: float = 1e-9
    """Adaptive integrator truncation-error tolerance."""
    max_light_time_iterations: int = 3
    num_threads: int = 0
    """``0`` = use all available cores."""
    frame: Frame = Frame.ICRF
    weighting: "WeightingConfig" = field(default_factory=lambda: WeightingConfig())
    """Observation weighting pipeline. Default = enabled + VFC17
    preset. See :class:`WeightingConfig` for full layered control."""
    debiasing: "DebiasingConfig" = field(default_factory=lambda: DebiasingConfig())
    """Catalog-bias-correction configuration. Default = EFCC2020
    standard resolution loaded from the engine's default data location.
    See :class:`DebiasingConfig`."""
    excluded_perturbers: list[Origin | str] = field(default_factory=list)
    """Bodies to omit from the perturber set. Pass :class:`Origin`
    instances (or canonical names). Useful when fitting an asteroid
    that the force model would otherwise include as a perturber —
    e.g. fitting Eros while excluding ``Origin.asteroid(433)``."""
    origin: OriginPolicy = field(default_factory=OriginPolicy)
    """Origin-policy selector. Default :attr:`OriginPolicyMode.AUTO`
    (heliocentric → geocentric Earth cascade). Set
    ``origin=OriginPolicy(mode=OriginPolicyMode.EXPLICIT, origin=Origin.EARTH)``
    to pin the pipeline to a specific central body for catalog
    satellites or regime-classified workflows."""

    # ── IOD (determine only) ────────────────────────────────
    iod: IODConfig = field(default_factory=IODConfig)

    # ── Differential correction ─────────────────────────────
    output_epoch: OutputEpoch = field(default_factory=OutputEpoch)
    max_iterations: int = 100
    convergence_tol: float = 1e-5
    use_stm_cache: bool = True
    solve_for: SolveForParams = SolveForParams.AUTO
    auto_escalation: AutoEscalationPolicy = field(default_factory=AutoEscalationPolicy)
    acceptability: AcceptabilityThresholds = field(default_factory=AcceptabilityThresholds)
    fit_station_biases: bool = False
    """Enable Schur-eliminated per-station RA/Dec bias fitting."""
    station_radec: StationRaDecConfig = field(default_factory=StationRaDecConfig)
    use_span_grouping: bool = False

    # ── Rejection ──────────────────────────────────────────
    rejection: RejectionConfig = field(default_factory=RejectionConfig)
    auto_force_model: bool = False
    """Auto-select force-model tier from IOD orbital elements."""
    output_representation: CovarianceRepresentation = CovarianceRepresentation.CARTESIAN

    def _to_wire_dict(self) -> dict[str, WireValue]:
        """Serialize to the nested dict shape the binding consumes.

        Internal — called by :func:`empyrean.determine` /
        :func:`empyrean.evaluate` / :func:`empyrean.refine` to marshal
        the config across the FFI boundary. For user-facing
        serialization (saving config to JSON, displaying it in a
        notebook, etc.), use :func:`dataclasses.asdict`.
        """
        return {
            "force_model": _enum_value(self.force_model),
            "epsilon": self.epsilon,
            "max_light_time_iterations": self.max_light_time_iterations,
            "num_threads": self.num_threads,
            "frame": _enum_value(self.frame),
            "weighting": _weighting_to_dict(self.weighting),
            "debiasing": _debiasing_to_dict(self.debiasing),
            "excluded_perturbers_naif": [_origin_to_naif(o) for o in self.excluded_perturbers],
            "origin": {
                "mode": _enum_value(self.origin.mode),
                "naif_id": (
                    _origin_to_naif(self.origin.origin) if self.origin.origin is not None else None
                ),
            },
            "iod": {
                "max_triplet_attempts": self.iod.max_triplet_attempts,
                "max_triplet_span_days": self.iod.max_triplet_span_days,
                "opposition_gap_days": self.iod.opposition_gap_days,
                "max_iod_arc_days": self.iod.max_iod_arc_days,
                "curvature_snr_threshold": self.iod.curvature_snr_threshold,
                "max_iod_fractional_sigma_a": self.iod.max_iod_fractional_sigma_a,
            },
            "output_epoch": {
                "mode": self.output_epoch.mode,
                "mjd_tdb": self.output_epoch.mjd_tdb,
            },
            "max_iterations": self.max_iterations,
            "convergence_tol": self.convergence_tol,
            "use_stm_cache": self.use_stm_cache,
            "solve_for": _enum_value(self.solve_for),
            "auto_escalation": {
                "reduced_chi2": self.auto_escalation.reduced_chi2,
                "at_ct_ratio": self.auto_escalation.at_ct_ratio,
                "min_arc_days": self.auto_escalation.min_arc_days,
                "min_n_obs": self.auto_escalation.min_n_obs,
            },
            "acceptability": {
                "reduced_chi2": self.acceptability.reduced_chi2,
                "rms_arcsec": self.acceptability.rms_arcsec,
                "at_ct_ratio": self.acceptability.at_ct_ratio,
                "min_arc_days": self.acceptability.min_arc_days,
                "fractional_sigma_a": self.acceptability.fractional_sigma_a,
            },
            "fit_station_biases": self.fit_station_biases,
            "station_radec": {
                "sigma_prior_arcsec": self.station_radec.sigma_prior_arcsec,
                "min_obs_per_station": self.station_radec.min_obs_per_station,
            },
            "use_span_grouping": self.use_span_grouping,
            "rejection": {
                "enabled": self.rejection.enabled,
                "kind": _enum_value(self.rejection.kind),
                "chi2_base": self.rejection.chi2_base,
                # Python alias `lambda_` keeps the keyword from
                # collising with the language; wire format uses bare
                # `lambda` so it round-trips through Rust unchanged.
                "lambda": self.rejection.lambda_,
                "max_threshold": self.rejection.max_threshold,
                "chi2_rej": self.rejection.chi2_rej,
                "chi2_rec": self.rejection.chi2_rec,
                "max_passes": self.rejection.max_passes,
            },
            "auto_force_model": self.auto_force_model,
            "output_representation": _enum_value(self.output_representation),
        }


def _weighting_to_dict(w: WeightingConfig) -> dict[str, WireValue]:
    """Serialize a :class:`WeightingConfig` to the wire dict the
    PyO3 bridge expects."""
    return {
        "enabled": w.enabled,
        "preset": _enum_value(w.preset),
        "default_sigma_arcsec": w.default_sigma_arcsec,
        "sigma_policy": _enum_value(w.sigma_policy) if w.sigma_policy is not None else None,
        "additional_layers": [_weighting_layer_to_dict(layer) for layer in w.additional_layers],
    }


def _weighting_layer_to_dict(layer: WeightingLayer) -> dict[str, WireValue]:
    return {
        "kind": _enum_value(layer.kind),
        "obs_code": layer.obs_code,
        "sigma": list(layer.sigma),
        "start_epoch_mjd_tdb": layer.start_epoch_mjd_tdb,
        "end_epoch_mjd_tdb": layer.end_epoch_mjd_tdb,
        "scale": layer.scale,
        "max_gap_days": layer.max_gap_days,
    }


def _debiasing_to_dict(d: DebiasingConfig) -> dict[str, WireValue]:
    return {
        "enabled": d.enabled,
        "resolution": _enum_value(d.resolution),
        "bias_dat_path": d.bias_dat_path,
    }


def _enum_value(v: Enum | str) -> str:
    """Accept either an Enum or a bare string; return a string."""
    return str(v.value) if isinstance(v, Enum) else str(v)


def _origin_to_naif(o: Origin | str) -> int:
    """Internal — resolve an :class:`Origin` (or canonical name) to the
    integer body code the binding wire format uses."""
    from empyrean._convert import origin_to_naif

    return origin_to_naif(o)


# ── Result types ─────────────────────────────────────────────

# Re-export StationBiases at the result module so callers can import
# it alongside the other OD types.
__all__ = []

# Any of the four orbit flavors that can come back from a determine /
# refine, depending on `ODConfig.output_representation`.
OrbitsTable = CartesianOrbits | KeplerianOrbits | CometaryOrbits | SphericalOrbits


@dataclass
class EvaluateResult:
    """Result of orbit evaluation (residuals only, no fitting)."""

    observations: ObservationResults
    summary: ResidualSummary


@dataclass
class DetermineResult:
    """Result of orbit determination — returned by both
    :func:`~empyrean.od.determine.determine` (full IOD + DC pipeline)
    and :func:`~empyrean.od.determine.refine` (Bayesian-prior fit
    against an existing orbit + covariance).

    Mirrors the Rust wrapper's ``empyrean::DetermineResult``.
    """

    orbit: OrbitsTable
    """Fitted orbit. Coordinate flavor matches
    :attr:`ODConfig.output_representation`."""
    observations: ObservationResults
    """Per-observation residuals + rejection / influence diagnostics."""
    summary: ResidualSummary
    iterations: int
    update_norm: float
    converged: bool
    covariance: np.ndarray
    """Fitted 6×6 state covariance, in :attr:`covariance_representation`."""
    covariance_representation: CovarianceRepresentation
    covariance_9x9: np.ndarray | None
    """Full 9×9 covariance over (state, A1, A2, A3) when solving for non-grav."""
    non_grav_delta: np.ndarray | None
    """Cumulative non-grav corrections (ΔA1, ΔA2, ΔA3) when present."""
    rejection_passes: int
    num_oppositions_fit: int
    force_model_used: ForceModelTier
    solve_for_used: SolveForParams
    acceptability: AcceptabilityReport
    station_biases: StationBiases
    """Per-station fitted nuisance biases when
    :attr:`ODConfig.fit_station_biases` was active. Empty quivr table
    otherwise."""
