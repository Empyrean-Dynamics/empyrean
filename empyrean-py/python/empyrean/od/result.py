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
    EXPLICIT = "explicit"
    """An explicit per-axis solve requested via ``solve_for_flags``
    (:class:`SolveFor`) — e.g. Marsden + DT, or state + AMRAT. Reported
    on :attr:`DetermineResult.solve_for_used` when the fit used the
    explicit flag surface rather than one of the coarse presets above."""


class CovarianceRepresentation(str, Enum):
    """Coordinate basis the OD output covariance is reported in."""

    CARTESIAN = "cartesian"
    KEPLERIAN = "keplerian"
    COMETARY = "cometary"
    SPHERICAL = "spherical"


class PhotometryModel(str, Enum):
    """Photometric model for the post-OD phase-function fit.

    Mirrors ``empyrean::PhotometryModel``. In ``AUTO`` the fit climbs a
    model ladder -- H-only -> HG12 -> HG1G2 -- admitting the richest
    model the arc's phase-angle coverage and magnitude count support,
    and a :class:`PhotometryResult` reports the model it actually fitted
    on ``model_used`` (never ``AUTO``). An explicit value pins a
    specific model. HG12 / HG1G2 follow Muinonen et al. (2010); H-only
    holds the slope fixed.
    """

    AUTO = "auto"
    """Auto-select up the ladder (H-only -> HG12 -> HG1G2) by data richness."""
    HONLY = "honly"
    """Fit only the absolute magnitude H (fixed slope)."""
    HG = "hg"
    """Two-parameter H, G."""
    HG12 = "hg12"
    """Two-parameter H, G12 (Muinonen et al. 2010)."""
    HG1G2 = "hg1g2"
    """Three-parameter H, G1, G2 (Muinonen et al. 2010)."""


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


# ── Wide solve-for + photometry request (mirrors empyrean::SolveFor /
#    empyrean::PhotometryConfig) ─────────────────────────────────


@dataclass
class SolveFor:
    """Per-axis wide solve-for selection. Mirrors ``empyrean::SolveFor``.

    Set on :attr:`ODConfig.solve_for_flags` to request an explicit
    multi-axis fit that the coarse :class:`SolveForParams` variants
    can't name. Each flag turns on one wide-STM axis, subject to its
    own precondition (a declared prior on the orbit) enforced by the
    engine.
    """

    marsden: bool = False
    """Solve the Marsden A1/A2/A3 block (requires a non-grav covariance)."""
    dt: bool = False
    """Solve the non-grav time delay DT (requires ``marsden`` + a DT prior)."""
    amrat: bool = False
    """Solve the SRP AMRAT (requires an SRP AMRAT prior)."""
    thrust_segments: int = 0
    """Number of thrust Δv segments to solve (3 columns each; 0 = none)."""


@dataclass
class PhotometryConfig:
    """Post-OD photometric-fit configuration. Mirrors
    ``empyrean::PhotometryConfig``.

    Attach via :attr:`ODConfig.photometry`. The fit runs after the
    orbit is solved and never touches the state. Sentinel rule:
    ``0`` / ``0.0`` on a tuning field requests the engine default.
    """

    model: PhotometryModel = PhotometryModel.AUTO
    """Model to fit. Default :attr:`PhotometryModel.AUTO`."""
    sigma_lightcurve: float = 0.0
    """1σ lightcurve scatter floor (mag). ``0.0`` → engine default (0.2)."""
    include_rejected: bool = False
    """Include astrometrically-rejected observations' magnitudes."""
    max_irls_iterations: int = 0
    """Max Huber-IRLS iterations. ``0`` → engine default (30)."""
    huber_k: float = 0.0
    """Huber tuning constant. ``0.0`` → engine default (1.5)."""


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
    solve_for_flags: SolveFor | None = None
    """Explicit per-axis wide solve request. When set, overrides the
    coarse :attr:`solve_for` and asks the engine for an ``Explicit``
    fit over the requested axes (Marsden / DT / AMRAT / thrust).
    ``None`` = use :attr:`solve_for`."""
    allow_unbracketed_maneuvers: bool = False
    """Permit solving a thrust Δv segment whose burn window is not
    bracketed by observations (the state absorbs it otherwise). Default
    ``False`` — refuse loudly."""
    photometry: PhotometryConfig | None = None
    """Post-OD photometric fit. ``None`` (default) disables it; the fit
    runs after the orbit is solved and never touches the state."""

    def _to_wire_dict(self) -> dict[str, WireValue]:
        """Serialize to the nested dict shape the binding consumes.

        Internal — called by :func:`empyrean.determine` /
        :func:`empyrean.evaluate` / :func:`empyrean.refine` to marshal
        the config across the FFI boundary. For user-facing
        serialization (saving config to JSON, displaying it in a
        notebook, etc.), use :func:`dataclasses.asdict`.
        """
        wire: dict[str, WireValue] = {
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
            "allow_unbracketed_maneuvers": self.allow_unbracketed_maneuvers,
        }
        # Explicit per-axis solve request and photometry config are only
        # emitted when set — the Rust parser reads these keys only when
        # present, so their absence leaves the coarse `solve_for` /
        # photometry-off defaults untouched.
        if self.solve_for_flags is not None:
            wire["solve_for_flags"] = {
                "marsden": self.solve_for_flags.marsden,
                "dt": self.solve_for_flags.dt,
                "amrat": self.solve_for_flags.amrat,
                "thrust_segments": self.solve_for_flags.thrust_segments,
            }
        if self.photometry is not None:
            wire["photometry"] = {
                "model": _enum_value(self.photometry.model),
                "sigma_lightcurve": self.photometry.sigma_lightcurve,
                "include_rejected": self.photometry.include_rejected,
                "max_irls_iterations": self.photometry.max_irls_iterations,
                "huber_k": self.photometry.huber_k,
            }
        return wire


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
class SolvedCovariance:
    """Full tagged solved-parameter covariance from a wide OD fit.
    Mirrors ``empyrean::SolvedCovariance``.

    :attr:`matrix` is the real solved covariance, sized
    ``width × width``; parameters are located by the slot fields, never
    by width (width 9 is Marsden-only OR one thrust segment). Canonical
    order is ``[state 6 | Marsden 3 | DT 1 | AMRAT 1 | thrust 3×k]``. The
    Δv axes are integration-frame components (see
    :attr:`DetermineResult.dv_frame`).
    """

    matrix: np.ndarray
    """The solved covariance, shaped ``(width, width)``."""
    width: int
    """Solved width (6..=17 under the current engine)."""
    marsden_slot: int | None
    """Column of the first Marsden coefficient, when Marsden was solved."""
    dt_slot: int | None
    """Column of the DT scalar, when DT was solved."""
    amrat_slot: int | None
    """Column of the AMRAT scalar, when AMRAT was solved."""
    thrust_slots: list[tuple[int, int, int]]
    """Column triples of each fitted thrust Δv segment (one
    ``(i, i+1, i+2)`` per solved segment). Empty when no thrust was
    solved."""


@dataclass
class BandStat:
    """Per-band photometric fit statistics. Mirrors ``empyrean::BandStat``."""

    band: str
    """Photometric band tag."""
    n: int
    """Number of observations in this band."""
    offset_applied: float
    """Band→V offset applied (mag)."""
    mean_residual: float
    """Mean residual in V (mag)."""
    rms: float
    """RMS residual in V (mag)."""


@dataclass
class GateRecord:
    """One model-ladder gate decision from the photometric fit. Mirrors
    ``empyrean::GateRecord``."""

    model: PhotometryModel
    """Model the gate evaluated."""
    passed: bool
    """Whether the model was admitted."""
    reason: str
    """Human-readable gate reason."""


@dataclass
class PhotometryResult:
    """Post-OD photometric solution — an H/G fit over the arc's
    magnitudes, run after the orbit is solved. Mirrors
    ``empyrean::PhotometryResult``.

    Photometry has no astrometric partials, so it never touches the
    state. H carries honest σ via :attr:`covariance`.
    """

    h: float
    """Fitted absolute magnitude H (mag)."""
    slope1: float
    """First slope parameter (G / G12 / G1 by model)."""
    slope2: float
    """Second slope parameter (G2 for HG1G2; unused otherwise)."""
    covariance: np.ndarray | None
    """Parameter covariance over (H, slope1, slope2), shaped ``(3, 3)``
    when available. ``None`` otherwise."""
    model_used: PhotometryModel
    """Model actually fitted (never :attr:`PhotometryModel.AUTO`)."""
    reduced_chi2: float
    """Reduced χ² of the photometric fit over its used magnitudes."""
    constraint_active: bool
    """Whether a simplex constraint was active on the fitted slopes."""
    n_mags_used: int
    """Magnitudes used in the fit."""
    n_mags_rejected_photometric: int
    """Magnitudes rejected by the photometric outlier pass."""
    n_obs_without_mags: int
    """Observations carrying no magnitude."""
    n_mags_from_astrometric_selected: int
    """Magnitudes drawn from astrometrically-selected observations."""
    n_mags_from_astrometric_rejected: int
    """Magnitudes drawn from astrometrically-rejected observations."""
    alpha_min_deg: float
    """Minimum phase angle of the fitted magnitudes (deg)."""
    alpha_max_deg: float
    """Maximum phase angle of the fitted magnitudes (deg)."""
    alpha_span_deg: float
    """Phase-angle span of the fitted magnitudes (deg)."""
    per_band: list[BandStat]
    """Per-band statistics."""
    gates: list[GateRecord]
    """Model-ladder gate records."""
    n_mags_dropped_unconvertible: int
    """Magnitudes excluded from the fit because their photometric band
    has no adopted V-band conversion (unknown/unspecified band codes,
    comet total/nuclear magnitudes). Never silent: each exclusion is
    counted here and the distinct offending band codes are listed in
    :attr:`dropped_bands`. The observations' astrometry is
    unaffected."""
    dropped_bands: list[str]
    """Distinct band codes that were dropped, sorted."""


@dataclass
class TrustGateEvent:
    """The intervening event named by an ``encounter_intervenes``
    covariance-trust verdict."""

    kind: str
    """``"close_approach"`` or ``"high_nonlinearity"``."""
    epoch_mjd_tdb: float
    """Epoch of the event (MJD TDB)."""
    body: str | None = None
    """Name of the approached body (close-approach events only)."""
    distance_au: float | None = None
    """Approach distance at the signal (AU; close-approach only)."""
    nonlinearity: float | None = None
    """Nonlinearity ratio at the crossing (high-nonlinearity only)."""
    threshold: float | None = None
    """Threshold the nonlinearity exceeded (high-nonlinearity only)."""


@dataclass
class CovarianceTrust:
    """Event-aware trust verdict on the delivered covariance, evaluated
    over its validity window on the converged orbit.

    ``trusted``: no intervening close approach and a 6-state solve — the
    linear covariance may be used as delivered. ``encounter_intervenes``:
    a close approach (or high-nonlinearity crossing) lies inside the
    window; do not extrapolate the linear covariance across it —
    escalate to nonlinear uncertainty propagation (second-order when
    :attr:`second_order_recoverable`, otherwise sampling).
    ``weakly_determined_high_n``: the fit solved more than the 6-state,
    so the delivered 6×6 is a marginal of a wider fit (conservative
    flag). A ``DetermineResult.covariance_trust`` of ``None`` means the
    call path ran no gate — absence of a verdict is not trust."""

    verdict: str
    """``"trusted"`` / ``"encounter_intervenes"`` /
    ``"weakly_determined_high_n"``."""
    solved_width: int | None = None
    """Solved-for width of the fit the verdict refers to (absent for
    ``trusted``)."""
    second_order_recoverable: bool | None = None
    """Whether a second-order state-only correction can recover the
    encounter (``encounter_intervenes`` only)."""
    event: TrustGateEvent | None = None
    """The earliest intervening event (``encounter_intervenes``
    only)."""


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
    solved_covariance: SolvedCovariance | None
    """Full tagged solved-parameter covariance when the fit solved any
    wide axis (Marsden / DT / AMRAT / thrust). ``None`` for a state-only
    fit — read it, not the width, to locate solved parameters."""
    dt_delta: float | None
    """Cumulative non-grav time-delay correction ΔDT (days), when DT was
    solved. ``None`` otherwise."""
    amrat_delta: float | None
    """Cumulative SRP AMRAT correction (m²/kg), when AMRAT was solved.
    ``None`` otherwise."""
    thrust_delta_m_per_s: np.ndarray | None
    """Per-segment fitted thrust Δv (m/s), shaped ``(k, 3)`` and
    expressed in :attr:`dv_frame`. ``None`` when no thrust was solved."""
    dv_frame: str | None
    """Integration frame the thrust Δv components are expressed in
    (``"icrf"`` / ``"eclipticj2000"`` / ``"itrf93"``). ``None`` when no
    thrust was solved."""
    photometry: PhotometryResult | None
    """Post-OD photometric solution when photometry was requested and
    ran. ``None`` otherwise."""
    covariance_trust: CovarianceTrust | None
    """Event-aware trust verdict on the delivered covariance. ``None``
    when the call path ran no trust gate — absence of a verdict is not
    trust."""
