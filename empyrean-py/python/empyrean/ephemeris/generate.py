"""Ephemeris generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from empyrean._convert import (
    _COORD_TYPE_MAP,
    AnyOrbits,
    coordinates_to_arrays,
    extract_non_grav_covariance,
    extract_photometry,
)

if TYPE_CHECKING:
    import pyarrow as pa

from empyrean.ephemeris.result import EphemerisConfig, EphemerisResult
from empyrean.observers.observers import Observers
from empyrean.propagation.config import (
    _DATACLASS_TO_INT,
    _FORCE_MODEL_TO_INT,
    _UNCERTAINTY_METHOD_TO_INT,
    ForceModelTier,
    MonteCarlo,
    PropagationConfig,
    SigmaPoint,
    UncertaintyMethod,
)

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
UncertaintyMethodLike = UncertaintyMethod | SigmaPoint | MonteCarlo | str


def generate_ephemeris(
    orbits: AnyOrbits,
    observers: Observers,
    config: EphemerisConfig | None = None,
    *,
    # Sugar for quick inline overrides on the embedded
    # PropagationConfig. Ignored when `config` is passed.
    #
    # Sugar mirrors top-level config knobs only — integrator-tuning
    # parameters nested under `config.propagation.advanced` (epsilon,
    # step bounds, loop guards) are deliberately not surfaced here.
    # Reach for the structured config when you need them.
    force_model: ForceModelTier | str | None = None,
    uncertainty_method: UncertaintyMethodLike | None = None,
    # Internal: a pre-built force-model handle
    # (``empyrean._empyrean_rs.BuiltSystem``). When supplied, ephemeris
    # generation runs through the frozen handle (identity-guarded, never a
    # silent rebuild). Set by :meth:`empyrean.BuiltSystem.generate_ephemeris`;
    # not part of the public call surface. Because the ephemeris pipeline
    # integrates in EclipticJ2000, the handle must be frozen at
    # ``Frame.ECLIPTICJ2000`` and the engine-default divisor.
    _builtsystem: Any = None,
) -> EphemerisResult:
    """Generate predicted ephemeris (RA/Dec) for orbits at observer locations.

    Parameters
    ----------
    orbits : CartesianOrbits | CometaryOrbits | KeplerianOrbits | SphericalOrbits
        Input orbits with optional covariance and non-gravitational
        parameters.
    observers : Observers
        Observer states from ``get_observer_states()``.
    config : EphemerisConfig, optional
        Full configuration. Construct with
        ``EphemerisConfig(propagation=PropagationConfig(...), ...)``.
        If omitted, one is built from the sugar kwargs below (or
        defaults).

    Other Parameters
    ----------------
    force_model : ForceModelTier or str, optional
        Quick override for ``config.propagation.force_model``. Ignored
        if ``config`` is given.
    uncertainty_method : UncertaintyMethod | SigmaPoint | MonteCarlo | str, optional
        Quick override for ``config.propagation.uncertainty_method``.
        ``SECOND_ORDER`` is what populates observation Hessians on the
        resulting :class:`~empyrean.types.ObservationSensitivity`.
        Ignored if ``config`` is given.

    Returns
    -------
    EphemerisResult
        Wraps the :class:`~empyrean.types.Ephemeris` table and, when
        input covariance is carried, the observation-partials
        :class:`~empyrean.types.ObservationSensitivity` container.
        Rows are orbit-major and, within each orbit, follow the
        **observer-input order** (sensitivity rows too). Each observer
        carries its own epoch, so positional pairing against the input
        observers is safe within an orbit block.

    Examples
    --------
    Defaults (Standard force model, FirstOrder uncertainty):

    >>> result = empyrean.generate_ephemeris(orbits, observers)

    With a config object:

    >>> cfg = EphemerisConfig(
    ...     propagation=PropagationConfig(
    ...         force_model=ForceModelTier.STANDARD,
    ...         uncertainty_method=UncertaintyMethod.SECOND_ORDER,
    ...     ),
    ...     compute_diagnostics=False,
    ... )
    >>> result = empyrean.generate_ephemeris(orbits, observers, cfg)
    """
    from empyrean._empyrean_rs import _generate_ephemeris
    from empyrean.ephemeris.result import Ephemeris, EphemerisResult
    from empyrean.ephemeris.sensitivity import ObservationSensitivities

    # ── Assemble EphemerisConfig ──────────────────────────────
    if config is None:
        # PropagationConfig.force_model is typed as ForceModelTier, while the
        # `force_model` sugar additionally accepts a str. Resolve a str tier
        # to its ForceModelTier member here so the constructed config carries
        # the precise enum type; the case-insensitive lookup mirrors the
        # downstream `_FORCE_MODEL_TO_INT` mapping.
        force_model_tier: ForceModelTier
        if force_model is None:
            force_model_tier = ForceModelTier.STANDARD
        elif isinstance(force_model, str):
            force_model_tier = ForceModelTier(force_model.lower())
        else:
            force_model_tier = force_model
        prop = PropagationConfig(
            force_model=force_model_tier,
            uncertainty_method=(
                uncertainty_method
                if uncertainty_method is not None
                else UncertaintyMethod.FIRST_ORDER
            ),
        )
        config = EphemerisConfig(propagation=prop)
    elif any(v is not None for v in (force_model, uncertainty_method)):
        raise TypeError(
            "generate_ephemeris(): pass either `config` or the sugar kwargs "
            "(force_model / uncertainty_method), not both"
        )

    # Pull fields off the config
    force_model = config.propagation.force_model
    uncertainty_method = config.propagation.uncertainty_method
    epsilon = config.propagation.epsilon

    # ── Extract coordinate arrays from orbits ────────────────
    coords = orbits.coordinates
    coord_type = type(coords)
    if coord_type not in _COORD_TYPE_MAP:
        raise TypeError(f"unsupported coordinate type: {coord_type}")

    (
        epochs_arr,
        elements_arr,
        covariances_arr,
        has_cov_arr,
        representations_arr,
        frames_arr,
        origins_arr,
    ) = coordinates_to_arrays(coords)

    # IDs
    orbit_ids = orbits.orbit_id.to_pylist()
    if orbits.object_id is not None:
        object_ids = [s if s else "" for s in orbits.object_id.to_pylist()]
    else:
        object_ids = [""] * len(orbits)

    # Non-grav parameters
    n = len(orbits)
    non_grav_dts: np.ndarray | None = None
    if orbits.non_grav is not None:
        ng = orbits.non_grav
        a1s = np.asarray(ng.a1.to_numpy(zero_copy_only=False), dtype=np.float64)
        a2s = np.asarray(ng.a2.to_numpy(zero_copy_only=False), dtype=np.float64)
        a3s = np.asarray(ng.a3.to_numpy(zero_copy_only=False), dtype=np.float64)
        a1s = np.nan_to_num(a1s, nan=0.0)
        a2s = np.nan_to_num(a2s, nan=0.0)
        a3s = np.nan_to_num(a3s, nan=0.0)
        # SBDB DT (days). NaN per-row → no delay; whole array
        # passed only when at least one row populated.
        dt_col = np.asarray(ng.dt.to_numpy(zero_copy_only=False), dtype=np.float64)
        if np.isfinite(dt_col).any():
            non_grav_dts = dt_col
    else:
        a1s = np.zeros(n, dtype=np.float64)
        a2s = np.zeros(n, dtype=np.float64)
        a3s = np.zeros(n, dtype=np.float64)

    # Photometric parameters
    phot_h, phot_g, phot_model = extract_photometry(orbits)

    # Fitted non-grav covariance — passed through only when a row carries one
    # (mirrors the OD output path) so a StateAndNonGrav-fitted orbit re-fed
    # into ephemeris generation keeps its prior. Gated like non_grav_dts so
    # the common no-cov case skips the FFI marshal.
    has_ng_cov_arr, ng_cov_arr = extract_non_grav_covariance(orbits)
    has_non_grav_cov: np.ndarray | None = has_ng_cov_arr if has_ng_cov_arr.any() else None
    non_grav_cov: np.ndarray | None = ng_cov_arr if has_ng_cov_arr.any() else None

    # ── Extract observer arrays ──────────────────────────────
    obs_codes = observers.obs_code.to_pylist()
    oc = observers.coordinates
    obs_epochs = np.asarray(oc.epoch.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_x = np.asarray(oc.x.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_y = np.asarray(oc.y.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_z = np.asarray(oc.z.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_vx = np.asarray(oc.vx.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_vy = np.asarray(oc.vy.to_numpy(zero_copy_only=False), dtype=np.float64)
    obs_vz = np.asarray(oc.vz.to_numpy(zero_copy_only=False), dtype=np.float64)

    # ── Map force model to int ───────────────────────────────
    if isinstance(force_model, str):
        fm_int = _FORCE_MODEL_TO_INT.get(force_model.lower())
        if fm_int is None:
            raise ValueError(f"unknown force model: {force_model}")
    elif isinstance(force_model, ForceModelTier):
        fm_int = _FORCE_MODEL_TO_INT[force_model]
    elif isinstance(force_model, int):
        fm_int = force_model
    else:
        raise TypeError(f"force_model must be ForceModelTier, str, or int, got {type(force_model)}")

    # ── Map uncertainty method to int + params (same dispatch
    # as `empyrean.propagate`) ───────────────────────────────────
    sigma_n_sigma = 1.0
    sigma_samples_per_plane = 8
    mc_n_samples = 1000
    mc_seed: int | None = 42
    # GaussianMixture knobs — same defaults as `empyrean.propagate`
    # since the ephemeris pipeline embeds a PropagationConfig and
    # the C ABI requires the GM params be threaded through even
    # when the uncertainty method isn't a mixture.
    gm_threshold = 1.0
    gm_max_depth = 3
    gm_components_per_split = 3

    if isinstance(uncertainty_method, (SigmaPoint, MonteCarlo)):
        um_int = _DATACLASS_TO_INT[type(uncertainty_method)]
        if isinstance(uncertainty_method, SigmaPoint):
            sigma_n_sigma = uncertainty_method.n_sigma
            sigma_samples_per_plane = uncertainty_method.samples_per_plane
        else:  # MonteCarlo
            mc_n_samples = uncertainty_method.n_samples
            mc_seed = uncertainty_method.seed
    elif isinstance(uncertainty_method, str):
        um_lookup = _UNCERTAINTY_METHOD_TO_INT.get(uncertainty_method.lower())
        if um_lookup is None:
            raise ValueError(f"unknown uncertainty method: {uncertainty_method}")
        um_int = um_lookup
    elif isinstance(uncertainty_method, UncertaintyMethod):
        um_int = _UNCERTAINTY_METHOD_TO_INT[uncertainty_method]
    elif isinstance(uncertainty_method, int):
        um_int = uncertainty_method
    else:
        raise TypeError(
            "uncertainty_method must be UncertaintyMethod, a SigmaPoint / "
            "MonteCarlo dataclass, str, or int; got "
            f"{type(uncertainty_method).__name__}"
        )

    # ── Call Rust ─────────────────────────────────────────────
    result = _generate_ephemeris(
        orbit_ids,
        object_ids,
        epochs_arr,
        elements_arr,
        covariances_arr,
        has_cov_arr,
        representations_arr,
        frames_arr,
        origins_arr,
        a1s,
        a2s,
        a3s,
        phot_h,
        phot_g,
        phot_model,
        obs_codes,
        obs_epochs,
        obs_x,
        obs_y,
        obs_z,
        obs_vx,
        obs_vy,
        obs_vz,
        fm_int,
        epsilon,
        uncertainty_method=um_int,
        non_grav_dts=non_grav_dts,
        has_non_grav_cov=has_non_grav_cov,
        non_grav_cov=non_grav_cov,
        gm_threshold=gm_threshold,
        gm_max_depth=gm_max_depth,
        gm_components_per_split=gm_components_per_split,
        sigma_n_sigma=sigma_n_sigma,
        sigma_samples_per_plane=sigma_samples_per_plane,
        mc_n_samples=mc_n_samples,
        mc_seed=mc_seed,
        # Thread the full nested EphemerisConfig (which embeds a full
        # PropagationConfig) so light-time iteration limits, diagnostics
        # toggles, integrator advanced knobs, and event-detection
        # settings all reach the C ABI.
        ephemeris_config_dict=config._to_wire_dict(),
        builtsystem=_builtsystem,
    )

    # ── Build Ephemeris from result ──────────────────────────
    #
    # The C ABI's flat ephemeris dict carries: orbit_id, object_id,
    # epoch, ra, dec, rho, vrho, vra, vdec, light_time, phase_angle,
    # elongation, heliocentric_distance, mag, mag_sigma, obs_code, and —
    # as of the parity extension — the local-horizon / sky-motion angles
    # zenith_angle, azimuth, hour_angle, lunar_elongation, position_angle,
    # sky_rate (all degrees; sky_rate is deg/day), NaN where the observer
    # geometry made them unavailable.
    #
    # Aberrated state and per-row observation covariance are still NOT in
    # the flat schema — those remain NaN / null placeholders here.
    from empyrean.coordinates.coordinates import (
        CartesianCoordinates,
        SphericalCoordinates,
    )
    from empyrean.coordinates.enums import Frame, Origin

    m = len(result["epoch"])
    object_id_list = [s if s else None for s in result["object_id"]]

    coordinates = SphericalCoordinates.from_kwargs(
        epoch=np.asarray(result["epoch"]),
        rho=np.asarray(result["rho"]),
        lon=np.asarray(result["ra"]),
        lat=np.asarray(result["dec"]),
        vrho=np.asarray(result["vrho"]),
        vlon=np.asarray(result["vra"]),
        vlat=np.asarray(result["vdec"]),
        frame=Frame.ICRF.value,
        origin=result["obs_code"],
    )

    def _nullable_float(key: str) -> pa.Array | FloatArray:
        arr: FloatArray = np.asarray(result[key], dtype=np.float64)
        mask = np.isnan(arr)
        if mask.any():
            import pyarrow as pa

            return pa.array(arr.tolist(), type=pa.float64(), mask=mask)
        return arr

    nan_col = np.full(m, np.nan)
    # `aberrated_state` is `nullable=True` but its inner CartesianCoordinates
    # still requires its `frame` StringAttribute to be set during quivr
    # validation, even when every row is null. Provide an all-NaN
    # placeholder with `Frame.ICRF` to satisfy the schema; the C ABI
    # doesn't carry aberrated state, so callers should treat these as
    # absent rather than meaningful values.
    aberrated_state = CartesianCoordinates.from_kwargs(
        epoch=np.asarray(result["epoch"]),
        x=nan_col,
        y=nan_col,
        z=nan_col,
        vx=nan_col,
        vy=nan_col,
        vz=nan_col,
        frame=Frame.ICRF.value,
        origin=[str(Origin.SSB)] * m,
    )
    ephemeris = Ephemeris.from_kwargs(
        orbit_id=result["orbit_id"],
        object_id=object_id_list,
        obs_code=result["obs_code"],
        coordinates=coordinates,
        aberrated_state=aberrated_state,
        light_time=_nullable_float("light_time"),
        phase_angle=_nullable_float("phase_angle"),
        elongation=_nullable_float("elongation"),
        heliocentric_distance=_nullable_float("heliocentric_distance"),
        mag=_nullable_float("mag"),
        mag_sigma=_nullable_float("mag_sigma"),
        zenith_angle=_nullable_float("zenith_angle"),
        azimuth=_nullable_float("azimuth"),
        hour_angle=_nullable_float("hour_angle"),
        lunar_elongation=_nullable_float("lunar_elongation"),
        position_angle=_nullable_float("position_angle"),
        sky_rate=_nullable_float("sky_rate"),
    )

    # ── Observation sensitivities ──
    # One row per (orbit, observer, epoch). jacobian/hessian are row-major-
    # flattened (6·n_params / 6·n_params²); hessian is null unless a
    # second-order method ran. Empty table on the f64-only path.
    n_sens = len(result.get("sensitivity_orbit_id", []))
    if n_sens == 0:
        sensitivity = ObservationSensitivities.empty()
    else:
        sensitivity = ObservationSensitivities.from_kwargs(
            orbit_id=result["sensitivity_orbit_id"],
            object_id=result["sensitivity_object_id"],
            obs_code=result["sensitivity_obs_code"],
            epoch_mjd_tdb=np.asarray(result["sensitivity_epoch_mjd_tdb"], dtype=np.float64),
            n_params=np.asarray(result["sensitivity_n_params"], dtype=np.uint8),
            jacobian=result["sensitivity_jacobian"],
            hessian=result["sensitivity_hessian"],
        )

    return EphemerisResult(ephemeris=ephemeris, sensitivity=sensitivity)
