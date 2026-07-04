"""Orbit propagation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pyarrow as pa
import quivr as qv

if TYPE_CHECKING:
    from empyrean.ephemeris.sensitivity import StateSensitivities
    from empyrean.orbits.thrust import ThrustParams
    from empyrean.propagation.tagged_covariance import TaggedCovariances

from empyrean._convert import (
    _COORD_TYPE_MAP,
    AnyOrbits,
    coordinates_to_arrays,
    extract_photometry,
    int_to_frame,
    naif_to_origin,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import (
    CartesianCovariance as _CartesianCovariance,
)
from empyrean.coordinates.covariance import _CovarianceTable
from empyrean.coordinates.epoch import Epochs
from empyrean.orbits.orbits import CartesianOrbits
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
from empyrean.propagation.events import (
    AtmosphericEntries,
    AtmosphericExits,
    CaptureEnds,
    CaptureStarts,
    CloseApproachEnds,
    CloseApproachStarts,
    CovarianceRegimeChanges,
    EventConfig,
    Events,
    EventSummary,
    Impacts,
    Periapses,
    PossibleImpacts,
    ShadowEntries,
    ShadowExits,
)
from empyrean.propagation.result import PropagationResult
from empyrean.propagation.tagged_covariance import _KIND_BY_CODE

# Quivr event sub-tables share the build helpers below; the helpers
# return the same concrete table subclass they are handed.
_EventTableT = TypeVar("_EventTableT", bound=qv.Table)

# ``CartesianCovariance`` is built dynamically by ``_make_covariance_class``,
# whose declared return type is the bare ``type``; that loses the injected
# ``from_matrix`` / ``from_kwargs`` constructors. Re-bind it through
# ``type[_CovarianceTable]`` (the Protocol describing exactly that
# dynamically-injected surface) so those constructors type-check at the call
# sites below. The runtime object is unchanged.
CartesianCovariance: type[_CovarianceTable] = _CartesianCovariance

UncertaintyMethodLike = UncertaintyMethod | SigmaPoint | MonteCarlo | str


def propagate(
    orbits: AnyOrbits,
    epochs: Epochs | np.ndarray | Sequence[float],
    config: PropagationConfig | None = None,
    *,
    # ── Sugar for quick, inline overrides ─────────────────────
    # Any of these will populate a fresh PropagationConfig when `config`
    # isn't supplied. Ignored when `config` is passed.
    #
    # Sugar mirrors top-level `PropagationConfig` fields only — knobs
    # nested under `config.advanced` (`epsilon`, step bounds, loop
    # guards) are deliberately not surfaced here. Reaching for those
    # means you're tuning the integrator itself, which is a structured-
    # config conversation, not an inline override.
    force_model: ForceModelTier | str | None = None,
    uncertainty_method: UncertaintyMethodLike | None = None,
    num_threads: int | None = None,
    events: EventConfig | None = None,
    tagged_covariance: bool = False,
    thrust_arcs: "Sequence[ThrustParams | None] | None" = None,
) -> PropagationResult:
    """Propagate orbits to target epochs.

    Parameters
    ----------
    orbits : CartesianOrbits | CometaryOrbits | KeplerianOrbits | SphericalOrbits
        Input orbits with optional covariance and non-gravitational
        parameters.
    epochs : Epochs | array-like
        Target epochs. An :class:`~empyrean.types.Epochs` table (converted
        to TDB internally), or a 1-D array of MJD TDB values.
    config : PropagationConfig, optional
        Full propagation configuration. Construct with
        ``PropagationConfig(force_model=..., uncertainty_method=...)``
        etc. If omitted, one is built from the sugar kwargs below (or
        defaults).
    Other Parameters
    ----------------
    force_model : ForceModelTier or str, optional
        Quick override for ``config.force_model``. Ignored if ``config``
        is given.
    uncertainty_method : UncertaintyMethod | SigmaPoint | MonteCarlo | str, optional
        Quick override for ``config.uncertainty_method``. Accepts either
        an enum / string (default parameters) or a parameterized
        dataclass (:class:`SigmaPoint(n_sigma=2.0)` etc.). Ignored if
        ``config`` is given.
    num_threads : int, optional
        Threads for multi-orbit propagation. ``None`` (default) =
        sequential; ``0`` = use all available cores; ``n`` > 0 = use
        exactly ``n`` cores. Each orbit is integrated on a single
        thread; parallelism is across orbits, not within a single
        trajectory.
    events : EventConfig, optional
        Event-detection toggles + body filter + dense-output cadence.
        Override individual flags here without rebuilding a full
        :class:`PropagationConfig`. See
        :class:`~empyrean.EventConfig`.
    tagged_covariance : bool, default False
        When ``True``, also read back the provenance-tagged,
        resolved-kind covariance at every output epoch (the honest
        covariance that distinguishes a second-order close-approach
        ellipsoid from the bare linear ``Φ Σ₀ Φᵀ`` mapping on the
        states). The result's
        :attr:`~empyrean.PropagationResult.tagged_covariance` table is
        populated and
        :meth:`~empyrean.PropagationResult.tagged_covariance_series`
        becomes usable. Off by default — the readback recomputes the
        resolved kind per orbit, so it isn't free.
    thrust_arcs : sequence of ThrustParams or None, optional
        Structured continuous-thrust / finite-burn input, one entry per
        orbit and positionally aligned with ``orbits`` (pass ``None`` for
        the gravity / non-grav-only orbits, or the whole argument
        ``None`` for a fully ballistic batch). Build each entry from
        :class:`~empyrean.ThrustParams` /
        :class:`~empyrean.ThrustArc` / a
        :class:`~empyrean.orbits.thrust.SteeringLaw` variant. A non-empty
        :attr:`~empyrean.ThrustParams.correction_covariances` triggers the
        burn-sensitivity propagation whose solved segments surface in the
        tagged-covariance
        :attr:`~empyrean.TaggedCovariance.thrust_segments` (requires
        ``tagged_covariance=True``). Length or arc/correction mismatches
        raise, never silently degrade.

    Returns
    -------
    PropagationResult
        Propagated states, detected events, and per-orbit state
        sensitivity chains.

    Examples
    --------
    Defaults (Standard force model, FirstOrder uncertainty):

    >>> result = empyrean.propagate(orbits, times)

    With a config object:

    >>> cfg = PropagationConfig(
    ...     force_model=ForceModelTier.FULL,
    ...     uncertainty_method=SigmaPoint(n_sigma=2.0),
    ...     num_threads=8,
    ... )
    >>> result = empyrean.propagate(orbits, times, cfg)

    With inline kwargs (sugar):

    >>> result = empyrean.propagate(orbits, times, force_model="full")
    """
    from empyrean._empyrean_rs import _propagate

    # ── Assemble PropagationConfig ────────────────────────────
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
        config = PropagationConfig(
            force_model=force_model_tier,
            uncertainty_method=(
                uncertainty_method
                if uncertainty_method is not None
                else UncertaintyMethod.FIRST_ORDER
            ),
            num_threads=num_threads,
            events=events if events is not None else EventConfig(),
        )
    elif any(v is not None for v in (force_model, uncertainty_method, num_threads, events)):
        raise TypeError(
            "propagate(): pass either `config` or the sugar kwargs "
            "(force_model / uncertainty_method / num_threads / events), "
            "not both"
        )

    # Pull fields off the config from here on
    force_model = config.force_model
    uncertainty_method = config.uncertainty_method
    num_threads = config.num_threads
    epsilon = config.epsilon
    events = config.events
    if events is None:
        events = EventConfig()

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
    # g(r) Marsden–Sekanina exponents. Passed only when a non-default g(r)
    # is present (any non-zero α/r0/m/n/k); all-zero is the inverse-square
    # asteroid default that the engine applies without a marshal.
    ng_alphas: np.ndarray | None = None
    ng_r0s: np.ndarray | None = None
    ng_ms: np.ndarray | None = None
    ng_ns: np.ndarray | None = None
    ng_ks: np.ndarray | None = None
    if orbits.non_grav is not None:
        ng = orbits.non_grav
        # Handle nullable columns: fill None with 0.0
        a1s = np.asarray(ng.a1.to_numpy(zero_copy_only=False), dtype=np.float64)
        a2s = np.asarray(ng.a2.to_numpy(zero_copy_only=False), dtype=np.float64)
        a3s = np.asarray(ng.a3.to_numpy(zero_copy_only=False), dtype=np.float64)
        a1s = np.nan_to_num(a1s, nan=0.0)
        a2s = np.nan_to_num(a2s, nan=0.0)
        a3s = np.nan_to_num(a3s, nan=0.0)
        # SBDB non-grav DT (days). NaN entries → no delay; pass the
        # whole array only when at least one row populated, so the
        # asteroid-only case avoids an FFI marshal.
        dt_col = np.asarray(ng.dt.to_numpy(zero_copy_only=False), dtype=np.float64)
        if np.isfinite(dt_col).any():
            non_grav_dts = dt_col
        # g(r) exponents — carry the comet Marsden–Sekanina g(r) so a fitted
        # or SBDB comet orbit isn't silently propagated with inverse-square.
        alpha_col = np.nan_to_num(
            np.asarray(ng.alpha.to_numpy(zero_copy_only=False), dtype=np.float64), nan=0.0
        )
        r0_col = np.nan_to_num(
            np.asarray(ng.r0.to_numpy(zero_copy_only=False), dtype=np.float64), nan=0.0
        )
        m_col = np.nan_to_num(
            np.asarray(ng.m.to_numpy(zero_copy_only=False), dtype=np.float64), nan=0.0
        )
        n_col = np.nan_to_num(
            np.asarray(ng.n.to_numpy(zero_copy_only=False), dtype=np.float64), nan=0.0
        )
        k_col = np.nan_to_num(
            np.asarray(ng.k.to_numpy(zero_copy_only=False), dtype=np.float64), nan=0.0
        )
        if (
            (alpha_col != 0).any()
            or (r0_col != 0).any()
            or (m_col != 0).any()
            or (n_col != 0).any()
            or (k_col != 0).any()
        ):
            ng_alphas = alpha_col
            ng_r0s = r0_col
            ng_ms = m_col
            ng_ns = n_col
            ng_ks = k_col
    else:
        a1s = np.zeros(n, dtype=np.float64)
        a2s = np.zeros(n, dtype=np.float64)
        a3s = np.zeros(n, dtype=np.float64)

    # Photometric parameters
    phot_h, phot_g, phot_model = extract_photometry(orbits)

    # ── Extract times ────────────────────────────────────────
    if isinstance(epochs, Epochs):
        # Convert to TDB if needed
        tdb = epochs.to_tdb()
        times_mjd_tdb = np.asarray(tdb.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)
    else:
        times_mjd_tdb = np.asarray(epochs, dtype=np.float64)

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

    # ── Map uncertainty method to int + extract params ─────────
    #
    # Three input shapes:
    #   1. str / UncertaintyMethod enum → default parameters
    #   2. SigmaPoint / MonteCarlo dataclass → method + params
    #   3. int (legacy) → default parameters
    sigma_n_sigma = 1.0
    sigma_samples_per_plane = 8
    mc_n_samples = 1000
    mc_seed: int | None = 42

    if isinstance(uncertainty_method, (SigmaPoint, MonteCarlo)):
        um_int = _DATACLASS_TO_INT[type(uncertainty_method)]
        if isinstance(uncertainty_method, SigmaPoint):
            sigma_n_sigma = uncertainty_method.n_sigma
            sigma_samples_per_plane = uncertainty_method.samples_per_plane
        else:  # MonteCarlo
            mc_n_samples = uncertainty_method.n_samples
            mc_seed = uncertainty_method.seed
    elif isinstance(uncertainty_method, str):
        um_int_opt = _UNCERTAINTY_METHOD_TO_INT.get(uncertainty_method.lower())
        if um_int_opt is None:
            raise ValueError(f"unknown uncertainty method: {uncertainty_method}")
        um_int = um_int_opt
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

    # ── Structured thrust input ──────────────────────────────
    # One ThrustParams (or None) per orbit, positionally aligned with the
    # batch. The binding reconstructs each into a wrapper ThrustParams and
    # attaches it per orbit; None entries stay gravity / non-grav only.
    thrust_arg: list[ThrustParams | None] | None = None
    if thrust_arcs is not None:
        thrust_arg = list(thrust_arcs)
        if len(thrust_arg) != n:
            raise ValueError(
                f"thrust_arcs must have one entry per orbit (got {len(thrust_arg)} for {n} orbits)"
            )

    # ── Call Rust ─────────────────────────────────────────────
    # Thread the full nested PropagationConfig as a single dict so that
    # advanced fields (events.dense_output, diagnostics.*, advanced.*,
    # excluded_perturbers, max_propagation_time_days, etc.) are honored
    # without growing _propagate's flat-arg signature.
    result = _propagate(
        orbit_ids,
        object_ids,
        epochs_arr,
        elements_arr,
        covariances_arr,
        has_cov_arr,
        representations_arr,
        frames_arr,
        origins_arr,
        times_mjd_tdb,
        fm_int,
        um_int,
        a1s,
        a2s,
        a3s,
        phot_h,
        phot_g,
        phot_model,
        num_threads=num_threads,
        epsilon=epsilon,
        thrust_arcs=thrust_arg,
        non_grav_dts=non_grav_dts,
        ng_alphas=ng_alphas,
        ng_r0s=ng_r0s,
        ng_ms=ng_ms,
        ng_ns=ng_ns,
        ng_ks=ng_ks,
        # These mixture kwargs are accepted by the binding for ABI
        # compatibility but ignored in this distribution. Pass benign
        # placeholders.
        gm_threshold=1.0,
        gm_max_depth=3,
        gm_components_per_split=3,
        sigma_n_sigma=sigma_n_sigma,
        sigma_samples_per_plane=sigma_samples_per_plane,
        mc_n_samples=mc_n_samples,
        mc_seed=mc_seed,
        propagation_config_dict=config._to_wire_dict(),
        with_tagged_covariance=tagged_covariance,
    )

    # ── Build CartesianOrbits from result ─────────────────────
    states = _build_cartesian_orbits(result)
    detected_events = _build_events(result)

    # ── Build per-orbit StateSensitivity chains ───────────────
    sensitivity = _build_state_sensitivity(result)

    # ── Build provenance-tagged covariance table (opt-in) ─────
    tagged = _build_tagged_covariance(result) if tagged_covariance else None

    return PropagationResult(
        states=states,
        events=detected_events,
        sensitivity=sensitivity,
        tagged_covariance=tagged,
    )


def _build_tagged_covariance(
    result: dict[str, Any],
) -> "TaggedCovariances | None":
    """Build a :class:`TaggedCovariances` table from the pyo3 result.

    The per-``(orbit, epoch)`` tagged arrays in ``result`` are aligned
    1:1 with the states, so the orbit / object ids and epochs are reused
    straight from the flat state columns. Returns ``None`` when the
    extension emitted no tagged sub-dict.
    """
    from empyrean.propagation.tagged_covariance import build_tagged_covariances

    if "tagged_covariance" not in result:
        return None

    orbit_ids: list[str] = list(result["orbit_ids"])
    object_ids: list[str | None] = [s if s else None for s in result["object_ids"]]
    epochs_arr = np.asarray(result["epochs"], dtype=np.float64)
    return build_tagged_covariances(result, orbit_ids, object_ids, epochs_arr)


def _build_state_sensitivity(result: dict[str, Any]) -> "StateSensitivities | None":
    """Build a :class:`StateSensitivities` table from the pyo3 result.

    Flattens the per-row (6, 6) STM and (6, 6, 6) STT arrays into the
    row-major lists the table expects (length 36 / 216 per row), with
    ``None`` per row when ``has_stm`` / ``has_stt`` is false on that row.

    Returns ``None`` if no row has an STM (sample-based methods or
    FirstOrder without input covariance).
    """
    from empyrean.ephemeris.sensitivity import StateSensitivities

    stms = np.asarray(result["stms"]) if "stms" in result else None
    has_stm = np.asarray(result["has_stm"]) if "has_stm" in result else None
    stts = np.asarray(result["stts"]) if "stts" in result else None
    has_stt = np.asarray(result["has_stt"]) if "has_stt" in result else None

    if has_stm is None or not bool(has_stm.any()):
        return None

    orbit_ids = list(result["orbit_ids"])
    object_ids = [s if s else None for s in result["object_ids"]]
    epochs_arr = np.asarray(result["epochs"], dtype=np.float64)
    n = len(orbit_ids)

    # Flatten STMs to length-36 lists; null per-row where has_stm is false.
    if stms is not None:
        stms_flat = stms.reshape(n, 36).tolist()
        stm_col = [
            row if (has_stm is None or bool(has_stm[i])) else None
            for i, row in enumerate(stms_flat)
        ]
    else:
        stm_col = [None] * n

    # STTs: only populate rows where has_stt is true; null elsewhere.
    if stts is not None and has_stt is not None and bool(has_stt.any()):
        stts_flat = stts.reshape(n, 216).tolist()
        stt_col = [row if bool(has_stt[i]) else None for i, row in enumerate(stts_flat)]
    else:
        stt_col = [None] * n

    # Per-row resolved covariance kind (linear / second_order / …).
    rk_codes = result.get("resolved_kind")
    if rk_codes is not None:
        resolved_kind_col: list[str | None] = [
            _KIND_BY_CODE[int(c)].value if int(c) >= 0 else None for c in np.asarray(rk_codes)
        ]
    else:
        resolved_kind_col = [None] * n

    return StateSensitivities.from_kwargs(
        orbit_id=orbit_ids,
        object_id=object_ids,
        epoch_mjd_tdb=epochs_arr,
        stm=stm_col,
        stt=stt_col,
        resolved_kind=resolved_kind_col,
    )


def _build_cartesian_orbits(result: dict[str, Any]) -> CartesianOrbits:
    """Build CartesianOrbits from the Rust result dict."""
    m = len(result["epochs"])

    out_epochs = np.asarray(result["epochs"])
    out_x = np.asarray(result["x"])
    out_y = np.asarray(result["y"])
    out_z = np.asarray(result["z"])
    out_vx = np.asarray(result["vx"])
    out_vy = np.asarray(result["vy"])
    out_vz = np.asarray(result["vz"])
    out_frames = np.asarray(result["frames"])
    out_origins = np.asarray(result["origins"])
    out_covariances = np.asarray(result["covariances"])
    out_has_cov = np.asarray(result["has_covariance"])
    out_orbit_ids = result["orbit_ids"]
    out_object_ids = result["object_ids"]

    # Build frame attribute (all rows should have the same frame)
    from empyrean.coordinates.enums import Frame as FrameEnum

    frame = int_to_frame(int(out_frames[0])) if m > 0 else FrameEnum.ICRF

    # Build origin column
    origin_strs = [naif_to_origin(int(o)) for o in out_origins]

    # Build covariance
    if out_has_cov.any():
        cov = CartesianCovariance.from_matrix(out_covariances)
    else:
        cov = None

    frame_value = frame.value if hasattr(frame, "value") else frame
    if cov is not None:
        cart_coords = CartesianCoordinates.from_kwargs(
            epoch=out_epochs,
            x=out_x,
            y=out_y,
            z=out_z,
            vx=out_vx,
            vy=out_vy,
            vz=out_vz,
            frame=frame_value,
            origin=origin_strs,
            covariance=cov,
        )
    else:
        cart_coords = CartesianCoordinates.from_kwargs(
            epoch=out_epochs,
            x=out_x,
            y=out_y,
            z=out_z,
            vx=out_vx,
            vy=out_vy,
            vz=out_vz,
            frame=frame_value,
            origin=origin_strs,
        )

    # Convert empty strings to None for nullable object_id
    object_id_list = [s if s else None for s in out_object_ids]

    return CartesianOrbits.from_kwargs(
        orbit_id=out_orbit_ids,
        object_id=object_id_list,
        coordinates=cart_coords,
    )


def _nullable_float(values: np.ndarray) -> pa.Array | np.ndarray:
    """Convert a list of floats to a pyarrow array with NaN -> null."""
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isnan(arr)
    if mask.any():
        return pa.array(arr.tolist(), type=pa.float64(), mask=mask)
    return arr


def _nullable_str_list(values: Sequence[str | None]) -> list[str | None]:
    """Convert a list of strings to a list with empty string -> None."""
    return [s if s else None for s in values]


def _build_events(result: dict[str, Any]) -> Events:
    """Build Events container from the Rust result dict.

    The Rust extension returns a single flat events sub-dict with an
    ``event_types`` discriminator column carrying ``distance_au`` /
    ``distance_km`` / ``relative_velocity_au_day``. We dispatch each row
    into the appropriate per-subtype quivr table. Subtype-specific fields
    that the flat schema *does* carry are read across (e.g. the atmospheric
    entry altitude rides in ``distance_km``); fields it does not carry
    (latitude / longitude on impacts, jacobi constants on captures,
    illumination on shadow events) are filled with NaN / null.
    """
    ev = result.get("events")
    if ev is None or len(ev["event_types"]) == 0:
        return Events(
            summary=EventSummary.empty(),
            close_approach_starts=CloseApproachStarts.empty(),
            close_approach_ends=CloseApproachEnds.empty(),
            periapses=Periapses.empty(),
            impacts=Impacts.empty(),
            possible_impacts=PossibleImpacts.empty(),
            atmospheric_entries=AtmosphericEntries.empty(),
            atmospheric_exits=AtmosphericExits.empty(),
            capture_starts=CaptureStarts.empty(),
            capture_ends=CaptureEnds.empty(),
            shadow_entries=ShadowEntries.empty(),
            shadow_exits=ShadowExits.empty(),
            covariance_regime_changes=CovarianceRegimeChanges.empty(),
        )

    orbit_ids = list(ev["orbit_ids"])
    object_ids = list(ev["object_ids"])
    event_types = list(ev["event_types"])
    bodies = list(ev["bodies"])
    epochs = np.asarray(ev["epochs"], dtype=np.float64)
    distance_au = np.asarray(ev["distance_au"], dtype=np.float64)
    distance_km = np.asarray(ev["distance_km"], dtype=np.float64)
    rel_v = np.asarray(ev["relative_velocity_au_day"], dtype=np.float64)
    # Subtype payload columns the C ABI now carries (NaN / -1 sentinels on
    # rows the field doesn't apply to). `.get` with a sentinel fallback so
    # older result dicts (pre-extension) still load.
    n_all = len(event_types)

    def _ev_f(key: str) -> np.ndarray:
        return np.asarray(ev.get(key, np.full(n_all, np.nan)), dtype=np.float64)

    two_body_energy = _ev_f("two_body_energy")
    jacobi = _ev_f("jacobi_constant")
    jacobi_sigma = _ev_f("jacobi_constant_sigma")
    jacobi_l1 = _ev_f("jacobi_constant_l1")
    jacobi_l2 = _ev_f("jacobi_constant_l2")
    n_periapses = np.asarray(ev.get("n_periapses", np.full(n_all, -1)), dtype=np.int32)
    impact_lat = _ev_f("impact_latitude_deg")
    impact_lon = _ev_f("impact_longitude_deg")
    impact_alt = _ev_f("impact_altitude_km")
    shadow_fraction = _ev_f("shadow_fraction")
    illumination = _ev_f("illumination")
    relative_x = _ev_f("relative_x")
    relative_y = _ev_f("relative_y")
    relative_z = _ev_f("relative_z")
    relative_vx = _ev_f("relative_vx")
    relative_vy = _ev_f("relative_vy")
    relative_vz = _ev_f("relative_vz")
    pi_effective_radius_au = _ev_f("effective_radius_au")
    pi_effective_radius_km = _ev_f("effective_radius_km")
    pi_sigma_distance_au = _ev_f("sigma_distance_au")
    pi_ip_linear = _ev_f("ip_linear")
    pi_ip_second_order = _ev_f("ip_second_order")
    pi_nonlinearity = _ev_f("nonlinearity")
    pi_ip_agm = _ev_f("ip_agm")
    pi_ip_mc = _ev_f("ip_mc")
    previous_kind = np.asarray(ev.get("previous_kind", np.full(n_all, -1)), dtype=np.int64)
    regime_resolved_kind = np.asarray(
        ev.get("regime_resolved_kind", np.full(n_all, -1)), dtype=np.int64
    )
    kappa = _ev_f("kappa")
    threshold_below = _ev_f("threshold_below")
    threshold_above = _ev_f("threshold_above")

    # Cross-cutting summary table — every event lands here.
    summary = EventSummary.from_kwargs(
        orbit_id=orbit_ids,
        object_id=_nullable_str_list(object_ids),
        event_type=event_types,
        body=bodies,
        epoch=epochs,
    )

    # Filter helpers ---------------------------------------------------
    def _idx(tag: str) -> list[int]:
        return [i for i, t in enumerate(event_types) if t == tag]

    def _str(values: Sequence[str], idx: list[int]) -> list[str]:
        return [values[i] for i in idx]

    def _str_opt(values: Sequence[str], idx: list[int]) -> list[str | None]:
        return [values[i] if values[i] else None for i in idx]

    def _arr(values: np.ndarray, idx: list[int]) -> np.ndarray:
        return values[idx] if len(idx) > 0 else np.zeros(0, dtype=values.dtype)

    def _common(tag: str, cls: type[_EventTableT]) -> _EventTableT:
        idx = _idx(tag)
        if not idx:
            return cls.empty()
        return cls.from_kwargs(
            orbit_id=_str(orbit_ids, idx),
            object_id=_str_opt(object_ids, idx),
            body=_str(bodies, idx),
            epoch=_arr(epochs, idx),
            distance_au=_arr(distance_au, idx),
            distance_km=_arr(distance_km, idx),
        )

    close_approach_starts = _common("close_approach_start", CloseApproachStarts)
    close_approach_ends = _common("close_approach_end", CloseApproachEnds)

    # Periapses carry relative state vectors wired through the C ABI.
    per_idx = _idx("periapsis")
    if per_idx:
        periapses = Periapses.from_kwargs(
            orbit_id=_str(orbit_ids, per_idx),
            object_id=_str_opt(object_ids, per_idx),
            body=_str(bodies, per_idx),
            epoch=_arr(epochs, per_idx),
            distance_au=_arr(distance_au, per_idx),
            distance_km=_arr(distance_km, per_idx),
            relative_velocity_au_day=_arr(rel_v, per_idx),
            relative_x=_arr(relative_x, per_idx),
            relative_y=_arr(relative_y, per_idx),
            relative_z=_arr(relative_z, per_idx),
            relative_vx=_arr(relative_vx, per_idx),
            relative_vy=_arr(relative_vy, per_idx),
            relative_vz=_arr(relative_vz, per_idx),
        )
    else:
        periapses = Periapses.empty()

    # Impacts: planetodetic surface-intercept lat/lon/alt now carried by
    # the flat schema (NaN -> null where the impact geometry was
    # unresolved).
    imp_idx = _idx("impact")
    if imp_idx:
        impacts = Impacts.from_kwargs(
            orbit_id=_str(orbit_ids, imp_idx),
            object_id=_str_opt(object_ids, imp_idx),
            body=_str(bodies, imp_idx),
            epoch=_arr(epochs, imp_idx),
            latitude_deg=_nullable_float(_arr(impact_lat, imp_idx)),
            longitude_deg=_nullable_float(_arr(impact_lon, imp_idx)),
            altitude_km=_nullable_float(_arr(impact_alt, imp_idx)),
        )
    else:
        impacts = Impacts.empty()

    # Possible impacts: probabilistic fields not in the flat schema.
    pi_idx = _idx("possible_impact")
    if pi_idx:
        # PossibleImpact probability payload is wired through the C ABI.
        # The second-order / AGM / MC probabilities are NaN unless the
        # matching uncertainty method ran.
        possible_impacts = PossibleImpacts.from_kwargs(
            orbit_id=_str(orbit_ids, pi_idx),
            object_id=_str_opt(object_ids, pi_idx),
            body=_str(bodies, pi_idx),
            epoch=_arr(epochs, pi_idx),
            miss_distance_au=_arr(distance_au, pi_idx),
            miss_distance_km=_arr(distance_km, pi_idx),
            effective_radius_au=_arr(pi_effective_radius_au, pi_idx),
            effective_radius_km=_arr(pi_effective_radius_km, pi_idx),
            sigma_distance_au=_arr(pi_sigma_distance_au, pi_idx),
            ip_linear=_arr(pi_ip_linear, pi_idx),
            relative_velocity_au_day=_arr(rel_v, pi_idx),
            ip_second_order=_arr(pi_ip_second_order, pi_idx),
            nonlinearity=_arr(pi_nonlinearity, pi_idx),
            ip_agm=_arr(pi_ip_agm, pi_idx),
            ip_mc=_arr(pi_ip_mc, pi_idx),
        )
    else:
        possible_impacts = PossibleImpacts.empty()

    # Atmospheric entries: distance_au is the body-CENTER crossing
    # distance (the Karman radius), NOT an altitude. The true altitude
    # above the reference ellipsoid and the surface lat/lon come from the
    # planetodetic ground track (impact_altitude_km / impact_*_deg on the
    # flat event), NaN -> null when the ground track is unresolved. The
    # entry speed rides in relative_velocity_au_day.
    ae_idx = _idx("atmospheric_entry")
    if ae_idx:
        atmospheric_entries = AtmosphericEntries.from_kwargs(
            orbit_id=_str(orbit_ids, ae_idx),
            object_id=_str_opt(object_ids, ae_idx),
            body=_str(bodies, ae_idx),
            epoch=_arr(epochs, ae_idx),
            distance_au=_arr(distance_au, ae_idx),
            altitude_km=_nullable_float(_arr(impact_alt, ae_idx)),
            relative_velocity_au_day=_nullable_float(_arr(rel_v, ae_idx)),
            latitude_deg=_nullable_float(_arr(impact_lat, ae_idx)),
            longitude_deg=_nullable_float(_arr(impact_lon, ae_idx)),
        )
    else:
        atmospheric_entries = AtmosphericEntries.empty()

    def _simple_entry_exit(tag: str, cls: type[_EventTableT]) -> _EventTableT:
        idx = _idx(tag)
        if not idx:
            return cls.empty()
        return cls.from_kwargs(
            orbit_id=_str(orbit_ids, idx),
            object_id=_str_opt(object_ids, idx),
            body=_str(bodies, idx),
            epoch=_arr(epochs, idx),
            distance_au=_arr(distance_au, idx),
        )

    atmospheric_exits = _simple_entry_exit("atmospheric_exit", AtmosphericExits)

    # Capture starts/ends: two-body energy + CR3BP Jacobi constants (and
    # the escape periapsis count) now carried by the flat schema.
    def _capture(tag: str, cls: type[_EventTableT], with_n_periapses: bool = False) -> _EventTableT:
        idx = _idx(tag)
        if not idx:
            return cls.empty()
        kwargs: dict[str, Any] = dict(
            orbit_id=_str(orbit_ids, idx),
            object_id=_str_opt(object_ids, idx),
            body=_str(bodies, idx),
            epoch=_arr(epochs, idx),
            distance_au=_arr(distance_au, idx),
            distance_km=_arr(distance_km, idx),
            relative_velocity_au_day=_arr(rel_v, idx),
            two_body_energy=_arr(two_body_energy, idx),
            jacobi_constant=_nullable_float(_arr(jacobi, idx)),
            jacobi_constant_sigma=_nullable_float(_arr(jacobi_sigma, idx)),
            jacobi_constant_l1=_nullable_float(_arr(jacobi_l1, idx)),
            jacobi_constant_l2=_nullable_float(_arr(jacobi_l2, idx)),
        )
        if with_n_periapses:
            kwargs["n_periapses"] = _arr(n_periapses, idx)
        return cls.from_kwargs(**kwargs)

    capture_starts = _capture("capture_start", CaptureStarts)
    capture_ends = _capture("capture_end", CaptureEnds, with_n_periapses=True)

    # Shadow events: shadow_fraction / illumination are carried through the
    # C ABI.
    def _shadow(tag: str, cls: type[_EventTableT]) -> _EventTableT:
        idx = _idx(tag)
        if not idx:
            return cls.empty()
        return cls.from_kwargs(
            orbit_id=_str(orbit_ids, idx),
            object_id=_str_opt(object_ids, idx),
            body=_str(bodies, idx),
            epoch=_arr(epochs, idx),
            shadow_fraction=_arr(shadow_fraction, idx),
            illumination=_arr(illumination, idx),
        )

    shadow_entries = _shadow("shadow_entry", ShadowEntries)
    shadow_exits = _shadow("shadow_exit", ShadowExits)

    # Covariance-regime-change events: the UncertaintyMethod.AUTO audit
    # trail (linear <-> second-order transitions at CA-window boundaries).
    # Kind codes: -1 = not applicable, else EMPYREAN_COVARIANCE_KIND_*.
    def _kind_label(code: int) -> str | None:
        return _KIND_BY_CODE[int(code)].value if int(code) >= 0 else None

    crc_idx = _idx("covariance_regime_change")
    if crc_idx:
        covariance_regime_changes = CovarianceRegimeChanges.from_kwargs(
            orbit_id=_str(orbit_ids, crc_idx),
            object_id=_str_opt(object_ids, crc_idx),
            body=_str_opt(bodies, crc_idx),
            epoch=_arr(epochs, crc_idx),
            previous_kind=[_kind_label(c) for c in _arr(previous_kind, crc_idx)],
            resolved_kind=[_kind_label(c) for c in _arr(regime_resolved_kind, crc_idx)],
            kappa=_nullable_float(_arr(kappa, crc_idx)),
            threshold_below=_nullable_float(_arr(threshold_below, crc_idx)),
            threshold_above=_nullable_float(_arr(threshold_above, crc_idx)),
        )
    else:
        covariance_regime_changes = CovarianceRegimeChanges.empty()

    return Events(
        summary=summary,
        close_approach_starts=close_approach_starts,
        close_approach_ends=close_approach_ends,
        periapses=periapses,
        impacts=impacts,
        possible_impacts=possible_impacts,
        atmospheric_entries=atmospheric_entries,
        atmospheric_exits=atmospheric_exits,
        capture_starts=capture_starts,
        capture_ends=capture_ends,
        shadow_entries=shadow_entries,
        shadow_exits=shadow_exits,
        covariance_regime_changes=covariance_regime_changes,
    )
