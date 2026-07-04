"""Multi-method impact-probability and B-plane computation.

Two functions live here, both wrapping
:func:`empyrean_core::impact::compute_impact_probabilities` and
:func:`empyrean_core::impact::compute_b_planes` (via the C ABI). Each
runs one full propagation per supplied :class:`UncertaintyMethod`
variant and returns a typed quivr table tagged with the method that
produced each row — exactly what you want when comparing linear,
second-order, and Monte-Carlo IP / B-plane breakdowns on the same
encounter.

The companion quivr classes :class:`ImpactProbabilities` and
:class:`BPlanes` carry a ``method`` string column (rather than an
opaque int tag) so consumers can group / filter without consulting a
mapping table — `ips.where(pc.field("method") == "second_order")`
(with `import pyarrow.compute as pc`) reads
exactly like the kind of query you actually want to write.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from empyrean._convert import (
    AnyOrbits,
    coordinates_to_arrays,
)
from empyrean.coordinates.enums import Origin
from empyrean.coordinates.epoch import Epochs
from empyrean.propagation.config import (
    _DATACLASS_TO_INT,
    _UNCERTAINTY_METHOD_TO_INT,
    MonteCarlo,
    SigmaPoint,
    UncertaintyMethod,
)

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# Value type for the flat-array orbit-field dict assembled by
# `_common_orbit_args`. The fields are heterogeneous numpy arrays —
# float64 element/covariance/non-grav arrays, a bool `has_covariance`
# mask, int32 representation/frame/origin tag arrays — plus an
# optional float64 `non_grav_dts` (None when no row carries a DT).
_OrbitArg = FloatArray | npt.NDArray[np.bool_] | npt.NDArray[np.int32] | None

# ── Method-name canonical strings ────────────────────────────
#
# Stable text labels that show up in the `method` column of the
# tables below. They line up with the `UncertaintyMethod` Python
# enum's lowercase names, so a user who already has a method enum
# can do `ips.where(pc.field("method") == method.value)` (or the analogous
# ``str(method)``) without an extra mapping step.

METHOD_FIRST_ORDER = "first_order"
METHOD_SECOND_ORDER = "second_order"
METHOD_SIGMA_POINT = "sigma_point"
METHOD_MONTE_CARLO = "monte_carlo"
METHOD_AUTO = "auto"

# Internal: maps the Rust-side integer tag returned by
# `_compute_impact_probabilities` / `_compute_b_planes` to the
# canonical Python label. The Rust side uses 0=First, 1=Second,
# 2=SigmaPoint, 3=MonteCarlo, 4=Auto (matches the EMPYREAN_UNCERTAINTY_*
# constants in empyrean-c/src/propagate.rs). A missing tag-4 entry silently
# mislabelled Auto IP/B-plane rows.
_TAG_TO_METHOD = {
    0: METHOD_FIRST_ORDER,
    1: METHOD_SECOND_ORDER,
    2: METHOD_SIGMA_POINT,
    3: METHOD_MONTE_CARLO,
    4: METHOD_AUTO,
}


# ── Quivr tables ──────────────────────────────────────────────


class ImpactProbabilities(qv.Table):
    """Probabilistic impact assessments tagged by uncertainty method.

    One row per (method × orbit × body) close-approach encounter.
    Mirrors :class:`empyrean.propagation.events.PossibleImpacts` plus
    a ``method`` column and the Monte-Carlo bookkeeping (sample
    count, impact count) that's nullable on the per-encounter table
    but populated when MC is among the requested methods.

    The closest-approach time is carried as an :class:`Epochs`
    sub-table (always emitted in TDB) rather than a raw MJD float
    so consumers can do ``ips.epochs.to_utc()`` and get back the
    same row alignment.
    """

    method = qv.LargeStringColumn()
    """Uncertainty method that produced this row. One of
    ``"first_order"`` / ``"second_order"`` / ``"sigma_point"`` /
    ``"monte_carlo"``."""
    orbit_id = qv.LargeStringColumn()
    """Orbit primary key — matches the input ``Orbits.orbit_id``."""
    object_id = qv.LargeStringColumn(nullable=True)
    """Object metadata label, if carried on the input orbit."""
    body = qv.LargeStringColumn()
    """Body name as the canonical :class:`Origin` string
    (``"Earth"`` / ``"Moon"`` / ``"asteroid_99942"``). Use
    :meth:`Origin.from_string` to lift the column value into a typed
    :class:`Origin`."""
    epochs = Epochs.as_column()
    """Closest-approach epoch as an :class:`Epochs` sub-table (TDB)."""
    miss_distance_au = qv.Float64Column()
    """Closest-approach geocentric (or body-centric) distance at the
    nominal trajectory, in AU."""
    miss_distance_km = qv.Float64Column()
    """Closest-approach distance in km — convenience copy."""
    effective_radius_au = qv.Float64Column()
    """Body radius inflated for atmospheric capture / gravitational
    focusing (AU): :math:`R_\\mathrm{eff}^2 = R^2(1 + (v_\\mathrm{esc}/v_\\infty)^2)`.
    Impact requires the orbit pierce a sphere of this radius."""
    effective_radius_km = qv.Float64Column()
    """Effective radius in km."""
    sigma_distance_au = qv.Float64Column()
    """1σ uncertainty along the miss-distance direction (AU);
    linearised even for Monte-Carlo rows."""
    sigma_distance_km = qv.Float64Column()
    """1σ miss-distance uncertainty in km."""
    ip_linear = qv.Float64Column()
    """Linear (Φ Σ Φᵀ-mapped) impact probability. Always populated."""
    relative_velocity_au_day = qv.Float64Column()
    """Hyperbolic-excess velocity magnitude at the close approach
    (AU/day). Independent of method."""
    ip_second_order = qv.Float64Column(nullable=True)
    """Park-Scheeres second-order Gaussian impact probability.
    Populated when the propagation carried STTs (i.e. ``method`` is
    ``"second_order"`` or higher)."""
    nonlinearity = qv.Float64Column(nullable=True)
    """Local nonlinearity diagnostic at the close-approach epoch —
    a scalar measure of how much the second-order STT contribution
    would shift the propagated mean relative to the linear map.
    Populated when STTs are available. Treat qualitatively: large
    values indicate :attr:`ip_linear` may disagree with sample-based
    estimates."""
    ip_agm = qv.Float64Column(nullable=True)
    """Reserved; not populated by any uncertainty method exposed in
    this release."""
    ip_mc = qv.Float64Column(nullable=True)
    """Monte-Carlo impact probability —
    :attr:`mc_n_impacts` / :attr:`mc_n_samples`. Populated only when
    ``method = "monte_carlo"``."""
    mc_n_samples = qv.UInt64Column(nullable=True)
    """Number of virtual-asteroid samples drawn (MC rows only)."""
    mc_n_impacts = qv.UInt64Column(nullable=True)
    """Sample count that intersected the effective-radius sphere
    (MC rows only)."""


class BPlanes(qv.Table):
    """B-plane geometry breakdowns tagged by uncertainty method.

    One row per (method × orbit × body) close-approach encounter.
    Mirrors :class:`empyrean_core.impact.BPlaneData` (= villeneuve's
    upstream type) flattened to columns: :math:`B \\cdot R`,
    :math:`B \\cdot T`, miss distance, hyperbolic excess velocity,
    the projected covariance, and the 3σ uncertainty ellipse
    (semi-major / semi-minor / rotation angle).

    Closest-approach time carried as an :class:`Epochs` sub-table —
    same convention as :class:`ImpactProbabilities`.
    """

    method = qv.LargeStringColumn()
    """Uncertainty method tag — see :class:`ImpactProbabilities`."""
    body = qv.LargeStringColumn()
    """Body name; B-plane is defined relative to this body's
    hyperbolic-excess-velocity asymptote at closest approach."""
    epochs = Epochs.as_column()
    """Closest-approach epoch (TDB)."""
    b_dot_t_km = qv.Float64Column()
    """Öpik :math:`B \\cdot T` coordinate in km. T points along the
    projection of the planet's heliocentric velocity onto the
    B-plane; controls the *along-track* encounter geometry and the
    resonant-return / keyhole structure."""
    b_dot_r_km = qv.Float64Column()
    """Öpik :math:`B \\cdot R` coordinate in km. R completes a
    right-handed frame with the inbound asymptote; controls the
    *cross-track* miss component."""
    b_mag_km = qv.Float64Column()
    """Magnitude :math:`|B| = \\sqrt{(B \\cdot T)^2 + (B \\cdot R)^2}`
    in km. Impact requires :math:`|B| < R_\\mathrm{eff}`."""
    v_inf_km_s = qv.Float64Column()
    """Hyperbolic excess velocity :math:`v_\\infty` at the close
    approach (km/s)."""
    effective_radius_km = qv.Float64Column()
    """Gravitational-focusing-inflated radius
    :math:`R_\\mathrm{eff}^2 = R^2 (1 + (v_\\mathrm{esc} / v_\\infty)^2)`
    in km — the radius :math:`|B|` is compared against."""
    body_radius_km = qv.Float64Column()
    """Body radius (km), pre-inflation."""
    cov_tt_km2 = qv.Float64Column(nullable=True)
    """B-plane projected covariance, T-T component (km²)."""
    cov_tr_km2 = qv.Float64Column(nullable=True)
    """B-plane projected covariance, T-R off-diagonal (km²)."""
    cov_rr_km2 = qv.Float64Column(nullable=True)
    """B-plane projected covariance, R-R component (km²)."""
    semi_major_3sig_km = qv.Float64Column(nullable=True)
    """Semi-major axis of the 3σ uncertainty ellipse on the B-plane
    (km), eigenvector of the projected covariance."""
    semi_minor_3sig_km = qv.Float64Column(nullable=True)
    """Semi-minor axis of the 3σ uncertainty ellipse on the B-plane
    (km)."""
    ellipse_angle_rad = qv.Float64Column(nullable=True)
    """Rotation angle of the uncertainty ellipse from the +T axis
    (radians)."""
    ip_linear = qv.Float64Column(nullable=True)
    """Linear impact probability evaluated against the projected
    B-plane covariance — convenience copy of the IP that matches
    this B-plane row."""


# ── Helpers ───────────────────────────────────────────────────

UncertaintyMethodLike = UncertaintyMethod | SigmaPoint | MonteCarlo | str | int


def _method_to_tag(m: UncertaintyMethodLike) -> int:
    """Map a Python-level method spec to the int tag the Rust side expects."""
    if isinstance(m, (SigmaPoint, MonteCarlo)):
        return _DATACLASS_TO_INT[type(m)]
    if isinstance(m, str):
        tag = _UNCERTAINTY_METHOD_TO_INT.get(m.lower())
        if tag is None:
            raise ValueError(f"unknown uncertainty method: {m}")
        return tag
    if isinstance(m, UncertaintyMethod):
        return _UNCERTAINTY_METHOD_TO_INT[m]
    if isinstance(m, int):
        return m
    raise TypeError(f"unsupported method spec: {type(m).__name__}")


def _tags_to_method_strings(tags: npt.NDArray[np.integer[Any]]) -> list[str]:
    """Convert the Rust-side integer tag column to the canonical
    string labels exposed on the quivr tables. Unknown tags (which
    shouldn't occur — every value comes from a Rust match arm) fall
    back to a stable ``"unknown_<n>"`` string rather than raising,
    so a downstream consumer doesn't lose the rest of the table to
    one corrupt entry."""
    return [_TAG_TO_METHOD.get(int(t), f"unknown_{int(t)}") for t in tags]


def _common_orbit_args(orbits: AnyOrbits) -> dict[str, _OrbitArg]:
    """Pull the flat-array orbit fields the Rust side needs.

    Mirrors what :func:`empyrean.propagate` extracts before it
    dispatches to ``_propagate`` — same orbit shape, same fields,
    same units.
    """
    (
        epochs_arr,
        elements_arr,
        covariances_arr,
        has_cov_arr,
        reps_arr,
        frames_arr,
        origins_arr,
    ) = coordinates_to_arrays(orbits.coordinates)

    n = len(orbits)
    a1s = np.zeros(n, dtype=np.float64)
    a2s = np.zeros(n, dtype=np.float64)
    a3s = np.zeros(n, dtype=np.float64)
    non_grav_dts: FloatArray | None = None
    ng_alphas: FloatArray | None = None
    ng_r0s: FloatArray | None = None
    ng_ms: FloatArray | None = None
    ng_ns: FloatArray | None = None
    ng_ks: FloatArray | None = None
    # `orbits.non_grav` is a nullable sub-table. quivr returns a
    # zero-or-all-null `NonGravParams` instance even when the caller
    # never passed `non_grav` to `from_kwargs`, so `is not None` alone
    # is not enough to gate. Read the columns with `zero_copy_only=False`
    # so arrow nulls promote to NaN, then normalize via `nan_to_num`.
    # Mirrors the pattern in `propagation/propagate.py`.
    if orbits.non_grav is not None:
        ng = orbits.non_grav
        a1s = np.nan_to_num(
            np.asarray(ng.a1.to_numpy(zero_copy_only=False), dtype=np.float64),
            nan=0.0,
        )
        a2s = np.nan_to_num(
            np.asarray(ng.a2.to_numpy(zero_copy_only=False), dtype=np.float64),
            nan=0.0,
        )
        a3s = np.nan_to_num(
            np.asarray(ng.a3.to_numpy(zero_copy_only=False), dtype=np.float64),
            nan=0.0,
        )
        # SBDB non-grav DT — surface as a NaN-sentineled array; the
        # Rust binding treats NaN as "no delay" per orbit. Skip the
        # whole array when no row has a finite DT (saves the FFI
        # marshal cost on the asteroid common case).
        dt_col = np.asarray(ng.dt.to_numpy(zero_copy_only=False), dtype=np.float64)
        if np.isfinite(dt_col).any():
            non_grav_dts = dt_col
        # Marsden g(r) exponents. Without them, a comet's custom g(r)
        # silently collapses to inverse-square on the IP / B-plane input
        # path (same class as the c37m propagate-input fix). Surface all
        # five together only when at least one row carries a custom g(r)
        # (alpha != 0); the binding's g(r) override needs the full set.
        alpha_col = np.nan_to_num(
            np.asarray(ng.alpha.to_numpy(zero_copy_only=False), dtype=np.float64),
            nan=0.0,
        )
        if (alpha_col != 0.0).any():

            def _col(name: str) -> FloatArray:
                return np.nan_to_num(
                    np.asarray(
                        getattr(ng, name).to_numpy(zero_copy_only=False),
                        dtype=np.float64,
                    ),
                    nan=0.0,
                )

            ng_alphas = alpha_col
            ng_r0s = _col("r0")
            ng_ms = _col("m")
            ng_ns = _col("n")
            ng_ks = _col("k")
    return {
        "epochs": epochs_arr,
        "elements": elements_arr,
        "covariances": covariances_arr,
        "has_covariance": has_cov_arr,
        "representations": reps_arr,
        "frames": frames_arr,
        "origins": origins_arr,
        "a1s": a1s,
        "a2s": a2s,
        "a3s": a3s,
        "non_grav_dts": non_grav_dts,
        "ng_alphas": ng_alphas,
        "ng_r0s": ng_r0s,
        "ng_ms": ng_ms,
        "ng_ns": ng_ns,
        "ng_ks": ng_ks,
    }


def _coerce_end_mjd_tdb(epoch: float | Epochs) -> float:
    """Accept either a plain MJD float or an :class:`Epochs` of length 1."""
    if isinstance(epoch, Epochs):
        tdb = epoch.to_tdb()
        arr = tdb.mjd.to_numpy(zero_copy_only=False)
        if len(arr) != 1:
            raise ValueError("end_epoch must be a single epoch (Epochs of length 1 or a float MJD)")
        return float(arr[0])
    return float(epoch)


def _nan_to_null(arr: FloatArray) -> pa.Array:
    """Convert a float64 numpy array with NaN sentinels to a nullable
    pyarrow array — quivr nullable columns expect arrow nulls, not
    NaN, for downstream consumers (pandas, polars, joins, …)."""
    mask = np.isnan(arr)
    return pa.array(arr, mask=mask)


def _zero_to_null(arr: npt.NDArray[np.uint64]) -> pa.Array:
    """Convert a uint64 numpy array with 0 sentinels to a nullable
    pyarrow array — used for the MC sample / impact counts which
    return 0 when the row's method wasn't Monte-Carlo."""
    mask = arr == 0
    return pa.array(arr, mask=mask)


# ── Public API ────────────────────────────────────────────────


def _recover_user_ids(
    fabricated_orbit_ids: Sequence[str],
    user_orbit_ids: Sequence[str],
    user_object_ids: Sequence[str | None] | None,
) -> tuple[list[str], list[str | None]]:
    """Parse the C ABI's fabricated ``"orbit_{i}"`` strings back to
    indices and return the corresponding user-supplied orbit_id and
    object_id strings. Falls back to the fabricated value for orbit_id
    and ``None`` for object_id when the parse fails (defensive — every
    row should match the pattern in practice).
    """
    out_orbit_ids: list[str] = []
    out_object_ids: list[str | None] = []
    for fab in fabricated_orbit_ids:
        idx: int | None = None
        if isinstance(fab, str) and fab.startswith("orbit_"):
            try:
                idx = int(fab[len("orbit_") :])
            except ValueError:
                idx = None
        if idx is not None and 0 <= idx < len(user_orbit_ids):
            out_orbit_ids.append(user_orbit_ids[idx])
            obj = (
                user_object_ids[idx]
                if user_object_ids is not None and idx < len(user_object_ids)
                else None
            )
            out_object_ids.append(obj if obj else None)
        else:
            out_orbit_ids.append(fab)
            out_object_ids.append(None)
    return out_orbit_ids, out_object_ids


def compute_impact_probabilities(
    orbits: AnyOrbits,
    end_epoch: float | Epochs,
    methods: Sequence[UncertaintyMethodLike],
    body_filter: Sequence[Origin | str] | None = None,
) -> ImpactProbabilities:
    """Run impact-probability detection over a propagation window with
    one full propagation per supplied :class:`UncertaintyMethod`.

    Parameters
    ----------
    orbits : CartesianOrbits | CometaryOrbits | KeplerianOrbits | SphericalOrbits
        Input orbits with optional covariance and non-gravitational
        parameters. Same shape :func:`empyrean.propagate` accepts.
    end_epoch : float | Epochs
        End of the propagation window. MJD TDB float or a length-1
        :class:`Epochs` (any time scale — converted to TDB internally).
    methods : sequence of UncertaintyMethod / str / dataclass
        Which uncertainty methods to run. One full propagation runs
        per method (in order); the result rows are tagged with the
        method via the ``method`` string column.
    body_filter : sequence of Origin | str, optional
        Restrict event monitoring to specific bodies. Pass
        :class:`Origin` instances (e.g. ``[Origin.EARTH, Origin.MOON]``)
        or canonical names. Default monitors every body in the
        ephemeris.

    Returns
    -------
    ImpactProbabilities
        Quivr table — one row per (method × orbit × body) encounter.
        See the class for the full column list. ``method`` takes
        ``"first_order"`` / ``"second_order"`` / ``"sigma_point"`` /
        ``"monte_carlo"``.

    Notes
    -----
    Each method's result is computed with a separate propagation run —
    different uncertainty backings (linear, second-order, sample cloud)
    don't yet share an integration step. The cost scales linearly with
    ``len(methods)``.
    """
    from empyrean._convert import origin_to_naif
    from empyrean._empyrean_rs import _compute_impact_probabilities

    args = _common_orbit_args(orbits)
    method_tags = [_method_to_tag(m) for m in methods]
    end_mjd = _coerce_end_mjd_tdb(end_epoch)
    filter_arg = [origin_to_naif(o) for o in body_filter] if body_filter else None

    out = _compute_impact_probabilities(
        epochs=args["epochs"],
        elements=args["elements"],
        covariances=args["covariances"],
        has_covariance=args["has_covariance"],
        representations=args["representations"],
        frames=args["frames"],
        origins=args["origins"],
        end_mjd_tdb=end_mjd,
        a1s=args["a1s"],
        a2s=args["a2s"],
        a3s=args["a3s"],
        method_tags=method_tags,
        body_filter_naif=filter_arg,
        non_grav_dts=args["non_grav_dts"],
        ng_alphas=args["ng_alphas"],
        ng_r0s=args["ng_r0s"],
        ng_ms=args["ng_ms"],
        ng_ns=args["ng_ns"],
        ng_ks=args["ng_ks"],
    )

    # The C ABI fabricates each row's orbit_id as `"orbit_{i}"` and
    # leaves object_id empty. Recover the user-supplied IDs by parsing
    # the index out of the fabricated
    # string and looking up the orbits batch.
    user_orbit_ids = orbits.orbit_id.to_pylist()
    user_object_ids = orbits.object_id.to_pylist() if orbits.object_id is not None else None
    fixed_orbit_ids, fixed_object_ids = _recover_user_ids(
        out["orbit_id"], user_orbit_ids, user_object_ids
    )

    return ImpactProbabilities.from_kwargs(
        method=_tags_to_method_strings(out["method_tag"]),
        orbit_id=fixed_orbit_ids,
        object_id=fixed_object_ids,
        body=out["body"],
        epochs=Epochs.from_kwargs(mjd=out["epoch_mjd_tdb"], scale="tdb"),
        miss_distance_au=out["miss_distance_au"],
        miss_distance_km=out["miss_distance_km"],
        effective_radius_au=out["effective_radius_au"],
        effective_radius_km=out["effective_radius_km"],
        sigma_distance_au=out["sigma_distance_au"],
        sigma_distance_km=out["sigma_distance_km"],
        ip_linear=out["ip_linear"],
        relative_velocity_au_day=out["relative_velocity_au_day"],
        ip_second_order=_nan_to_null(out["ip_second_order"]),
        nonlinearity=_nan_to_null(out["nonlinearity"]),
        ip_agm=_nan_to_null(out["ip_agm"]),
        ip_mc=_nan_to_null(out["ip_mc"]),
        mc_n_samples=_zero_to_null(out["mc_n_samples"]),
        mc_n_impacts=_zero_to_null(out["mc_n_impacts"]),
    )


def compute_b_planes(
    orbits: AnyOrbits,
    end_epoch: float | Epochs,
    methods: Sequence[UncertaintyMethodLike],
    body_filter: Sequence[Origin | str] | None = None,
) -> BPlanes:
    """Run B-plane breakdown extraction over a propagation window with
    one full propagation per supplied :class:`UncertaintyMethod`.

    Same call shape as :func:`compute_impact_probabilities`, but the
    output table carries the B-plane geometry (B·R, B·T, miss
    distance, 3σ ellipse, projected covariance) for every detected
    close approach instead of the IP record.

    Returns
    -------
    BPlanes
        Quivr table — one row per (method × orbit × body) close
        approach. See the class for the full column list.
    """
    from empyrean._convert import origin_to_naif
    from empyrean._empyrean_rs import _compute_b_planes

    args = _common_orbit_args(orbits)
    method_tags = [_method_to_tag(m) for m in methods]
    end_mjd = _coerce_end_mjd_tdb(end_epoch)
    filter_arg = [origin_to_naif(o) for o in body_filter] if body_filter else None

    out = _compute_b_planes(
        epochs=args["epochs"],
        elements=args["elements"],
        covariances=args["covariances"],
        has_covariance=args["has_covariance"],
        representations=args["representations"],
        frames=args["frames"],
        origins=args["origins"],
        end_mjd_tdb=end_mjd,
        a1s=args["a1s"],
        a2s=args["a2s"],
        a3s=args["a3s"],
        method_tags=method_tags,
        body_filter_naif=filter_arg,
        non_grav_dts=args["non_grav_dts"],
        ng_alphas=args["ng_alphas"],
        ng_r0s=args["ng_r0s"],
        ng_ms=args["ng_ms"],
        ng_ns=args["ng_ns"],
        ng_ks=args["ng_ks"],
    )

    return BPlanes.from_kwargs(
        method=_tags_to_method_strings(out["method_tag"]),
        body=out["body"],
        epochs=Epochs.from_kwargs(mjd=out["epoch_mjd_tdb"], scale="tdb"),
        b_dot_t_km=out["b_dot_t_km"],
        b_dot_r_km=out["b_dot_r_km"],
        b_mag_km=out["b_mag_km"],
        v_inf_km_s=out["v_inf_km_s"],
        effective_radius_km=out["effective_radius_km"],
        body_radius_km=out["body_radius_km"],
        cov_tt_km2=_nan_to_null(out["cov_tt_km2"]),
        cov_tr_km2=_nan_to_null(out["cov_tr_km2"]),
        cov_rr_km2=_nan_to_null(out["cov_rr_km2"]),
        semi_major_3sig_km=_nan_to_null(out["semi_major_3sig_km"]),
        semi_minor_3sig_km=_nan_to_null(out["semi_minor_3sig_km"]),
        ellipse_angle_rad=_nan_to_null(out["ellipse_angle_rad"]),
        ip_linear=_nan_to_null(out["ip_linear"]),
    )
