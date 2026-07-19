"""Orbit determination: read observations, determine orbits, evaluate, refine."""

from __future__ import annotations

import os
from typing import Any, TypeAlias

import numpy as np
import pyarrow as pa
import quivr as qv

from empyrean._convert import (
    _COORD_TYPE_MAP,
    AnyOrbits,
    coordinates_to_arrays,
    int_to_frame,
    naif_to_origin,
)
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.covariance import (
    CartesianCovariance as _CartesianCovariance,
)
from empyrean.coordinates.covariance import _CovarianceTable
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.radar_observations import ADESRadarObservations
from empyrean.od.residuals import (
    AcceptabilityReport,
    ObservationResults,
    ResidualSummary,
    StationBiases,
)
from empyrean.od.result import (
    BandStat,
    DetermineResult,
    EvaluateResult,
    GateRecord,
    ODConfig,
    PhotometryModel,
    PhotometryResult,
    SolvedCovariance,
)
from empyrean.orbits.orbits import CartesianOrbits
from empyrean.propagation.config import _FORCE_MODEL_TO_INT, ForceModelTier

# ``CartesianCovariance`` is built dynamically by ``_make_covariance_class``,
# whose declared return type is the bare ``type``; that loses the injected
# ``from_matrix`` / ``from_kwargs`` constructors. Re-bind it through
# ``type[_CovarianceTable]`` (the Protocol describing exactly that
# dynamically-injected surface) so those constructors type-check at the call
# sites below. The runtime object is unchanged.
CartesianCovariance: type[_CovarianceTable] = _CartesianCovariance

# Values crossing the FFI boundary arrive as an untyped dict keyed by name;
# every consumer here treats them positionally so the value type is open.
ResultDict: TypeAlias = dict[str, Any]

# Accepted value types for ``qv.Table.from_kwargs`` keyword arguments: a
# data source (pyarrow array, list, nested Table, or numpy array) or a
# scalar attribute value (int / float / str).
KwargValue: TypeAlias = pa.Array | list[Any] | qv.Table | np.ndarray | int | float | str


def _force_model_to_int(force_model: ForceModelTier | str | int) -> int:
    """Convert a force model to its integer code."""
    if isinstance(force_model, str):
        fm_int = _FORCE_MODEL_TO_INT.get(force_model.lower())
        if fm_int is None:
            raise ValueError(f"unknown force model: {force_model}")
        return fm_int
    elif isinstance(force_model, ForceModelTier):
        return _FORCE_MODEL_TO_INT[force_model]
    elif isinstance(force_model, int):
        return force_model
    else:
        raise TypeError(f"force_model must be ForceModelTier, str, or int, got {type(force_model)}")


def _obs_to_dict(observations: ADESObservations) -> dict[str, list[Any] | np.ndarray]:
    """Convert an :class:`ADESObservations` quivr table to the flat-dict
    shape consumed by the Rust ``_determine`` / ``_evaluate_single`` /
    ``_refine_single`` entry points.

    Carries the full ADES schema the C ABI's :c:type:`EmpyreanObservation`
    struct supports — perm_id / prov_id / trk_sub / stn / mode /
    obs_time / ra / dec / rms_ra / rms_dec / rms_corr / mag / rms_mag /
    band / ast_cat / sys / ctr / pos1-3.
    """

    def _f(col: pa.Array) -> np.ndarray:
        # quivr nullable Float64Column.to_numpy(zero_copy_only=False)
        # returns NaN for nulls already.
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)

    n_stars_list = observations.n_stars.to_pylist()
    n_stars_arr = np.asarray([v if v is not None else -1 for v in n_stars_list], dtype=np.int32)

    return {
        # ── Identification ──
        "perm_id": observations.perm_id.to_pylist(),
        "prov_id": observations.prov_id.to_pylist(),
        "trk_sub": observations.trk_sub.to_pylist(),
        "obs_id": observations.obs_id.to_pylist(),
        "obs_sub_id": observations.obs_sub_id.to_pylist(),
        "trk_id": observations.trk_id.to_pylist(),
        # ── Observer ──
        "stn": observations.stn.to_pylist(),
        "mode": observations.mode.to_pylist(),
        "prog": observations.prog.to_pylist(),
        # ── Observer location ──
        "sys": observations.sys.to_pylist(),
        "ctr": _f(observations.ctr),
        "pos1": _f(observations.pos1),
        "pos2": _f(observations.pos2),
        "pos3": _f(observations.pos3),
        # ── Astrometry ──
        "obs_time": observations.obs_time.to_pylist(),
        "ra": _f(observations.ra),
        "dec": _f(observations.dec),
        # ── Uncertainties ──
        "rms_ra": _f(observations.rms_ra),
        "rms_dec": _f(observations.rms_dec),
        "rms_corr": _f(observations.rms_corr),
        # ── Catalog ──
        "ast_cat": observations.ast_cat.to_pylist(),
        # ── Photometry ──
        "mag": _f(observations.mag),
        "rms_mag": _f(observations.rms_mag),
        "band": observations.band.to_pylist(),
        "phot_cat": observations.phot_cat.to_pylist(),
        "phot_ap": _f(observations.phot_ap),
        # ── Supplementary diagnostics ──
        "log_snr": _f(observations.log_snr),
        "seeing": _f(observations.seeing),
        "exp": _f(observations.exp),
        "rms_fit": _f(observations.rms_fit),
        "n_stars": n_stars_arr,
        "notes": observations.notes.to_pylist(),
        "remarks": observations.remarks.to_pylist(),
    }


def _radar_to_dict(radar: ADESRadarObservations) -> dict[str, list[Any] | np.ndarray]:
    """Convert an :class:`ADESRadarObservations` quivr table to the flat-
    dict shape consumed by the Rust ``_determine`` radar entry point.

    Mirrors :func:`_obs_to_dict`. All values are ADES-native — delay in
    **seconds**, ``rms_delay`` in **microseconds**, ``doppler`` /
    ``rms_doppler`` in **Hz**, ``frq`` in **MHz** — and no conversion is
    applied; the single SI normalization happens once downstream. The
    ``observable`` discriminator (``"delay"`` / ``"doppler"``) crosses the
    boundary explicitly so a 0.0-Hz Doppler is never mistaken for an
    absent one.
    """

    def _f(col: pa.Array) -> np.ndarray:
        # quivr nullable Float64Column.to_numpy(zero_copy_only=False)
        # returns NaN for nulls already — matches the Rust NaN-inactive
        # convention.
        return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)

    # com: Option<bool> tri-state -> i8 with -1 = absent (never 0).
    com_list = radar.com.to_pylist()
    com_arr = np.asarray([1 if v else 0 if v is not None else -1 for v in com_list], dtype=np.int8)

    return {
        # ── Identification ──
        "perm_id": radar.perm_id.to_pylist(),
        "prov_id": radar.prov_id.to_pylist(),
        "trk_sub": radar.trk_sub.to_pylist(),
        # ── Bistatic geometry ──
        "trx": radar.trx.to_pylist(),
        "rcv": radar.rcv.to_pylist(),
        # ── Core measurement ──
        "obs_time": radar.obs_time.to_pylist(),
        "observable": radar.observable.to_pylist(),
        "delay": _f(radar.delay),
        "rms_delay": _f(radar.rms_delay),
        "doppler": _f(radar.doppler),
        "rms_doppler": _f(radar.rms_doppler),
        # ── Reduction metadata ──
        "frq": _f(radar.frq),
        "com": com_arr,
        "log_snr": _f(radar.log_snr),
        "remarks": radar.remarks.to_pylist(),
    }


def _orbits_to_dict(orbits: AnyOrbits) -> dict[str, list[Any] | np.ndarray]:
    """Convert an orbit table to a dict of arrays for the Rust boundary.

    Orbits must be in Cartesian representation. Returns arrays suitable
    for building ``Orbits<AU>`` on the Rust side.
    """
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

    orbit_ids = orbits.orbit_id.to_pylist()
    if orbits.object_id is not None:
        object_ids = [s if s else "" for s in orbits.object_id.to_pylist()]
    else:
        object_ids = [""] * len(orbits)

    # Non-grav parameters — thread the seed orbit's full force model so
    # evaluate / refine operate on the actual non-grav (A1/A2/A3 + g(r) + dt)
    # rather than silently gravity-only, and carry the fitted non-grav
    # covariance so a StateAndNonGrav refine keeps its prior.
    n = len(orbits)
    a1s = np.zeros(n, dtype=np.float64)
    a2s = np.zeros(n, dtype=np.float64)
    a3s = np.zeros(n, dtype=np.float64)
    ng_alphas = np.zeros(n, dtype=np.float64)
    ng_r0s = np.zeros(n, dtype=np.float64)
    ng_ms = np.zeros(n, dtype=np.float64)
    ng_ns = np.zeros(n, dtype=np.float64)
    ng_ks = np.zeros(n, dtype=np.float64)
    non_grav_dts = np.full(n, np.nan, dtype=np.float64)  # NaN = no thermal-lag delay
    has_non_grav_cov = np.zeros(n, dtype=bool)
    non_grav_cov = np.zeros((n, 3, 3), dtype=np.float64)
    if orbits.non_grav is not None:
        ng = orbits.non_grav

        def _col(name: str) -> np.ndarray:
            return np.nan_to_num(
                np.asarray(getattr(ng, name).to_numpy(zero_copy_only=False), dtype=np.float64),
                nan=0.0,
            )

        a1s, a2s, a3s = _col("a1"), _col("a2"), _col("a3")
        ng_alphas, ng_r0s = _col("alpha"), _col("r0")
        ng_ms, ng_ns, ng_ks = _col("m"), _col("n"), _col("k")
        non_grav_dts = np.asarray(ng.dt.to_numpy(zero_copy_only=False), dtype=np.float64)
        for i, c in enumerate(ng.covariance.to_pylist()):
            if c is not None:
                non_grav_cov[i] = np.asarray(c, dtype=np.float64).reshape(3, 3)
                has_non_grav_cov[i] = True

    return {
        "orbit_ids": orbit_ids,
        "object_ids": object_ids,
        "epochs": epochs_arr,
        "elements": elements_arr,
        "covariances": covariances_arr,
        "has_covariance": has_cov_arr,
        "representations": representations_arr,
        "frames": frames_arr,
        "origins": origins_arr,
        "a1s": a1s,
        "a2s": a2s,
        "a3s": a3s,
        "ng_alphas": ng_alphas,
        "ng_r0s": ng_r0s,
        "ng_ms": ng_ms,
        "ng_ns": ng_ns,
        "ng_ks": ng_ks,
        "non_grav_dts": non_grav_dts,
        "has_non_grav_cov": has_non_grav_cov,
        "non_grav_cov": non_grav_cov,
    }


def _nullable_float(values: np.ndarray) -> pa.Array | np.ndarray:
    """Convert a numpy array to a pyarrow array with NaN -> null."""
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isnan(arr)
    if mask.any():
        return pa.array(arr.tolist(), type=pa.float64(), mask=mask)
    return arr


def _build_observation_results(result: ResultDict) -> ObservationResults:
    """Build ObservationResults from a Rust result dict.

    Mirrors the full ``EmpyreanObservationResult`` surface — every
    upstream field crosses the boundary, including obs_id (cross-match
    key), rejection diagnostics, influence stats, and AT/CT
    decomposition.
    """
    return ObservationResults.from_kwargs(
        # Identification
        obs_id=list(result["obs_ids"]),
        obs_code=list(result["obs_codes"]),
        ast_cat=list(result["ast_cats"]),
        epoch_mjd_tdb=np.asarray(result["obs_epochs"]),
        # Core residuals
        ra_residual=np.asarray(result["ra_residuals"]),
        dec_residual=np.asarray(result["dec_residuals"]),
        chi2=np.asarray(result["chi2s"]),
        dof=np.asarray(result["dofs"], dtype=np.int32),
        probability=np.asarray(result["probabilities"]),
        selected=np.asarray(result["selecteds"]),
        # Residual covariance
        residual_cov_ra=_nullable_float(np.asarray(result["residual_cov_ras"])),
        residual_cov_dec=_nullable_float(np.asarray(result["residual_cov_decs"])),
        residual_cov_corr=_nullable_float(np.asarray(result["residual_cov_corrs"])),
        # Rejection
        rejection_reason=list(result["rejection_reasons"]),
        rejection_criterion=_nullable_float(np.asarray(result["rejection_criterions"])),
        rejection_threshold=_nullable_float(np.asarray(result["rejection_thresholds"])),
        rejection_effective_threshold=_nullable_float(
            np.asarray(result["rejection_effective_thresholds"])
        ),
        rejection_information_loss=_nullable_float(
            np.asarray(result["rejection_information_losses"])
        ),
        # Influence
        cooks_distance=_nullable_float(np.asarray(result["cooks_distances"])),
        leverage=_nullable_float(np.asarray(result["leverages"])),
        fractional_information=_nullable_float(np.asarray(result["fractional_informations"])),
        # Along/cross-track
        along_track=_nullable_float(np.asarray(result["along_tracks"])),
        cross_track=_nullable_float(np.asarray(result["cross_tracks"])),
        along_track_error=_nullable_float(np.asarray(result["along_track_errors"])),
        cross_track_error=_nullable_float(np.asarray(result["cross_track_errors"])),
        track_position_angle_deg=_nullable_float(np.asarray(result["track_position_angles"])),
    )


def _build_residual_summary(result: ResultDict, prefix: str = "") -> ResidualSummary:
    """Build ResidualSummary from a Rust result dict.

    Mirrors the full ``EmpyreanResidualSummary`` surface — including
    weighted RMS, mean / std, AT / CT RMS, and dof.
    """
    p = prefix
    return ResidualSummary(
        num_obs=int(result[f"{p}num_obs"]),
        num_selected=int(result[f"{p}num_selected"]),
        num_rejected=int(result[f"{p}num_rejected"]),
        chi2=float(result[f"{p}chi2"]),
        dof=int(result[f"{p}dof"]),
        reduced_chi2=float(result[f"{p}reduced_chi2"]),
        rms_ra_arcsec=float(result[f"{p}rms_ra"]),
        rms_dec_arcsec=float(result[f"{p}rms_dec"]),
        rms_combined_arcsec=float(result[f"{p}rms_combined"]),
        weighted_rms_ra_arcsec=float(result[f"{p}weighted_rms_ra"]),
        weighted_rms_dec_arcsec=float(result[f"{p}weighted_rms_dec"]),
        weighted_rms_combined_arcsec=float(result[f"{p}weighted_rms_combined"]),
        mean_ra_arcsec=float(result[f"{p}mean_ra"]),
        mean_dec_arcsec=float(result[f"{p}mean_dec"]),
        std_ra_arcsec=float(result[f"{p}std_ra"]),
        std_dec_arcsec=float(result[f"{p}std_dec"]),
        rms_along_track_arcsec=float(result[f"{p}rms_along_track"]),
        rms_cross_track_arcsec=float(result[f"{p}rms_cross_track"]),
    )


def _build_acceptability_report(
    result: ResultDict, prefix: str = "acceptability_"
) -> AcceptabilityReport:
    """Build :class:`AcceptabilityReport` from a Rust result dict."""

    p = prefix
    return AcceptabilityReport(
        fit_acceptable=bool(result[f"{p}fit_acceptable"]),
        extrapolation_acceptable=bool(result[f"{p}extrapolation_acceptable"]),
        converged_ok=bool(result[f"{p}converged_ok"]),
        reduced_chi2_ok=bool(result[f"{p}reduced_chi2_ok"]),
        reduced_chi2_value=float(result[f"{p}reduced_chi2_value"]),
        reduced_chi2_threshold=float(result[f"{p}reduced_chi2_threshold"]),
        rms_ok=bool(result[f"{p}rms_ok"]),
        rms_value_arcsec=float(result[f"{p}rms_value_arcsec"]),
        rms_threshold_arcsec=float(result[f"{p}rms_threshold_arcsec"]),
        residual_isotropy_ok=bool(result[f"{p}residual_isotropy_ok"]),
        at_ct_ratio_value=float(result[f"{p}at_ct_ratio_value"]),
        at_ct_ratio_threshold=float(result[f"{p}at_ct_ratio_threshold"]),
        covariance_ok=bool(result[f"{p}covariance_ok"]),
        arc_coverage_ok=bool(result[f"{p}arc_coverage_ok"]),
        arc_days_value=float(result[f"{p}arc_days_value"]),
        arc_days_threshold=float(result[f"{p}arc_days_threshold"]),
        fractional_sigma_a_ok=bool(result[f"{p}fractional_sigma_a_ok"]),
        fractional_sigma_a_value=float(result[f"{p}fractional_sigma_a_value"]),
        fractional_sigma_a_threshold=float(result[f"{p}fractional_sigma_a_threshold"]),
    )


def _build_cartesian_orbits_single(result: ResultDict, prefix: str = "") -> CartesianOrbits:
    """Build CartesianOrbits (single orbit) from a Rust result dict."""

    p = prefix
    epoch = np.asarray([result[f"{p}epoch"]])
    x = np.asarray([result[f"{p}x"]])
    y = np.asarray([result[f"{p}y"]])
    z = np.asarray([result[f"{p}z"]])
    vx = np.asarray([result[f"{p}vx"]])
    vy = np.asarray([result[f"{p}vy"]])
    vz = np.asarray([result[f"{p}vz"]])
    frame = int_to_frame(int(result[f"{p}frame"]))
    origin = naif_to_origin(int(result[f"{p}origin"]))

    cov_flat = result.get(f"{p}covariance")
    if cov_flat is not None:
        cov_matrix = np.asarray(cov_flat).reshape(1, 6, 6)
        cov = CartesianCovariance.from_matrix(cov_matrix)
    else:
        cov = None

    coord_kwargs: dict[str, KwargValue] = {
        "epoch": epoch,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "frame": frame.value if hasattr(frame, "value") else frame,
        "origin": [origin],
    }
    if cov is not None:
        coord_kwargs["covariance"] = cov

    cart_coords = CartesianCoordinates.from_kwargs(
        validate=True, permit_nulls=False, **coord_kwargs
    )

    orbit_id = result.get(f"{p}orbit_id", "0")
    object_id = result.get(f"{p}object_id")
    object_id_list = [object_id if object_id else None]

    orbits_kwargs: dict[str, KwargValue] = {
        "orbit_id": [orbit_id],
        "object_id": object_id_list,
        "coordinates": cart_coords,
    }

    # Fitted **absolute** non-grav, so the returned orbit is re-feedable
    # into propagate / evaluate / refine / compute_b_planes without silently
    # dropping the force model. The Rust wrapper folds A1/A2/A3 + g(r) + dt
    # onto the result orbit; mirror that here.
    a1 = result.get(f"{p}a1")
    if a1 is not None:
        from empyrean.orbits.nongrav import NonGravParams

        alpha = result.get(f"{p}ng_alpha", 0.0)
        r0 = result.get(f"{p}ng_r0", 0.0)
        m = result.get(f"{p}ng_m", 0.0)
        n_exp = result.get(f"{p}ng_n", 0.0)
        k = result.get(f"{p}ng_k", 0.0)
        # All-zero g(r) exponents are the inverse-square asteroid default;
        # any non-zero value is an explicit Marsden–Sekanina g(r).
        is_inverse_square = alpha == 0.0 and r0 == 0.0 and m == 0.0 and n_exp == 0.0 and k == 0.0
        model = "inverse_square" if is_inverse_square else "marsden_sekanina"
        dt = result.get(f"{p}non_grav_dt")
        # Fitted non-grav 3×3 covariance (row-major flat, 9), present only for
        # StateAndNonGrav fits. Carried so the orbit re-feeds into a
        # StateAndNonGrav refine without losing its prior.
        ng_cov = result.get(f"{p}non_grav_cov")
        orbits_kwargs["non_grav"] = NonGravParams.from_kwargs(
            a1=[a1],
            a2=[result[f"{p}a2"]],
            a3=[result[f"{p}a3"]],
            model=[model],
            alpha=[alpha],
            r0=[r0],
            m=[m],
            n=[n_exp],
            k=[k],
            dt=[dt],
            covariance=[list(ng_cov) if ng_cov is not None else None],
        )

    return CartesianOrbits.from_kwargs(validate=True, permit_nulls=False, **orbits_kwargs)


def _radar_from_result(result: ResultDict) -> ADESRadarObservations:
    """Build an :class:`ADESRadarObservations` table from the nested
    ``"radar"`` dict surfaced by the Rust ``_read_ades`` entry point.

    The inactive value column is NaN per the Rust NaN-inactive
    convention; :func:`_nullable_float` maps those NaNs back to nulls so
    the quivr table reflects the ``observable`` discriminator rather than
    a magic sentinel.
    """
    radar = result.get("radar")
    if radar is None or len(radar["obs_time"]) == 0:
        return ADESRadarObservations.empty()

    def _str_list(key: str) -> list[str | None]:
        return [s if s else None for s in radar[key]]

    # com: i8 tri-state (-1 absent, 0 false, 1 true) -> Optional[bool].
    com_arr = np.asarray(radar["com"], dtype=np.int8)
    com_list = [None if v < 0 else bool(v) for v in com_arr]

    return ADESRadarObservations.from_kwargs(
        # ── Identification ──
        perm_id=_str_list("perm_id"),
        prov_id=_str_list("prov_id"),
        trk_sub=_str_list("trk_sub"),
        # ── Bistatic geometry ──
        trx=radar["trx"],
        rcv=radar["rcv"],
        # ── Core measurement ──
        obs_time=radar["obs_time"],
        observable=radar["observable"],
        delay=_nullable_float(np.asarray(radar["delay"])),
        rms_delay=_nullable_float(np.asarray(radar["rms_delay"])),
        doppler=_nullable_float(np.asarray(radar["doppler"])),
        rms_doppler=_nullable_float(np.asarray(radar["rms_doppler"])),
        # ── Reduction metadata ──
        frq=np.asarray(radar["frq"]),
        com=com_list,
        log_snr=_nullable_float(np.asarray(radar["log_snr"])),
        remarks=_str_list("remarks"),
    )


def read_ades(
    path_or_string: str | os.PathLike[str],
) -> tuple[ADESObservations, ADESRadarObservations]:
    """Read ADES PSV observations into optical + radar tables.

    Parameters
    ----------
    path_or_string : str | os.PathLike
        Either a filesystem path to an ADES PSV / MPC80 file, or the
        PSV / MPC80 content directly as a string. A `pathlib.Path`
        instance is always treated as a path; a plain `str` is treated
        as a path when it exists on disk and as content otherwise.

    Returns
    -------
    tuple[ADESObservations, ADESRadarObservations]
        ``(optical, radar)``. The radar table is empty when the file
        carries no ``<radar>`` block. ADES models radar as its own
        top-level table, so both are returned together; unpack the
        tuple — e.g. ``optical, radar = read_ades(path)``.
    """
    import os
    from pathlib import Path

    from empyrean._empyrean_rs import _read_ades

    # Path-vs-content detection: a `Path` is unambiguous; a `str` is a
    # path iff the file exists. Multi-line strings (the common shape
    # for inline PSV content) won't accidentally hit the filesystem
    # because no filename can contain a newline on POSIX or Windows.
    content: str
    if isinstance(path_or_string, Path) or (
        isinstance(path_or_string, str)
        and "\n" not in path_or_string
        and os.path.isfile(path_or_string)
    ):
        with open(path_or_string, encoding="utf-8") as f:
            content = f.read()
    else:
        content = str(path_or_string)

    result = _read_ades(content)

    radar = _radar_from_result(result)

    n = len(result["obs_time"])
    if n == 0:
        return ADESObservations.empty(), radar

    def _str_list(key: str) -> list[str | None]:
        return [s if s else None for s in result[key]]

    n_stars_arr = np.asarray(result["n_stars"], dtype=np.int32)
    n_stars_list = [int(v) if v >= 0 else None for v in n_stars_arr]

    optical = ADESObservations.from_kwargs(
        # ── Identification ──
        perm_id=_str_list("perm_id"),
        prov_id=_str_list("prov_id"),
        trk_sub=_str_list("trk_sub"),
        obs_id=_str_list("obs_id"),
        obs_sub_id=_str_list("obs_sub_id"),
        trk_id=_str_list("trk_id"),
        # ── Observer ──
        stn=result["stn"],
        mode=_str_list("mode"),
        prog=_str_list("prog"),
        # ── Observer location ──
        sys=_str_list("sys"),
        ctr=_nullable_float(np.asarray(result["ctr"])),
        pos1=_nullable_float(np.asarray(result["pos1"])),
        pos2=_nullable_float(np.asarray(result["pos2"])),
        pos3=_nullable_float(np.asarray(result["pos3"])),
        # ── Astrometry ──
        obs_time=result["obs_time"],
        ra=np.asarray(result["ra"]),
        dec=np.asarray(result["dec"]),
        # ── Uncertainties ──
        rms_ra=_nullable_float(np.asarray(result["rms_ra"])),
        rms_dec=_nullable_float(np.asarray(result["rms_dec"])),
        rms_corr=_nullable_float(np.asarray(result["rms_corr"])),
        # ── Catalog ──
        ast_cat=_str_list("ast_cat"),
        # ── Photometry ──
        mag=_nullable_float(np.asarray(result["mag"])),
        rms_mag=_nullable_float(np.asarray(result["rms_mag"])),
        band=_str_list("band"),
        phot_cat=_str_list("phot_cat"),
        phot_ap=_nullable_float(np.asarray(result["phot_ap"])),
        # ── Supplementary diagnostics ──
        log_snr=_nullable_float(np.asarray(result["log_snr"])),
        seeing=_nullable_float(np.asarray(result["seeing"])),
        exp=_nullable_float(np.asarray(result["exp"])),
        rms_fit=_nullable_float(np.asarray(result["rms_fit"])),
        n_stars=n_stars_list,
        notes=_str_list("notes"),
        remarks=_str_list("remarks"),
    )

    return optical, radar


def determine(
    observations: ADESObservations,
    initial_orbits: dict[str, CartesianOrbits] | None = None,
    *,
    radar: ADESRadarObservations | None = None,
    config: ODConfig | None = None,
) -> DetermineResult:
    """Run the full orbit determination pipeline (IOD + DC).

    Parameters
    ----------
    observations : ADESObservations
        ADES optical observations for a single object.
    initial_orbits : dict[str, CartesianOrbits], optional
        Map of object ID to initial seed orbit. When provided, IOD is
        skipped and the seed becomes the DC starting point.
    radar : ADESRadarObservations, optional
        ADES radar (delay / Doppler) observations for the same object,
        as returned alongside the optical table by :func:`read_ades`.
        When omitted or empty, the fit is optical-only — every existing
        caller keeps working unchanged.
    config : ODConfig, optional
        Pipeline configuration. Default: ``ODConfig()``.

    Returns
    -------
    DetermineResult
        Single fitted orbit + residuals + summary. The C ABI's
        determine surface returns one fit per call (single-target);
        for multi-object batches loop over per-object ADESObservations
        slices in Python.
    """
    from empyrean._empyrean_rs import _determine

    if config is None:
        config = ODConfig()

    obs_dict = _obs_to_dict(observations)
    radar_dict = _radar_to_dict(radar) if radar is not None and len(radar) > 0 else None

    initial_orbits_dict = None
    if initial_orbits is not None:
        initial_orbits_dict = {}
        for oid, orbit in initial_orbits.items():
            initial_orbits_dict[oid] = _orbits_to_dict(orbit)

    result = _determine(obs_dict, config._to_wire_dict(), initial_orbits_dict, radar_dict)
    return _build_determine_result(result)


def evaluate(
    orbit: AnyOrbits,
    observations: ADESObservations,
    *,
    config: ODConfig | None = None,
) -> EvaluateResult:
    """Evaluate residuals for an orbit against observations.

    Propagates the orbit to each observation epoch and computes
    residuals. No fitting is performed.

    Parameters
    ----------
    orbit : CartesianOrbits
        Single orbit (must contain exactly one entry). Parameter name
        is singular to match the Rust wrapper's `Context::evaluate`;
        the type is a quivr table because Python orbits are always
        table-shaped, but only the first row is used.
    observations : ADESObservations
        ADES optical observations.
    config : ODConfig, optional
        Configuration. Defaults to :class:`ODConfig` defaults
        (Standard force model, etc.).

    Returns
    -------
    EvaluateResult
    """
    from empyrean._empyrean_rs import _evaluate_single

    if config is None:
        config = ODConfig()

    orbit_dict = _orbits_to_dict(orbit)
    obs_dict = _obs_to_dict(observations)
    result = _evaluate_single(orbit_dict, obs_dict, config._to_wire_dict())

    obs_results = _build_observation_results(result)
    summary = _build_residual_summary(result, prefix="summary_")

    return EvaluateResult(observations=obs_results, summary=summary)


def refine(
    orbit: AnyOrbits,
    observations: ADESObservations,
    *,
    config: ODConfig | None = None,
) -> DetermineResult:
    """Refine an orbit with new observations using a Bayesian prior.

    The orbit must carry a covariance matrix; the seed orbit's
    covariance is always used as the prior constraint by the underlying
    ``empyrean_refine`` C ABI entry point. (Unlike ``determine``, there
    is no opt-out — calling ``refine`` IS the prior-based path.)

    Parameters
    ----------
    orbit : CartesianOrbits
        Single orbit with covariance (must contain exactly one entry).
        Parameter name is singular to match the Rust wrapper's
        `Context::refine`; the type is a quivr table because Python
        orbits are always table-shaped, but only the first row is used.
    observations : ADESObservations
        ADES optical observations.
    config : ODConfig, optional
        Configuration. Defaults to :class:`ODConfig` defaults.

    Returns
    -------
    DetermineResult
        Refined orbit, residuals, and summary statistics.
    """
    from empyrean._empyrean_rs import _refine_single

    if config is None:
        config = ODConfig()

    orbit_dict = _orbits_to_dict(orbit)
    obs_dict = _obs_to_dict(observations)
    result = _refine_single(orbit_dict, obs_dict, config._to_wire_dict())

    return _build_determine_result(result)


def _build_solved_covariance(d: ResultDict | None) -> SolvedCovariance | None:
    """Build a :class:`SolvedCovariance` from the nested ``solved_covariance``
    dict, or ``None`` when the fit solved only the 6-element state.

    The slot fields are read with ``.get`` so an unsolved axis reads as
    ``None`` (never 0) — a slot of 0 would be a bug since the state
    always occupies columns 0..5.
    """
    if d is None:
        return None
    width = int(d["width"])
    matrix = np.asarray(d["matrix"], dtype=np.float64).reshape(width, width)

    def _slot(key: str) -> int | None:
        v = d.get(key)
        return int(v) if v is not None else None

    thrust_slots = [tuple(int(i) for i in s) for s in d.get("thrust_slots", [])]
    return SolvedCovariance(
        matrix=matrix,
        width=width,
        marsden_slot=_slot("marsden_slot"),
        dt_slot=_slot("dt_slot"),
        amrat_slot=_slot("amrat_slot"),
        thrust_slots=thrust_slots,
    )


def _build_photometry_result(d: ResultDict | None) -> PhotometryResult | None:
    """Build a :class:`PhotometryResult` from the nested ``photometry``
    dict, or ``None`` when photometry did not run.

    Per-band / gate arrays arrive as parallel struct-of-arrays and are
    zipped back into :class:`BandStat` / :class:`GateRecord` rows.
    """
    if d is None:
        return None

    cov_flat = d.get("covariance")
    covariance = (
        np.asarray(cov_flat, dtype=np.float64).reshape(3, 3) if cov_flat is not None else None
    )

    per_band = [
        BandStat(
            band=band,
            n=int(n),
            offset_applied=float(offset),
            mean_residual=float(mean),
            rms=float(rms),
        )
        for band, n, offset, mean, rms in zip(
            d.get("band", []),
            d.get("band_n", []),
            np.asarray(d.get("band_offset_applied", []), dtype=np.float64),
            np.asarray(d.get("band_mean_residual", []), dtype=np.float64),
            np.asarray(d.get("band_rms", []), dtype=np.float64),
            strict=True,
        )
    ]
    gates = [
        GateRecord(model=PhotometryModel(model), passed=bool(passed), reason=reason)
        for model, passed, reason in zip(
            d.get("gate_model", []),
            d.get("gate_passed", []),
            d.get("gate_reason", []),
            strict=True,
        )
    ]

    return PhotometryResult(
        h=float(d["h"]),
        slope1=float(d["slope1"]),
        slope2=float(d["slope2"]),
        covariance=covariance,
        model_used=PhotometryModel(d["model_used"]),
        reduced_chi2=float(d["reduced_chi2"]),
        constraint_active=bool(d["constraint_active"]),
        n_mags_used=int(d["n_mags_used"]),
        n_mags_rejected_photometric=int(d["n_mags_rejected_photometric"]),
        n_obs_without_mags=int(d["n_obs_without_mags"]),
        n_mags_from_astrometric_selected=int(d["n_mags_from_astrometric_selected"]),
        n_mags_from_astrometric_rejected=int(d["n_mags_from_astrometric_rejected"]),
        alpha_min_deg=float(d["alpha_min_deg"]),
        alpha_max_deg=float(d["alpha_max_deg"]),
        alpha_span_deg=float(d["alpha_span_deg"]),
        per_band=per_band,
        gates=gates,
    )


def _build_determine_result(result: ResultDict) -> DetermineResult:
    """Assemble a :class:`DetermineResult` from a Rust _determine /
    _refine result dict."""
    from empyrean.od.result import (
        CovarianceRepresentation,
    )
    from empyrean.od.result import (
        ForceModelTier as _FMT,
    )
    from empyrean.od.result import (
        SolveForParams as _SFP,
    )

    orbit = _build_cartesian_orbits_single(result, prefix="orbit_")
    obs_results = _build_observation_results(result)
    summary = _build_residual_summary(result, prefix="summary_")
    acceptability = _build_acceptability_report(result, prefix="acceptability_")
    cov = np.asarray(result["covariance"], dtype=np.float64).reshape(6, 6)
    cov_rep = CovarianceRepresentation(result["covariance_representation"])
    cov_9x9 = result.get("covariance_9x9")
    if cov_9x9 is not None:
        cov_9x9 = np.asarray(cov_9x9, dtype=np.float64).reshape(9, 9)
    ng_delta = result.get("non_grav_delta")
    if ng_delta is not None:
        ng_delta = np.asarray(ng_delta, dtype=np.float64)

    # Per-station fitted biases (empty quivr table when fit_station_biases=False).
    sb_codes = list(result.get("station_bias_obs_codes", []))
    if sb_codes:
        station_biases = StationBiases.from_kwargs(
            obs_code=sb_codes,
            n_obs=np.asarray(result["station_bias_n_obs"], dtype=np.uint64),
            bias_ra_arcsec=np.asarray(result["station_bias_ra_arcsec"]),
            sigma_ra_arcsec=np.asarray(result["station_bias_sigma_ra_arcsec"]),
            bias_dec_arcsec=np.asarray(result["station_bias_dec_arcsec"]),
            sigma_dec_arcsec=np.asarray(result["station_bias_sigma_dec_arcsec"]),
            bias_timing_sec=list(result["station_bias_timing_sec"]),
            sigma_timing_sec=list(result["station_bias_sigma_timing_sec"]),
            significance=np.asarray(result["station_bias_significance"]),
        )
    else:
        station_biases = StationBiases.empty()

    # ── Wide fitting surface (v0.9.0) ───────────────────────────────
    # Every key is read with `.get` so a dropped / unsolved axis reads
    # as None (never 0 / NaN) — the loud-None contract.
    solved_covariance = _build_solved_covariance(result.get("solved_covariance"))
    dt_delta = result.get("dt_delta")
    dt_delta = float(dt_delta) if dt_delta is not None else None
    amrat_delta = result.get("amrat_delta")
    amrat_delta = float(amrat_delta) if amrat_delta is not None else None
    thrust_delta = result.get("thrust_delta_m_per_s")
    thrust_delta_arr = (
        np.asarray(thrust_delta, dtype=np.float64).reshape(-1, 3)
        if thrust_delta is not None
        else None
    )
    dv_frame = result.get("dv_frame")
    photometry = _build_photometry_result(result.get("photometry"))

    return DetermineResult(
        orbit=orbit,
        observations=obs_results,
        summary=summary,
        iterations=int(result["iterations"]),
        update_norm=float(result["update_norm"]),
        converged=bool(result["converged"]),
        covariance=cov,
        covariance_representation=cov_rep,
        covariance_9x9=cov_9x9,
        non_grav_delta=ng_delta,
        rejection_passes=int(result["rejection_passes"]),
        num_oppositions_fit=int(result["num_oppositions_fit"]),
        force_model_used=_FMT(result["force_model_used"]),
        solve_for_used=_SFP(result["solve_for_used"]),
        acceptability=acceptability,
        station_biases=station_biases,
        solved_covariance=solved_covariance,
        dt_delta=dt_delta,
        amrat_delta=amrat_delta,
        thrust_delta_m_per_s=thrust_delta_arr,
        dv_frame=dv_frame,
        photometry=photometry,
    )
