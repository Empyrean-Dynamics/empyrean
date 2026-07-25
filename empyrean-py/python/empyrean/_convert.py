"""Internal conversion utilities between Rust dicts and quivr types."""

from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import quivr as qv

from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from empyrean.coordinates.covariance import (
    CartesianCovariance,
    CometaryCovariance,
    KeplerianCovariance,
    SphericalCovariance,
    _CovarianceTable,
)
from empyrean.coordinates.enums import Frame, Origin
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)
from empyrean.orbits.srp import SRPParams

# Float64 numpy array alias used throughout this module.
FloatArray = npt.NDArray[np.float64]

# Type alias for any of the four coordinate flavors.
AnyCoordinates = (
    CartesianCoordinates | KeplerianCoordinates | CometaryCoordinates | SphericalCoordinates
)

# Type alias for any of the four orbit flavors.
AnyOrbits = CartesianOrbits | KeplerianOrbits | CometaryOrbits | SphericalOrbits


def _covariance_from_matrix(cov_cls: type[_CovarianceTable], matrix: FloatArray) -> qv.Table:
    """Build a covariance table from ``(N, 6, 6)`` matrices.

    The covariance classes are synthesised at runtime via ``type(...)`` in
    :mod:`empyrean.coordinates.covariance`; the :class:`_CovarianceTable`
    Protocol describes their injected ``from_matrix`` classmethod so it is
    visible to static analysis. The concrete result is a ``qv.Table``
    subclass instance at runtime, so we narrow the structural Protocol type
    back to ``qv.Table`` for callers (mirroring the ``to_matrix`` cast
    below). Funnel every covariance construction through this one typed
    helper so call sites stay strict.
    """
    return cast("qv.Table", cov_cls.from_matrix(matrix))


class _SupportsToMatrix(Protocol):
    """Structural surface of a covariance table's ``to_matrix`` method.

    The covariance classes are synthesised at runtime via ``type(...)`` in
    :mod:`empyrean.coordinates.covariance`, so their ``to_matrix`` method is
    invisible to static analysis (the ``covariance`` column statically resolves
    to bare ``qv.Table``). This protocol captures exactly the member the
    extraction helper touches.
    """

    def to_matrix(self) -> FloatArray: ...


def _covariance_to_matrix(covariance: qv.Table) -> FloatArray:
    """Extract ``(N, 6, 6)`` matrices from a covariance table.

    Inverse of :func:`_covariance_from_matrix`. Funnel every matrix extraction
    through this one typed helper so call sites stay strict.
    """
    matrix: FloatArray = cast("_SupportsToMatrix", covariance).to_matrix()
    return matrix


def _covariance_row_reference(index: int, orbit_ids: list[str] | None) -> str:
    """Name one covariance row in an error message.

    Includes the ``orbit_id`` when the caller has one (the orbit entry point),
    so a user with a large batch can find the offending record without
    re-deriving the row order.
    """
    if orbit_ids is not None and index < len(orbit_ids) and orbit_ids[index]:
        return f"row {index} (orbit_id={orbit_ids[index]!r})"
    return f"row {index}"


def _covariance_state_labels(covariance: qv.Table) -> list[str]:
    """The six state-variable labels of a covariance table, for error text."""
    return list(cast("_CovarianceTable", covariance)._state_labels)


def _covariance_matrices_and_presence(
    covariance: qv.Table,
    *,
    entry_point: str,
    n_rows: int,
    orbit_ids: list[str] | None = None,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Marshal a covariance sub-table to ``(n, 6, 6)`` matrices + presence mask.

    Single chokepoint for every covariance the wrapper hands to the engine.
    A covariance that cannot be marshaled is **raised**, never replaced: this
    helper has no fallback path, no zero-fill for malformed input, and no
    ``has_covariance = False`` consolation prize. A caller who asked for a
    covariance-bearing run and got a silently zeroed covariance would get
    confidently wrong uncertainties out the far end.

    Absence versus corruption
    -------------------------
    ``from_matrix`` writes NaN *values* for an absent covariance, and an unset
    nullable column reads back as NaN too, so "null" and "NaN" are the same
    signal here and a row can be fully non-null yet fully NaN. A row whose 36
    entries are all null or NaN is genuinely absent: it is zero-filled and
    reported as ``has_covariance = False``. Anything else must be complete
    enough to use.

    Partial rows
    ------------
    A row is rejected when any of its **six diagonal** entries is null or NaN
    while the row is not fully absent. Null **off-diagonal** entries are
    accepted and filled with zero, which is an explicit statement that those
    state variables are uncorrelated — the standard reading of an omitted
    off-diagonal, and the natural result of building a covariance from only
    the six ``cov_<a>_<a>`` columns. A zero *diagonal* entry is different in
    kind: it asserts zero variance on that state variable, a delta function
    the caller never supplied.

    Parameters
    ----------
    covariance : qv.Table
        The covariance sub-table to marshal.
    entry_point : str
        Name of the calling entry point, quoted verbatim in every error. Both
        entry points see the same coordinate classes, so the class name alone
        would not tell a user which call failed.
    n_rows : int
        Expected row count — the length of the parent coordinate table.
    orbit_ids : list of str, optional
        Per-row orbit identifiers, quoted in the error when available so the
        offending record is findable in a large batch.

    Returns
    -------
    (matrices, has_covariance)
        ``(n, 6, 6)`` float64 matrices with absent rows and null off-diagonals
        zero-filled, and the per-row presence mask.

    Raises
    ------
    ValueError
        If the sub-table cannot be converted, is not shaped ``(n, 6, 6)``,
        carries a non-finite entry, or carries a null/NaN diagonal entry on a
        row that is not fully absent.
    """
    try:
        matrices = _covariance_to_matrix(covariance)
    except Exception as exc:
        raise ValueError(
            f"{entry_point}: could not convert the covariance sub-table to "
            f"(n, 6, 6) matrices ({type(exc).__name__}: {exc})"
        ) from exc

    expected_shape = (n_rows, 6, 6)
    if matrices.shape != expected_shape:
        raise ValueError(
            f"{entry_point}: covariance must be shaped {expected_shape} "
            f"(one 6x6 per coordinate row), got {matrices.shape}"
        )

    matrices = np.asarray(matrices, dtype=np.float64)
    if n_rows == 0:
        return matrices, np.zeros(0, dtype=bool)

    # NaN marks absence; +/-inf marks corruption. Separate the two before
    # anything is zero-filled — `np.nan_to_num(x, nan=0.0)` leaves
    # `posinf` alone at its default, which would turn +inf into 1.797e308
    # and hand the engine a finite-looking variance.
    is_nan = np.isnan(matrices)
    is_inf = np.isinf(matrices)
    if is_inf.any():
        bad_rows = np.flatnonzero(is_inf.any(axis=(1, 2)))
        first = int(bad_rows[0])
        raise ValueError(
            f"{entry_point}: covariance {_covariance_row_reference(first, orbit_ids)} "
            f"contains a non-finite (+/-inf) entry; "
            f"{len(bad_rows)} of {n_rows} row(s) affected. A covariance must be "
            "finite — fix the input rather than letting an infinite variance "
            "reach the engine."
        )

    absent = is_nan.all(axis=(1, 2))
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    partial = np.isnan(diagonal).any(axis=1) & ~absent
    if partial.any():
        bad_rows = np.flatnonzero(partial)
        first = int(bad_rows[0])
        labels = _covariance_state_labels(covariance)
        missing = [labels[k] for k in np.flatnonzero(np.isnan(diagonal[first]))]
        raise ValueError(
            f"{entry_point}: covariance {_covariance_row_reference(first, orbit_ids)} "
            f"has a null or NaN variance on {missing} but is not fully absent; "
            f"{len(bad_rows)} of {n_rows} row(s) affected. Supply every "
            "cov_<a>_<a> diagonal entry, or leave the whole row null/NaN to "
            "declare the row has no covariance. A missing diagonal entry "
            "cannot be filled with zero: that would assert zero variance on "
            "that state variable. (Null off-diagonal entries are accepted and "
            "read as uncorrelated.) If this covariance came from "
            "transform_coordinates, the element is undefined at that geometry "
            "rather than missing from your input — an angular element loses "
            "its definition as the orbit degenerates (argument of perihelion "
            "for a circular orbit, node and inclination for an equatorial "
            "one), so its variance is genuinely NaN and the transform, not "
            "this call, is where the degeneracy arose."
        )

    # Every surviving NaN is either a fully-absent row or an off-diagonal
    # zero-correlation fill; no infinities remain past the check above.
    return np.nan_to_num(matrices, nan=0.0), ~absent


# ── Representation / frame / origin mappings ─────────────────

_REP_TO_INT = {
    "cartesian": 0,
    "keplerian": 1,
    "cometary": 2,
    "spherical": 3,
}

_INT_TO_REP = {v: k for k, v in _REP_TO_INT.items()}

_FRAME_TO_INT = {
    Frame.ICRF: 0,
    Frame.ECLIPTICJ2000: 1,
    Frame.ITRF93: 2,
    # Canonical lowercase strings (the Frame enum's `value`s).
    "icrf": 0,
    "eclipticj2000": 1,
    "itrf93": 2,
    # Common variants users actually type.
    "ecliptic_j2000": 1,
    "ecliptic j2000": 1,
    "ecliptic": 1,
    "itrf_93": 2,
}


def _normalize_frame_key(frame: Frame | str | int) -> Frame | str | int:
    """Canonicalise a frame argument into the keys used by ``_FRAME_TO_INT``.

    Accepts a :class:`Frame` enum, an empyrean wrapper Frame, or a
    string. Strings are lower-cased and stripped before lookup.
    """
    if isinstance(frame, Frame):
        return frame
    if isinstance(frame, str):
        return frame.strip().lower()
    return frame


_INT_TO_FRAME = {
    0: Frame.ICRF,
    1: Frame.ECLIPTICJ2000,
    2: Frame.ITRF93,
}

# Canonical-name → NAIF ID. Keyed by the string value an Origin
# serializes to; `origin_to_naif` accepts the typed Origin too.
_NAME_TO_NAIF = {
    "SSB": 0,
    "Sun": 10,
    "Mercury": 199,
    "Venus": 299,
    "Earth": 399,
    "Moon": 301,
    "Mars Barycenter": 4,
    "Jupiter Barycenter": 5,
    "Saturn Barycenter": 6,
    "Uranus Barycenter": 7,
    "Neptune Barycenter": 8,
    "Pluto Barycenter": 9,
}

_NAIF_TO_NAME = {v: k for k, v in _NAME_TO_NAIF.items()}


def origin_to_naif(origin: str | Origin | int) -> int:
    """Convert an [`Origin`] / canonical name / NAIF int to a NAIF ID.

    The user-facing API takes :class:`empyrean.Origin`. This helper
    accepts the legacy string + bare int forms too, so I/O paths that
    re-hydrate from a Parquet column (string) or interop with code
    that already speaks NAIF can route through one entry point.
    """
    if isinstance(origin, (int, np.integer)):
        return int(origin)
    if isinstance(origin, Origin):
        name = origin.name
    else:
        name = str(origin)
    if name in _NAME_TO_NAIF:
        return _NAME_TO_NAIF[name]
    if name.startswith("asteroid_"):
        return 2_000_000 + int(name[9:])
    raise ValueError(f"unknown origin: {origin!r}")


def naif_to_origin(naif_id: int) -> str:
    """Convert a NAIF ID to its canonical origin name (string).

    Returns the same string an :class:`Origin` serializes to via
    ``str(origin)`` — so values written into a quivr ``origin`` column
    always round-trip cleanly.
    """
    if naif_id in _NAIF_TO_NAME:
        return _NAIF_TO_NAME[naif_id]
    if naif_id >= 2_000_000:
        return f"asteroid_{naif_id - 2_000_000}"
    return str(naif_id)


def frame_to_int(frame: Frame | str | int) -> int:
    """Convert a frame enum / canonical name / int to its integer code.

    Accepts the lowercase canonical name (``"icrf"``), legacy
    underscore-separated forms (``"ecliptic_j2000"``), and bare ints
    (passed through).
    """
    if isinstance(frame, (int, np.integer)):
        return int(frame)
    key = _normalize_frame_key(frame)
    if key in _FRAME_TO_INT:
        return _FRAME_TO_INT[key]
    raise ValueError(f"unknown frame: {frame!r}")


def int_to_frame(val: int) -> Frame:
    """Convert an integer to a Frame enum."""
    return _INT_TO_FRAME[val]


def rep_to_int(rep: str | int) -> int:
    """Convert a representation string to an integer."""
    if isinstance(rep, (int, np.integer)):
        return int(rep)
    return _REP_TO_INT[rep.lower()]


# ── Coordinate type detection ────────────────────────────────

# Map coordinate class → representation string (for transform_coordinates target)
_CLASS_TO_REP = {
    CartesianCoordinates: "cartesian",
    KeplerianCoordinates: "keplerian",
    CometaryCoordinates: "cometary",
    SphericalCoordinates: "spherical",
}

# Map coordinate type → (representation string, element column names)
_COORD_TYPE_MAP: dict[type[AnyCoordinates], tuple[str, list[str]]] = {
    CartesianCoordinates: ("cartesian", ["x", "y", "z", "vx", "vy", "vz"]),
    KeplerianCoordinates: ("keplerian", ["a", "e", "i", "raan", "ap", "ma"]),
    CometaryCoordinates: ("cometary", ["q", "e", "i", "raan", "ap", "tp"]),
    SphericalCoordinates: ("spherical", ["rho", "lon", "lat", "vrho", "vlon", "vlat"]),
}

_REP_TO_COORD_TYPE: dict[str, type[AnyCoordinates]] = {
    "cartesian": CartesianCoordinates,
    "keplerian": KeplerianCoordinates,
    "cometary": CometaryCoordinates,
    "spherical": SphericalCoordinates,
}

# Values are the dynamically-generated covariance classes. They are
# ``qv.Table`` subclasses at runtime; their ``from_matrix`` classmethod is
# reached via :func:`_covariance_from_matrix`.
_REP_TO_COV_TYPE: dict[str, type[_CovarianceTable]] = {
    "cartesian": CartesianCovariance,
    "keplerian": KeplerianCovariance,
    "cometary": CometaryCovariance,
    "spherical": SphericalCovariance,
}

_REP_TO_ELEMENT_NAMES = {
    "cartesian": ["x", "y", "z", "vx", "vy", "vz"],
    "keplerian": ["a", "e", "i", "raan", "ap", "ma"],
    "cometary": ["q", "e", "i", "raan", "ap", "tp"],
    "spherical": ["rho", "lon", "lat", "vrho", "vlon", "vlat"],
}


def _column_to_numpy(col: pa.Array) -> FloatArray:
    """Extract a quivr column's pyarrow array to a float64 numpy array.

    On a quivr ``Table`` instance, accessing a column attribute returns
    the underlying :class:`pyarrow.Array` (not the descriptor), so the
    argument is a pyarrow array.
    """
    arr: FloatArray = col.to_numpy(zero_copy_only=False).astype(np.float64)
    return arr


# ── Coordinates → arrays ─────────────────────────────────────


def coordinates_to_arrays(
    coords: AnyCoordinates,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    npt.NDArray[np.bool_],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Convert a quivr coordinate table to numpy arrays for Rust.

    Returns (epochs, elements, covariances, has_covariance,
             representations, frames, origins).
    """
    coord_type = type(coords)
    if coord_type not in _COORD_TYPE_MAP:
        raise TypeError(f"unsupported coordinate type: {coord_type}")

    rep_str, col_names = _COORD_TYPE_MAP[coord_type]
    rep_int = rep_to_int(rep_str)
    n = len(coords)

    epochs = _column_to_numpy(coords.epoch)
    elements = np.column_stack([_column_to_numpy(getattr(coords, c)) for c in col_names])

    frame_int = frame_to_int(coords.frame)
    frames = np.full(n, frame_int, dtype=np.int32)

    origins = np.array(
        [origin_to_naif(o) for o in coords.origin.to_pylist()],
        dtype=np.int32,
    )

    representations = np.full(n, rep_int, dtype=np.int32)

    # Covariance. Every class in `_COORD_TYPE_MAP` declares a
    # `covariance = <Rep>Covariance.as_column(nullable=True)` column, and quivr
    # materialises an all-null sub-table rather than `None` when the caller
    # never supplied one — so there is no "no covariance column" branch to
    # take. If a fifth representation is ever added, give it a covariance
    # column too, or reinstate a guard here that raises rather than
    # substituting zeros.
    cov_matrices, has_cov = _covariance_matrices_and_presence(
        coords.covariance,
        entry_point="coordinates_to_arrays",
        n_rows=n,
    )

    return (
        epochs,
        elements,
        cov_matrices,
        has_cov,
        representations,
        frames,
        origins,
    )


# ── Arrays → coordinates ─────────────────────────────────────


def arrays_to_coordinates(
    result_dict: dict[str, npt.NDArray[Any]], representation: str
) -> AnyCoordinates:
    """Convert a Rust result dict to a quivr coordinate table.

    Parameters
    ----------
    result_dict : dict
        Dict from _transform_coordinates with epochs, elements,
        covariances, has_covariance, frames, origins arrays.
    representation : str
        Target representation name.
    """
    rep = representation.lower()
    coord_cls = _REP_TO_COORD_TYPE[rep]
    cov_cls = _REP_TO_COV_TYPE[rep]
    col_names = _REP_TO_ELEMENT_NAMES[rep]

    epochs = np.asarray(result_dict["epochs"])
    elements = np.asarray(result_dict["elements"])
    covariances = np.asarray(result_dict["covariances"], dtype=np.float64)
    has_cov = np.asarray(result_dict["has_covariance"])
    frames = np.asarray(result_dict["frames"])
    origins = np.asarray(result_dict["origins"])
    n = len(epochs)

    # Build the frame attribute (all rows should have the same frame)
    frame = int_to_frame(int(frames[0])) if n > 0 else Frame.ICRF

    # Build origin column
    origin_strs = [naif_to_origin(int(o)) for o in origins]

    # Build covariance
    cov: qv.Table | None
    if has_cov.any():
        cov = _covariance_from_matrix(cov_cls, covariances)
    else:
        cov = None

    kwargs: dict[str, npt.NDArray[Any] | list[str] | str | qv.Table] = {
        "epoch": epochs,
        "frame": frame.value,
        "origin": origin_strs,
    }
    for j, name in enumerate(col_names):
        kwargs[name] = elements[:, j]
    if cov is not None:
        kwargs["covariance"] = cov

    # Bind ``validate`` / ``permit_nulls`` explicitly so the ``**kwargs``
    # splat maps only onto quivr's trailing column ``**kwargs`` parameter
    # (whose keys are dynamic element/column names), not its leading
    # ``bool`` parameters.
    return coord_cls.from_kwargs(validate=True, permit_nulls=False, **kwargs)


# ── OrbitBatch dict ↔ quivr Orbits ───────────────────────────


def orbit_batch_dict_to_orbits(result: dict[str, Any]) -> AnyOrbits:
    """Convert an OrbitBatch dict from the Rust extension into a
    quivr Orbits table (Cartesian / Keplerian / Cometary / Spherical
    depending on the dict's ``representation`` column).

    Used by:

    - :func:`empyrean.query_sbdb` — SBDB returns cometary by default.
    - :func:`empyrean.io.read_orbits_*` — representation matches the
      file's stored schema.

    Non-grav columns (``a1`` / ``a2`` / ``a3`` / ``g_*``) are attached
    to the ``non_grav`` field when at least one row has a non-zero
    coefficient. Photometric columns are not exposed on this batch dict.
    Continuous-thrust is not an Orbits-table column at all — thrust arcs
    are variable-length per orbit, so they are supplied as structured
    input at propagation time via the ``thrust_arcs`` keyword of
    :func:`empyrean.propagate` (see
    :class:`~empyrean.ThrustParams`), not carried on the orbit rows.
    """
    from empyrean.orbits.nongrav import NonGravParams
    from empyrean.orbits.orbits import (
        CartesianOrbits,
        CometaryOrbits,
        KeplerianOrbits,
        SphericalOrbits,
    )

    representations = result["representation"]
    n = len(result["orbit_ids"])
    if n == 0:
        # Default to Cartesian for empty batches; caller can dispatch
        # if they have a stronger expectation.
        return CartesianOrbits.from_kwargs(
            orbit_id=[],
            object_id=[],
            coordinates=_REP_TO_COORD_TYPE["cartesian"].from_kwargs(
                epoch=np.zeros(0),
                x=np.zeros(0),
                y=np.zeros(0),
                z=np.zeros(0),
                vx=np.zeros(0),
                vy=np.zeros(0),
                vz=np.zeros(0),
                frame=Frame.ICRF.value,
                origin=[],
            ),
        )

    rep = representations[0].lower()
    if not all(r.lower() == rep for r in representations):
        raise ValueError(
            "mixed representations in orbit batch — Orbits requires homogeneous schema"
        )

    coord_cls = _REP_TO_COORD_TYPE[rep]
    cov_cls = _REP_TO_COV_TYPE[rep]
    col_names = _REP_TO_ELEMENT_NAMES[rep]
    orbits_by_rep: dict[str, type[AnyOrbits]] = {
        "cartesian": CartesianOrbits,
        "keplerian": KeplerianOrbits,
        "cometary": CometaryOrbits,
        "spherical": SphericalOrbits,
    }
    orbits_cls = orbits_by_rep[rep]

    epochs = np.asarray(result["epoch_mjd_tdb"], dtype=np.float64)
    elements = np.asarray(result["elements"], dtype=np.float64)
    has_cov = np.asarray(result["has_covariance"], dtype=bool)
    cov_matrices = np.asarray(result["covariance"], dtype=np.float64)
    frames = result["frame"]  # list of strings ("icrf" / "ecliptic_j2000")
    origins = np.asarray(result["origin"], dtype=np.int32)

    # Frame is shared across the whole table (quivr column attribute,
    # not per-row). Validate homogeneity and pick the first.
    if not all(f.lower() == frames[0].lower() for f in frames):
        raise ValueError("mixed frames in orbit batch")
    frame_str = frames[0].lower()
    frame = {
        "icrf": Frame.ICRF,
        "ecliptic_j2000": Frame.ECLIPTICJ2000,
        "eclipticj2000": Frame.ECLIPTICJ2000,
    }[frame_str]
    origin_strs = [naif_to_origin(int(o)) for o in origins]

    coords_kwargs: dict[str, npt.NDArray[Any] | list[str] | str | qv.Table] = {
        "epoch": epochs,
        "frame": frame.value,
        "origin": origin_strs,
    }
    for j, name in enumerate(col_names):
        coords_kwargs[name] = elements[:, j]
    if has_cov.any():
        coords_kwargs["covariance"] = _covariance_from_matrix(cov_cls, cov_matrices)
    # Bind validate / permit_nulls explicitly so the splat maps only onto
    # quivr's trailing column ``**kwargs``, not its leading bool params.
    coordinates = coord_cls.from_kwargs(validate=True, permit_nulls=False, **coords_kwargs)

    a1 = np.asarray(result.get("a1", np.zeros(n)), dtype=np.float64)
    a2 = np.asarray(result.get("a2", np.zeros(n)), dtype=np.float64)
    a3 = np.asarray(result.get("a3", np.zeros(n)), dtype=np.float64)

    orbits_kwargs: dict[str, list[str] | list[str | None] | qv.Table] = {
        "orbit_id": list(result["orbit_ids"]),
        "object_id": [s if s else None for s in result["object_ids"]],
        "coordinates": coordinates,
    }
    if (a1 != 0).any() or (a2 != 0).any() or (a3 != 0).any():
        # Carry the full Marsden–Sekanina g(r) the C ABI emitted — the
        # exponents (alpha/r0/m/n/k) AND the thermal-lag dt — not just
        # A1/A2/A3. Dropping the exponents silently re-propagates a comet
        # as inverse-square with no lag (the 67P divergence). All-zero
        # exponents are the established "inverse-square" sentinel; NaN dt
        # means "no thermal lag".
        ng_alpha = np.asarray(result.get("ng_alpha", np.zeros(n)), dtype=np.float64)
        ng_r0 = np.asarray(result.get("ng_r0", np.zeros(n)), dtype=np.float64)
        ng_m = np.asarray(result.get("ng_m", np.zeros(n)), dtype=np.float64)
        ng_n = np.asarray(result.get("ng_n", np.zeros(n)), dtype=np.float64)
        ng_k = np.asarray(result.get("ng_k", np.zeros(n)), dtype=np.float64)
        ng_dt = np.asarray(result.get("non_grav_dt", np.full(n, np.nan)), dtype=np.float64)
        ng_dt_variance = np.asarray(
            result.get("non_grav_dt_variance", np.full(n, np.nan)), dtype=np.float64
        )
        models: list[str] = []
        for i in range(n):
            if ng_alpha[i] != 0.0:
                # Custom SBDB g(r) exponents present → "marsden" (the
                # documented NonGravParams vocabulary for custom g(r)).
                models.append("marsden")
            elif a1[i] != 0.0 or a2[i] != 0.0 or a3[i] != 0.0:
                # No custom g(r) but a real coefficient: the inverse-square
                # default. Includes pure-transverse Yarkovsky asteroids
                # (A2 only, A1 == 0) — labelling those "" would emit an
                # invalid model the parquet/round-trip consumers choke on.
                models.append("inverse_square")
            else:
                # All coefficients zero for this row (a gravity-only orbit
                # riding in a mixed batch). No non-grav model applies.
                models.append("")
        orbits_kwargs["non_grav"] = NonGravParams.from_kwargs(
            a1=a1,
            a2=a2,
            a3=a3,
            model=models,
            alpha=ng_alpha,
            r0=ng_r0,
            m=ng_m,
            n=ng_n,
            k=ng_k,
            dt=ng_dt,
            dt_variance=ng_dt_variance,
        )

    # SRP force slot — reconstruct ``orbits.srp`` when any row carries an SRP
    # force (has_srp == 1). Additive with, and independent of, the Marsden
    # non-grav above (a State+AMRAT fit has no Marsden coefficients), so this
    # is deliberately outside the non-grav block. Absent rows get a NaN
    # amrat/cr sentinel that :func:`extract_srp` reads back as "no slot".
    has_srp = np.asarray(result.get("has_srp", np.zeros(n, dtype=np.uint8)), dtype=np.uint8)
    if has_srp.any():
        srp_amrat = np.asarray(result.get("srp_amrat", np.full(n, np.nan)), dtype=np.float64)
        srp_cr = np.asarray(result.get("srp_cr", np.full(n, np.nan)), dtype=np.float64)
        srp_var = np.asarray(result.get("srp_amrat_variance", np.full(n, np.nan)), dtype=np.float64)
        present = has_srp != 0
        orbits_kwargs["srp"] = SRPParams.from_kwargs(
            amrat=np.where(present, srp_amrat, np.nan),
            cr=np.where(present, srp_cr, np.nan),
            amrat_variance=np.where(present, srp_var, np.nan),
        )

    # Photometry — populated when the upstream source (e.g. SBDB's
    # phys_par) carried H + slope params. Empty list / all-NaN H means
    # no row had usable photometry, in which case we leave the
    # ``photometric`` column unset on the orbit so downstream consumers
    # see ``orbit.photometric is None``.
    phot_system = result.get("phot_system", [None] * n)
    if any(pf is not None for pf in phot_system):
        from empyrean.orbits.photometry import PhotometricParams

        phot_h = np.asarray(result.get("phot_h", np.full(n, np.nan)), dtype=np.float64)
        phot_slope1 = np.asarray(result.get("phot_slope1", np.zeros(n)), dtype=np.float64)
        phot_slope2 = np.asarray(result.get("phot_slope2", np.zeros(n)), dtype=np.float64)
        # Map (model, slope1, slope2) → (g, g1, g2, g12) per the upstream
        # PhotometricParams slot-mapping convention.
        h_list: list[float | None] = []
        g_list: list[float | None] = []
        g1_list: list[float | None] = []
        g2_list: list[float | None] = []
        g12_list: list[float | None] = []
        model_list: list[str | None] = []
        for i in range(n):
            pf = phot_system[i]
            h = float(phot_h[i])
            s1 = float(phot_slope1[i])
            s2 = float(phot_slope2[i])
            if pf is None or not np.isfinite(h):
                h_list.append(None)
                g_list.append(None)
                g1_list.append(None)
                g2_list.append(None)
                g12_list.append(None)
                model_list.append(None)
                continue
            h_list.append(h)
            model_list.append(pf)
            if pf == "HG":
                g_list.append(s1)
                g1_list.append(None)
                g2_list.append(None)
                g12_list.append(None)
            elif pf == "HG1G2":
                g_list.append(None)
                g1_list.append(s1)
                g2_list.append(s2)
                g12_list.append(None)
            elif pf == "HG12":
                g_list.append(None)
                g1_list.append(None)
                g2_list.append(None)
                g12_list.append(s1)
            else:
                g_list.append(None)
                g1_list.append(None)
                g2_list.append(None)
                g12_list.append(None)
        orbits_kwargs["photometric"] = PhotometricParams.from_kwargs(
            model=model_list,
            h=h_list,
            g=g_list,
            g1=g1_list,
            g2=g2_list,
            g12=g12_list,
        )

    # Bind validate / permit_nulls explicitly so the splat maps only onto
    # quivr's trailing column ``**kwargs``, not its leading bool params.
    return orbits_cls.from_kwargs(validate=True, permit_nulls=False, **orbits_kwargs)


def orbits_to_orbit_batch_dict(orbits: AnyOrbits) -> dict[str, Any]:
    """Convert a quivr Orbits table (Cartesian / Keplerian / Cometary
    / Spherical) into the OrbitBatch dict shape consumed by the Rust
    extension's I/O writers.

    Inverse of :func:`orbit_batch_dict_to_orbits`.
    """
    coord_type = type(orbits.coordinates)
    if coord_type not in _COORD_TYPE_MAP:
        raise TypeError(f"unsupported coordinate type: {coord_type}")
    rep, col_names = _COORD_TYPE_MAP[coord_type]

    # NonGravParams is Marsden-only; reject a stray model='srp' / cr before
    # marshaling (SRP rides its own slot below).
    validate_non_grav_marsden_only(orbits)

    n = len(orbits)
    epochs = _column_to_numpy(orbits.coordinates.epoch)
    elements = np.column_stack(
        [_column_to_numpy(getattr(orbits.coordinates, c)) for c in col_names]
    )

    # Same contract as `coordinates_to_arrays`: every class in
    # `_COORD_TYPE_MAP` declares a nullable `covariance` sub-table column and
    # quivr materialises it as all-null rather than `None`, so there is no
    # "no covariance column" branch. A fifth representation must declare one
    # too, or gain a guard here that raises rather than substituting zeros.
    orbit_ids = orbits.orbit_id.to_pylist()
    cov_matrices, cov_present = _covariance_matrices_and_presence(
        orbits.coordinates.covariance,
        entry_point="orbits_to_orbit_batch_dict",
        n_rows=n,
        orbit_ids=orbit_ids,
    )
    has_cov = cov_present.astype(np.uint8)

    # ``orbits.coordinates.frame`` is the stored string (``Frame`` is a
    # ``str`` enum, so its value serialises directly). Key by those value
    # strings to map to the OrbitBatch frame label.
    frame_label_map: dict[str, str] = {
        Frame.ICRF.value: "icrf",
        Frame.ECLIPTICJ2000.value: "ecliptic_j2000",
    }
    frame_value = orbits.coordinates.frame
    frame_str = frame_label_map.get(frame_value, str(frame_value))
    origin_naif = np.array(
        [origin_to_naif(o) for o in orbits.coordinates.origin.to_pylist()],
        dtype=np.int32,
    )

    a1 = np.zeros(n)
    a2 = np.zeros(n)
    a3 = np.zeros(n)
    ng_alpha = np.zeros(n)
    ng_r0 = np.zeros(n)
    ng_m = np.zeros(n)
    ng_n = np.zeros(n)
    ng_k = np.zeros(n)
    ng_dt = np.full(n, np.nan)
    ng_dt_variance = np.full(n, np.nan)
    if orbits.non_grav is not None:
        ng = orbits.non_grav
        a1 = np.nan_to_num(_column_to_numpy(ng.a1), nan=0.0)
        a2 = np.nan_to_num(_column_to_numpy(ng.a2), nan=0.0)
        a3 = np.nan_to_num(_column_to_numpy(ng.a3), nan=0.0)
        # g(r) Marsden–Sekanina exponents — carry the comet g(r) so a fitted
        # or SBDB comet orbit re-fed into evaluate/refine/determine isn't
        # silently treated as inverse-square. All-zero = inverse-square.
        ng_alpha = np.nan_to_num(_column_to_numpy(ng.alpha), nan=0.0)
        ng_r0 = np.nan_to_num(_column_to_numpy(ng.r0), nan=0.0)
        ng_m = np.nan_to_num(_column_to_numpy(ng.m), nan=0.0)
        ng_n = np.nan_to_num(_column_to_numpy(ng.n), nan=0.0)
        ng_k = np.nan_to_num(_column_to_numpy(ng.k), nan=0.0)
        # Preserve NaN as "no thermal-lag delay" — 0.0 is a real delay,
        # so do NOT nan_to_num this one.
        ng_dt = _column_to_numpy(ng.dt)
        # DT prior variance (opens the DT column in a StateAndNonGravAndDT
        # solve). Preserve NaN as "no prior" — do NOT nan_to_num.
        ng_dt_variance = _column_to_numpy(ng.dt_variance)

    # SRP force slot (has_srp explicit switch + amrat/cr/variance). Additive
    # with the Marsden non-grav above; validated + gated in extract_srp.
    has_srp, srp_amrat, srp_cr, srp_amrat_variance = extract_srp(orbits)

    object_ids = [s if s else "" for s in orbits.object_id.to_pylist()]
    return {
        "orbit_ids": orbit_ids,
        "object_ids": object_ids,
        "epoch_mjd_tdb": epochs,
        "elements": elements,
        "covariance": cov_matrices,
        "has_covariance": has_cov,
        "representation": [rep] * n,
        "frame": [frame_str] * n,
        "origin": origin_naif,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "ng_alpha": ng_alpha,
        "ng_r0": ng_r0,
        "ng_m": ng_m,
        "ng_n": ng_n,
        "ng_k": ng_k,
        "non_grav_dt": ng_dt,
        "non_grav_dt_variance": ng_dt_variance,
        "has_srp": has_srp,
        "srp_amrat": srp_amrat,
        "srp_cr": srp_cr,
        "srp_amrat_variance": srp_amrat_variance,
    }


def extract_photometry(
    orbits: AnyOrbits,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int32]]:
    """Extract photometric parameters from orbits.

    Returns (h, g, model_ints) arrays. model_int: 0=HG, 1=HG1G2, 2=HG12, -1=none.
    """
    n = len(orbits)
    if orbits.photometric is not None:
        p = orbits.photometric
        h = np.asarray(p.h.to_numpy(zero_copy_only=False), dtype=np.float64)
        g = np.asarray(p.g.to_numpy(zero_copy_only=False), dtype=np.float64)
        h = np.nan_to_num(h, nan=0.0)
        g = np.nan_to_num(g, nan=0.0)
        # PhotometricParams.model carries lowercase tags ("hg", "hg1g2",
        # "hg12") per python/empyrean/orbits/photometry.py:18. Match case-
        # insensitively so callers using either convention get photometry
        # threaded through; an unknown tag falls through to -1, which the
        # binding interprets as "no photometry — emit NaN mag".
        models = p.model.to_pylist()
        model_map = {"hg": 0, "hg1g2": 1, "hg12": 2}
        model_ints = np.array(
            [model_map.get(m.lower(), -1) if m else -1 for m in models],
            dtype=np.int32,
        )
    else:
        h = np.full(n, np.nan, dtype=np.float64)
        g = np.full(n, np.nan, dtype=np.float64)
        model_ints = np.full(n, -1, dtype=np.int32)
    return h, g, model_ints


def extract_non_grav_covariance(
    orbits: AnyOrbits,
) -> tuple[npt.NDArray[np.bool_], FloatArray]:
    """Extract the fitted non-grav 3x3 covariance from orbits.

    Returns ``(has_non_grav_cov, non_grav_cov)`` — a ``(n,)`` bool mask and a
    ``(n, 3, 3)`` float64 array (row-major, zeros where absent). Mirrors the OD
    output path's marshal (``empyrean.od.determine._orbits_to_dict``) so a
    ``StateAndNonGrav``-fitted orbit re-fed into the forward model
    (propagate / generate_ephemeris / impact) keeps its non-grav prior instead
    of silently dropping it. All-False + zeros when the orbit has no
    ``non_grav`` sub-table or no row carries a fitted covariance.
    """
    n = len(orbits)
    has_non_grav_cov = np.zeros(n, dtype=bool)
    non_grav_cov = np.zeros((n, 3, 3), dtype=np.float64)
    if orbits.non_grav is not None:
        for i, c in enumerate(orbits.non_grav.covariance.to_pylist()):
            if c is not None:
                non_grav_cov[i] = np.asarray(c, dtype=np.float64).reshape(3, 3)
                has_non_grav_cov[i] = True
    return has_non_grav_cov, non_grav_cov


# ── SRP force slot (shared marshal helpers) ──────────────────────────
#
# SRP is a first-class, additive force slot carried on ``orbits.srp``
# (:class:`~empyrean.orbits.srp.SRPParams`) — NOT a NonGravParams model.
# These two helpers are shared by every orbit-marshaling entry point
# (propagate / generate_ephemeris / impact / od.determine / io) so none of
# them silently drops the caller's SRP slot or silently applies the old,
# physically-wrong ``model="srp"`` Marsden path.

# The Marsden g(r) models NonGravParams may carry. SRP is deliberately NOT in
# this set — it moved to its own force slot. The reverse-marshal paths emit
# "marsden", "marsden_sekanina", "inverse_square", and "" (a gravity-only row
# in a mixed batch), so all four round-trip cleanly.
_VALID_NON_GRAV_MODELS = frozenset(
    {"marsden_water", "inverse_square", "marsden", "marsden_sekanina", ""}
)


def validate_non_grav_marsden_only(orbits: AnyOrbits) -> None:
    """Reject a ``NonGravParams`` that is not a pure Marsden non-grav.

    Shared by every orbit-marshaling entry point. NonGravParams is
    Marsden-only; solar radiation pressure is a separate additive slot on
    ``orbits.srp`` (:class:`~empyrean.orbits.srp.SRPParams`). This raises
    :class:`ValueError` when:

    - ``model == "srp"`` — the old in-band SRP tag, which the forward model
      silently applied as a *radial Marsden* acceleration (``a1`` treated as
      A1, not AMRAT). Any orbit fit or propagated under it was wrong.
    - a non-null ``cr`` appears on ``NonGravParams`` — Cr belongs on
      ``SRPParams`` now.
    - an unknown ``model`` string.

    A no-op when ``orbits.srp`` alone is set (that is the correct surface).
    """
    if orbits.non_grav is None:
        return
    ng = orbits.non_grav

    # Cr moved to SRPParams. Reject any lingering non-null cr loudly (old data
    # or a caller still setting it on the wrong table).
    if "cr" in ng.table.column_names:
        cr_col = ng.column("cr")
        if cr_col.null_count < len(cr_col):
            raise ValueError(
                "NonGravParams carries a non-null `cr` — the radiation-pressure "
                "coefficient moved to SRPParams. Set `orbits.srp = "
                "SRPParams(amrat=..., cr=...)` instead of a `cr` on NonGravParams."
            )

    models = ng.model.to_pylist()
    for model in models:
        tag = (model or "").lower()
        if tag == "srp":
            raise ValueError(
                "NonGravParams model='srp' is no longer supported: solar "
                "radiation pressure is a separate additive force slot. Use "
                "`orbits.srp = SRPParams(amrat=<m^2/kg>, cr=<coefficient>)` "
                "instead. NOTE: any orbit previously fit or propagated with "
                "model='srp' was computed as a radial Marsden acceleration "
                "(a1 treated as A1, not AMRAT) and is INVALID — re-run it with "
                "the SRPParams slot."
            )
        if tag not in _VALID_NON_GRAV_MODELS:
            raise ValueError(
                f"unknown NonGravParams model={model!r}; expected one of "
                f"{sorted(_VALID_NON_GRAV_MODELS)} (SRP now lives on orbits.srp)"
            )


def extract_srp(
    orbits: AnyOrbits,
) -> tuple[npt.NDArray[np.uint8], FloatArray, FloatArray, FloatArray]:
    """Extract the SRP force slot from ``orbits.srp`` as parallel arrays.

    Returns ``(has_srp, srp_amrat, srp_cr, srp_amrat_variance)`` — a ``(n,)``
    ``uint8`` switch and three ``(n,)`` float64 arrays. ``has_srp[i] == 1``
    only where the ``srp`` sub-table is present for row ``i`` (a finite
    ``amrat``); SRP is never value-inferred. Absent rows read
    ``has_srp = 0`` with ``amrat = cr = 0`` and ``amrat_variance = NaN``.

    Validates loudly (mirrors the C-ABI ``empyrean_orbit_srp_params`` rules):
    a present row must have a finite ``amrat`` and ``cr`` both ``> 0``, and a
    finite ``amrat_variance`` must be ``> 0`` (NaN / null = no prior).
    """
    n = len(orbits)
    has_srp = np.zeros(n, dtype=np.uint8)
    srp_amrat = np.zeros(n, dtype=np.float64)
    srp_cr = np.zeros(n, dtype=np.float64)
    srp_amrat_variance = np.full(n, np.nan, dtype=np.float64)
    if orbits.srp is None:
        return has_srp, srp_amrat, srp_cr, srp_amrat_variance

    srp = orbits.srp
    amrat = np.asarray(srp.amrat.to_numpy(zero_copy_only=False), dtype=np.float64)
    cr = np.asarray(srp.cr.to_numpy(zero_copy_only=False), dtype=np.float64)
    # Nullable column: nulls come back as NaN = "no prior".
    variance = np.asarray(srp.amrat_variance.to_numpy(zero_copy_only=False), dtype=np.float64)

    # A row is present exactly when its (non-nullable) amrat is non-null. quivr
    # returns NaN for a null sub-table row (and an all-null SRPParams even when
    # the caller never set `srp`), so a non-NaN amrat is the "real slot" gate.
    # Gating on non-NaN rather than finiteness means an explicit Inf amrat is
    # still "present" and gets rejected loudly below (never a silent drop).
    present = ~np.isnan(amrat)
    for i in np.nonzero(present)[0]:
        if not (np.isfinite(amrat[i]) and amrat[i] > 0.0):
            raise ValueError(
                f"orbits.srp.amrat[{i}] must be finite and > 0 (area-to-mass "
                f"ratio, m^2/kg); got {amrat[i]}"
            )
        if not (np.isfinite(cr[i]) and cr[i] > 0.0):
            raise ValueError(
                f"orbits.srp.cr[{i}] must be finite and > 0 (radiation-pressure "
                f"coefficient); got {cr[i]}"
            )
        if np.isfinite(variance[i]) and variance[i] <= 0.0:
            raise ValueError(
                f"orbits.srp.amrat_variance[{i}] must be > 0 when set (AMRAT "
                f"prior variance, (m^2/kg)^2); got {variance[i]} — use null for "
                f"no prior"
            )
        has_srp[i] = 1
        srp_amrat[i] = amrat[i]
        srp_cr[i] = cr[i]
        srp_amrat_variance[i] = variance[i]
    return has_srp, srp_amrat, srp_cr, srp_amrat_variance
