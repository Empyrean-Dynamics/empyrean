"""Coordinate transformations between representations and frames."""

from empyrean._convert import (
    _CLASS_TO_REP,
    AnyCoordinates,
    arrays_to_coordinates,
    coordinates_to_arrays,
    frame_to_int,
    origin_to_naif,
    rep_to_int,
)
from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from empyrean.coordinates.enums import Frame, Origin


def transform_coordinates(
    coordinates: AnyCoordinates,
    target: type[CartesianCoordinates]
    | type[KeplerianCoordinates]
    | type[CometaryCoordinates]
    | type[SphericalCoordinates]
    | str,
    frame: Frame | str | None = None,
    origin: Origin | str | None = None,
) -> AnyCoordinates:
    """Transform coordinates to a different representation, frame, or origin.

    Parameters
    ----------
    coordinates : Cartesian / Keplerian / Cometary / SphericalCoordinates
        Input coordinates (quivr table). Angles must be in degrees.
    target : type | str
        Target coordinate class (e.g. ``CartesianCoordinates``) or a
        representation name (``"cartesian"`` / ``"keplerian"`` /
        ``"cometary"`` / ``"spherical"``).
    frame : Frame | str, optional
        Target frame. ``None`` keeps the current frame.
    origin : Origin | str, optional
        Target origin. ``None`` keeps the current origin.

    Returns
    -------
    CartesianCoordinates | KeplerianCoordinates | CometaryCoordinates | SphericalCoordinates
        Transformed coordinates with covariance propagated (if present).

    Examples
    --------
    >>> cart = transform_coordinates(cometary_coords, CartesianCoordinates)
    >>> kep = transform_coordinates(cart, KeplerianCoordinates, frame=Frame.ICRF)
    """
    from empyrean._empyrean_rs import _transform_coordinates

    # Resolve target representation from class or string
    if isinstance(target, type) and target in _CLASS_TO_REP:
        representation = _CLASS_TO_REP[target]
    elif isinstance(target, str):
        representation = target.lower()
    else:
        raise TypeError(
            f"target must be a coordinate class (e.g. CartesianCoordinates) "
            f"or a string, got {target!r}"
        )

    (
        epochs,
        elements,
        covariances,
        has_covariance,
        representations,
        frames,
        origins,
    ) = coordinates_to_arrays(coordinates)

    target_rep = rep_to_int(representation)

    # Default: keep current frame/origin
    if frame is not None:
        target_frame = frame_to_int(frame)
    else:
        target_frame = int(frames[0]) if len(frames) > 0 else 0

    if origin is not None:
        target_origin = origin_to_naif(origin)
    else:
        target_origin = int(origins[0]) if len(origins) > 0 else 10

    result = _transform_coordinates(
        epochs,
        elements,
        covariances,
        has_covariance,
        representations,
        frames,
        origins,
        target_rep,
        target_frame,
        target_origin,
    )

    return arrays_to_coordinates(result, representation)
