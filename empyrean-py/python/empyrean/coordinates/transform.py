"""Coordinate transformations between representations and frames."""

from empyrean._convert import (
    _CLASS_TO_REP,
    AnyCoordinates,
    AnyOrbits,
    arrays_to_coordinates,
    coordinates_to_arrays,
    frame_to_int,
    orbit_batch_dict_to_orbits,
    orbits_to_orbit_batch_dict,
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
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)

# Orbit class → representation name, for resolving a ``transform_orbits``
# target given as an :class:`Orbits` class rather than a string.
_ORBIT_CLASS_TO_REP: dict[type, str] = {
    CartesianOrbits: "cartesian",
    KeplerianOrbits: "keplerian",
    CometaryOrbits: "cometary",
    SphericalOrbits: "spherical",
}


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
    from empyrean.errors import translate_push_errors

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

    # A covariance-bearing state is refused here (the −50 push code) — move a
    # covariance between bases with :func:`transform_orbits` instead, which
    # transports the full packed joint. The refusal surfaces through the one
    # translation point as a :class:`~empyrean.PushError` whose variant is
    # :attr:`~empyrean.PushErrorVariant.TRANSFORM_COVARIANCE_UNSUPPORTED`, the
    # same type every packed-joint refusal takes, and whose message names
    # ``transform_orbits``.
    with translate_push_errors():
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


def transform_orbits(
    orbits: AnyOrbits,
    target: type[CartesianOrbits]
    | type[KeplerianOrbits]
    | type[CometaryOrbits]
    | type[SphericalOrbits]
    | str,
    frame: Frame | str | None = None,
    origin: Origin | str | None = None,
) -> AnyOrbits:
    """Transform a batch of orbits — coordinate state plus its packed joint
    covariance — to a different representation, frame, or origin.

    The orbit's joint is transported through the \\(6 \\times 6\\) Jacobian of
    the transformation; this is the entry point for moving a covariance
    between bases. The state-level :func:`transform_coordinates` transforms
    bare states and refuses a covariance-bearing input, raising a
    :class:`~empyrean.PushError` whose variant is
    :attr:`~empyrean.PushErrorVariant.TRANSFORM_COVARIANCE_UNSUPPORTED`.

    Parameters
    ----------
    orbits : Cartesian / Keplerian / Cometary / SphericalOrbits
        Input orbits (quivr table). Angles must be in degrees.
    target : type | str
        Target orbit class (e.g. ``CartesianOrbits``) or a representation
        name (``"cartesian"`` / ``"keplerian"`` / ``"cometary"`` /
        ``"spherical"``).
    frame : Frame | str, optional
        Target frame. ``None`` keeps the current frame.
    origin : Origin | str, optional
        Target origin. ``None`` keeps the current origin.

    Returns
    -------
    CartesianOrbits | KeplerianOrbits | CometaryOrbits | SphericalOrbits
        Transformed orbits with the joint covariance transported.
    """
    from empyrean._empyrean_rs import _transform_orbits
    from empyrean.errors import translate_push_errors

    if isinstance(target, type) and target in _ORBIT_CLASS_TO_REP:
        representation = _ORBIT_CLASS_TO_REP[target]
    elif isinstance(target, type) and target in _CLASS_TO_REP:
        representation = _CLASS_TO_REP[target]
    elif isinstance(target, str):
        representation = target.lower()
    else:
        raise TypeError(
            f"target must be an orbit class (e.g. CartesianOrbits) or a string, got {target!r}"
        )

    orbit_dict = orbits_to_orbit_batch_dict(orbits)
    target_rep = rep_to_int(representation)

    # Default: keep the current frame / origin (homogeneous across the batch).
    if frame is not None:
        target_frame = frame_to_int(frame)
    else:
        cur = orbit_dict["frame"]
        target_frame = frame_to_int(cur[0]) if len(cur) > 0 else 0

    if origin is not None:
        target_origin = origin_to_naif(origin)
    else:
        cur_origin = orbit_dict["origin"]
        target_origin = int(cur_origin[0]) if len(cur_origin) > 0 else 10

    with translate_push_errors():
        result = _transform_orbits(orbit_dict, target_rep, target_frame, target_origin)

    return orbit_batch_dict_to_orbits(result)
