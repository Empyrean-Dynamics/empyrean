"""Query body states from SPK ephemeris."""

from collections.abc import Sequence

import numpy as np

from empyrean._convert import frame_to_int, int_to_frame, naif_to_origin, origin_to_naif
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs


def get_states(
    target: Origin | str,
    center: Origin | str,
    epochs: Epochs | np.ndarray | Sequence[float],
    frame: Frame | str = Frame.ECLIPTICJ2000,
) -> CartesianCoordinates:
    """Query Cartesian states of a body from the SPK ephemeris.

    Parameters
    ----------
    target : Origin | str
        Target body. Pass an :class:`Origin` (e.g. ``Origin.EARTH``) or
        the canonical name (e.g. ``"Earth"``).
    center : Origin | str
        Center body. Same shape as ``target``.
    epochs : Epochs | array-like
        Epochs as an Epochs table or MJD TDB array.
    frame : Frame | str
        Reference frame (default: EclipticJ2000).

    Returns
    -------
    CartesianCoordinates
        Cartesian states (AU, AU/day) at each epoch.
    """
    from empyrean._empyrean_rs import _get_states

    target_naif = origin_to_naif(target)
    center_naif = origin_to_naif(center)
    frame_int = frame_to_int(frame)

    if isinstance(epochs, Epochs):
        tdb = epochs.to_tdb()
        epochs_mjd = np.asarray(tdb.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)
    else:
        epochs_mjd = np.asarray(epochs, dtype=np.float64)

    result = _get_states(target_naif, center_naif, epochs_mjd, frame_int)

    out_frame = int_to_frame(int(result["frame"][0])) if len(result["frame"]) > 0 else frame
    origin_strs = [naif_to_origin(int(o)) for o in result["origin"]]

    return CartesianCoordinates.from_kwargs(
        epoch=np.asarray(result["epoch"]),
        x=np.asarray(result["x"]),
        y=np.asarray(result["y"]),
        z=np.asarray(result["z"]),
        vx=np.asarray(result["vx"]),
        vy=np.asarray(result["vy"]),
        vz=np.asarray(result["vz"]),
        frame=out_frame.value if hasattr(out_frame, "value") else out_frame,
        origin=origin_strs,
    )
