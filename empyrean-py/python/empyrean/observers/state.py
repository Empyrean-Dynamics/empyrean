"""Compute observer states from MPC observatory codes."""

from collections.abc import Sequence

import numpy as np

from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs
from empyrean.observers.observers import Observers


def get_observer_states(
    obs_codes: Sequence[str],
    epochs: Epochs | np.ndarray | Sequence[float],
    frame: Frame | str | int = Frame.ICRF,
    origin: Origin | str | int = Origin.SSB,
) -> Observers:
    """Compute observer Cartesian states in a chosen frame and origin.

    Cross product: ``N`` obs codes × ``M`` epochs = ``N*M`` observers.

    Thin wrapper around :meth:`Observers.from_codes` — prefer the
    classmethod when writing new code.

    Parameters
    ----------
    obs_codes : list[str]
        MPC observatory codes (e.g. ``["W84", "F51"]``).
    epochs : Epochs | array-like
        Observation epochs. :class:`~empyrean.coordinates.epoch.Epochs`
        table or MJD TDB array.
    frame : Frame | str | int
        Reference frame for the returned states. Default
        :attr:`Frame.ICRF`.
    origin : Origin | str | int
        Body the states are relative to. Default :attr:`Origin.SSB`.

    Notes
    -----
    ``(Frame.ICRF, Origin.SSB)`` is the construction basis, required by
    ephemeris generation and orbit determination and returned untouched.
    See :meth:`Observers.from_codes` for when another basis is the right
    request.

    Returns
    -------
    Observers
        Observer states with ``obs_code``, ``epoch``, position,
        velocity, and ``observing_night``.
    """
    return Observers.from_codes(obs_codes, epochs, frame=frame, origin=origin)
