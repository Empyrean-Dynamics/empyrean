"""Compute observer states from MPC observatory codes."""

from collections.abc import Sequence

import numpy as np

from empyrean.coordinates.epoch import Epochs
from empyrean.observers.observers import Observers


def get_observer_states(
    obs_codes: Sequence[str],
    epochs: Epochs | np.ndarray | Sequence[float],
) -> Observers:
    """Compute observer Cartesian states in ICRF relative to SSB.

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

    Returns
    -------
    Observers
        Observer states with ``obs_code``, ``epoch``, position,
        velocity, and ``observing_night``.
    """
    return Observers.from_codes(obs_codes, epochs)
