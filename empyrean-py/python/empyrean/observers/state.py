"""Compute observer states from MPC observatory codes."""

from collections.abc import Sequence

from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs, _require_epochs
from empyrean.observers.observers import Observers


def get_observer_states(
    obs_codes: Sequence[str],
    epochs: Epochs,
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
    epochs : Epochs
        Observation epochs as an
        :class:`~empyrean.coordinates.epoch.Epochs` table, converted to
        TDB internally. Build one with
        ``Epochs.from_mjd(values, scale="tdb")`` (or ``scale="utc"``); a
        bare array is refused, as it carries no time scale.
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

    Raises
    ------
    TypeError
        ``epochs`` is not an :class:`Epochs` table.
    """
    # Validated here rather than only in `from_codes` so the refusal names
    # the entry point the caller actually used.
    _require_epochs(epochs, "get_observer_states() epochs")
    return Observers.from_codes(obs_codes, epochs, frame=frame, origin=origin)
