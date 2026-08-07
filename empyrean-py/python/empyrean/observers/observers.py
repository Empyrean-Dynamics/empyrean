"""Observer state type."""

from collections.abc import Sequence

import numpy as np
import quivr as qv

from empyrean._convert import frame_to_int, int_to_frame, naif_to_origin, origin_to_naif
from empyrean.coordinates.coordinates import CartesianCoordinates
from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs


class Observers(qv.Table):
    """Precomputed observer states as Cartesian coordinates.

    Each row is one MPC observatory at one epoch. The kinematic state
    lives inside a nested :class:`CartesianCoordinates` table — same
    schema as orbit positions — so frame and origin are explicit and
    consistent with the rest of the API. By default that's
    :attr:`Frame.ICRF` and :attr:`Origin.SSB` — the **construction
    basis**, and the one every downstream consumer (ephemeris
    generation, orbit determination) requires. Pass ``frame`` /
    ``origin`` to :meth:`from_code` / :meth:`from_codes` to get the
    states in another basis instead, for geometry work that wants one
    (e.g. heliocentric ecliptic site positions).

    Construct via :meth:`from_code` (single observatory, ``N`` epochs)
    or :meth:`from_codes` (cartesian product of ``N`` observatories ×
    ``M`` epochs). The top-level shortcut
    :func:`empyrean.get_observer_states` is the same as
    :meth:`from_codes`.
    """

    obs_code = qv.LargeStringColumn()
    coordinates = CartesianCoordinates.as_column()
    observing_night = qv.Int32Column(nullable=True)  # YYYYMMDD or null

    @classmethod
    def from_code(
        cls,
        obs_code: str,
        epochs: Epochs | np.ndarray | Sequence[float],
        frame: Frame | str | int = Frame.ICRF,
        origin: Origin | str | int = Origin.SSB,
    ) -> "Observers":
        """Observer states for a **single** observatory at ``N`` epochs.

        Returns an ``N``-row table, one row per epoch.

        Parameters
        ----------
        obs_code : str
            MPC observatory code (e.g. ``"W84"``, ``"F51"``, ``"274"``).
        epochs : Epochs | array-like
            ``N`` observation epochs. An :class:`Epochs` table is
            converted to TDB internally; an array is treated as MJD TDB.
        frame : Frame | str | int
            Reference frame for the returned states. Default
            :attr:`Frame.ICRF`. See :meth:`from_codes`.
        origin : Origin | str | int
            Body the states are relative to. Default
            :attr:`Origin.SSB`. See :meth:`from_codes`.

        Returns
        -------
        Observers
            Length-``N`` table — row ``i`` is the observer at ``epochs[i]``.

        Examples
        --------
        >>> times = Epochs.from_mjd([60500.0, 60501.0])
        >>> obs = Observers.from_code("W84", times)
        >>> len(obs)
        2
        """
        return cls.from_codes([obs_code], epochs, frame=frame, origin=origin)

    @classmethod
    def from_codes(
        cls,
        obs_codes: Sequence[str],
        epochs: Epochs | np.ndarray | Sequence[float],
        frame: Frame | str | int = Frame.ICRF,
        origin: Origin | str | int = Origin.SSB,
    ) -> "Observers":
        """Observer states for the **cartesian product** of ``N``
        observatory codes × ``M`` epochs.

        Returns an ``N * M``-row table in **code-major** order: all
        ``M`` epochs for ``obs_codes[0]``, then all ``M`` epochs for
        ``obs_codes[1]``, and so on. Indexing into the result is
        therefore ``i * M + j`` for the (code ``i``, epoch ``j``) row.

        Parameters
        ----------
        obs_codes : list[str]
            ``N`` MPC observatory codes.
        epochs : Epochs | array-like
            ``M`` observation epochs. An :class:`Epochs` table is
            converted to TDB internally; an array is treated as MJD TDB.
        frame : Frame | str | int
            Reference frame for the returned states. Default
            :attr:`Frame.ICRF`.
        origin : Origin | str | int
            Body the states are relative to. Default :attr:`Origin.SSB`.

        Notes
        -----
        ``(Frame.ICRF, Origin.SSB)`` is the **construction basis** — the
        one ephemeris generation and orbit determination require, and the
        one to keep when the observers are headed there. Requesting it
        takes no transform at all: the states come back exactly as
        constructed, bit for bit. Any other basis rotates and/or
        translates them, which is what a consumer plotting observer
        geometry wants (e.g. heliocentric ecliptic site positions via
        ``frame=Frame.ECLIPTICJ2000, origin=Origin.SUN``).

        The ``frame`` / ``origin`` stamped on the returned table are read
        off the states the engine returned, not echoed from the request,
        so a table always reports the basis that actually produced it.

        Returns
        -------
        Observers
            ``N * M``-row table, code-major.

        Examples
        --------
        >>> times = Epochs.from_mjd([60000.0, 60001.0])
        >>> obs = Observers.from_codes(["W84", "F51", "274"], times)
        >>> len(obs)
        6
        >>> obs.obs_code.to_pylist()
        ['W84', 'W84', 'F51', 'F51', '274', '274']
        """
        from empyrean._empyrean_rs import _get_observers

        if isinstance(epochs, Epochs):
            tdb = epochs.to_tdb()
            epochs_mjd = np.asarray(tdb.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)
        else:
            epochs_mjd = np.asarray(epochs, dtype=np.float64)

        result = _get_observers(
            list(obs_codes), epochs_mjd, frame_to_int(frame), origin_to_naif(origin)
        )

        nights = result["observing_night"]
        night_list = [int(n) if n >= 0 else None for n in nights]

        # Basis as reported by the returned states, never echoed from the
        # request. `frame` is a table-level attribute in quivr, so a
        # mixed-frame return would be unrepresentable — surface it loudly
        # instead of silently keeping the first row's value.
        frames = np.asarray(result["frame"])
        origins = np.asarray(result["origin"])
        if len(frames) == 0:
            out_frame = int_to_frame(frame_to_int(frame))
        elif np.all(frames == frames[0]):
            out_frame = int_to_frame(int(frames[0]))
        else:
            raise ValueError(
                "observer states came back in more than one frame "
                f"({sorted({int(f) for f in frames})}); Observers carries a single "
                "table-level frame and cannot represent a mixed-frame table"
            )

        coordinates = CartesianCoordinates.from_kwargs(
            epoch=np.asarray(result["epoch"]),
            x=np.asarray(result["x"]),
            y=np.asarray(result["y"]),
            z=np.asarray(result["z"]),
            vx=np.asarray(result["vx"]),
            vy=np.asarray(result["vy"]),
            vz=np.asarray(result["vz"]),
            frame=out_frame.value,
            origin=[naif_to_origin(int(o)) for o in origins],
        )

        return cls.from_kwargs(
            obs_code=result["obs_code"],
            coordinates=coordinates,
            observing_night=night_list,
        )
