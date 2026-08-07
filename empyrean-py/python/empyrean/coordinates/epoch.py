"""Epochs table with time scale awareness and ISO 8601 interop."""

import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import quivr as qv

if TYPE_CHECKING:
    from astropy.time import Time as AstropyTime

    from empyrean._convert import AnyOrbits


class TimeScale(str, enum.Enum):
    """Time scale for epoch values."""

    TDB = "tdb"
    """Barycentric Dynamical Time — the standard for orbital mechanics."""

    UTC = "utc"
    """Coordinated Universal Time — used for observations."""


# JD = MJD + 2400000.5
_JD_MJD_OFFSET = 2400000.5

ScaleArg = str | TimeScale


def _scale_str(scale: ScaleArg) -> str:
    """Normalize a scale argument to a lowercase ``"utc"`` / ``"tdb"`` string.

    Accepts either a :class:`TimeScale` enum value or a string
    (case-insensitive). Raises :class:`ValueError` on anything else.
    """
    if isinstance(scale, TimeScale):
        return scale.value
    if isinstance(scale, str):
        s = scale.lower()
        if s not in ("utc", "tdb"):
            raise ValueError(f"unknown time scale {scale!r}. Supported: 'utc', 'tdb'.")
        return s
    raise TypeError(f"scale must be str or TimeScale, got {type(scale).__name__}")


class Epochs(qv.Table):
    """Epochs as Modified Julian Dates with an explicit time scale.

    The time scale is a table-level attribute (not per-row) because
    mixing scales within a single coordinate set is not meaningful.
    All ``scale=`` arguments throughout this class accept either a
    string (``"utc"`` / ``"tdb"``, case-insensitive) or a
    :class:`TimeScale` enum value.

    Parameters
    ----------
    mjd : array-like
        Modified Julian Date values.
    scale : str or TimeScale
        Time scale: ``"tdb"`` or ``"utc"``.

    Examples
    --------
    >>> epochs = Epochs.from_mjd([60200.0, 60201.0], scale="tdb")
    >>> epochs.scale
    'tdb'
    """

    mjd = qv.Float64Column()
    scale = qv.StringAttribute()

    # ── Scale conversions ────────────────────────────────────

    def to_tdb(self) -> "Epochs":
        """Convert to TDB.

        Returns self unchanged if already TDB. Uses villeneuve's
        leap-second + TDB-TT secular term conversion.
        """
        if self.scale == TimeScale.TDB.value:
            return self

        from empyrean._empyrean_rs import _convert_epochs

        mjd_tdb = _convert_epochs(
            np.asarray(self.mjd.to_numpy(zero_copy_only=False), dtype=np.float64),
            self.scale,
            TimeScale.TDB.value,
        )
        return Epochs.from_kwargs(mjd=np.asarray(mjd_tdb), scale=TimeScale.TDB.value)

    def to_utc(self) -> "Epochs":
        """Convert to UTC.

        Returns self unchanged if already UTC.
        """
        if self.scale == TimeScale.UTC.value:
            return self

        from empyrean._empyrean_rs import _convert_epochs

        mjd_utc = _convert_epochs(
            np.asarray(self.mjd.to_numpy(zero_copy_only=False), dtype=np.float64),
            self.scale,
            TimeScale.UTC.value,
        )
        return Epochs.from_kwargs(mjd=np.asarray(mjd_utc), scale=TimeScale.UTC.value)

    def to_scale(self, scale: ScaleArg) -> "Epochs":
        """Convert to the named scale (``"utc"`` or ``"tdb"``)."""
        target = _scale_str(scale)
        if target == TimeScale.TDB.value:
            return self.to_tdb()
        return self.to_utc()

    # ── ISO 8601 ─────────────────────────────────────────────

    @classmethod
    def from_iso(
        cls,
        iso_strings: Sequence[str],
        scale: ScaleArg = TimeScale.UTC,
    ) -> "Epochs":
        """Create Epochs from ISO 8601 UTC strings.

        Parameters
        ----------
        iso_strings : list[str]
            ISO 8601 UTC timestamps, e.g.
            ``["2029-04-13T21:46:00.000Z"]``. The trailing ``Z`` is
            required.
        scale : str or TimeScale, default ``"utc"``
            Output scale. ``"utc"`` returns MJD UTC; ``"tdb"`` runs the
            UTC→TDB leap-second + Fairhead/Bretagnon conversion before
            returning MJD TDB.

        Returns
        -------
        Epochs
            Length-``N`` table.
        """
        from empyrean._empyrean_rs import _iso_to_mjd

        target = _scale_str(scale)
        if isinstance(iso_strings, str):
            iso_strings = [iso_strings]
        mjd = _iso_to_mjd(list(iso_strings), target)
        return cls.from_kwargs(mjd=np.asarray(mjd), scale=target)

    def to_iso(self, scale: ScaleArg | None = None) -> list[str]:
        """Format epochs as ISO 8601 **UTC wall-clock** strings.

        The output is always the UTC wall-clock time of the stored
        instant, interpreting the stored MJD in the table's own
        :attr:`scale`. A TDB table therefore comes back as UTC ISO with
        the TDB→UTC offset applied (≈69 s at 2026 epochs) — **not** the
        raw TDB clock reading relabelled ``Z``.

        Parameters
        ----------
        scale : str or TimeScale, optional
            Guard only. If given it must equal the table's stored
            :attr:`scale`; ``to_iso`` does not reinterpret the stored
            instant in a different scale. A mismatched ``scale`` raises
            rather than silently relabelling the clock reading. To format
            the instant *as if* it lived in another scale, convert first —
            ``epochs.to_scale(x).to_iso()`` (or
            ``epochs.to_utc().to_iso()`` / ``epochs.to_tdb().to_iso()``),
            which apply the real leap-second + TDB−TT conversion.

        Returns
        -------
        list[str]
            One ISO string per row, always with the trailing ``Z``.

        Raises
        ------
        ValueError
            If ``scale`` is given and differs from the table's stored
            :attr:`scale`.
        """
        from empyrean._empyrean_rs import _mjd_to_iso

        # Option A (honest surface): the stored MJD is always interpreted
        # in the table's own scale, so `to_iso` emits the UTC wall-clock
        # of the actual instant. A different `scale` used to be forwarded
        # as a *reinterpretation* of the stored MJD — a silent relabel
        # worth ~69 s at 2026 epochs — so it is now rejected loudly (no
        # hidden fallback). `scale == self.scale` (or None) is the correct
        # path and is preserved.
        if scale is not None and _scale_str(scale) != self.scale:
            requested = _scale_str(scale)
            raise ValueError(
                f"to_iso() always emits the UTC wall-clock time of the stored "
                f"instant, interpreting the stored MJD in the table's own scale "
                f"({self.scale!r}); it will not reinterpret it as {requested!r} "
                f"(that would relabel the clock reading, a silent ~69 s error at "
                f"2026 epochs). Convert first, then format: "
                f".to_scale({requested!r}).to_iso() — or .to_utc().to_iso() / "
                f".to_tdb().to_iso()."
            )
        iso_strings: list[str] = _mjd_to_iso(
            np.asarray(self.mjd.to_numpy(zero_copy_only=False), dtype=np.float64),
            self.scale,
        )
        return iso_strings

    # ── Astropy interop (optional) ───────────────────────────

    @classmethod
    def from_astropy(cls, time: "AstropyTime") -> "Epochs":
        """Create Epochs from an ``astropy.time.Time`` object.

        Parameters
        ----------
        time : astropy.time.Time
            The astropy scale must be ``"tdb"`` or ``"utc"``.

        Returns
        -------
        Epochs

        Raises
        ------
        ImportError
            If astropy is not installed.
        TypeError
            If the input is not an astropy Time object.
        ValueError
            If the time scale is not ``"tdb"`` or ``"utc"``.
        """
        try:
            from astropy.time import Time
        except ImportError as e:
            raise ImportError(
                "astropy is required for Epochs.from_astropy(). Install with: pip install astropy"
            ) from e

        if not isinstance(time, Time):
            raise TypeError(f"expected astropy.time.Time, got {type(time)}")

        scale = time.scale
        if scale not in ("tdb", "utc"):
            raise ValueError(f"unsupported time scale {scale!r}. Supported: 'tdb', 'utc'.")

        mjd = time.mjd
        if np.ndim(mjd) == 0:
            mjd = np.array([float(mjd)])
        else:
            mjd = np.asarray(mjd, dtype=np.float64)

        return cls.from_kwargs(mjd=mjd, scale=scale)

    def to_astropy(self) -> "AstropyTime":
        """Convert to an ``astropy.time.Time`` object.

        Returns
        -------
        astropy.time.Time

        Raises
        ------
        ImportError
            If astropy is not installed.
        """
        try:
            from astropy.time import Time
        except ImportError as e:
            raise ImportError(
                "astropy is required for Epochs.to_astropy(). Install with: pip install astropy"
            ) from e

        mjd = np.asarray(self.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)
        return Time(mjd, format="mjd", scale=self.scale)

    @classmethod
    def from_orbits(
        cls,
        orbits: "AnyOrbits",
        dt: np.ndarray | Sequence[float],
    ) -> "Epochs":
        """Create epochs offset from the orbits' common epoch.

        All orbits must share the same epoch. The output has one
        epoch per ``dt`` value, shared across all orbits during
        propagation.

        Parameters
        ----------
        orbits : CartesianOrbits | CometaryOrbits | KeplerianOrbits | SphericalOrbits
            Orbits table. All orbits must share the same epoch.
        dt : array-like
            Time offsets in days from the orbit epoch.

        Returns
        -------
        Epochs
            Epochs in TDB at ``orbit_epoch + dt``.
        """
        t0s = np.asarray(orbits.coordinates.epoch.to_numpy(zero_copy_only=False), dtype=np.float64)
        if len(t0s) > 1 and not np.allclose(t0s, t0s[0]):
            raise ValueError(
                f"from_orbits requires all orbits to share the same epoch. Got epochs: {t0s}"
            )
        t0 = float(t0s[0])
        dt_arr = np.asarray(dt, dtype=np.float64)
        return cls.from_kwargs(mjd=t0 + dt_arr, scale=TimeScale.TDB.value)

    # ── Range constructors ───────────────────────────────────

    @classmethod
    def linspace(
        cls,
        start: float,
        end: float,
        num: int = 50,
        scale: ScaleArg = TimeScale.TDB,
    ) -> "Epochs":
        """Create evenly spaced epochs between ``start`` and ``end``."""
        scale_str = _scale_str(scale)
        mjd = np.linspace(float(start), float(end), num)
        return cls.from_kwargs(mjd=mjd, scale=scale_str)

    @classmethod
    def arange(
        cls,
        start: float,
        end: float,
        step: float = 1.0,
        scale: ScaleArg = TimeScale.TDB,
    ) -> "Epochs":
        """Create epochs from ``start`` to ``end`` (exclusive) with a fixed step."""
        scale_str = _scale_str(scale)
        mjd = np.arange(float(start), float(end), float(step))
        return cls.from_kwargs(mjd=mjd, scale=scale_str)

    # ── Numpy / Arrow accessors ───────────────────────────────

    def to_numpy(self) -> np.ndarray:
        """Return the MJD column as a numpy ``float64`` array."""
        return np.asarray(self.mjd.to_numpy(zero_copy_only=False), dtype=np.float64)

    def mjd_tdb(self) -> np.ndarray:
        """Return MJD values in TDB as a numpy array.

        Converts internally if stored in another scale; returns the
        existing column directly when already TDB (no copy).
        """
        if self.scale == TimeScale.TDB.value:
            return self.to_numpy()
        return self.to_tdb().to_numpy()

    def mjd_utc(self) -> np.ndarray:
        """Return MJD values in UTC as a numpy array."""
        if self.scale == TimeScale.UTC.value:
            return self.to_numpy()
        return self.to_utc().to_numpy()

    def jd(self) -> np.ndarray:
        """Return Julian Date values in the stored scale (= MJD + 2400000.5)."""
        return self.to_numpy() + _JD_MJD_OFFSET

    # ── Convenience constructors ─────────────────────────────

    @classmethod
    def from_mjd(
        cls,
        mjd: float | Sequence[float] | np.ndarray,
        scale: ScaleArg = TimeScale.TDB,
    ) -> "Epochs":
        """Construct from MJD values + an explicit scale.

        Single-line shorthand for ``Epochs.from_kwargs(mjd=..., scale=...)``.

        >>> Epochs.from_mjd(60500.0)
        >>> Epochs.from_mjd([60500.0, 60501.0], scale="utc")
        """
        scale_str = _scale_str(scale)
        arr = np.atleast_1d(np.asarray(mjd, dtype=np.float64))
        return cls.from_kwargs(mjd=arr, scale=scale_str)

    @classmethod
    def from_jd(
        cls,
        jd: float | Sequence[float] | np.ndarray,
        scale: ScaleArg = TimeScale.TDB,
    ) -> "Epochs":
        """Construct from Julian Date values (converts to MJD = JD - 2400000.5)."""
        scale_str = _scale_str(scale)
        arr = np.atleast_1d(np.asarray(jd, dtype=np.float64)) - _JD_MJD_OFFSET
        return cls.from_kwargs(mjd=arr, scale=scale_str)

    @classmethod
    def now(cls, scale: ScaleArg = TimeScale.UTC) -> "Epochs":
        """Construct a single-row Epochs at "right now" in the requested scale.

        Uses the system clock (``datetime.now(timezone.utc)``) and the
        native ISO→MJD converter — no astropy dependency.
        """
        from datetime import datetime, timezone

        scale_str = _scale_str(scale)
        iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return cls.from_iso([iso], scale=scale_str)

    @classmethod
    def concat(cls, *epochs: "Epochs") -> "Epochs":
        """Concatenate multiple :class:`Epochs` tables.

        All inputs must share the same time scale.
        """
        if not epochs:
            return cls.from_kwargs(mjd=np.zeros(0), scale=TimeScale.TDB.value)
        scale = epochs[0].scale
        for e in epochs[1:]:
            if e.scale != scale:
                raise ValueError(f"cannot concat Epochs with mixed scales: {scale} vs {e.scale}")
        mjd = np.concatenate(
            [np.asarray(e.mjd.to_numpy(zero_copy_only=False), dtype=np.float64) for e in epochs]
        )
        return cls.from_kwargs(mjd=mjd, scale=scale)
