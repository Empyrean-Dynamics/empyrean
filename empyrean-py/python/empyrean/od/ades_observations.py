"""ADES optical observation types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

if TYPE_CHECKING:

    class _IsIn(Protocol):
        """Static signature for :func:`pyarrow.compute.is_in`.

        pyarrow generates its compute kernels (``is_in`` among them) into the
        module namespace at import time, so they are invisible to static
        analysis even though the module itself resolves. This Protocol gives
        mypy the true signature; the runtime binding below is the real
        ``pc.is_in`` function, so behavior is unchanged.
        """

        def __call__(
            self,
            values: pa.ChunkedArray | pa.Array,
            /,
            value_set: pa.Array,
            *,
            skip_nulls: bool = ...,
            options: Any = ...,
            memory_pool: pa.MemoryPool | None = ...,
        ) -> pa.BooleanArray: ...


# Typed handle onto the dynamically-generated kernel (see _IsIn above). The
# attribute is injected into pyarrow.compute's namespace at import time, so it
# is fetched from the module dict and re-typed through the Protocol for static
# callers.
_is_in: _IsIn = pc.__dict__["is_in"]


class ADESObservations(qv.Table):
    """ADES optical observations — full schema.

    Every named PSV column round-trips losslessly across the Python ↔
    wrapper ↔ C ABI boundary. The class name signals that the schema is
    the MPC ADES astrometric data exchange standard, not a generic
    observations record.
    """

    # ── Identification ────────────────────────────────
    perm_id = qv.LargeStringColumn(nullable=True)
    prov_id = qv.LargeStringColumn(nullable=True)
    trk_sub = qv.LargeStringColumn(nullable=True)
    obs_id = qv.LargeStringColumn(nullable=True)
    obs_sub_id = qv.LargeStringColumn(nullable=True)
    trk_id = qv.LargeStringColumn(nullable=True)

    # ── Observer ──────────────────────────────────────
    stn = qv.LargeStringColumn()  # MPC obs code
    mode = qv.LargeStringColumn(nullable=True)
    prog = qv.LargeStringColumn(nullable=True)

    # ── Observer location (roving / spacecraft) ──────
    sys = qv.LargeStringColumn(nullable=True)
    ctr = qv.Float64Column(nullable=True)
    pos1 = qv.Float64Column(nullable=True)
    pos2 = qv.Float64Column(nullable=True)
    pos3 = qv.Float64Column(nullable=True)

    # ── Astrometry ────────────────────────────────────
    obs_time = qv.LargeStringColumn()  # ISO 8601 UTC
    ra = qv.Float64Column()  # degrees
    dec = qv.Float64Column()  # degrees

    # ── Uncertainties ────────────────────────────────
    rms_ra = qv.Float64Column(nullable=True)  # arcsec
    rms_dec = qv.Float64Column(nullable=True)  # arcsec
    rms_corr = qv.Float64Column(nullable=True)  # correlation [-1, 1]

    # ── Astrometric catalog ──────────────────────────
    ast_cat = qv.LargeStringColumn(nullable=True)

    # ── Photometry ───────────────────────────────────
    mag = qv.Float64Column(nullable=True)
    rms_mag = qv.Float64Column(nullable=True)
    band = qv.LargeStringColumn(nullable=True)
    phot_cat = qv.LargeStringColumn(nullable=True)
    phot_ap = qv.Float64Column(nullable=True)  # arcsec

    # ── Supplementary diagnostics ────────────────────
    log_snr = qv.Float64Column(nullable=True)
    seeing = qv.Float64Column(nullable=True)  # arcsec FWHM
    exp = qv.Float64Column(nullable=True)  # seconds
    rms_fit = qv.Float64Column(nullable=True)  # arcsec
    n_stars = qv.Int32Column(nullable=True)
    notes = qv.LargeStringColumn(nullable=True)
    remarks = qv.LargeStringColumn(nullable=True)

    # ── Selection helpers ────────────────────────────────

    def select_station(self, codes: str | list[str]) -> ADESObservations:
        """Rows from one or more MPC observatory codes (the ``stn``
        column)."""
        wanted = [codes] if isinstance(codes, str) else list(codes)
        mask = _is_in(self.column("stn"), value_set=pa.array(wanted))
        return self.apply_mask(mask)

    # ── Arc statistics ───────────────────────────────────

    @staticmethod
    def _obs_times_to_days(times: list[str]) -> np.ndarray:
        """Parse ISO-8601 UTC strings to a 1-D float64 array of MJD-
        agnostic days-since-epoch.

        ADES `obs_time` carries the trailing ``Z`` UTC marker; numpy's
        `datetime64` rejects timezone-aware strings (raises a deprecation
        warning), so strip the suffix before parsing. The downstream
        consumers care only about deltas / gaps, not the absolute epoch,
        so the days-since-Unix-epoch normalization is fine.
        """
        cleaned = [t.rstrip("Z") if isinstance(t, str) else t for t in times]
        return np.array(cleaned, dtype="datetime64[ns]").astype("int64") / 1e9 / 86400.0

    @property
    def time_span_days(self) -> float:
        """Arc length in days, max(obs_time) − min(obs_time). Empty /
        single-row tables return 0.0."""
        times = self.obs_time.to_pylist()
        if len(times) < 2:
            return 0.0
        days = self._obs_times_to_days(times)
        return float(days.max() - days.min())

    def n_oppositions(self, gap_days: float = 90.0) -> int:
        """Count of distinct apparitions, defined as ``1 + (number of
        consecutive-observation gaps exceeding ``gap_days``)``.

        90-day default matches scott's :attr:`ODConfig.opposition_gap_days`
        and the conventional planetary-science threshold for
        Sun-synodic visibility windows. Empty tables return 0;
        single-row tables return 1.
        """
        times = self.obs_time.to_pylist()
        if len(times) == 0:
            return 0
        if len(times) == 1:
            return 1
        days = np.sort(self._obs_times_to_days(times))
        gaps = np.diff(days)
        return int(1 + np.sum(gaps > gap_days))
