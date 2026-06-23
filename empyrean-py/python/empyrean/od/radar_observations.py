"""ADES radar observation types."""

from collections.abc import Callable

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv

# pyarrow's compute functions are generated at runtime into the
# ``pyarrow.compute`` module namespace (see ``_make_global_functions``),
# so the bundled type stubs do not declare them and mypy cannot resolve
# ``pc.is_in`` / ``pc.equal`` as attributes. Bind the ones we use to
# precisely-typed module-level aliases via the module ``__dict__`` (the
# alias is the exact same function object — runtime behavior is unchanged)
# so every call site gets a real signature.
_is_in: Callable[..., pa.BooleanArray] = pc.__dict__["is_in"]
_equal: Callable[[pa.ChunkedArray | pa.Array, pa.Scalar], pa.BooleanArray] = pc.__dict__["equal"]


class ADESRadarObservations(qv.Table):
    """ADES radar observations — full schema.

    ADES models radar astrometry as its own top-level table, parallel to
    the optical :class:`~empyrean.od.ades_observations.ADESObservations`
    (not as an optical ``mode``). Each record carries a round-trip time
    **delay** *or* a **Doppler** shift — the ADES ``RadarValue`` is an XSD
    ``<choice>`` — referred to a transmitting (``trx``) and receiving
    (``rcv``) station (equal for a monostatic observation, distinct for a
    bistatic one).

    Units are ADES-native end to end — no conversion is applied in this
    layer (the single SI normalization happens once downstream):

    * ``delay`` — round-trip time delay in **seconds**
    * ``rms_delay`` — its 1σ uncertainty in **microseconds** (the
      asymmetry vs ``delay`` is intentional in the ADES schema)
    * ``doppler`` / ``rms_doppler`` — Doppler shift and 1σ in **Hz**
      (signed value)
    * ``frq`` — transmit carrier reference frequency in **MHz**

    The ``observable`` discriminator column (``"delay"`` / ``"doppler"``)
    — *not* a NaN probe — records which measurement each row carries, so a
    genuine 0.0-Hz Doppler is never confused with an absent one. The
    inactive value column is null for that row.
    """

    # ── Identification ────────────────────────────────
    perm_id = qv.LargeStringColumn(nullable=True)
    prov_id = qv.LargeStringColumn(nullable=True)
    trk_sub = qv.LargeStringColumn(nullable=True)

    # ── Bistatic geometry ─────────────────────────────
    trx = qv.LargeStringColumn()  # MPC station code of the transmitter
    rcv = qv.LargeStringColumn()  # MPC station code of the receiver

    # ── Core measurement ──────────────────────────────
    obs_time = qv.LargeStringColumn()  # ISO 8601 UTC (receive epoch)
    # Discriminator: which RadarValue choice this row carries.
    observable = qv.LargeStringColumn()  # "delay" | "doppler"
    delay = qv.Float64Column(nullable=True)  # seconds
    rms_delay = qv.Float64Column(nullable=True)  # MICROSECONDS (ADES rmsDelay)
    doppler = qv.Float64Column(nullable=True)  # Hz (signed)
    rms_doppler = qv.Float64Column(nullable=True)  # Hz

    # ── Reduction metadata ────────────────────────────
    frq = qv.Float64Column()  # MHz (transmit carrier reference)
    com = qv.BooleanColumn(nullable=True)  # center-of-mass flag (ADES com)
    log_snr = qv.Float64Column(nullable=True)
    remarks = qv.LargeStringColumn(nullable=True)

    # ── Selection helpers ────────────────────────────────

    def select_station(self, codes: str | list[str]) -> "ADESRadarObservations":
        """Rows whose receiving station (``rcv``) is one of ``codes``."""
        wanted = [codes] if isinstance(codes, str) else list(codes)
        mask = _is_in(self.column("rcv"), value_set=pa.array(wanted))
        return self.apply_mask(mask)

    def delays(self) -> "ADESRadarObservations":
        """Rows carrying a delay measurement (``observable == "delay"``)."""
        mask = _equal(self.column("observable"), pa.scalar("delay"))
        return self.apply_mask(mask)

    def dopplers(self) -> "ADESRadarObservations":
        """Rows carrying a Doppler measurement (``observable == "doppler"``)."""
        mask = _equal(self.column("observable"), pa.scalar("doppler"))
        return self.apply_mask(mask)
