"""Per-row sensitivity tables for state-space and observation-space partials.

Two flat quivr Tables, one row per ``(orbit, epoch)`` (or
``(orbit, observer, epoch)`` for the observation side):

- :class:`StateSensitivities` — STM (and optional STT) at each output
  epoch of a propagation. ``stm`` / ``stt`` ride as row-major flattened
  ``LargeListColumn`` values (length 36 / 216).
- :class:`ObservationSensitivities` — observation Jacobian (and
  optional Hessian) at each ephemeris epoch. ``n_params`` column
  documents the inner shape (6 for state-only DC, 9 with non-grav);
  ``jacobian`` / ``hessian`` are variable-length flattened lists
  (length ``6·n_params`` / ``6·n_params²``). Their leading axis is the
  observable — range, RA, Dec, and their rates, in that order; index it
  with the ``SENSITIVITY_ROW_*`` constants defined here.

Filter to one chain with the standard quivr pattern before calling
the per-chain accessors::

    chain = sens.select("orbit_id", oid)  # one orbit
    obs_chain = obs.select("orbit_id", oid).select("obs_code", "F51")

Helper accessors on the filtered table (``stms_array``,
``jacobians_array``, ``index_at``, ``propagate_covariance``,
``kappa``) reshape the flat lists back to numpy matrices on demand.
"""

from datetime import datetime
from typing import Literal

import numpy as np
import quivr as qv

# ``pyarrow.compute`` generates its comparison wrappers (``less_equal`` …)
# dynamically at import time, so they are invisible to the static type
# checker. ``call_function`` is the public, statically-typed entry point
# these wrappers delegate to; import it from its defining module so the
# checker can resolve it.
from pyarrow._compute import call_function

from empyrean.coordinates.epoch import Epochs

EpochLike = float | str | datetime | Epochs
"""Anything :func:`StateSensitivities.index_at` accepts — a scalar MJD
TDB, a length-1 :class:`Epochs`, an ISO-8601 string, or a ``datetime``."""


# ── Observation-sensitivity row order ─────────────────────────────────
#
# Row indices into the observation Jacobian's leading axis, shared by
# the Hessian. The angles are in degrees and the range in AU, so a
# wrong row is wrong in unit as well as in observable — reading row 0
# as RA yields a range partial in AU, which is finite, plausible, and
# silently wrong. Use these instead of literals.

SENSITIVITY_ROW_RANGE = 0
"""Row of the range (topocentric distance) partials, in AU per input unit."""

SENSITIVITY_ROW_RA = 1
"""Row of the right-ascension partials, in degrees per input unit."""

SENSITIVITY_ROW_DEC = 2
"""Row of the declination partials, in degrees per input unit."""

SENSITIVITY_ROW_VRANGE = 3
"""Row of the range-rate partials, in AU/day per input unit."""

SENSITIVITY_ROW_VRA = 4
"""Row of the RA-rate partials, in deg/day per input unit. The rate is
dRA/dt, not scaled by cos(Dec)."""

SENSITIVITY_ROW_VDEC = 5
"""Row of the Dec-rate partials, in deg/day per input unit."""


# ── State-space sensitivity ───────────────────────────────────────────


class StateSensitivities(qv.Table):
    """Per-``(orbit, epoch)`` state-transition matrices and tensors.

    One row per output epoch; rows are grouped contiguously by
    ``orbit_id`` (matching propagation's orbit-major output). Filter
    to one chain with quivr's standard ``select`` before calling the
    per-chain accessors::

        chain = sens.select("orbit_id", "2020 CD3")
        phi = chain.stms_array()[chain.index_at(60750.0)]

    Notes
    -----
    Matrices are stored row-major flattened in ``LargeListColumn`` s:

    - ``stm`` is the 6×6 STM Φ flattened to length 36
      (``stm[6·r + c] = Φ[r, c]``). ``None`` per row when STMs were not
      computed for that row.
    - ``stt`` is the 6×6×6 STT Ψ flattened to length 216
      (``stt[36·k + 6·a + b] = Ψ[k, a, b]``). ``None`` when the
      propagation method did not carry STTs (anything other than
      ``UncertaintyMethod.SECOND_ORDER``).

    The accessors below operate on whatever rows are present —
    typically a single chain after ``select("orbit_id", oid)``, but
    work on the full table too when shapes are uniform.
    """

    orbit_id = qv.LargeStringColumn()
    """Orbit primary key (matches the input ``Orbits.orbit_id``)."""
    object_id = qv.LargeStringColumn(nullable=True)
    """Object metadata label, if carried on the input orbit."""
    epoch_mjd_tdb = qv.Float64Column()
    """Output epoch (MJD TDB)."""
    stm = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    """Row-major flattened 6×6 STM (length 36 per row), or ``None``."""
    stt = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    """Row-major flattened 6×6×6 STT (length 216 per row), or ``None``."""
    resolved_kind = qv.LargeStringColumn(nullable=True)
    """Resolved covariance kind at this output epoch
    (:class:`~empyrean.propagation.tagged_covariance.CovarianceKind` value:
    ``linear`` / ``second_order`` / …), or ``None`` if the propagation did
    not resolve a covariance for this row."""

    # ── Introspection ─────────────────────────────────────────

    def orbit_ids_unique(self) -> list[str]:
        """Unique ``orbit_id`` values, in first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for v in self.orbit_id.to_pylist():
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    # ── Matrix reshaping ──────────────────────────────────────

    def stms_array(self) -> np.ndarray | None:
        """Reshape ``stm`` to ``(n_t, 6, 6)``.

        Returns ``None`` when every row has a null STM. Null rows
        within an otherwise-populated chain are filled with NaN.
        Raises :class:`ValueError` if the table holds more than one
        unique ``orbit_id`` — filter via ``select`` first.
        """
        _require_single_state_chain(self, "stms_array")
        col = self.column("stm")
        if col.null_count == len(col):
            return None
        rows = col.to_pylist()
        n = len(rows)
        out = np.empty((n, 36), dtype=np.float64)
        for i, row in enumerate(rows):
            if row is None:
                out[i, :] = np.nan
            else:
                out[i, :] = row
        return out.reshape(n, 6, 6)

    def stts_array(self) -> np.ndarray | None:
        """Reshape ``stt`` to ``(n_t, 6, 6, 6)``.

        Returns ``None`` when every row has a null STT.
        Raises :class:`ValueError` if the table holds more than one
        unique ``orbit_id`` — filter via ``select`` first.
        """
        _require_single_state_chain(self, "stts_array")
        col = self.column("stt")
        if col.null_count == len(col):
            return None
        rows = col.to_pylist()
        n = len(rows)
        out = np.empty((n, 216), dtype=np.float64)
        for i, row in enumerate(rows):
            if row is None:
                out[i, :] = np.nan
            else:
                out[i, :] = row
        return out.reshape(n, 6, 6, 6)

    # ── Epoch lookup ──────────────────────────────────────────

    def index_at(self, epoch: EpochLike, *, atol: float = 1e-9) -> int:
        """Row index at the given epoch.

        ``epoch`` is converted to MJD TDB and matched within ``atol``.
        Raises :class:`ValueError` if no row matches, or if the table
        holds more than one unique ``orbit_id`` — filter via ``select``
        first.
        """
        _require_single_state_chain(self, "index_at")
        target = _to_mjd_tdb(epoch)
        mjd = self.column("epoch_mjd_tdb").to_numpy(zero_copy_only=False)
        diffs = np.abs(mjd - target)
        i = int(np.argmin(diffs))
        if diffs[i] > atol:
            raise ValueError(
                f"epoch MJD TDB {target} not found "
                f"(nearest row {i} at MJD TDB {mjd[i]}, Δ={diffs[i]:.3e} > "
                f"atol={atol:.3e})"
            )
        return i

    def up_to(self, epoch: EpochLike) -> "StateSensitivities":
        """Subset including rows with ``epoch_mjd_tdb ≤`` the target."""
        target = _to_mjd_tdb(epoch)
        mask = call_function("less_equal", [self.column("epoch_mjd_tdb"), target])
        return self.apply_mask(mask)

    # ── Covariance propagation ────────────────────────────────

    def propagate_covariance(
        self,
        cov_in: np.ndarray,
        *,
        i: int | None = None,
        order: Literal[1, 2, "auto"] = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward-propagate a covariance through the chain.

        Filter to a single chain via ``select("orbit_id", oid)`` first
        — this method assumes the chain's STMs share a common t₀.

        Parameters
        ----------
        cov_in : np.ndarray
            Input 6×6 covariance at the chain's start epoch.
        i : int, optional
            Row index to evaluate at. ``None`` (default) returns the
            covariance at every chain epoch, shape ``(n_t, 6, 6)``.
        order : {1, 2, "auto"}
            ``1``: linear (``Σ = Φ Σ_0 Φᵀ``, ``Δμ = 0``).
            ``2``: Jet2 second-order Gaussian correction (requires STTs).
            ``"auto"``: order 2 when STTs are present, else order 1.

        Returns
        -------
        (cov_out, delta_mu) : (np.ndarray, np.ndarray)
            ``(6, 6)`` / ``(6,)`` for scalar ``i``;
            ``(n_t, 6, 6)`` / ``(n_t, 6)`` for ``i=None``.
        """
        stms = self.stms_array()
        if stms is None:
            raise ValueError(
                "chain has no STMs — propagation method did not compute them "
                "(Monte Carlo / SigmaPoint, or FirstOrder without input covariance)"
            )
        cov_in = np.asarray(cov_in, dtype=np.float64)
        if cov_in.shape != (6, 6):
            raise ValueError(f"cov_in must be (6, 6), got {cov_in.shape}")

        stts = self.stts_array()
        if order == "auto":
            order = 2 if stts is not None else 1
        if order == 2 and stts is None:
            raise ValueError(
                "order=2 requires STTs; chain has none "
                "(run propagate with uncertainty_method=UncertaintyMethod.SECOND_ORDER)"
            )
        if order not in (1, 2):
            raise ValueError(f"order must be 1, 2, or 'auto'; got {order!r}")

        sub_stms = stms if i is None else stms[i : i + 1]
        sub_stts: np.ndarray | None = None
        if order == 2:
            # stts is guaranteed non-None here: order==2 with stts is None
            # raised above, and order=="auto" resolved to 2 only when stts
            # was non-None.
            assert stts is not None
            sub_stts = stts if i is None else stts[i : i + 1]

        cov_out, delta_mu = _propagate_cov_batch(sub_stms, sub_stts, cov_in, order=order)
        if i is None:
            return cov_out, delta_mu
        return cov_out[0], delta_mu[0]

    def kappa(
        self,
        cov_in: np.ndarray,
        *,
        i: int | None = None,
    ) -> float | np.ndarray:
        """Jet2 nonlinearity diagnostic κ.

        Approximates the departure of the true distribution from a
        Gaussian centered at the nominal state. Small κ (≲ 0.1) means
        first-order covariance is adequate; larger κ warrants Jet2 SOG
        or Gaussian-mixture splitting. Requires STTs. Filter to one
        chain first.
        """
        stts = self.stts_array()
        if stts is None:
            raise ValueError(
                "kappa requires STTs; chain has none "
                "(run propagate with uncertainty_method=UncertaintyMethod.SECOND_ORDER)"
            )
        cov_in = np.asarray(cov_in, dtype=np.float64)
        sub_stts = stts if i is None else stts[i : i + 1]
        kap = _kappa_batch(sub_stts, cov_in)
        return float(kap[0]) if i is not None else kap


# ── Observation-space sensitivity ─────────────────────────────────────


class ObservationSensitivities(qv.Table):
    """Per-``(orbit, observer, epoch)`` observation Jacobians and Hessians.

    Holds ∂h/∂x₀ at every ephemeris epoch for each ``(orbit, observer)``
    pair, plus the observation Hessians when the underlying propagation
    carried STTs. The Jacobian composes ∂(obs)/∂(state at t_obs) ·
    Φ(t_obs, t₀) and omits the light-time terms (the −v·∂τ/∂x partial;
    the STM is sampled at t_obs rather than emission t_obs − τ): both
    are O(τ), landing in the velocity columns of the angle rows with
    fractional error ≈ τ/Δt (τ ≈ 0.006–0.017 d) — negligible for
    multi-night arcs, growing as the arc shrinks toward intra-night.
    Filter to one chain via two ``select`` calls before using the
    per-chain accessors::

        chain = obs.select("orbit_id", oid).select("obs_code", "F51")
        H = chain.jacobians_array()[chain.index_at(60750.0)]

    Notes
    -----
    Matrix payloads are row-major flattened ``LargeListColumn`` values:

    - ``jacobian`` is ``(6, n_params)`` flattened to length
      ``6·n_params`` (``jacobian[n_params·r + c] = J[r, c]``).
    - ``hessian`` is ``(6, n_params, n_params)`` flattened to length
      ``6·n_params²``.

    The ``n_params`` column documents which: ``6`` for a state-only DC,
    ``9`` when non-gravitational parameters (A1, A2, A3) are also free
    variables. All rows within a single chain share the same
    ``n_params``. Columns ``0:6`` are the input Cartesian state; any
    further columns are the extra solved-for parameters.

    The leading axis — of both the Jacobian and the Hessian — is the
    topocentric spherical observable, in this order:

    ==== ============================ ============ ====================
    row  constant                     observable   unit per input unit
    ==== ============================ ============ ====================
    0    ``SENSITIVITY_ROW_RANGE``    range        AU
    1    ``SENSITIVITY_ROW_RA``       RA           deg
    2    ``SENSITIVITY_ROW_DEC``      Dec          deg
    3    ``SENSITIVITY_ROW_VRANGE``   range rate   AU/day
    4    ``SENSITIVITY_ROW_VRA``      RA rate      deg/day
    5    ``SENSITIVITY_ROW_VDEC``     Dec rate     deg/day
    ==== ============================ ============ ====================

    The RA rate is dRA/dt, **not** scaled by cos(Dec). Projecting an
    input covariance onto the sky plane therefore contracts rows 1 and
    2; row 0 is range, and reading it as RA gives a number in the wrong
    observable and the wrong unit. Index with the module-level
    ``SENSITIVITY_ROW_*`` constants rather than with literals::

        from empyrean import SENSITIVITY_ROW_DEC, SENSITIVITY_ROW_RA

        J = chain.jacobians_array()[chain.index_at(60750.0)]
        h_ra = J[SENSITIVITY_ROW_RA, :6]
        h_dec = J[SENSITIVITY_ROW_DEC, :6]
    """

    orbit_id = qv.LargeStringColumn()
    """Orbit primary key."""
    object_id = qv.LargeStringColumn(nullable=True)
    """Object metadata label."""
    obs_code = qv.LargeStringColumn()
    """MPC observatory code."""
    epoch_mjd_tdb = qv.Float64Column()
    """Observation epoch (MJD TDB)."""
    n_params = qv.UInt8Column()
    """Last-axis dimension of Jacobian / Hessian. 6 for state-only DC,
    9 when non-grav A1/A2/A3 are free. Constant within a chain."""
    jacobian = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    """Row-major flattened (6, n_params) Jacobian."""
    hessian = qv.LargeListColumn(qv.Float64Column(), nullable=True)
    """Row-major flattened (6, n_params, n_params) Hessian."""

    # ── Introspection ─────────────────────────────────────────

    def chain_keys(self) -> list[tuple[str, str]]:
        """Unique ``(orbit_id, obs_code)`` pairs, in first-seen order."""
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for oid, obs in zip(self.orbit_id.to_pylist(), self.obs_code.to_pylist(), strict=False):
            key = (oid, obs)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    # ── Matrix reshaping ──────────────────────────────────────

    def jacobians_array(self) -> np.ndarray | None:
        """Reshape ``jacobian`` to ``(n_t, 6, n_params)``.

        Axis 1 is the observable, in ``SENSITIVITY_ROW_*`` order
        (range, RA, Dec, and their rates); axis 2 is the input
        parameter, state first. Returns ``None`` when every row has a
        null Jacobian.
        """
        _require_single_obs_chain(self, "jacobians_array")
        col = self.column("jacobian")
        if col.null_count == len(col):
            return None
        n_p = int(self.column("n_params")[0].as_py())
        rows = col.to_pylist()
        n = len(rows)
        out = np.empty((n, 6 * n_p), dtype=np.float64)
        for i, row in enumerate(rows):
            if row is None:
                out[i, :] = np.nan
            else:
                out[i, :] = row
        return out.reshape(n, 6, n_p)

    def hessians_array(self) -> np.ndarray | None:
        """Reshape ``hessian`` to ``(n_t, 6, n_params, n_params)``.

        Axis 1 is the observable, in the same ``SENSITIVITY_ROW_*``
        order as :meth:`jacobians_array`. Returns ``None`` when every
        row has a null Hessian.
        """
        _require_single_obs_chain(self, "hessians_array")
        col = self.column("hessian")
        if col.null_count == len(col):
            return None
        n_p = int(self.column("n_params")[0].as_py())
        rows = col.to_pylist()
        n = len(rows)
        out = np.empty((n, 6 * n_p * n_p), dtype=np.float64)
        for i, row in enumerate(rows):
            if row is None:
                out[i, :] = np.nan
            else:
                out[i, :] = row
        return out.reshape(n, 6, n_p, n_p)

    # ── Epoch lookup ──────────────────────────────────────────

    def index_at(self, epoch: EpochLike, *, atol: float = 1e-9) -> int:
        """Row index at the given epoch.

        Filter to a single chain via two ``select`` calls first if
        epochs repeat across chains.
        """
        _require_single_obs_chain(self, "index_at")
        target = _to_mjd_tdb(epoch)
        mjd = self.column("epoch_mjd_tdb").to_numpy(zero_copy_only=False)
        diffs = np.abs(mjd - target)
        i = int(np.argmin(diffs))
        if diffs[i] > atol:
            raise ValueError(
                f"epoch MJD TDB {target} not found "
                f"(nearest row {i} at MJD TDB {mjd[i]}, Δ={diffs[i]:.3e} > "
                f"atol={atol:.3e})"
            )
        return i

    def up_to(self, epoch: EpochLike) -> "ObservationSensitivities":
        """Subset including rows with ``epoch_mjd_tdb ≤`` the target."""
        target = _to_mjd_tdb(epoch)
        mask = call_function("less_equal", [self.column("epoch_mjd_tdb"), target])
        return self.apply_mask(mask)

    # ── Covariance propagation ────────────────────────────────

    def propagate_covariance(
        self,
        cov_in: np.ndarray,
        *,
        i: int | None = None,
        order: Literal[1, 2, "auto"] = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map an initial-state covariance into observation-frame
        covariance.

        Filter to a single ``(orbit_id, obs_code)`` chain first.

        Returns ``(Σ_obs, Δμ_obs)`` with shapes:
        ``i=None``: ``(n_t, 6, 6)``, ``(n_t, 6)``
        ``i=int``:  ``(6, 6)``, ``(6,)``

        Both are in the observable basis — the ``SENSITIVITY_ROW_*``
        order (range, RA, Dec, and their rates), with AU on the range
        axes and degrees on the angle axes. A sky-plane 2×2 is the
        ``[SENSITIVITY_ROW_RA, SENSITIVITY_ROW_DEC]`` submatrix of
        ``Σ_obs``, in deg², before any cos(Dec) scaling.
        """
        jacs = self.jacobians_array()
        if jacs is None:
            raise ValueError(
                "chain has no Jacobians — ephemeris generation did not carry "
                "observation partials (likely no input covariance)"
            )
        n_p = jacs.shape[-1]
        cov_in = np.asarray(cov_in, dtype=np.float64)
        if cov_in.shape != (n_p, n_p):
            raise ValueError(f"cov_in must be ({n_p}, {n_p}) for this chain, got {cov_in.shape}")

        hess = self.hessians_array()
        if order == "auto":
            order = 2 if hess is not None else 1
        if order == 2 and hess is None:
            raise ValueError(
                "order=2 requires Hessians; chain has none (propagation wasn't SECOND_ORDER)"
            )

        sub_jacs = jacs if i is None else jacs[i : i + 1]
        sub_hess: np.ndarray | None = None
        if order == 2:
            # hess is guaranteed non-None here: order==2 with hess is None
            # raised above, and order=="auto" resolved to 2 only when hess
            # was non-None.
            assert hess is not None
            sub_hess = hess if i is None else hess[i : i + 1]

        cov_out, delta_mu = _propagate_obs_cov_batch(sub_jacs, sub_hess, cov_in, order=order)
        if i is None:
            return cov_out, delta_mu
        return cov_out[0], delta_mu[0]


# ── Internal helpers ──────────────────────────────────────────────────


def _to_mjd_tdb(epoch: EpochLike) -> float:
    """Coerce any accepted epoch input to a scalar MJD TDB."""
    if isinstance(epoch, Epochs):
        if len(epoch) != 1:
            raise ValueError(f"expected a single-row Epochs, got length {len(epoch)}")
        return float(epoch.to_tdb().mjd.to_numpy(zero_copy_only=False)[0])
    if isinstance(epoch, str):
        return float(Epochs.from_iso([epoch]).to_tdb().mjd.to_numpy(zero_copy_only=False)[0])
    if isinstance(epoch, datetime):
        return float(
            Epochs.from_iso([epoch.isoformat()]).to_tdb().mjd.to_numpy(zero_copy_only=False)[0]
        )
    return float(epoch)


def _require_single_state_chain(table: StateSensitivities, method: str) -> None:
    """Guard for per-chain methods on :class:`StateSensitivities`.

    Raises :class:`ValueError` if the table contains more than one
    unique ``orbit_id``, with a hint to filter via ``select`` first.
    """
    oids = table.orbit_ids_unique()
    if len(oids) > 1:
        preview = ", ".join(repr(o) for o in oids[:3])
        more = f" (+{len(oids) - 3} more)" if len(oids) > 3 else ""
        raise ValueError(
            f"{method}() requires a single chain but got {len(oids)} unique "
            f"orbit_ids: {preview}{more}. Filter to one chain first: "
            f'sens.select("orbit_id", "<orbit_id>").{method}(...)'
        )


def _require_single_obs_chain(table: ObservationSensitivities, method: str) -> None:
    """Guard for per-chain methods on :class:`ObservationSensitivities`.

    Raises :class:`ValueError` if the table contains more than one
    unique ``(orbit_id, obs_code)`` pair, with a hint to filter via
    chained ``select`` calls first.
    """
    keys = table.chain_keys()
    if len(keys) > 1:
        preview = ", ".join(repr(k) for k in keys[:3])
        more = f" (+{len(keys) - 3} more)" if len(keys) > 3 else ""
        raise ValueError(
            f"{method}() requires a single chain but got {len(keys)} unique "
            f"(orbit_id, obs_code) pairs: {preview}{more}. Filter to one "
            f"chain first: "
            f'obs.select("orbit_id", "<oid>").select("obs_code", "<code>").{method}(...)'
        )


def _propagate_cov_batch(
    stms: np.ndarray,  # (k, 6, 6)
    stts: np.ndarray | None,  # (k, 6, 6, 6) or None
    cov_in: np.ndarray,  # (6, 6)
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized state-space covariance propagation.

    Returns ``(cov_out, delta_mu)`` with shapes ``(k, 6, 6)`` and ``(k, 6)``.
    """
    cov_out = np.einsum("tij,jk,tlk->til", stms, cov_in, stms)
    k = stms.shape[0]
    delta_mu = np.zeros((k, 6), dtype=np.float64)
    if order == 2 and stts is not None:
        term1 = np.einsum("tkab,tlcd,ac,bd->tkl", stts, stts, cov_in, cov_in)
        term2 = np.einsum("tkab,tlcd,ad,bc->tkl", stts, stts, cov_in, cov_in)
        cov_out = cov_out + 0.5 * (term1 + term2)
        delta_mu = 0.5 * np.einsum("tkab,ab->tk", stts, cov_in)
    return cov_out, delta_mu


def _kappa_batch(
    stts: np.ndarray,  # (k, 6, 6, 6)
    cov_in: np.ndarray,  # (6, 6)
) -> np.ndarray:
    """Per-epoch κ_t = ‖Ψ_t · Σ_0‖_F / 6."""
    quad = np.einsum("tkab,ab->tk", stts, cov_in)
    return np.asarray(np.linalg.norm(quad, axis=1) / 6.0, dtype=np.float64)


def _propagate_obs_cov_batch(
    jacs: np.ndarray,  # (k, 6, n)
    hess: np.ndarray | None,  # (k, 6, n, n) or None
    cov_in: np.ndarray,  # (n, n)
    *,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized observation-space covariance propagation."""
    cov_out = np.einsum("tij,jk,tlk->til", jacs, cov_in, jacs)
    k = jacs.shape[0]
    delta_mu = np.zeros((k, 6), dtype=np.float64)
    if order == 2 and hess is not None:
        term1 = np.einsum("tkab,tlcd,ac,bd->tkl", hess, hess, cov_in, cov_in)
        term2 = np.einsum("tkab,tlcd,ad,bc->tkl", hess, hess, cov_in, cov_in)
        cov_out = cov_out + 0.5 * (term1 + term2)
        delta_mu = 0.5 * np.einsum("tkab,ab->tk", hess, cov_in)
    return cov_out, delta_mu
