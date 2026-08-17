"""Retained adaptive-Gaussian-mixture (AGM) components.

Under ``GAUSSIAN_MIXTURE`` — and inside ``AUTO``'s close-approach
windows — the engine splits the input Gaussian into a mixture and
retains the resulting components at every close approach where the
splitter actually fired. This module is the read-back for those
components: the mixture itself, not its moment collapse. A consumer can
evaluate ``sum_k w_k * N(x | mu_k, Sigma_k)`` directly at a retained CA
epoch, with no further propagation.

Two shapes, mirroring
:mod:`~empyrean.propagation.tagged_covariance`:

- :class:`MixtureChains` — a flat quivr Table, one row per
  ``(orbit, CA epoch, component)``, rows grouped contiguously by
  ``orbit_id``. Empty (zero rows) when nothing split — never
  zero-filled placeholder rows.
- :class:`MixtureComponent` — a small per-component dataclass with the
  mean and covariance re-materialized as contiguous ``np.ndarray`` and
  the basis names decoded, returned by
  :meth:`MixtureChains.to_chains`.

Scope of what is retained
-------------------------

Four limits apply, and each is a real property of the engine's
retention rather than a marshaling shortfall:

- **Depth-0 only.** Only the initial split is retained; recursive AGM
  calls (depth > 0) are not captured.
- **Only CA epochs where AGM fired.** An orbit that never triggered a
  split contributes no rows, not a one-component chain.
- **Component covariance is the linear map.** Each component's
  covariance is ``Phi Sigma_k Phi^T``; the second-order mean correction
  is intentionally omitted.
- **Retained weights may sum to less than 1.** A sub-Gaussian whose own
  sub-propagation missed the close approach (or failed to integrate)
  contributes no component, and the deficit is not recorded anywhere.
  Do not assume ``sum_k w_k == 1``; sum ``weight`` and check.

Note the name: :class:`MixtureComponent` here is the basis-tagged AGM
read-back component, a different type from the top-level
``empyrean.MixtureComponent``, which is the
:func:`~empyrean.split_gaussian` primitive at t0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import quivr as qv


@dataclass
class MixtureComponent:
    """One Gaussian sub-component of a retained AGM decomposition.

    Attributes
    ----------
    weight : float
        Prior split weight from the Gaussian splitting library — never
        likelihood-reweighted. See the module docstring: the retained
        weights of one CA epoch may sum to less than 1.
    mean : np.ndarray
        Propagated sub-Gaussian centroid ``[x, y, z, vx, vy, vz]`` (AU,
        AU/day) at the CA epoch, shape ``(6,)``, in the basis given by
        ``frame`` / ``origin``.
    covariance : np.ndarray
        Linearly-mapped component covariance ``Phi Sigma_k Phi^T``,
        contiguous, shape ``(6, 6)``, same basis as ``mean``.
    frame : str
        Reference frame of the basis (canonical name, e.g. ``"icrf"``).
    origin : str
        Canonical origin (center body) name of the basis. Matches the
        propagation origin at the split's close-approach epoch, so it can
        differ between CA epochs of the same chain when origin switching
        occurred.
    """

    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    frame: str
    origin: str


class MixtureChains(qv.Table):
    """Flat per-``(orbit, CA epoch, component)`` AGM component readback.

    One row per retained component. Rows are grouped contiguously by
    ``orbit_id`` (matching propagation's orbit-major output); within an
    orbit they are grouped by ``ca_epoch_mjd_tdb`` and ordered by
    ``component_index``.

    An orbit whose splitter never fired contributes **zero rows**. A
    ``FIRST_ORDER`` propagation therefore yields an empty table, which is
    the honest answer — a zero-filled placeholder row would read as a
    one-component mixture of weight 0.

    Read the module docstring's scope notes before consuming these:
    depth-0 only, CA epochs only, linear component covariance, and
    retained weights that may sum to less than 1.
    """

    orbit_id = qv.LargeStringColumn()
    """Orbit primary key (matches the input ``Orbits.orbit_id``)."""
    orbit_index = qv.UInt32Column()
    """Zero-based index into the input orbits (orbit-major order)."""
    ca_epoch_mjd_tdb = qv.Float64Column()
    """Close-approach epoch the components were retained at (MJD TDB)."""
    component_index = qv.UInt32Column()
    """Zero-based index of this component within its CA epoch's group."""
    weight = qv.Float64Column()
    """Prior split weight. See the class docstring on the weight sum."""

    # Propagated sub-Gaussian centroid [x, y, z, vx, vy, vz].
    mean_x = qv.Float64Column()
    mean_y = qv.Float64Column()
    mean_z = qv.Float64Column()
    mean_vx = qv.Float64Column()
    mean_vy = qv.Float64Column()
    mean_vz = qv.Float64Column()

    covariance = qv.LargeListColumn(qv.Float64Column())
    """6x6 row-major component covariance, 36 values, in the basis named
    by ``frame`` / ``origin``."""

    origin = qv.LargeStringColumn()
    """Canonical origin (center body) name of the basis."""
    frame = qv.LargeStringColumn()
    """Reference frame of the basis (canonical name)."""

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

    def ca_epochs(self, orbit_index: int) -> list[float]:
        """Retained CA epochs (MJD TDB) for one orbit, in table order."""
        idx = self.orbit_index.to_numpy(zero_copy_only=False)
        epochs = self.ca_epoch_mjd_tdb.to_numpy(zero_copy_only=False)
        out: list[float] = []
        for i in range(len(self)):
            if int(idx[i]) != orbit_index:
                continue
            e = float(epochs[i])
            if not out or out[-1] != e:
                out.append(e)
        return out

    # ── Per-chain view ────────────────────────────────────────

    def to_chains(self, orbit_index: int) -> list[list[MixtureComponent]]:
        """One orbit's retained components, grouped by CA epoch.

        Returns one inner list per retained CA epoch of the orbit at
        ``orbit_index``, in the order :meth:`ca_epochs` reports, with
        each mean/covariance re-materialized as a contiguous array and
        the basis names decoded.

        An orbit that retained nothing returns an empty list — a real
        answer, not an error.
        """
        idx = self.orbit_index.to_numpy(zero_copy_only=False)
        epochs = self.ca_epoch_mjd_tdb.to_numpy(zero_copy_only=False)
        weights = self.weight.to_numpy(zero_copy_only=False)
        covs = self.covariance.to_pylist()
        origins = self.origin.to_pylist()
        frames = self.frame.to_pylist()
        means = np.column_stack(
            [
                self.column(f"mean_{lab}").to_numpy(zero_copy_only=False)
                for lab in ("x", "y", "z", "vx", "vy", "vz")
            ]
        )

        out: list[list[MixtureComponent]] = []
        current: float | None = None
        for i in range(len(self)):
            if int(idx[i]) != orbit_index:
                continue
            e = float(epochs[i])
            if current is None or e != current:
                current = e
                out.append([])
            out[-1].append(
                MixtureComponent(
                    weight=float(weights[i]),
                    mean=np.ascontiguousarray(means[i], dtype=np.float64),
                    covariance=np.ascontiguousarray(
                        np.asarray(covs[i], dtype=np.float64).reshape(6, 6)
                    ),
                    frame=str(frames[i]),
                    origin=str(origins[i]),
                )
            )
        return out

    # ── Persistence ───────────────────────────────────────────

    def to_dir(self, path: str) -> None:
        """Write this table to ``{path}/mixtures.parquet``.

        The directory is created if it does not exist. An empty table is
        still written, so ``from_dir`` round-trips "nothing split" as an
        empty table rather than as absence.
        """
        os.makedirs(path, exist_ok=True)
        self.to_parquet(os.path.join(path, "mixtures.parquet"))

    @classmethod
    def from_dir(cls, path: str) -> MixtureChains:
        """Read a table written by :meth:`to_dir`.

        Returns an empty table when the directory holds no
        ``mixtures.parquet`` — the same shape a propagation that split
        nothing produces.
        """
        fpath = os.path.join(path, "mixtures.parquet")
        if not os.path.exists(fpath):
            return cls.empty()
        return cls.from_parquet(fpath)


def build_mixture_chains(result: dict[str, object]) -> MixtureChains:
    """Build a :class:`MixtureChains` table from the Rust result dict.

    ``result`` is the dict returned by the ``_propagate`` extension,
    which sets ``"mixtures"`` unconditionally — present and carrying
    zero rows is how "nothing split" is spelled.

    A MISSING key is a different fault and raises: the only way to reach
    it is a compiled extension older than this Python package, and
    returning an empty table would spell that skew exactly like a
    genuine "the splitter never fired", so a mixture run that did split
    would report no components and ``to_dir`` would persist the wrong
    claim.
    """
    from empyrean._convert import int_to_frame, naif_to_origin

    if "mixtures" not in result:
        raise RuntimeError(
            "the compiled empyrean extension emitted no 'mixtures' key; it is older than "
            "this Python package — rebuild the wheel (maturin develop / pip install -e .). "
            "An empty table here would be indistinguishable from a propagation that split "
            "nothing."
        )
    mix = result["mixtures"]
    if not isinstance(mix, dict):
        raise TypeError(
            f"the compiled empyrean extension emitted 'mixtures' as {type(mix).__name__}, "
            "expected a dict of parallel columns — the extension and this Python package "
            "disagree about the propagation result shape; rebuild the wheel."
        )

    orbit_index = np.asarray(mix["mixture_orbit_index"], dtype=np.uint32)
    orbit_id = list(mix["mixture_orbit_id"])
    ca_epoch = np.asarray(mix["mixture_ca_epoch_mjd_tdb"], dtype=np.float64)
    component_index = np.asarray(mix["mixture_component_index"], dtype=np.uint32)
    weight = np.asarray(mix["mixture_weight"], dtype=np.float64)
    mean = np.asarray(mix["mixture_mean"], dtype=np.float64)  # (n, 6)
    covariance = np.asarray(mix["mixture_covariance"], dtype=np.float64)  # (n, 6, 6)
    frame_codes = np.asarray(mix["mixture_frame"], dtype=np.int64)
    origin_codes = np.asarray(mix["mixture_origin"], dtype=np.int64)

    n = len(orbit_id)
    return MixtureChains.from_kwargs(
        orbit_id=orbit_id,
        orbit_index=orbit_index,
        ca_epoch_mjd_tdb=ca_epoch,
        component_index=component_index,
        weight=weight,
        mean_x=mean[:, 0],
        mean_y=mean[:, 1],
        mean_z=mean[:, 2],
        mean_vx=mean[:, 3],
        mean_vy=mean[:, 4],
        mean_vz=mean[:, 5],
        covariance=[covariance[i].reshape(36).tolist() for i in range(n)],
        origin=[naif_to_origin(int(o)) for o in origin_codes],
        frame=[int_to_frame(int(f)).value for f in frame_codes],
    )
