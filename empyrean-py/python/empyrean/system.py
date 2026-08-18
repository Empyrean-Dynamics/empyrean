"""Reusable pre-built force-model handle.

A :class:`BuiltSystem` assembles the force model — the perturber set,
spherical-harmonic gravity, the post-Newtonian relativistic correction,
and the integration frame — **once** for a frozen
``{force_model, frame, encounter_timescale_divisor}`` key, then reuses it
across every :meth:`~BuiltSystem.propagate` /
:meth:`~BuiltSystem.generate_ephemeris` call. For workloads dominated by
many short forward models (fit / predict / screen loops) the per-call
force-model assembly is the dominant cost; the handle amortizes it away.

Results are bit-identical to the one-shot :func:`empyrean.propagate` /
:func:`empyrean.generate_ephemeris` on the matching key — the handle only
changes *when* the force model is assembled, never the numerics.

Identity guard
--------------
Every forward-model call validates, before it runs, that the handle's
frozen key matches the call's configuration and that the source data has
not changed since the handle was built. Any mismatch — frame, force
model, divisor, source-data identity, or staleness — raises a loud,
distinct exception. The handle **never** silently rebuilds and never
serves wrong physics. **Rebuild the handle after any**
:func:`empyrean.initialize` / data (re)load.

Sharing across threads
----------------------
The underlying native handle is ``Send + Sync`` and every forward-model
call releases the GIL around the native propagation, so a single handle
may be shared by reference across Python threads and run concurrently.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from empyrean._convert import frame_to_int, int_to_frame
from empyrean.coordinates.enums import Frame
from empyrean.propagation.config import (
    _FORCE_MODEL_TO_INT,
    ForceModelTier,
    PropagationConfig,
)

if TYPE_CHECKING:
    from empyrean._convert import AnyOrbits
    from empyrean.coordinates.epoch import Epochs
    from empyrean.ephemeris.result import EphemerisConfig, EphemerisResult
    from empyrean.observers.observers import Observers
    from empyrean.orbits.thrust import ThrustParams
    from empyrean.propagation.result import PropagationResult

__all__ = [
    "BuiltSystem",
    "KernelKind",
    "KernelProvenance",
    "KernelRecord",
    "SystemDescription",
    "build_system",
    "od_system",
]

# Force-model tier code (the Rust boundary) → the Python enum. Inverse of
# ``_FORCE_MODEL_TO_INT`` restricted to the canonical integer keys.
_INT_TO_FORCE_MODEL = {
    0: ForceModelTier.APPROXIMATE,
    1: ForceModelTier.BASIC,
    2: ForceModelTier.STANDARD,
}


# ── Provenance description ────────────────────────────────────


class KernelKind(str, enum.Enum):
    """Category of a loaded data file in a handle's kernel manifest."""

    SPK = "spk"
    """SPK ephemeris kernel (planetary / small-body / spacecraft states)."""
    BPC = "bpc"
    """Binary PCK body-orientation kernel (Earth / Moon rotation)."""
    TPC = "tpc"
    """Text PCK of gravitational parameters."""
    GRAVITY = "gravity"
    """Gravity-field coefficient model (file or built-in field)."""
    OBSCODES = "obscodes"
    """Observatory-code registry."""


class KernelProvenance(str, enum.Enum):
    """Where a kernel in a handle's manifest came from."""

    FILE = "file"
    """Loaded from a file on disk (``path`` / ``sha256`` / ``bytes`` set)."""
    IN_MEMORY = "in_memory"
    """Handed over pre-loaded in memory (no path or hash known)."""
    BUILT_IN = "built_in"
    """Synthesized from constants compiled into the engine (``name`` set)."""


@dataclass(frozen=True)
class KernelRecord:
    """One entry of the kernel-identity manifest a handle captured at
    construction. Names the provenance of each loaded data file — never
    any tuning rationale.

    Only the fields the provenance variant supplies are populated; the
    rest are ``None`` (a ``FILE`` record carries ``path`` / ``sha256`` /
    ``bytes``; a ``BUILT_IN`` record carries ``name``).
    """

    kind: KernelKind
    provenance: KernelProvenance
    path: str | None = None
    """Absolute path the kernel was loaded from (``FILE`` only)."""
    sha256: str | None = None
    """Lowercase-hex SHA-256 of the file's bytes, 64 chars (``FILE`` only)."""
    bytes: int | None = None
    """Hashed file size in bytes (``FILE`` only)."""
    name: str | None = None
    """Human-readable model name (``BUILT_IN`` only)."""


@dataclass(frozen=True)
class SystemDescription:
    """A reproducibility summary of a :class:`BuiltSystem`'s frozen force
    model plus the kernel-identity manifest it captured.

    Every field is populated — nothing is defaulted. Names the citable
    menu (tier, frame, GR, perturbers, BPC, kernel hashes) so a run can
    be reproduced and audited.
    """

    force_model: ForceModelTier
    """The frozen force-model tier."""
    frame: Frame
    """The frozen integration/output frame."""
    encounter_timescale_divisor: float
    """The frozen encounter-timescale divisor."""
    relativistic: bool
    """Whether the post-Newtonian (EIH) relativistic correction is enabled."""
    asteroids: bool
    """Whether the 16 asteroid perturbers are included."""
    has_bpc: bool
    """Whether a BPC (body-fixed rotation) kernel is loaded."""
    perturber_origins: list[int]
    """NAIF ids of the perturbing bodies included in the force model."""
    kernels: list[KernelRecord]
    """The captured kernel-identity manifest, one record per loaded file."""

    @classmethod
    def _from_raw(cls, raw: Any) -> SystemDescription:
        """Build from the flat dict the ``_empyrean_rs`` binding returns."""
        kernels = [
            KernelRecord(
                kind=KernelKind(k["kind"]),
                provenance=KernelProvenance(k["provenance"]),
                path=k["path"],
                sha256=k["sha256"],
                bytes=None if k["bytes"] is None else int(k["bytes"]),
                name=k["name"],
            )
            for k in raw["kernels"]
        ]
        return cls(
            force_model=_INT_TO_FORCE_MODEL[int(raw["force_model"])],
            frame=int_to_frame(int(raw["frame"])),
            encounter_timescale_divisor=float(raw["encounter_timescale_divisor"]),
            relativistic=bool(raw["relativistic"]),
            asteroids=bool(raw["asteroids"]),
            has_bpc=bool(raw["has_bpc"]),
            perturber_origins=[int(x) for x in raw["perturber_origins"]],
            kernels=kernels,
        )


# ── Input normalization ───────────────────────────────────────


def _resolve_force_model(force_model: ForceModelTier | str | int) -> ForceModelTier:
    """Normalize a force-model argument to a :class:`ForceModelTier`."""
    if isinstance(force_model, ForceModelTier):
        return force_model
    if isinstance(force_model, str):
        return ForceModelTier(force_model.lower())
    if isinstance(force_model, int):
        tier = _INT_TO_FORCE_MODEL.get(force_model)
        if tier is None:
            raise ValueError(f"unknown force model tier code: {force_model}")
        return tier
    raise TypeError(f"force_model must be ForceModelTier, str, or int, got {type(force_model)}")


def _resolve_frame(frame: Frame | str | int) -> Frame:
    """Normalize a frame argument to a :class:`Frame` (round-tripped
    through the shared int table so it can't desync from the propagate /
    ephemeris FFI boundary)."""
    return int_to_frame(frame_to_int(frame))


# ── The handle ────────────────────────────────────────────────


class BuiltSystem:
    """A reusable pre-built force-model handle.

    Construct with :func:`build_system` (or directly). See the module
    docstring for the identity guard and the cross-thread reuse story.

    Parameters
    ----------
    force_model : ForceModelTier | str | int
        Force-model tier to freeze. Default :attr:`ForceModelTier.STANDARD`.
    frame : Frame | str | int
        Frame to freeze. Default :attr:`Frame.ECLIPTICJ2000`. For
        :meth:`generate_ephemeris` the frame must be
        :attr:`Frame.ECLIPTICJ2000` (the ephemeris integration frame).
    encounter_timescale_divisor : float, optional
        Encounter-timescale divisor to freeze into the key. ``None``
        (default) freezes the engine default.
    """

    def __init__(
        self,
        force_model: ForceModelTier | str | int = ForceModelTier.STANDARD,
        frame: Frame | str | int = Frame.ECLIPTICJ2000,
        encounter_timescale_divisor: float | None = None,
    ) -> None:
        from empyrean._empyrean_rs import BuiltSystem as _RsBuiltSystem

        self._force_model: ForceModelTier = _resolve_force_model(force_model)
        self._frame: Frame = _resolve_frame(frame)
        fm_int = _FORCE_MODEL_TO_INT[self._force_model]
        frame_int = frame_to_int(self._frame)
        divisor = 0.0 if encounter_timescale_divisor is None else float(encounter_timescale_divisor)
        self._rs: Any = _RsBuiltSystem(fm_int, frame_int, divisor)
        # The engine resolves the 0.0 sentinel to its default; read it back
        # so the recorded key reflects the actual frozen divisor.
        self._encounter_timescale_divisor: float = float(self._rs.encounter_timescale_divisor)

    @classmethod
    def _adopt(cls, rs: Any) -> BuiltSystem:
        """Wrap an already-assembled native handle, reading the frozen key
        back off it rather than restating it here.

        Used by :func:`od_system`, whose whole point is that the recipe
        lives in the engine and not in a second copy on this side.
        """
        self = cls.__new__(cls)
        self._rs = rs
        self._force_model = _INT_TO_FORCE_MODEL[int(rs.force_model)]
        self._frame = int_to_frame(int(rs.frame))
        self._encounter_timescale_divisor = float(rs.encounter_timescale_divisor)
        return self

    # ── Frozen key (read-only) ────────────────────────────────

    @property
    def force_model(self) -> ForceModelTier:
        """The frozen force-model tier."""
        return self._force_model

    @property
    def frame(self) -> Frame:
        """The frozen integration/output frame."""
        return self._frame

    @property
    def encounter_timescale_divisor(self) -> float:
        """The frozen encounter-timescale divisor (engine default resolved)."""
        return self._encounter_timescale_divisor

    # ── Forward models ────────────────────────────────────────

    def propagate(
        self,
        orbits: AnyOrbits,
        epochs: Epochs,
        config: PropagationConfig | None = None,
        *,
        tagged_covariance: bool = False,
        thrust_arcs: Sequence[ThrustParams | None] | None = None,
    ) -> PropagationResult:
        """Propagate orbits to target epochs through the frozen force model.

        Identical to :func:`empyrean.propagate` but reuses the pre-built
        force model. When ``config`` is omitted a default is built from
        this handle's frozen ``force_model`` / ``frame`` so the call
        always matches the key. If you pass an explicit ``config`` whose
        force model or frame diverges from the frozen key, the identity
        guard raises — it never silently rebuilds. The result is
        bit-identical to the one-shot on the matching key.

        Parameters
        ----------
        orbits, epochs, config, tagged_covariance, thrust_arcs
            As :func:`empyrean.propagate`. ``config`` defaults to
            ``PropagationConfig(force_model=self.force_model,
            frame=self.frame)``.

        Returns
        -------
        PropagationResult
        """
        from empyrean.coordinates.epoch import _require_epochs
        from empyrean.propagation.propagate import propagate as _propagate_fn

        # Validated here rather than only in `propagate` so the refusal
        # names the entry point the caller actually used.
        _require_epochs(epochs, "BuiltSystem.propagate() epochs")

        if config is None:
            config = PropagationConfig(force_model=self._force_model, frame=self._frame)
        return _propagate_fn(
            orbits,
            epochs,
            config,
            tagged_covariance=tagged_covariance,
            thrust_arcs=thrust_arcs,
            _builtsystem=self._rs,
        )

    def generate_ephemeris(
        self,
        orbits: AnyOrbits,
        observers: Observers,
        config: EphemerisConfig | None = None,
    ) -> EphemerisResult:
        """Generate predicted ephemeris through the frozen force model.

        Identical to :func:`empyrean.generate_ephemeris` but reuses the
        pre-built force model. The ephemeris pipeline integrates in
        EclipticJ2000, so this handle must be frozen at
        :attr:`Frame.ECLIPTICJ2000` (the default) and the engine-default
        divisor; a handle frozen otherwise is rejected loudly by the
        identity guard rather than served under the wrong dynamics.

        Parameters
        ----------
        orbits, observers, config
            As :func:`empyrean.generate_ephemeris`. ``config`` defaults to
            an :class:`EphemerisConfig` carrying this handle's frozen
            ``force_model`` / ``frame``.

        Returns
        -------
        EphemerisResult
        """
        from empyrean.ephemeris.generate import generate_ephemeris as _generate_ephemeris_fn
        from empyrean.ephemeris.result import EphemerisConfig

        if config is None:
            config = EphemerisConfig(
                propagation=PropagationConfig(force_model=self._force_model, frame=self._frame)
            )
        return _generate_ephemeris_fn(orbits, observers, config, _builtsystem=self._rs)

    # ── Orbit determination ───────────────────────────────────
    #
    # The three verbs mirror the module-level :func:`empyrean.determine` /
    # :func:`empyrean.evaluate` / :func:`empyrean.refine` argument for
    # argument. Build the handle with :func:`od_system` — a fit picks its
    # own integration frame (EclipticJ2000) and encounter-timescale
    # divisor, and a handle frozen at, say, ``frame="icrf"`` is refused
    # here rather than served under dynamics the fit did not ask for.
    #
    # The guard refuses; it does not degrade. A caller who named this
    # surface asked to reuse a *specific* frozen force model, so fitting
    # under a differently-assembled one while reporting success is exactly
    # the silent substitution this package will not make.
    #
    # ``ODConfig.auto_force_model`` is refused here, not tolerated: it lets
    # the fit re-pick its own tier part-way through, which no frozen handle
    # can follow, so the engine would drop the handle mid-fit and reassemble
    # per solve while the call still returned a result. All three verbs
    # raise instead, naming the field. Clear it and freeze the handle at the
    # tier you want, or use the module-level :func:`empyrean.determine`,
    # which is free to choose the tier.

    def determine(
        self,
        observations: Any,
        initial_orbits: Any = None,
        *,
        radar: Any = None,
        config: Any = None,
    ) -> Any:
        """Run the full OD pipeline over every object, through the frozen
        force model.

        Identical to :func:`empyrean.determine`, which see for the batch
        semantics, the seeding rules, and the return shape. Results are
        bit-identical to that call on a matching handle.
        """
        from empyrean.od.determine import determine as _determine_fn

        return _determine_fn(
            observations,
            initial_orbits,
            radar=radar,
            config=config,
            _builtsystem=self._rs,
        )

    def evaluate(
        self,
        orbit: Any,
        observations: Any,
        *,
        config: Any = None,
    ) -> Any:
        """Evaluate residuals for an orbit against observations, through
        the frozen force model.

        Identical to :func:`empyrean.evaluate`; no fitting is performed.
        """
        from empyrean.od.determine import evaluate as _evaluate_fn

        return _evaluate_fn(orbit, observations, config=config, _builtsystem=self._rs)

    def refine(
        self,
        orbit: Any,
        observations: Any,
        *,
        config: Any = None,
    ) -> Any:
        """Refine an orbit with new observations using a Bayesian prior,
        through the frozen force model.

        Identical to :func:`empyrean.refine`. This is the loop the whole
        surface exists for: assemble one handle, refine many times through
        it.
        """
        from empyrean.od.determine import refine as _refine_fn

        return _refine_fn(orbit, observations, config=config, _builtsystem=self._rs)

    # ── Provenance ────────────────────────────────────────────

    def describe(self) -> SystemDescription:
        """Return a full reproducibility summary of the frozen force model
        plus the kernel manifest captured at construction.

        Every field of the returned :class:`SystemDescription` is
        populated — nothing is defaulted.
        """
        return SystemDescription._from_raw(self._rs.describe())

    def __repr__(self) -> str:
        return (
            f"BuiltSystem(force_model={self._force_model.value!r}, "
            f"frame={self._frame.value!r}, "
            f"encounter_timescale_divisor={self._encounter_timescale_divisor})"
        )


def build_system(
    force_model: ForceModelTier | str | int = ForceModelTier.STANDARD,
    frame: Frame | str | int = Frame.ECLIPTICJ2000,
    encounter_timescale_divisor: float | None = None,
) -> BuiltSystem:
    """Assemble a reusable :class:`BuiltSystem` force-model handle.

    Builds the force model once for ``(force_model, frame)`` and freezes
    ``encounter_timescale_divisor`` into its key, then reuse the returned
    handle across many :meth:`~BuiltSystem.propagate` /
    :meth:`~BuiltSystem.generate_ephemeris` calls. **Rebuild the handle
    after any** :func:`empyrean.initialize` / data (re)load — a stale
    handle is rejected loudly by every forward-model call, never silently
    reused.

    Parameters
    ----------
    force_model : ForceModelTier | str | int
        Force-model tier to freeze. Default :attr:`ForceModelTier.STANDARD`.
    frame : Frame | str | int
        Frame to freeze. Default :attr:`Frame.ECLIPTICJ2000`.
    encounter_timescale_divisor : float, optional
        Encounter-timescale divisor to freeze. ``None`` (default) freezes
        the engine default.

    Returns
    -------
    BuiltSystem

    Examples
    --------
    >>> system = empyrean.build_system(force_model="standard", frame="icrf")
    >>> result = system.propagate(orbits, epochs)  # reuses the force model
    >>> desc = system.describe()  # provenance: tier, GR, perturbers, kernels
    """
    return BuiltSystem(
        force_model=force_model,
        frame=frame,
        encounter_timescale_divisor=encounter_timescale_divisor,
    )


def od_system(force_model: ForceModelTier | str | int = ForceModelTier.STANDARD) -> BuiltSystem:
    """Assemble a :class:`BuiltSystem` keyed for the orbit-determination
    entry points.

    :func:`build_system` asks for a frame and an encounter-timescale
    divisor. An OD fit picks both itself — :class:`~empyrean.ODConfig`
    exposes no knob for either — so naming them at the call site is an
    invitation to name them wrong, and the wrong frame is the likely
    mistake: OD integrates in :attr:`Frame.ECLIPTICJ2000`, while
    propagation examples routinely freeze ``"icrf"``. This function
    freezes the recipe a fit actually runs under, so the handle it returns
    is accepted by the OD identity guard as long as ``force_model``
    matches the config's.

    Parameters
    ----------
    force_model : ForceModelTier | str | int
        Force-model tier to freeze. Must equal the ``force_model`` on the
        :class:`~empyrean.ODConfig` you pass to the fit. Default
        :attr:`ForceModelTier.STANDARD`, which is ``ODConfig``'s default.

    Returns
    -------
    BuiltSystem
        An ordinary handle: ``describe``-able, shareable across threads,
        and usable for propagation too when the config asks for the same
        frame.

    Examples
    --------
    >>> system = empyrean.od_system()
    >>> for arc in arcs:  # doctest: +SKIP
    ...     fit = system.refine(orbit, arc)  # one force-model assembly total
    """
    from empyrean._empyrean_rs import BuiltSystem as _RsBuiltSystem

    tier = _resolve_force_model(force_model)
    return BuiltSystem._adopt(_RsBuiltSystem.for_od(_FORCE_MODEL_TO_INT[tier]))
