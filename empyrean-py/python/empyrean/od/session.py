"""Stateful orbit-determination session.

Wraps the Rust extension's ``Session`` PyClass with quivr-typed
inputs / outputs. The typical workflow is *fit → look at residuals →
mask one bad night → re-fit → compare χ²*::

    import empyrean
    from empyrean import Session, query_observations

    empyrean.initialize()

    # Three input forms — pick whichever fits your pipeline:
    sess1 = Session("apophis_2004_2021.psv")  # path
    sess2 = Session(open("apophis.psv").read())  # PSV string
    obs = query_observations(["99942"])
    sess3 = Session(obs)  # ADESObservations

    fit0 = sess3.refine()  # initial IOD → DC
    print(f"χ²_red = {fit0.summary.reduced_chi2:.2f}")

    # Drop the 7th observation and refit:
    sess3.mask(7)
    fit1 = sess3.refine()
    diff = sess3.diff(0)
    print(f"Δχ²_red = {diff.reduced_chi2_delta:+.3f}")

The session's history is kept on the Rust side; each call to
:meth:`refine` pushes a new fit onto the history list and returns it
as a :class:`DetermineResult`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from empyrean.od.ades_observations import ADESObservations
from empyrean.od.result import DetermineResult, ODConfig


class _RustSessionProtocol(Protocol):
    """Typed view of the ``_empyrean_rs.Session`` PyClass surface.

    The compiled pyo3 extension ships no type information, so its
    methods would otherwise resolve to ``Any``. This protocol pins the
    exact subset used here to concrete return types, mirroring the
    Rust-side signatures (``usize`` → :class:`int`, the ``refine`` /
    ``history`` / ``diff`` dicts → ``dict[str, Any]``).
    """

    def n_observations(self) -> int: ...

    def n_masked(self) -> int: ...

    def n_active(self) -> int: ...

    def mask(self, idx: int) -> None: ...

    def unmask(self, idx: int) -> None: ...

    def unmask_all(self) -> None: ...

    def is_masked(self, idx: int) -> bool: ...

    def refine(self) -> dict[str, Any]: ...

    def history_len(self) -> int: ...

    def history(self, idx: int) -> dict[str, Any]: ...

    def diff(self, prior_idx: int) -> dict[str, Any]: ...


@dataclass
class SessionDiff:
    """Pairwise diagnostic between two fits in the same session."""

    reduced_chi2_delta: float
    """Δ reduced χ² (positive ⇒ current fit is worse)."""
    iterations_delta: int
    """Δ iteration count between current and prior fits."""
    n_observations_delta: int
    """Δ number of observations used (negative when masked between refines)."""
    update_norm_current: float
    """Final update-norm convergence metric on the current fit."""
    update_norm_prior: float
    """Final update-norm convergence metric on the prior fit."""


SessionInput = str | Path | ADESObservations


class Session:
    """Stateful orbit-determination handle."""

    _inner: _RustSessionProtocol

    def __init__(
        self,
        source: SessionInput,
        config: ODConfig | None = None,
    ):
        """Construct a session from a path, PSV content, or quivr table.

        Parameters
        ----------
        source : str | pathlib.Path | ADESObservations
            One of:

            - A filesystem path to an ADES PSV / MPC80 file
              (``str`` or :class:`pathlib.Path`).
            - The PSV content as a string.
            - An :class:`ADESObservations` quivr table — typically the
              return value of :func:`empyrean.query_observations` or
              :func:`empyrean.read_ades`. No PSV round-trip.
        config : ODConfig, optional
            Full nested configuration mirroring scott's ``ODConfig``.
            Defaults to :class:`ODConfig` defaults (Standard force model,
            VFC17 + EFCC2020 weighting, AUTO solve-for, adaptive
            rejection on).
        """
        from empyrean._empyrean_rs import Session as _RustSession
        from empyrean.od.result import ODConfig

        if config is None:
            config = ODConfig()
        config_dict = config._to_wire_dict()

        if isinstance(source, ADESObservations):
            # Already parsed — feed the flat-dict shape directly to the
            # Rust extension, bypassing PSV parsing.
            from empyrean.od.determine import _obs_to_dict

            self._inner = _RustSession.from_observations_dict(_obs_to_dict(source), config_dict)
        elif isinstance(source, Path):
            self._inner = _RustSession(str(source), config_dict)
        elif isinstance(source, str):
            self._inner = _RustSession(source, config_dict)
        else:
            raise TypeError(
                "Session source must be str, pathlib.Path, or "
                f"ADESObservations; got {type(source).__name__}"
            )

    @classmethod
    def from_ades(
        cls,
        path: str | Path,
        config: ODConfig | None = None,
    ) -> "Session":
        """Construct a session from an ADES PSV / MPC80 file path."""
        return cls(path, config)

    @classmethod
    def from_observations(
        cls,
        observations: ADESObservations,
        config: ODConfig | None = None,
    ) -> "Session":
        """Construct a session from an :class:`ADESObservations` table."""
        return cls(observations, config)

    @property
    def n_observations(self) -> int:
        """Total observations in the session, masked or not."""
        return self._inner.n_observations()

    @property
    def n_masked(self) -> int:
        """Number of currently masked observations."""
        return self._inner.n_masked()

    @property
    def n_active(self) -> int:
        """Number of observations active in the next refine."""
        return self._inner.n_active()

    def mask(self, idx: int) -> None:
        """Mask the observation at the given index."""
        self._inner.mask(idx)

    def unmask(self, idx: int) -> None:
        """Unmask the observation at the given index."""
        self._inner.unmask(idx)

    def unmask_all(self) -> None:
        """Clear all masks."""
        self._inner.unmask_all()

    def is_masked(self, idx: int) -> bool:
        """Whether the observation at ``idx`` is masked."""
        return self._inner.is_masked(idx)

    def refine(self) -> DetermineResult:
        """Run an OD refine using the current mask state.

        On the first call runs the full IOD → DC pipeline. On
        subsequent calls reuses the previously-fit orbit as the IOD
        seed (skipping the IOD step). Pushes the new fit onto the
        session's history.
        """
        return _result_dict_to_determine(self._inner.refine())

    @property
    def history_len(self) -> int:
        """Number of fits stored in the session history."""
        return self._inner.history_len()

    def history(self, idx: int) -> DetermineResult:
        """Retrieve the i-th history entry."""
        return _result_dict_to_determine(self._inner.history(idx))

    def diff(self, prior_idx: int) -> SessionDiff:
        """Diff the current fit against the ``prior_idx``-th history entry."""
        d = self._inner.diff(prior_idx)
        return SessionDiff(
            reduced_chi2_delta=d["reduced_chi2_delta"],
            iterations_delta=d["iterations_delta"],
            n_observations_delta=d["n_observations_delta"],
            update_norm_current=d["update_norm_current"],
            update_norm_prior=d["update_norm_prior"],
        )


def _result_dict_to_determine(d: dict[str, Any]) -> DetermineResult:
    """Map the flat OD-result dict from `_empyrean_rs.Session.refine`
    (and `Session.history`) into a quivr :class:`DetermineResult`.

    Delegates to :func:`empyrean.od.determine._build_determine_result`
    so the per-row schema and acceptability surface are identical
    across the determine / refine / session paths.
    """
    from empyrean.od.determine import _build_determine_result

    return _build_determine_result(d)
