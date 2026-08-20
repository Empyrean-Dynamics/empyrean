"""Empyrean: high-fidelity orbital mechanics for Python."""

import pathlib

from empyrean import io

# ── Type re-exports (organized by subpackage) ────────────────
from empyrean.coordinates.coordinates import (
    CartesianCoordinates,
    CometaryCoordinates,
    KeplerianCoordinates,
    SphericalCoordinates,
)
from empyrean.coordinates.covariance import (
    CartesianCovariance,
    CometaryCovariance,
    KeplerianCovariance,
    SphericalCovariance,
)
from empyrean.coordinates.enums import Frame, Origin
from empyrean.coordinates.epoch import Epochs, TimeScale

# ── Function re-exports (organized by subpackage) ────────────
from empyrean.coordinates.transform import transform_coordinates
from empyrean.ephemeris.generate import generate_ephemeris
from empyrean.ephemeris.result import Ephemeris, EphemerisConfig, EphemerisResult
from empyrean.ephemeris.sensitivity import (
    SENSITIVITY_ROW_DEC,
    SENSITIVITY_ROW_RA,
    SENSITIVITY_ROW_RANGE,
    SENSITIVITY_ROW_VDEC,
    SENSITIVITY_ROW_VRA,
    SENSITIVITY_ROW_VRANGE,
    ObservationSensitivities,
    StateSensitivities,
)
from empyrean.impact import (
    BPlanes,
    ImpactProbabilities,
    compute_b_planes,
    compute_impact_probabilities,
)
from empyrean.io.horizons import query_horizons, query_horizons_vectors
from empyrean.io.observations import query_observations, query_radar
from empyrean.io.sbdb import query_sbdb

# NAME COLLISION, resolved by module path. This top-level
# ``MixtureComponent`` is the ``split_gaussian`` primitive at t0 (weight /
# mean / covariance, no basis tags). The AGM *read-back* component — the
# basis-tagged one — is ``empyrean.propagation.mixtures.MixtureComponent``
# and is deliberately NOT re-exported here: flattening both names would
# need one of them renamed away from the name the engine uses.
from empyrean.math import MixtureComponent, eigenvector_max_6x6, split_gaussian
from empyrean.observers.observers import Observers
from empyrean.observers.state import get_observer_states
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.determine import determine, evaluate, read_ades, refine
from empyrean.od.disposition import ParamDisposition
from empyrean.od.radar_observations import ADESRadarObservations
from empyrean.od.residuals import (
    AcceptabilityReport,
    FitSummary,
    ObservationResults,
    ResidualSummary,
    StationBiases,
)
from empyrean.od.result import (
    AcceptabilityThresholds,
    AutoEscalationPolicy,
    BandStat,
    CovarianceRepresentation,
    CovarianceTrust,
    DebiasingConfig,
    DebiasingResolution,
    DetermineFailure,
    DetermineResult,
    DetermineResults,
    EvaluateResult,
    GateRecord,
    IODConfig,
    ODConfig,
    OriginPolicy,
    OriginPolicyMode,
    OutputEpoch,
    OutputEpochMode,
    PhotometryConfig,
    PhotometryModel,
    PhotometryResult,
    RejectionConfig,
    RejectionKind,
    SigmaPolicy,
    SolvedCovariance,
    SolveFor,
    SolveForParams,
    StallDelivery,
    StationRaDecConfig,
    TrustGateEvent,
    WeightingConfig,
    WeightingLayer,
    WeightingLayerKind,
    WeightingPreset,
)
from empyrean.od.session import Session, SessionDiff
from empyrean.orbits.nongrav import NonGravParams
from empyrean.orbits.orbits import (
    CartesianOrbits,
    CometaryOrbits,
    KeplerianOrbits,
    SphericalOrbits,
)
from empyrean.orbits.photometry import PhotometricParams
from empyrean.orbits.srp import SRPParams
from empyrean.orbits.thrust import (
    ConstantRTN,
    InertialFixed,
    SteeringLaw,
    ThrustArc,
    ThrustParams,
    VelocityTangent,
)
from empyrean.orbits.wide_cross import WideCross
from empyrean.planning.plan import evaluate_plan
from empyrean.planning.result import (
    STAGE_POSTERIOR,
    STAGE_PRIOR,
    ObservatoryConfig,
    PlanCandidates,
    PlanEphemeris,
    PlanMetrics,
    PlannedObservation,
    PlannedObservationKind,
    PlanningConfig,
    PlanResult,
    RadarMode,
    RadarStation,
)
from empyrean.propagation.config import (
    AdvancedIntegratorConfig,
    Auto,
    DiagnosticsConfig,
    EphemerisOverlapPolicy,
    ForceModelTier,
    GaussianMixture,
    IntegratorChoice,
    MonteCarlo,
    OriginSwitchingConfig,
    PropagationConfig,
    SigmaPoint,
    UncertaintyMethod,
)
from empyrean.propagation.events import (
    AtmosphericEntries,
    AtmosphericExits,
    CaptureEnds,
    CaptureStarts,
    CloseApproachEnds,
    CloseApproachStarts,
    CovarianceRegimeChanges,
    EventConfig,
    Events,
    EventSummary,
    Impacts,
    Periapses,
    PossibleImpacts,
    ShadowEntries,
    ShadowExits,
)
from empyrean.propagation.mixtures import MixtureChains
from empyrean.propagation.propagate import propagate
from empyrean.propagation.result import PropagationResult
from empyrean.propagation.tagged_covariance import (
    CovarianceKind,
    CovarianceQuality,
    TaggedCovariance,
    TaggedCovariances,
    TargetFunctional,
)
from empyrean.states import get_states
from empyrean.system import (
    BuiltSystem,
    KernelKind,
    KernelProvenance,
    KernelRecord,
    SystemDescription,
    build_system,
    od_system,
)


def version_string() -> str:
    """Return the multi-line version report for the empyrean stack.

    Format::

        empyrean-core <ver>
        villeneuve    <ver>
        scott         <ver>
        nolan         <ver>

    Where each upstream version is the git-populated ``<tag>+<sha>``
    string baked into the cdylib at build time. Use this for build-
    provenance reporting in logs / crash dumps / `--version`-style
    output.

    Returns
    -------
    str
        Multi-line version report.
    """
    from empyrean._empyrean_rs import _version_string

    result: str = _version_string()
    return result


def versions() -> dict[str, str]:
    """Return per-crate versions of the empyrean stack.

    Returns
    -------
    dict[str, str]
        Mapping of crate name (``empyrean_core`` / ``villeneuve`` /
        ``scott`` / ``nolan``) to its version string. ``empyrean_core``
        is its semver from ``Cargo.toml``; the upstream physics crates
        carry git-populated ``<tag>+<sha>`` strings.
    """
    from empyrean._empyrean_rs import _versions

    core, villeneuve, scott, nolan = _versions()
    return {
        "empyrean_core": core,
        "villeneuve": villeneuve,
        "scott": scott,
        "nolan": nolan,
    }


def default_data_dir() -> pathlib.Path:
    """Return the OS-appropriate XDG data directory empyrean uses by default.

    Resolution order:

    1. ``EMPYREAN_DATA_DIR`` environment variable, if set.
    2. The OS XDG data location:

       - Linux: ``$XDG_DATA_HOME/empyrean/data/`` (default
         ``~/.local/share/empyrean/data/``)
       - macOS: ``~/Library/Application Support/empyrean/data/``
       - Windows: ``%APPDATA%\\empyrean\\data\\``

    Cheap to call — does not touch the filesystem.

    Returns
    -------
    pathlib.Path
        Path to the data directory.
    """
    from pathlib import Path

    from empyrean._empyrean_rs import _default_data_dir

    return Path(_default_data_dir())


def _bundled_gm_path() -> str:
    """Return the path to the gm_de440.tpc bundled inside the wheel."""
    from importlib.resources import files

    # `joinpath` on `Traversable` only accepts a single child segment per
    # call (despite the `MultiplexedPath` overload accepting varargs); chain
    # to compose the relative path portably.
    return str(files("empyrean").joinpath("_data").joinpath("gm_de440.tpc"))


def _discover_b612_data() -> dict[str, str]:
    """Detect B612 Foundation SPICE kernel pip packages and return paths.

    Returns a dict mapping a stable kernel name to the file path of
    every detected package. Empty dict if none are installed.
    """
    paths: dict[str, str] = {}
    try:
        import naif_de440

        paths["de440"] = naif_de440.de440
    except ImportError:
        pass
    try:
        import jpl_small_bodies_de441_n16

        paths["sb441_n16"] = jpl_small_bodies_de441_n16.de441_n16
    except ImportError:
        pass
    try:
        import naif_eop_high_prec

        paths["earth_high_prec"] = naif_eop_high_prec.eop_high_prec
    except ImportError:
        pass
    try:
        import naif_eop_historical

        paths["earth_historical"] = naif_eop_historical.eop_historical
    except ImportError:
        pass
    try:
        import naif_eop_predict

        paths["earth_predict"] = naif_eop_predict.eop_predict
    except ImportError:
        pass
    try:
        import mpc_obscodes

        paths["mpc_obscodes"] = mpc_obscodes.mpc_obscodes
    except ImportError:
        pass
    return paths


# Maps B612 kernel name → filename expected by villeneuve's DataManager.
# See villeneuve/src/data.rs for the authoritative filename list.
_B612_TO_VILLENEUVE_FILENAME = {
    "de440": "de440.bsp",
    "sb441_n16": "sb441-n16.bsp",
    "earth_high_prec": "earth_latest_high_prec.bpc",
    "earth_historical": "earth_620120_250826.bpc",
    "earth_predict": "earth_2025_250826_2125_predict.bpc",
    "mpc_obscodes": "obscodes_extended.json",
}


def _ensure_data_dir(cache: pathlib.Path) -> None:
    """Create the data directory, naming a broken path instead of ``Errno 17``.

    ``Path.mkdir(exist_ok=True)`` re-raises ``FileExistsError`` whenever the
    path exists but is not a directory — and ``mkdir(2)`` never follows a
    trailing symbolic link, so a data dir that is a link to nowhere (or to a
    file, or to itself) raises a bare ``[Errno 17] File exists`` naming
    nothing the user can act on. Re-check what is actually there and say so.

    Raises:
        NotADirectoryError: the path exists but does not resolve to a
            directory. The message names the path, and the link target when
            the path is a symbolic link.
    """
    import os

    try:
        cache.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if cache.is_dir():
            # Raced with another process that created it; nothing is wrong.
            return
        if cache.is_symlink():
            try:
                target = os.readlink(cache)
            except OSError as read_err:  # pragma: no cover - unreadable link
                target = f"<unreadable: {read_err}>"
            raise NotADirectoryError(
                f"the empyrean data directory {str(cache)!r} is a symbolic link to "
                f"{target!r} that does not resolve to a directory. Repoint or remove "
                f"that link, or set EMPYREAN_DATA_DIR to a directory that already "
                f"contains the kernels."
            ) from None
        raise NotADirectoryError(
            f"the empyrean data directory {str(cache)!r} exists but is not a "
            f"directory. Remove or replace that path, or set EMPYREAN_DATA_DIR to a "
            f"directory that already contains the kernels."
        ) from None


def _stage_b612_cache(b612: dict[str, str]) -> pathlib.Path:
    """Stage B612-provided kernel symlinks inside the platform data directory.

    Links each B612-provided kernel into villeneuve's XDG-compliant
    data directory (``~/Library/Application Support/empyrean/data/`` on
    macOS, ``~/.local/share/empyrean/data/`` on Linux, ``%APPDATA%\\empyrean\\data\\``
    on Windows) under the filename villeneuve expects, so the SPICE /
    asteroid / Earth-orientation kernels shipped by the B612 PyPI
    packages are reused without redownload.

    Linking *into* the canonical data directory (not a sibling
    ``b612-cache/``) keeps villeneuve and scott in agreement: villeneuve
    downloads anything missing — ``bias.dat`` is the practical case —
    next to the symlinks, and scott's catalog-debiasing loader
    (``DataManager::new().data_dir()``) finds the bias table at the same
    XDG default. Honors ``EMPYREAN_DATA_DIR`` via the same logic
    :func:`Context.from_data_dir(None) <Context.from_data_dir>` uses.

    Existing real files at a target path take precedence — only stale
    symlinks are replaced, so a user who already downloaded a fresh
    kernel does not have it silently swapped for the (possibly older)
    version that ships with a B612 release.

    Returns the data directory path.
    """
    from pathlib import Path

    from empyrean._empyrean_rs import _default_data_dir

    cache = Path(_default_data_dir())
    _ensure_data_dir(cache)

    def _link_if_safe(target: Path, link: Path) -> None:
        # Replace stale symlinks (e.g. when a B612 package updated and
        # the previous version was unlinked from site-packages) but
        # never overwrite a real file the user fetched themselves.
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            return
        link.symlink_to(target)

    for key, filename in _B612_TO_VILLENEUVE_FILENAME.items():
        if key not in b612:
            continue
        _link_if_safe(Path(b612[key]), cache / filename)

    # Bundled gm_de440.tpc (not available from B612)
    gm_src = Path(_bundled_gm_path())
    if gm_src.exists():
        _link_if_safe(gm_src, cache / "gm_de440.tpc")

    return cache


def initialize(
    *,
    data_dir: str | pathlib.Path | None = None,
    de440_path: str | pathlib.Path | None = None,
    gm_path: str | pathlib.Path | None = None,
    refresh: bool = True,
) -> None:
    """Initialize empyrean with SPICE kernel data.

    On first call, loads ephemeris data into a global context. Subsequent
    calls are no-ops — including their ``refresh``, so the first call in
    a process is the one that decides whether the network is reachable.

    If the B612 Foundation data packages (``naif-de440``,
    ``jpl-small-bodies-de441-n16``, ``naif-eop-high-prec``,
    ``naif-eop-historical``, ``naif-eop-predict``, ``mpc-obscodes``) are
    installed and no explicit paths are provided, empyrean stages a
    symlinked cache under the platform XDG data directory
    (``$XDG_DATA_HOME/empyrean/data/`` on Linux,
    ``~/Library/Application Support/empyrean/data/`` on macOS,
    ``%APPDATA%\\empyrean\\data\\`` on Windows; honors
    ``EMPYREAN_DATA_DIR``) and uses that as the data directory — zero
    network access required. Falls back to ``data_dir`` (default: the
    same XDG ``.../empyrean/data/`` location) plus :func:`download_data`
    otherwise.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing kernel files. Overrides B612 detection.
    de440_path : str, optional
        Explicit path to ``de440.bsp``. Overrides B612 detection.
    gm_path : str, optional
        Explicit path to ``gm_de440.tpc``.
    refresh : bool
        Whether initialization may reach the network. ``True``
        (default) downloads any required kernel that is missing and
        re-downloads any whose upstream copy moved. ``False`` is
        **strict offline**: kernels are resolved from the data directory
        alone and initialization fails, naming every absent file, if any
        is missing. There is no try-the-network-and-tolerate path and no
        degrade-to-a-lower-tier path.

        Passing both ``de440_path`` and ``gm_path`` loads exactly those
        two files and never reaches the network on either value, so
        ``refresh=False`` is already satisfied on that branch.

        Setting the environment variable ``EMPYREAN_OFFLINE=1`` acts as a
        **floor**: it downgrades ``refresh=True`` to ``False`` and says so
        on stderr. It can never turn ``False`` into ``True``, so an
        operator asserting "this machine must not reach the network"
        cannot have that reversed by a library call.

    Raises
    ------
    FileNotFoundError
        Under ``refresh=False`` when the data directory is missing a
        required kernel. The exception carries a ``missing_data_files``
        attribute — the list of absent filenames — so a caller can fetch
        or report exactly that set without re-parsing the message.
    RuntimeError
        Any other initialization failure.

    Examples
    --------
    >>> empyrean.initialize(refresh=False)  # air-gapped / reproducible run
    """
    from empyrean._empyrean_rs import _initialize

    if data_dir is None and de440_path is None:
        b612 = _discover_b612_data()
        if b612:
            data_dir = str(_stage_b612_cache(b612))

    _initialize(
        data_dir=None if data_dir is None else str(data_dir),
        de440_path=None if de440_path is None else str(de440_path),
        gm_path=None if gm_path is None else str(gm_path),
        refresh=refresh,
    )


def download_data(*, data_dir: str | pathlib.Path | None = None) -> str:
    """Provision a usable data directory with the required SPICE kernels.

    Provisions the OS-appropriate XDG data directory by default (see
    :func:`default_data_dir`); pass ``data_dir`` to target another. Idempotent:
    files already present are kept; only missing files are downloaded.

    If the B612 Foundation data packages (``naif-de440``,
    ``jpl-small-bodies-de441-n16``, ``naif-eop-high-prec``,
    ``naif-eop-historical``, ``naif-eop-predict``, ``mpc-obscodes``) are
    installed and no explicit ``data_dir`` is given, their kernels are staged
    from the installed wheels with **zero network access**, and only what they
    do not supply (e.g. ``bias.dat``) is downloaded.

    Parameters
    ----------
    data_dir : str, optional
        Target directory. Defaults to the value returned by
        :func:`default_data_dir` (honors ``EMPYREAN_DATA_DIR``).

    Returns
    -------
    str
        Path to the provisioned data directory.

    Raises
    ------
    RuntimeError
        If ``EMPYREAN_OFFLINE=1`` is set. That variable is a floor on the
        process, and it downgrades a context construction from "refresh"
        to "resolve what is already here" — but provisioning has no such
        second mode, because reaching the network *is* the call. So it
        refuses rather than ignoring the assertion, naming the variable.
        Build against an already-provisioned directory with
        :func:`initialize` and ``refresh=False`` instead, or unset the
        variable for the process that must provision.
    RuntimeError
        If a kernel fetch was attempted and failed — a 404 from an upstream
        that rotated or withdrew a pinned kernel, a refused connection, a
        mid-transfer failure. The message leads with ``"Data download
        failed: "`` and carries the request context (``GET <url>: ...``),
        so the kernel that could not be fetched is named by its URL. The
        remedy is connectivity, or — when the URL 404s — staging that file
        by hand into ``data_dir``, or moving to a release whose kernel pin
        is still served. Retrying this call does not help, and neither does
        local file repair: nothing is wrong on disk.
    """
    # Prefer installed B612 data packages — symlink the kernels they ship into
    # the data dir (no network) and let the engine fetch only the remainder.
    if data_dir is None:
        b612 = _discover_b612_data()
        if b612:
            data_dir = str(_stage_b612_cache(b612))

    from empyrean._empyrean_rs import _download_data

    # The binding takes Option<&str>; coerce an explicit pathlib.Path.
    result: str = _download_data(data_dir=None if data_dir is None else str(data_dir))
    return result
