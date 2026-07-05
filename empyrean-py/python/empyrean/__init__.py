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
from empyrean.math import MixtureComponent, eigenvector_max_6x6, split_gaussian
from empyrean.observers.observers import Observers
from empyrean.observers.state import get_observer_states
from empyrean.od.ades_observations import ADESObservations
from empyrean.od.determine import determine, evaluate, read_ades, refine
from empyrean.od.radar_observations import ADESRadarObservations
from empyrean.od.residuals import (
    AcceptabilityReport,
    ObservationResults,
    ResidualSummary,
    StationBiases,
)
from empyrean.od.result import (
    AcceptabilityThresholds,
    AutoEscalationPolicy,
    CovarianceRepresentation,
    DebiasingConfig,
    DebiasingResolution,
    DetermineResult,
    EvaluateResult,
    IODConfig,
    ODConfig,
    OriginPolicy,
    OriginPolicyMode,
    OutputEpoch,
    OutputEpochMode,
    RejectionConfig,
    RejectionKind,
    SigmaPolicy,
    SolveForParams,
    StationRaDecConfig,
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
from empyrean.orbits.thrust import (
    ConstantRTN,
    InertialFixed,
    SteeringLaw,
    ThrustArc,
    ThrustParams,
    VelocityTangent,
)
from empyrean.propagation.config import (
    AdvancedIntegratorConfig,
    DiagnosticsConfig,
    ForceModelTier,
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
    cache.mkdir(parents=True, exist_ok=True)

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
) -> None:
    """Initialize empyrean with SPICE kernel data.

    On first call, loads ephemeris data into a global context. Subsequent
    calls are no-ops.

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
