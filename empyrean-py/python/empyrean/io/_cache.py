"""Default cache directory resolution for external queries.

Every network query (``query_sbdb``, ``query_horizons``,
``query_horizons_vectors``, ``query_observations``) caches its JSON
response under
``~/.empyrean/cache/<service>/`` by default so repeated lookups for the
same object don't hit JPL / MPC again.

Override the base directory globally with the ``EMPYREAN_CACHE_DIR``
environment variable; pass ``cache_dir=False`` (or any falsy non-None
value) to a query function to disable caching entirely; pass an explicit
path to override on a per-call basis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

# Sentinel for "I want the default cache" — using None makes that the
# natural default in function signatures.
_USE_DEFAULT = object()


def resolve_cache_dir(
    cache_dir: str | Path | bool | None,
    service: str,
) -> str | None:
    """Resolve a user-supplied ``cache_dir`` argument to an actual path.

    Parameters
    ----------
    cache_dir : str | Path | bool | None
        - ``None``: use the default location
          (``$EMPYREAN_CACHE_DIR/<service>`` or ``~/.empyrean/cache/<service>``).
        - ``False`` (or any other falsy value): disable caching;
          returns ``None`` so the C ABI sees no cache pointer.
        - ``str`` / ``Path``: use this path verbatim.
    service : str
        Subdirectory name under the cache root (e.g. ``"sbdb"``,
        ``"horizons"``, ``"mpc"``).

    Returns
    -------
    str or None
        Path string for the underlying C ABI call, or ``None`` to skip.
    """
    if cache_dir is False:
        return None
    if cache_dir is None:
        base = os.environ.get("EMPYREAN_CACHE_DIR")
        if base is None:
            base = str(Path.home() / ".empyrean" / "cache")
        path = Path(base) / service
    else:
        # ``cache_dir is False`` and ``None`` are handled above, leaving
        # ``str | Path`` as the supported inputs. The identity check on
        # ``False`` does not narrow ``bool`` away (``bool`` subclasses
        # ``int``), so mypy still sees a residual ``Literal[True]``; the
        # explicit ``cast`` documents the intended runtime contract
        # without altering it.
        path = Path(cast("str | Path", cache_dir))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
