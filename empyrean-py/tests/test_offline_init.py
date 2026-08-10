"""``initialize(refresh=False)`` is a strict-offline context construction.

``refresh`` mirrors the Rust wrapper's ``DataDirOptions.refresh`` and the
C ABI's ``EmpyreanDataDirOptions.refresh``. ``False`` means: resolve the
kernel set from the data directory alone, download nothing, and fail —
naming every absent file — if anything is missing. There is no
try-the-network-and-tolerate path and no degrade-to-a-lower-tier path.

Every test here runs in a **subprocess**. The context is process-global
and built exactly once, so a failed initialization inside the test
process would either be masked by the session fixture's existing context
or poison it for the rest of the suite.
"""

import inspect
import json
import os
import subprocess
import sys
import textwrap

import empyrean
import pytest


def _run(body: str, *, env: dict | None = None) -> dict:
    """Run `body` in a fresh interpreter; it prints one JSON object."""
    full_env = dict(os.environ)
    # Never let an ambient data directory make an "empty directory" test
    # accidentally find real kernels.
    full_env.pop("EMPYREAN_DATA_DIR", None)
    full_env.pop("EMPYREAN_OFFLINE", None)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=600,
        check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    assert lines, (
        f"subprocess produced no JSON result\n"
        f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(lines[-1])


# ── Signature parity ──────────────────────────────────────────────────


def test_initialize_exposes_refresh_defaulting_to_true():
    """`refresh` is a keyword-only parameter defaulting to today's
    behaviour, so every existing call site is unaffected."""
    sig = inspect.signature(empyrean.initialize)
    assert "refresh" in sig.parameters, "initialize must expose the offline switch"
    param = sig.parameters["refresh"]
    assert param.default is True, "the default must stay the network-enabled path"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


# ── The offline path ──────────────────────────────────────────────────


def test_offline_init_fails_loudly_and_names_the_missing_files(tmp_path):
    """An empty data directory under ``refresh=False`` must fail with the
    full list of absent files, as a structured attribute — not as text a
    caller has to split on a separator a filename may itself contain."""
    result = _run(f"""
        import json
        import empyrean
        out = {{}}
        try:
            empyrean.initialize(data_dir={str(tmp_path)!r}, refresh=False)
            out["outcome"] = "unexpected-success"
        except FileNotFoundError as e:
            out["outcome"] = "FileNotFoundError"
            out["message"] = str(e)
            out["missing"] = list(getattr(e, "missing_data_files", []))
        except Exception as e:
            out["outcome"] = type(e).__name__
            out["message"] = str(e)
            out["missing"] = list(getattr(e, "missing_data_files", []))
        print(json.dumps(out))
    """)

    assert result["outcome"] == "FileNotFoundError", (
        f"strict offline against an empty directory must raise "
        f"FileNotFoundError, got {result['outcome']}: {result.get('message')}"
    )
    missing = result["missing"]
    assert missing, "the absent files must ride the exception as a real list"
    assert all(isinstance(f, str) and f for f in missing)
    # The list is the remedy: it has to be complete, not the first failure.
    assert "de440.bsp" in missing
    assert "gm_de440.tpc" in missing
    # And it must agree with the message, so neither channel is stale.
    for name in missing:
        assert name in result["message"], f"{name} is in the list but not the message"


def test_offline_init_downloads_nothing(tmp_path):
    """`refresh=False` must leave the data directory exactly as it found
    it. A partial download followed by a failure would be the worst of
    both worlds."""
    result = _run(f"""
        import json, os
        import empyrean
        d = {str(tmp_path)!r}
        before = sorted(os.listdir(d))
        try:
            empyrean.initialize(data_dir=d, refresh=False)
            outcome = "unexpected-success"
        except Exception as e:
            outcome = type(e).__name__
        print(json.dumps({{
            "outcome": outcome,
            "before": before,
            "after": sorted(os.listdir(d)),
        }}))
    """)
    assert result["outcome"] == "FileNotFoundError"
    assert result["after"] == result["before"] == [], (
        "strict offline wrote into the data directory; it must not touch the network"
    )


def test_offline_init_succeeds_against_a_complete_directory():
    """The flag gates the *network*, not the load: pointed at a directory
    that already has the kernels, ``refresh=False`` builds a working
    context and the engine runs."""
    data_dir = os.environ.get("EMPYREAN_DATA_DIR")
    if not data_dir:
        # Mirror the wrapper's own default resolution rather than guessing.
        try:
            data_dir = empyrean.default_data_dir()
        except Exception:  # noqa: BLE001 - no resolvable data dir at all
            pytest.skip("no resolvable data directory")
    if not os.path.isdir(data_dir) or not os.path.exists(os.path.join(data_dir, "de440.bsp")):
        pytest.skip(f"data directory {data_dir} is not populated")

    result = _run(
        f"""
        import json
        import numpy as np
        import empyrean
        out = {{}}
        try:
            empyrean.initialize(data_dir={str(data_dir)!r}, refresh=False)
            obs = empyrean.Observers.from_code("500", [60000.0])
            out["outcome"] = "ok"
            out["rows"] = len(obs)
            out["finite"] = bool(np.isfinite(obs.coordinates.x.to_numpy()).all())
        except Exception as e:
            out["outcome"] = type(e).__name__
            out["message"] = str(e)
        print(json.dumps(out))
    """,
        env={"EMPYREAN_DATA_DIR": data_dir},
    )
    assert result["outcome"] == "ok", (
        f"offline init against a populated directory must succeed: {result.get('message')}"
    )
    assert result["rows"] == 1
    assert result["finite"]


def test_explicit_kernel_paths_are_already_offline(tmp_path):
    """The two-explicit-paths branch loads exactly what it is handed and
    never reaches the network, so ``refresh=False`` is honoured there
    rather than being quietly inapplicable — it just has nothing to
    forbid. Pinned so the branch cannot start downloading later."""
    result = _run(f"""
        import json
        import empyrean
        out = {{}}
        try:
            empyrean.initialize(
                de440_path={str(tmp_path / "nope-de440.bsp")!r},
                gm_path={str(tmp_path / "nope-gm.tpc")!r},
                refresh=False,
            )
            out["outcome"] = "unexpected-success"
        except Exception as e:
            out["outcome"] = type(e).__name__
            out["message"] = str(e)
        import os
        out["dir"] = sorted(os.listdir({str(tmp_path)!r}))
        print(json.dumps(out))
    """)
    assert result["outcome"] != "unexpected-success", "loading kernels that do not exist must fail"
    assert result["dir"] == [], "the explicit-paths branch must not fetch anything"
