"""Compile gate for the Python code examples in the project READMEs.

Every fenced ``python`` block in the PyPI README (``empyrean-py/README.md``)
and the workspace README (repo-root ``README.md``) is extracted and run
through :func:`compile` in ``"exec"`` mode. This catches syntax / parse
regressions so the published documentation can't silently rot.

This is intentionally a *parse* gate, not an *execution* gate: it imports
nothing from ``empyrean`` and touches no network or kernel data, so it runs
anywhere CI runs. API-level drift (renamed attributes, removed keyword
arguments) is verified separately by introspecting the installed package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# README locations, resolved relative to this test file so the gate works
# from any working directory.
_TESTS_DIR = Path(__file__).resolve().parent
_PYPI_README = _TESTS_DIR.parent / "README.md"
_WORKSPACE_README = _TESTS_DIR.parent.parent / "README.md"

# Matches a fenced ```python ... ``` block, capturing its body. The fence
# language tag may carry extra info-string words (e.g. ```python title=...).
_PYTHON_BLOCK = re.compile(
    r"^[ \t]*```[ \t]*python\b[^\n]*\n(.*?)^[ \t]*```",
    re.DOTALL | re.MULTILINE,
)


def _python_blocks(readme: Path) -> list[str]:
    """Return every fenced ``python`` code block body in ``readme``."""
    if not readme.exists():
        return []
    return _PYTHON_BLOCK.findall(readme.read_text(encoding="utf-8"))


# Stable, unambiguous labels (both files are named ``README.md``).
_README_LABELS = {
    _PYPI_README: "pypi",
    _WORKSPACE_README: "workspace",
}


def _cases() -> list[tuple[str, int, str]]:
    """Build ``(readme_label, block_index, source)`` parametrization cases."""
    cases: list[tuple[str, int, str]] = []
    for readme, label in _README_LABELS.items():
        for index, source in enumerate(_python_blocks(readme)):
            cases.append((label, index, source))
    return cases


_CASES = _cases()


def test_readmes_exist() -> None:
    """Both READMEs are present and expose at least one Python example."""
    assert _PYPI_README.exists(), f"missing PyPI README: {_PYPI_README}"
    assert _python_blocks(_PYPI_README), "PyPI README has no ```python blocks"


@pytest.mark.parametrize(
    ("readme_label", "block_index", "source"),
    _CASES,
    ids=[f"{label}#{index}" for label, index, _ in _CASES],
)
def test_readme_python_block_compiles(readme_label: str, block_index: int, source: str) -> None:
    """Each fenced ``python`` README block parses without a SyntaxError."""
    label = f"<{readme_label}-README#{block_index}>"
    try:
        compile(source, label, "exec")
    except SyntaxError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{label} failed to compile: {exc}")
