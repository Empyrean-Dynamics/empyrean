"""Tests for the agentdocs generator (docs -> SKILL.md tail, references, llms.txt).

These are fast and self-contained: no kernels, no network. They import the
extension module directly from ``empyrean-py/docs/_ext`` rather than through
Sphinx, and they double as the docs-vs-package drift guard (``api_index``).
"""

import shutil
import sys
from pathlib import Path

import empyrean
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "empyrean-py" / "docs" / "_ext"))

import agentdocs  # noqa: E402  (path to _ext is set up just above)

DOCS = REPO / "empyrean-py" / "docs"
SKILL_DIR = REPO.joinpath("plugins", "empyrean", "skills", "empyrean")

STALE_MSG = (
    "plugins/empyrean/skills/empyrean is stale: run `make html` in empyrean-py/docs "
    "(or `python empyrean-py/docs/_ext/agentdocs.py --skill`) and include the result"
)

EXPECTED_ORDER = [
    "index",
    "about",
    "cookbook/propagation",
    "cookbook/impact-probability",
    "cookbook/orbit-determination",
    "cookbook/sensitivity",
    "cookbook/data-setup",
    "agents",
    "reference/frames-units",
    "reference/glossary",
    "reference/validation",
    "api",
]

FORBIDDEN = ("{func}", "{class}", "{doc}", "toctree")


def _titles():
    """Map every docname to its page title using only the public helpers."""
    return {
        docname: agentdocs.page_title((DOCS / (docname + ".md")).read_text(encoding="utf-8"))
        for docname in agentdocs.page_order(DOCS)
    }


def _skill(name="empyrean", desc="A short skill description.", body="Use `empyrean.propagate`."):
    """Build a minimal SKILL.md text (head + empty generated region) for head checks."""
    return (
        f'---\nname: {name}\ndescription: "{desc}"\n---\n\n'
        f"# empyrean\n\n{body}\n\n{agentdocs.START_MARKER}\n{agentdocs.END_MARKER}\n"
    )


# ── (a) strip_myst ───────────────────────────────────────────────────


def test_strip_myst_tilde_role_keeps_qualified_name():
    assert agentdocs.strip_myst("{func}`~empyrean.propagate`") == "`empyrean.propagate`"
    assert agentdocs.strip_myst(":class:`~empyrean.Session`") == "`empyrean.Session`"


def test_strip_myst_removes_toctree_block():
    text = "before\n\n```{toctree}\n:maxdepth: 2\n\nabout\napi\n```\n\nafter"
    out = agentdocs.strip_myst(text)
    assert "toctree" not in out
    assert "about\napi" not in out
    assert "before" in out and "after" in out


def test_strip_myst_note_becomes_blockquote():
    out = agentdocs.strip_myst("```{note}\nBe careful here.\n```")
    assert "**Note:**" in out
    assert out.strip().startswith(">")


def test_strip_myst_flattens_tab_set():
    text = "::::{tab-set}\n:::{tab-item} Python\ncode here\n:::\n::::"
    out = agentdocs.strip_myst(text)
    assert "**Python**" in out
    assert ":::" not in out
    assert "tab-set" not in out
    assert "code here" in out


def test_strip_myst_leaves_python_fence_untouched():
    fence = "```python\nx = 1  # {func}`not a role`\n```"
    assert agentdocs.strip_myst(fence) == fence


def test_strip_myst_idempotent_on_plain_markdown():
    plain = "# Title\n\nJust prose here.\n\n- item one\n- item two\n"
    once = agentdocs.strip_myst(plain)
    assert once == plain
    assert agentdocs.strip_myst(once) == once


# ── (b) page_order ───────────────────────────────────────────────────


def test_page_order_matches_toctree_and_files_exist():
    assert agentdocs.page_order(DOCS) == EXPECTED_ORDER
    for docname in EXPECTED_ORDER:
        assert (DOCS / (docname + ".md")).exists(), docname


# ── (c) api_index drift check ────────────────────────────────────────


def test_api_index_resolves_and_renders_propagate():
    index = agentdocs.api_index((DOCS / "api.md").read_text(encoding="utf-8"), empyrean)
    assert "## Pipeline entry points" in index
    prop = [line for line in index.splitlines() if line.startswith("`propagate(")]
    assert len(prop) == 1
    assert "orbits" in prop[0]
    assert "epochs" in prop[0]
    assert "*" in prop[0]
    assert "_builtsystem" not in index


# ── (d) check_skill_head ─────────────────────────────────────────────


def test_check_skill_head_passes_on_checked_in_skill():
    agentdocs.check_skill_head((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"), empyrean)


def test_skill_md_at_most_500_lines():
    assert (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8").count("\n") <= 500


def test_check_skill_head_rejects_wrong_name():
    with pytest.raises(ValueError):
        agentdocs.check_skill_head(_skill(name="not-empyrean"), empyrean)


def test_check_skill_head_rejects_overlong_description():
    with pytest.raises(ValueError):
        agentdocs.check_skill_head(_skill(desc="x" * 1100), empyrean)


def test_check_skill_head_rejects_unresolved_reference():
    with pytest.raises(ValueError):
        agentdocs.check_skill_head(_skill(body="Call `empyrean.does_not_exist_xyz`."), empyrean)


# ── (e) freshness ────────────────────────────────────────────────────


def test_skill_is_fresh(tmp_path):
    tmp_skill = tmp_path.joinpath("plugins", "empyrean", "skills", "empyrean")
    shutil.copytree(SKILL_DIR, tmp_skill)
    (tmp_path / "empyrean-py").mkdir()
    (tmp_path / "empyrean-py" / "docs").symlink_to(DOCS)
    # The generator reads the version from pyproject.toml beside the docs.
    pyproject = REPO / "empyrean-py" / "pyproject.toml"
    (tmp_path / "empyrean-py" / "pyproject.toml").symlink_to(pyproject)

    agentdocs.write_skill(tmp_path, empyrean)

    regenerated = sorted(p.relative_to(tmp_skill) for p in tmp_skill.rglob("*") if p.is_file())
    checked_in = sorted(p.relative_to(SKILL_DIR) for p in SKILL_DIR.rglob("*") if p.is_file())
    assert regenerated == checked_in, STALE_MSG
    for rel in regenerated:
        assert (tmp_skill / rel).read_bytes() == (SKILL_DIR / rel).read_bytes(), (
            f"{STALE_MSG} ({rel})"
        )


# ── (f) write_llms ───────────────────────────────────────────────────


def test_write_llms(tmp_path):
    agentdocs.write_llms(
        tmp_path, DOCS, empyrean, "https://docs.empyrean-dynamics.com", "9.9.9", "Test description."
    )
    llms = (tmp_path / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# empyrean")

    documentation = llms.split("## Documentation", 1)[1].split("## Verification", 1)[0]
    prefix = "](https://docs.empyrean-dynamics.com/"
    links = [
        line.split(prefix, 1)[1].split(")", 1)[0]
        for line in documentation.splitlines()
        if prefix in line
    ]
    assert links
    for rel in links:
        assert (tmp_path / rel).exists(), rel

    assert "validation_report.html" in llms
    assert "empyrean-scenarios" in llms

    full = (tmp_path / "llms-full.txt").read_text(encoding="utf-8")
    for title in _titles().values():
        assert ("# " + title) in full, title

    for path in tmp_path.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for token in FORBIDDEN:
                assert token not in text, f"{token} leaked into {path.name}"


def test_tail_states_the_documented_version():
    """The generated region opens with the version from pyproject.toml, not the venv."""
    fields = agentdocs._pyproject_project_fields(REPO / "empyrean-py" / "pyproject.toml")
    version = fields["version"]
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    tail = text.split("agentdocs:generated:start", 1)[1]
    assert "This skill documents empyrean " + version + "." in tail
