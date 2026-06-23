"""Sphinx configuration for the empyrean-py docs site."""

from __future__ import annotations

import importlib.metadata as md

# ── Project metadata ─────────────────────────────────────────────────

_meta = md.metadata("empyrean")
project = _meta["Name"]
release = _meta["Version"]
version = release
author = _meta["Author-email"].split("<")[0].strip()
copyright = "2024-2026, Joachim Moeyens"

# ── Extensions ───────────────────────────────────────────────────────

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # numpy/scipy-style docstrings
    "sphinx.ext.intersphinx",       # cross-link to numpy / pyarrow / quivr docs
    "sphinx.ext.viewcode",          # [source] links next to API entries
    "sphinx_autodoc_typehints",     # render type hints in the signature
    "sphinx_copybutton",            # copy-button on code blocks
    "sphinx_design",                # cards / tabs / dropdowns for the cookbook
    "sphinxcontrib.katex",          # KaTeX as the math renderer
    "myst_parser",                  # accept Markdown source files alongside .rst
]

# MyST extensions — `dollarmath` parses `$...$` (inline) and `$$...$$`
# (display) in .md sources; sphinxcontrib-katex then renders them.
# The previous `\\(...\\)` / `\\[...\\]` LaTeX-style delimiters survive
# RST docstrings but get passed through verbatim by MyST without this.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "smartquotes",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "both"
autodoc_typehints_format = "short"
# Hide internal module nesting in cross-reference labels: render
# `CartesianOrbits` instead of `empyrean.orbits.orbits.CartesianOrbits`
# in the API ref tree and signature hover text. Public API users
# import everything from the top-level `empyrean` package; the deeper
# module structure is implementation detail.
add_module_names = False
python_use_unqualified_type_names = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# Intersphinx targets — let docstrings link out to the relevant ecosystem docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# Allow both .rst (default) and .md (via myst) source files.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML output ──────────────────────────────────────────────────────

html_theme = "empyrean"  # registered via empyrean-sphinx-theme entry point
html_title = f"empyrean {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "default_mode": "dark",   # initial theme on first visit (user can toggle)
    "homepage_url": "https://empyrean-dynamics.com",
}

# ── KaTeX setup — matches the Rust /// docs convention ───────────────

katex_prerender = True
katex_options = r"""{
    macros: {
        "\\R": "\\mathbb{R}"
    }
}"""
