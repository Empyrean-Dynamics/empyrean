# Minimal Sphinx Makefile.

# Note: --keep-going only (no -W). Sphinx surfaces a handful of
# "Field list ends without a blank line" warnings from inherited
# quivr.Table docstrings (the parent class's `column` / `apply_mask`
# Examples blocks end on a doctest output line, which docutils flags
# but doesn't actually mis-render). Those aren't ours to fix without
# patching quivr; treat warnings as informational rather than fatal.
SPHINXOPTS    ?= --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"

# Live-reload server: `pip install sphinx-autobuild` and run `make livehtml`.
livehtml:
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
