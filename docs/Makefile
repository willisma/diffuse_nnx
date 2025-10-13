# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the "new" style
# where we invoke sphinx-build; take the -M flag to list all Sphinx targets.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for common tasks
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf api/

html: clean
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild -b html $(SOURCEDIR) $(BUILDDIR)/html --host 0.0.0.0 --port 8000

apidoc:
	@echo "API documentation is generated automatically by Sphinx autodoc extension"

build: apidoc html

serve: html
	cd $(BUILDDIR)/html && python -m http.server 8000

# Development targets
dev: livehtml

# Production build
prod: clean apidoc html
	@echo "Documentation built successfully!"
	@echo "Open $(BUILDDIR)/html/index.html in your browser"
