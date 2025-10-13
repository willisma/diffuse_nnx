# DiffuseNNX Documentation

This directory contains the Sphinx documentation for the DiffuseNNX library.

## Quick Start

### Prerequisites

Make sure you have installed the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Building the Documentation

#### Option 1: Using the Makefile (Recommended)

```bash
# Build HTML documentation
make html

# Build and serve locally
make serve

# Clean and rebuild
make clean html

# Live reload during development
make livehtml
```

#### Option 2: Using the Build Script

```bash
# Build documentation
python build_docs.py

# Build and open in browser
python build_docs.py --open
```

#### Option 3: Using Sphinx Directly

```bash
# Generate API documentation
sphinx-apidoc -o api/ ../interfaces ../networks ../samplers ../trainers ../eval ../utils ../data --separate

# Build HTML
sphinx-build -b html . _build/html
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── examples.rst         # Usage examples
├── contributing.rst     # Contributing guidelines
├── api/                 # Auto-generated API documentation
│   ├── interfaces.rst
│   ├── networks.rst
│   ├── samplers.rst
│   ├── trainers.rst
│   ├── eval.rst
│   ├── utils.rst
│   └── data.rst
├── interfaces/          # Interface documentation
│   └── index.rst
├── networks/            # Network documentation
│   └── index.rst
├── samplers/            # Sampler documentation
│   └── index.rst
├── Makefile             # Build commands
├── build_docs.py        # Build script
└── README.md            # This file
```

## Development

### Adding New Documentation

1. **New modules**: Add to the appropriate `api/*.rst` file
2. **New sections**: Create new `.rst` files and add to `index.rst`
3. **Examples**: Add to `examples.rst` or create new example files

### Updating API Documentation

The API documentation is automatically generated from docstrings. To update:

1. Ensure your code has proper docstrings
2. Run `make apidoc` to regenerate API docs
3. Build the documentation with `make html`

### Writing Documentation

Follow these guidelines:

- Use reStructuredText (`.rst`) format
- Include code examples with syntax highlighting
- Use proper cross-references between sections
- Keep examples up-to-date with the codebase

## Configuration

The Sphinx configuration is in `conf.py`. Key settings:

- **Project info**: Name, version, author
- **Extensions**: Autodoc, napoleon, myst-parser, etc.
- **Theme**: Read the Docs theme
- **Math support**: MathJax for equations

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the project is installed (`pip install -e .`)
2. **Missing modules**: Check that all modules are included in `sphinx-apidoc`
3. **Build errors**: Check the Sphinx output for specific error messages

### Getting Help

- Check the `conf.py` file for configuration options
- See the [Sphinx documentation](https://www.sphinx-doc.org/) for advanced usage
- Check the project's GitHub issues for known problems

## Publishing

The documentation can be published to:

- **GitHub Pages**: Enable in repository settings
- **Read the Docs**: Connect your GitHub repository
- **Local hosting**: Use `make serve` for local development

For GitHub Pages, the built documentation in `_build/html/` should be committed to the `gh-pages` branch.
