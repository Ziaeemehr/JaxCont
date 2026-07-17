# JaxCont Documentation Structure

## Documentation Tree

```
docs/
├── Makefile                    # Unix build commands
├── make.bat                    # Windows build commands
├── README.md                   # Documentation build instructions
├── requirements.txt            # Documentation dependencies
├── build/                      # Generated documentation (git-ignored)
│   └── html/                   # HTML output
│       ├── index.html          # Main documentation page
│       ├── api/                # API reference pages
│       ├── examples/           # Example pages
│       ├── tutorials/          # Tutorial pages
│       └── user_guide/         # User guide pages
└── source/                     # Documentation source files
    ├── conf.py                 # Sphinx configuration
    ├── index.rst               # Main documentation page
    ├── installation.rst        # Installation instructions
    ├── quickstart.rst          # Quick start guide
    ├── contributing.rst        # Contributing guidelines
    ├── development.rst         # Developer guide
    ├── roadmap.rst             # Project roadmap
    ├── changelog.rst           # Version history
    ├── api/                    # API reference
    │   ├── index.rst           # API overview
    │   └── core.rst            # Core module docs
    ├── user_guide/             # User guides
    │   └── index.rst           # User guide index
    ├── examples/               # Example documentation
    │   ├── index.rst           # Examples index
    │   ├── pitchfork_bifurcation.rst
    │   └── notebooks/          # Jupyter notebooks (to be added)
    ├── tutorials/              # Step-by-step tutorials
    │   └── index.rst           # Tutorials index
    ├── _static/                # Static assets (CSS, images)
    │   └── custom.css          # Custom styling
    └── _templates/             # Custom Sphinx templates
```

## Built Documentation Pages

### Core Pages (✅ Complete)
- **index.html** - Main landing page with overview
- **installation.html** - Installation instructions
- **quickstart.html** - Quick start guide
- **contributing.html** - How to contribute
- **development.html** - Developer documentation
- **roadmap.html** - Project roadmap and future plans
- **changelog.html** - Version history

### API Reference (⏳ Partially Complete)
- **api/index.html** - API overview
- **api/core.html** - Core module reference
- **api/problems.html** - 🔴 To be created
- **api/bifurcations.html** - 🔴 To be created
- **api/solvers.html** - 🔴 To be created
- **api/stability.html** - 🔴 To be created
- **api/utils.html** - 🔴 To be created

### Examples (⏳ Partially Complete)
- **examples/index.html** - Examples gallery
- **examples/pitchfork_bifurcation.html** - Pitchfork example
- **examples/lorenz_system.html** - 🔴 To be created
- **examples/van_der_pol_oscillator.html** - 🔴 To be created

### Tutorials (🔴 To be Created)
- **tutorials/index.html** - Tutorial index (placeholder)
- **tutorials/tutorial_01_basics.html** - 🔴 To be created
- **tutorials/tutorial_02_continuation_methods.html** - 🔴 To be created
- **tutorials/tutorial_03_bifurcations.html** - 🔴 To be created
- **tutorials/tutorial_04_stability.html** - 🔴 To be created
- **tutorials/tutorial_05_periodic_orbits.html** - 🔴 To be created

### User Guide (🔴 To be Created)
- **user_guide/index.html** - User guide index (placeholder)
- Content pages to be added

## Documentation Features

### Configured Extensions
1. **sphinx.ext.autodoc** - Automatic API documentation from docstrings
2. **sphinx.ext.napoleon** - Support for NumPy/Google style docstrings
3. **sphinx.ext.viewcode** - Add links to source code
4. **sphinx.ext.intersphinx** - Link to external docs (NumPy, JAX, etc.)
5. **nbsphinx** - Jupyter notebook integration
6. **myst_parser** - Markdown support in documentation
7. **sphinx_rtd_theme** - Read the Docs theme

### Intersphinx Links
- Python: https://docs.python.org/3/
- NumPy: https://numpy.org/doc/stable/
- SciPy: https://docs.scipy.org/doc/scipy/
- Matplotlib: https://matplotlib.org/stable/
- JAX: https://docs.jax.dev/en/latest/

### Theme
- **sphinx_rtd_theme** (Read the Docs)
- Custom CSS in `_static/custom.css`
- Responsive design
- Built-in search functionality

## Building the Documentation

### Local Build

```bash
# From project root
cd docs

# Clean build
make clean

# Build HTML
make html

# Open in browser
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
start build/html/index.html  # Windows
```

### Live Rebuild (Auto-refresh)

```bash
cd docs
make livehtml
# Access at http://localhost:8000
```

### Alternative Build Command

```bash
cd docs
sphinx-build -b html source build/html
```

## Hosting Options

### 1. Read the Docs (Recommended)
- Free hosting for open source projects
- Automatic builds on git push
- Versioned documentation
- Search enabled
- Configuration: `.readthedocs.yml` already created

### 2. GitHub Pages
- Free hosting
- Requires GitHub Actions workflow
- Can serve from `docs/` folder or gh-pages branch

### 3. Self-Hosted
- Upload `docs/build/html/` to any web server
- Requires manual rebuilds

## Next Steps

### Documentation Content

1. **Complete API Reference**
   - Add api/problems.rst
   - Add api/bifurcations.rst
   - Add api/solvers.rst
   - Add api/stability.rst
   - Add api/utils.rst

2. **Add Example Documentation**
   - examples/lorenz_system.rst
   - examples/van_der_pol_oscillator.rst
   - More complex examples

3. **Create Tutorials**
   - Jupyter notebooks for interactive learning
   - Step-by-step walkthroughs
   - Video tutorials (optional)

4. **Expand User Guide**
   - Continuation methods explained
   - Bifurcation theory basics
   - Best practices
   - Troubleshooting

5. **Theory Section**
   - Mathematical background
   - Algorithm details
   - Comparisons with other tools

### Enhancements

- [ ] Add more diagrams and figures
- [ ] Create video tutorials
- [ ] Add interactive examples
- [ ] Set up Read the Docs hosting
- [ ] Add "Edit on GitHub" links
- [ ] Improve search optimization
- [ ] Add multilanguage support (optional)

## Status Summary

**✅ Complete:**
- Documentation structure and build system
- Sphinx configuration with all extensions
- Main landing pages (installation, quickstart, contributing, development, roadmap, changelog)
- API reference for core module
- Example documentation for pitchfork bifurcation
- Custom styling
- Build scripts (Makefile, make.bat)
- Read the Docs configuration

**⏳ In Progress:**
- API reference for remaining modules
- Example documentation for additional examples
- Tutorial framework (index created)
- User guide framework (index created)

**🔴 To Do:**
- Complete API reference pages
- Create tutorial notebooks
- Write detailed user guide content
- Add theory section
- Create more example documentation
- Add images and diagrams

## Documentation Statistics

- **Total RST files:** 13
- **Generated HTML pages:** 12
- **API modules documented:** 1/6
- **Examples documented:** 1/3
- **Build warnings:** 87 (mostly for missing planned pages)
- **Build time:** ~5-10 seconds

The documentation is now ready for content expansion!
