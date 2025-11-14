Changelog
=========

All notable changes to JaxCont will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
^^^^^
- Documentation system with Sphinx
- Comprehensive API documentation
- Tutorial notebooks
- Example gallery

Changed
^^^^^^^
- Improved error messages
- Better plot aesthetics

[0.1.0] - 2025-11-13
--------------------

Added
^^^^^
- Initial release of JaxCont
- Core continuation framework
  
  - Natural parameter continuation
  - Pseudo-arclength continuation
  - Adaptive step size control
  - Predictor-corrector base class

- Problem definitions
  
  - Equilibrium problems
  - Periodic orbit problem framework
  - Boundary value problem framework

- Numerical solvers
  
  - Newton solver with JAX autodiff
  - Corrector methods

- Bifurcation detection
  
  - Fold bifurcation detector
  - Hopf bifurcation detector
  - Period-doubling detector
  - Bifurcation point framework

- Stability analysis
  
  - Eigenvalue computation
  - Stability classification
  - Floquet multiplier framework

- Utilities
  
  - Configuration system
  - Plotting functions
  - Bifurcation diagram generation

- Examples
  
  - Pitchfork bifurcation
  - Lorenz system
  - Van der Pol oscillator

- Testing
  
  - Test framework with pytest
  - Core functionality tests
  - Example validation

- Documentation
  
  - README with quick start
  - Installation guide
  - Development guide
  - Contributing guidelines
  - MIT License

Known Issues
^^^^^^^^^^^^
- Bifurcation location needs refinement
- Normal form computation incomplete
- Limited periodic orbit support
- No two-parameter continuation yet

[0.0.1] - 2025-11-10
--------------------

Added
^^^^^
- Project initialization
- Package structure
- Basic framework design

---

Legend
------

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.
