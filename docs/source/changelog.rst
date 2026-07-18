Changelog
=========

All notable changes to JaxCont will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.1.0] - Unreleased
--------------------

Added
^^^^^
- Initial supported release of JaxCont's equilibrium-continuation API
- Functional ``bif_problem`` / ``continuation`` interface
- Whole-loop compiled pseudo-arclength engine, used by default
- Batched continuation sweeps with ``jax.vmap``
- Differentiable fold locations with implicit reverse-mode gradients
- Core continuation framework
  
  - Natural parameter continuation
  - Pseudo-arclength continuation
  - Adaptive step size control
  - Predictor-corrector base class

- Problem definitions
  
  - Equilibrium problems

- Numerical solvers
  
  - Newton solver with JAX autodiff
  - Corrector methods

- Bifurcation detection
  
  - Fold bifurcation detector
  - Hopf bifurcation detector
  - Bifurcation point framework

- Stability analysis
  
  - Eigenvalue computation
  - Stability classification

- Utilities
  
  - Configuration system
  - Plotting functions
  - Bifurcation diagram generation

- Examples
  
  - Pitchfork bifurcation
  - Lorenz system
  - Neural-mass model
  - Batched imperfect-pitchfork sweep
  - Differentiable fold inverse design

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
- Periodic orbits, Floquet multipliers, boundary-value problems, normal forms,
  branch switching, and two-parameter continuation are outside the supported
  v0.1 scope.

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
