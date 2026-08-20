Roadmap
=======

Current Version: 0.3.1
-----------------------

**Status**: Alpha/Beta -- broad functional coverage (equilibrium and
periodic-orbit continuation, codimension-one and codimension-two events,
two-parameter continuation, phase response curves) with extensive
analytic, MatCont, and BifurcationKit cross-validation. Under active
hardening before a stable 1.0 release.

Completed (through v0.3.1)
---------------------------

- Natural and pseudo-arclength continuation, both as fully JIT-compiled,
  ``vmap``-safe whole-loop engines
- Newton solver with JAX autodiff; adaptive step-size control
- Equilibrium and periodic-orbit (limit-cycle) problem definitions, the
  latter via Gauss-Legendre orthogonal collocation
- Codimension-one event detection and refinement: fold, Hopf,
  period-doubling, Neimark-Sacker
- Floquet multiplier computation via the collocation monodromy matrix
- Hopf normal-form classification (first Lyapunov coefficient,
  criticality) and five direct codimension-two point solvers (cusp,
  Bogdanov-Takens, generalized Hopf, zero-Hopf, double Hopf), all
  differentiable via the implicit function theorem
- Two-parameter continuation: fold-curve and Hopf-curve problems with
  their own codimension-two event detection
- Infinitesimal phase response curves (iPRC) and their
  parameter-derivative sensitivity (dPRC)
- 2D phase-plane visualization and stability-aware bifurcation-diagram
  plotting
- Differentiable bifurcation locations (``jax.grad``/``jax.jacfwd``
  through fold/Hopf solvers) for inverse design
- Batched continuation sweeps with ``jax.vmap``
- Analytic, MatCont 7.6, and BifurcationKit cross-validation suites

Next: Advanced Features
------------------------

Features:

- ⬜ Branch switching
- ⬜ Homoclinic orbits
- ⬜ Heteroclinic connections
- ⬜ Invariant tori
- ⬜ Symmetry exploitation
- ⬜ Parallel continuation

Later: Performance & Polish
----------------------------

Features:

- ⬜ GPU optimization
- ⬜ Sparse matrix support
- ⬜ Adaptive mesh refinement
- ⬜ Parallel branch computation
- ⬜ Interactive visualization
- ⬜ Web interface

Version 1.0: Production Ready
------------------------------

Goals:

- ⬜ Feature complete
- ⬜ Comprehensive documentation
- ⬜ >90% test coverage
- ⬜ Extensive examples
- ⬜ Benchmark comparison with MATCONT/AUTO
- ⬜ Community adoption
- ⬜ Published paper

Future Directions
-----------------

Long-term goals beyond v1.0:

**Extended Applications**

- Delay differential equations (DDEs)
- Partial differential equations (PDEs via MOL)
- Differential-algebraic equations (DAEs)
- Stochastic systems
- Fractional-order systems

**Integration**

- Neural ODE frameworks
- Optimization libraries
- Machine learning workflows
- Scientific workflow systems

**Advanced Analysis**

- Parameter sensitivity analysis
- Uncertainty quantification
- Model reduction
- Control design

**Visualization**

- 3D phase portraits
- Interactive parameter exploration
- Animation generation
- Publication-quality figures

**Community**

- Plugin system for extensions
- User-contributed examples
- Tutorial workshops
- Conference presentations

Contributing to the Roadmap
----------------------------

We welcome input on priorities! If you have specific needs:

1. Open an issue on GitHub
2. Describe your use case
3. Suggest implementation approach
4. Offer to contribute

Priority is given to:

- Widely requested features
- Features with contributors
- Features enabling other features
- Bug fixes and stability

Version History
---------------

**v0.3.1** (August 2026)
   - Floquet near-unit-circle detection margin widened (bugfix)
   - Read the Docs build fix

**v0.3.0** (August 2026)
   - Hopf normal-form classification and five direct codimension-two solvers
   - 2D phase-plane visualization

**v0.2.0** (July 2026)
   - First PyPI release
   - Periodic-orbit continuation, Floquet multipliers, period-doubling/
     Neimark-Sacker detection

**v0.1.0** (July 2026)
   - Initial release
   - Core continuation framework
   - Basic bifurcation detection
   - Example gallery

Get Involved
------------

Help shape the future of JaxCont:

- ⭐ Star on GitHub
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 💬 Join discussions

See :doc:`contributing` for more details.
