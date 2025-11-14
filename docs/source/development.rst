Development Guide
=================

Architecture Overview
---------------------

JaxCont is organized into several key modules:

Core Module
^^^^^^^^^^^

The core module contains the fundamental continuation algorithms:

- ``ContinuationProblem``: Defines the problem to solve
- ``ContinuationSolution``: Stores the computed solution branch
- ``PredictorCorrector``: Base class for continuation methods
- ``NaturalContinuation``: Simple parameter continuation
- ``PseudoArclengthContinuation``: Robust method that can pass folds

Problems Module
^^^^^^^^^^^^^^^

Defines different types of continuation problems:

- ``EquilibriumProblem``: For finding equilibrium points
- ``PeriodicOrbitProblem``: For periodic orbit continuation
- ``BoundaryValueProblem``: For BVP formulations

Bifurcations Module
^^^^^^^^^^^^^^^^^^^

Handles bifurcation detection and analysis:

- ``BifurcationDetector``: Main detection engine
- ``FoldBifurcation``: Fold/saddle-node bifurcations
- ``HopfBifurcation``: Hopf bifurcations  
- ``PeriodDoublingBifurcation``: Period-doubling bifurcations

Design Principles
-----------------

1. **JAX-First**: Leverage automatic differentiation and JIT compilation
2. **Modular**: Easy to extend with new methods
3. **Type-Safe**: Use type hints throughout
4. **Well-Tested**: Comprehensive test coverage
5. **User-Friendly**: Simple API for common tasks

Adding New Features
-------------------

Adding a Continuation Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Subclass ``PredictorCorrector``
2. Implement ``predict()``, ``correct()``, and ``compute_tangent()``
3. Add tests in ``tests/``
4. Document in user guide

Example:

.. code-block:: python

   from jaxcont.core.predictor_corrector import PredictorCorrector

   class MyNewMethod(PredictorCorrector):
       def predict(self, u, param, tangent, ds):
           # Your prediction step
           return u_pred, param_pred
       
       def correct(self, problem, u_pred, param_pred, ...):
           # Your correction step
           return u_corrected, param_corrected, converged, iters
       
       def compute_tangent(self, problem, u, param, ...):
           # Compute tangent vector
           return tangent

Adding a Bifurcation Type
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a new class in ``bifurcations/``
2. Implement ``test_function()`` and ``detect()`` methods
3. Add to ``BifurcationDetector``
4. Write tests
5. Document the mathematics

Code Organization
-----------------

Directory Structure::

    src/jaxcont/
    ├── core/              # Core algorithms
    ├── problems/          # Problem definitions
    ├── bifurcations/      # Bifurcation detection
    ├── solvers/           # Numerical solvers
    ├── stability/         # Stability analysis
    └── utils/             # Utilities

Testing Strategy
----------------

Unit Tests
^^^^^^^^^^

Test individual components:

.. code-block:: python

   def test_newton_solver():
       solver = NewtonSolver(tol=1e-6)
       def f(x):
           return x**2 - 4.0
       x, converged, iters = solver.solve(f, jnp.array([1.0]))
       assert converged
       assert jnp.isclose(x, 2.0)

Integration Tests
^^^^^^^^^^^^^^^^^

Test complete workflows:

.. code-block:: python

   def test_pitchfork_continuation():
       problem = create_pitchfork_problem()
       solution = equilibrium_continuation(problem, (-1, 2))
       assert solution.n_points > 10
       assert any(b['type'] == 'fold' for b in solution.bifurcations)

Performance Optimization
------------------------

Using JIT
^^^^^^^^^

JAX's JIT compilation can significantly speed up computations:

.. code-block:: python

   from jax import jit

   @jit
   def compute_residual(u, param):
       return rhs(u, param)

However, be careful with:

- Dynamic shapes
- Conditionals (use ``jax.lax.cond``)
- Loops (use ``jax.lax.fori_loop``)

GPU Acceleration
^^^^^^^^^^^^^^^^

JAX automatically uses GPU when available:

.. code-block:: python

   import jax
   print(jax.devices())  # Check available devices

For large systems, consider:

- Batching computations
- Using ``jax.vmap`` for vectorization
- Sparse matrix operations

Debugging
---------

Common Issues
^^^^^^^^^^^^^

**Tracer Errors**

If you see "Abstract tracer value encountered", you're mixing JAX arrays with Python control flow:

.. code-block:: python

   # Bad
   if jnp.sum(x) > 0:
       ...
   
   # Good
   from jax import lax
   lax.cond(jnp.sum(x) > 0, true_fn, false_fn)

**Non-convergence**

If Newton doesn't converge:

- Check initial guess
- Reduce step size
- Check Jacobian computation
- Try different tolerance

Documentation
-------------

We use Sphinx with:

- NumPy-style docstrings
- Mathematical notation (LaTeX)
- Code examples
- Cross-references

Build docs locally:

.. code-block:: bash

   cd docs
   make html
   # Open build/html/index.html

Release Process
---------------

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Run full test suite
4. Build and test package
5. Tag release in git
6. Push to PyPI

Continuous Integration
----------------------

GitHub Actions workflow:

- Run tests on multiple Python versions
- Check code formatting
- Build documentation
- Measure coverage

Community
---------

- GitHub Issues: Bug reports and feature requests
- Discussions: Q&A and general discussion
- Pull Requests: Code contributions

References
----------

- JAX Documentation: https://jax.readthedocs.io/
- MATCONT Manual: Theoretical background
- AUTO Documentation: Classical continuation methods
