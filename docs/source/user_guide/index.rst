User Guide
==========

Complete guide to using JaxCont for continuation and bifurcation analysis.

.. toctree::
   :maxdepth: 2

   introduction
   continuation_methods
   bifurcation_analysis
   periodic_orbits
   stability_analysis
   visualization
   advanced_topics

Introduction
------------

JaxCont is designed for numerical continuation and bifurcation analysis of dynamical systems.
It provides tools for:

- Computing solution branches as parameters vary
- Detecting and classifying bifurcation points
- Analyzing stability of equilibria and periodic orbits
- Visualizing bifurcation diagrams

Key Concepts
^^^^^^^^^^^^

**Dynamical System**: A system of ODEs of the form:

.. math::

   \\frac{du}{dt} = f(u, p)

where :math:`u \\in \\mathbb{R}^n` is the state and :math:`p` is a parameter.

**Equilibrium**: A solution where :math:`f(u, p) = 0`.

**Bifurcation**: A qualitative change in the dynamics as parameters vary.

**Continuation**: Following a branch of solutions as a parameter changes.

Basic Workflow
^^^^^^^^^^^^^^

1. Define your dynamical system
2. Create a problem with ``jc.bif_problem``
3. Choose a continuation algorithm (``jc.PseudoArclength()`` or ``jc.Natural()``)
4. Run the continuation with ``jc.continuation``
5. Analyze results and bifurcations
6. Visualize the bifurcation diagram

Example workflow:

.. code-block:: python

   import jaxcont as jc

   # 1. Define system: f(state, param, args) -> residual
   def my_system(u, p, args):
       # ... implement your RHS
       return f_value

   # 2. Create problem
   problem = jc.bif_problem(my_system, u0=initial_state, p0=p_min)

   # 3-4. Run continuation
   result = jc.continuation(
       problem,
       jc.PseudoArclength(),
       p_span=(p_min, p_max),
       settings=jc.ContinuationPar(),
       events=[jc.Fold(), jc.Hopf()],
   )

   # 5. Analyze
   solution = result._solution
   bifurcations = solution.bifurcations

   # 6. Visualize
   from jaxcont.utils.plotting import plot_continuation
   plot_continuation(solution)
