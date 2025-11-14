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
2. Create a ContinuationProblem
3. Choose a continuation method
4. Run the continuation
5. Analyze results and bifurcations
6. Visualize the bifurcation diagram

Example workflow:

.. code-block:: python

   # 1. Define system
   def my_system(state, params):
       # ... implement your RHS
       return f_value
   
   # 2. Create problem
   problem = ContinuationProblem(
       rhs=my_system,
       u0=initial_state,
       params=parameters,
       continuation_param='parameter_name'
   )
   
   # 3. Run continuation
   solution = equilibrium_continuation(
       problem,
       param_range=(p_min, p_max)
   )
   
   # 4. Analyze
   bifurcations = detect_bifurcations(solution)
   
   # 5. Visualize
   solution.plot()
