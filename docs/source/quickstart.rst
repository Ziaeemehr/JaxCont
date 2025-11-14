Quick Start Guide
=================

This guide will get you started with JaxCont in minutes.

Basic Continuation
------------------

Here's a simple example using a pitchfork bifurcation:

.. code-block:: python

   import jax.numpy as jnp
   from jaxcont import ContinuationProblem, equilibrium_continuation

   # Define the system: dx/dt = r*x - x^3
   def pitchfork(state, params):
       x = state[0]
       r = params['r']
       return jnp.array([r * x - x**3])

   # Create the continuation problem
   problem = ContinuationProblem(
       rhs=pitchfork,
       u0=jnp.array([0.1]),
       params={'r': -1.0},
       continuation_param='r',
       problem_type='equilibrium'
   )

   # Run continuation from r=-1 to r=2
   solution = equilibrium_continuation(
       problem,
       param_range=(-1.0, 2.0),
       ds=0.05,
       max_steps=200
   )

   # Plot the bifurcation diagram
   solution.plot()

Multi-dimensional System
------------------------

For systems with multiple state variables:

.. code-block:: python

   def lorenz(state, params):
       x, y, z = state
       sigma = params['sigma']
       rho = params['rho']
       beta = params['beta']
       
       dx = sigma * (y - x)
       dy = x * (rho - z) - y
       dz = x * y - beta * z
       
       return jnp.array([dx, dy, dz])

   problem = ContinuationProblem(
       rhs=lorenz,
       u0=jnp.array([0.0, 0.0, 0.0]),
       params={'sigma': 10.0, 'rho': 1.0, 'beta': 8/3},
       continuation_param='rho'
   )

   solution = equilibrium_continuation(problem, param_range=(1.0, 30.0))

Continuation Methods
--------------------

JaxCont supports different continuation methods:

Natural Continuation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jaxcont import NaturalContinuation

   continuation = NaturalContinuation(ds=0.01, max_steps=500)
   solution = continuation.run(problem, param_range=(0.0, 5.0))

Pseudo-arclength Continuation (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from jaxcont import PseudoArclengthContinuation

   continuation = PseudoArclengthContinuation(
       ds=0.01,
       ds_min=1e-5,
       ds_max=0.1,
       adaptive_stepsize=True
   )
   solution = continuation.run(problem, param_range=(0.0, 5.0))

Bifurcation Detection
----------------------

Enable bifurcation detection:

.. code-block:: python

   from jaxcont import BifurcationDetector

   detector = BifurcationDetector(
       detect_fold=True,
       detect_hopf=True,
       detect_branch_point=False
   )

   bifurcations = detector.detect_along_branch(solution)
   
   for bif in bifurcations:
       print(f"Found {bif['type']} bifurcation at parameter = {bif['parameter']}")

Stability Analysis
------------------

Compute eigenvalues along the branch:

.. code-block:: python

   from jaxcont.stability import compute_eigenvalues_along_branch

   eigenvalues = compute_eigenvalues_along_branch(problem, solution)
   
   # Plot eigenvalue trajectories
   from jaxcont.utils.plotting import plot_eigenvalues
   plot_eigenvalues(solution)

Customizing Plots
-----------------

Control plot appearance:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   
   solution.plot(
       state_index=0,
       ax=ax,
       show_bifurcations=True,
       stable_color='blue',
       unstable_color='red'
   )
   
   ax.set_title('My Bifurcation Diagram')
   ax.set_xlabel('Parameter')
   ax.set_ylabel('State')
   plt.savefig('bifurcation.png', dpi=300)

Configuration
-------------

Set global configuration:

.. code-block:: python

   from jaxcont.utils import Config, set_config

   # Use a fast configuration
   config = Config.fast()
   set_config(config)

   # Or customize
   custom_config = Config(
       use_jit=True,
       default_ds=0.05,
       default_tolerance=1e-6,
       compute_stability=True
   )
   set_config(custom_config)

Next Steps
----------

- Check out the :doc:`tutorials/index` for detailed examples
- Read the :doc:`user_guide/index` for in-depth explanations
- Explore the :doc:`api/index` for complete API documentation
- See :doc:`examples/index` for more complex use cases
