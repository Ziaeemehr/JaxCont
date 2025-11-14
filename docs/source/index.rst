JaxCont Documentation
=====================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/yourusername/JaxCont/blob/main/LICENSE
   :alt: License

**JaxCont** is a high-performance continuation and bifurcation analysis package implemented in JAX, 
designed for analyzing dynamical systems with automatic differentiation and GPU acceleration.

Features
--------

- **High Performance**: Leverages JAX's JIT compilation and automatic differentiation
- **Robust Continuation**: Pseudo-arclength continuation for passing fold bifurcations
- **Bifurcation Detection**: Automatic detection of fold, Hopf, and period-doubling bifurcations
- **Stability Analysis**: Eigenvalue and Floquet multiplier computation
- **GPU Ready**: Seamless GPU acceleration through JAX
- **Modern API**: Clean, intuitive Python interface
- **Extensible**: Modular design for easy customization

Quick Start
-----------

Installation:

.. code-block:: bash

   pip install jaxcont

Basic usage:

.. code-block:: python

   import jax.numpy as jnp
   from jaxcont import ContinuationProblem, equilibrium_continuation

   # Define system: dx/dt = r*x - x^3
   def pitchfork(state, params):
       x = state[0]
       r = params['r']
       return jnp.array([r * x - x**3])

   # Setup and run continuation
   problem = ContinuationProblem(
       rhs=pitchfork,
       u0=jnp.array([0.1]),
       params={'r': -1.0},
       continuation_param='r'
   )
   
   solution = equilibrium_continuation(problem, param_range=(-1.0, 2.0))
   solution.plot()

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   user_guide/index
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   development
   roadmap
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Theory

   theory/continuation
   theory/bifurcations
   theory/stability

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
