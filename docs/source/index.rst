JaxCont documentation
=====================

**Vectorize continuation sweeps with** ``jax.vmap`` **and differentiate fold
locations with** ``jax.grad``.

JaxCont is an equilibrium continuation and bifurcation-analysis library whose
default pseudo-arclength engine runs the whole continuation loop as a compiled
JAX computation. It supports natural and pseudo-arclength continuation, fold
and Hopf detection with refinement, equilibrium stability, and bifurcation
diagrams.

The v0.1 API intentionally excludes periodic orbits, Floquet multipliers,
boundary-value problems, branch switching, and two-parameter continuation.

Start here
----------

Install from PyPI:

.. code-block:: bash

   pip install jaxcont

Then follow the :doc:`quickstart` for the functional API, batched sweeps, and
differentiable fold locations.

.. toctree::
   :maxdepth: 2
   :caption: Using JaxCont

   installation
   quickstart
   auto_examples/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
