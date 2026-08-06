JaxCont documentation
=====================

**Vectorize continuation sweeps with** ``jax.vmap`` **and differentiate fold
locations with** ``jax.grad``.

JaxCont is a continuation and bifurcation-analysis library whose default
pseudo-arclength engine runs the whole continuation loop as a compiled JAX
computation. It supports equilibrium and periodic-orbit continuation,
principal codimension-one events, direct codimension-two point solvers, and
phase response curves: infinitesimal PRC (``prc_curve``, cross-validated
against MatCont) and its parameter-derivative sensitivity (``dprc_curve``,
validated independently of MatCont, which does not compute this quantity).
Branch switching, continuation of two-parameter curves, general
boundary-value problems, and connecting orbits remain unsupported.

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
   validation
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
