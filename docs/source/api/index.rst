API reference
=============

The functional equilibrium API is the supported v0.1 surface.

Problem and continuation
------------------------

.. autofunction:: jaxcont.bif_problem

.. autofunction:: jaxcont.continuation

.. autoclass:: jaxcont.BifProblem
   :members:

.. autoclass:: jaxcont.ContinuationPar
   :members:

Algorithms and events
---------------------

.. autoclass:: jaxcont.PseudoArclength

.. autoclass:: jaxcont.Natural

.. autoclass:: jaxcont.Fold

.. autoclass:: jaxcont.Hopf

Results
-------

.. autoclass:: jaxcont.ContinuationResult
   :members:

.. autoclass:: jaxcont.Branch
   :members:

.. autoclass:: jaxcont.EventHit
   :members:

Differentiable fold solver
--------------------------

.. autofunction:: jaxcont.fold_point

.. autofunction:: jaxcont.fold_parameter

Low-level scan engine
---------------------

The fixed-shape low-level result is useful when applying ``jax.vmap`` or
``jax.jacfwd`` to an entire sweep.

.. autofunction:: jaxcont.core.scan_continuation.pseudo_arclength_scan

.. autofunction:: jaxcont.core.scan_continuation.branch_eigenvalues
