Core Module
===========

The core module contains the fundamental continuation algorithms and data structures.

Continuation Problem and Solution
----------------------------------

.. autoclass:: jaxcont.core.continuation.ContinuationProblem
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: jaxcont.core.continuation.ContinuationSolution
   :members:
   :special-members: __init__
   :show-inheritance:

Continuation Engines
--------------------

Both algorithms available through :func:`jaxcont.continuation`
(:class:`jaxcont.PseudoArclength` and :class:`jaxcont.Natural`) dispatch to
one of these fully JIT-compiled, ``vmap``-safe whole-loop engines.

.. autofunction:: jaxcont.core.scan_continuation.pseudo_arclength_scan

.. autofunction:: jaxcont.core.scan_continuation.natural_scan
