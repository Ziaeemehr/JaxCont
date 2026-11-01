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
one of these fully JIT-compiled, ``vmap``-safe whole-loop engines. With
``ContinuationPar(loop="while")`` (the default) they use the
``lax.while_loop`` engines below; with ``loop="scan"`` they use
``pseudo_arclength_lax_scan`` / ``natural_lax_scan``, a masked ``lax.scan``
with static iteration and attempt bounds that supports reverse-mode
differentiation (see :doc:`index`).

.. autofunction:: jaxcont.core.scan_continuation.pseudo_arclength_scan

.. autofunction:: jaxcont.core.scan_continuation.natural_scan
