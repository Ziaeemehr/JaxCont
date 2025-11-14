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

Continuation Functions
----------------------

.. autofunction:: jaxcont.core.continuation.equilibrium_continuation

.. autofunction:: jaxcont.core.continuation.periodic_continuation

Predictor-Corrector Base Class
-------------------------------

.. autoclass:: jaxcont.core.predictor_corrector.PredictorCorrector
   :members:
   :special-members: __init__
   :show-inheritance:

Natural Continuation
--------------------

.. autoclass:: jaxcont.core.natural_continuation.NaturalContinuation
   :members:
   :special-members: __init__
   :show-inheritance:

Pseudo-arclength Continuation
------------------------------

.. autoclass:: jaxcont.core.pseudo_arclength.PseudoArclengthContinuation
   :members:
   :special-members: __init__
   :show-inheritance:
