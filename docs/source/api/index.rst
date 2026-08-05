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

Differentiable Hopf solver
---------------------------

.. autofunction:: jaxcont.hopf_point

.. autofunction:: jaxcont.hopf_parameter

.. autofunction:: jaxcont.lyapunov_coefficient

Codim-2 point solvers
---------------------

Direct solvers for codimension-2 equilibrium bifurcations. These take a
parameter array ``p`` of shape ``(2,)`` (codim-2 needs two free parameters)
and are differentiable in ``args`` via the implicit function theorem, like
their codim-1 counterparts above. Each returns a trailing ``converged``
flag rather than raising.

.. autofunction:: jaxcont.fold_coefficient

.. autofunction:: jaxcont.cusp_point

.. autofunction:: jaxcont.cusp_parameters

.. autofunction:: jaxcont.bogdanov_takens_point

.. autofunction:: jaxcont.bogdanov_takens_parameters

.. autofunction:: jaxcont.generalized_hopf_point

.. autofunction:: jaxcont.generalized_hopf_parameters

.. autofunction:: jaxcont.zero_hopf_point

.. autofunction:: jaxcont.zero_hopf_parameters

.. autofunction:: jaxcont.double_hopf_point

.. autofunction:: jaxcont.double_hopf_parameters

Low-level scan engine
---------------------

The fixed-shape low-level result is useful when applying ``jax.vmap`` or
``jax.jacfwd`` to an entire sweep.

.. autofunction:: jaxcont.core.scan_continuation.pseudo_arclength_scan

.. autofunction:: jaxcont.core.scan_continuation.branch_eigenvalues

Visualization
-------------

.. autofunction:: jaxcont.viz.plot_continuation

.. autofunction:: jaxcont.viz.plot_eigenvalues

.. autofunction:: jaxcont.viz.plot_branch_states

2D phase planes
~~~~~~~~~~~~~~~

Phase-plane plots support two-dimensional autonomous systems only. Slices of
higher-dimensional systems are deliberately not drawn: on a slice the
zero-contours are not nullclines of the full system and their intersections
are not generally equilibria.

.. autofunction:: jaxcont.viz.plot_phase_plane

.. autofunction:: jaxcont.viz.plot_nullclines

.. autofunction:: jaxcont.viz.plot_vector_field

.. autofunction:: jaxcont.viz.plot_streamlines

.. autofunction:: jaxcont.viz.plot_equilibria

.. autofunction:: jaxcont.viz.plot_trajectory
