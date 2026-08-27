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

.. autoclass:: jaxcont.ContinuationAlgorithm

.. autoclass:: jaxcont.PseudoArclength

.. autoclass:: jaxcont.Natural

.. autoclass:: jaxcont.Event

.. autoclass:: jaxcont.Fold

.. autoclass:: jaxcont.Hopf

.. autoclass:: jaxcont.BP

.. autoclass:: jaxcont.PeriodDoubling

.. autoclass:: jaxcont.NeimarkSacker

Branch switching
-----------------

Switch off a detected bifurcation onto the branch that crosses there, rather
than continuing to follow the original branch through it. See
``docs/superpowers/specs/2026-08-26-branch-point-switching-design.md`` (an
equilibrium branch point, from ``BP``) and
``docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md`` (a
period-doubled cycle, from ``PeriodDoubling``).

.. autofunction:: jaxcont.branch_switch

.. autofunction:: jaxcont.branch_switch_pd

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

.. autofunction:: jaxcont.fold_coefficient

Differentiable Hopf solver
---------------------------

.. autofunction:: jaxcont.hopf_point

.. autofunction:: jaxcont.hopf_parameter

.. autofunction:: jaxcont.lyapunov_coefficient

Codim-2 point solvers
---------------------

Direct solvers for codimension-2 equilibrium bifurcations, differentiable in
``args`` via the implicit function theorem, like their codim-1 counterparts
above. The ``*_point`` functions below take a parameter array ``p`` of
shape ``(2,)`` (codim-2 needs two free parameters) and return a trailing
``converged`` flag rather than raising; the corresponding ``*_parameters``
functions return a bare ``p*`` array with no flag, for direct use under
``jax.grad``.

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

Codim-2 events
--------------

``Event`` implementations usable via ``continuation(..., events=[...])`` on a
fold or Hopf curve (see "Two-parameter continuation" below): each is a test
function built on the matching direct solver above.

.. autoclass:: jaxcont.Cusp

.. autoclass:: jaxcont.BogdanovTakens

.. autoclass:: jaxcont.GeneralizedHopf

.. autoclass:: jaxcont.ZeroHopf

.. autoclass:: jaxcont.DoubleHopf

Two-parameter continuation
---------------------------

Curves of folds or Hopf points through a two-parameter plane, traced by the
same scan engine as ordinary equilibrium continuation -- see
``docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md``.

.. autofunction:: jaxcont.fold_curve_problem

.. autofunction:: jaxcont.hopf_curve_problem

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

.. autofunction:: jaxcont.viz.plot_bifurcation_diagram

.. autofunction:: jaxcont.viz.plot_two_parameter_diagram

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

Stability analysis
-------------------

.. autofunction:: jaxcont.analyze_stability

.. autofunction:: jaxcont.compute_eigenvalues

Lower-level problem and solver infrastructure
-----------------------------------------------

The core dataclasses ``bif_problem``/``continuation`` build on, and the
pluggable linear-algebra/corrector protocols behind ``Solvers`` -- most users
should not need these directly.

.. autoclass:: jaxcont.EquilibriumProblem
   :members:

.. autoclass:: jaxcont.ContinuationProblem
   :members:

.. autoclass:: jaxcont.ContinuationSolution
   :members:

.. autoclass:: jaxcont.Solvers

.. autoclass:: jaxcont.LinearSolver

.. autoclass:: jaxcont.Dense

.. autoclass:: jaxcont.EigenSolver

.. autoclass:: jaxcont.DenseEigen

.. autoclass:: jaxcont.Corrector
   :members:

.. autoclass:: jaxcont.NewtonSolver
   :members:

.. autoclass:: jaxcont.Config
   :members:
