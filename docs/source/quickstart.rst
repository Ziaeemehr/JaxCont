Quickstart
==========

Functional continuation
-----------------------

The public API separates the problem, algorithm, numerical settings, and
events. This example follows a saddle-node branch through its fold:

.. code-block:: python

   import jax.numpy as jnp
   import jaxcont as jc

   def saddle_node(u, p, args):
       return u**2 + p

   problem = jc.bif_problem(saddle_node, u0=jnp.array([1.0]), p0=-1.0)
   result = jc.continuation(
       problem,
       p_span=(-1.0, 0.2),
       settings=jc.ContinuationPar(ds=0.03, max_steps=200),
       events=[jc.Fold()],
   )

   print(result.branch.states)
   print(result.branch.params)
   print([(event.kind, event.p) for event in result.events])

The default is ``PseudoArclength()``. It passes folds, runs the whole
bounded continuation loop as a compiled JAX computation, computes stability
in a vectorized post-pass, and refines requested fold/Hopf events. Select
another algorithm explicitly when needed:

.. code-block:: python

   jc.continuation(problem, jc.Natural(), p_span=(-1.0, -0.1))

Many diagrams in one kernel with vmap
-------------------------------------

The fixed-shape scan result can be transformed directly. Here each value of
``b`` produces one imperfect-pitchfork branch:

.. code-block:: python

   import jax
   from jaxcont.core.scan_continuation import pseudo_arclength_scan

   def run_branch(b):
       def rhs(u, p, args):
           return jnp.array([p * u[0] - u[0] ** 3 + b])

       return pseudo_arclength_scan(
           rhs,
           None,  # step_update
           jnp.array([0.05]),
           jnp.array(-1.0),
           jnp.array(2.0),
           None,  # args
           jnp.array(0.05),
           jnp.array(1e-5),
           jnp.array(0.2),
           jnp.array(1e-6),
           120,
           jnp.array(20),
       )

   bs = jnp.linspace(-0.4, 0.4, 256)
   branches = jax.vmap(run_branch)(bs)
   print(branches.states.shape)  # (256, 121, 1)

``n_valid`` records how many slots are valid for each fixed-size result. See
:doc:`auto_examples/example_06_vmap_sweep` for timing and plotting.

Differentiate a fold location
-----------------------------

``fold_parameter`` solves the fold extended system and uses implicit
differentiation, so its output supports reverse-mode AD:

.. code-block:: python

   def fold_system(u, p, theta):
       return jnp.array([u[0] ** 2 - theta * u[0] + p])

   def fold_p(theta):
       return jc.fold_parameter(
           fold_system,
           u_guess=jnp.array([0.4]),
           p_guess=jnp.array(0.2),
           args=theta,
       )

   theta = jnp.array(1.0)
   print(fold_p(theta))             # 0.25
   print(jax.grad(fold_p)(theta))   # 0.5

Use ``jax.jacfwd`` for sensitivities through the whole-loop scan itself;
reverse-mode differentiation through JAX's dynamic ``lax.while_loop`` is not
supported. :doc:`auto_examples/example_07_differentiable` demonstrates both
paths and a gradient-based inverse-design loop.

Precision
---------

For analysis work, enable 64-bit arithmetic before importing anything else
from JAX or ``jaxcont``:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

Single precision (``float32``, the JAX default) is faster and uses less
memory, but its roundoff limits how tightly Newton's method can converge,
so residuals stall around ``1e-6`` and bifurcation points are located less
accurately. JaxCont never changes this setting itself; the choice is yours.

Tolerances follow the dtype automatically. ``ContinuationPar.newton_tol``
and the ``tol`` arguments of the solvers default to ``None``, which means
``1e-9`` for ``float64`` and ``1e-5`` for ``float32``. Pass a number to
override it.

Next steps
----------

Browse the generated :doc:`auto_examples/index` and :doc:`api/index`.

Explicit curve layouts
----------------------

Curve factories attach a static ``CurveLayout(kind, formulation, n, free)``
to the problem and continuation branch. Curve events and two-parameter plots
require this metadata; manually constructed curve ``BranchPoint`` and ``Branch``
objects must set ``curve_layout`` explicitly. Ordinary equilibrium points may
leave it as ``None``. This is a breaking change in 0.5.0: dimensions are never
inferred from packed state lengths. Existing ``unpack_fold_curve`` and
``unpack_hopf_curve`` helpers apply only to extended states.

``decode_curve_state(X, q, layout, raw_f, user_args)`` returns the original
state, full two-parameter vector and diagnostic ``CurveModes``. Diagnostic
fold vectors and complex Hopf modes have unit norm; scans align their signs
or phases along each branch without changing the Newton unknowns.

Event candidates under JAX transformations
------------------------------------------

With events enabled, ``jit`` and ``vmap`` return unvalidated detections in
``result.event_candidates``: one fixed-shape buffer per event, with a
``bracket_mask`` indexed by bracket start and mode. Only valid consecutive
points can open a bracket. Traced ``events`` and ``rejected_events`` stay empty.
Eager calls use these same scanners and validate roots automatically.
Custom Python ``scan(points)`` hooks remain eager only; implement
``scan_arrays(data, args)`` or an array-pure scalar ``test_function`` for tracing.
Eager custom events without ``scan_arrays`` use the Python compatibility path,
including pointwise evaluation of Python scalar tests, and have ``None`` in
their candidate tuple slot.

Slice a batched result pytree before refining outside the trace::

   single = jax.tree_util.tree_map(lambda leaf: leaf[batch_index], batched_result)
   refined = jc.refine_event_candidates(single, problem, events, solvers=solvers)

Use the corresponding unbatched problem and the same solver bundle as detection.
Refinement applies the problem's ``step_update`` at each bracket's left point,
then validates locality and deduplicates adjacent roots. Candidates alone do
not certify a bifurcation. Bothside continuation remains eager only.

Detection masks and assignment permutations have stopped gradients; derivatives
at a crossing or spectral degeneracy are not promised. Smooth branch outputs
retain forward-mode support with ``loop="while"``. Reverse-mode differentiation
requires ``ContinuationPar(loop="scan")``; eager refinement has no reverse-mode
contract.

Floquet multipliers of periodic orbits
--------------------------------------

Periodic branches compute Floquet multipliers with the strategy in
``Solvers.floquet``:

- ``PeriodicSchur(max_sweeps=None)``, the default, decomposes the ordered collocation
  interval propagators by a complex periodic Schur iteration and never forms
  their product. Small physical multipliers keep their relative accuracy.
- ``ProductFloquet(eigen_solver=DenseEigen())`` eigendecomposes the formed
  monodromy matrix with a matrix ``EigenSolver`` and keeps that solver's
  ordering. Multipliers much smaller than the largest are lost to roundoff.

``Solvers.eigen`` serves equilibrium spectra only. A ``PeriodDoubling`` or
``NeimarkSacker`` event may set its own ``floquet_solver``; it then detects and
refines with that strategy, while branch stability keeps the bundle strategy.
Pass the same ``solvers`` to ``refine_event_candidates``.

``floquet_spectrum(raw_f, mesh, U, p, floquet_solver, args=user_args)`` returns
``FloquetSpectrum(values, phase, logabs, trivial_index, converged)``. In 0.5.0
the fifth positional argument of ``floquet_multipliers`` and
``branch_floquet_multipliers`` is this strategy, no longer a matrix
``eigen_solver``. The trivial index is the multiplier whose right eigenvector
is most aligned with the flow ``raw_f(x0, p, args)``, not the value nearest one.
Branches store ``floquet_logabs`` and ``floquet_converged``; stability tests
the nontrivial ``logabs < 0``.

Values equal ``phase * exp(logabs)`` and may underflow to zero or overflow to
infinity while ``logabs`` stays finite; only an exactly zero multiplier has
``logabs = -inf``. PD/NS detection uses finite near-unit values. A spectrum that
does not converge raises ``RuntimeError`` in eager public calls. Under ``jit``,
and inside continuation, it reports NaN values, index ``-1``, false status and
stability, and never opens a bracket.

Derivatives of ``PeriodicSchur`` multipliers come from a custom JVP on
periodic left and right eigenvectors, which also transposes for reverse mode.
Only simple, finite, nonzero multipliers are differentiable. Repeated or
defective multipliers have no derivative, and ``loop="scan"`` does not remove
this spectral limitation.

Minimal fold curves
-------------------

In 0.5.0, ``fold_curve_problem`` defaults to ``formulation="minimal"``.
Its state is ``(u, p_fixed)``, of width ``n+1``; import
``unpack_minimal_fold_curve`` from ``jaxcont.bifurcations.curves`` to split it.
The bordered residual freezes its borders during correction and refreshes them
at accepted points. Problem args are ``FoldCurveArgs``; the original RHS
receives only ``user_args``. Diagnostic null vectors are reconstructed by the
layout decoder for events and plotting. To retain the previous packed
``(u, p_fixed, v)`` state and ``unpack_fold_curve`` helper, pass
``formulation="extended"`` explicitly.

Minimal positive-frequency Hopf curves
--------------------------------------

In 0.5.0, ``hopf_curve_problem`` defaults to ``formulation="minimal"``.
Its state is ``(u, p_fixed, omega)``, of width ``n+2``; use
``unpack_minimal_hopf_curve(X, n)`` from ``jaxcont.bifurcations.curves``.
Complex borders are frozen during correction and refreshed with normalized
right and adjoint modes at accepted points. ``HopfCurveArgs`` preserves the
original RHS data in ``user_args``. Events reconstruct unit complex modes
through the layout decoder, and plotting uses the same layout metadata.

This chart requires ``omega > 10*default_tol(dtype)`` and cannot cross a
Bogdanov--Takens point or continue a zero/negative-frequency segment. An eager
seed below that floor is rejected; residual evaluations outside the chart
return nonfinite values so continuation cannot accept them. Select
``formulation="extended"`` explicitly for the previous
``(u, p_fixed, q1, q2, omega)`` representation and ``unpack_hopf_curve`` helper.

Hopf curves through BT
----------------------

Select ``hopf_curve_problem(..., formulation="bt")`` to continue through a
Bogdanov-Takens point using the real state ``(u, p_fixed, v, w, kappa)``
(width ``3*n+2``, larger than the default minimal chart's ``n+2``).
The seed must have positive Hopf frequency. The seed-fixed gauge remains local;
loss of chart regularity reports solver failure.

Positive kappa describes an oscillatory Hopf pair. Negative kappa describes
a neutral-saddle extension with real eigenvalues ``+/-sqrt(-kappa)``, rather
than an oscillatory Hopf bifurcation. Two-parameter plots label this extension
and draw it dashed. BT events use kappa crossings followed by original-system
root validation; GH, ZH and HH events require strictly positive kappa.
