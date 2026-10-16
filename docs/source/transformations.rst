JAX transformations
===================

``ContinuationPar(loop="while")`` is the default. Use ``loop="scan"`` for
reverse-mode differentiation of continuation. Both ``PseudoArclength`` and
``Natural`` support this option. The scan runs exactly ``max_attempts``
iterations (default ``4 * max_steps``), retaining the carry unchanged after
termination. Its Newton loops also use bounded, masked scans. Buffer sizes,
attempt bounds and Newton iteration bounds are static in this variant.
Statuses, accepted/rejected masking and accepted-point ``step_update``
semantics match the while engine.

Differentiate a scalar reduction of ``branch.states * branch.valid[:, None]``
inside ``jax.jit`` to use the fixed-size traced representation. Derivatives
follow the locally selected acceptance and stopping decisions; changes to
those discrete decisions need not be smooth. Reverse-mode differentiation of
the while engine raises ``ValueError`` naming ``loop='scan'``.

The matrix records only transformations exercised by tests run for Task 4.1.
A pass describes the tested fixture, not all problems or option combinations.
``untested`` makes no support claim.

.. list-table:: Tested transformation coverage
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - Operation
     - jit
     - vmap
     - jacfwd
     - grad
   * - continuation (while)
     - pass [1]
     - pass [1]
     - pass [2]
     - ValueError [2]
   * - continuation (scan)
     - pass [1]
     - pass [1]
     - pass [3]
     - pass [2, 4]
   * - events
     - untested
     - rejected [5]
     - untested
     - untested
   * - fold_point
     - untested
     - untested
     - untested
     - pass [6]
   * - hopf_point
     - untested
     - untested
     - untested
     - pass [7]
   * - cusp_point
     - untested
     - untested
     - untested
     - pass [8]
   * - bogdanov_takens_point
     - untested
     - untested
     - untested
     - pass [8]
   * - generalized_hopf_point
     - untested
     - untested
     - untested
     - pass [8]
   * - zero_hopf_point
     - untested
     - untested
     - untested
     - pass [8]
   * - double_hopf_point
     - untested
     - untested
     - untested
     - pass [8]
   * - periodic_orbit_problem
     - pass [9]
     - pass [9]
     - untested
     - pass [9]
   * - prc_curve
     - untested
     - untested
     - untested
     - untested
   * - dprc_curve
     - untested
     - untested
     - untested
     - untested
   * - branch_switch
     - untested
     - pass [10]
     - untested
     - pass [10]

Evidence
--------

Tests in ``tests/test_functional_api.py``:

* **[1]** ``test_continuation_loop_jit_vmap`` (both algorithms and both loop options).
* **[2]** ``test_grad_through_scan_loop`` (both algorithms; reverse scan compared with
   forward while, rtol ``1e-5``), and
   ``test_grad_through_while_loop_raises_clear_error``.
* **[3]** ``test_scan_forward_derivative_matches_while``.
* **[5]** ``TestVmapSafety::test_vmap_with_events_raises_clearly`` (event orchestration
   requires eager Python execution).
* **[6]** ``TestDifferentiableFold::test_reverse_mode_grad_matches_analytic``.

Test in ``tests/test_differentiable_root.py``:

* **[4]** ``test_scan_natural_corrector_reverse_mode_matches_forward`` (nonlinear
   Newton corrections, including an inexact seed).

The engine-level scan entry points are ``pseudo_arclength_lax_scan`` and
``natural_lax_scan`` in ``jaxcont.core.scan_continuation``. The original
``pseudo_arclength_scan`` and ``natural_scan`` names retain the default
while behavior and dynamic iteration/attempt bounds.

Additional default-suite evidence:

* **[7]** ``tests/test_hopf_normal_form.py::test_hopf_parameter_grad_matches_finite_difference``
   differentiates ``hopf_parameter``, which calls ``hopf_point``.
* **[8]** ``tests/test_codim2.py::test_<name>_parameters_grad_matches_finite_difference``
   for ``cusp``, ``bogdanov_takens``, ``generalized_hopf``, ``zero_hopf``
   and ``double_hopf``; each parameter wrapper calls the corresponding point solver.
* **[9]** ``tests/test_periodic_orbit_problem.py``:
   ``test_jit_over_construction_matches_eager_radius``,
   ``test_vmap_over_construction_matches_eager_radii`` and
   ``test_construction_grad_matches_exact_radius_derivative_with_fixed_guess``.
* **[10]** ``tests/test_branch_switch.py``:
    ``test_branch_switch_vmap_batches_over_a_second_parameter`` and
    ``test_branch_switch_is_differentiable_in_args``.
