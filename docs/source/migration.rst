Migrating from 0.4 to 0.5
=========================

This page shows, for each user-visible breaking change in 0.5.0, the old
form and the new one. The complete list, including internal changes, is in
the changelog. The "after" examples on this page run as written, one after
the other (in float64, which ``import jaxcont`` enables; see below).

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np

   import jaxcont as jc


   def cubic(u, p, args):
       return jnp.array([p + u[0] - u[0] ** 3 / 3])

Removed classes: use ``bif_problem`` and ``continuation``
---------------------------------------------------------

``NewtonSolver``, ``Corrector``, ``EquilibriumProblem``,
``BoundaryValueProblem`` and ``Config`` (with its global helpers) are gone.

.. illustrative

.. code-block:: python

   # 0.4
   problem = EquilibriumProblem(rhs, u0, p0)
   solver = NewtonSolver(tol=1e-8)

.. code-block:: python

   # 0.5
   problem = jc.bif_problem(cubic, u0=jnp.array([-2.1038034027355366]), p0=-1.0)
   settings = jc.ContinuationPar(ds=0.01, max_steps=600)
   result = jc.continuation(problem, jc.PseudoArclength(), p_span=(-1.0, 1.0),
                            settings=settings, events=[jc.Fold()])
   assert result.status == "reached_end"

Installation diagnostics remain in ``jaxcont.utils.config``.

float64 only
------------

0.4 ran in whatever precision JAX was set to. 0.5 computes in float64 only:
``import jaxcont`` turns on ``jax_enable_x64`` if it is off and warns once
(``jaxcont.PrecisionWarning``). Enable it yourself first to silence the
warning:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

``JAXCONT_NO_AUTO_X64=1`` keeps the global setting untouched; JaxCont entry
points then raise ``RuntimeError``. Float32 inputs are converted to float64.

Tolerances follow the precision
-------------------------------

``ContinuationPar.newton_tol`` and the ``tol`` arguments of the point
solvers, curve factories, ``branch_switch`` and ``dprc_curve`` default to
``None``: ``1e-9`` in float64 and ``1e-5`` in float32. The old fixed defaults
(``1e-6``/``1e-8``) are gone, and periodic problems no longer need an explicit
``newton_tol=1e-5`` in float32. Pass a number only to override.

Arclength metric and ``ds``
---------------------------

Pseudo-arclength continuation weights the state by ``1/n_state`` and the
parameter by 1 (``ContinuationPar(theta=None)``). For a system with several
states this changes the physical length of a step of size ``ds`` and the
number of steps. ``theta=1.0`` restores the unweighted state geometry. A
concrete ``theta`` that is not finite and positive raises ``ValueError``.

.. code-block:: python

   unweighted = jc.ContinuationPar(ds=0.01, max_steps=600, theta=1.0)
   try:
       jc.ContinuationPar(theta=0.0)
   except ValueError as error:
       print(error)

Step budget: ``max_steps`` counts accepted points
-------------------------------------------------

Rejected steps no longer consume ``max_steps``. Total corrector attempts are
bounded by ``ContinuationPar.max_attempts`` (default ``4 * max_steps``); the
initial ``ds`` is clipped to ``[ds_min, ds_max]``, and a run stalls only when
a step already attempted at ``ds <= ds_min`` fails. A seed that does not
converge raises ``ValueError`` in eager mode. A zero-length ``p_span`` raises
``ValueError``.

.. code-block:: python

   try:
       jc.continuation(problem, p_span=(-1.0, -1.0), settings=settings)
   except ValueError as error:
       print(error)

Events: only validated hits, with reasons for the rest
------------------------------------------------------

``result.events`` holds only validated hits (``EventHit.status ==
"validated"``). A refinement that does not converge, leaves its bracket or
fails the spectral re-check is an ``EventHit(status="failed")`` with
``p = nan`` and ``info["reason"]`` in ``result.rejected_events``; the old
``info["converged"] = False`` form is gone. ``result.status`` reports
``"reached_end"``, ``"max_steps"``, ``"stalled"`` or ``"nonfinite"``.

.. illustrative

.. code-block:: python

   # 0.4
   folds = [e for e in result.events if e.info.get("converged", True)]

.. code-block:: python

   # 0.5
   folds = [e for e in result.events if e.kind == "fold"]
   assert all(e.status == "validated" for e in folds) and len(folds) == 2
   for hit in result.rejected_events:
       print(hit.kind, hit.info["reason"])

Spectral events (``Hopf``, ``BP``, ``PeriodDoubling``, ``NeimarkSacker``,
``BogdanovTakens``, ``ZeroHopf``, ``DoubleHopf``) detect on continuously
tracked modes and expose ``scan(points)`` instead of ``test_function``;
``select_candidate``, ``test_value``, ``near_unit_circle`` and ``near_critical``
are removed. Their ``tolerance`` (the ``|Im|`` threshold between real and
complex) defaults to ``sqrt`` of the dtype's default tolerance. Event
deduplication merges two hits only when their brackets are adjacent and their
states and parameters agree, so hits on different sheets at the same ``p``
are both kept. ``hopf_point(..., eigenvalue=...)`` and
``fold_point(..., null_vector=...)`` accept a seed for the tracked mode.

``BP.rank_tolerance`` now thresholds a scale-invariant ratio (smallest
singular value of ``[f_u | f_p]`` at the refined point over the largest on
the bracket); its default ``None`` resolves to ``1e3 * default_tol``.
Absolute thresholds tuned for 0.4 must be re-tuned.

First Lyapunov coefficient doubles
----------------------------------

``lyapunov_coefficient`` and the Hopf event's ``info["l1"]`` use the
MatCont/Kuznetsov convention (unit-norm ``q``, denominator ``2 * omega0``).
Values are twice the 0.4 values: the standard cubic Hopf gives ``-2``, not
``-1``. Multiply a tuned ``Hopf.l1_tolerance`` by two; the default is
unchanged.

.. code-block:: python

   def hopf_normal_form(u, mu, args):
       x, y = u
       r2 = x * x + y * y
       return jnp.array([mu * x - y - x * r2, x + mu * y - y * r2])


   hopf = jc.continuation(jc.bif_problem(hopf_normal_form, u0=jnp.zeros(2), p0=-0.5),
                          p_span=(-0.5, 0.5), settings=jc.ContinuationPar(ds=0.02, max_steps=200),
                          events=[jc.Hopf()])
   (hit,) = hopf.events
   assert abs(float(hit.info["l1"]) + 2.0) < 1e-4 and hit.info["criticality"] == "supercritical"

Reverse-mode derivatives need ``loop="scan"``
---------------------------------------------

Reverse mode through the default ``loop="while"`` engine raises a clear
``ValueError``; select ``ContinuationPar(loop="scan")``. Forward mode works
with both. Corrector iterations are not differentiated: a converged root
carries the implicit-function derivative.

Under ``jit`` and ``vmap``, branch arrays have fixed-size buffers with zero
padding. Index the endpoint with ``branch.n_valid - 1``; indexing ``[-1]``
can select padding. This example returns NaN when seed failure leaves no valid
point, so a failed solve cannot masquerade as a zero endpoint.

.. code-block:: python

   def endpoint(a, loop):
       prob = jc.bif_problem(lambda u, p, args: jnp.array([p * u[0] - u[0] ** 3 + args]),
                             u0=jnp.array([1.0]), p0=1.0, args=a)
       run = jc.continuation(prob, p_span=(1.0, 2.0),
                             settings=jc.ContinuationPar(ds=0.05, max_steps=40, loop=loop))
       branch = run.branch
       return jnp.where(branch.n_valid > 0,
                        branch.states[jnp.maximum(branch.n_valid - 1, 0), 0], jnp.nan)


   gradient = jax.grad(lambda a: endpoint(a, "scan"))(0.1)
   try:
       jax.grad(lambda a: endpoint(a, "while"))(0.1)
   except ValueError as error:
       print(error)

Periodic orbits: ``PeriodicArgs`` and Floquet results
-----------------------------------------------------

A periodic problem's ``args`` is a ``PeriodicArgs`` module (dynamic
``u_ref_coll``, ``uref_prime_coll`` and ``user_args``; static ``raw_f`` and
``mesh``) instead of the tuple ``(u_ref_coll, uref_prime_coll, raw_f, mesh)``.
Your own data is ``args.user_args``, and ``raw_f`` is always called as
``raw_f(u, p, user_args)``. ``PeriodDoubling`` and ``NeimarkSacker`` no longer
take ``raw_f``/``mesh``; ``continuation`` passes them the problem's
``PeriodicArgs``, and direct refinement without it raises ``ValueError``.
PD/NS refinement now has a budget of 50 iterations and rejects an unfinished
or unresolved bracket with ``reason="not_converged"``; the corrected point and
its critical multiplier (``info["multiplier"]``) are re-checked.
``branch_switch_pd`` scales its seed and verifies the doubling with the
mean-removed orbit ``u - mean(u)``, so results change for orbits that are not
centred on the origin.

.. illustrative

.. code-block:: python

   # 0.4
   u_ref, du_ref, raw_f, mesh = problem.args
   multipliers = floquet_multipliers(raw_f, mesh, U, p, eigen_solver)
   events = [jc.PeriodDoubling(raw_f=raw_f, mesh=mesh)]

.. code-block:: python

   # 0.5
   from jaxcont.stability.floquet import ProductFloquet, floquet_multipliers, floquet_stable


   def radial(u, rho, args):
       x, y = u
       r2 = x * x + y * y
       return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


   t = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
   mesh = jc.Collocation(ntst=10, ncol=4)
   orbit = jc.periodic_orbit_problem(radial, np.stack([np.cos(t), np.sin(t)], axis=1), t,
                                     2 * np.pi, 1.0, mesh)
   user_args = orbit.args.user_args
   values, trivial_index = floquet_multipliers(radial, mesh, orbit.u0, orbit.p0, args=user_args)
   assert bool(floquet_stable(values[None], jnp.array([trivial_index]))[0])
   product = floquet_multipliers(radial, mesh, orbit.u0, orbit.p0, ProductFloquet(), args=user_args)
   events = [jc.PeriodDoubling(), jc.NeimarkSacker()]

``floquet_multipliers`` returns ``(multipliers, trivial_index)`` and
``branch_floquet_multipliers`` the batched pair; the trivial multiplier is the
one whose eigenvector aligns with the flow, not the value nearest one, and
``floquet_stable(multipliers, trivial_index)`` needs those indices. The fifth
positional argument is a Floquet strategy (default ``PeriodicSchur()``); wrap
a matrix eigen solver as ``ProductFloquet(eigen_solver)`` to keep the old
monodromy-matrix path. ``interval_propagators(..., args=...)`` and
``monodromy_matrix(..., args=...)`` now pass ``args`` to the right-hand side.

Eager ``periodic_orbit_problem`` construction raises ``ValueError`` for a
nonpositive ``period0``, a trajectory shorter than ``0.95 * period0``, failed
refinement, or a refined orbit of near-zero amplitude.

Two-parameter curves: minimal charts by default
-----------------------------------------------

``fold_curve_problem`` and ``hopf_curve_problem`` default to
``formulation="minimal"``: a fold curve state is ``(u, p_fixed)`` (width
``n + 1``) and a Hopf curve state ``(u, p_fixed, omega)`` (width ``n + 2``).
Decode them with ``unpack_minimal_fold_curve`` / ``unpack_minimal_hopf_curve``,
or select ``formulation="extended"`` for the 0.4 eigenvector charts. The
minimal Hopf chart cannot cross a Bogdanov-Takens point; use
``formulation="bt"`` to pass it. The problem's ``args`` wraps your data
(``FoldCurveArgs`` / ``HopfCurveArgs``, with your data in ``user_args``), and
curve layouts are carried as static ``CurveLayout`` metadata. On an extended
Hopf chart, ``HopfCurveArgs`` also holds the eigenvector anchors
``q1_anchor`` / ``q2_anchor`` of the phase condition, which are re-anchored
after every accepted point; build the problem with the factory rather than by
hand. Manually built curve ``BranchPoint``/``Branch`` objects must supply a
``CurveLayout``.

.. illustrative

.. code-block:: python

   # 0.4
   u, p_fixed, v = unpack_fold_curve(state, n)

.. code-block:: python

   # 0.5
   from jaxcont.bifurcations.curves import unpack_minimal_fold_curve


   def cusp_normal_form(u, p, args):
       return jnp.array([p[0] + p[1] * u[0] - u[0] ** 3])


   fold_curve = jc.fold_curve_problem(cusp_normal_form, jnp.array([1.0]), jnp.array([-2.0, 3.0]))
   assert fold_curve.curve_layout.formulation == "minimal" and fold_curve.u0.shape == (2,)
   u, p_fixed = unpack_minimal_fold_curve(fold_curve.u0, n=fold_curve.curve_layout.n)

Low-level engines and persistence
---------------------------------

``pseudo_arclength_scan`` / ``natural_scan`` take a static ``step_update`` (or
``None``) right after ``f``, dynamic ``args`` right after ``p_end`` and a
trailing ``max_attempts``; ``f`` is called as ``f(u, p, args)``. Positional
calls must insert the new arguments. Saved event metadata stores complex
values as ``[real, imaginary]`` pairs; decode with ``complex(*pair)``.
``analyze_stability`` selects the dominant eigenvalue by maximum real part,
also for unsorted input.
