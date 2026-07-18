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

The default is ``PseudoArclength(engine="scan")``. It passes folds, runs the
whole bounded continuation loop as a compiled JAX computation, computes
stability in a vectorized post-pass, and refines requested fold/Hopf events.
Select another algorithm explicitly when needed:

.. code-block:: python

   jc.continuation(problem, jc.Natural(), p_span=(-1.0, -0.1))
   jc.continuation(
       problem,
       jc.PseudoArclength(engine="legacy"),
       p_span=(-1.0, 0.2),
   )

Many diagrams in one kernel with vmap
-------------------------------------

The fixed-shape scan result can be transformed directly. Here each value of
``b`` produces one imperfect-pitchfork branch:

.. code-block:: python

   import jax
   from jaxcont.core.scan_continuation import pseudo_arclength_scan

   def run_branch(b):
       def rhs(u, p):
           return jnp.array([p * u[0] - u[0] ** 3 + b])

       return pseudo_arclength_scan(
           rhs,
           jnp.array([0.05]),
           jnp.array(-1.0),
           jnp.array(2.0),
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

Next steps
----------

Browse the generated :doc:`auto_examples/index` and :doc:`api/index`.
