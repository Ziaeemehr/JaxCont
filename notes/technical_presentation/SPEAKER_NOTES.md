# Speaker notes

These notes support a beginner-friendly technical presentation. Explain the
purpose and picture before the equation. Do not assume the audience already
knows numerical continuation, bifurcation terminology, or JAX transformations.

## Suggested routes

### Short route: 15–20 minutes

Cover the research problem, branch geometry, fold failure, pseudo-arclength
prediction and correction, the v0.2 capability map, the periodic-orbit workflow,
the JAX benefits table, validation, and main takeaways.

### Full route: 55–70 minutes

Use all main slides. Pause after the bordered Newton system and after the
collocation residual explanation. Use the appendix only for derivations, method
comparison, or implementation questions. For a 40-minute version, omit the
lower-level fixed-buffer/JIT slides or the detailed periodic-event examples
according to the audience.

## Teaching sequence

### 1. Begin with the scientific question

Continuation asks where equilibria exist as a model parameter changes.
Simulation asks where one trajectory goes in physical time. A root solver finds
one equilibrium; continuation orders many nearby equilibria into a branch.

Use the cubic example throughout. Point out that the branch remains smooth at a
fold even though the parameter stops being a good coordinate.

### 2. Motivate pseudo-arclength geometrically

Natural continuation takes fixed-parameter slices. At a fold the slice becomes
tangent to the branch and the state Jacobian becomes singular.

For pseudo-arclength, identify three points in the picture:

1. the last accepted point;
2. the tangent prediction;
3. the Newton-corrected point on the branch.

Explain the augmented equations in words:

- the model residual says “be an equilibrium”;
- the arclength constraint says “be approximately one step ahead.”

The bordered Newton system is the mathematical core. The extra parameter column
and geometric row can keep the larger system nonsingular at an ordinary fold.

### 3. Explain the implementation without unnecessary jargon

Separate three ideas:

- pseudo-arclength is the numerical method;
- predictor–corrector is the repeated algorithmic pattern;
- the scan engine is JaxCont’s compiled whole-branch implementation.

The current outer loop uses `jax.lax.while_loop`; “scan engine” does not mean
that it calls `jax.lax.scan`.

Describe fixed buffers as reserved output capacity. `n_valid` or the `valid`
mask distinguishes real points from unused capacity. This predictable shape is
what makes JIT compilation and `vmap` batching practical.

One branch is sequential because every point depends on the preceding point.
`vmap` provides parallelism across independent branches, not across the steps of
one branch.

### 4. Connect features to research applications

Continuation can map equilibria, locate candidate transition thresholds,
identify multistable regions, continue limit-cycle shape and period, follow
equilibrium or Floquet stability changes, and repeat the same analysis over
uncertain parameters or design variants.

For periodic orbits, stress the abstraction before the formulas: collocation
turns one entire cycle into one large root. The continuation engine does not
need a separate branch-following algorithm; it sees a different residual.

Keep the two stability rules visibly separate:

- equilibria are stable when every Jacobian eigenvalue has negative real part;
- periodic orbits are stable when every nontrivial Floquet multiplier lies
  inside the unit circle.

The multiplier nearest +1 is excluded because it represents motion along the
cycle itself.

### 5. Explain what JAX contributes

Map every JAX mechanism to a package benefit. Automatic differentiation builds
Jacobians, JIT reduces orchestration overhead, `lax.while_loop` keeps adaptive
loops on device, `vmap` batches independent branches and stability calculations,
and the custom VJP differentiates converged roots implicitly.

Do not describe these as accuracy improvements. They make a carefully specified
numerical program composable and executable across CPU/GPU/TPU backends; they
do not repair conditioning, precision, initialization, or event semantics.

Do not imply that a bifurcation diagram validates a model or establishes
causality. It describes mathematical solutions of the supplied model.

### 6. End with a trustworthy workflow

Emphasize five habits:

1. verify the starting equilibrium or refined cycle;
2. begin with a small conservative run;
3. inspect residuals, termination, and step sizes;
4. repeat with changed settings or direction;
5. validate important conclusions independently.

Repository tests establish software behavior. They do not replace validation of
the researcher’s model, parameter range, initial solution, or interpretation.

## Likely questions

### Why not always use pseudo-arclength?

It is more robust at folds but solves a slightly larger system and needs a
tangent. Natural continuation is useful for simple monotone branches and as a
reference method.

### Is pseudo-arclength exact arclength?

No. Its extra constraint is a local linear approximation based on the previous
tangent.

### Does JIT make the numerical answer more accurate?

No. JIT changes execution, not the mathematical conditioning or convergence of
the problem.

### Does automatic differentiation make everything exact?

It avoids a finite-difference step choice for Jacobian construction, but it
does not fix poor conditioning, low precision, non-convergence, or the wrong
branch.

### If continuation stops, did the physical branch end?

Not necessarily. Check termination reason, residuals, step-size limits,
non-finite model values, and the attempt budget before interpreting the stop.

### Does finding a Hopf point give the oscillation amplitude?

No. It identifies a local stability transition of an equilibrium. JaxCont v0.2
can follow a periodic orbit once the user supplies a coarse one-cycle trajectory
and period guess, but it does not automatically branch-switch from the Hopf
point or integrate the ODE to find that initial cycle.

### Why does a periodic orbit need a phase condition?

The same closed orbit can be represented with any point chosen as time zero.
That time-shift freedom makes the collocation Jacobian singular unless one
extra scalar condition pins a representative phase.

### Why is periodic stability not based on negative real parts?

A Floquet multiplier measures growth over one full period, so stability is a
magnitude condition: nontrivial multipliers must lie inside the unit circle.
Negative-real-part tests apply to continuous-time equilibrium eigenvalues.
