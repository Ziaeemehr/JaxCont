# Speaker notes

These notes are written for someone new to numerical continuation. Do not try
to read every equation aloud. First explain the picture and purpose; then use
the equation to make the statement precise.

## Suggested routes

### 15-minute route

Use slides 1–3, 4–7, 9–16, 19, 21–22, 27–33, 38, 41–44. The central message is:
problem → fold failure → pseudo-arclength → whole-loop JIT → fixed buffers →
`vmap` → current limitations.

### 35–45-minute route

Use every main slide. Pause after the bordered system and ask the audience to
identify which new row/column removes the fold singularity. Use the appendix
only for questions.

## Opening and framing (slides 1–3)

Start with: “There are two stories here. One is a numerical mathematics story:
how to follow a curve through a fold. The other is a software execution story:
how to express that repeated method so JAX can compile and batch it.”

Stress that `scan` is a project naming convention for the whole-sweep engine.
The implementation uses `lax.while_loop`. This prevents the most likely source
of confusion at the beginning.

## Continuation fundamentals (slides 4–8)

For slide 4, distinguish equilibrium computation from simulation:

- simulation asks where a trajectory goes as physical time increases;
- continuation asks where equilibria exist as a model parameter changes.

For slides 5–6, point to successive dots on the curve. A root solver computes
one dot; continuation orders many dots and reuses local information.

For slide 7, say that `F_u` measures how the residual changes when the state is
perturbed. When it is invertible, a small parameter change has a locally unique
state correction. At a fold one state direction produces almost no first-order
residual change, so the inverse problem becomes singular.

For slide 8, physically trace the S-shaped curve with your finger. The curve is
not broken at a fold; only `p` is a bad coordinate there.

## Natural continuation (slides 9–10)

Describe natural continuation as a sequence of vertical slices in the branch
diagram. It fixes the next parameter value and asks Newton to find the state.
Near a fold that slice becomes tangent to the branch. Reducing the step helps
approach the fold but does not let the method turn around it.

Make clear that this failure is expected mathematical behavior, not a bug in
`natural_scan`. Natural continuation remains useful on monotone branches and as
a teaching/reference algorithm.

## Pseudo-arclength mathematics (slides 11–17)

On slide 11, define the tangent as an arrow in combined state–parameter space.
Both the state and parameter components can change.

On slide 12, identify three points:

1. the last accepted point;
2. the tangent prediction, which is cheap but usually off the branch;
3. the corrected point, where the branch meets the correction hyperplane.

On slide 13, explain the two equations in words before symbols:

- the model equation says “be an equilibrium”;
- the arclength equation says “be approximately one step ahead, not another
  arbitrary equilibrium.”

On slide 14, explain why tangent orientation matters. A null space has no
preferred sign. Without comparison to the previous tangent, the direction
could flip and the method could walk backwards.

Slide 15 is the mathematical core. At the fold, the upper-left block `F_u` is
singular. The additional parameter column and arclength row make a larger
system that is generically nonsingular at an ordinary fold. JaxCont solves the
full bordered system; it does not first invert `F_u`.

For slides 16–17, frame the method as an accept/reject state machine. Adaptive
step size is not changing the equations; it changes how far the next prediction
travels.

## Scan engines and consolidation (slides 18–25)

Slide 18 is the terminology slide. Repeat it if needed:

- pseudo-arclength = mathematics;
- scan engine = implementation structure;
- `lax.while_loop` = concrete JAX primitive currently used.

For slides 19–20, describe duplication as a correctness problem, not merely a
code-style complaint. A real fix reached one old implementation and not its
sibling. The consolidation leaves two mathematical engines, because natural
and pseudo-arclength genuinely have different correctors, but one result and
reassembly spine.

On slide 21, walk from the user's `jc.continuation` call down to the selected
engine and back into common result handling. Slide 22 is an important migration
lesson: the starting equilibrium is paired with `p_span[0]`, not silently with
`problem.p0`. These two values should describe the same equilibrium.

On slides 23–24, explain the carry and buffers as the compiled equivalent of
local variables and output lists. “Immutable” means each iteration returns the
next logical state; JAX/XLA may still optimize storage internally.

Slide 25 explains why the program is guaranteed to end. Mention both the outer
branch budget and inner Newton budget.

## Performance strategy (slides 26–37)

For slide 26, avoid saying that JAX operations themselves were slow. The issue
was many tiny dispatches coordinated by Python. Small linear algebra can finish
faster than the orchestration around it.

On slide 27, contrast many host-side loop iterations with one device program.
The continuation remains sequential mathematically, but the sequencing happens
inside compiled control flow.

On slide 28, explain cold versus warm timing. JAX compilation is an investment.
For a one-off tiny branch it may not pay back; for repeated/batched runs it often
does.

On slide 29, explain static arguments as part of the program's structure.
`max_steps` determines array dimensions, so changing it changes the compiled
shape.

On slides 30–31, fixed buffers and `jnp.where` are the main transformation
techniques. They replace variable-length Python lists and runtime Python `if`
statements with uniform array programs.

For slide 32, be precise: one branch's steps still depend on previous steps and
are not parallel. `vmap` parallelizes independent branches, such as different
design values or initial conditions.

Slide 33 shows a second pattern: after the sequential branch exists,
independent eigenvalue calculations across points are vectorized.

Slides 34–35 distinguish forward-mode differentiation through the scan from
reverse-mode implicit differentiation of a selected fold. Do not imply that
`jax.grad` can simply backpropagate through every `while_loop` iteration.

Slide 36 explains the eager/traced split. Convenience operations such as Python
list sorting and variable-length trimming stay outside the transformed path.
Events are therefore currently an eager-only feature.

For slide 37, give measured values only with their context. The safest external
claim is that the architecture reduces dispatch and enables batching. Re-run
benchmarks before publishing a speedup number.

## Correctness and scope (slides 38–44)

On slide 38, emphasize that white-box mathematical tests are valuable during a
large API migration. They ensure the new functional surface did not weaken the
actual tangent/corrector guarantees. Then give the final evidence: 75 default
tests plus 12 slow-marked tests, all six migrated gallery examples, and the two
BifurcationKit.jl cross-checks.

Slides 39–40 separate completed engine consolidation from the future
periodic-orbit feature. The merged work removes old code and creates a reliable
spine; it is not the complete v0.2 roadmap. Mention that some loop-body
duplication is intentionally retained until a third predictor provides a clear
abstraction target.

Slide 41 is deliberately candid. A technically strong presentation states
transformation and scaling boundaries explicitly.

Slide 42 can be used verbatim as the final spoken summary. Slide 43 is a quick
reference if terminology questions arise. End on slide 44 with the six main
takeaways.

## Likely questions

### Why not always use pseudo-arclength?

It is more robust at folds but solves a slightly larger linear system and needs
a tangent. Natural continuation is simpler and can be useful when the parameter
is known to remain a good coordinate. In JaxCont the default is
pseudo-arclength because bifurcation diagrams commonly contain folds.

### Is pseudo-arclength exact arclength?

No. The added constraint is a local linear approximation based on the previous
tangent. That is why the method is called pseudo-arclength.

### Is `lax.while_loop` parallel?

Not across dependent continuation steps. Its main benefit here is compiled
control flow with low dispatch overhead. Parallelism comes from `vmap` across
independent branches and from vectorized post-processing.

### Why fixed buffers instead of dynamic arrays?

JIT and `vmap` need predictable output shapes. The method reserves a maximum
capacity and returns `n_valid`/a mask to distinguish used entries.

### Does automatic differentiation guarantee accuracy?

It removes finite-difference truncation choices for Jacobian construction, but
it does not fix poor conditioning, an unconverged Newton solve, low floating
point precision, or following the wrong branch.

### What happens at a branch point rather than a fold?

The null space has additional structure and branch switching is required.
Pseudo-arclength can follow the current branch, but selecting a new branch is a
separate capability and is outside this consolidation.
