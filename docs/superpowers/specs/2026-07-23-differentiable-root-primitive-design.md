# Extract `differentiable_root` primitive from `fold_solve.py`

**Date:** 2026-07-23

## Goal

Resolve v0.2 engineering-prep item 3 (`ROADMAP.md` "Engineering / architecture
recommendations for v0.2"): the Newton-in-`lax.while_loop` + `jax.custom_vjp`
pattern that makes a fold's location a reverse-mode-differentiable function of
design parameters is currently bespoke to `fold_solve.py`. `ARCHITECTURE.md`
§3.2 already calls Hopf/codim-2 extended-system solvers "natural follow-ups."
Before writing the second one, extract the shared scaffolding into
`solvers/implicit.py: differentiable_root(G, x0, theta) -> x*`, so each future
differentiable event (Hopf, then LPC/PD/NS in v0.2/v0.3) is a new `G`
function, not a new `custom_vjp` implementation.

This is prep work, not feature work. It ships no new public API surface
beyond the internal primitive, and changes no observable behavior of
`fold_point`/`fold_parameter`.

## Scope

**In scope:**
1. A new `differentiable_root(G, x0, theta, *, tol, max_iter) -> x*` function
   in `src/jaxcont/solvers/implicit.py`, generalizing `fold_solve.py`'s
   `_make_fold_solver`: Newton's method in `lax.while_loop`, wrapped in
   `jax.custom_vjp` implementing the implicit function theorem
   (`dx*/dθ = -G_x⁻¹ G_θ`) for the backward pass.
2. Refactor `fold_solve.py` to build its extended-system `G` and packed
   `x0 = (u, p, v)` as before, then delegate the solve to
   `differentiable_root` instead of its own `_make_fold_solver`/`solve`.
3. New unit tests for `differentiable_root` itself, independent of the fold
   domain (toy `G`, not an equilibrium/fold problem).
4. `fold_solve.py`'s existing public API (`fold_point`, `fold_parameter`) and
   its existing tests (`tests/test_functional_api.py::TestDifferentiableFold`)
   are unchanged from the outside and must keep passing unmodified — this is
   a refactor, not a behavior change.

**Explicitly out of scope:**
- A Hopf (or LPC/PD/NS) extended-system solver. `differentiable_root` is
  built and proven generic via its own tests plus the fold refactor; a real
  second consumer is future work once Hopf's extended system is designed.
- Making the Newton step's linear solve pluggable (`LinearSolver` protocol).
  `differentiable_root` keeps today's hardcoded `jnp.linalg.solve` for both
  the Newton step and the backward implicit-diff solve — that swap is v0.2
  prep item 5, a separate task.
- Any change to `BifurcationDetector`/`Event` (item 4) or to the v0.1
  equilibrium types.
- Exporting `differentiable_root` from `jaxcont.__init__` — it is an internal
  building block for `solvers/`/`bifurcations/` modules, not user-facing API
  (consistent with `fold_solve.py`'s own `_make_fold_solver`/`newton`/`_pack`
  not being exported today).

## Design

### 1. `solvers/implicit.py`

```python
def differentiable_root(
    G: Callable[[Array, PyTree], Array],
    x0: Array | Callable[[PyTree], Array],
    theta: PyTree,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """Solve G(x, theta) = 0 for x via Newton, differentiable in theta.

    x0 is the Newton seed. Pass a plain Array for a theta-independent seed,
    or a callable `theta -> Array` for a seed that depends on theta (as
    fold's null-vector guess does) — see "x0 as callable" below for why the
    callable form is required rather than precomputing a theta-dependent
    array and passing it in. The gradient dx*/dtheta uses the implicit
    function theorem, so it does not differentiate through the inner Newton
    loop.
    """
```

Internals are today's `newton()` + `custom_vjp` forward/backward pair from
`fold_solve.py`, lifted out and generalized:
- `newton(theta)`: resolves `x_seed = x0(theta) if callable(x0) else x0`,
  then `lax.while_loop` over Newton steps on `G(x, theta)` starting from
  `x_seed`, dense `jnp.linalg.solve` for the step, same
  convergence/divergence `done` condition (`norm(residual) < tol` or
  non-finite residual) as today.
- `custom_vjp` forward: run `newton`, save `(x_star, theta)` as residuals.
- `custom_vjp` backward: `Gx = jacobian(G, argnums=0)(x_star, theta)`,
  solve `Gx.T @ y = x_bar`, then `theta_bar = -vjp(lambda t: G(x_star, t), theta)(y)`
  — unchanged math from `fold_solve.py`'s `solve_bwd`, just renamed/generalized.

No class, no protocol — a plain function. This is a direct lift of working
code, not new design surface; a config-object/protocol wrapper (to mirror
item 5's upcoming `LinearSolver` style) would add abstraction this doesn't
need and was considered and rejected on YAGNI grounds.

**`x0` as callable (found during prototyping, not in the original ROADMAP
wording):** `fold_solve.py`'s actual seed (`_initial_v`, the SVD-based null
vector) genuinely depends on `theta`/`args` — it's not a fixed constant.
Prototyping the naive `differentiable_root(G, x0, theta)` signature (`x0` a
plain, precomputed `Array`) against this case surfaced a real
`jax.errors.UnexpectedTracerError`: computing `x0` from `theta` *outside*
`differentiable_root` and closing it into the `custom_vjp`-wrapped `newton`
leaks a `LinearizeTracer` across `lax.while_loop`'s internal trace boundary
during `jax.grad`. A `theta`-*independent* `x0` (verified separately) does
not hit this — the leak is specific to a seed that both (a) depends on the
differentiated argument and (b) is computed in a different function scope
than the one `custom_vjp` traces. The fix, verified against both a toy case
and the real fold extended system (exact match to the `theta/2` /
`theta**2/4` analytics `TestDifferentiableFold` checks): accept `x0` as
`Array | Callable[[theta], Array]`; when callable, `newton(theta)` evaluates
it as its first step, so the theta-dependence is computed *inside* the
traced primal (matching how `fold_solve.py` already did this today, before
this refactor — `_initial_v` runs inside `newton(args)` in the current
code). Callers with a genuinely constant seed pass a plain `Array`, unchanged
from the original signature.

### 2. `fold_solve.py` refactor

- Delete `_make_fold_solver` (its `newton`/`custom_vjp` body moves to
  `differentiable_root`).
- `fold_point` keeps `_pack`/`_unpack`/`_extended_residual`/`_initial_v`
  as-is (these are fold-domain-specific: building `G` and the initial
  `x0 = (u, p, v)`), then calls:
  ```python
  G = lambda x, theta: _extended_residual(x, f, theta, n)
  x0 = lambda theta: _pack(u_guess, p_guess, _initial_v(f, u_guess, p_guess, theta, n))
  x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
  ```
  (`x0` is passed as a callable, not a precomputed array — see "`x0` as
  callable" above; `_initial_v` depends on `args`/`theta`, same as today.)
- `fold_parameter` is unchanged (already just delegates to `fold_point`).
- Public signatures, defaults, and docstrings of `fold_point`/`fold_parameter`
  are unchanged.

### 3. Tests

- New `tests/test_differentiable_root.py`, domain-independent of bifurcation
  theory:
  - A toy scalar `G(x, theta) = x**2 - theta` (root `x* = sqrt(theta)`):
    correctness of `differentiable_root` against the analytic root, and
    `jax.grad` against the analytic `dx*/dtheta = 1/(2*sqrt(theta))`, with
    `x0` passed as a plain (theta-independent) `Array`.
  - A toy vector-valued case with `theta` as an array of two scalars,
    checking `jax.jacobian` over `differentiable_root` matches a
    hand-derived Jacobian — mirrors `TestDifferentiableFold`'s
    `test_vector_parameter_jacobian` but without fold machinery.
  - A case with `x0` passed as a **callable** `theta -> Array` (a seed that
    depends on `theta`, e.g. `x0 = lambda theta: jnp.array([jnp.sqrt(theta)])`)
    — this is the path `fold_solve.py`'s refactor actually exercises, and the
    one that surfaced the tracer-leak bug during design, so it needs its own
    direct regression test rather than relying solely on the fold tests.
- Existing `tests/test_functional_api.py::TestDifferentiableFold` re-run
  unmodified as the regression check that the refactor preserves behavior.
- Full suite (`pytest tests/ -q`) green, same pass count as before plus the
  new file's tests.

## Error handling

Unchanged from today: Newton divergence is signaled only by the existing
`isfinite`-based early-exit in the `while_loop`'s `done` condition (the loop
returns whatever `x` it last had at `max_iter` if it never converges or hits
a non-finite residual) — no new validation or error surface is introduced by
this refactor.

## Non-goals / future work this unblocks

- A real Hopf extended-system solver (`hopf_solve.py`?) built on top of
  `differentiable_root`, once Hopf's extended system (`f=0`, complex
  eigenvalue pair crossing condition) is designed — likely alongside or
  after item 4's `Event` protocol rewrite, since `hopf.py` currently only
  does sign-change scanning, not an extended-system solve.
- Swapping `differentiable_root`'s dense `jnp.linalg.solve` calls for a
  pluggable `LinearSolver` (item 5).
