# Phase 1 Correctness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four Phase-1 correctness findings from the 2026-08-19 project review: invalid continuation seeds silently marked converged, root solvers silently returning failed/non-finite solutions, periodic-orbit event refinement evaluating uncorrected interpolated orbits, and the MC-LC-002 capability claim being inaccurate.

**Architecture:** No new modules. Each fix slots into the existing pattern already proven elsewhere in this codebase: seed correction reuses the existing `_natural_correct` Newton helper inside the jitted scan engines; solver convergence reporting reuses the `(x_star, converged)` pattern that `bifurcations/codim2.py`'s private `_solve_and_check` already established for the codim-2 solvers (promoted to a shared, public helper in `solvers/implicit.py` so `fold_solve.py`/`hopf_normal_form.py`/`codim2.py`/the periodic event refiners all share one implementation); periodic refinement correction reuses `differentiable_root` against the exact same collocation residual closure the continuation engine already runs (`rhs` is already available inside `Event.refine`).

**Tech Stack:** Python, JAX (`jax.numpy`, `lax.while_loop`, `jacfwd`), pytest.

**Spec:** `notes/PROJECT_REVIEW_2026-08.md` (findings #1, #2, #3, #6 and their "Recommended fix" sections; "Phase 1: Correctness hardening" in "Recommended implementation order").

## Global Constraints

- Float32 is the project default (no `jax_enable_x64` anywhere) — any new Newton tolerance must sit above the float32 residual floor already measured for that residual (e.g. periodic collocation ~3.4e-6, see `problems/periodic.py`'s own `tol=1e-5` choice), or correction will spuriously "fail" every time.
- Convergence flags returned from Newton primitives must be JAX arrays (`jnp.bool_`), not Python `bool`, until the point they are consumed in eager (non-traced) code — this keeps every solver still usable inside `jax.grad`/`jax.jit`/`jax.vmap` per the project's `jit`/`vmap`-first design.
- Do not add try/except-based error recovery inside anything that runs inside `lax.while_loop` or under `jax.jit`'s trace — non-convergence is signaled by a returned boolean, never an exception, inside traced code. Eager-only call sites (`curves.py`'s `*_curve_problem` factories, `events.py`'s `Event.refine` methods) may raise/branch on the concrete `bool(...)` of that flag.
- Follow the existing `_solve_and_check`/`cusp_point`-style convention already in `bifurcations/codim2.py`: a "point" solver returns `(..., converged)`; a "parameter"-only convenience wrapper returns a bare differentiable array with no convergence flag, documented as such, so `jax.grad` applies directly.
- Every touched file already has an established docstring-heavy, "explain the why" commenting style (see any file in `src/jaxcont/bifurcations/`) — match it; do not add throwaway comments.

---

## Task 1: Continuation engines Newton-correct and validate their seed

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py:156-269` (`pseudo_arclength_scan`), `:317-412` (`natural_scan`)
- Modify: `src/jaxcont/api.py:360-386` (`_run_scan`'s `p_span[0] == problem.p0` check)
- Modify: `tests/test_functional_api.py:31-34` (`_max_residual` helper)
- Test: `tests/test_functional_api.py` (new tests, appended)

**Interfaces:**
- Consumes: `_natural_correct(f, u_pred, p_fixed, tol, max_iter, linear_solver) -> (u_f, converged, iters)`, already defined at `src/jaxcont/core/scan_continuation.py:282-314` — no signature change, just a new call site.
- Produces: no public signature changes. `ScanResult.converged[0]` now reflects a real Newton-correction result instead of a hardcoded `True`; `ScanResult.states[0]`/`ScanResult.params[0]` now hold the corrected seed instead of the raw, possibly-invalid `u0`.

Currently both scan engines write `u0` into buffer slot 0 and hardcode `C[0] = True` (`.at[0].set(True)`) without ever evaluating `f(u0, p0)`. `_natural_correct` (a plain fixed-`p` Newton corrector, already used by `natural_scan`'s main loop) is the right tool to reuse for the seed too: the seed correction is exactly "Newton on `f(u, p0)=0` holding `p0` fixed."

- [ ] **Step 1: Newton-correct the seed in `pseudo_arclength_scan`**

In `src/jaxcont/core/scan_continuation.py`, inside `pseudo_arclength_scan` (around line 178-197), insert the correction before the tangent/buffers are built:

```python
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    # The scan buffer's slot 0 is the branch's starting point. u0 is caller-
    # supplied and not guaranteed to satisfy f(u0, p0) = 0 (e.g. a stale
    # guess, or api.py's BifProblem.u0 built by hand rather than refined).
    # Correct it the same way natural_scan corrects every other point on
    # the branch -- via _natural_correct, plain Newton with p held fixed at
    # p0 -- instead of writing the raw guess into slot 0 and marking it
    # converged unconditionally.
    u0_seed, seed_converged, _ = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)

    # Initial tangent: seed prev with the parameter axis pointing in `direction`,
    # so the branch is traversed toward p_end.
    seed = jnp.zeros(n + 1, dtype).at[-1].set(direction)
    tan0 = _tangent(f, u0_seed, p0, seed, linear_solver)

    # Fixed-size output buffers; slot 0 is the initial point.
    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0_seed)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype).at[0].set(tan0)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(seed_converged)
```

Delete the old `u0 = jnp.asarray(u0)` / `P = ...set(u0)` / `C = ...set(True)` lines this replaces (they are the ones currently at file lines ~178-194).

- [ ] **Step 2: Newton-correct the seed in `natural_scan`**

Same idea, in `natural_scan` (around line 346-358):

```python
    u0 = jnp.asarray(u0)
    n = u0.shape[0]
    dtype = u0.dtype
    p0 = jnp.asarray(p0, dtype)
    p_end = jnp.asarray(p_end, dtype)
    direction = jnp.sign(p_end - p0)

    u0_seed, seed_converged, _ = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)

    P = jnp.zeros((max_steps + 1, n), dtype).at[0].set(u0_seed)
    Q = jnp.zeros((max_steps + 1,), dtype).at[0].set(p0)
    T = jnp.zeros((max_steps + 1, n + 1), dtype)
    C = jnp.zeros((max_steps + 1,), dtype=bool).at[0].set(seed_converged)
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
```

(`_natural_correct` is defined earlier in the same file at line 282, so it is already in scope for both call sites — Python resolves the name at call time, not at `natural_scan`'s definition time, and `pseudo_arclength_scan` calling it before its own textual definition is fine for the same reason.)

- [ ] **Step 3: Apply the `p_span[0] == problem.p0` check to every problem kind, not just curve kinds**

In `src/jaxcont/api.py`, `_run_scan` (lines 360-386) currently nests the equality check inside `if problem.kind in ("fold_curve", "hopf_curve"):`. Move it out so it runs for every kind:

```python
    if problem.kind in ("fold_curve", "hopf_curve"):
        if settings.compute_stability:
            raise ValueError(
                f"compute_stability=True is not meaningful for "
                f"kind={problem.kind!r}: the branch state is an extended-"
                f"system vector, so eigendecomposing its Jacobian does not "
                f"give the original system's spectrum. Pass "
                f"ContinuationPar(compute_stability=False); the codim-2 "
                f"events in bifurcations/codim2_events.py carry their own "
                f"raw_f and compute the original spectrum themselves."
            )

    try:
        # Check p_span[0] == problem.p0 only in eager mode; traced calls
        # (jax.vmap/jax.jit) have p_start as a tracer and would raise
        # TracerBoolConversionError on the jnp.allclose result.
        if not jnp.allclose(jnp.asarray(p_start, dtype), problem.p0):
            raise ValueError(
                f"p_span[0]={float(p_start)} must equal the problem's starting "
                f"parameter p0={float(problem.p0)}. continuation() treats "
                f"p_span[0] as the literal starting value rather than reading "
                f"it off the problem, so a mismatch would start the run at a "
                f"point that is not the problem's actual seed."
            )
    except jax.errors.ConcretizationTypeError:
        # Traced call: p_start cannot be concretized to test equality, so
        # the check is deferred to eager execution. Skip it here.
        pass
```

This is the same two blocks that already existed, just un-nested: the `compute_stability` guard stays scoped to curve kinds, the `p_span[0]` equality check now applies to every kind.

- [ ] **Step 4: Stop hiding slot 0 from residual assertions in tests**

In `tests/test_functional_api.py`, flip `_max_residual`'s default now that slot 0 is a real corrected point:

```python
def _max_residual(f, states, params, args=None, skip_first=False):
    start = 1 if skip_first else 0
    vals = [jnp.abs(f(states[i], params[i], args)).max() for i in range(start, states.shape[0])]
    return float(jnp.max(jnp.array(vals))) if vals else 0.0
```

(Keep the `skip_first` parameter for any future caller that legitimately needs it; just flip which behavior is opt-in.)

- [ ] **Step 5: Add regression tests for both failure modes**

Append to `tests/test_functional_api.py` (in a class or at module level, matching the surrounding style):

```python
def test_p_span_mismatch_raises_for_equilibrium_problem():
    # Regression for finding #1: p_span[0] == problem.p0 used to be checked
    # only for fold_curve/hopf_curve kinds, so an ordinary equilibrium
    # problem silently started its branch at a point that was never on the
    # refined curve.
    def f(u, p, args):
        return u - p

    prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=jnp.array(0.0))
    with pytest.raises(ValueError, match="p_span"):
        jc.continuation(prob, p_span=(1.0, 1.2))


def test_invalid_seed_is_newton_corrected_not_silently_accepted():
    # Regression for finding #1: an invalid u0 (residual != 0 at p0) used to
    # be copied into branch slot 0 and marked converged unconditionally.
    def f(u, p, args):
        return u - p - 1.0  # true equilibrium: u = p + 1

    prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=jnp.array(0.0))  # f(0, 0) = -1, invalid
    result = jc.continuation(
        prob, p_span=(0.0, 0.5), settings=jc.ContinuationPar(max_steps=20),
    )
    assert result.branch.n_valid >= 1
    assert float(result.branch.states[0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert _max_residual(f, result.branch.states, result.branch.params) < 1e-5
```

- [ ] **Step 6: Run the affected tests**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_functional_api.py tests/test_adaptive_stepsize.py -v`
Expected: all pass, including the two new tests. (`test_adaptive_stepsize.py` is included because it also exercises `pseudo_arclength_scan`/`natural_scan` directly and must still pass unchanged.)

- [ ] **Step 7: Run the full fast suite to catch any other test relying on the old slot-0 behavior**

Run: `env JAX_PLATFORMS=cpu python -m pytest -q`
Expected: same pass/skip/xfail counts as the review's baseline (`333 passed, 3 skipped, 1 xfailed`) plus the 2 new tests, i.e. `335 passed, 3 skipped, 1 xfailed`. Investigate and fix any new failure before moving on — it means some other test depended on slot 0 being the raw, uncorrected `u0`.

- [ ] **Step 8: Commit**

```bash
git add src/jaxcont/core/scan_continuation.py src/jaxcont/api.py tests/test_functional_api.py
git commit -m "fix: Newton-correct continuation seed and validate p_span[0] for all problem kinds"
```

---

## Task 2: Root solvers report convergence instead of silently returning failed solutions

**Files:**
- Modify: `src/jaxcont/solvers/implicit.py` (new shared helper)
- Modify: `src/jaxcont/bifurcations/fold_solve.py` (`fold_point`/`fold_parameter`)
- Modify: `src/jaxcont/bifurcations/hopf_normal_form.py` (`hopf_point`/`hopf_parameter`)
- Modify: `src/jaxcont/bifurcations/codim2.py` (dedupe: reuse the shared helper instead of its own private copy)
- Modify: `src/jaxcont/bifurcations/curves.py` (`fold_curve_problem`/`hopf_curve_problem`: raise on non-converged seed)
- Modify: `src/jaxcont/bifurcations/events.py` (`Fold.refine`/`Hopf.refine`)
- Modify: `tests/test_hopf_normal_form.py`, `tests/test_codim2.py`, `tests/test_bifurcations.py`, `tests/test_functional_api.py`
- Modify: `examples/example_07_differentiable.py`, `examples/MatCont/python_cases/equilibrium.py`, `examples/MatCont/python_cases/transforms.py`
- Test: `tests/test_bifurcations.py`, `tests/test_hopf_normal_form.py` (new tests)

**Interfaces:**
- Produces: `differentiable_root_checked(G, x0, theta, *, tol=1e-8, max_iter=50) -> tuple[Array, Array]` in `src/jaxcont/solvers/implicit.py` — `(x_star, converged)`, `converged` a JAX bool.
- Produces: `fold_point(f, u_guess, p_guess, args=None, *, tol=1e-8, max_iter=50) -> (u, p, v, converged)` (was `(u, p, v)`).
- Produces: `hopf_point(f, u_guess, p_guess, args=None, *, tol=1e-8, max_iter=50) -> (u, p, q1, q2, omega0, converged)` (was `(u, p, q1, q2, omega0)`).
- Consumes (in `events.py`): the above two new 4-/6-tuples, and the existing `EventHit`/`BranchPoint` types unchanged.

This task follows the pattern `bifurcations/codim2.py`'s private `_solve_and_check` already established (residual re-check after `differentiable_root`, JAX-bool `converged`) — promoted to a shared public helper so `fold_solve.py` and `hopf_normal_form.py` stop being the inconsistent odd ones out.

- [ ] **Step 1: Add the shared checked-root helper**

In `src/jaxcont/solvers/implicit.py`, append after `differentiable_root`:

```python
def differentiable_root_checked(
    G: Callable[[Array, PyTree], Array],
    x0: Array | Callable[[PyTree], Array],
    theta: PyTree,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Array, Array]:
    """
    Solve ``G(x, theta) = 0`` like :func:`differentiable_root`, and
    additionally report whether the result is trustworthy.

    ``differentiable_root``'s Newton loop exits on a non-finite residual as
    well as on convergence, so the caller cannot otherwise tell success from
    failure without re-checking the final residual itself. ``converged`` is
    a JAX boolean (not a Python ``bool``), so this stays ``jit``/``vmap``-safe;
    callers in eager code should call ``bool(converged)`` themselves.
    """
    x_star = differentiable_root(G, x0, theta, tol=tol, max_iter=max_iter)
    residual = jnp.linalg.norm(G(x_star, theta))
    converged = (
        jnp.isfinite(residual)
        & (residual < tol)
        & jnp.all(jnp.isfinite(x_star))
    )
    return x_star, converged
```

- [ ] **Step 2: Run the codim-2 tests to confirm the pattern is understood correctly before reusing it**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_codim2.py -v`
Expected: all pass (no code changed yet in this step — this is a baseline check).

- [ ] **Step 3: Switch `fold_point` to report convergence**

In `src/jaxcont/bifurcations/fold_solve.py`, change the import and `fold_point`/`fold_parameter`:

```python
from jaxcont.solvers.implicit import differentiable_root_checked
```

```python
def fold_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Array, Array, Array, Array]:
    """
    Locate a fold near ``(u_guess, p_guess)``, differentiable in ``args``.

    Returns ``(u*, p*, v*, converged)`` where ``v*`` is the (unit) null
    vector of ``f_u`` and ``converged`` is a JAX bool reporting whether the
    extended-system residual actually reached ``tol`` (not just whether the
    Newton loop's iterate stayed finite) -- see
    :func:`jaxcont.solvers.implicit.differentiable_root_checked`.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _extended_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        return _pack(u_guess, p_guess, v0)

    x_star, converged = differentiable_root_checked(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v = _unpack(x_star, n)
    return u, p, v, converged


def fold_parameter(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Parameter value ``p*`` at the fold -- a scalar, differentiable in
    ``args``. Returns a bare array with no convergence flag so
    ``jax.grad(...)`` applies directly; use :func:`fold_point` when you need
    convergence info.

    ``jax.grad(lambda a: fold_parameter(f, u0, p0, a))(theta)`` gives the exact
    sensitivity of the fold location to the design parameters.
    """
    _, p, _, _ = fold_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p
```

- [ ] **Step 4: Switch `hopf_point` to report convergence**

In `src/jaxcont/bifurcations/hopf_normal_form.py`, add the import and update `hopf_point`/`hopf_parameter`:

```python
from jaxcont.solvers.implicit import differentiable_root_checked
```

```python
def hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """
    Locate a Hopf point near ``(u_guess, p_guess)``, differentiable in ``args``.

    Returns ``(u*, p*, q1*, q2*, omega0*, converged)``: the equilibrium,
    parameter, real and imaginary parts of the (unit) critical eigenvector,
    the critical frequency, and a JAX bool reporting whether the extended-
    system residual actually reached ``tol`` -- see
    :func:`jaxcont.solvers.implicit.differentiable_root_checked`.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _extended_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        q1_0, q2_0, omega_0 = _seed(f, u_guess, p_guess, theta, n)
        return _pack(u_guess, p_guess, q1_0, q2_0, omega_0)

    x_star, converged = differentiable_root_checked(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1, q2, omega = _unpack(x_star, n)
    return u, p, q1, q2, omega, converged


def hopf_parameter(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Parameter value ``p*`` at the Hopf point -- a scalar, differentiable in
    ``args``. Returns a bare array with no convergence flag so
    ``jax.grad(...)`` applies directly; use :func:`hopf_point` when you need
    convergence info.
    """
    _, p, _, _, _, _ = hopf_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p
```

- [ ] **Step 5: Dedupe `codim2.py`'s private helper against the new shared one**

In `src/jaxcont/bifurcations/codim2.py`, replace the import and delete the now-redundant local `_solve_and_check` (lines 44-67):

```python
from jaxcont.solvers.implicit import differentiable_root_checked as _solve_and_check
```

(Keeping the local name `_solve_and_check` as an alias means none of the ~10 call sites further down this file need to change.) Delete the old `def _solve_and_check(...): ...` body it replaces.

- [ ] **Step 6: Run the codim-2 tests again to confirm the dedupe is behavior-preserving**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_codim2.py -v`
Expected: identical pass count to Step 2 (same tests, same assertions, now calling the shared helper through an alias).

- [ ] **Step 7: Make `fold_curve_problem`/`hopf_curve_problem` reject a non-converged seed**

In `src/jaxcont/bifurcations/curves.py`, update the `fold_point` call (around line 117):

```python
    u_star, p_star, v_star, seed_converged = fold_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )
    if not bool(seed_converged):
        raise ValueError(
            f"fold_curve_problem: the initial fold-point refinement did not "
            f"converge near u_guess={u_guess}, p_guess={p_guess}. Pass a "
            f"guess closer to an actual fold, or increase tol/max_iter."
        )
```

And the `hopf_point` call (around line 184):

```python
    u_star, p_star, q1_star, q2_star, omega_star, seed_converged = hopf_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )
    if not bool(seed_converged):
        raise ValueError(
            f"hopf_curve_problem: the initial Hopf-point refinement did not "
            f"converge near u_guess={u_guess}, p_guess={p_guess}. Pass a "
            f"guess closer to an actual Hopf point, or increase tol/max_iter."
        )
```

- [ ] **Step 8: Update `Fold.refine`/`Hopf.refine` in `events.py` to use the convergence flag**

In `src/jaxcont/bifurcations/events.py`, `Fold.refine` (lines 104-117):

```python
    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        # fold_point expects f(u, p, args) (3-arg, per fold_solve.py); `rhs`
        # here is the 2-arg (u, p) -> Array callable used throughout this
        # module (matches api.py's rhs2), so adapt with an ignored 3rd arg.
        u_bif, p_bif, null_vector, converged = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        # A bracket sign-change is not a convergence guarantee: the same
        # "silent unchecked bifurcation location" risk Hopf.refine already
        # guards against applies here (Codim-2's Cusp.refine established
        # this fallback shape -- see bifurcations/codim2_events.py).
        if not bool(converged):
            return EventHit(
                kind="fold", p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind="fold", p=float(p_bif), u=u_bif, index=index,
            info={"null_vector": null_vector, "converged": True, "method": "extended_system"},
        )
```

`Hopf.refine` (lines 155-185): change the `hopf_point` call to unpack `converged`, and require it in the existing `finite` gate, and fall back to `right.p`/`right.u` (not the possibly non-finite solved values) when not ok:

```python
    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u, p, q1, q2, omega0, converged = hopf_point(
            lambda u, p, _args: rhs(u, p), u_guess, p_guess,
            tol=tolerance, max_iter=max_iterations,
        )
        l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
        # hopf_point's Newton solve (via differentiable_root) has no
        # convergence guarantee: if the bracket's sign change wasn't a real
        # Hopf point (a known occurrence -- see the "no close match --
        # spurious" branches in examples/example_05_neural_mass.py), p/l1/
        # omega0 can come back non-finite (e.g. p=-inf, l1=nan), or --the
        # gap `converged` closes -- come back finite but never actually
        # satisfy the extended-system residual within tol. Both `abs(nan) <
        # tol` and `nan < 0` are False, so without these guards a
        # non-convergent solve would silently fall through to the
        # "subcritical" else-branch below -- a confident-looking label for
        # a result that isn't a Hopf point at all. omega0 > 0 is checked too
        # since a genuine Hopf point always has a nonzero critical frequency.
        finite = jnp.isfinite(p) & jnp.isfinite(l1) & jnp.isfinite(omega0)
        ok = bool(converged) and bool(finite) and (float(omega0) > 0.0)
        if not ok:
            return EventHit(
                kind="hopf", p=float(right.p), u=right.u, index=index,
                info={"omega0": float(omega0), "l1": float(l1),
                      "criticality": "unknown", "converged": False,
                      "method": "extended_system"},
            )
        criticality = (
            "degenerate" if abs(l1) < self.l1_tolerance
            else "supercritical" if l1 < 0 else "subcritical"
        )
        return EventHit(
            kind="hopf", p=float(p), u=u, index=index,
            info={"omega0": float(omega0), "l1": float(l1),
                  "criticality": criticality, "converged": True,
                  "method": "extended_system"},
        )
```

- [ ] **Step 9: Update the monkeypatch-based Hopf tests in `test_bifurcations.py` for the new 6-tuple**

In `tests/test_bifurcations.py`, `test_hopf_refine_marks_non_finite_result_as_unknown`'s `fake_hopf_point` (lines 117-121) — non-finite `p`, so `converged=False`:

```python
    def fake_hopf_point(f, u_guess, p_guess, args=None, **kwargs):
        return (
            jnp.zeros(2), jnp.array(-jnp.inf),
            jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]), jnp.array(1.0),
            jnp.array(False),
        )
```

`test_hopf_refine_zero_omega_marks_unknown`'s `fake_hopf_point` (lines 145-149) — finite result, so `converged=True` (the test exists specifically to prove the `omega0 > 0` gate fires independently of Newton convergence):

```python
    def fake_hopf_point(f, u_guess, p_guess, args=None, **kwargs):
        return (
            jnp.zeros(2), jnp.array(0.02),
            jnp.array([1.0, 0.0]), jnp.array([0.0, 0.0]), jnp.array(0.0),
            jnp.array(True),
        )
```

- [ ] **Step 10: Add regression tests for the new convergence gate in `Fold.refine`/`Hopf.refine`**

Append to `tests/test_bifurcations.py`:

```python
def test_fold_refine_reports_not_converged_when_no_fold_exists(monkeypatch):
    # Regression for finding #2: a bracket sign-change is not a convergence
    # guarantee. Drive refine() with a fake fold_point that reports
    # converged=False and confirm the EventHit says so instead of silently
    # publishing an unchecked (u_bif, p_bif).
    import jaxcont.bifurcations.events as events_mod

    def fake_fold_point(f, u_guess, p_guess, args=None, **kwargs):
        return jnp.zeros(1), jnp.array(jnp.inf), jnp.zeros(1), jnp.array(False)

    monkeypatch.setattr(events_mod, "fold_point", fake_fold_point)

    def rhs(u, p):
        return u + p

    fold = Fold()
    left = BranchPoint(p=-0.05, u=jnp.zeros(1))
    right = BranchPoint(p=0.05, u=jnp.zeros(1))
    hit = fold.refine(left, right, (0, 1), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.info["converged"] is False
    assert hit.p == right.p


def test_hopf_refine_reports_converged_true_on_a_real_hopf_point():
    # Companion to the two "unknown" monkeypatched tests: confirm a genuine
    # convergent solve now also carries converged=True in info.
    def rhs(u, p):
        x, y = u[0], u[1]
        return jnp.array([p * x - 0.1 * y, 0.1 * x + p * y])

    def eigs_at(u, p):
        jac = jacfwd(lambda u_: rhs(u_, p))(u)
        return jnp.linalg.eigvals(jac)

    hopf = Hopf()
    left = BranchPoint(p=-0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), -0.05))
    right = BranchPoint(p=0.05, u=jnp.zeros(2), eigenvalues=eigs_at(jnp.zeros(2), 0.05))
    hit = hopf.refine(left, right, (3, 4), rhs, tolerance=1e-8, max_iterations=50)
    assert hit.info["converged"] is True
```

- [ ] **Step 11: Update every remaining call site to the new tuple shapes**

Mechanical unpacking updates (no logic changes) -- each adds one `, _converged` (or `, converged`) to an existing unpack:

`tests/test_hopf_normal_form.py` lines 28, 72, 106, 122, 138, 162 — e.g. line 28:
```python
    u, p, q1, q2, omega0, _converged = hopf_point(
```
(repeat the same trailing-name addition at each of the other 5 call sites in this file).

`tests/test_codim2.py` line 252:
```python
    _, p_h, _, _, om_h, _converged = hopf_point(
```

`tests/test_functional_api.py` line 154:
```python
        u, p, v, _converged = jc.fold_point(self._f, jnp.array([0.4]), jnp.array(0.2),
```
(keep whatever continuation the line already had after this point unchanged).

`examples/example_07_differentiable.py` line 52:
```python
u, p, v, _converged = jc.fold_point(f_fold, jnp.array([0.4]), jnp.array(0.2), theta0)
```

`examples/MatCont/python_cases/equilibrium.py`:
- Line 67-70, the list comprehension now yields 4-tuples:
```python
    refined = [jc.fold_point(_cubic_rhs, event.u, event.p) for event in fold_events]
    actual = sorted(
        [(float(state[0]), float(parameter), vector)
         for state, parameter, vector, _converged in refined],
        key=lambda item: item[0],
    )
```
- Line 169:
```python
        state, parameter, q1, q2, omega, _converged = jc.hopf_point(
            _van_der_pol_rhs, hopf_events[0].u, hopf_events[0].p, tol=1e-7
        )
```
- Line 259: apply the same `, _converged` addition (same call shape as line 169 -- read the surrounding 5 lines first to match local variable names exactly).

`examples/MatCont/python_cases/transforms.py`:
- Line 78-80 (`fold_parameter` closure) is unaffected -- it already indexes `[1]`, and `fold_point`'s `p` is still at index 1.
- Line 99-102 (`hopf_parameter` closure) is unaffected for the same reason.
- Line 108-113:
```python
    def lyapunov(scale):
        state, parameter, q1, q2, omega, _converged = jc.hopf_point(
            _lyapunov_family,
            jnp.zeros(2),
            jnp.array(0.05),
            scale,
            tol=1e-7,
        )
```

- [ ] **Step 12: Run the full affected-file test set**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_bifurcations.py tests/test_hopf_normal_form.py tests/test_codim2.py tests/test_functional_api.py -v`
Expected: all pass, including the 2 new tests from Step 10.

- [ ] **Step 13: Run the example scripts that were touched, to catch anything pytest doesn't cover**

Run: `env JAX_PLATFORMS=cpu python examples/example_07_differentiable.py`
Expected: runs to completion without a `TypeError`/unpacking error (compare printed output shape to a `git stash` run if in doubt).

- [ ] **Step 14: Run the full fast suite**

Run: `env JAX_PLATFORMS=cpu python -m pytest -q`
Expected: `337 passed, 3 skipped, 1 xfailed` (Task 1's 2 new tests + this task's 2 new tests, on top of the `333 passed` baseline).

- [ ] **Step 15: Commit**

```bash
git add src/jaxcont/solvers/implicit.py src/jaxcont/bifurcations/fold_solve.py \
        src/jaxcont/bifurcations/hopf_normal_form.py src/jaxcont/bifurcations/codim2.py \
        src/jaxcont/bifurcations/curves.py src/jaxcont/bifurcations/events.py \
        tests/test_hopf_normal_form.py tests/test_codim2.py tests/test_bifurcations.py \
        tests/test_functional_api.py examples/example_07_differentiable.py \
        examples/MatCont/python_cases/equilibrium.py examples/MatCont/python_cases/transforms.py
git commit -m "fix: propagate root-solver convergence status through fold_point/hopf_point and their callers"
```

---

## Task 3: Periodic event refinement Newton-corrects interpolated orbits before evaluating multipliers

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py` (`PeriodDoubling.refine`, `NeimarkSacker.refine`)
- Test: `tests/test_period_doubling_neimark_sacker.py` (new tests, appended)

**Interfaces:**
- Consumes: `differentiable_root_checked` from `src/jaxcont/solvers/implicit.py` (Task 2, Step 1) — `G(x, theta)` here is `rhs(U, p)`, already the exact periodic collocation residual closure (`api.py`'s `rhs2 = lambda u, p: problem.f(u, p, args)`, passed straight through `detect_events` into `Event.refine` as the `rhs` argument), so no wrapper lambda is needed (unlike `Fold`/`Hopf.refine`, which wrap for the 3-arg `fold_point`/`hopf_point` signature).
- Produces: no signature change to `refine()`. `EventHit.info` gains a `"corrected"` bool for both event kinds.

Both `PeriodDoubling.refine` and `NeimarkSacker.refine` currently linearly interpolate `u_mid = u_left + alpha * (u_right - u_left)` and hand that straight to `floquet_multipliers` — the interpolated point is not on the collocation residual manifold, so the multipliers describing it can be wrong on a curved branch (finding #6; a plausible contributor to the MC-LC-002 discrepancy Task 4 addresses next).

- [ ] **Step 1: Correct `PeriodDoubling.refine`'s interpolation midpoint**

In `src/jaxcont/bifurcations/events.py`, add the import at the top of the file:

```python
from jaxcont.solvers.implicit import differentiable_root_checked
```

Replace `PeriodDoubling.refine`'s body (lines 360-393):

```python
    def refine(
        self, left, right, index, rhs, *, tolerance, max_iterations, prev_value=None,
    ) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        v_left = prev_value if prev_value is not None else self.select_candidate(left, None)
        v_right = self.select_candidate(right, v_left)
        t_left = self.test_value(v_left)
        t_right = self.test_value(v_right)
        corrected_all = True
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_interp = u_left + alpha * (u_right - u_left)
            # A linear interpolation between two collocation states does not
            # itself satisfy the nonlinear collocation residual on a curved
            # branch. Correct it back onto the residual manifold at the
            # interpolated parameter before trusting its Floquet multipliers
            # -- otherwise a narrow reported parameter bracket does not by
            # itself imply an accurate multiplier crossing. tol=1e-5 matches
            # problems/periodic.py's own calibrated float32 residual floor
            # for this exact collocation residual (~3.4e-6); a tighter
            # default would spuriously report `corrected=False` every step.
            u_mid, corrected = differentiable_root_checked(rhs, u_interp, p_mid, tol=1e-5)
            if not bool(corrected):
                corrected_all = False
                break
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            v_mid = self.select_candidate(mid_point, v_left)
            t_mid = self.test_value(v_mid)
            # Three-way branch, not "left-half or else" -- see this file's
            # existing Global Constraints (Hopf has the same shape, for the
            # same reason: a two-way version degenerates whenever t_mid
            # lands on an exact zero).
            if t_left * t_mid < 0:
                p_right, u_right, t_right, v_right = p_mid, u_mid, t_mid, v_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left, v_left = p_mid, u_mid, t_mid, v_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="period_doubling", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection", "corrected": corrected_all},
        )
```

- [ ] **Step 2: Apply the identical fix to `NeimarkSacker.refine`**

Same change, same reasoning, in `NeimarkSacker.refine` (lines 449-478) — `kind="neimark_sacker"` in the returned `EventHit`, everything else identical to Step 1's body.

- [ ] **Step 3: Add a regression test proving the corrected orbit actually satisfies the collocation residual**

Append to `tests/test_period_doubling_neimark_sacker.py` (reuses `_build_problem`/`_sweep`/constants already defined in that file):

```python
def test_period_doubling_refine_result_satisfies_collocation_residual():
    # Regression for finding #6: refine() used to hand floquet_multipliers
    # a linearly-interpolated (uncorrected) orbit. The event's own .u/.p
    # must now satisfy the same nonlinear collocation residual the branch
    # itself is held to.
    prob, mesh, rhs = _build_problem(BETA_PD, alpha0=-0.05)
    sol = jc.continuation(
        prob, p_span=(-0.05, 0.05),
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.02, max_steps=50, newton_tol=1e-5
        ),
        events=[jc.PeriodDoubling(raw_f=rhs, mesh=mesh)],
    )
    assert len(sol.events) == 1
    hit = sol.events[0]
    assert hit.info.get("corrected", False) is True
    residual = prob.f(hit.u, jnp.asarray(hit.p, hit.u.dtype), prob.args)
    assert float(jnp.abs(residual).max()) < 1e-4


def test_neimark_sacker_refine_result_satisfies_collocation_residual():
    prob, mesh, rhs = _build_problem(BETA_NS, alpha0=-0.05)
    sol = jc.continuation(
        prob, p_span=(-0.05, 0.05),
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.02, max_steps=50, newton_tol=1e-5
        ),
        events=[jc.NeimarkSacker(raw_f=rhs, mesh=mesh)],
    )
    assert len(sol.events) == 1
    hit = sol.events[0]
    assert hit.info.get("corrected", False) is True
    residual = prob.f(hit.u, jnp.asarray(hit.p, hit.u.dtype), prob.args)
    assert float(jnp.abs(residual).max()) < 1e-4
```

- [ ] **Step 4: Run the period-doubling/Neimark-Sacker test file**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_period_doubling_neimark_sacker.py -v`
Expected: all pass, including the 2 new tests. If the existing detection-location assertions (`abs(sol.events[0].p) < 1e-4`) start failing, the correction step is changing which bracket direction gets taken — re-read Step 1's three-way branch logic against the original before debugging further, do not loosen the test tolerance to make it pass.

- [ ] **Step 5: Run the full fast suite**

Run: `env JAX_PLATFORMS=cpu python -m pytest -q`
Expected: `339 passed, 3 skipped, 1 xfailed`.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/events.py tests/test_period_doubling_neimark_sacker.py
git commit -m "fix: Newton-correct interpolated periodic orbits before evaluating Floquet multipliers in PD/NS refinement"
```

---

## Task 4: Verify MC-LC-002 against the periodic-refinement fix, and accurately state its status

**Files:**
- Modify: `examples/MatCont/cases.json:116-134` (`MC-LC-002` entry)
- Modify: `docs/source/validation.md:1-25`

**Interfaces:**
- Consumes: nothing new. Reads Task 3's fix's actual effect on the existing MATLAB-validated `MC-LC-002` comparison (`tests/test_matcont_suite.py::test_torbpc_jaxcont_matches_all_matcont_diagnostics`, currently `xfail(strict=True)`).
- Produces: no code interface changes — this task only makes `cases.json`/`docs/source/validation.md` state the true, current, MATLAB-verified status. It does not touch `examples/MatCont/registry.py`'s binary `support` field semantics (`"supported"`/`"unsupported"`) — those are consumed by equality checks (`case["support"] == "unsupported"`) elsewhere and must not gain a third value.

MATLAB (required to regenerate/compare the `MC-LC-002` reference artifacts) is available in this environment at `/home/ziaee/prog/Matlab/R2020a/bin/matlab`, so this task can actually run the comparison rather than guessing. Task 3's periodic-refinement fix is a plausible (not certain) contributor to the documented discrepancy — this task finds out which, empirically, rather than assuming.

- [ ] **Step 1: Re-run the strict `MC-LC-002` comparison with Task 3's fix in place**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_matcont_suite.py -k torbpc -v --runxfail`
Expected: one of two outcomes.
- If `test_torbpc_jaxcont_matches_all_matcont_diagnostics` now **passes**: the discrepancy is resolved. Proceed to Step 2a.
- If it **still fails**: the discrepancy has another cause beyond the interpolation-correction fix. Proceed to Step 2b.

- [ ] **Step 2a (fix resolved it): remove the `xfail` marker and flip the validation snapshot to PASS**

In `tests/test_matcont_suite.py`, remove the `@pytest.mark.xfail(strict=True, reason=...)` decorator (lines 258-262) from `test_torbpc_jaxcont_matches_all_matcont_diagnostics`, leaving the `@pytest.mark.slow` marker in place.

In `docs/source/validation.md`, update the snapshot line and table row:

```markdown
**Validation snapshot (<today's date>): all seven supported cases pass.**
```

```markdown
| `MC-LC-002` | `torBPC1` limit-cycle LPC/NS/PD locations, periods, extrema, and multipliers | PASS |
```

Remove the now-stale `MC-LC-002 is the current limitation: ...` paragraph and the `Exit status 1 is expected while MC-LC-002 remains failing.` sentence.

Skip to Step 3.

- [ ] **Step 2b (fix did not resolve it): keep the `xfail`, but make `cases.json`'s capability claim match reality**

In `examples/MatCont/cases.json`, `MC-LC-002`'s entry (lines 115-134) currently claims `"features": [..., "fold-of-cycles", "neimark-sacker", "period-doubling", "floquet-multipliers"]` under `"support": "supported"` with no caveat that the strict comparison is a known, tracked failure. Add a `"caveats"` field (an additive, optional key — `registry.py`'s `_REQUIRED_CASE_FIELDS` check only rejects *missing* required fields, so this is safe for `load_registry`/`select_cases` to ignore):

```json
      "id": "MC-LC-002",
      "title": "MatCont torBPC1 limit-cycle bifurcations",
      "support": "supported",
      "features": ["periodic-orbit", "fold-of-cycles", "neimark-sacker", "period-doubling", "floquet-multipliers"],
      "caveats": "The strict MatCont comparison (test_torbpc_jaxcont_matches_all_matcont_diagnostics) is a tracked xfail: JaxCont is missing LPC/PD event labels MatCont reports, and its maximum Floquet-multiplier error is approximately 1.10e-2. Treat fold-of-cycles/period-doubling/Neimark-Sacker support as experimental until this closes.",
```

(keep every other existing key in the entry unchanged — this only adds the one new key after `"features"`).

In `docs/source/validation.md`, strengthen the existing (already-transparent) caveat paragraph so it explicitly states the capability-maturity consequence the review flagged, right after the existing `MC-LC-002 is the current limitation: ...` sentence:

```markdown
Because this comparison remains an open, tracked failure, treat JaxCont's fold-of-cycles, period-doubling, and Neimark-Sacker detection as experimental rather than fully validated until it closes.
```

- [ ] **Step 3: Run the affected test files**

Run: `env JAX_PLATFORMS=cpu python -m pytest tests/test_matcont_suite.py -v` (full file, not just `-k torbpc`, to confirm nothing else in that file broke)
Expected: matches whichever of Step 2a/2b's outcome actually occurred — either the `torbpc` test now passes outright, or it remains `xfail` (not `XPASS`, which would mean the marker should have been removed after all).

- [ ] **Step 4: Commit**

```bash
git add examples/MatCont/cases.json docs/source/validation.md tests/test_matcont_suite.py
git commit -m "docs: reconcile MC-LC-002 capability claims with its actual MatCont-validated status"
```

(If Step 2a's path was taken, `tests/test_matcont_suite.py` is included in the `git add`; if Step 2b's path was taken, that file did not change and `git add` will simply have nothing to stage for it.)

---

## Final check across all four tasks

- [ ] **Run the complete test suite one more time end to end**

Run: `env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu python -m pytest -p no:cacheprovider -m ''`
Expected: no regressions versus the review's original baseline (`333 passed, 3 skipped, 1 xfailed`) plus this plan's new tests (6 from Tasks 1-3), and, depending on Task 4's outcome, either one fewer `xfailed` (now passing) or the same `xfailed` count with a more accurate `cases.json`/`docs/source/validation.md`.

- [ ] **Run flake8's enforced (non-`--exit-zero`) selection**

Run: `env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -m flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics`
Expected: `0`.
