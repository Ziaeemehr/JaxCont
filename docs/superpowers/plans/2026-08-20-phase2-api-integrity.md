# Phase 2 API Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase-2 "API integrity" findings from the 2026-08-19 project review: `ContinuationPar(adaptive=False)` is a no-op, `newton_iters` diagnostics are fabricated as `0` and `verbose=True` prints nothing, solution persistence is broken (a `None` optional field raises `TypeError` on load) and unsafe (`allow_pickle=True` on untrusted files), and the BVP placeholder API isn't clearly marked as unimplemented.

**Architecture:** No new modules. `adaptive` and real per-step Newton-iteration counts both slot into the same fixed-size buffer/`Carry` pattern the scan engines already use for `ds` (a new `iters` buffer alongside the existing `ds` buffer, written every accepted step); `adaptive` becomes a new traced (non-static) scalar argument consumed only inside `_adapt_ds`'s existing `jnp.where` selection, so it stays `jit`/`vmap`-safe without touching `static_argnums`. Persistence moves from ad hoc `np.savez`/pickle-on-load to a versioned schema: numeric arrays stay native NumPy arrays (optional ones simply omitted, never stored as `None`), and non-numeric metadata (`bifurcations`, `convergence_info`, `state_names`, `param_name`) is JSON-encoded into one `metadata_json` unicode-array entry — NumPy stores unicode arrays natively, so nothing in the new format ever needs `allow_pickle=True`. The BVP fix is documentation/status-only; no behavior changes.

**Tech Stack:** Python, JAX (`jax.numpy`, `lax.while_loop`), NumPy (`np.savez`/`np.load`), `json`, pytest.

**Spec:** `notes/PROJECT_REVIEW_2026-08.md` (findings #5, #7, #10, #13 and their "Recommended fix" sections; "Phase 2: API integrity" in "Recommended implementation order").

## Global Constraints

- Prefer additive, non-breaking signature changes (new parameters with defaults; new optional dict/NamedTuple fields). None of this plan's tasks require a breaking change; if an implementer finds one is unavoidable, stop and add a `### Changed` / **Breaking:** bullet to `CHANGELOG.md`'s `[Unreleased]` section, matching the Phase 1 precedent already there, rather than making the change silently.
- `ContinuationSolution.load()` must default to `allow_pickle=False` and never widen that default — untrusted `.npz` files must never be able to trigger pickle execution (2026-08-19 review finding #7).
- Every touched file already has an established docstring-heavy, "explain the why" commenting style (see any file in `src/jaxcont/core/` or `src/jaxcont/bifurcations/`) — match it; do not add throwaway comments.
- Run tests with:
  ```
  PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu \
    /home/ziaee/envs/jaxcont/bin/python -m pytest -n auto -m '' <path>
  ```
  From inside a git worktree, `PYTHONPATH` **must** point at that worktree's own `src/` — the venv's editable install otherwise silently resolves `import jaxcont` to the main checkout's `src/`, so a worktree test run without it exercises the wrong code and produces confusing, unrelated-looking failures. Drop `-m ''` only for a single already-fast test file; the full-suite command above needs it to include `@pytest.mark.slow` tests like `tests/test_adaptive_stepsize.py`.

---

## Task 1: Real per-step diagnostics — `adaptive=False` and truthful `newton_iters`/`verbose`

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py` (`ScanResult`, `_adapt_ds`, `pseudo_arclength_scan`, `natural_scan`)
- Modify: `src/jaxcont/api.py` (`_run_scan`)
- Modify: `tests/test_adaptive_stepsize.py` (remove two stale "not wired" NOTEs; add a new regression test)
- Test: `tests/test_functional_api.py` (new tests, appended)
- Modify: `CHANGELOG.md` (`[Unreleased]` section)

**Interfaces:**
- Consumes: `_natural_correct(f, u_pred, p_fixed, tol, max_iter, linear_solver) -> (u_f, converged, iters)` (unchanged, already defined at `src/jaxcont/core/scan_continuation.py:299-331`) — its third return value, currently discarded as `_` at both scan engines' seed-correction call sites, becomes the seed's `iters` value.
- Produces: `ScanResult` gains a new field `iters: Array` (`(max_steps + 1,)`, `int32`, Newton iterations used to reach each point — mirrors the existing `ds` field's shape/semantics). `pseudo_arclength_scan`/`natural_scan` gain a new trailing parameter `adaptive: Array = jnp.array(True)` (after the existing `linear_solver` parameter; not added to `static_argnums`, since it is only ever used inside `jnp.where`). `_run_scan`'s built `convergence_info` dicts now have a real `"newton_iters"` int instead of the constant `0`. `continuation(..., verbose=True)` now prints one summary line via `_run_scan` (eager path only — the traced path, `_run_scan_traced`, never receives `verbose` today and this task does not change that).

Currently `_adapt_ds` runs unconditionally every step regardless of `ContinuationPar.adaptive` — a `ContinuationPar(adaptive=False)` run still grows/shrinks `ds` exactly like an adaptive one, because `_run_scan` never reads `settings.adaptive` at all. Separately, every accepted step's Newton iteration count (`iters`, already computed by `_newton_correct`/`_natural_correct` every step) is thrown away; `api.py` hardcodes `"newton_iters": 0` for every point. And `continuation(verbose=True)` accepts the flag but never acts on it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_stepsize.py`, inside `class TestAdaptiveVsFixed:` (after `test_adaptive_handles_difficult_regions`):

```python
    def test_disabled_adaptive_keeps_step_constant_after_success(self):
        """`adaptive=False` must preserve the requested fixed step after
        every successful correction, instead of silently growing/shrinking
        it (2026-08-19 review finding #5)."""
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, adaptive=False, max_steps=200, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        converged_ds = [
            info["ds"] for info in sol._solution.convergence_info[:n]
            if info["converged"]
        ]
        assert len(converged_ds) > 5, "should have several converged fixed steps"
        assert all(ds == pytest.approx(0.01) for ds in converged_ds), (
            f"adaptive=False must keep every successful step at ds=0.01, "
            f"got distinct values {sorted(set(converged_ds))}"
        )
```

Append to `tests/test_functional_api.py` (module already has `import jaxcont as jc` and `import jax.numpy as jnp`):

```python
def test_convergence_info_reports_real_newton_iteration_counts():
    """Regression test for the fabricated newton_iters=0 diagnostic
    (2026-08-19 review finding #10)."""
    def rhs(u, p, args):
        return u ** 3 - p * u

    prob = jc.bif_problem(rhs, u0=jnp.array([0.01]), p0=-1.0)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=50, compute_stability=False),
    )

    iters = [info["newton_iters"] for info in sol._solution.convergence_info]
    assert any(i > 0 for i in iters), (
        "at least one accepted step should report a nonzero Newton iteration "
        "count -- every entry was 0"
    )


def test_verbose_prints_a_bifurcation_summary(capsys):
    """Regression test for the unused verbose=True flag (2026-08-19 review
    finding #10)."""
    def rhs(u, p, args):
        return u ** 2 - p

    prob = jc.bif_problem(rhs, u0=jnp.array([0.5]), p0=0.25)
    jc.continuation(
        prob, jc.PseudoArclength(), p_span=(0.25, -0.25),
        settings=jc.ContinuationPar(ds=0.05, max_steps=50, compute_stability=True),
        events=[jc.Fold()],
        verbose=True,
    )

    captured = capsys.readouterr()
    assert captured.out.strip() != "", "verbose=True should print a summary"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_adaptive_stepsize.py::TestAdaptiveVsFixed::test_disabled_adaptive_keeps_step_constant_after_success tests/test_functional_api.py::test_convergence_info_reports_real_newton_iteration_counts tests/test_functional_api.py::test_verbose_prints_a_bifurcation_summary -v`

Expected: all three FAIL — the step-size test on the constant-`ds` assertion (values grow instead), the iteration-count test on `any(i > 0 ...)` (every entry is `0`), the verbose test on `captured.out.strip() != ""` (nothing was printed).

- [ ] **Step 3: Add `iters` to `ScanResult` and make `_adapt_ds` respect `adaptive`**

In `src/jaxcont/core/scan_continuation.py`, change:

```python
class ScanResult(NamedTuple):
    """Fixed-length buffers from a jitted continuation run."""

    params: Array        # (max_steps + 1,)
    states: Array        # (max_steps + 1, n)
    tangents: Array      # (max_steps + 1, n + 1)
    converged: Array     # (max_steps + 1,) bool  (step accepted)
    ds: Array            # (max_steps + 1,) step size used to reach each point
    n_valid: Array       # scalar int; entries [:n_valid] are real points
```

to:

```python
class ScanResult(NamedTuple):
    """Fixed-length buffers from a jitted continuation run."""

    params: Array        # (max_steps + 1,)
    states: Array        # (max_steps + 1, n)
    tangents: Array      # (max_steps + 1, n + 1)
    converged: Array     # (max_steps + 1,) bool  (step accepted)
    ds: Array            # (max_steps + 1,) step size used to reach each point
    iters: Array         # (max_steps + 1,) Newton iterations used to reach each point
    n_valid: Array       # scalar int; entries [:n_valid] are real points
```

Change `_adapt_ds`:

```python
def _adapt_ds(ds_mag, iters, converged, ds_min, ds_max):
    """Grow ds on fast convergence, shrink on slow/failed — branch-free."""
    grow = ds_mag * 1.5
    shrink_slow = ds_mag * 0.8
    shrink_fail = ds_mag * 0.5
    new = jnp.where(
        converged,
        jnp.where(iters < 3, grow, jnp.where(iters > 6, shrink_slow, ds_mag)),
        shrink_fail,
    )
    return jnp.clip(new, ds_min, ds_max)
```

to:

```python
def _adapt_ds(ds_mag, iters, converged, ds_min, ds_max, adaptive=True):
    """Grow ds on fast convergence, shrink on slow/failed — branch-free.

    When ``adaptive`` is false, a converged step keeps ``ds_mag`` unchanged
    (the caller's requested fixed step) instead of growing/shrinking it. A
    rejected step still backs off by the same ``shrink_fail`` factor as the
    adaptive path either way, so a fixed-step run that cannot converge at
    the requested size still terminates via the existing ``stalled``
    (``ds <= ds_min``) condition below instead of retrying the same failing
    step until ``max_steps`` runs out.
    """
    grow = ds_mag * 1.5
    shrink_slow = ds_mag * 0.8
    shrink_fail = ds_mag * 0.5
    adaptive_choice = jnp.where(
        converged,
        jnp.where(iters < 3, grow, jnp.where(iters > 6, shrink_slow, ds_mag)),
        shrink_fail,
    )
    fixed_choice = jnp.where(converged, ds_mag, shrink_fail)
    new = jnp.where(jnp.asarray(adaptive), adaptive_choice, fixed_choice)
    return jnp.clip(new, ds_min, ds_max)
```

(The `adaptive=True` default keeps the three existing direct-call tests in
`class TestAdaptiveStepsizeAlgorithm` in `tests/test_adaptive_stepsize.py`,
which call `_adapt_ds(...)` positionally with 5 args, passing unmodified.)

- [ ] **Step 4: Thread `iters` and `adaptive` through `pseudo_arclength_scan`**

In `src/jaxcont/core/scan_continuation.py`, in `pseudo_arclength_scan`:

Change the signature (add `adaptive` after `linear_solver`):

```python
def pseudo_arclength_scan(
    f: Callable[[Array, Array], Array],
    u0: Array,
    p0: Array,
    p_end: Array,
    ds0: Array,
    ds_min: Array,
    ds_max: Array,
    tol: Array,
    max_steps: int,
    max_iter: Array,
    linear_solver: LinearSolver = Dense(),
    adaptive: Array = jnp.array(True),
) -> ScanResult:
```

Change the seed correction line (currently discards the seed's iteration count as `_`):

```python
    u0_corrected, seed_converged, _ = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)
```

to:

```python
    u0_corrected, seed_converged, seed_iters = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)
```

Add an `I` buffer next to the existing `D` (ds) buffer:

```python
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
```

to:

```python
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
    I = jnp.zeros((max_steps + 1,), jnp.int32).at[0].set(seed_iters.astype(jnp.int32))
```

Add `I` to the `Carry` NamedTuple (after `D`):

```python
    class Carry(NamedTuple):
        u: Array
        p: Array
        tan: Array
        ds: Array         # positive magnitude; direction lives in the tangent
        idx: Array        # int; number of accepted points so far (write pointer)
        stop: Array       # bool
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array
        I: Array          # Newton iterations used to reach each accepted point
```

In `body`, write `I` alongside `D` and thread `adaptive` into `_adapt_ds`:

```python
        write = c.idx + 1  # slot for the next accepted point
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_new, c.Q[write]))
        T = c.T.at[write].set(jnp.where(converged, tan_new, c.T[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))
```

to:

```python
        write = c.idx + 1  # slot for the next accepted point
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_new, c.Q[write]))
        T = c.T.at[write].set(jnp.where(converged, tan_new, c.T[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))
        I = c.I.at[write].set(jnp.where(converged, iters, c.I[write]).astype(jnp.int32))
```

and change:

```python
        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)
```

to:

```python
        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max, adaptive)
```

then update the `return Carry(...)` at the end of `body`:

```python
        return Carry(u, p, tan, ds, idx, stop, P, Q, T, C, D)
```

to:

```python
        return Carry(u, p, tan, ds, idx, stop, P, Q, T, C, D, I)
```

Update `init` and the final `ScanResult`:

```python
    init = Carry(
        u=u0_seed, p=p0, tan=tan0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,   # +1 for the initial point in slot 0
    )
```

to:

```python
    init = Carry(
        u=u0_seed, p=p0, tan=tan0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D, I=I,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        iters=final.I,
        n_valid=final.idx + 1,   # +1 for the initial point in slot 0
    )
```

- [ ] **Step 5: Mirror Step 4 in `natural_scan`**

In `src/jaxcont/core/scan_continuation.py`, in `natural_scan`, apply the same shape of changes:

Signature — add `adaptive: Array = jnp.array(True)` after `linear_solver: LinearSolver = Dense(),`.

Seed correction — change:

```python
    u0_corrected, seed_converged, _ = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)
```

to:

```python
    u0_corrected, seed_converged, seed_iters = _natural_correct(f, u0, p0, tol, max_iter, linear_solver)
```

`I` buffer — after the existing:

```python
    ds_mag0 = jnp.asarray(ds0, dtype)
    D = jnp.zeros((max_steps + 1,), dtype).at[0].set(ds_mag0)
```

add:

```python
    I = jnp.zeros((max_steps + 1,), jnp.int32).at[0].set(seed_iters.astype(jnp.int32))
```

`Carry` — add `I: Array` after `D: Array`:

```python
    class Carry(NamedTuple):
        u: Array
        p: Array
        ds: Array
        idx: Array
        stop: Array
        P: Array
        Q: Array
        T: Array
        C: Array
        D: Array
        I: Array
```

`body` — change:

```python
        write = c.idx + 1
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_pred, c.Q[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))

        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_pred, c.p)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max)

        reached = jnp.where(direction >= 0, p >= p_end, p <= p_end)
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, ds, idx, stop, P, Q, c.T, C, D)
```

to:

```python
        write = c.idx + 1
        P = c.P.at[write].set(jnp.where(converged, u_new, c.P[write]))
        Q = c.Q.at[write].set(jnp.where(converged, p_pred, c.Q[write]))
        C = c.C.at[write].set(converged)
        D = c.D.at[write].set(jnp.where(converged, c.ds, c.D[write]))
        I = c.I.at[write].set(jnp.where(converged, iters, c.I[write]).astype(jnp.int32))

        u = jnp.where(converged, u_new, c.u)
        p = jnp.where(converged, p_pred, c.p)
        idx = c.idx + converged.astype(c.idx.dtype)

        ds = _adapt_ds(c.ds, iters, converged, ds_min, ds_max, adaptive)

        reached = jnp.where(direction >= 0, p >= p_end, p <= p_end)
        stalled = jnp.logical_and(jnp.logical_not(converged), ds <= ds_min)
        nonfinite = jnp.logical_not(jnp.all(jnp.isfinite(u)))
        stop = jnp.logical_or(reached, jnp.logical_or(stalled, nonfinite))

        return Carry(u, p, ds, idx, stop, P, Q, c.T, C, D, I)
```

`init` and final `ScanResult` — change:

```python
    init = Carry(
        u=u0_seed, p=p0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        n_valid=final.idx + 1,
    )
```

to:

```python
    init = Carry(
        u=u0_seed, p=p0, ds=ds_mag0,
        idx=jnp.array(0, jnp.int32), stop=jnp.array(False),
        P=P, Q=Q, T=T, C=C, D=D, I=I,
    )
    final = lax.while_loop(cond_fun, body, init)

    return ScanResult(
        params=final.Q,
        states=final.P,
        tangents=final.T,
        converged=final.C,
        ds=final.D,
        iters=final.I,
        n_valid=final.idx + 1,
    )
```

- [ ] **Step 6: Pass `settings.adaptive` and real `newton_iters` through `api.py`, implement `verbose`**

In `src/jaxcont/api.py`, in `_run_scan`, change the `scan_fn(...)` call:

```python
    res = scan_fn(
        rhs2,
        u0,
        jnp.asarray(p_start, dtype),
        jnp.asarray(p_end, dtype),
        jnp.asarray(settings.ds, dtype),
        jnp.asarray(settings.ds_min, dtype),
        jnp.asarray(settings.ds_max, dtype),
        jnp.asarray(settings.newton_tol, dtype),
        int(settings.max_steps),
        jnp.asarray(settings.newton_max_iter),
        solvers.linear,
    )
```

to:

```python
    res = scan_fn(
        rhs2,
        u0,
        jnp.asarray(p_start, dtype),
        jnp.asarray(p_end, dtype),
        jnp.asarray(settings.ds, dtype),
        jnp.asarray(settings.ds_min, dtype),
        jnp.asarray(settings.ds_max, dtype),
        jnp.asarray(settings.newton_tol, dtype),
        int(settings.max_steps),
        jnp.asarray(settings.newton_max_iter),
        solvers.linear,
        jnp.asarray(settings.adaptive),
    )
```

Change the `convergence_info` construction:

```python
    convergence_info = [
        {
            "step": i,
            "converged": bool(res.converged[i]),
            "newton_iters": 0,
            "ds": float(res.ds[i]),
        }
        for i in range(n)
    ]
```

to:

```python
    convergence_info = [
        {
            "step": i,
            "converged": bool(res.converged[i]),
            "newton_iters": int(res.iters[i]),
            "ds": float(res.ds[i]),
        }
        for i in range(n)
    ]
```

Finally, implement `verbose`. Find the end of `_run_scan`, just before `return _to_result(sol)`:

```python
        sol.bifurcations = [
            {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
            for h in hits
        ]

    return _to_result(sol)
```

change to:

```python
        sol.bifurcations = [
            {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
            for h in hits
        ]

    if verbose:
        n_converged = sum(1 for c in convergence_info if c["converged"])
        kind_counts: dict[str, int] = {}
        for b in (sol.bifurcations or []):
            kind_counts[b["type"]] = kind_counts.get(b["type"], 0) + 1
        bif_summary = (
            ", ".join(f"{count} {kind}" for kind, count in sorted(kind_counts.items()))
            or "none"
        )
        print(
            f"continuation(): {n} points ({n_converged} converged), "
            f"bifurcations: {bif_summary}"
        )

    return _to_result(sol)
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_adaptive_stepsize.py::TestAdaptiveVsFixed::test_disabled_adaptive_keeps_step_constant_after_success tests/test_functional_api.py::test_convergence_info_reports_real_newton_iteration_counts tests/test_functional_api.py::test_verbose_prints_a_bifurcation_summary -v`

Expected: PASS.

- [ ] **Step 8: Update the two stale "not wired" NOTEs**

In `tests/test_adaptive_stepsize.py`, `test_adaptive_uses_fewer_steps`'s docstring currently reads:

```python
        """
        Test that a looser step-size-bound configuration can use fewer steps
        than a tighter one on a smooth problem.

        NOTE: `adaptive=False` is not wired into the scan engine (same gap
        documented above for the dropped test_disabled_adaptive_returns_same
        -- `_run_scan` never reads `settings.adaptive`, and `_adapt_ds` runs
        unconditionally every step). So `sol_fixed` below is NOT a true
        fixed-step run; both runs actually use the same always-on adaptation.
        What's really being compared is two different (ds, ds_min, ds_max)
        configurations, one of which happens to be labeled "fixed". The
        assertions still hold and are meaningful for that narrower claim.
        """
```

Replace with:

```python
        """
        Test that a looser step-size-bound configuration can use fewer steps
        than a tighter one on a smooth problem, compared against a true
        fixed-step run (`adaptive=False`, wired in via
        test_disabled_adaptive_keeps_step_constant_after_success above).
        """
```

`test_adaptive_handles_difficult_regions`'s docstring has an analogous `NOTE: as in test_adaptive_uses_fewer_steps above, ...` paragraph at its end — delete that paragraph, keeping the rest of the docstring (the "Reformulated from the pre-migration version..." explanation above it stays; it documents an unrelated, still-true limitation of what the engine exposes).

- [ ] **Step 9: Run the full adaptive-stepsize and functional-API suites**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_adaptive_stepsize.py tests/test_functional_api.py -v`

Expected: all PASS. `adaptive=False` is now a true fixed step, so
`test_adaptive_uses_fewer_steps` and `test_adaptive_handles_difficult_regions`
compare a real fixed-step run against a real adaptive one for the first
time — their existing numeric assertions (`n_valid` bounds) were written
generously enough to hold either way, but if either fails, tighten or
loosen only the specific numeric bound that fails, in this task's commit,
with a one-line comment on why (preserving the test's original intent:
"adaptive should not use significantly more steps than fixed" /
"adaptive should reach at least as far as fixed in a difficult region").
Do not weaken an assertion to merely match whatever the implementation
happens to produce.

- [ ] **Step 10: Run the full suite**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' -n auto`

Expected: same pass/skip/xfail counts as the pre-task baseline, plus the 3 new tests passing.

- [ ] **Step 11: Update CHANGELOG.md**

In `CHANGELOG.md`, under the existing `## [Unreleased]` / `### Fixed` section, add two bullets:

```markdown
- `ContinuationPar(adaptive=False)` now actually disables step-size adaptation: a
  successful step keeps the requested `ds` unchanged instead of silently growing it. A
  failed fixed-size step still backs off (and the run still terminates via the existing
  `ds <= ds_min` stall condition) rather than retrying forever.
- `convergence_info` entries now report the real Newton iteration count for each accepted
  step instead of a hardcoded `0`, and `continuation(..., verbose=True)` now prints a
  one-line summary instead of doing nothing.
```

- [ ] **Step 12: Commit**

```bash
git add src/jaxcont/core/scan_continuation.py src/jaxcont/api.py \
  tests/test_adaptive_stepsize.py tests/test_functional_api.py CHANGELOG.md
git commit -m "fix: wire ContinuationPar.adaptive through the scan engines, report real Newton iteration counts, implement verbose"
```

---

## Task 2: Versioned, pickle-free solution persistence

**Files:**
- Modify: `src/jaxcont/core/continuation.py` (`ContinuationSolution.save`/`.load`)
- Create: `tests/test_solution_persistence.py`
- Modify: `CHANGELOG.md` (`[Unreleased]` section)

**Interfaces:**
- Consumes: `ContinuationSolution`'s existing field set (`states`, `parameters`, `eigenvalues`, `stability`, `bifurcations`, `tangent_vectors`, `convergence_info`, `state_names`, `param_name`) — unchanged, no dataclass field changes.
- Produces: `ContinuationSolution.save(filename)` writes a `format_version=1` `.npz` archive; `ContinuationSolution.load(filename)` reads it back with `allow_pickle=False` and raises `ValueError` on a missing/unsupported `format_version`. This is a clean break from the previous (broken) format — no repository file, test, or doc relies on the old format, so no migration path is needed.

The current `save()` passes `eigenvalues=None`/`stability=None` straight into `np.savez(**data)`, which NumPy stores as a 0-d `object` array *containing* `None` rather than omitting the key — so `load()`'s `data["eigenvalues"] is not None` check is always `True` (it's an array, not `None`), and the subsequent `jnp.array(...)` on that object array raises `TypeError`. `load()` also always passes `allow_pickle=True`, and neither method preserves `tangent_vectors`, `convergence_info`, `state_names`, or `param_name` at all.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_solution_persistence.py`:

```python
"""
Tests for ContinuationSolution.save()/.load() (2026-08-19 review finding #7:
persistence was broken for the common case of optional fields set to None,
and unsafe by defaulting to allow_pickle=True).
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc


def _sample_solution(*, with_optional_fields: bool) -> jc.ContinuationSolution:
    states = jnp.array([[0.0], [0.1], [0.2]])
    parameters = jnp.array([0.0, 0.5, 1.0])
    if not with_optional_fields:
        return jc.ContinuationSolution(states=states, parameters=parameters)
    return jc.ContinuationSolution(
        states=states,
        parameters=parameters,
        eigenvalues=jnp.array([[-1.0], [-0.5], [0.1]]),
        stability=jnp.array([True, True, False]),
        tangent_vectors=jnp.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        bifurcations=[
            {
                "type": "fold",
                "parameter": 0.9,
                "state": jnp.array([0.19]),
                "index": 2,
                "null_vector": jnp.array([1.0, 0.0]),
            }
        ],
        convergence_info=[
            {"step": 0, "converged": True, "newton_iters": 3, "ds": 0.1},
            {"step": 1, "converged": True, "newton_iters": 2, "ds": 0.1},
        ],
        state_names=("x",),
        param_name="p",
    )


def test_round_trip_preserves_all_fields(tmp_path):
    sol = _sample_solution(with_optional_fields=True)
    path = tmp_path / "sol.npz"
    sol.save(str(path))
    loaded = jc.ContinuationSolution.load(str(path))

    assert jnp.allclose(loaded.states, sol.states)
    assert jnp.allclose(loaded.parameters, sol.parameters)
    assert jnp.allclose(loaded.eigenvalues, sol.eigenvalues)
    assert jnp.array_equal(loaded.stability, sol.stability)
    assert jnp.allclose(loaded.tangent_vectors, sol.tangent_vectors)
    assert loaded.state_names == sol.state_names
    assert loaded.param_name == sol.param_name
    assert loaded.convergence_info == sol.convergence_info

    assert len(loaded.bifurcations) == 1
    bif = loaded.bifurcations[0]
    assert bif["type"] == "fold"
    assert bif["parameter"] == pytest.approx(0.9)
    assert bif["index"] == 2
    assert bif["state"] == pytest.approx([0.19])
    assert bif["null_vector"] == pytest.approx([1.0, 0.0])


def test_round_trip_with_none_optional_fields_does_not_raise(tmp_path):
    """Direct regression test for the reviewed TypeError: saving/loading a
    solution with eigenvalues=None/stability=None/tangent_vectors=None must
    round-trip those fields as None, not crash."""
    sol = _sample_solution(with_optional_fields=False)
    assert sol.eigenvalues is None
    assert sol.stability is None
    assert sol.tangent_vectors is None

    path = tmp_path / "sol.npz"
    sol.save(str(path))
    loaded = jc.ContinuationSolution.load(str(path))

    assert loaded.eigenvalues is None
    assert loaded.stability is None
    assert loaded.tangent_vectors is None
    assert jnp.allclose(loaded.states, sol.states)


def test_load_rejects_a_file_with_no_format_version(tmp_path):
    path = tmp_path / "not_a_solution.npz"
    np.savez(path, states=np.zeros((1, 1)), parameters=np.zeros((1,)))
    with pytest.raises(ValueError, match="format_version"):
        jc.ContinuationSolution.load(str(path))


def test_load_rejects_pickled_array_payloads(tmp_path):
    """allow_pickle=False must be the load() default: an untrusted archive
    that smuggles a pickled object into a field this code actually reads
    must fail to load, not silently execute the pickle."""
    path = tmp_path / "malicious.npz"
    metadata = json.dumps(
        {
            "bifurcations": [],
            "convergence_info": None,
            "state_names": None,
            "param_name": None,
        }
    )
    np.savez(
        path,
        format_version=np.array(1),
        states=np.array([object()], dtype=object),
        parameters=np.zeros((1,)),
        metadata_json=np.array(metadata),
    )
    with pytest.raises(ValueError):
        jc.ContinuationSolution.load(str(path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_solution_persistence.py -v`

Expected: `test_round_trip_preserves_all_fields` and
`test_round_trip_with_none_optional_fields_does_not_raise` FAIL (missing
fields / `TypeError` from the `None`-as-object-array bug);
`test_load_rejects_a_file_with_no_format_version` and
`test_load_rejects_pickled_array_payloads` FAIL because `load()` doesn't
look for `format_version` yet and happily loads pickled payloads today.

- [ ] **Step 3: Replace `save()`/`load()`**

In `src/jaxcont/core/continuation.py`, add this module-level helper above the
`ContinuationSolution` class (after the existing imports):

```python
def _json_safe(value):
    """Recursively convert JAX/NumPy arrays into plain Python lists so a
    value is safe to pass to ``json.dumps`` — used for the metadata blob
    :meth:`ContinuationSolution.save` embeds in its ``.npz`` archive."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value
```

Then replace the existing `save`/`load` methods:

```python
    def save(self, filename: str):
        """
        Save solution to file.
        
        Args:
            filename: Path to save file
        """
        import numpy as np
        data = {
            "states": np.array(self.states),
            "parameters": np.array(self.parameters),
            "eigenvalues": np.array(self.eigenvalues) if self.eigenvalues is not None else None,
            "stability": np.array(self.stability) if self.stability is not None else None,
            "bifurcations": self.bifurcations,
        }
        np.savez(filename, **data)
    
    @classmethod
    def load(cls, filename: str) -> "ContinuationSolution":
        """
        Load solution from file.
        
        Args:
            filename: Path to load file
        
        Returns:
            ContinuationSolution object
        """
        import numpy as np
        data = np.load(filename, allow_pickle=True)
        return cls(
            states=jnp.array(data["states"]),
            parameters=jnp.array(data["parameters"]),
            eigenvalues=jnp.array(data["eigenvalues"]) if data["eigenvalues"] is not None else None,
            stability=jnp.array(data["stability"]) if data["stability"] is not None else None,
            bifurcations=list(data["bifurcations"]) if "bifurcations" in data else [],
        )
```

with:

```python
    def save(self, filename: str):
        """
        Save this solution to ``filename`` (an ``.npz`` archive), using a
        versioned, pickle-free schema (format_version=1).

        Numeric arrays (states, parameters, tangent_vectors, eigenvalues,
        stability) are stored as native NumPy arrays; an optional array
        that is ``None`` is simply omitted from the archive rather than
        stored as ``None`` -- NumPy has no null array value, and storing
        ``None`` via ``np.array(None)`` both requires pickling on load and
        round-trips as a 0-d object array, not ``None``. Non-numeric
        metadata (bifurcations, convergence_info, state_names, param_name)
        is JSON-encoded into a single ``metadata_json`` entry, which NumPy
        stores as a plain unicode array -- no pickling needed there either.

        Args:
            filename: Path to save file
        """
        import json

        import numpy as np

        arrays: dict[str, Any] = {"format_version": np.array(1)}
        arrays["states"] = np.asarray(self.states)
        arrays["parameters"] = np.asarray(self.parameters)
        if self.tangent_vectors is not None:
            arrays["tangent_vectors"] = np.asarray(self.tangent_vectors)
        if self.eigenvalues is not None:
            arrays["eigenvalues"] = np.asarray(self.eigenvalues)
        if self.stability is not None:
            arrays["stability"] = np.asarray(self.stability)

        metadata = {
            "bifurcations": _json_safe(self.bifurcations),
            "convergence_info": _json_safe(self.convergence_info),
            "state_names": list(self.state_names) if self.state_names is not None else None,
            "param_name": self.param_name,
        }
        arrays["metadata_json"] = np.array(json.dumps(metadata))

        np.savez(filename, **arrays)

    @classmethod
    def load(cls, filename: str) -> "ContinuationSolution":
        """
        Load a solution saved by :meth:`save`.

        Reads with ``allow_pickle=False``: the format_version=1 schema
        never needs pickling, so this refuses to execute arbitrary pickled
        payloads from an untrusted ``.npz`` file rather than trusting them
        by default.

        Args:
            filename: Path to load file

        Returns:
            ContinuationSolution object
        """
        import json

        import numpy as np

        with np.load(filename, allow_pickle=False) as data:
            if "format_version" not in data.files:
                raise ValueError(
                    f"{filename!r} has no 'format_version' entry -- it is not "
                    f"a ContinuationSolution.save() archive (or predates this "
                    f"schema) and cannot be loaded."
                )
            version = int(data["format_version"])
            if version != 1:
                raise ValueError(
                    f"{filename!r} uses save/load format_version={version}, "
                    f"but this version of JaxCont only supports version 1."
                )

            metadata = json.loads(data["metadata_json"].item())
            state_names = metadata["state_names"]

            return cls(
                states=jnp.array(data["states"]),
                parameters=jnp.array(data["parameters"]),
                tangent_vectors=(
                    jnp.array(data["tangent_vectors"])
                    if "tangent_vectors" in data.files else None
                ),
                eigenvalues=(
                    jnp.array(data["eigenvalues"])
                    if "eigenvalues" in data.files else None
                ),
                stability=(
                    jnp.array(data["stability"])
                    if "stability" in data.files else None
                ),
                bifurcations=metadata["bifurcations"],
                convergence_info=metadata["convergence_info"],
                state_names=tuple(state_names) if state_names is not None else None,
                param_name=metadata["param_name"],
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_solution_persistence.py -v`

Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' -n auto`

Expected: same pass/skip/xfail counts as the pre-task baseline, plus the 4 new tests passing.

- [ ] **Step 6: Update CHANGELOG.md**

In `CHANGELOG.md`, under `## [Unreleased]`, add a `### Changed` bullet (create the subsection if `### Fixed` is currently the only one under `[Unreleased]`) and a `### Fixed` bullet:

```markdown
### Changed
- **Breaking:** `ContinuationSolution.save()`/`.load()` now use a versioned (`format_version=1`),
  pickle-free `.npz` schema. Archives written by the previous implementation cannot be loaded
  (the previous format was broken for any solution with an optional field left as `None` -- see
  Fixed, below -- so no working archives from it exist to migrate).

### Fixed
- `ContinuationSolution.save()`/`.load()` no longer raises `TypeError` when `eigenvalues`,
  `stability`, or `tangent_vectors` is `None`; these fields now round-trip correctly, along with
  `convergence_info`, `state_names`, and `param_name`, none of which the previous format
  preserved at all.
- `ContinuationSolution.load()` now defaults to `allow_pickle=False`, so loading an untrusted
  `.npz` file can no longer trigger arbitrary pickle execution.
```

- [ ] **Step 7: Commit**

```bash
git add src/jaxcont/core/continuation.py tests/test_solution_persistence.py CHANGELOG.md
git commit -m "fix: replace unsafe pickle-based solution persistence with a versioned, JSON-safe schema"
```

---

## Task 3: Clearly mark the BVP placeholder API as unimplemented

**Files:**
- Modify: `src/jaxcont/problems/bvp.py` (`BoundaryValueProblem` class docstring)
- Modify: `docs/source/development.rst` (BVP bullet)
- Test: `tests/test_bvp_placeholder.py` (new)
- Modify: `CHANGELOG.md` (`[Unreleased]` section)

**Interfaces:**
- Consumes: nothing new — `BoundaryValueProblem` keeps its existing dataclass fields (`rhs`, `boundary_conditions`, `params`, `t_span`, `initial_guess`) and both methods keep raising `NotImplementedError`, unchanged.
- Produces: nothing new is exported. This task only changes docstrings/docs text and adds a regression test locking in today's documented-placeholder behavior.

`BoundaryValueProblem` is exported from `jaxcont.problems` (`from jaxcont.problems import BoundaryValueProblem` works today, per `src/jaxcont/problems/__init__.py`'s `__all__`) but its class-level docstring gives no indication that `solve_collocation`/`solve_shooting` are unimplemented — only an inline comment inside each method body says so, which a user only sees by opening the source, not `help(BoundaryValueProblem)`. `docs/source/development.rst` lists it as "For BVP formulations" alongside real, working problem types, with no caveat.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bvp_placeholder.py`:

```python
"""
Regression test locking in BoundaryValueProblem's documented placeholder
status (2026-08-19 review finding #13): solve_collocation/solve_shooting
must keep raising NotImplementedError, and the class docstring must say so
up front, until a real implementation lands.
"""

import jax.numpy as jnp
import pytest

from jaxcont.problems.bvp import BoundaryValueProblem


def _sample_problem() -> BoundaryValueProblem:
    return BoundaryValueProblem(
        rhs=lambda t, u, params: u,
        boundary_conditions=lambda u0, uT: u0 - uT,
        params={},
        t_span=(0.0, 1.0),
        initial_guess=jnp.zeros(2),
    )


def test_class_docstring_states_placeholder_status():
    assert "placeholder" in BoundaryValueProblem.__doc__.lower()


def test_solve_collocation_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _sample_problem().solve_collocation()


def test_solve_shooting_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _sample_problem().solve_shooting()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_bvp_placeholder.py -v`

Expected: `test_class_docstring_states_placeholder_status` FAILS (the current
docstring never says "placeholder"); the two `NotImplementedError` tests
already PASS today (that behavior is correct and pre-existing) — leave
them as-is, they exist to make sure this task never accidentally changes it.

- [ ] **Step 3: Update the class docstring**

In `src/jaxcont/problems/bvp.py`, change:

```python
@dataclass
class BoundaryValueProblem:
    """
    Two-point boundary value problem.
    
    Solve: du/dt = f(t, u, params)
    Subject to: g(u(0), u(T)) = 0
    
    Attributes:
        rhs: Right-hand side f(t, u, params)
        boundary_conditions: Boundary condition function g(u0, uT)
        params: System parameters
        t_span: Time span (t0, tF)
        initial_guess: Initial guess for solution
    """
```

to:

```python
@dataclass
class BoundaryValueProblem:
    """
    Two-point boundary value problem.

    **Status: unimplemented placeholder.** This dataclass exists to fix the
    intended problem-definition shape, but ``solve_collocation`` and
    ``solve_shooting`` both always raise ``NotImplementedError`` -- there is
    no working BVP solver in JaxCont yet. Constructing this class is safe;
    calling either solve method is not.

    Solve: du/dt = f(t, u, params)
    Subject to: g(u(0), u(T)) = 0
    
    Attributes:
        rhs: Right-hand side f(t, u, params)
        boundary_conditions: Boundary condition function g(u0, uT)
        params: System parameters
        t_span: Time span (t0, tF)
        initial_guess: Initial guess for solution
    """
```

- [ ] **Step 4: Update `docs/source/development.rst`**

Change:

```rst
- ``EquilibriumProblem``: For finding equilibrium points
- ``periodic_orbit_problem``: For periodic orbit continuation
- ``BoundaryValueProblem``: For BVP formulations
```

to:

```rst
- ``EquilibriumProblem``: For finding equilibrium points
- ``periodic_orbit_problem``: For periodic orbit continuation
- ``BoundaryValueProblem``: BVP problem definition only -- an unimplemented
  placeholder; ``solve_collocation``/``solve_shooting`` always raise
  ``NotImplementedError``
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' tests/test_bvp_placeholder.py -v`

Expected: PASS (all 3 tests).

- [ ] **Step 6: Run the full suite**

Run: `PYTHONPATH=<worktree-path>/src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest -q -m '' -n auto`

Expected: same pass/skip/xfail counts as the pre-task baseline, plus the 3 new tests passing.

- [ ] **Step 7: Update CHANGELOG.md**

In `CHANGELOG.md`, under `## [Unreleased]` / `### Fixed`, add:

```markdown
- `BoundaryValueProblem`'s class docstring now states up front that it is an unimplemented
  placeholder (`solve_collocation`/`solve_shooting` always raise `NotImplementedError`),
  instead of only saying so in an inline comment inside each method body.
```

- [ ] **Step 8: Commit**

```bash
git add src/jaxcont/problems/bvp.py docs/source/development.rst \
  tests/test_bvp_placeholder.py CHANGELOG.md
git commit -m "docs: clearly mark BoundaryValueProblem as an unimplemented placeholder"
```
