# LinearSolver / EigenSolver Protocols — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** v0.2 engineering-prep item 5 (`notes/ROADMAP.md`, "Engineering /
architecture recommendations for v0.2").

## Motivation

Every linear solve and eigendecomposition in JaxCont's live continuation path is a
hardcoded `jnp.linalg.solve`/`jnp.linalg.eigvals` call. ARCHITECTURE.md §3's
"GPU-ready" / "matrix-free, iterative solvers" claims are aspirational until there is
an actual seam to swap in something other than dense LAPACK calls. ARCHITECTURE.md
§4.6 already sketches the target shape (`LinearSolver`/`EigenSolver` protocols, a
`Dense()` default, a `Solvers` bundle passed to `continuation()`); this item makes
that sketch real, with `Dense()` as the *only* concrete implementation for now —
behavior must be numerically identical to today's hardcoded calls.

This also makes ARCHITECTURE.md §10.2's reserved DDE eigensolver seam
(`ChebyshevDDE()` swapped in for `Dense()`, continuation loop untouched) and any
future large-system/GPU work (`GMRES()`/`Arnoldi()`) a pure solver swap instead of a
continuation-loop change.

## Scope

**In scope:** the live jitted engine in `src/jaxcont/core/scan_continuation.py`
(`pseudo_arclength_scan`, `natural_scan`, and the private helpers they call:
`_tangent`, `_newton_correct`, `_natural_correct`, `branch_eigenvalues`), plus the
public `continuation()` entry point in `src/jaxcont/api.py`.

**Out of scope (explicit, per roadmap text and current codebase state):**
- `src/jaxcont/solvers/implicit.py` (`differentiable_root`, used by
  `fold_solve.py`'s `Fold` event refinement) — a separate concern (post-hoc fold
  refinement, not the continuation loop), and the site of a recent tracer-leak fix;
  touching it again is not justified by this item's stated motivation.
- `src/jaxcont/stability/eigenvalue.py`, `src/jaxcont/stability/floquet.py`,
  `src/jaxcont/bifurcations/period_doubling.py` — not called from any live path
  today (no import outside their own tests); reserved for future periodic-orbit
  work. Adding solver plumbing to unused code is premature.
- A `NewtonSolver` protocol — ARCHITECTURE.md §4.6 sketches one, but the roadmap
  item only asks for `LinearSolver`/`EigenSolver`, and the corrector algorithm
  itself is already selected via `PseudoArclength()`/`Natural()`. Adding it now
  would be scope creep beyond what's requested.
- A new gallery example script — see "Testing", below, for why.

## Design

### 1. New module: `src/jaxcont/solvers/protocols.py`

```python
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jax import Array


@runtime_checkable
class LinearSolver(Protocol):
    def __call__(self, A: Array, b: Array) -> Array: ...


@runtime_checkable
class EigenSolver(Protocol):
    def __call__(self, A: Array) -> Array: ...


@dataclass(frozen=True)
class Dense:
    """Default LinearSolver: jnp.linalg.solve. Zero fields -- value-hashable,
    safe as a jax.jit static argument (see scan_continuation.py wiring)."""

    def __call__(self, A: Array, b: Array) -> Array:
        return jnp.linalg.solve(A, b)


@dataclass(frozen=True)
class DenseEigen:
    """Default EigenSolver: jnp.linalg.eigvals. Zero fields -- value-hashable,
    safe as a jax.jit static argument."""

    def __call__(self, A: Array) -> Array:
        return jnp.linalg.eigvals(A)
```

**Naming deviation from ARCHITECTURE.md §4.6:** the doc's sketch reuses `Dense()`
as the default for both `linear` and `eigen` fields. A single class cannot cleanly
satisfy both protocols (`__call__(self, A, b)` vs. `__call__(self, A)` are
different signatures; branching on whether `b` was passed is a bad way to encode
"which protocol am I right now"). This spec uses two honestly-named classes,
`Dense` (linear) and `DenseEigen` (eigen). ARCHITECTURE.md §4.6 will be updated to
match as part of this work's documentation task.

**Why plain `@dataclass(frozen=True)`, not `eqx.Module`:** `Dense`/`DenseEigen`
have no fields. A plain frozen dataclass gets value-based `__eq__`/`__hash__` from
the dataclass machinery for free — two independently-constructed `Dense()`
instances compare equal and hash equal. This matters because they are passed as
`jax.jit` **static** arguments (see below): if equality were identity-based (the
default for a bare class, or for an `eqx.Module` without a matching `__eq__`),
every fresh `Dense()` — e.g. the shared default on `Solvers` — could still work by
sharing one instance, but any caller constructing a fresh `Dense()` explicitly
would force a spurious recompile. Value equality avoids that footgun entirely.

### 2. Wiring into `src/jaxcont/core/scan_continuation.py`

- `_tangent(f, u, p, prev_tangent, linear_solver: LinearSolver = Dense())` —
  replace `jnp.linalg.solve(M, rhs)` with `linear_solver(M, rhs)`.
- `_newton_correct(..., linear_solver: LinearSolver = Dense())` — same swap for
  its `jnp.linalg.solve(M, rhs)`.
- `_natural_correct(..., linear_solver: LinearSolver = Dense())` — same swap for
  its `jnp.linalg.solve(jac_u, -f_val)`.
- `branch_eigenvalues(f, states, params, eigen_solver: EigenSolver = DenseEigen())`
  — replace `jnp.linalg.eigvals(...)` with `eigen_solver(...)`.
- `pseudo_arclength_scan(..., linear_solver: LinearSolver = Dense())` and
  `natural_scan(..., linear_solver: LinearSolver = Dense())` — both gain this as a
  **trailing** parameter (positional index 10, after today's `max_iter` at index
  9), added to their `@partial(jax.jit, static_argnums=(0, 8))` decorator as
  `static_argnums=(0, 8, 10)`. Trailing + defaulted means every existing
  10-positional-arg call site (examples, `test_functional_api.py`,
  `test_gpu_smoke.py`) keeps working unmodified.
- `_tangent`/`_newton_correct`/`_natural_correct` are called from inside the
  `body` closures of `pseudo_arclength_scan`/`natural_scan`; those call sites pass
  the outer function's `linear_solver` through.

### 3. Public surface: `src/jaxcont/api.py`

```python
@dataclass(frozen=True)
class Solvers:
    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()
```

`continuation()` gains `solvers: Solvers = Solvers()`, threaded through
`_run_scan`/`_run_scan_traced`:
- `scan_fn(..., linear_solver=solvers.linear)` (the `pseudo_arclength_scan`/
  `natural_scan` call).
- `branch_eigenvalues(rhs2, states, params, eigen_solver=solvers.eigen)` (both the
  eager and traced reassembly paths call this).

`Solvers` is a plain Python-level config object (like `ContinuationPar` already
is) — it is never itself passed across a `jax.jit` boundary; only its `.linear`/
`.eigen` fields are, as the jit-static arguments described above.

### 4. Exports

- `src/jaxcont/solvers/__init__.py`: add `LinearSolver`, `EigenSolver`, `Dense`,
  `DenseEigen` to the existing imports and `__all__` (alongside `NewtonSolver`,
  `Corrector` — note `solvers/newton.py`'s `NewtonSolver` is an unrelated
  pre-existing concrete class, not the ARCHITECTURE.md §4.6 `NewtonSolver`
  protocol sketch, which is out of scope per above).
- `src/jaxcont/api.py`: add `Solvers` to `__all__`.
- `src/jaxcont/__init__.py`: import and re-export `Solvers` from `jaxcont.api`
  alongside `PseudoArclength`/`Natural`; import and re-export `LinearSolver`,
  `EigenSolver`, `Dense`, `DenseEigen` from `jaxcont.solvers`.

## Testing

No new gallery example script. `Dense()` reproduces today's exact numerics, so
there is nothing new to *demonstrate* visually — every existing example already
exercises the protocol invisibly the moment it lands, the same way the
`differentiable_root` extraction and Event protocol rewrite (this item's two
predecessors) shipped without new example scripts, since both were internal
architecture changes with no new user-facing behavior.

What needs proving instead is that the seam is *real* — that swapping `solvers=`
actually changes what runs, not just decorative wrapping around the same hardcoded
calls. Required tests:

1. **Custom-solver routing test:** a test-local `LinearSolver` implementation that
   counts calls (or records the arguments it was invoked with) plugged into
   `Solvers(linear=...)`, passed to `continuation()`; assert the custom solver was
   actually invoked (call count > 0) and that swapping it changes nothing about
   correctness (the counting solver still calls `jnp.linalg.solve` internally, so
   results match the `Dense()` baseline). Same pattern for a custom `EigenSolver`
   plugged into `Solvers(eigen=...)`.
2. **Numerical-equivalence regression test:** for representative `A`/`b`
   matrices, assert `Dense()(A, b) == jnp.linalg.solve(A, b)` and
   `DenseEigen()(A) == jnp.linalg.eigvals(A)` exactly (`jnp.array_equal`), plus a
   full-branch test running a known problem (e.g. the pitchfork from
   `test_functional_api.py`) through `pseudo_arclength_scan` with the default
   `Dense()`/`DenseEigen()` and asserting identical `states`/`params`/
   `eigenvalues` to a captured pre-change baseline. Guards against a future
   accidental behavior change in `Dense`/`DenseEigen`.
3. **`jax.jit` static-argument correctness:** a test that calls
   `pseudo_arclength_scan` twice with two independently-constructed `Dense()`
   instances (`Dense()` and `Dense()`, not the same object) and confirms no
   error and matching results — this is what exercises the value-equality
   requirement from the design section above. A `jax.vmap` test analogous to
   `TestScanEngine.test_vmap_batch` in `test_functional_api.py`, run with the new
   trailing `linear_solver` argument passed explicitly, is also required (existing
   vmap coverage only exercises the *default*-args path).
4. **Full existing suite green**, including `test_functional_api.py`,
   `test_bordered_newton.py`, `test_gpu_smoke.py`, and a headless re-run of
   `example_01_pitchfork.py`, `example_02_lorenz.py`, `example_05_neural_mass.py`,
   `example_06_vmap_sweep.py`, `example_07_differentiable.py` confirming identical
   output to before this change (in particular, `example_02`/`05`'s
   BifurcationKit.jl comparison tables must still match).

## Global Constraints

- `Dense()`/`DenseEigen()` must be numerically identical to today's hardcoded
  `jnp.linalg.solve`/`jnp.linalg.eigvals` calls — this is a pure refactor of the
  solve boundary, not a numerics change.
- All new function parameters are trailing and defaulted; no existing call site
  (library code, tests, or examples) may require modification to keep working.
- `Dense`/`DenseEigen` must be `@dataclass(frozen=True)` with no fields, to get
  value-based `__eq__`/`__hash__` and remain valid as `jax.jit` static arguments.
- `Solvers` is a plain Python-level dataclass (like `ContinuationPar`), never
  itself crossing a `jax.jit` boundary.
- No `NewtonSolver` protocol, no new example script, no changes to
  `solvers/implicit.py`, `stability/eigenvalue.py`, `stability/floquet.py`, or
  `bifurcations/period_doubling.py` in this item.
