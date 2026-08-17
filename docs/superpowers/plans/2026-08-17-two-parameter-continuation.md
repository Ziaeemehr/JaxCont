# Two-Parameter Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trace fold and Hopf curves in two parameters and detect codim-2 points (CP/BT/ZH/GH/HH) along them, making the shipped direct codim-2 solvers reachable without a hand-supplied guess.

**Architecture:** No new engine. A fold curve is the existing fold extended system (`bifurcations/fold_solve.py`) of a *reduced* one-parameter problem, continued in the second parameter — an ordinary `F(X, q) = 0` that `core/scan_continuation.py:pseudo_arclength_scan` already solves. Two pure factories return ordinary `BifProblem`s (the `periodic_orbit_problem` pattern), so `jit`/`vmap`/`grad` come for free. Codim-2 events are `Event` implementations that carry their own `raw_f` and recompute the *original* system's spectrum.

**Tech Stack:** JAX 0.11.0 (float32 default, CUDA), pytest, matplotlib. Cross-validation against MatCont 7.6 (MATLAB R2020a) and BifurcationKit.jl v0.5.2.

**Spec:** `docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md`

**Branch:** `feat/two-parameter-continuation` (already created; the spec is committed there as `3dac60b`).

## Global Constraints

- **Python env:** use `/home/ziaee/envs/jaxcont/bin/python`. The system `python`/`python3` is a different install and cannot import JAX.
- **Run tests with** `JAX_PLATFORMS=cpu` for determinism, e.g. `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`. Verify the full suite is green after every task (baseline: 239 passed as of v0.3.1).
- **float32 is the default dtype.** Never set `jax_enable_x64=True`, not even temporarily in a prototype script — a stray enable contaminated the periodic-orbit verification numbers once already.
- **Never set a `tol`/`newton_tol` below ~`1e-6`.** ROADMAP issue #12: a tolerance under the float32 floor reports `converged=False` forever with no error raised. Measure the achievable floor; do not inherit one from another system.
- **`p` has shape `(2,)` throughout this feature**, matching `bifurcations/codim2.py`'s existing convention. `f` keeps the `f(u, p, args)` signature; only `p`'s shape changes.
- **`free` is a Python `int` (0 or 1), captured at factory time — always static**, never a traced value.
- **Commit after every task** with the message given in that task's final step.

---

### Task 1: Fold-curve factory

**Files:**
- Create: `src/jaxcont/bifurcations/curves.py`
- Modify: `src/jaxcont/api.py` (widen `BifProblem.kind`; add two guards in `_run_scan`)
- Test: `tests/test_curves.py`

**Interfaces:**
- Consumes: `fold_solve._extended_residual(x, f, args, n)`, `fold_solve._pack(u, p, v)`, `fold_solve.fold_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, v)`.
- Produces:
  - `fold_curve_problem(f, u_guess, p_guess, *, free=1, args=None, tol=1e-6, max_iter=50) -> BifProblem` with `kind="fold_curve"`, `u0` the packed `(2n+1,)` state `(u, p_fixed, v)`, `p0 = p_guess[free]`.
  - `_assemble_p(p_fixed, q, free) -> Array` of shape `(2,)`.
  - `unpack_fold_curve(X, n) -> (u, p_fixed, v)`.

**Note — spec correction.** Spec §3 says the *factory* validates `p_span[0] == p_guess[free]`. It cannot: the factory never sees `p_span`, which is an argument to `continuation()`. The check belongs in `api.py:_run_scan`. Step 7 fixes the spec text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curves.py`:

```python
"""Two-parameter curve factories (bifurcations/curves.py)."""

import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.bifurcations.curves import fold_curve_problem, unpack_fold_curve


def _cusp_shifted(u, p, args):
    """u' = (p0-0.3) + (p1-1.2)*(u-0.7) - (u-0.7)**3.

    The cusp normal form, affinely shifted off the origin (origin-centred
    normal forms have no discriminating power -- a stub returning zeros
    passes them all). Its fold set is the exact discriminant
        27*(p0-0.3)**2 == 4*(p1-1.2)**3
    derived by eliminating u from f=0 and df/du=0.
    """
    x = u[0] - 0.7
    a = p[0] - 0.3
    b = p[1] - 1.2
    return jnp.array([a + b * x - x**3])


def _cusp_discriminant(p0, p1):
    return 27.0 * (p0 - 0.3) ** 2 - 4.0 * (p1 - 1.2) ** 3


def test_fold_curve_traces_the_exact_cusp_discriminant():
    # Seed: b = p1-1.2 = 3  ->  x = sqrt(b/3) = 1 -> u = 1.7,
    # a = p0-0.3 = -2*x*b/3 = -2 -> p0 = -1.7.
    prob = fold_curve_problem(
        _cusp_shifted,
        jnp.array([1.7]),
        jnp.array([-1.7, 4.2]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 6.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _ = unpack_fold_curve(sol.branch.states[i], n=1)
        assert abs(_cusp_discriminant(float(p0), p1)) < 1e-2


def test_fold_curve_problem_rejects_wrong_p_shape():
    with pytest.raises(ValueError, match="shape"):
        fold_curve_problem(_cusp_shifted, jnp.array([1.7]), jnp.array([-1.7]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.bifurcations.curves'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/bifurcations/curves.py`:

```python
"""
Two-parameter continuation of codim-1 curves (fold curve, Hopf curve).

A fold of ``f(u, p) = 0`` is the root of the extended system in
``fold_solve.py``. A fold *curve* is that same system continued in a second
parameter. With ``X = (u, p_fixed, v)`` and continuation parameter
``q = p[free]``, ``F(X, q) = 0`` is an ordinary residual in a scalar
parameter -- exactly what ``core/scan_continuation.py`` already solves.

The reduction that makes this free: define
``f_reduced(u, p_fixed, args) = f(u, assemble(p_fixed, q), args)``. From
``f_reduced``'s point of view ``p_fixed`` is an ordinary scalar parameter,
so ``fold_solve._extended_residual`` applies UNCHANGED. Same trick that let
periodic-orbit collocation reuse the equilibrium engine.

See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array

from jaxcont.api import BifProblem
from jaxcont.bifurcations.fold_solve import (
    _extended_residual as _fold_extended_residual,
)
from jaxcont.bifurcations.fold_solve import _pack as _fold_pack
from jaxcont.bifurcations.fold_solve import fold_point

PyTree = Any


def _assemble_p(p_fixed: Array, q: Array, free: int) -> Array:
    """Rebuild the shape-(2,) parameter vector from its solved and
    continued components. ``free`` is a Python int, so this is static."""
    parts = [None, None]
    parts[free] = jnp.reshape(q, ())
    parts[1 - free] = jnp.reshape(p_fixed, ())
    return jnp.stack(parts)


def _validate(u_guess: Array, p_guess: Array, free: int) -> None:
    if p_guess.shape != (2,):
        raise ValueError(
            f"p_guess must have shape (2,) for two-parameter continuation, "
            f"got {p_guess.shape}. Codim-2 work needs two free parameters "
            f"(see bifurcations/codim2.py)."
        )
    if free not in (0, 1):
        raise ValueError(f"free must be 0 or 1, got {free!r}")
    if u_guess.ndim != 1:
        raise ValueError(f"u_guess must be 1-D, got shape {u_guess.shape}")


def unpack_fold_curve(X: Array, n: int) -> Tuple[Array, Array, Array]:
    """Split a packed fold-curve state into ``(u, p_fixed, v)``."""
    return X[:n], X[n], X[n + 1:]


def fold_curve_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    *,
    free: int = 1,
    args: PyTree = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of folds in the
    ``(p[0], p[1])`` plane.

    ``free`` indexes which component of ``p`` is continued; the other is
    solved for and lives inside the packed state. ``p_guess`` has shape
    ``(2,)``.

    The caller's guess is refined to a genuine fold via ``fold_point``
    before the problem is returned -- the scan engines do not Newton-correct
    their starting point, so an unrefined guess would silently be marked
    ``converged=True`` (the same reason ``periodic_orbit_problem`` refines).

    ``tol``/``max_iter`` govern that initial refinement only; the
    continuation-time tolerance is ``ContinuationPar.newton_tol`` (use
    ``1e-5``, see the module notes and this feature's design spec).

    Pass ``p_span=(p_guess[free], ...)`` to ``continuation()``: its
    ``p_span[0]`` is the literal starting parameter value, and a mismatch is
    rejected there.
    """
    u_guess = jnp.asarray(u_guess)
    p_guess = jnp.asarray(p_guess)
    _validate(u_guess, p_guess, free)

    n = u_guess.shape[0]
    q0 = p_guess[free]
    fixed0 = p_guess[1 - free]

    def reduced(u, p_fixed, a, q):
        return f(u, _assemble_p(p_fixed, q, free), a)

    # Refine the seed onto the curve at q = q0.
    u_star, p_star, v_star = fold_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )

    def F(X, q, a):
        return _fold_extended_residual(
            X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a, n
        )

    X0 = _fold_pack(u_star, p_star, v_star)
    return BifProblem(
        f=F,
        u0=X0,
        p0=jnp.asarray(q0, X0.dtype),
        args=args,
        kind="fold_curve",
        param_name=f"p[{free}]",
    )
```

- [ ] **Step 4: Widen `kind` and add the two guards**

In `src/jaxcont/api.py`, change `BifProblem.kind`'s annotation (currently line ~87):

```python
    kind: Literal[
        "equilibrium", "periodic", "bvp", "fold_curve", "hopf_curve"
    ] = "equilibrium"
```

Then, at the top of `_run_scan` (immediately after `dtype = u0.dtype`), add:

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
        if not jnp.allclose(jnp.asarray(p_start, dtype), problem.p0):
            raise ValueError(
                f"p_span[0]={float(p_start)} must equal the curve's starting "
                f"parameter p0={float(problem.p0)}. continuation() treats "
                f"p_span[0] as the literal starting value rather than reading "
                f"it off the problem, so a mismatch would start the run at a "
                f"point that is not on the refined curve."
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py -v`
Expected: PASS (both tests)

- [ ] **Step 6: Measure the achievable residual floor**

Do not guess `newton_tol`. Run:

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python - <<'EOF'
import jax.numpy as jnp
from tests.test_curves import _cusp_shifted
from jaxcont.bifurcations.curves import fold_curve_problem
prob = fold_curve_problem(_cusp_shifted, jnp.array([1.7]), jnp.array([-1.7, 4.2]), free=1)
print("residual at refined seed:", float(jnp.linalg.norm(prob.f(prob.u0, prob.p0, None))))
EOF
```

Record the printed value in `fold_curve_problem`'s docstring as the measured floor, and confirm the `newton_tol=1e-5` used in the tests sits above it. If the floor is above `1e-5`, raise the tests' `newton_tol` to the next decade and note why.

- [ ] **Step 7: Correct the spec's validation claim**

In `docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md` §3, replace "The factories validate this and raise on mismatch" with "`continuation()` validates this and raises on mismatch (the factories never see `p_span`)".

- [ ] **Step 8: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/curves.py src/jaxcont/api.py tests/test_curves.py docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md
git commit -m "feat(curves): fold-curve continuation in two parameters"
```

---

### Task 2: Hopf-curve factory

**Files:**
- Modify: `src/jaxcont/bifurcations/curves.py`
- Test: `tests/test_curves.py`

**Interfaces:**
- Consumes: `_assemble_p`, `_validate` (Task 1); `hopf_normal_form._extended_residual(x, f, args, n, u_guess, p_guess)`, `hopf_normal_form._pack(u, p, q1, q2, omega)`, `hopf_normal_form.hopf_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, q1, q2, omega)`.
- Produces:
  - `hopf_curve_problem(f, u_guess, p_guess, *, free=1, args=None, tol=1e-6, max_iter=50) -> BifProblem` with `kind="hopf_curve"`, `u0` the packed `(3n+2,)` state.
  - `unpack_hopf_curve(X, n) -> (u, p_fixed, q1, q2, omega)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_curves.py`:

```python
from jaxcont.bifurcations.curves import hopf_curve_problem, unpack_hopf_curve


def _hopf_parabola(u, p, args):
    """Hopf normal form with equilibrium shifted to (0.5, -0.3) and
    mu = p0 + p1**2 - 2, so the exact Hopf curve is the parabola
    p0 = 2 - p1**2 and the critical frequency is exactly omega = 1."""
    x = u[0] - 0.5
    y = u[1] + 0.3
    mu = p[0] + p[1] ** 2 - 2.0
    r2 = x * x + y * y
    return jnp.array([mu * x - y - x * r2, x + mu * y - y * r2])


def test_hopf_curve_traces_the_exact_parabola():
    prob = hopf_curve_problem(
        _hopf_parabola,
        jnp.array([0.5, -0.3]),
        jnp.array([2.0, 0.0]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 1.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _, _, omega = unpack_hopf_curve(sol.branch.states[i], n=2)
        assert abs(float(p0) + p1**2 - 2.0) < 1e-3
        assert abs(float(omega) - 1.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py::test_hopf_curve_traces_the_exact_parabola -v`
Expected: FAIL with `ImportError: cannot import name 'hopf_curve_problem'`

- [ ] **Step 3: Write the implementation**

Add to the imports in `src/jaxcont/bifurcations/curves.py`:

```python
from jaxcont.bifurcations.hopf_normal_form import (
    _extended_residual as _hopf_extended_residual,
)
from jaxcont.bifurcations.hopf_normal_form import _pack as _hopf_pack
from jaxcont.bifurcations.hopf_normal_form import hopf_point
```

Then append:

```python
def unpack_hopf_curve(
    X: Array, n: int
) -> Tuple[Array, Array, Array, Array, Array]:
    """Split a packed Hopf-curve state into ``(u, p_fixed, q1, q2, omega)``."""
    return (
        X[:n], X[n], X[n + 1:2 * n + 1], X[2 * n + 1:3 * n + 1], X[3 * n + 1]
    )


def hopf_curve_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    *,
    free: int = 1,
    args: PyTree = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
    """
    Build a ``BifProblem`` whose solution branch is a curve of Hopf points
    in the ``(p[0], p[1])`` plane. See :func:`fold_curve_problem` for the
    shared ``free``/``p_guess``/``tol`` conventions.

    Known limitation: the phase condition anchors to a seed eigenvector
    recomputed from the refined starting point at each ``q``, so it tracks
    gently along the curve. If the eigenvector rotates far enough to become
    orthogonal to that seed, the phase row degenerates and the curve stalls.
    Fixed-shape buffers (which the jit/vmap story depends on) rule out
    MatCont-style adaptive re-anchoring; the remedy is restarting from a
    later point.
    """
    u_guess = jnp.asarray(u_guess)
    p_guess = jnp.asarray(p_guess)
    _validate(u_guess, p_guess, free)

    n = u_guess.shape[0]
    q0 = p_guess[free]
    fixed0 = p_guess[1 - free]

    def reduced(u, p_fixed, a, q):
        return f(u, _assemble_p(p_fixed, q, free), a)

    u_star, p_star, q1_star, q2_star, omega_star = hopf_point(
        lambda u, p_fixed, a: reduced(u, p_fixed, a, q0),
        u_guess, fixed0, args, tol=tol, max_iter=max_iter,
    )

    def F(X, q, a):
        return _hopf_extended_residual(
            X, lambda u, p_fixed, aa: reduced(u, p_fixed, aa, q), a, n,
            u_star, p_star,
        )

    X0 = _hopf_pack(u_star, p_star, q1_star, q2_star, omega_star)
    return BifProblem(
        f=F,
        u0=X0,
        p0=jnp.asarray(q0, X0.dtype),
        args=args,
        kind="hopf_curve",
        param_name=f"p[{free}]",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Measure the Hopf-curve residual floor**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python - <<'EOF'
import jax.numpy as jnp
from tests.test_curves import _hopf_parabola
from jaxcont.bifurcations.curves import hopf_curve_problem
prob = hopf_curve_problem(_hopf_parabola, jnp.array([0.5,-0.3]), jnp.array([2.0,0.0]), free=1)
print("residual at refined seed:", float(jnp.linalg.norm(prob.f(prob.u0, prob.p0, None))))
EOF
```

Record it in `hopf_curve_problem`'s docstring. The Hopf system is `3n+2` versus the fold's `2n+1`, so expect a looser floor; if it exceeds `1e-5`, raise the test's `newton_tol` and document the measured value as the reason.

- [ ] **Step 6: Check GPU numerics against CPU (spec §5)**

The TF32 issue that corrupted the periodic-orbit collocation Jacobian came
from large `einsum` contractions; curve residuals are `jacfwd` plus matvecs,
and the Floquet recursion's small solves needed no fix. Check rather than
assume in either direction — run both curve tests on the **real GPU
backend** (no `JAX_PLATFORMS` override):

```bash
/home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py -v
```

Expected: PASS, matching the CPU run. If the exact-curve assertions fail
only on GPU, the fix is `jax.default_matmul_precision("float32")` around the
residual — apply it and record the measured discrepancy in the docstring.
If they pass, record in `curves.py`'s module docstring that no precision fix
was needed and that this was verified, not assumed.

- [ ] **Step 7: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/curves.py tests/test_curves.py
git commit -m "feat(curves): Hopf-curve continuation in two parameters"
```

---

### Task 3: `Cusp` event

**Files:**
- Create: `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_codim2_events.py`

**Interfaces:**
- Consumes: `events.BranchPoint`, `events.Event`, `events.EventHit`; `fold_normal_form.fold_coefficient(f, u, p, v, args) -> Array`; `codim2.cusp_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, v, converged)`; `curves.unpack_fold_curve`, `curves._assemble_p`.
- Produces: `Cusp(raw_f, free, args=None, kind="cusp", tolerance=1e-6)` with `test_function`/`refine`; module-level helper `_curve_p(point, raw_f_free, n)`.

**Note:** `Cusp` needs no eigenvalues — `fold_coefficient` is already a scalar that changes sign at a cusp. This task establishes the module and the `raw_f`-carrying pattern; Tasks 4-7 add the eigenvalue-based events on top.

- [ ] **Step 1: Write the failing test**

Create `tests/test_codim2_events.py`:

```python
"""Codim-2 events detected along two-parameter curves."""

import jax.numpy as jnp

import jaxcont as jc
from jaxcont.bifurcations.codim2_events import Cusp
from jaxcont.bifurcations.curves import fold_curve_problem


def _cusp_shifted(u, p, args):
    """Same shifted cusp normal form as tests/test_curves.py. Its cusp point
    is exactly u=0.7, p=(0.3, 1.2) (where the discriminant's two fold
    branches meet)."""
    x = u[0] - 0.7
    a = p[0] - 0.3
    b = p[1] - 1.2
    return jnp.array([a + b * x - x**3])


def test_cusp_detected_on_the_fold_curve():
    # Trace the fold curve DOWN toward the cusp at p1 = 1.2. Start at
    # b = p1-1.2 = 3 (x = 1, a = -2), continue p1 from 4.2 down past 1.2.
    prob = fold_curve_problem(
        _cusp_shifted, jnp.array([1.7]), jnp.array([-1.7, 4.2]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[Cusp(raw_f=_cusp_shifted, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "cusp"]
    assert len(hits) == 1
    assert abs(hits[0].p - 1.2) < 5e-2
    assert abs(float(hits[0].info["p_fixed"]) - 0.3) < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.bifurcations.codim2_events'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/bifurcations/codim2_events.py`:

```python
"""
Codim-2 events along two-parameter curves (see bifurcations/curves.py).

Kept separate from ``events.py`` deliberately: that module covers codim-1
events along ordinary branches, these are only meaningful along curves.

Every event here carries its own ``raw_f`` and ``free`` index.
``detect_events``'s generic ``rhs`` parameter is the EXTENDED-system
residual for a curve problem, not the original ODE, so reusing it would
reproduce -- in reverse -- the equilibrium-only footgun ``Hopf`` originally
had. This is the same reason ``PeriodDoubling``/``NeimarkSacker`` carry
``raw_f``/``mesh``.

See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array

from jaxcont.bifurcations.codim2 import cusp_point
from jaxcont.bifurcations.curves import _assemble_p, unpack_fold_curve
from jaxcont.bifurcations.events import BranchPoint, Event, EventHit
from jaxcont.bifurcations.fold_normal_form import fold_coefficient

PyTree = Any


@dataclass(frozen=True)
class Cusp(Event):
    """A cusp (CP) point on a fold curve.

    Test function: the fold's quadratic normal-form coefficient ``a``
    (``fold_normal_form.fold_coefficient``). ``a != 0`` is the fold's
    non-degeneracy condition and ``a == 0`` is precisely the cusp
    condition, so this needs no eigenvalues at all.

    Abbreviation **CP**, see ``bifurcations.taxonomy.describe("CP")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "cusp"
    tolerance: float = 1e-6

    def _decode(self, point: BranchPoint):
        """Recover ``(u, p, v)`` in the ORIGINAL system's coordinates from a
        packed fold-curve branch point."""
        n = (point.u.shape[0] - 1) // 2
        u, p_fixed, v = unpack_fold_curve(point.u, n)
        p = _assemble_p(p_fixed, jnp.asarray(point.p), self.free)
        return u, p, v

    def test_function(self, point: BranchPoint) -> float:
        u, p, v = self._decode(point)
        return float(fold_coefficient(self.raw_f, u, p, v, self.args))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v, converged = cusp_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        # codim2 solvers return `converged` as a JAX bool and can return
        # non-finite values on a failed solve. Both `nan < 0` and
        # `abs(nan) < tol` are False, so an unguarded fall-through would
        # emit a confident-looking hit for a point that isn't a cusp --
        # the exact bug Hopf.refine() grew its own guard for.
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "converged": True,
                "method": "extended_system",
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/codim2_events.py tests/test_codim2_events.py
git commit -m "feat(codim2-events): detect cusp points along a fold curve"
```

---

### Task 4: `BogdanovTakens` event + the trivial-eigenvalue exclusion

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_codim2_events.py`

**Interfaces:**
- Consumes: Task 3's `Cusp._decode` pattern; `codim2.bogdanov_takens_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, v0, v1, converged)`; `curves.unpack_hopf_curve`.
- Produces:
  - `_CurveEvent` (decoding mixin) and `_drop_nearest(values, target) -> mask` — shared by Tasks 4-7 (see Step 3's code for the authoritative signatures; this line only names them).
  - `BogdanovTakens(raw_f, free, curve, args=None, kind="bogdanov_takens", tolerance=1e-6, near_critical=2.0)`.
  - `_CurveEvent` mixin providing `_decode_fold` / `_decode_hopf` / `_jacobian`.

**This is the correctness core of the feature.** On a fold curve one eigenvalue is pinned at zero at *every* point (the curve's defining condition), so a naive "an eigenvalue reaches zero" test is satisfied everywhere and detects nothing. The test function must exclude the pinned eigenvalue and watch the next one. Ground truth below makes this checkable exactly.

**Ground truth derivation** (`_bt_shifted`, already verified in `tests/test_codim2.py`): with `x=X-5, y=Y-2, b1=p0-3, b2=p1+1`, the system is `x'=y`, `y'=b1+b2*x+x^2+x*y`. At an equilibrium (`y=0`) the Jacobian is `[[0,1],[b2+2x, x]]`, so `det = -(b2+2x)` and `trace = x`. Fold means `det=0`, i.e. `x=-b2/2`; substituting into `b1+b2*x+x^2=0` gives the exact fold curve **`b1 = b2**2/4`**, i.e. `p0 = 3 + (p1+1)**2/4`. On that curve the eigenvalues are `0` and `trace = x = -(p1+1)/2`. The second one crosses zero at **`p1 = -1`**, which is exactly the known BT at `u*=(5,2), p*=(3,-1)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_codim2_events.py`:

```python
from jaxcont.bifurcations.codim2_events import BogdanovTakens


def _bt_shifted(u, p, args):
    """Same system as tests/test_codim2.py: BT at u*=(5,2), p*=(3,-1).
    Exact fold curve p0 = 3 + (p1+1)**2/4; the non-trivial eigenvalue along
    it is exactly -(p1+1)/2, crossing zero at p1 = -1."""
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


def _bt_fold_curve_seed():
    # p1 = -2 -> b2 = -1 -> x = -b2/2 = 0.5 -> X = 5.5, Y = 2
    # b1 = b2**2/4 = 0.25 -> p0 = 3.25
    return jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0])


def _bt_fold_curve_solution(events):
    u_guess, p_guess = _bt_fold_curve_seed()
    prob = fold_curve_problem(_bt_shifted, u_guess, p_guess, free=1)
    return jc.continuation(
        prob,
        p_span=(-2.0, 0.0),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=events,
    )


def test_fold_curve_matches_the_exact_bt_parabola():
    sol = _bt_fold_curve_solution([])
    from jaxcont.bifurcations.curves import unpack_fold_curve
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _ = unpack_fold_curve(sol.branch.states[i], n=2)
        assert abs(float(p0) - (3.0 + (p1 + 1.0) ** 2 / 4.0)) < 1e-3


def test_bogdanov_takens_detected_on_the_fold_curve():
    sol = _bt_fold_curve_solution(
        [BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold")]
    )
    hits = [h for h in sol.events if h.kind == "bogdanov_takens"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - (-1.0)) < 5e-2
    assert jnp.allclose(hits[0].u, jnp.array([5.0, 2.0]), atol=5e-2)


def test_bt_test_function_ignores_the_pinned_zero_eigenvalue():
    """DISCRIMINATING POWER: one eigenvalue is identically zero along the
    whole fold curve. If the exclusion in _drop_nearest were
    removed, the test function would be identically ~0, never change sign,
    and detect nothing. This asserts it instead tracks -(p1+1)/2."""
    from jaxcont.bifurcations.events import BranchPoint
    sol = _bt_fold_curve_solution([])
    ev = BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold")
    saw_positive = saw_negative = False
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        val = ev.test_function(
            BranchPoint(p=p1, u=sol.branch.states[i])
        )
        assert abs(val - (-(p1 + 1.0) / 2.0)) < 5e-2
        saw_positive |= val > 0.05
        saw_negative |= val < -0.05
    assert saw_positive and saw_negative, "no sign change -> nothing to detect"


def test_bt_near_critical_filter_is_not_too_aggressive():
    """DISCRIMINATING POWER (the other direction): a pre-filter tight enough
    to reject the genuine candidate would silently drop the detection. The
    candidate reaches |-(p1+1)/2| = 0.5 at the ends of this span, so a
    filter narrower than that would zero out the detection."""
    sol = _bt_fold_curve_solution(
        [BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold",
                        near_critical=2.0)]
    )
    assert len([h for h in sol.events if h.kind == "bogdanov_takens"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v -k "bt or bogdanov or parabola"`
Expected: FAIL with `ImportError: cannot import name 'BogdanovTakens'`

- [ ] **Step 3: Write the shared decode/exclusion helpers**

Add to `src/jaxcont/bifurcations/codim2_events.py` (imports first):

```python
from jax import jacfwd

from jaxcont.bifurcations.codim2 import bogdanov_takens_point
from jaxcont.bifurcations.curves import unpack_hopf_curve
```

Then insert before `Cusp`:

```python
class _CurveEvent:
    """Shared decoding for events that live on a two-parameter curve.

    Deliberately NOT a dataclass: it declares no fields and exists only to
    share methods. Making it one would put an empty dataclass in the MRO
    ahead of each event's own field list for no benefit.

    Subclasses set ``curve`` to ``"fold"`` or ``"hopf"``. The curve type is
    NOT inferred: an Event only ever receives a BranchPoint, which carries
    no problem metadata, so it must be supplied at construction.
    """

    def _decode(self, point: BranchPoint):
        """Return ``(u, p, extra)`` in the ORIGINAL system's coordinates.

        ``extra`` is the null vector ``v`` for a fold curve, or
        ``(q1, q2, omega)`` for a Hopf curve.
        """
        if self.curve == "fold":
            n = (point.u.shape[0] - 1) // 2
            u, p_fixed, v = unpack_fold_curve(point.u, n)
            extra = v
        elif self.curve == "hopf":
            n = (point.u.shape[0] - 2) // 3
            u, p_fixed, q1, q2, omega = unpack_hopf_curve(point.u, n)
            extra = (q1, q2, omega)
        else:
            raise ValueError(
                f"curve must be 'fold' or 'hopf', got {self.curve!r}"
            )
        p = _assemble_p(p_fixed, jnp.asarray(point.p), self.free)
        return u, p, extra

    def _eigenvalues(self, u: Array, p: Array) -> Array:
        """Spectrum of the ORIGINAL system's Jacobian -- never the extended
        system's, which is what the branch state actually holds."""
        jac = jacfwd(lambda uu: self.raw_f(uu, p, self.args))(u)
        return jnp.linalg.eigvals(jac)


def _drop_nearest(values: Array, target: complex) -> Array:
    """Mask out the single entry closest to ``target``.

    On a fold curve one eigenvalue is pinned at 0 at every point; on a Hopf
    curve the pair +-i*omega is pinned to the axis. Those are the curve's
    OWN defining conditions, so they carry no information and must be
    excluded before looking for a codim-2 crossing -- the same rule
    stability.floquet.floquet_stable uses to exclude the trivial Floquet
    multiplier 1.
    """
    idx = jnp.argmin(jnp.abs(values - target))
    return jnp.arange(values.shape[0]) != idx
```

- [ ] **Step 4: Write the `BogdanovTakens` event**

Append to `src/jaxcont/bifurcations/codim2_events.py`:

```python
@dataclass(frozen=True)
class BogdanovTakens(_CurveEvent, Event):
    """A Bogdanov-Takens (BT) point, where fold and Hopf curves meet.

    On a FOLD curve: the second eigenvalue reaches zero (the first is
    pinned at zero by the curve's own defining condition).
    On a HOPF curve: the critical frequency ``omega`` reaches zero, i.e.
    the imaginary pair collides into a double zero.

    Abbreviation **BT**, see ``bifurcations.taxonomy.describe("BT")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "fold"
    args: PyTree = None
    kind: str = "bogdanov_takens"
    tolerance: float = 1e-6
    # How far a candidate may sit from the critical set before it is
    # treated as "never relevant". Serves the same purpose as
    # PeriodDoubling.near_unit_circle: without a pre-filter, argmin can
    # silently latch onto an unrelated, always-far eigenvalue and fire a
    # false positive; with too tight a window, the genuine candidate is
    # dropped and the detection is silently lost.
    #
    # This is a PLAIN absolute bound on |eigenvalue|, NOT the log-magnitude
    # window PeriodDoubling needs. The difference is real: Floquet
    # multipliers cluster multiplicatively around 1, so a linear window
    # there is capped below 1.0 by construction (a ~0-magnitude multiplier
    # sits at exactly distance 1), which is what caused the v0.3.1 false
    # negative. Eigenvalues cluster additively around 0, so distance from
    # zero is already the natural measure and has no such ceiling.
    near_critical: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        u, p, extra = self._decode(point)
        if self.curve == "hopf":
            _q1, _q2, omega = extra
            return float(omega)
        eigs = self._eigenvalues(u, p)
        keep = _drop_nearest(eigs, 0.0 + 0.0j)
        near = jnp.abs(eigs) < self.near_critical
        mask = keep & near & (jnp.abs(jnp.imag(eigs)) < self.tolerance)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        u_star, p_star, _v0, _v1, converged = bogdanov_takens_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "converged": True,
                "method": "extended_system",
            },
        )
```

Also update `Cusp` to inherit the shared decoding: change its declaration to
`class Cusp(_CurveEvent, Event):`, add the field `curve: str = "fold"`, and
delete its own `_decode` method (the mixin's supersedes it).

- [ ] **Step 5: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: PASS (all tests, including the two discriminating-power tests)

- [ ] **Step 6: Prove the discriminating-power tests actually discriminate**

Temporarily replace `keep = _drop_nearest(eigs, 0.0 + 0.0j)` with
`keep = jnp.ones(eigs.shape[0], dtype=bool)` and re-run:

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v -k "pinned or detected_on_the_fold"`
Expected: **FAIL** — confirming the tests have real discriminating power rather than passing vacuously. Revert the edit and confirm they pass again. Do not commit the temporary edit.

- [ ] **Step 7: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/codim2_events.py tests/test_codim2_events.py
git commit -m "feat(codim2-events): Bogdanov-Takens detection with trivial-eigenvalue exclusion"
```

---

### Task 5: `ZeroHopf` event

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_codim2_events.py`

**Interfaces:**
- Consumes: `_CurveEvent`, `_drop_nearest` (Task 4); `codim2.zero_hopf_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, v, q1, q2, omega, converged)`.
- Produces: `ZeroHopf(raw_f, free, curve, args=None, kind="zero_hopf", tolerance=1e-6, near_critical=2.0)`.

**Note:** `zero_hopf_point` requires `n >= 3`, so the test system is 3-D: a 2-D Hopf block (supplying the imaginary pair) plus a decoupled linear `z` block (supplying the real eigenvalue). This is the same decoupled-block construction the period-doubling work used to build ground truth the 2-D system could not exhibit.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2_events.py`:

```python
from jaxcont.bifurcations.codim2_events import ZeroHopf
from jaxcont.bifurcations.curves import hopf_curve_problem


def _zh_system(u, p, args):
    """2-D Hopf block (+- i at criticality) decoupled from a linear z block.

    Equilibrium is pinned at (1, -2, 3) for every p. The Hopf block's
    mu = p0 + p1**2 - 1 gives the exact Hopf curve p0 = 1 - p1**2; along
    it the z block contributes the real eigenvalue lam = p1 - 0.5, so the
    zero-Hopf point sits exactly at p1 = 0.5, p0 = 0.75.
    """
    x = u[0] - 1.0
    y = u[1] + 2.0
    z = u[2] - 3.0
    mu = p[0] + p[1] ** 2 - 1.0
    lam = p[1] - 0.5
    r2 = x * x + y * y
    return jnp.array([
        mu * x - y - x * r2,
        x + mu * y - y * r2,
        lam * z,
    ])


def test_zero_hopf_detected_on_the_hopf_curve():
    # Seed at p1 = 0 -> p0 = 1, equilibrium (1,-2,3).
    prob = hopf_curve_problem(
        _zh_system, jnp.array([1.0, -2.0, 3.0]), jnp.array([1.0, 0.0]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 1.0),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[ZeroHopf(raw_f=_zh_system, free=1, curve="hopf")],
    )
    hits = [h for h in sol.events if h.kind == "zero_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.5) < 5e-2
    assert abs(float(hits[0].info["p_fixed"]) - 0.75) < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py::test_zero_hopf_detected_on_the_hopf_curve -v`
Expected: FAIL with `ImportError: cannot import name 'ZeroHopf'`

- [ ] **Step 3: Write the implementation**

Add the import `from jaxcont.bifurcations.codim2 import zero_hopf_point`, then append:

```python
@dataclass(frozen=True)
class ZeroHopf(_CurveEvent, Event):
    """A zero-Hopf (ZH) point: a zero eigenvalue coincides with an
    imaginary pair. Requires ``n >= 3``.

    On a HOPF curve: a REAL eigenvalue crosses zero (the imaginary pair is
    pinned to the axis by the curve's defining condition).
    On a FOLD curve: a complex pair's real part crosses zero (the zero
    eigenvalue is pinned by the curve's defining condition).

    Abbreviation **ZH**, see ``bifurcations.taxonomy.describe("ZH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    curve: str = "hopf"
    args: PyTree = None
    kind: str = "zero_hopf"
    tolerance: float = 1e-6
    near_critical: float = 2.0

    def test_function(self, point: BranchPoint) -> float:
        u, p, extra = self._decode(point)
        eigs = self._eigenvalues(u, p)
        near = jnp.abs(eigs) < self.near_critical
        if self.curve == "hopf":
            # Exclude the pinned imaginary pair; watch a real eigenvalue.
            _q1, _q2, omega = extra
            keep = _drop_nearest(eigs, 1j * omega) & _drop_nearest(eigs, -1j * omega)
            mask = keep & near & (jnp.abs(jnp.imag(eigs)) < self.tolerance)
        else:
            # Exclude the pinned zero; watch a complex pair's real part.
            keep = _drop_nearest(eigs, 0.0 + 0.0j)
            mask = keep & near & (jnp.abs(jnp.imag(eigs)) > self.tolerance)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = zero_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        u_star, p_star, _v, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega": float(omega_star),
                "converged": True,
                "method": "extended_system",
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/codim2_events.py tests/test_codim2_events.py
git commit -m "feat(codim2-events): zero-Hopf detection on fold and Hopf curves"
```

---

### Task 6: `GeneralizedHopf` event

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_codim2_events.py`

**Interfaces:**
- Consumes: `_CurveEvent`; `hopf_normal_form.lyapunov_coefficient(f, u, p, q1, q2, omega0, args) -> Array`; `codim2.generalized_hopf_point(f, u_guess, p_guess, args, *, tol, max_iter) -> (u, p, q1, q2, omega, converged)`.
- Produces: `GeneralizedHopf(raw_f, free, args=None, kind="generalized_hopf", curve="hopf", l1_tolerance=1e-6)`.

**Note:** like `Cusp`, this needs no eigenvalues — `l1` is already a scalar that changes sign at a GH point. The test system is the Bautin normal form, whose `l1` is a positive multiple of its cubic coefficient.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_codim2_events.py`:

```python
from jaxcont.bifurcations.codim2_events import GeneralizedHopf


def _bautin(u, p, args):
    """Bautin (generalized-Hopf) normal form, equilibrium shifted to
    (0.4, -0.6). mu = p0 so the Hopf curve is exactly p0 = 0; the cubic
    coefficient b = p1 - 0.4 sets sign(l1), so GH sits exactly at p1 = 0.4.
    """
    x = u[0] - 0.4
    y = u[1] + 0.6
    mu = p[0]
    b = p[1] - 0.4
    r2 = x * x + y * y
    return jnp.array([
        mu * x - y + b * x * r2 - x * r2 * r2,
        x + mu * y + b * y * r2 - y * r2 * r2,
    ])


def test_generalized_hopf_detected_on_the_hopf_curve():
    prob = hopf_curve_problem(
        _bautin, jnp.array([0.4, -0.6]), jnp.array([0.0, 0.0]), free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[GeneralizedHopf(raw_f=_bautin, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "generalized_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.4) < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py::test_generalized_hopf_detected_on_the_hopf_curve -v`
Expected: FAIL with `ImportError: cannot import name 'GeneralizedHopf'`

- [ ] **Step 3: Write the implementation**

Add the imports `from jaxcont.bifurcations.codim2 import generalized_hopf_point` and `from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient`, then append:

```python
@dataclass(frozen=True)
class GeneralizedHopf(_CurveEvent, Event):
    """A generalized-Hopf / Bautin (GH) point on a Hopf curve: the first
    Lyapunov coefficient ``l1`` crosses zero, so the Hopf's criticality
    flips between supercritical and subcritical.

    Needs no eigenvalues -- ``l1`` is already a scalar that changes sign.

    Abbreviation **GH**, see ``bifurcations.taxonomy.describe("GH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "generalized_hopf"
    curve: str = "hopf"
    # Absolute threshold on a scale-dependent float32 quantity; see
    # events.py:Hopf.l1_tolerance for why this is not a universal constant.
    l1_tolerance: float = 1e-6

    def test_function(self, point: BranchPoint) -> float:
        u, p, (q1, q2, omega) = self._decode(point)
        return float(lyapunov_coefficient(self.raw_f, u, p, q1, q2, omega, self.args))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        result = generalized_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
        )
        u_star, p_star, _q1, _q2, omega_star, converged = result
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega": float(omega_star),
                "converged": True,
                "method": "extended_system",
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/codim2_events.py tests/test_codim2_events.py
git commit -m "feat(codim2-events): generalized-Hopf detection via l1 sign change"
```

---

### Task 7: `DoubleHopf` event (with automatic `seed_b`)

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_codim2_events.py`

**Interfaces:**
- Consumes: `_CurveEvent`, `_drop_nearest`; `codim2.double_hopf_point(f, u_guess, p_guess, args, *, seed_b, tol, max_iter, separation_tolerance) -> Tuple[Array, ...]`.
- Produces: `DoubleHopf(raw_f, free, args=None, kind="double_hopf", curve="hopf", tolerance=1e-6, near_critical=2.0, separation_tolerance=1e-3)`.

**Note — the payoff.** `double_hopf_point` requires a caller-supplied `seed_b` with no default, because it cannot guess the second Hopf pair and returns `nan` if both blocks seed onto the same physical pair. Curve detection produces that second pair naturally, so this event supplies `seed_b` automatically — making HH reachable without hand-construction for the first time. `double_hopf_point` returns a variable-length tuple (`Tuple[Array, ...]`); unpack it positionally as `(u, p, q1a, q2a, omega_a, q1b, q2b, omega_b, converged)` and verify against the live signature in Step 1.

- [ ] **Step 1: Confirm the return arity, then write the failing test**

First confirm the exact tuple:

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -c "
from jaxcont.bifurcations.codim2 import double_hopf_point as d
print(d.__doc__)"
```

Adjust the unpacking in Step 3 to match what the docstring states. Then append to `tests/test_codim2_events.py`:

```python
from jaxcont.bifurcations.codim2_events import DoubleHopf


def _hh_system(u, p, args):
    """Two decoupled 2-D Hopf blocks with distinct frequencies (1 and 2, so
    they clear double_hopf_point's separation check). Equilibrium pinned at
    (0.2, -0.1, 0.3, -0.4). Block A's mu_a = p0 gives the Hopf curve
    p0 = 0; along it block B's mu_b = p1 - 0.3 crosses zero exactly at
    p1 = 0.3, which is the double-Hopf point."""
    xa, ya = u[0] - 0.2, u[1] + 0.1
    xb, yb = u[2] - 0.3, u[3] + 0.4
    mu_a = p[0]
    mu_b = p[1] - 0.3
    ra2 = xa * xa + ya * ya
    rb2 = xb * xb + yb * yb
    return jnp.array([
        mu_a * xa - 1.0 * ya - xa * ra2,
        1.0 * xa + mu_a * ya - ya * ra2,
        mu_b * xb - 2.0 * yb - xb * rb2,
        2.0 * xb + mu_b * yb - yb * rb2,
    ])


def test_double_hopf_detected_with_automatic_seed_b():
    prob = hopf_curve_problem(
        _hh_system,
        jnp.array([0.2, -0.1, 0.3, -0.4]),
        jnp.array([0.0, 0.0]),
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 0.7),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[DoubleHopf(raw_f=_hh_system, free=1)],
    )
    hits = [h for h in sol.events if h.kind == "double_hopf"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert abs(hits[0].p - 0.3) < 5e-2
    # The two frequencies must be genuinely distinct -- a collapsed pair is
    # the degenerate case double_hopf_point returns nan for.
    assert abs(hits[0].info["omega_a"] - hits[0].info["omega_b"]) > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py::test_double_hopf_detected_with_automatic_seed_b -v`
Expected: FAIL with `ImportError: cannot import name 'DoubleHopf'`

- [ ] **Step 3: Write the implementation**

Add the import `from jaxcont.bifurcations.codim2 import double_hopf_point`, then append:

```python
@dataclass(frozen=True)
class DoubleHopf(_CurveEvent, Event):
    """A double-Hopf (HH) point on a Hopf curve: a SECOND complex pair's
    real part crosses zero while the first pair is pinned to the axis.

    ``double_hopf_point`` requires a caller-supplied ``seed_b`` (no
    default) because it cannot guess the second pair and degenerates to
    ``nan`` if both blocks land on the same physical pair. Detection along
    the curve produces that second pair naturally, so this event supplies
    ``seed_b`` itself.

    Abbreviation **HH**, see ``bifurcations.taxonomy.describe("HH")``.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    free: int = 1
    args: PyTree = None
    kind: str = "double_hopf"
    curve: str = "hopf"
    tolerance: float = 1e-6
    near_critical: float = 2.0
    separation_tolerance: float = 1e-3

    def _second_pair(self, point: BranchPoint):
        """Return ``(eigs, mask)`` for the non-pinned complex candidates."""
        u, p, (_q1, _q2, omega) = self._decode(point)
        eigs = self._eigenvalues(u, p)
        keep = _drop_nearest(eigs, 1j * omega) & _drop_nearest(eigs, -1j * omega)
        near = jnp.abs(jnp.real(eigs)) < self.near_critical
        mask = keep & near & (jnp.abs(jnp.imag(eigs)) > self.tolerance)
        return eigs, mask

    def test_function(self, point: BranchPoint) -> float:
        eigs, mask = self._second_pair(point)
        if not jnp.any(mask):
            return float("nan")
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        return float(jnp.real(eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_l, p_l, _ = self._decode(left)
        u_r, p_r, _ = self._decode(right)
        # seed_b: the second pair's frequency, taken from the bracket side
        # where it is cleanest. This is exactly what double_hopf_point
        # cannot guess for itself.
        eigs, mask = self._second_pair(right)
        candidates = jnp.where(mask, eigs, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates)))
        seed_b = jnp.abs(jnp.imag(eigs[idx]))

        result = double_hopf_point(
            self.raw_f, (u_l + u_r) / 2, (p_l + p_r) / 2, self.args,
            seed_b=seed_b,
            tol=max(tolerance, 1e-6), max_iter=max_iterations,
            separation_tolerance=self.separation_tolerance,
        )
        u_star, p_star = result[0], result[1]
        omega_a, omega_b, converged = result[-3], result[-2], result[-1]
        ok = bool(converged) and bool(jnp.all(jnp.isfinite(p_star)))
        if not ok:
            return EventHit(
                kind=self.kind, p=float(right.p), u=right.u, index=index,
                info={"converged": False, "method": "extended_system"},
            )
        return EventHit(
            kind=self.kind, p=float(p_star[self.free]), u=u_star, index=index,
            info={
                "p": p_star,
                "p_fixed": p_star[1 - self.free],
                "omega_a": float(omega_a),
                "omega_b": float(omega_b),
                "converged": True,
                "method": "extended_system",
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_codim2_events.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/bifurcations/codim2_events.py tests/test_codim2_events.py
git commit -m "feat(codim2-events): double-Hopf detection with automatic seed_b"
```

---

### Task 8: Top-level exports, taxonomy labels, plot styles

**Files:**
- Modify: `src/jaxcont/__init__.py`, `src/jaxcont/viz/styles.py`
- Test: `tests/test_curves.py`

**Interfaces:**
- Produces: `jc.fold_curve_problem`, `jc.hopf_curve_problem`, `jc.Cusp`, `jc.BogdanovTakens`, `jc.ZeroHopf`, `jc.GeneralizedHopf`, `jc.DoubleHopf`; `BIFURCATION_STYLES` entries for the five new `kind` strings.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_curves.py`:

```python
def test_two_parameter_surface_is_exported_top_level():
    for name in [
        "fold_curve_problem", "hopf_curve_problem",
        "Cusp", "BogdanovTakens", "ZeroHopf", "GeneralizedHopf", "DoubleHopf",
    ]:
        assert hasattr(jc, name), f"jc.{name} missing"
        assert name in jc.__all__, f"{name} missing from __all__"


def test_every_codim2_event_kind_has_a_plot_style():
    from jaxcont.viz.styles import BIFURCATION_STYLES
    for kind in [
        "cusp", "bogdanov_takens", "zero_hopf",
        "generalized_hopf", "double_hopf",
    ]:
        assert kind in BIFURCATION_STYLES, f"no style for {kind!r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py -v -k "exported or plot_style"`
Expected: FAIL with `AssertionError: jc.fold_curve_problem missing`

- [ ] **Step 3: Add the exports**

In `src/jaxcont/__init__.py`, after the existing `from jaxcont.bifurcations.codim2 import (...)` block, add:

```python
# Two-parameter continuation: fold/Hopf curves + codim-2 events along them.
# See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md
from jaxcont.bifurcations.curves import fold_curve_problem, hopf_curve_problem
from jaxcont.bifurcations.codim2_events import (
    BogdanovTakens, Cusp, DoubleHopf, GeneralizedHopf, ZeroHopf,
)
```

and add these seven names to `__all__`:

```python
    "fold_curve_problem",
    "hopf_curve_problem",
    "Cusp",
    "BogdanovTakens",
    "ZeroHopf",
    "GeneralizedHopf",
    "DoubleHopf",
```

- [ ] **Step 4: Add the plot styles**

In `src/jaxcont/viz/styles.py`, add to `BIFURCATION_STYLES` (labels are
`taxonomy.py`'s standard abbreviations, matching the existing entries'
convention):

```python
    "cusp": BifStyle("P", "#0072B2", "CP"),
    "bogdanov_takens": BifStyle("*", "#D55E00", "BT"),
    "zero_hopf": BifStyle("X", "#56B4E9", "ZH"),
    "generalized_hopf": BifStyle("d", "#F0E442", "GH"),
    "double_hopf": BifStyle("h", "#8C564B", "HH"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py tests/test_viz.py tests/test_taxonomy.py -v`
Expected: PASS

- [ ] **Step 6: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/__init__.py src/jaxcont/viz/styles.py tests/test_curves.py
git commit -m "feat: export two-parameter curve API and codim-2 event styles"
```

---

### Task 9: `plot_two_parameter_diagram`

**Files:**
- Create: `src/jaxcont/viz/two_parameter.py`
- Modify: `src/jaxcont/viz/__init__.py`, `src/jaxcont/__init__.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: `viz.styles.style_for(bif_type) -> BifStyle`; `curves.unpack_fold_curve` / `unpack_hopf_curve`; `ContinuationResult`.
- Produces: `plot_two_parameter_diagram(results, *, free=1, labels=None, ax=None, annotate=True) -> matplotlib.axes.Axes`, where `results` is a sequence of `(ContinuationResult, curve_kind)` pairs with `curve_kind` in `{"fold", "hopf"}`.

**Note:** `plot_phase_portrait` once silently ignored an `ax=` argument because it had no such parameter and swallowed it into `**kwargs`, so a two-panel figure never saved correctly. Give this a real `ax` parameter and test it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
def test_plot_two_parameter_diagram_draws_curves_and_codim2_markers():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    import jaxcont as jc
    from jaxcont.viz.two_parameter import plot_two_parameter_diagram
    from tests.test_codim2_events import _bt_shifted, _bt_fold_curve_seed

    u_guess, p_guess = _bt_fold_curve_seed()
    prob = jc.fold_curve_problem(_bt_shifted, u_guess, p_guess, free=1)
    sol = jc.continuation(
        prob, p_span=(-2.0, 0.0),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[jc.BogdanovTakens(raw_f=_bt_shifted, free=1, curve="fold")],
    )

    fig, ax = plt.subplots()
    returned = plot_two_parameter_diagram([(sol, "fold")], free=1, ax=ax)
    # The ax argument must be honoured, not swallowed into **kwargs (the
    # bug plot_phase_portrait once had).
    assert returned is ax
    assert len(ax.lines) >= 1
    # One codim-2 marker was drawn.
    assert len(ax.collections) + sum(
        1 for ln in ax.lines if ln.get_linestyle() == "None"
    ) >= 1
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py::test_plot_two_parameter_diagram_draws_curves_and_codim2_markers -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.viz.two_parameter'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/viz/two_parameter.py`:

```python
"""Two-parameter (codim-2) bifurcation diagrams: curves of folds and Hopf
points in the (p[0], p[1]) plane, with codim-2 points marked.

Reuses viz/styles.py's shared BIFURCATION_STYLES table so these plots use
the same markers/colors/abbreviations as every other diagram.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from jaxcont.bifurcations.curves import unpack_fold_curve, unpack_hopf_curve
from jaxcont.viz.styles import style_for

_CURVE_STYLE = {
    "fold": {"color": "#009E73", "label": "LP curve"},
    "hopf": {"color": "#CC79A7", "label": "H curve"},
}


def _curve_points(result, curve_kind: str, free: int):
    """Return ``(p_free, p_fixed)`` arrays for a traced curve."""
    states = result.branch.states
    params = result.branch.params
    n_state = states.shape[1]
    fixed = []
    for i in range(states.shape[0]):
        if curve_kind == "fold":
            n = (n_state - 1) // 2
            _u, p_fixed, _v = unpack_fold_curve(states[i], n)
        elif curve_kind == "hopf":
            n = (n_state - 2) // 3
            _u, p_fixed, _q1, _q2, _w = unpack_hopf_curve(states[i], n)
        else:
            raise ValueError(
                f"curve_kind must be 'fold' or 'hopf', got {curve_kind!r}"
            )
        fixed.append(float(p_fixed))
    return [float(p) for p in params], fixed


def plot_two_parameter_diagram(
    results: Sequence[Tuple[object, str]],
    *,
    free: int = 1,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
) -> plt.Axes:
    """
    Plot one or more two-parameter curves with their codim-2 points.

    ``results`` is a sequence of ``(ContinuationResult, curve_kind)`` pairs
    where ``curve_kind`` is ``"fold"`` or ``"hopf"``. ``free`` must match
    the ``free`` passed to the curve factory.

    ``ax`` is a real parameter: pass an existing axis to compose this into a
    multi-panel figure.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))

    for i, (result, curve_kind) in enumerate(results):
        p_free, p_fixed = _curve_points(result, curve_kind, free)
        style = _CURVE_STYLE[curve_kind]
        label = labels[i] if labels is not None else style["label"]
        ax.plot(p_free, p_fixed, "-", color=style["color"], lw=1.8, label=label)

        for hit in result.events:
            bif = style_for(hit.kind)
            p_vec = hit.info.get("p")
            if p_vec is None:
                continue
            x, y = float(p_vec[free]), float(p_vec[1 - free])
            ax.plot(
                x, y, bif.marker, color=bif.color, markersize=11,
                markeredgecolor="black", markeredgewidth=0.6,
                linestyle="None", label=bif.label, zorder=5,
            )
            if annotate:
                ax.annotate(
                    bif.label, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=bif.color,
                )

    ax.set_xlabel(f"p[{free}]")
    ax.set_ylabel(f"p[{1 - free}]")
    ax.grid(alpha=0.3)
    # De-duplicate legend entries (each codim-2 marker adds its own label).
    handles, legend_labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, legend_labels):
        seen.setdefault(lab, h)
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best", fontsize=9)
    return ax
```

Add to `src/jaxcont/viz/__init__.py`:

```python
from jaxcont.viz.two_parameter import plot_two_parameter_diagram
```

and to `src/jaxcont/__init__.py`'s imports and `__all__`:

```python
from jaxcont.viz.two_parameter import plot_two_parameter_diagram
```
```python
    "plot_two_parameter_diagram",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add src/jaxcont/viz/two_parameter.py src/jaxcont/viz/__init__.py src/jaxcont/__init__.py tests/test_viz.py
git commit -m "feat(viz): two-parameter bifurcation diagram with codim-2 markers"
```

---

### Task 10: BifurcationKit.jl cross-validation (Lorenz-84 BT)

**Files:**
- Modify: `examples/BifurcationKit/05_codim2.jl`
- Test: `tests/test_curves.py`

**Interfaces:**
- Consumes: `fold_curve_problem`, `BogdanovTakens`.
- Produces: `tests/test_curves.py::test_lorenz84_bt_matches_bifurcationkit`.

**Note:** `examples/BifurcationKit/05_codim2.jl` already continues the Lorenz-84 fold curve (`sn_codim2`) and Newton-refines its BT point; those refined values are already asserted in `tests/test_codim2.py`. This task asserts that our *detection along the traced curve* lands on the same point — the direct solver already matches it, so this tests the detector, not the solver.

- [ ] **Step 1: Read the existing reference values**

```bash
command grep -n "lorenz84\|bogdanov" -A 20 tests/test_codim2.py | command grep -n "assert\|jnp.array\|def test" | head -30
```

Copy the Lorenz-84 model definition and the asserted BT `u*`/`p*` values from `tests/test_codim2.py` — reuse them verbatim rather than re-deriving, so both tests are anchored to the same Julia output.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_curves.py`, substituting the model and reference values read in Step 1 (named here `_lorenz84`, `BK_BT_U`, `BK_BT_P`):

```python
def test_lorenz84_bt_matches_bifurcationkit():
    """Detection along a traced fold curve must land on the BT point
    BifurcationKit.jl v0.5.2 finds independently (examples/BifurcationKit/
    05_codim2.jl). tests/test_codim2.py asserts the direct SOLVER matches
    these values; this asserts the DETECTOR finds them without a guess."""
    from tests.test_codim2 import _lorenz84, BK_BT_U, BK_BT_P  # Step 1

    # Seed the fold curve at a fold point away from the BT, then continue
    # toward it. Exact seed values come from 05_codim2.jl's printed output.
    prob = jc.fold_curve_problem(
        _lorenz84, U_SEED, P_SEED, free=1,   # from Step 3's Julia run
    )
    sol = jc.continuation(
        prob, p_span=(P_SEED_FREE, P_TARGET_FREE),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[jc.BogdanovTakens(raw_f=_lorenz84, free=1, curve="fold")],
    )
    hits = [h for h in sol.events if h.kind == "bogdanov_takens"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert jnp.allclose(hits[0].u, BK_BT_U, atol=1e-3)
    assert jnp.allclose(hits[0].info["p"], BK_BT_P, atol=1e-3)
```

- [ ] **Step 3: Extend the Julia script to print curve endpoints**

In `examples/BifurcationKit/05_codim2.jl`, after the existing
`sn_codim2 = continuation(...)` call, add:

```julia
# --- Fold-curve endpoints, for JaxCont's two-parameter continuation test ---
println("=== LP curve endpoints (seed values for tests/test_curves.py) ===")
println("first point x  = ", sn_codim2.sol[1].x)
println("first point p  = ", sn_codim2.sol[1].p)
println("last  point p  = ", sn_codim2.sol[end].p)
println("n curve points = ", length(sn_codim2.sol))
```

Run it and copy the printed values into the test as `U_SEED`, `P_SEED`,
`P_SEED_FREE`, `P_TARGET_FREE`:

```bash
julia examples/BifurcationKit/05_codim2.jl
```

If Julia or BifurcationKit.jl is unavailable, mark the test
`@pytest.mark.skip(reason="requires offline Julia run; see 05_codim2.jl")`
and record the reason in the commit message — do NOT invent seed values.

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_curves.py::test_lorenz84_bt_matches_bifurcationkit -v`
Expected: PASS

- [ ] **Step 5: Confirm the test discriminates**

Perturb one component of `BK_BT_P` by `0.05` and re-run: the test must FAIL. Revert. This is the same injected-error check the codim-2 work used to prove its cross-validation was not a tautology.

- [ ] **Step 6: Run the full suite and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add examples/BifurcationKit/05_codim2.jl tests/test_curves.py
git commit -m "test: cross-validate Lorenz-84 BT detection against BifurcationKit.jl"
```

---

### Task 11: Promote MatCont case `US-C2-LP-001` to supported

**Files:**
- Move: `examples/MatCont/matlab/unsupported/run_two_parameter_fold_curve.m` → `examples/MatCont/matlab/run_two_parameter_fold_curve.m`
- Create: `examples/MatCont/matlab/export_curve_run.m`
- Create: `examples/MatCont/python_cases/curves.py`
- Modify: `examples/MatCont/cases.json`, `examples/MatCont/matlab/run_supported.m`
- Create: `examples/MatCont/reference/US-C2-LP-001_branch.csv`, `US-C2-LP-001_events.csv`, `US-C2-LP-001_metadata.json`

**Interfaces:**
- Consumes: `fold_curve_problem`, `BogdanovTakens`, `Cusp`; the existing `run_validation.py` / `compare.py` harness.
- Produces: `examples.MatCont.python_cases.curves:run_two_parameter_fold_curve` returning the harness's standard result dict.

**Note:** the existing driver runs MatCont's own `Testruns/testLPcataloscill.m` catalytic-oscillator model with `init_LP_LP` + `cont(@limitpoint, ...)` in free parameters `[2, 7]`, and is already marked `'unsupported_execution': 'executable'` — the MATLAB half runs today. It only `assert`s and prints; it needs an export step.

- [ ] **Step 1: Read the harness contract**

```bash
command sed -n '1,80p' examples/MatCont/matlab/export_equilibrium_run.m
command sed -n '1,60p' examples/MatCont/python_cases/equilibrium.py
command sed -n '1,60p' examples/MatCont/compare.py
```

Note the exact CSV column names, the metadata JSON keys, and the dict shape a `python_cases` function must return. Mirror them exactly — do not invent a new format.

- [ ] **Step 2: Add the export helper and promote the driver**

Create `examples/MatCont/matlab/export_curve_run.m` modelled on
`export_equilibrium_run.m` (same columns/keys as read in Step 1), writing a
two-parameter curve as `<case>_branch.csv` (one row per curve point, with the
two parameter columns plus the state columns) and detected singularities as
`<case>_events.csv`.

Then move the driver and add the export call:

```bash
git mv examples/MatCont/matlab/unsupported/run_two_parameter_fold_curve.m \
       examples/MatCont/matlab/run_two_parameter_fold_curve.m
```

In the moved file, delete the `%UNSUPPORTED_BY_JAXCONT` marker comment,
replace the final `fprintf('UNSUPPORTED_BY_JAXCONT ...')` line with:

```matlab
export_curve_run('US-C2-LP-001', curve, s, [2, 7]);
```

and add `run_two_parameter_fold_curve();` to `run_supported.m`.

- [ ] **Step 3: Generate the reference**

```bash
/home/ziaee/envs/jaxcont/bin/python -m examples.MatCont.run_validation \
    --case US-C2-LP-001 --regenerate-matcont
```

Review the generated files, then promote them into `reference/`. If MATLAB
fails to launch, STOP and report — do not hand-transcribe reference values.

- [ ] **Step 4: Write the JaxCont half**

Create `examples/MatCont/python_cases/curves.py` with
`run_two_parameter_fold_curve()` building the same catalytic-oscillator
model (transcribe it from `matlab/systems/cataloscill.m`), seeding
`fold_curve_problem` from the same LP the MATLAB driver seeds from, running
`jc.continuation` with `events=[Cusp(...), BogdanovTakens(..., curve="fold")]`,
and returning the harness dict shape from Step 1.

- [ ] **Step 5: Flip the case to supported**

In `examples/MatCont/cases.json`, for `US-C2-LP-001` set:

```json
  "support": "supported",
  "python": "examples.MatCont.python_cases.curves:run_two_parameter_fold_curve",
  "references": [
    "US-C2-LP-001_branch.csv",
    "US-C2-LP-001_events.csv",
    "US-C2-LP-001_metadata.json"
  ],
```

and remove its `"unsupported_execution"` key.

- [ ] **Step 6: Run the case and the suite**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m examples.MatCont.run_validation --case US-C2-LP-001
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_matcont_suite.py -v
```
Expected: the case passes within its declared `parameter_atol` of `0.001`.
If it fails only on tolerance, do NOT loosen the tolerance silently — record
the measured discrepancy in the commit message and raise it deliberately.

- [ ] **Step 7: Commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
git add examples/MatCont/ tests/
git commit -m "test(matcont): promote US-C2-LP-001 two-parameter fold curve to supported"
```

---

### Task 12: Promote MatCont case `US-C2-H-001` to supported

**Files:**
- Move: `examples/MatCont/matlab/unsupported/run_two_parameter_hopf_curve.m` → `examples/MatCont/matlab/run_two_parameter_hopf_curve.m`
- Modify: `examples/MatCont/python_cases/curves.py`, `examples/MatCont/cases.json`, `examples/MatCont/matlab/run_supported.m`
- Create: `examples/MatCont/reference/US-C2-H-001_*.{csv,json}`

**Interfaces:**
- Consumes: `export_curve_run.m` (Task 11), `hopf_curve_problem`, `GeneralizedHopf`, `BogdanovTakens`, `ZeroHopf`.
- Produces: `examples.MatCont.python_cases.curves:run_two_parameter_hopf_curve`.

- [ ] **Step 1: Promote the driver and add the export call**

```bash
git mv examples/MatCont/matlab/unsupported/run_two_parameter_hopf_curve.m \
       examples/MatCont/matlab/run_two_parameter_hopf_curve.m
```

Delete the `%UNSUPPORTED_BY_JAXCONT` marker, replace the final `fprintf`
with `export_curve_run('US-C2-H-001', curve, s, [2, 7]);`, and add
`run_two_parameter_hopf_curve();` to `run_supported.m`.

- [ ] **Step 2: Generate the reference**

```bash
/home/ziaee/envs/jaxcont/bin/python -m examples.MatCont.run_validation \
    --case US-C2-H-001 --regenerate-matcont
```

Review, then promote into `reference/`.

- [ ] **Step 3: Write the JaxCont half**

Add `run_two_parameter_hopf_curve()` to
`examples/MatCont/python_cases/curves.py`, reusing the catalytic-oscillator
model already transcribed in Task 11 (do not duplicate it — import it from
the module level), seeding `hopf_curve_problem` from the same H point the
MATLAB driver seeds from, with
`events=[GeneralizedHopf(...), BogdanovTakens(..., curve="hopf"), ZeroHopf(..., curve="hopf")]`.

- [ ] **Step 4: Flip the case to supported**

In `cases.json`, for `US-C2-H-001` set `"support": "supported"`,
`"python": "examples.MatCont.python_cases.curves:run_two_parameter_hopf_curve"`,
list its three reference files, and remove `"unsupported_execution"`.

- [ ] **Step 5: Run the case and the suite**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m examples.MatCont.run_validation --case US-C2-H-001
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
```

- [ ] **Step 6: Commit**

```bash
git add examples/MatCont/ tests/
git commit -m "test(matcont): promote US-C2-H-001 two-parameter Hopf curve to supported"
```

---

### Task 13: Gallery example with `vmap` and `grad`

**Files:**
- Create: `examples/example_12_two_parameter_diagram.py`

**Interfaces:**
- Consumes: the full public surface from Tasks 1-9.
- Produces: a Sphinx-Gallery example script (needs a title docstring, or the Read the Docs build breaks — that exact failure caused the v0.3.1 patch release).

**Note:** the roadmap's standing mandate is that every new curve/event type ships with a `vmap`-batched demonstration *and* a differentiable one. Because curves are ordinary `BifProblem`s, both come free — one script covers both rather than padding the gallery.

- [ ] **Step 1: Write the example**

Create `examples/example_12_two_parameter_diagram.py`:

```python
"""
Two-parameter continuation: curves of folds and Hopf points
============================================================

Traces a *curve* of fold points and a *curve* of Hopf points through the
(p0, p1) plane, marks the codim-2 points where they degenerate, and then
shows the two things MatCont and BifurcationKit.jl cannot do at all: the
whole curve batched under ``jax.vmap``, and the exact gradient of a
codim-2 location under ``jax.grad``.
"""

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxcont as jc


def bt_system(u, p, args):
    """x' = y ; y' = b1 + b2*x + x^2 + x*y, shifted so the BT sits at
    u = (5, 2), p = (3, -1). Its fold curve is the exact parabola
    p0 = 3 + (p1 + 1)^2 / 4."""
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


# %%
# Trace the fold curve and detect the Bogdanov-Takens point
# ---------------------------------------------------------
prob = jc.fold_curve_problem(
    bt_system, jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0]), free=1,
)
sol = jc.continuation(
    prob,
    p_span=(-2.0, 0.0),
    settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    events=[jc.BogdanovTakens(raw_f=bt_system, free=1, curve="fold")],
)

print(f"curve points: {sol.branch.params.shape[0]}")
for hit in sol.events:
    print(f"  {hit.kind:18s} p = {hit.info['p']}  (exact: [3, -1])")

ax = jc.plot_two_parameter_diagram([(sol, "fold")], free=1)
ax.set_title("Fold curve with its Bogdanov-Takens point")
plt.savefig("example_12_two_parameter_diagram.png", dpi=140,
            bbox_inches="tight")

# %%
# Batched: 64 curves in one kernel
# --------------------------------
# The curve is an ordinary BifProblem, so the existing scan engine's
# vmap support applies unchanged. Events stay outside the trace (they use
# Python control flow), so we batch the curve geometry itself.
def curve_endpoint(shift):
    def shifted(u, p, args):
        return bt_system(u, p + jnp.array([shift, 0.0]), args)
    prob_s = jc.fold_curve_problem(
        shifted, jnp.array([5.5, 2.0]), jnp.array([3.25 - shift, -2.0]),
        free=1,
    )
    res = jc.continuation(
        prob_s, p_span=(-2.0, 0.0),
        settings=jc.ContinuationPar(compute_stability=False,
                                    newton_tol=1e-5, max_steps=200),
    )
    return res.branch.states[-1][2]   # the solved p0 at the curve's end


shifts = jnp.linspace(0.0, 0.5, 64)
batched = jax.vmap(curve_endpoint)(shifts)
print(f"64 curves batched in one kernel; endpoint spread: "
      f"{float(batched.min()):.4f} .. {float(batched.max()):.4f}")

# %%
# Differentiable: exact sensitivity of a codim-2 location
# -------------------------------------------------------
# bogdanov_takens_parameters is built on the same implicit-function-theorem
# primitive as the rest of the library, so jax.grad skips the iteration.
def bt_p0_given_shift(shift):
    def shifted(u, p, args):
        return bt_system(u, p + jnp.array([shift, 0.0]), args)
    p_star = jc.bogdanov_takens_parameters(
        shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
    )
    return p_star[0]


g = jax.grad(bt_p0_given_shift)(0.0)
fd = (bt_p0_given_shift(1e-3) - bt_p0_given_shift(-1e-3)) / 2e-3
print(f"d(BT p0)/d(shift): grad = {float(g):.5f}, "
      f"finite-diff = {float(fd):.5f}")
assert abs(float(g) - float(fd)) < 1e-2, "gradient disagrees with FD"
```

- [ ] **Step 2: Run the example**

Run: `cd examples && MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python example_12_two_parameter_diagram.py`
Expected: exit 0; BT printed near `[3, -1]`; the gradient assertion passes.

If `curve_endpoint`'s `vmap` raises `ConcretizationTypeError`, the factory's internal `fold_point` refinement is being traced — note it, and move the seed refinement outside the batched function (build one problem, then `vmap` only `jc.continuation`). Report which form was needed.

- [ ] **Step 3: Verify the gallery build sees a title**

Run: `command head -5 examples/example_12_two_parameter_diagram.py`
Confirm the module docstring's title-and-underline is present. A title-less gallery page broke the Read the Docs build once already (v0.3.1).

- [ ] **Step 4: Commit**

```bash
git add examples/example_12_two_parameter_diagram.py
git commit -m "docs(examples): two-parameter diagram with vmap and grad demonstrations"
```

---

### Task 14: Update the roadmap and architecture docs

**Files:**
- Modify: `notes/ROADMAP.md`, `notes/ARCHITECTURE.md`

**Interfaces:**
- Consumes: the completed feature.
- Produces: documentation consistent with the shipped API.

- [ ] **Step 1: Tick the roadmap item**

In `notes/ROADMAP.md`'s "v0.3.0+ — Advanced (demand-driven)" section, change
`- [ ] Two-parameter continuation` to `- [x]` and add a writeup in the style
of the neighbouring entries covering: the no-new-engine reduction; the two
factories; the five events and the trivial-eigenvalue exclusion; the two
independent cross-validations (MatCont `US-C2-LP-001`/`US-C2-H-001`,
BifurcationKit Lorenz-84 BT); the measured `newton_tol` floors from Tasks 1
and 2; and the explicit scope cuts (adaptive phase re-anchoring,
bialternate products, `US-C2-PD-001`/`LPC`/`NS` staying unsupported,
codim-2 normal-form coefficients).

Also update the "Next up" list at the bottom: two-parameter continuation is
no longer open; branch switching remains.

- [ ] **Step 2: Replace the ARCHITECTURE §6 sketch**

In `notes/ARCHITECTURE.md` §6, replace the provisional block

```python
  sol2 = jc.continuation(jc.codim2(prob, event=jc.Fold()),
                         p_span=..., p2_span=..., events=[jc.Cusp(), jc.BogdanovTakens()])
```

with the shipped API:

```python
  prob2 = jc.fold_curve_problem(f, u_guess, p_guess=jnp.array([0.3, 1.0]), free=1)
  sol2  = jc.continuation(prob2, p_span=(1.0, 4.0),
                          settings=jc.ContinuationPar(compute_stability=False),
                          events=[jc.Cusp(raw_f=f, free=1),
                                  jc.BogdanovTakens(raw_f=f, free=1, curve="fold")])
```

and move the entry out of "Provisional API for later versions" into the
shipped-features section, noting that `jc.codim2(...)` was never built —
the factory pattern replaced it.

- [ ] **Step 3: Verify no stale references remain**

```bash
command grep -rn "jc.codim2\|two-parameter continuation, which is a separate (unstarted)" \
  src/ notes/ docs/source/ examples/ | command grep -v auto_examples
```

Expected: only `bifurcations/codim2.py`'s module docstring, which must also be
updated — change "requires two-parameter continuation, which is a separate
(unstarted) roadmap item" to point at `bifurcations/curves.py`.

- [ ] **Step 4: Final full verification and commit**

```bash
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto
JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m examples.MatCont.run_validation
git add notes/ src/jaxcont/bifurcations/codim2.py
git commit -m "docs: record two-parameter continuation as shipped"
```

---

## Notes for the executor

- **`docs/source/roadmap.rst` is badly stale** (still says "Current Version: 0.1.0" and lists shipped features as planned). Fixing it is out of scope for this plan — flag it, do not silently rewrite it.
- **If a codim-2 event fires zero times** where a test expects one, check the `near_critical` pre-filter before touching anything else. Both known failure directions run through it: too wide and `argmin` latches onto an unrelated eigenvalue (false positive), too narrow and the genuine candidate is dropped (false negative). The v0.3.1 patch release was exactly this bug in `PeriodDoubling`.
- **Never weaken a filter or loosen a tolerance to make a test pass** without first establishing that the *test case* is wrong. An implementer weakened `near_unit_circle` during the period-doubling work and it was reverted — the real bug was a hand-built test case sitting exactly on the filter boundary.
