# Equilibrium Branch-Point Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect equilibrium branch points (pitchfork/transcritical crossings, taxonomy label `BP`) and compute the crossing branch's tangent so a caller can continue onto it, closing the last item on ROADMAP.md's v0.3.0+ "match MatCont's most-used subset" list.

**Architecture:** A new `BranchPoint` `Event` (alongside `Fold`/`Hopf`/`PeriodDoubling`/`NeimarkSacker` in `bifurcations/events.py`) whose cheap per-step test function fires on the same real-eigenvalue-crossing signal a plain fold gives, and whose `refine()` does the expensive, decisive classification: Newton-converge via `fold_solve.py`'s existing extended system, then check whether the augmented Jacobian `[f_u | f_p]` has also dropped rank there (true BP) or not (ordinary fold — `refine()` returns `None`, a new capability the `Event` protocol needs). A new `branch_switch()` factory in `bifurcations/branch_switch.py` computes the crossing branch's tangent via SVD of that same augmented matrix, takes one predictor step, Newton-corrects via `solvers/implicit.py:differentiable_root`, and returns an ordinary `BifProblem` ready for `continuation()`.

**Tech Stack:** JAX (float32 default), pytest, this codebase's existing extended-system pattern (`fold_solve.py`, `solvers/implicit.py`).

**Spec:** `docs/superpowers/specs/2026-08-26-branch-point-switching-design.md`

**Execution note:** Each task below is written to be handed to a fresh agent (Codex CLI via `/codex:rescue`) with no memory of the brainstorming that produced this plan. Every task carries the exact file paths, exact function/class names from the current codebase, and the verification command. Read the spec first if anything here is ambiguous — the spec has the full rationale; this plan has the exact steps.

## Global Constraints

- **Python env:** use `/home/ziaee/envs/jaxcont/bin/python`. The system `python`/`python3` is a different install and cannot import JAX.
- **Run tests with** `JAX_PLATFORMS=cpu`, e.g. `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`. Verify the full suite is green after every task.
- **float32 is the default dtype.** Never set `jax_enable_x64=True`.
- **Never set a `tol`/`newton_tol` below ~`1e-6`.** A tolerance under the float32 floor reports `converged=False` forever with no error raised (ROADMAP issue #12). Measure the achievable floor for each new solve; do not assume one from another feature.
- **`differentiable_root`'s `theta` argument is always `args`** in this codebase's convention (see `fold_solve.py:fold_point`'s call), never `p`. When a seed or intermediate value used inside `G` depends on `args`, it must be recomputed *inside* the closure passed to `differentiable_root`/its `x0` callable — never precomputed outside and captured, which leaks a tracer under `jax.grad` (see `solvers/implicit.py:differentiable_root`'s own docstring).
- **Commit after every task** with the message given in that task's final step.

---

### Task 1: `BranchPoint` event — detection and classification

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py`
- Test: `tests/test_bifurcations.py` (existing file — check it exists and follow its existing test style for `Fold`/`Hopf`; if it does not exist, create it)

**Interfaces:**
- Consumes: `fold_solve.fold_point(f, u_guess, p_guess, args=None, *, tol, max_iter) -> (u, p, v, converged)` (already exists, `src/jaxcont/bifurcations/fold_solve.py`). `BranchPoint(left, right)` are `events.py`'s existing `BranchPoint`... **naming collision warning:** `events.py` already has an unrelated dataclass named `BranchPoint` (one point along a branch, fields `p`, `u`, `tangent`, `eigenvalues` — see the top of `events.py`). **This task's new `Event` subclass must NOT be named `BranchPoint`** to avoid shadowing it. Name it `BP` (matching the taxonomy abbreviation directly, and avoiding any confusion with the existing per-point dataclass).
- Produces: `BP` class (`kind: str = "branch_point"`) with `test_function`/`refine`; `Event.refine()`'s return type widened to `Optional[EventHit]`; `detect_events`'s hit-collection loop updated to skip `None`.

**Read first:** `src/jaxcont/bifurcations/events.py` in full (528 lines) — especially the existing `BranchPoint` per-point dataclass near the top (do not confuse with this task's new `BP` event class), `Fold`'s `refine()` (~line 105-125, the `fold_point` call pattern and the `converged=False` fallback shape), `Hopf`'s `test_function` (the real/complex eigenvalue masking pattern you'll mirror for real eigenvalues instead), and `detect_events` (the loop that calls `event.refine(...)` and appends to `hits`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_bifurcations.py` (create the file with this header if it doesn't exist; check first with `grep -l "^import jaxcont" tests/test_bifurcations.py` — if a similar file already exists with different fixtures, add these to it instead of duplicating imports):

```python
import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.bifurcations.events import BP, BranchPoint as EventPoint


def _pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


def _saddle_node(u, p, args):
    # f = u^2 + p: an ordinary fold at (u, p) = (0, 0), NOT a branch point.
    return jnp.array([u[0] ** 2 + p])


def test_bp_fires_on_pitchfork_but_not_on_a_plain_fold():
    prob = jc.bif_problem(_pitchfork, u0=jnp.array([0.0]), p0=-0.5)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-0.5, 0.5),
        settings=jc.ContinuationPar(ds=0.02, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    bp_hits = [e for e in sol.events if e.kind == "branch_point"]
    assert len(bp_hits) == 1
    assert bp_hits[0].p == pytest.approx(0.0, abs=1e-3)
    assert bp_hits[0].u[0] == pytest.approx(0.0, abs=1e-3)
    assert "tangent" in bp_hits[0].info

    # Negative control: a plain fold must NOT be classified as a branch
    # point, even though it gives the same real-eigenvalue-crossing signal.
    fold_prob = jc.bif_problem(_saddle_node, u0=jnp.array([1.0]), p0=-1.0)
    fold_sol = jc.continuation(
        fold_prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    assert [e for e in fold_sol.events if e.kind == "branch_point"] == []


def test_refine_returning_none_is_skipped_by_detect_events():
    # Direct unit test of the widened protocol, independent of continuation().
    left = EventPoint(p=-0.1, u=jnp.array([0.1]), tangent=jnp.array([1.0, 0.0]))
    right = EventPoint(p=0.1, u=jnp.array([-0.1]), tangent=jnp.array([1.0, 0.0]))
    hit = BP().refine(
        left, right, (0, 1), lambda u, p: _saddle_node(u, p, None),
        tolerance=1e-6, max_iterations=50,
    )
    assert hit is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_bifurcations.py -k "test_bp_fires or test_refine_returning_none" -v`
Expected: FAIL with `ImportError: cannot import name 'BP'` (it doesn't exist yet).

- [ ] **Step 3: Widen the `Event` protocol and `detect_events`**

In `src/jaxcont/bifurcations/events.py`, change the `Event` protocol's `refine` signature (near the top of the file, in the `@runtime_checkable class Event(Protocol):` block) from:

```python
    def refine(
        self,
        left: BranchPoint,
        right: BranchPoint,
        index: tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> EventHit:
        """Precisely locate the event between `left` and `right`."""
        ...
```

to:

```python
    def refine(
        self,
        left: BranchPoint,
        right: BranchPoint,
        index: tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> Optional[EventHit]:
        """Precisely locate the event between `left` and `right`, or
        return None if what looked like a candidate isn't a real hit
        (e.g. BP's classification step rejecting an ordinary fold)."""
        ...
```

(`Optional` is already imported at the top of `events.py` — check with `grep "^from typing" src/jaxcont/bifurcations/events.py`; add it to the existing `from typing import (...)` block if missing.)

In `detect_events`, find the loop:

```python
        for i in range(len(points) - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                kwargs = {} if select is None else {"prev_value": selected[i]}
                hits.append(event.refine(
                    points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations, **kwargs,
                ))
```

and change it to:

```python
        for i in range(len(points) - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                kwargs = {} if select is None else {"prev_value": selected[i]}
                hit = event.refine(
                    points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations, **kwargs,
                )
                if hit is not None:
                    hits.append(hit)
```

- [ ] **Step 4: Add the `BP` event class**

Add this class to `events.py`, after the existing `Hopf` class (before `PeriodDoubling`). Import `jacfwd` at the top of the file if not already imported (check `grep "^import jax\|^from jax" src/jaxcont/bifurcations/events.py` — add `from jax import jacfwd` or extend the existing `jax`/`jax.numpy` import if it uses a different style; match the file's existing import style):

```python
@dataclass(frozen=True)
class BP(Event):
    """A branch point: two equilibrium branches crossing (pitchfork,
    transcritical). See docs/superpowers/specs/2026-08-26-branch-point-
    switching-design.md for the full derivation.

    Test function: the real eigenvalue nearest zero -- the SAME signal an
    ordinary fold gives (df/du singular). This is deliberate, not an
    oversight: distinguishing a branch point from a plain fold requires a
    strictly stronger condition (the augmented [df/du | df/dp] matrix ALSO
    dropping rank, i.e. df/dp lying in the range of df/du) that only
    refine() can afford to check. test_function will fire at every plain
    LP too; refine() rejects those by returning None.

    Abbreviation **BP**, see jaxcont.bifurcations.taxonomy.describe("BP").
    """

    kind: str = "branch_point"
    tolerance: float = 1e-6       # |Im(eigenvalue)| below this counts as "real"
    rank_tolerance: float = 1e-4  # smallest singular value of [f_u|f_p] below
                                   # this counts as "rank-deficient" (a true BP);
                                   # calibrated in Step 5 below against the
                                   # pitchfork/transcritical/saddle-node fixtures

    def test_function(self, point: BranchPoint) -> float:
        eigs = point.eigenvalues
        real_mask = jnp.abs(jnp.imag(eigs)) < self.tolerance
        if not jnp.any(real_mask):
            return float("nan")
        real_eigs = jnp.real(eigs[real_mask])
        idx = jnp.argmin(jnp.abs(real_eigs))
        return float(real_eigs[idx])

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> Optional[EventHit]:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        u_bif, p_bif, _null_vector, converged = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        if not bool(converged):
            return None

        n = u_bif.shape[0]
        f_u = jacfwd(lambda u: rhs(u, p_bif))(u_bif)
        f_p = jacfwd(lambda p: rhs(u_bif, p))(p_bif)
        augmented = jnp.concatenate([f_u, f_p.reshape(n, 1)], axis=1)
        smallest_singular_value = jnp.linalg.svd(augmented, compute_uv=False)[-1]
        if not bool(smallest_singular_value < self.rank_tolerance):
            return None  # ordinary fold, not a branch point

        return EventHit(
            kind="branch_point", p=float(p_bif), u=u_bif, index=index,
            info={"tangent": right.tangent, "converged": True, "method": "extended_system"},
        )
```

`fold_point` is already imported at the top of `events.py` (used by `Fold.refine`) — no new import needed for it.

- [ ] **Step 5: Run tests, calibrate `rank_tolerance`, and check for regressions**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_bifurcations.py -k "test_bp_fires or test_refine_returning_none" -v`

If `test_bp_fires_on_pitchfork_but_not_on_a_plain_fold` fails because the saddle-node case's `smallest_singular_value` isn't clearly above `rank_tolerance=1e-4` (float32 noise could push a genuinely non-degenerate augmented matrix's smallest singular value lower than expected for some systems), print the actual `smallest_singular_value` for both the pitchfork's BP and the saddle-node's LP in a scratch script and pick a `rank_tolerance` with real margin between them (matches this project's standing practice of measuring, not guessing, float32 tolerance floors — see the Global Constraints section).

Then run the full suite to confirm no regression from widening the `Event` protocol:

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass (same count as before this task, plus the 2 new tests).

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/events.py tests/test_bifurcations.py
git commit -m "feat: add BP branch-point event, widen Event.refine() to Optional[EventHit]"
```

---

### Task 2: `branch_switch()` factory

**Files:**
- Create: `src/jaxcont/bifurcations/branch_switch.py`
- Test: `tests/test_branch_switch.py`

**Interfaces:**
- Consumes: `BP` event and its `EventHit` shape from Task 1 (`kind="branch_point"`, `info["tangent"]` an `(n+1,)` array whose last entry is the `dp/ds` component, matching every other `BranchPoint`-per-point `.tangent` field in this codebase). `solvers/implicit.py:differentiable_root(G, x0, theta, *, tol, max_iter) -> Array` (already exists). `api.py:bif_problem(f, u0, p0, *, args=None, ...) -> BifProblem` (already exists).
- Produces: `branch_switch(f, event, *, args=None, ds=0.01, tol=1e-6, max_iter=50) -> BifProblem`.

**Read first:** `src/jaxcont/solvers/implicit.py` in full (120 lines — small) for `differentiable_root`'s exact contract, especially the docstring paragraph on why `x0` must be a callable when the seed depends on `theta`. `src/jaxcont/bifurcations/fold_solve.py:fold_point` (lines ~62-92) as the reference example of `differentiable_root`'s usage convention: `theta` is always `args`, never `p`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_branch_switch.py`:

```python
"""branch_switch() -- bifurcations/branch_switch.py."""

import jax
import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.bifurcations.branch_switch import branch_switch
from jaxcont.bifurcations.events import BP


def _pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


def _transcritical(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 2])


def _find_bp(f, u0, p0, p_span):
    prob = jc.bif_problem(f, u0=u0, p0=p0)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=p_span,
        settings=jc.ContinuationPar(ds=0.02, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    hits = [e for e in sol.events if e.kind == "branch_point"]
    assert len(hits) == 1
    return hits[0]


def test_switch_lands_on_the_pitchfork_second_branch():
    event = _find_bp(_pitchfork, jnp.array([0.0]), -0.5, (-0.5, 0.5))
    new_prob = branch_switch(_pitchfork, event, ds=0.05)
    # u = +-sqrt(p) -- sign is a genuine ambiguity (SVD's arbitrary sign
    # convention), so check magnitude only.
    assert abs(float(new_prob.u0[0])) == pytest.approx(
        float(jnp.sqrt(jnp.abs(new_prob.p0))), rel=0.2
    )
    assert float(new_prob.p0) > 0.0  # the pitchfork's second branch only exists for p > 0

    # The seed must actually be a valid equilibrium, not just "off the
    # trivial branch".
    residual = _pitchfork(new_prob.u0, new_prob.p0, None)
    assert float(jnp.max(jnp.abs(residual))) < 1e-4

    # And it must be continuable: continuing further should stay on
    # u != 0 (the new branch), not snap back onto the trivial one.
    sol = jc.continuation(
        new_prob, jc.PseudoArclength(), p_span=(float(new_prob.p0), float(new_prob.p0) + 0.3),
        settings=jc.ContinuationPar(ds=0.02, max_steps=30, newton_tol=1e-6),
    )
    assert float(jnp.min(jnp.abs(sol.branch.states[:, 0]))) > 0.05


def test_switch_lands_on_the_transcritical_second_branch():
    event = _find_bp(_transcritical, jnp.array([0.0]), -0.5, (-0.5, 0.5))
    new_prob = branch_switch(_transcritical, event, ds=0.05)
    # second branch: u = p
    assert float(new_prob.u0[0]) == pytest.approx(float(new_prob.p0), abs=0.1)
    residual = _transcritical(new_prob.u0, new_prob.p0, None)
    assert float(jnp.max(jnp.abs(residual))) < 1e-4


def test_branch_switch_rejects_a_non_branch_point_event():
    from jaxcont.bifurcations.events import EventHit
    fake = EventHit(kind="fold", p=0.0, u=jnp.array([0.0]), index=(0, 1), info={})
    with pytest.raises(ValueError, match="branch_point"):
        branch_switch(_pitchfork, fake)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_branch_switch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.bifurcations.branch_switch'`.

- [ ] **Step 3: Write `branch_switch.py`**

Create `src/jaxcont/bifurcations/branch_switch.py`:

```python
"""
Branch switching: from a detected equilibrium branch point (BP), compute
the crossing branch's tangent and return a seeded BifProblem on it.

See docs/superpowers/specs/2026-08-26-branch-point-switching-design.md.

At a confirmed BP (jaxcont.bifurcations.events.BP), the augmented Jacobian
[f_u | f_p] (n x (n+1)) has a 2-D null space: one direction is the incoming
branch's own known tangent (carried in EventHit.info["tangent"] -- EventHit
itself has no tangent field), the other is the crossing branch's direction.
Both are recovered via SVD; the crossing direction is whichever of the two
smallest right-singular vectors survives projecting out the known tangent.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jacfwd

from jaxcont.api import BifProblem, bif_problem
from jaxcont.bifurcations.events import EventHit
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _switching_seed(f, u_star, p_star, incoming_tangent, args, ds, n):
    """(u_pred, p_pred): one predictor step along the crossing branch's
    tangent, recomputed here (not precomputed by the caller) so it stays
    theta(=args)-dependent when this is called from inside branch_switch's
    differentiable_root closures -- see the module docstring and
    solvers/implicit.py's "seed must be callable when theta-dependent" rule.
    """
    f_u = jacfwd(lambda u: f(u, p_star, args))(u_star)
    f_p = jacfwd(lambda p: f(u_star, p, args))(p_star)
    augmented = jnp.concatenate([f_u, f_p.reshape(n, 1)], axis=1)
    _, _, vh = jnp.linalg.svd(augmented)
    v1, v2 = vh[-1], vh[-2]

    incoming_dir = incoming_tangent / jnp.linalg.norm(incoming_tangent)

    def project_out(v):
        return v - jnp.dot(v, incoming_dir) * incoming_dir

    c1, c2 = project_out(v1), project_out(v2)
    t_new_raw = jnp.where(jnp.linalg.norm(c1) > jnp.linalg.norm(c2), c1, c2)
    t_new = t_new_raw / jnp.linalg.norm(t_new_raw)

    du, dp = t_new[:n], t_new[n]
    return u_star + ds * du, p_star + ds * dp


def branch_switch(
    f: Callable[[Array, Array, PyTree], Array],
    event: EventHit,
    *,
    args: PyTree = None,
    ds: float = 0.01,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
    """
    From a detected BP (``jaxcont.bifurcations.events.BP``), compute the
    crossing branch's tangent, take one predictor step, Newton-correct back
    onto ``f(u, p, args) = 0``, and return a ready-to-continue
    :class:`BifProblem`.

    ``event`` must be a ``kind="branch_point"`` hit (from
    ``continuation(..., events=[BP()])``); anything else raises
    ``ValueError`` -- eagerly, except under a traced call
    (``jax.jit``/``jax.vmap``), where the same deferral
    ``fold_curve_problem``/``hopf_curve_problem`` already use applies (the
    check becomes a no-op rather than raising, since ``bool()`` on a
    tracer would itself raise ``TracerBoolConversionError``).

    The switching direction's sign is arbitrary (an SVD sign convention,
    not a physical choice) -- callers that care which of the two crossing
    directions they land on should inspect ``new_problem.u0``/``.p0`` and
    negate ``ds`` and retry if it's the wrong one.
    """
    if event.kind != "branch_point":
        raise ValueError(
            f"branch_switch: expected a BP hit (kind='branch_point'), "
            f"got kind={event.kind!r}. Call continuation(..., events=[BP()]) "
            f"and pass one of the resulting kind='branch_point' events."
        )

    u_star = jnp.asarray(event.u)
    p_star = jnp.asarray(event.p)
    n = u_star.shape[0]
    incoming_tangent = jnp.asarray(event.info["tangent"])

    def G(u, theta):
        u_pred, p_pred = _switching_seed(f, u_star, p_star, incoming_tangent, theta, ds, n)
        return f(u, p_pred, theta)

    def x0(theta):
        u_pred, _ = _switching_seed(f, u_star, p_star, incoming_tangent, theta, ds, n)
        return u_pred

    u_corrected = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    _, p_corrected = _switching_seed(f, u_star, p_star, incoming_tangent, args, ds, n)

    return bif_problem(f, u0=u_corrected, p0=p_corrected, args=args)
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_branch_switch.py -v`
Expected: PASS. If `test_switch_lands_on_the_pitchfork_second_branch`'s continuation assertion fails because the corrected seed is numerically too close to the trivial branch (Newton correcting back onto `u=0` instead of the intended branch), increase `ds` in the test (e.g. `0.1`) -- the predictor step needs to be large enough to land in the new branch's basin of attraction; this is a real tuning parameter, not a bug, and is exactly why `branch_switch`'s `ds` defaults to `0.01` but is caller-overridable.

Then the full suite:

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/bifurcations/branch_switch.py tests/test_branch_switch.py
git commit -m "feat: add branch_switch() factory for equilibrium branch points"
```

---

### Task 3: Top-level exports and plot styling

**Files:**
- Modify: `src/jaxcont/api.py`
- Modify: `src/jaxcont/__init__.py`
- Modify: `src/jaxcont/viz/styles.py`
- Test: `tests/test_functional_api.py`, `tests/test_viz.py`

**Interfaces:**
- Consumes: `BP` (Task 1), `branch_switch` (Task 2).
- Produces: `jc.BP`, `jc.branch_switch` at the top-level package surface.

**Read first:** `src/jaxcont/api.py` lines 25-54 (the `from jaxcont.bifurcations.events import (...)` block and `__all__` list -- `Fold`/`Hopf`/`PeriodDoubling`/`NeimarkSacker` are re-exported through `api.py`, not directly from `events.py`, in `__init__.py`). `src/jaxcont/__init__.py` lines 14-31 (the `from jaxcont.api import (...)` block) and its `__all__` list (alphabetically sorted -- keep it that way). `src/jaxcont/viz/styles.py` lines 24-32 (`BIFURCATION_STYLES` dict) -- note the existing `"branch-point"` key uses a HYPHEN while every `Event.kind` in this codebase uses underscores (`"fold"`, `"hopf"`, `"period_doubling"`, `"neimark_sacker"`, and now `"branch_point"`); this is a pre-existing inconsistency for `period-doubling`/`period_doubling` too (out of scope to fix here -- do not touch the `"period-doubling"`/`"branch-point"` hyphenated keys), so just ADD a new `"branch_point"` (underscore) key so this feature's own plots resolve correctly, rather than fixing the older mismatch.

**Note on the spec's §7 deliverables list:** the spec also lists updating `taxonomy.py`'s `BP` entry status. That turns out to be unnecessary: `taxonomy.py`'s own docstring says a `"v0.3"` status already means "landing in the current development series" (not "not yet implemented") -- `BT`/`GH`/`ZH`/`HH` all carry `"v0.3"` and are long since implemented. No `taxonomy.py` change is needed; skip it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_functional_api.py` (near the other top-level export smoke tests, or as a new small test if none exist -- check with `grep -n "jc\.Fold\b" tests/test_functional_api.py` for the existing pattern):

```python
def test_bp_and_branch_switch_are_exported_at_top_level():
    assert hasattr(jc, "BP")
    assert hasattr(jc, "branch_switch")
    assert jc.BP().kind == "branch_point"
```

Add to `tests/test_viz.py` (check `grep -n "BIFURCATION_STYLES\[" tests/test_viz.py` for the existing per-kind style test pattern and mirror it):

```python
def test_branch_point_style_is_registered():
    from jaxcont.viz.styles import BIFURCATION_STYLES, style_for
    assert "branch_point" in BIFURCATION_STYLES
    assert style_for("branch_point").label == "BP"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py tests/test_viz.py -k "bp_and_branch_switch or branch_point_style" -v`
Expected: FAIL (`AttributeError`/`AssertionError`).

- [ ] **Step 3: Wire up the exports**

In `src/jaxcont/api.py`, add `BP` to the existing import block:

```python
from jaxcont.bifurcations.events import (
    BP,
    Event,
    EventHit,
    Fold,
    Hopf,
    NeimarkSacker,
    PeriodDoubling,
    detect_events,
)
```

and add `"BP"` to `api.py`'s `__all__` list (alphabetically), alongside the existing `"Event"`, `"EventHit"`, `"Fold"`, etc.

In `src/jaxcont/__init__.py`:
- Add `BP` to the existing `from jaxcont.api import (...)` block (alphabetically, between `BifProblem` and `Branch`).
- Add a new import line, near the existing `from jaxcont.bifurcations.curves import fold_curve_problem, hopf_curve_problem` line:

```python
# Branch-point switching: detect a pitchfork/transcritical crossing (BP)
# and compute the crossing branch's tangent. See
# docs/superpowers/specs/2026-08-26-branch-point-switching-design.md
from jaxcont.bifurcations.branch_switch import branch_switch
```

- Add `"BP"` and `"branch_switch"` to `__init__.py`'s `__all__` list, keeping it alphabetically sorted (matches the file's existing convention -- `"BP"` goes right after `"BogdanovTakens"`/before `"Branch"`; `"branch_switch"` goes in the lowercase-sorted section near `"bif_problem"`/`"bogdanov_takens_parameters"`).

In `src/jaxcont/viz/styles.py`, add one line to `BIFURCATION_STYLES` (after the existing `"branch-point"` entry, do not remove or rename that one):

```python
    "branch_point": BifStyle("D", "#7B61A8", "BP"),
```

- [ ] **Step 4: Run tests**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py tests/test_viz.py -v`
Expected: PASS.

Then the full suite:

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/api.py src/jaxcont/__init__.py src/jaxcont/viz/styles.py tests/test_functional_api.py tests/test_viz.py
git commit -m "feat: export jc.BP/jc.branch_switch, add branch_point plot style"
```

---

### Task 4: `vmap` + differentiable verification, example script

**Files:**
- Test: `tests/test_branch_switch.py` (extend)
- Create: `examples/example_21_pitchfork_branch_switch.py` (check `ls examples/ | tail -5` first to confirm the next free number -- use whatever is actually next if `example_21` is already taken)

**Interfaces:**
- Consumes: `BP`, `branch_switch` (Tasks 1-2).

**Read first:** `examples/example_07_differentiable.py` (the existing pattern for a `jax.grad`-through-an-extended-system example, including how it cross-checks against finite differences) and `examples/example_06_vmap_sweep.py` (the existing `vmap`-batched pattern). Follow both conventions.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_branch_switch.py`:

```python
def test_branch_switch_is_differentiable_in_args():
    def imperfect_pitchfork(u, p, b):
        return jnp.array([p * u[0] - u[0] ** 3 + b])

    event = _find_bp(lambda u, p, a: imperfect_pitchfork(u, p, 0.0), jnp.array([0.0]), -0.5, (-0.5, 0.5))

    def new_branch_start_state(b):
        new_prob = branch_switch(imperfect_pitchfork, event, args=b, ds=0.05)
        return new_prob.u0[0]

    b0 = jnp.array(0.0)
    analytic = jax.grad(new_branch_start_state)(b0)

    eps = 1e-3
    finite_diff = (new_branch_start_state(b0 + eps) - new_branch_start_state(b0 - eps)) / (2 * eps)

    assert jnp.isfinite(analytic)
    assert float(jnp.abs(analytic - finite_diff)) < 2e-2


def test_branch_switch_vmap_batches_over_a_second_parameter():
    def imperfect_pitchfork(u, p, b):
        return jnp.array([p * u[0] - u[0] ** 3 + b])

    def switch_for_b(b):
        event = _find_bp(lambda u, p, a: imperfect_pitchfork(u, p, b), jnp.array([0.0]), -0.5, (-0.5, 0.5))
        new_prob = branch_switch(imperfect_pitchfork, event, args=b, ds=0.05)
        return new_prob.u0[0], new_prob.p0

    # _find_bp calls continuation() with events=[...], which is eager-only
    # (see events.py's module docstring) -- vmap only the branch_switch()
    # call itself, using a pre-detected event per batch element built
    # outside the trace. This matches how events=[...] under vmap is
    # documented as unsupported elsewhere in this codebase.
    bs = jnp.linspace(-0.05, 0.05, 4)
    events = [
        _find_bp(lambda u, p, a: imperfect_pitchfork(u, p, float(b)), jnp.array([0.0]), -0.5, (-0.5, 0.5))
        for b in bs
    ]
    u_stars = jnp.array([e.u[0] for e in events])
    p_stars = jnp.array([e.p for e in events])
    tangents = jnp.array([e.info["tangent"] for e in events])

    def switch_one(u_star, p_star, tangent, b):
        from jaxcont.bifurcations.events import EventHit
        fake_event = EventHit(kind="branch_point", p=float(p_star), u=jnp.array([u_star]),
                               index=(0, 1), info={"tangent": tangent})
        new_prob = branch_switch(imperfect_pitchfork, fake_event, args=b, ds=0.05)
        return new_prob.u0[0], new_prob.p0

    batched = jax.vmap(switch_one)(u_stars, p_stars, tangents, bs)
    assert batched[0].shape == (4,)
    for i, b in enumerate(bs):
        serial_u, serial_p = switch_for_b(float(b))
        assert float(batched[0][i]) == pytest.approx(float(serial_u), abs=1e-4)
        assert float(batched[1][i]) == pytest.approx(float(serial_p), abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail or pass for the wrong reason**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_branch_switch.py -k "differentiable or vmap" -v`

If `test_branch_switch_is_differentiable_in_args` fails with a shape/tracer error rather than a numeric mismatch, the likely cause is `_switching_seed`'s SVD or the `jnp.where` sign-selection not tracing cleanly under `jax.grad` -- debug by first checking `jax.grad` on `_switching_seed` alone (a standalone scratch script, not part of the test suite) before touching `branch_switch`'s structure. Do not add `lax.stop_gradient` anywhere to make this pass -- that would silently produce a wrong (not just approximate) gradient, since the switching direction genuinely depends on `args` here; the earlier codebase precedent for `stop_gradient` (GH/ZH/HH's Hopf-seed) applies to a `jnp.linalg.eig` gradient-rule gap that does not exist for `jnp.linalg.svd`, so it does not apply here.

- [ ] **Step 3: Fix any real numerical issues found, then confirm both pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_branch_switch.py -v`
Expected: all PASS.

- [ ] **Step 4: Write the example script**

Create `examples/example_21_pitchfork_branch_switch.py` (adjust the number per Step 1's `ls` check), following `examples/example_07_differentiable.py`'s structure (docstring header, `if __name__ == "__main__":` guard, `MPLBACKEND=Agg`-safe plotting): detect the pitchfork's BP via `continuation(..., events=[jc.BP()])`, call `jc.branch_switch(...)`, continue the returned problem, and plot both branches together using `jc.plot_continuation` (or `jc.plot_bifurcation_diagram` if that is the existing convention for multi-branch plots -- check `examples/example_01_pitchfork.py` for how the existing pitchfork example already plots the trivial branch, and follow the same style so the two branches look consistent).

- [ ] **Step 5: Verify the example runs**

Run: `MPLBACKEND=Agg JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python examples/example_21_pitchfork_branch_switch.py`
Expected: exits 0, no exceptions.

- [ ] **Step 6: Full suite and commit**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

```bash
git add tests/test_branch_switch.py examples/example_21_pitchfork_branch_switch.py
git commit -m "test: verify branch_switch is differentiable and vmap-batchable, add example"
```

---

### Task 5: MatCont cross-validation (or documented fallback)

**Files:**
- Create: `examples/MatCont/python_cases/branch_switch.py` (only if a suitable MatCont reference is found -- see Step 1)
- Modify: `examples/MatCont/cases.json`, `examples/MatCont/validation.py`, `tests/test_matcont_suite.py` (only if Step 1 finds a suitable case)
- Modify: `docs/superpowers/specs/2026-08-26-branch-point-switching-design.md` (record the outcome either way)

**Read first:** `examples/MatCont/python_cases/codim2_curves.py` (this session's own precedent: hardcoded reference values from a live MatCont run, checked via `validation.py`'s `case_result_passes` predicates, not a `compare.py` extension) -- follow this exact pattern if Step 1 finds a match.

- [ ] **Step 1: Search for a plain single-parameter equilibrium BP testrun in MatCont**

```bash
grep -rl "init_BP_EP" /home/ziaee/prog/MatCont/MatCont7p6/Testruns/*.m
```

Two BP-related MatCont testruns are already known from this feature's planning and do NOT fit (do not re-investigate them, go straight past):
- `Testruns/testbratu2.m` uses `init_BP_EP` but on a BVP-discretized Bratu system -- JaxCont's BVP support is an unimplemented placeholder (`problems/bvp.py`), and this exact system is already `cases.json`'s `US-BP-001` ("Bratu branch switching", `unsupported`, features `["boundary-value-problem", "branch-switching"]") for that reason. Out of scope for this task.
- `Testruns/cstr3.m` finds branch points on the `cstr` equilibrium system but its own branch-switching workflow jumps straight to a two-parameter branch-point-CURVE continuation (`init_BP_BP`/`cont(@branchpoint,...)`), not the single-parameter switch this feature implements.

If the `grep` above finds a different, plain-ODE (non-BVP) testrun using `init_BP_EP`: read it, transcribe the model to Python, and follow `python_cases/codim2_curves.py`'s exact methodology from this session (independently detect the BP via JaxCont's own equilibrium continuation, run the MATLAB driver live against `/home/ziaee/prog/MatCont/MatCont7p6` to get real reference numbers, hardcode them, compare, wire into `cases.json`/`validation.py`/`tests/test_matcont_suite.py` exactly as `US-C2-LP-001`/`US-C2-H-001` were wired in). MATLAB is available at `/home/ziaee/prog/Matlab/R2020a/bin/matlab`.

If the `grep` finds nothing else usable: skip straight to Step 2 (documented fallback) -- do not force a fit using `cstr3.m`/`testbratu2.m`.

- [ ] **Step 2: Record the outcome in the spec**

Edit `docs/superpowers/specs/2026-08-26-branch-point-switching-design.md`'s §6 (Verification), adding a note after the "MatCont cross-validation" bullet recording what was found: either the new supported case's ID, or (if Step 1 found nothing suitable) a short paragraph stating that no plain-ODE MatCont `init_BP_EP` testrun exists among the ones checked, that the feature's correctness evidence rests on the closed-form pitchfork/transcritical fixtures (Tasks 1-2) plus the differentiability/`vmap` checks (Task 4), and that `US-BP-001` (Bratu, BVP-based) remains `unsupported` pending BVP support, unrelated to this feature's own completeness.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-26-branch-point-switching-design.md
# If Step 1 found a usable case, also:
# git add examples/MatCont/python_cases/branch_switch.py examples/MatCont/cases.json examples/MatCont/validation.py tests/test_matcont_suite.py
git commit -m "docs: record MatCont cross-validation outcome for branch-point switching"
```

---

### Task 6: Update the roadmap and architecture docs

**Files:**
- Modify: `notes/ROADMAP.md`
- Modify: `notes/ARCHITECTURE.md`
- Modify: `CHANGELOG.md`

**Read first:** `notes/ROADMAP.md`'s v0.3.0+ section (the `- [ ] Branch switching` line) and its "Do this next" numbered list (the most recent entries, near the end of the file, follow a `N. ✅ **Title** (done DATE) -- summary` format -- match it). `notes/ARCHITECTURE.md` §6 (the provisional `jc.branch_switch` sketch this feature implements -- update it from "provisional/do not implement" to reflect the real, shipped signature).

- [ ] **Step 1: Update ROADMAP.md**

In the v0.3.0+ section, change:
```
- [ ] Branch switching
```
to:
```
- [x] Branch switching (equilibrium BP only; period-doubling switching remains a
      future item) -- *(done <DATE>, see
      [plan](../docs/superpowers/plans/2026-08-26-branch-point-switching.md) and its
      [design spec](../docs/superpowers/specs/2026-08-26-branch-point-switching-design.md))*.
      New `BP` event (`bifurcations/events.py`) distinguishes a genuine branch point from
      an ordinary fold via the augmented Jacobian `[f_u | f_p]`'s rank (both give the same
      real-eigenvalue-crossing signal); `branch_switch()` (`bifurcations/branch_switch.py`)
      computes the crossing branch's tangent via SVD of that same matrix and returns a
      seeded `BifProblem`. `Event.refine()`'s protocol widened to `Optional[EventHit]` so a
      false-positive candidate (a plain fold) can report "not a real hit" -- the first
      `Event` to need this. Verified against closed-form pitchfork/transcritical fixtures
      (with a plain-fold negative control confirming the classification actually
      discriminates), `vmap`-batched, and differentiable in `args`.
```

Add a new numbered "Do this next" entry (matching the most recent entry's number + 1) documenting completion, following the exact format of the entries immediately before it in the file.

- [ ] **Step 2: Update ARCHITECTURE.md §6**

Replace the provisional sketch:
```
- **v0.3 Branch switching** (from a detected event):
  ```python
  new = jc.branch_switch(sol, event=sol.events[0], settings=...)
  ```
```
with the shipped signature (matching Task 2's actual implementation):
```
- **Branch switching** (shipped -- see
  docs/superpowers/specs/2026-08-26-branch-point-switching-design.md):
  ```python
  sol = jc.continuation(prob, p_span=..., events=[jc.BP()])
  bp_event = [e for e in sol.events if e.kind == "branch_point"][0]
  new_prob = jc.branch_switch(f, bp_event, args=..., ds=0.01)
  new_sol = jc.continuation(new_prob, p_span=...)
  ```
```

- [ ] **Step 3: Update CHANGELOG.md**

Add to the `[Unreleased]` section's `### Added` block:
```
- `jc.BP`/`jc.branch_switch`: detect equilibrium branch points (pitchfork/transcritical
  crossings) and compute the crossing branch's tangent, closing the last v0.3.0+ roadmap
  item. Period-doubling switching remains a separate future item.
```

- [ ] **Step 4: Full suite, then commit**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

```bash
git add notes/ROADMAP.md notes/ARCHITECTURE.md CHANGELOG.md
git commit -m "docs: close out the branch-switching roadmap item"
```
