# Equinox Adoption (v0.2 Periodic-Orbit Prep) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve `ARCHITECTURE.md` §4's open `eqx.Module` decision by adding `equinox` as a dependency and proving, via one throwaway scaffold type, that its static/traced field split works end-to-end through `jit`/`vmap` — giving future periodic-orbit work (v0.2) a proven pattern to build on instead of a blank page.

**Architecture:** One new internal (non-exported) module, `src/jaxcont/core/_periodic_eqx_scaffold.py`, defines `CollocationMeshScaffold(eqx.Module)` with `ntst`/`ncol` as `eqx.field(static=True)` compile-time constants and `mesh` as a traced `Array`. A test file exercises it under `jit` and `vmap`. No existing code changes; two docs get a status update.

**Tech Stack:** `equinox` (new dependency), `jax`, `pytest`.

## Global Constraints

- Add `equinox` to `pyproject.toml`'s main `dependencies` list (not `dev`/`docs` extras) — verbatim from the spec, since the scaffold lives in `src/jaxcont/core/` and is a real runtime import.
- The scaffold module must NOT be exported from `src/jaxcont/__init__.py` — matches the existing convention (`ARCHITECTURE.md` §8) of keeping not-yet-real v0.2/v0.3 types out of the top-level surface.
- No changes to the v0.1 equilibrium types (`BifProblem`, `Branch`, `ContinuationResult`) — they stay hand-rolled `register_pytree_node` dataclasses, per the existing 2026-07-19 decision.
- Out of scope: no `Collocation` predictor, no `PeriodDoubling`/`LPC`/`NS` events, no `BifProblem.kind="periodic"` wiring — this plan is prep only.
- Reference spec: [docs/superpowers/specs/2026-07-22-equinox-adoption-design.md](../specs/2026-07-22-equinox-adoption-design.md).

---

### Task 1: Add `equinox` as a runtime dependency

**Files:**
- Modify: `pyproject.toml:29-35` (`dependencies` list)

**Interfaces:**
- Produces: `equinox` importable as `eqx` anywhere in `src/jaxcont/`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, change (currently lines 29-35):

```toml
dependencies = [
    "jax>=0.3.0",
    "jaxlib>=0.3.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
]
```

to:

```toml
dependencies = [
    "jax>=0.3.0",
    "jaxlib>=0.3.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "equinox>=0.11.0",
]
```

- [ ] **Step 2: Reinstall the package in editable mode so the new dependency is honored**

Run: `pip install -e . --no-deps` (equinox is already installed in this environment at 0.13.2, satisfying `>=0.11.0`, so no download is needed — this just re-reads `pyproject.toml`'s metadata).

- [ ] **Step 3: Verify `equinox` imports cleanly in this environment**

Run: `python3 -c "import equinox as eqx; print(eqx.__version__)"`
Expected: prints a version string (e.g. `0.13.2`), no `ImportError`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add equinox as a runtime dependency for v0.2 periodic-orbit types"
```

---

### Task 2: Build the `CollocationMeshScaffold` pattern-proof and its test

**Files:**
- Create: `src/jaxcont/core/_periodic_eqx_scaffold.py`
- Test: `tests/test_equinox_scaffold.py`

**Interfaces:**
- Consumes: `equinox` (Task 1).
- Produces: `CollocationMeshScaffold(eqx.Module)` with fields `ntst: int` (static), `ncol: int` (static), `mesh: Array` (traced) — importable as `from jaxcont.core._periodic_eqx_scaffold import CollocationMeshScaffold`. Not exported from `jaxcont/__init__.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_equinox_scaffold.py`:

```python
"""
Proves the eqx.Module static/traced field-split pattern that v0.2's real
periodic-orbit types (Collocation predictor, mesh/ntst/ncol) will need,
ahead of those types existing. See
docs/superpowers/specs/2026-07-22-equinox-adoption-design.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcont.core._periodic_eqx_scaffold import CollocationMeshScaffold


def _make_scaffold(ntst=4, ncol=3):
    return CollocationMeshScaffold(
        ntst=ntst, ncol=ncol, mesh=jnp.linspace(0.0, 1.0, 5)
    )


def _eval(m: CollocationMeshScaffold, period):
    # Uses ntst/ncol in a *shape* position -- only legal if they are static,
    # not traced. A wrong static/traced split raises a
    # TracerArrayConversionError/ConcretizationTypeError here.
    pad = jnp.zeros((m.ntst, m.ncol))
    return m.mesh * period + pad.sum()


def test_scaffold_runs_eagerly():
    m = _make_scaffold()
    result = _eval(m, jnp.array(2.0))
    expected = jnp.linspace(0.0, 1.0, 5) * 2.0
    assert jnp.allclose(result, expected)


def test_scaffold_jits():
    m = _make_scaffold()
    result = jax.jit(_eval)(m, jnp.array(2.0))
    expected = jnp.linspace(0.0, 1.0, 5) * 2.0
    assert jnp.allclose(result, expected)


def test_scaffold_vmaps_over_traced_input_with_static_config_fixed():
    m = _make_scaffold()
    periods = jnp.array([1.0, 2.0, 3.0])

    batch = jax.vmap(lambda p: _eval(m, p))(periods)

    expected = jnp.stack([jnp.linspace(0.0, 1.0, 5) * p for p in [1.0, 2.0, 3.0]])
    assert batch.shape == (3, 5)
    assert jnp.allclose(batch, expected)


def test_ntst_ncol_are_static_python_ints_not_traced():
    m = _make_scaffold(ntst=4, ncol=3)
    assert isinstance(m.ntst, int)
    assert isinstance(m.ncol, int)

    # Changing a static field changes the pytree's structure (it's part of
    # the jit cache key), unlike a traced field -- confirm this holds.
    m2 = _make_scaffold(ntst=6, ncol=3)
    leaves1, treedef1 = jax.tree_util.tree_flatten(m)
    leaves2, treedef2 = jax.tree_util.tree_flatten(m2)
    assert treedef1 != treedef2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu pytest tests/test_equinox_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.core._periodic_eqx_scaffold'`

- [ ] **Step 3: Implement the scaffold module**

Create `src/jaxcont/core/_periodic_eqx_scaffold.py`:

```python
"""
Throwaway scaffold proving the `eqx.Module` static/traced field split works
end-to-end (jit + vmap) for the v0.2 periodic-orbit types.

This is NOT a real predictor and is not exported from `jaxcont.__init__`.
Delete and replace with a real `Collocation` predictor once periodic-orbit
continuation is actually built. See
docs/superpowers/specs/2026-07-22-equinox-adoption-design.md for the
rationale and docs/superpowers/plans/2026-07-22-equinox-adoption.md for how
this was built.
"""

import equinox as eqx
from jax import Array


class CollocationMeshScaffold(eqx.Module):
    """Static mesh-size config (ntst/ncol) + a traced mesh-point array."""

    ntst: int = eqx.field(static=True)
    ncol: int = eqx.field(static=True)
    mesh: Array
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu pytest tests/test_equinox_scaffold.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `JAX_PLATFORMS=cpu pytest tests/ -q`
Expected: same pass count as before this task, plus the 4 new tests, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/core/_periodic_eqx_scaffold.py tests/test_equinox_scaffold.py
git commit -m "feat: add throwaway eqx.Module scaffold proving static/traced split for v0.2"
```

---

### Task 3: Mark the `eqx.Module` decision resolved in the docs

**Files:**
- Modify: `notes/ARCHITECTURE.md:174-181` (§4 "Decision (2026-07-19)" callout)
- Modify: `notes/ROADMAP.md:426-434` (item 2 of "Engineering / architecture recommendations for v0.2")

**Interfaces:**
- Consumes: nothing (doc-only task; references Task 2's file by name).
- Produces: nothing consumed by later tasks (this is the last task in the plan).

- [ ] **Step 1: Update `ARCHITECTURE.md`**

In `notes/ARCHITECTURE.md`, change (currently lines 174-181):

```markdown
> **Decision (2026-07-19):** stay with hand-rolled `register_pytree_node` dataclasses for the v0.1
> equilibrium types (`BifProblem`, `Branch`, `ContinuationResult`) — they shipped, are tested, and
> are flat enough not to need more. **Adopt `equinox` starting with the v0.2 periodic-orbit types**
> (mesh, `ntst`/`ncol`, phase condition, static-vs-traced fields) and the growing set of pluggable
> protocol implementations (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events, solver
> variants) — that is exactly the case `Module`/`field(static=...)` exists for, and by then the
> extra dependency pays for itself. Do not churn the already-working v0.1 types to match. See
> [ROADMAP.md "Engineering recommendations for v0.2"](ROADMAP.md) for the full rationale.
```

to:

```markdown
> **Decision (2026-07-19):** stay with hand-rolled `register_pytree_node` dataclasses for the v0.1
> equilibrium types (`BifProblem`, `Branch`, `ContinuationResult`) — they shipped, are tested, and
> are flat enough not to need more. **Adopt `equinox` starting with the v0.2 periodic-orbit types**
> (mesh, `ntst`/`ncol`, phase condition, static-vs-traced fields) and the growing set of pluggable
> protocol implementations (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events, solver
> variants) — that is exactly the case `Module`/`field(static=...)` exists for, and by then the
> extra dependency pays for itself. Do not churn the already-working v0.1 types to match. See
> [ROADMAP.md "Engineering recommendations for v0.2"](ROADMAP.md) for the full rationale.
>
> **Resolved (2026-07-22):** `equinox` is now a runtime dependency and the static/traced field
> split is proven end-to-end (jit + vmap) by a throwaway scaffold,
> [`core/_periodic_eqx_scaffold.py`](../src/jaxcont/core/_periodic_eqx_scaffold.py) (not exported
> from `jaxcont.__init__`). The real periodic-orbit types (`Collocation` predictor,
> `PeriodDoubling`/`LPC`/`NS` events) still don't exist — this only removes the open decision and
> gives them a proven pattern to build on. See
> [docs/superpowers/specs/2026-07-22-equinox-adoption-design.md](../docs/superpowers/specs/2026-07-22-equinox-adoption-design.md).
```

- [ ] **Step 2: Update `ROADMAP.md`**

In `notes/ROADMAP.md`, change (currently lines 426-434):

```markdown
2. **Resolve the `eqx.Module` "open decision" (ARCHITECTURE.md §4, line ~170) now, before periodic
   orbits land.** v0.1's `BifProblem`/`Branch` are flat enough that hand-rolled
   `register_pytree_node` dataclasses work fine (and that's what's shipped, zero new deps — good
   call for v0.1). Periodic-orbit problems add real structure (mesh, `ntst`/`ncol`, a phase
   condition, static vs. traced fields) and v0.2/v0.3 add several more pluggable protocol
   implementations (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events, more solver
   variants) — exactly the case equinox's `Module`/`field(static=...)` idiom exists for.
   Recommendation: adopt `equinox` starting with the v0.2 periodic-orbit types, leave the already-
   shipped v0.1 equilibrium types as-is (not worth churning a working, tested surface).
```

to:

```markdown
2. ✅ **Resolve the `eqx.Module` "open decision" (ARCHITECTURE.md §4, line ~170) now, before
   periodic orbits land.** *(done 2026-07-22 — see
   [docs/superpowers/plans/2026-07-22-equinox-adoption.md](../docs/superpowers/plans/2026-07-22-equinox-adoption.md)
   and its [design spec](../docs/superpowers/specs/2026-07-22-equinox-adoption-design.md))*
   `equinox` is now a runtime dependency; a throwaway `CollocationMeshScaffold`
   (`core/_periodic_eqx_scaffold.py`, not exported from `jaxcont.__init__`) proves the
   static-vs-traced field split (`eqx.field(static=True)` for `ntst`/`ncol`, traced `mesh` array)
   works end-to-end under `jit` and `vmap` — see `tests/test_equinox_scaffold.py`. The real
   periodic-orbit types (`Collocation` predictor, `PeriodDoubling`/`LPC`/`NS` events) are still not
   built; this only removes the open decision and gives that future work a proven pattern. v0.1's
   `BifProblem`/`Branch` are untouched, per the original recommendation.
```

- [ ] **Step 3: Verify the docs render sensibly**

Run: `grep -n "Resolved (2026-07-22)" notes/ARCHITECTURE.md && grep -n "done 2026-07-22" notes/ROADMAP.md`
Expected: both greps print one matching line each (confirms the edits landed).

- [ ] **Step 4: Commit**

```bash
git add notes/ARCHITECTURE.md notes/ROADMAP.md
git commit -m "docs: mark eqx.Module decision resolved for v0.2 periodic-orbit prep"
```

---

## Final verification

- [ ] Run the full suite one more time: `JAX_PLATFORMS=cpu pytest tests/ -q` — expect all green.
- [ ] Confirm `jaxcont.core._periodic_eqx_scaffold` is NOT importable from the top-level package: `python3 -c "import jaxcont as jc; assert not hasattr(jc, 'CollocationMeshScaffold')"` — expect no `AssertionError`.
