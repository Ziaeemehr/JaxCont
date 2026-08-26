# Period-Doubling Branch Switching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch from a detected `PeriodDoubling` event on a periodic-orbit branch onto the period-doubled cycle that bifurcates there, completing the branch-switching feature (the equilibrium half shipped 2026-08-26).

**Architecture:** A new `branch_switch_pd()` factory in a new `bifurcations/pd_switch.py`. It recomputes the monodromy matrix at the event point to get the `-1` Floquet eigenvector, builds a trajectory guess over the doubled period as `u_base(t mod T) + eps*cos(pi*t/T)*v0`, and hands that to the **existing, unchanged** `problems/periodic.py:periodic_orbit_problem` for resampling and fixed-`p` Newton correction on a doubled mesh. Because an under-sized seed converges silently onto the wrong branch, every candidate is verified by a doubling detector, and the factory searches a `(dp sign) x (eps ladder)` grid, raising rather than returning an unverified problem.

**Tech Stack:** JAX (float32 default), pytest, this codebase's existing periodic-orbit collocation machinery (`problems/periodic.py`, `core/collocation.py`).

**Spec:** `docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md`

**Execution note:** Each task below is written to be handed to a fresh agent (Codex CLI) with no memory of the brainstorming that produced this plan. Every task carries exact file paths, exact function names from the current codebase, complete code, and the verification command. Read the spec first if anything is ambiguous — the spec has the full rationale and the probe data; this plan has the exact steps.

## Global Constraints

- **Python env:** use `/home/ziaee/envs/jaxcont/bin/python`. The system `python`/`python3` is a different install and cannot import JAX.
- **Run tests with** `JAX_PLATFORMS=cpu`, e.g. `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`. Verify the full suite is green after every task.
- **float32 is the default dtype.** Never set `jax_enable_x64=True`.
- **Never set a `tol`/`newton_tol` below ~`1e-6`.** A tolerance under the float32 floor reports `converged=False` forever with no error raised (ROADMAP issue #12). The periodic collocation residual's own float32 floor is ~`3.4e-6`, and `periodic_orbit_problem` uses `tol=1e-5` internally.
- **Do NOT modify** `src/jaxcont/bifurcations/events.py`, `src/jaxcont/core/collocation.py`, `src/jaxcont/problems/periodic.py`, or `src/jaxcont/solvers/protocols.py`. This feature is designed to need no changes to any of them. If you believe one is required, stop and report rather than editing.
- **This feature is eager-only.** It is NOT `vmap`/`grad`-batched, because it calls `periodic_orbit_problem`, whose *construction* is documented as unsafe under an outer `jax.grad`/`jax.vmap` (its phase-condition reference is computed before an internal `lax.while_loop`, so a tracer can leak). This is a knowing, documented scope limit — do not add `jax.vmap`/`jax.grad` tests or examples for this feature, and do not attempt to fix `periodic_orbit_problem`.
- **Commit after every task** with the message given in that task's final step.

## Design facts established by numerical probing (do not contradict these)

These were measured during design against the fixture in Task 2, not assumed. Code that contradicts them is wrong:

1. Correcting at the bifurcation parameter `p*` **fails** — it collapses to the trivial branch (amplitude ~1e-2, float32 noise) for every seed amplitude. The parameter **must** be stepped to `p* + sign*dp` before correcting.
2. Once `p` is stepped off, an ordinary **fixed-`p`** Newton solve converges to the exact doubled branch to ~1e-6. **No pseudo-arclength corrector is needed** — unlike the equilibrium branch-point sub-project, which did need one. Do not build one here.
3. A **constant** `-1` Floquet eigenvector seed is sufficient; the exact eigenfunction is not needed. Square-wave and smooth-cosine modulation gave identical accuracy. **Do not build a variational eigenfunction solver.**
4. An **under-sized `eps` is a silent wrong answer**: it converges cleanly (residual ~7e-6, indistinguishable from success) onto the *original orbit traversed twice*. The doubling detector is mandatory. Measured at `alpha=0.05` (exact doubled amplitude 0.2236): `eps/exact` in 0.1–0.5 collapses; 0.75–5 is correct; ≥10 diverges with residual ~3e3.
5. The doubled branch exists on **only one side** of the bifurcation and which side is system-dependent. Probed directly: `dp=+0.05` found it at every sufficient `eps`; `dp=-0.05` collapsed at every `eps`. A wrong sign is externally indistinguishable from a too-small `eps`, so the search must span both signs.
6. The detector separates the two outcomes by five orders of magnitude: `d_rel ~ 0.436` (genuine doubled) vs `~1e-6` (collapsed).

---

## File Structure

- **Create** `src/jaxcont/bifurcations/pd_switch.py` — `_pd_eigenvector`, `_doubling_measure`, `branch_switch_pd`. One responsibility: turning a PD event into a seeded doubled-period `BifProblem`.
- **Modify** `src/jaxcont/api.py`, `src/jaxcont/__init__.py` — export `branch_switch_pd`.
- **Create** `tests/test_pd_switch.py` — the cubic-saturated fixture and all tests.
- **Create** `examples/example_22_period_doubling_switch.py`.
- **Modify** `notes/ROADMAP.md`, `notes/ARCHITECTURE.md`, `CHANGELOG.md`.

---

### Task 1: `_pd_eigenvector` and `_doubling_measure` helpers

**Files:**
- Create: `src/jaxcont/bifurcations/pd_switch.py`
- Test: `tests/test_pd_switch.py` (create)

**Interfaces:**
- Consumes: `core.collocation.collocation_matrices(ncol) -> (D, E, gauss, gw)` (numpy arrays), `core.collocation.monodromy_matrix(raw_f, D, E, h, mesh_states, coll_states, T, p) -> (n, n)` Array, `core.collocation.Collocation(ntst, ncol)`.
- Produces: `_unpack(U, mesh) -> (mesh_states, coll_states, T, n)`, `_pd_eigenvector(raw_f, U, p, mesh, *, pd_tol=1e-2, imag_tol=1e-3) -> Array` (shape `(n,)`, real, unit norm), and `_doubling_measure(U, mesh, n) -> float`. Task 2 uses all three.

**Read first:** `src/jaxcont/stability/floquet.py` (lines 21–48) — `floquet_multipliers` shows the exact unpacking pattern this task must mirror: how `n` is derived algebraically from `U`, how `mesh_states`/`coll_states`/`T` are sliced out of the flat vector, and how `monodromy_matrix` is called. Also skim `src/jaxcont/core/collocation.py:132-141` for `monodromy_matrix`'s signature.

Context: a periodic `BifProblem`'s state vector `U` packs `ntst` mesh-point states, then `ntst*ncol` collocation-point states, then the scalar period `T` — so `n = (len(U) - 1) // (ntst * (1 + ncol))`.

The detector's rationale (spec §2): for a doubled orbit on a uniform mesh of `ntst_d` points over `2T`, the shift `t -> t + T` is an index roll of `ntst_d/2`. A genuine doubled orbit differs from its own half-period shift; the original orbit traversed twice is identical to it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pd_switch.py`:

```python
"""Tests for period-doubling branch switching (jc.branch_switch_pd).

Fixture: a cubic-saturated variant of the repo's 4D "circle (+) transverse"
system. The repo's EXISTING PD fixture in
tests/test_period_doubling_neimark_sacker.py has a LINEAR transverse block, so
w=0 solves it for every parameter and NO period-doubled branch exists to switch
onto -- correct for detection, useless here. Adding the cubic saturation gives a
genuine doubled branch with closed-form ground truth:

    x' = (1 - r^2) x - y            r^2 = x^2 + y^2
    y' = (1 - r^2) y + x            -> circle, period T = 2*pi
    w' = (alpha + i*beta) w - c |w|^2 w    w = w1 + i*w2, beta = pi/T, c = 1

The linear part rotates w by exactly pi per period, so w(T) = -w(0):
anti-periodic, i.e. a period-doubling at alpha = 0. For alpha > 0 the exact
doubled orbit is |w| = sqrt(alpha/c) with period 2T = 4*pi.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.bifurcations.pd_switch import _doubling_measure, _pd_eigenvector
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT
C_SAT = 1.0
N_STATE = 4


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        p * w1 - BETA * w2 - C_SAT * w1 * m2,
        BETA * w1 + p * w2 - C_SAT * w2 * m2,
    ])


def base_problem(alpha, ntst=10, ncol=4):
    """The w=0 circle orbit at parameter `alpha`, period T."""
    t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
    u = np.stack([np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    mesh = Collocation(ntst=ntst, ncol=ncol)
    prob = periodic_orbit_problem(
        rhs, jnp.asarray(u), jnp.asarray(t), T_EXACT, alpha, mesh
    )
    return prob, mesh


def test_pd_eigenvector_is_supported_on_the_transverse_block():
    # At alpha=0 the transverse multiplier is -exp(0)= -1, and its eigenvector
    # lies entirely in the (w1, w2) plane -- components 2 and 3.
    prob, mesh = base_problem(0.0)
    v = _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)
    v = np.asarray(v)
    assert v.shape == (N_STATE,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    assert abs(v[0]) < 1e-3
    assert abs(v[1]) < 1e-3
    assert np.linalg.norm(v[2:]) > 0.99


def test_pd_eigenvector_rejects_a_point_that_is_not_period_doubling():
    # At alpha=0.5 the transverse multiplier is -exp(0.5*T) ~ -23, nowhere
    # near -1, so this is not a PD point and must be rejected loudly.
    prob, mesh = base_problem(0.5)
    with pytest.raises(ValueError, match="period-doubling"):
        _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)


def _pack_mesh_only(mesh_states, mesh, n):
    """Build a U vector whose mesh-state block is `mesh_states`.

    _doubling_measure only reads the mesh-state block, so the collocation
    block and period can be zeros for this unit test.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    return jnp.concatenate([
        jnp.asarray(mesh_states).flatten(),
        jnp.zeros(ntst * ncol * n),
        jnp.zeros(1),
    ])


def test_doubling_measure_separates_a_doubled_orbit_from_a_retraced_one():
    mesh = Collocation(ntst=20, ncol=4)
    t = np.linspace(0.0, 2 * T_EXACT, 20, endpoint=False)

    # The ORIGINAL orbit traversed twice: identical under a t -> t+T shift.
    retraced = np.stack(
        [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    d_retraced = _doubling_measure(
        _pack_mesh_only(retraced, mesh, N_STATE), mesh, N_STATE
    )
    assert d_retraced < 1e-3

    # A genuine doubled orbit: w rotates by pi per period, so w(t+T) = -w(t).
    amp = np.sqrt(0.05 / C_SAT)
    doubled = np.stack([
        np.cos(t), np.sin(t),
        amp * np.cos(BETA * t), amp * np.sin(BETA * t),
    ], axis=1)
    d_doubled = _doubling_measure(
        _pack_mesh_only(doubled, mesh, N_STATE), mesh, N_STATE
    )
    # Closed form: d = 2*amp, scale = sqrt(1 + amp^2)  ->  ~0.436
    assert d_doubled == pytest.approx(2 * amp / np.sqrt(1 + amp**2), abs=1e-3)
    assert d_doubled > 0.4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_pd_switch.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'jaxcont.bifurcations.pd_switch'`.

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/bifurcations/pd_switch.py`:

```python
"""
Period-doubling branch switching -- see
docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md.

Switches from a detected PeriodDoubling event onto the period-doubled cycle
that bifurcates there. The corrector is the existing
``problems/periodic.py:periodic_orbit_problem`` used unchanged: this module
only builds the seed and verifies the result.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcont.core.collocation import (
    Collocation,
    collocation_matrices,
    monodromy_matrix,
)

PyTree = Any


def _unpack(U: Array, mesh: Collocation):
    """Split a periodic ``U`` into ``(mesh_states, coll_states, T, n)``.

    Mirrors ``stability/floquet.py:floquet_multipliers``: ``mesh`` carries no
    state dimension, so ``n`` is derived algebraically from ``U``'s length.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    n = (U.shape[-1] - 1) // (ntst * (1 + ncol))
    mesh_states = U[: ntst * n].reshape(ntst, n)
    coll_states = U[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    return mesh_states, coll_states, U[-1], n


def _pd_eigenvector(
    raw_f: Callable[[Array, Array, PyTree], Array],
    U: Array,
    p: Array,
    mesh: Collocation,
    *,
    pd_tol: float = 1e-2,
    imag_tol: float = 1e-3,
) -> Array:
    """The real, unit-norm ``-1`` Floquet eigenvector at a period-doubling point.

    Recomputes the monodromy matrix rather than reading a cached eigenvector
    off the event: ``PeriodDoubling.refine`` does not store one, and
    ``branch_switch_pd`` already needs ``raw_f``/``mesh`` for the seed, so
    recomputation costs one small ``n x n`` eigendecomposition and keeps
    ``events.py`` untouched.

    ``pd_tol`` is deliberately loose (1e-2): the event point is a bisection
    midpoint sitting at the float32 collocation residual floor, not a
    Newton-refined point.
    """
    mesh_states, coll_states, T, _n = _unpack(U, mesh)
    D_np, E_np, _, _ = collocation_matrices(mesh.ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    Phi = monodromy_matrix(
        raw_f, D, E, 1.0 / mesh.ntst, mesh_states, coll_states, T, p
    )
    values, vectors = jnp.linalg.eig(Phi)
    idx = int(jnp.argmin(jnp.abs(values + 1.0)))
    lam = values[idx]
    if float(jnp.abs(lam + 1.0)) > pd_tol or float(jnp.abs(jnp.imag(lam))) > imag_tol:
        raise ValueError(
            "not a period-doubling point: the Floquet multiplier nearest -1 is "
            f"{complex(lam):.6g} (needs |lambda + 1| < {pd_tol} and "
            f"|Im lambda| < {imag_tol}). Check that this event came from a "
            "jc.PeriodDoubling() detection on a periodic-orbit branch."
        )
    v = jnp.real(vectors[:, idx])
    norm = float(jnp.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError(
            "the -1 Floquet eigenvector is numerically zero; cannot build a "
            "switching direction."
        )
    return v / norm


def _doubling_measure(U: Array, mesh: Collocation, n: int) -> float:
    """How much the orbit differs from its own half-period shift, relative to
    its scale.

    On a uniform mesh of ``ntst`` points over the doubled period ``2T``, the
    shift ``t -> t + T`` is an index roll of ``ntst // 2`` (``ntst`` is even by
    construction, so the roll is exact). A genuine period-doubled orbit differs
    from that shift; the original orbit traversed twice is identical to it.
    This is the only signal that separates the two -- an under-sized seed
    converges onto the retraced original with a perfectly healthy residual.
    """
    mesh_states = U[: mesh.ntst * n].reshape(mesh.ntst, n)
    shifted = jnp.roll(mesh_states, mesh.ntst // 2, axis=0)
    spread = float(jnp.max(jnp.linalg.norm(mesh_states - shifted, axis=1)))
    scale = float(jnp.max(jnp.linalg.norm(mesh_states, axis=1)))
    if scale < 1e-12:
        return 0.0
    return spread / scale
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_pd_switch.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/pd_switch.py tests/test_pd_switch.py
git commit -m "feat: add PD eigenvector extraction and the doubling detector"
```

---

### Task 2: `branch_switch_pd` — seed construction and verified search

**Files:**
- Modify: `src/jaxcont/bifurcations/pd_switch.py`
- Test: `tests/test_pd_switch.py`

**Interfaces:**
- Consumes: `_pd_eigenvector(raw_f, U, p, mesh, *, pd_tol=1e-2, imag_tol=1e-3) -> Array` and `_doubling_measure(U, mesh, n) -> float` and `_unpack(U, mesh) -> (mesh_states, coll_states, T, n)` from Task 1. `problems.periodic.periodic_orbit_problem(f, u_trajectory, t_trajectory, period0, p0, mesh) -> BifProblem`. `bifurcations.events.EventHit` (fields: `kind: str`, `p: float`, `u: Array`, `index`, `info: dict`).
- Produces: `branch_switch_pd(raw_f, event, mesh, *, dp=0.05, dp_sign=None, eps_ladder=(0.05, 0.1, 0.2, 0.4, 0.8), doubling_tol=1e-3, residual_tol=1e-4, n_samples=None, pd_tol=1e-2) -> BifProblem`. Task 3 exports it; Task 4 uses it.

**Read first:** `src/jaxcont/problems/periodic.py` in full (155 lines) — it is the corrector this task reuses **unchanged**. Note in particular: it takes a *trajectory* guess `(u_trajectory, t_trajectory, period0)` and resamples onto `mesh` itself, it Newton-corrects at fixed `p0` internally with `tol=1e-5`, and it returns `BifProblem(f=residual, u0=U0, p0=p0_arr, args=args, kind="periodic")` so the residual can be evaluated as `prob.f(prob.u0, prob.p0, prob.args)`.

Why the parameter must be stepped off the bifurcation (spec §1, probe 1 vs 2): correcting at `p*` collapses to the trivial branch for every seed amplitude, because the doubled branch has zero amplitude exactly at `p*` — the same pitchfork geometry the equilibrium sub-project hit. Stepping to `p* + sign*dp` first makes an ordinary fixed-`p` solve converge to ~1e-6. **Do not add a pseudo-arclength corrector.**

Why the search grid (spec §2–3): an under-sized `eps` converges cleanly onto the original orbit traversed twice (residual ~7e-6 — no error signal), and the doubled branch exists on only one side of the bifurcation, which side being system-dependent. Both failures look identical from outside, so the factory tries `(sign, eps)` combinations, verifies each with the detector, and raises if none pass. **Returning an unverified problem is not an acceptable outcome.**

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pd_switch.py`:

```python
from jaxcont.bifurcations.events import EventHit
from jaxcont.bifurcations.pd_switch import branch_switch_pd


def pd_event_at_origin():
    """An EventHit at the exact PD point (alpha = 0), built directly.

    Building the event directly rather than running a continuation keeps these
    tests fast and deterministic; Task 4's example covers the detect-then-switch
    path end to end.
    """
    prob, mesh = base_problem(0.0)
    event = EventHit(kind="period_doubling", p=float(prob.p0), u=prob.u0, index=None)
    return event, mesh, prob


def transverse_amplitude(prob, mesh):
    """max |w| over the mesh points of a doubled-period problem."""
    ntst = mesh.ntst
    ms = np.asarray(prob.u0[: ntst * N_STATE].reshape(ntst, N_STATE))
    return float(np.max(np.sqrt(ms[:, 2] ** 2 + ms[:, 3] ** 2)))


def test_switch_lands_on_the_period_doubled_branch():
    event, mesh, _ = pd_event_at_origin()
    dp = 0.05
    switched = branch_switch_pd(rhs, event, mesh, dp=dp)

    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    # Closed-form ground truth: |w| = sqrt(alpha/c), period = 2T.
    assert transverse_amplitude(switched, doubled_mesh) == pytest.approx(
        np.sqrt(dp / C_SAT), abs=1e-3
    )
    assert float(switched.u0[-1]) == pytest.approx(2 * T_EXACT, abs=1e-3)
    assert float(switched.p0) == pytest.approx(dp, abs=1e-6)
    assert switched.kind == "periodic"


def test_switch_doubles_the_mesh():
    event, mesh, _ = pd_event_at_origin()
    switched = branch_switch_pd(rhs, event, mesh, dp=0.05)
    expected_len = 2 * mesh.ntst * (1 + mesh.ncol) * N_STATE + 1
    assert switched.u0.shape[-1] == expected_len


def test_switch_finds_the_branch_without_being_told_which_side_it_is_on():
    # The doubled branch exists only for alpha > 0 here. The default search
    # spans both signs, so it must succeed without a dp_sign hint.
    event, mesh, _ = pd_event_at_origin()
    switched = branch_switch_pd(rhs, event, mesh, dp=0.05, dp_sign=None)
    assert float(switched.p0) > 0


def test_switch_raises_on_the_wrong_side_when_the_sign_is_pinned():
    # Pinned to the side with no doubled branch: every eps collapses, so this
    # must raise rather than hand back the retraced original orbit.
    event, mesh, _ = pd_event_at_origin()
    with pytest.raises(ValueError, match="could not find"):
        branch_switch_pd(rhs, event, mesh, dp=0.05, dp_sign=-1)


def test_switch_raises_rather_than_returning_a_collapsed_branch():
    # THE key regression test for this feature's main hazard: an under-sized
    # seed converges cleanly (residual ~7e-6) onto the ORIGINAL orbit traversed
    # twice. Without the detector this returns a plausible, wrong BifProblem.
    event, mesh, _ = pd_event_at_origin()
    with pytest.raises(ValueError, match="collapsed"):
        branch_switch_pd(rhs, event, mesh, dp=0.05, eps_ladder=(0.01, 0.02))


def test_switch_rejects_a_non_period_doubling_event():
    _event, mesh, prob = pd_event_at_origin()
    fold_event = EventHit(kind="fold", p=float(prob.p0), u=prob.u0, index=None)
    with pytest.raises(ValueError, match="period_doubling"):
        branch_switch_pd(rhs, fold_event, mesh, dp=0.05)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_pd_switch.py -v`
Expected: `ImportError: cannot import name 'branch_switch_pd'`.

- [ ] **Step 3: Write the implementation**

Append to `src/jaxcont/bifurcations/pd_switch.py` (add `from jaxcont.api import BifProblem` and `from jaxcont.bifurcations.events import EventHit` and `from jaxcont.problems.periodic import periodic_orbit_problem` to the imports at the top):

```python
def _periodic_base_samples(mesh_states: Array, T: float, t_samples: np.ndarray) -> np.ndarray:
    """Sample the base orbit at ``t mod T`` by periodic linear interpolation.

    The base orbit's mesh points sit at ``tau = i/ntst`` over ``[0, T)``. The
    wrap-around point is appended so a time just below ``T`` interpolates back
    to ``mesh_states[0]`` rather than clamping.
    """
    base = np.asarray(mesh_states)
    ntst = base.shape[0]
    t_knots = np.concatenate([np.arange(ntst) * (T / ntst), np.array([T])])
    u_knots = np.concatenate([base, base[:1]], axis=0)
    wrapped = np.mod(t_samples, T)
    return np.stack(
        [np.interp(wrapped, t_knots, u_knots[:, c]) for c in range(base.shape[1])],
        axis=1,
    )


def branch_switch_pd(
    raw_f: Callable[[Array, Array, PyTree], Array],
    event: EventHit,
    mesh: Collocation,
    *,
    dp: float = 0.05,
    dp_sign: Optional[int] = None,
    eps_ladder: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.8),
    doubling_tol: float = 1e-3,
    residual_tol: float = 1e-4,
    n_samples: Optional[int] = None,
    pd_tol: float = 1e-2,
) -> BifProblem:
    """Seed the period-doubled cycle bifurcating at a ``PeriodDoubling`` event.

    ``raw_f`` is the ODE right-hand side (``f(u, p, args)``), not the assembled
    collocation residual. ``mesh`` is the ``Collocation`` the original branch was
    continued on; the returned problem lives on ``Collocation(ntst=2*ntst,
    ncol=ncol)``, which preserves per-unit-time resolution across the twice-as-long
    period.

    The parameter is stepped to ``event.p + sign*dp`` before correcting: at the
    bifurcation parameter itself the doubled branch has zero amplitude, so a
    correction there collapses to the trivial branch no matter how it is seeded.
    Once stepped off, the existing ``periodic_orbit_problem`` fixed-``p`` solve
    reaches the doubled branch directly.

    Neither the correct seed amplitude nor the side the doubled branch lies on is
    knowable in advance, and both failures look identical from outside -- a clean
    convergence onto the original orbit traversed twice. So this tries
    ``(sign, eps)`` combinations, verifies each against
    :func:`_doubling_measure`, and returns the first that passes. If none does it
    raises, rather than returning a plausible but wrong branch.

    This factory is eager-only: it calls ``periodic_orbit_problem``, whose
    construction is not safe under an outer ``jax.grad``/``jax.vmap``. The
    returned ``BifProblem`` is itself fine to pass to ``jc.continuation``.

    Raises:
        ValueError: if ``event`` is not a period-doubling event, if the point is
            not actually a period-doubling point, or if no ``(sign, eps)``
            candidate produced a verified period-doubled orbit.
    """
    if event.kind != "period_doubling":
        raise ValueError(
            f"branch_switch_pd needs a period_doubling event, got {event.kind!r}. "
            "Detect one with jc.continuation(..., events=[jc.PeriodDoubling(...)])."
        )

    U_star = jnp.asarray(event.u)
    p_star = float(event.p)
    mesh_states, _coll_states, T_arr, n = _unpack(U_star, mesh)
    T = float(T_arr)

    v0 = np.asarray(
        _pd_eigenvector(raw_f, U_star, jnp.asarray(p_star), mesh, pd_tol=pd_tol)
    )
    orbit_scale = float(jnp.max(jnp.linalg.norm(mesh_states, axis=1)))

    if n_samples is None:
        n_samples = max(240, 12 * mesh.ntst)
    t_samples = np.linspace(0.0, 2 * T, n_samples, endpoint=False)
    base_samples = _periodic_base_samples(mesh_states, T, t_samples)
    # cos(pi*t/T) is anti-periodic over T -- it flips sign between the two
    # copies of the base orbit, which is what makes the guess a period-DOUBLED
    # object rather than a period-T one.
    modulation = np.cos(np.pi * t_samples / T)[:, None] * v0[None, :]

    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    signs = (1, -1) if dp_sign is None else (int(np.sign(dp_sign)),)

    attempts: list[str] = []
    for sign in signs:
        for eps_rel in eps_ladder:
            eps = eps_rel * orbit_scale
            u_seed = base_samples + eps * modulation
            label = f"sign={sign:+d} eps={eps:.4g}"
            try:
                candidate = periodic_orbit_problem(
                    raw_f,
                    jnp.asarray(u_seed),
                    jnp.asarray(t_samples),
                    2 * T,
                    p_star + sign * dp,
                    doubled_mesh,
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                attempts.append(f"{label}: correction raised {type(exc).__name__}")
                continue
            residual = float(
                jnp.linalg.norm(
                    candidate.f(candidate.u0, candidate.p0, candidate.args)
                )
            )
            if not np.isfinite(residual) or residual > residual_tol:
                attempts.append(f"{label}: did not converge (residual {residual:.3g})")
                continue
            measure = _doubling_measure(candidate.u0, doubled_mesh, n)
            if measure > doubling_tol:
                return candidate
            attempts.append(
                f"{label}: collapsed onto the original orbit "
                f"(doubling measure {measure:.3g} <= {doubling_tol})"
            )

    raise ValueError(
        "branch_switch_pd could not find a period-doubled orbit at "
        f"p = {p_star} +/- {dp}. Every candidate either failed to converge or "
        "collapsed back onto the original orbit traversed twice. Try a larger "
        "dp, a wider eps_ladder, or check that this really is a period-doubling "
        "point.\nAttempts:\n  " + "\n  ".join(attempts)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_pd_switch.py -v`
Expected: 9 passed.

If `test_switch_lands_on_the_period_doubled_branch` fails on the amplitude, print the actual value and the `attempts` list before changing any numbers — the ladder and tolerances here were measured, not guessed (see "Design facts" above). Do not loosen `doubling_tol` to make a test pass; that tolerance is what makes the collapse detectable.

- [ ] **Step 5: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/pd_switch.py tests/test_pd_switch.py
git commit -m "feat: add branch_switch_pd for period-doubling branch switching"
```

---

### Task 3: Top-level exports

**Files:**
- Modify: `src/jaxcont/api.py`
- Modify: `src/jaxcont/__init__.py`
- Test: `tests/test_functional_api.py`

**Interfaces:**
- Consumes: `bifurcations.pd_switch.branch_switch_pd` from Task 2.
- Produces: `jc.branch_switch_pd`.

**Read first:** `src/jaxcont/__init__.py` — find the existing line `from jaxcont.bifurcations.branch_switch import branch_switch` (added by the equilibrium sub-project) and the `__all__` list. This task follows exactly that precedent.

Note: `src/jaxcont/bifurcations/taxonomy.py` does **not** need changes — period-doubling already has an entry there from the v0.2 detection work, and its status field means "implemented", not "todo".

- [ ] **Step 1: Write the failing test**

Append to `tests/test_functional_api.py`:

```python
def test_branch_switch_pd_is_exported_at_top_level():
    import jaxcont as jc

    assert hasattr(jc, "branch_switch_pd")
    assert "branch_switch_pd" in jc.__all__
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py::test_branch_switch_pd_is_exported_at_top_level -v`
Expected: FAIL — `AssertionError` on `hasattr`.

- [ ] **Step 3: Add the exports**

In `src/jaxcont/__init__.py`, next to the existing `from jaxcont.bifurcations.branch_switch import branch_switch` line, add:

```python
from jaxcont.bifurcations.pd_switch import branch_switch_pd
```

and add `"branch_switch_pd"` to `__all__` (next to the existing `"branch_switch"` entry).

- [ ] **Step 4: Run the test to verify it passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_functional_api.py -v`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/__init__.py tests/test_functional_api.py
git commit -m "feat: export jc.branch_switch_pd at the top level"
```

---

### Task 4: End-to-end detect-then-switch test and example

**Files:**
- Modify: `tests/test_pd_switch.py`
- Create: `examples/example_22_period_doubling_switch.py`

**Interfaces:**
- Consumes: `jc.branch_switch_pd(raw_f, event, mesh, *, dp=...)` from Tasks 2–3, `jc.PeriodDoubling(raw_f=..., mesh=...)`, `jc.continuation(problem, p_span=..., settings=..., events=...)`, `jc.ContinuationPar(ds=..., max_steps=..., newton_tol=..., compute_stability=...)`.
- Produces: nothing later tasks consume.

**Read first:** `examples/example_08_period_doubling.py` — the existing PD *detection* example on the linear-transverse-block system. It shows the `jc.PeriodDoubling(raw_f=rhs, mesh=mesh)` construction, the `ContinuationPar` settings that work for periodic problems, and the `result.branch.params` / `result.branch.states` accessors. This example is the switching counterpart and uses the cubic-saturated fixture instead, because the linear system has no doubled branch to switch onto.

Important: periodic problems need `newton_tol=1e-5` (not the default `1e-6`) — the collocation residual's float32 floor is ~`3.4e-6`, and a tighter tolerance stalls continuation (every step rejected, branch terminates at the first point).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pd_switch.py`:

```python
import jaxcont as jc


def test_detect_then_switch_end_to_end():
    """The full path: continue the base branch, detect the PD event with
    jc.PeriodDoubling, switch on the detected event, and continue the result."""
    prob, mesh = base_problem(-0.1)
    result = jc.continuation(
        prob,
        p_span=(-0.1, 0.1),
        settings=jc.ContinuationPar(
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=True
        ),
        events=[jc.PeriodDoubling(raw_f=rhs, mesh=mesh)],
    )
    pd_events = [e for e in result.events if e.kind == "period_doubling"]
    assert pd_events, "no period-doubling event detected on the base branch"

    event = pd_events[0]
    assert event.p == pytest.approx(0.0, abs=2e-2)

    dp = 0.05
    switched = jc.branch_switch_pd(rhs, event, mesh, dp=dp)
    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    amp = transverse_amplitude(switched, doubled_mesh)
    # Ground truth is sqrt((p_switched - p_bif)/c); the detected p_bif carries
    # the event's own bisection error, so this is looser than the exact-point test.
    expected = np.sqrt(max(float(switched.p0) - event.p, 0.0) / C_SAT)
    assert amp == pytest.approx(expected, abs=2e-2)
    assert float(switched.u0[-1]) == pytest.approx(2 * T_EXACT, abs=1e-2)
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_pd_switch.py::test_detect_then_switch_end_to_end -v`

This exercises only already-implemented code, so it may pass immediately — that is a valid TDD outcome for an integration test over shipped units. If it FAILS, the failure is real: report the actual event parameter, the amplitude, and the `attempts` list from any `ValueError` before changing tolerances.

- [ ] **Step 3: Write the example**

Create `examples/example_22_period_doubling_switch.py`:

```python
"""
Switch onto a period-doubled cycle
==================================

Detect a period-doubling (PD) bifurcation on a limit-cycle branch, seed the
period-doubled cycle with :func:`jaxcont.branch_switch_pd`, and continue it.

The system is a circle oscillator with a cubically-saturated transverse block::

    x' = (1 - r^2) x - y            r^2 = x^2 + y^2
    y' = (1 - r^2) y + x            -> circle, period T = 2*pi
    w' = (alpha + i*beta) w - |w|^2 w    w = w1 + i*w2, beta = pi/T

The linear part rotates ``w`` by exactly ``pi`` per period, so ``w(T) = -w(0)``:
a period-doubling at ``alpha = 0``. For ``alpha > 0`` the doubled cycle has the
closed form ``|w| = sqrt(alpha)`` with period ``2T``, which is what the right
panel checks against.

Note: period-doubling switching is eager-only -- it builds a new
periodic-orbit problem, and that construction is not safe under an outer
``jax.vmap``/``jax.grad``.
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        p * w1 - BETA * w2 - w1 * m2,
        BETA * w1 + p * w2 - w2 * m2,
    ])


def main():
    os.makedirs("images", exist_ok=True)

    t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
    u_guess = np.stack(
        [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    base = periodic_orbit_problem(
        rhs, jnp.asarray(u_guess), jnp.asarray(t), T_EXACT, -0.1, mesh
    )

    # newton_tol=1e-5, not the 1e-6 default: the collocation residual's float32
    # floor is ~3.4e-6, and a tighter tolerance stalls continuation.
    result = jc.continuation(
        base,
        p_span=(-0.1, 0.1),
        settings=jc.ContinuationPar(
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=True
        ),
        events=[jc.PeriodDoubling(raw_f=rhs, mesh=mesh)],
    )

    pd_events = [e for e in result.events if e.kind == "period_doubling"]
    if not pd_events:
        raise RuntimeError("no period-doubling event detected")
    event = pd_events[0]
    print(f"PD detected at alpha = {event.p:.6f} (exact: 0)")

    switched = jc.branch_switch_pd(rhs, event, mesh, dp=0.05)
    doubled_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    print(
        f"switched seed: alpha = {float(switched.p0):.6f}, "
        f"period = {float(switched.u0[-1]):.6f} (exact: {2 * T_EXACT:.6f})"
    )

    doubled = jc.continuation(
        switched,
        p_span=(float(switched.p0), 0.3),
        settings=jc.ContinuationPar(
            ds=0.02, max_steps=40, newton_tol=1e-5, compute_stability=False
        ),
    )

    n_state = 4
    ntst_d = doubled_mesh.ntst
    amps = [
        float(
            np.max(
                np.sqrt(
                    np.asarray(U[: ntst_d * n_state].reshape(ntst_d, n_state))[:, 2] ** 2
                    + np.asarray(U[: ntst_d * n_state].reshape(ntst_d, n_state))[:, 3] ** 2
                )
            )
        )
        for U in np.asarray(doubled.branch.states)
    ]
    alphas = np.asarray(doubled.branch.params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.asarray(result.branch.params), np.zeros(result.branch.n_valid),
                 "o-", color="C0", label="base cycle (|w| = 0)")
    axes[0].plot(alphas, amps, "o-", color="C1", label="period-doubled cycle")
    axes[0].axvline(event.p, color="black", ls=":", lw=1, label="PD")
    axes[0].set_xlabel(r"$\alpha$")
    axes[0].set_ylabel(r"$\max |w|$")
    axes[0].set_title("Period-doubling branch switching")
    axes[0].legend()

    axes[1].plot(alphas, amps, "o", color="C1", label="JaxCont")
    axes[1].plot(alphas, np.sqrt(np.clip(alphas, 0, None)), "--", color="black",
                 label=r"exact $\sqrt{\alpha}$")
    axes[1].set_xlabel(r"$\alpha$")
    axes[1].set_ylabel(r"$\max |w|$")
    axes[1].set_title("Doubled-cycle amplitude vs closed form")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("images/example_22_period_doubling_switch.png", dpi=150,
                bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the example**

Run: `cd /home/ziaee/git/JaxCont && JAX_PLATFORMS=cpu MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_22_period_doubling_switch.py`
Expected: exits 0, prints a PD parameter near 0 and a switched period near 12.566, writes `images/example_22_period_doubling_switch.png`.

- [ ] **Step 5: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_pd_switch.py examples/example_22_period_doubling_switch.py
git commit -m "test: add end-to-end PD switching test and example"
```

---

### Task 5: MatCont cross-validation (or documented fallback)

**Files:**
- Modify: `docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md`
- Possibly create: `examples/MatCont/python_cases/pd_switch.py`
- Possibly modify: `examples/MatCont/cases.json`, `examples/MatCont/validation.py`, `tests/test_matcont_suite.py`

**Interfaces:**
- Consumes: `jc.branch_switch_pd` from Tasks 2–3.
- Produces: either a real cross-validation case or a documented fallback note in the spec.

MatCont 7.6 is installed at `/home/ziaee/prog/MatCont/MatCont7p6`; MATLAB is at `/home/ziaee/prog/Matlab/R2020a/bin/matlab`.

MatCont's period-doubling switching initializer is `init_PD_LC`.

- [ ] **Step 1: Search for a usable testrun**

Run:

```bash
grep -rl "init_PD_LC" /home/ziaee/prog/MatCont/MatCont7p6/Testruns/*.m
```

Decision rule — apply it honestly, do not force a fit:

- **If a plain-ODE (non-BVP, non-PDE-discretization) testrun is found**, read `examples/MatCont/python_cases/codim2_curves.py` first and follow its exact methodology: transcribe the model to Python, independently locate the PD point with JaxCont's own periodic continuation, switch with `jc.branch_switch_pd`, and compare against hardcoded MatCont reference values exported from a live MATLAB run. Then promote the corresponding `cases.json` entry to `"supported"` and add a test to `tests/test_matcont_suite.py`. Do not build a generic comparison helper for a single case — hardcoded reference rows are this repo's established pattern (see `MC-C2-001`).
- **If nothing plain-ODE and usable is found**, go straight to Step 2. Do not stretch a BVP or otherwise unsuitable case into a comparison.

- [ ] **Step 2: If no usable case exists, document the fallback**

Add to `docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md`, at the end of §6 (Testing):

```markdown
  **MatCont cross-validation outcome (2026-08-27):** no plain-ODE MatCont
  `init_PD_LC` testrun was found among the MatCont 7.6 testruns. The feature's
  correctness evidence therefore rests on the closed-form doubled-cycle fixture
  above (exact amplitude `sqrt(alpha/c)` and exact period `2T`), the collapse
  regression test, and the end-to-end detect-then-switch test, rather than on a
  MatCont reference comparison.
```

Record the *actual* outcome — if Step 1 did find a usable case, write what was validated instead.

- [ ] **Step 3: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 4: Commit**

If the fallback path was taken:

```bash
git add docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md
git commit -m "docs: record MatCont cross-validation outcome for PD switching"
```

If a real case was found:

```bash
git add docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md \
        examples/MatCont/python_cases/pd_switch.py examples/MatCont/cases.json \
        examples/MatCont/validation.py tests/test_matcont_suite.py
git commit -m "test: cross-validate PD branch switching against MatCont"
```

---

### Task 6: Close out the roadmap and architecture docs

**Files:**
- Modify: `notes/ROADMAP.md`
- Modify: `notes/ARCHITECTURE.md`
- Modify: `CHANGELOG.md`

**Read first:** `notes/ROADMAP.md`'s v0.3.0+ section — find the `- [x] Branch switching (equilibrium BP only; period-doubling switching remains a future item)` entry added by the previous sub-project, and the numbered "Do this next" list near the end of the file (most recent entry is `25.`, format `N. ✅ **Title** (done DATE) -- summary`). In `notes/ARCHITECTURE.md` §6, find the `- **Branch switching** (shipped ...)` block.

- [ ] **Step 1: Update ROADMAP.md's v0.3.0+ entry**

Change the opening line of the existing branch-switching entry from:

```
- [x] Branch switching (equilibrium BP only; period-doubling switching remains a
      future item) -- *(done 2026-08-26, see
```

to:

```
- [x] Branch switching (equilibrium BP and period-doubling; branch points on
      cycles remain out of scope) -- *(equilibrium BP done 2026-08-26,
      period-doubling done 2026-08-27, see
```

- [ ] **Step 2: Add a "Do this next" completion entry**

Append after entry `25.`:

```
26. ✅ **Period-doubling branch switching** — done 2026-08-27, see
    [plan](../docs/superpowers/plans/2026-08-27-period-doubling-switching.md) and its
    [design spec](../docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md).
    New `bifurcations/pd_switch.py`: `branch_switch_pd()` turns a detected
    `PeriodDoubling` event into a seeded doubled-period `BifProblem`. Design was settled by
    numerical probing rather than assumption, and two probe results shaped it: (1) correcting
    at the bifurcation parameter collapses to the trivial branch, but stepping `p` off it
    first lets the **existing** `periodic_orbit_problem` fixed-`p` solve reach the doubled
    branch directly — so no pseudo-arclength corrector (unlike the equilibrium sub-project)
    and no variational eigenfunction solve were needed, and `events.py`/`collocation.py`/
    `periodic.py` are untouched; (2) an under-sized seed converges *cleanly* (residual ~7e-6)
    onto the original orbit traversed twice — a silent wrong answer with no error signal — so
    a doubling detector (`max_i ||u[i] - u[i + ntst/2]|| / scale`, ~0.44 genuine vs ~1e-6
    collapsed) is mandatory, and the factory searches both `dp` signs × an `eps` ladder and
    **raises** rather than returning an unverified branch. Validated against a
    cubic-saturated fixture with closed-form ground truth (`|w| = sqrt(alpha/c)`, period
    `2T`). Eager-only: inherits `periodic_orbit_problem`'s construction-time
    `vmap`/`grad` limitation (see item 27). Branch points on cycles (BPC) remain out of scope.
27. **Next up:** make `periodic_orbit_problem` construction `vmap`/`grad`-safe. Its
    phase-condition reference derivative is computed from `p0` *before* the internal
    refinement's `lax.while_loop`, so wrapping the factory in `jax.grad`/`jax.vmap` can leak a
    tracer across that boundary (documented in `problems/periodic.py`'s own docstring). Fixing
    it means recomputing that reference inside the differentiable solve. This blocks the usual
    "ship a vmap-batched + differentiable example" convention for both periodic-orbit
    continuation and period-doubling switching, so it benefits more than one feature — but it
    touches shared machinery current tests depend on, which is why it is its own item rather
    than folded into the switching work.
```

- [ ] **Step 3: Update ARCHITECTURE.md §6**

Directly below the existing `- **Branch switching** (shipped ...)` code block, add:

```
- **Period-doubling switching** (shipped -- see
  docs/superpowers/specs/2026-08-27-period-doubling-switching-design.md):
  ```python
  sol = jc.continuation(periodic_prob, p_span=...,
                        events=[jc.PeriodDoubling(raw_f=f, mesh=mesh)])
  pd = [e for e in sol.events if e.kind == "period_doubling"][0]
  doubled_prob = jc.branch_switch_pd(f, pd, mesh, dp=0.05)
  doubled_sol = jc.continuation(doubled_prob, p_span=...)
  ```
  Eager-only (inherits `periodic_orbit_problem`'s construction-time
  `vmap`/`grad` limitation).
```

- [ ] **Step 4: Update CHANGELOG.md**

Add to the `[Unreleased]` section's `### Added` block:

```
- `jc.branch_switch_pd`: switch from a detected period-doubling bifurcation onto the
  period-doubled cycle, completing the branch-switching feature. Verifies the switched
  orbit is genuinely period-doubled rather than the original orbit traversed twice, and
  raises instead of returning an unverified branch. Eager-only.
```

- [ ] **Step 5: Run the full suite**

Run: `JAX_PLATFORMS=cpu /home/ziaee/envs/jaxcont/bin/python -m pytest tests/ -n auto`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add notes/ROADMAP.md notes/ARCHITECTURE.md CHANGELOG.md
git commit -m "docs: close out the period-doubling switching roadmap item"
```
