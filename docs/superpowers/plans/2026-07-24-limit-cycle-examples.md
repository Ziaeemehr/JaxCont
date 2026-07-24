# Limit-Cycle Examples (Van der Pol, Brusselator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two example scripts demonstrating real limit-cycle continuation (via the existing
`periodic_orbit_problem`/`jc.continuation()` API) for Van der Pol and the Brusselator, closing the
last v0.2.0 ROADMAP checklist item.

**Architecture:** Each script simulates a short trajectory with `scipy.integrate.solve_ivp` (the
user's own simulation — JaxCont doesn't integrate ODEs itself), extracts one period from the tail via
`scipy.signal.find_peaks`, refines it into an exact periodic orbit via `periodic_orbit_problem`, then
continues it with `jc.continuation()` and plots the result. Pure example/documentation content — no
changes to `src/jaxcont/`.

**Tech Stack:** `scipy.integrate.solve_ivp`/`scipy.signal.find_peaks` (already a project dependency),
the existing `periodic_orbit_problem`/`Collocation`/`jc.continuation()` API, `matplotlib`.

## Global Constraints

- No changes to `src/jaxcont/*` — example content only, using the existing public API as documented.
- `scipy` usage is confined to the example scripts (the user's own simulation) — the library itself
  still does not integrate ODEs; this matches, not violates, that architecture principle.
- `t_trajectory` passed to `periodic_orbit_problem` **must** be re-based to start at `0` before the
  call. This is required, not a style preference: `periodic_orbit_problem`'s internal resampling
  computes `t = τ·period0` for `τ∈[0,1]`, so an un-rebased `t_trajectory` (e.g. starting at some
  `t0>0` from a long simulation) falls entirely outside `jnp.interp`'s domain and silently clamps to
  a constant — verified during design: this produces a degenerate `T≈0` "solution" with deceptively
  small residual, not an error.
- `newton_tol` is per-example and must **not** be assumed to transfer between systems: `1e-5` for Van
  der Pol, `1e-4` for Brusselator (same mesh size, different systems — verified during design that
  Brusselator's achievable float32 residual floor is looser than Van der Pol's at this mesh).
- Both examples assert `Branch.stable` holds throughout via `raise` (matching `example_08`/`09`'s
  style), not a silent print.
- The Makefile is not touched (its `examples` target is already stale/incomplete — a separate,
  pre-existing issue, out of scope here).
- All code in this plan was prototyped and numerically verified before being written here — both
  scripts were run end-to-end against the real `periodic_orbit_problem`/`jc.continuation()` pipeline,
  including finding and fixing the two issues above. Exact verified values appear in this plan's
  run-and-verify steps.

## Task Granularity Decision

Two tasks, not three: the `example_03_van_der_pol.py` docstring fix (one line, thematically tied to
Van der Pol) folds into Task 1 rather than getting its own task — it has no independent test cycle
worth a separate reviewer gate, and splitting it out would just add review overhead for one doc line.
Task 2 (Brusselator) is fully independent of Task 1 and could be reviewed in either order.

---

## Background: reading the existing code

- `examples/example_03_van_der_pol.py` already defines `van_der_pol_rhs(u, p, args)` (3-arg
  convention, matching `periodic_orbit_problem`'s expected `f(u, p, args)` signature) and continues
  the *equilibrium* through its degenerate Hopf crossing at `μ=0`. Its docstring's last line,
  "Periodic-orbit continuation is outside JaxCont's current scope," is now stale.
- `examples/example_08_period_doubling.py`/`example_09_neimark_sacker.py` are the structural/stylistic
  template both new examples follow: docstring with `.. math::` LaTeX, `%%`-delimited sections,
  `periodic_orbit_problem`/`Collocation`/`jc.continuation()` calls, a `raise RuntimeError` check
  (not a silent print) for the "did it work" condition, a two-panel `matplotlib` figure saved to
  `images/`, `plt.show()` at the end.
- `src/jaxcont/problems/periodic.py`'s `periodic_orbit_problem(f, u_trajectory, t_trajectory,
  period0, p0, mesh) -> BifProblem` is the function both examples call. `mesh: Collocation` fixes
  `ntst`/`ncol`; the returned `BifProblem.u0` is a flat vector: `ntst` mesh-point states, then
  `ntst*ncol` collocation-point states, then the period `T` (last entry).

---

### Task 1: Van der Pol limit cycle example (+ `example_03` docstring fix)

**Files:**
- Create: `examples/example_10_van_der_pol_limit_cycle.py`
- Modify: `examples/example_03_van_der_pol.py` (one line)

**Interfaces:**
- Consumes: `jaxcont.problems.periodic.periodic_orbit_problem`, `jaxcont.core.collocation.Collocation`,
  `jaxcont.continuation`/`jc.ContinuationPar` (all existing, unmodified).
- Produces: `images/example_10_van_der_pol_limit_cycle.png` (a runnable artifact, not consumed by
  Task 2).

- [ ] **Step 1: Fix the stale docstring line in `example_03_van_der_pol.py`**

In `examples/example_03_van_der_pol.py`, change:

```python
generic, vanishing-amplitude Hopf cycle. Periodic-orbit continuation is
outside JaxCont's current scope.
"""
```

to:

```python
generic, vanishing-amplitude Hopf cycle. See
``examples/example_10_van_der_pol_limit_cycle.py`` for the real limit cycle,
continued away from this degenerate point via periodic-orbit collocation.
"""
```

- [ ] **Step 2: Create `examples/example_10_van_der_pol_limit_cycle.py`**

```python
r"""
Van der Pol limit cycle
========================

Continue the actual Van der Pol limit cycle -- not just the equilibrium's
degenerate Hopf crossing at :math:`\mu=0` shown in
``example_03_van_der_pol.py`` -- via periodic-orbit collocation.

.. math::

    \dot{x} &= y \\
    \dot{y} &= \mu (1 - x^2) y - x

As explained in ``example_03``, the cycle does not emerge as a
vanishing-amplitude Hopf cycle at :math:`\mu=0` (the crossing is degenerate:
zero first Lyapunov coefficient). Starting instead from a short simulation
at :math:`\mu=1` -- comfortably past that degenerate point -- and refining
it into an exact periodic orbit gives a real starting point to continue
into the classic large-:math:`\mu` relaxation-oscillator regime.
"""

# %%
# Setup
# -----
# JaxCont does not integrate ODEs itself: the initial trajectory guess below
# comes from the user's own simulation (``scipy.integrate.solve_ivp``), and
# ``periodic_orbit_problem`` only resamples and refines it.

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

NTST, NCOL, N_STATE = 20, 4, 2


# %%
# Define the system
# ------------------
# Same right-hand side as ``example_03_van_der_pol.py``.


def van_der_pol_rhs(u, p, args):
    x, y = u
    mu = p
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


def _van_der_pol_rhs_np(t, u, mu):
    x, y = u
    return [y, mu * (1.0 - x**2) * y - x]


# %%
# Simulate to find an initial trajectory guess
# ---------------------------------------------
# Simulate at :math:`\mu=1` until the transient dies out, then locate one
# full period in the tail via successive peaks in :math:`x(t)`.
# ``t_trajectory`` must start at 0 -- ``periodic_orbit_problem`` resamples
# using ``t = tau * period0`` for ``tau`` in ``[0, 1]``, so an un-rebased
# time array (still carrying the simulation's raw offset) falls entirely
# outside ``jnp.interp``'s domain and silently clamps to a constant,
# producing a degenerate near-zero-period "solution" with deceptively small
# residual rather than an error -- verified during design.

mu_start = 1.0
raw_solution = solve_ivp(
    _van_der_pol_rhs_np, [0.0, 60.0], [0.5, 0.0], args=(mu_start,),
    max_step=0.01, dense_output=True,
)
tail_times = np.linspace(40.0, 60.0, 4000)
tail_x = raw_solution.sol(tail_times)[0]
peak_indices, _ = find_peaks(tail_x)
peak_times = tail_times[peak_indices]
period_guess = float(np.mean(np.diff(peak_times)))

cycle_start_time = peak_times[-2]
t_trajectory = np.linspace(0.0, period_guess, 60, endpoint=False)
u_trajectory = raw_solution.sol(t_trajectory + cycle_start_time).T

# %%
# Refine into an exact periodic orbit
# -------------------------------------

mesh = Collocation(ntst=NTST, ncol=NCOL)
problem = periodic_orbit_problem(
    van_der_pol_rhs,
    jnp.asarray(u_trajectory),
    jnp.asarray(t_trajectory),
    period_guess,
    mu_start,
    mesh,
)

# %%
# Continue the limit cycle
# --------------------------
# ``compute_stability=True`` stores Floquet multipliers so ``Branch.stable``
# is available to confirm the cycle remains stable throughout.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(mu_start, 4.0),
    settings=jc.ContinuationPar(
        ds=0.05,
        max_steps=250,
        newton_tol=1e-5,
        compute_stability=True,
    ),
)

if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Van der Pol limit cycle to remain stable throughout")

mus = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
print(f"Continued mu in [{mus[0]:.3f}, {mus[-1]:.3f}] over {n_valid} points, stable throughout.")

# %%
# Extract mesh-point states for plotting
# -----------------------------------------
# ``Branch.states`` packs each point's full collocation unknown vector:
# ``ntst`` mesh-point states first, then collocation-point states, then the
# period. The first ``ntst*N_STATE`` entries reshape into the mesh-point
# states -- dense enough (``ntst=20``) for a clean phase-portrait polygon.

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(np.abs(mesh_states[:, :, 0]), axis=1)
print(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}] (expected ~flat, near 2)")
print(f"Period range: [{periods.min():.4f}, {periods.max():.4f}] (expected to grow with mu)")

# %%
# Plot the cycle's shape change and period growth
# ---------------------------------------------------
# Amplitude barely changes for Van der Pol at this normalization -- period
# growth and waveform sharpening are the salient relaxation-oscillator
# signatures, not amplitude growth (contrast with the Brusselator example).

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


def _closed_loop(mesh_pts):
    return np.concatenate([mesh_pts, mesh_pts[:1]], axis=0)


first_loop = _closed_loop(mesh_states[0])
last_loop = _closed_loop(mesh_states[-1])
axes[0].plot(first_loop[:, 0], first_loop[:, 1], "o-", label=f"mu={mus[0]:.2f}")
axes[0].plot(last_loop[:, 0], last_loop[:, 1], "o-", label=f"mu={mus[-1]:.2f}")
axes[0].set(xlabel="$x$", ylabel="$y$", title="Limit cycle shape", aspect="equal")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(mus, periods, "o-")
axes[1].set(xlabel=r"$\mu$", ylabel="period", title="Period growth")
axes[1].grid(alpha=0.25)

fig.tight_layout()
plt.savefig("images/example_10_van_der_pol_limit_cycle.png")
plt.show()
```

- [ ] **Step 3: Run and verify**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_10_van_der_pol_limit_cycle.py`

Expected: exits 0, prints something matching (values within these ranges, verified during design —
this is a real simulation-seeded continuation, not a closed-form check, so match approximately, not
bit-exact):
```
Continued mu in [1.000, ~4.02] over ~138 points, stable throughout.
Amplitude range: [~1.99, ~2.01] (expected ~flat, near 2)
Period range: [~6.66, ~10.25] (expected to grow with mu)
```
and creates `images/example_10_van_der_pol_limit_cycle.png` (check with `ls -la
images/example_10_van_der_pol_limit_cycle.png` — file must exist and be non-empty).

- [ ] **Step 4: Commit**

```bash
git add examples/example_03_van_der_pol.py examples/example_10_van_der_pol_limit_cycle.py images/example_10_van_der_pol_limit_cycle.png
git commit -m "docs: add Van der Pol limit-cycle example, fix stale example_03 scope claim"
```

---

### Task 2: Brusselator limit cycle example

**Files:**
- Create: `examples/example_11_brusselator_limit_cycle.py`

**Interfaces:**
- Consumes: same as Task 1 (`periodic_orbit_problem`, `Collocation`, `jc.continuation`) — fully
  independent of Task 1's file, no shared state.
- Produces: `images/example_11_brusselator_limit_cycle.png`.

- [ ] **Step 1: Create `examples/example_11_brusselator_limit_cycle.py`**

```python
r"""
Brusselator limit cycle
=========================

Continue the Brusselator's limit cycle via periodic-orbit collocation.

.. math::

    \dot{x} &= a - (b+1)x + x^2 y \\
    \dot{y} &= bx - x^2 y

With :math:`a=1` fixed, the origin-shifted fixed point
:math:`(x,y)=(a, b/a)` undergoes a Hopf bifurcation at :math:`b=1+a^2=2`.
Starting a short simulation at :math:`b=2.5` (past the Hopf point) and
refining it into an exact periodic orbit gives a starting point to continue
the cycle as :math:`b` grows.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

NTST, NCOL, N_STATE = 20, 4, 2
A = 1.0


# %%
# Define the system


def brusselator_rhs(u, p, args):
    x, y = u
    b = p
    return jnp.array([A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y])


def _brusselator_rhs_np(t, u, b):
    x, y = u
    return [A - (b + 1.0) * x + x**2 * y, b * x - x**2 * y]


# %%
# Simulate to find an initial trajectory guess
# ---------------------------------------------
# Same rationale as the Van der Pol example: ``t_trajectory`` must start at
# 0 before ``periodic_orbit_problem`` resamples it (see that example's
# comment, or the design spec, for what silently goes wrong otherwise).

b_start = 2.5
raw_solution = solve_ivp(
    _brusselator_rhs_np, [0.0, 80.0], [1.1, 1.1], args=(b_start,),
    max_step=0.01, dense_output=True,
)
tail_times = np.linspace(60.0, 80.0, 6000)
tail_x = raw_solution.sol(tail_times)[0]
peak_indices, _ = find_peaks(tail_x)
peak_times = tail_times[peak_indices]
period_guess = float(np.mean(np.diff(peak_times)))

cycle_start_time = peak_times[-2]
t_trajectory = np.linspace(0.0, period_guess, 60, endpoint=False)
u_trajectory = raw_solution.sol(t_trajectory + cycle_start_time).T

# %%
# Refine into an exact periodic orbit

mesh = Collocation(ntst=NTST, ncol=NCOL)
problem = periodic_orbit_problem(
    brusselator_rhs,
    jnp.asarray(u_trajectory),
    jnp.asarray(t_trajectory),
    period_guess,
    b_start,
    mesh,
)

# %%
# Continue the limit cycle
# --------------------------
# ``newton_tol=1e-4``, not Van der Pol's ``1e-5``: verified during design
# that this system's achievable float32 residual floor at this mesh size is
# looser than Van der Pol's -- ``1e-5`` here stalls continuation (adaptive
# step size collapses toward ``ds_min``, the branch barely advances) even
# though a fresh, independent refinement at a nearby parameter value
# converges cleanly. Do not assume one system's verified tolerance transfers
# to another.

result = jc.continuation(
    problem,
    jc.PseudoArclength(),
    p_span=(b_start, 4.0),
    settings=jc.ContinuationPar(
        ds=0.05,
        max_steps=350,
        newton_tol=1e-4,
        compute_stability=True,
    ),
)

if not bool(jnp.all(result.branch.stable)):
    raise RuntimeError("Expected the Brusselator limit cycle to remain stable throughout")

bs = np.asarray(result.branch.params)
n_valid = result.branch.n_valid
print(f"Continued b in [{bs[0]:.3f}, {bs[-1]:.3f}] over {n_valid} points, stable throughout.")

# %%
# Extract mesh-point states for plotting

states = np.asarray(result.branch.states)
mesh_states = states[:, : NTST * N_STATE].reshape(n_valid, NTST, N_STATE)
periods = states[:, -1]
amplitude = np.max(mesh_states[:, :, 0], axis=1) - np.min(mesh_states[:, :, 0], axis=1)
print(f"Amplitude range: [{amplitude.min():.4f}, {amplitude.max():.4f}] (expected to grow with b)")
print(f"Period range: [{periods.min():.4f}, {periods.max():.4f}]")

# %%
# Plot the cycle's growth
# --------------------------
# Unlike Van der Pol (near-constant amplitude, growing period), the
# Brusselator's amplitude grows substantially with ``b`` -- the salient
# feature to plot here.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


def _closed_loop(mesh_pts):
    return np.concatenate([mesh_pts, mesh_pts[:1]], axis=0)


first_loop = _closed_loop(mesh_states[0])
last_loop = _closed_loop(mesh_states[-1])
axes[0].plot(first_loop[:, 0], first_loop[:, 1], "o-", label=f"b={bs[0]:.2f}")
axes[0].plot(last_loop[:, 0], last_loop[:, 1], "o-", label=f"b={bs[-1]:.2f}")
axes[0].set(xlabel="$x$", ylabel="$y$", title="Limit cycle shape")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(bs, amplitude, "o-")
axes[1].set(xlabel="$b$", ylabel="amplitude (max$-$min of $x$)", title="Amplitude growth")
axes[1].grid(alpha=0.25)

fig.tight_layout()
plt.savefig("images/example_11_brusselator_limit_cycle.png")
plt.show()
```

- [ ] **Step 2: Run and verify**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_11_brusselator_limit_cycle.py`

Expected: exits 0, prints something matching (approximately, as verified during design):
```
Continued b in [2.500, ~4.00] over ~256 points, stable throughout.
Amplitude range: [~2.02, ~4.70] (expected to grow with b)
Period range: [~6.58, ~9.04]
```
and creates `images/example_11_brusselator_limit_cycle.png` (check with `ls -la
images/example_11_brusselator_limit_cycle.png` — file must exist and be non-empty).

- [ ] **Step 3: Commit**

```bash
git add examples/example_11_brusselator_limit_cycle.py images/example_11_brusselator_limit_cycle.png
git commit -m "docs: add Brusselator limit-cycle example"
```

---

## Post-plan state

After this plan: `examples/` has real limit-cycle demonstrations for both systems the v0.2.0 ROADMAP
item names, `example_03_van_der_pol.py`'s stale scope claim is fixed, and the v0.2.0 "Periodic
orbits" epic (periodic-orbit collocation continuation, Floquet multipliers, period-doubling/
Neimark–Sacker detection, and now limit-cycle examples) is complete. No `src/jaxcont/` changes.
