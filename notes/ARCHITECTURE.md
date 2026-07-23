# JaxCont Architecture & API Design

**Status:** Draft / proposal (2026-07-18)
**Decision:** Functional, diffrax-style top-level API ("Sketch A").
**Relationship to roadmap:** This file defines the *shape* the code should grow into.
[ROADMAP.md](ROADMAP.md) defines *what ships when*. If they disagree on scope, ROADMAP wins;
if they disagree on API shape, this file wins.

---

## 1. Design principles

1. **Pure functions at the core.** The user's system is `f(u, p, args) -> residual`, a pure
   JAX function. No hidden state, no mutation. This is what makes `jit`/`vmap`/`grad` work.
2. **Autodiff, not finite differences.** All Jacobians (`df/du`, `df/dp`) come from
   `jax.jacfwd`/`jacrev`. (Done for the corrector and tangent as of 2026-07-18.)
3. **Composition over flags.** Behaviour is selected by passing *objects*
   (`alg=PseudoArclength()`, `events=[Fold(), Hopf()]`), not by boolean constructor flags.
   Objects compose and extend; flags explode combinatorially.
4. **Pluggable solvers.** Linear solve, eigensolve, and Newton are protocols with a dense
   default. This is the seam that later enables matrix-free / iterative / GPU-scale solves
   without touching the continuation logic. (BifurcationKit's biggest architectural win.)
5. **Lean into JAX — this is the reason the project exists.** The point is not "Julia but in
   Python". It is the set of capabilities that fall out of a pure-functional, autodiff-native,
   XLA-compiled core: batched continuation via `vmap`, **differentiable** bifurcation analysis
   via `grad`, and one code path across CPU/GPU/TPU. See §3, which is the design's north star.
6. **The JAX-ecosystem sibling is diffrax.** Match its feel (`diffeqsolve(terms, solver, ...)`
   → `continuation(problem, alg, ...)`, PyTree state, everything jittable). Mine MATCONT for
   its bifurcation *taxonomy and naming*, not its API.

---

## 2. Performance model (why the API is shaped this way)

**Finding (2026-07-18):** JIT-compiling the corrector inner loop alone did **not** measurably
speed up continuation on small systems (~21 ms/point before and after, n=1 and n=3). The cost
is dominated by the **Python-level outer loop**: one dispatch per step for the corrector *plus*
separate dispatches for tangent and eigenvalues, plus per-step host syncs (`float()`/`bool()`
on JAX scalars) and Python bookkeeping. The systems are dispatch/sync-bound, not compute-bound.

**Implication:** The "high-performance JAX" thesis is *not* delivered by JIT-ing one kernel.
It needs one of:

- **(a) Whole-loop JIT** — express the *entire* predictor-corrector sweep as a single
  `lax.while_loop` / `lax.scan` so one continuation run is one dispatched program. Requires the
  pure functional core this document specifies (fixed-size buffers, no Python-side branching on
  traced values, detection folded into the scan). This is the real win for a single long run.
- **(b) `vmap` batching** — run many continuations (parameter sweeps, ensembles, initial
  conditions) in one vectorized call. This is the *bigger* win and needs `continuation()` to be
  a pure function of its inputs — which the functional API gives for free and the OO API does
  not.

The functional design is therefore not cosmetic: it is a precondition for both (a) and (b), and
for everything in §3. Sequencing: ship a correct functional API over the current Python loop
first (v0.1.0), then migrate the loop internals to `lax.scan` (a) behind the same API (v0.1.x),
with `vmap` (b) as a documented capability throughout.

---

## 3. The JAX advantage — capabilities Julia / MATCONT don't offer

This is the design's reason to exist. Be honest about the split: some of this is *genuinely hard
to match* in the Julia/MATLAB tools; some is merely *cleaner or free* here. Both matter, but we
should market and prioritize the first row.

| Capability | JaxCont | BifurcationKit.jl | MATCONT |
|---|---|---|---|
| **`vmap`-batched continuation** (many sweeps as ONE compiled kernel) | ✅ native, GPU-parallel | ⚠️ threads/`EnsembleProblem`, not one kernel | ❌ scripted loop |
| **`grad` *through* a bifurcation diagram** (differentiate a fold/Hopf location w.r.t. params) | ✅ native via implicit diff | ⚠️ partial (AD of Jacobians, not end-to-end) | ❌ |
| **Optimize over bifurcation properties** (optax/inverse design) | ✅ composes with JAX ecosystem | ⚠️ manual | ❌ |
| Autodiff Jacobians / `df/dp` | ✅ | ✅ | ❌ (FD) |
| GPU / TPU, same code path | ✅ free | ✅ (CUDA.jl, some effort) | ❌ |
| Matrix-free iterative solves (large systems) | ✅ `jax.scipy.sparse.linalg` | ✅ (KrylovKit) | ❌ |

### 3.1 Batched continuation (`vmap`) — the flagship

Continuation is embarrassingly parallel across *anything that parametrizes the problem*: a second
(unfolded) parameter, initial guesses for multistart, ensembles of a stochastic/learned system,
or a grid for a two-parameter scan. Because `continuation()` is a pure function of a PyTree
`BifProblem`, batching is one `vmap` — a **single** XLA program, ideal for GPU:

```python
# Sweep a SECOND parameter b (held in args), 500 values, in one vectorized call:
def run_for_b(b):
    return jc.continuation(prob.at(args={"b": b}), p_span=(0.0, 5.0))
diagrams = jax.vmap(run_for_b)(jnp.linspace(-1.0, 1.0, 500))   # CPU or GPU, one dispatch

# Multistart from many initial guesses (find disconnected branches):
diagrams = jax.vmap(lambda u0: jc.continuation(prob.at(u0=u0), p_span=...))(u0_grid)
```

Design requirement this imposes: **fixed-shape outputs**. A `vmap`'d run cannot early-exit at a
Python level or return ragged branches — so `Branch` uses fixed-length buffers of size
`max_steps` plus a `n_valid` count / validity mask (§4.3). This is exactly the discipline
whole-loop `lax.scan` (§2a) already requires, so the two goals reinforce each other.

### 3.2 Differentiable bifurcation analysis (`grad`) — the genuinely novel one

Once a bifurcation point is defined implicitly by `G(x*, θ) = 0` (extended system for a
fold/Hopf, θ = system parameters), the **implicit function theorem** gives `dx*/dθ` without
differentiating through Newton's iterations. Wrapping the solve in `jax.lax.custom_root` /
`jax.custom_vjp` makes the *location of a bifurcation* a differentiable function of the
parameters:

```python
def fold_parameter(theta):
    prob = jc.bif_problem(f, u0, p0, args=theta)
    sol  = jc.continuation(prob, p_span=..., events=[jc.Fold()])
    return sol.events[0].p          # parameter value at the fold

dfold_dtheta = jax.grad(fold_parameter)(theta0)     # sensitivity of the fold to design params
```

This unlocks use cases the Julia/MATLAB tools don't do natively:
- **Inverse design / control:** optimize θ so a fold/Hopf sits at a target value
  (`optax` + `jax.grad` over `fold_parameter`).
- **Sensitivity & robustness:** rank which parameters most move a stability boundary.
- **Learned vector fields:** `f` can be a neural network (equinox/flax); continue and
  differentiate the equilibria of a *trained* dynamical system — bifurcation analysis inside a
  training loop.

Implementation note: the differentiable seam lives in the corrector/event solve, not the outer
Python loop — so it can be delivered incrementally and is compatible with (§2a) whole-loop scan.

**What works today** (verified in `examples/example_07_differentiable.py`, tests in
`tests/test_functional_api.py::TestDifferentiableFold`):
- **Reverse-mode `jax.grad` of a fold location — DONE** via `jc.fold_parameter` /
  `jc.fold_point` ([bifurcations/fold_solve.py](../src/jaxcont/bifurcations/fold_solve.py)). The
  fold is the extended system `G(u,p,v;θ)=0` (`f=0`, `f_u·v=0`, `‖v‖=1`), solved by Newton and
  wrapped in `jax.custom_vjp` implementing the implicit function theorem — so the inner
  `while_loop` is irrelevant to the gradient. Exact to analytic on `f=u²−θu+p`
  (`dp*/dθ = θ/2`), including full Jacobians w.r.t. vector-valued `θ`:

  ```python
  fold_p = lambda theta: jc.fold_parameter(f, u_guess, p_guess, theta)
  jax.grad(fold_p)(theta0)        # exact sensitivity of the fold location
  ```

  Note this differentiates the *isolated fold solve*, not a `continuation()` call. The naive
  `jax.grad` over a whole `continuation(...events=[Fold()])` run still won't work (see below);
  the supported path is: continue to find/seed the fold, then refine + differentiate with
  `fold_point`. Analogous Hopf/codim-2 extended-system solvers are the natural follow-ups.
- **Forward-mode `jacfwd` works *through* the whole-loop engine** — `lax.while_loop` supports
  forward-mode AD, so the sensitivity of a computed branch to a parameter is available out of the
  box (example matches finite differences to 5 digits).
- **Reverse-mode `jax.grad` does *not* differentiate through `lax.while_loop`** — so reverse-mode
  over the full continuation *sweep* is unsupported; use the implicit-diff solvers above for
  reverse-mode gradients of bifurcation quantities.

### 3.3 One code path, many devices

No CUDA-specific code. The same `continuation()` runs on CPU for a 1-D toy and on GPU/TPU for a
10⁴-dimensional discretized PDE, chosen by `JAX_PLATFORMS` / device placement. Combined with
§3.1, a GPU can run hundreds of continuations of a large system concurrently.

**Consistency with lyapax:** these are the same three levers lyapax already exposes (`vmap`
sweeps, differentiable exponents, matrix-free scaling). Presenting them identically in both
packages is what makes the two read as one ecosystem (§9).

---

## 4. The stable core contract (design now, keep stable)

These names/signatures are the public spine. Changing them post-1.0 is expensive, so they get
designed up front. Everything in §6 is deferred.

> **Note on `eqx.Module`:** used below as shorthand for "an immutable, JAX-registered PyTree".
> Concretely this can be a `jax.tree_util.register_pytree_node`-decorated frozen dataclass (zero
> new deps) or [equinox](https://github.com/patrick-kidger/equinox)'s `Module` (adds a dep, but
> is the diffrax-native choice and removes boilerplate). Whichever we choose, the requirement is
> the same: problems, algorithms, and results must be PyTrees so they can cross `jit`/`vmap`
> boundaries.
>
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

### 4.1 Problem — ONE type

Replaces the current split between `ContinuationProblem` (core) and
`EquilibriumProblem`/`PeriodicOrbitProblem` (problems/).

```python
class BifProblem(eqx.Module):        # a PyTree; frozen dataclass-like
    f: Callable[[Array, Array, PyTree], Array]   # f(u, p, args) -> residual, pure
    u0: Array                                    # initial state
    p0: float | Array                            # initial continuation-parameter value
    args: PyTree = None                          # extra static/dynamic params
    kind: Literal["equilibrium", "periodic", "bvp"] = "equilibrium"

    def at(self, *, u0=None, p0=None, args=None) -> "BifProblem": ...
    # cheap functional "copy-with-overrides"; enables vmap over p0 / u0 / args (§3.1)
    def as_rhs(self, p) -> Callable[[Array], Array]: ...
    # autonomous rhs(u) frozen at p — the lyapax bridge (§9)
```

Notes:
- The continuation parameter is the explicit second argument `p`, not a string key into a
  params dict. This removes the `evaluate_rhs` dict-rebuild per call and makes `df/dp` a plain
  `jacfwd(f, argnums=1)`.
- Other parameters live in `args` (any PyTree). This is deliberately the axis you `vmap`/`grad`
  over (§3): a second parameter, a design vector θ, or neural-net weights all live in `args`.
- Continue a *different* parameter by writing `f` to take that one as `p` — or, later, a
  `reparametrize()` helper.

### 4.2 Entry point — ONE function

```python
def continuation(
    problem: BifProblem,
    alg: ContinuationAlgorithm = PseudoArclength(),
    *,
    settings: ContinuationPar = ContinuationPar(),
    p_span: tuple[float, float],
    events: Sequence[Event] = (),
    solvers: Solvers = Solvers(),         # linear/eigen bundle
) -> ContinuationResult: ...
```

Good defaults keep the simple call short:

```python
sol = jc.continuation(prob, p_span=(0.5, 1.5))          # simplest
sol = jc.continuation(prob, jc.Natural(), p_span=(0,1)) # swap algorithm
```

An optional thin OO facade may exist for discoverability, but the functional form is canonical
and is the one that is `jit`/`vmap`/`grad`-safe.

### 4.3 Result / Branch

```python
class Branch(eqx.Module):
    params: Array          # (max_steps,)     fixed-length buffer (§3.1)
    states: Array          # (max_steps, state_dim)
    tangents: Array        # (max_steps, state_dim + 1)
    eigenvalues: Array | None
    stable: Array | None   # (max_steps,) bool
    n_valid: Array         # scalar int; entries [:n_valid] are real, rest are padding

class ContinuationResult(eqx.Module):
    branch: Branch
    events: list[EventHit]        # detected folds/hopf/user events, with refined locations
    stats: RunStats               # steps, accepts/rejects, newton iters, wall time
    # convenience: .plot(), .save(), .get_point(i), .branch.at_param(p)
```

Fixed-length buffers + `n_valid` are what let a whole run be one `lax.scan` (§2a) and let many
runs stack under `vmap` (§3.1). Ragged/growing lists would forfeit both.

### 4.4 Algorithms (predictor + corrector strategy)

```python
class ContinuationAlgorithm(Protocol): ...
class Natural(ContinuationAlgorithm): ...          # natural-parameter
class PseudoArclength(ContinuationAlgorithm):      # default; passes folds
    predictor: Predictor = Secant()                # Tangent() | Secant() | Polynomial()
```

### 4.5 Settings

```python
class ContinuationPar(eqx.Module):
    ds: float = 0.01
    ds_min: float = 1e-5
    ds_max: float = 0.1
    max_steps: int = 1000        # also the Branch buffer length (§4.3)
    adaptive: bool = True
    newton_tol: float = 1e-6
    newton_max_iter: int = 20
    compute_stability: bool = True
```

### 4.6 Solver protocols (pluggable; dense default now, matrix-free later)

```python
class LinearSolver(Protocol):
    def __call__(self, A, b) -> Array: ...     # Dense() default; GMRES()/BiCGStab() later
class EigenSolver(Protocol):
    def __call__(self, A) -> Array: ...        # DenseEigen() default; Arnoldi()/shift-invert later

@dataclass(frozen=True)                        # plain dataclass bundle, not eqx.Module --
class Solvers:                                 # never crosses a jax.jit boundary itself
    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()
```

(`NewtonSolver`/`BorderedNewton` removed from this sketch — not implemented as
part of the `Solvers` bundle; see `src/jaxcont/solvers/protocols.py` for the
implemented `LinearSolver`/`EigenSolver`/`Dense`/`DenseEigen`.)

### 4.7 Events (unifies detection; replaces bespoke BifurcationDetector)

```python
class Event(Protocol):
    def test_function(self, state) -> Array: ...  # scalar; zero-crossing = event
    def refine(self, ...) -> EventHit: ...        # bisection to locate precisely

class Fold(Event): ...     # test: tangent's dp component / det(df/du) sign change
class Hopf(Event): ...     # test: pair of eigenvalues crossing imaginary axis
# users subclass Event for custom detections
```

---

## 5. Minimal end-to-end (target UX)

```python
import jaxcont as jc
import jax, jax.numpy as jnp

def pitchfork(u, p, args):
    return p * u - u**3

prob = jc.BifProblem(pitchfork, u0=jnp.array([0.1]), p0=0.5)

sol = jc.continuation(prob, jc.PseudoArclength(),
                      p_span=(0.5, 1.5), events=[jc.Fold()])
sol.branch.params, sol.branch.states, sol.events

# Batched sweep — the JAX payoff (one vectorized call, CPU or GPU; §3.1):
solve_one = lambda p0: jc.continuation(prob.at(p0=p0), p_span=(p0, p0 + 1.0))
sweep = jax.vmap(solve_one)(jnp.linspace(0.0, 5.0, 500))
```

---

## 6. Provisional API for later versions (signatures only — internals WILL change)

Sketched so the surface stays coherent as features land. **Do not implement or over-specify now.**

- **v0.2 Periodic orbits & Floquet**
  ```python
  prob = jc.BifProblem(f, u0=..., p0=..., kind="periodic")
  sol  = jc.continuation(prob, jc.PseudoArclength(),
                         alg_kw={"mesh": jc.Collocation(n=20, degree=4)},
                         events=[jc.PeriodDoubling(), jc.Fold()])
  # Floquet multipliers land in Branch.eigenvalues for kind="periodic"
  ```
- **v0.3 Branch switching** (from a detected event):
  ```python
  new = jc.branch_switch(sol, event=sol.events[0], settings=...)
  ```
- **v0.3 Two-parameter continuation** (codim-2 curve of a codim-1 point):
  ```python
  sol2 = jc.continuation(jc.codim2(prob, event=jc.Fold()),
                         p_span=..., p2_span=..., events=[jc.Cusp(), jc.BogdanovTakens()])
  ```
- **Normal forms / Lyapunov coefficient**: `jc.normal_form(sol, event)` -> criticality.

---

## 7. Migration from the current code

| Current | Target |
|---------|--------|
| `ContinuationProblem` + `EquilibriumProblem`/`PeriodicOrbitProblem` | single `BifProblem` |
| `rhs(u, params_dict)` + string `continuation_param` | `f(u, p, args)` |
| `PseudoArclengthContinuation(...).run(prob, range)` | `continuation(prob, PseudoArclength(), p_span=range)` |
| `equilibrium_continuation(...)` free fn | folded into `continuation` |
| flags `detect_bifurcations=`, `compute_stability=` | `events=[...]`, `settings.compute_stability` |
| `BifurcationDetector` | `Event` protocol + `Fold()/Hopf()` |
| growing Python lists in `run()` | fixed-length `Branch` buffers + `n_valid` (§4.3) |
| FD `df/dp` | `jacfwd(f, argnums=1)` *(done)* |
| block-elimination corrector | full bordered solve in `lax.while_loop` *(done)* |

The already-completed corrector rewrite is API-agnostic and carries over unchanged; the new
`continuation()` will call the same `_correct_jit` internally.

## 8. Immediate reconciliation (do alongside v0.1.0)

- ✅ **`__init__.py` trimmed to the equilibrium spine** (2026-07-18). The stubs
  (`PeriodicOrbitProblem`, `BoundaryValueProblem`, `compute_floquet_multipliers`,
  `PeriodDoublingBifurcation`, `periodic_continuation`) are no longer exported at top level; they
  remain importable from their submodules (e.g. `jaxcont.problems.periodic`) until their version
  ships. Top-level `__all__` is now the 32-name equilibrium surface.
- Introduce `BifProblem` + `continuation()` as the blessed surface; keep the OO class as a
  deprecated thin wrapper for one release to avoid breaking the examples.

---

## 9. Ecosystem: lyapax interop (Lyapunov exponents)

**lyapax** (`~/git/lyapunov`) computes the **Lyapunov spectrum** of ODE/DDE/network systems in
JAX. It is a *sibling* package, not a JaxCont feature — different job, and its API is already the
same functional pattern we adopt here (`ode_problem(rhs, state0, dt, integrator=...)` +
`lyapunov_spectrum(problem, n_steps=...)` → `result.exponents`). The three JAX levers in §3 are
exposed identically by lyapax, which is what makes them one ecosystem.

**Terminology (do not conflate):**
- *Lyapunov coefficient* `l₁` = Hopf criticality (super/subcritical). A **bifurcation** invariant
  → JaxCont v0.3 "normal forms" (§6).
- *Lyapunov exponents* (spectrum) = trajectory divergence / chaos. A **dynamics** invariant →
  **lyapax**. For equilibria these are just `Re(eig(df/du))` (JaxCont has them); for limit cycles
  they are `log|μ|/T` from Floquet multipliers (JaxCont v0.2). lyapax's unique value is the
  chaotic/quasiperiodic regime *along a branch* (transition-to-chaos, Kaplan–Yorke dim vs param).

**Design decisions:**
1. Keep the packages **separate**; do not vendor lyapax into JaxCont.
2. **Align idioms** so they read as one ecosystem: functional front door, PyTree problems, `x64`,
   `vmap`. Mirror lyapax's lowercase `*_problem(...)` factory (`jc.bif_problem(...)` alongside the
   `BifProblem` type) and its `result.<attr>` convention (`sol.branch...`, `result.exponents`).
3. Provide **one thin bridge**: `BifProblem.as_rhs(p) -> Callable[[Array], Array]` returns a
   lyapax-compatible autonomous `rhs(state)` frozen at continuation-parameter value `p`. Workflow:
   continue with JaxCont → pick a branch point → compute its spectrum with lyapax.

   ```python
   prob = jc.bif_problem(f, u0, p0=0.5)
   sol  = jc.continuation(prob, p_span=(0.5, 3.0))
   p_star, u_star = sol.branch.at_param(2.4)
   lyap = lyapax.lyapunov_spectrum(
       lyapax.ode_problem(prob.as_rhs(p_star), state0=u_star, dt=1e-3, integrator="rk4"),
       n_steps=20_000,
   )
   ```
4. **Possible future shared core** (do NOT extract yet): jittable integrators and the
   `ModelSpec`/`build_jax_dfun` symbolic builder are needed by JaxCont v0.2 (periodic-orbit
   shooting/collocation) *and* already exist in lyapax. If duplication becomes real, factor a
   small common dependency then — premature now.

## 10. Delay differential equations (DDEs) — out of scope, seams reserved

### 10.1 What "a system with delay" means

A delay differential equation makes the rate of change depend on the state at an *earlier* time,
`x(t − τ)`, not just the present:

```
dx/dt = f( x(t), x(t − τ), p )          # single constant delay τ
```

Unlike an ODE, the "initial condition" is an entire **history function** `x(s)` for
`s ∈ [−τ, 0]`, so the true state is infinite-dimensional (a function segment, not a point). That
is the whole reason DDE bifurcation analysis is its own field and tool (DDE-BIFTOOL).

**Concrete example systems** (why anyone would want this):
- **Mackey–Glass** `dx/dt = β·x(t−τ)/(1 + x(t−τ)ⁿ) − γ·x(t)` — physiological control; a
  textbook route-to-chaos as τ increases.
- **Delayed logistic (Hutchinson)** `dx/dt = r·x(t)·(1 − x(t−τ)/K)` — population dynamics; Hopf
  bifurcation to oscillations at a critical delay.
- **Pyragas delayed feedback control** — stabilizing unstable periodic orbits via a `x(t)−x(t−τ)`
  feedback term.
- **Neural-mass / brain-network models with conduction delays** — directly relevant here: our
  `example_05_neural_mass.py` is delay-free today, but inter-regional axonal conduction delays are
  exactly where DDE continuation would matter for that use case.

### 10.2 Decision and reserved seams

**Decision:** DDE continuation is **out of scope for v0.1–v0.3.** Supporting the
infinite-dimensional history state is the largest scope-creep risk to "equilibria done well", and
DDE-BIFTOOL shows it is a package-sized effort on its own.

**But the architecture reserves three extension points, at zero present cost:**
1. **`BifProblem.kind`** — a future `"dde"` kind with signature `f(u, u_delayed, p, args)`
   (constant delays carried in `args`) fits the existing surface without breaking it. Crucially,
   DDE **equilibria** satisfy the *same* algebraic system as ODE equilibria (all delayed terms
   collapse to the steady state `x* = x(t−τ)`), so the bordered corrector applies **unchanged** —
   only stability differs.
2. **`EigenSolver` protocol (§4.6)** — DDE stability is governed by roots of a *transcendental*
   characteristic equation, e.g. `det(λI − A₀ − Σₖ Aₖ e^{−λτₖ}) = 0`, which has infinitely many
   roots. In practice one Chebyshev/pseudospectral-discretizes the infinitesimal generator and
   takes eigenvalues of a large matrix. That is *purely* an eigensolver swap
   (`ChebyshevDDE()` in place of `DenseEigen()`); the continuation loop is untouched. **This is the
   single strongest concrete justification for the pluggable-solver seam in §4.6.**
3. **lyapax already handles DDE trajectories** (ring-buffer history, `lyapunov_spectrum_dde`), so
   trajectory-level DDE analysis (chaos, spectra) has a home today — same division of labor as §9.

Revisit as "v0.4+ / separate package" once equilibrium + periodic continuation ship; DDE-BIFTOOL
and Knut are the prior art to study then.

## References
- **BifurcationKit.jl** — unified `BifurcationProblem`, pluggable solvers, event API, predictors,
  branch switching. Primary model for the spine.
- **diffrax** — the JAX-ecosystem template for functional, PyTree-state, jittable solver APIs.
- **MATCONT** — bifurcation taxonomy & naming (EP/LP/H/BP/CP/BT/...).
- **DDE-BIFTOOL**, **Knut** — DDE continuation prior art (§10).
- **AUTO-07p**, **PyDSTool** — prior art; constants/monolithic style, less relevant to API shape.
- Kuznetsov, *Elements of Applied Bifurcation Theory*.
