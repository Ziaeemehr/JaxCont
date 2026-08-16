# Two-parameter continuation — design spec

**Date:** 2026-08-17
**Status:** approved, ready for implementation planning
**Roadmap item:** v0.3.0+ "Two-parameter continuation" (`notes/ROADMAP.md`)

## 1. Problem

JaxCont ships five direct codim-2 point solvers (`cusp_point`,
`bogdanov_takens_point`, `generalized_hopf_point`, `zero_hopf_point`,
`double_hopf_point`, v0.3.0). Every one of them is a *refinement* tool: it needs a
guess already near the codim-2 point and none of them search.
`bifurcations/codim2.py`'s own module docstring says so explicitly — "Finding
codim-2 points you cannot already approximate requires two-parameter
continuation, which is a separate (unstarted) roadmap item."

This spec is that item. It adds continuation of **codim-1 curves in two
parameters** (a fold curve and a Hopf curve in the `(p1, p2)` plane) and
**codim-2 event detection along those curves**, which is what turns the shipped
point solvers into something reachable without a hand-supplied guess.

Scope was chosen deliberately over two narrower alternatives (fold curve only;
curves without codim-2 detection). Detection is what closes the roadmap's
explicitly-deferred "`Event`/`events=[...]` integration" item, so it is in scope.

## 2. Key architectural insight: no new engine

A fold of `f(u, p) = 0` is the root of the extended system already written in
`bifurcations/fold_solve.py`:

```
G1: f(u, p)        = 0     (n eqs)   equilibrium
G2: f_u(u, p) · v  = 0     (n eqs)   singular Jacobian, null vector v
G3: vᵀv - 1        = 0     (1 eq)    normalization
```

A fold *curve* is that same system continued in a second parameter. With
`X = (u, p1, v)` and continuation parameter `q = p2`, `F(X, q) = 0` is an
ordinary residual in a scalar parameter — exactly what
`core/scan_continuation.py:pseudo_arclength_scan` already solves generically.

This is the same reduction that made periodic orbits work in v0.2.0 (a
collocation system is just a large `F(U, p) = 0`). The consequence is that
**two-parameter continuation needs no new engine, no new solver, and no change to
`continuation()`** — only a pure factory returning an ordinary `BifProblem`,
mirroring `periodic_orbit_problem`.

### 2.1 The reduction trick (no new residuals are written)

Define, at a fixed continuation value `q`:

```python
f_reduced(u, p1, args) = f(u, assemble(p1, q), args)
```

From `f_reduced`'s point of view `p1` is an ordinary scalar parameter.
Therefore `fold_solve._extended_residual` and
`hopf_normal_form._extended_residual` apply **completely unchanged**. A fold
curve is literally "the fold extended system of the reduced one-parameter
problem, continued in `q`". Both curve residuals are a handful of lines each.

Both curves inherit `jit`, `vmap` and `grad` from the existing engine at no
additional cost.

### 2.2 Verified prerequisite

The Hopf extended system's phase condition calls
`hopf_normal_form._seed`, which uses `jnp.linalg.eig`. Under this design that
call lands **inside the scan engine's jitted Newton loop**. JAX has
historically not supported general (non-symmetric) `eig` on GPU, which would
have sunk the Hopf curve entirely.

Checked rather than assumed, on this project's env (JAX 0.11.0, `CudaDevice`):

| case | result |
|---|---|
| eager `jnp.linalg.eig` | OK |
| `eig` under `jax.jit` | OK |
| `eig` inside `lax.while_loop` under `jit` | OK (value verified correct) |
| `vmap` over that whole construction | OK (value verified correct) |

The approach is viable end to end. Performance of `eig`-in-loop is a separate
question and is a measurement task for the plan, not an assumption here; if it
proves too slow, the documented fallback is hoisting the seed to a constant
computed once at factory time (at the cost of the seed no longer tracking along
the curve).

## 3. Public API

New module: `src/jaxcont/bifurcations/curves.py`. Two pure factories:

```python
fold_curve_problem(f, u_guess, p_guess, *, free=1, args=None,
                   tol=1e-6, max_iter=50) -> BifProblem
hopf_curve_problem(f, u_guess, p_guess, *, free=1, args=None,
                   tol=1e-6, max_iter=50) -> BifProblem
```

(`tol`/`max_iter` govern the initial seed refinement of §3.2 only, and match
`fold_point`/`hopf_point`'s own defaults. They are unrelated to the
continuation-time `ContinuationPar.newton_tol` calibrated in §5.)

Usage:

```python
prob2 = jc.fold_curve_problem(f, u_guess, p_guess=jnp.array([0.3, 1.0]), free=1, args=args)
sol2  = jc.continuation(prob2, p_span=(1.0, 4.0),
                        events=[jc.Cusp(raw_f=f, free=1),
                                jc.BogdanovTakens(raw_f=f, free=1, curve="fold")])
```

- `p_guess` has shape `(2,)`, matching the convention `codim2.py` already
  established for codim-2 work. No other function's `p` handling changes.
- `free` indexes which component of `p` is the continuation parameter; the other
  becomes part of the solved state `X`. With `free=1`, `p[1]` is continued and
  `p[0]` is solved for.
- `p_span[0]` must equal `p_guess[free]`. This is not redundant: `continuation()`
  treats `p_span[0]` as the *literal* starting parameter value rather than
  reading it off the problem — a pre-existing `api.py` design point documented
  during the engine consolidation, and a known footgun. The factories validate
  this and raise on mismatch rather than silently continuing from a point that
  is not on the refined curve.
- Both factories are exported top level (`jc.fold_curve_problem`,
  `jc.hopf_curve_problem`), as are all five events of §4 (`jc.Cusp`,
  `jc.BogdanovTakens`, `jc.ZeroHopf`, `jc.GeneralizedHopf`, `jc.DoubleHopf`),
  matching how `Fold`/`Hopf`/`PeriodDoubling`/`NeimarkSacker` are already
  surfaced.

This supersedes the `jc.codim2(prob, event=jc.Fold())` sketch in
ARCHITECTURE.md §6, which predates both the codim-2 solvers and the
`periodic_orbit_problem` factory pattern. ARCHITECTURE.md §6 is updated to the
shipped API as part of this work.

### 3.1 Packing and dimensions

| curve | packed state `X` | dim | equations |
|---|---|---|---|
| fold | `(u, p1, v)` | `2n+1` | `2n+1` |
| Hopf | `(u, p1, q1, q2, ω)` | `3n+2` | `3n+2` |

Both square. Both reuse the existing `_pack`/`_unpack` helpers in
`fold_solve.py` / `hopf_normal_form.py`.

### 3.2 Seeding

Each factory refines the caller's guess to a genuine point on the curve before
returning, using `fold_point` / `hopf_point` (both already exist, both already
built on `solvers/implicit.py:differentiable_root`). This mirrors
`periodic_orbit_problem` refining a raw trajectory guess before handing back a
`BifProblem`: a bad guess fails loudly at construction rather than producing a
plausible-looking wrong curve.

## 4. Codim-2 events

New module: `src/jaxcont/bifurcations/codim2_events.py`, importing the `Event`
protocol from `events.py`. Kept separate deliberately: `events.py` is already 391
lines and covers codim-1 events along ordinary branches; codim-2 events are only
meaningful along curves.

| Event | Curve | Test function | `refine()` calls |
|---|---|---|---|
| `Cusp` | fold | `fold_coefficient` crosses 0 | `cusp_point` |
| `BogdanovTakens` | fold / Hopf | 2nd eigenvalue → 0 / `ω` → 0 | `bogdanov_takens_point` |
| `ZeroHopf` | fold / Hopf | complex pair's `Re` crosses 0 / real eigenvalue crosses 0 | `zero_hopf_point` |
| `GeneralizedHopf` | Hopf | `lyapunov_coefficient` crosses 0 | `generalized_hopf_point` |
| `DoubleHopf` | Hopf | 2nd complex pair's `Re` crosses 0 | `double_hopf_point` |

`Cusp` and `GeneralizedHopf` need no eigenvalues at all: `fold_coefficient` and
`lyapunov_coefficient` already ship and are already scalars that change sign.

`BogdanovTakens` and `ZeroHopf` occur on both curves with genuinely different
mathematical conditions, so they take an explicit `curve="fold"|"hopf"` field at
construction. This is not inferred: an `Event` only ever receives a
`BranchPoint`, so there is nothing to infer it from.

### 4.1 Each event carries its own `raw_f`

`detect_events`'s generic `rhs` parameter is the **extended-system residual**
here, not the original ODE. Reusing it would reproduce, in reverse, the
equilibrium-only footgun that `Hopf` originally had. Every codim-2 event
therefore carries its own `raw_f` and `free` index and recomputes the original
system's Jacobian itself — the precedent `PeriodDoubling` / `NeimarkSacker`
already set for exactly this reason.

This is why `_run_scan`'s `problem.kind` dispatch needs no third branch: nothing
about curve problems flows through `Branch.eigenvalues`.

### 4.2 Excluding the always-critical eigenvalue (the correctness core)

On a fold curve **one eigenvalue is pinned at zero at every point** — that is
the curve's defining condition. On a Hopf curve the pair `±iω` is pinned to the
imaginary axis at every point. So a naive "BT = an eigenvalue reaches zero" test
is satisfied at literally every point of a fold curve and detects nothing.

Every test function therefore:

1. Excludes the pinned eigenvalue(s) — `argmin` distance to the critical set,
   the same rule `stability.floquet.floquet_stable` uses to exclude the trivial
   Floquet multiplier `1`.
2. Applies a "was this candidate ever near critical?" pre-filter, ported from
   `PeriodDoubling`/`NeimarkSacker`'s `near_unit_circle`, **including its v0.3.1
   fix to a log-magnitude window** (the original linear form produced false
   negatives). Without the pre-filter, `argmin` can silently latch onto an
   unrelated, always-far eigenvalue and fire a false positive.

This is the single most likely source of a silent false positive in the whole
feature. It has bitten this codebase twice already; both the trap and its fixed
form are known going in.

### 4.3 `refine()` guards

Every `codim2.py` solver returns `converged` as a JAX boolean. Each `refine()`
checks `converged` **and** `isfinite` before emitting a hit, mirroring the guard
`Hopf.refine()` grew after a non-convergent solve silently mislabeled itself
`"subcritical"` (`nan < 0` is `False` in Python/NumPy/JAX). A failed refine
reports a non-hit; it never emits a plausible-looking wrong point.

### 4.4 Double-Hopf payoff

`double_hopf_point` requires a caller-supplied `seed_b` (keyword-only, no
default) because it cannot guess the second Hopf pair and degenerates to `nan`
if both blocks seed onto the same physical pair. Curve detection produces that
second pair naturally, so `DoubleHopf.refine()` supplies `seed_b` automatically
— making HH reachable for the first time without hand-construction.

## 5. Guards and numerical calibration

**`compute_stability=True` on a curve problem raises**, with a message pointing
at the codim-2 events. Eigendecomposing the extended Jacobian is not a
meaningful spectrum; per §4.1 the events compute the original system's spectrum
themselves. This is the same guard clause periodic-orbit problems originally
carried, for the same reason.

**`newton_tol` is measured per curve type, not inherited.** The extended systems
are larger (`2n+1`, `3n+2`), so their achievable float32 residual floor will be
looser than the equilibrium case. This codebase has been bitten three separate
times by assuming a transferable tolerance (periodic orbits needed `1e-5`,
Brusselator `1e-4`, codim-2 solvers `1e-6`; ROADMAP issue #12 is the general
pattern: a `tol` below the float32 floor reports `converged=False` forever, with
no error raised). The plan measures the floor for each curve type before picking
a default.

**GPU matmul precision is checked, not assumed in either direction.** The TF32
issue that corrupted the periodic-orbit collocation Jacobian came from large
`einsum` contractions; curve residuals are `jacfwd` plus matvecs, and the
Floquet recursion's small solves needed no such fix. Verify; do not assume.

## 6. Verification

Three tiers. Tiers 1 and 2 are this project's established pattern; tier 3 exists
because a detector that never fires trivially passes any "no false positives"
test.

### 6.1 Closed-form exact

Test systems are **affinely shifted**, per the codim-2 lesson that
origin-centered normal forms have no discriminating power (every standard normal
form places its codim-2 point at `u=0, p=(0,0)`, so a stub returning zeros would
pass them all).

The cusp normal form's fold curve is given analytically by its discriminant, so
the traced curve is comparable against an exact algebraic answer, not merely a
plausible shape.

### 6.2 Independent tools — two of them, on two different models

**MatCont 7.6 (catalytic oscillator).** The validation suite in
`examples/MatCont/` already scaffolds both cases:

| case | title | current state |
|---|---|---|
| `US-C2-LP-001` | Two-parameter fold-curve continuation | `support: unsupported`, `python: None` |
| `US-C2-H-001` | Two-parameter Hopf-curve continuation | `support: unsupported`, `python: None` |

Both already have working MATLAB drivers in `matlab/unsupported/`
(`run_two_parameter_fold_curve.m`, `run_two_parameter_hopf_curve.m`), marked
`'unsupported_execution': 'executable'` — the MATLAB half already runs. They
derive from MatCont 7.6's own `Testruns/testLPcataloscill.m` and
`testLPHopfcataloscill.m`, so the model is MatCont's own test case, not one
chosen to flatter the answer. The toolchain was verified present on this machine
(`/home/ziaee/prog/Matlab/R2020a/bin/matlab`, `/home/ziaee/prog/MatCont/MatCont7p6/`),
so references can genuinely be regenerated rather than hand-transcribed.

Work required to promote both cases to `supported`:

1. Move the two drivers out of `matlab/unsupported/` and wire into
   `run_supported.m`.
2. Add the missing export step — they currently only `assert` and print. They
   need an `export_*_run.m` analogue of `export_equilibrium_run.m` emitting
   `_branch.csv` / `_events.csv` / `_metadata.json`.
3. Write the JaxCont half as `examples/MatCont/python_cases/curves.py`.
4. Flip `support` to `supported` in `cases.json` and fill the `python:` field.

**BifurcationKit.jl v0.5.2 (Lorenz-84).** `examples/BifurcationKit/05_codim2.jl`
already traces this exact fold curve and Newton-refines a BT point off it; those
refined values are already asserted in `tests/test_codim2.py`. Extend the script
to dump curve samples, compare point-by-point, and assert our BT *detection*
lands on the value the direct solver already matches.

Two independent tools on two different models is stronger than either alone.

### 6.3 Discriminating-power checks

Following the codim-2 work's deliberate-bug-injection practice:

- One test that **fails** if the trivial-eigenvalue exclusion (§4.2) is removed.
- One test that **fails** if the pre-filter is too aggressive (i.e. suppresses a
  genuine detection).

Without both, a do-nothing detector passes the suite.

## 7. Deliverables

- `src/jaxcont/bifurcations/curves.py` — the two factories.
- `src/jaxcont/bifurcations/codim2_events.py` — the five events.
- `examples/example_12_two_parameter_diagram.py` — fold and Hopf curves in the
  `(p1, p2)` plane with codim-2 points marked, on Lorenz-84 against the Julia
  reference. Because curves are ordinary `BifProblem`s, `vmap` over `args` and
  `jax.grad` of a codim-2 location come free; one example section demonstrates
  both. This satisfies the roadmap's "every new curve/event type ships batched
  *and* differentiable" mandate without padding the gallery with a second
  script.
- `viz/`: `plot_two_parameter_diagram`, reusing the existing
  `BIFURCATION_STYLES` table (`taxonomy.py` already carries CP/BT/GH/ZH/HH
  entries).
- `examples/MatCont/` promotions per §6.2.
- Docs: `notes/ROADMAP.md` updated; ARCHITECTURE.md §6 sketch replaced with the
  shipped API.

## 8. Explicitly out of scope

Listed so each is visibly a decision, not an oversight:

- **Adaptive re-anchoring of the Hopf phase seed.** The phase condition anchors
  to a seed eigenvector recomputed from `(u_guess, p_guess)` at each `q`, so it
  tracks gently along the curve. If the eigenvector rotates far enough to become
  orthogonal to the seed, the phase row degenerates and the curve stalls.
  Fixed-shape buffers (the discipline the whole `jit`/`vmap` story depends on)
  rule out MatCont-style adaptive re-anchoring. v1 documents this; the remedy is
  restarting from a later point.
- **Bialternate-product / determinant test functions** (MatCont's approach).
  More robust for "second eigenvalue crosses" conditions without eigenvector
  bookkeeping, but bialternate products are `n(n-1)/2`-dimensional and would need
  their own `LinearSolver` story. Disproportionate cost here.
- **Two-parameter curves of periodic bifurcations** — `US-C2-PD-001`,
  `US-C2-LPC-001`, `US-C2-NS-001` stay `unsupported`. These need the collocation
  system nested inside the extended system and are their own epic.
- **Codim-2 normal-form coefficients beyond the defining conditions** — BT's
  `(a, b)` pair, GH's second Lyapunov coefficient `l2`. Still their own roadmap
  items.
- **Branch switching.** Separate roadmap item, unaffected by this work.

## 9. References

- `notes/ROADMAP.md` — v0.3.0+ "Two-parameter continuation"
- `notes/ARCHITECTURE.md` §4.7 (`Event` protocol), §6 (provisional API)
- `docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md`
- `docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md`
- `docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md`
  (the factory-returning-`BifProblem` precedent)
- Kuznetsov, *Elements of Applied Bifurcation Theory*, Ch. 8 (codim-2)
- MatCont 7.6 manual §§ on `init_LP_LP` / `limitpoint` curve continuation
