# Hopf normal form / first Lyapunov coefficient — design spec

**Date:** 2026-08-04
**Status:** approved, ready for implementation planning
**Roadmap item:** v0.3.0+ "Normal forms / Lyapunov coefficient l1" (`notes/ROADMAP.md`); prerequisite
for the `GH` (Generalized Hopf/Bautin) taxonomy entry already reserved in
`bifurcations/taxonomy.py`. API target sketched in `notes/ARCHITECTURE.md` §6/§9.

## Motivation

JaxCont detects Hopf points (`jc.Hopf()`) but cannot classify them: whether the branching limit
cycle is stable (supercritical, `l1<0`) or unstable (subcritical, `l1>0`) is exactly the piece of
information `notes/ARCHITECTURE.md` §9 calls out as the "bifurcation invariant" version of a
Lyapunov coefficient (not to be confused with lyapax's Lyapunov *exponent* spectrum — a different,
sibling-package concept). No code for this exists anywhere in the repo today; the only prior
placeholder (`FoldBifurcation.compute_normal_form`/`HopfBifurcation.compute_first_lyapunov_
coefficient`) was deleted outright in the `65347cd` event-protocol rewrite and never rebuilt.

This also fixes a latent weakness in `Hopf.refine()`: today it locates a Hopf point by bisection
and returns a **linearly-interpolated** `u` between two branch points, which is not itself a
converged solution of `f(u,p)=0`. The new extended-system solve produces a genuine equilibrium.

## Scope

1. `hopf_point(f, u_guess, p_guess, args=None, *, tol, max_iter) -> (u*, p*, q1*, q2*, omega0*)`
   — differentiable (implicit-function-theorem) Hopf-point solver, mirroring
   `bifurcations/fold_solve.py`'s `fold_point`.
2. `hopf_parameter(f, u_guess, p_guess, args=None, *, tol, max_iter) -> p*` — scalar, grad-ready,
   mirrors `fold_parameter`.
3. `lyapunov_coefficient(f, u, p, q1, q2, omega0, args=None) -> l1` — Kuznetsov's first Lyapunov
   coefficient formula, pure algebra (no Newton solve), differentiable in its inputs.
4. `Hopf.refine()` rewired onto (1)+(3), replacing bisection; `EventHit.info` gains
   `omega0`/`l1`/`criticality`.
5. Top-level exports of (1)-(3) from `jaxcont/__init__.py`, alongside `fold_point`/`fold_parameter`.

Out of scope (explicitly deferred, not forgotten): the fold's own normal-form coefficient `a`,
`jc.normal_form(sol, event)` dispatcher, GH/codim-2 detection, branch switching, two-parameter
continuation. Each is a separate future roadmap item.

## Design

### Why two functions, not one

`hopf_point` is the only piece needing Newton iteration / `custom_vjp` (via the existing
`solvers/implicit.py:differentiable_root`, extracted specifically so future extended-system events
could reuse it). `lyapunov_coefficient` is a closed-form algebraic formula — it needs no
eigendecomposition and no iteration, so it stays a plain differentiable JAX function. Composing
them (`l1 = lyapunov_coefficient(f, *hopf_point(f, ...), args)`) gives correct end-to-end gradients
through ordinary chain-rule composition, without `lyapunov_coefficient` ever calling
`jnp.linalg.eig` — which has no reliable gradient rule for complex eigenvectors under phase
ambiguity. `hopf_point`, like `fold_point`'s SVD seed, only uses `eig` as an **undifferentiated
Newton seed** (`x0` as a `theta -> Array` callable, evaluated inside `differentiable_root`'s traced
primal, per that function's existing documented contract), never in the differentiated result path.

### `hopf_point`: extended system

Standard formulation (Kuznetsov §10.2 / Govaerts; same shape MatCont's `hopf` curve uses).
Unknowns `x = (u, p, q1, q2, ω) ∈ ℝ^(3n+2)`, packed/unpacked like `fold_solve.py`'s `_pack`/
`_unpack`:

```
G1: f(u, p)              = 0        (n)   equilibrium
G2: J(u,p)·q1 + ω·q2      = 0        (n)   Re[(J - iω)(q1+iq2)] = 0
G3: J(u,p)·q2 - ω·q1      = 0        (n)   Im[(J - iω)(q1+iq2)] = 0
G4: q1ᵀq1 + q2ᵀq2 - 1     = 0        (1)   unit norm
G5: q1ᵀq2                 = 0        (1)   phase normalization
```

`(q1+iq2)` is the right eigenvector for eigenvalue `iω`. Seed: `J = jacfwd(f, argnums=0)(u_guess,
p_guess, args)`, eigendecompose with `jnp.linalg.eig` (undifferentiated, seed-only), pick the
complex pair with smallest `|Re|` — the same selection rule `Hopf.test_function` already uses —
set `q1, q2 = Re(vec), Im(vec)` (normalized), `ω = |Im(eigval)|`.

### `lyapunov_coefficient`: Kuznetsov's formula

Needs a **left** eigenvector `p ∈ ℂⁿ` with `Aᵀp = -iωp`, normalized so `p̄ᵀq = 1` (a different
vector from `q̄` in general — equal only if `A` is normal). Solved as a plain null-space problem:
`jnp.linalg.svd` on the real 2n×2n block form of `(Aᵀ + iωI)`, smallest singular vector, then
rescaled by division so `p̄ᵀq=1` holds exactly — this division removes the SVD's arbitrary
phase/sign ambiguity, so the result is a well-defined differentiable function of `(A, q1, q2, ω)`;
no `custom_vjp` needed for this step.

`B(x,y)`/`C(x,y,z)` (bilinear/trilinear parts of `f`'s 2nd/3rd derivatives) are computed as
directional derivatives via nested `jax.jvp` — never forming the full n²/n³ tensor:

```python
B = lambda x, y: jvp(lambda u: jvp(lambda u: f(u, p, args), (u,), (y,))[1], (u,), (x,))[1]
```

```
l1 = (1/(2ω)) · Re[ ⟨p, C(q,q,q̄)⟩ − 2⟨p, B(q, A⁻¹B(q,q̄))⟩ + ⟨p, B(q̄, (2iωI−A)⁻¹B(q,q))⟩ ]
```
where `⟨p,x⟩ = p̄ᵀx`, `q = q1+iq2`. Sign/conjugate conventions will be nailed down empirically
against the closed-form ground truth below (§Verification item 1) during implementation — this is
this project's established practice for every prior formula-heavy feature (Floquet, PD/NS all
required a "found and fixed via end-to-end verification" pass; the formula on paper is not trusted
until it reproduces a known-exact answer).

### `Hopf.refine()` rewrite

```python
def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
    u_guess, p_guess = (left.u + right.u) / 2, (left.p + right.p) / 2
    u, p, q1, q2, omega0 = hopf_point(
        lambda u, p, _args: rhs(u, p), u_guess, p_guess,
        tol=tolerance, max_iter=max_iterations,
    )
    l1 = lyapunov_coefficient(lambda u, p, _args: rhs(u, p), u, p, q1, q2, omega0)
    if abs(l1) < self.l1_tolerance:
        criticality = "degenerate"
    else:
        criticality = "supercritical" if l1 < 0 else "subcritical"
    return EventHit(
        kind="hopf", p=float(p), u=u, index=index,
        info={"omega0": float(omega0), "l1": float(l1),
              "criticality": criticality, "method": "extended_system"},
    )
```
(mirrors `Fold.refine()`'s existing `lambda u, p, _args: rhs(u, p)` adapter — `detect_events`'s
generic `rhs` is 2-arg, `hopf_point` wants the 3-arg `f(u,p,args)` shape.)

### Edge cases

- **Newton failure to converge** (coarse bracket, degenerate case): `differentiable_root`'s
  `lax.while_loop` runs to `max_iter` and returns whatever `x` it has — identical failure mode to
  `fold_point`'s existing behavior, not a regression versus bisection (which also just runs to
  `max_iterations` and returns its last bracket).
- **`l1 ≈ 0`** (Bautin/GH point): `Hopf` gains an `l1_tolerance: float = 1e-6` field (mirroring its
  existing `tolerance` field); `|l1| < l1_tolerance` is labeled `"degenerate"` rather than an exact
  `l1 == 0` comparison, which floating-point `l1` would essentially never hit. Honest about the
  boundary case, and a breadcrumb toward the `GH` taxonomy entry already reserved for v0.3.
- **Regression risk**: `example_02_lorenz.py`/`example_05_neural_mass.py`'s detected Hopf
  locations may shift slightly now that `u` is a converged equilibrium rather than an
  interpolation. Both scripts must be re-run after the change and their existing
  `BifurcationKit.jl` comparison tables re-verified to still match — required, not optional, since
  those numbers are cited elsewhere in the roadmap.

## Testing / verification plan

New `tests/test_hopf_normal_form.py`:

1. **Closed-form ground truth**: `ẋ=−y+x(μ−x²−y²), ẏ=x+y(μ−x²−y²)` — exact Hopf at `(u,p)=(0,0)`,
   `ω0=1`, `l1=−1` (standard supercritical textbook example, Kuznetsov §3.2/§3.4). Assert
   `hopf_point` recovers `(0,0,ω0=1)` and `lyapunov_coefficient` recovers `l1=−1` to
   float32-achievable precision.
2. **Gradient check**: `jax.grad` of `hopf_parameter` and of the composed
   `lyapunov_coefficient(f, *hopf_point(...))` w.r.t. an `args`-supplied parameter (e.g. perturb
   the cubic coefficient), cross-checked against finite differences — same pattern as
   `example_07_differentiable.py`'s fold gradient check.
3. **Independent cross-check**: run BifurcationKit.jl v0.5.2 (confirmed installed in this dev
   environment — `julia -e 'using Pkg; Pkg.status()'` shows it) on the same closed-form system, get
   its own normal-form `l1`, hardcode the reference value into the test with a comment on how it
   was produced — matching `example_02`/`05`'s existing "offline reference value" pattern, kept
   self-contained in this new test rather than touching those files' own tables.
4. **Regression check**: re-run `example_02_lorenz.py` and `example_05_neural_mass.py` end-to-end
   after `Hopf.refine()` is rewired; confirm their existing `bk_reference` comparison tables still
   match.
5. **Degenerate case**: a synthetic system with `l1` designed near zero; assert
   `criticality == "degenerate"` rather than an arbitrary sign.

## Out of scope / explicit non-goals

- Fold's own normal-form coefficient `a` (a separate, smaller piece of math — not bundled here).
- `jc.normal_form(sol, event)` dispatcher from `ARCHITECTURE.md` §6 — superseded by the simpler
  "automatic in `EventHit.info` + standalone grad-ready `lyapunov_coefficient`" delivery chosen
  here; revisit only if a unified dispatcher proves genuinely needed later.
- GH/codim-2 detection as an `Event` — this spec only unlocks the prerequisite (`l1`).
- Making `Hopf`/events.py trace-safe (`jax.vmap`/`jax.jit`) — out of scope, same as today; this
  module stays eager-only, consistent with `events.py`'s existing documented contract.
