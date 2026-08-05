# Codim-2 direct point solvers — design spec

**Date:** 2026-08-05
**Status:** approved, ready for implementation planning
**Roadmap item:** v0.3.0+ "Codim-2 bifurcations (cusp, Bogdanov-Takens, ...)"
(`notes/ROADMAP.md`). Implements the `CP`/`BT`/`ZH`/`HH`/`GH` labels reserved in
`bifurcations/taxonomy.py` since v0.1. Direct successor to the Hopf normal-form work
(`2026-08-04-hopf-normal-form-design.md`), whose `lyapunov_coefficient` is a prerequisite for `GH`.

## Motivation

JaxCont locates and classifies codim-1 points (`fold_point`, `hopf_point`, `lyapunov_coefficient`)
but has no codim-2 capability at all. Codim-2 points are the *organizing centres* of a bifurcation
diagram: they are where codim-1 curves meet, terminate, or change criticality, and knowing where
they sit explains the global structure that scanning one parameter at a time cannot. All five
labels have been reserved in `taxonomy.py` with status `"v0.3"` since v0.1; no code has ever
backed them.

This also delivers something no other continuation tool offers: because the solve goes through
`solvers/implicit.py:differentiable_root`, a codim-2 *location* becomes differentiable with
respect to `args`. `jax.grad` of "where is the Bogdanov-Takens point" with respect to a design
vector is exactly the gradient-based bifurcation design that `notes/ARCHITECTURE.md` §3.2 names as
the reason this project exists.

## Scope

Five direct codim-2 point solvers in a new `src/jaxcont/bifurcations/codim2.py`, plus the fold's
quadratic normal-form coefficient in a new `src/jaxcont/bifurcations/fold_normal_form.py`:

| Function | Label | Defining condition beyond the codim-1 system |
|---|---|---|
| `cusp_point` | `CP` | fold with quadratic coefficient `a = 0` |
| `bogdanov_takens_point` | `BT` | double zero eigenvalue (Jordan block) |
| `generalized_hopf_point` | `GH` | Hopf with `l₁ = 0` |
| `zero_hopf_point` | `ZH` | simultaneous zero eigenvalue and imaginary pair |
| `double_hopf_point` | `HH` | two distinct imaginary pairs |
| `fold_coefficient` | — | `a = ½·wᵀB(v,v)`, the fold's own normal-form coefficient (needed by `CP`) |

Each point solver has a `*_parameters` companion returning just `p*` (shape `(2,)`), mirroring
`fold_parameter`/`hopf_parameter`, for the grad-ready case.

Exact signatures, so there is no ambiguity about return arity:

```python
cusp_point(f, u_guess, p_guess, args=None, *, tol, max_iter)
    -> (u, p, v, converged)
bogdanov_takens_point(f, u_guess, p_guess, args=None, *, tol, max_iter)
    -> (u, p, v0, v1, converged)
generalized_hopf_point(f, u_guess, p_guess, args=None, *, tol, max_iter)
    -> (u, p, q1, q2, omega, converged)
zero_hopf_point(f, u_guess, p_guess, args=None, *, tol, max_iter)
    -> (u, p, v, q1, q2, omega, converged)
double_hopf_point(f, u_guess, p_guess, args=None, *, tol, max_iter,
                  separation_tolerance=1e-3)
    -> (u, p, q1a, q2a, omega_a, q1b, q2b, omega_b, converged)

# Companions: p* only (shape (2,)), NO converged flag -- a bare array keeps
# jax.grad(...) usable directly, which is the whole point of these. Callers
# needing the flag use the full solver above.
cusp_parameters(...)              -> p
bogdanov_takens_parameters(...)   -> p
generalized_hopf_parameters(...)  -> p
zero_hopf_parameters(...)         -> p
double_hopf_parameters(...)       -> p

# Pure algebra, no Newton solve, so no convergence flag.
fold_coefficient(f, u, p, v, args=None) -> a
```

**Explicitly out of scope**, deferred rather than forgotten:

- **Two-parameter continuation.** `notes/ARCHITECTURE.md` §6 sketches codim-2 as events along a
  continued codim-1 curve (`jc.codim2(prob, event=jc.Fold())` + `p2_span`). That remains its own
  separate, unstarted roadmap item. This spec deliberately takes the direct-solve route instead:
  it needs no new engine, reuses the proven `differentiable_root` pattern, and inherits the
  `grad`/`vmap` story for free. The two approaches are complementary, not competing — direct
  solves refine a point you can already guess at; curve continuation finds points you cannot.
- **New `Event` subclasses.** A codim-2 point cannot be detected along a single-parameter branch
  (it requires a codim-1 curve in two parameters), so an `Event` would have nothing to fire
  against today and would ship as untested dead surface. `taxonomy.py`'s status strings are
  updated; `events=[...]` is untouched.
- **Codim-2 normal-form coefficients** (the BT `(a,b)` pair, the GH second Lyapunov coefficient
  `l₂`, cusp `c`). This spec locates codim-2 points; it does not classify their sub-cases.
- **Branch switching** from a codim-2 point, and all `"out of scope"`-status cycle codim-2 labels
  in `taxonomy.py` (`R1`–`R4`, `CH`, `LPNS`, `PDNS`, `LPPD`, `NSNS`, `GPD`, `CPC`, `BPC`).

## Design

### The second parameter: `p` becomes shape `(2,)`

Codim-2 points need two free parameters. Rather than introduce a second RHS calling convention,
these solvers take the *same* `f(u, p, args)` signature with `p` shaped `(2,)` instead of scalar:

```python
def f(u, p, args):
    a, b = p          # the two free parameters
    ...
```

This is a generalization of the existing convention, not a competitor: `jacfwd(f, argnums=0)`,
the `args` PyTree, and every derivative helper work unchanged. Existing scalar-`p` code is
untouched — nothing in `api.py`, `scan_continuation.py`, or the codim-1 solvers changes.

### Shared harness

All five solvers are the same shape: build an extended residual `G(x, θ) = 0` that is square by
construction, seed it, hand it to `differentiable_root`, unpack. A private
`_solve_codim2(residual_fn, seed_fn, args, *, tol, max_iter)` in `codim2.py` owns that flow so
each public solver is only its residual, its seed, and its unpacking.

Differentiability comes entirely from `differentiable_root`'s existing `custom_vjp` implicit-
function-theorem wrapper — no new gradient code. Per that primitive's documented contract, any
`θ`-dependent seed **must** be passed as a callable `theta -> Array` and resolved inside the
traced primal, never precomputed and closed over (which leaks a tracer under `jax.grad`). All
five seeds are θ-dependent (they call `jnp.linalg.eig`/`svd` on a `θ`-dependent Jacobian), so all
five use the callable form under `lax.stop_gradient`, exactly as `hopf_normal_form.py` does.

### The five extended systems

Write `J = f_u(u, p, args)`, `u ∈ Rⁿ`, `p ∈ R²`. Each system is square; the counts below were
verified numerically (see "Verified during planning").

**`CP` — cusp.** Unknowns `x = (u, p, v)`, dimension `2n+2`.
```
f(u, p)              (n)
J v                  (n)
vᵀv − 1              (1)
a(u, p, v)           (1)     <- the cusp condition
```

**`BT` — Bogdanov-Takens.** Unknowns `x = (u, p, v₀, v₁)`, dimension `3n+2`. Uses the Jordan-chain
formulation (`v₁` is the generalized eigenvector), *not* the left/right-null-vector form
`f=0, Jv=0, Jᵀw=0, ‖v‖=1, ‖w‖=1, wᵀv=0`, which is overdetermined by one equation:
```
f(u, p)              (n)
J v₀                 (n)
J v₁ − v₀            (n)
v₀ᵀv₀ − 1            (1)
v₀ᵀv₁                (1)
```

**`GH` — generalized Hopf (Bautin).** Unknowns `x = (u, p, q₁, q₂, ω)`, dimension `3n+3`. This is
`hopf_normal_form.py`'s existing Hopf system with one row appended:
```
f(u, p)                          (n)
J q₁ + ω q₂                      (n)
J q₂ − ω q₁                      (n)
q₁ᵀq₁ + q₂ᵀq₂ − 1                (1)
q₁ˢᵉᵉᵈ·q₂ − q₂ˢᵉᵉᵈ·q₁            (1)     <- phase condition
l₁(u, p, q₁, q₂, ω)              (1)     <- the GH condition
```
The phase condition **must** be the seed-based form above, matching
`hopf_normal_form.py:_extended_residual`'s `g5`. The naive `q₁·q₂ = 0` alternative is recorded as
broken in the Hopf design spec and must not be reintroduced here.

**`ZH` — zero-Hopf.** Unknowns `x = (u, p, v, q₁, q₂, ω)`, dimension `4n+3`: the fold rows
(`J v`, `vᵀv − 1`) and the Hopf rows together on one `f(u,p) = 0`.

**`HH` — double Hopf.** Unknowns `x = (u, p, q₁ᴬ, q₂ᴬ, ωᴬ, q₁ᴮ, q₂ᴮ, ωᴮ)`, dimension `5n+4`: one
`f(u,p) = 0` plus two independent Hopf blocks, each with its own normalization and its own
distinct phase seed.

### The fold coefficient `a`

`fold_normal_form.py:fold_coefficient(f, u, p, v, args) -> a` implements Kuznetsov's quadratic
coefficient `a = ½·⟨w, B(v,v)⟩`, where `J v = 0`, `Jᵀ w = 0`, and `w` is normalized so
`⟨w, v⟩ = 1`. `B(v,v)` is the second directional derivative `D²f(u)[v,v]`, computed with the same
nested-`jax.jvp` technique `lyapunov_coefficient` already uses — no finite differences, no
`jnp.linalg.eig` inside the differentiated path. `w` comes from an SVD nullspace of `Jᵀ`, which is
never itself differentiated (same treatment as `fold_solve.py`'s `_initial_v` seed).

Unlike `lyapunov_coefficient`, `a` needs only real arithmetic and only second derivatives, so it
carries no holomorphy requirement.

### Sign convention for ω

The Hopf-block residual is *exactly* invariant under `(ω, q₂) → (−ω, −q₂)`: substituting flips
the sign of the `J q₂ − ω q₁` row and leaves `J q₁ + ω q₂` unchanged, so both signs are genuine
roots. Newton picks whichever the seed falls toward, and in planning it reliably chose the
negative one even from positive seeds (see below). Every solver returning an `ω` (`GH`, `ZH`,
`HH`) therefore applies a **post-solve normalization**: if `ω < 0`, negate both `ω` and `q₂`. This
is exact, not a heuristic — it selects the conjugate of the same eigenvector — and costs one
`jnp.where`. It must be applied outside the residual, not as an extra equation, which would break
squareness.

This matters beyond cosmetics: `bifurcations/events.py:Hopf.refine()` already guards on
`omega0 > 0` and reports `"unknown"` otherwise, so an unnormalized negative ω would be reported
as a failed solve.

### Failure and degeneracy reporting

Every solver returns a trailing boolean `converged` rather than raising, keeping the whole surface
`jit`/`vmap`-safe and consistent with `Hopf.refine()`'s existing `"unknown"`-instead-of-throw
behaviour:

```python
u, p, q1, q2, omega, ok = jc.double_hopf_point(f, u_guess, p_guess, args)
```

`converged` is the conjunction of:

1. **Finiteness** of every returned array — the non-finite-result footgun the Hopf work fixed
   (`nan < 0` and `abs(nan) < tol` are both `False`, so an unguarded sign test silently mislabels
   a failed solve).
2. **Residual below `tol`** at the returned point.
3. **`HH` only — pair separation.** `abs(|ωᴬ| − |ωᴮ|)` must exceed `separation_tolerance`
   (default `1e-3`). Note both the inner and outer absolute values: the inner ones because ω's
   sign is not meaningful (see below), the outer one because the two blocks may converge in
   either order. If both Hopf blocks are seeded onto the same physical pair the Jacobian is
   structurally singular; in planning this produced `nan` rather than a plausible wrong answer,
   but the check makes the diagnosis explicit rather than leaving a bare `nan`. The distinct-seed
   requirement is a documented caller contract for `HH`, mirroring how `hopf_point` documents that
   its guess must be near an actual Hopf point.

### Public surface

The five point solvers, their five `*_parameters` companions, and `fold_coefficient` are exported
from `jaxcont/__init__.py` alongside `fold_point`/`hopf_point`, and documented in
`docs/source/api/index.rst` in a new "Codim-2 point solvers" section mirroring the existing fold
and Hopf sections.

**`taxonomy.py` needs no status changes**, contrary to an earlier draft of this spec. Its `status`
field holds the version a label *lands in*, not a boolean — `LC` reads `"v0.2"` and is implemented
today. `CP`/`BT`/`ZH`/`HH`/`GH` already read `"v0.3"`, which is exactly right for work landing in
v0.3. What *is* stale is the field's own docstring (`"v0.1" (implemented today), "v0.2"/"v0.3"
(planned)`), written before v0.2 shipped; it should be reworded to describe the field as the
landing version. This is a one-line comment fix, not a data change.

## Verified during planning (2026-08-05)

Each extended system was built as a standalone float64 prototype and Newton-solved from a
deliberately perturbed guess against a closed-form normal form whose codim-2 point is known
exactly. This is design-time evidence, not a substitute for the implementation's own tests.

| System | Unknowns = Equations | Final residual | cond(J) at solution | Recovered exact point |
|---|---|---|---|---|
| `BT` | 8 = 8 | 8.6e-26 | 9.5 | ✅ |
| `GH` | 9 = 9 | 2.3e-16 | 4.1 | ✅ |
| `ZH` | 15 = 15 | 6.6e-16 | 2.0 | ✅ |
| `HH` | 24 = 24 | 3.1e-14 | 4.0 | ✅ |

Three findings came out of this, each of which changed the design above:

1. **ω's sign is unconstrained.** `GH`, `ZH`, and `HH` all converged to *negative* ω (`−1.0`,
   `−1.0`, and `−1.0`/`−2.0` respectively) from positive seeds. This is the exact symmetry
   documented under "Sign convention for ω" and is why post-solve normalization is required rather
   than optional.
2. **`HH` is genuinely degenerate when both pairs are seeded onto the same physical pair** — the
   probe returned `nan` for the residual, both frequencies, and the condition number. It fails
   loudly rather than silently, but with no diagnosis, hence the separation check.
3. **Conditioning is uniformly benign** (2.0–9.5 across all four probed systems), so no
   preconditioning, row scaling, or trust-region machinery is warranted. The plain Newton solve
   `differentiable_root` already provides is sufficient.
4. **The obvious test systems have no discriminating power.** Caught while reviewing the first
   draft of this spec: every textbook normal form places its codim-2 point at `u = 0, p = (0,0)`,
   so a stub returning zeros passes all of them. Re-running the `BT` probe on an affinely shifted
   system (exact answer `u* = (5,2)`, `p* = (3,−1)`) recovered it to 6.5e-13 with the condition
   number unchanged at 9.4559 — confirming the shift costs nothing numerically while making the
   test actually able to fail. Hence the mandatory shifted-form requirement in the testing plan.

The BT overdetermination noted in the design section was also found here: the textbook
left/right-null-vector formulation yields `3n+3` equations for `3n+2` unknowns, which is why the
Jordan-chain form is specified instead.

## Testing / verification plan

Mirrors `tests/test_hopf_normal_form.py`'s structure, in a new `tests/test_codim2.py` plus
`tests/test_fold_normal_form.py`.

**Closed-form ground truth**, one system per solver, each with an exactly-known codim-2 point:

- `CP`: `x' = β₁ + β₂x + x³` — cusp at `x = 0`, `β = (0,0)`.
- `BT`: `x' = y`, `y' = β₁ + β₂x + x² + xy` — BT at `u = 0`, `β = (0,0)`.
- `GH`: Bautin normal form `r' = r(β₁ + β₂r² − r⁴)`, `θ' = 1` in Cartesian coordinates — GH at
  `u = 0`, `β = (0,0)`, `ω = 1`.
- `ZH`: `w' = β₁ + w²` decoupled from `x' = β₂x − y`, `y' = x + β₂y` — ZH at `u = 0`, `β = (0,0)`,
  `ω = 1`.
- `HH`: two decoupled rotations with frequencies 1 and 2 — HH at `u = 0`, `β = (0,0)`,
  `ωᴬ = 1`, `ωᴮ = 2`.

**Every one of the above must ALSO be tested in an affinely shifted form**, and this is not
optional. All five normal forms place their codim-2 point at `u = 0, p = (0,0)`, so an
implementation that simply returned zeros would pass all five — they have no discriminating power
on their own. Substituting `u → u − u₀`, `p → p − p₀` moves the answer to a non-trivial known
point while keeping ground truth exact and conditioning unchanged. Verified during planning on
`BT` shifted to `u* = (5,2)`, `p* = (3,−1)`: recovered to 6.5e-13, `cond(J)` identical to the
origin-centred version (9.4559 vs 9.4559), while a `return zeros` stub scores error 0.0 on the
origin-centred system and 5.0 on the shifted one.

**Independent cross-validation against BifurcationKit.jl.** Closed-form normal forms verify the
mathematics but cannot catch a *convention* mismatch — sign and normalization choices for `a` and
`l₁` are exactly where independent implementations disagree, and this project already cross-checks
`lyapunov_coefficient` this way
(`test_lyapunov_coefficient_matches_bifurcationkit_jl_independent_run`). The plan must add an
equivalent for codim-2, following the established pattern: an offline Julia run whose results are
hardcoded as reference values in the Python test, with the generating script committed as
`examples/BifurcationKit/05_codim2.jl` (matching `04_hopf_normal_form.jl` from the predecessor
work). Prototype/probe scripts stay out of `docs/superpowers/` — that tree is markdown-only, and
the numeric findings below are recorded in prose rather than as committed code.

Status, honestly: BifurcationKit.jl v0.5.2 is installed and working in this dev environment, and a
codim-2 run was attempted during planning on Bazykin's predator-prey model. In the parameter
regime tried it exposed only a transcritical point, no BT/Hopf, so **no reference values were
obtained and this remains an open plan task, not a solved one.** Two viable routes, in preference
order: (a) drive BifurcationKit's `continuation(br, i, lens2; detect_codim2_bifurcation=2)` on an
applied model with documented codim-2 structure — a real independent check on a system neither
implementation was tuned for; (b) failing that, run it on the shifted normal forms above, which
still cross-checks the two implementations against each other even though ground truth is already
known analytically. Route (a) is worth real effort before falling back to (b). MATCONT is an
acceptable substitute for either if the Julia route proves intractable.

**Beyond location accuracy:**

- **Gradient correctness:** `jax.grad` of each `*_parameters` against central finite differences,
  on a variant whose codim-2 point moves with a scalar in `args`. This is the feature's headline
  claim and must be tested directly, not inferred from `differentiable_root` being tested
  elsewhere.
- **`fold_coefficient` sign and scaling:** `a` must flip sign with the quadratic term's sign and
  scale linearly with its magnitude — the same shape of test `lyapunov_coefficient` uses.
- **ω normalization:** assert every returned `ω > 0`, seeded from *both* signs. This is the
  regression test for finding 1; without it the bug reappears invisibly.
- **`converged=False` paths:** a guess far from any codim-2 point, and (for `HH`) two pairs seeded
  onto the same physical pair. Assert the flag is `False` rather than asserting on the garbage
  values.
- **Consistency with codim-1 solvers:** at a `CP`, `fold_point` seeded nearby must return the same
  `(u, p₁)`; at a `GH`, `hopf_point` likewise, and `lyapunov_coefficient` there must be ≈ 0.
- **float32 floor:** tolerances are calibrated against what float32 actually achieves, per this
  project's repeated finding that the achievable residual floor is system-specific and does not
  transfer between systems.

## Implementation phasing

Five solvers is more than one plan should land at once. Suggested order, each independently
verifiable and leaving the tree green:

1. `_solve_codim2` harness + `fold_normal_form.fold_coefficient` (+ its tests).
2. `CP` and `BT` — the two real-arithmetic systems, no ω, no complex seeds.
3. `GH` — introduces the ω normalization and reuses `lyapunov_coefficient`.
4. `ZH` and `HH` — the combined systems, plus the separation check and degeneracy tests.
5. BifurcationKit.jl cross-validation (see the testing plan) — deliberately last, because it is
   the one task with an unresolved research component and must not block the four above. It is
   still required, not optional.

## Risks

- **Seeding is the whole game.** Every solver needs a guess already near the codim-2 point;
  none of them search. This is the same documented contract as `hopf_point` and is acceptable, but
  it means these functions are a *refinement* tool. Finding codim-2 points you cannot already
  approximate needs two-parameter continuation, which is out of scope here.
- **`HH` and `ZH` have no demand behind them.** They were included by explicit choice; if their
  degenerate-case handling proves expensive during implementation, dropping them to a follow-up is
  a reasonable scope cut that leaves `CP`/`BT`/`GH` intact.
- **float32 by default.** Planning used float64 to separate design errors from precision floors.
  The shipped tolerances must be recalibrated under float32 — the periodic-orbit and limit-cycle
  work both hit this, and both found the floor to be system-specific.

## References

- Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed. — Ch. 8 (codim-2 equilibrium
  bifurcations), §8.3 (BT), §8.4 (ZH/HH), §5.4 (normal-form coefficients).
- MATCONT manual §4 — the `CP`/`BT`/`ZH`/`HH`/`GH` defining systems and label conventions
  `taxonomy.py` already mirrors.
- `docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md` — the immediate predecessor;
  `lyapunov_coefficient`, the seed-based phase condition, and the non-finite-result guard all
  originate there.
