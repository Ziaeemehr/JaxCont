# Equilibrium branch-point switching — design spec

**Date:** 2026-08-26
**Status:** approved, ready for implementation planning
**Roadmap item:** v0.3.0+ "Branch switching" (`notes/ROADMAP.md`) — the last item on the
"match MatCont's most-used subset" list still unbuilt as of v0.4.0.

## 1. Problem

`taxonomy.py` already reserves the abbreviation `BP` ("Branch Point", equilibrium,
status `"v0.3"`) but nothing detects or acts on one. `ARCHITECTURE.md` §6 sketches the
target surface (`jc.branch_switch(sol, event=sol.events[0], settings=...)`) but is
explicitly a provisional signature, "do not implement or over-specify now" — this spec is
that "now".

A branch point (pitchfork/transcritical-type) is where **two distinct equilibrium
branches cross**. Continuing through one with the ordinary predictor-corrector just
follows whichever branch the tangent happens to be aligned with; it never discovers the
other one. Branch switching is the operation that does: detect the crossing, compute the
other branch's tangent direction, take one step onto it, and hand back a seed the caller
can feed into an ordinary `continuation()` call.

**Scope for this spec: equilibrium BP only** (pitchfork/transcritical points on an
`f(u, p, args) = 0` branch). Two structurally different follow-ups are explicitly
deferred, each its own future spec:
- **Period-doubling switching** (branching onto the new period-doubled limit cycle from a
  detected `PeriodDoubling` event) — different extended system (built on the collocation
  residual, not the plain equilibrium residual), different `Branch` object. Scoped as a
  separate sub-project during this feature's brainstorming; not designed here.
- **`BPC`** (Branch Point of Cycles, the equilibrium-BP analogue for periodic orbits) —
  already tagged `"out of scope"` in `taxonomy.py`; this spec does not change that.

## 2. Why detection is the hard part, not switching

A branch point and an ordinary fold (`LP`) give the **same primary signal**: a real
eigenvalue of `f_u = df/du` crosses zero. `Fold`'s own test function deliberately avoids
eigenvalues (issue #7: a Hopf pair's real part could masquerade as a fold) and uses the
pseudo-arclength tangent's `dp` sign change instead — but that doesn't help distinguish
BP from LP either, since **at a BP the incoming branch's own tangent does not reverse in
`p`** (unlike a genuine turning point). Both LP and BP make `f_u` singular; what
distinguishes them is a strictly stronger condition on the augmented Jacobian:

- At a plain fold: `f_u` is singular (rank `n-1`), but `f_p = df/dp` is **not** in the
  range of `f_u`. The augmented matrix `[f_u | f_p]` (`n x (n+1)`) still has full row
  rank `n` — this is exactly why the fold's bordered extended system (`fold_solve.py`)
  is solvable and unique there.
- At a branch point: `f_u` is singular **and** `f_p` **is** in the range of `f_u`. The
  augmented matrix `[f_u | f_p]` also drops to rank `n-1` — its null space is
  2-dimensional in `(u, p)`-augmented space: one direction is the incoming branch's own
  tangent, the other is the crossing branch's tangent.

So detection is a two-stage process: a cheap per-step test function flags *candidate*
points (real eigenvalue crossing zero — same signal a plain fold gives), and `refine()`
does the expensive, decisive classification (augmented-rank check) that separates true
BPs from ordinary folds.

## 3. Detection: the `BranchPoint` event

New `bifurcations/events.py` class, alongside `Fold`/`Hopf`/`PeriodDoubling`/`NeimarkSacker`:

```python
@dataclass(frozen=True)
class BranchPoint(Event):
    kind: str = "branch_point"
    rank_tolerance: float = 1e-4   # calibrated during implementation, see §5

    def test_function(self, point: BranchPoint) -> float:
        # sign of the real eigenvalue nearest zero -- same signal Fold's
        # eigenvalue-based test would give (deliberately NOT what Fold uses,
        # for the issue #7 reason above); real-valued eigenvalues only
        # (nan sentinel when none exist, matching Hopf's convention)
        ...

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> Optional[EventHit]:
        # 1. Newton-converge via fold_solve.py's EXISTING extended system
        #    (f=0, f_u . v=0, |v|=1) -- both LP and BP satisfy this, it's
        #    just "df/du is singular", not yet the BP-specific condition.
        # 2. At the converged point, compute the smallest singular value of
        #    the augmented [f_u | f_p] (n x (n+1)) matrix.
        # 3. If it's not below rank_tolerance: this was an ordinary fold,
        #    not a branch point -- return None (see §3.1).
        # 4. Otherwise: package info={"tangent": point.tangent} -- the
        #    INCOMING branch's tangent at the confirmed point. EventHit has
        #    no tangent field of its own, and it's the one piece
        #    branch_switch() cannot recompute from (f, u*, p*, args) alone
        #    (see §4 step 3).
        ...
```

`test_function` will fire at *every* real-eigenvalue-crossing point, including every
plain `LP` the `Fold` event already reports — that is intentional (§2: the cheap test
cannot distinguish them) and is exactly why `refine()` must be allowed to say "not
actually a hit".

### 3.1 Protocol change: `refine()` may return `None`

`Event.refine()` currently always returns an `EventHit`. This spec changes the protocol
type to `Optional[EventHit]`, and `detect_events`'s collection loop (`events.py`, near
the `hits.append(event.refine(...))` line) changes to skip `None`:

```python
hit = event.refine(points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations, **kwargs)
if hit is not None:
    hits.append(hit)
```

`Fold`/`Hopf`/`PeriodDoubling`/`NeimarkSacker`'s existing `refine()` implementations are
unaffected (they keep returning `EventHit` unconditionally) — this is a widening of the
protocol, not a breaking change to any existing `Event`.

## 4. Computing the switching tangent and seed

At a `refine()`-confirmed BP `(u*, p*)`:

1. Build the augmented Jacobian `A = [f_u(u*, p*) | f_p(u*, p*)]` (`n x (n+1)`, via
   `jacfwd`, matching every other extended-system module's autodiff-only-derivatives
   convention). `branch_switch` has `event.u`/`event.p` from the `EventHit`; `u*, p*`
   below just abbreviate those.
2. `_, _, vh = jnp.linalg.svd(A)`; the last two rows of `vh` span the 2-D null space
   (mirrors `fold_solve._initial_v`'s existing smallest-right-singular-vector pattern,
   just keeping two singular vectors instead of one).
3. The incoming branch's own known tangent — read from `event.info["tangent"]`, which
   `BranchPoint.refine()` populates from the `point.tangent` it had at detection time
   (`EventHit` itself carries no tangent field; this is the one piece of context
   `branch_switch` cannot recompute from `f`/`u*`/`p*`/`args` alone) — is one vector in
   that 2-D space, up to normalization. Project it out of the two SVD vectors and
   normalize the remainder — this is the new branch's direction, `t_new`.
4. Take one predictor step `x_pred = (u*, p*) + ds * t_new` (small fixed `ds`, a
   `branch_switch` keyword, not reusing `ContinuationPar.ds` — this is a one-off hop, not
   a continuation run), then Newton-correct **jointly in `(u, p)`** via a pseudo-arclength
   corrector, not a fixed-`p` corrector: solve the `(n+1)`-equation system
   `[f(u, p, args); t_new . ((u,p) - x_pred)] = 0` for `x = (u, p)`. **This is a real
   correctness requirement, not a simplification available to skip:** for a pitchfork,
   `t_new` is exactly `(1, 0, ..., 0)` in `(u, p)`-order — the new branch is tangent to
   the `p`-axis at the BP, i.e. `dp/ds = 0` there — so a fixed-`p` corrector holding
   `p = p*` would try to solve `f(u, p*, args) = 0`, which at `p* = 0` has no root other
   than the trivial one (found and verified during this feature's implementation: a
   fixed-`p` corrector silently snaps back onto the original branch for exactly this
   case, passing no test that doesn't check against it explicitly). The pseudo-arclength
   corrector is the same style the engine's own predictor-corrector already uses
   (`core/scan_continuation.py`), just done once here instead of iterated.
5. Return `jc.bif_problem(f, u0=u_corrected, p0=p_corrected, args=...)` — an ordinary
   `BifProblem`, `kind="equilibrium"`, ready for `continuation()` exactly like any other
   problem. No new `BifProblem.kind` needed.

### 4.1 Public API

```python
def branch_switch(
    f: Callable[[Array, Array, PyTree], Array],
    event: EventHit,          # a BranchPoint hit from sol.events
    *,
    args: PyTree = None,
    ds: float = 0.01,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> BifProblem:
```

in a new `bifurcations/branch_switch.py`, re-exported as `jc.branch_switch` — matching
`ARCHITECTURE.md`'s sketched name, but taking `f`/`args` explicitly (like
`fold_curve_problem` does) rather than the whole `ContinuationResult`, since that is all
it actually needs and keeps the factory pure. `event` must be a `BranchPoint` hit
(`event.kind == "branch_point"`); a wrong-kind `EventHit` raises `ValueError` eagerly
(mirrors `fold_curve_problem`'s existing seed-convergence `ValueError`, including the
same "no-op under a traced call" deferral for the same `ConcretizationTypeError` reason).

## 5. Numerical calibration

Every extended-system feature in this codebase has needed a measured, not assumed,
float32 tolerance floor (`fold_curve_problem`: 2.4e-7; `hopf_curve_problem`: 5.96e-8;
codim-2 solvers: 4.4e-8 to 5.96e-8). This spec is no different — `rank_tolerance` (the
augmented-matrix smallest-singular-value cutoff distinguishing BP from LP) and the
corrector's `newton_tol` both get measured against the closed-form fixtures in §6 during
implementation, not guessed. Expect `rank_tolerance` to need to sit comfortably above the
float32 noise floor of an ordinary (non-BP) fold's smallest singular value, so a plain
`LP` never gets misclassified — this margin is exactly what the pitchfork/transcritical
fixtures' "must NOT fire on an ordinary fold" test (§6) is for.

## 6. Verification

Matches this project's standing bar: closed-form fixtures with a known second branch, not
just "it ran".

- **Pitchfork** `f(u, p) = p*u - u**3`: trivial branch `u=0` for all `p`, BP at
  `(u,p)=(0,0)`, second branch `u = ±sqrt(p)` for `p > 0`. `branch_switch` from the
  detected BP must land on (and then continue onto) `u = sqrt(p)` (or `-sqrt(p)`,
  whichever the SVD's sign convention picks — assert `|u| ≈ sqrt(p)`, not a specific
  sign).
- **Transcritical** `f(u, p) = p*u - u**2`: trivial branch `u=0`, BP at `(0,0)`, second
  branch `u = p`. `branch_switch` must land on `u ≈ p`.
- **Negative control**: a plain fold system (e.g. `saddle_node` from
  `tests/test_functional_api.py`, `f(u,p) = u**2 + p`) must **not** produce a
  `BranchPoint` hit — `BranchPoint.refine()` returns `None` there, confirming the
  augmented-rank check actually discriminates rather than firing on every real-eigenvalue
  crossing.
- **`vmap`-batched example**, matching "go further than MatCont everywhere": batch the
  pitchfork's growth-rate parameter, switch-and-continue every batch element in one
  kernel.
- **Differentiable variant**: `jax.grad`/`jax.jacfwd` through `branch_switch`'s output
  parameter (e.g. gradient of the new branch's starting state w.r.t. a design parameter
  in `args`), matching every other extended-system feature's "ship differentiable" bar —
  achieved for free if the corrector step (§4 step 4) reuses
  `solvers/implicit.py:differentiable_root` rather than a bespoke non-differentiable
  Newton loop; this is a real design constraint on the implementation, not just a nice-to-have.
- **MatCont cross-validation**: MatCont's own manual has genuine BP test cases (its
  pitchfork/branch-point testruns). Add one following `python_cases/codim2_curves.py`'s
  precedent from the two-parameter-continuation/MatCont-follow-up work earlier this
  session (hardcoded reference values checked against a live MatCont run, not a
  `compare.py` extension) rather than inventing a third comparison mechanism.

  No plain-ODE MatCont `init_BP_EP` testrun was found among the checked MatCont 7.6
  testruns. The feature's correctness evidence therefore rests on the closed-form
  pitchfork/transcritical fixtures above plus the differentiability and `vmap` checks.
  `US-BP-001` (Bratu, BVP-based) remains `unsupported` pending BVP support; that
  limitation is unrelated to this feature's own completeness.

## 7. Deliverables

- `bifurcations/events.py`: `BranchPoint` event; `Event.refine()` protocol widened to
  `Optional[EventHit]`; `detect_events`'s collection loop updated to skip `None`.
- `bifurcations/branch_switch.py`: `branch_switch()`.
- `taxonomy.py`: `BP`'s `status` field updated from `"v0.3"` to reflect it's now
  implemented (matching how other entries track status, e.g. `LPC`/`NS`/`PD` read
  `"v0.2"` once shipped).
- `viz/styles.py`: `"branch-point"` style entry already exists (`BifStyle("D",
  "#7B61A8", "BP")`) — verify `plot_continuation`/`plot_bifurcation_diagram` actually
  route a `kind="branch_point"` `EventHit` to it; wire up if not.
- Top-level export: `jc.branch_switch`, `jc.BranchPoint`.
- New example script (pitchfork branch-switch, following the numbered `examples/`
  convention) and the `vmap`/differentiable coverage from §6.
- `ROADMAP.md`: close out the "Branch switching" v0.3.0+ item; note period-doubling
  switching as the explicit next follow-up (§1).

## 8. Explicitly out of scope

- Period-doubling switching (own future spec, §1).
- `BPC` / branch points of cycles (`taxonomy.py`: `"out of scope"`, unchanged).
- Automatic re-detection/de-duplication across repeated `branch_switch` calls on the same
  physical BP from multiple approach directions — the caller drives `continuation()`
  themselves after `branch_switch`, same as every other curve/point factory in this
  codebase; no new bookkeeping layer.
- Higher-codimension branch points (e.g. where three or more branches meet) — MatCont
  treats these as a separate, rarer case; not addressed here.

## 9. References

- Kuznetsov, *Elements of Applied Bifurcation Theory*, §4 (branch points, the augmented-
  Jacobian rank characterization).
- MatCont manual, branch-point continuation and switching sections.
- `notes/ARCHITECTURE.md` §6 (the provisional `jc.branch_switch` sketch this spec
  formalizes).
- `bifurcations/fold_solve.py` (the extended system and `_initial_v` SVD pattern this
  design reuses/extends).
