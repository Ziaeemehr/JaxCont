# Period-Doubling / Neimark–Sacker Detection — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** v0.2.0 "Periodic orbits" (`notes/ROADMAP.md`), third checklist item — "Period-doubling
detection." Third sub-project of the same epic as periodic-orbit collocation continuation and Floquet
multipliers (both already shipped — see their
[design](2026-07-24-periodic-orbit-collocation-design.md)/[plan](../plans/2026-07-24-periodic-orbit-collocation.md)
and [design](2026-07-24-floquet-multipliers-design.md)/[plan](../plans/2026-07-24-floquet-multipliers.md)).

## Motivation

Floquet multipliers (`Branch.eigenvalues` for `kind="periodic"` branches, via `compute_stability=True`)
now exist, but nothing detects the two standard codim-1 bifurcations they signal: a real multiplier
crossing `-1` (period-doubling / flip) and a complex-conjugate pair crossing the unit circle away from
the real axis (Neimark–Sacker / torus). The Floquet design spec explicitly flagged both as "the natural
next `Event` implementations, not built here." This spec is that feature, for both — scoping analysis
during brainstorming found the code and verification cost for adding both together is close to the cost
of adding one (see Scope and Verification System below), so both are in scope rather than splitting them
across two sub-projects.

## Scope

**In scope:** two new `Event` implementations, `PeriodDoubling` and `NeimarkSacker`, added to
`bifurcations/events.py` alongside the existing `Fold`/`Hopf`, detecting the two Floquet-multiplier
crossing conditions along a periodic branch, using the existing `Event` protocol / `detect_events`
infrastructure unchanged.

**Out of scope (explicit):**
- Any change to `Event`, `BranchPoint`, `EventHit`, or `detect_events` themselves — both new events are
  ordinary implementations of the existing protocol.
- Any change to `core/scan_continuation.py`, `core/collocation.py`, `stability/floquet.py`,
  `solvers/protocols.py`, or `problems/periodic.py`.
- Enforcing the `PeriodDoubling()`/`NeimarkSacker()`-on-equilibrium-branch footgun with a raise (documented
  only, matching how `Hopf()`-on-periodic is already handled — see Symmetric Footgun below).
- Limit-cycle example scripts (Van der Pol, Brusselator) — the last remaining v0.2.0 checklist item,
  a separate future sub-project.
- Branch switching at a detected period-doubling/Neimark–Sacker point (i.e. continuing the *new*
  branch that emerges) — detection only, not continuation onto the bifurcating branch.

## Mathematical background: why 3+ state dimensions

For a **2D** autonomous periodic orbit, the single non-trivial Floquet multiplier is
`exp(∫₀ᵀ div(f) dt)` — the exponential of a real number, hence always positive real. It can never be
negative (period-doubling) or complex (Neimark–Sacker). Both bifurcations are only possible for periodic
orbits of systems with **3 or more state dimensions**. This means the existing 2D circle system
(`r'=r(ρ-r²), θ'=1`), used to verify collocation and Floquet multipliers, cannot itself exhibit either
bifurcation — a new verification system is required (see below).

## Architecture

`PeriodDoubling` and `NeimarkSacker` join `Fold`/`Hopf` as two more `@dataclass(frozen=True)` `Event`
implementations in `bifurcations/events.py`. No changes to the `Event` protocol, `BranchPoint`,
`EventHit`, or `detect_events` — both are drop-in consumers of `BranchPoint.eigenvalues`, exactly the way
`Hopf` already is, since for `kind="periodic"` branches with `compute_stability=True` that field now holds
real Floquet multipliers (shipped in the prior sub-project).

Both events identify and exclude the trivial multiplier the same way `stability/floquet.floquet_stable`
already does — `argmin(|multiplier - 1|)` — before applying their own test:

- **`PeriodDoubling`**: among the remaining multipliers, filter to those both (a) real (`|imag| <
  tolerance`) and (b) near the unit circle (`|magnitude - 1| < near_unit_circle`, see below), then pick
  the one closest to `-1` (`argmin(|multiplier + 1|)`), test function = `real(multiplier) + 1`. Returns
  `nan` (never `inf`, for the same reason `Hopf` uses `nan` — avoids a false sign-change when the
  branch's multiplier structure merely transitions from real to complex) if no candidate remains after
  filtering.
- **`NeimarkSacker`**: among the remaining multipliers, filter to those both (a) complex (`|imag| >
  tolerance`) and (b) near the unit circle (`|magnitude - 1| < near_unit_circle`), then pick the pair
  with `|multiplier|` closest to `1` (`argmin(||multiplier| - 1|)`), test function = `|multiplier| - 1`.
  Returns `nan` if no candidate remains after filtering.

**The near-unit-circle filter is required, not optional — found by end-to-end verification, not design
reasoning alone.** Without it, `argmin(|multiplier + 1|)`/`argmin(||multiplier| - 1|)` can silently
*switch* which physical multiplier it tracks as the branch evolves: once the true transverse multiplier
moves far enough past `-1` (e.g. `real = -2.78`, `|real + 1| = 1.78`), an unrelated multiplier that
merely sits at a roughly constant distance from `-1` (e.g. the circle system's always-decaying multiplier
`≈ 3.4e-6`, `|3.4e-6 + 1| ≈ 1.0`) can become the argmin instead, purely because `1.0 < 1.78` — with no
physical crossing anywhere nearby. This produces a real false-positive `PeriodDoubling` detection: verified
directly against `jc.continuation()` on the `circle ⊕ transverse` system (`β=π/T`, `α` sweep `-0.1→0.1`)
BEFORE adding this filter, which reported two hits — the genuine one at `p≈-2.8e-7` (correct) and a
spurious second one at `p≈0.110` (exactly where `|−exp(αT) + 1|` first exceeds `|3.4e-6 + 1| ≈ 1`,
confirming the mechanism). Adding the filter (`near_unit_circle`, a new field on both classes, default
`0.5`) eliminated the false positive with the same sweep, and a follow-up check across a *wider* sweep
(`-0.1 → 0.3`, deliberately re-exercising the exact range that produced the false positive) confirmed
exactly one detection, at the correct point. `near_unit_circle=0.5` was chosen empirically during this
verification (not a value from the dead stub, which used `0.1` — too tight for even a moderate `α` sweep,
since `|magnitude - 1|` grows quickly with `α·T`) as wide enough to track a multiplier through a
realistic sweep range while still excluding multipliers that were never near the circle to begin with.

Both `refine()` methods bisect exactly like `Hopf.refine` does today: three-way bisection (not a
two-way "left-half-or-else", which degenerates whenever the midpoint test value lands on an exact zero —
see `events.py`'s existing Global Constraints), recomputing Floquet multipliers at each midpoint via
`stability.floquet.floquet_multipliers` (not `_eigenvalues_at`, which is the equilibrium-only
`jacfwd(df/du)` helper). `floquet_multipliers`'s signature is `(raw_f, mesh, U, p, eigen_solver=...)` —
incompatible with `detect_events`'s existing `rhs` parameter (the 2-arg equilibrium convention,
`rhs(u, p) -> Array`, which for a periodic problem is the assembled collocation *residual*, not the raw
ODE — using it here would repeat `Hopf`'s exact footgun in reverse). Rather than changing `Event`,
`BranchPoint`, or `detect_events` to thread `raw_f`/`mesh` through generically (which the Global
Constraints below rule out), `PeriodDoubling`/`NeimarkSacker` each carry `raw_f`/`mesh` as their own
required constructor fields (plus `tolerance: float = 1e-6` and `near_unit_circle: float = 0.5`,
both defaulted), exactly the way `Hopf` already carries `tolerance` — e.g.
`jc.PeriodDoubling(raw_f=my_rhs, mesh=my_mesh)`. `refine` ignores the generic `rhs` argument `detect_events`
passes in and uses `self.raw_f`/`self.mesh` instead. This keeps the `Event` protocol and `detect_events`
completely unchanged, at the cost of the caller supplying `raw_f`/`mesh` a second time (redundant with what
they already gave `periodic_orbit_problem`, but avoids a special-cased protocol just for these two events).

The dead `bifurcations/period_doubling.py` stub (pre-`Event`-protocol, pre-collocation,
`ContinuationSolution`-based) is deleted outright — its only reusable idea, the
`real(multiplier) + 1` test-function formula, is already folded into `PeriodDoubling` above.

## Verification system

A single shared 4D system, `circle ⊕ transverse`, extends the existing verified 2D circle system with a
decoupled linear block:

```
x' = (ρ - x²-y²)x - y      y' = (ρ - x²-y²)y + x     (unchanged circle; ρ=1 fixed → T=2π)
w1' = α·w1 - β·w2           w2' = β·w1 + α·w2          (decoupled linear "transverse" block)
```

`w ≡ 0` is an exact periodic solution for *any* `α`, `β` (the transverse block is homogeneous and fully
decoupled from `x,y` and from itself doesn't feed back), so the test fixture is the existing circle
trajectory zero-padded to 4D — no new simulation needed. Because the transverse block is linear and
time-invariant, its exact contribution to the monodromy matrix is the matrix exponential of a constant
matrix, `exp((α ± iβ)T)` — closed form:

- **`β = π/T`**: multiplier `= -exp(αT)`, real, crosses `-1` exactly at `α = 0` → period-doubling ground
  truth.
- **`β = 0.3`** (any value not a multiple of `π/T`): multiplier `= exp(αT)·e^{±iβT}`, genuine complex
  pair, `|multiplier|` crosses `1` exactly at `α = 0` → Neimark–Sacker ground truth.

`α` is the continuation/bifurcation parameter for both cases; `ρ` stays fixed at `1`. Verified during
design directly against the real pipeline (`periodic_orbit_problem` + `floquet_multipliers`, not a
hand-rolled recursion): all six checked cases (`α ∈ {-0.05, 0, +0.05}` × both `β` values) matched the
closed-form prediction to float32 precision, including both exact-bifurcation cases (`α=0`): the PD
system's transverse multiplier came out `-1.0000045` (predicted exactly `-1`), and the NS system's
transverse pair had magnitude `≈1.0000035` (predicted exactly `1`).

**End-to-end verification against real `PeriodDoubling`/`NeimarkSacker` `Event` code** (not just the
closed-form multipliers above) additionally confirmed, after the near-unit-circle fix described above:
sweeping `α` from `-0.05` to `0.05` with `events=[PeriodDoubling(...)]` on the `β=π/T` system detects
exactly one hit, at `p≈-4.6e-7`; the symmetric `NeimarkSacker(...)` sweep on the `β=0.3` system detects
exactly one hit, at `p≈-4.6e-7`; running each event type against the *other* system produces zero hits in
both directions; and re-running the originally-problematic wider sweep (`-0.1 → 0.3`, `β=π/T`,
`PeriodDoubling`) after the fix produces exactly one hit, at `p≈-2.8e-7` — matching `α=0` to well within
bisection tolerance in every case.

## Symmetric footgun documentation

`PeriodDoubling()`/`NeimarkSacker()` on an **equilibrium** branch is the mirror image of the existing
`Hopf()`-on-periodic footgun (documented in `periodic_orbit_problem`'s docstring): mathematically
meaningless (equilibrium eigenvalues aren't Floquet multipliers, so "crosses `-1`"/"crosses the unit
circle" has no dynamical meaning there), but not enforced by a raise — documented only, in each new
class's docstring, matching the existing precedent exactly. No new runtime check anywhere.

## Testing

Per this project's established standard: empirical verification against a known answer, not just design
reasoning.

1. **Test-function unit tests**: directly on hand-built `BranchPoint`/multiplier arrays at known `α`
   values (no full continuation run needed) — confirms the masking/argmin/sign logic in isolation for
   both `PeriodDoubling` and `NeimarkSacker`, mirroring how `Hopf`'s test function is unit-testable today.
2. **`refine()` accuracy**: run `jc.continuation()` on the `circle ⊕ transverse` system at `β=π/T`
   sweeping `α` across `0` with `events=[jc.PeriodDoubling(raw_f=rhs, mesh=mesh)]`, and separately at
   `β=0.3` with `events=[jc.NeimarkSacker(raw_f=rhs, mesh=mesh)]`; assert the detected `p` (i.e. `α`)
   matches `0.0` to bisection tolerance, and the reported multiplier is within tolerance of `-1` /
   magnitude `1` respectively.
3. **Zero false positives**: run `NeimarkSacker(raw_f=rhs, mesh=mesh)` detection on the `β=π/T` (PD)
   system and `PeriodDoubling(raw_f=rhs, mesh=mesh)` detection on the `β=0.3` (NS) system; assert zero
   detections in both — confirms each test function doesn't fire on the other's bifurcation type.
4. **Near-unit-circle filter regression**: sweep `α` from `-0.1` to `0.3` (`β=π/T`) with
   `events=[jc.PeriodDoubling(...)]` and assert exactly one detection (not two) — regression test for the
   candidate-switching false positive found and fixed during design (see Architecture above).
5. **Regression**: existing `Fold`/`Hopf` tests untouched; full suite green.

## File layout

- **`src/jaxcont/bifurcations/events.py`**: add `PeriodDoubling`, `NeimarkSacker` (dataclasses
  implementing `Event`). No changes to `Event`, `BranchPoint`, `EventHit`, or `detect_events`.
- **Delete `src/jaxcont/bifurcations/period_doubling.py`** (dead, pre-`Event`-protocol stub) outright.
- **`src/jaxcont/api.py`**: re-export `PeriodDoubling`/`NeimarkSacker` alongside the existing
  `Fold`/`Hopf` re-exports (import line, `__all__`), same pattern.
- **Untouched**: `core/scan_continuation.py`, `core/collocation.py`, `stability/floquet.py`,
  `solvers/protocols.py`, `problems/periodic.py`.

## Global Constraints

- `Event` protocol, `BranchPoint`, `EventHit`, and `detect_events` are unchanged — both new events are
  ordinary implementations of the existing protocol, not a protocol extension.
- `core/scan_continuation.py`, `core/collocation.py`, `stability/floquet.py`, `solvers/protocols.py`,
  `problems/periodic.py` are not touched in this sub-project.
- No branch switching / continuation onto the bifurcating branch — detection only.
- No enforcement (raise) of the equilibrium-branch footgun for `PeriodDoubling`/`NeimarkSacker` —
  documented only, matching the existing `Hopf`-on-periodic precedent.
- No limit-cycle example scripts — separate future sub-project.
- The verification system and its closed-form predictions (above) must be re-confirmed against the exact
  code written into the implementation plan before the plan is finalized — not assumed to still hold from
  this spec's own verification alone.
