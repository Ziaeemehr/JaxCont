# Period-Doubling Branch Switching — Design

**Status:** approved (2026-08-27)
**Scope:** switch from a detected `PeriodDoubling` event on a periodic-orbit
branch onto the period-doubled cycle that bifurcates there.

This is the second and final sub-project of the branch-switching feature. The
first, equilibrium branch-point switching, shipped 2026-08-26 — see
`2026-08-26-branch-point-switching-design.md`. The two share a philosophy
("cheap linear seed, let Newton do the rest") but almost no code: the
equilibrium case returns a seeded state vector, this one returns a seeded
*collocation* problem on a doubled mesh.

## 1. Why this is a thin wrapper, not new machinery

The obvious worry going in was that period-doubling would need its own
collocation surgery: a doubled mesh, an anti-periodic variational solve for
the true eigenfunction, and a pseudo-arclength corrector (the fix the
equilibrium pitchfork needed, because `dp/ds = 0` exactly at the bifurcation).

Numerical probing during design established that none of that is required.
Four probes were run against the closed-form fixture in §5:

1. **Correcting at `p*` fails, as expected.** Fixed-`p` correction exactly at
   the bifurcation parameter collapses to the trivial branch (amplitude
   ~1e-2, float32 noise) for every seed amplitude tried. This is the same
   pitchfork geometry the equilibrium case hit.
2. **Correcting at `p* + dp` succeeds.** Stepping the parameter off the
   bifurcation first, then running an ordinary fixed-`p` Newton solve,
   converges to the exact doubled amplitude to ~1e-6 and is robust to a 2x
   error in the seed amplitude. **No pseudo-arclength corrector is needed**,
   which means `periodic_orbit_problem` can be reused unchanged.
3. **A generic constant-eigenvector seed is sufficient.** Seeding with
   `eps * s(t) * v0` for a constant `-1` Floquet eigenvector `v0` and an
   anti-periodic scalar `s(t)` works as well as the exact eigenfunction.
   Square-wave and smooth-cosine `s(t)` gave identical accuracy, so **no
   variational eigenfunction solve is needed**.
4. **The extracted eigenvector works end to end.** Recomputing the monodromy
   matrix at the event point and taking the eigenvector for the multiplier
   nearest `-1` reproduces the exact doubled branch.

The consequence: `branch_switch_pd` builds a trajectory guess and hands it to
the existing `periodic_orbit_problem`. There is no new collocation code, no
new solver, and no change to `events.py`.

## 2. The silent-wrong-answer hazard (the reason for §3)

Probe 3 found the failure mode that dominates this design. An under-sized
`eps` does not diverge and does not report a bad residual. It converges
cleanly — residual ~7e-6, indistinguishable from success — onto **the
original orbit traversed twice**, which is a perfectly valid solution of the
doubled-period collocation system. The caller receives a converged
`BifProblem` describing the wrong branch, with no error signal.

Measured behavior at `alpha = 0.05` (exact doubled amplitude 0.2236):

| `eps` / exact amplitude | outcome | residual |
|---|---|---|
| 0.1 – 0.5 | collapsed to original branch | ~8e-6 (healthy) |
| 0.75 – 5 | correct doubled branch | ~8e-6 |
| ≥ 10 | Newton diverges | ~3e3 (loud) |

Divergence is self-reporting. Collapse is not. A detector is therefore
mandatory, not optional hardening.

**Detector.** For a doubled orbit sampled on a uniform mesh of `ntst_d`
points over `2T`, the time shift `t -> t + T` is an index roll of
`ntst_d / 2`. Define

```
d = max_i || mesh_states[i] - mesh_states[i + ntst_d/2] ||
d_rel = d / max_i || mesh_states[i] ||
```

A genuine doubled orbit has `d_rel ~ 0.44` in the fixture; the collapsed case
has `d_rel ~ 1e-6`. The separation is five orders of magnitude, so a threshold
of `1e-3` is unambiguous. `ntst_d` is even by construction (§4), so the roll is
exact.

**The parameter step's sign is also not knowable in advance.** The doubled
branch exists on only one side of the bifurcation (supercritical vs
subcritical, system-dependent). Probe 4 confirmed this directly: `dp = +0.05`
found the branch at every sufficient `eps`, while `dp = -0.05` collapsed at
every `eps`. The wrong sign is indistinguishable from a too-small `eps` from
the outside — both just collapse — so the search must span both.

## 3. Search strategy

`branch_switch_pd` tries `(sign, eps)` combinations and accepts the first
whose corrected result passes the detector:

- `sign` over `(+1, -1)` — unless the caller pins `dp_sign`. The step
  *magnitude* `dp` defaults to `0.05` and is a caller parameter: it is in the
  caller's own parameter units, so no library default can be universally
  right. A poor choice is caught loudly by the detector rather than silently
  returning a wrong branch, which is what makes a plain default acceptable
  here.
- `eps` over a geometric ladder relative to the base orbit's own scale
  (`max_i ||u_base[i]||`), default `(0.05, 0.1, 0.2, 0.4, 0.8)`. The ladder is
  relative because the correct `eps` is system-dependent; the upper end stops
  well below the ~10x-exact divergence threshold measured in §2.

Each candidate costs one collocation Newton solve. If every candidate fails,
the function raises `ValueError` naming what was tried and both failure modes
(collapse vs. non-convergence) per candidate, rather than returning a
plausible-looking wrong branch. **Returning an unverified problem is not an
acceptable outcome of this function.**

## 4. Algorithm

Input: the ODE right-hand side `raw_f(u, p, args)`, a `PeriodDoubling`
`EventHit`, and the `Collocation` mesh the original branch was continued on.

1. **Unpack the event point.** `event.u` is the flat collocation vector in
   `problems/periodic.py`'s layout (`ntst` mesh states, then `ntst*ncol`
   collocation states, then the period `T`); `event.p` is the bifurcation
   parameter. Derive `n` algebraically as
   `n = (len(U) - 1) // (ntst * (1 + ncol))`, matching
   `stability/floquet.py:floquet_multipliers`.
2. **Extract the switching direction.** Build the monodromy matrix via
   `core.collocation.monodromy_matrix(raw_f, D, E, h, mesh_states,
   coll_states, T, p)` with `D, E` from `collocation_matrices(ncol)`, then
   `jnp.linalg.eig`. Select the eigenvalue nearest `-1`. Verify it is
   genuinely a period-doubling multiplier — `|lambda + 1| < pd_tol` (default
   `1e-2`, loose because the event point is a bisection midpoint at the
   float32 residual floor, not a Newton-refined point) and
   `|Im(lambda)| < 1e-3` — and raise if not, since that means the caller
   passed something that is not a PD point. Take the real part of the
   corresponding eigenvector and normalize.

   This deliberately recomputes the monodromy rather than reading a cached
   eigenvector off `EventHit.info`. `PeriodDoubling.refine` does not currently
   store one, and `branch_switch_pd` already requires `raw_f` and `mesh` for
   step 4 — so recomputation costs one small `n x n` eigendecomposition and
   keeps `events.py` untouched.
3. **Build the doubled trajectory guess.** The base orbit's mesh points sit at
   `tau = i/ntst` over `[0, T)`. Sample `n_samples` points uniformly over
   `[0, 2T)` (default `max(240, 12 * ntst)`, comfortably oversampling the
   doubled mesh that `periodic_orbit_problem` will resample onto) and
   evaluate

   ```
   u_seed(t) = u_base(t mod T) + eps * cos(pi * t / T) * v0
   ```

   `u_base` is periodic linear interpolation of `mesh_states` (append the
   wrap-around point so `t` just below `T` interpolates back to
   `mesh_states[0]`). `cos(pi t / T)` is anti-periodic over `T` — it changes
   sign between the two copies, which is what makes the guess a
   period-*doubled* rather than period-`T` object.
4. **Correct.** Call `periodic_orbit_problem(raw_f, u_seed, t_samples, 2*T,
   event.p + sign*dp, Collocation(ntst=2*ntst, ncol=ncol))`. Doubling `ntst`
   preserves per-unit-time resolution across the 2x longer period (matching
   MatCont, and what the probes verified); `ncol` is unchanged.
   `periodic_orbit_problem` does the resampling and the fixed-`p` Newton
   refinement internally, so this step introduces no new solver code.
5. **Verify.** Compute `d_rel` per §2 on the returned problem's `u0`, and the
   residual norm. Accept only if `d_rel > doubling_tol` and the residual is at
   the expected float32 collocation floor. Otherwise move to the next
   candidate.
6. **Return** the accepted `BifProblem` (`kind="periodic"`, doubled mesh),
   ready to pass to `jc.continuation`.

## 5. Test fixture

The repo's existing PD fixture (`tests/test_period_doubling_neimark_sacker.py`)
has a **linear** transverse block, so `w = 0` is an exact solution for every
parameter value and no period-doubled branch exists to switch onto. It is
correct for detection and useless for switching. This work adds a
cubic-saturated variant with a closed-form doubled branch:

```
x' = (1 - r^2) x - y                    r^2 = x^2 + y^2
y' = (1 - r^2) y + x                    -> circle, period T = 2*pi
w' = (alpha + i*beta) w - c |w|^2 w     w = w1 + i*w2, beta = pi/T, c = 1
```

The linear part rotates `w` by exactly `pi` over one period, so `w(T) =
-w(0)`: anti-periodic, giving a period-doubling at `alpha = 0`. For
`alpha > 0` the exact doubled orbit is

```
|w| = sqrt(alpha / c)     period = 2T = 4*pi
```

so both the amplitude and the period of the switched branch have closed-form
ground truth. Verified during design: the switch reproduces `|w|` to ~1e-6 and
the period to float precision.

## 6. Testing

- **Positive:** switch at the detected PD point, assert amplitude matches
  `sqrt(dp/c)` and period matches `2T`.
- **Collapse regression (the important one):** force a ladder of only
  under-sized `eps` values and assert `branch_switch_pd` **raises** rather
  than returning the collapsed trivial branch. This pins the §2 hazard.
- **Wrong-sign coverage:** assert the automatic both-signs search finds the
  branch regardless of which side it lies on.
- **Rejection:** assert a non-`period_doubling` event raises, and that a point
  whose nearest multiplier is not near `-1` raises.
- **Continuation:** continue the returned problem and assert a valid branch of
  the expected doubled period results.
- **Cross-validation:** search MatCont 7.6 testruns for a plain-ODE PD
  switching case (`init_PD_LC`). If none is usable, document the fallback the
  way the equilibrium spec did rather than forcing a fit.

  **MatCont cross-validation outcome (2026-08-27):** `Testruns/testadapt2.m`
  is a plain three-state ODE case using `TestSystems/adaptx.m` and
  `init_PD_LC`. A live MatCont 7.6 run located the source PD at
  `alpha=0.630302040837`, `T=6.3640713323`, and the first PD on the switched
  doubled branch at `alpha=0.579629869689`, `T=12.7343655405`. JaxCont
  independently continued the transcribed `adaptx` cycle, located the
  Floquet `-1` crossing, then switched it with `jc.branch_switch_pd` at
  `dp=-0.05`; it obtained `alpha=0.629574656`, `T=6.36421783` before the
  switch and a doubled period `T=12.73437309` after it. The supported
  `MC-LC-003` registry case and `test_adaptx_period_doubling_switch_matches_matcont`
  now pin this comparison, including the non-collapsed doubled-orbit check.

  **What is and is not compared.** JaxCont's *switched parameter* is just the
  detected PD offset by this case's fixed `dp`, so it is checked only for
  internal consistency — comparing it to any MatCont value would test
  arithmetic, not the switch. The switch itself is validated three ways
  instead: the doubled period against MatCont's doubled-branch start
  (`T = 12.7281426646`, which is *exactly* twice the base period); the
  period ratio against 2 (phase-invariant and `dp`-independent, measured
  error 9.3e-4); and, most tightly, the period against MatCont's reported
  point at the same alpha (measured error 7.6e-6) — guarded by an explicit
  assertion that the two alphas still coincide, so a future `dp` change fails
  loudly instead of quietly comparing unrelated points.

## 7. Deliverables

- `bifurcations/pd_switch.py`: `branch_switch_pd()`. No changes to
  `events.py`, `collocation.py`, `periodic.py`, or the `EigenSolver` protocol.
- `api.py` / `__init__.py`: export `branch_switch_pd`.
- `tests/test_pd_switch.py`: the cases in §6.
- `examples/example_22_period_doubling_switch.py`: detect, switch, continue
  both branches, plot.
- Docs: `ROADMAP.md`, `ARCHITECTURE.md`, `CHANGELOG.md`.

## 8. Explicit scope limits

- **Not `vmap`/`grad`-batched.** `problems/periodic.py` documents that
  `periodic_orbit_problem`'s *construction* is unsafe under an outer
  `jax.grad`/`jax.vmap` (its phase-condition reference is computed before an
  internal `lax.while_loop`, so a tracer can leak across that boundary); the
  returned `BifProblem` is fully safe. `branch_switch_pd` calls that
  construction and therefore inherits the limitation, so this feature ships
  eager-only and its example is eager-only. This breaks the project's usual
  "ship a vmap-batched + differentiable example" convention **knowingly**, not
  by oversight. Lifting it is a separate future item (§9) — an isolated fix to
  `periodic_orbit_problem` that would benefit periodic-orbit continuation
  generally, not just this feature, and that touches shared machinery current
  tests depend on.
- **Branch points on cycles (BPC) are out of scope.** MatCont also supports
  switching at branch points *of* limit cycles. That is a third case, distinct
  from both shipped sub-projects, and is not planned here.
- **No adaptive mesh.** `ntst` doubles once at construction; the doubled branch
  is then continued on a fixed mesh, consistent with the rest of the
  periodic-orbit machinery.

## 9. Follow-up item created by this work

**Make `periodic_orbit_problem` construction `vmap`/`grad`-safe.** Recompute
the phase-condition reference derivative inside the differentiable solve
rather than before it, so the whole factory can be batched and
differentiated. This would let both periodic-orbit continuation and PD
switching ship the batched/differentiable examples the rest of the library
has. To be tracked on `ROADMAP.md` as its own item, not folded into this
feature.
