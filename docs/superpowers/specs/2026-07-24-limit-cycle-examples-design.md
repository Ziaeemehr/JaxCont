# Limit-Cycle Examples (Van der Pol, Brusselator) — Design Spec

**Status:** Approved for implementation planning.
**Roadmap item:** v0.2.0 "Periodic orbits" (`notes/ROADMAP.md`), final checklist item —
"Limit-cycle examples (Van der Pol, Brusselator)." This closes the v0.2.0 epic: periodic-orbit
collocation continuation, Floquet multipliers, and period-doubling/Neimark–Sacker detection all
shipped earlier this session (see their specs/plans under `docs/superpowers/specs/`/`plans/`), and
the user added `examples/example_08_period_doubling.py`/`example_09_neimark_sacker.py` directly,
demonstrating the new `Event` types on a synthetic verification system. Those don't demonstrate a
*real* limit cycle from a classic textbook system, which is what this ROADMAP item asks for.

## Motivation

`examples/example_03_van_der_pol.py` (pre-existing, v0.1.0) only continues the Van der Pol
oscillator's *equilibrium* through its Hopf crossing at `μ=0`, and its docstring explicitly (now
incorrectly) says "Periodic-orbit continuation is outside JaxCont's current scope." Neither Van der
Pol's nor Brusselator's actual limit cycle — continued via the real `periodic_orbit_problem`
collocation machinery — has an example anywhere in the repo. This spec adds both, purely as
example/documentation content (no library changes).

## Scope

**In scope:** two new example scripts (`example_10_van_der_pol_limit_cycle.py`,
`example_11_brusselator_limit_cycle.py`) demonstrating real limit-cycle continuation for both named
systems, plus a one-line docstring fix to `example_03_van_der_pol.py`'s stale scope claim.

**Out of scope:** any change to `src/jaxcont/*` (this is example content only); updating the
Makefile's `examples` target (already stale — missing examples 04–09 — a pre-existing gap unrelated
to this spec); Sphinx-Gallery/docs build wiring beyond what the existing `example_NN_*.py` naming
convention already provides for free.

## Architecture

Both new examples follow `example_08`/`example_09`'s exact structure: docstring with the system's
math, `%%`-delimited sections, build problem, continue, print results, plot, save to `images/`.

Since JaxCont doesn't integrate ODEs itself, each script needs an initial trajectory guess from the
user's own simulation before handing it to `periodic_orbit_problem` for collocation refinement —
this is the intended usage pattern, not a workaround. `scipy.integrate.solve_ivp` (already a project
dependency, unused elsewhere in `examples/` so far) runs a short forward simulation from a
perturbation off the unstable equilibrium, long enough to settle onto the limit cycle; the last full
period (located via `scipy.signal.find_peaks` on the tail) becomes `(u_trajectory, t_trajectory,
period0)`. **The trajectory's time array must be re-based to start at `0`** before being passed to
`periodic_orbit_problem` — its internal resampling computes `t = τ·period0` for `τ∈[0,1]`, so a
`t_trajectory` that doesn't start at `0` falls outside `jnp.interp`'s domain and silently clamps to a
constant (found during design verification: this exact mistake produced a degenerate `T≈0` "solution"
that still reported a deceptively small residual).

- **Van der Pol** (`ẋ=y, ẏ=μ(1-x²)y-x`, reusing `example_03`'s exact rhs): guess simulation starts
  at `μ=1` (comfortably past the degenerate `μ=0` Hopf crossing `example_03`'s docstring already
  explains), continuation sweeps `μ: 1→4`. `mesh=Collocation(ntst=20, ncol=4)`,
  `settings=jc.ContinuationPar(ds=0.05, max_steps=250, newton_tol=1e-5, compute_stability=True)`.
- **Brusselator** (`ẋ=a-(b+1)x+x²y, ẏ=bx-x²y`, `a=1` fixed, Hopf at `b=1+a²=2`): guess simulation
  starts at `b=2.5`, continuation sweeps `b: 2.5→4.0`. Same mesh, but
  `settings=jc.ContinuationPar(ds=0.05, max_steps=350, newton_tol=1e-4, compute_stability=True)` —
  see Verification below for why `newton_tol` differs from Van der Pol's.

Both pass `compute_stability=True` and print/assert (via a plain `if`/`raise`, matching
`example_08`/`09`'s error-on-unexpected-result style, not a silent print) that `Branch.stable` holds
at every point — a light tie-in to the Floquet work, not a new focal point. Both plot a phase
portrait (at the first and last continued parameter value, to show shape change) and
amplitude-vs-parameter / period-vs-parameter.

`example_03_van_der_pol.py`'s docstring line "Periodic-orbit continuation is outside JaxCont's
current scope" is replaced with a pointer to the new `example_10`.

## Verification

Prototyped end-to-end (`scipy.integrate.solve_ivp` guess → `periodic_orbit_problem` →
`jc.continuation()`) before writing this plan, not just reasoned about.

**Real issue found and fixed during design** (a plan-writing mistake, not a library bug): passing
the simulation's raw (non-zero-based) time array as `t_trajectory` silently produces a degenerate
`T≈0` guess (see Architecture above) — `differentiable_root` "converges" (residual ~1e-5, looks
fine) to this degenerate point instead of the real orbit. Fixed by re-basing `t_trajectory` to start
at `0`; re-verified clean afterward.

**Second real finding**: Brusselator's collocation system (same `ntst=20` mesh as Van der Pol) would
not advance past its starting point under `newton_tol=1e-5` — continuation's adaptive step size
collapsed toward `ds_min` every step even though a fresh, independent refinement at a nearby
parameter value converged cleanly in isolation under the same tolerance. This is the same class of
issue as the periodic-orbit-collocation sub-project's original precision finding (achievable float32
residual floor tighter than the requested tolerance stalls continuation silently, no error raised)
but is **not** simply a function of mesh size — Van der Pol used the identical `ntst=20` mesh and
converged fine under `1e-5`. The achievable floor is system-specific and must be checked per example,
not assumed to transfer from a previously-verified value. Fixed by loosening Brusselator's
`newton_tol` to `1e-4`.

Final verified results (`compute_stability=True`, `Branch.stable` true throughout in both cases):

| System | Parameter sweep | `n_valid` | Amplitude | Period |
|---|---|---|---|---|
| Van der Pol | `μ: 1.0 → 4.02` | 138 | `≈2.01 → ≈1.99` (flat — correct for this normalization) | `6.66 → 10.25` |
| Brusselator | `b: 2.5 → 4.00` | 256 | `2.02 → 4.70` (visible growth) | `6.58 → 9.04` |

The two systems make a deliberately different pair: Van der Pol's relaxation-oscillator character
shows up as period growth and waveform sharpening with near-constant amplitude; Brusselator's shows
up as substantial amplitude growth. Worth keeping both, not redundant.

## File layout

- **Create** `examples/example_10_van_der_pol_limit_cycle.py`.
- **Create** `examples/example_11_brusselator_limit_cycle.py`.
- **Modify** `examples/example_03_van_der_pol.py`: one docstring line.
- **Untouched**: everything under `src/jaxcont/`; the Makefile's `examples` target (pre-existing
  staleness, separate issue).

## Global Constraints

- No changes to `src/jaxcont/*` — this is example content only, using the existing, already-shipped
  public API (`periodic_orbit_problem`, `Collocation`, `jc.continuation`) exactly as documented.
- `scipy.integrate.solve_ivp`/`scipy.signal.find_peaks` are used only inside the example scripts
  (the user's own simulation, per the architecture's intended usage pattern) — not inside the
  library, which does not integrate ODEs itself.
- `t_trajectory` passed to `periodic_orbit_problem` must be re-based to start at `0` — verified
  necessary, not a stylistic choice; omitting this silently produces a degenerate result rather than
  an error.
- `newton_tol` is set per example based on its own verified achievable floor (`1e-5` for Van der
  Pol, `1e-4` for Brusselator) — do not assume one value transfers to the other system.
- Both examples must assert (raise on failure, matching `example_08`/`09`'s style) that
  `Branch.stable` holds throughout, not just print it silently.
