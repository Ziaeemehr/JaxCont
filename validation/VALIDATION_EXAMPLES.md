# JaxCont cross-validation suite

**Status:** proposed validation specification  
**JaxCont baseline:** v0.1.0 (equilibrium-only public API)  
**Local reference tools:** MATLAB R2020a and MatCont 7.6  
**Taxonomy source:** *MatCont and CL_MatCont Manual*, August 2019, 124 pages
(``/home/ziaee/Desktop/ManualMatcontAug2019.pdf``)

This document defines examples for validating JaxCont against analytic results,
MatCont, and BifurcationKit.jl. It is deliberately versioned by the JaxCont
release that can support each example. A case belongs to a release only when the
required curve type, event detector, and reported invariants are part of that
release's **supported public API**. The presence of an experimental class or stub
does not count as support.

## Is Markdown the right format?

Yes—for the design, rationale, equations, expected results, and review checklist.
Markdown alone is not sufficient for reproducible numerical validation. The
recommended layout is:

```text
validation/
├── VALIDATION_EXAMPLES.md       # human-readable specification (this file)
├── cases.yaml                   # future machine-readable case registry
├── jaxcont/                     # Python producers
├── matcont/                     # MATLAB/CL_MatCont producers
├── bifurcationkit/              # pinned Julia producers
├── reference/                   # reviewed reference CSV/JSON artifacts
└── compare/                     # tolerance-aware comparison code and reports
```

Keep expected numbers out of prose once a case is automated: store them in a
versioned CSV/JSON artifact with tool versions, precision, solver settings, and a
source commit. A future `cases.yaml` should drive both local runs and pytest
parameterization. Do **not** commit MATLAB `.mat` files as the only reference;
they are opaque in reviews and can vary by MATLAB release.

## What constitutes validation?

Use independent evidence in this order:

1. **Closed-form theory**, when available. This is stronger than agreement
   between two software packages that might implement the same algorithm.
2. **MatCont/CL_MatCont**, using the installed MatCont 7.6 and the test-run
   taxonomy documented in the August 2019 manual.
3. **BifurcationKit.jl**, preferably a pinned Julia project and a script using
   exactly the same equations and parameters as JaxCont.
4. **A high-accuracy direct calculation**, such as a SciPy root solve or direct
   eigenvalue calculation, for a narrow invariant.

A plot is diagnostic evidence, not a pass criterion. Every automated case must
check at least the residual and one problem-specific invariant.

## Version map

These are capability gates, not release promises. They follow the current
[JaxCont roadmap](../notes/ROADMAP.md).

| Suite | Earliest JaxCont version | Required supported capability | Status now |
|---|---:|---|---|
| V01 | 0.1.x | Equilibrium continuation; fold/Hopf detection; eigenvalue stability | Run now |
| V02 | 0.2.x | Periodic-orbit continuation; Floquet multipliers; LPC/PD/NS detection | Planned |
| V03 | 0.3.x | Branch switching; two-parameter curves; normal forms; codim-2 events | Planned/demand-driven |
| V04 | 0.4+ or external-only | General BVP, homoclinic/heteroclinic, PRC/dPRC | Reference-only unless roadmap changes |

The BifurcationKit documentation is useful as a second, independently developed
reference suite: its [tutorial index](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/tutorials/tutorials/)
groups equilibrium, codimension-2, periodic-orbit, homoclinic, PDE, and symmetry
examples, while its [capability overview](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/capabilities/)
states exactly which equilibrium and periodic-orbit events it supports. Use
pinned scripts, not values copied from a changing documentation build.

## Common output contract

Each producer should write one branch file and one event file.

`branch.csv`:

```text
case_id,tool,tool_version,point,arclength,parameter,state_0,...,residual_norm,stable
```

`events.csv`:

```text
case_id,tool,tool_version,event_index,event_type,parameter,state_0,...,frequency,critical_value
```

Also write `metadata.json` containing:

- equation revision/hash and state/parameter ordering;
- JaxCont/MatCont/BifurcationKit and Python/MATLAB/Julia versions;
- float precision and backend (CPU/GPU);
- continuation algorithm and all step/Newton/eigensolver settings;
- timestamp, Git commit, and whether the artifact is reviewed reference data.

Never compare branches row-by-row: pseudo-arclength implementations choose
different arclength grids. Compare an interpolated observable at common parameter
values on single-valued segments, or compare point sets in scaled `(state, p)`
space. Match events by type and minimum parameter/state distance.

## Default tolerances

Use scale-aware tests, `|a-b| <= atol + rtol*max(|a|,|b|)`. Case-specific
tolerances below override these defaults.

| Quantity | float64 reference | JaxCont float32 | Notes |
|---|---:|---:|---|
| Equilibrium residual, infinity norm | `1e-9` | `2e-5` | Scale residual first for stiff systems |
| Regular branch state | `atol=1e-7`, `rtol=1e-6` | `atol=2e-4`, `rtol=2e-3` | Interpolate away from events |
| Fold/Hopf parameter | `1e-6` | `max(5e-4, 0.25*ds)` | Refined events should beat the raw step |
| Eigenvalue real/imaginary parts | `1e-7` | `2e-4` | Match spectra with an assignment, not sorting by raw complex value |
| Period | `rtol=1e-5` | `rtol=2e-3` | V02+
| Floquet multiplier | `1e-5` | `2e-3` | Remove the trivial multiplier near `+1` first |

All event tests must additionally check the defining local condition: a fold has
one near-zero real eigenvalue; a Hopf has a conjugate pair with near-zero real
part and nonzero imaginary part; PD has a multiplier near `-1`; LPC has a second
multiplier near `+1`; NS has a unit-modulus non-real pair.

---

## V01 — supported by JaxCont 0.1.x

### V01-EQ-001: scalar linear branch (smoke test)

**System:** `f(x,p) = x - p = 0`, start `(x,p)=(-1,-1)`, continue to `p=1`.

**Why:** isolates Newton correction, parameter direction, fixed-size branch
buffers, `valid` mask, and stability without a bifurcation.

**Checks:** every valid point satisfies `x=p`; residual `<2e-6`; Jacobian
eigenvalue is `+1`; the branch is classified unstable for the ODE `xdot=f`.
Run eager, `jax.jit`, `jax.vmap`, CPU, and (when available) GPU variants. This
case needs no external package because the analytic answer is exact.

### V01-EQ-002: cubic S-curve with two folds

**System:**

```math
\dot{x}=r+x-x^3/3.
```

Start at `(x,r)=(-2,-1)` and use pseudo-arclength continuation through the full
connected component. The folds are exactly

```math
(x,r)=(-1,2/3),\qquad (1,-2/3).
```

**Sources:** analytic result; repository examples
[`example_01_pitchfork.py`](../examples/example_01_pitchfork.py) and
[`01_pitchfork.jl`](../examples/BifurcationKit/01_pitchfork.jl); BifurcationKit's
[getting-started example](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/gettingstarted/).

**Checks:** both folds found once and labeled `fold`; event parameter and state
within `5e-4`; maximum scaled residual `<2e-5`; stability changes at each fold;
natural continuation is expected to stop/fail at the turning point while PALC
passes it. The name “pitchfork” in the existing files is historical: this is a
perturbed cubic/S-curve, not the symmetric pitchfork normal form.

**External scripts:** [`run_cubic_fold.m`](matcont/run_cubic_fold.m) and
[`cubic_fold.m`](matcont/systems/cubic_fold.m) export MatCont branch/events CSV.

### V01-EQ-003: Van der Pol equilibrium Hopf

**System:**

```math
\dot{x}=y,\qquad \dot{y}=\mu(1-x^2)y-x.
```

The origin is an equilibrium for every `mu`. Its Jacobian is
`[[0,1],[-1,mu]]`; therefore a Hopf occurs exactly at `mu=0`, with frequency
`omega=1`. Continue from `mu=-2` through `mu=2`.

**Checks:** exactly one Hopf near zero; state norm `<1e-6`; eigenvalues near
`+/- i`; stable for `mu<0`, unstable for `mu>0`. Do not require periodic-orbit
output in V01. See [`example_03_van_der_pol.py`](../examples/example_03_van_der_pol.py).

**External scripts:** [`run_vanderpol_hopf.m`](matcont/run_vanderpol_hopf.m)
and [`vanderpol_hopf.m`](matcont/systems/vanderpol_hopf.m).

### V01-EQ-004: Lorenz-84 equilibrium branch

Use the exact equations, constants, starting equilibrium, and `F` continuation
parameter in [`example_02_lorenz.py`](../examples/example_02_lorenz.py) and
[`02_lorenz84.jl`](../examples/BifurcationKit/02_lorenz84.jl).

Pinned BifurcationKit v0.5.2 reference parameters currently recorded by the
example are approximately `1.546648` (branch/fold), then Hopf points
`1.619658`, `2.467222`, and `2.859876`.

**Checks:** residual and spectral conditions at each matched event; parameter
agreement within `0.005`; no unmatched events. The final condition is an
intentional expected failure for the current detector because the repository
roadmap records duplicate/mislabeled detections. Mark it `xfail(strict=True)`
until the event-protocol rewrite fixes the problem; do not weaken the oracle.

### V01-EQ-005: neural-mass equilibrium branch

Use the three-dimensional Tsodyks–Markram-style model and `E0` continuation in
[`example_05_neural_mass.py`](../examples/example_05_neural_mass.py), paired
with [`03_neural_mass.jl`](../examples/BifurcationKit/03_neural_mass.jl). The
BifurcationKit tutorial describes the same model and its later periodic branch
([neural-mass tutorial](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/tutorials/ode/tutorialsODE/)).

Pinned equilibrium references: folds near `-1.865224` and `-1.463027`; Hopf
points near `-1.850125` and `-1.151059`. A run that does not reach the last
Hopf must report **incomplete coverage**, not pass by silently omitting it.

**Checks:** equilibria and eigenvalues, event location within `0.002`, unique
event matching, and branch-range coverage. Duplicate fold/Hopf reports and the
known spurious fold near `E0=-1.55` are strict expected failures in v0.1.

### V01-EQ-006: MatCont Bratu four-point discretization (partial)

Use MatCont manual section 6.5 and the installed `Testruns/testbratu.m`. The
manual case demonstrates equilibrium continuation, a user function, a branch
point, and switching to a secondary equilibrium branch.

**V01 scope:** compare the primary equilibrium branch and residuals only.
JaxCont 0.1 has no supported branch-point locator/switcher, so BP labeling and
the secondary branch are V03 requirements. Keep those assertions as strict
`xfail` tests rather than pretending the entire MatCont example is supported.

### V01-JAX-001: batched imperfect-pitchfork sweep

Run [`example_06_vmap_sweep.py`](../examples/example_06_vmap_sweep.py). Validate
each batch element against the analytic/serial result, including `n_valid` and
the `Branch.valid` mask. Check permutation invariance of batch order. This is a
JaxCont differentiation/vectorization validation; MatCont is not the right
oracle because it does not expose an equivalent batched transformation.

### V01-JAX-002: differentiable fold inverse design

Run [`example_07_differentiable.py`](../examples/example_07_differentiable.py).
Compare `jax.grad(fold_parameter)` with the closed-form derivative and a
centered float64 finite difference. Test scalar and vector design parameters.
This must be separate from continuation-branch agreement: a correct fold value
does not prove a correct custom VJP.

---

## V02 — periodic orbits and Floquet data

These cases stay skipped until the periodic API, collocation residual, phase
condition, Floquet calculation, and cycle-event detectors are supported and
public. BifurcationKit documents collocation, shooting, and trapezoidal routes
from Hopf points to cycles in its
[periodic branch-switching guide](https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/abs-from-hopf/).

### V02-PO-001: Van der Pol cycle from Hopf

Start the periodic branch at the `mu=0` Hopf from V01-EQ-003. Compare JaxCont
collocation against MatCont `init_H_LC`/`limitcycle` and BifurcationKit
collocation. Check the BVP residual, phase condition, period, extrema, and
nontrivial Floquet multipliers. Near Hopf, verify `T -> 2*pi`.

### V02-PO-002: neural-mass cycles from Hopf

Continue the cycle born from V01-EQ-005. Compare period and the min/max of `E`
at common `E0` values, then compare nontrivial Floquet multipliers. This is the
natural V02 extension of the existing BifurcationKit neural-mass example.

### V02-PO-003: MatCont adaptive-control cycle and period doubling

Use manual sections 7.7 and 8.3.5 and installed test runs `testadapt.m`,
`testadapt1.m`, and `testadapt2.m`. The chain is:

```text
equilibrium -> Hopf -> limit-cycle branch -> LPC/PD -> doubled cycle
```

Gate the stages independently so a PD-detector failure does not hide a correct
cycle branch. Match period, cycle extrema, critical multipliers, and event
parameters. Continuation of the **PD curve in two parameters** (`testadapt3.m`)
belongs to V03, not V02.

### V02-PO-004: fast Morris–Lecar cycles

Use manual section 8.4.4 and installed `testEquilMLfast.m`/`testLCMLfast.m`.
V02 covers the equilibrium-to-cycle chain and detection of the fold of cycles
(LPC). The two-parameter LPC curve from `testLPCMLfast.m` belongs to V03.

### V02-PO-005: periodic predator–prey or Brusselator

Adopt one maintained BifurcationKit tutorial only after reproducing it in a
pinned Julia environment. Prefer the periodic predator–prey tutorial for
PD/NS and shooting-vs-collocation comparisons. The one-dimensional
Brusselator is useful later for large discretized systems but should not be the
first periodic validation because spatial discretization errors complicate the
oracle.

---

## V03 — branches, two parameters, normal forms, codimension two

### V03-BP-001: symmetric pitchfork branch switching

Use `xdot = mu*x - x^3`. Continue the trivial equilibrium through `(0,0)`,
locate the branch point, and switch to both nontrivial branches
`x=+/-sqrt(mu)`. This analytic case precedes Bratu. Compare with
BifurcationKit's [equilibrium branch-switching examples](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/abs-from-eq/).

### V03-BP-002: MatCont Bratu branch switching

Complete the V01-EQ-006 expected failures using manual section 6.5 and
`testbratu.m`/`testbratu2.m`: BP location, tangent of the switched branch, both
orientations, and branch-set agreement.

### V03-C2-001: catalytic oscillator fold/Hopf curves

Use manual sections 8.1.5 and 8.2.5 and installed
`testequilcataloscill.m`, `testLPcataloscill.m`, and
`testLPHopfcataloscill.m`. Validate one-parameter equilibria first, then the
two-parameter LP/H curves, and only then BT/cusp/GH classifications and normal
form coefficients.

### V03-C2-002: extended Lorenz-84 codimension-two diagram

Extend V01-EQ-004 to fold/Hopf continuation in two parameters and compare
against the maintained BifurcationKit Lorenz-84 tutorial. This is a strong
cross-tool case because the same vector field already has a validated V01
equilibrium branch.

### V03-C2-003: cycle bifurcation curves

Promote V02 events to two-parameter curves: PD (`testadapt3.m`), LPC
(`testLPCMLfast.m`), and NS (`testtorBPC3.m`). First compare curve geometry;
then validate codim-2 labels and critical multipliers. Do not make normal-form
coefficient agreement a blocker until derivative conventions and eigenvector
normalizations are documented for all three tools.

---

## V04/reference-only — intentionally outside the current roadmap

Keep these as a catalog, not skipped tests that make the normal CI suite noisy:

| MatCont source | Capability | Why deferred |
|---|---|---|
| Manual section 10.5, `testmyml.m`, `homoc1.m` | Homoclinic continuation | Separate numerical specialization; roadmap says external-only |
| Manual section 7.8, `testadaptPRC.m` | PRC and derivative | Valuable but self-contained and not requested by current users |
| Appendix B, `testbrusselator.m` | General BVP/PDE equilibrium continuation | Requires a supported general BVP discretization API |
| `testtorBPC4.m` through `testtorBPC7.m` | BPC detection/switching/three-parameter continuation | Beyond the proposed V03 minimum |
| GUI sessions | Interactive GUI reproduction | Conflicts with JaxCont's script/notebook focus |

If demand moves one of these into scope, assign it a real release gate and add
a small analytic precursor before importing the full MatCont case.

## Running the initial MatCont producers

The included scripts discover MatCont from the `MATCONT_ROOT` environment
variable and otherwise use this machine's current install at
`/home/ziaee/prog/MatCont/MatCont7p6`.

```bash
cd validation/matcont
MATCONT_ROOT=/home/ziaee/prog/MatCont/MatCont7p6 \
  /home/ziaee/prog/Matlab/R2020a/bin/matlab -batch "run_cubic_fold"

MATCONT_ROOT=/home/ziaee/prog/MatCont/MatCont7p6 \
  /home/ziaee/prog/Matlab/R2020a/bin/matlab -batch "run_vanderpol_hopf"
```

Outputs go to `validation/reference/generated/` and are intentionally ignored
until reviewed. To promote an artifact to a reference, verify its residuals,
record the MatCont/MATLAB versions in metadata, and commit a normalized text
copy. Never overwrite reviewed reference data automatically in CI.

## CI policy

- **Per commit:** analytic V01 cases, existing unit tests, and comparisons to
  small reviewed text references; CPU float32 and CPU float64 where enabled.
- **Nightly/manual:** MATLAB/MatCont and Julia regeneration, full examples,
  accelerator tests, and branch-set comparisons.
- **Release gate:** no unexpected `xfail -> pass` (the test must be reviewed and
  unmarked), no new unmatched events, and all artifacts identify exact tool
  versions.
- Keep performance benchmarks separate from numerical correctness. Warm-up,
  compilation, and device transfer must be reported separately for JAX.

## Implementation order

1. Automate V01-EQ-001 through V01-EQ-003 with analytic assertions.
2. Normalize the existing BifurcationKit scripts into the common CSV contract.
3. Fix JaxCont event de-duplication, then enable strict Lorenz-84/neural-mass
   event matching.
4. Add Bratu primary-branch data as a reviewed MatCont reference.
5. When V02 starts, implement Van der Pol periodic continuation first; only
   after it passes should the neural-mass and adaptive-control chains become
   release gates.

