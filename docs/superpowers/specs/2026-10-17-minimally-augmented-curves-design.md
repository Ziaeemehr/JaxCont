# Minimally Augmented Fold and Hopf Curves — Design Spec

**Status:** Proposed; documentation only, against `61e0a6c`.
**Scope:** Phase 6.2; review M4/M10. Reduce continuation unknowns without changing direct point solvers.

## Problem

`src/jaxcont/bifurcations/curves.py:157` continues a `2n+1` fold state;
`:262` continues a `3n+2` Hopf state. Dense differentiation/solves include all
null-vector unknowns. `HopfCurveArgs` and its accepted-step anchor update
already exist (`:173–193`), as does the engine's shape-preserving
`_apply_step_update` (`src/jaxcont/core/scan_continuation.py:243`). Codim-2
`_decode` infers dimensions from those old layouts
(`src/jaxcont/bifurcations/codim2_events.py:63`); visualization does likewise
(`src/jaxcont/viz/two_parameter.py:28`). They must migrate together.

## Chosen design and alternatives

For a fold, freeze real borders `b,c` within a corrector and solve
`[[A,b],[c.T,0]] [v,g] = [0,1]`, where `A=f_u`.
Continue `X=(u,p_fixed)` with residual `(f,g)`, dimension `n+1`.
For Hopf, use complex borders and `A-i*omega*I` in the same construction,
with bottom row `c.conj().T`; continue `X=(u,p_fixed,omega)` and residual
`(f,Re(g),Im(g))`, dimension `n+2`. This is a positive-frequency chart.
A complex system has one scalar complex constraint, not two unrelated
real eigenvalue selections. The bordered approach follows the
[BifurcationKit formulation](https://github.com/bifurcationkit/BifurcationKitDocs.jl/blob/main/docs/src/codim2Continuation.md).

Initialize right/left borders from the refined critical right/left modes,
with unit norms. In `step_update`, use right and adjoint bordered solves
at the accepted point to obtain new normalized borders; align signs/phases
with the old borders. Stop gradients through borders, but differentiate
through the bordered residual solve. Rejected attempts leave borders unchanged.
Evaluate borders at the accepted point before computing the next tangent so
the tangent belongs to the new residual chart; recompute tangent after update
in both engines if necessary. The root set is independent of a valid border.

Rejected: determinants (scaling/overflow), recomputing eigenvectors inside
every residual (nonsmooth ordering/derivatives), and a squared-Jacobian
single-border Hopf test (the Hopf nullspace is two-dimensional over the reals).
Keep the fully extended formulation as an explicit oracle and the foundation
for the BT chart specified separately; minimal complex Hopf cannot cross BT.

## API and invariants

Add factory keyword `formulation='minimal'|'extended'`, default `'minimal'`.
The existing extended representation and `unpack_*_curve` remain explicitly
extended-only helpers. Add `unpack_minimal_fold_curve(X,n)->(u,p_fixed)` and
`unpack_minimal_hopf_curve(X,n)->(u,p_fixed,omega)`.
Add `FoldCurveArgs(b,c,user_args)`; extend `HopfCurveArgs` with `b,c` and
static `formulation,n`, preserving `q1_anchor,q2_anchor,user_args` for extended
mode. Fold args also hold static `formulation,n`. Border fields always have
fixed shapes/dtypes per problem. User RHS receives only `user_args`.

Add static `BifProblem.curve_layout` and `Branch.curve_layout`, default `None`,
registered as pytree metadata. `CurveLayout(kind,formulation,n,free)` is a frozen
hashable descriptor. Add `decode_curve_state(X,q,layout,raw_f,user_args)` returning
original `u,p` and a `CurveModes` pytree (`v`, `w`, `q1`, `q2`, `omega`, `kappa`,
`oscillatory`, with unavailable vector fields `None`). Minimal vector modes
are reconstructed with bordered solves using pointwise eigenvector seeds;
these are diagnostic inputs, not new Newton unknowns. `w` is the real BT
Jordan-chain vector, `None` in non-BT layouts; BT always returns its stored
`w`, including at zero/negative kappa, never `q2/omega`.
Before fold coefficient scans, normalize each reconstructed `v` and align its
sign by a branchwise `lax.scan`: first valid vector has its largest-magnitude
component positive (lowest-index tie); later vectors choose the sign with
nonnegative dot product against the previous valid aligned vector. Near-zero
alignment (absolute dot <= `default_tol(dtype)`) breaks coefficient eligibility
on that adjacent bracket rather than inventing an orientation. Invalid/padded
slots do not update carry and cannot bracket; each vmapped branch has independent
carry. Apply the same orientation pass to eager and array CP scanning, including
extended states. Normalize Hopf `q=q1+i*q2` to unit Hermitian norm before the
Lyapunov-coefficient interface; align phase to the previous valid mode when
available. These are diagnostic transformations only. Events receive layout
through scan input / eager `BranchPoint.curve_layout`; plotters use branch
metadata. Never guess dimensions from packed lengths. Layout is required
for curve decoding; it remains optional on ordinary equilibrium points.

Seed refinement still uses `fold_point`/`hopf_point` with `(user_args,q0)`
as dynamic solve data, retaining the current closure fix at `curves.py:135`.
A failed seed is never made valid. Borders must couple to critical left/right
modes; nonfinite bordered solves propagate failure, never a fake zero.
Documentation must describe the state/API break and the explicit extended opt-in.

## Verification

`tests/test_minimal_curves.py`:
- `test_minimal_fold_shifted_cusp`: `x=u-0.7`,
  `f=(p0-.3)+(p1-1.2)*x-x**3`; seed `(u,p)=(1.7,[-1.7,4.2])`,
  assert state width `2`, `27*(p0-.3)**2=4*(p1-1.2)**3`, scaled error
  below `100*default_tol(dtype)` over `p1=4.2..5.2`.
- `test_minimal_hopf_shifted_parabola`: equilibrium `(.5,-.3)`,
  damping `p0+p1**2-2`, frequency `1`; width `4`, `p0=2-p1**2`,
  `omega=1`, maximum errors `<100*default_tol(dtype)`.
- `test_minimal_and_extended_agree_at_fixed_parameters`: compare both
  formulations on natural samples `p1=[0,.25,.5,.75,1]`, same tolerance.
- `test_border_refresh_preserves_root_and_reject_keeps_args`: multiply
  valid borders by `2` and `-3`; roots unchanged; rejected engine update
  leaves every array leaf bitwise unchanged.
- `test_minimal_curve_decoding_events_and_plotting`: decode `n=2` Hopf
  state of length `4` as `n=2`, never as extended `n=0`; plotted solved
  parameters are `[2,1.9375,1.75,1.4375,1]` on the grid above.
- `test_minimal_curve_seed_q_sensitivity`: the parabola seed has
  `d(p_fixed)/dq=-2*q`, exactly `-1` at `q=.5`, tolerance
  `100*default_tol(dtype)`; `jit`/`vmap` factories and `loop='scan'`
  continuation retain finite derivatives away from degeneracies.

- `test_fold_diagnostic_orientation_no_false_cp`: alternate raw eigenvector
  seed signs on a fold segment with nonzero coefficient; after alignment no CP
  bracket appears. Include a genuine coefficient crossing, invalid interior
  samples and trailing padding, and independent vmapped branches with opposite
  initial raw signs; zero ownership and valid crossing counts remain correct.
- `test_hopf_diagnostic_unit_norm`: rescale/phase-rotate raw seeds and assert
  unit decoded coefficient-input norms and invariant Lyapunov coefficients.

## Migration ownership

Task 6.2a introduces mandatory extended layout metadata and the shared extended
diagnostic orientation/normalization pass before any array curve scanner. Its
layout tests already cover alternating raw signs/padding and unit Hopf modes;
6.1b reuses that pass, and 6.2b/c add minimal-formulation coverage. It migrates manually constructed curve points in `test_codim2_events.py`,
`test_curves.py` and `test_viz.py` at the same checkpoint. Task 6.2b switches the
fold default and makes legacy fold consumers explicitly extended; Task 6.2c
owns the equivalent Hopf migration. Each task lists and runs all three regression
modules, with minimal-default integration coverage added at the corresponding
switch. No packed-length inference is allowed during migration.
