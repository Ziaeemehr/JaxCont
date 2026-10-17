# Accurate Floquet Multipliers by Periodic Schur — Design Spec

**Status:** Proposed; numerical prototype is an implementation acceptance gate.
**Scope:** Phase 6.3; review M1/M5/M10. Preserve collocation interval maps.

## Problem

`src/jaxcont/core/collocation.py:147–149` multiplies all interval propagators
into one matrix. `src/jaxcont/stability/floquet.py:52–60` eigendecomposes that
product twice, once for flow alignment and once through `EigenSolver`.
Small physical multipliers disappear relative to large ones before either
solver runs. x64 postpones this loss but does not remove it. PRC also forms
a product (`src/jaxcont/stability/prc.py:60`); its scaling redesign belongs
to Phase 7, not this change.

## Chosen design and alternatives

Implement a JAX-native complex periodic Schur decomposition of the ordered
factors `M_i` returned by `interval_propagators`. Compute unitary `Q_i` and
upper triangular `T_i` with `M_i Q_i = Q_(i+1) T_i`, cyclically
`Q_m=Q_0`. Use periodic Hessenberg–triangular reduction and shifted periodic
QR with deflation, following the factor-preserving algorithmic structure
of [SLICOT MB03WD](https://www.slicot.org/objects/software/shared/doc/MB03WD.html).
This is an algorithm reference, not a new Fortran dependency or a claim that
ordinary QR on a product is equivalent. MB03WD documents a real algorithm;
the complex scaled implementation and its derivative require the independent
prototype gates below, rather than being established by that reference.

Return `phase_j=product(T_i[j,j]/abs(T_i[j,j]))` and
`logabs_j=sum(log(abs(T_i[j,j])))`; zero factors give phase zero and logabs
`-inf`. Do not form a product even for shift selection: use scaled trailing
small blocks. Represent shifts with mantissa/exponent and apply periodic
bulge chasing to factors. Use a static sweep budget `max_sweeps=200*n`,
dtype-scaled deflation, and return an explicit convergence boolean.
Only the sole Hessenberg factor `H_(m-1)` has subdiagonals; the other factors
are triangular. Delete its subdiagonal `h=h[k+1,k]` only when two tests hold.
Local test: after scaling `h`, `h[k,k]` and `h[k+1,k+1]` by their maximum
magnitude (to avoid under/overflow), `abs(h)<=100*eps*(abs(h[k,k])+abs(h[k+1,k+1]))`.
If all three are zero it passes; if both diagonals are zero but `h` is nonzero
it fails. Cyclic test: accumulate the `2x2` product block `[[a,b],[c,d]]` of
rows `k,k+1` from every factor's diagonal block, renormalizing by the largest
entry after each factor (the test is homogeneous, so this is exact). `c` is `h`
times the triangular diagonals at `k`, and `b` carries every factor's coupling.
Deleting `h` moves the block's eigenvalues by about
`abs(c*b)/max(abs(a-d),sqrt(abs(c*b)))`, which must be at most
`100*eps*max(abs(a),abs(d))`. A coupling that is small in its own factor but
meets a reciprocally large coupling in another factor therefore stays (loop-5
audit). Only the `2x2` diagonal blocks enter, as in the LAPACK Ahues–Tisseur
test; contributions through the subdiagonal above, `h[k,k-1]`, are ignored.
Never use an absolute unit-scale floor that would erase tiny physical modes.
Apply both tests to every subdiagonal after the Hessenberg reduction and after
each sweep, and verify cyclic backward error. Test cyclic reconstruction before
accepting values. Spectrum derivatives are published only for modes whose local
multipliers through the true factors reproduce `phase*exp(logabs)` within
`500*eps*(m+sum_i abs(log(abs(d_i))))`; other modes get a NaN tangent.

Shift heuristics (deliberate extension beyond MB03WD's plain Wilkinson
shift). The shift comes from the scaled `2x2` product block at rows `hi-1,hi`.
It is the block eigenvalue nearest the block's last diagonal entry
(Wilkinson), except that it is zero when the other block eigenvalue is smaller
in modulus by more than `1/_ORDER_RATIO` (`_ORDER_RATIO=1e-2`). That zero shift
is a periodic orthogonal-iteration step: a block can deflate in factored form
only when no much smaller multiplier sits above a much larger one. Exceptional
shifts: on the 10th, 20th, ... consecutive sweep with the same bottom row `hi`
of the active window, the shift is alternately zero (first) and the scaled
block's last diagonal entry plus `0.75` times its subdiagonal modulus.

Rejected: re-integrating an IVP (different discretization), scalar rescaling
of the whole product (still loses relative modes), unshifted orthogonal
iteration (fails to separate equal-modulus conjugate modes), and a full
`m*n` cyclic pencil (large memory plus unavailable native generalized-QZ
path). Cyclic reduction is viable but requires equally careful scaling;
periodic Schur exposes the phase-zero basis needed for trivial-mode identity.

## API, derivatives, and invariants

Add `PeriodicSchur(max_sweeps=None)` and `ProductFloquet(eigen_solver=DenseEigen())`
strategies. Retain `Solvers.floquet=ProductFloquet()` through the prototype;
change it to `PeriodicSchur()` only at the separate acceptance decision in 6.3d
after all 6.3a–c gates pass in both precisions. The existing
`Solvers.eigen` continues to serve equilibrium spectra. Explicit product mode
retains custom matrix eigensolvers and their ordering. Replace the positional
`eigen_solver` argument on both Floquet functions with `floquet_solver`;
update all callers, including PD/NS refinement. Add optional
`floquet_solver=None` fields to PD/NS events: an explicit event strategy wins,
otherwise continuation/refinement orchestration binds `Solvers.floquet`. The
eager candidate-refinement helper takes the same `solvers` bundle.
Detection computes/caches `FloquetSpectrum` per distinct effective static
strategy (event override or bundle default), including its own trivial index,
convergence and logabs. Event input uses that strategy's spectrum, never the
branch's shared spectrum for an overriding event. The branch stability fields
still use the bundle strategy; candidate refinement rebinds the same effective
event strategy. Failures mask only events using the failed strategy. Keep `(values,trivial_index)`
returns. Add `floquet_spectrum(...) -> FloquetSpectrum(values,phase,logabs,
trivial_index,converged)`. `Branch` gains optional `floquet_logabs` and
`floquet_converged` array fields, included in pytree registration.

Find the trivial mode using reconstructed phase-zero right eigenvectors,
via triangular periodic back substitution, normalized and aligned with
`raw_f(x0,p,PeriodicArgs.user_args)`. Schur vectors alone are not right
eigenvectors. Preserve the resulting index in the returned ordering; never
use distance to `+1`. Stability uses nontrivial `logabs<0`; nonconvergence
sets stable false and all event eligibility false. Eager public evaluation
raises `RuntimeError` on nonconvergence; traced evaluation returns NaN values,
index `-1`, and false status. Padding is replaced by a valid seed before
spectral evaluation and masked afterward. A zero flow is invalid, not index 0.

Representable multipliers equal `phase*exp(logabs)`; under/overflow may yield
zero/infinity in values, while logabs remains usable. PD/NS only use finite
near-unit values. Differentiate simple finite multipliers with a custom JVP:
propagate scaled periodic left/right eigenvectors and sum factor perturbation
contributions, using `l_i^H r_i=1` locally; never differentiate QR iteration
or discrete deflation. The transposed JVP supports reverse mode. Multiple or
defective multipliers have no promised derivative. `loop='scan'` does not
remove this spectral limitation. The prototype must establish eigenvector
and JVP accuracy before switching the default.

## Verification

New `tests/test_periodic_schur.py`, with exact-factor tolerance
`err=200*default_tol(dtype)` for log/phase checks. Spectrum comparisons are
permutation-independent, matching isolated modes before derivative checks.
The `100*eps` deflation and `500*eps` backward-error checks are explicit
machine-epsilon algorithmic exceptions to the default_tol policy, not user
nonlinear solve tolerances; physical discretization bounds remain separate:
- `test_periodic_schur_rotating_scaled_factors`: `m=32`,
  `M_i=Q_(i+1) diag(exp(-2),1,exp(2)) Q_i.T`, `Q_i` rotations in the
  first/third plane by `2*pi*i/32`. Expect logabs `[-64,0,64]`, phases
  all `1`, each log error `<err`, and nonzero smallest value. Reconstruction
  relative error `<500*eps`; zero is not an acceptable stable multiplier.
- `test_periodic_schur_complex_equal_modulus`: `16` identical blocks
  `0.9*rotation(.2)`, expect `16*log(.9)` and phases `exp(+-3.2j)`.
- `test_periodic_schur_zero_and_budget_status`: one diagonal factor
  `diag(0,1,2)` gives `[-inf,0,log(2)]`; zero sweeps on a nontriangular
  three-dimensional fixture reports false, never stable true.
- `test_periodic_schur_jvp_log_multiplier`: replace `-2` above with `-a`,
  derivative of contracting logabs is exactly `-32`; derivative of its
  value is `-32*exp(-32*a)` at `a=1`, relative error `<20*err`.

`tests/test_floquet.py::test_circle_log_multipliers_mesh_convergence` uses
`r'=r*(rho-r*r), theta'=1`, `rho=1`, meshes `(ntst,ncol)=(20,4),(40,4)`:
expected `{1,exp(-4*pi)}`, logabs `{-12.566370614359172,0}`; fine-mesh
log error `<2e-3` and not worse than coarse error plus `500*eps`.
`test_periodic_schur_flow_index_with_near_unit_physical_mode` supplies flow
`e0` and diagonal factors with total values `[.9999,1.00001]`; index `0`,
stable false. `test_periodic_schur_branch_jit_vmap_and_pd_ns` checks branch
shapes and existing PD/NS analytic fixtures in both precisions. Test the
explicit product custom-order path separately, preserving its old gradient tests.

- `test_periodic_schur_nonnormal_complex_derivatives` (mandatory before default):
  use four factors `M_i(a)=S_(i+1)(a)*D(a)*inv(S_i(a))`, cyclic `S_4=S_0`,
  `S_i(a)=[[1, .4+.1*i+a*(.2+.05j)],[.1j,1+.05*i]]`,
  `D(a)=diag(exp(-.3*a+1j*.2*a),exp(.1*a-1j*.4*a))`, `a=.7`.
  Multipliers are exactly `exp(4*(-.3+.2j)*a)` and
  `exp(4*(.1-.4j)*a)`; logabs derivatives `[-1.2,.4]`, phase derivatives
  `[.8j*phase[0],-1.6j*phase[1]]`, and value derivatives follow their
  analytic products. Test JVP and reverse gradients of real/imaginary scalar
  objectives against these formulas to `<20*err` relative scaled error.
  At every phase verify right `M_i*r_i=r_(i+1)*d_i` and left
  `M_i^H*l_(i+1)=conj(d_i)*l_i` residuals `<500*eps` relative to norms,
  finite nonzero local scaling and `l_i^H*r_i=1` to `<500*eps`.
  Here `d_i` are reconstructed local multipliers with product equal to the
  matched global multiplier, not just the Schur diagonal for nonnormal modes.
- `test_periodic_schur_mixed_status_batch`: jit/vmap convergent diagonal and
  nontriangular zero-budget lanes; converged lane retains its spectrum and
  gradient, failed lane reports false. In 6.3d integration, require NaN public
  values, index `-1`, false
  stability and no events. Safe surrogate inputs must keep masked failed-lane
  gradients finite (zero); eager public failure raises `RuntimeError`, while
  internal orchestration consumes status without Python branching on arrays.
- `test_periodic_schur_zero_underflow_overflow`: include an exactly zero factor
  (phase zero, logabs `-inf`), and 32 diagonal factors with exponents `+-30`
  (finite logabs `+-960`, values zero/infinity at representability limits).
  No NaN phase/logabs for converged lanes; underflowed zero differs from an
  exact-zero multiplier in logabs. Ineligible PD/NS values never bracket.
- `test_floquet_event_override_detection_and_refinement`: configure branch
  `ProductFloquet` and explicit PD/NS `PeriodicSchur` overrides, then reverse
  them. Instrument distinct strategies with distinguishable ordering and
  trivial indices on analytic crossing fixtures. Assert scan calls, tracked
  spectra, raw-column exclusions, candidate counts and refinement calls all
  use each event's override; unspecified events use the bundle. A failure in
  one effective strategy masks only its events. Shared effective strategies
  are computed once per point; branch reporting follows its own strategy.
