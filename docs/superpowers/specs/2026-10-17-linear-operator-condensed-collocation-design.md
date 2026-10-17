# Linear Operators and Condensed Collocation — Design Spec

**Status:** Proposed, spec only. No Phase 7 implementation tasks are authorized by this plan.
**Scope:** Review M10 and Phase 7 of the remediation plan; prerequisite for large BVP work.

## Problem

`src/jaxcont/solvers/protocols.py:24` requires a dense matrix.
`src/jaxcont/core/scan_continuation.py:126–180` assembles bordered Jacobians;
`src/jaxcont/solvers/implicit.py:62–63,91–92` bypasses the solver in primal
and adjoint. `src/jaxcont/core/collocation.py:132` directly solves interval
systems; `src/jaxcont/stability/prc.py:60–70` uses a product and a dense
bordered solve. A periodic residual has
`N=ntst*(1+ncol)*n+1` unknowns in its current layout, including independent
mesh states, collocation states, and period. With `(100,4,10)`, N is `5001`.
A JVP facade over an already assembled `N*N` matrix would not solve this problem.

## Chosen design and alternatives

Replace the linear protocol with
`LinearSolver.__call__(matvec,b,*,diag_blocks=None)->Array`.
`matvec` is a pure linear callable on vectors `(N,)`; `b` may be `(N,)`
or `(N,r)`. `diag_blocks` is optional structured acceleration metadata,
not permission to discard off-diagonal couplings. `None` means a generic
operator. A `BlockDiagonal` pytree holds preconditioner blocks; a
`CollocationBlocks` pytree additionally holds local interior factors,
endpoint couplings, global border rows/columns, permutations, and a static
mesh descriptor. Both supply diagonal blocks; only the latter permits exact
condensation. Do not pass anonymous arrays whose meaning depends on a caller.

`Dense` materializes columns with `vmap(matvec)(eye(N)).T` as the explicit
small-system fallback. `CondensedCollocation` requires `CollocationBlocks`;
missing/incompatible structure raises `ValueError`, with no silent dense
allocation. Ordinary iterative solvers may use just block preconditioning;
adding GMRES is outside this first seam proposal.

For each interval linearize defect equations as
`D_i dz_i + L_i dx_i + H_i dgamma = r_i`, where `gamma` contains period
and, for pseudo-arclength, the continued parameter. Factor `D_i` and eliminate
interior `dz_i`. Substitute into continuity, phase and arclength rows to form
a cyclic mesh system plus a small global border. Solve that system by
pivoted block cyclic elimination, then back-substitute interior states.
Carry the global border throughout elimination; do not invert the singular
unbordered periodic closure and then apply a Schur complement. Preserve
periodic wraparound in forward and transposed solves. Complexity is linear
in ntst for fixed n/ncol and border width, without an `N*N` array.

Rejected: generic sparse assembly as the only seam (still requires a stored
Jacobian and backend sparse support), full monodromy condensation (poor
conditioning on strongly separated cycles), and treating diagonal blocks
as the entire operator (incorrect coupled solve).

## API, differentiation, and invariants

Add `linear_solver=Dense()` to both root primitives and pass it through all
point-solver callers. Operators on complex arrays must be complex-linear;
general nonholomorphic residuals are represented as real vectors of stacked
real/imaginary components, with real-linear operators and the ordinary real
Euclidean pairing. Never pass a non-complex-linear complex callable through
the complex solver contract.

For a complex-linear `matvec(x)=A*x`, JAX's `jax.linear_transpose` (and VJP)
returns `A.T*c` using the bilinear cotangent pairing `c.T*x`, with no implicit
Hermitian conjugation. Newton uses the JVP `A*x`; implicit backward and
`lax.custom_linear_solve`'s `transpose_solve(transpose_matvec,b)` solve
`A.T*x=b`, including structured metadata transposed with `.T`. Do not replace
this callback with `A.conj().T`. If a mathematical Hermitian-adjoint solve is
needed (e.g. complex PRC), adapt explicitly:
`adjoint_matvec(x)=conj(transpose_matvec(conj(x)))`, and obtain the
`A^H` solution as `conj(solve_A_T(conj(b)))`. Conversely adapt a Hermitian
solver to the required transpose using conjugation on its RHS and solution.
Use `lax.custom_linear_solve` with explicit `transpose_solve`, `symmetric=False`,
so factorization internals are not differentiated. Transposed condensed
metadata must preserve this exact pairing and cyclic border convention.

Engine solver strategies stay static/hashable; operator captures and block
values stay dynamic. Problem-specific structure is supplied through an
optional static `BifProblem.linearize(u,p,args)` hook returning an operator
and metadata; generic residuals use `jax.linearize`. Border adapters add the
engine's tangent/parameter constraints. Eager and `loop='scan'` paths share
this seam. `step_update` is frozen within each Newton solve; `PeriodicArgs`
and `HopfCurveArgs.user_args` remain dynamic and unchanged in meaning.

Route interval propagator solves (including periodic Schur's local maps)
and PRC bordered/adjoint solves through the seam. For PRC use a cyclic
adjoint BVP with phase normalization, avoiding its product. `EigenSolver`
and `PeriodicSchur` are spectral protocols, not linear solvers; only their
linear subproblems use this seam. `trivial_index` retains flow identity.

A solver returns the same shaped solution and caller dtype. Callers check
relative linear residual and propagate existing failure statuses. No source
precision override, no finite differences, and no dense fallback in the
condensed path. Arbitrary singular local interior blocks may require a
larger pivot window; until supported, report failure rather than produce an
unchecked step. Failure handling and factorization workspace need a future
implementation plan before any coding.

## Acceptance tests for the later Phase 7 plan

- `test_operator_dense_vector_and_multiple_rhs`: matrix `[[4,1],[2,3]]`,
  RHS `[1,2]`, solution `[.1,.6]`; multiple RHS identity gives inverse
  `[[.3,-.1],[-.2,.4]]`, error `<100*default_tol(dtype)`.
- `test_root_operator_adjoint`: `G(x,a)=A*x-a*[1,2]`, objective `sum(x)`;
  exact derivative `.7`. A spy solver observes primal and transpose calls.
- `test_cyclic_condensation_keeps_global_border`: three scalar mesh blocks
  `x1-x0=1`, `x2-x1=1`, `x0-x2+gamma=-2`, phase `sum(x)=0` have exact
  solution `[-1,0,1,0]` for `(x0,x1,x2,gamma)`. Test forward and transpose
  against the full four-by-four matrix; closure alone is singular.
- `test_condensed_periodic_newton_matches_dense`: shifted circle
  `(x-.5,y+.3)`, `r'=r*(1-r*r), theta'=1`, mesh `(10,4)`;
  Newton increments and JVP/VJP agree with dense to `100*default_tol(dtype)`
  times `max(1,norm(reference))`; period `2*pi` to collocation error `<1e-3`.
- `test_condensed_prc_normalization`: on that circle,
  `Z=(-sin(t),cos(t))`, `Z*f=1`; mesh `(40,4)`, error `<2e-3`.
- `test_condensed_large_mesh_never_materializes_global_jacobian`: `(100,4,10)`
  uses N=`5001`, no intermediate shape `(5001,5001)` in the traced jaxpr;
  solver fails explicitly on missing metadata. Check transpose memory too.

All acceptance cases run under both precision modes and `jit`/`vmap`;
root gradients use implicit differentiation, branch reverse mode uses
`loop='scan'`. These are future acceptance criteria, not tasks in the Phase 6 plan.

- `test_complex_operator_transpose_convention`: use nonsymmetric
  `A=[[2+1j,1-2j],[3+.5j,4-1j]]`, `b=[1+2j,-.5+1j]`.
  Verify `jax.linear_transpose(matvec,zeros_like(b))(b)[0] == A.T@b`,
  transpose solves against `solve(A.T,b)`, and explicit Hermitian adapter
  against `solve(A.conj().T,b)`; these reference answers must differ.
  Check bilinear identity `c.T@(A@x)==(A.T@c).T@x` for complex `c,x`.
  For `G(x,a)=A@x-a*b`, real scalar `a`, and real objective
  `Re(c.T@x)` with `c=[.3-.7j,1+.2j]`, compare JVP and reverse `grad`
  to `Re(c.T@solve(A,b))` within `100*default_tol(dtype)`. Spy on the
  custom-linear-solve transpose callback and structured transpose path;
  also compare stacked-real representation for the same objective. Run both
  precisions under jit/vmap. A separate nonholomorphic test uses stacked-real
  input for `A@z+.1*conj(z)` and verifies real transpose/gradient agreement.
