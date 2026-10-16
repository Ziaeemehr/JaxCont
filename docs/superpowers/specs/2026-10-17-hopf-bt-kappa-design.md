# Hopf Curves Through BT with Kappa — Design Spec

**Status:** Proposed; the separate BT chart is a deliberate size/regularity tradeoff.
**Scope:** Phase 6.4; review Phase 6 omega limitation. Depends on curve-layout decoding in 6.2.

## Problem

`src/jaxcont/bifurcations/curves.py:196–202` stores signed `omega` in a
complex eigenvector chart. At BT its real/imaginary eigenvectors coalesce;
changing the sign of frequency does not create the real neutral-saddle
segment. `src/jaxcont/bifurcations/codim2_events.py:249–266` instead detects
an omega sign or records `curve_terminated`, which is not a located BT.
The moving `HopfCurveArgs` anchor at `curves.py:187` fixes phase drift but
cannot fix the singular chart.

## Chosen design and alternatives

Add a real regular chart with state `X=(u,p_fixed,v,w,kappa)` and equations
`f=0`, `A*v+kappa*w=0`, `A*w-v=0`, `v.T*v-1=0`,
`v_anchor.T*w=0`. There are `3n+2` real equations/unknowns. At BT,
`A*v=0, A*w=v` is a Jordan chain with nonzero `v`. For `kappa>0`,
`q=v+i*sqrt(kappa)*w` satisfies `A*q=i*sqrt(kappa)*q`.
For `kappa<0` the invariant plane contains the real pair
`+-sqrt(-kappa)`. All residuals remain polynomial in kappa at zero.
Initialize from a positive-frequency direct Hopf seed, rotate its phase to
make real and imaginary parts orthogonal, normalize the real part, and set
`w=Im(q)/omega`, `kappa=omega**2`.

Keep a locally valid fixed `v_anchor`, initialized from the normalized seed
`v` with stopped gradients. For formulation `bt`, `step_update` leaves this
anchor unchanged on accepted and rejected steps, including left-endpoint
refinement setup. The gauge is `v_anchor.T*w=0`; it does not imply `v.T*w=0`.
Changing args alone cannot refresh this chart without moving an accepted root.
Monitor corrector/Jacobian validity using the existing failure statuses; if this
local chart loses regularity, report failure rather than silently reanchor.
Automatic chart switching is outside this design: it would require a coordinated
state transformation, root revalidation and tangent recomputation.

Rejected: `omega=sqrt(abs(kappa))` in the complex equations (wrong neutral
segment and nondifferentiability), `A*A+kappa*I` with one border (rank-two
Hopf nullspace), and using the old imaginary eigenvector at zero (vanishes).
A fully minimal real two-border chart is a possible later optimization;
its two independent scalar constraints need a separately verified chart
selection algorithm. This release favors a verifiable Jordan-chain chart.
Thus 6.2 minimal Hopf remains the small positive-frequency option; through-BT
continuation intentionally opts into a larger representation.

## API and invariants

Extend `hopf_curve_problem(...,formulation='minimal'|'extended'|'bt')`.
Default remains `'minimal'`; `'extended'` retains omega and `'bt'` uses kappa.
Add `unpack_bt_hopf_curve(X,n)->(u,p_fixed,v,w,kappa)` and
`hopf_curve_frequency(kappa)->(omega,oscillatory)`: omega is
`sqrt(max(kappa,0))`, oscillatory is `kappa>0`; the frequency helper is for
reporting only, never the residual. `CurveLayout.formulation='bt'` identifies
the state; never reinterpret old omega buffers as kappa.

Shared decoding reports `CurveModes.kappa` and `oscillatory`; real vectors
`v,w` remain available at/after BT. The `q1,q2` diagnostic arrays keep fixed shapes: use `q1=v`,
`q2=sqrt(max(kappa,0))*w` and require the `oscillatory` mask before treating
them as Hopf eigenvectors. Never switch an array field to `None` based on
a traced kappa value. Original-system spectra exclude the
two pinned modes `+-i*sqrt(kappa)` or `+-sqrt(-kappa)` by distinct column
identity. GH/ZH/HH eligibility requires positive kappa, and normal-form
routines receive safe surrogate inputs before any batched conditional. For
ineligible lanes, select the complete surrogate RHS `f_s(z)=A_s*z`,
`A_s=[[0,-1],[1,0]]` direct-summed with `-2*I` on extra dimensions,
equilibrium zero, frequency `1`, and unit mode `(e0-i*e1)/sqrt(2)`;
all higher derivatives are zero. Select inputs before square roots, divisions,
and linear solves; merely masking an already singular output is forbidden.
Use `kappa_safe=where(eligible,kappa,1)` before square roots and reconstruct
coefficient modes from stored `v,w` using this safe frequency, rather than
backpropagating through reporting-only `sqrt(max(kappa,0))`. Select safe
Jacobian and higher derivative tensors consistently before any coefficient
solve; the surrogate has `A_s` and zero second/third tensors. Normalize eligible
complex modes to unit norm; use the unit surrogate mode for other lanes. Evaluate the selected RHS and
its derivatives consistently, then mask coefficient outputs to zero and mark
ineligible. `lax.cond` may still express this choice, but correctness must hold
when `vmap` lowers it to selection. No derivative flows through the surrogate
selection mask; ineligible lanes have finite zero coefficient gradients.

BT's scalar test is kappa; use the shared zero-ownership kernel. Refinement
still calls `bogdanov_takens_point` on original coordinates and validates
convergence/locality. A sign change alone is only a candidate. For the BT
chart remove the omega-based termination fallback; keep it for explicitly
selected legacy extended/minimal positive-frequency charts. Plots draw the
negative-kappa part dashed and describe it as a neutral-saddle extension.
Initial construction still requires a positive-frequency Hopf seed. This
spec does not promise regularity at higher degeneracies, e.g. a zero Jordan
block of length three or another simultaneous zero mode.

## Verification

Use shifted BT system `x=u0-.7`, `y=u1+.4`,
`f=[y,(p0-.3)+(p1-1.2)*y+x*x+x*y]`. Along the trace-zero locus,
`x=t`, `y=0`, `p1=1.2-t`, `p0=.3-t*t`, `kappa=-2*t`.
The BT is exactly `u=(.7,-.4), p=(.3,1.2)`.
Start `t=-.1`: `u=(.6,-.4), p=(.29,1.3), kappa=.2`;
end `t=.1`: `u=(.8,-.4), p=(.29,1.1), kappa=-.2`.

`tests/test_hopf_bt_curve.py`:
- `test_bt_chart_residual_and_jacobian_at_zero`: at BT use `v=(1,0)`,
  `w=(0,1)` and anchor `v`. Residual zero to `100*default_tol(dtype)`;
  the square fixed-`p1` Jacobian is nonsingular (smallest singular value
  `>0.05`).
- `test_hopf_bt_kappa_crosses_shifted_bt`: natural samples `p1=1.3..1.1`
  in steps `-.01` (`ds=.01`, `adaptive=False`, `max_steps=20`),
  all `21` points valid; state/parameter/kappa errors
  `<100*default_tol(dtype)`; kappa has both signs.
- `test_bt_kappa_event_once_and_validated`: exactly one validated BT,
  both parameter errors `<100*default_tol(dtype)`, no `curve_terminated`.
- `test_bt_chart_masks_hopf_only_events`: no GH/ZH/HH candidates at
  `kappa<=0`, and no NaN from coefficient evaluation there.
- `test_bt_chart_jit_vmap_scan_gradient`: replace `.3` by user scalar `a`;
  with `loop='scan'`, derivative of mean solved `p0` with respect to `a`
  is `1` within `100*default_tol(dtype)`, including samples on both sides.

- `test_bt_fixed_anchor_nonnormal_similarity`: additionally transform the entire
  shifted vector field by `u=c+S(p1)*z`, `c=(.7,-.4)`,
  `S(p1)=[[1,.2+(p1-1.2)],[.3*(p1-1.2),1]]`,
  `f_new(u,p)=S(p1)*f_base(S(p1)^(-1)*(u-c),p)` with base coordinates
  `(x,y)` and no second shift. This invertible, parameter-dependent,
  nonorthogonal transform preserves the analytic parameter/kappa locus.
  Initialize the gauge once at the positive-kappa seed. Across all 21 samples,
  verify transformed states, all five residual blocks, constant anchor leaves,
  and full-rank corrector Jacobians; decoder returns both `v,w` at zero and
  negative kappa. Include the review's exact nilpotent matrix
  `[[.2,1],[-.04,-.2]]`, anchor `e0`, `w=e1/sqrt(1.04)`, `v=A*w`:
  the fixed-anchor residual stays zero; replacing the anchor by `v` produces
  gauge residual `-.1923076923` and must never occur.
- `test_bt_mixed_kappa_coefficient_grad`: `jit(vmap(...))` over
  `kappa=[.2,0,-.2]`, with eligible lane a shifted positive-frequency Hopf
  normal form `f(z)=sqrt(.2)*rotation(pi/2)*z-a*(z.T*z)*z` at `z=0`,
  `a=.7`, differentiating with respect to cubic strength `a` and other lanes the BT fixture. Compare positive-lane coefficients
  and gradients to isolated evaluation. Reverse-mode `grad` of the summed
  masked coefficients is finite; ineligible lanes return zero values and zero
  gradients, with no GH/ZH/HH candidates. Run both precisions.
