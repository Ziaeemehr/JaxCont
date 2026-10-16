# Phase 6 Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement trace-safe detection, minimally augmented curves, accurate Floquet spectra, and a regular Hopf-to-neutral-saddle BT chart.

**Architecture:** Keep the existing fixed-buffer engines, `step_update(args,u,p)` hook, `PeriodicArgs`, and `HopfCurveArgs`. Add array-only detection and explicit curve layout metadata; use bordered minimal systems for ordinary curves, periodic Schur for factor spectra, and a real Jordan-chain chart for BT crossing.

**Tech Stack:** Python ≥ 3.11, JAX ≥ 0.9, equinox, NumPy/SciPy for independent host references, pytest + pytest-xdist.

**Specs:**
- [6.1 Trace-safe event scanning](../specs/2026-10-17-trace-safe-event-scanning-design.md)
- [6.2 Minimally augmented curves](../specs/2026-10-17-minimally-augmented-curves-design.md)
- [6.3 Periodic Schur Floquet](../specs/2026-10-17-periodic-schur-floquet-design.md)
- [6.4 Hopf through BT](../specs/2026-10-17-hopf-bt-kappa-design.md)

**Baseline:** `review-remediation-2026-10`, `61e0a6c`, Phases 0–5 complete. All source/test changes below are future work. This revision is limited to the six requested design/plan Markdown files. Pre-existing constraint/progress edits and `.clinerules` are outside its scope. Do not commit during this revision.

## Global Constraints

- Pre-1.0 policy: remove, don't deprecate. Every breaking change gets a `CHANGELOG.md` entry under `## [0.5.0] - Unreleased` → `### Breaking`.
- Never call `jax.config.update("jax_enable_x64", ...)` at import time inside `jaxcont`; precision is the caller's choice.
- Tolerances derive from dtype through one helper, `jaxcont.utils.precision.default_tol(dtype)`: `1e-9` for float64, `1e-5` for float32. No new hard-coded `1e-5`/`1e-6` literals in `src/`.
- Schur machine-epsilon deflation/backward-error criteria are explicit algorithmic exceptions: `100*eps` and `500*eps`, with scale-safe local comparisons from spec 6.3. Exact-factor acceptance uses `err=200*default_tol(dtype)`; discretization bounds are separate.
- Every test in the default suite must pass under both `JAXCONT_TEST_X64=0` and `JAXCONT_TEST_X64=1`.
- No Python control flow on traced values in the engine or in anything reachable under `jit`/`vmap`.
- New source comments state invariants, not incident history; history goes in `CHANGELOG.md`.
- Run tests with `python -m pytest -q -n 8`; slow suite with `python -m pytest -q -m slow`.

## Review Focus

1. Equal-cost assignments and repeated unit multipliers must retain deterministic column identity (6.1a/6.1b).
2. Invalid branch padding must not feed singular periodic spectral evaluations (6.1c/6.3d).
3. Accepted-step anchor updates change the residual chart; next tangents must use refreshed args, rejected attempts must preserve args (6.2b/6.2c).
4. A Schur basis is not an eigenvector basis; flow identity and multiplier derivatives need independent reconstruction checks (6.3c).
5. Negative kappa is a neutral-saddle segment; Hopf-only coefficients use safe surrogate inputs in ineligible lanes before batched evaluation (6.4b).

## Execution and verification policy

Dependencies: 6.2a → 6.1a → 6.1b → 6.1c; 6.2a → 6.2b → 6.2c;
6.3a → 6.3b → 6.3c → 6.3d; 6.4a requires 6.2c;
6.4b additionally requires 6.1c. Complete these in the order printed below.
Each task is one implementer's reviewable deliverable. No Phase 7 implementation.

Every test command below is run with both `JAXCONT_TEST_X64=0` and `=1`.
Use the environment's Python executable (the current shell has `python3`, not
`python`); `python` in copied global constraints means that environment's Python.
`tol=default_tol(dtype)`; exact-factor Floquet `err=200*tol`. Test values below override neither source tolerance policy nor caller precision.
Run a new test before its implementation and record its expected failure; then
run the stated passing command. Commit steps are omitted because `.git` is
read-only; an executing developer may record changes after checks in a writable checkout.

---
### Task 6.2a: Explicit curve layouts and decoder migration

**Files:**
- Create: `src/jaxcont/bifurcations/curve_layout.py`
- Modify: `src/jaxcont/api.py`, `src/jaxcont/bifurcations/curves.py`, `src/jaxcont/bifurcations/events.py`, `src/jaxcont/bifurcations/codim2_events.py`, `src/jaxcont/viz/two_parameter.py`
- Test/modify: `tests/test_minimal_curves.py`, `tests/test_curves.py`, `tests/test_codim2_events.py`, `tests/test_viz.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Existing extended factories and unpack helpers; no array scanner dependency.
- Produces: Frozen `CurveLayout(kind,formulation,n,free)`; `BifProblem.curve_layout`, `Branch.curve_layout`, `BranchPoint.curve_layout`; `CurveModes(v,w,q1,q2,omega,kappa,oscillatory)` pytree; `decode_curve_state(X,q,layout,raw_f,user_args)->(u,p,modes)`.

- [ ] **Step 1:** Write `test_curve_layout_roundtrip_extended`: shifted parabola at free parameter `.5` decodes `u=(.5,-.3)`, full `p=(1.75,.5)`, `omega=1` to `<100*tol`; metadata survives jit/result flatten and bothside assembly. Migrate every manually constructed curve point in the listed regression modules to explicit extended layout metadata, including synthetic Hopf points; ordinary equilibrium points still allow None. Write `test_curve_layout_required_for_curve_decoding` expecting `ValueError` on missing layout. Add extended-layout sign-flip/padding and Hopf unit-norm coefficient tests from spec 6.2 to the layout test group.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_minimal_curves.py -k layout`; expect absent metadata.
- [ ] **Step 3:** Implement descriptors and extended decoding, updating all curve event/plot consumers and result constructors together. Keep original unpack helpers extended-only. Implement the shared diagnostic orientation/normalization pass from spec 6.2 for extended eager scans now, including padding and per-branch carry; later minimal decoding reuses it. Define `w=None` for non-BT modes; the BT decoder will return its stored Jordan vector without dividing by frequency. Migrate metadata-dependent consumers/tests in this task, before requiring layout. Add Breaking documentation to `docs/source/quickstart.rst` and `CHANGELOG.md` for required curve metadata. For arrays, layout is static metadata; it is never inferred from a buffer length.
- [ ] **Step 4:** Run Step 2 and `python3 -m pytest -q tests/test_curves.py tests/test_codim2_events.py tests/test_viz.py`; expect PASS in both precisions.

### Task 6.1a: Array mode assignment and zero ownership

**Files:**
- Create: `src/jaxcont/bifurcations/array_modes.py`
- Test: `tests/test_event_scan.py`

**Interfaces:**
- Consumes: `modes.py::track_modes` predictive semantics and `crossings` zero ownership.
- Produces: `track_modes_array(spectra: Array, valid: Array) -> (tracked, permutation)`; `crossing_mask(values: Array, eligible: Array, valid: Array) -> Array`, shape `(B-1,K)`.

- [ ] **Step 1:** Write `test_array_assignment_matches_host_predictor` and `test_crossing_masks_zeros_padding_nonfinite` exactly as spec 6.1: 21 samples, modes `p,-p,.2+.1*p+2j,-.2+.1*p+3j`; tied all-zero cost chooses identity. Mask `[-1,0,1,0,NaN]` with validity `[1,1,1,0,0]` is `[1,0,0,0]`; all-zero values never bracket; `[0,1]` brackets once.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_event_scan.py -k "assignment or crossing_masks"`; expect missing-module failure.
- [ ] **Step 3:** Implement bounded JAX Hungarian assignment with lowest-index ties, predictive carry, integer permutation, and the owned-zero mask. Use a finite dtype maximum divided by 4 for nonfinite costs, matching host semantics. Do not change the host oracle.
- [ ] **Step 4:** Repeat Step 2 in both precisions; expect PASS under eager and `jax.jit`, including singleton/empty-bracket buffers.

### Task 6.1b: Built-in array scan protocol

**Files:**
- Create: `src/jaxcont/bifurcations/event_scan.py`
- Modify: `src/jaxcont/bifurcations/events.py`, `src/jaxcont/bifurcations/codim2_events.py`
- Test: `tests/test_event_scan.py`

**Interfaces:**
- Consumes: 6.1a array assignment and masks; 6.2a extended decoder/layout; existing scalar tests and spectral eligibility rules.
- Produces: `EventScanInput(params,states,tangents,eigenvalues,trivial_index,valid,curve_layout=None,floquet_strategies=(),floquet_spectra=())`; `EventCandidates(values,eligible,bracket_mask,spectra,permutation)` pytrees; `scan_event_candidates(events,data,args) -> tuple[EventCandidates,...]`; built-in `scan_arrays(data,args)`.

- [ ] **Step 1:** Write `test_scan_preserves_raw_trivial_identity`: raw `[1,1,1.1]`, trivial index `1`, permutation `[2,0,1]` excludes tracked column `2` only. Add `test_array_scan_two_modes_one_bracket`: two real PD modes `[-.8,-.9] -> [-1.2,-1.1]` yield two mask entries, with a separate trivial `+1` excluded.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_event_scan.py -k "trivial_identity or two_modes"`; expect missing interfaces.
- [ ] **Step 3:** Implement array scan dispatch for Fold/Hopf/BP/PD/NS/CP/BT/ZH/GH/HH. Preserve original-system spectral decoding, conjugate representative selection, and real/complex masks. Reuse 6.2a diagnostic orientation/normalization before coefficient scans, with independent carry per vmapped branch. Remove scalar `float()` from array-pure coefficient tests and port `_drop_nearest_pinned_pair` without Python branching on array values. Define the per-effective-strategy spectrum tuple now (empty until 6.3d); do not infer layouts from packed length. Preserve custom eager `scan(points)`; raise a named `TypeError` for unsupported traced Python scanners. Use static Python dispatch only on event types, never on array values.
- [ ] **Step 4:** Run Step 2 and `python3 -m pytest -q tests/test_modes.py tests/test_bifurcations.py tests/test_codim2_events.py tests/test_period_doubling_neimark_sacker.py`; expect PASS in both precisions.

### Task 6.1c: Expose candidates and retain eager refinement

**Files:**
- Modify: `src/jaxcont/api.py`, `src/jaxcont/__init__.py`, `src/jaxcont/bifurcations/events.py`
- Test: `tests/test_event_scan.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: 6.1b scan tuple; current `detect_events` refinement validation, locality and dedup.
- Produces: `ContinuationResult.event_candidates` dynamic pytree child; `refine_event_candidates(result,problem,events,*,tolerance=None,max_iterations=50,solvers=Solvers()) -> ContinuationResult` (unbatched eager only).

- [ ] **Step 1:** Write `test_jit_vmap_detects_shifted_hopf`, `test_candidates_refine_with_left_anchor`, `test_custom_python_scan_rejected_under_jit`, and `test_loop_scan_event_detection_keeps_smooth_branch_grad`. Add `test_bothside_candidate_assembly` and `test_scalar_candidate_refinement_dispatch` with spec 6.1 assertions: crossings on each arm and seed once, assembled bracket indices, padding masks, and direct scalar CP/BT refinement without `TrackedMode`. Hopf block is `[[p-a,-2],[2,p-a]]`, equilibrium `(.5,-.3)`, `a=[-.25,.25]`, natural grid step `.125` on `[-.5,.5]`: one candidate per batch, eager root error `<20*tol`. For `u=a*p`, five samples `[0,.25,.5,.75,1]` (`ds=.25, adaptive=False, max_steps=4`) give summed-state derivative `2.5`.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_event_scan.py`; expect the current traced-events rejection.
- [ ] **Step 3:** Replace the rejection with fixed-shape detection. Populate needed spectra even when events require them independently of `compute_stability`; use a valid seed in padded spectral slots, masking outputs afterward. Adapt spectral candidates to existing `TrackedMode`; dispatch scalar candidates directly. Preserve left-point `step_update` inputs, including the unchanged fixed BT anchor. Assemble eager bothside states first (reverse backward valid points, share seed once), then rescan the assembled buffer for candidate rows, permutations, orientation and zero ownership; never concatenate arm masks. Include candidates in all result pytree constructors. Keep traced bothside rejection and eager custom scanners.
- [ ] **Step 4:** Document unvalidated candidates, slicing before eager refinement, stopped mask gradients, and the `loop="scan"` reverse-mode requirement. Add a release entry for the new protocol.
- [ ] **Step 5:** Run `python3 -m pytest -q tests/test_event_scan.py tests/test_functional_api.py tests/test_loop3_regressions.py`; expect PASS in both precisions.

### Task 6.2b: Minimally augmented fold chart

**Files:**
- Modify: `src/jaxcont/bifurcations/curves.py`, `src/jaxcont/bifurcations/curve_layout.py`, `src/jaxcont/core/scan_continuation.py`
- Test/modify: `tests/test_minimal_curves.py`, `tests/test_curves.py`, `tests/test_codim2_events.py`, `tests/test_viz.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: 6.2a layout; `fold_point` seeding; accepted-step hook.
- Produces: `FoldCurveArgs(b,c,user_args,formulation,n)`; `fold_curve_problem(...,formulation="minimal")`; `unpack_minimal_fold_curve(X,n)->(u,p_fixed)`; `fold_curve_step_update(args,X,q)->FoldCurveArgs`.

- [ ] **Step 1:** Write `test_minimal_fold_shifted_cusp` and fold half of `test_border_refresh_preserves_root_and_reject_keeps_args`. Width `2`; shifted cubic seed `(1.7,[-1.7,4.2])`; normalized discriminant error `<100*tol` on `p1=4.2..5.2`. Scale borders by `2,-3`; same root, rejected update leaves all leaves identical. Add `test_fold_diagnostic_orientation_no_false_cp` from spec 6.2: alternating raw signs, genuine CP crossing, padding and independent batched carries.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_minimal_curves.py -k "fold or border"`; expect missing formulation.
- [ ] **Step 3:** Implement fold residual `(f,g)` using the `(n+1)` bordered solve and adjoint border refresh. Retain `formulation="extended"`. Compute accepted-point tangents using updated args in pseudo-arclength; natural continuation has no stored tangent update. Preserve seed tangent/status behavior and rejected args. Add minimal fold decoding with reconstructed null mode and branchwise sign alignment before eager/array CP scans per spec 6.2. Invalid slots do not update carry; ambiguous alignment disables the adjacent bracket. In this task migrate legacy fold factory calls/unpack consumers in all listed modules to `formulation="extended"`, retain minimal-default integration tests, and document the fold default break in quickstart/Breaking changelog.
- [ ] **Step 4:** Run Step 2 and `python3 -m pytest -q tests/test_curves.py tests/test_codim2_events.py tests/test_viz.py tests/test_event_scan.py tests/test_adaptive_stepsize.py tests/test_periodic_orbit_continuation.py`; expect PASS with old periodic reference behavior preserved.

### Task 6.2c: Minimal positive-frequency Hopf chart and consumers

**Files:**
- Modify: `src/jaxcont/bifurcations/curves.py`, `src/jaxcont/bifurcations/curve_layout.py`, `src/jaxcont/bifurcations/codim2_events.py`, `src/jaxcont/viz/two_parameter.py`
- Test: `tests/test_minimal_curves.py`, `tests/test_curves.py`, `tests/test_codim2_events.py`, `tests/test_viz.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: 6.2a layouts, 6.2b border-update conventions; existing `HopfCurveArgs`.
- Produces: `hopf_curve_problem(...,formulation="minimal")`; expanded `HopfCurveArgs` with complex `b,c`, static `formulation,n` and existing anchors/user args; `unpack_minimal_hopf_curve(X,n)->(u,p_fixed,omega)`.

- [ ] **Step 1:** Write `test_minimal_hopf_shifted_parabola`, `test_minimal_and_extended_agree_at_fixed_parameters`, `test_minimal_curve_decoding_events_and_plotting`, `test_minimal_curve_seed_q_sensitivity`. Width `4`, equilibrium `(.5,-.3)`, `omega=1`, solved parameters `[2,1.9375,1.75,1.4375,1]`; errors `<100*tol`. Seed derivative at `q=.5` is `-1`. Add `test_hopf_diagnostic_unit_norm` with rescaled/phase-rotated raw seeds and invariant coefficient assertions from spec 6.2.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_minimal_curves.py`; expect missing minimal Hopf support.
- [ ] **Step 3:** Implement complex bordered scalar residual `(f,Re(g),Im(g))`, normalized adjoint/right border refresh, and diagnostic mode reconstruction. Normalize reconstructed complex modes to unit Hermitian norm before Lyapunov evaluation and align phase per spec 6.2. Migrate legacy Hopf factory calls in all listed test modules to explicit `formulation="extended"` in this default-switch task (fold migrations already belong to 6.2b, metadata to 6.2a), add minimal integration coverage, and preserve user args and dynamic seed q dependence. Restrict this chart to positive frequency.
- [ ] **Step 4:** Document default representation change, extended opt-in and minimum-frequency chart limitation; add Breaking entry under the copied release policy.
- [ ] **Step 5:** Run `python3 -m pytest -q tests/test_minimal_curves.py tests/test_curves.py tests/test_codim2_events.py tests/test_viz.py`; expect PASS in both precisions.

### Task 6.3a: Periodic Hessenberg–triangular reduction

**Files:**
- Create: `src/jaxcont/stability/periodic_schur.py`
- Test: `tests/test_periodic_schur.py`

**Interfaces:**
- Consumes: Ordered `(m,n,n)` interval factors from existing `interval_propagators`.
- Produces: `periodic_hessenberg(factors: Array)->(H,Q)` with `M_i Q_i=Q_(i+1) H_i`, all but the last factor triangular, last upper Hessenberg.

- [ ] **Step 1:** Write `test_periodic_hessenberg_factor_reconstruction`: `m=4,n=3`, entries `sin(arange(36).reshape(4,3,3)+1)+2*I`; verify each cyclic reconstruction and orthogonality with relative error `<500*eps`, forbidden subdiagonals `<500*eps*norm(H)`. Include factors of the 32-factor rotating-scale fixture from spec 6.3.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_periodic_schur.py -k hessenberg`; expect missing module.
- [ ] **Step 3:** Implement complex Householder periodic reduction with cyclic unitary accumulation, no product, no host callbacks. Cast constants to factor dtype. Keep original factors available for reconstruction checks.
- [ ] **Step 4:** Run Step 2 under `jit`/`vmap` and both precisions; expect PASS.

### Task 6.3b: Shifted periodic Schur and scaled spectra

**Files:**
- Modify: `src/jaxcont/stability/periodic_schur.py`
- Test: `tests/test_periodic_schur.py`

**Interfaces:**
- Consumes: 6.3a factors and cyclic basis.
- Produces: `periodic_schur(factors,*,max_sweeps=None)->(T,Q,phase,logabs,converged)`; default budget `200*n`, deflation multiplier `100*eps`.

- [ ] **Step 1:** Write `test_periodic_schur_rotating_scaled_factors`, `test_periodic_schur_complex_equal_modulus`, `test_periodic_schur_zero_and_budget_status`. Compare spectra permutation-independently. Exact logabs `[-64,0,64]`, phases `1`; equal-modulus block gives `16*log(.9)` and `exp(+-3.2j)`; diagonal zero case gives `[-inf,0,log(2)]`. All finite log/phase errors `<err`; nontriangular fixture with zero sweeps reports false.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_periodic_schur.py`; expect missing Schur driver.
- [ ] **Step 3:** Implement scaled small-block shifts, periodic bulge chasing and scale-safe last-Hessenberg-factor subdiagonal deflation (including zero/tiny diagonal safeguards), following the factor equations/reference in spec 6.3. Use bounded JAX loops and explicit convergence status, log diagonal accumulation and zero handling. Ordinary QR on the product is prohibited. Preserve complex phases and conjugate pairs.
- [ ] **Step 4:** Run Step 2 in both precisions; additionally require cyclic factor reconstruction `<500*eps` and a nonzero `exp(-64)` value. This is a numerical prototype gate: do not proceed/default-switch if any assertion fails.

### Task 6.3c: Periodic eigenvectors and multiplier differentiation

**Files:**
- Modify: `src/jaxcont/stability/periodic_schur.py`
- Test: `tests/test_periodic_schur.py`

**Interfaces:**
- Consumes: 6.3b converged T/Q and spectrum.
- Produces: `periodic_eigenvectors(factors,T,Q,phase,logabs)->(right,left)` in per-phase normalized/scaled form; differentiable `periodic_spectrum(factors,*,max_sweeps=None)->(phase,logabs,converged)` with custom JVP.

- [ ] **Step 1:** Write `test_periodic_eigenvectors_are_not_schur_columns`: factors built from upper triangular `[[1,2],[0,.5]]` conjugated by cyclic rotations; reconstructed phase-zero right modes satisfy the eigen-equation to `<500*eps`. Write `test_periodic_schur_jvp_log_multiplier`: contracting diagonal `exp(-a)` over 32 factors gives log derivative `-32`, value derivative `-32*exp(-32)` at `a=1`, relative error `<20*err`. Check JVP and reverse-mode scalar objective. Add `test_periodic_schur_nonnormal_complex_derivatives`, `test_periodic_schur_mixed_status_batch`, and `test_periodic_schur_zero_underflow_overflow` exactly as spec 6.3 (low-level spectrum/status assertions here; public Floquet exception/stability/event assertions in 6.3d): four cyclic parameter-dependent nonorthogonal factors, analytic log/phase/value derivatives at `a=.7`, left AND right residuals at all phases, nonzero dual pairings/local scales, mixed failed lanes, zero factor and logs `+-960` with zero/infinity values.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_periodic_schur.py -k "eigenvectors or jvp or nonnormal_complex_derivatives or mixed_status_batch or zero_underflow_overflow"`; expect absent derivative/reconstruction support.
- [ ] **Step 3:** Implement scaled periodic triangular back substitution for left/right eigenvectors. Normalize local dual pairings and sum per-factor eigenvalue perturbations in a custom JVP; stop derivatives through mode ordering/deflation. Preserve complex phases. Do not advertise derivatives for repeated or defective modes.
- [ ] **Step 4:** Run Step 2 and the complete periodic Schur test module; expect PASS in both precisions. Record every eigenvector, nonnormal JVP/reverse-mode, complex-phase, mixed-status and representability gate before the separate default decision; diagonal derivatives alone cannot authorize it.

### Task 6.3d: Floquet strategy, flow identity and branch integration

**Files:**
- Modify: `src/jaxcont/stability/floquet.py`, `src/jaxcont/api.py`, `src/jaxcont/bifurcations/events.py`, `src/jaxcont/bifurcations/event_scan.py`
- Test: `tests/test_floquet.py`, `tests/test_period_doubling_neimark_sacker.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: 6.3c spectrum/eigenvectors and 6.1 candidate eligibility.
- Produces: `PeriodicSchur`, `ProductFloquet`, `FloquetSpectrum`; `Solvers.floquet`; optional `Branch.floquet_logabs`, `Branch.floquet_converged`; Floquet functions take `floquet_solver` instead of matrix `eigen_solver`.

- [ ] **Step 1:** Write `test_circle_log_multipliers_mesh_convergence`, `test_periodic_schur_flow_index_with_near_unit_physical_mode`, `test_periodic_schur_branch_jit_vmap_and_pd_ns` per spec. Circle fine mesh `(40,4)` has logs `[-12.566370614359172,0]` to `<2e-3`. Flow `e0`, values `[.9999,1.00001]`: index `0`, stable false. Add `test_floquet_event_override_detection_and_refinement` from spec 6.3, testing both override directions, distinct ordering/trivial indices, per-strategy failure and once-per-point caching. Add `test_failed_schur_masks_events_and_padding`: failed status produces no candidate and false stability, invalid slots never evaluated as collapsed orbits.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_floquet.py -k "periodic_schur or log_multipliers or failed_schur or event_override_detection_and_refinement"`; expect missing strategy.
- [ ] **Step 3:** Implement strategy dispatch and spectrum reporting; retain `ProductFloquet()` as bundle default until Step 5 acceptance; preserve tuple returns and raw `PeriodicArgs.user_args`. Use reconstructed right modes for flow identity. Use logabs for stability, explicit nonconvergence policy from spec, and populate all Branch pytree fields. Add optional `floquet_solver=None` fields on PD/NS events; bind an unspecified strategy from `Solvers.floquet` in continuation and the eager candidate-refinement helper, preserving explicit event overrides. Compute/cache spectra per distinct effective static strategy, supplying each event its matching spectrum/trivial-index/convergence buffers through the 6.1b tuple; branch stability/reporting uses the bundle strategy. Refinement uses the identical effective event strategy. Internal orchestration consumes status arrays; only eager public Floquet evaluation raises on failure. Move existing product monkeypatch/custom-order tests to explicit `ProductFloquet`.
- [ ] **Step 4:** Document strategy selection, changed positional argument, representability limits and spectral derivative scope; add Breaking entry.
- [ ] **Step 5:** Run `python3 -m pytest -q tests/test_floquet.py tests/test_periodic_schur.py tests/test_event_scan.py tests/test_period_doubling_neimark_sacker.py tests/test_prc.py`; expect PASS in both precisions. PRC remains unchanged until Phase 7. This is the separate default acceptance decision: switch `Solvers.floquet` to `PeriodicSchur()` only after all 6.3a–c prototype gates and this integration suite pass in both precisions; then rerun this integration command in both precisions. On any gate failure keep ProductFloquet as default and report the failed gate.

### Task 6.4a: Regular real kappa chart

**Files:**
- Modify: `src/jaxcont/bifurcations/curves.py`, `src/jaxcont/bifurcations/curve_layout.py`
- Test: `tests/test_hopf_bt_curve.py`

**Interfaces:**
- Consumes: 6.2c `HopfCurveArgs` and static layout; BT uses fixed-anchor semantics, not minimal border refresh.
- Produces: `hopf_curve_problem(...,formulation="bt")`; `unpack_bt_hopf_curve(X,n)->(u,p_fixed,v,w,kappa)`; `hopf_curve_frequency(kappa)->(omega,oscillatory)`; args field `v_anchor` for BT chart.

- [ ] **Step 1:** Write `test_bt_chart_residual_and_jacobian_at_zero` and `test_hopf_bt_kappa_crosses_shifted_bt` using spec 6.4 shifted system. BT is `u=(.7,-.4),p=(.3,1.2)`, `v=(1,0),w=(0,1)`; fixed-p1 Jacobian smallest singular value `>.05`. Natural grid `p1=1.3..1.1`, `ds=.01, adaptive=False, max_steps=20`, gives 21 valid points, `x=1.2-p1`, `p0=.3-x*x`, `kappa=-2*x` to `<100*tol`. Add `test_bt_fixed_anchor_nonnormal_similarity` from spec 6.4, including its parameter-dependent transform and exact nilpotent counterexample; all residuals remain valid after accepted steps with unchanged anchor, including at BT.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_hopf_bt_curve.py -k "residual or crosses"`; expect unsupported formulation.
- [ ] **Step 3:** Implement the exact five residual blocks from spec, orthogonalized positive-frequency initialization, a seed-fixed `v_anchor` and decoder with both real `CurveModes.v,w` at every kappa. The BT step hook is a no-op on the anchor on accepted/rejected steps and refinement setup; do not set it to accepted v. Report loss of local chart regularity via failure status; no automatic reanchoring. Square-root frequency is reporting only. Preserve minimal/default and legacy extended representations.
- [ ] **Step 4:** Run Step 2 and `python3 -m pytest -q tests/test_minimal_curves.py tests/test_curves.py`; expect PASS in both precisions.

### Task 6.4b: BT detection, neutral-segment eligibility and presentation

**Files:**
- Modify: `src/jaxcont/bifurcations/codim2_events.py`, `src/jaxcont/bifurcations/event_scan.py`, `src/jaxcont/viz/two_parameter.py`
- Test: `tests/test_hopf_bt_curve.py`
- Modify: `docs/source/quickstart.rst`, `CHANGELOG.md`

**Interfaces:**
- Consumes: 6.4a kappa chart and 6.1c candidate/refinement bridge.
- Produces: Kappa-owned BT brackets for BT layout, guarded positive-kappa GH/ZH/HH scans, and dashed neutral-saddle visualization.

- [ ] **Step 1:** Write `test_bt_kappa_event_once_and_validated`, `test_bt_chart_masks_hopf_only_events`, `test_bt_chart_jit_vmap_scan_gradient`. Expect one BT with parameter errors `<100*tol`, no `curve_terminated`, no GH/ZH/HH at `kappa<=0`. Replacing `.3` by user args `a` gives derivative of mean solved p0 equal to `1` to `<100*tol` with `loop="scan"`. Add `test_bt_mixed_kappa_coefficient_grad` from spec 6.4: `jit(vmap(...))` with `[.2,0,-.2]`, positive-lane reference agreement, finite reverse-mode gradient and exactly zero ineligible coefficient gradients. Add `test_bt_plot_neutral_segment_dashed`: samples kappa `[.2,0,-.2]` produce separately labelled oscillatory/neutral segments with negative-kappa line style `--`.
- [ ] **Step 2:** Run `python3 -m pytest -q tests/test_hopf_bt_curve.py`; expect missing detection/masks.
- [ ] **Step 3:** Use kappa scalar crossings and existing original-system BT root validation/locality. Exclude two distinct pinned columns on either side. Sanitize all ineligible coefficient inputs before any square root/division/solve using the complete canonical linear Hopf surrogate from spec 6.4 (unit mode, omega 1, stable extra dimensions, zero higher derivatives). Normalize eligible modes. Select `kappa_safe=where(eligible,kappa,1)` before taking square roots. Only then evaluate batched coefficients and mask output to zero; `lax.cond` alone is insufficient under vmap; retain omega termination fallback only in the other explicit charts. Document neutral-saddle meaning and larger state width.
- [ ] **Step 4:** Run the complete new modules and `python3 -m pytest -q -n 8`, then `python3 -m pytest -q -m slow`, each in both precision modes; expect PASS. Record unavailable external reference tools separately from numerical failures; do not invent reference outputs.

## Phase 7: Spec only

See [Linear operators and condensed collocation](../specs/2026-10-17-linear-operator-condensed-collocation-design.md).
There are no Phase 7 implementation tasks in this plan. Its acceptance cases,
solver failure policy and structured metadata contract must be reviewed in a
separate future implementation plan.


## Preflight resolutions

| Finding | Resolution | Where it now lives |
|---|---|---|
| C1 — BT anchor breaks roots | Seed-fixed locally valid anchor; no args-only refresh; transformed nonnormal fixture and exact counterexample | BT spec, Chosen design and Verification; Tasks 6.4a/b |
| I1 — Diagnostic orientation/normalization | Branchwise fold sign carry, padding/batch/ambiguous-overlap rules, unit Hopf coefficient modes; tests | Minimal spec, API and Verification; Tasks 6.2a, 6.1b, 6.2b/c |
| I2 — Missing Jordan vector | `CurveModes(v,w,q1,q2,omega,kappa,oscillatory)`; stored w at all BT kappa | Minimal and BT specs; Tasks 6.2a, 6.4a |
| I3 — Batched cond safety | Safe complete surrogate inputs before singular operations; mixed-kappa jit(vmap) and grad gate | BT spec, API and Verification; Task 6.4b |
| I4 — Migration timing | Metadata migration in 6.2a, fold default migration in 6.2b, Hopf default migration in 6.2c; all affected test files/checks included | Minimal spec, Migration ownership; Tasks 6.2a/b/c |
| I5 — Decoder dependency | Extended decoder task printed/executed before array scans; no length inference | Execution dependencies; Task 6.2a before 6.1a/b/c; trace-safe spec |
| I6 — Floquet override spectra | Per-effective-strategy cache and event input; matching detection/refinement and differing-strategy test | Trace-safe and Schur specs; Tasks 6.1b, 6.3d |
| I7 — Weak Schur gate | Nonnormal parameterized complex factors, left/right residuals/dual scales, analytic phase/value derivatives, mixed failure and zero/range checks; separate default decision | Schur spec, Verification; Tasks 6.3b/c/d |
| I8 — Complex transpose ambiguity | Exact JAX bilinear transpose A.T, explicit Hermitian adapters, stacked-real nonholomorphic convention; nonsymmetric complex test | Phase 7 operator spec, API and Acceptance tests (future plan only) |
| Minor — Tolerance policy | Explicit machine-epsilon algorithmic exceptions; exact-factor err=200*default_tol(dtype); separate discretization error | Global Constraints and execution policy; Schur spec |
| Minor — Bothside candidates | Assemble states once then rescan for reversal/index/seed ownership; explicit regression | Trace-safe spec, Eager bothside assembly; Task 6.1c |
| Minor — Deflation details | Last Hessenberg factor's subdiagonal/local diagonals; scale-safe zero/tiny rules | Schur spec, Chosen design; Task 6.3b |
| Minor — Repeated grid parenthesis | Removed duplicate ds/adaptive/max_steps text | Trace-safe spec, Verification |
| Minor — Working-tree claim | Scope statement distinguishes these six docs from pre-existing edits/untracked file | Plan Baseline |
| Audit — Source citations | Correct scalar fallback, mode tracking and hook invocation references | Trace-safe spec, Problem and API |
| Compatibility — Scalar/mode refinement | TrackedMode only for spectral candidates; direct scalar dispatch and test | Trace-safe spec; Task 6.1c |
| Constraints — Traced pinned-mode drop | Port host pinned-pair selection to distinct-column array logic | Trace-safe spec; Task 6.1b |

All findings are resolved as design/plan requirements. Numerical prototype and
implementation acceptance remain future work; this documentation revision does
not claim those tests have run or passed. Phase 7 still requires its own reviewed
implementation plan for failure handling and workspace before coding.
