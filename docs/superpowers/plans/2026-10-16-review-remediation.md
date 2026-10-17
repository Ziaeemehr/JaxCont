# Review Remediation (2026-10) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the correctness, transformation, and API defects found in the 2026-10 review so that JaxCont's three promises hold: (1) results you can trust without supervision, (2) `jit`/`vmap`/`grad` across the analysis, (3) agreement with MatCont/BifurcationKit.

**Architecture:** Keep the existing spine (pure residual → whole-loop `lax.while_loop` engine → eager post-pass → `custom_vjp` extended-system solvers). Changes:
- the engine reports explicit status and takes `args` as a dynamic input;
- event detection moves to tracked, continuous test functions with validated or failed outcomes;
- periodic problems get a real pytree structure and a per-step reference update;
- float64 becomes the documented analysis default, with dtype-derived tolerances.

**Tech Stack:** Python ≥ 3.11, JAX ≥ 0.9, equinox, NumPy/SciPy (host-side mode matching), pytest + pytest-xdist.

**Spec:** [notes/PROJECT_REVIEW_2026-10.md](../../../notes/PROJECT_REVIEW_2026-10.md). Finding IDs (H1–H10, M1–M13) below refer to it.

**Scope note:** This plan covers several subsystems. Phases 0–3 are specified to task level and ship together as **v0.5.0** (breaking). Phases 4–6 are specified at task level but are independent and can ship as v0.5.x/v0.6.0. Phase 7 (scaling seam) needs its own design spec first and is only outlined here.

## Global Constraints

- Pre-1.0 policy: remove, don't deprecate. Every breaking change gets a `CHANGELOG.md` entry under `## [0.5.0] - Unreleased` → `### Breaking`.
- Never call `jax.config.update("jax_enable_x64", ...)` at import time inside `jaxcont`; precision is the caller's choice.
- Tolerances derive from dtype through one helper, `jaxcont.utils.precision.default_tol(dtype)`: `1e-9` for float64, `1e-5` for float32. No new hard-coded `1e-5`/`1e-6` literals in `src/`.
- Every test in the default suite must pass under both `JAXCONT_TEST_X64=0` and `JAXCONT_TEST_X64=1`.
- No Python control flow on traced values in the engine or in anything reachable under `jit`/`vmap`.
- New source comments state invariants, not incident history; history goes in `CHANGELOG.md`.
- Run tests with `python -m pytest -q -n 8`; slow suite with `python -m pytest -q -m slow`.

## Review Focus

1. **Two modes trading places** (e.g. two complex pairs whose real parts swap order) must produce no event. Pinned in Task 2.2 (`test_hopf_no_event_when_pairs_swap_order`).
2. **Bifurcation exactly on a sample point** (test value exactly 0) must be reported once. Pinned in Task 2.3 (`test_exact_zero_reported_once`).
3. **Same-kind bifurcations at the same p on different sheets** of a branch (after a fold) must both be reported. Pinned in Task 2.3 (`test_dedup_keeps_hits_on_different_sheets`).
4. **A trajectory guess whose time axis does not start at 0** must work exactly like the rebased one. Pinned in Task 3.2 (`test_trajectory_time_offset_is_rebased`).
5. **Repeated identical `continuation()` calls** must not recompile. Pinned in Task 1.1 (`test_repeated_calls_hit_compile_cache`).

---

## Phase 0: Test infrastructure and precision policy (M1)

### Task 0.1: Dual-precision test harness and `default_tol`

**Files:**
- Create: `src/jaxcont/utils/precision.py`
- Create: `tests/conftest.py`, or modify it if present
- Modify: `.github/workflows/tests.yml` (add `x64: [0, 1]` to the matrix, export `JAXCONT_TEST_X64`)
- Test: `tests/test_precision.py`

**Interfaces:**
- Produces: `default_tol(dtype) -> float` (`1e-9` for float64, `1e-5` for float32 and anything narrower); `is_x64() -> bool`.

- [ ] **Step 1:** Write `tests/test_precision.py::test_default_tol_values`, asserting `default_tol(jnp.float64) == 1e-9` and `default_tol(jnp.float32) == 1e-5`.
- [ ] **Step 2:** Run it; expect FAIL (module missing).
- [ ] **Step 3:** Implement `precision.py`. In `conftest.py`, read `JAXCONT_TEST_X64` before any `jax` import and call `jax.config.update("jax_enable_x64", True)` when it is `"1"`.
- [ ] **Step 4:** Run the full suite under `JAXCONT_TEST_X64=0` and `=1`. Record every test that fails under x64 in the task's commit message. These are the existing dtype bugs (Codex finding: `_tangent`'s `jnp.zeros` without dtype at `scan_continuation.py:76`).
- [ ] **Step 5:** Fix dtype leaks by passing `dtype=u.dtype` to every `jnp.zeros`/`jnp.eye`/`jnp.ones` in `core/`, `solvers/`, `stability/`, `bifurcations/`, and `problems/`. Add `test_precision.py::test_scan_float32_inputs_under_x64`: with x64 on and float32 `u0`, `pseudo_arclength_scan` returns float32 `states`.
- [ ] **Step 6:** Both precisions green → commit `test: run the suite under float32 and float64; add default_tol`.

### Task 0.2: Dtype-derived defaults in `ContinuationPar` and the solvers

**Files:**
- Modify: `src/jaxcont/api.py` (`ContinuationPar.newton_tol: Optional[float] = None`)
- Modify: `src/jaxcont/solvers/implicit.py`, `bifurcations/fold_solve.py`, `bifurcations/hopf_normal_form.py`, `bifurcations/codim2.py`, `problems/periodic.py`, `bifurcations/events.py` (every `tol=` default becomes `None` → `default_tol(dtype)`)
- Modify: `docs/source/quickstart.rst` (new "Precision" section recommending x64 for analysis)
- Test: `tests/test_precision.py`

- [ ] **Step 1:** Write `test_newton_tol_none_resolves_by_dtype`: `continuation()` with the default settings converges on `u² + p` in both precisions, and in float64 the max residual over the branch is `< 1e-9`.
- [ ] **Step 2:** Implement the resolution in `_call_scan`: `tol = settings.newton_tol or default_tol(dtype)`.
- [ ] **Step 3:** Delete the periodic-specific "pass `newton_tol=1e-5`" advice from the `periodic_orbit_problem` docstring and the examples (now automatic).
- [ ] **Step 4:** Suite green in both precisions → commit.

---

## Phase 1: Engine status, step control, compile caching (H4, H7, M3, M7, M9)

### Task 1.1: Stable kernel with dynamic `args`

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py` (`pseudo_arclength_scan`, `natural_scan`, `_newton_correct`, `_natural_correct`, `_tangent`, `branch_eigenvalues`)
- Modify: `src/jaxcont/api.py:491` (drop `rhs2`; pass `problem.f` and `problem.args`)
- Modify: every direct caller (`grep -rn "pseudo_arclength_scan\|natural_scan\|branch_eigenvalues" src examples tests`)
- Test: `tests/test_functional_api.py`

**Interfaces:**
- Produces: `pseudo_arclength_scan(f, u0, p0, p_end, args, ds0, ds_min, ds_max, tol, max_steps, max_iter, linear_solver=Dense(), adaptive=True) -> ScanResult` (Task 1.2 appends `max_attempts=None`, Task 3.3 adds static `step_update` after `f`). Here `f(u, p, args)` is static and `args` is a dynamic pytree. `natural_scan` gets the same signature. `branch_eigenvalues(f, states, params, args, eigen_solver)`.

- [ ] **Step 1:** Write `test_repeated_calls_hit_compile_cache`: three identical `jc.continuation()` calls on `u² + p`. Assert that `pseudo_arclength_scan._cache_size()` is unchanged after the first call, and that a fourth call with different `args` values (same structure) also leaves it unchanged.
- [ ] **Step 2:** Run it; expect FAIL (cache grows by 1 per call today).
- [ ] **Step 3:** Thread `args` through as a traced argument. `f` stays the static argument. Users with a lambda `f` still recompile per new lambda; document that.
- [ ] **Step 4:** Write `test_grad_wrt_args_through_engine_forward`: `jax.jacfwd` of `sum(states * valid)` with respect to `args`, for `u − a·p`, matches the analytic value.
- [ ] **Step 5:** Suite green → commit `perf: make args a dynamic engine input so continuation() reuses its compiled kernel`.

### Task 1.2: Termination status, seed failure, attempt budget

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py`
- Modify: `src/jaxcont/api.py` (`ContinuationPar.max_attempts: Optional[int] = None`; `ContinuationResult.status: str`; raise on seed failure in eager mode)
- Test: `tests/test_adaptive_stepsize.py`, `tests/test_functional_api.py`

**Interfaces:**
- Produces: `class Status(enum.IntEnum)` in `scan_continuation.py` with `REACHED_END = 0, MAX_STEPS = 1, STALLED = 2, NONFINITE = 3, SEED_FAILED = 4, MAX_ATTEMPTS = 5`. `ScanResult` gains `status: Array` (int32) and `n_attempts: Array`. `ContinuationResult.status` is the lower-case enum name (`"reached_end"`, …); `ContinuationResult.stats` holds `{"n_points", "n_attempts", "status"}`.

- [ ] **Step 1:** Write the tests:
  - `test_seed_failure_raises_eagerly`: for `u² + p`, `u0 = [0.0]`, `p0 = 1.0`, `jc.continuation` raises `ValueError` matching `"seed"`.
  - `test_seed_failure_traced_marks_nothing_valid`: under `jax.jit`, `branch.valid.sum() == 0` and `status == Status.SEED_FAILED`.
- [ ] **Step 2:** Write the step-control tests:
  - `test_stall_only_after_failure_at_ds_min`: build a problem whose corrector fails for every `ds > 0.012` (e.g. `max_iter = 1` on a curved branch). With `ds = 0.015` and `ds_min = 0.01`, assert that some point is accepted with `ds == 0.01`.
  - `test_initial_ds_is_clipped`: `ds = 0.5, ds_max = 0.1` gives `res.ds[1] <= 0.1`.
  - `test_max_attempts_bounds_rejections`: `status == Status.MAX_ATTEMPTS` and `n_attempts == max_attempts`.
- [ ] **Step 3:** Run; expect FAIL.
- [ ] **Step 4:** Implement in the engine:
  - Clip `ds0` to `[ds_min, ds_max]`.
  - Declare a stall only when the *attempted* `ds` was already `<= ds_min` and failed.
  - Count attempts in the carry; loop while `idx < max_steps and attempts < max_attempts` (default `4 * max_steps`).
  - When `seed_converged` is false, run zero iterations, set `n_valid = 0`, and set `status = SEED_FAILED`.
  - Set `status` from the stop reason.
- [ ] **Step 5:** Fix the module docstring paragraph about rejected steps to describe the new budget.
- [ ] **Step 6:** Suite green → commit.

### Task 1.3: Masked `Branch` semantics, `bothside` stitching, batched diagnostics

**Files:**
- Modify: `src/jaxcont/api.py` (`Branch.n_valid`, `Branch.at_param`, `_run_scan_bothside`, `_assemble_leg`)
- Test: `tests/test_functional_api.py`

**Interfaces:**
- Produces: `Branch.n_valid -> int | Array`: the concrete count when eager, `valid.sum()` when traced. `Branch.at_param` ignores invalid slots. `EventHit.index` refers to the stitched branch.

- [ ] **Step 1:** Write the tests:
  - `test_traced_n_valid_counts_mask`: for `u − p` with `max_steps = 400` under `jit`, `int(out.branch.n_valid) == int(out.branch.valid.sum())`.
  - `test_at_param_ignores_padding`.
  - `test_bothside_event_index_points_at_event`: for each event, `abs(branch.params[e.index[0]] - e.p) <= abs(branch.params[e.index[0]] - branch.params[e.index[1]])`.
  - `test_bothside_tangents_follow_stitched_direction`: consecutive stitched tangents have a positive dot product.
- [ ] **Step 2:** Implement:
  - Negate the reversed leg's tangents.
  - Remap backward-leg indices `i → n_back - i`.
  - Build `convergence_info` from one `np.asarray` per buffer instead of per-point `float()`.
- [ ] **Step 3:** Suite green → commit.

---

## Phase 2: Event detection rebuild (H1, H2, H3, H10, M6)

### Task 2.1: Validated vs. failed outcomes

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py` (`EventHit.status`), `bifurcations/codim2_events.py`, `src/jaxcont/api.py` (`ContinuationResult.rejected_events`), `src/jaxcont/viz/core.py` (plot only validated hits)
- Modify: `src/jaxcont/solvers/implicit.py:69` (initial `done` from the seed residual)
- Test: `tests/test_bifurcations.py`, `tests/test_differentiable_root.py`

**Interfaces:**
- Produces: `EventHit.status: Literal["validated", "failed"]`. `ContinuationResult.events` holds only validated hits; `ContinuationResult.rejected_events` holds failed ones together with `info["reason"]` (`"not_converged"`, `"outside_bracket"`, `"spectral_check_failed"`). `refine()` keeps returning `Optional[EventHit]`, but never a `kind`-labelled hit at `right.p` on failure.

- [ ] **Step 1:** Write `test_differentiable_root_keeps_exact_seed`: `G(x) = x²` with seed `x0 = 0.0` returns exactly `0.0`, with no NaN and `converged=True`.
- [ ] **Step 2:** Write `test_failed_refine_not_in_events`. Monkeypatch `fold_point` to return `converged=False`, then assert `result.events == []` and `result.rejected_events[0].info["reason"] == "not_converged"`.
- [ ] **Step 3:** Write `test_refine_outside_bracket_rejected`. For a system with Hopf points at p = ±2 and a forced candidate bracket at [−0.1, 0.1], assert there is no validated event. Locality rule: refined `p` must lie within the bracket widened by one bracket width on each side.
- [ ] **Step 4:** Implement, then commit.

### Task 2.2: Mode tracking and continuous spectral test functions

**Files:**
- Create: `src/jaxcont/bifurcations/modes.py`
- Modify: `src/jaxcont/bifurcations/events.py` (`Hopf`, `BP`, `PeriodDoubling`, `NeimarkSacker`)
- Modify: `src/jaxcont/bifurcations/codim2_events.py` (`BogdanovTakens`, `ZeroHopf`, `DoubleHopf`)
- Test: `tests/test_modes.py`, `tests/test_bifurcations.py`, `tests/test_period_doubling_neimark_sacker.py`, `tests/test_codim2_events.py`

**Interfaces:**
- Produces: `track_modes(spectra: np.ndarray) -> np.ndarray`. It takes `(N, n)` complex values and returns them reordered so that column `k` is one continuous mode, using `scipy.optimize.linear_sum_assignment` on `|λ_i(t) − λ_j(t+1)|` between consecutive points. Host-side NumPy only; it runs in the eager post-pass.
- Produces: `crossings(values: np.ndarray, where: np.ndarray) -> list[int]`, the bracket starts `i` where `values[:, k]` changes sign or hits exactly 0, restricted to steps where `where[i, k] & where[i+1, k]`.
- Event rules (column `k` is a tracked mode; trivial Floquet mode excluded by Task 3.4):
  - Hopf: `Re λ_k` crosses 0 with `|Im λ_k| > tol` at both ends.
  - BP: real `λ_k` crosses 0, followed by the rank check from Task 2.4.
  - PD: real `μ_k` (`|Im| ≤ tol` at both ends) with `Re μ_k + 1` crossing 0.
  - NS: complex `μ_k` with `|μ_k| − 1` crossing 0.
  - Every refined point is re-checked spectrally: the eigenvalue nearest the critical value must satisfy the rule's real/complex condition.

- [ ] **Step 1:** Write `test_modes.py`:
  - `test_track_modes_follows_crossing_eigenvalues`: two real eigenvalues `p` and `−p` on `p ∈ [−1, 1]`; each output column is monotone.
  - `test_track_modes_handles_conjugate_pairs`.
- [ ] **Step 2:** Write the event tests:
  - `test_hopf_no_event_when_pairs_swap_order`: pairs at Re = 0.2 + 0.1p and Re = −0.2 + 0.1p on `p ∈ [−1, 1]`; expect zero events.
  - `test_hopf_spurious_case_from_review`: the 4-D system from review H1, a fixed pair at −0.1 and a moving pair at p; expect exactly one Hopf, at `|p| < 1e-3`.
  - `test_pd_second_multiplier_crossing_detected`: multipliers −0.8 (fixed) and −0.5 → −1.2; expect one PD.
  - `test_pd_not_reported_for_complex_pair_crossing_re_minus_one`.
- [ ] **Step 3:** Run; expect FAIL on the swap and second-multiplier cases.
- [ ] **Step 4:** Implement `modes.py`. Rewrite the four events' detection on top of it, using a new `Event.scan(points) -> list[tuple[int, int, Any]]` hook (bracket and the mode identity passed to `refine`). The orchestrator in `detect_events` calls `scan` when it is present and otherwise the plain `test_function` loop (Fold, Cusp, GeneralizedHopf already have continuous test functions). Delete `select_candidate`, `near_unit_circle`, and `near_critical`.
- [ ] **Step 5:** Port BT/ZH/HH in `codim2_events.py` to `track_modes` over the original-system spectrum. Hopf-curve BT stays `omega` → 0, plus a `rejected_events` entry with reason `"curve_terminated"` when the curve stalls with `omega < 10 * tol` (see Phase 6, Task 6.4, for the ω² formulation).
- [ ] **Step 6:** Suite green, including `-m slow` (the MatCont suite) → commit.

### Task 2.3: Orchestrator: exact zeros, sheet-aware dedup, batched test values

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py` (`detect_events`)
- Test: `tests/test_bifurcations.py`

- [ ] **Step 1:** Write the tests:
  - `test_exact_zero_reported_once`: a fixed-step branch of `u − p` with an event whose test function is exactly `p` and a sample landing on `p = 0`; expect one hit.
  - `test_dedup_keeps_hits_on_different_sheets`: an S-curve with a Hopf on the lower and the upper sheet at the same p (construct with `f = [p + u − u³/3, …]` plus an oscillatory block whose damping is `u`); expect two Hopf hits.
- [ ] **Step 2:** Implement:
  - A zero at sample `i` owns brackets `(i−1, i)` and `(i, i+1)`; report it once.
  - Dedup merges same-kind hits only when their brackets are adjacent (`|i − j| ≤ 1`) **and** `‖(u, p) − (u′, p′)‖ ≤ 2·max(ds_i, ds_j)`, using the per-point `ds` from `ScanResult.ds`.
  - Evaluate all `test_function` values in one `jax.vmap`-ed call where the event is array-pure, falling back to per-point evaluation otherwise.
- [ ] **Step 3:** Suite green → commit.

### Task 2.4: Scale-invariant BP classification

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py` (`BP.refine`)
- Test: `tests/test_branch_switch.py`

- [ ] **Step 1:** Write `test_bp_invariant_under_rhs_scaling`: `f = 1e-6·(u² + p)` produces no BP, and `f = 1e-6·u·(p − u)` (transcritical) produces one BP.
- [ ] **Step 2:** Change the criterion to `σ_min / σ_max < rank_tolerance` on `[f_u | f_p]`, with the default `rank_tolerance = 1e3 · finfo(dtype).eps ** 0.5`.
- [ ] **Step 3:** Suite green → commit.

---

## Phase 3: Periodic orbits (H5, H6, M4, M5)

### Task 3.1: `PeriodicArgs` pytree and user `args`

**Files:**
- Create: `src/jaxcont/problems/periodic_args.py`
- Modify: `src/jaxcont/problems/periodic.py`, `stability/floquet.py`, `stability/prc.py`, `bifurcations/pd_switch.py`, `bifurcations/events.py` (PD/NS take their raw RHS from `PeriodicArgs`), `src/jaxcont/api.py:398` (`_, _, raw_f, mesh = problem.args` → attribute access)
- Modify: `src/jaxcont/__init__.py` (export `periodic_orbit_problem`, `Collocation`, `prc_curve`, `branch_prc`, `dprc_curve`)
- Test: `tests/test_periodic_orbit_problem.py`, `tests/test_floquet.py`, `tests/test_prc.py`

**Interfaces:**
- Produces: `class PeriodicArgs(eqx.Module)` with dynamic fields `u_ref_coll`, `uref_prime_coll`, `user_args` and static fields `raw_f`, `mesh`. `periodic_orbit_problem(f, u_trajectory, t_trajectory, period0, p0, mesh, *, args=None, tol=None)`. `raw_f` is always called as `raw_f(u, p, periodic_args.user_args)`.

- [ ] **Step 1:** Write the tests:
  - `test_periodic_problem_is_jit_argument`: `jax.jit(lambda pr: pr.u0.sum())(problem)` runs.
  - `test_periodic_rhs_receives_user_args`: Hopf normal form with `args = ω`; the converged period equals `2π/ω`.
  - `test_grad_period_wrt_user_args`: `jax.grad` of the converged period with respect to ω equals `−2π/ω²` within `1e-3` relative.
- [ ] **Step 2:** Implement; update the PRC/Floquet/PD-switch call sites and examples 10–14, 19–20, 22–23.
- [ ] **Step 3:** Suite green → commit.

### Task 3.2: Validated periodic construction

**Files:**
- Modify: `src/jaxcont/problems/periodic.py`
- Test: `tests/test_periodic_orbit_problem.py`

- [ ] **Step 1:** Write the tests:
  - `test_trajectory_time_offset_is_rebased`: `t + 100` gives the same `u0` as `t` within `default_tol`.
  - `test_collapsed_orbit_raises`: a constant trajectory raises `ValueError` matching `"amplitude"`.
  - `test_nonconverged_construction_raises`: a guess one period off in shape and `max_iter = 1` raises `ValueError` matching `"converge"`.
- [ ] **Step 2:** Implement:
  - Rebase with `t − t[0]`.
  - Require `t[-1] − t[0] ≥ 0.95·period0`.
  - Interpolate periodically (wrap with `jnp.interp(..., period=period0)`).
  - Use `differentiable_root_checked`.
  - Check, eagerly only, that `T > 0` and that `max_t ‖u(t) − mean(u)‖ > 1e3·tol·(1 + ‖mean(u)‖)`.
- [ ] **Step 3:** Commit.

### Task 3.3: Per-step reference update (engine "aux" hook)

**Files:**
- Modify: `src/jaxcont/api.py` (`BifProblem.step_update: Optional[Callable[[PyTree, Array, Array], PyTree]] = None`, static in the pytree aux)
- Modify: `src/jaxcont/core/scan_continuation.py` (carry `args`; after each accepted step, `args = step_update(args, u_new, p_new)` when provided)
- Modify: `src/jaxcont/problems/periodic.py` (`step_update` replaces `u_ref_coll`/`uref_prime_coll` with the accepted orbit)
- Modify: `src/jaxcont/bifurcations/curves.py` (Hopf curve: the anchor `(q1_seed, q2_seed)` moves into `args`; `step_update` sets it to the accepted `(q1, q2)`)
- Modify: `src/jaxcont/bifurcations/events.py` (refinement uses `step_update(args, left.u, left.p)` as its `args`)
- Test: `tests/test_periodic_orbit_continuation.py`, `tests/test_curves.py`

**Interfaces:**
- Consumes: Task 1.1's dynamic `args` in the engine.
- Produces: `BifProblem.step_update`. The engine signature gains `step_update` as a static argument (after `f`).

- [ ] **Step 1:** Write the tests:
  - `test_long_branch_non_symmetric_orbit`: Van der Pol `μ` from 0.5 to 4.0 (strong shape change) reaches `p_end` with `status == "reached_end"`. Run it on `main` first and record the failure in the commit message. If it already passes, pick the Brusselator from example 11 over its full documented range.
  - `test_hopf_curve_survives_eigenvector_rotation`: a Hopf curve over a range where the critical eigenvector rotates by more than 90°; it reaches `p_end`.
- [ ] **Step 2:** Implement the hook; it must not change buffer shapes. Remove the "fixed-shape buffers rule out re-anchoring" paragraph from `hopf_curve_problem`'s docstring.
- [ ] **Step 3:** Suite and slow suite green → commit.

### Task 3.4: Trivial-multiplier identification by flow direction

**Files:**
- Modify: `src/jaxcont/stability/floquet.py`
- Test: `tests/test_floquet.py`

**Interfaces:**
- Produces: `floquet_multipliers(...)` also returns `trivial_index` (the eigenvector with maximal `|cos∠(v, f(x0))|`). `floquet_stable(multipliers, trivial_index)`. `Branch` gains `trivial_index` for periodic branches; PD/NS exclude by it.

- [ ] **Step 1:** Write `test_trivial_multiplier_by_eigenvector`: a constructed monodromy with eigenvalues `{0.9999 (along f), 1.00001}`; the orbit is reported unstable.
- [ ] **Step 2:** Tighten `test_floquet.py:76` to a relative check: `abs(log|μ| − log|μ_exact|) < 1e-3`.
- [ ] **Step 3:** Implement → commit.

---

## Phase 4: Differentiation story (H8, H9, M2, M11)

### Task 4.1: Transformation matrix and a reverse-mode path

**Files:**
- Modify: `src/jaxcont/core/scan_continuation.py` (`ContinuationPar.loop: Literal["while", "scan"] = "while"`; the `"scan"` variant runs `lax.scan` for exactly `max_attempts` iterations, with accepted/rejected masking as today)
- Create: `docs/source/transformations.rst` (a matrix of rows × columns `jit` / `vmap` / `jacfwd` / `grad`)
  - Rows: `continuation` (while), `continuation` (scan), `events`, `fold_point`, `hopf_point`, the codim-2 points, `periodic_orbit_problem`, `prc_curve`, `dprc_curve`, `branch_switch`.
- Modify: `README.md`, the `src/jaxcont/api.py:9-12` docstring, the `scan_continuation.py` docstring (stop claiming `grad` through the while engine)
- Test: `tests/test_differentiable_root.py`, `tests/test_functional_api.py`

- [ ] **Step 1:** Write the tests:
  - `test_grad_through_scan_loop`: for `u − a·p`, `jax.grad` of `sum(states * valid)` under `loop="scan"` equals `jax.jacfwd` under `loop="while"` (rtol `1e-5` in x64).
  - `test_grad_through_while_loop_raises_clear_error`: `pytest.raises(ValueError)`.
- [ ] **Step 2:** Implement. Write every cell of the matrix from a test that exercises it; a cell without a test reads "untested".
- [ ] **Step 3:** Commit.

### Task 4.2: Repair the differentiation boundaries

**Files:**
- Modify: `src/jaxcont/bifurcations/branch_switch.py:40` (crossing tangent from a bordered null-space solve: `[f_u f_p; t_inᵀ; r_kᵀ]`, with `r_k` the unit vector maximizing the conditioning; no full rectangular SVD inside the differentiated residual)
- Modify: `src/jaxcont/bifurcations/curves.py:135,221` (thread `q0` through `theta = (args, q0)`)
- Modify: `src/jaxcont/bifurcations/hopf_normal_form.py:241` (left eigenvector from the bordered complex system `[[Aᵀ + iωI, q̄], [qᵀ, 0]]`)
- Test: `tests/test_branch_switch.py`, `tests/test_curves.py`, `tests/test_hopf_normal_form.py`

- [ ] **Step 1:** Write the tests:
  - `test_branch_switch_grad_when_augmented_jacobian_depends_on_args`: `f = [u₁(p − u₁), u₂ − a·u₁]`; `jax.grad` with respect to `a` of the seeded tangent's first component is finite and matches central differences (x64, rtol `1e-4`).
  - `test_fold_curve_seed_grad_wrt_start_parameter`: finite, and matches central differences.
  - `test_l1_grad_with_repeated_stable_modes`: Hopf normal form plus two identical modes `z′ = −a·z`; `jax.grad(l1)(a)` is finite and ≈ 0.
- [ ] **Step 2:** Implement each → commit per file.

### Task 4.3: `l1` convention

**Files:**
- Modify: `src/jaxcont/bifurcations/hopf_normal_form.py:267` (denominator `2·ω0`, MatCont/Kuznetsov eq. 3.20 with unit-norm `q`)
- Modify: `examples/MatCont/python_cases/equilibrium.py:282` (delete the `2.0 *` adapter)
- Modify: `tests/test_hopf_normal_form.py` (expected `−2.0` for `ż = (μ+i)z − z|z|²`), `examples/BifurcationKit/04_hopf_normal_form.jl` comparison note, `CHANGELOG.md` (Breaking)

- [ ] **Step 1:** Change the test expectation, run it, and see it FAIL.
- [ ] **Step 2:** Change the denominator; scale `Hopf.l1_tolerance`'s doc accordingly.
- [ ] **Step 3:** Run the MatCont slow suite; MC-EQ-003 must pass without the adapter → commit.

### Task 4.4: Real-derivative normal forms

**Files:**
- Modify: `src/jaxcont/bifurcations/hopf_normal_form.py` (build `B(x, y)` and `C(x, y, z)` from real `jvp`s on `Re`/`Im` parts using multilinearity: `B(a + ib, c + id) = B(a,c) − B(b,d) + i(B(a,d) + B(b,c))`)
- Modify: `src/jaxcont/bifurcations/events.py` (`Hopf(classify: bool = True)`; when `False`, skip `l1`)
- Test: `tests/test_hopf_normal_form.py`

- [ ] **Step 1:** Write `test_l1_with_non_holomorphic_rhs`: the textbook system written with `jnp.maximum(r2, 0.0)`, where it is inactive near the Hopf point, gives the same `l1` as the polynomial form.
- [ ] **Step 2:** Implement → commit.

---

## Phase 5: Smaller numerics and hygiene (M3 arclength, M12, M13, low items)

### Task 5.1: Weighted arclength

- **Files:** `core/scan_continuation.py` (`ContinuationPar.theta: Optional[float] = None` → the state part weighted by `1/n_state` when `None`, as AUTO does), `tests/test_pseudo_arclength.py`.
- [ ] Test `test_periodic_step_count_independent_of_mesh`: the same orbit branch with `ntst = 10` and `ntst = 40` takes accepted-step counts within 20 % of each other. Implement → commit.

### Task 5.2: `analyze_stability` and `pd_switch` scale fixes

- **Files:** `stability/eigenvalue.py:47` (dominant by `argmax(Re)`), `bifurcations/pd_switch.py:104,187` (measure on `u − mean(u)`), plus tests.
- [ ] Tests:
  - `test_dominant_eigenvalue_is_max_real_part`: `[-5, 1] → 1`.
  - `test_pd_switch_invariant_under_translation`: the system shifted by `1e4` still switches.
- [ ] Implement → commit.

### Task 5.3: Remove the legacy surface

- **Files:** delete `solvers/corrector.py`, `solvers/newton.py`, `problems/equilibrium.py`, `problems/bvp.py`, and the `Config` class in `utils/config.py`. Keep its installation-diagnostics functions; `examples/check_installation.py` uses them. Prune `__init__.py` exports and `docs/source/api*.rst`.
- [ ] Step 1: `grep -rn "NewtonSolver\|Corrector\|EquilibriumProblem\|BoundaryValueProblem\|ContinuationProblem\|\bConfig\b" src tests examples docs/source`, and migrate every hit.
- [ ] Step 2: Delete the modules and their tests (`tests/test_newton.py`, plus the matching parts of `tests/test_solver_wiring.py`).
- [ ] Step 3: Suite and Sphinx build (`make -C docs html SPHINXOPTS=-W`) green → commit `refactor!: remove legacy OO surface`.

### Task 5.4: Strengthen the weak tests and gate the release

- [ ] Fix `test_curves.py:116`: assert `valid.sum() > 10` plus a residual bound.
- [ ] Fix `test_bifurcation_workflow.py:59`: assert an exact event count.
- [ ] Make `publish.yml` `needs:` a job that runs `pytest` on the built wheel (`pip install dist/*.whl && pytest`).
- [ ] Update the README capability list (branch switching is shipped) and link `transformations.rst`.
- [ ] Commit.

---

## Phase 6: Follow-ups that need a short design spec first

These are not task-specified, because each has a real design choice. Write a spec in `docs/superpowers/specs/` before planning.

1. **6.1 Trace-safe event scanning.** Compute all test values and bracket masks inside the jitted program (fixed size), and keep only `refine` eager, so that `events=[...]` works under `vmap` for detection (not refinement).
2. **6.2 Minimally augmented fold/Hopf curves.** Bordered systems instead of the 2n+1 / 3n+2 fully extended ones. This becomes possible after Task 3.3's aux hook, which carries the bordering vectors.
3. **6.3 Accurate Floquet multipliers.** Periodic Schur or cyclic-reduction instead of the plain propagator product, for strongly stable or unstable cycles.
4. **6.4 Hopf curves through BT.** Use `κ = ω²` as the unknown so the curve continues into the neutral-saddle part and BT becomes a sign change of `κ`.

## Phase 7: Scaling seam (M10), spec only

Change `LinearSolver`'s input from a materialized matrix to `(matvec, diag_blocks | None)`. Route `differentiable_root`, Floquet, and PRC through it. Add a condensed-collocation solver (eliminate the interior collocation unknowns per interval, then solve the `ntst`-block cyclic system). Prerequisite for BVP/homoclinic work and for `n ≥ 50` systems.

---

## Suggested release grouping

| Release | Phases | Breaking? |
|---|---|---|
| v0.5.0 | 0, 1, 2, 3, 4.3 | Yes: engine signature, `EventHit.status`, `PeriodicArgs`, `l1` value, dtype defaults |
| v0.5.x | 4.1, 4.2, 4.4, 5 | Task 5.3 removes exports |
| v0.6.0 | 6.x as specced | Depends on the specs |
| later | 7 | Internal solver protocol |
