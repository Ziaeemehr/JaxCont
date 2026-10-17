# JaxCont Project Review — 2026-10

**Reviewers:** Claude Opus 5.5 (Claude Code) + an independent OpenAI Codex pass (read-only), findings merged and cross-checked.
**Scope:** design and implementation of `src/jaxcont` at `2497820` (v0.4.1.dev), checked against the project's own goals (README, `notes/ROADMAP.md`, `notes/ARCHITECTURE.md`).
**Test baseline:** the default suite passes, 374 tests, 0 failures (CPU, JAX 0.11.2, Python 3.12, float32).
**Remediation plan:** [docs/superpowers/plans/2026-10-16-review-remediation.md](../docs/superpowers/plans/2026-10-16-review-remediation.md)

Evidence tags:
- **[run]**: reproduced by executing code (probe scripts in the review session).
- **[read]**: confirmed by reading the source; the failure follows directly from the code.
- **[suspected]**: plausible from the code but not reproduced.

---

## 1. Verdict against the project's goals

ARCHITECTURE.md §1 sets the goals: pure functions, autodiff instead of finite differences, composable objects, pluggable solvers, and "lean into JAX" (`jit`/`vmap`/`grad` across the whole analysis).

| Goal | State | Main gap |
|---|---|---|
| Correct continuation (equilibria) | Good | Step-control edge cases, silent seed failure |
| Correct event detection | **Weak** | Discontinuous test functions, failed refinements published as events, parameter-only dedup |
| Periodic orbits | Fair | Fixed phase reference, unchecked construction, `args` not passed to the RHS, not a valid pytree |
| `jit` / `vmap` | Good for the engine | Public API recompiles on every eager call; events are eager-only |
| `grad` | **Partly overstated** | Reverse mode through `continuation()` is impossible (dynamic `while_loop`); only forward mode and the point solvers support it |
| Precision | **Structural risk** | float32 everywhere; much code and many comments exist to work around the float32 floor |
| Scaling (BVP, large n) | Not ready | Dense Jacobians everywhere; the solver seam takes materialized matrices |
| API hygiene | Fair | Legacy classes (`Config`, `Corrector`, `NewtonSolver`, `EquilibriumProblem`, `BoundaryValueProblem` stub) still exported |

The core design (a pure residual, a whole-loop `lax.while_loop` engine over fixed buffers, and extended systems solved with `custom_vjp` implicit differentiation) is sound and worth keeping. Most problems sit in the layers around it: event orchestration, problem construction, result assembly, and precision policy.

---

## 2. Findings

### 2.1 Critical / high

**H1. Spectral test functions are discontinuous, so they invent and miss bifurcations.** [run] for Hopf, [read] for the others.
`Hopf.test_function` (`bifurcations/events.py:158`), `BP.test_function` (`events.py:232`), and the BT/ZH/HH tests in `codim2_events.py` re-select "the candidate nearest critical" independently at every point. When two modes trade places, the selected value jumps sign without any mode crossing.
- Reproduced: a 4-D system with a fixed pair at Re = −0.1 and a moving pair at Re = p yields a spurious `hopf` at p = 0.1625 (`converged=False`) next to the real one at p = 0.
- PD/NS (`events.py:392`, `:518`) do the opposite: they track a single multiplier forever, so a crossing by any *other* multiplier is missed. After tracking starts they also drop the real/complex condition, so a complex pair crossing Re(μ) = −1 can be labelled PD.
- Refinement is not checked for locality: Newton can converge to a genuine bifurcation outside the bracket and report it there.

**H2. Failed refinements are published as events.** [run]
`Fold.refine`, `Hopf.refine`, and every codim-2 `refine` return a hit of the named kind at `right.p` with `info["converged"]=False` instead of dropping it. Plots, `result.events`, and event counts do not filter these. (The spurious Hopf in H1 is exactly such a hit.)

**H3. Event orchestration has three structural defects.** [read]
- Exact zeros are lost: `test_vals[i] * test_vals[i+1] < 0` (`events.py:320`) misses a test value of exactly 0 at a sample, which is common with symmetric problems and fixed steps.
- Dedup by parameter only (`events.py:329`): hits of the same kind within `2*settings.ds` in **p** are merged even when they lie on different sheets of the branch (after a fold the branch revisits the same p). It also uses the *initial* `ds`, not the adaptive one.
- Python-level per-point loops with `float()` on device scalars mean thousands of host syncs per branch.

**H4. A failed seed is published as a valid branch.** [run]
`pseudo_arclength_scan`/`natural_scan` fall back to the raw `u0` when seed correction fails (`core/scan_continuation.py:223`), still count it in `n_valid`, and the traced `valid` mask marks it valid. Reproduced: `u² + p` at p0 = 1 (no real root) returns a 1-point "branch" with residual 1.0, with no error and no status.

**H5. Periodic-orbit construction accepts degenerate or unconverged orbits.** [run]
`periodic_orbit_problem` (`problems/periodic.py:85`, `:167`) assumes `t_trajectory` starts at 0, calls the *unchecked* `differentiable_root`, and never validates period > 0 or non-zero amplitude. Reproduced: the same circular orbit with `t + 100` converges to **period ≈ 9.5e-7**, a collapsed solution that is returned silently.

**H6. A periodic `BifProblem` is not a valid pytree, and its RHS never sees `args`.** [run]
`args = (coll_guess, uref_prime_coll, f, mesh)` puts a Python function in the dynamic children, so `jax.jit(g)(periodic_problem)` raises `TypeError`. Separately, the raw RHS is always called with `args=None` (`periodic.py:123`, `floquet.py`, `prc.py`), so periodic problems cannot `vmap`/`grad` over model arguments, which is the axis the README sells for equilibria.

**H7. Every eager `continuation()` call recompiles the engine.** [run]
`api.py:491` builds a fresh `rhs2 = lambda ...` per call and passes it as a static argument to the jitted engine. Measured: the jit cache grows 1 → 2 → 3 over three identical calls, ~0.5 s per call for a 1-D problem, versus 0.1 ms for a warmed engine call with a stable `f`. `args` is also baked into the program as a constant, so new `args` values always recompile.

**H8. Reverse-mode `grad` through `continuation()` is impossible, and the docs claim it works.** [run]
`lax.while_loop` with a dynamic trip count has no reverse-mode rule. Reproduced: `jax.grad` through `pseudo_arclength_scan` raises `ValueError: Reverse-mode differentiation does not work for lax.while_loop`; `jax.jacfwd` works. `api.py:9-12`, the scan docstring ("`grad`-friendly"), and the README imply otherwise. Reverse mode exists only for the extended-system point solvers (`fold_point`, `hopf_point`, codim-2), which is a real and valuable capability but a different one.

**H9. Differentiation boundaries that break or will break.** [read] / [suspected]
- `branch_switch` differentiates a full SVD of a rectangular `n×(n+1)` matrix inside the residual (`bifurcations/branch_switch.py:40`, `:109`); JAX has no derivative for that. Existing tests pick an `args` that leaves the augmented Jacobian unchanged, so they never hit it. [read]
- `fold_curve_problem`/`hopf_curve_problem` capture `q0 = p_guess[free]` in a closure passed into the `custom_vjp` solve (`curves.py:135`, `:221`); differentiating through it either leaks a tracer or silently drops that dependency. [suspected]
- `lyapunov_coefficient` takes the left eigenvector from a full SVD (`hopf_normal_form.py:241`); repeated singular values elsewhere in the spectrum make its gradient NaN. [suspected]

**H10. `differentiable_root` always takes at least one Newton step.** [read]
The initial `done` is `False` (`solvers/implicit.py:69`). For an exactly converged seed with a singular Jacobian (a symmetric bracket whose midpoint is the pitchfork itself, refined through the fold extended system used by `BP`), the solve with `J dx = 0` can destroy a correct seed. The scan correctors already check the initial residual; this one does not.

### 2.2 Medium

**M1. Float32 by default is a design-level liability.** [run]
`jax_enable_x64` is never enabled. The roadmap, comments, and tests show repeated failures from tolerances near the float32 floor (issues #9, #12, #14; `newton_tol=1e-5` for periodic orbits; PD/NS "hardware-dependent" false negatives; `near_unit_circle` windows). Floquet multipliers from a product of interval propagators lose all information below ~1e-7 of the largest one. MatCont, AUTO, and BifurcationKit all work in float64. Recommendation: float64 as the documented default for analysis, float32 as an opt-in performance mode, and every tolerance expressed relative to `jnp.finfo(dtype).eps`.

**M2. Lyapunov coefficient normalization differs from the formula it cites.** [run]
The docstring cites Kuznetsov eq. 3.20 (unit-norm `q`, denominator `2ω`); the code divides by `4ω` (`hopf_normal_form.py:267`). For `ż = (μ+i)z − z|z|²` it returns −1.0, while Kuznetsov's unit-norm formula and MatCont give −2. The MatCont validation adapter multiplies by 2 to compensate (`examples/MatCont/python_cases/equilibrium.py:282`). The sign, and therefore criticality, is unaffected; magnitudes, sensitivities, and the absolute `l1_tolerance` are affected. Pick one convention (recommended: MatCont-compatible `2ω`), document it, and remove the adapter factor.

**M3. Step control edge cases.** [run] / [read]
- After a failure, the stall test runs on the *shrunk* step (`scan_continuation.py:292`): with `ds = 0.015` and `ds_min = 0.01`, one failure clamps to 0.01 and stops without ever trying 0.01. [run]
- The first step uses an unclipped `ds0` (`scan_continuation.py:236`), even though the `_adapt_ds` docstring says it is clamped. [read]
- `max_steps` bounds *accepted* points (`idx` counts accepts), contrary to the module docstring, which says rejected steps consume iterations. [read]
- The arclength constraint is the unweighted Euclidean norm, so for collocation states (hundreds of components) `ds` effectively changes meaning with mesh size. Standard practice is to weight the state part, for example by 1/N or an L2 integral norm.

**M4. The periodic phase reference and the Hopf-curve eigenvector anchor are frozen for the whole branch.** [read]
The integral phase condition uses the initial *guess* as reference for every step. The Hopf-curve phase row uses the seed eigenvector, recomputed at the fixed `(u*, p*)` for each `q`. The docstring says fixed-shape buffers rule out re-anchoring, but that is not true: re-anchoring changes values, not shapes. The engine lacks a carried "auxiliary state" updated after each accepted step. That one mechanism would fix the periodic reference, the Hopf-curve anchor, and enable minimally augmented (bordered) fold/Hopf systems later. (A probe on a rotationally symmetric orbit did not fail, because the symmetry hides the problem; non-symmetric orbits are the risk.)

**M5. The trivial Floquet multiplier is identified as "closest to +1".** [read]
`floquet.py:75`. Near a fold of cycles or a weakly unstable mode, a physical multiplier at 1.00001 can be discarded instead of a numerically perturbed trivial one at 0.9999, so the orbit is labelled stable. Identify the trivial mode from the eigenvector aligned with `f(x0)` (Φ·f(x0) = f(x0)).

**M6. `BP` uses an absolute rank threshold.** [read]
`rank_tolerance = 1e-4` on the smallest singular value of `[f_u | f_p]` (`events.py:255`). `10⁻⁶·(u² + p)` (a plain fold) is classified as a BP. Use a relative criterion, scaled by the largest singular value or by row/column scaling. A standard alternative test function is det of the bordered tangent matrix `[f_u f_p; tᵀ]`.

**M7. Result-object semantics.** [run] / [read]
- Under `jit`/`vmap`, `Branch.n_valid` returns the buffer length: 401 reported versus 16 real points. [run]
- `at_param` searches the padding. [read]
- `bothside=True`: the reversed leg's tangents are not negated, and event `index` values stay relative to each leg, so they point at the wrong points of the stitched branch. [read]
- `ScanResult` has no termination status (reached end / stalled / non-finite / max steps / seed failed). [read]

**M8. `periodic_orbit_problem`, `prc_curve`, `dprc_curve`, and `Collocation` are not exported at top level**, while legacy classes are. [read]

**M9. Python-level diagnostics in eager mode.** [read]
`_assemble_leg` builds `convergence_info` with `bool()`/`int()`/`float()` per point (`api.py:407`), and `detect_events` does the same per test function. One `np.asarray` transfer would replace thousands of syncs.

**M10. Dense linear algebra end to end.** [read]
The `LinearSolver` protocol receives a materialized matrix, and `differentiable_root` and Floquet bypass it (`jnp.linalg.solve` directly). Collocation with `ntst = 100`, `ncol = 4`, `n = 10` means 5,001 unknowns and a dense LU per Newton iteration. Before BVP/large-n work, the seam needs JVP/linear-operator inputs and a block (condensed) collocation solve.

**M11. Normal forms require a holomorphic RHS.** [read]
`lyapunov_coefficient` feeds complex states into the user RHS. A real-smooth RHS that uses `maximum`/`where`/`clip` breaks or silently gives wrong values, and `Hopf.refine` always calls it. Use real directional derivatives and extend the multilinear forms to complex arguments algebraically, and allow location without classification.

**M12. `analyze_stability` reports `eigenvalues[0]` as "dominant"** (unsorted) and uses Python control flow. [read]

**M13. `pd_switch` doubling measure depends on the coordinate origin** (absolute state norm). [read]

### 2.3 Low / hygiene

- **Legacy surface.** `Config`, `Corrector`, `NewtonSolver`, `EquilibriumProblem` (whose `parameter_derivative` uses central differences with `eps = 1e-7` in float32, which is pure noise), the `BoundaryValueProblem` stub (always raises `NotImplementedError`), and `ContinuationProblem` are all exported from `jaxcont`. This contradicts ARCHITECTURE principle 2 and the project's own "remove, don't deprecate" rule.
- **Weak tests.** Codex identified tests that pass for the wrong behaviour:
  - `test_curves.py:116` asserts `n_valid > 0`, which is always true under tracing.
  - `test_branch_switch.py:72` uses a gradient parameter that leaves the augmented Jacobian unchanged.
  - `test_floquet.py:76` uses `atol = 1e-5` against an expected multiplier of ≈ 3.5e-6, so zero passes.
  - `test_bifurcation_workflow.py:59` passes when no event is detected.
- **Docs.** The README still lists branch switching as future work, and there is no capability × transformation matrix.
- **Release workflow.** `publish.yml` does not depend on tests passing.
- **Comment density.** Many comments are long incident narratives ("found during final whole-branch review…"). These belong in the CHANGELOG or design specs; source comments should state the invariant.

---

## 3. What is solid (keep)

- The whole-loop bordered pseudo-arclength engine and its fold-safe `(n+1)×(n+1)` corrector.
- The `differentiable_root` + `custom_vjp` implicit-diff primitive, and its reuse across fold/Hopf/codim-2 solvers.
- Collocation scaling (`T·h`), the periodic continuity closure, monodromy via interval propagators, the iPRC adjoint direction and `Z·f = 2π/T` normalization (all checked by Codex).
- The fold normal-form coefficient via a bordered adjoint solve, and the BT Jordan-chain equations.
- The external validation culture (MatCont/BifurcationKit cases).

---

## 4. Merged priority list

1. Make failure explicit: seed status, termination status, validated vs. failed events. (H2, H4, H5, M7)
2. Rebuild event detection on continuous test functions with mode tracking and locality checks. (H1, H3, M6)
3. Precision policy: float64 default for analysis, eps-relative tolerances. (M1)
4. Stable compiled kernel: static `f` plus dynamic `args`; batched diagnostics. (H7, M9)
5. Periodic problem structure: real pytree, user `args`, validated construction, per-step phase reference. (H5, H6, M4, M5)
6. Honest differentiation story: transformation matrix, fixes at the SVD/closure boundaries, `differentiable_root` initial check. (H8, H9, H10, M11)
7. Small numeric fixes: step control and the `l1` convention. (M2, M3)
8. API cleanup: remove the legacy surface, export the periodic/PRC API, docs. (M8, M12, low items)
9. Scaling seam: linear operators and condensed collocation, as a design spec before the BVP work. (M10)
