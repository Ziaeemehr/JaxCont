# Test tolerance audit (2026-10-29)

Audited the initial branch-changed file inventory in filename order through `tests/test_codim2.py`, including helpers and files without tolerance assertions. Measured the original assertion operands on CPU in `~/envs/jaxcont`, in separate processes with `JAX_ENABLE_X64=0` and `=1`; temporary pytest instrumentation and logs live outside the repository in `/tmp/jaxcont-tol-audit/`. Errors below are maximum absolute errors across all inputs/repetitions at each assertion site, measured on the host without rounding the analytic scalar reference back to float32. References are analytic normal forms, defining residuals, finite differences, serial/batched self-consistency, Python arithmetic, or the reviewed BifurcationKit BT bundle. Tolerances were checked against Newton residual budgets, conditioning, finite-difference cancellation/truncation, or roundoff. This audit changed no expected values, inputs, assertions, skips, or production code and made no git writes.

**Verdict counts (assertion sites, not parametrizations): 101 audited; 79 LOOSE tightened; 22 OK; 0 SUSPECT.** Thirteen present branch-changed files were completed. The rest is explicitly deferred below to keep this audit within budget; preliminary measurements alone do not count as an audit.

## Findings

| ID | Severity | File/line | Evidence | Fix | Status |
|---|---|---|---|---|---|

No SUSPECT tolerance/reference findings in the completed scope.

## Tightened

`eps_J` is machine epsilon for the assertion's JAX dtype (float32: `1.1920929e-7`; float64: `2.2204460e-16`). `eps_Py` is Python/NumPy float64 epsilon in both runs. `_POINT_ATOL` and `refine_atol` are `1e-5` for float32 and `1e-9` for float64, matching the actual default Newton stopping tolerance; these well-scaled normal forms admit that location/residual budget. `step_rtol = 8*eps_J`; `lane_atol = 8*eps_J` at unit problem scale. For clipping inequalities, the measured error is the excess/shortfall beyond the requested bound (zero when inside), not the distance from the bound. A zero measurement still retains a roundoff or solver budget. Every new effective bound is at least 10 times its measured error, and every effective bound is no larger than its old value.

| File/line | Old | New | Measured x32 | Measured x64 |
|---|---|---|---|---|
| [tests/test_adaptive_stepsize.py:96](../../tests/test_adaptive_stepsize.py#L96) | `ds_min * 0.99` | `ds_min * (1 - step_rtol)` | 0 | 0 |
| [tests/test_adaptive_stepsize.py:114](../../tests/test_adaptive_stepsize.py#L114) | `ds_max * 1.01` | `ds_max * (1 + step_rtol)` | 7.45e-10 | 0 |
| [tests/test_adaptive_stepsize.py:236](../../tests/test_adaptive_stepsize.py#L236) | `atol=1e-8, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 3.73e-10 | 0 |
| [tests/test_adaptive_stepsize.py:242](../../tests/test_adaptive_stepsize.py#L242) | `atol=1e-8, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 6.71e-10 | 0 |
| [tests/test_adaptive_stepsize.py:310](../../tests/test_adaptive_stepsize.py#L310) | `0.005 * 0.99` | `0.005 * (1 - step_rtol)` | 0 | 0 |
| [tests/test_adaptive_stepsize.py:312](../../tests/test_adaptive_stepsize.py#L312) | `0.1 * 1.01` | `0.1 * (1 + step_rtol)` | 1.49e-09 | 0 |
| [tests/test_benchmark_batching.py:76](../../tests/test_benchmark_batching.py#L76) | `abs=1e-12, rel=1e-6` | `abs=0, rel=8 * eps_Py` | 0 | 0 |
| [tests/test_benchmark_harness.py:62](../../tests/test_benchmark_harness.py#L62) | `abs=1e-12, rel=1e-6` | `abs=0, rel=8 * eps_Py` | 0 | 0 |
| [tests/test_bifurcations.py:51](../../tests/test_bifurcations.py#L51) | `abs=0.001, rel=0` | `abs=refine_atol, rel=0` | 0 | 0 |
| [tests/test_bifurcations.py:53](../../tests/test_bifurcations.py#L53) | `abs=0.001, rel=0` | `abs=refine_atol, rel=0` | 0 | 0 |
| [tests/test_bifurcations.py:83](../../tests/test_bifurcations.py#L83) | `atol=1e-06, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 4.47e-10 | 0 |
| [tests/test_bifurcations.py:134](../../tests/test_bifurcations.py#L134) | `atol=0.0001, rtol=1e-5` | `atol=0, rtol=32 * eps_J` | 1.49e-09 | 0 |
| [tests/test_bifurcations.py:136](../../tests/test_bifurcations.py#L136) | `atol=0.0001, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 0 | 0 |
| [tests/test_bifurcations.py:234](../../tests/test_bifurcations.py#L234) | `atol=0.0001, rtol=1e-5` | `atol=0, rtol=32 * eps_J` | 2.38e-07 | 4.44e-16 |
| [tests/test_bifurcations.py:258](../../tests/test_bifurcations.py#L258) | `atol=0.0001, rtol=1e-5` | `atol=0, rtol=32 * eps_J` | 2.38e-07 | 4.44e-16 |
| [tests/test_bifurcations.py:445](../../tests/test_bifurcations.py#L445) | `abs=1e-12, rel=1e-6` | `abs=0, rel=8 * eps_Py` | 0 | 0 |
| [tests/test_bifurcations.py:459](../../tests/test_bifurcations.py#L459) | `abs=0.001, rel=0` | `abs=1e-09 if x64 else 1e-05, rel=0` | 0 | 0 |
| [tests/test_bifurcations.py:517](../../tests/test_bifurcations.py#L517) | `0.001` | `1e-09 if x64 else 1e-05` | 1.86e-09 | 0 |
| [tests/test_bifurcations.py:814](../../tests/test_bifurcations.py#L814) | `0.001` | `2 * (1e-09 if x64 else 1e-05)` | 5.96e-08 | 1.72e-10 |
| [tests/test_bifurcations.py:884](../../tests/test_bifurcations.py#L884) | `atol=0, rtol=1e-7` | `atol=0, rtol=8 * eps_Py` | 0 | 0 |
| [tests/test_bifurcations.py:886](../../tests/test_bifurcations.py#L886) | `atol=0, rtol=1e-7` | `atol=0, rtol=8 * eps_Py` | 0 | 0 |
| [tests/test_bordered_newton.py:60](../../tests/test_bordered_newton.py#L60) | `atol=1e-06, rtol=1e-5` | `atol=1e-06, rtol=0` | 0 | 0 |
| [tests/test_bordered_newton.py:138](../../tests/test_bordered_newton.py#L138) | `atol=1e-06, rtol=1e-5` | `atol=1e-06, rtol=0` | 0 | 0 |
| [tests/test_bordered_newton.py:140](../../tests/test_bordered_newton.py#L140) | `atol=1e-06, rtol=1e-5` | `atol=1e-06, rtol=0` | 0 | 0 |
| [tests/test_bordered_newton.py:228](../../tests/test_bordered_newton.py#L228) | `atol=1e-10, rtol=1e-5` | `atol=0, rtol=32 * eps_J` | 0 | 2.22e-16 |
| [tests/test_branch_switch.py:38](../../tests/test_branch_switch.py#L38) | `abs=1e-12, rel=0.2` | `abs=0, rel=32 * eps_J` | 0 | 0 |
| [tests/test_branch_switch.py:47](../../tests/test_branch_switch.py#L47) | `0.0001` | `1e-09 if x64 else 1e-05` | 0 | 0 |
| [tests/test_branch_switch.py:63](../../tests/test_branch_switch.py#L63) | `abs=0.1, rel=0` | `abs=0, rel=32 * eps_J` | 3.73e-09 | 6.94e-18 |
| [tests/test_branch_switch.py:66](../../tests/test_branch_switch.py#L66) | `0.0001` | `1e-09 if x64 else 1e-05` | 0 | 0 |
| [tests/test_branch_switch.py:94](../../tests/test_branch_switch.py#L94) | `0.02` | `32 * eps_J * 0.05 / eps` | 0 | 0 |
| [tests/test_branch_switch.py:134](../../tests/test_branch_switch.py#L134) | `abs=0.0001, rel=0` | `abs=lane_atol, rel=0` | 0 | 0 |
| [tests/test_branch_switch.py:136](../../tests/test_branch_switch.py#L136) | `abs=0.0001, rel=0` | `abs=lane_atol, rel=0` | 0 | 0 |
| [tests/test_branch_switch.py:211](../../tests/test_branch_switch.py#L211) | `atol=1e-09, rtol=0.0001` | `atol=1e-09, rtol=0` | 1.71e-11 | 1.71e-11 |
| [tests/test_codim2.py:51](../../tests/test_codim2.py#L51) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:53](../../tests/test_codim2.py#L53) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:55](../../tests/test_codim2.py#L55) | `atol=1e-05, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:71](../../tests/test_codim2.py#L71) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:123](../../tests/test_codim2.py#L123) | `atol=0.001, rtol=1e-5` | `atol=2e-08, rtol=0` | 0 | 4.66e-13 |
| [tests/test_codim2.py:143](../../tests/test_codim2.py#L143) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 2.66e-15 |
| [tests/test_codim2.py:145](../../tests/test_codim2.py#L145) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 2.56e-13 |
| [tests/test_codim2.py:147](../../tests/test_codim2.py#L147) | `atol=1e-05, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:159](../../tests/test_codim2.py#L159) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 2.22e-16 |
| [tests/test_codim2.py:161](../../tests/test_codim2.py#L161) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 2.66e-15 |
| [tests/test_codim2.py:163](../../tests/test_codim2.py#L163) | `atol=1e-05, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 3.05e-16 |
| [tests/test_codim2.py:204](../../tests/test_codim2.py#L204) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 1.14e-13 | 7.58e-14 |
| [tests/test_codim2.py:206](../../tests/test_codim2.py#L206) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 5.64e-12 |
| [tests/test_codim2.py:208](../../tests/test_codim2.py#L208) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:235](../../tests/test_codim2.py#L235) | `0.0001` | `_POINT_ATOL` | 1.94e-25 | 1.13e-11 |
| [tests/test_codim2.py:277](../../tests/test_codim2.py#L277) | `atol=0.001, rtol=1e-5` | `atol=2 * max(1e-06, _POINT_ATOL), rtol=0` | 0 | 5.66e-14 |
| [tests/test_codim2.py:279](../../tests/test_codim2.py#L279) | `atol=0.001, rtol=1e-5` | `atol=2 * max(1e-06, _POINT_ATOL), rtol=0` | 0 | 0 |
| [tests/test_codim2.py:299](../../tests/test_codim2.py#L299) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:301](../../tests/test_codim2.py#L301) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:303](../../tests/test_codim2.py#L303) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:315](../../tests/test_codim2.py#L315) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:317](../../tests/test_codim2.py#L317) | `atol=1e-05, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:320](../../tests/test_codim2.py#L320) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 5.55e-17 | 0 |
| [tests/test_codim2.py:322](../../tests/test_codim2.py#L322) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 5.55e-17 | 0 |
| [tests/test_codim2.py:369](../../tests/test_codim2.py#L369) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 1.19e-11 |
| [tests/test_codim2.py:374](../../tests/test_codim2.py#L374) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:376](../../tests/test_codim2.py#L376) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 2.08e-13 |
| [tests/test_codim2.py:388](../../tests/test_codim2.py#L388) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:390](../../tests/test_codim2.py#L390) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:392](../../tests/test_codim2.py#L392) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 8.44e-12 |
| [tests/test_codim2.py:394](../../tests/test_codim2.py#L394) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 8.44e-12 |
| [tests/test_codim2.py:457](../../tests/test_codim2.py#L457) | `atol=0.0001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:460](../../tests/test_codim2.py#L460) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:462](../../tests/test_codim2.py#L462) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 0 | 9.33e-15 |
| [tests/test_codim2.py:478](../../tests/test_codim2.py#L478) | `atol=1e-06, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:480](../../tests/test_codim2.py#L480) | `atol=1e-06, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:489](../../tests/test_codim2.py#L489) | `atol=1e-06, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 4.77e-08 | 0 |
| [tests/test_codim2.py:491](../../tests/test_codim2.py#L491) | `atol=1e-05, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:493](../../tests/test_codim2.py#L493) | `atol=1e-05, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 0 | 0 |
| [tests/test_codim2.py:495](../../tests/test_codim2.py#L495) | `atol=1e-06, rtol=1e-5` | `atol=8 * eps_J, rtol=8 * eps_J` | 0 | 2.22e-16 |
| [tests/test_codim2.py:497](../../tests/test_codim2.py#L497) | `atol=1e-05, rtol=1e-5` | `atol=8 * eps_J, rtol=0` | 5.96e-08 | 5.55e-17 |
| [tests/test_codim2.py:507](../../tests/test_codim2.py#L507) | `atol=1e-8, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 0 | 0 |
| [tests/test_codim2.py:509](../../tests/test_codim2.py#L509) | `atol=1e-8, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 0 | 0 |
| [tests/test_codim2.py:511](../../tests/test_codim2.py#L511) | `atol=1e-8, rtol=1e-5` | `atol=0, rtol=8 * eps_J` | 0 | 0 |
| [tests/test_codim2.py:595](../../tests/test_codim2.py#L595) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 1.19e-07 | 4.16e-17 |
| [tests/test_codim2.py:597](../../tests/test_codim2.py#L597) | `atol=0.001, rtol=1e-5` | `atol=_POINT_ATOL, rtol=0` | 1.3e-08 | 3.47e-18 |

The codim-2 location and defining-equation checks use the Newton budget, not an arbitrary fit to their often much smaller errors. The codim-1/codim-2 Hopf cross-check allows twice the larger of their Newton tolerances. The cusp fold parameter cross-check uses `2e-8`: its parameter residual is bounded by `1e-8` plus the cubic state term. Pure `_normalize_omega` checks use epsilon bounds. For its unit-norm assertion, both absolute and relative epsilon terms provide a 16-epsilon budget at unit scale. The independent Lorenz-84 BT reference is the reviewed `examples/validation/reference/lorenz84/bifurcationkit/lorenz84.bt-ft` bundle (float64, Newton `1e-12`), compared with dtype conversion as in the original test.

The bordered linear solve comparison has matrix condition number approximately 4.93; 32 epsilon relative error is a conservative roundoff budget. The branch-switch derivative test with additive bias has exactly constant state `u=ds`, so its central difference has zero truncation error; its new bound is `32*eps_J*0.05/h`, with `h=1e-3`. The other branch-switch derivative check explicitly enables x64 in both environment runs: its two measured errors are `1.71e-11` and `7.85e-13`; `atol=1e-9` retains more than 10 times either and covers `h^2` truncation at `h=1e-5` plus cancellation. None of these tolerances relies on a time-discretization mesh.

## Checked OK

Errors are shown as x32 / x64. These are the 22 unchanged tolerance sites; tests with both tightened and retained assertions appear here too.

- `test_adaptive_stepsize.py::TestAdaptiveVsFixed::test_disabled_adaptive_keeps_step_constant_after_success` — `2.24e-10 / 0`; default approx bound `1e-8` is about 45 times the float32 error.
- `test_adaptive_stepsize.py::TestAdaptiveStepsizeAlgorithm::test_adapt_stepsize_fixed_mode_still_shrinks_on_failure` — `3.73e-10 / 0`; approx bound `2.5e-8` is about 67 times the float32 error.
- `test_benchmark_batching.py::{test_every_lane_matches_serial_valid_points,test_batch_permutation_preserves_results}` — the shared `_compare` assertion is one site; maximum error `0 / 0` over 72 comparisons in each precision. Eight epsilon times reference scale is a principled serial/batched roundoff allowance.
- `test_bifurcation_workflow.py::test_with_and_without_refinement` — analytic fold parameter zero, error `8.65e-6 / 0`; `1e-4` is about 12 times the float32 error and consistent with default Newton `1e-5`.
- `test_bifurcations.py::{test_hopf_refine_converges_via_extended_system,test_detect_events_finds_hopf_not_fold_synthetic_repro}` — two parameter assertions, errors `0 / 0`; `1e-6` matches the explicit or default float32 Newton budget for the extended systems.
- `test_bordered_newton.py::test_bordered_system_simple` — two residual assertions: equilibrium `0 / 0`, arclength `2.98e-8 / 9.71e-17`; `1e-6` is the explicit Newton stopping tolerance.
- `test_bordered_newton.py::test_bordered_system_nonlinear` — three sites: equilibrium `4.77e-7 / 5.71e-12`, arclength `1.79e-7 / 4.44e-16`, and `u^2=p` error `4.77e-7 / 5.71e-12`. The residual bounds match Newton `1e-6`; the existing allclose-style bound, including implicit `rtol=1e-5`, is about 96 times the float32 error.
- `test_bordered_newton.py::test_bordered_system_2d` — two defining residuals: equilibrium `0 / 0`, arclength `1.49e-8 / 5.55e-17`; explicit Newton `1e-6` justifies both bounds.
- `test_bordered_newton.py::test_bordered_system_continuation_branch` — maximum equilibrium residual `2.98e-8 / 3.94e-10` over all five steps; bound `1e-6` matches Newton.
- `test_bordered_newton.py::test_bordered_system_block_elimination` — retained vector comparison error `2.98e-8 / 5.55e-17`. Componentwise evaluation matters: the approximately `0.0824` component has error `2.98e-8`, so its effective original bound is only about 28 times that error. The scalar comparison was tightened separately.
- `test_branch_switch.py::test_switch_lands_on_the_pitchfork_second_branch` — retained continuation lower bound `0.05-1e-6`; shortfall below `0.05` is `0 / 0`, with the bound using the explicit continuation Newton budget.
- `test_codim2.py::test_cusp_agrees_with_the_codim1_fold_solver_there` — retained state comparison error `4.89e-5 / 4.88e-5`; effective bound approximately `1.02e-3`, about 21 times the error. At the cusp, the fold constraint is `3*(u-2)^2=0`: Newton residual `1e-8` allows state error `sqrt(1e-8/3)=5.77e-5`. The observed error agrees with that degenerate conditioning; it is not evidence of a solver defect.
- `test_codim2.py::{test_cusp_parameters_grad_matches_finite_difference,test_bogdanov_takens_parameters_grad_matches_finite_difference,test_generalized_hopf_parameters_grad_matches_finite_difference,test_zero_hopf_parameters_grad_matches_finite_difference,test_double_hopf_parameters_grad_matches_finite_difference}` — five central differences at `h=1e-3` of analytically translated parameter locations (derivative one). Errors respectively `1.29e-5 / 8.88e-16`, `4.67e-5 / 1.10e-13`, `4.68e-5 / 1.12e-13`, `1.66e-4 / 1.10e-13`, `1.66e-4 / 1.10e-13`; existing bounds approximately `1.01e-3` are between 6 and 79 times the float32 errors and cover subtraction roundoff at parameter scales 1–5.

Also inspected and found no numeric-tolerance assertion sites: `tests/conftest.py`, `tests/reference_bundles.py`, all tests in `test_benchmark_contracts.py`, `test_candidate_provenance.py`, and `test_check_docs.py`. Count/shape/type checks, sign tests, and exact schema-value equalities are outside numeric-tolerance verdict counts. Deleted branch files `test_bvp_placeholder.py`, `test_newton.py`, and `test_newton_jit.py` have no current assertions to audit.

## Verification

Final full-suite results (both commands exited **0**, with no failures or errors):

| Precision | Final pytest summary |
|---|---|
| `JAX_ENABLE_X64=0` | **913 passed, 1 skipped** |
| `JAX_ENABLE_X64=1` | **914 passed** |

Commands, in the requested venv:

```sh
source ~/envs/jaxcont/bin/activate
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 python -m pytest -q -n 4 tests
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -q -n 4 tests
```

A reporting-only plugin in `/tmp/jaxcont-tol-audit/` recorded the final counts because the repository's `-q` plus the command's `-q` suppress normal passing totals. It did not instrument assertions or change execution/selection. **All 79 tightened assertion sites passed in both precisions; none required reverting.** `git diff --check` also passed.

Explicit slow-module checks after tightening: float32 **14 passed in 4.08s**; float64 **14 passed in 4.10s** (`-o addopts='' tests/test_adaptive_stepsize.py`). The required full-suite commands retain repository marker defaults (`not slow and not gpu and not matlab and not julia`), so these extra runs cover the audited slow assertions.

The workspace contained concurrent benchmark edits and a commit by another actor; original assertion text in this report is anchored to `58223c2`, before that commit incorporated the two benchmark tolerance edits. `test_benchmark_reporting.py` appeared during the audit and is deferred with the additional files below. Initial instrumented measurements encountered `ValueError: unknown tool 'matcont'` in the existing pending-row test in both precisions (100 passed, 1 failed per run); that nonnumeric failure is separate from the tolerance verdicts. A subsequent intermediate snapshot also failed the warm-worker fake (dictionary versus the new `.branch.n_valid` protocol); the concurrent benchmark work updated that fake before the successful final runs. Those edits were preserved. These nonnumeric transient failures did not require any tolerance changes; the final results above apply to the stable observed working tree.

## Not yet audited

Budget stop after `tests/test_codim2.py`. Full-suite execution of later files is verification, not a tolerance audit. Resume with branch-changed files first:

- `tests/test_codim2_events.py`
- `tests/test_curves.py`
- `tests/test_differentiable_root.py`
- `tests/test_event_scan.py`
- `tests/test_examples.py`
- `tests/test_examples_migration_map.py`
- `tests/test_external_runners.py`
- `tests/test_floquet.py`
- `tests/test_functional_api.py`
- `tests/test_gallery_sources.py`
- `tests/test_gpu_smoke.py`
- `tests/test_hopf_bt_curve.py`
- `tests/test_hopf_normal_form.py`
- `tests/test_julia_exports.py`
- `tests/test_legacy_surface.py`
- `tests/test_loop1_regressions.py`
- `tests/test_loop2_regressions.py`
- `tests/test_loop3_regressions.py`
- `tests/test_loop4_regressions.py`
- `tests/test_loop5_regressions.py`
- `tests/test_loop6_regressions.py`
- `tests/test_lorenz84_cli.py`
- `tests/test_lorenz84_matcont.py`
- `tests/test_lorenz84_producer.py`
- `tests/test_lorenz84_three_tool.py`
- `tests/test_matcont_suite.py`
- `tests/test_matcont_visualization.py`
- `tests/test_minimal_curves.py`
- `tests/test_modes.py`
- `tests/test_offline_examples.py`
- `tests/test_pd_switch.py`
- `tests/test_period_doubling_neimark_sacker.py`
- `tests/test_periodic_matcont.py`
- `tests/test_periodic_orbit_continuation.py`
- `tests/test_periodic_orbit_problem.py`
- `tests/test_periodic_schur.py`
- `tests/test_periodic_validation.py`
- `tests/test_precision.py`
- `tests/test_pseudo_arclength.py`
- `tests/test_reference_promotion.py`
- `tests/test_reference_systems.py`
- `tests/test_solver_wiring.py`
- `tests/test_stability.py`
- `tests/test_validation_figures.py`
- `tests/test_validation_reports.py`
- `tests/test_validation_schema.py`
- `tests/test_viz.py`

Then remaining present `tests/*.py` files (including the benchmark reporting tests added concurrently after the initial inventory):

- `tests/__init__.py`
- `tests/test_benchmark_reporting.py`
- `tests/test_collocation.py`
- `tests/test_continuation.py`
- `tests/test_fold_normal_form.py`
- `tests/test_installation.py`
- `tests/test_prc.py`
- `tests/test_solution_persistence.py`
- `tests/test_solver_protocols.py`
- `tests/test_taxonomy.py`
- `tests/test_van_der_pol_validation.py`
- `tests/test_viz_prc.py`
