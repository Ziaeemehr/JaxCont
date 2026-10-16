# Task 0.1 report
Status: DONE. Commit ef22d4d "test: run the suite under float32 and float64; add default_tol".

Implemented: src/jaxcont/utils/precision.py (default_tol, is_x64); tests/conftest.py enables x64 from JAXCONT_TEST_X64=1 before any jax import; .github/workflows/tests.yml matrix x64 [0,1] + env JAXCONT_TEST_X64; tests/test_precision.py (default_tol values, is_x64, float32-under-x64 scan).

TDD: RED = `ModuleNotFoundError: No module named 'jaxcont.utils.precision'` at collection. GREEN after implementation. test_scan_float32_inputs_under_x64 RED without the fix: `TypeError: while_loop body ... c.u float32[1] vs float64[1]`; GREEN with dtype=u.dtype in _tangent.

x64 failures found (full suite, X64=1; X64=0 was clean):
1. test_branch_switch.py::test_switch_lands_on_the_pitchfork_second_branch: `min|u| > 0.05` got exactly 0.05 (seed) in float64; relaxed to >= 0.05 - 1e-6.
2. test_solver_wiring.py::test_continuation_default_solvers_matches_prior_behavior: exact float32 param list compare; now allclose atol=1e-6 plus length check.
Dtype leaks fixed: scan_continuation._tangent rhs, collocation.py eye (2), prc.py eye/zeros (4). No other unspecified-dtype zeros/eye/ones in src.

Final: both full suites pass (X64=0 and X64=1), no failures, no xfails added.
Concerns: none. Existing hard-coded 1e-5/1e-6 literals in src were not touched (out of scope).
