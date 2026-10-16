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

