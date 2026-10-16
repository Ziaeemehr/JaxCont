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

