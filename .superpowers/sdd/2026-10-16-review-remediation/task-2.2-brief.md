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

