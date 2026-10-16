# Task 0.2 report

Implemented dtype-derived tolerances.
- ContinuationPar.newton_tol: Optional[float]=None; resolved in `_call_scan` via default_tol(dtype).
- implicit.differentiable_root(_checked), fold_solve, hopf_normal_form, codim2, curves, branch_switch, prc.dprc_curve: tol default None (resolved from x0/x_star dtype eagerly).
- events.py: detect_events tolerance default None; api no longer passes 1e-6; PD/NS refine use `collocation_tol` (new helper in utils/precision.py: max(default_tol(dtype), default_tol(float32))) because stiff collocation systems keep a ~1e-6 residual floor even in float64 (found via examples 20 / test_registered_torbpc_renderer_reports_known_failure under X64=1). codim2_events: `_refine_tol`.
- periodic_orbit_problem seeds with collocation_tol; periodic-specific newton_tol=1e-5 advice removed from docstring and examples 08/09/10/22 and MatCont radial case (verified examples 08/09/10/22 still run correctly in float64). The MatCont torBPC case keeps explicit newton_tol=1e-5 (float64 corrector floor ~1e-6).
- utils/__init__.py: viz names now lazy (PEP 562) to break an import cycle introduced by importing utils.precision from the engine.
- quickstart.rst Precision section; CHANGELOG 0.5.0 Breaking entry.
- Tests: tests/test_precision.py (two new tests). RED evidence: old 1e-6 default gives max residual 8.9e-07 under x64 (> 1e-9).
- Test edits: test_codim2 cusp/fold cross-check passes explicit tol=1e-8 (degenerate fold stops early at the f32 default); slow test_adaptive_handles_difficult_regions passes explicit newton_tol=1e-6.
Results: default suite passes under X64=0 and X64=1. Slow: X64=0 and X64=1 only test_adaptx_period_doubling_switch_matches_matcont fails (also fails on the untouched baseline, both precisions).

## Fix round 1
1. Root cause of the "float64 floor": measured Newton residual history of `periodic_orbit_problem` in float64 (script hist.py, scratchpad). No float32 artifact: trajectory, D/E/gauss, p0 and period are float64; the seed dtype is float64.
   - Hopf normal form (ntst=10, ncol=4): residual 1.1e-1, 7.7e-5, 3.1e-10, 4.5e-15 (floor ~4e-15, cond 3e2).
   - torBPC seed: residual 1.7e-4, 4.3e-7, 2.8e-4 (step blows up), 7e-5, ... hovering 2e-7 to 2e-6 after 12 iterations; Jacobian cond 1.6e8 (small orbit near a Hopf point). So it is genuine conditioning for that case, not a general floor.
   Kept one helper, `collocation_tol`, with a docstring recording this evidence (no longer claiming a generic 1e-6 floor).
2. PD/NS refine: the parameter-bracket width is now `default_tol(dtype)` (1e-9 in float64); the corrector check uses `collocation_tol`. Previously both shared one value.
3. Minors: replaced the stale "~3.4e-6 float32 floor" comment in periodic.py residual with a one-line invariant; added `test_pd_ns_refine_locates_event_tightly_in_float64` (skipped under float32; PD and NS fixtures locate p within 1e-6).
Tests: default suite passes under JAXCONT_TEST_X64=1 and =0 (one skip in float32).

## Fix round 2
- Deleted `collocation_tol` (and its docstring). Periodic seeds and PD/NS correctors use `default_tol(dtype)`.
- `periodic_orbit_problem(..., *, tol=None)` overrides the seed tolerance; PD/NS events gained `corrector_tol: Optional[float] = None`. The torBPC MatCont case passes `tol=1e-5` and `corrector_tol=1e-5` with a comment (condition number ~1.6e8), plus newton_tol=1e-5.
- Examples 08/09/10/22 rerun in float64 with defaults: PD/NS errors now 1.4e-8 / 2.2e-8 / n.a. / 3e-6.
- Location test uses named constant `EVENT_ALPHA` (fixtures expose none).
- Default suite passes under X64=0 and X64=1. Slow suite: X64=0 and X64=1 both fail only test_adaptx_period_doubling_switch_matches_matcont, with RuntimeError "found no adaptx PD crossing" at pd_switch.py:96, identical to the baseline failure.
