# MatCont Validation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible MatCont/analytic validation suite for every currently supported JaxCont numerical feature, with explicit MatCont-only markers for unsupported families.

**Architecture:** `examples/MatCont` is a small Python package and MATLAB script collection. A JSON registry declares cases, support status, producers, references, and tolerances; focused Python modules implement registry loading, comparison primitives, case execution, and the CLI. MATLAB producers export normalized CSV/JSON artifacts that can be regenerated locally, while committed reviewed references allow offline JaxCont validation.

**Tech Stack:** Python 3.9+, JAX/JaxCont, NumPy, SciPy, pytest, MATLAB R2020a, MatCont 7.6, JSON and CSV.

## Global Constraints

- Work only on `feat/matcont-validation-suite`, based on `6dd4e620b18835334f953dbbddfe78ab46feb830`; do not merge or push.
- Canonical suite path is `examples/MatCont`; `validation/VALIDATION_EXAMPLES.md` remains only as a compatibility pointer.
- Default executables are `/home/ziaee/prog/Matlab/R2020a/bin/matlab` and `/home/ziaee/prog/MatCont/MatCont7p6`, overridable by CLI options or `MATLAB_BIN`/`MATCONT_ROOT`.
- Commit only normalized CSV/JSON references and human documentation; never commit `.mat`, generated plots, or frozen JaxCont outputs.
- Supported numerical mismatches fail visibly. Never loosen a tolerance or change JaxCont library code merely to make validation pass.
- Unsupported scripts are excluded by default and report `UNSUPPORTED_BY_JAXCONT`; a MATLAB wrapper failure is still an error.
- Compare branches by interpolation on monotone segments or scaled point-set distance, events by unique kind/location matching, and spectra by assignment after excluding the trivial multiplier nearest `+1`.
- Every Python behavior change follows RED/GREEN TDD with focused tests before implementation.

---

### Task 1: Registry, Comparison Primitives, and CLI Skeleton

**Files:**
- Create: `examples/MatCont/__init__.py`
- Create: `examples/MatCont/cases.json`
- Create: `examples/MatCont/registry.py`
- Create: `examples/MatCont/compare.py`
- Create: `examples/MatCont/run_validation.py`
- Create: `tests/test_matcont_suite.py`

**Interfaces:**
- Produces `load_registry(path: Path | None = None) -> dict`, `select_cases(registry, ids=None, include_unsupported=False) -> list[dict]`.
- Produces `scaled_close`, `interpolate_observable`, `match_events`, and `match_spectrum` returning comparison diagnostics or raising `ValidationMismatch`.
- Produces CLI options `--case`, `--regenerate-matcont`, `--verify-references`, `--include-unsupported`, `--matlab-bin`, and `--matcont-root`.

- [ ] **Step 1: Write failing registry and comparator tests**

```python
def test_default_selection_excludes_unsupported():
    selected = select_cases(load_registry())
    assert selected
    assert all(case["support"] != "unsupported" for case in selected)

def test_match_events_rejects_duplicate_reuse():
    reference = [{"kind": "LP", "parameter": -1.0}, {"kind": "LP", "parameter": 1.0}]
    actual = [{"kind": "LP", "parameter": -1.0}]
    with pytest.raises(ValidationMismatch):
        match_events(actual, reference, atol=1e-3)

def test_match_spectrum_ignores_only_one_trivial_multiplier():
    actual = np.array([1.0, 0.5, -1.0])
    reference = np.array([1.0, -1.0, 0.5])
    assert match_spectrum(actual, reference, atol=1e-8)["max_error"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run RED**

Run: `python3 -m pytest tests/test_matcont_suite.py -q`

Expected: import failure because the suite modules do not exist.

- [ ] **Step 3: Implement minimal focused modules and registry schema**

Registry entries must contain `id`, `title`, `support`, `features`, `python`, `matlab`, `references`, `manual_source`, and `tolerances`. Use `scipy.optimize.linear_sum_assignment` for unique event/spectrum matching and return maximum absolute errors plus assignments.

- [ ] **Step 4: Run GREEN and CLI help smoke test**

Run: `python3 -m pytest tests/test_matcont_suite.py -q`

Run: `python3 -m examples.MatCont.run_validation --help`

Expected: tests pass and help lists all six options.

- [ ] **Step 5: Commit**

```bash
git add examples/MatCont tests/test_matcont_suite.py
git commit -m "feat: add MatCont validation registry and comparators"
```

### Task 2: Equilibrium, Transform, and Codimension-Two Python Validators

**Files:**
- Create: `examples/MatCont/python_cases/equilibrium.py`
- Create: `examples/MatCont/python_cases/transforms.py`
- Create: `examples/MatCont/python_cases/codim2.py`
- Create: `examples/MatCont/python_cases/__init__.py`
- Modify: `examples/MatCont/cases.json`
- Modify: `tests/test_matcont_suite.py`

**Interfaces:**
- Each case exposes `run_<case_id>() -> CaseResult`, using a shared `CaseResult` dataclass containing `case_id`, `checks`, `observations`, and `artifacts`.
- Cases: `MC-EQ-001` cubic S-curve; `MC-EQ-002` Van der Pol; `MC-EQ-003` adaptive-control Hopf; `MC-JAX-001` transforms; `MC-C2-001` shifted CP/BT/GH/ZH/HH direct solvers.

- [ ] **Step 1: Write failing analytic-case tests**

```python
def test_cubic_case_finds_both_analytic_folds():
    result = run_cubic_fold()
    assert result.checks["fold_count"] == 2
    assert result.checks["max_fold_error"] < 5e-4

def test_codim2_case_recovers_all_shifted_points():
    result = run_codim2_points()
    assert result.checks["all_converged"]
    assert result.checks["max_parameter_error"] < 1e-3
    assert result.checks["bt_bifurcationkit_error"] < 1e-3
```

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q -k 'cubic or codim2 or transform or vanderpol or adaptive'`

Expected: failures because case modules and `CaseResult` do not exist.

- [ ] **Step 3: Implement cases using current public APIs**

Use `jc.continuation`, `Fold`, `Hopf`, `fold_point`, `hopf_point`, `fold_coefficient`, `lyapunov_coefficient`, and direct `cusp_point`, `bogdanov_takens_point`, `generalized_hopf_point`, `zero_hopf_point`, `double_hopf_point` APIs. Reuse the shifted equations and Lorenz-84 BT constants from current tests without importing test modules. Check gradients against hand-derived values and centered finite differences.

- [ ] **Step 4: Run GREEN**

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q -k 'cubic or codim2 or transform or vanderpol or adaptive'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/MatCont/python_cases examples/MatCont/cases.json tests/test_matcont_suite.py
git commit -m "feat: add equilibrium and codim-2 validation cases"
```

### Task 3: Periodic-Orbit Validators and MATLAB Producers

**Files:**
- Create: `examples/MatCont/python_cases/periodic.py`
- Create: `examples/MatCont/matlab/setup_matcont.m`
- Create: `examples/MatCont/matlab/export_equilibrium_run.m`
- Create: `examples/MatCont/matlab/export_cycle_run.m`
- Create: `examples/MatCont/matlab/run_supported.m`
- Create: `examples/MatCont/matlab/systems/*.m`
- Modify: `examples/MatCont/cases.json`
- Modify: `tests/test_matcont_suite.py`

**Interfaces:**
- `run_radial_cycle() -> CaseResult` checks radius, period, collocation residual, stability, and exact Floquet multipliers.
- `run_torbpc_cycle(reference_dir: Path) -> CaseResult` compares LPC/NS/PD locations, periods, extrema, and multipliers.
- MATLAB producers write `<case>_branch.csv`, `<case>_events.csv`, `<case>_multipliers.csv`, and `<case>_metadata.json` into a caller-provided output directory.

- [ ] **Step 1: Write failing periodic and exporter-schema tests**

```python
def test_radial_cycle_matches_exact_floquet_formula():
    result = run_radial_cycle()
    assert result.checks["max_period_error"] < 5e-3
    assert result.checks["max_multiplier_error"] < 5e-3

def test_cycle_reference_has_normalized_columns(reviewed_reference_dir):
    branch = read_csv(reviewed_reference_dir / "MC-LC-002_branch.csv")
    assert {"point", "parameter", "period", "residual_norm"} <= set(branch.dtype.names)
```

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q -k 'radial or cycle_reference'`

Expected: failures because the periodic module/reference data do not exist.

- [ ] **Step 3: Implement periodic cases and MATLAB exporters**

Use fixed Gauss-Legendre collocation, exclude the trivial Floquet multiplier nearest `+1`, and match remaining spectra uniquely. For `torBPC1`, use fixed parameters from MatCont's installed test system and compare LPC `-0.5844928424`, NS `-0.5957504315`, and PD `-0.6146816596` with `2e-3` parameter tolerance.

- [ ] **Step 4: Run MATLAB producers and GREEN tests**

Run: `/home/ziaee/prog/Matlab/R2020a/bin/matlab -batch "cd('examples/MatCont/matlab'); run_supported"`

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q -k 'radial or cycle_reference'`

Expected: producers exit zero and selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/MatCont/python_cases/periodic.py examples/MatCont/matlab examples/MatCont/cases.json tests/test_matcont_suite.py
git commit -m "feat: add periodic MatCont validation producers"
```

### Task 4: Reviewed References, Unsupported Wrappers, and Documentation

**Files:**
- Create: `examples/MatCont/reference/**/*.{csv,json}`
- Create: `examples/MatCont/generated/.gitignore`
- Create: `examples/MatCont/matlab/unsupported/*.m`
- Create: `examples/MatCont/README.md`
- Modify: `examples/MatCont/run_validation.py`
- Modify: `validation/VALIDATION_EXAMPLES.md`
- Modify: `README.md`
- Modify: `examples/example_03_van_der_pol.py`
- Delete: `validation/matcont/*.m`
- Delete: `validation/matcont/systems/*.m`
- Modify: `tests/test_matcont_suite.py`

**Interfaces:**
- Unsupported wrappers cover Bratu branch switching, two-parameter LP/H/PD/LPC/NS curves, general BVP, homoclinic/heteroclinic continuation, and PRC/dPRC.
- Default CLI validates supported cases from committed references; `--include-unsupported` runs wrappers and reports `UNSUPPORTED_BY_JAXCONT` without treating absence of a JaxCont comparator as success.
- `--verify-references` compares regenerated artifacts with reviewed references and never overwrites them.

- [ ] **Step 1: Write failing CLI integration and metadata tests**

```python
def test_offline_cli_validates_supported_references(tmp_path):
    completed = run_cli("--case", "MC-EQ-001", "--reference-dir", reviewed_reference_dir)
    assert completed.returncode == 0
    assert "PASS MC-EQ-001" in completed.stdout

def test_unsupported_case_is_explicitly_reported():
    completed = run_cli("--case", "US-BP-001", "--include-unsupported", "--dry-run")
    assert "UNSUPPORTED_BY_JAXCONT" in completed.stdout
```

- [ ] **Step 2: Run RED**

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q -k 'offline_cli or unsupported_case or metadata'`

Expected: failures because reference promotion, wrappers, and completed CLI behavior do not exist.

- [ ] **Step 3: Add reviewed artifacts, wrappers, documentation, and consolidation**

Metadata must record case ID, provenance, manual section or analytic source, MATLAB/MatCont/JaxCont/Python/JAX versions, precision, mesh, solver settings, timestamp, and equation hash. Rewrite the old validation specification as a pointer to `examples/MatCont/README.md` and update repository links.

- [ ] **Step 4: Run GREEN and reference verification**

Run: `JAX_PLATFORMS=cpu python3 -m pytest tests/test_matcont_suite.py -q`

Run: `python3 -m examples.MatCont.run_validation --verify-references`

Expected: tests and supported validations pass; unsupported cases are excluded unless requested.

- [ ] **Step 5: Commit**

```bash
git add examples/MatCont validation/VALIDATION_EXAMPLES.md README.md examples/example_03_van_der_pol.py tests/test_matcont_suite.py
git commit -m "docs: consolidate MatCont validation suite"
```

### Task 5: Full Verification and Review Handoff

**Files:**
- Modify only files required to correct validation-suite defects found by verification; do not change `src/jaxcont`.

**Interfaces:**
- Produces a clean local review branch with logical commits and no merge/push.

- [ ] **Step 1: Regenerate all supported MatCont artifacts**

Run: `python3 -m examples.MatCont.run_validation --regenerate-matcont`

Expected: MATLAB exits zero for every supported producer.

- [ ] **Step 2: Verify regenerated versus reviewed references**

Run: `python3 -m examples.MatCont.run_validation --verify-references`

Expected: all reviewed artifacts agree within registry tolerances.

- [ ] **Step 3: Run supported JaxCont validation**

Run: `JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-jaxcont-validation python3 -m examples.MatCont.run_validation`

Expected: every supported case reports `PASS`.

- [ ] **Step 4: Run full project tests**

Run: `JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-jaxcont-validation python3 -m pytest -m ''`

Expected: zero failures.

- [ ] **Step 5: Inspect branch state**

Run: `git status --short --branch && git log --oneline 6dd4e62..HEAD`

Expected: clean worktree, branch `feat/matcont-validation-suite`, logical local commits, nothing pushed or merged.
