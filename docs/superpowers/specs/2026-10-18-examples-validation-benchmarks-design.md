# Examples, validation, and benchmarks design

Date: 2026-10-18. Baseline: JaxCont v0.5.0, branch `examples-validation-plan`.
Status: proposed. The owner decisions are listed at the end of the plan
(`docs/superpowers/plans/2026-10-18-examples-validation-benchmarks.md`).
Nothing described here has been implemented, installed, or re-run yet. PASS/FAIL
statements about the current suite are taken from the repository and have not
been re-run.

## Outcome and binding rules

1. **One shared system.** Each comparison runs the SAME system in JaxCont,
   MatCont, and BifurcationKit: the same equations, fixed parameters, seeds,
   windows, and observables. Lorenz-84 is the single flagship. adaptx (PD) and
   torBPC (NS) are named periodic supplements. Cubic and radial are cheap
   controls. Tutorials are short and separate from evidence.
2. **Every figure has numbers behind it.** Each validation figure has a
   `.checks.json` sidecar that records the measured value, limit, and verdict
   for every metric. Each tutorial figure has at least one numeric assertion in
   the test suite.
3. **Fair benchmarks.** First-call and compile time are kept separate from warm
   time. Resolved tolerances must be equal across tools. Accuracy is reported
   next to every timing. Batched/vmap throughput is its own category. There
   are no headline cross-tool speed claims. `demo_jit_speedup.py` is dropped,
   and `profile_continuation.py` is rewritten as a developer diagnostic.
4. **The docs are audited against 0.5.0.** Every 0.5.0 changelog entry gets a
   docs disposition and a test.
5. **External runtimes are set up early.** Julia, BifurcationKit, MATLAB
   R2024a, and MatCont 7.6 are verified in Phase 0. External tests are opt-in
   and gated by environment variables, just like the existing
   `MATLAB_BIN`/`MATCONT_ROOT`.
6. **Offline by construction.** Every tutorial, validation page, validation
   figure, and docs benchmark page runs without MATLAB and without Julia. It
   reads committed reference data that carries provenance. MATLAB and Julia
   are used only by opt-in regeneration commands and tests, and those write
   candidates, never reviewed data.
7. **Three execution loops.** The work is executed as three run → debug → fix
   loops, and the best-scoring snapshot is returned.

No placeholder values and no JaxCont output may stand in for external data. If
an external route fails, its row stays visibly FAIL or PENDING.

## Survey findings

- The 23 numbered examples mix lessons, MatCont overlays (16–20), independent
  shooting validation (14), and timing output. They are 68–324 lines long.
  Example 01 and Julia 01 are a two-fold cubic S-curve despite the "pitchfork"
  name. Example 21 is the real pitchfork. Example 03 rightly calls Van der
  Pol's crossing degenerate. Examples 08/09 are analytic PD/NS controls, not
  applied evidence.
- `examples/MatCont` already provides registry selection, schema checks,
  sheet-preserving interpolation, one-to-one event and spectrum matching,
  provenance, a regenerate/verify CLI (`--regenerate-matcont`,
  `--verify-references`), and renderers that keep failures visible. Reuse all
  of it. The registry has 18 cases. There are six committed bundles (22
  files): EQ-001/002/003, LC-001/002, and PRC-001. PRC-001 is the adaptx
  limit cycle. Historical metadata (for example JaxCont 0.2.0, MATLAB R2020a)
  is preserved as history.
- Overlays 16–20 already run offline from committed MatCont CSVs. Rule 6
  extends that pattern to Julia.
- MC-LC-003 (adaptx PD switch) and the catalytic curve cases embed captured
  constants in Python with no bundle. Julia evidence currently exists only as
  constants copied by hand from printed output into `tests/test_codim2.py`,
  `tests/test_curves.py`, `tests/test_hopf_normal_form.py`, and the table in
  example 02 (BifurcationKit v0.5.2). Rule 6 requires these to become
  committed bundles.
- The five Julia files have no Project/Manifest, export schema, or runner. They
  load Revise/Plots, and 05 selects special points by numeric index.
- MC-LC-002 torBPC is a deliberate failure: missing LPC/PD labels, plus a
  multiplier error of about 1.13e-2 against a 5e-3 limit. It is tracked as a
  strict xfail and stays that way.
- Two supported catalytic cases keep `US-` IDs and live under
  `matlab/unsupported`. The MatCont README and `docs/source/validation.md`
  (an eight-case snapshot) are stale.
- `docs/source/conf.py` scans all of `examples/` and blacklists helper
  basenames. It forces CPU on RTD, executes the gallery on RTD or when
  `JAXCONT_DOCS_EXECUTE_GALLERY=1`, aborts on errors, and orders entries by
  filename. Examples write `images/` into the current directory.
  `docs/source/auto_examples` is generated output.
- Direct consumers that must keep working: `tests/test_matcont_suite.py`,
  `tests/test_matcont_visualization.py`, and the example path in
  `tests/test_loop6_regressions.py`. Default pytest uses
  `-m 'not slow and not gpu'`.

## Target layout

```text
examples/
  README.md  __init__.py
  tutorials/   GALLERY_HEADER.rst  plot_NN_<topic>.py   # public-API lessons only
validation/                      # repo tooling, not part of the jaxcont wheel
  README.md registry.py artifacts.py compare.py validation.py
  visualize.py run_validation.py
  gallery/  GALLERY_HEADER.rst  plot_<run>.py   # 10–30 line evidence pages
  cases/
    cases.json
    systems/                     # named equations, parameter order, seeds
    python_cases/                # JaxCont producers, no plotting
  regenerate/                   # opt-in MATLAB and Julia tooling
    matcont/                    # MATLAB drivers/exporters/systems
      capability_gaps/ templates/
    bifurcationkit/             # .jl producers, Project/Manifest, setup.jl, runner.py
  reference/<system>/<tool>/<run>/   # committed reviewed bundles (rule 6)
  generated/                     # git-ignored candidates, reports, figures
benchmarks/
  README.md cases.json contracts.py run_benchmarks.py batching.py reporting.py
  adapters/{jaxcont,matcont,bifurcationkit}.py
  results/reviewed/              # committed reviewed benchmark records
  results/raw/                   # git-ignored
scripts/
  check_installation.py profile_continuation.py
  check_examples_migration_map.py build_validation_figures.py check_docs.py
docs/source/  validation.md benchmarks.md  auto_tutorials/ auto_validation/
```

There is no new public API in `src/jaxcont`. Commands run from the repository
root with the package installed. Tutorials keep the system, seed, problem,
`ContinuationPar`, continuation, and plot visible. The budget is ≤60
executable lines, or ≤90 for periodic lessons with a stated reason. Export,
matching, and report logic stay in tooling. All outputs go to an explicit
directory, and nothing is written to repository-root `images/`.

Sphinx-Gallery gets two explicit roots (`examples/tutorials` →
`auto_tutorials`, `validation/gallery` → `auto_validation`) with a `plot_`
pattern. This structural whitelist replaces the basename blacklist. The
current CPU forcing, execution opt-in, abort-on-error, downloads, and ordering
are kept. Moves are direct, with no Python alias modules. Old gallery URLs get
redirects. A durable `docs/source/examples-migration.json` maps every old path
to its new path.

## Offline reference data (rule 6)

- Every external result that a page, figure, or default test uses lives under
  `validation/reference/<system>/<tool>/<run>/` as a committed v2 bundle with
  `metadata.json` provenance. External tools: `matcont`, `bifurcationkit`.
  Analytic values live in `validation/cases/systems/` with their derivation.
- Default code paths never resolve `MATLAB_BIN`, `MATCONT_ROOT`, `JULIA_BIN`, or
  `BIFURCATIONKIT_PROJECT`. A guard test runs every tutorial and every
  validation entry in a subprocess. In that subprocess the variables are unset,
  `PATH` excludes `matlab`/`julia`, and any `subprocess` launch of them raises.
- Regeneration is opt-in only:
  `python3 -m validation.run_validation --regenerate {matcont,bifurcationkit}
  --case <id> --output-dir <dir>`. The existing `--regenerate-matcont` flag
  remains as an alias. Regeneration writes candidates, `--verify-references`
  compares them numerically with the committed bundles, and promotion is an
  explicit reviewed copy with a diff.
- Hand-copied external constants in tests and examples are replaced by reads
  from committed bundles. Their historical values are kept as v1 bundles with
  a provenance entry that says how they were obtained.
- Fresh JaxCont output is computed live in pages and figures, because it needs
  no external runtime. Captions say which series is fresh and which is
  committed.
- `docs/source/benchmarks.md` renders only committed records from
  `benchmarks/results/reviewed/`. No timing runs during a docs build.

## Inventory and dispositions

### Python examples (current path under `examples/`)

| Current | Verdict | Destination / reason |
|---|---|---|
| `example_01_pitchfork.py` + `example_04_continuation_methods.py` | merge | `tutorials/plot_01_cubic_fold.py`: two-fold cubic, natural vs PALC |
| `example_21_pitchfork_branch_switch.py` | move | `tutorials/plot_02_pitchfork_switch.py`: real BP switch |
| `example_02_lorenz.py` | move | `tutorials/plot_03_lorenz84.py`; the BK table moves to evidence |
| `example_03_van_der_pol.py` | move | `tutorials/plot_04_vanderpol_equilibrium.py`; keep the degeneracy caveat |
| `example_05_neural_mass.py` | move | `tutorials/plot_05_neural_mass.py` |
| `example_15_two_parameter_diagram.py` | move | `tutorials/plot_06_two_parameter.py`; 0.5.0 `CurveLayout`/unpack |
| `example_06_vmap_sweep.py` | move | `tutorials/plot_07_vmap_sweep.py`; timing moves to benchmarks |
| `example_07_differentiable.py` | move | `tutorials/plot_08_differentiable.py`; scan vs while limits |
| `example_10_van_der_pol_limit_cycle.py` | move | `tutorials/plot_09_vanderpol_cycle.py` |
| `example_11_brusselator_limit_cycle.py` | move | `tutorials/plot_10_brusselator_cycle.py` |
| `example_12_fitzhugh_nagumo_phase_plane.py` | move | `tutorials/plot_11_phase_plane.py` |
| `example_13_phase_response_curve.py` | move | `tutorials/plot_12_prc.py` |
| `example_08_period_doubling.py` | move | `tutorials/plot_13_period_doubling.py`; analytic control |
| `example_22_period_doubling_switch.py` | move | `tutorials/plot_14_pd_switch.py`; embedded MatCont numbers move to a bundle |
| `example_09_neimark_sacker.py` | move | `tutorials/plot_15_neimark_sacker.py`; analytic control |
| `example_23_periodic_vmap_grad.py` | move | `tutorials/plot_16_periodic_transforms.py` |
| `example_14_prc_shooting_validation.py` | split | `validation/cases/python_cases/shooting.py` + `validation/gallery/plot_prc_shooting.py` |
| `example_16`–`example_20` overlays | merge | `validation/gallery/plot_{cubic_fold,vanderpol_hopf,adaptx_hopf,radial_cycle,torbpc_ns}.py`, all failures kept |
| `demo_jit_speedup.py` | drop | its hard-coded 10–50x claim and toy tangent are not continuation evidence; replaced by benchmarks |
| `profile_continuation.py` | move | `scripts/profile_continuation.py`; public results only, synchronized timing, host-cProfile label |
| `check_installation.py` | move | `scripts/check_installation.py` |
| `GALLERY_HEADER.rst`, `__init__.py` | rewrite | one header per gallery root; package docstring updated |

New evidence pages: `plot_lorenz84_equilibria.py`, `plot_lorenz84_codim2.py`,
`plot_adaptx_pd.py`.

### Tooling, external scripts, and references (rules)

- `examples/MatCont/*.py`, `cases.json`, `python_cases/`, and
  `generated/.gitignore` move to `validation/`, with `cases.json` and
  `python_cases/` under `validation/cases/`, unchanged in behaviour. Drop
  `GALLERY_HEADER.txt`. Merge the README after fixing stale claims.
- `examples/MatCont/matlab/**` moves to `validation/regenerate/matcont/`.
  Exceptions: the catalytic fold/Hopf curve drivers leave `unsupported/` for
  the adapter root; Bratu, connecting-orbit, LPC/NS/PD-curve wrappers go to
  `capability_gaps/`; the general-BVP and heteroclinic files go to
  `templates/` (non-executable).
- Julia 01/02/05 merge into `cubic_fold.jl`, `lorenz84_equilibria.jl`, and
  `lorenz84_codim2.jl` (headless, exported data, events selected by type and
  window). 03 and 04 move as `neural_mass.jl` and `hopf_normal_form.jl`.
- Reference files move byte-identical to
  `validation/reference/<system>/matcont/<legacy-id>/`, with system `cubic`,
  `vanderpol`, `adaptx`, `radial`, or `torbpc`. PRC-001 goes to `adaptx`. Old
  and new SHA-256 hashes go into the migration map. v2 conversion writes a new
  bundle plus a conversion manifest.
- Generated images and `docs/source/auto_examples/**` are regenerated, never
  edited. `docs/images/matcont_cubic_overlay.png` (README) is rebuilt by the
  figure pipeline with a sidecar.

### Canonical run IDs (new; legacy IDs stay as aliases)

| Canonical ID | Legacy alias | Role / required tools |
|---|---|---|
| `lorenz84.eq-f` | — | flagship F-branch with fold/Hopf events; JC, MC, BK |
| `lorenz84.fold-ft`, `lorenz84.hopf-ft` | — | flagship (F,T) curves; JC, MC, BK |
| `lorenz84.bt-ft` | — | flagship refined BT; JC, MC, BK |
| `cubic.eq-fold` | MC-EQ-001 | control; JC, MC, BK |
| `radial.cycle-floquet` | MC-LC-001 | analytic periodic control; JC, MC, BK |
| `adaptx.eq-hopf` | MC-EQ-003 | applied Hopf; JC, MC, BK |
| `adaptx.cycle-pd-switch` | MC-LC-003 | PD and doubled branch; JC, MC, BK |
| `torbpc.cycle-lpc-pd-ns` | MC-LC-002 | full LPC/PD/NS branch; JC, MC, BK; known FAIL |
| `adaptx.prc` | MC-PRC-001 | iPRC vs MatCont; dPRC is not cross-validated |
| `shooting.prc-check` | — | PRC/dPRC vs independent IVP shooting (sheared circle, Van der Pol) |
| `vanderpol.eq-degenerate-hopf` | MC-EQ-002 | degenerate control; JC, MC |
| `hopf-normal-form.l1` | — | asymmetric l1 coefficient; JC, BK |
| `analytic.codim2` | MC-C2-001 | aggregate analytic CP/BT/GH/ZH/HH; not an alias of `lorenz84.bt-ft` |
| `jax.transforms` | MC-JAX-001 | JAX transform control; JC only |
| `catalytic.fold-curve`, `catalytic.hopf-curve` | US-C2-LP-001, US-C2-H-001 | secondary regression; JC, MC |

The IDs US-BP-001, US-C2-PD/LPC/NS-001, US-BVP-001, US-HOM-001, and US-HET-001
stay as they are, as capability-gap or template entries with no success claim.
Alias collisions are rejected.

## Reference systems

- **Extended Lorenz-84 (4D):** the flagship. It covers the equilibrium F-branch
  (T=0.04, from Julia 02), fold and Hopf curves in (F,T), and BT refinement
  with spectra in all three tools. The MatCont driver is new work. For BT, the
  checks are the residual, two small eigenvalues, geometric nullity one, and
  nondegeneracy coefficients where they are available. A Hopf chart that ends
  near BT is reported as a chart boundary.
- **adaptx (3D):** Hopf, the base cycle, simple PD, and the doubled branch.
  `T_doubled/T_base ≈ 2`, and the mean-removed half-period difference is
  nonzero. The MatCont PD export and Julia periodic producer are new.
- **torBPC (3D):** the full LPC/PD/NS branch with no window trimming. It stays
  an honest stress case and is not promised to pass.
- **Cubic and radial:** cheap calibration controls for exporters and matching.

If the owner chooses a different flagship, it needs a new feasibility record.
The model is never substituted partway through a report.

## Validation-figure standard

- Each figure shows the run ID, tool versions, precision, bundle hashes,
  compared domain, thresholds, report hash, and a visible PASS / FAIL / PENDING
  / UNSUPPORTED status. A single `ValidationReport` drives both the CLI and the
  renderer, so a plot cannot override a failed predicate. FAIL makes the CLI
  exit nonzero. A missing tool is PENDING, never PASS.
- The `<figure>.checks.json` sidecar records, for each metric, the measured
  value, limit, sample count, coverage fraction, verdict, and source hashes. A
  publication build requires the full expected domain and event set.
  Pairwise JC–MC, JC–BK, and MC–BK results plus independent residual or
  analytic checks are all reported.
- **Equilibria:** the branch on each sheet, the spectral abscissa, error versus
  arclength, the ∞-norm residual, endpoint coverage, unique event type and
  location, frequency, instability count, and fold/l1 coefficients.
- **Periodic:** extrema, period, nontrivial multipliers (as log-modulus and
  phase), the collocation defect AND an independent shooting/dense-grid
  defect, closure, phase-aligned orbit distance, and a mesh-refinement
  summary. The trivial multiplier is identified by flow alignment. Legacy
  bundles that use nearest-to-+1 identification record that limitation.
- **PD/NS:** a simple real −1 for PD. For NS, a nonresonant unit-circle pair
  with its angle and a local crossing check.
- **Codim-2:** curves in both parameters, matched labels, the
  spectral/nondegeneracy conditions, and the chart termination status.
- **PRC:** the phase and unit conventions, the adjoint residual, and Z·f
  normalization. d(PRC)/dp is checked against shooting finite differences and
  is never compared with MatCont's d(PRC)/dt column.
- The comparison rule is `|a−b| ≤ atol + rtol·max(|a|,|b|)`, with per-metric
  budgets in the registry. The current MC limits are the migration floor (EQ
  parameter 5e-4; torBPC parameter 2e-3, period/extrema/multiplier 5e-3,
  residual 1e-7). Limits are never widened to make a case pass. Stricter x64
  budgets come from two refinements. Matching is one-to-one, mode
  multiplicity is preserved, and branches are never sorted or deduplicated
  across a fold.

## Reference format and provenance

A v2 bundle (UTF-8 CSV plus JSON, `schema_version: 2`) contains:

- `branch.csv`: segment and point IDs, order or arclength, parameters, states
  or extrema, period, residual, instability count, and status.
- `events.csv`: type, status, bracket, refined values, frequency or period,
  coefficients with their convention, and the critical mode.
- `spectra.csv`: linkage to the branch or event, kind, real/imag,
  logabs/phase, and the trivial-mode method.
- `orbits.csv` and `prc.csv` where needed.
- `metadata.json`: tool, runtime, and package versions with lock hashes;
  OS/CPU/BLAS; precision; equation and driver SHA-256; parameter order;
  scaling; seeds; mesh; resolved tolerances and step controls; solver
  strategies; command; UTC time; generator revision; output hashes; reviewer
  and date; conversion chain.

Unknown historical fields are recorded as `unknown`, never invented. Hashes
prove identity, not correctness. Raw external logs are kept as candidate
artifacts. Committed bundles are capped at ≤10 MiB per run; larger orbits are
stored externally with a hash.

## Fair benchmark methodology

Benchmarks time complete, accuracy-qualified workloads from the validation
cases (cubic/radial calibration; Lorenz-84, adaptx, and torBPC). Results are
reported per workload, with no rankings or aggregate speedups.

| Category | Boundary |
|---|---|
| Startup/import | fresh OS process until the tool is ready, timed externally |
| First call | ready runtime through the first complete synchronized solve (JAX compile, Julia specialization, MATLAB first use); fresh process per replicate, cache policy declared |
| Compile/preparation | JAX lower/compile where it can be measured; otherwise `null` with a reason, never 0. "First minus warm" is labelled an overhead estimate |
| Warm | same process, same callable, shape, dtype, and options; seed reset each repetition; all result leaves synchronized |
| End-to-end (optional) | seed + continuation + stability/events + export, with the same stages for every tool |
| Batched throughput | B=1,8,32,128: JAX serial vs `jit(vmap)`; Julia/MATLAB serial or explicitly configured threads/parfor; each lane checked against its serial result; batch compile reported separately; GPU is a separate experiment |

- **Equal controls.** Before timing, every adapter resolves the case contract:
  Newton atol/rtol and iteration limit, event tolerance, eigenvalue accuracy,
  bounds, direction, `ds`/`ds_min`/`ds_max`, adaptation, step limits, the
  physical arclength metric and weights, stability/event cadence, and
  ntst/ncol. JaxCont weights the state by 1/n_state, so an equal numeric `ds`
  alone is not equal control. The adapter stores `resolved_controls[tool]`
  after defaults and unit conversion. Any missing, unequal, or
  unrepresentable field sets `comparable=false` and records the field, tool,
  and value. A fixed-step matched baseline comes first. Adaptive runs are a
  separate experiment and report accepted/rejected steps and Newton
  iterations.
- **Equal accuracy.** Each timing row carries its errors, computed by the
  validation report against the committed reference bundle, not values the
  tool reports about itself. A row is eligible for cross-tool display only
  when every tool passes the same `accuracy_limits`, which are separate from
  solver stopping tolerances. Rows that fail or are not comparable stay in
  the raw records with no cross-tool comparison.
- **Environment.** float64 on CPU, one named machine, pinned thread and BLAS
  counts, tools run sequentially in randomized order, versions and lockfiles
  recorded. Starting controls: Newton atol 1e-9, rtol 0, 25 iterations; event
  atol 1e-7; ds=1e-2, ds_min=1e-5, ds_max=2e-2. Case manifests may override
  these before any run.
- **Sampling.** Five fresh-process first calls and ≥20 warm repetitions after
  three untimed warm-ups. All samples are kept. Reports give the median, IQR,
  p95, n, timer resolution, and bootstrap CIs. No timing assertions run in CI.
  Timeouts and memory limits are the same for every tool.
- **Docs.** Measuring MATLAB/Julia timings is an opt-in run. Reviewed records
  are committed to `benchmarks/results/reviewed/`, and the docs render only
  those records (rule 6).

## External runtimes

- **Julia.** Install user-local with juliaup (no sudo), choosing versions from
  the official BifurcationKit compatibility notes at execution time. Do not
  rely on old v0.5.2 comments. `validation/regenerate/bifurcationkit/` holds
  `Project.toml` and `Manifest.toml` (BifurcationKit, Accessors, CSV, JSON,
  LinearAlgebra; periodic packages only if they are used; no Revise or
  Plots), plus `setup.jl` (instantiate, precompile, version report) and
  `runner.py`. Nothing installs during pytest or a docs build. Julia is never a
  wheel dependency.
- **MATLAB.** R2024a and MatCont 7.6 come from `MATLAB_BIN` and `MATCONT_ROOT`.
  Verify them in Phase 0. New flagship and adaptx-PD bundles must be produced
  by real local runs. Historical bundles keep their original versions.
- **Opt-in contract.** Runtime resolution is CLI flag > environment variable
  (`MATLAB_BIN`/`MATCONT_ROOT`, `JULIA_BIN`/`BIFURCATIONKIT_PROJECT`/
  `JULIA_DEPOT_PATH`) > `PATH`. The registered markers `matlab` and `julia` are
  deselected by default. When a marker is selected explicitly, a missing or
  broken runtime is a failure, not a skip. Subprocesses have timeouts,
  captured logs, absolute paths, and nonzero exit propagation. Data is never
  parsed from printed output.

## Execution: three run → debug → fix loops

The plan's tasks are built in Loop 1. All three loops then run the same gate
list (G1–G8 in the plan) and record a scorecard of gates passed and open
Critical/Important findings. Each loop also has a docs audit focus:

- **Loop 1 (build + API truth).** Implement Phases 0–4 and run every gate.
  Audit README and every authored docs page against each 0.5.0 Breaking,
  Added, and Fixed entry.
- **Loop 2 (fresh checkout + reproducibility).** Re-run every gate from a clean
  clone with MATLAB and Julia hidden. Re-run opt-in regeneration once and
  verify it numerically against the committed bundles. Run an executed strict
  gallery build and check every docs snippet.
- **Loop 3 (fresh-reader review).** Re-read the full docs against code,
  reports, and changelog, and repeat all gates.

Each loop's findings ledger records file/line, evidence, severity, fix, and
re-review verdict before it closes. Each loop ends with a snapshot. The
returned result is the snapshot with the best scorecard; ties go to the later
loop. A required solver change becomes a separate proposal. Known numerical
failures stay visible.

## Work split

Codex owns contracts, JaxCont producers, integration, reports, and docs fixes.
Copilot owns the Julia/MATLAB setup and producers in separate files, and
independently re-reviews each loop ledger. Claude gets three narrow reviews of
≤2,000 output tokens each: (1) the Lorenz-84 model contract, (2) the resolved
benchmark controls, and (3) the final high-risk docs claims. Each review sees
only the relevant excerpts. Only Codex edits the registry and report files.

## Risks

- The MatCont Lorenz-84 route and the Julia periodic routes are new. If one
  fails, only that row is blocked.
- torBPC may need solver work outside this scope. It stays FAIL with a strict
  xfail.
- Arclength, l1, Floquet mode, PRC phase, and BT chart conventions can cause
  false disagreement. Conventions are recorded in each bundle.
- Moving tooling breaks imports, test paths, notebooks, and URLs. The
  migration map, path updates, and clean builds catch this.
- MATLAB licensing or Julia network limits can block regeneration. They never
  block offline examples.
- Thread policy and caching can dominate small cases. They are disclosed, and
  such rows are marked non-comparable.
