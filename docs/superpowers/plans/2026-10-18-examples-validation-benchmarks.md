# Examples, Validation, and Benchmarks Implementation Plan

**Goal:** Short JaxCont 0.5.0 tutorials. A separate evidence gallery comparing
the same systems in JaxCont, MatCont, and BifurcationKit, with quantitative
checks behind every figure. Fair, accuracy-qualified benchmarks. Docs audited
against 0.5.0. Everything user-facing runs without MATLAB or Julia.

**Architecture:** Two gallery roots, `examples/tutorials` and
`examples/validation/gallery`, both read committed reference bundles under
`examples/validation/reference/`. External tools only regenerate those bundles through
opt-in commands. `benchmarks/` reuses the validation workloads and accuracy
checks.

**Tech stack:** Python ≥3.11, the pinned JAX stack, pytest, and Sphinx-Gallery.
MATLAB R2024a and MatCont 7.6 come from `MATLAB_BIN`/`MATCONT_ROOT`. Julia and
BifurcationKit are pinned in Phase 0.

**Spec:** [design](../specs/2026-10-18-examples-validation-benchmarks-design.md).
Status: proposed, not yet executed. Test names below are new unless marked
"existing".

## Ground rules

- **Offline (rule 6).** Every tutorial, validation page and figure, docs
  build, and default test runs with MATLAB and Julia absent. They read
  committed bundles. External runtimes are reached only through
  `--regenerate {matcont,bifurcationkit}` and the `matlab`/`julia` pytest
  markers, and those write candidates only.
- **Bundles are immutable.** Promotion is an explicit reviewed copy with a
  numerical diff. Historical bytes and versions never change. Hand-copied
  external constants are replaced by bundle reads.
- **No weaker checks.** Do not change `src/jaxcont` algorithms or widen limits
  to make something pass. torBPC stays FAIL with its existing strict xfail.
- **Benchmarks.** No headline cross-tool speed claims, rankings, or CI timing
  assertions. Startup, first call, compile, warm, and batched results are
  separate categories. Cross-tool rows require equal resolved controls and
  passing accuracy.
- **Tests.** New schema and report tests use small synthetic fixtures. Only the
  scientific tests run expensive continuation. Never edit generated gallery
  copies.
- **Ownership.** Copilot is not available (owner, 2026-10-28). Codex
  implements every task, including the Julia/MATLAB setup and producers that
  were Copilot's. Claude reviews each Codex diff, runs the tests and gates,
  does the three focused reviews in Tasks 2.3, 3.1, and 5.2, and takes over
  the ledger re-reviews. Only Codex edits `examples/validation/cases/cases.json`
  and the report modules.
- **Language and pages.** All files are in English. Each validation gallery
  page is one self-contained comparison page per reference system, built to
  the spec's "Comparison-page standard": system, what to look for, the
  JaxCont computation, comparison figures (primary), a report-driven table
  (supporting), and a short takeaway.
- **Runtimes on the owner's machine.** `MATLAB_BIN=~/programms/Matlab/R2024a/bin/matlab`,
  `MATCONT_ROOT=~/programms/MatCont/MatCont7p6`. Julia is installed user-local
  with juliaup in Task 0.1.
- **Usage quota.** Keep about 10% of each model's usage quota in reserve. When
  it is reached, record where work stopped, wait for the reset, and resume.

## Execution structure: three run → debug → fix loops

- **Loop 1.** Build Phases 0–4 in task order, then run G1–G8.
- **Loops 2 and 3.** Tasks 5.1 and 5.2 run the gates again.

Each loop follows the same cycle. *Run* all gates. *Debug*: triage each failure
into the loop ledger `docs/audits/2026-10-18-loop-N.md` with a cause and a
severity. *Fix*. Then re-run the failed gates and record a **scorecard**: the
gates passed and the open Critical/Important findings. Each loop ends with a
snapshot (a commit on the feature branch). The final result is the snapshot
with the best scorecard; ties go to the later loop.

| Gate | Command (from the repository root) | Pass condition |
|---|---|---|
| G1 | `JAXCONT_TEST_X64=0 python3 -m pytest -n 8` and `JAXCONT_TEST_X64=1 python3 -m pytest -n 8` | green |
| G2 | `python3 -m pytest tests/test_offline_examples.py -m 'slow or not slow' -q` | every entry in `examples/tutorials` and `examples/validation/gallery` runs with MATLAB/Julia hidden |
| G3 | `JAXCONT_TEST_X64=1 python3 -m pytest tests/test_matcont_suite.py tests/test_validation_reports.py tests/test_lorenz84_three_tool.py tests/test_periodic_validation.py tests/test_curves.py tests/test_codim2.py -m slow` | green; torBPC xfail strict |
| G4 | `python3 scripts/build_validation_figures.py --output-dir /tmp/jaxcont-evidence --check` | every figure has a sidecar that agrees with its report |
| G5 | `JAXCONT_DOCS_EXECUTE_GALLERY=1 JAX_PLATFORMS=cpu MPLBACKEND=Agg python3 -m sphinx -E -a -W --keep-going -b html docs/source /tmp/jaxcont-html` on a clean tree with MATLAB/Julia hidden | no warnings; no gallery source outside the two configured roots is executed or listed |
| G6 | `python3 scripts/check_docs.py --snippets --changelog 0.5.0 --ledger docs/audits/2026-10-18-loop-N.md` | snippets run; every changelog entry has a disposition; ledger closed |
| G7 | `python3 -m benchmarks.reporting --check benchmarks/results/reviewed` | categories, accuracy, and comparability valid |
| G8 (opt-in) | `JAXCONT_TEST_X64=1 python3 -m pytest -m matlab` and `-m julia` | real runs; candidates verify numerically against committed bundles |

## Phase 0: External environments (early)

### Task 0.1: Verify MATLAB and pin the Julia environment — Codex

**Files:** Create `examples/validation/regenerate/bifurcationkit/{Project.toml,Manifest.toml,setup.jl,README.md}`.

**Interfaces:** `julia --project=examples/validation/regenerate/bifurcationkit examples/validation/regenerate/bifurcationkit/setup.jl`
instantiates and precompiles the environment, then prints a JSON version
report: Julia, packages, and the manifest hash.

- [ ] **Step 1:** Confirm that `MATLAB_BIN` and `MATCONT_ROOT` resolve to
  R2024a and 7.6, and save the version log. Install Julia user-local with
  juliaup (Decision 7). Choose versions from the official BifurcationKit
  compatibility notes.
- [ ] **Step 2:** Create a headless project (BifurcationKit, Accessors, CSV,
  JSON, LinearAlgebra; no Revise or Plots). Instantiate it and commit the
  manifest.
- [ ] **Step 3:** Run `setup.jl` and save its report as a candidate artifact.
  Acceptance: both runtimes report their versions; nothing in pytest or docs
  installs packages.

### Task 0.2: Opt-in runners and markers — Codex

**Files:** Create `examples/validation/__init__.py` (merged in Task 1.2),
`examples/validation/regenerate/bifurcationkit/runner.py`, and `tests/test_external_runners.py`.
Modify `tests/conftest.py` (register the `matlab` and `julia` markers) and
`pyproject.toml` (addopts `-m 'not slow and not gpu and not matlab and not julia'`).

**Interfaces:** `resolve_julia(cli=None)` and `run_julia(script, args, timeout)`.
Resolution order is CLI > `JULIA_BIN`/`BIFURCATIONKIT_PROJECT`/`JULIA_DEPOT_PATH`
> `PATH`. Under an explicitly selected marker, a missing runtime is a failure.
Existing MATLAB-running tests also get the `matlab` marker.

- [ ] **Step 1:** Add offline tests with subprocess fakes:
  `tests/test_external_runners.py::test_default_run_never_launches_external_tools`,
  `::test_julia_resolution_precedence`, `::test_selected_marker_missing_runtime_fails`,
  and `::test_timeout_and_nonzero_exit_keep_logs`.
- [ ] **Step 2:** Implement the runner. Add the `julia` smoke test
  `tests/test_external_runners.py::test_julia_environment_smoke`, which checks
  imports and the manifest hash and installs nothing.
- [ ] **Step 3:** Run `python3 -m pytest tests/test_external_runners.py -q`, then
  `python3 -m pytest tests/test_external_runners.py -m julia -q`. Acceptance:
  default runs skip cleanly; existing MATLAB resolution tests are unchanged.

## Phase 1: Restructure

### Task 1.1: Migration map — Codex

**Files:** Create `docs/source/examples-migration.json` and `scripts/check_examples_migration_map.py`.

**Interfaces:** Each entry maps an old path to a new path with its verdict and
coverage ID. The map covers the spec's inventory tables, the 18 legacy case IDs
plus the canonical IDs, and old/new SHA-256 for the 22 reference files.
`--planned` validates the map's content; `--completed` also checks that the
destinations exist and the sources are gone.

- [ ] **Step 1:** Write the map by hand from the spec tables, including the
  `MC-PRC-001` → `adaptx.prc` correction.
- [ ] **Step 2:** Implement the check. It rejects unmapped sources, alias
  collisions, merges without a rationale, and unmapped case IDs.
- [ ] **Step 3:** Run `python3 scripts/check_examples_migration_map.py --planned`.
  Acceptance: the map accounts for 23 numbered examples, 3 other scripts, 5
  Julia files, 18 cases, and 22 reference files.

### Task 1.2: Move validation tooling and references — Codex

**Files:** Move `examples/MatCont/{*.py,cases.json,python_cases/,generated/.gitignore}`
to `examples/validation/`, with `cases.json` and `python_cases/` under `examples/validation/cases/`. Move `reference/*` to `examples/validation/reference/<system>/matcont/<id>/`.
Update the imports in `tests/test_matcont_suite.py`, `tests/test_matcont_visualization.py`,
and overlays 16–20. Create `examples/validation/README.md`.

**Interfaces:** `CaseResult`, registry selection, and the comparison/renderer
signatures do not change. The CLI becomes `python3 -m examples.validation.run_validation --case MC-EQ-001`.

- [ ] **Step 1:** Move the files and update the dotted producer paths and test
  imports in one change, using `examples.validation.cases.python_cases` for producers
  and `examples.validation.regenerate` for external tooling. Reference bytes stay identical.
- [ ] **Step 2:** Run the existing tests
  `tests/test_matcont_suite.py::test_offline_cli_validates_supported_reference_case`,
  `::test_folded_branch_comparison_preserves_duplicate_parameter_arms`, and
  `tests/test_matcont_visualization.py::test_render_case_overlay_reuses_registered_case_and_writes_png`.
- [ ] **Step 3:** Run `python3 -m pytest tests/test_matcont_suite.py tests/test_matcont_visualization.py`
  in both precisions. Acceptance: hashes and verdicts are unchanged, and no
  `examples.MatCont` import remains.

### Task 1.3: Move external adapters — Codex

**Files:** Move `examples/MatCont/matlab/**` to `examples/validation/regenerate/matcont/`
following the spec's folder rules. Move `examples/BifurcationKit/*.jl` to their
spec names. Update the producer paths in `examples/validation/cases/cases.json`  and the MATLAB directory in `tests/test_matcont_suite.py`.

**Interfaces:** MATLAB resolution and CLI precedence do not change. The
catalytic curve drivers leave `unsupported/`. Capability gaps and templates
are separated. Legacy `US-` IDs are kept.

- [ ] **Step 1:** Move the files and fix the MATLAB `addpath` calls, export
  paths, and source-hash lookups. Correct the comments that mislabel
  capability support.
- [ ] **Step 2:** Run the existing tests
  `tests/test_matcont_suite.py::test_unsupported_template_is_never_reported_as_executed`,
  `::test_explicit_unsupported_case_requires_opt_in`, and
  `::test_relative_generated_directory_resolves_before_matlab_changes_directory`.
- [ ] **Step 3:** Run the existing
  `tests/test_matcont_suite.py::test_matlab_equilibrium_exporters_preserve_spectra_and_normal_forms -m matlab`.
  Acceptance: a real MATLAB run, with candidates that match the committed
  bundles.

### Task 1.4: Equilibrium and transform tutorials — Codex

**Files:** Create `examples/tutorials/plot_01`…`plot_08` per the spec table.
Create `examples/validation/cases/systems/{cubic,lorenz84,vanderpol}.py` and
`tests/test_examples.py`. Update the path in `tests/test_loop6_regressions.py`.

**Interfaces:** Public `bif_problem`, `ContinuationPar`, `continuation`, the
curve unpackers, and `CurveLayout`. A tutorial runs as
`python3 examples/tutorials/plot_01_cubic_fold.py --output-dir <dir>`.

- [ ] **Step 1:** Rewrite each tutorial to ≤60 executable lines. Keep the
  equations, seed, and settings visible. Move reference tables and timing out.
  Keep the Van der Pol caveat and the scan-vs-while limits.
- [ ] **Step 2:** Add one numeric check per tutorial figure in
  `tests/test_examples.py`. For example: `::test_cubic_fold_has_two_folds`,
  `::test_pitchfork_switch_finds_both_arms`,
  `::test_lorenz84_branch_has_fold_and_hopf`, and
  `::test_vmap_sweep_matches_serial_valid_points`. Update the existing
  `tests/test_loop6_regressions.py::test_vmap_example_does_not_claim_events_are_unsupported`
  to the new path.
- [ ] **Step 3:** Run `python3 -m pytest tests/test_examples.py tests/test_curves.py tests/test_branch_switch.py tests/test_differentiable_root.py`
  in both precisions. Acceptance: each lesson has one purpose and its numeric
  check passes.

### Task 1.5: Periodic tutorials, evidence entries, and scripts — Codex

**Files:** Create `examples/tutorials/plot_09`…`plot_16`,
`examples/validation/cases/python_cases/shooting.py`, and `examples/validation/gallery/plot_*.py` for
overlays 16–20 and the PRC shooting page. Move `check_installation.py` and
`profile_continuation.py` to `scripts/`. Update `examples/__init__.py`.

**Interfaces:** Each comparison page follows the spec's "Comparison-page
standard" and renders its figures through `examples/validation/visualize.py`
(`render_case_overlay(case_id, output_dir=...)` and the new per-family
panels). In this task the pages for cubic, Van der Pol, adaptx Hopf, radial,
torBPC, and PRC shooting are written with the system and "what to look for"
text and the existing MatCont overlays. Task 2.13 adds the BifurcationKit
series, the report tables, and the new pages. The shooting checker is a
producer and keeps its numerical method unchanged.

- [ ] **Step 1:** Port the periodic lessons (≤90 lines). Keep the analytic
  PD/NS "control" labels and the existing failure banners.
- [ ] **Step 2:** Add `tests/test_examples.py::test_periodic_tutorials_have_positive_period_and_amplitude`
  and `::test_pd_switch_period_ratio_near_two`. Parameterize the existing
  `tests/test_matcont_visualization.py::test_gallery_example_runs_from_outside_repository_root`
  over the new entries.
- [ ] **Step 3:** Run them together with `tests/test_installation.py` and
  `tests/test_pd_switch.py`. Acceptance: no orphaned source and no writes to
  root `images/`.

### Task 1.6: Gallery roots and the offline guard — Codex

**Files:** Modify `docs/source/conf.py`, `docs/source/index.rst`, `README.md`
(links and paths), and `examples/README.md`. Create the two gallery headers,
`examples/tutorials/GALLERY_HEADER.rst` and `examples/validation/gallery/GALLERY_HEADER.rst`,
`tests/test_gallery_sources.py`, and `tests/test_offline_examples.py`. Remove
the old root scripts and the `examples/MatCont`/`examples/BifurcationKit`
directories.

**Interfaces:** `docs/source/conf.py` currently uses the single gallery root
`examples_dirs='../../examples'`. Switch to exactly
`examples_dirs=['../../examples/tutorials', '../../examples/validation/gallery']`
with matching `gallery_dirs=['auto_tutorials', 'auto_validation']`; the pattern is `plot_`.
The `examples/validation/cases/`, `examples/validation/reference/`, and
`examples/validation/regenerate/` trees are never scanned or executed.
Execution semantics stay the same. Old URLs redirect.

- [ ] **Step 1:** Add `tests/test_gallery_sources.py::test_execution_semantics_survive_root_split`
  (covering both environment branches) and `::test_only_examples_tutorials_and_examples_validation_gallery_plot_sources_become_pages`.
  Add `::test_sphinx_build_executes_and_lists_only_the_two_examples_roots`: run
  an executed Sphinx build with sentinel `plot_*.py` sources in cases/, reference/,
  and regenerate/; assert no source outside the two configured roots is executed
  or listed in generated gallery pages or downloads.
- [ ] **Step 2:** Add `tests/test_offline_examples.py::test_examples_tutorials_and_validation_gallery_run_without_matlab_or_julia`
  (slow), covering `examples/tutorials` and `examples/validation/gallery`. Each entry runs in a subprocess with the MATLAB/Julia environment
  variables unset, those binaries removed from `PATH`, and a guard that fails
  on any external launch. This is gate G2.
- [ ] **Step 3:** Run both test files (including the executed-build acceptance test),
  `python3 scripts/check_examples_migration_map.py --completed`,
  and `JAXCONT_DOCS_EXECUTE_GALLERY=0 python3 -m sphinx -W --keep-going -b html docs/source /tmp/jaxcont-layout`.
  Acceptance: helper files cannot become pages; the executed build lists and
  executes sources only from the two roots, and every entry passes G2.

## Phase 2: Reference data and quantitative validation

### Task 2.1: v2 schema, provenance, and the regeneration CLI — Codex

**Files:** Modify `examples/validation/{registry,artifacts,run_validation}.py` and
`examples/validation/cases/cases.json`. Create `examples/validation/reference/schema-v2.json` and
`tests/test_validation_schema.py`.

**Interfaces:** Each registry run has `canonical_id`, `aliases`, `system`,
`tool_coverage`, `evidence_status`, the expected domain and events, and
per-metric limits. `--regenerate {matcont,bifurcationkit} --case --output-dir`
writes candidates only; `--regenerate-matcont` remains an alias.
`--verify-references` compares candidates numerically.

- [ ] **Step 1:** Add `::test_v2_requires_tool_equation_and_driver_provenance`,
  `::test_bundle_hash_mutation_is_rejected`,
  `::test_legacy_v1_bundle_reads_with_unknown_fields`,
  `::test_pending_tool_cannot_satisfy_coverage`, and
  `::test_regeneration_never_writes_into_reference`.
- [ ] **Step 2:** Implement v1 reading, v2 conversion with a manifest, and
  alias resolution.
- [ ] **Step 3:** Run the new tests plus the existing
  `tests/test_matcont_suite.py::test_reference_metadata_requires_complete_reproducibility_provenance`
  and `::test_default_reference_rejects_unreviewed_and_explicit_rejects_malformed_hash`.

### Task 2.2: One report gate — Codex

**Files:** Modify `examples/validation/{compare,validation}.py`. Create `tests/test_validation_reports.py`.

**Interfaces:** `ValidationReport` holds the per-tool coverage, pairwise and
independent per-metric value/limit/verdict, source hashes, and the overall
status. The CLI exits nonzero on FAIL.

- [ ] **Step 1:** Add `::test_three_pairwise_checks_required_when_three_tools_present`,
  `::test_partial_overlap_cannot_pass_full_domain`, `::test_fold_sheets_are_not_merged`,
  and `::test_missing_event_or_nonfinite_value_fails`. Also add the slow
  `::test_offline_suite_verdicts_match_registry`, which expects torBPC FAIL.
- [ ] **Step 2:** Generalize the existing matching to N tools and add
  flow-mode identification with a legacy fallback. Reuse the existing
  primitives.
- [ ] **Step 3:** Run them together with the existing
  `tests/test_matcont_suite.py::test_match_spectrum_prefers_a_tolerance_feasible_assignment`
  and `::test_offline_cli_rejects_corrupted_matcont_fold_event`.

### Task 2.3: Lorenz-84 model contract — Codex

**Files:** Complete `examples/validation/cases/systems/lorenz84.py`. Create `tests/test_reference_systems.py`.
Add the four `lorenz84.*` runs to `examples/validation/cases/cases.json`.

**Interfaces:** The right-hand side, parameter order, fixed constants, seeds,
and windows (T=0.04 F-sweep from Julia 02), plus RHS/Jacobian values at sample
states for the MATLAB and Julia transcriptions to match. Events are selected by
type and window, never by index.

- [ ] **Step 1:** Add `::test_lorenz84_rhs_matches_independent_samples` and
  `::test_event_selection_uses_type_and_window`.
- [ ] **Step 2:** Claude review 1: the contract and conventions only. Resolve
  its findings before any producer starts.
- [ ] **Step 3:** Run `python3 -m pytest tests/test_reference_systems.py -q`.

### Task 2.4: JaxCont Lorenz-84 producer — Codex

**Files:** Create `examples/validation/cases/python_cases/lorenz84.py` and `tests/test_lorenz84_producer.py`.

**Interfaces:** Fresh v2-shaped results for the four flagship runs, with
residual and BT structure checks.

- [ ] **Step 1:** Add the slow tests `::test_branch_has_expected_domain_and_events`
  and `::test_bt_has_jordan_structure`.
- [ ] **Step 2:** Implement the producer with the public API. Run the existing
  `tests/test_codim2.py::test_bogdanov_takens_matches_bifurcationkit_jl_independent_run`
  and `tests/test_curves.py::test_lorenz84_bt_matches_bifurcationkit` with `-m slow`.
- [ ] **Step 3:** Run `JAXCONT_TEST_X64=1 python3 -m pytest tests/test_lorenz84_producer.py -m slow -q`.

### Task 2.5: MatCont Lorenz-84 producer — Codex

**Files:** Create `examples/validation/regenerate/matcont/systems/lorenz84.m`,
`run_lorenz84_equilibria.m`, `run_lorenz84_codim2.m`, and `tests/test_lorenz84_matcont.py`.

**Interfaces:** An independent transcription. The drivers export v2 candidates
for the four runs and record R2024a/7.6, the resolved controls, hashes, and
raw logs.

- [ ] **Step 1:** Make the drivers export RHS/Jacobian samples. Add the
  `matlab`-marked `::test_flagship_exports_all_four_runs` and
  `::test_matlab_rhs_samples_match_contract`. The latter also runs offline on
  the bundle once Task 2.12 has promoted it.
- [ ] **Step 2:** Run `python3 -m examples.validation.run_validation --regenerate matcont --case lorenz84.eq-f --output-dir /tmp/jc-cand`
  for each run, then `pytest -m matlab tests/test_lorenz84_matcont.py`.
- [ ] **Step 3:** Hand the candidates and logs to Task 2.12. Acceptance: a real
  local run with the full domain and events. License or path errors are
  blockers, never skips.

### Task 2.6: Julia controls and Lorenz-84 equilibria — Codex

**Files:** Rewrite `cubic_fold.jl`, `lorenz84_equilibria.jl`, and
`hopf_normal_form.jl`. Create `export.jl` and `radial_cycle.jl` in
`examples/validation/regenerate/bifurcationkit/`. Create `tests/test_julia_exports.py`.

**Interfaces:** Headless scripts take `--case` and `--output-dir`, write v2
candidates, and exit nonzero on failure. They do not read JaxCont solutions.

- [ ] **Step 1:** Build the common exporter, then cubic and radial with
  explicit tolerances, mesh, and provenance.
- [ ] **Step 2:** Export the Lorenz-84 F-branch and the asymmetric l1. Add the
  offline `::test_julia_rhs_matches_contract_samples` and
  `::test_equilibrium_export_schema_and_conventions` on small fixtures.
- [ ] **Step 3:** Run `--regenerate bifurcationkit` for these cases, then
  `pytest -m julia tests/test_julia_exports.py`.

### Task 2.7: Julia Lorenz-84 codim-2 — Codex

**Files:** Rewrite `examples/validation/regenerate/bifurcationkit/lorenz84_codim2.jl`.
Extend `export.jl` and `tests/test_julia_exports.py`.

**Interfaces:** (F,T) fold and Hopf curves and the refined BT, selected by type
and window, with the chart settings recorded.

- [ ] **Step 1:** Replace the index-based special-point selection.
- [ ] **Step 2:** Export the curve and BT candidates. Add
  `::test_codim2_export_preserves_parameter_order_and_event_identity`.
- [ ] **Step 3:** Run it with `-m julia`. Acceptance: a complete bundle, or a
  recorded failed route; never a printed scalar.

### Task 2.8: JaxCont periodic checks — Codex

**Files:** Modify `examples/validation/cases/python_cases/{periodic,pd_switch,prc}.py`. Create
`tests/test_periodic_validation.py`.

**Interfaces:** adaptx Hopf/PD/doubled-branch records and the complete torBPC
LPC/PD/NS record. Mesh refinement and an independent shooting defect (not the
collocation residual).

- [ ] **Step 1:** Add `::test_pd_switch_requires_genuine_doubled_shape`,
  `::test_periodic_reference_requires_complete_event_set`,
  `::test_radial_mesh_refinement_reduces_independent_defect`, and
  `::test_prc_parameter_derivative_is_not_time_derivative`.
- [ ] **Step 2:** Implement them. The embedded PD constants are read from the
  bundle produced in Task 2.9.
- [ ] **Step 3:** Run the existing `tests/test_matcont_suite.py::test_adaptx_period_doubling_switch_matches_matcont`,
  `::test_radial_cycle_matches_exact_floquet_formula`, and
  `::test_torbpc_known_disagreements_are_explicit_failures` with `-m slow`.
  Acceptance: the strict xfail is still in place and no limit is widened.

### Task 2.9: MatCont adaptx PD export — Codex

**Files:** Create `examples/validation/regenerate/matcont/run_adaptx_pd.m`. Extend
`export_cycle_run.m` for orbits and flow-mode data.

**Interfaces:** A v2 candidate for `adaptx.cycle-pd-switch` (base branch, PD,
doubled branch, orbits, spectra) from a real R2024a run.

- [ ] **Step 1:** Implement the driver with the case's window, mesh, and controls.
- [ ] **Step 2:** Run `--regenerate matcont --case adaptx.cycle-pd-switch`.
- [ ] **Step 3:** Hand the candidate and logs to Task 2.12. Acceptance: the
  period ratio, PD multiplier, and doubled shape are present.

### Task 2.10: Julia adaptx PD — Codex

**Files:** Create `examples/validation/regenerate/bifurcationkit/adaptx_pd.jl`. Extend
`tests/test_julia_exports.py`.

**Interfaces:** The same adaptx parameters and window. The independent Hopf →
cycle → PD → doubled-branch route is exported with orbits, spectra, defect,
and refinement.

- [ ] **Step 1:** Seed from the Hopf point with no JaxCont input. Match the
  mesh, norm, and step controls, or declare the run non-comparable.
- [ ] **Step 2:** Add `::test_pd_export_has_base_and_doubled_orbit_identity`
  and export the candidate.
- [ ] **Step 3:** Run with `-m julia`. Acceptance: a complete candidate, or a
  recorded failed route, which then stays PENDING.

### Task 2.11: Julia torBPC — Codex

**Files:** Create `examples/validation/regenerate/bifurcationkit/torbpc_ns.jl`. Extend
`tests/test_julia_exports.py`.

**Interfaces:** The full existing torBPC window with every expected LPC/PD/NS
label, plus the NS angle and crossing.

- [ ] **Step 1:** Freeze the seed and controls before comparing.
- [ ] **Step 2:** Add `::test_torbpc_export_requires_full_expected_event_domain`
  and export the candidate.
- [ ] **Step 3:** Run with `-m julia`. Acceptance: a complete candidate or a
  recorded failure. The window is never trimmed.

### Task 2.12: Promote bundles and remove hand-copied constants — Codex

**Files:** Add `examples/validation/reference/{lorenz84,cubic,radial,adaptx,torbpc}/{matcont,bifurcationkit}/**`.
Create `tests/test_lorenz84_three_tool.py`. Modify `tests/test_codim2.py`,
`tests/test_curves.py`, and `tests/test_hopf_normal_form.py` so that their
BifurcationKit constants are read from bundles. Historical values become v1
bundles with provenance.

**Interfaces:** Each promotion has a reviewed manifest: candidate hashes,
command, logs, a numerical diff against the other tools, and the reviewer and
date (Decision 6).

- [ ] **Step 1:** Add `::test_flagship_requires_all_three_bundles` and
  `::test_aliases_and_canonical_ids_select_identical_runs`. Also add the
  offline `tests/test_validation_schema.py::test_no_hand_copied_external_constants_in_tests`,
  which flags external-tool literals that have no bundle source.
- [ ] **Step 2:** Review each candidate (provenance, coverage, pairwise diffs)
  and promote it. A failed route is recorded as PENDING/FAIL in `examples/validation/cases/cases.json`.
- [ ] **Step 3:** Run G1 and G3. Acceptance: every claimed three-tool run has
  committed, reviewed bundles, and the offline tests pass without runtimes.

### Task 2.13: Comparison pages and publication gate — Codex

**Files:** Modify `examples/validation/visualize.py` and the Phase 1 pages.
Create `examples/validation/gallery/plot_{lorenz84,adaptx,hopf_normal_form,catalytic}.py`
(adaptx absorbs the Hopf, PD, and PRC runs into one page),
`scripts/build_validation_figures.py`, and `tests/test_validation_figures.py`.
Rebuild `docs/images/matcont_cubic_overlay.png`.

**Interfaces:** The renderer takes a `ValidationReport` and the fresh JaxCont
result. It writes PNG plus `.checks.json`, both sharing the status and hashes.
External series come only from committed bundles.

- [ ] **Step 1:** Add `::test_figure_and_sidecar_share_status_and_hashes`,
  `::test_failed_report_cannot_render_pass_banner`, `::test_pending_tool_is_visibly_pending`,
  and `::test_publication_requires_complete_domain`.
- [ ] **Step 2:** Add the spec's panels per family, with versions, thresholds,
  and a fresh/committed caption. Complete every page to the
  Comparison-page standard: three-tool overlays with labelled special points,
  an error or zoom panel where curves coincide, and a report-driven table.
  Add `::test_every_page_has_the_six_sections` (system, what to look for,
  computation, figures, table, takeaway) and
  `::test_page_numbers_come_from_the_report` (no hand-written numeric claims).
- [ ] **Step 3:** Run the tests and G4. Acceptance: every validation figure has
  a sidecar, torBPC renders FAIL, and each page reads as a self-contained
  tutorial (Claude reads every rendered page once).

## Phase 3: Fair benchmarks

### Task 3.1: Workload and controls contract — Codex

**Files:** Create `benchmarks/{__init__,contracts}.py`, `benchmarks/{README.md,cases.json}`,
and `tests/test_benchmark_contracts.py`.

**Interfaces:** `BenchmarkCase` holds the validation canonical ID, the frozen
controls, `accuracy_limits`, the stages, and the sampling plan.
`ResolvedControls` is per tool. Comparability uses exact equality of
normalized decimals and records the field, tool, and value when they differ.

- [ ] **Step 1:** Add `::test_equal_ds_with_different_metric_is_not_comparable`,
  `::test_mismatched_resolved_tolerance_disables_comparison`,
  `::test_accuracy_limits_are_separate_from_stopping_tolerances`, and
  `::test_compile_unavailable_is_null_not_zero`.
- [ ] **Step 2:** Write manifests for the cubic, radial, Lorenz-84, adaptx, and
  torBPC workloads, each with a fixed-step row and a separate adaptive row.
- [ ] **Step 3:** Claude review 2: this contract and its tests only. Resolve
  its findings before any measurement.

### Task 3.2: First-call and warm harness — Codex

**Files:** Create `benchmarks/run_benchmarks.py`,
`benchmarks/adapters/{jaxcont,matcont,bifurcationkit}.py` (including the
MATLAB and Julia workers), and `tests/test_benchmark_harness.py`.

**Interfaces:** An external supervisor times startup. A resident worker does a
ready handshake, then emits synchronized samples. Each row's accuracy is
computed by `ValidationReport` against the committed bundle.

- [ ] **Step 1:** Add the fake-clock tests `::test_timer_waits_for_all_result_leaves`,
  `::test_first_call_uses_fresh_process`, `::test_warm_keeps_callable_identity_and_resets_seed`,
  and `::test_failed_accuracy_row_is_ineligible`.
- [ ] **Step 2:** Implement the monotonic timers, JAX synchronization, thread
  pinning, timeouts, and raw records.
- [ ] **Step 3:** Run a JaxCont CPU smoke test (5 cold, 20 warm) and an opt-in
  three-tool run on the cubic workload. Acceptance: categories are separate
  and accuracy is shown on every row.

### Task 3.3: Batched throughput — Codex

**Files:** Create `benchmarks/batching.py` and `tests/test_benchmark_batching.py`.

**Interfaces:** B=1,8,32,128. Serial and `jit(vmap)` rows record per-lane
status, valid points, memory, and batch compile. External strategies are named
explicitly.

- [ ] **Step 1:** Add `::test_every_lane_matches_serial_valid_points`,
  `::test_batch_permutation_preserves_results`, and
  `::test_failed_lane_counts_in_denominator`.
- [ ] **Step 2:** Implement batching for the equilibrium and periodic workloads.
- [ ] **Step 3:** Run in both precisions. Acceptance: throughput is never
  labelled a single-branch speedup.

### Task 3.4: Reports, docs page, and performance scripts — Codex

**Files:** Delete `examples/demo_jit_speedup.py`. Finish `scripts/profile_continuation.py`.
Create `benchmarks/reporting.py`, `tests/test_benchmark_reporting.py`,
`docs/source/benchmarks.md`, and `benchmarks/results/reviewed/` records.

**Interfaces:** A report gives median, IQR, p95, CIs, accuracy versus time, the
controls, and the reasons for non-comparability. `--check` validates the
records. The docs page renders only the reviewed records.

- [ ] **Step 1:** Add `::test_report_includes_accuracy_and_sampling_metadata`
  and `::test_noncomparable_rows_have_no_cross_tool_table`.
- [ ] **Step 2:** Run the opt-in measurement on the named machine (Decision 4),
  then review and commit the records. Rewrite the profiler with public
  results and a host-cProfile label.
- [ ] **Step 3:** Run G7 and a profiler smoke test. Acceptance: no "expected
  speedup" text remains, and the docs page builds offline.

## Phase 4: Documentation (Loop 1 audit)

### Task 4.1: 0.5.0 audit tooling and Loop 1 ledger — Codex

**Files:** Create `scripts/check_docs.py` and `docs/audits/2026-10-18-loop-1.md`.

**Interfaces:** `check_docs.py --snippets` runs the README and docs code blocks
(illustrative blocks are marked as such). `--changelog 0.5.0` requires a
disposition for every entry. `--ledger` requires every finding to be closed or
accepted.

- [ ] **Step 1:** Map every 0.5.0 Breaking, Added, and Fixed entry to docs text
  and a test. This includes precision, statuses and masks, tracked events,
  dynamic args, scan vs while, PeriodicArgs, l1, CurveLayout, BT termination,
  PeriodicSchur, and persistence.
- [ ] **Step 2:** Record the findings before making fixes, including the stale
  `validation.md` snapshot and the MatCont README claims.
- [ ] **Step 3:** Run `python3 scripts/check_docs.py --changelog 0.5.0`.

### Task 4.2: Loop 1 docs fixes — Codex, re-reviewed by Claude

**Files:** Modify `README.md`, `docs/source/{quickstart.rst,transformations.rst,api/*.rst,index.rst,validation.md,installation.rst}`,
and the other pages the audit identifies. Update the Loop 1 ledger.

**Interfaces:** Accurate 0.5.0 API and migration text. `validation.md` is
generated from the registry and reports, covering every case and status.

- [ ] **Step 1:** Fix every finding. Breaking changes get before/after examples.
- [ ] **Step 2:** Run the existing `tests/test_loop6_regressions.py::test_api_reference_lists_the_0_5_0_surface`,
  `::test_changelog_0_5_0_has_one_section_per_kind`, and `tests/test_legacy_surface.py`, then G6.
- [ ] **Step 3:** Claude re-reviews the ledger. Then run G1–G8 and record the
  Loop 1 scorecard and snapshot.

## Phase 5: Loops 2 and 3

### Task 5.1: Loop 2 — fresh checkout and reproducibility

**Files:** Create `docs/audits/2026-10-18-loop-2.md`. Modify only the files
that the findings point to.

**Interfaces:** The reproduction matrix lists each command, its working
directory, precision, required runtime, and expected status.

- [ ] **Step 1:** *Run:* clone into a temporary directory, install, and run
  G1–G7 with MATLAB/Julia hidden. Run G8 once with the runtimes present.
- [ ] **Step 2:** *Debug:* record each failure in the ledger with its root
  cause before fixing. Check the gallery downloads, links, and images.
- [ ] **Step 3:** *Fix:* apply the fixes and re-run the failed gates. Claude
  re-reviews. Record the scorecard and snapshot.

### Task 5.2: Loop 3 — fresh-reader review and final selection

**Files:** Create `docs/audits/2026-10-18-loop-3.md`. Modify only the files
that the findings point to.

**Interfaces:** A full read of the README and docs against the code, reports,
and changelog. The final handoff names the selected snapshot.

- [ ] **Step 1:** *Run:* read through install → first branch → periodic →
  validation → benchmarks as a new user, then run G1–G8. Claude review 3
  covers the high-risk 0.5.0, evidence, and performance claims.
- [ ] **Step 2:** *Debug:* record the findings, including any speed
  exaggeration or capability over-claim.
- [ ] **Step 3:** *Fix,* re-run the gates, and have Claude re-review. Compare
  the three scorecards and return the best snapshot (ties go to the later
  loop). The handoff lists the gate results, which external runs were real,
  open PENDING/FAIL rows, and limitation IDs.

## Decisions for the owner

1. **Reference systems:** Lorenz-84 as the single flagship (equilibria, (F,T) curves, BT), with adaptx PD and torBPC NS as periodic supplements and cubic/radial as controls. *Recommended: approve.*
2. **Layout (owner decision):** `examples/` holds `examples/tutorials/` and the whole `examples/validation/` tree: `examples/validation/gallery/` (user-facing comparison scripts that build figures), `examples/validation/cases/` (case definitions), `examples/validation/reference/<system>/<tool>/<run>/` (committed reference bundles with provenance), and `examples/validation/regenerate/` (opt-in MATLAB and Julia regeneration tooling). Sphinx scans exactly the tutorials and validation gallery roots. `benchmarks/` stays top-level. Direct moves with no Python alias modules, gallery URL redirects, and legacy case IDs kept as aliases.
3. **Performance scripts:** drop `demo_jit_speedup.py`; keep `profile_continuation.py`, rewritten under `scripts/` as a developer-only diagnostic. *Recommended: approve.*
4. **Benchmark budget:** float64 CPU on one named machine; 5 fresh-process first calls and 20 warm repetitions; batch sizes 1/8/32/128; GPU as a separate optional experiment; no cross-tool rankings. *Recommended: approve.*
5. **Known failures:** torBPC stays FAIL with its strict xfail, and any failed external route stays visibly FAIL or PENDING. Solver fixes go to a separate proposal. *Recommended: approve.*
6. **Reference promotion:** the owner reviews and approves each promoted bundle from Codex's diff manifest; committed bundles are capped at ≤10 MiB per run, with larger raw data hashed and stored externally. *Recommended: approve, with the owner as reviewer.*
7. **Julia install:** user-local juliaup with the default depot, a Julia version chosen from the current BifurcationKit compatibility notes, and exact versions pinned in the committed `Manifest.toml`. *Recommended: approve.*

8. **Answered 2026-10-28:** comparison pages are one self-contained page per system (plots first, table as support, English); Sphinx-Gallery `.py`; the whole plan is executed; Julia is installed user-local; MATLAB/MatCont paths as in Ground rules; Codex replaces Copilot.

## Review history

- v0 (Codex): first draft of the spec and plan.
- Review 1 (Copilot): Julia was deferred too late, there was no work split, tolerance equality was untested, MATLAB was optional, and canonical IDs were missing. Codex fixed these in v1.
- Review 2 (Claude): enforced rule 6 (offline committed bundles, a guard gate, and hand-copied constants removed), restructured execution into three run → debug → fix loops with gates G1–G8, renumbered and split the Phase 2 tasks, fixed the `MC-PRC-001` system (adaptx, not radial), unified the external markers, required benchmark accuracy against committed bundles, consolidated helper scripts, and trimmed both documents.
- Owner decision (2026-10-18): move the whole validation tree inside `examples/validation/`, alongside `examples/tutorials/`; Sphinx scans exactly those tutorials and `examples/validation/gallery/`.
- Owner decisions (2026-10-28): per-system comparison pages as self-contained tutorials, English only, Copilot replaced by Codex (implementation) and Claude (reviews), MATLAB/MatCont paths recorded, whole plan approved for execution.
