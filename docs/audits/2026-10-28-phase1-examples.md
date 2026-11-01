# Phase 1 examples migration: Tasks 1.4–1.6

Date: 2026-10-28. Branch: `examples-validation-plan`. No commit or staging was performed.

## Final verification

Every shell command started with `source ~/envs/jaxcont/bin/activate &&`.
All tests and builds exported `JAX_PLATFORMS=cpu`. Later complete runs used CPU affinity 0–7, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and a persistent compilation cache under `/tmp/jaxcont-validation-cache`; the heavy gates ran sequentially. These settings do not change numerical limits.

The table is generated from the final JUnit XML records. Test commands also wrote JUnit XML under `/tmp` (`--junitxml` for targeted runs and `PYTEST_ADDOPTS` for the two exact full-suite commands).

| Gate | Precision | Command | Passed | Failed | Skipped |
|---|---|---|---:|---:|---:|
| 1.4 | `JAXCONT_TEST_X64=0` | `python3 -m pytest tests/test_examples.py tests/test_curves.py tests/test_branch_switch.py tests/test_differentiable_root.py -n 4 -q` | 56 | 0 | 0 |
| 1.4 | `JAXCONT_TEST_X64=1` | `python3 -m pytest tests/test_examples.py tests/test_curves.py tests/test_branch_switch.py tests/test_differentiable_root.py -n 8 -q` | 56 | 0 | 0 |
| 1.5 | `JAXCONT_TEST_X64=0` | `python3 -m pytest tests/test_matcont_visualization.py::test_gallery_example_runs_from_outside_repository_root tests/test_installation.py tests/test_pd_switch.py tests/test_examples.py::test_periodic_tutorials_have_positive_period_and_amplitude tests/test_examples.py::test_pd_switch_period_ratio_near_two -n 8 -q` | 27 | 0 | 0 |
| 1.5 | `JAXCONT_TEST_X64=1` | `python3 -m pytest tests/test_matcont_visualization.py::test_gallery_example_runs_from_outside_repository_root tests/test_installation.py tests/test_pd_switch.py tests/test_examples.py::test_periodic_tutorials_have_positive_period_and_amplitude tests/test_examples.py::test_pd_switch_period_ratio_near_two -n 8 -q` | 27 | 0 | 0 |
| 1.6 (all slow entries and executed sentinel build included) | `JAXCONT_TEST_X64=0` | `python3 -m pytest tests/test_gallery_sources.py tests/test_offline_examples.py -m 'slow or not slow' -n 2 -q` | 54 | 0 | 0 |
| 1.6 (all slow entries and executed sentinel build included) | `JAXCONT_TEST_X64=1` | `python3 -m pytest tests/test_gallery_sources.py tests/test_offline_examples.py -m 'slow or not slow' -n 2 -q` | 54 | 0 | 0 |
| Full default suite | `JAXCONT_TEST_X64=0` | `python3 -m pytest -n 8 -q` | 761 | 0 | 1 |
| Full default suite | `JAXCONT_TEST_X64=1` | `python3 -m pytest -n 8 -q` | 762 | 0 | 0 |
| Moved slow direct consumers | `JAXCONT_TEST_X64=0` | `python3 -m pytest tests/test_loop2_regressions.py::test_example14_completes_in_scratch tests/test_loop4_regressions.py::test_shipped_gallery_example -m 'slow or not slow' -n 2 -q` | 4 | 0 | 0 |
| Moved slow direct consumers | `JAXCONT_TEST_X64=1` | `python3 -m pytest tests/test_loop2_regressions.py::test_example14_completes_in_scratch tests/test_loop4_regressions.py::test_shipped_gallery_example -m 'slow or not slow' -n 2 -q` | 4 | 0 | 0 |
| CLI/legacy-renderer dispatch regression | `JAXCONT_TEST_X64=0` | `python3 -m pytest tests/test_matcont_suite.py::test_offline_cli_validates_supported_reference_case tests/test_matcont_visualization.py::test_render_case_overlay_reuses_registered_case_and_writes_png -q` | 2 | 0 | 0 |
| CLI/legacy-renderer dispatch regression | `JAXCONT_TEST_X64=1` | `python3 -m pytest tests/test_matcont_suite.py::test_offline_cli_validates_supported_reference_case tests/test_matcont_visualization.py::test_render_case_overlay_reuses_registered_case_and_writes_png -q` | 2 | 0 | 0 |

The float32 full-suite skip is the existing float64-only PD/NS accuracy check.

| Build/other command | Outcome |
|---|---|
| `JAXCONT_DOCS_EXECUTE_GALLERY=0 python3 -m sphinx -W --keep-going -b html docs/source /tmp/jaxcont-layout-accepted` | PASS: 1 build, 0 failures, 0 warnings |
| `JAX_ENABLE_X64=1 JAXCONT_DOCS_EXECUTE_GALLERY=1 MPLBACKEND=Agg python3 -m sphinx -E -a -W --keep-going -b html docs/source /tmp/jaxcont-html-executed-accepted` | PASS: 1 build, 0 failures, 0 warnings; 22/22 gallery sources executed, 0 excluded as stale |
| `JAX_ENABLE_X64=0/1 python3 scripts/check_installation.py` (two runs) | PASS: 2 runs, 0 failures |
| `JAX_ENABLE_X64=0/1 python3 scripts/profile_continuation.py` (two runs) | PASS: 2 runs, 0 failures |
| `git diff --check` | PASS |
| `python3 scripts/check_examples_migration_map.py --completed` | FAIL: exit 1, four later-phase destinations absent (listed below); checker and migration map were not weakened |

The two standalone Sphinx builds unset MATLAB/Julia configuration variables. The offline subprocess gate additionally removes their binaries from PATH and installs a launch guard. The executed sentinel tests verify only the two configured roots become pages/downloads and that every gallery source has a captured figure. The old numbered URLs, old index, and dropped demo URL have HTML redirects.

## Remaining audit failure / out of scope

The durable migration map includes creations assigned to later tasks. Completing this audit requires those actual implementations; no placeholder was added:

- `examples/validation/gallery/plot_lorenz84.py` — Task 2.13.
- `examples/validation/gallery/plot_hopf_normal_form.py` — Task 2.13.
- `examples/validation/gallery/plot_catalytic.py` — Task 2.13.
- `benchmarks/run_benchmarks.py` — Task 3.2.

No BifurcationKit series, new external reference bundles, or benchmark runner was fabricated. All requested six Phase 1 comparison pages are present. All 22 historical reference-file hashes remain unchanged; the completed audit reports only the four absent later-phase files.

## File dispositions

The following root example sources were rewritten, merged, or split according to the existing migration map:

| Old source | Disposition | Destination |
|---|---|---|
| `examples/example_01_pitchfork.py` | merge | `examples/tutorials/plot_01_cubic_fold.py` |
| `examples/example_02_lorenz.py` | move | `examples/tutorials/plot_03_lorenz84.py` |
| `examples/example_03_van_der_pol.py` | move | `examples/tutorials/plot_04_vanderpol_equilibrium.py` |
| `examples/example_04_continuation_methods.py` | merge | `examples/tutorials/plot_01_cubic_fold.py` |
| `examples/example_05_neural_mass.py` | move | `examples/tutorials/plot_05_neural_mass.py` |
| `examples/example_06_vmap_sweep.py` | move | `examples/tutorials/plot_07_vmap_sweep.py` |
| `examples/example_07_differentiable.py` | move | `examples/tutorials/plot_08_differentiable.py` |
| `examples/example_08_period_doubling.py` | move | `examples/tutorials/plot_13_period_doubling.py` |
| `examples/example_09_neimark_sacker.py` | move | `examples/tutorials/plot_15_neimark_sacker.py` |
| `examples/example_10_van_der_pol_limit_cycle.py` | move | `examples/tutorials/plot_09_vanderpol_cycle.py` |
| `examples/example_11_brusselator_limit_cycle.py` | move | `examples/tutorials/plot_10_brusselator_cycle.py` |
| `examples/example_12_fitzhugh_nagumo_phase_plane.py` | move | `examples/tutorials/plot_11_phase_plane.py` |
| `examples/example_13_phase_response_curve.py` | move | `examples/tutorials/plot_12_prc.py` |
| `examples/example_14_prc_shooting_validation.py` | split | `examples/validation/cases/python_cases/shooting.py`, `examples/validation/gallery/plot_prc_shooting.py` |
| `examples/example_15_two_parameter_diagram.py` | move | `examples/tutorials/plot_06_two_parameter.py` |
| `examples/example_16_matcont_cubic_overlay.py` | merge | `examples/validation/gallery/plot_cubic.py` |
| `examples/example_17_matcont_vanderpol_overlay.py` | merge | `examples/validation/gallery/plot_vanderpol.py` |
| `examples/example_18_matcont_adaptive_control_overlay.py` | merge | `examples/validation/gallery/plot_adaptx.py` |
| `examples/example_19_matcont_radial_cycle_overlay.py` | merge | `examples/validation/gallery/plot_radial.py` |
| `examples/example_20_matcont_torbpc_overlay.py` | merge | `examples/validation/gallery/plot_torbpc.py` |
| `examples/example_21_pitchfork_branch_switch.py` | move | `examples/tutorials/plot_02_pitchfork_switch.py` |
| `examples/example_22_period_doubling_switch.py` | move | `examples/tutorials/plot_14_pd_switch.py` |
| `examples/example_23_periodic_vmap_grad.py` | move | `examples/tutorials/plot_16_periodic_transforms.py` |

Moves used plain `mv` because `.git` is read-only:

- `examples/check_installation.py` → `scripts/check_installation.py` (present and moved).
- `examples/profile_continuation.py` → `scripts/profile_continuation.py` (present; rewritten as a synchronized public-result diagnostic with an explicit host-cProfile label).

Created support files:

- `examples/README.md`.
- `examples/tutorials/GALLERY_HEADER.rst`.
- `examples/validation/gallery/GALLERY_HEADER.rst`.
- `examples/validation/cases/systems/cubic.py`.
- `examples/validation/cases/systems/lorenz84.py`.
- `examples/validation/cases/systems/vanderpol.py`.
- `examples/validation/cases/python_cases/shooting.py`.
- `examples/validation/reporting.py`.
- `tests/test_examples.py`.
- `tests/test_gallery_sources.py`.
- `tests/test_offline_examples.py`.
- `docs/audits/2026-10-28-phase1-examples.md`.

Deleted the 23 old root `example_*.py` scripts, `examples/demo_jit_speedup.py`, `examples/GALLERY_HEADER.rst`, and the obsolete `examples/MatCont` and `examples/BifurcationKit` directories.

Updated gallery rendering, the optional precomputed-result plumbing and CLI dispatch, README links, Sphinx configuration/navigation/redirects, `.gitignore`, validation documentation and the compatibility pointer. Updated the named test paths and the additional slow direct consumers in the loop-2/loop-4 regressions. Old-source comments in the other affected tests now point at the migrated lessons.

The only change in `src/jaxcont` is the explicitly authorized docstring path in `bifurcations/hopf_normal_form.py`. `CHANGELOG.md`, `notes/`, and the pre-existing untracked `.clinerules` were left untouched.

## Scientific and execution details

- All 16 lessons have module titles/introductions, English text cells, visible systems/seeds/settings/public calls, explicit output directories, and numeric assertions on the plotted data. Equilibrium lessons use at most 59 executable lines; periodic lessons use at most 90, with their reason stated when the allowance is used.
- Figures remain open for the Sphinx-Gallery matplotlib scraper. No script defaults to repository-root `images/`.
- All six comparison pages use fresh JaxCont computations, existing committed MatCont data where applicable, status banners, error panels, report-driven tables, and matching `.checks.json` sidecars. Shooting uses fresh independent IVP comparisons and explicitly states its shared-orbit scope.
- Van der Pol degeneracy, scan/while differentiation limits, analytic PD/NS control scope, and torBPC FAIL remain visible. Existing scientific tolerances and strict torBPC xfail were not changed.
- The neural-mass assertions check stationary balances in state units to avoid magnifying float32 cancellation by division by its small time constant. The vector field is unchanged.
- The Brusselator lesson uses the original refined float64 mesh, a smaller float32 mesh that converges at the same solve tolerance, and a target of b=3. It states that the corrector can stop before the target; its figures show the computed domain.
- PRC derivative assertions compare independently re-converged orbits: a frozen-phase analytic derivative would omit phase-anchor drift.

## Debugging history

Early runs exposed the trimmed eager-vs-batched validity-mask contract, PRC phase-anchor drift, float32 seed refinement and neural-mass cancellation, CLI dispatch of the new optional computed arguments, and the absence of `__file__` during Sphinx-Gallery execution. These were fixed without editing solver algorithms or widening numerical limits.

An early nonexecuted docs build produced 26 warnings (old execution-time links, a title underline, and the absent optional presentation PDF); the final builds have none. The optional PDF page now shows its real LaTeX source link when PDF tooling is unavailable.

Some early overlapping heavy runs timed out or were interrupted. They are superseded by the complete passing runs above. An initial whitelist assertion looked for figures in the gallery subdirectory instead of Sphinx’s `_images`; the corrected test checks actual numbered figure files and the download archives. No unresolved test or docs-build failure remains beyond the explicit migration-audit destinations listed above.
