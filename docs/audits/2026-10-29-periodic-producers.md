# Phase 2 periodic producers (Tasks 2.8–2.11)

This work changes validation tooling only. No solver files or reviewed
reference data are changed. Candidates are unreviewed and remain under
`examples/validation/generated/candidates/`. Task 2.12 promotion remains
separate and requires the owner's review.

## Numerical reconnaissance before implementation

The native probes are retained under `generated/candidates/periodic-probe`.
The frozen windows and their numerical justification are in
`cases/systems/periodic.json` (paths relative to `examples/validation`).

- MatCont adaptx: Hopf alpha=1, period=2*pi; base PD alpha=0.63030202310037,
  period=6.3640713357497; doubled start period=12.728142671499;
  doubled PD alpha=0.57962986032678, period=12.73436554085.
- Real MatCont torBPC: LPC nu=-0.584492842421877, period=8.41186621425809;
  NS nu=-0.595750403989607, period=8.86109929893306;
  PD nu=-0.614681653707091, period=9.25684575645854.
- Independent Julia adaptx reconnaissance: PD alpha=0.6303020048703702;
  genuine doubled cycles continue through alpha=0.57.
- Independent Julia torBPC reconnaissance reaches nu=-0.75 and finds all
  LPC/NS/PD locations. The registry domain is unchanged.

## Findings and resolutions

1. BK's default bisection can stop on the trivial zero Floquet exponent.
   Set `tol_bisection_eigenvalue=0` and explicitly bound the inversion count,
   minimum bisection step, and iteration budget. A positive eigenvalue stop
   produced a PD marked `guess` at alpha=0.62716; that is not accepted.
2. At torBPC LPC, BK may label the converged double +1 as a tiny-angle `:ns`,
   followed by a `:bp` with status `guessL`. Select by native type, status and
   computed window, then certify two +1 multipliers and the parameter turn.
   Preserve the native type and every additional native event in the export.
3. FloquetColl supplies no eigenvectors on this route. Native FloquetGEV
   recomputes spectra and eigenvectors at each stored point and refined event;
   it selects the flow mode by physical alignment. No nearest-to-+1 fallback
   is presented as flow identification. Near a defective double +1 both
   eigenvectors can align with the flow; native numerical precision warnings
   remain in the log.
4. MatCont's unweighted coordinate norm needs a separate step budget. With
   ds_max=0.02 the initial 300 points did not reach adaptx PD. The frozen
   MatCont budget is 500 (an exploratory 1200-point run measured 359 base and 134 doubled points to cover the windows); no domain or tolerance is relaxed. Export keeps
   the first traversal through each complete contract window, including the
   first point beyond its lower boundary.
5. Integration diagnostics must evaluate the RHS in float64. Running a
   1e-11 IVP tolerance against rounded float32 RHS outputs stalls the
   integrator. The first diagnostic test was interrupted; the corrected
   float64 independent integration passes in both test precisions.
6. Historical embedded adaptx numbers have been removed. An explicit MatCont
   candidate can be compared immediately. Default offline comparisons remain
   PENDING until the separately reviewed bundle is promoted. Internal
   switching checks still run; missing external values are null and cannot
   pass. The existing numerical limits are retained.
7. Native event spectra link to their exact refined branch identities.
   All output CSVs, including orbits, are hashed. Single-row MATLAB JSON
   objects are wrapped at the Python boundary.
8. Failed routes retain logs and `failure.json`; metadata is invalidated.
   Julia soft-scope warnings are errors at the runner boundary. All new Julia
   continuation code executes inside functions, avoiding top-level soft scope.

## Validation record

The final command matrix and measured candidate differences are recorded
below after the runtime runs finish. All commands use the activated
`~/envs/jaxcont` environment, `JAX_PLATFORMS=cpu`, and closed subprocess stdin.
Reference production uses float64. No packages are added and the pinned
Julia Manifest is unchanged. torBPC retains its strict xfail and visible FAIL.

## Selected candidates and numerical differences

Unreviewed candidates are under `examples/validation/generated/candidates/periodic-handoff-adaptx-20261029/<tool>/` and `periodic-handoff-torbpc-20261029/<tool>/`. The complete machine-readable table, hashes and commands are in `examples/validation/generated/reports/periodic-20261029.json`. No reference promotion was performed.

All adaptx tools produced complete candidates. MatCont and BifurcationKit produced complete torBPC candidates; JaxCont produced a full raw FAIL record with a missing PD label and unresolved native spectra. The existing MatCont torBPC exporter remains v1; the new external producers are v2.

### adaptx

| Tool | Hopf alpha | Hopf period | Base period at alpha=0.9 | PD alpha | PD period | Critical multiplier |
|---|---:|---:|---:|---:|---:|---:|
| jaxcont | 1 | 6.28318530718 | 6.3062350574 | 0.630302004637 | 6.36407133945 | -1.00000001167 |
| matcont | 1.00000000681 | 6.28318531788 | 6.30623504735 | 0.630302005377 | 6.36407133932 | -1.000000006 |
| bifurcationkit | 0.999999959999 | 6.28318524435 | 6.30623303006 | 0.630302004937 | 6.36407133941 | -1.00000001002 |

The MatCont and BK base periods at alpha=0.9 use interpolation between adjacent continuation points; JaxCont has an exact parameter sample. The roughly 2e-6 BK base-period difference includes interpolation error. PD values are refined directly in all three tools.

| Tool | Doubled seed period | Period ratio | Maximum mean-removed half-period difference | Doubled-branch PD alpha / period |
|---|---:|---:|---:|---|
| jaxcont | 12.7343695993 | 2.00097844919 | 0.568171145929 | not located as a labelled event |
| matcont | 12.7281426786 | 2 | 0.563686132895 | 0.579629840517 / 12.7343655416 |
| bifurcationkit | 12.7281628159 | 2.00000316419 | 0.633188361209 | 0.57962984058 / 12.7343655417 |

Ratios refer to each tool's switched seed: MatCont starts at PD, BK at PD minus 1e-4, and the JaxCont switch regression at its raw crossing endpoint minus 0.05. These are structural doubling checks, not period comparisons at identical alpha. The full doubled comparison domain [0.57, 0.63] is covered. JaxCont separately refines the base PD for its exported event; its original switch-regression seed and limits are preserved.

| Pair | PD parameter difference | PD period difference | Maximum matched multiplier difference | Base period difference at alpha=0.9 |
|---|---:|---:|---:|---:|
| jaxcont--matcont | 7.39464711685e-10 | 1.32356348104e-10 | 5.66378011158e-09 | 1.00500443523e-08 |
| jaxcont--bifurcationkit | 2.99342439725e-10 | 4.4148684708e-11 | 1.64989866214e-09 | 2.02733545862e-06 |
| matcont--bifurcationkit | 4.40122271961e-10 | 8.82076633957e-11 | 4.01388144944e-09 | 2.01728541427e-06 |

### torBPC

| Tool | Event | Parameter | Period | Multipliers (real, imaginary) |
|---|---|---:|---:|---|
| jaxcont | LPC | -0.584492840453 | 8.41185501795 | (2.85763548243, -3.00488029733e-16); (0.999969102489, -6.73717790184e-13); (1.00003090541, 6.7441101665e-13) |
| jaxcont | NS | -0.59575328527 | 8.86115640257 | (0.57421887258, -0.818701828748); (0.57421887258, 0.818701828748); (0.999958957072, 0) |
| jaxcont | PD (unlabelled nearest sample) | -0.614788879233 | 9.2589500123 | (-1.0383873656, 0); (-0.50513865559, 0); (0.999999972975, 0) |
| matcont | LPC | -0.584492842422 | 8.41186621426 | (2.85752263993, 0); (1.00000404805, 0); (0.999999999825, 0) |
| matcont | NS | -0.59575040399 | 8.86109929893 | (0.574302412219, 0.818646781394); (0.574302412219, -0.818646781394); (1, 0) |
| matcont | PD | -0.614681653707 | 9.25684575646 | (1, 0); (-0.999991340476, 0); (-0.526304902903, 0) |
| bifurcationkit | LPC | -0.584492842389 | 8.41185495113 | (2.85763608762, -0); (1.00000000996, 1.38635243349e-05); (1.00000000996, -1.38635243349e-05) |
| bifurcationkit | NS | -0.595750552876 | 8.86110274775 | (0.999999998749, -0); (0.574292782195, 0.818649982429); (0.574292782195, -0.818649982429) |
| bifurcationkit | PD | -0.614681676883 | 9.25684621142 | (-1.0000000103, -1.22464681176e-16); (1.00000000037, -0); (-0.526299955951, -6.44531556447e-17) |

NS angles: jaxcont=0.959146602539, matcont=0.959046599546, bifurcationkit=0.959056321467. BK exports a two-sided crossing check. Its auxiliary native `:bp` with status `guessL` is retained and is not claimed as an additional validated bifurcation.

| Pair | Event | Parameter difference | Period difference | Maximum matched multiplier difference |
|---|---|---:|---:|---:|
| jaxcont--matcont | LPC | 1.96855198809e-09 | 1.11963114868e-05 | 0.000112842501829 |
| jaxcont--matcont | NS | 2.88128084358e-06 | 5.71036405432e-05 | 0.000100045401427 |
| jaxcont--matcont | PD | 0.000107225525848 | 0.00210425583735 | 0.0383960251204 |
| jaxcont--bifurcationkit | LPC | 1.93536497939e-09 | 6.68186910247e-08 | 3.38743127024e-05 |
| jaxcont--bifurcationkit | NS | 2.73239468862e-06 | 5.36548236543e-05 | 9.02810721792e-05 |
| jaxcont--bifurcationkit | PD | 0.000107202349458 | 0.00210380087459 | 0.0383873552978 |
| matcont--bifurcationkit | LPC | 3.31870086967e-11 | 1.12631301779e-05 | 0.000113447695278 |
| matcont--bifurcationkit | NS | 1.4888615496e-07 | 3.44881688896e-06 | 1.01481019871e-05 |
| matcont--bifurcationkit | PD | 2.31763893632e-08 | 4.54962762575e-07 | 8.66982260983e-06 |

JaxCont PD entries above are explicitly unlabelled estimates and cannot satisfy event coverage. The unchanged legacy whole-branch comparison remains FAIL: maximum multiplier error 0.0383960251205 exceeds 0.005. There are 14 rows with unresolved native spectra; none is removed or replaced. Native failed spectra remain NaN in CSV and null in the auxiliary strict JSON.

JaxCont's 300-step raw torBPC record reaches -0.746807721986; the unchanged expected lower endpoint is -0.747263543778. The endpoint gap is 0.000455821791984, within the existing 0.002 parameter allowance, and is reported rather than trimmed away. The expected LPC/PD/NS set is unchanged.

### Independent refinement evidence

JaxCont adaptx base-cycle independent dense defect at alpha=0.9 decreases from about 1.15e-7 to 3.63e-9 on 20 to 40 intervals. BK adaptx at alpha=0.55 decreases from 4.71e-7 to 1.45e-8 on 20 to 40 intervals. JaxCont torBPC decreases from 3.08e-5 to 1.36e-6 on 10 to 20 intervals; BK torBPC decreases from 3.72e-7 to 1.53e-8 on 25 to 50 intervals. These are original-ODE integration checks, not collocation residuals. MatCont adaptx exports an independent full-period shooting defect and physical variational flow alignment at every stored point.

All new candidates explicitly decline benchmark comparability because of metric, mesh or resolved-control differences. Pairwise numerical agreement does not establish absolute continuous-orbit accuracy.


## Commands and outcomes

Each pytest command below starts with `source ~/envs/jaxcont/bin/activate && export JAX_PLATFORMS=cpu`, sets `JAXCONT_TEST_X64` to the indicated value, and closes stdin. The real-runtime matrix also sets `MATLAB_BIN`, `MATCONT_ROOT`, `JULIA_BIN`, and the writable-first `JULIA_DEPOT_PATH` specified by the owner. Final runs set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and a writable `MPLCONFIGDIR`. JUnit XML was added to capture exact counts despite quiet pytest output.

| Command / selection | Precision | Passed | Failed / errors | Skipped | Strict xfail |
|---|---|---:|---:|---:|---:|
| `python3 -m pytest tests/test_periodic_validation.py -m 'slow or not slow' -q (initial five-test revision)` | 0 | 5 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_periodic_validation.py -m 'slow or not slow' -q (initial five-test revision)` | 1 | 5 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_external_runners.py tests/test_validation_schema.py tests/test_validation_reports.py -q` | 0 | 39 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_external_runners.py tests/test_validation_schema.py tests/test_validation_reports.py -q` | 1 | 39 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_matcont_suite.py::{test_adaptx_period_doubling_switch_matches_matcont,test_radial_cycle_matches_exact_floquet_formula,test_torbpc_known_disagreements_are_explicit_failures,test_torbpc_jaxcont_matches_all_matcont_diagnostics} -m slow -q (four explicit node IDs; notation expanded in results.json)` | 0 | 2 | 0 | 1 | 1 |
| `python3 -m pytest tests/test_matcont_suite.py::{test_adaptx_period_doubling_switch_matches_matcont,test_radial_cycle_matches_exact_floquet_formula,test_torbpc_known_disagreements_are_explicit_failures,test_torbpc_jaxcont_matches_all_matcont_diagnostics} -m slow -q (four explicit node IDs; notation expanded in results.json)` | 1 | 2 | 0 | 1 | 1 |
| `python3 -m pytest -m matlab -q` | 0 | 7 | 0 | 0 | 0 |
| `python3 -m pytest -m matlab -q` | 1 | 7 | 0 | 0 | 0 |
| `python3 -m pytest -m julia -q` | 0 | 10 | 0 | 0 | 0 |
| `python3 -m pytest -m julia -q` | 1 | 10 | 0 | 0 | 0 |
| `python3 -m pytest -n 8 -q` | 0 | 819 | 0 | 2 | 0 |
| `python3 -m pytest -n 8 -q` | 1 | 820 | 0 | 1 | 0 |
| `python3 -m pytest tests/test_periodic_validation.py -m 'slow or not slow' -q` (final six-test periodic suite, including genuine shape, independent refinement, PRC derivative, failed native modes and refined PD) | 0 | 6 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_validation_reports.py tests/test_validation_schema.py tests/test_external_runners.py -q` (final schema/report/runner suite after v2 dispatch wiring) | 0 | 39 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_periodic_validation.py -m 'slow or not slow' -q` (final six-test periodic suite, including genuine shape, independent refinement, PRC derivative, failed native modes and refined PD) | 1 | 6 | 0 | 0 | 0 |
| `python3 -m pytest tests/test_validation_reports.py tests/test_validation_schema.py tests/test_external_runners.py -q` (final schema/report/runner suite after v2 dispatch wiring) | 1 | 39 | 0 | 0 | 0 |

The offline adaptx skip occurs only after the internal switch checks pass: its reviewed reference does not exist yet and promotion is explicitly outside this task. The same unchanged MatCont comparison limits pass against an explicit real candidate in `tests/test_periodic_matcont.py::test_jaxcont_switch_matches_explicit_matcont_candidate`, in both precisions. The torBPC strict xfail is unchanged and remains an actual numerical/event failure, not a missing-runtime setup failure.

Exact expanded commands, elapsed times and counts are in `examples/validation/generated/candidates/periodic-acceptance-20261029/logs/results.json`; each command has its raw log and XML beside it. The final full-suite commands ran after all code changes.

### Regeneration and diagnostic attempts

| Command / route | Outcome |
|---|---|
| `python3 -m examples.validation.run_validation --regenerate matcont --case adaptx.cycle-pd-switch --output-dir examples/validation/generated/candidates/periodic-complete-20261029` | 1 candidate produced, 0 failed routes; moved with plain mv to the handoff folder |
| `python3 -m examples.validation.run_validation --regenerate bifurcationkit --case adaptx.cycle-pd-switch --case torbpc.cycle-lpc-pd-ns --output-dir examples/validation/generated/candidates/periodic-reviewed-20261029` | 2 candidates produced, 0 failed routes; moved with plain mv to handoff folders |
| `python3 -m examples.validation.run_validation --regenerate matcont --case torbpc.cycle-lpc-pd-ns --output-dir examples/validation/generated/candidates/periodic-handoff-torbpc-20261029/matcont` | 1 complete legacy candidate produced, 0 failed routes |
| Float64 calls to `produce_adaptx_periodic()` / `produce_torbpc_periodic()` plus `write_candidate` | 1 complete adaptx candidate and 1 explicit torBPC FAIL record; generator scripts and logs retained inside the handoff bundles |
| Preliminary default full suites, `python3 -m pytest -n 8 -q` | precision 0: 818 passed, 2 skipped; precision 1: 819 passed, 1 skipped; 0 failures in either run (before the additional failed-spectrum test) |
| Preliminary `python3 -m pytest -m julia -q` | 10 passed, 0 failed, both precisions; final runs above also verify the strengthened doubled-PD and full-domain checks |
| Preliminary `python3 -m pytest -m matlab -q` with the original 300-second periodic timeout | precision 0: 5 passed, 2 failed; precision 1: 6 passed, 1 failed. Failures were actual periodic-route timeouts; final 500-point / 600-second runs above pass |
| First periodic diagnostic run | interrupted after 2 passes when a float32 RHS stalled the high-accuracy IVP integration; no completed verdict claimed |
| First corrected diagnostic run | 3 passed, 1 failed because the scalar-parameter dPRC array was indexed as a vector-parameter array; corrected and rerun: 4 passed in both precisions |
| First legacy periodic selection without runtime environment variables | 1 passed, 1 skipped, 1 setup error, 1 xfailed; superseded by the real-runtime selections above |
| Tooling and periodic diagnostics during development | 39 tooling tests and the combined 43-test diagnostic selection passed in both precisions; five-test periodic records passed before the final sixth failure-preservation test was added |
| Obsolete duplicate-spectrum producer/test runs | interrupted after replacing redundant batched Floquet recomputation with the public Branch spectra, log magnitudes and flow indices; final tests and candidates use those native fields |
| Early native probe/export attempts | recorded nonzero outcomes for a MATLAB contset call, an exhausted 300-point budget, unhashed auxiliary CSVs, Julia parser/API/eigenvector/linkage errors, and a 300-second MATLAB timeout. These were debugged and rerun, not replaced with JaxCont output. Raw candidate/probe logs remain under generated/candidates/periodic-probe and the intermediate periodic run folders; the findings above identify the corrections |

Final inspection, parsing, compileall, bundle-hash validation and `git diff --check` commands exited zero and collect no tests. No extra Julia packages were needed, no Manifest was resolved, and there are no external-runtime blockers.

## Files created or changed

Created:

- `examples/validation/cases/python_cases/periodic_checks.py`
- `examples/validation/cases/systems/periodic.json`
- `examples/validation/regenerate/matcont/run_adaptx_pd.m`
- `examples/validation/regenerate/bifurcationkit/adaptx_pd.jl`
- `examples/validation/regenerate/bifurcationkit/torbpc_ns.jl`
- `examples/validation/regenerate/bifurcationkit/periodic_export.jl`
- `tests/test_periodic_validation.py`
- `tests/test_periodic_matcont.py`
- this audit.

Changed:

- `examples/validation/cases/python_cases/{periodic,pd_switch,prc}.py`
- `examples/validation/regenerate/matcont/export_cycle_run.m`
- `examples/validation/regenerate/bifurcationkit/{export.jl,runner.py}`
- `examples/validation/{candidates,run_validation,validation}.py`
- `examples/validation/cases/cases.json`
- `tests/test_julia_exports.py`
- `tests/test_matcont_suite.py`.

No `src/jaxcont`, reviewed reference or pinned Manifest files changed. The existing untracked `.clinerules` file was left untouched. Nothing was committed. Candidate promotion remains Task 2.12.

