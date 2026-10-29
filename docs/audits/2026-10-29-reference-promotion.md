# Task 2.12 promotion manifest gate (2026-10-29)

Current status: Round 3 partially complete; see the Round 3 section for promotions, remaining failures, and verification.

Historical round 1 status: STOPPED at the owner-requested manifest gate. No promotion, registry status rewrite, historical bundle migration, or test constant replacement was performed.

Generated candidates and manifests are under `examples/validation/generated/promotion/`. The numerical producers were run from commit `3dd787c13a76e00e0e5552614ba1772d0259a84d`, in float64 on CPU. The old generated `candidates` and `candidates2` folders were removed. No solver source, pinned environment, legacy reviewed bundle, or tolerance was changed. Nothing was committed.

## Manifest results

| Run | Tools present | Worst measured pairwise metric / limit | Verdict | Promoted |
|---|---|---|---|---|
| cubic.eq-fold | jaxcont,matcont,bifurcationkit | matcont--bifurcationkit.sheet_2.endpoint_difference=3.1740845164295797 / 0.0005 | FAIL | No |
| adaptx.eq-hopf | jaxcont,matcont | jaxcont--matcont.sheet_0.endpoint_difference=10.761240000000099 / 0.0005 | FAIL | No |
| radial.cycle-floquet | jaxcont,matcont,bifurcationkit | jaxcont--matcont.sheet_0.endpoint_difference=0.5000000002212699 / 0.0005 | FAIL | No |
| torbpc.cycle-lpc-pd-ns | jaxcont,matcont,bifurcationkit | jaxcont--matcont.sheet_count_difference=5.0 / 0; 2 unavailable | FAIL | No |
| adaptx.cycle-pd-switch | jaxcont,bifurcationkit,matcont | jaxcont--bifurcationkit.PD.count_difference=1.0 / 0 | FAIL | No |
| catalytic.fold-curve | none | unavailable | PENDING | No |
| catalytic.hopf-curve | none | unavailable | PENDING | No |
| adaptx.prc | matcont | unavailable | FAIL | No |
| lorenz84.eq-f | jaxcont,matcont,bifurcationkit | jaxcont--matcont.sheet_count_difference=1.0 / 0 | FAIL | No |
| lorenz84.fold-ft | jaxcont,matcont,bifurcationkit | jaxcont--matcont.sheet_count_difference=1.0 / 0 | FAIL | No |
| lorenz84.hopf-ft | jaxcont,matcont,bifurcationkit | jaxcont--bifurcationkit.sheet_0.spectrum=0.0007094835764555598 / 0.0005 | FAIL | No |
| lorenz84.bt-ft | jaxcont,matcont,bifurcationkit | jaxcont--bifurcationkit.sheet_0.spectrum=9.285017002636761e-09 / 0.0005 | FAIL | No |
| hopf-normal-form.l1 | jaxcont,bifurcationkit | jaxcont--bifurcationkit.H_0.parameter=5.392241290654013e-07 / 0.0005; 2 unavailable | FAIL | No |

Unexpected FAIL runs: cubic.eq-fold, adaptx.eq-hopf, radial.cycle-floquet, adaptx.prc.

The cubic registry requires parameter coverage [-1, 4.17408451642958]. Fresh JaxCont reaches 1.0040569485788358 and BifurcationKit reaches 1.0; MatCont reaches the required endpoint. This is a measured domain failure, independently of schema or provenance limitations.

Every manifest also embeds the existing ValidationReport gate. Detailed metrics include unavailable evidence and unchanged thresholds; a small error on a shared subinterval cannot establish full coverage. Native legacy MatCont exports remain v1 candidates and require complete v2 provenance before promotion. Missing Julia adaptx Hopf/PRC producers and the catalytic drivers without exporters are retained as failed routes with logs. The JaxCont PRC and catalytic native results are retained, but have no complete v2 exporter.

The registry has no coefficient budget or normal-form conversion convention for hopf-normal-form.l1; promotion tooling does not invent one. JaxCont torBPC retains its native FAIL and nonfinite modes. The adaptx MatCont candidate directory totals approximately 15.154 MiB including a 9.8 MiB raw JSON duplicate; a full copy exceeds the 10 MiB cap and is refused by promotion tooling. No oversized bundle was promoted.

## Commands and outcomes

Shell commands used the activated ~/envs/jaxcont environment, JAX_PLATFORMS=cpu, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. External regeneration used the owner-specified MATLAB/MatCont/Julia paths, the pinned project, and JULIA_DEPOT_PATH=$PWD/.julia-depot:$HOME/.julia. Every numerical subprocess had closed stdin.

| Native CLI command | Produced | Failed | Exit |
|---|---:|---:|---:|
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case cubic.eq-fold --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/cubic.eq-fold` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case cubic.eq-fold --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/cubic.eq-fold` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case adaptx.eq-hopf --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/adaptx.eq-hopf` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case adaptx.eq-hopf --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/adaptx.eq-hopf` | 0 | 1 | 1 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case radial.cycle-floquet --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/radial.cycle-floquet` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case radial.cycle-floquet --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/radial.cycle-floquet` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case torbpc.cycle-lpc-pd-ns --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/torbpc.cycle-lpc-pd-ns` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case torbpc.cycle-lpc-pd-ns --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/torbpc.cycle-lpc-pd-ns` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case adaptx.cycle-pd-switch --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/adaptx.cycle-pd-switch` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case adaptx.cycle-pd-switch --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/adaptx.cycle-pd-switch` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case catalytic.fold-curve --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/catalytic.fold-curve` | 0 | 1 | 1 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case catalytic.hopf-curve --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/catalytic.hopf-curve` | 0 | 1 | 1 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case adaptx.prc --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/adaptx.prc` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case adaptx.prc --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/adaptx.prc` | 0 | 1 | 1 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case lorenz84.eq-f --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/lorenz84.eq-f` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case lorenz84.eq-f --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/lorenz84.eq-f` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case lorenz84.fold-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/lorenz84.fold-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case lorenz84.fold-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/lorenz84.fold-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case lorenz84.hopf-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/lorenz84.hopf-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case lorenz84.hopf-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/lorenz84.hopf-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate matcont --case lorenz84.bt-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/matcont/lorenz84.bt-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case lorenz84.bt-ft --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/lorenz84.bt-ft` | 1 | 0 | 0 |
| `/home/ziaee/envs/jaxcont/bin/python3 -m examples.validation.run_validation --regenerate bifurcationkit --case hopf-normal-form.l1 --output-dir /home/ziaee/git/JaxCont/examples/validation/generated/promotion/staging/bifurcationkit/hopf-normal-form.l1` | 1 | 0 | 0 |

| Test command | Passed | Failed/errors | Skipped |
|---|---:|---:|---:|
| `JAXCONT_TEST_X64=0 python3 -m pytest tests/test_reference_promotion.py -q (initial seven tests)` (promotion-tests) | 7 | 0 | 0 |
| `JAXCONT_TEST_X64=0 python3 -m pytest tests/test_reference_promotion.py tests/test_validation_schema.py tests/test_validation_reports.py -q` (tooling) | 46 | 0 | 0 |
| `JAXCONT_TEST_X64=0 python3 -m pytest tests/test_reference_promotion.py tests/test_validation_schema.py tests/test_validation_reports.py -q` (tooling-x0-final) | 46 | 0 | 0 |
| `JAXCONT_TEST_X64=1 python3 -m pytest tests/test_reference_promotion.py tests/test_validation_schema.py tests/test_validation_reports.py -q` (tooling-x1) | 46 | 0 | 0 |
| `JAXCONT_TEST_X64=0 python3 -m pytest -n 8 -q` (g1-x0) | 829 | 0 | 2 |
| `JAXCONT_TEST_X64=1 python3 -m pytest -n 8 -q` (g1-x1) | 831 | 0 | 1 |
| `JAXCONT_TEST_X64=0 python3 -m pytest -n 8 -q` (g1-x0-final) | 830 | 0 | 2 |

The test-first checks intentionally failed before implementation: promotion module absent (6 failures), missing branch-spectrum comparison (1 failure), unresolved status/PARTIAL coverage (2 failures), and absent coefficient budget (1 failure). The first intermediate green attempt had 2 failures because the synthetic event lacked its spectrum; the fixture was completed. Final focused tests pass.

JaxCont commands: `python3 -m examples.validation.generated.promotion.regenerate_jaxcont` retained all native results; the initial missing l1 producer was superseded by `python3 -m examples.validation.generated.promotion.hopf_jaxcont` (84 native points, l1=-1.7499999999999996). The final tree has 10 complete JaxCont candidates and 3 native-result-only routes. `python3 -m examples.validation.generated.promotion.complete_jaxcont_provenance` attached source identities, known configurations, and captured logs without numerical changes.

`python3 -m compileall -q scripts/promote_reference.py tests/test_reference_promotion.py` and `git diff --check` exited zero. These commands collect no tests.

Manifest command: `python3 -m examples.validation.generated.promotion.write_manifests` writes all 13 manifests and returns 1 for the unexpected FAIL gate. No promotion command was run.

G3, -m matlab, and -m julia acceptance suites are not run: the owner required stopping after all manifests on unexpected FAIL. The requested three-tool/schema constant tests, constant migration, adaptx comparison re-enablement, registry updates and promotion belong to the blocked post-gate steps.

## Files changed

- scripts/promote_reference.py: explicit manifest creation and immutable candidate promotion; coverage, provenance, hash, size, numerical and shared-report guards.
- tests/test_reference_promotion.py: eleven synthetic regression tests for the promotion boundary.
- docs/audits/2026-10-29-reference-promotion.md: this handoff.

The pre-existing untracked .clinerules was left untouched. An unrelated untracked benchmarks/contracts.py appeared during the run and was also left untouched. Generated scripts, raw results, logs, and manifests are ignored artifacts, not reviewed reference data.


## Round 3

Round 3 is partially complete. No solver source or tolerance changed and no commit was made. Only PASS manifests were promoted, with reviewer `Claude (review delegated by owner, 2026-10-28)` and date 2026-10-29. Existing reviewed bundles remain unchanged.

### Defects, causes, and fixes

- Curve comparisons previously used the union of both grids, interpolating sparse evidence onto dense samples and counting interpolation error as disagreement. Both the promotion/window comparator and the legacy branch/spectrum comparator now evaluate at the sparser tool's native samples inside the shared window, independently for each matched sheet. No synthetic dense grid or decimation is used. Synthetic quadratic curves exercise states, parameters, and spectra in both pair orders: identical curves PASS and a true offset FAILs.
- The third JaxCont equilibrium sheet is the outward continuation from the seed at F=3.0. Its intersection with the old window was only the endpoint. Boundary-only intersections no longer create comparison sheets. The required two-sheet count is unchanged. The comparison window is now [1.5467, 2.99], strictly below MatCont's upper-sheet endpoint F=2.997528062547353. All independent equilibrium checks now PASS, but pairwise values near the fold still FAIL: JaxCont/MatCont state_0 error 0.00561990697 / 0.0002, spectrum error 0.06070634247 / 0.0005. This run was not promoted.
- Lorenz84 Hopf FAIL → PASS after correcting interpolation direction (worst spectrum difference 8.9464e-5 / 5e-4). Lorenz84 fold remains FAIL: MatCont's native export has explicit segments 0 and 1 meeting around T=0, whereas the other tools have one sheet. Its chart subdivisions require investigation without rewriting candidate data or merging physical arms. An exploratory in-memory join also leaves MatCont/BifurcationKit state discrepancies above budget; it was not applied.
- torBPC retains JaxCont FAIL (14 unresolved points, missing PD, residual and sheet-count failures). The independently measured MatCont/BifurcationKit-only manifest also FAILs: spectrum errors 0.02200558488 and 0.05076158314 / 0.005 on the two sheets. Neither external bundle was promoted. The registry explicitly records JaxCont FAIL. The existing strict solver xfail remains.
- The adaptx MatCont candidate exceeded 10 MiB because `raw.json` duplicated the numerical exports. A separate packaged candidate retains every CSV and native log byte for byte; `packaging-provenance.json` records every original file hash and the omitted redundant raw JSON hash. The original candidate remains intact. The new package was measured again and its PASS manifest promoted under the unchanged size cap.

### Promotion and migration

Ten external bundles were promoted for five runs (MatCont and BifurcationKit for each): cubic.eq-fold, radial.cycle-floquet, adaptx.cycle-pd-switch, lorenz84.hopf-ft, and lorenz84.bt-ft. Every promoted bundle contains its reviewed manifest, original numerical hashes, native commands, and logs. Fresh JaxCont results remain generated evidence.

The BT solver test now reads the reviewed BifurcationKit BT bundle. The original BT values, curve seed, and Hopf coefficient are preserved as explicitly unreviewed historical v1 bundles with source-test hashes and commit provenance; unavailable runtime facts remain unknown. The curve seed and Hopf test read these historical bundles with hash verification. No historical evidence is represented as a fresh promotion. The adaptx PD comparison now requires the promoted MatCont bundle and no longer skips for PENDING.

Offline Lorenz84 tests check the reviewed flagship bundles, missing-tool gate, and alias/canonical selection. The schema suite rejects numeric literals assigned to the migrated external expectation/seed names and checks its detector against synthetic offending and bundle-backed assignments.

### Integration fixes found by verification

Promotion exposed a CLI assumption that every producer returns a v2 payload: cubic and radial still return legacy `CaseResult` objects. A report-only adapter now preserves their native branch/event/spectrum values and identities, using the producer's maximum residual as a conservative residual bound; it does not write or substitute evidence. Registry assertions now distinguish the reviewed PASS runs from measured FAIL runs. The radial legacy CLI independently checks every native reference period against its existing exact-period budget so endpoint corruption cannot escape the sparse comparison grid. The gallery source assertion accepts the concurrent page's public `ReportTable(report)` renderer as well as the older renderer; no page code was changed by Round 3.

### Per-run manifests

| Run | Round 2 / supplied verdict | Round 3 | Promoted |
|---|---|---|---|
| adaptx.cycle-pd-switch | PASS | PASS | MatCont, BifurcationKit |
| adaptx.eq-hopf | PENDING | PENDING | None |
| adaptx.prc | PENDING | PENDING | None |
| catalytic.fold-curve | PENDING | PENDING | None |
| catalytic.hopf-curve | PENDING | PENDING | None |
| cubic.eq-fold | PASS | PASS | MatCont, BifurcationKit |
| hopf-normal-form.l1 | PENDING | PENDING | None |
| lorenz84.bt-ft | PASS | PASS | MatCont, BifurcationKit |
| lorenz84.eq-f | FAIL | FAIL | None |
| lorenz84.fold-ft | FAIL | FAIL | None |
| lorenz84.hopf-ft | FAIL | PASS | MatCont, BifurcationKit |
| radial.cycle-floquet | PASS | PASS | MatCont, BifurcationKit |
| torbpc.cycle-lpc-pd-ns | FAIL | FAIL | None |

### Remaining work

Resolve the Lorenz84 fold native segment/chart correspondence and remaining state differences; resolve the equilibrium near-fold interpolation discrepancies without changing tolerances or the required sheets. Re-measure those runs and promote only if PASS. torBPC needs the existing solver proposal plus investigation of external spectrum disagreement before any promotion. Missing adaptx Hopf/PRC and catalytic exporters and the Hopf coefficient convention/budget remain PENDING. Passing runs and historical migration are complete.

Other processes edited gallery/reporting/visualization files during this session; those edits and their temporary sentinel files are outside Round 3's changes and were not modified by this work.

### Verification

Commands use `source ~/envs/jaxcont/bin/activate &&`, `JAX_PLATFORMS=cpu`, closed stdin, and one BLAS/OpenMP thread. Logs are `/tmp/round3-*.log`.

| Final check | Result |
|---|---|
| Promotion/report/schema and Lorenz84 tests, x64=0 (plus gallery source checks) | 83 PASS |
| Promotion/report/schema and Lorenz84 tests, x64=1 | 55 PASS |
| CLI/registry/offline-suite integration checks, x64=1 | 8 PASS |
| G1: `JAXCONT_TEST_X64=0 python3 -m pytest -n 8 -q` | 855 PASS, 2 SKIP; exit 0 |
| G1: `JAXCONT_TEST_X64=1 python3 -m pytest -n 8 -q` | 856 PASS, 1 SKIP; exit 0 |

Initial G1 runs identified the CLI adapter defect, the obsolete PENDING registry assertion, and the concurrent gallery renderer assertion; these were fixed before the final green runs. The first G3 command omitted MATLAB/MatCont environment variables and had three runtime setup errors plus three integration failures (one corruption check and two CLI adapter checks), with the strict torBPC xfail retained. The second G3 process had the correct runtime variables but imported the CLI before the adapter fix, so its two remaining adapter failures were stale; the eight final integration checks pass. A complete G3 run from the final source was then started with all owner-specified runtime paths.

Final G3 command (all runtime paths set as above): `JAXCONT_TEST_X64=1 python3 -m pytest tests/test_matcont_suite.py tests/test_validation_reports.py tests/test_lorenz84_three_tool.py tests/test_periodic_validation.py tests/test_curves.py tests/test_codim2.py -m slow -q` — **24 PASS, 1 strict XFAIL, exit 0**. Log: `/tmp/round3-g3-verified.log`. This full run started after every implementation and integration fix.

Final integrity verification loaded all ten promoted bundles with `require_reviewed=True`, verified every numerical output hash, and enforced the unchanged 10 MiB cap. `git diff --check` passes. No `src/jaxcont` file changed. Final state has green focused tests, both G1 precisions, and G3; the numerical FAIL/PENDING runs in the table remain unpromoted and explicitly unresolved.
