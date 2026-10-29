# Task 2.12 promotion manifest gate (2026-10-29)

Status: STOPPED at the owner-requested manifest gate. No promotion, registry status rewrite, historical bundle migration, or test constant replacement was performed.

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
