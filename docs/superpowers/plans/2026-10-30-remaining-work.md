# Remaining work (as of 2026-10-30)

State: branch `examples-validation-plan`. The examples/validation/benchmarks
plan (`2026-10-18-examples-validation-benchmarks.md`) is complete, three
review loops (Codex reviews + fixes, reports were in the session scratchpad)
are merged, and JaxCont is float64 only (`986ea9c`). Full suite green
(float64 only).

## Before the next release (docs; deliberately postponed by the owner)

- Rebuild the executed docs/gallery (`JAXCONT_DOCS_EXECUTE_GALLERY=1 make docs`)
  so the generated torBPC page shows PASS (local `docs/source/auto_*` is stale
  and untracked).
- Docs pass for float64 only beyond what is already updated (quickstart,
  migration, CHANGELOG, tutorials header, RELEASING, development): README,
  installation, transformations/presentation pages, gallery prose that still
  mentions float32; tutorial code that still branches on precision.
- `C-05`: the migration inventory still promises
  `examples/validation/gallery/plot_hopf_normal_form.py` and
  `plot_catalytic.py`; write them or drop the promise
  (`docs/source/examples-migration.json`, Task 2.13 of the plan).
- Re-measure the reviewed benchmark records after the event-refinement
  speedup (L2-F1, `4f6f0c3`): warm cubic continuation 0.218 s -> 0.009 s;
  `benchmarks/results/reviewed/*.json` still hold the old timings, and
  `docs/source/benchmarks.md` is generated from them.
- CHANGELOG/ledger: every new 0.5.0 entry needs its row in
  `docs/audits/2026-10-18-loop-1.md` (`python scripts/check_docs.py
  --changelog 0.5.0 --ledger docs/audits/2026-10-18-loop-1.md`).

## Open validation evidence

- torBPC three-tool promotion stays PENDING: BifurcationKit's PD has native
  status `guess` (its event bisection does not converge), so no MatCont v2 or
  BifurcationKit bundle is promoted for `torbpc.cycle-lpc-pd-ns`. Options:
  tune `periodic_options` bisection in
  `examples/validation/regenerate/bifurcationkit/periodic_export.jl`, or
  declare a refined PD comparison elsewhere. The legacy MatCont comparison
  passes.
- PENDING cases with no reviewed three-tool evidence: `adaptx.eq-hopf`,
  `adaptx.prc` (MC-PRC-001), `hopf-normal-form.l1` (needs a coefficient
  convention/budget), `catalytic.*`, `analytic.codim2`, `jax.transforms`.
- Known limitation: spurious folds near a Hopf seed at very small amplitude
  (torBPC start point, documented).

## Test-suite follow-ups

- Tolerance audit `docs/audits/2026-10-29-test-tolerances.md`: findings
  T01-T03 (output-captured snapshots in `tests/test_solver_wiring.py`) are
  open; derive the expected schedule independently.
- Dead float32 branches in tests: removed after `986ea9c` (see the commit
  following it). Library code still carries dtype-generic helpers
  (`default_tol` float32 branch); harmless, simplify only if wanted.

## Housekeeping

- Local branches `review-fixes`, `review-fixes-codex`,
  `review-remediation-2026-10` are merged; delete when no longer needed.
- `git stash@{0}`: the superseded limited-scope PrecisionWarning WIP; drop it
  once nobody needs it.
- Open a PR from `examples-validation-plan` to `main` when ready.
