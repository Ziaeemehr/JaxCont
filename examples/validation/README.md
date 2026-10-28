# Validation tooling

Run from the repository root with JaxCont installed:

```sh
python3 -m examples.validation.run_validation --case MC-EQ-001
python3 -m examples.validation.run_validation --dry-run
```

The registry is `cases/cases.json`. Its 18 legacy case IDs are preserved,
including the supported catalytic fold and Hopf curve cases with `US-` IDs.
The migration plan and canonical IDs are recorded in
`docs/source/examples-migration.json`; canonical selection is later work.

Default supported cases use analytic checks or committed MatCont evidence.
The six historical bundles (22 files) are under
`reference/<system>/matcont/<legacy-id>/`. Their filenames, bytes, provenance,
and historical version strings are unchanged. `MC-PRC-001` is **adaptx**.
The PRC comparison checks MatCont's time derivative separately from JaxCont's
parameter derivative; it does not claim an external d(PRC)/dp validation.

The torBPC comparison retains its known failures and strict xfail. A failed
comparison remains FAIL in the CLI and renderer. No tolerance was widened.
The registry distinguishes executable capability gaps from non-executable
BVP and heteroclinic templates. Unsupported cases require
`--include-unsupported`; templates never report successful execution.

External regeneration is opt-in. MATLAB R2024a and MatCont 7.6 are configured
through `MATLAB_BIN` and `MATCONT_ROOT` (CLI overrides take precedence):

```sh
python3 -m examples.validation.run_validation --case MC-EQ-001 \
  --regenerate-matcont --generated-dir /tmp/jaxcont-candidates
python3 -m examples.validation.run_validation --case MC-EQ-001 \
  --verify-references --generated-dir /tmp/jaxcont-candidates
```

Candidates remain separate from the reviewed references. A custom flat
`--reference-dir` remains supported for candidate checks. MATLAB drivers are
under `regenerate/matcont/`, with capability gaps and templates separated.
Julia scripts are under `regenerate/bifurcationkit/`; their headless export
rewrite is a later task. The [comparison gallery](gallery/) now contains cubic, Van der Pol,
adaptx Hopf/iPRC, radial, torBPC and independent PRC-shooting pages. Each page
runs fresh JaxCont computations, renders report-driven tables and writes PNG
figures with matching `.checks.json` sidecars. For example:

```sh
python3 examples/validation/gallery/plot_cubic.py --output-dir /tmp/jaxcont-evidence
```

Figures remain open for Sphinx-Gallery capture. The 16 public API lessons
live separately in `../tutorials/`. Later comparison pages and external
series remain future work; no missing result is substituted with JaxCont data.

## v2 bundles and opt-in runtimes

`metadata.json` and `branch.csv`, `events.csv`, `spectra.csv` form a v2 bundle.
The metadata contract is in `reference/schema-v2.json`; `load_bundle` also
checks file hashes, table columns, point identities and linkages. Optional
orbit/PRC CSVs must also be hashed. The legacy readers remain available;
`convert_v1_bundle` writes a new candidate with a conversion manifest and an
unchanged copy of the historical files. Unknown historical fields stay
`unknown`. Conversion never grants review approval.

Registry selection accepts canonical IDs and legacy aliases. Alias collisions
are errors. Regenerate only into a temporary or generated directory:

```sh
python3 -m examples.validation.run_validation --regenerate matcont --case cubic.eq-fold --output-dir /tmp/jaxcont-candidates
python3 -m examples.validation.run_validation --regenerate bifurcationkit --case cubic.eq-fold --output-dir /tmp/jaxcont-candidates
```

`--regenerate-matcont` and `--generated-dir` remain aliases. MATLAB_BIN and
MATCONT_ROOT, or JULIA_BIN, BIFURCATIONKIT_PROJECT and JULIA_DEPOT_PATH, are
resolved only for explicit external work. CLI overrides take precedence.
`--verify-references` compares candidates numerically with the reviewed
bundles. A producer that has not been implemented is visibly PENDING and
returns nonzero; historical interactive Julia scripts are not executed by the
new runner. Nothing installs during tests or documentation builds.

`ValidationReport` records every present tool pair, independent residual and
coverage checks, limits, verdicts and source hashes. Any failed predicate
makes the report FAIL; missing required evidence makes it PENDING. The v2 CLI
uses that report as its gate. Legacy producers retain their established
case-specific numerical checks; their reports expose incomplete external
coverage separately, so a legacy two-tool PASS is not a three-tool claim.
Publication and new renderers must use the report's overall status.

The `matlab` and `julia` markers are deselected by default. Explicit selection
requires a working runtime and fails when one is missing. See
`regenerate/bifurcationkit/README.md` for the pinned Julia setup command.

## Lorenz-84 contract awaiting owner review

Review `cases/systems/lorenz84.py`, the four `lorenz84.*` registry records,
and `tests/test_reference_systems.py` before starting any flagship producer.
The contract fixes state order X,Y,Z,U, parameter order F,T, the original time
scale, model constants, the T=0.04 F sweep from Julia 02, Newton correction of
the historical starting guess, selection windows, and chart conventions.
The tests contain hand-evaluated RHS and analytic Jacobian samples independent
of continuation results. Selection requires exactly one matching event by
type and named window; numeric special-point indices are not an interface.
The codimension-two comparison domain is the existing positive-T BT chart;
Hopf termination at a chart boundary is recorded rather than called BT.
The contract passed Claude review 1 (2026-10-28): the Hopf seed window was
narrowed to the Hopf point nearest the BT, the full F-sweep event set (one LP,
three H) was recorded, and branches are clipped to the F window per sheet.
All four runs remain PENDING until their bundles are produced and promoted.
