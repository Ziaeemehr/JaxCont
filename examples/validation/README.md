# Validation tooling

Run from the repository root with JaxCont installed:

```sh
python3 -m examples.validation.run_validation --case MC-EQ-001
python3 -m examples.validation.run_validation --dry-run
```

The registry is `cases/cases.json` (23 cases). Cases are selected by
canonical ID (for example `cubic.eq-fold`, `lorenz84.bt-ft`) or by a legacy
alias (`MC-EQ-001`); alias collisions are errors. The status of every case,
its committed bundles and their promotion verdicts are listed on the
generated docs page `docs/source/validation.md`
(`python3 -m examples.validation.docs_page`).

Reference bundles live under `reference/<system>/<tool>/<run>/`. Current v2
bundles from MatCont 7.6 (MATLAB R2024a) and BifurcationKit 0.8.8 (Julia
1.13.3) carry a `promotion-manifest.json` with the reviewed verdict; they
exist for cubic, radial, adaptx PD switching and the four Lorenz-84 runs.
The six historical MatCont bundles (22 files, MATLAB 9.8) keep their legacy
IDs (`reference/<system>/matcont/<legacy-id>/`) with unchanged bytes and
provenance. `MC-PRC-001` is **adaptx**. The PRC comparison checks MatCont's
time derivative separately from JaxCont's parameter derivative; it does not
claim an external d(PRC)/dp validation.

The torBPC comparison passes on the external tools' mesh at JaxCont's
refined events; a failed comparison remains FAIL in the CLI and the
renderer, and no tolerance was widened. The registry distinguishes executable capability gaps from
non-executable BVP and heteroclinic templates. Unsupported cases require
`--include-unsupported`; templates never report successful execution.

External regeneration is opt-in. MATLAB R2024a and MatCont 7.6 are configured
through `MATLAB_BIN` and `MATCONT_ROOT` (CLI overrides take precedence):

```sh
python3 -m examples.validation.run_validation --case MC-EQ-001 \
  --regenerate-matcont --generated-dir /tmp/jaxcont-candidates
python3 -m examples.validation.run_validation --case MC-EQ-001 \
  --verify-references --generated-dir /tmp/jaxcont-candidates
```

Candidates remain separate from the reviewed references; promotion is a
reviewed step (`scripts/promote_reference.py`). MATLAB drivers are under
`regenerate/matcont/`, with capability gaps and templates separated; the
headless Julia exporters are under `regenerate/bifurcationkit/`.

The [comparison gallery](gallery/) has one page per system: cubic, Van der
Pol, adaptx (Hopf and iPRC), radial, torBPC, independent PRC shooting and the
three-tool Lorenz-84 page. Each page runs a fresh JaxCont computation, reads
the committed bundles (never MATLAB or Julia), renders report-driven tables
and writes PNG figures with matching `.checks.json` sidecars. For example:

```sh
python3 examples/validation/gallery/plot_cubic.py --output-dir /tmp/jaxcont-evidence
```

Figures remain open for Sphinx-Gallery capture. The public API lessons live
separately in `../tutorials/`. No missing external result is substituted with
JaxCont data.

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

## Lorenz-84 contract

`cases/systems/lorenz84.py` fixes state order X,Y,Z,U, parameter order F,T,
the original time scale, model constants, the T=0.04 F sweep (one LP and
three H), Newton correction of the historical starting guess, the selection
windows (fold seed F in [1.50, 1.58], Hopf seed F in [1.59, 1.65], BT in
F [1.40, 1.50] with T in [0.01, 0.03]) and the chart conventions. Selection
requires exactly one matching event by type and named window; numeric
special-point indices are not an interface. The codimension-two comparison
domain is the positive-T BT chart; Hopf termination at a chart boundary is
recorded rather than called BT. BifurcationKit's BT on its fold and Hopf
curves has native status `guess`; only the refined BT (`lorenz84.bt-ft`) is
compared. All four runs (`eq-f`, `fold-ft`, `hopf-ft`, `bt-ft`) PASS against
both external tools; the producers run in float64 (Newton tolerance 1e-12).
