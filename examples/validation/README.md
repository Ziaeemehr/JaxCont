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
rewrite is a later task. The tutorials and comparison galleries are planned
in the migration map and have not yet been created by this migration.
