# Examples

The [tutorials](tutorials/) each teach one public API workflow. Run from the
repository root with JaxCont installed, for example:

```bash
python3 examples/tutorials/plot_01_cubic_fold.py --output-dir /tmp/jaxcont-tutorials
```

The [validation gallery](validation/gallery/) compares fresh JaxCont results
with committed reference series. Its plots, report tables and `.checks.json`
sidecars keep failed comparisons visible. See the
[validation README](validation/README.md) for case IDs, provenance and opt-in
regeneration. MATLAB and Julia are unnecessary for tutorials or gallery pages.

Sphinx scans only `tutorials/` and `validation/gallery/`. Output defaults to
`/tmp/jaxcont/generated` (the platform temporary directory); use `--output-dir`
to select another directory. Figures remain open for Sphinx-Gallery capture.
Developer diagnostics live in `scripts/check_installation.py` and
`scripts/profile_continuation.py`. The [migration map](../docs/source/examples-migration.json)
records the old paths and gallery redirects.
