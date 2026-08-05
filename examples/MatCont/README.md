# MatCont validation suite

This directory contains reproducible cross-validation cases for JaxCont. The
reviewed references were regenerated with MATLAB R2020a and MatCont 7.6 and are
stored as normalized CSV/JSON text in `reference/`; `.mat` files, plots, and
frozen JaxCont outputs are intentionally excluded.

## Run the suite

From the repository root:

```bash
# Run every supported JaxCont/analytic comparison from reviewed references.
JAX_PLATFORMS=cpu python3 -m examples.MatCont.run_validation

# Run one case.
python3 -m examples.MatCont.run_validation --case MC-EQ-001

# Regenerate MatCont output without overwriting reviewed references.
python3 -m examples.MatCont.run_validation --regenerate-matcont

# Compare generated/ with the reviewed reference/ oracle.
python3 -m examples.MatCont.run_validation --verify-references
```

The defaults are
`/home/ziaee/prog/Matlab/R2020a/bin/matlab` and
`/home/ziaee/prog/MatCont/MatCont7p6`. Override them with `--matlab-bin` and
`--matcont-root`, or the `MATLAB_BIN` and `MATCONT_ROOT` environment variables.
Use `--reference-dir` and `--generated-dir` for alternate artifact locations.
`--case` may be repeated.

MATLAB producers are also standalone:

```matlab
cd examples/MatCont/matlab
run_supported('../generated')
run_cubic_fold('../generated')
run_torbpc_cycle('../generated')
```

Each producer writes `<case>_branch.csv`, `<case>_events.csv`,
`<case>_multipliers.csv`, and `<case>_metadata.json`. Metadata records equation
hashes, source/provenance, MATLAB, MatCont, JaxCont, Python and JAX versions,
precision, mesh and solver settings. Regeneration writes only to `generated/`;
verification never promotes or edits reviewed files.

## Supported coverage

| Case | Validation |
|---|---|
| `MC-EQ-001` | Cubic S-curve, natural versus pseudo-arclength continuation, two folds, residuals, stability and fold coefficients |
| `MC-EQ-002` | Van der Pol Hopf at `mu=0`, frequency 1, stability change and degenerate `l1=0` |
| `MC-EQ-003` | MatCont adaptive-control Hopf near `alpha=1` with `l1=-0.3` |
| `MC-LC-001` | Radial cycle collocation, radius, period `2*pi`, residual and exact Floquet spectrum |
| `MC-LC-002` | MatCont `torBPC1` LPC/NS/PD locations, periods, extrema and critical multipliers |
| `MC-JAX-001` | Eager/JIT equivalence, `vmap`, permutation invariance, fold/Hopf gradients and finite differences |
| `MC-C2-001` | Direct shifted CP/BT/GH/ZH/HH solves, sensitivities, fold/GH normal forms and Lorenz-84 BifurcationKit BT reference |

`MC-LC-002` deliberately reports the current JaxCont mismatch: MatCont's LPC,
NS and PD references are retained, while JaxCont presently misses the correctly
located LPC/PD event labels and exceeds the critical-multiplier tolerance at
NS. The CLI exits nonzero for this case. Do not hide that result by changing
tolerances.

## Unsupported matrix

These MatCont-only wrappers live in `matlab/unsupported`. They are excluded by
default. `--include-unsupported --dry-run` prints the matrix without running
MATLAB; omitting `--dry-run` executes selected wrappers but reports
`UNSUPPORTED_BY_JAXCONT`, never `PASS`.

| Registry case | MatCont capability not yet supported by JaxCont |
|---|---|
| `US-BP-001` | Bratu equilibrium branch switching |
| `US-C2-LP-001` | Two-parameter fold curves |
| `US-C2-H-001` | Two-parameter Hopf curves |
| `US-C2-PD-001` | Two-parameter period-doubling curves |
| `US-C2-LPC-001` | Two-parameter limit-point-of-cycles curves |
| `US-C2-NS-001` | Two-parameter Neimark-Sacker curves |
| `US-BVP-001` | General BVP continuation beyond supported periodic collocation |
| `US-HOM-001` | Homoclinic continuation |
| `US-HET-001` | Heteroclinic continuation (requires problem-specific endpoint seeds) |
| `US-PRC-001` | PRC/dPRC calculation |

Direct CP, BT, GH, ZH and HH point solvers are supported and are therefore not
in this table. Continuation of their two-parameter curves remains unsupported.

## Layout and comparison policy

- `cases.json` is the capability/case registry.
- `python_cases/` contains the public-API JaxCont and analytic validators.
- `matlab/` contains shared exporters and standalone producers.
- `reference/` contains reviewed normalized oracles.
- `generated/` is ignored scratch output.

Branches are compared by interpolation on monotone segments or by scaled
point-set distance. Events require unique type/location assignments. Spectra
are matched by a tolerance-feasible assignment after removing exactly one
trivial multiplier nearest `+1`. Numerical disagreements are failures, not a
reason to relax the registry tolerances.
