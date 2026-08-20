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

# Generate the complete visual-comparison gallery.
JAX_PLATFORMS=cpu python3 -m examples.example_16_matcont_cubic_overlay
JAX_PLATFORMS=cpu python3 -m examples.example_17_matcont_vanderpol_overlay
JAX_PLATFORMS=cpu python3 -m examples.example_18_matcont_adaptive_control_overlay
JAX_PLATFORMS=cpu python3 -m examples.example_19_matcont_radial_cycle_overlay
JAX_PLATFORMS=cpu python3 -m examples.example_20_matcont_torbpc_overlay

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
`--case` may be repeated. Relative reference/generated overrides are resolved
against the caller's working directory before MATLAB changes directory.

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

## Visual comparison gallery

The five root modules below are Sphinx-Gallery-compatible visual companions to
the systematic checks. Each runs the registered JaxCont producer and overlays
its fresh result on reviewed MatCont 7.6 CSV artifacts; no MATLAB runtime is
needed. Run them from the repository root with the commands in the previous
section. Each command writes its PNG below `images/` relative to the current
working directory.

| Module | Output | Panels and visual meaning |
|---|---|---|
| `examples.example_16_matcont_cubic_overlay` | `images/matcont_cubic_overlay.png` | Cubic equilibrium branch with JaxCont and MatCont fold (`LP`) markers. |
| `examples.example_17_matcont_vanderpol_overlay` | `images/matcont_vanderpol_overlay.png` | Equilibrium branch and spectral abscissa, with the Hopf (`H`) stability crossing. |
| `examples.example_18_matcont_adaptive_control_overlay` | `images/matcont_adaptive_control_overlay.png` | Adaptive-control equilibrium branch and spectral abscissa, with the Hopf (`H`) crossing. |
| `examples.example_19_matcont_radial_cycle_overlay` | `images/matcont_radial_cycle_overlay.png` | Periodic-orbit envelope, period, and nontrivial Floquet-multiplier modulus. |
| `examples.example_20_matcont_torbpc_overlay` | `images/matcont_torbpc_overlay.png` | torBPC envelope and period panels plus an event-centred Floquet multiplier plane. |

The packages retain their native adaptive meshes, so their plotted samples do
not need to coincide point-for-point. Every panel is limited to the shared
continuation-parameter domain; visual agreement means that the independently
sampled geometry, stability, and event locations are consistent. The
validation CLI remains authoritative: it performs interpolated branch, event,
stability, spectrum, and tolerance checks against the reviewed references.

The torBPC page (`MC-LC-002`) is deliberately a known-failing diagnostic, not
evidence of complete correspondence. It retains MatCont LPC, NS, and PD
references. JaxCont reports raw LPC and NS detections, but its detected LPC is
displaced and therefore is not a correctly located match; PD is not detected.
The strict comparison consequently lists LPC and PD as missing correctly
located event types. Its `JaxCont near <type>` multiplier points are spectra
at the nearest JaxCont branch samples, not event detections. The failure banner
calls their location differences proxy discrepancies and preserves the
unchanged numerical tolerances described by the
[supported-coverage warning](#supported-coverage) below.

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
| `MC-PRC-001` | adaptx Hopf limit cycle, iPRC adjoint-method curve against MatCont's PRC/Input processor output (phase-in-radians vs. phase-in-cycle-fractions and phase-origin conventions reconciled; dPRC is *not* cross-checked here -- MatCont's exported dPRC is `d(PRC)/dt`, confirmed via `LimitCycle/calcPRC.m`, a different quantity from JaxCont's `dprc_curve` = `d(PRC)/d(alpha)`, which is validated instead by `tests/test_prc.py`) |

`MC-LC-002` deliberately reports the current JaxCont mismatch: MatCont's LPC,
NS and PD references are retained. JaxCont's detected LPC is displaced enough
not to count as a correctly located match, PD is not detected, and the strict
missing-event list is therefore LPC and PD. JaxCont also exceeds the
critical-multiplier tolerance at NS. The CLI exits nonzero for this case. Do
not hide that result by changing tolerances.

## Validation gaps and unsupported matrix

JaxCont v0.4.0 supports equilibrium fold and Hopf curves through
`fold_curve_problem` and `hopf_curve_problem`. The MatCont wrappers
`US-C2-LP-001` and `US-C2-H-001` remain in the registry as validation gaps:
the suite does not yet pair those MatCont runs with JaxCont producers and
reviewed comparison artifacts. They must not be read as library capability
gaps.

These MatCont-only wrappers live in `matlab/unsupported`. They are excluded by
default, and explicitly requesting one without `--include-unsupported` is an
error. `--include-unsupported --dry-run` prints the matrix without running
MATLAB. Executable wrappers report `UNSUPPORTED_BY_JAXCONT`, never `PASS`.
General-BVP and heteroclinic entries are honest non-executable setup templates:
selecting one without `--dry-run` exits nonzero with `NON_EXECUTABLE_TEMPLATE`.

| Registry case | MatCont capability not yet supported by JaxCont |
|---|---|
| `US-BP-001` | Bratu equilibrium branch switching |
| `US-C2-PD-001` | Two-parameter period-doubling curves |
| `US-C2-LPC-001` | Two-parameter limit-point-of-cycles curves |
| `US-C2-NS-001` | Two-parameter Neimark-Sacker curves |
| `US-BVP-001` | General BVP setup template beyond supported periodic collocation (non-executable) |
| `US-HOM-001` | Homoclinic continuation |
| `US-HET-001` | Heteroclinic setup template requiring problem-specific endpoint seeds (non-executable) |

Direct CP, BT, GH, ZH and HH point solvers are supported and are therefore not
in this table. Fold and Hopf curves, including their supported codimension-two
events, are also available. Period-doubling, limit-point-of-cycles, and
Neimark-Sacker curve continuation remain unsupported.

## Layout and comparison policy

- `cases.json` is the capability/case registry.
- `python_cases/` contains the public-API JaxCont and analytic validators.
- `matlab/` contains shared exporters and standalone producers.
- `reference/` contains reviewed normalized oracles.
- `generated/` is ignored scratch output.

Branches are compared by interpolation on monotone segments. Events require
unique type/location assignments. Eigenvalues and Floquet spectra are matched
by tolerance-feasible assignment; exactly one trivial multiplier nearest `+1`
is removed only for Floquet comparisons. The offline CLI compares the JaxCont
branch, event and spectrum data numerically with the committed MatCont files;
analytic self-checks alone cannot make a case pass. Numerical disagreements are
failures, not a reason to relax the registry tolerances.
