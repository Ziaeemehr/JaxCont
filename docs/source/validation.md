# Validation against MatCont

**Validation snapshot (20 August 2026): seven of the eight supported cases pass.**

| Case | What is checked | Result |
| --- | --- | --- |
| `MC-EQ-001` | Cubic equilibrium continuation, folds, residuals, stability, and fold coefficients | PASS |
| `MC-EQ-002` | Van der Pol Hopf location, frequency, stability change, and Lyapunov coefficient | PASS |
| `MC-EQ-003` | Adaptive-control Hopf location, frequency, residual, and Lyapunov coefficient | PASS |
| `MC-JAX-001` | Eager/JIT and `vmap` agreement, permutation invariance, and sensitivities | PASS |
| `MC-C2-001` | Direct CP/BT/GH/ZH/HH point solvers, sensitivities, and normal forms | PASS |
| `MC-LC-001` | Radial periodic orbit radius, period, collocation residual, multipliers, and stability | PASS |
| `MC-LC-002` | `torBPC1` limit-cycle LPC/NS/PD locations, periods, extrema, and multipliers | FAIL |
| `MC-PRC-001` | adaptx iPRC curve against MatCont's PRC/Input processor output | PASS |

`MC-LC-002` is the current limitation: event-location errors remain small, but JaxCont is missing the LPC and PD event labels, and its maximum critical-multiplier error is approximately `1.13e-2`.

Because this comparison remains an open, tracked failure, treat JaxCont's fold-of-cycles, period-doubling, and Neimark-Sacker detection as experimental rather than fully validated until it closes.

The reviewed references were generated with MATLAB R2020a and MatCont 7.6, then committed as normalized CSV/JSON artifacts. Comparisons use the declared case tolerances and the suite's policy for interpolated branch segments, unique event type/location assignments, and tolerance-feasible eigenvalue/Floquet-spectrum matching; Floquet comparisons remove exactly one trivial multiplier nearest `+1`.

Reproduce the CPU validation snapshot from the repository root:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/mpl-jaxcont-validation python3 -m examples.MatCont.run_validation
```

Exit status `1` is expected while `MC-LC-002` remains failing. See the [complete MatCont validation-suite README](https://github.com/Ziaeemehr/JaxCont/blob/main/examples/MatCont/README.md) for case definitions, artifacts, the full comparison policy, and the unsupported-capabilities matrix.
