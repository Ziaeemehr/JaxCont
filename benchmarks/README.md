# Benchmarks

Fair, accuracy-qualified timings of the validation workloads in JaxCont,
MatCont and BifurcationKit. There are no rankings and no aggregate speedups:
results are reported per workload and per stage.

## Contract

`cases.json` lists the workloads. Each row is a validation run (its
`canonical_id`), a fixed-step or adaptive step mode, the frozen target
controls, and the accuracy limits. `contracts.py` loads and validates it.

- **Equal controls.** Before timing, each adapter resolves the controls its
  tool actually used (after defaults and unit conversion) into
  `ResolvedControls`. `compare_controls` compares them by exact equality of
  normalized decimals. Any missing, unequal or unrepresentable field makes the
  row non-comparable and records the field, tool and value. The arclength
  metric is part of the controls: JaxCont weights the state by 1/n_state and
  the parameter by 1, so an equal numeric `ds` under another metric is a
  different physical step.
- **Equal accuracy.** `accuracy_limits` are validation-report metrics (errors
  against the committed reference bundles), equal to the registry's
  `metric_limits` for the run. They are separate from solver stopping
  tolerances. A row is shown across tools only when every tool passes them.
- **Separate stages.** Startup, first call, compile, warm, end-to-end and
  batched throughput are separate categories. An unavailable stage is `null`
  with a reason, never 0.

Fixed-step rows come first; adaptive rows are a separate experiment.
Measuring MATLAB and Julia timings is opt-in. Reviewed records are committed
to `results/reviewed/`, and the docs render only those records.
