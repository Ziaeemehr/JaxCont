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

## Running

```sh
export MATLAB_BIN=... MATCONT_ROOT=... JULIA_BIN=...   # opt-in external tools
python3 -m benchmarks.run_benchmarks --case cubic.eq-fold/fixed \
  --tool jaxcont --tool matcont --tool bifurcationkit < /dev/null
python3 -m benchmarks.reporting --check benchmarks/results/raw
python3 -m benchmarks.reporting --markdown benchmarks/results/raw
```

The supervisor starts one fresh worker per first call and one for the warm
repetitions, and times startup from launch to the worker's `READY` line. Tools
run in a randomized order recorded with its seed. Threads are pinned to one
(`OMP_NUM_THREADS=1` and XLA flags for JAX, `-singleCompThread` for MATLAB,
`--threads=1` for Julia). A worker that hangs, fails or times out is recorded
as `FAILED`; a tool, workload or install that is not available is `PENDING`.
The deadline covers both stdout reads and process exit.

New records retain every first-call and warm replicate's timing, status,
point count, controls, accuracy, and versions under `replicates`. Inconsistent
work, controls, versions, or unsuccessful replicates fail the row without a
successful timing summary. Accuracy uses the worst error across all replicates;
an unavailable measurement keeps the row ineligible. Nonfinite or negative
errors are invalid evidence. Accuracy extraction happens after each timed solve.

| Tool | Worker | Workloads | Notes |
|---|---|---|---|
| JaxCont | `adapters/jaxcont.py` | cubic, Lorenz-84 equilibria | periodic workloads PENDING |
| MatCont 7.6 | `adapters/matcont.py` + `matcont_worker.m` | cubic | no parameter bounds (stops at `MaxNumPoints`); Newton bounds both residual and step |
| BifurcationKit 0.8.8 | `adapters/bifurcationkit.py` + `bifurcationkit_worker.jl` | cubic | `PALC(θ=0.5)`, `ds` scaled by √0.5 to the contract metric; reports the cubic folds as `bp` |

Accuracy is computed by `accuracy.py` from each worker's raw branch and
events against one shared reference per workload (analytic folds for cubic,
the committed MatCont bundle otherwise), never from the tool's own report.
The resolved controls of the three tools differ in what the Newton tolerance
bounds (and MatCont in bounds), so the cubic rows are reported per tool
without a cross-tool table.

## Batched throughput

Run `python -m benchmarks.batching --case cubic.eq-fold/fixed --batch 1 --batch 8`
for serial Python loops of a jitted lane and `jit(vmap)` (select either with
`--strategy`). Defaults cover B=1,8,32,128. Both equilibrium systems and step
modes use deterministic seed-parameter perturbations; lane 0 is the serial
case. Periodic workloads and opt-in external `--tool` rows are PENDING.

Each row runs in a fresh CPU process with the harness's one-thread environment
and float64 by default (`JAX_ENABLE_X64=0` selects float32). JSON records in
`results/raw/` keep the synchronized first-call batch cost separate from warm
samples, per-lane valid counts and termination status, serial agreement (a
lane that differs from its serial run beyond a few ulps makes the row
`MISMATCH`, with no throughput), and
peak memory. CPU memory falls back to process-lifetime `host-maxrss`, including
compilation. Truncated and failed lanes remain in the success denominator;
only converged lanes contribute valid points per warm median second. These
records describe throughput; not a single-branch speedup. Event candidates
remain fixed buffers, without eager event refinement; these rows are not
eligible for cross-tool comparisons.

## Reviewed records

`results/reviewed/` holds the cubic (fixed and adaptive) and Lorenz-84
`eq-f` (fixed) rows measured on 2026-10-29 in float64 on one machine (CPU
model in each record), tools in randomized order with seed 20261029. A light
read-only review job was running at the same time. Reviewed by Claude: every
record passes `--check`, every tool does the same work in each warm
repetition, and accuracy is measured for every tool except the fold
coefficient, so no cross-tool table is drawn.

What the numbers show: on the scalar cubic, JaxCont's continuation itself
takes about 5 ms warm (8 ms with stability), but its warm total is about
0.22 s because eager refinement of the two folds re-traces `fold_point`
(about 0.14 s per call; profile in the Loop 2 ledger). MatCont needs about
10 s to start MATLAB; BifurcationKit's first call includes about 4 s of Julia
compilation. MatCont and BifurcationKit Lorenz-84 rows are PENDING (no
worker for that workload).

