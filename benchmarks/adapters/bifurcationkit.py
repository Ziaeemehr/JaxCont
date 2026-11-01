"""BifurcationKit benchmark worker: Julia runs ``bifurcationkit_worker.jl``; Python measures accuracy.

Only ``cubic.eq-fold`` is implemented, in the pinned BifurcationKit 0.8.8
environment of ``examples/validation/regenerate/bifurcationkit``. Other
workloads, and a machine without Julia (``JULIA_BIN``), raise
``NotImplementedError`` so the supervisor records the row as PENDING.

The Julia process prints ``READY`` after loading packages, so the
supervisor's startup covers Python, Julia launch and package load; the first
call includes Julia's compilation of the continuation, as JaxCont's does.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.accuracy import branch_residual, measure
from benchmarks.contracts import ResolvedControls, controls_record, load_cases

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "examples" / "validation" / "regenerate" / "bifurcationkit"
SUPPORTED = {"cubic.eq-fold"}


def resolved_controls(case) -> ResolvedControls:
    """The controls ``bifurcationkit_worker.jl`` sets, converted to the contract's metric.

    With PALC(θ=0.5) BK's norm is sqrt(0.5) times the contract norm; the worker
    passes ds * sqrt(0.5), so in contract units the step and weights are equal.
    """
    target = case.controls
    n_state = 1
    return ResolvedControls(
        newton_atol=target.newton_atol, newton_rtol=0.0, newton_max_iter=target.newton_max_iter,
        # Folds are located by bisection on ds (dsmin_bisection), not to an event tolerance.
        event_atol=None,
        ds=target.ds, ds_min=target.ds_min, ds_max=target.ds_max,
        adaptive=not target.ds_min == target.ds_max, max_steps=target.max_steps,
        direction=target.direction, bounds=tuple(target.bounds),
        state_weight=1.0 / n_state, parameter_weight=1.0, stability_every_step=True,
        ntst=None, ncol=None,
        # Palc.jl: normAC(resf, resn) = max(norm(resf), abs(resn)).
        newton_criterion="augmented-residual-max(2norm-equations,abs-constraint)",
    )


def _julia():
    julia = os.environ.get("JULIA_BIN") or shutil.which("julia")
    if not julia or not shutil.which(julia):
        raise NotImplementedError("Julia unavailable (set JULIA_BIN)")
    return julia


def worker(benchmark_id, mode, repetitions, warmups):
    (case,) = [c for c in load_cases() if c.benchmark_id == benchmark_id]
    if case.canonical_id not in SUPPORTED:
        raise NotImplementedError(f"{case.canonical_id}: BifurcationKit benchmark worker covers cubic.eq-fold only")
    command = [_julia(), f"--project={PROJECT}", "--threads=1", str(HERE / "bifurcationkit_worker.jl"),
               mode, str(repetitions), str(warmups), json.dumps(controls_record(case.controls))]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        if line.strip() == "READY":
            print("READY", flush=True)
        elif line.startswith("{"):
            row = json.loads(line)
            if "folds" in row:
                states, params = row.pop("states"), row.pop("params")
                residual = branch_residual(states, params, lambda u, p: p + u - u**3 / 3)
                row["accuracy"] = measure(case, {"LP": row.pop("folds")}, residual, {"LP": row.pop("fold_states")})
                row["controls"] = controls_record(resolved_controls(case))
            print(json.dumps(row), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"Julia exited with status {process.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("first_call", "warm"), required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    options = parser.parse_args()
    worker(options.case, options.mode, options.repetitions, options.warmups)
    sys.exit(0)
