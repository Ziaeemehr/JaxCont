"""MatCont benchmark worker: MATLAB runs ``matcont_worker.m``; Python measures accuracy.

Only ``cubic.eq-fold`` is implemented. Other workloads, and a machine without
MATLAB (``MATLAB_BIN``) or MatCont (``MATCONT_ROOT``), raise
``NotImplementedError`` so the supervisor records the row as PENDING.

The MATLAB process prints ``READY`` after path setup; this wrapper forwards it
at once, so the supervisor's startup time covers Python, MATLAB launch and
MatCont setup. Timings come from MATLAB's ``tic``/``toc`` around ``cont``.
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
SUPPORTED = {"cubic.eq-fold"}


def resolved_controls(case) -> ResolvedControls:
    """The controls ``matcont_worker.m`` sets, in the contract's terms."""
    target = case.controls
    return ResolvedControls(
        newton_atol=target.newton_atol, newton_rtol=0.0,  # absolute tolerances only
        newton_max_iter=target.newton_max_iter,  # MaxCorrIters
        event_atol=target.event_atol,  # TestTolerance
        ds=target.ds, ds_min=target.ds_min, ds_max=target.ds_max,
        # MatCont always adapts between MinStepsize and MaxStepsize.
        adaptive=not target.ds_min == target.ds_max, max_steps=target.max_steps,
        direction=target.direction,
        # MatCont has no parameter bounds; it stops after MaxNumPoints.
        bounds=None,
        # Euclidean norm on (x, p): state and parameter weight 1.
        state_weight=1.0, parameter_weight=1.0, stability_every_step=True,
        ntst=None, ncol=None,
        # newtcorr requires both ||F|| < FunTolerance and ||dx|| < VarTolerance.
        newton_criterion="residual-2norm-and-step-2norm",
    )


def _matlab():
    matlab = os.environ.get("MATLAB_BIN") or shutil.which("matlab")
    if not matlab or not shutil.which(matlab) or not os.environ.get("MATCONT_ROOT"):
        raise NotImplementedError("MATLAB or MatCont unavailable (set MATLAB_BIN and MATCONT_ROOT)")
    return matlab


def _cubic_residual(states, params):
    return branch_residual(states, params, lambda u, p: p + u - u**3 / 3)


def _as_list(value):
    return value if isinstance(value, list) else [value]


def worker(benchmark_id, mode, repetitions, warmups):
    (case,) = [c for c in load_cases() if c.benchmark_id == benchmark_id]
    if case.canonical_id not in SUPPORTED:
        raise NotImplementedError(f"{case.canonical_id}: MatCont benchmark worker covers cubic.eq-fold only")
    controls = controls_record(case.controls)
    call = (f"addpath('{HERE}'); matcont_worker('{mode}', {repetitions}, {warmups}, "
            f"'{json.dumps(controls)}')")
    # -singleCompThread matches the supervisor's one-thread pinning for the Python tools.
    process = subprocess.Popen([_matlab(), "-singleCompThread", "-batch", call], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        if line.strip() == "READY":
            print("READY", flush=True)
        elif line.startswith("{"):
            row = json.loads(line)
            if "folds" in row:
                states, params = _as_list(row.pop("states")), _as_list(row.pop("params"))
                row["accuracy"] = measure(case, {"LP": _as_list(row.pop("folds"))},
                                          _cubic_residual(states, params),
                                          {"LP": _as_list(row.pop("fold_states"))})
                row["controls"] = controls_record(resolved_controls(case))
            print(json.dumps(row), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"MATLAB exited with status {process.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("first_call", "warm"), required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    options = parser.parse_args()
    worker(options.case, options.mode, options.repetitions, options.warmups)
    sys.exit(0)
