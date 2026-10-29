"""Benchmark supervisor: separate startup, first-call and warm timings per tool.

  python3 -m benchmarks.run_benchmarks --case cubic.eq-fold/fixed --tool jaxcont

Startup is timed here, from process launch to the worker's ``READY`` line.
Each first call runs in its own fresh process; warm repetitions run in one
process after untimed warm-ups. Raw records go to ``benchmarks/results/raw/``
(git-ignored); only reviewed records are committed. No timing assertions run
in CI.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.contracts import load_cases

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks" / "results" / "raw"
WORKERS = {"jaxcont": "benchmarks.adapters.jaxcont", "matcont": "benchmarks.adapters.matcont",
           "bifurcationkit": "benchmarks.adapters.bifurcationkit"}
# Workers follow one READY/JSON protocol. A tool without a worker, a workload
# a worker does not cover, or a missing MATLAB/Julia install is PENDING,
# never timed with other controls.
PENDING_TOOLS: dict[str, str] = {}


def _environment():
    """Pinned threads make CPU timings comparable across runs."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _cpu():
    """CPU model and logical core count: the named machine, without a hostname."""
    try:
        model = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        model = platform.processor() or "unknown"
    return {"model": model, "logical_cores": os.cpu_count()}


def run_worker(tool, benchmark_id, mode, timeout, extra=(), clock=time.perf_counter):
    """Launch one worker; return (startup_seconds, records).

    Stdout is read on a thread so a silent, hung worker still hits the
    deadline; stderr goes to a temporary file so it can never fill a pipe.
    """
    command = [sys.executable, "-m", WORKERS[tool], "--case", benchmark_id, "--mode", mode, *extra]
    with tempfile.TemporaryFile("w+") as stderr:
        launched = clock()
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, env=_environment(), cwd=ROOT)
        lines = queue.Queue()
        threading.Thread(target=lambda: [*map(lines.put, process.stdout), lines.put(None)], daemon=True).start()
        startup, records = None, []
        deadline = launched + timeout
        while True:
            try:
                line = lines.get(timeout=max(0.0, deadline - clock()))
            except queue.Empty:
                line = ""
            if line is None:
                break
            if line.strip() == "READY":
                startup = clock() - launched
            elif line.startswith("{"):
                records.append(json.loads(line))
            if clock() > deadline:
                process.kill()
                process.wait()
                raise TimeoutError(f"{tool} {benchmark_id} {mode} exceeded {timeout} s")
        if process.wait() != 0 or startup is None:
            stderr.seek(0)
            raise RuntimeError(f"{tool} {benchmark_id} {mode} failed: {stderr.read()[-2000:]}")
    return startup, records


def eligible(row, limits):
    """A row enters cross-tool tables only if every accuracy limit is measured and met."""
    accuracy = row.get("accuracy") or {}
    return all(accuracy.get(metric) is not None and accuracy[metric] <= limit for metric, limit in limits.items())


def summarize(samples):
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    return {"n": len(ordered), "median": statistics.median(ordered), "iqr": quartiles[2] - quartiles[0],
            "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))], "samples": ordered}


def benchmark(benchmark_id, tool):
    (case,) = [c for c in load_cases() if c.benchmark_id == benchmark_id]
    sampling = case.sampling
    record = {"benchmark_id": benchmark_id, "canonical_id": case.canonical_id, "tool": tool,
              "generated_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "cpu": _cpu(),
              "python": sys.version.split()[0], "timer": "time.perf_counter", "accuracy_limits": dict(case.accuracy_limits)}
    if tool in PENDING_TOOLS:
        return record | {"status": "PENDING", "reason": PENDING_TOOLS[tool]}
    try:
        startups, first_calls, accuracy, controls = [], [], None, None
        for _ in range(sampling.first_calls):
            startup, (row,) = run_worker(tool, benchmark_id, "first_call", sampling.timeout_seconds)
            startups.append(startup)
            first_calls.append(row["seconds"])
            accuracy, controls, versions = row["accuracy"], row["controls"], row.get("versions")
            first_points = row.get("points")
        _, warm = run_worker(tool, benchmark_id, "warm", sampling.timeout_seconds,
                             ("--repetitions", str(sampling.warm_repetitions), "--warmups", str(sampling.untimed_warmups)))
    except NotImplementedError as exc:
        return record | {"status": "PENDING", "reason": str(exc)}
    except RuntimeError as exc:
        if "NotImplementedError" in str(exc):
            return record | {"status": "PENDING", "reason": str(exc).splitlines()[-1]}
        return record | {"status": "FAILED", "reason": str(exc)}
    except TimeoutError as exc:
        # A failure is a recorded outcome, never a timing and never a crash of the sweep.
        return record | {"status": "FAILED", "reason": str(exc)}
    warm_seconds = [row["seconds"] for row in warm]
    # Every timed repetition must do the same work as the first call.
    points = {"first_call": first_points, "warm": sorted({row.get("points") for row in warm}, key=str)}
    record |= {
        "status": "MEASURED", "points": points, "resolved_controls": controls, "accuracy": accuracy, "versions": versions,
        "sampling": {"first_calls": sampling.first_calls, "warm_repetitions": sampling.warm_repetitions,
                     "untimed_warmups": sampling.untimed_warmups, "timeout_seconds": sampling.timeout_seconds},
        "timer_resolution": time.get_clock_info("perf_counter").resolution,
        "eligible_for_cross_tool": eligible({"accuracy": accuracy}, case.accuracy_limits),
        "startup": summarize(startups), "first_call": summarize(first_calls), "warm": summarize(warm_seconds),
        # Labelled as an estimate: first call minus warm also includes Python-side work.
        "compile_overhead_estimate": {"seconds": statistics.median(first_calls) - statistics.median(warm_seconds),
                                      "estimate": True},
    }
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--tool", action="append", choices=("jaxcont", "matcont", "bifurcationkit"), required=True)
    parser.add_argument("--output-dir", type=Path, default=RAW)
    parser.add_argument("--seed", type=int, default=0, help="seed for the randomized tool order")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = random.Random(args.seed)
    for benchmark_id in args.case:
        # Tools run sequentially in a randomized, recorded order (no fixed first-runner advantage).
        tools = order.sample(args.tool, len(args.tool))
        for position, tool in enumerate(tools):
            record = benchmark(benchmark_id, tool) | {"tool_order": tools, "order_seed": args.seed,
                                                      "order_position": position}
            name = f"{benchmark_id.replace('/', '__')}__{tool}.json"
            (args.output_dir / name).write_text(json.dumps(record, indent=2) + "\n")
            print(f"{record['status']:9s} {benchmark_id} {tool}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
