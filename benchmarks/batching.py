"""Equilibrium throughput with independent, deterministic continuation lanes.

Lane i starts at p0 + direction * 0.01 * i / 127, with the adapter's state
seed Newton-corrected at that parameter. Thus every prefix B=1,8,32,128 uses
the same lanes, lane 0 is the serial case, and all seeds stay within 0.01 of
the well-posed reference seed. Equations, fixed arguments, end parameter,
settings and event candidates are unchanged for both step modes.

Serial means a Python loop over one jitted single-lane callable; jit(vmap)
compiles the mapped continuation. Both return fixed buffers and event
candidates, without eager event refinement. Batch compile seconds measure
the synchronized first call (including execution), not isolated compilation.
Each CLI row uses a fresh CPU worker with the supervisor's pinned threads.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.contracts import controls_record, load_cases
from benchmarks.run_benchmarks import PENDING_TOOLS, RAW, ROOT, _environment

BATCH_SIZES = (1, 8, 32, 128)
STRATEGIES = ("serial", "jit(vmap)")
LABEL = "throughput; not a single-branch speedup"


def build_lanes(case, B):
    """Build a traceable single-lane solve using the serial adapter's controls."""
    import jax.numpy as jnp
    import jaxcont as jc
    from benchmarks.adapters.jaxcont import build

    problem, settings, _ = build(case)
    direction = case.controls.direction
    inputs = jnp.asarray(problem.p0) + direction * (0.01 / 127) * jnp.arange(B)
    low, high = case.controls.bounds
    p_end = high if direction > 0 else low
    events = [jc.Fold()] + ([jc.Hopf()] if case.canonical_id == "lorenz84.eq-f" else [])

    def single(p0):
        return jc.continuation(problem.at(p0=p0), jc.PseudoArclength(),
                               p_span=(p0, p_end), settings=settings, events=events)

    return inputs, single


def lane_atol(reference):
    """A few ulps of the reference scale: batched lanes are bitwise equal to serial runs in practice."""
    import numpy as np

    reference = np.asarray(reference)
    return 8 * np.finfo(reference.dtype).eps * max(1.0, float(np.max(np.abs(reference), initial=0.0)))


def make_batch(single, strategy):
    """Keep callable identity across warm samples and reset all lane seeds."""
    import jax

    if strategy == "jit(vmap)":
        return jax.jit(jax.vmap(single))
    if strategy != "serial":
        raise ValueError(f"unknown strategy {strategy!r}")
    solve = jax.jit(single)

    def serial(inputs):
        # Lanes stay a list here; stacking is host work done by as_batch, outside the timer.
        return [solve(value) for value in inputs]

    return serial


def as_batch(result):
    """Stack a serial list of lane results into the batched layout; pass a batch through."""
    import jax
    import jax.numpy as jnp

    if isinstance(result, list):
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *result)
    return result


def lane_summary(result, warm_seconds):
    """Only completed, finite lanes contribute points; all lanes count as trials."""
    import numpy as np
    from jaxcont.core.scan_continuation import Status

    masks = np.asarray(result.branch.valid, dtype=bool)
    counts = masks.sum(axis=1)
    reported = np.asarray(result.stats["n_valid"])
    codes = np.asarray(result.stats["status"])
    states, params = np.asarray(result.branch.states), np.asarray(result.branch.params)
    statuses = []
    for i, (count, code) in enumerate(zip(counts, codes)):
        finite = np.isfinite(states[i][masks[i]]).all() and np.isfinite(params[i][masks[i]]).all()
        if count == 0 or count != reported[i] or not finite:
            status = "failed"
        elif code == Status.REACHED_END:
            status = "converged"
        elif code in (Status.MAX_STEPS, Status.MAX_ATTEMPTS):
            status = "truncated"
        else:
            status = "failed"
        statuses.append(status)
    ok = [status == "converged" for status in statuses]
    points = sum(int(count) for count, success in zip(counts, ok) if success)
    return {"lane_status": statuses, "valid_points": counts.tolist(),
            "lanes_total": len(statuses), "lanes_ok": sum(ok),
            "success_fraction": sum(ok) / len(statuses), "successful_valid_points": points,
            "throughput": points / statistics.median(warm_seconds)}


def peak_memory():
    """Device peak if exposed, otherwise process-lifetime host high-water RSS."""
    import jax

    reason = "device peak memory statistics unavailable"
    try:
        stats = jax.devices()[0].memory_stats()
        peak = (stats or {}).get("peak_bytes_in_use")
        if peak is not None and peak > 0:
            return {"bytes": int(peak), "source": "device-memory-stats", "reason": None,
                    "scope": "worker lifetime, including compilation"}
    except (RuntimeError, NotImplementedError) as exc:
        reason = f"device memory statistics unavailable: {exc}"
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 0:
            return {"bytes": int(rss * (1 if sys.platform == "darwin" else 1024)),
                    "source": "host-maxrss", "reason": reason,
                    "scope": "worker lifetime, including compilation"}
    except (OSError, ValueError) as exc:
        reason += f"; host maxrss unavailable: {exc}"
    return {"bytes": None, "source": "host-maxrss", "reason": reason + "; no positive host maxrss"}


def measure(case, B, strategy, tool="jaxcont", clock=time.perf_counter):
    """Measure one fresh callable; validation and host conversion are untimed."""
    if B < 1 or strategy not in STRATEGIES:
        raise ValueError("positive B and a supported strategy are required")
    record = {"benchmark_id": case.benchmark_id, "canonical_id": case.canonical_id,
              "tool": tool, "strategy": strategy, "B": B, "label": LABEL,
              "step_mode": case.step_mode, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "batch_compile_seconds": None, "warm_seconds": [],
              "lane_status": None, "valid_points": None, "lanes_total": B, "lanes_ok": None,
              "success_fraction": None, "throughput": None,
              "peak_memory": {"bytes": None, "source": None, "reason": "row not measured"},
              "eligible_for_cross_tool": False}
    reason = PENDING_TOOLS.get(tool)
    if tool != "jaxcont" and reason is None:
        raise ValueError(f"unknown tool {tool!r}")
    if reason:
        return record | {"status": "PENDING", "reason": reason}
    try:
        inputs, single = build_lanes(case, B)
    except NotImplementedError as exc:
        return record | {"status": "PENDING", "reason": str(exc)}

    import jax
    import numpy as np
    from benchmarks.adapters.jaxcont import build, resolved_controls, synchronize

    run = make_batch(single, strategy)
    synchronize(inputs)
    start = clock()
    result = synchronize(run(inputs))
    compile_seconds = clock() - start
    for _ in range(case.sampling.untimed_warmups):
        synchronize(run(inputs))
    samples = []
    for _ in range(case.sampling.warm_repetitions):
        start = clock()
        result = synchronize(run(inputs))
        samples.append(clock() - start)
    memory = peak_memory()
    # Compare valid branch points outside the timed region, including padding masks.
    result = as_batch(result)
    reference = as_batch(synchronize(make_batch(single, "serial")(inputs)))
    matches = []
    for i, mask in enumerate(np.asarray(result.branch.valid)):
        want_p, want_u = (np.asarray(reference.branch.params)[i][mask], np.asarray(reference.branch.states)[i][mask])
        matches.append(bool(np.array_equal(mask, np.asarray(reference.branch.valid)[i])
                            and np.allclose(np.asarray(result.branch.params)[i][mask], want_p,
                                            atol=lane_atol(want_p), rtol=0)
                            and np.allclose(np.asarray(result.branch.states)[i][mask], want_u,
                                            atol=lane_atol(want_u), rtol=0)))
    problem, settings, _ = build(case)
    summary = lane_summary(result, samples)
    if not all(matches):
        # A batched lane that differs from its serial run is a wrong result, not a faster one.
        summary |= {"throughput": None}
    return record | {"status": "MEASURED" if all(matches) else "MISMATCH", "batch_compile_seconds": compile_seconds,
                     "batch_compile_boundary": "first synchronized call, includes execution",
                     "warm_seconds": samples, "warm_median_seconds": statistics.median(samples),
                     "peak_memory": memory, "serial_matches": matches,
                     "lane_seed_parameters": np.asarray(inputs).tolist(),
                     "resolved_controls": controls_record(resolved_controls(case, problem, settings)),
                     "sampling": asdict(case.sampling), "platform": platform.platform(),
                     "python": sys.version.split()[0], "jax": jax.__version__,
                     "dtype": str(inputs.dtype), "device": str(jax.devices()[0]),
                     "timer": "time.perf_counter", "timer_resolution": time.get_clock_info("perf_counter").resolution,
                     **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=[c.benchmark_id for c in load_cases()])
    parser.add_argument("--batch", action="append", type=int, choices=BATCH_SIZES)
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--tool", action="append", choices=("jaxcont", *PENDING_TOOLS))
    parser.add_argument("--output-dir", type=Path, default=RAW)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    case = next(c for c in load_cases() if c.benchmark_id == args.case)
    if args.worker:
        print(json.dumps(measure(case, args.batch[0], args.strategy[0], args.tool[0]), allow_nan=False))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tool in args.tool or ["jaxcont"]:
        for B in args.batch or BATCH_SIZES:
            for strategy in args.strategy or STRATEGIES:
                command = [sys.executable, "-m", "benchmarks.batching", "--worker", "--case", args.case,
                           "--batch", str(B), "--strategy", strategy, "--tool", tool]
                env = _environment()
                env.setdefault("JAX_ENABLE_X64", "1")
                try:
                    worker = subprocess.run(command, env=env, cwd=ROOT, text=True, capture_output=True,
                                            stdin=subprocess.DEVNULL, timeout=case.sampling.timeout_seconds)
                    if worker.returncode != 0:
                        raise RuntimeError(worker.stderr[-2000:])
                    record = json.loads(worker.stdout)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    # A failed row is recorded and the sweep continues.
                    record = {"benchmark_id": args.case, "tool": tool, "strategy": strategy, "B": B,
                              "label": LABEL, "status": "FAILED", "reason": str(exc)}
                name = f"{args.case.replace('/', '__')}__{tool}__B{B}__{strategy.replace('(', '_').replace(')', '')}.json"
                path = args.output_dir / name
                path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                print(f"{record['status']:9s} {args.case} {tool} B={B} {strategy} "
                      f"lanes_ok={record.get('lanes_ok')}/{B} throughput={record.get('throughput')} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
