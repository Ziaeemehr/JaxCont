"""JaxCont benchmark adapter: build, resolve controls, solve, measure accuracy.

Equilibrium workloads (cubic.eq-fold, lorenz84.eq-f) are implemented. The
periodic workloads raise ``NotImplementedError`` and are reported as PENDING
rather than timed with controls that were never resolved.

Run as a worker: ``python -m benchmarks.adapters.jaxcont --case ID --mode
first_call|warm`` prints one JSON line per measurement after a ``READY`` line,
so a supervisor can time startup externally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.accuracy import measure
from benchmarks.contracts import ResolvedControls, StageTiming, controls_record, load_cases



def _system(canonical_id):
    if canonical_id == "cubic.eq-fold":
        from examples.validation.cases.systems import cubic
        return cubic.rhs, jnp.array(cubic.SEED_U), cubic.SEED_P, None
    if canonical_id == "lorenz84.eq-f":
        from examples.validation.cases.systems import lorenz84
        sweep = lorenz84.F_SWEEP
        return lorenz84.lorenz84_rhs, jnp.array(sweep["seed_state"]), sweep["seed_F"], lorenz84.FIXED_ARGS
    raise NotImplementedError(f"{canonical_id}: periodic JaxCont adapter is pending")


def build(case):
    """Problem, settings and the continuation callable for one benchmark row."""
    import jaxcont as jc

    rhs, u0, p0, args = _system(case.canonical_id)
    controls = case.controls
    problem = jc.bif_problem(rhs, u0=u0, p0=p0, args=args)
    settings = jc.ContinuationPar(
        ds=controls.ds, ds_min=controls.ds_min, ds_max=controls.ds_max, max_steps=controls.max_steps,
        adaptive=controls.adaptive, newton_tol=controls.newton_atol,
        newton_max_iter=controls.newton_max_iter, compute_stability=controls.stability_every_step,
    )
    low, high = controls.bounds
    p_end = high if controls.direction > 0 else low
    events = [jc.Fold()] + ([jc.Hopf()] if case.canonical_id == "lorenz84.eq-f" else [])

    def solve():
        return jc.continuation(problem, jc.PseudoArclength(), p_span=(float(p0), p_end),
                               settings=settings, events=events)

    return problem, settings, solve


def resolved_controls(case, problem, settings) -> ResolvedControls:
    """The controls JaxCont actually used, after its own defaults."""
    n_state = int(np.size(problem.u0))
    target = case.controls
    return ResolvedControls(
        newton_atol=float(settings.newton_tol), newton_rtol=0.0,
        newton_max_iter=int(settings.newton_max_iter),
        # Event refinement uses JaxCont's own default tolerance; it has no
        # separate user-facing event atol, so this field is unrepresentable.
        event_atol=None,
        ds=float(settings.ds), ds_min=float(settings.ds_min), ds_max=float(settings.ds_max),
        adaptive=bool(settings.adaptive), max_steps=int(settings.max_steps),
        direction=target.direction, bounds=tuple(target.bounds),
        state_weight=float(settings.theta) if settings.theta is not None else 1.0 / n_state,
        parameter_weight=1.0, stability_every_step=bool(settings.compute_stability),
        ntst=None, ncol=None, newton_criterion="augmented-residual-2norm",
    )


def synchronize(result):
    """Block on every array leaf so a timing includes all device work."""
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


def accuracy(case, result) -> dict:
    """Errors against the shared reference (``benchmarks.accuracy``), never self-reported.

    ``residual_atol`` is the largest equilibrium residual on the fresh branch.
    """
    events, states = {}, {}
    for event in result.events:
        kind = {"fold": "LP", "hopf": "H"}.get(event.kind)
        if kind:
            events.setdefault(kind, []).append(float(event.p))
            states.setdefault(kind, []).append(float(jnp.ravel(event.u)[0]))
    rhs, _, _, args = _system(case.canonical_id)
    branch = result.branch
    residual = max(float(jnp.max(jnp.abs(rhs(u, p, args)))) for u, p in zip(branch.states, branch.params))
    return measure(case, events, residual, states)


def _case(benchmark_id):
    (case,) = [c for c in load_cases() if c.benchmark_id == benchmark_id]
    return case


def worker(benchmark_id, mode, repetitions, warmups):
    import jaxcont as jc

    case = _case(benchmark_id)
    problem, settings, solve = build(case)
    print("READY", flush=True)
    if mode == "first_call":
        start = time.perf_counter()
        result = synchronize(solve())
        timing = StageTiming("first_call", time.perf_counter() - start)
        print(json.dumps({"stage": timing.stage, "seconds": timing.seconds, "points": int(result.branch.n_valid),
                          "controls": controls_record(resolved_controls(case, problem, settings)),
                          "accuracy": accuracy(case, result),
                          "versions": {"jaxcont": jc.__version__, "jax": jax.__version__,
                                       "x64": bool(jax.config.x64_enabled)}}), flush=True)
        return
    for _ in range(warmups):
        synchronize(solve())
    for _ in range(repetitions):
        start = time.perf_counter()
        result = synchronize(solve())
        seconds = time.perf_counter() - start
        # The point count shows every repetition did the same work.
        print(json.dumps({"stage": "warm", "seconds": seconds, "points": int(result.branch.n_valid)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("first_call", "warm"), required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    options = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    worker(options.case, options.mode, options.repetitions, options.warmups)
    sys.exit(0)
