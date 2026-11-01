"""Benchmark harness mechanics with fakes (no real timing assertions)."""

import io
import json
import threading
import types

import jax.numpy as jnp
import numpy as np

import pytest

import benchmarks.run_benchmarks as harness
import benchmarks.adapters.jaxcont as adapter
from benchmarks.adapters.jaxcont import synchronize


class _Leaf:
    def __init__(self):
        self.blocked = False

    def block_until_ready(self):
        self.blocked = True


def test_timer_waits_for_all_result_leaves():
    leaves = {"branch": [_Leaf(), _Leaf()], "events": (_Leaf(),), "meta": "not an array"}
    synchronize(leaves)
    assert all(leaf.blocked for leaf in leaves["branch"] + list(leaves["events"]))


class _FakeProcess:
    launched = 0

    def __init__(self, command, **kwargs):
        type(self).launched += 1
        self.stdout = io.StringIO('READY\n{"seconds": 0.5, "accuracy": {}, "controls": {}}\n')

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _clock(*ticks):
    """Return the given ticks in order, then repeat the last one."""
    values = iter(ticks)
    last = [ticks[-1]]
    return lambda: next(values, last[0])


def test_first_call_uses_fresh_process(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "Popen", _FakeProcess)
    _FakeProcess.launched = 0
    for _ in range(2):
        # launch at 0.0, a queue wait at 0.1, READY seen at 0.4.
        startup, records = harness.run_worker("jaxcont", "x", "first_call", 60, clock=_clock(0.0, 0.1, 0.4, 0.5))
        assert records == [{"seconds": 0.5, "accuracy": {}, "controls": {}}]
    assert _FakeProcess.launched == 2
    # Measured absolute error: 0; launch -> READY uses Python floats.
    assert startup == pytest.approx(0.4, rel=8 * np.finfo(float).eps, abs=0)


def test_silent_hung_worker_times_out(monkeypatch):
    released = threading.Event()

    class Hung(_FakeProcess):
        def __init__(self, command, **kwargs):
            self.stdout = iter(lambda: released.wait() and None, None)

        def kill(self):
            released.set()

    monkeypatch.setattr(harness.subprocess, "Popen", Hung)
    with pytest.raises(TimeoutError):
        harness.run_worker("jaxcont", "x", "first_call", 0.2)
    assert released.is_set()


def test_worker_failure_is_an_error_not_a_timing(monkeypatch):
    class Failing(_FakeProcess):
        def wait(self, timeout=None):
            return 1
    monkeypatch.setattr(harness.subprocess, "Popen", Failing)
    with pytest.raises(RuntimeError, match="failed"):
        harness.run_worker("jaxcont", "x", "first_call", 60)


def test_warm_keeps_callable_identity_and_resets_seed(monkeypatch, capsys):
    """One build per worker; every repetition calls the same callable from the same seed."""
    builds, seeds = [], []

    def fake_build(case):
        problem = types.SimpleNamespace(u0=jnp.array([0.25]))

        def solve():
            seeds.append((id(solve), float(problem.u0[0])))
            return types.SimpleNamespace(branch=types.SimpleNamespace(n_valid=3))

        builds.append(solve)
        return problem, None, solve

    monkeypatch.setattr(adapter, "_case", lambda benchmark_id: None)
    monkeypatch.setattr(adapter, "build", fake_build)
    monkeypatch.setattr(adapter, "accuracy", lambda *args: {})
    monkeypatch.setattr(adapter, "resolved_controls", lambda *args: None)
    monkeypatch.setattr(adapter, "controls_record", lambda *args: {})
    adapter.worker("x", "warm", repetitions=4, warmups=2)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "READY" and len(lines) == 1 + 4
    assert len(builds) == 1
    assert len(seeds) == 6 and len(set(seeds)) == 1
    assert all(json.loads(line)["points"] == 3 for line in lines[1:])


def test_failed_accuracy_row_is_ineligible():
    limits = {"parameter_atol": 5e-4, "residual_atol": 2e-5}
    assert harness.eligible({"accuracy": {"parameter_atol": 1e-6, "residual_atol": 1e-9}}, limits)
    assert not harness.eligible({"accuracy": {"parameter_atol": 1e-3, "residual_atol": 1e-9}}, limits)
    # An unmeasured metric is not a pass.
    assert not harness.eligible({"accuracy": {"parameter_atol": 1e-6, "residual_atol": None}}, limits)


def test_missing_julia_is_pending_not_timed(monkeypatch):
    monkeypatch.setenv("JULIA_BIN", "/nonexistent/julia")
    record = harness.benchmark("cubic.eq-fold/fixed", "bifurcationkit")
    assert record["status"] == "PENDING" and "Julia" in record["reason"] and "first_call" not in record


def test_bifurcationkit_controls_are_converted_to_the_contract_metric():
    """PALC(θ=0.5) halves the squared norm; the worker scales ds by sqrt(0.5), so contract units match."""
    import benchmarks.adapters.bifurcationkit as bk
    from benchmarks.contracts import compare_controls, load_cases
    from benchmarks.adapters.jaxcont import build, resolved_controls
    (case,) = [c for c in load_cases() if c.benchmark_id == "cubic.eq-fold/fixed"]
    problem, settings, _ = build(case)
    result = compare_controls({"jaxcont": resolved_controls(case, problem, settings),
                               "bifurcationkit": bk.resolved_controls(case)})
    # The step metric matches. Neither tool exposes an event atol (both None);
    # what the Newton tolerance bounds differs, so the row stays non-comparable.
    assert [m.field for m in result.mismatches] == ["newton_criterion"] and not result.comparable


def test_missing_matlab_is_pending_not_timed(monkeypatch):
    monkeypatch.setenv("MATLAB_BIN", "/nonexistent/matlab")
    record = harness.benchmark("cubic.eq-fold/fixed", "matcont")
    assert record["status"] == "PENDING" and "MATLAB" in record["reason"] and "first_call" not in record


def test_matcont_worker_rejects_uncovered_workloads():
    import benchmarks.adapters.matcont as matcont
    with pytest.raises(NotImplementedError, match="cubic.eq-fold only"):
        matcont.worker("lorenz84.eq-f/fixed", "first_call", 1, 0)


def test_summary_keeps_every_sample():
    summary = harness.summarize([3.0, 1.0, 2.0, 4.0])
    assert summary["samples"] == [1.0, 2.0, 3.0, 4.0] and summary["median"] == 2.5 and summary["n"] == 4


def test_failed_run_is_recorded_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("jaxcont cubic.eq-fold/fixed first_call exceeded 1 s")
    monkeypatch.setattr(harness, "run_worker", boom)
    record = harness.benchmark("cubic.eq-fold/fixed", "jaxcont")
    assert record["status"] == "FAILED" and "exceeded" in record["reason"] and "warm" not in record


def test_worker_closing_stdout_still_has_a_deadline(monkeypatch):
    import subprocess
    import sys

    real_popen, children = subprocess.Popen, []

    def start(*args, **kwargs):
        process = real_popen([sys.executable, "-c",
                              "import os,time; print('READY',flush=True); os.close(1); time.sleep(1.2)"],
                             **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(harness.subprocess, "Popen", start)
    with pytest.raises(TimeoutError, match="exceeded"):
        harness.run_worker("jaxcont", "x", "first_call", 0.2)
    assert children[0].poll() is not None


@pytest.mark.parametrize("stage,index,change", [
    ("first_call", 0, "points"), ("first_call", 2, "controls"),
    ("first_call", 1, "versions"), ("first_call", 0, "status"),
    ("warm", 0, "points"), ("warm", 1, "controls"),
])
def test_every_timed_replicate_must_do_consistent_work(monkeypatch, stage, index, change):
    record = _replicate_benchmark(monkeypatch, stage, index, change)
    assert record["status"] == "FAILED"
    assert record["reason"]
    assert "first_call" not in record  # No successful timing summary for invalid work.
    assert record["replicates"][stage][index][change] != record["replicates"][stage][-1][change]


@pytest.mark.parametrize("stage", ["first_call", "warm"])
def test_accuracy_uses_worst_replicate_not_last(monkeypatch, stage):
    record = _replicate_benchmark(monkeypatch, stage, 0, "accuracy")
    assert record["status"] == "MEASURED"
    assert not record["eligible_for_cross_tool"]
    assert record["accuracy"]["parameter_atol"] == 100.
    assert len(record["replicates"]["first_call"]) == 5
    assert len(record["replicates"]["warm"]) == 20


def _replicate_benchmark(monkeypatch, stage, index, change):
    from benchmarks.contracts import controls_record, load_cases
    case = next(c for c in load_cases() if c.benchmark_id == "cubic.eq-fold/fixed")
    counters = {"first_call": 0, "warm": 0}

    def row(mode):
        i = counters[mode]
        counters[mode] += 1
        value = dict(seconds=.01, points=120, accuracy={k: 0. for k in case.accuracy_limits},
                     controls=controls_record(case.controls), versions={"tool": "1"}, status="MEASURED")
        if mode == stage and i == index:
            if change == "accuracy":
                value[change]["parameter_atol"] = 100.
            elif change == "controls":
                value[change]["ds"] = 2.
            else:
                value[change] = {"points": 3, "versions": {"tool": "2"}, "status": "FAILED"}[change]
        return value

    def run(tool, benchmark_id, mode, *args):
        return .1, [row(mode) for _ in range(1 if mode == "first_call" else case.sampling.warm_repetitions)]

    monkeypatch.setattr(harness, "run_worker", run)
    return harness.benchmark(case.benchmark_id, "jaxcont")
