"""Benchmark harness mechanics with fakes (no real timing assertions)."""

import io
import threading
import types

import jax.numpy as jnp

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

    def wait(self):
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
    assert startup == pytest.approx(0.4)  # launch -> READY, timed outside the worker


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
        def wait(self):
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
            return {"branch": jnp.zeros(3)}

        builds.append(solve)
        return problem, None, solve

    monkeypatch.setattr(adapter, "_case", lambda benchmark_id: None)
    monkeypatch.setattr(adapter, "build", fake_build)
    adapter.worker("x", "warm", repetitions=4, warmups=2)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "READY" and len(lines) == 1 + 4
    assert len(builds) == 1
    assert len(seeds) == 6 and len(set(seeds)) == 1


def test_failed_accuracy_row_is_ineligible():
    limits = {"parameter_atol": 5e-4, "residual_atol": 2e-5}
    assert harness.eligible({"accuracy": {"parameter_atol": 1e-6, "residual_atol": 1e-9}}, limits)
    assert not harness.eligible({"accuracy": {"parameter_atol": 1e-3, "residual_atol": 1e-9}}, limits)
    # An unmeasured metric is not a pass.
    assert not harness.eligible({"accuracy": {"parameter_atol": 1e-6, "residual_atol": None}}, limits)


def test_external_tools_without_workers_are_pending_not_timed():
    record = harness.benchmark("cubic.eq-fold/fixed", "matcont")
    assert record["status"] == "PENDING" and "first_call" not in record


def test_summary_keeps_every_sample():
    summary = harness.summarize([3.0, 1.0, 2.0, 4.0])
    assert summary["samples"] == [1.0, 2.0, 3.0, 4.0] and summary["median"] == 2.5 and summary["n"] == 4


def test_failed_run_is_recorded_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("jaxcont cubic.eq-fold/fixed first_call exceeded 1 s")
    monkeypatch.setattr(harness, "run_worker", boom)
    record = harness.benchmark("cubic.eq-fold/fixed", "jaxcont")
    assert record["status"] == "FAILED" and "exceeded" in record["reason"] and "warm" not in record
