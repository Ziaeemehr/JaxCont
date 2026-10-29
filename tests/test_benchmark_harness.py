"""Benchmark harness mechanics with fakes (no real timing assertions)."""

import io

import pytest

import benchmarks.run_benchmarks as harness
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
        self.stderr = io.StringIO("")

    def wait(self):
        return 0


def test_first_call_uses_fresh_process(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "Popen", _FakeProcess)
    _FakeProcess.launched = 0
    # Per run: launch, READY, then one deadline check per output line.
    ticks = iter([0.0, 0.4, 0.5, 0.6] * 2)
    for _ in range(2):
        startup, records = harness.run_worker("jaxcont", "x", "first_call", 60, clock=lambda: next(ticks))
        assert records == [{"seconds": 0.5, "accuracy": {}, "controls": {}}]
    assert _FakeProcess.launched == 2
    assert startup == pytest.approx(0.4)  # launch -> READY, timed outside the worker


def test_worker_failure_is_an_error_not_a_timing(monkeypatch):
    class Failing(_FakeProcess):
        def wait(self):
            return 1
    monkeypatch.setattr(harness.subprocess, "Popen", Failing)
    with pytest.raises(RuntimeError, match="failed"):
        harness.run_worker("jaxcont", "x", "first_call", 60)


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
