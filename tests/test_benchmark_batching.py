"""Batched branches preserve lane identity; no timing assertions."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import batching
from benchmarks.adapters.jaxcont import build, synchronize
from benchmarks.contracts import load_cases
from jaxcont.core.scan_continuation import Status


@pytest.fixture(params=["cubic.eq-fold/fixed", "cubic.eq-fold/adaptive",
                        "lorenz84.eq-f/fixed", "lorenz84.eq-f/adaptive"])
def lanes(request):
    case = next(c for c in load_cases() if c.benchmark_id == request.param)
    tol = 1e-9 if jax.config.x64_enabled else 1e-6
    case = replace(case, controls=replace(case.controls, max_steps=60, newton_atol=tol))
    inputs, single = batching.build_lanes(case, 4)
    return case, inputs, single


def _compare(actual, expected):
    """Lanes are measured bitwise identical to serial runs; allow a few ulps only."""
    np.testing.assert_array_equal(actual.branch.valid, expected.branch.valid)
    mask = np.asarray(actual.branch.valid)
    assert mask.any()
    for name in ("params", "states"):
        got, want = np.asarray(getattr(actual.branch, name))[mask], np.asarray(getattr(expected.branch, name))[mask]
        np.testing.assert_allclose(got, want, rtol=0, atol=batching.lane_atol(want))


def test_every_lane_matches_serial_valid_points(lanes):
    case, inputs, single = lanes
    serial = batching.as_batch(synchronize(batching.make_batch(single, "serial")(inputs)))
    batch = synchronize(batching.make_batch(single, "jit(vmap)")(inputs))
    for i in range(4):
        lane = lambda result: jax.tree_util.tree_map(lambda x: x[i], result)
        _compare(lane(batch), lane(serial))
    _, _, solve = build(case)
    _compare(jax.tree_util.tree_map(lambda x: x[0], serial),
             synchronize(jax.jit(solve)()))
    assert len(np.unique(np.asarray(inputs))) == 4
    assert len(np.unique(np.asarray(batch.branch.params)[:, 0])) == 4


def test_batch_permutation_preserves_results(lanes):
    _, inputs, single = lanes
    run = batching.make_batch(single, "jit(vmap)")
    original = synchronize(run(inputs))
    order = jnp.array([2, 0, 3, 1])
    permuted = synchronize(run(inputs[order]))
    for i, index in enumerate(np.asarray(order)):
        _compare(jax.tree_util.tree_map(lambda x: x[i], permuted),
                 jax.tree_util.tree_map(lambda x: x[index], original))


def test_failed_lane_counts_in_denominator():
    result = SimpleNamespace(
        branch=SimpleNamespace(valid=np.array([[True, True, False], [True, False, False],
                                               [True, True, True], [False, False, False]]),
                               states=np.zeros((4, 3, 1)), params=np.zeros((4, 3))),
        stats={"n_valid": np.array([2, 1, 3, 0]),
               "status": np.array([Status.REACHED_END, Status.STALLED, Status.MAX_STEPS, Status.SEED_FAILED])},
    )
    record = batching.lane_summary(result, [2.0, 4.0, 3.0])
    assert record["lane_status"] == ["converged", "failed", "truncated", "failed"]
    assert record["valid_points"] == [2, 1, 3, 0]
    assert record["lanes_total"] == 4 and record["lanes_ok"] == 1
    assert record["success_fraction"] == 0.25
    assert record["throughput"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("tool,case_id", [("matcont", "cubic.eq-fold/fixed"),
                                          ("bifurcationkit", "cubic.eq-fold/fixed"),
                                          ("jaxcont", "radial.cycle-floquet/fixed")])
def test_unimplemented_rows_are_pending(tool, case_id):
    case = next(c for c in load_cases() if c.benchmark_id == case_id)
    record = batching.measure(case, 8, "serial", tool=tool)
    assert record["status"] == "PENDING" and record["reason"]
    assert record["strategy"] == "serial" and record["B"] == 8
    assert record["batch_compile_seconds"] is None and not record["warm_seconds"]
    assert record["lanes_total"] == 8 and record["lanes_ok"] is None
    assert record["peak_memory"]["bytes"] is None and record["peak_memory"]["reason"]


@pytest.mark.parametrize("device_peak,rss,source,expected", [
    (4096, 2, "device-memory-stats", 4096),
    (None, 2, "host-maxrss", 2048),
    (0, 0, "host-maxrss", None),
])
def test_peak_memory_labels_fallback_and_unavailable(monkeypatch, device_peak, rss, source, expected):
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(
        memory_stats=lambda: {"peak_bytes_in_use": device_peak})])
    monkeypatch.setattr(batching.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=rss))
    monkeypatch.setattr(batching.sys, "platform", "linux")
    record = batching.peak_memory()
    assert record["source"] == source and record["bytes"] == expected
    if source == "host-maxrss":
        assert record["reason"]


def test_lane_mismatch_marks_the_row(monkeypatch):
    case = next(c for c in load_cases() if c.benchmark_id == "cubic.eq-fold/fixed")
    case = replace(case, controls=replace(case.controls, max_steps=5,
                                          newton_atol=1e-9 if jax.config.x64_enabled else 1e-6),
                   sampling=replace(case.sampling, warm_repetitions=1, untimed_warmups=0))
    make_batch = batching.make_batch

    def corrupted(single, strategy):
        run = make_batch(single, strategy)
        if strategy == "serial":
            return run
        return lambda inputs: jax.tree_util.tree_map(
            lambda x: x + 1e-3 if jnp.issubdtype(x.dtype, jnp.floating) else x, run(inputs))

    monkeypatch.setattr(batching, "make_batch", corrupted)
    record = batching.measure(case, 2, "jit(vmap)")
    assert record["status"] == "MISMATCH" and record["throughput"] is None
