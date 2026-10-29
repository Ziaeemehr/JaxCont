"""Invalid raw results must never become successful accuracy evidence."""
import io
import json
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from benchmarks.accuracy import measure
from benchmarks.contracts import load_cases

CASE = next(c for c in load_cases() if c.benchmark_id == "cubic.eq-fold/fixed")


@pytest.mark.parametrize("field", ["events", "states"])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_event_samples_are_unavailable(field, index, invalid):
    events, states = [-2 / 3, 2 / 3], [1., -1.]
    (events if field == "events" else states)[index] = invalid
    result = measure(CASE, {"LP": events}, 0., {"LP": states})
    assert all(result[key] is None for key in CASE.accuracy_limits)


@pytest.mark.parametrize("states", [[], [1.], [1., -1., 0.]])
def test_event_state_cardinality_must_match(states):
    result = measure(CASE, {"LP": [-2 / 3, 2 / 3]}, 0., {"LP": states})
    assert all(result[key] is None for key in CASE.accuracy_limits)


@pytest.mark.parametrize("states,params", [([0., float("nan")], [0., 0.]),
                                          ([float("nan"), 0.], [0., 0.]),
                                          ([0., 0.], [0.]), ([], [])])
@pytest.mark.parametrize("tool", ["jaxcont", "matcont", "bifurcationkit"])
def test_invalid_branch_samples_are_unavailable(monkeypatch, states, params, tool, capsys):
    if tool == "jaxcont":
        from benchmarks.adapters import jaxcont as adapter
        events = [SimpleNamespace(kind="fold", p=p, u=jnp.array([u]))
                  for p, u in [(-2 / 3, 1.), (2 / 3, -1.)]]
        result = SimpleNamespace(events=events, branch=SimpleNamespace(
            states=jnp.asarray(states).reshape(-1, 1), params=jnp.asarray(params)))
        assert adapter.accuracy(CASE, result)["residual_atol"] is None
        return
    from benchmarks.adapters import matcont, bifurcationkit
    adapter = matcont if tool == "matcont" else bifurcationkit
    row = dict(states=states, params=params, folds=[-2 / 3, 2 / 3], fold_states=[1., -1.])
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(row) + "\n"), wait=lambda: 0)
    monkeypatch.setattr(adapter.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(adapter, "_matlab" if tool == "matcont" else "_julia", lambda: "unused")
    adapter.worker(CASE.benchmark_id, "first_call", 1, 0)
    assert json.loads(capsys.readouterr().out)["accuracy"]["residual_atol"] is None
