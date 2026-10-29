"""Benchmark workload and controls contract (no timing runs)."""

import dataclasses
import json

import pytest

from benchmarks.contracts import (
    BenchmarkCase,
    ResolvedControls,
    StageTiming,
    compare_controls,
    load_cases,
)
from examples.validation.registry import load_registry, select_cases


def _controls(**overrides):
    base = dict(
        newton_atol=1e-9, newton_rtol=0.0, newton_max_iter=25, event_atol=1e-7,
        ds=0.01, ds_min=1e-5, ds_max=0.02, adaptive=True, max_steps=1000,
        direction=1, bounds=(-1.0, 1.0), state_weight=1.0, parameter_weight=1.0,
        stability_every_step=True,
    )
    return ResolvedControls(**{**base, **overrides})


def test_equal_ds_with_different_metric_is_not_comparable():
    # BifurcationKit's theta=0.5 metric weights the state by 0.5/n and the
    # parameter by 0.5: the same numeric ds is a different physical step.
    jaxcont = _controls(state_weight=1.0, parameter_weight=1.0)
    bifurcationkit = _controls(state_weight=0.5, parameter_weight=0.5)
    result = compare_controls({"jaxcont": jaxcont, "bifurcationkit": bifurcationkit})
    assert not result.comparable
    assert {(m.field, m.tool) for m in result.mismatches} == {
        ("state_weight", "bifurcationkit"),
        ("parameter_weight", "bifurcationkit"),
    }


def test_mismatched_resolved_tolerance_disables_comparison():
    result = compare_controls({
        "jaxcont": _controls(),
        "matcont": _controls(newton_atol=1e-6),
        "bifurcationkit": _controls(),
    })
    assert not result.comparable
    (mismatch,) = result.mismatches
    assert (mismatch.field, mismatch.tool, mismatch.value) == ("newton_atol", "matcont", 1e-6)
    assert (mismatch.baseline_tool, mismatch.baseline_value) == ("jaxcont", 1e-9)


def test_normalized_decimals_ignore_representation_but_not_value():
    same = compare_controls({"jaxcont": _controls(ds=0.01, max_steps=1000),
                             "matcont": _controls(ds=1e-2, max_steps=1000.0)})
    assert same.comparable
    assert not compare_controls({"jaxcont": _controls(ds=0.01),
                                 "matcont": _controls(ds=0.0100000001)}).comparable


def test_nonfinite_controls_are_never_equal():
    nan = float("nan")
    assert not compare_controls({"jaxcont": _controls(ds=nan), "matcont": _controls(ds=nan)}).comparable


def test_unresolved_or_unrepresentable_control_is_not_comparable():
    unresolved = compare_controls({"jaxcont": _controls(), "matcont": None})
    assert not unresolved.comparable and unresolved.mismatches[0].tool == "matcont"
    # A tool without a control (None) differs from one that sets it.
    missing = compare_controls({"jaxcont": _controls(event_atol=1e-7), "matcont": _controls(event_atol=None)})
    assert not missing.comparable and missing.mismatches[0].field == "event_atol"


def test_accuracy_limits_are_separate_from_stopping_tolerances():
    controls = _controls(ds=0.01, ds_min=0.01, ds_max=0.01, adaptive=False)
    with pytest.raises(ValueError, match="not controls"):
        BenchmarkCase("x/fixed", "cubic.eq-fold", "fixed", controls,
                      accuracy_limits={"newton_atol": 1e-9}, stages=("warm",))
    with pytest.raises(ValueError, match="required"):
        BenchmarkCase("x/fixed", "cubic.eq-fold", "fixed", controls, accuracy_limits={}, stages=("warm",))
    case = BenchmarkCase("x/fixed", "cubic.eq-fold", "fixed", controls,
                         accuracy_limits={"parameter_atol": 5e-4}, stages=("warm",))
    # Tightening a stopping tolerance leaves the accuracy requirement unchanged.
    tighter = dataclasses.replace(case, controls=dataclasses.replace(controls, newton_atol=1e-12))
    assert tighter.accuracy_limits == case.accuracy_limits


def test_compile_unavailable_is_null_not_zero():
    assert StageTiming("compile", None, reason="MATLAB has no separable compile step").seconds is None
    with pytest.raises(ValueError, match="reason"):
        StageTiming("compile", None)
    with pytest.raises(ValueError, match="positive"):
        StageTiming("compile", 0.0)


def test_fixed_step_row_requires_equal_step_bounds():
    with pytest.raises(ValueError, match="fixed-step"):
        BenchmarkCase("x/fixed", "cubic.eq-fold", "fixed", _controls(),
                      accuracy_limits={"parameter_atol": 5e-4}, stages=("warm",))


def test_manifest_rows_match_the_validation_registry():
    cases = load_cases()
    registry = load_registry()
    workloads = {"cubic.eq-fold", "radial.cycle-floquet", "lorenz84.eq-f",
                 "adaptx.cycle-pd-switch", "torbpc.cycle-lpc-pd-ns"}
    assert {case.canonical_id for case in cases} == workloads
    for canonical_id in workloads:
        modes = sorted(c.step_mode for c in cases if c.canonical_id == canonical_id)
        assert modes == ["adaptive", "fixed"]
    for case in cases:
        (run,) = select_cases(registry, [case.canonical_id])
        # Accuracy limits are the registry's report limits, never looser.
        assert dict(case.accuracy_limits) == run["metric_limits"]
        assert case.controls.state_weight == "1/n_state"


def test_manifest_is_valid_json_with_unique_ids(tmp_path):
    data = json.loads((load_cases.__defaults__[0]).read_text())
    ids = [row["benchmark_id"] for row in data["cases"]]
    assert len(ids) == len(set(ids))
    data["cases"].append(dict(data["cases"][0]))
    duplicate = tmp_path / "cases.json"
    duplicate.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(duplicate)
