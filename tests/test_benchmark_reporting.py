"""Benchmark reports: metadata, comparability gating and record checks (no timing assertions)."""

import copy

import pytest

from benchmarks import reporting
from benchmarks.contracts import controls_record, load_cases

CASE = next(c for c in load_cases() if c.benchmark_id == "cubic.eq-fold/fixed")


def _summary(samples):
    import statistics
    ordered = sorted(samples)
    q = statistics.quantiles(ordered, n=4)
    return {"n": len(ordered), "median": statistics.median(ordered), "iqr": q[2] - q[0],
            "p95": ordered[-1], "samples": ordered}


def _record(tool, **controls):
    accuracy = {metric: limit / 10 for metric, limit in CASE.accuracy_limits.items()}
    return {
        "benchmark_id": CASE.benchmark_id, "canonical_id": CASE.canonical_id, "tool": tool,
        "status": "MEASURED", "platform": "Linux", "cpu": {"model": "test cpu", "logical_cores": 1},
        "timer": "time.perf_counter", "versions": {tool: "1.0"},
        "resolved_controls": controls_record(CASE.controls.__class__(**{**CASE.controls.__dict__, **controls})),
        "accuracy": accuracy, "eligible_for_cross_tool": True,
        "startup": _summary([1.0, 1.1, 1.2, 1.3, 1.4]), "first_call": _summary([2.0, 2.1, 2.2, 2.3, 2.4]),
        "warm": _summary([0.2 + 0.001 * i for i in range(20)]),
        "compile_overhead_estimate": {"seconds": 1.9, "estimate": True},
        "points": {"first_call": 120, "warm": [120]},
    }


def test_report_includes_accuracy_and_sampling_metadata():
    text = reporting.render_markdown([_record("jaxcont"), _record("matcont")])
    for metric, limit in CASE.accuracy_limits.items():
        assert f"| {metric} | {limit / 10:.4g} | {limit:g} |" in text
    assert "| warm | 20 |" in text and "| first_call | 5 |" in text and "median 95% CI" in text
    assert "timer time.perf_counter" in text and "test cpu" in text
    assert "Resolved controls are equal" in text


def test_noncomparable_rows_have_no_cross_tool_table():
    records = [_record("jaxcont"), _record("matcont", newton_criterion="residual-2norm-and-step-2norm", bounds=None)]
    text = reporting.render_markdown(records)
    assert "No cross-tool table" in text and "| tool | first call median" not in text
    assert "newton_criterion: jaxcont='augmented-residual-2norm', matcont='residual-2norm-and-step-2norm'" in text
    assert "bounds:" in text


def test_ineligible_accuracy_blocks_the_cross_tool_table():
    failing = _record("matcont")
    failing["accuracy"]["parameter_atol"] = 1.0
    failing["eligible_for_cross_tool"] = False
    verdict = reporting.comparison([_record("jaxcont"), failing])
    assert not verdict["comparable"] and "matcont: parameter_atol = 1 exceeds 0.0005" in verdict["reasons"]
    unmeasured = _record("matcont")
    unmeasured["accuracy"]["state_atol"] = None
    assert "matcont: state_atol not measured" in reporting.comparison([_record("jaxcont"), unmeasured])["reasons"]


def test_check_accepts_a_complete_record_and_flags_tampering():
    good = _record("jaxcont")
    assert reporting.check([good]) == []
    bad_median = copy.deepcopy(good)
    bad_median["warm"]["median"] = 0.1
    short = copy.deepcopy(good)
    short["warm"] = _summary([0.2] * 3)
    lying = copy.deepcopy(good)
    lying["accuracy"]["residual_atol"] = None
    pending = {"benchmark_id": CASE.benchmark_id, "tool": "matcont", "status": "PENDING"}
    boast = copy.deepcopy(good)
    boast["note"] = "10x speedup"
    batched = {"benchmark_id": CASE.benchmark_id, "tool": "jaxcont", "status": "MEASURED",
               "strategy": "jit(vmap)", "B": 8, "label": "batched"}
    different_work = copy.deepcopy(good)
    different_work["points"] = {"first_call": 120, "warm": [3, 120]}
    problems = reporting.check([bad_median, short, lying, pending, boast, batched, different_work])
    assert any("did not all compute" in p for p in problems)
    assert any("throughput label" in p for p in problems)
    assert any("median does not match" in p for p in problems)
    assert any("3 samples, sampling asks for 20" in p for p in problems)
    assert any("eligibility flag disagrees" in p for p in problems)
    assert any("PENDING without a reason" in p for p in problems)
    assert any("speedup wording" in p for p in problems)


def test_bootstrap_ci_brackets_the_median():
    samples = [0.2 + 0.001 * i for i in range(20)]
    low, high = reporting.bootstrap_ci(samples)
    assert low <= sorted(samples)[10] <= high and low < high


def test_docs_page_matches_the_reviewed_records():
    """docs/source/benchmarks.md is generated; regenerate with `python3 -m benchmarks.reporting --docs`."""
    assert reporting.DOCS_PAGE.read_text() == reporting.render_docs()
    assert not list(reporting.REVIEWED.glob("*.json")) or reporting.check(reporting.load(reporting.REVIEWED)) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -1.0])
def test_invalid_accuracy_is_rejected_by_check_and_table(value):
    from benchmarks.run_benchmarks import eligible

    bad = _record("matcont")
    bad["accuracy"]["parameter_atol"] = value
    bad["eligible_for_cross_tool"] = False
    assert not eligible(bad, CASE.accuracy_limits)
    assert any("invalid accuracy" in p for p in reporting.check([bad]))
    assert not reporting.comparison([_record("jaxcont"), bad])["comparable"]
    text = reporting.render_markdown([_record("jaxcont"), bad])
    assert "No cross-tool table" in text
    assert "| tool | first call median" not in text


@pytest.mark.parametrize("field", ["resolved_controls", "points", "versions", "first_call", "warm", "startup"])
def test_reported_aggregates_must_match_retained_replicates(field):
    record = _record("matcont")
    record["replicates"] = {}
    for stage in ("first_call", "warm"):
        record["replicates"][stage] = [
            dict(seconds=seconds, status="MEASURED", points=120,
                 accuracy=copy.deepcopy(record["accuracy"]), versions=record["versions"].copy(),
                 controls=copy.deepcopy(record["resolved_controls"]), startup_seconds=1.0 + .1 * i)
            for i, seconds in enumerate(record[stage]["samples"])]
    assert reporting.check([record]) == []
    if field == "resolved_controls":
        record[field]["ds"] *= 2
    elif field == "points":
        record[field] = {"first_call": 3, "warm": [3]}
    elif field == "versions":
        record[field] = {"matcont": "2"}
    else:
        record[field] = _summary([v * 2 for v in record[field]["samples"]])
    assert any("retained replicates" in p for p in reporting.check([record]))
    assert not reporting.comparison([_record("jaxcont"), record])["comparable"]
