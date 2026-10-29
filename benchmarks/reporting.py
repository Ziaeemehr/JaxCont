"""Benchmark reports from raw or reviewed records.

  python3 -m benchmarks.reporting --check benchmarks/results/reviewed
  python3 -m benchmarks.reporting --markdown benchmarks/results/raw
  python3 -m benchmarks.reporting --docs        # rewrite docs/source/benchmarks.md

Per workload and tool a report gives n, median, IQR, p95 and a bootstrap 95%
CI of the median for each timing category, the measured accuracy next to the
limits, and the resolved controls. A cross-tool table is drawn only when the
tools' resolved controls are equal and every row meets every accuracy limit;
otherwise the report lists the reasons and shows each tool on its own. There
are no rankings and no speedup ratios. The docs page renders only the records
committed to ``benchmarks/results/reviewed/``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

from benchmarks.accuracy import eligible as _eligible, meets_limit, valid_accuracy
from benchmarks.contracts import ResolvedControls, compare_controls, load_cases
from benchmarks.replicates import record_problems

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "benchmarks" / "results" / "reviewed"
DOCS_PAGE = ROOT / "docs" / "source" / "benchmarks.md"
STATUSES = {"MEASURED", "PENDING", "FAILED", "MISMATCH"}
CATEGORIES = ("startup", "first_call", "warm")
SUMMARY_KEYS = ("n", "median", "iqr", "p95", "samples")


def load(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(Path(directory).glob("*.json"))]


def bootstrap_ci(samples, resamples=2000, level=0.95, seed=0):
    """Percentile bootstrap CI of the median; deterministic for a given seed."""
    draw = random.Random(seed)
    medians = sorted(statistics.median(draw.choices(samples, k=len(samples))) for _ in range(resamples))
    tail = (1 - level) / 2
    return [medians[int(tail * (resamples - 1))], medians[int((1 - tail) * (resamples - 1))]]


def check(records: list[dict]) -> list[str]:
    """Problems that make a record unfit for review; empty means it can be committed."""
    cases = {case.benchmark_id: case for case in load_cases()}
    problems = []
    for record in records:
        name = f"{record.get('benchmark_id')}/{record.get('tool')}"
        status = record.get("status")
        if status not in STATUSES:
            problems.append(f"{name}: unknown status {status!r}")
            continue
        if "speedup" in json.dumps(record).replace("not a single-branch speedup", ""):
            problems.append(f"{name}: speedup wording")
        case = cases.get(record.get("benchmark_id"))
        if case is None:
            problems.append(f"{name}: unknown benchmark_id")
            continue
        if status != "MEASURED":
            if not record.get("reason") and status != "MISMATCH":
                problems.append(f"{name}: {status} without a reason")
            continue
        if "strategy" in record:  # batched throughput rows are reported separately
            if record.get("label") != "throughput; not a single-branch speedup":
                problems.append(f"{name}: batched row without the throughput label")
            continue
        problems.extend(f"{name}: {issue}" for issue in record_problems(record, case))
        accuracy = record.get("accuracy") or {}
        for metric, value in accuracy.items():
            if value is not None and not valid_accuracy(value):
                problems.append(f"{name}: invalid accuracy {metric}={value!r}")
        missing = sorted(set(case.accuracy_limits) - set(accuracy))
        if missing:
            problems.append(f"{name}: accuracy lacks {missing}")
        if record.get("eligible_for_cross_tool") != _eligible(record, case.accuracy_limits):
            problems.append(f"{name}: eligibility flag disagrees with the measured accuracy")
        if not record.get("resolved_controls"):
            problems.append(f"{name}: no resolved controls")
        expected_n = {"startup": case.sampling.first_calls, "first_call": case.sampling.first_calls,
                      "warm": case.sampling.warm_repetitions}
        for category in CATEGORIES:
            summary = record.get(category) or {}
            if any(key not in summary for key in SUMMARY_KEYS):
                problems.append(f"{name}: {category} summary incomplete")
                continue
            samples = summary["samples"]
            if summary["n"] != len(samples) or summary["n"] != expected_n[category]:
                problems.append(f"{name}: {category} has {len(samples)} samples, sampling asks for {expected_n[category]}")
            elif abs(summary["median"] - statistics.median(samples)) > 1e-12:
                problems.append(f"{name}: {category} median does not match its samples")
        points = record.get("points") or {}
        if points.get("first_call") is None or points.get("warm") != [points["first_call"]]:
            problems.append(f"{name}: warm repetitions did not all compute the first call's {points.get('first_call')} points")
        for key in ("timer", "platform", "versions"):
            if not record.get(key):
                problems.append(f"{name}: no {key}")
    return problems


def _controls(record):
    values = dict(record["resolved_controls"])
    values["bounds"] = None if values["bounds"] is None else tuple(values["bounds"])
    return ResolvedControls(**values)


def comparison(records: list[dict]) -> dict:
    """Comparability of one workload's harness rows across tools, with reasons."""
    measured = [r for r in records if r.get("status") == "MEASURED" and "strategy" not in r]
    reasons = [f"{r['tool']}: {r['status']} ({r.get('reason', '')})" for r in records
               if r.get("status") != "MEASURED" and "strategy" not in r]
    if len(measured) < 2:
        reasons.append("fewer than two measured tools")
        return {"comparable": False, "reasons": reasons}
    result = compare_controls({r["tool"]: _controls(r) for r in measured})
    reasons += [f"{m.field}: {m.baseline_tool}={m.baseline_value!r}, {m.tool}={m.value!r}" for m in result.mismatches]
    limits = {case.benchmark_id: case.accuracy_limits for case in load_cases()}
    for record in measured:
        case = next(c for c in load_cases() if c.benchmark_id == record["benchmark_id"])
        reasons += [f"{record['tool']}: {issue}" for issue in record_problems(record, case)]
        accuracy = record.get("accuracy") or {}
        for metric, limit in limits[record["benchmark_id"]].items():
            if accuracy.get(metric) is None:
                reasons.append(f"{record['tool']}: {metric} not measured")
            elif not valid_accuracy(accuracy[metric]):
                reasons.append(f"{record['tool']}: invalid accuracy {metric}={accuracy[metric]!r}")
            elif not meets_limit(accuracy[metric], limit):
                reasons.append(f"{record['tool']}: {metric} = {accuracy[metric]:.3g} exceeds {limit:g}")
    return {"comparable": not reasons, "reasons": reasons}


def _seconds(value):
    return "—" if value is None else f"{value:.4g}"


def _timing_rows(record):
    rows = []
    for category in CATEGORIES:
        summary = record[category]
        low, high = bootstrap_ci(summary["samples"])
        rows.append(f"| {category} | {summary['n']} | {_seconds(summary['median'])} | {_seconds(summary['iqr'])} "
                    f"| {_seconds(summary['p95'])} | [{_seconds(low)}, {_seconds(high)}] |")
    return rows


def _accuracy_rows(record, limits):
    accuracy = record.get("accuracy") or {}
    return [f"| {metric} | {_seconds(accuracy.get(metric)) if accuracy.get(metric) is not None else 'not measured'} "
            f"| {limit:g} |" for metric, limit in limits.items()]


def render_markdown(records: list[dict]) -> str:
    cases = {case.benchmark_id: case for case in load_cases()}
    by_case: dict[str, list[dict]] = {}
    for record in records:
        if "strategy" not in record:
            by_case.setdefault(record["benchmark_id"], []).append(record)
    lines = []
    for benchmark_id in sorted(by_case):
        rows, case = by_case[benchmark_id], cases[benchmark_id]
        verdict = comparison(rows)
        lines += [f"## `{benchmark_id}`", ""]
        if verdict["comparable"]:
            lines += ["Resolved controls are equal and every tool meets every accuracy limit.", "",
                      "| tool | first call median (s) | warm median (s) | warm 95% CI (s) |", "|---|---|---|---|"]
            for record in sorted(rows, key=lambda r: r["tool"]):
                low, high = bootstrap_ci(record["warm"]["samples"])
                lines.append(f"| {record['tool']} | {_seconds(record['first_call']['median'])} "
                             f"| {_seconds(record['warm']['median'])} | [{_seconds(low)}, {_seconds(high)}] |")
            lines.append("")
        else:
            lines += ["No cross-tool table: the rows are not comparable.", ""]
            lines += [f"- {reason}" for reason in verdict["reasons"]] + [""]
        for record in sorted(rows, key=lambda r: r["tool"]):
            lines += [f"### {record['tool']}: {record['status']}", ""]
            if record["status"] != "MEASURED":
                lines += [record.get("reason", ""), ""]
                continue
            versions = ", ".join(f"{k} {v}" for k, v in (record.get("versions") or {}).items())
            lines += [f"{versions}; {record['cpu']['model'] if record.get('cpu') else record['platform']}; "
                      f"timer {record['timer']}.", "",
                      "| category | n | median (s) | IQR (s) | p95 (s) | median 95% CI (s) |",
                      "|---|---|---|---|---|---|", *_timing_rows(record), "",
                      f"Compile overhead estimate (first call minus warm median): "
                      f"{_seconds(record['compile_overhead_estimate']['seconds'])} s.", "",
                      "| accuracy metric | measured | limit |", "|---|---|---|",
                      *_accuracy_rows(record, case.accuracy_limits), ""]
    return "\n".join(lines)


DOCS_HEADER = """# Benchmarks

<!-- Generated by `python3 -m benchmarks.reporting --docs` from
benchmarks/results/reviewed/. Do not edit by hand. -->

These tables show committed, reviewed benchmark records only; a docs build
never runs a timing. Each timing category is kept separate: startup (a fresh
process until the tool is ready), first call (including JAX compilation or
Julia specialization), and warm repetitions in one process. Accuracy is
measured by the harness against a shared reference, not reported by the tool.

A cross-tool table appears only when the tools resolved to equal controls and
every row meets every accuracy limit. Otherwise each tool is shown on its own
with the reasons, and the numbers must not be compared across tools. There
are no rankings. The method is described in `benchmarks/README.md`.

"""


def render_docs(directory: Path = REVIEWED) -> str:
    records = load(directory) if Path(directory).is_dir() else []
    body = render_markdown(records) if records else "No reviewed benchmark records have been committed yet.\n"
    return DOCS_HEADER + body.rstrip() + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", type=Path, metavar="DIR")
    parser.add_argument("--markdown", type=Path, metavar="DIR")
    parser.add_argument("--docs", action="store_true")
    args = parser.parse_args(argv)
    if not (args.check or args.markdown or args.docs):
        parser.error("nothing to do")
    ok = True
    if args.check:
        problems = check(load(args.check))
        for problem in problems:
            print(f"FAIL  {problem}")
        print(f"{'PASS' if not problems else 'FAIL'}  {len(load(args.check))} records in {args.check}")
        ok = not problems
    if args.markdown:
        print(render_markdown(load(args.markdown)))
    if args.docs:
        DOCS_PAGE.write_text(render_docs())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
