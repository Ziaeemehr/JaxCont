"""Retained evidence and consistency checks for every timed solve."""
from benchmarks.accuracy import valid_accuracy
from benchmarks.contracts import ResolvedControls, compare_controls


def worst_accuracy(replicates, limits):
    rows = [row for group in replicates.values() for row in group]
    result = {}
    for metric in limits:
        values = [(row.get("accuracy") or {}).get(metric) for row in rows]
        result[metric] = max(values) if values and all(valid_accuracy(v) for v in values) else None
    return result


def problems(replicates, case):
    issues, rows, controls = [], [], {}
    for stage, count in (("first_call", case.sampling.first_calls), ("warm", case.sampling.warm_repetitions)):
        group = replicates.get(stage, [])
        if len(group) != count:
            issues.append(f"{stage}: expected {count} replicates, got {len(group)}")
        for index, row in enumerate(group):
            name = f"{stage}[{index}]"
            rows.append(row)
            if row.get("status") != "MEASURED":
                issues.append(f"{name}: unsuccessful status {row.get('status')!r}")
            if not valid_accuracy(row.get("seconds")) or row["seconds"] <= 0:
                issues.append(f"{name}: invalid timing")
            if not isinstance(row.get("points"), int) or row["points"] <= 0:
                issues.append(f"{name}: missing or invalid point count")
            if not row.get("versions"):
                issues.append(f"{name}: missing versions")
            accuracy = row.get("accuracy") or {}
            if set(case.accuracy_limits) - set(accuracy):
                issues.append(f"{name}: missing accuracy")
            if any(value is not None and not valid_accuracy(value) for value in accuracy.values()):
                issues.append(f"{name}: invalid accuracy")
            try:
                controls[name] = ResolvedControls(**row["controls"])
            except (KeyError, TypeError):
                issues.append(f"{name}: missing or invalid controls")
    if rows:
        for key in ("points", "versions"):
            if any(row.get(key) != rows[0].get(key) for row in rows[1:]):
                issues.append(f"replicates disagree on {key}")
    if len(controls) >= 2 and not compare_controls(controls).comparable:
        issues.append("replicates disagree on resolved controls or have nonfinite controls")
    return issues


def record_problems(record, case):
    """Reported aggregates must describe the retained timed solves."""
    if "replicates" not in record:  # Reviewed legacy records predate this evidence.
        return []
    replicates = record["replicates"]
    issues = problems(replicates, case)
    if record.get("accuracy") != worst_accuracy(replicates, case.accuracy_limits):
        issues.append("accuracy does not match retained replicates' worst error")
    first = replicates.get("first_call", [])
    warm = replicates.get("warm", [])
    if not first or not warm:
        return issues
    if record.get("versions") != first[0].get("versions"):
        issues.append("versions does not match retained replicates")
    try:
        controls = {"reported": ResolvedControls(**record["resolved_controls"]),
                    "retained": ResolvedControls(**first[0]["controls"])}
        if not compare_controls(controls).comparable:
            issues.append("resolved_controls does not match retained replicates")
    except (KeyError, TypeError):
        issues.append("resolved_controls cannot be matched to retained replicates")
    points = {"first_call": first[0].get("points"),
              "warm": sorted({row.get("points") for row in warm}, key=str)}
    if record.get("points") != points:
        issues.append("points does not match retained replicates")
    for stage, rows, key in (("startup", first, "startup_seconds"),
                             ("first_call", first, "seconds"), ("warm", warm, "seconds")):
        values = [row.get(key) for row in rows]
        if (not all(valid_accuracy(v) and v > 0 for v in values)
                or (record.get(stage) or {}).get("samples") != sorted(values)):
            issues.append(f"{stage} samples do not match retained replicates")
    return issues
