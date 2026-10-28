"""Shared case-level gates for CLI and visual MatCont validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import compare_case_result_to_reference
from .compare import ValidationMismatch
from .cases.python_cases import CaseResult

_REFERENCE_COMPARISON_CASES = {
    "MC-EQ-001",
    "MC-EQ-002",
    "MC-EQ-003",
    "MC-LC-001",
}


@dataclass(frozen=True)
class CaseValidation:
    """Outcome from the validation gate shared by every presentation layer."""

    passed: bool
    producer_passed: bool
    checks: dict[str, Any]
    diagnostics: dict[str, float] | None
    numerical_mismatch: str | None
    report: ValidationReport | None = None


def case_result_passes(case_id: str, checks: dict[str, Any]) -> bool:
    """Apply the authoritative producer-specific predicate for one case."""
    predicates = {
        "MC-EQ-001": lambda: (
            checks["fold_count"] == 2
            and checks["max_fold_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["natural_stalled_at_fold"]
            and checks["palc_traversed_both_folds"]
            and checks["max_fold_coefficient_error"] < 1e-4
            and checks["stability_transition_count"] == 2
        ),
        "MC-EQ-002": lambda: (
            checks["hopf_count"] == 1
            and checks["max_hopf_error"] < 5e-4
            and checks["frequency_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["lyapunov_error"] < 1e-4
            and checks["stable_for_negative_parameter"]
            and checks["unstable_for_positive_parameter"]
        ),
        "MC-EQ-003": lambda: (
            checks["hopf_count"] == 1
            and checks["max_hopf_error"] < 5e-4
            and checks["frequency_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["lyapunov_error"] < 5e-4
        ),
        "MC-JAX-001": lambda: (
            checks["all_finite"]
            and checks["max_analytic_gradient_error"] < 2e-3
            and checks["max_finite_difference_error"] < 2e-3
            and checks["jit_matches_eager"]
            and checks["vmap_valid_masks_present"]
            and checks["vmap_valid_masks_match_serial"]
            and checks["vmap_valid_parameters_match_serial"]
            and checks["vmap_valid_states_match_serial"]
            and checks["permutation_invariant"]
        ),
        "MC-C2-001": lambda: (
            checks["all_converged"]
            and checks["max_parameter_error"] < 1e-3
            and checks["bt_bifurcationkit_error"] < 1e-3
            and checks["frequency_error"] < 1e-3
            and checks["gh_lyapunov_error"] < 1e-4
            and checks["parameter_gradients_finite"]
            and checks["max_analytic_gradient_error"] < 2e-3
            and checks["max_finite_difference_error"] < 2e-3
        ),
        "MC-LC-001": lambda: (
            checks["max_radius_error"] < 5e-3
            and checks["max_period_error"] < 5e-3
            and checks["max_collocation_residual"] < 2e-5
            and checks["max_multiplier_error"] < 5e-3
            and checks["all_stable"]
        ),
        "MC-LC-002": lambda: bool(checks["all_comparisons_pass"]),
        "MC-LC-003": lambda: (
            checks["pd_found"]
            and checks["pd_parameter_error"] < 2e-3
            and checks["pd_period_error"] < 5e-3
            and checks["switched_parameter_offset_error"] < 1e-5
            and checks["doubled_period_error"] < 2e-2
            and checks["doubled_period_ratio_error"] < 5e-3
            and checks["doubled_branch_alphas_match"]
            and checks["doubled_branch_period_error"] < 1e-3
            and checks["switch_residual"] < 2e-5
            and checks["genuine_doubled_orbit"]
        ),
        "MC-PRC-001": lambda: checks["prc_matches_matcont"],
        "US-C2-LP-001": lambda: (
            checks["lp_found"]
            and checks["lp_seed_error"] < 1e-3
            and checks["max_curve_error"] < 1e-3
        ),
        "US-C2-H-001": lambda: (
            checks["hopf_found"]
            and checks["hopf_seed_error"] < 1e-3
            and checks["max_curve_error"] < 1e-3
        ),
    }
    return bool(predicates[case_id]())


def evaluate_case_result(
    case: dict[str, Any],
    result: CaseResult,
    reference_dir: Path,
) -> CaseValidation:
    """Combine producer checks with the applicable MatCont comparison.

    Numerical ``ValidationMismatch`` failures are returned as diagnostic state
    so a caller can render them. Structural exceptions retain their native
    types and are expected to be rejected by artifact validation or plotting.
    """
    checks = dict(result.checks)
    diagnostics = None
    numerical_mismatch = None
    if case["id"] in _REFERENCE_COMPARISON_CASES:
        try:
            diagnostics = compare_case_result_to_reference(
                case,
                result,
                reference_dir,
            )
        except ValidationMismatch as exc:
            numerical_mismatch = str(exc)
        checks["matcont_reference"] = (
            diagnostics if diagnostics is not None else {"mismatch": numerical_mismatch}
        )

    producer_passed = case_result_passes(case["id"], checks)
    report = _legacy_report(case, reference_dir, producer_passed, diagnostics, numerical_mismatch)
    checks["validation_report"] = report.to_dict()
    return CaseValidation(
        passed=producer_passed and numerical_mismatch is None,
        producer_passed=producer_passed,
        checks=checks,
        diagnostics=diagnostics,
        numerical_mismatch=numerical_mismatch,
        report=report,
    )


__all__ = ["CaseValidation", "case_result_passes", "evaluate_case_result"]


@dataclass(frozen=True)
class ValidationReport:
    """Serializable evidence shared by the CLI and figure renderers."""

    canonical_id: str
    coverage: dict
    pairwise: dict
    independent: dict
    source_hashes: dict
    status: str

    @property
    def report_hash(self):
        import hashlib
        import json
        from dataclasses import asdict

        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        )

    def to_dict(self):
        from dataclasses import asdict

        return {**asdict(self), "report_hash": self.report_hash}


def coverage_status(coverage, required_tools):
    """Missing evidence is PENDING; an observed failure takes precedence."""
    states = [coverage.get(tool, "PENDING") for tool in required_tools]
    if "FAIL" in states:
        return "FAIL"
    if not states or any(state != "PASS" for state in states):
        return "PENDING"
    return "PASS"


def _metric(value, limit, *, passed=True, samples=0, coverage=1.0, reason=None):
    return {
        "value": value,
        "limit": limit,
        "verdict": "PASS" if passed else "FAIL",
        "sample_count": samples,
        "coverage_fraction": coverage,
        "reason": reason,
    }


def _event_rows(data):
    return [
        {**row, "event_type": row.get("event_type", row.get("kind"))}
        for row in data.get("events", [])
    ]


def _check_payload(case, data):
    """Independent finite/residual/domain/event predicates, before any pair."""
    from collections import Counter
    import numpy as np

    rows = data.get("branch", [])
    if not rows:
        raise ValidationMismatch("empty branch")
    if case.get("spectrum_required", True) and not data.get("spectra"):
        raise ValidationMismatch("required spectra are absent")
    for table in (rows, data.get("events", []), data.get("spectra", [])):
        for row in table:
            for key, value in row.items():
                if key in {
                    "parameter",
                    "period",
                    "residual_norm",
                    "real",
                    "imag",
                    "frequency",
                    "fold_coefficient",
                    "first_lyapunov",
                    "logabs",
                    "phase",
                    "unstable_dimension",
                    "order",
                    "arclength",
                } or key.startswith(("state_", "parameter_")):
                    if not np.isfinite(float(value)):
                        raise ValidationMismatch(f"nonfinite {key}")
            if "status" in row and row["status"] not in {"CONVERGED", "PASS", "converged"}:
                raise ValidationMismatch(f"unresolved point status: {row['status']}")
    parameters = np.array([float(r["parameter"]) for r in rows])
    domain = case.get("expected_domain")
    fraction = 1.0
    if domain:
        low, high = domain
        atol = case["tolerances"].get("parameter_atol", 0.0)
        intervals = []
        for segment in _segments(data):
            coordinates = [float(row["parameter"]) for row in segment]
            if high > low and len(set(coordinates)) < 2:
                raise ValidationMismatch("singleton sheet cannot cover a continuation interval")
            intervals.append((min(coordinates), max(coordinates)))
        end = low
        covered = 0.0
        gap = False
        for start, stop in sorted(intervals):
            start, stop = max(low, start), min(high, stop)
            if stop < low or start > high:
                continue
            if start > end + atol:
                gap = True
            covered += max(0.0, stop - max(start, end))
            end = max(end, stop)
        fraction = (
            min(1.0, covered / (high - low))
            if high > low
            else float(parameters.min() <= low <= parameters.max())
        )
        if gap or parameters.min() > low + atol or end < high - atol:
            raise ValidationMismatch(f"incomplete expected domain {domain}; coverage={fraction}")
    expected = Counter(case.get("expected_events", []))
    present = Counter(r["event_type"] for r in _event_rows(data))
    if expected - present:
        raise ValidationMismatch(f"missing expected events: {dict(expected - present)}")
    residual = max(abs(float(r["residual_norm"])) for r in rows)
    limit = case["tolerances"].get("residual_atol", 1e-7)
    if residual > limit:
        raise ValidationMismatch(f"residual {residual} exceeds {limit}")
    return _metric(residual, limit, samples=len(rows), coverage=fraction)


def _segments(data):
    """Keep explicit sheets in continuation order, independent of parameter sorting."""
    from .artifacts import _monotone_segments

    groups = {}
    for row in data["branch"]:
        groups.setdefault(str(row.get("segment_id", "0")), []).append(row)
    segments = []
    for rows in groups.values():
        segments.extend(_monotone_segments(rows, "parameter"))
    return segments


def _compare_payloads(case, left, right):
    """Reuse tolerance-feasible matching and interpolation on individual sheets."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from .artifacts import _compare_branch_rows, _compare_event_rows, _compare_spectrum_segments
    from .compare import match_spectrum, identify_flow_mode

    a, b = _segments(left), _segments(right)
    if len(a) != len(b):
        raise ValidationMismatch("branch sheet count differs")
    penalty = 1e100
    cost = np.full((len(a), len(b)), penalty)
    errors = {}
    omitted = {"segment_id", "point_id", "order", "arclength", "status"}
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            x = [{k: v for k, v in r.items() if k not in omitted} for r in x]
            y = [{k: v for k, v in r.items() if k not in omitted} for r in y]
            try:
                d = _compare_branch_rows(
                    case,
                    list(x[0]),
                    x,
                    list(y[0]),
                    y,
                    require_same_extent=True,
                    require_same_schema=True,
                )
                cost[i, j] = d["max_error"]
                errors[i, j] = d["max_error"]
            except ValidationMismatch:
                pass
    ia, ib = linear_sum_assignment(cost)
    if any(cost[i, j] == penalty for i, j in zip(ia, ib)):
        raise ValidationMismatch("no tolerance-feasible matching for branch sheets")
    branch_error = max((errors[i, j] for i, j in zip(ia, ib)), default=0.0)
    event_diagnostics, assignments = _compare_event_rows(
        case, _event_rows(left), _event_rows(right)
    )
    spectrum_error = 0.0
    methods = set()
    for i, j in zip(ia, ib):
        curves = []
        kinds = set()
        for data, segment in ((left, a[i]), (right, b[j])):
            curve = []
            for row in segment:
                point = str(row.get("point_id", row.get("point")))
                spectrum = [
                    r
                    for r in data.get("spectra", [])
                    if str(r.get("point_id", r.get("point"))) == point
                    and str(r.get("segment_id", "0")) == str(row.get("segment_id", "0"))
                    and not r.get("event_id")
                ]
                if not spectrum:
                    continue
                kind = spectrum[0].get("kind", spectrum[0].get("spectrum_kind", "EIGENVALUE"))
                kinds.add(kind)
                values = np.array([complex(float(r["real"]), float(r["imag"])) for r in spectrum])
                if kind == "FLOQUET":
                    point_key = f"{row.get('segment_id', '0')}:{point}"
                    flow_data = data.get("flow_modes", {}).get(point_key, {})
                    mode = identify_flow_mode(
                        values, flow_data.get("vectors"), flow_data.get("flow")
                    )
                    methods.add(mode["method"])
                    values = np.delete(values, mode["index"])
                curve.append({"parameter": float(row["parameter"]), "values": values})
            curves.append(curve)
        if len(kinds) > 1 or bool(curves[0]) != bool(curves[1]):
            raise ValidationMismatch("branch spectrum availability or kind differs")
        if curves[0]:
            # Floquet flow mode was removed above, so no second removal here.
            d = _compare_spectrum_segments(
                case, curves[0], curves[1], "parameter", require_same_extent=True, kind="EIGENVALUE"
            )
            spectrum_error = max(spectrum_error, d["max_error"])
    # Event spectra retain one-to-one event identity and mode multiplicity.
    for i, j in assignments:
        event_spectra = []
        kinds = set()
        for data, index in ((left, i), (right, j)):
            event = data["events"][index]
            identity = str(event.get("event_id", event.get("event_index", index)))
            rows = [r for r in data.get("spectra", []) if str(r.get("event_id", "")) == identity]
            event_spectra.append([complex(float(r["real"]), float(r["imag"])) for r in rows])
            kinds.update(r.get("kind", "EIGENVALUE") for r in rows)
        if bool(event_spectra[0]) != bool(event_spectra[1]) or len(kinds) > 1:
            raise ValidationMismatch("event spectrum availability or kind differs")
        if event_spectra[0]:
            d = match_spectrum(
                *event_spectra,
                atol=case["tolerances"].get(
                    "spectrum_atol", case["tolerances"].get("multiplier_atol", 1e-9)
                ),
                rtol=case["tolerances"].get("rtol", 0),
                exclude_trivial="FLOQUET" in kinds,
            )
            spectrum_error = max(spectrum_error, d["max_error"])
            if "FLOQUET" in kinds:
                methods.add("nearest-to-+1")
    return {
        "branch": _metric(branch_error, case["tolerances"], samples=len(left["branch"])),
        "events": _metric(
            event_diagnostics["max_error"], case["tolerances"], samples=len(left.get("events", []))
        ),
        "spectra": {
            **_metric(spectrum_error, case["tolerances"], samples=len(left.get("spectra", []))),
            "trivial_mode_methods": sorted(methods),
        },
    }


def build_validation_report(case, tool_results):
    """Evaluate all present tools and all unordered pairs; never hide failures."""
    from itertools import combinations

    independent, pairwise, hashes = {}, {}, {}
    coverage = {tool: "PENDING" for tool in case.get("tool_coverage", {})}
    for tool, data in tool_results.items():
        hashes[tool] = data.get("source_hashes", data.get("metadata", {}).get("output_hashes", {}))
        try:
            independent[tool] = _check_payload(case, data)
            declared = case.get("tool_coverage", {}).get(tool, "PASS")
            coverage[tool] = declared if declared != "PASS" else "PASS"
        except (ValueError, KeyError, ValidationMismatch) as exc:
            coverage[tool] = "FAIL"
            independent[tool] = _metric(
                None, case["tolerances"], passed=False, reason=str(exc), coverage=0.0
            )
    for a, b in combinations(tool_results, 2):
        key = f"{a}--{b}"
        try:
            pairwise[key] = _compare_payloads(case, tool_results[a], tool_results[b])
        except (ValueError, KeyError, ValidationMismatch) as exc:
            pairwise[key] = {
                "comparison": _metric(
                    None, case["tolerances"], passed=False, reason=str(exc), coverage=0.0
                )
            }
    failed = any(
        metric["verdict"] == "FAIL" for pair in pairwise.values() for metric in pair.values()
    )
    status = "FAIL" if failed else coverage_status(coverage, coverage.keys())
    return ValidationReport(
        case.get("canonical_id", case["id"]), coverage, pairwise, independent, hashes, status
    )


def _legacy_report(case, reference_dir, producer_passed, diagnostics, mismatch):
    """Expose historical gates through the report without inventing evidence.

    Legacy producer verdicts remain their established compatibility interface.
    The report separately exposes missing tools and historical limitations.
    """
    from .artifacts import reference_directory, file_hash

    directory = reference_directory(reference_dir, case["id"])
    hashes = {
        p.name: file_hash(p)
        for name in case.get("references", [])
        if (p := directory / name).is_file()
    }
    coverage = {tool: "PENDING" for tool in case.get("tool_coverage", {})}
    coverage["jaxcont"] = "PASS" if producer_passed else "FAIL"
    pairwise = {}
    if diagnostics is not None or mismatch is not None:
        coverage["matcont"] = "FAIL" if mismatch else "PASS"
        pairwise["jaxcont--matcont"] = {
            name: _metric(value, case["tolerances"], passed=mismatch is None, reason=mismatch)
            for name, value in (diagnostics or {"comparison": None}).items()
        }
    independent = {
        "jaxcont": _metric(
            int(producer_passed),
            {"equals": 1},
            passed=producer_passed,
            reason="legacy case-specific predicates",
        )
    }
    return ValidationReport(
        case.get("canonical_id", case["id"]),
        coverage,
        pairwise,
        independent,
        {"matcont": hashes},
        coverage_status(coverage, coverage.keys()),
    )
