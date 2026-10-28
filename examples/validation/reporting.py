"""Report records shared by offline gallery figures and their check sidecars."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import jax
import jaxcont as jc

from .artifacts import reference_directory
from .registry import load_registry
from .validation import evaluate_case_result


@dataclass(frozen=True)
class ValidationReport:
    case_id: str
    status: str
    metrics: list[dict]
    checks: dict
    diagnostics: dict | None
    mismatch: str | None
    precision: str
    reference_hashes: dict
    compared_domain: dict = field(default_factory=dict)

    def record(self):
        record = asdict(self)
        record['tool_versions'] = {'jaxcont_fresh': jc.__version__, 'jax': jax.__version__}
        if self.reference_hashes:
            record['tool_versions']['matcont_committed'] = '7.6'
        else:
            import scipy
            record['tool_versions']['scipy_shooting_fresh'] = scipy.__version__
        payload = json.dumps(record, sort_keys=True, default=str).encode()
        record["report_hash"] = hashlib.sha256(payload).hexdigest()
        return record


def make_report(result, reference_dir):
    """Retain the authoritative gate; table rows use its existing limits."""
    case = next(c for c in load_registry()["cases"] if c["id"] == result.case_id)
    reference_dir = reference_directory(reference_dir, result.case_id)
    validation = evaluate_case_result(case, result, reference_dir)
    c = result.checks
    rows = []

    def row(metric, value, limit, passed=None):
        rows.append(
            dict(
                metric=metric,
                measured=float(value),
                limit=float(limit),
                passed=bool(value < limit if passed is None else passed),
            )
        )

    if result.case_id.startswith("MC-EQ-"):
        key = "max_fold_error" if result.case_id == "MC-EQ-001" else "max_hopf_error"
        row("event location (analytic)", c[key], 5e-4)
        row("equilibrium residual", c["max_residual"], 2e-5)
        coefficient = (
            "max_fold_coefficient_error" if result.case_id == "MC-EQ-001" else "lyapunov_error"
        )
        limit = 5e-4 if result.case_id == "MC-EQ-003" else 1e-4
        row("normal-form coefficient (analytic)", c[coefficient], limit)
        # These differences come from the existing sheet-preserving comparison.
        if validation.diagnostics:
            row(
                "JaxCont–MatCont event parameter",
                validation.diagnostics["event_max_error"],
                case["tolerances"]["parameter_atol"],
            )
            row(
                "JaxCont–MatCont spectrum",
                validation.diagnostics["spectrum_max_error"],
                case["tolerances"]["spectrum_atol"],
            )
    elif result.case_id == "MC-LC-001":
        for key in ("max_radius_error", "max_period_error", "max_multiplier_error"):
            row(key.removeprefix("max_") + " (exact)", c[key], 5e-3)
        row("collocation residual", c["max_collocation_residual"], 2e-5)
        if validation.diagnostics:
            row("JaxCont–MatCont period", validation.diagnostics["branch_max_error"],
                case["tolerances"]["period_atol"])
            row("JaxCont–MatCont Floquet spectrum", validation.diagnostics["spectrum_max_error"],
                case["tolerances"]["multiplier_atol"])
    elif result.case_id == "MC-LC-002":
        for key, limit in [
            ("jaxcont_lpc_parameter_error", 2e-3),
            ("jaxcont_ns_parameter_error", 2e-3),
            ("jaxcont_pd_parameter_error", 2e-3),
            ("jaxcont_max_period_error", 5e-3),
            ("jaxcont_max_extrema_error", 5e-3),
            ("jaxcont_max_multiplier_error", 5e-3),
        ]:
            row(key.removeprefix("jaxcont_"), c[key], limit)
        missing = len(c["jaxcont_missing_event_types"])
        row("missing correctly located event types", missing, 0, missing == 0)
    elif result.case_id == "MC-PRC-001":
        # The original gate is pointwise atol + rtol*abs(reference), not a
        # comparison against the largest tolerance anywhere on the curve.
        from .cases.python_cases.prc import _PRC_ATOL, _PRC_RTOL
        import numpy as np

        a = result.artifacts
        scaled = np.abs(a["jaxcont_prc"] - a["reference_prc"]) / (
            _PRC_ATOL + _PRC_RTOL * np.maximum(np.abs(a["jaxcont_prc"]), np.abs(a["reference_prc"]))
        )
        row("JaxCont–MatCont max scaled iPRC error", np.max(scaled), 1, c["prc_matches_matcont"])
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(reference_dir.glob(result.case_id + "*"))
        if p.is_file()
    }
    if result.case_id == 'MC-PRC-001':
        domain = {'periodic_phase_fraction': [0.0, 1.0]}
    else:
        import csv
        import numpy as np
        parameters = np.asarray(result.artifacts.get('parameters', result.artifacts.get('jaxcont_parameters')))
        with (reference_dir / f'{result.case_id}_branch.csv').open() as handle:
            ref_parameters = [float(r['parameter']) for r in csv.DictReader(handle)]
        domain = {'jaxcont_parameter': [float(parameters.min()), float(parameters.max())],
                  'matcont_parameter': [min(ref_parameters), max(ref_parameters)]}
    return ValidationReport(
        result.case_id,
        "PASS" if validation.passed else "FAIL",
        rows,
        validation.checks,
        validation.diagnostics,
        validation.numerical_mismatch,
        "float64" if jax.config.x64_enabled else "float32",
        hashes,
        domain,
    )


def save_checked_figure(figure, report, path):
    """A figure and its sidecar carry exactly the same report identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = report.record()
    figure.text(
        0.5,
        0.995,
        f"{report.case_id} | {report.status} | {report.precision} | report {record['report_hash'][:12]}",
        ha="center",
        va="top",
        fontsize=8,
        color="darkred" if report.status == "FAIL" else "darkgreen",
    )
    bundle = hashlib.sha256(
        json.dumps(report.reference_hashes, sort_keys=True).encode()
    ).hexdigest()
    limits = ", ".join(f"{r['limit']:.2g}" for r in report.metrics)
    domain = '; '.join(f"{key}: [{lo:.4g}, {hi:.4g}]" for key, (lo, hi) in report.compared_domain.items())
    figure.text(
        0.5,
        -0.035,
        f"JaxCont {jc.__version__} fresh | "
        + (
            f"MatCont 7.6 committed | bundle {bundle[:12]}\n"
            if report.reference_hashes
            else "independent IVP shooting fresh\n"
        )
        + f"Metric limits (table order): {limits}\n{domain}",
        ha="center",
        fontsize=7,
    )
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Description": json.dumps(record, default=str)},
    )
    path.with_suffix(".checks.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
