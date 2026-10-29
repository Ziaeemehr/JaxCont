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

    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if any(r.get("verdict", "PASS" if r["passed"] else "FAIL") == "FAIL" for r in self.metrics):
            object.__setattr__(self, "status", "FAIL")
        else:
            from .promotion_metrics import covered_elsewhere
            pending = [m for m in self.metrics if m.get("verdict") == "PENDING"]
            open_items = [m for m in pending if not covered_elsewhere(m)]
            coverage = (self.evidence or {}).get("gate", {}).get("coverage", {})
            if self.status == "PASS" and open_items:
                object.__setattr__(self, "status", "PENDING")
            elif (self.status == "PENDING" and pending and not open_items
                  and "PENDING" not in coverage.values()):
                # PENDING only because of items another PASS run compares
                # (never because a tool's bundle is missing).
                object.__setattr__(self, "status", "PASS")

    def record(self):
        record = asdict(self)
        record['tool_versions'] = {'jaxcont_fresh': jc.__version__, 'jax': jax.__version__}
        if self.evidence:
            record['tool_versions'] = self.evidence['tool_versions']
        elif self.reference_hashes:
            record['tool_versions']['matcont_committed'] = '7.6'
        else:
            import scipy
            record['tool_versions']['scipy_shooting_fresh'] = scipy.__version__
        payload = json.dumps(record, sort_keys=True, default=str).encode()
        record["report_hash"] = hashlib.sha256(payload).hexdigest()
        return record


def make_report(result, reference_dir):
    """Retain the authoritative gate; table rows use its existing limits."""
    case = next(c for c in load_registry()["cases"]
                if result.case_id in (c["id"], c.get("canonical_id")))
    if any((Path(reference_dir) / case["system"] / tool / case["canonical_id"] / "metadata.json").is_file()
           for tool in ("matcont", "bifurcationkit")) or "bundle" in result.artifacts:
        return _make_three_tool_report(case, result, reference_dir)
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
        # Same rule as the three-tool path: a declared window clips each range.
        bounds = next((c.get("comparison_window") for c in load_registry()["cases"]
                       if c["id"] == result.case_id), None)
        if isinstance(bounds, list):
            domain = {k: [max(lo, bounds[0]), min(hi, bounds[1])] for k, (lo, hi) in domain.items()}
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


def save_checked_figure(figure, report, path, *, publication=False):
    """A figure and its sidecar carry exactly the same report identity."""
    if publication:
        require_publication(report)
    # The report is authoritative even if a caller supplied a stale banner.
    for artist in figure.texts:
        if "PASS" in artist.get_text() and report.status != "PASS":
            artist.set_text(artist.get_text().replace("PASS", report.status))
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
        color={"FAIL": "darkred", "PASS": "darkgreen"}.get(report.status, "#92400e"),
    )
    bundle = hashlib.sha256(
        json.dumps(report.reference_hashes, sort_keys=True).encode()
    ).hexdigest()
    import textwrap
    domain = textwrap.fill('; '.join(f"{key}: [{lo:.4g}, {hi:.4g}]" for key, (lo, hi) in report.compared_domain.items()), width=145)
    # Reserve a strip below the axes so the footer never overlaps tick labels;
    # a figure with a comparison summary needs room for both texts.
    figure.tight_layout(rect=(0, 0.14 if getattr(figure, "_validation_summary", False) else 0.07, 1, 0.97))
    figure.text(
        0.5,
        0.005,
        f"JaxCont {jc.__version__} fresh | "
        + (
            (" | ".join(f"{tool}: {version} committed" for tool, version in record["tool_versions"].items()
                         if tool not in ("jaxcont_fresh", "jax")) + f" | bundle {bundle[:12]}")
            if report.reference_hashes
            else "independent IVP shooting fresh"
        )
        + (f"\ncompared domain {domain}" if domain else ""),
        ha="center",
        va="bottom",
        fontsize=7,
    )
    footer = figure.texts[-1]
    footer_top = (footer.get_text().count("\n") + 1) * 9 / (figure.get_figheight() * 72)
    for artist in figure.texts:
        if artist.get_text().startswith("Systematic comparison:"):
            artist.set_y(footer_top + .025)
    figure.tight_layout(rect=(0, footer_top + (.10 if getattr(figure, "_validation_summary", False) else .035), 1, .97))
    figure.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Description": json.dumps(record, default=str)},
    )
    path.with_suffix(".checks.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def comparison_inputs(result, reference_dir):
    """Load only reviewed v2 references, alongside the supplied fresh series."""
    from .artifacts import load_bundle, case_result_payload
    case = next(c for c in load_registry()["cases"]
                if result.case_id in (c["id"], c.get("canonical_id")))
    fresh = result.artifacts.get("bundle") or case_result_payload(case, result)
    fresh = {**fresh, "metadata": dict(fresh.get("metadata", {}))}
    fresh["metadata"]["output_hashes"] = {
        name: hashlib.sha256(json.dumps(fresh.get(name, []), sort_keys=True, default=str).encode()).hexdigest()
        for name in ("branch", "events", "spectra")}
    tools = {"jaxcont": fresh}
    for tool in ("matcont", "bifurcationkit"):
        path = Path(reference_dir) / case["system"] / tool / case["canonical_id"]
        if (path / "metadata.json").exists():
            tools[tool] = load_bundle(path, case_id=case["canonical_id"], tool=tool, require_reviewed=True)
    return case, tools


def _make_three_tool_report(case, result, reference_dir):
    from .validation import build_validation_report
    from .promotion_metrics import _pairwise, _sheets, window
    from itertools import combinations
    case, tools = comparison_inputs(result, reference_dir)
    gate = build_validation_report(case, tools)
    rows, samples = [], {}
    for group, metrics in {**gate.pairwise, **gate.independent}.items():
        for name, metric in metrics.items():
            rows.append(dict(metric=f"{group}: {name}", measured=metric["value"],
                             limit=metric["limit"], passed=metric["verdict"] == "PASS",
                             verdict=metric["verdict"], sample_count=len(tools.get(group, {}).get("branch", [])),
                             coverage_fraction=0.0 if metric["verdict"] == "FAIL" and "coverage" in name else 1.0,
                             source_hashes=gate.source_hashes, reason=metric.get("reason")))
    if case['id'].startswith('MC-'):
        from .validation import case_result_passes
        passed = case_result_passes(case['id'], result.checks)
        rows.append(dict(metric='jaxcont: producer predicates', measured=0. if passed else 1.,
                         limit=0., passed=passed, verdict='PASS' if passed else 'FAIL',
                         sample_count=len(tools['jaxcont']['branch']), coverage_fraction=1.,
                         source_hashes=gate.source_hashes))
    coverage = {}
    for tool, data in tools.items():
        fractions = []
        for sheet in _sheets(data, case):
            bounds = window(case, sheet[0].get('segment_id'))
            if not bounds or isinstance(bounds, dict) or bounds[1] == bounds[0]:
                fractions.append(1.)
                continue
            coordinates = [float(r['parameter']) for r in sheet]
            fraction = max(0., min(max(coordinates), bounds[1])-max(min(coordinates), bounds[0])) / (bounds[1]-bounds[0])
            gap = max(min(coordinates)-bounds[0], bounds[1]-max(coordinates), 0.)
            fractions.append(1. if gap <= case["tolerances"]["parameter_atol"] else min(1., fraction))
        coverage[tool] = min(fractions, default=0.)
    for row in rows:
        group = row['metric'].split(': ', 1)[0]
        row['coverage_fraction'] = min(coverage[t] for t in group.split('--'))
    for a,b in combinations(tools, 2):
        sampled = {}
        _pairwise(case, tools[a], tools[b], samples=sampled)
        samples[f"{a}--{b}"] = sampled
    for row in rows:
        group, name = row['metric'].split(': ', 1)
        sampled = samples.get(group, {})
        if name in sampled:
            row['sample_count'] = len(sampled[name]['x'])
        elif group in tools:
            row['sample_count'] = len(tools[group]['events'] if '.count' in name else tools[group]['branch'])
        else:
            row['sample_count'] = min(len(tools[t]['events'] if not name.startswith('sheet_') else tools[t]['branch']) for t in group.split('--'))
    versions = {"jaxcont_fresh": jc.__version__, "jax": jax.__version__}
    domains = {}
    for tool, data in tools.items():
        values = [float(r["parameter"]) for r in data["branch"]]
        bounds = case.get("comparison_window")
        if isinstance(bounds, list):
            domains[tool] = [max(min(values), bounds[0]), min(max(values), bounds[1])]
        else:
            domains[tool] = [min(values), max(values)]
        if tool != "jaxcont":
            versions[tool] = str(next((v for k,v in data["metadata"].get("package_versions", {}).items() if k.lower() == tool), "unknown"))
    return ValidationReport(case["canonical_id"], gate.status, rows, result.checks,
        gate.to_dict(), None, "float64" if jax.config.x64_enabled else "float32",
        gate.source_hashes, domains, {"gate": gate.to_dict(), "samples": samples, "tool_versions": versions})


def require_publication(report):
    """Publication requires every expected tool, event and domain predicate."""
    if report.status != "PASS" or any(not r["passed"] or r.get("coverage_fraction", 1.) < 1. for r in report.metrics):
        raise ValueError("publication requires complete domain, events and passing evidence")


def combine_reports(reports, *, case_id=None):
    """A combined figure and table have one immutable evidence identity."""
    status = next((s for s in ('FAIL', 'PENDING', 'UNSUPPORTED') if any(r.status == s for r in reports)), 'PASS')
    identity = case_id or ' + '.join(r.case_id for r in reports)
    return ValidationReport(identity, status,
        [{**row, 'metric': f'{r.case_id}: {row["metric"]}'} for r in reports for row in r.metrics],
        {}, None, None, reports[0].precision,
        {r.case_id: r.reference_hashes for r in reports},
        {r.case_id.split('.')[-1]: [min(b[0] for b in r.compared_domain.values()), max(b[1] for b in r.compared_domain.values())] for r in reports if not r.case_id.endswith('eq-f')},
        {'reports': [r.record() for r in reports], 'tool_versions': reports[0].record()['tool_versions']})
