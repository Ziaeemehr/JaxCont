"""The shared three-tool gate must fail on broken evidence (review B-01..B-03).

Every tool receives the same committed MatCont bundle (a passing control);
one copy is then damaged. The gate is the one CLI and promotion use.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from examples.validation.artifacts import load_bundle
from examples.validation.registry import load_registry, select_cases
from examples.validation.validation import build_validation_report

REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "validation" / "reference"


def _report(case_id, damage):
    case = deepcopy(select_cases(load_registry(), [case_id])[0])
    data = load_bundle(REFERENCE / case["system"] / "matcont" / case_id)
    tools = {tool: deepcopy(data) for tool in case["tool_coverage"]}
    case["tool_coverage"] = {tool: "PASS" for tool in tools}
    assert build_validation_report(case, tools).status == "PASS"
    damage(tools["jaxcont"])
    return build_validation_report(case, tools)


def _nan_residual_after_a_finite_one(data):
    row = next(r for r in data["branch"][1:] if -1 < float(r["parameter"]) < 1)
    row["residual_norm"] = "nan"


def _large_event_residual(data):
    for row in data["events"]:
        row["residual_norm"] = 100.0


def _missing_state_component(data):
    for table in ("branch", "events"):
        for row in data[table]:
            row.pop("state_3", None)


def _blank_fold_coefficients(data):
    for row in data["events"]:
        row["fold_coefficient"] = ""


def _nan_spectra(data):
    for row in data["spectra"]:
        row["real"] = row["imag"] = "nan"


def _base_orbit_traversed_twice(data):
    for row in data["branch"]:
        if row["segment_id"] == "doubled":
            row["half_period_difference"] = 0.0


def _non_numeric_coefficient(data):
    data["events"][0]["fold_coefficient"] = "not-a-number"


@pytest.mark.parametrize("case_id, damage, status", [
    ("cubic.eq-fold", _nan_residual_after_a_finite_one, "FAIL"),          # B-01
    ("cubic.eq-fold", _large_event_residual, "FAIL"),                     # B-01
    # B-02: a value one tool does not export is missing evidence (PENDING,
    # never PASS); a value that is present but not a number fails.
    ("lorenz84.eq-f", _missing_state_component, "PENDING"),
    ("cubic.eq-fold", _blank_fold_coefficients, "PENDING"),
    ("cubic.eq-fold", _non_numeric_coefficient, "FAIL"),
    ("adaptx.cycle-pd-switch", _nan_spectra, "FAIL"),                     # B-03
    ("adaptx.cycle-pd-switch", _base_orbit_traversed_twice, "FAIL"),      # B-03
], ids=["nan-residual", "event-residual", "missing-state", "blank-coefficient",
        "non-numeric-coefficient", "nan-spectra", "not-doubled"])
def test_damaged_evidence_never_passes_the_shared_gate(case_id, damage, status):
    assert _report(case_id, damage).status == status


def test_cli_and_gallery_apply_the_same_producer_predicates(monkeypatch):
    """Review B-06: matching tables with a failing producer predicate (the
    natural continuation did not stall at the fold) fail in both callers."""
    from examples.validation import run_validation as cli
    from examples.validation.cases.python_cases import CaseResult
    from examples.validation.reporting import make_report

    case = select_cases(load_registry(), ["cubic.eq-fold"])[0]
    data = load_bundle(REFERENCE / "cubic" / "matcont" / "cubic.eq-fold")
    checks = dict(fold_count=2, max_fold_error=0, max_residual=0, natural_stalled_at_fold=False,
                  palc_traversed_both_folds=True, max_fold_coefficient_error=0,
                  stability_transition_count=2)
    result = CaseResult("MC-EQ-001", checks, {}, dict(bundle=data))
    monkeypatch.setattr(cli, "_load_case_callable", lambda _: lambda: result)
    passed, info = cli._validate_case(case, REFERENCE)
    assert not passed
    assert info["validation_report"]["status"] == "FAIL"
    assert make_report(result, REFERENCE).status == "FAIL"


def _synthetic(grid, *, bump=None, eigenvalue=None, value=lambda p: 0.0):
    rows = [dict(segment_id="s", point_id=str(i), parameter=float(p), state_0=float(p),
                 state_1=float(value(p)), residual_norm=0.) for i, p in enumerate(grid)]
    spectra = [dict(segment_id="s", point_id=r["point_id"], event_id="", real=-1., imag=0.) for r in rows]
    if bump is not None:
        rows[bump]["state_1"] = 100.
    if eigenvalue is not None:
        spectra[eigenvalue[0]]["real"] = eigenvalue[1]
    return dict(branch=rows, spectra=spectra, metadata={})


_CASE = dict(id="synthetic", canonical_id="synthetic.eq", comparison_window=[0, 1], expected_events=[],
             tolerances=dict(parameter_atol=1e-4, state_atol=1e-4, spectrum_atol=1e-4, residual_atol=1e-7))


def test_a_disagreement_between_sparse_nodes_is_found():
    """Review B-04: a dense-only node far from the sparse interpolant fails;
    comparing only at the sparse nodes reported 0 and PASS."""
    from examples.validation.promotion_metrics import _pairwise
    sparse, dense = _synthetic([0, 1]), _synthetic([0, .25, .5, .75, 1], bump=2, eigenvalue=(2, -2.))
    for left, right in ((sparse, dense), (dense, sparse)):
        metrics = _pairwise(_CASE, left, right)
        assert metrics["sheet_0.state_1.dense_nodes"]["verdict"] == "FAIL"
        assert metrics["sheet_0.spectrum.dense_nodes"]["verdict"] == "FAIL"


def test_dense_nodes_agreeing_with_a_smooth_sparse_series_pass():
    from examples.validation.promotion_metrics import _pairwise
    smooth = lambda p: np.sin(p)
    sparse = _synthetic(np.linspace(0, 1, 21), value=smooth)
    dense = _synthetic(np.linspace(0, 1, 201), value=smooth)
    metrics = _pairwise(_CASE, sparse, dense)
    assert metrics["sheet_0.state_1.dense_nodes"]["verdict"] == "PASS"
    assert metrics["sheet_0.spectrum.dense_nodes"]["verdict"] == "PASS"


def test_a_sparse_series_too_coarse_for_the_budget_is_unresolved_not_passed():
    """Where only the sparse series' own interpolation error could explain the
    difference, no verdict is claimed."""
    from examples.validation.promotion_metrics import _pairwise
    wiggle = lambda p: 1e-2 * np.sin(12 * p)
    sparse = _synthetic(np.linspace(0, 1, 6), value=wiggle)
    dense = _synthetic(np.linspace(0, 1, 201), value=wiggle)
    assert _pairwise(_CASE, sparse, dense)["sheet_0.state_1.dense_nodes"]["verdict"] == "PENDING"


def test_a_pending_item_delegated_to_a_passing_run_does_not_leave_the_tool_pending():
    """lorenz84.fold-ft: BifurcationKit's BT is an unrefined guess whose
    refined point is compared in lorenz84.bt-ft (PASS); that delegated item
    must not keep the tool's coverage, and with it the case, PENDING."""
    case = select_cases(load_registry(), ["lorenz84.fold-ft"])[0]
    tools = {tool: load_bundle(REFERENCE / "lorenz84" / tool / "lorenz84.fold-ft")
             for tool in ("matcont", "bifurcationkit")}
    tools["jaxcont"] = deepcopy(tools["matcont"])  # a passing stand-in for the fresh run
    report = build_validation_report(case, tools)
    assert report.coverage["bifurcationkit"] == "PASS"
    assert report.status == "PASS"


def test_the_verdict_does_not_depend_on_the_tool_order():
    """Round 2: with equally many in-window nodes the sparse grid was the first
    argument's, and a disagreement visible only at the other tool's nodes
    passed in one order and failed in the other. Both tools' nodes count."""
    case = dict(id="synthetic", canonical_id="synthetic.eq", comparison_window=[.2, .8],
                expected_events=[], tool_coverage={"a": "PASS", "b": "PASS"},
                tolerances=dict(parameter_atol=1e-4, state_atol=1e-4, spectrum_atol=1e-4,
                                residual_atol=1e-7))
    # b's state_1 vanishes at a's nodes (.25, .5, .75) but not at its own (.2, .8).
    a = _synthetic([0, .25, .5, .75, 1])
    b = _synthetic([.1, .2, .5, .8, .9], value=lambda p: 100 * (p - .25) * (p - .5) * (p - .75))
    statuses = {build_validation_report(case, dict(zip("ab", order))).status for order in ((a, b), (b, a))}
    assert statuses == {"FAIL"}
