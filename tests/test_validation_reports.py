"""One report governs coverage, all pairs, independent checks, and verdicts."""

from copy import deepcopy
import itertools

import numpy as np
import pytest


def payload(offset=0):
    return {
        "branch": [
            {
                "segment_id": "a",
                "point_id": "0",
                "order": 0,
                "parameter": 0.0,
                "state_0": offset,
                "residual_norm": 0.0,
                "unstable_dimension": 0,
                "status": "CONVERGED",
            },
            {
                "segment_id": "a",
                "point_id": "1",
                "order": 1,
                "parameter": 1.0,
                "state_0": 1 + offset,
                "residual_norm": 0.0,
                "unstable_dimension": 0,
                "status": "CONVERGED",
            },
        ],
        "events": [{"kind": "LP", "parameter": 0.5}],
        "spectra": [
            {
                "point_id": "0",
                "segment_id": "a",
                "event_id": "",
                "kind": "EIGENVALUE",
                "real": -1.0,
                "imag": 0.0,
            }
        ],
        "source_hashes": {"branch": "sha256:" + "a" * 64},
    }


def contract():
    return {
        "id": "synthetic",
        "canonical_id": "synthetic.eq",
        "expected_domain": [0.0, 1.0],
        "expected_events": ["LP"],
        "tool_coverage": {"jaxcont": "PASS", "matcont": "PASS", "bifurcationkit": "PASS"},
        "tolerances": {
            "parameter_atol": 1e-5,
            "state_atol": 1e-5,
            "residual_atol": 1e-7,
            "spectrum_atol": 1e-5,
        },
    }


def test_three_pairwise_checks_required_when_three_tools_present():
    from examples.validation.validation import build_validation_report

    report = build_validation_report(
        contract(), {t: payload() for t in contract()["tool_coverage"]}
    )
    assert report.status == "PASS"
    assert set(report.pairwise) == {
        "jaxcont--matcont",
        "jaxcont--bifurcationkit",
        "matcont--bifurcationkit",
    }
    assert report.independent
    assert report.source_hashes
    assert report.to_dict()["report_hash"] == report.report_hash
    for checks in report.pairwise.values():
        assert {"value", "limit", "verdict", "sample_count", "coverage_fraction"} <= checks[
            "branch"
        ].keys()


def test_partial_overlap_cannot_pass_full_domain():
    from examples.validation.validation import build_validation_report

    data = {"jaxcont": payload(), "matcont": payload()}
    data["matcont"]["branch"][0]["parameter"] = 0.2
    assert build_validation_report(contract(), data).status == "FAIL"


def test_fold_sheets_are_not_merged():
    from examples.validation.validation import build_validation_report

    a = payload()
    a["branch"] += [{**r, "segment_id": "b", "state_0": 10 + r["state_0"]} for r in a["branch"]]
    b = deepcopy(a)
    for r in b["branch"]:
        if r["segment_id"] == "b":
            r["state_0"] = r["parameter"]
    case = contract()
    case["tool_coverage"] = {"jaxcont": "PASS", "matcont": "PASS"}
    assert build_validation_report(case, {"jaxcont": a, "matcont": a}).status == "PASS"
    assert build_validation_report(case, {"jaxcont": a, "matcont": b}).status == "FAIL"


@pytest.mark.parametrize("mutation", ["event", "nan", "residual", "spectrum"])
def test_missing_event_or_nonfinite_value_fails(mutation):
    from examples.validation.validation import build_validation_report

    a = payload()
    if mutation == "event":
        a["events"] = []
    if mutation == "nan":
        a["branch"][0]["state_0"] = np.nan
    if mutation == "residual":
        a["branch"][0]["residual_norm"] = 1e-3
    if mutation == "spectrum":
        a["spectra"][0]["real"] = np.inf
    assert (
        build_validation_report(contract(), {"jaxcont": payload(), "matcont": a}).status == "FAIL"
    )


def test_flow_mode_uses_alignment_and_records_legacy_fallback():
    from examples.validation.compare import identify_flow_mode, match_spectrum

    modes = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert identify_flow_mode([1.00001, 0.99], modes, [1.0, 0.0])["index"] == 1
    assert identify_flow_mode([1.00001, 0.99])["method"] == "nearest-to-+1"
    result = match_spectrum(
        [1.00001, 0.99],
        [1.00001, 0.99],
        atol=1e-8,
        actual_modes=modes,
        reference_modes=modes,
        actual_flow=[1.0, 0.0],
        reference_flow=[1.0, 0.0],
    )
    assert result["excluded"]["actual"] == 1
    assert result["trivial_mode_method"]["actual"] == "flow-alignment"


@pytest.mark.slow
@pytest.mark.parametrize(
    "case_id", ["MC-EQ-001", "MC-EQ-002", "MC-EQ-003", "MC-LC-001", "MC-LC-002", "MC-PRC-001"]
)
def test_offline_suite_verdicts_match_registry(case_id):
    from examples.validation.registry import load_registry
    from examples.validation.run_validation import _validate_case, _DEFAULT_REFERENCE_DIR

    case = next(c for c in load_registry()["cases"] if c["id"] == case_id)
    passed, checks = _validate_case(case, _DEFAULT_REFERENCE_DIR)
    assert passed == (case["evidence_status"] == "PASS")
    if case_id == "MC-LC-002":
        assert checks["validation_report"]["status"] == "FAIL"


def test_v2_cli_uses_report_and_exits_nonzero_on_fail(tmp_path, monkeypatch, capsys):
    from examples.validation import run_validation as cli
    from tests.test_validation_schema import make_bundle

    case = contract()
    case.update(
        title="synthetic test",
        system="synthetic",
        support="supported",
        python="fake:produce",
        references=[],
    )
    case["tool_coverage"] = {"jaxcont": "PASS", "matcont": "PASS"}
    directory = make_bundle(tmp_path / "synthetic" / "matcont" / "synthetic.eq")
    # Match the fixture's domain and fold location.
    fresh = payload(1.0)
    fresh["events"][0]["parameter"] = 0.0
    monkeypatch.setattr(cli, "load_registry", lambda: {"cases": [case]})
    monkeypatch.setattr(cli, "_load_case_callable", lambda spec: lambda: fresh)
    assert cli.main(["--case", "synthetic", "--reference-dir", str(tmp_path)]) == 0
    assert "PASS synthetic" in capsys.readouterr().out
    fresh["branch"][0]["state_0"] += 0.1
    assert cli.main(["--case", "synthetic", "--reference-dir", str(tmp_path)]) == 1
    assert "FAIL synthetic" in capsys.readouterr().err


@pytest.mark.parametrize(
    "table,column",
    [
        ("events", "fold_coefficient"),
        ("events", "first_lyapunov"),
        ("spectra", "logabs"),
        ("spectra", "phase"),
        ("branch", "unstable_dimension"),
    ],
)
def test_nonfinite_diagnostics_cannot_pass_even_when_both_tools_agree(table, column):
    from examples.validation.validation import build_validation_report

    data = payload()
    data[table][0][column] = np.nan
    assert build_validation_report(contract(), {"jaxcont": data, "matcont": data}).status == "FAIL"


def test_disconnected_points_cannot_claim_full_domain():
    from examples.validation.validation import build_validation_report

    data = payload()
    data["branch"][0]["segment_id"] = "s0"
    data["branch"][1]["segment_id"] = "s1"
    assert build_validation_report(contract(), {"jaxcont": data, "matcont": data}).status == "FAIL"


def test_cli_rejects_tool_identity_substitution(tmp_path, monkeypatch, capsys):
    from examples.validation import run_validation as cli
    from tests.test_validation_schema import make_bundle

    case = contract()
    case.update(
        title="synthetic",
        system="synthetic",
        support="supported",
        python="fake:produce",
        references=[],
    )
    make_bundle(tmp_path / "synthetic" / "matcont" / "synthetic.eq")
    make_bundle(tmp_path / "synthetic" / "bifurcationkit" / "synthetic.eq", tool="matcont")
    fresh = payload(1.0)
    fresh["events"][0]["parameter"] = 0.0
    monkeypatch.setattr(cli, "load_registry", lambda: {"cases": [case]})
    monkeypatch.setattr(cli, "_load_case_callable", lambda spec: lambda: fresh)
    assert cli.main(["--case", "synthetic", "--reference-dir", str(tmp_path)]) == 1
    assert "tool" in capsys.readouterr().err


def test_absent_required_spectra_cannot_pass():
    from examples.validation.validation import build_validation_report

    data = payload()
    data["spectra"] = []
    assert build_validation_report(contract(), {"jaxcont": data, "matcont": data}).status == "FAIL"
