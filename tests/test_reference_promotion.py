"""Promotion must preserve evidence and reject unsafe or failing copies."""
import json

import pytest

from tests.test_validation_schema import make_bundle

import numpy as np


def candidates(tmp_path):
    paths = {tool: make_bundle(tmp_path / tool, tool=tool)
             for tool in ("jaxcont", "matcont", "bifurcationkit")}
    for path in paths.values():
        metadata = json.loads((path / "metadata.json").read_text())
        metadata["generator_revision"] = "a" * 40
        with (path / "spectra.csv").open("a") as stream:
            stream.write("s,0,e,EIGENVALUE,0,0,0,-700,0,not-applicable\n")
            stream.write("s,1,,EIGENVALUE,0,-1,0,0,3.141592653589793,not-applicable\n")
        from examples.validation.artifacts import file_hash
        metadata["output_hashes"]["spectra.csv"] = file_hash(path / "spectra.csv")
        (path / "metadata.json").write_text(json.dumps(metadata))
        (path / "stdout.log").write_text("Native producer completed\n")
    return paths


def case():
    return dict(id="synthetic.eq", canonical_id="synthetic.eq", system="synthetic",
                evidence_status="PASS", expected_domain=[0, 1], expected_events=["LP"],
                tool_coverage={t: "PASS" for t in ("jaxcont", "matcont", "bifurcationkit")},
                tolerances={"parameter_atol": 1e-4, "state_atol": 1e-4,
                            "spectrum_atol": 1e-4, "residual_atol": 1e-7})


def manifest(tmp_path, paths, selected=None):
    from scripts.promote_reference import write_manifest
    return write_manifest(selected or case(), paths, tmp_path / "manifest.json",
                          reviewer="Reviewer", review_date="2026-10-29")


def test_manifest_measures_every_pair_and_promotes_immutable_copy(tmp_path):
    from scripts.promote_reference import promote
    from examples.validation.artifacts import load_bundle
    paths = candidates(tmp_path)
    report = manifest(tmp_path, paths)
    assert report["verdict"] == "PASS"
    assert len(report["pairwise"]) == 3
    assert all(m["value"] == 0 and m["limit"] >= 0 and m["verdict"] == "PASS"
               for pair in report["pairwise"].values() for m in pair.values())
    dest = promote(report, "matcont", tmp_path / "reference")
    assert load_bundle(dest, require_reviewed=True)["metadata"]["reviewer"] == "Reviewer"
    assert (dest / "promotion-manifest.json").exists()
    assert json.loads((paths["matcont"] / "metadata.json").read_text())["reviewed"] is False
    with pytest.raises(ValueError, match="exists|immutable"):
        promote(report, "matcont", tmp_path / "reference")


@pytest.mark.parametrize("mutation", ["hash", "provenance", "partial", "domain"])
def test_unsafe_candidate_cannot_be_promoted(tmp_path, mutation):
    from scripts.promote_reference import promote
    paths = candidates(tmp_path)
    path = paths["matcont"]
    meta = json.loads((path / "metadata.json").read_text())
    selected = case()
    if mutation == "hash":
        (path / "branch.csv").write_text("corrupted")
    elif mutation == "provenance":
        meta["generator_revision"] = "unknown"
    elif mutation == "partial":
        meta["coverage_status"] = "PARTIAL"
    else:
        selected["expected_domain"] = [0, 2]
    (path / "metadata.json").write_text(json.dumps(meta))
    report = manifest(tmp_path, paths, selected)
    assert report["verdict"] != "PASS"
    with pytest.raises(ValueError):
        promote(report, "matcont", tmp_path / "reference")


def test_candidate_mutation_after_review_is_rejected(tmp_path):
    from scripts.promote_reference import promote
    paths = candidates(tmp_path)
    report = manifest(tmp_path, paths)
    (paths["matcont"] / "stdout.log").write_text("changed after review")
    with pytest.raises(ValueError, match="hash|changed"):
        promote(report, "matcont", tmp_path / "reference")


def test_branch_spectrum_disagreement_cannot_pass(tmp_path):
    from examples.validation.artifacts import file_hash
    paths = candidates(tmp_path)
    path = paths["bifurcationkit"]
    spectrum = path / "spectra.csv"
    spectrum.write_text(spectrum.read_text().replace(",0,-1,0,0,", ",0,-2,0,0,"))
    meta = json.loads((path / "metadata.json").read_text())
    meta["output_hashes"]["spectra.csv"] = file_hash(spectrum)
    (path / "metadata.json").write_text(json.dumps(meta))
    report = manifest(tmp_path, paths)
    assert report["verdict"] == "FAIL"
    metric = report["pairwise"]["jaxcont--bifurcationkit"]["sheet_0.spectrum"]
    # Measured |error|: 0; exact mocked report metric.
    assert metric['value'] == pytest.approx(1, abs=0, rel=8 * np.finfo(float).eps)
    assert metric["limit"] == 1e-4


@pytest.mark.parametrize("mutation", ["status", "nonfinite", "partial"])
def test_unresolved_native_evidence_never_passes(tmp_path, mutation):
    from examples.validation.artifacts import file_hash
    paths = candidates(tmp_path)
    path = paths["matcont"]
    meta = json.loads((path / "metadata.json").read_text())
    selected = case()
    selected["evidence_status"] = "PENDING"
    if mutation == "partial":
        meta["coverage_status"] = "PARTIAL"
    else:
        branch = path / "branch.csv"
        text = branch.read_text()
        text = text.replace("CONVERGED", "FAILED") if mutation == "status" else text.replace(",2,0,0,", ",nan,0,0,")
        branch.write_text(text)
        meta["output_hashes"]["branch.csv"] = file_hash(branch)
    (path / "metadata.json").write_text(json.dumps(meta))
    assert manifest(tmp_path, paths, selected)["verdict"] == "FAIL"


def test_normal_form_without_coefficient_budget_remains_pending(tmp_path):
    paths = candidates(tmp_path)
    selected = case()
    selected["canonical_id"] = selected["id"] = "hopf-normal-form.l1"
    for path in paths.values():
        meta = json.loads((path / "metadata.json").read_text())
        meta["canonical_id"] = selected["canonical_id"]
        (path / "metadata.json").write_text(json.dumps(meta))
    report = manifest(tmp_path, paths, selected)
    assert report["verdict"] == "PENDING"


@pytest.mark.parametrize("shift, verdict", [(0.0, "PASS"), (0.01, "FAIL")])
@pytest.mark.parametrize("reverse", [False, True])
def test_curve_metrics_evaluate_dense_curve_at_sparse_native_nodes(shift, verdict, reverse):
    import numpy as np
    from examples.validation.promotion_metrics import _pairwise

    def curve(samples, offset):
        rows = [dict(segment_id="s", point_id=str(i), parameter=float(p),
                     state_0=float(p*p + offset), F=float(p*p + offset))
                for i, p in enumerate(samples)]
        spectra = [dict(segment_id="s", point_id=r["point_id"],
                        real=r["state_0"], imag=0, event_index=-1) for r in rows]
        return dict(branch=rows, spectra=spectra)

    selected = case()
    selected.update(comparison_window=[0, 1], expected_events=[])
    coarse, dense = curve(np.linspace(0, 1, 5), 0), curve(np.linspace(0, 1, 101), shift)
    left, right = (dense, coarse) if reverse else (coarse, dense)
    metrics = _pairwise(selected, left, right)
    for name in ("state_0", "F", "spectrum"):
        assert metrics[f"sheet_0.{name}"]["verdict"] == verdict


def test_window_boundary_contact_is_not_an_extra_sheet():
    from examples.validation.promotion_metrics import _sheets
    selected = case()
    selected["comparison_window"] = [0, 1]
    data = dict(branch=[dict(segment_id="inward", parameter=p) for p in [1, 0]] +
                       [dict(segment_id="outward", parameter=p) for p in [1, 2]])
    assert len(_sheets(data, selected)) == 1


@pytest.mark.parametrize("shift, verdict", [(0., "PASS"), (0.01, "FAIL")])
@pytest.mark.parametrize("reverse", [False, True])
def test_fold_comparison_uses_regular_coordinate(shift, verdict, reverse):
    import numpy as np
    from examples.validation.promotion_metrics import _pairwise

    def curve(samples, offset):
        rows = [dict(segment_id="s", point_id=str(i), parameter=float(x*x + offset),
                     state_0=float(x), state_1=float(x*x*x)) for i, x in enumerate(samples)]
        return dict(branch=rows, spectra=[dict(segment_id="s", point_id=r["point_id"],
                    real=2*r["state_0"], imag=0, event_index=-1) for r in rows])

    selected = case()
    selected.update(comparison_window=[0, 1], expected_events=[])
    selected["comparison_window"] = [1e-5, .95]
    sparse = curve(np.sqrt(np.linspace(1e-5, .95, 12)), 0)
    dense = curve(np.sqrt(np.linspace(0, 1, 101)), shift)
    metrics = _pairwise(selected, *( (dense, sparse) if reverse else (sparse, dense)))
    assert all(m["verdict"] == verdict for k, m in metrics.items() if k.endswith("parameter"))
    assert metrics["sheet_0.spectrum"]["verdict"] == "PASS"
    if verdict == "PASS":
        assert all(m["verdict"] == "PASS" for m in metrics.values())
    else:
        assert any(m["verdict"] == "FAIL" for m in metrics.values())


def test_first_window_passage_excludes_and_reports_later_reentry():
    from examples.validation.promotion_metrics import _sheets, passage_information
    selected = case()
    selected.update(comparison_window=[0, 1], comparison_passage="first")
    data = dict(branch=[dict(segment_id="chart0" if i < 2 else "chart1", point_id=str(i),
                             parameter=p, state_0=i)
                        for i, p in enumerate([1, .5, .1, -.5, -1, .2, .8, 2])])
    sheets = _sheets(data, selected)
    assert len(sheets) == 1
    assert [r["point_id"] for r in sheets[0]] == ["0", "1", "2", "3"]
    data["events"] = [dict(segment_id="chart0", point_id="1", parameter=.5, event_type="H"),
                      dict(segment_id="chart1", point_id="5", parameter=.2, event_type="H")]
    from examples.validation.promotion_metrics import event_rows
    assert [r["point_id"] for r in event_rows(selected, data)] == ["1"]
    info = passage_information(data, selected)
    assert len(info["later_reentries"]) == 1
    assert info["later_reentries"][0]["point_ids"] == ["5", "6"]

    from examples.validation.validation import build_validation_report
    selected.update(expected_events=[], spectrum_required=False)
    data["metadata"] = {}
    for row in data["branch"]:
        row["residual_norm"] = 0
    report = build_validation_report(selected, {"jaxcont": data})
    assert report.to_dict()["information"]["passages"]["jaxcont"]["later_reentries"] == info["later_reentries"]


def test_pending_item_delegated_to_a_passing_run_is_information_only(monkeypatch):
    import scripts.promote_reference as promote
    registry = {"cases": [
        {"id": "a.pass", "canonical_id": "a.pass", "aliases": [], "support": "supported", "evidence_status": "PASS"},
        {"id": "a.open", "canonical_id": "a.open", "aliases": [], "support": "supported", "evidence_status": "PENDING"},
    ]}
    monkeypatch.setattr("examples.validation.registry.load_registry", lambda *a, **k: registry)
    delegated = {"verdict": "PENDING", "reason": "unrefined BT detection excluded; refined BT is compared in a.pass"}
    assert promote._covered_elsewhere(delegated)
    assert not promote._covered_elsewhere({**delegated, "reason": "refined BT is compared in a.open"})
    assert not promote._covered_elsewhere({**delegated, "reason": "refined BT is compared in a.missing"})
    assert not promote._covered_elsewhere({"verdict": "PENDING", "reason": "no refined comparison declared"})


@pytest.mark.parametrize("level, shift, verdict", [
    # torBPC: an unstable multiplier near -50, where MatCont and
    # BifurcationKit differ by 0.033 (relative 6e-4).
    (-50.0, 0.033, "PASS"),
    (-50.0, 0.5, "FAIL"),     # relative 1e-2 is a disagreement
    (0.98, 0.0049, "PASS"),   # |mu| <= 1 keeps the absolute limit
    (0.98, 0.006, "FAIL"),
])
def test_floquet_multipliers_compare_relative_to_their_size(level, shift, verdict):
    """A declared multiplier_rtol is absolute up to |mu| = 1 and relative
    above: a large multiplier exp(lambda*T) carries an error proportional to
    its size. Without the declaration multiplier_atol stays absolute."""
    from examples.validation.promotion_metrics import _pairwise

    def curve(samples, offset):
        rows = [dict(segment_id="s", point_id=str(i), parameter=float(p), state_0=float(p))
                for i, p in enumerate(samples)]
        spectra = [dict(segment_id="s", point_id=r["point_id"], real=level + offset,
                        imag=0, event_index=-1) for r in rows]
        return dict(branch=rows, spectra=spectra)

    selected = case()
    selected.update(comparison_window=[0, 1], expected_events=[],
                    tolerances={"parameter_atol": 1e-4, "state_atol": 1e-4,
                                "multiplier_atol": 5e-3, "multiplier_rtol": 5e-3, "residual_atol": 1e-7})
    metrics = _pairwise(selected, curve(np.linspace(0, 1, 5), 0), curve(np.linspace(0, 1, 9), shift))
    assert metrics["sheet_0.spectrum"]["verdict"] == verdict
    # Review R1-02: the absolute budget alone is never read as relative.
    del selected["tolerances"]["multiplier_rtol"]
    absolute = _pairwise(selected, curve(np.linspace(0, 1, 5), 0), curve(np.linspace(0, 1, 9), shift))
    assert absolute["sheet_0.spectrum"]["verdict"] == ("PASS" if shift < 5e-3 else "FAIL")
