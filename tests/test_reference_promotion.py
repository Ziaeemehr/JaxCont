"""Promotion must preserve evidence and reject unsafe or failing copies."""
import json

import pytest

from tests.test_validation_schema import make_bundle


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
    assert metric["value"] == pytest.approx(1)
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
