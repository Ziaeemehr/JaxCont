"""Small bundle fixtures exercise provenance, integrity, conversion, and CLI safety."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest
from examples.validation import artifacts, run_validation
from examples.validation.registry import load_registry, select_cases


def make_bundle(path, *, tool="matcont"):
    path.mkdir(parents=True)
    files = {
        "branch.csv": "segment_id,point_id,order,parameter,state_0,residual_norm,unstable_dimension,status\ns,0,0,0,1,0,0,CONVERGED\ns,1,1,1,2,0,0,CONVERGED\n",
        "events.csv": "event_id,event_type,status,segment_id,point_id,parameter,bracket_lower,bracket_upper\ne,LP,CONVERGED,s,0,0,0,1\n",
        "spectra.csv": "segment_id,point_id,event_id,kind,mode_id,real,imag,logabs,phase,trivial_mode_method\ns,0,,EIGENVALUE,0,-1,0,0,3.141592653589793,not-applicable\n",
    }
    for name, value in files.items():
        (path / name).write_text(value)
    metadata = dict(
        schema_version=2,
        canonical_id="synthetic.eq",
        tool=tool,
        runtime_version="test",
        package_versions={"test": "1"},
        lock_hash="unknown",
        platform={"os": "test", "cpu": "test", "blas": "test"},
        precision="float64",
        equation_hash="sha256:" + "a" * 64,
        driver_hash="sha256:" + "b" * 64,
        parameter_order=["p"],
        scaling={"state": [1]},
        seeds={"state": [1], "parameter": 0},
        mesh={"kind": "equilibrium"},
        resolved_controls={"newton_atol": 1e-9},
        solver_strategies={"method": "test"},
        command=["test"],
        generated_utc="2026-10-28T00:00:00Z",
        generator_revision="unknown",
        reviewed=False,
        reviewer="unknown",
        review_date="unknown",
        conversion_chain=[],
        output_hashes={
            name: "sha256:" + hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in files
        },
    )
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def test_v2_requires_tool_equation_and_driver_provenance(tmp_path):
    path = make_bundle(tmp_path / "bundle")
    assert artifacts.load_bundle(path)["metadata"]["schema_version"] == 2
    for field in ("tool", "equation_hash", "driver_hash"):
        data = json.loads((path / "metadata.json").read_text())
        del data[field]
        (path / "metadata.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match=field):
            artifacts.load_bundle(path)
        path = make_bundle(tmp_path / field)


def test_bundle_hash_mutation_is_rejected(tmp_path):
    path = make_bundle(tmp_path / "bundle")
    with (path / "branch.csv").open("a") as stream:
        stream.write("s,2,2,2,3,0,0,CONVERGED\n")
    with pytest.raises(ValueError, match="hash"):
        artifacts.load_bundle(path)


def test_legacy_v1_bundle_reads_with_unknown_fields(tmp_path):
    source = artifacts.reference_directory(run_validation._DEFAULT_REFERENCE_DIR, "MC-EQ-001")
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    bundle = artifacts.load_bundle(source, case_id="MC-EQ-001")
    assert bundle["metadata"]["schema_version"] == 1
    assert bundle["metadata"]["driver_hash"] == "unknown"
    output = tmp_path / "converted"
    artifacts.convert_v1_bundle(source, output, case_id="MC-EQ-001", canonical_id="cubic.eq-fold")
    converted = artifacts.load_bundle(output)
    assert converted["metadata"]["conversion_chain"]
    assert converted["metadata"]["reviewed"] is False
    assert json.loads((output / "conversion-manifest.json").read_text())["source_hashes"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_pending_tool_cannot_satisfy_coverage():
    from examples.validation.validation import coverage_status

    assert (
        coverage_status({"jaxcont": "PASS", "matcont": "PENDING"}, ["jaxcont", "matcont"])
        == "PENDING"
    )


@pytest.mark.parametrize("tool", ["matcont", "bifurcationkit"])
def test_regeneration_never_writes_into_reference(tool, monkeypatch, tmp_path):
    reference = run_validation._DEFAULT_REFERENCE_DIR
    link = tmp_path / "link"
    link.symlink_to(reference, target_is_directory=True)
    monkeypatch.setattr(run_validation, "_regenerate", lambda *a: pytest.fail("producer called"))
    for target in (reference, reference / "new", link / "new"):
        assert (
            run_validation.main(
                ["--regenerate", tool, "--case", "cubic.eq-fold", "--output-dir", str(target)]
            )
            == 2
        )


def test_aliases_are_identical_and_collisions_rejected(tmp_path):
    registry = load_registry()
    assert select_cases(registry, ["cubic.eq-fold"]) == select_cases(registry, ["MC-EQ-001"])
    registry["cases"][1]["aliases"].append("cubic.eq-fold")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="collision"):
        load_registry(path)


def test_v2_numeric_verification_is_not_byte_comparison(tmp_path):
    a = make_bundle(tmp_path / "a")
    b = make_bundle(tmp_path / "b")
    data = (b / "branch.csv").read_text().replace(",2,0,0,CONVERGED", ",2.00000001,0,0,CONVERGED")
    (b / "branch.csv").write_text(data)
    meta = json.loads((b / "metadata.json").read_text())
    meta["output_hashes"]["branch.csv"] = (
        "sha256:" + hashlib.sha256((b / "branch.csv").read_bytes()).hexdigest()
    )
    (b / "metadata.json").write_text(json.dumps(meta))
    assert artifacts.verify_v2_bundles(a, b, {"state_atol": 1e-6})["max_numeric_error"] > 0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("precision", ""),
        ("platform", {}),
        ("parameter_order", []),
        ("resolved_controls", {}),
        ("generated_utc", "yesterday"),
        ("reviewed", "yes"),
    ],
)
def test_v2_rejects_malformed_provenance_values(tmp_path, field, bad):
    path = make_bundle(tmp_path / "bundle")
    metadata = json.loads((path / "metadata.json").read_text())
    metadata[field] = bad
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        artifacts.load_bundle(path)


def test_verification_cannot_succeed_without_candidate(tmp_path, monkeypatch, capsys):
    case = select_cases(load_registry(), ["lorenz84.eq-f"])[0]
    monkeypatch.setattr(run_validation, "load_registry", lambda: {"cases": [case]})
    assert (
        run_validation.main(
            ["--verify-references", "--case", case["id"], "--output-dir", str(tmp_path)]
        )
        == 1
    )
    assert "REFERENCE_FAIL" in capsys.readouterr().err


def test_conversion_chain_cannot_bypass_provenance(tmp_path):
    path = make_bundle(tmp_path / "bundle")
    metadata = json.loads((path / "metadata.json").read_text())
    metadata.update(driver_hash="unknown", conversion_chain=["anything"])
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="conversion_chain"):
        artifacts.load_bundle(path)


def test_legacy_prc_conversion_preserves_data_without_inventing_branch(tmp_path):
    source = artifacts.reference_directory(run_validation._DEFAULT_REFERENCE_DIR, "MC-PRC-001")
    bundle = artifacts.load_bundle(source, case_id="MC-PRC-001")
    assert bundle["prc"]
    output = tmp_path / "converted"
    artifacts.convert_v1_bundle(source, output, case_id="MC-PRC-001", canonical_id="adaptx.prc")
    converted = artifacts.load_bundle(output)
    assert converted["prc"] == bundle["prc"]
    assert converted["branch"] == []
    assert converted["metadata"]["limitations"]


def test_verification_rejects_equation_identity_change(tmp_path):
    a, b = make_bundle(tmp_path / "a"), make_bundle(tmp_path / "b")
    metadata = json.loads((b / "metadata.json").read_text())
    metadata["equation_hash"] = "sha256:" + "c" * 64
    (b / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(artifacts.ValidationMismatch, match="equation_hash"):
        artifacts.verify_v2_bundles(a, b, {})


def _external_literal_assignments(source):
    """External expectations and seeds must be supplied by bundle readers."""
    import ast
    import re
    violations = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        if not any(re.match(r'(BK_|bk_|matcont_.*reference|bifurcationkit_|U_SEED$|P_SEED$)', name) for name in names):
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float, complex))
               for n in ast.walk(node.value)):
            violations.append(node.lineno)
    return violations


def test_no_hand_copied_external_constants_in_tests():
    root = Path(__file__).resolve().parent
    for name in ('test_codim2.py', 'test_curves.py', 'test_hopf_normal_form.py'):
        assert not _external_literal_assignments((root / name).read_text()), name
    assert _external_literal_assignments('BK_BT_P = (1.23456789, 0.12345678)')
    assert not _external_literal_assignments('BK_BT_P = read_bundle()["parameter"]')
