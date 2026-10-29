"""Gate G4: a figure, its sidecar and its embedded report record must agree."""

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_validation_figures", ROOT / "scripts" / "build_validation_figures.py")
g4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g4)


def _figure(path, record, sidecar=None):
    figure = plt.figure()
    figure.savefig(path, metadata={"Description": json.dumps(record)})
    plt.close(figure)
    path.with_suffix(".checks.json").write_text(json.dumps(record if sidecar is None else sidecar))


def _record(status="PASS"):
    record = {"case_id": "cubic.eq-fold", "status": status, "metrics": [{"metric": "m", "measured": 1e-9}]}
    return record | {"report_hash": g4._report_hash(record)}


def test_consistent_figures_pass(tmp_path):
    _figure(tmp_path / "a.png", _record())
    _figure(tmp_path / "b.png", _record("FAIL"))
    assert g4.check(tmp_path) == []


def test_missing_sidecar_and_disagreement_are_reported(tmp_path):
    _figure(tmp_path / "ok.png", _record())
    _figure(tmp_path / "lies.png", _record("FAIL"), sidecar=_record("PASS"))
    _figure(tmp_path / "orphan.png", _record())
    (tmp_path / "orphan.checks.json").unlink()
    tampered = _record()
    tampered["metrics"][0]["measured"] = 1e-3
    _figure(tmp_path / "tampered.png", tampered)
    problems = g4.check(tmp_path)
    assert any("lies.png: PNG record differs from its sidecar" in p for p in problems)
    assert any("orphan.png: no .checks.json sidecar" in p for p in problems)
    assert any("tampered.png: sidecar report_hash does not match" in p for p in problems)
    assert not any(p.startswith("ok.png") for p in problems)


def test_hash_matches_the_report_record():
    """The checker recomputes exactly what ValidationReport.record() writes."""
    from examples.validation.reporting import ValidationReport
    import dataclasses
    fields = {f.name: None for f in dataclasses.fields(ValidationReport)}
    fields.update(case_id="x", status="PENDING", metrics=[], checks={}, diagnostics={}, precision="float64",
                  reference_hashes={}, compared_domain={})
    record = json.loads(json.dumps(ValidationReport(**fields).record(), default=str))
    assert record["report_hash"] == g4._report_hash(record)
