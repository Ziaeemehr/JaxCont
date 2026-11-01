"""Measure candidate agreement before an explicit, immutable reference copy.

Run without --promote to write a manifest only. Missing evidence is retained
as PENDING; measured violations are FAIL. No tolerances are inferred or enlarged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.validation.artifacts import file_hash, load_bundle, _monotone_segments
from examples.validation.registry import load_registry, select_cases
from examples.validation.validation import build_validation_report


def _metric(value, limit, reason=None):
    finite = value is not None and np.isfinite(value)
    return dict(value=float(value) if finite else None, limit=limit,
                verdict="PENDING" if limit is None else "PASS" if finite and value <= limit else "FAIL", reason=reason)


def _hashes(path):
    return {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*"))
            if p.is_file()}


def _covered_elsewhere(metric):
    from examples.validation.promotion_metrics import covered_elsewhere
    return covered_elsewhere(metric)


def _provenance(path, data, case):
    meta = data["metadata"]
    if meta["schema_version"] != 2:
        raise ValueError("Native v1 candidate needs an explicit provenance-preserving v2 exporter")
    for field in ("generator_revision", "runtime_version", "precision", "command"):
        if meta[field] in ("unknown", "", None):
            raise ValueError(f"incomplete provenance: {field}")
    if meta["precision"] not in ("float64", "double"):
        raise ValueError("promotion requires float64")
    if not re.fullmatch(r"[0-9a-f]{40,64}", meta["generator_revision"]):
        raise ValueError("incomplete provenance: generator_revision")
    if not any(p.suffix == ".log" for p in path.rglob("*")):
        raise ValueError("incomplete provenance: native logs absent")
    if (meta.get("coverage_status") == "PARTIAL"
            and case.get("evidence_status") not in ("FAIL", "PENDING")):
        raise ValueError("PARTIAL coverage is not a known FAIL/PENDING run")


def _read_candidate(path, case, tool):
    if (path / "metadata.json").exists():
        return load_bundle(path, case_id=case["canonical_id"], tool=tool)
    # Inspect fresh legacy output without changing or substituting reviewed data.
    if (path / f"{case['id']}_metadata.json").exists():
        return load_bundle(path, case_id=case["id"], tool=tool)
    raise ValueError("no complete native candidate")


from examples.validation.promotion_metrics import _independent, _pairwise, _sheets


def normalize_fresh_matcont(source, output, case, *, generator_revision):
    """Normalize a successful fresh v1 export while preserving original bytes.

    This is deliberately separate from historical conversion: successful native
    invocation and current generator identity are mandatory, and the MatCont
    exporter stores only corrected branch points. No legacy reviewed data is
    rewritten and no unavailable diagnostics are synthesized.
    """
    from examples.validation.artifacts import convert_v1_bundle
    source, output = Path(source), Path(output)
    invocation = json.loads((source / "invocation.json").read_text())
    if invocation.get("returncode") != 0 or (source / "failure.json").exists():
        raise ValueError("fresh normalization requires a successful native invocation")
    if not re.fullmatch(r"[0-9a-f]{40,64}", generator_revision):
        raise ValueError("generator revision required")
    original = _read_candidate(source, case, "matcont")
    if original["metadata"]["schema_version"] != 1:
        raise ValueError("normalization requires native v1 data")
    convert_v1_bundle(source, output, case_id=case["id"], canonical_id=case["canonical_id"])
    import csv
    for table in ("branch", "events", "spectra"):
        path = output / (table + ".csv")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream); fields = list(reader.fieldnames); rows = list(reader)
        if table == "events" and "kind" not in fields:
            fields.append("kind")
        for row in rows:
            if table in ("branch", "events"):
                row["status"] = "CONVERGED" if table == "branch" else "REFINED"
            if table == "events": row["kind"] = row["event_type"]
            if table == "spectra":
                z = complex(float(row["real"]), float(row["imag"]))
                row["logabs"] = str(float(np.log(max(abs(z), np.finfo(float).tiny))))
                row["phase"] = str(float(np.angle(z)))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    meta_path = output / "metadata.json"
    meta = json.loads(meta_path.read_text()); native = original["metadata"]
    meta.update(runtime_version=native["matlab_version"], generator_revision=generator_revision,
                command=invocation["command"], generated_utc=native["generated_utc"],
                package_versions={"MatCont":native["matcont_version"]},
                mesh=native.get("mesh", "unknown"), resolved_controls=native.get("solver_settings", "unknown"),
                normalization_note="Successful fresh cont invocation stores corrected branch points; event rows are native localized special points. Unavailable v1 diagnostics remain unknown.")
    meta["output_hashes"] = {p.name:file_hash(p) for p in output.glob("*.csv")}
    meta["conversion_chain"][0]["output_hashes"] = meta["output_hashes"]
    meta_path.write_text(json.dumps(meta,indent=2)+"\n")
    (output / "conversion-manifest.json").write_text(json.dumps(meta["conversion_chain"][0],indent=2)+"\n")
    for path in source.glob("*.log"): shutil.copy2(path,output/path.name)
    shutil.copy2(source / "invocation.json", output / "invocation.json")
    load_bundle(output, case_id=case["canonical_id"], tool="matcont")
    return output


def write_manifest(case, candidates, output, *, reviewer, review_date):
    """Write all source identities and measured differences; never promote here."""
    date.fromisoformat(review_date)
    if not reviewer.strip():
        raise ValueError("reviewer required")
    report = dict(schema_version=1, canonical_id=case["canonical_id"], system=case["system"],
                  reviewer=reviewer, review_date=review_date, expected_status=case["evidence_status"],
                  registry_case=deepcopy(case), candidates={}, independent={}, pairwise={}, event_information={}, comparison_coordinates={}, passage_information={})
    bundles = {}
    for tool, candidate in candidates.items():
        path = Path(candidate).resolve()
        hashes = _hashes(path)
        invocation = path / "invocation.json"
        command = json.loads(invocation.read_text()).get("command", "unknown") if invocation.exists() else "unknown"
        entry = dict(path=str(path), hashes=hashes, promotable=False, command=command,
                     logs={k:v for k,v in hashes.items() if k.endswith(".log")})
        report["candidates"][tool] = entry
        try:
            data = _read_candidate(path, case, tool)
            bundles[tool] = data
            meta = data["metadata"]
            entry.update(command=meta.get("command", command),
                         logs={k:v for k,v in entry["hashes"].items() if k.endswith(".log")},
                         coverage=meta.get("coverage_status", "unspecified"))
            report["independent"][tool] = _independent(case, data)
            from examples.validation.promotion_metrics import event_rows, unrefined, passage_information
            report["passage_information"][tool] = passage_information(data, case)
            inside = event_rows(case,data,refined=False)
            report["event_information"][tool] = {
                "outside_window": [r for r in data.get("events",[]) if r not in inside],
                "unrefined_detections": [{**r,"refinement_status":"UNREFINED_DETECTION"} for r in inside if unrefined(r)]}
            _provenance(path, data, case)
            entry["promotable"] = True
        except (ValueError, KeyError, FileNotFoundError) as exc:
            entry["reason"] = str(exc)
    for a, b in combinations(bundles, 2):
        coordinates = {}
        report["pairwise"][f"{a}--{b}"] = _pairwise(case, bundles[a], bundles[b], coordinates=coordinates)
        report["comparison_coordinates"][f"{a}--{b}"] = coordinates
    # Promotion shares the CLI/publication gate. Registry PENDING describes
    # review state; it must not prevent measuring newly generated candidates.
    measured_case = deepcopy(case)
    measured_case["tool_coverage"] = {t: "PASS" if t in bundles else "PENDING"
                                      for t in case["tool_coverage"]}
    gate = build_validation_report(measured_case, bundles)
    report["validation_report"] = gate.to_dict()
    all_metrics = [m for group in (report["independent"], report["pairwise"]) for metrics in group.values() for m in metrics.values()]
    failed = gate.status == "FAIL" or any(m["verdict"] == "FAIL" for m in all_metrics)
    pending = any(m["verdict"] == "PENDING" and not _covered_elsewhere(m) for m in all_metrics)
    complete = all(t in bundles for t in case["tool_coverage"])
    eligible = all(e["promotable"] for e in report["candidates"].values())
    report["verdict"] = "FAIL" if failed else "PASS" if complete and eligible and not pending else "PENDING"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def promote(manifest, tool, reference_root):
    """Copy exactly the reviewed candidate; exclusive creation prevents overwrite."""
    if manifest["verdict"] != "PASS":
        raise ValueError("manifest does not pass")
    if tool == "jaxcont":
        raise ValueError("JaxCont evidence stays fresh")
    entry = manifest["candidates"][tool]
    if not entry["promotable"]:
        raise ValueError("candidate provenance is incomplete")
    source = Path(entry["path"])
    if _hashes(source) != entry["hashes"]:
        raise ValueError("candidate hashes changed after review")
    data = _read_candidate(source, manifest["registry_case"], tool)
    _provenance(source, data, manifest["registry_case"])
    destination = Path(reference_root) / manifest["system"] / tool / manifest["canonical_id"]
    if destination.exists():
        raise ValueError(f"immutable bundle already exists: {destination}")
    size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    if size + len(manifest_bytes) > 10 * 1024 * 1024:
        raise ValueError("bundle exceeds 10 MiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        target = Path(staging) / "bundle"
        shutil.copytree(source, target)
        meta = deepcopy(data["metadata"])
        meta.update(reviewed=True, reviewer=manifest["reviewer"], review_date=manifest["review_date"])
        (target / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (target / "promotion-manifest.json").write_bytes(manifest_bytes)
        load_bundle(target, require_reviewed=True)
        if sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) > 10 * 1024 * 1024:
            raise ValueError("bundle exceeds 10 MiB")
        # mkdir is exclusive even if another promotion raced our initial check.
        destination.mkdir()
        try:
            for item in target.iterdir():
                shutil.move(str(item), str(destination / item.name))
        except BaseException:
            shutil.rmtree(destination)
            raise
    return destination


def summary(manifest):
    metrics = [(f"{pair}.{name}", m) for pair, rows in manifest["pairwise"].items() for name, m in rows.items()]
    measured = [item for item in metrics if item[1]["value"] is not None and item[1]["limit"] is not None]
    def ratio(item):
        m = item[1]
        return float("inf") if m["value"] is None or m["limit"] is None or (m["limit"] == 0 and m["value"] != 0) else m["value"] / m["limit"] if m["limit"] else 0
    worst = max(measured, key=ratio) if measured else None
    label = f"{worst[0]}={worst[1]['value']} / {worst[1]['limit']}" if worst else "unavailable"
    tools = ",".join(manifest["independent"])
    unavailable = len(metrics) - len(measured)
    if unavailable:
        label += f"; {unavailable} unavailable"
    return f"{manifest['canonical_id']} | {tools or 'none'} | {label} | {manifest['verdict']}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--promote", action="append", choices=["matcont", "bifurcationkit"])
    parser.add_argument("--reference-root", type=Path, default=ROOT / "examples/validation/reference")
    args = parser.parse_args(argv)
    case = select_cases(load_registry(), [args.case])[0]
    paths = {t: args.candidate_root / case["system"] / t / case["canonical_id"] for t in case["tool_coverage"]}
    manifest = write_manifest(case, paths, args.manifest_dir / f"{case['canonical_id']}.json",
                              reviewer=args.reviewer, review_date=args.date)
    print(summary(manifest))
    for tool in args.promote or []:
        promote(manifest, tool, args.reference_root)
    return int(manifest["verdict"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
