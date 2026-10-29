"""Candidate-only serialization shared by native numerical producers.

No reference promotion is performed here. Native data must already contain
all numerical values; this module only writes tables and records provenance.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

from .artifacts import _V2_COLUMNS, candidate_directory, file_hash, load_bundle


def write_candidate(bundle, output_dir, *, tool, driver, equation, lock=None):
    output = candidate_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").unlink(missing_ok=True)
    hashes = {}
    for table in ("branch", "events", "spectra"):
        rows = bundle[table]
        name = table + ".csv"
        columns = sorted(set(_V2_COLUMNS[name]).union(*(row.keys() for row in rows)))
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        hashes[name] = file_hash(output / name)
    if "orbits" in bundle:
        rows = bundle["orbits"]
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (output / "orbits.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        hashes["orbits.csv"] = file_hash(output / "orbits.csv")
    hashes.update({path.name: file_hash(path) for path in output.glob("*.csv")})
    metadata = dict(bundle.get("metadata", {}))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, check=True).stdout.strip()
    metadata["source_hashes"] = {str(Path(p).name): file_hash(p) for p in Path(driver).parent.glob("lorenz84*.m")} if tool == "matcont" else {}
    if tool == "jaxcont" and bundle["canonical_id"] in ("adaptx.cycle-pd-switch", "torbpc.cycle-lpc-pd-ns"):
        metadata["source_hashes"].update({p.name: file_hash(p) for p in (Path(driver).parent / "periodic_checks.py", Path(driver).parents[1] / "systems/periodic.json")})
    if bundle["canonical_id"] == "adaptx.cycle-pd-switch" and tool == "matcont":
        metadata["source_hashes"].update({p.name: file_hash(p) for p in (Path(driver).parent / "export_cycle_run.m", Path(driver).parents[2] / "cases/systems/periodic.json")})
    if bundle["canonical_id"] in ("lorenz84.fold-ft", "lorenz84.hopf-ft"):
        from .cases.systems.lorenz84 import chart_coverage
        T_values = [float(r["T"]) for r in bundle["branch"]]
        expected, status = chart_coverage(bundle["canonical_id"], T_values)
        metadata.update(observed_domain=[min(T_values), max(T_values)], expected_domain=expected, coverage_status=status)
        if metadata["coverage_status"] == "PARTIAL":
            metadata["chart_status"] = "Native Hopf boundary or step budget reached" if bundle["canonical_id"].endswith("hopf-ft") else "Configured step budget reached before expected T domain"
    metadata.update(schema_version=2, canonical_id=bundle["canonical_id"], tool=tool,
                    equation_hash=file_hash(equation), driver_hash=file_hash(driver),
                    lock_hash=file_hash(lock) if lock else "unknown",
                    generated_utc=datetime.now(timezone.utc).isoformat(),
                    generator_revision=revision, output_hashes=hashes,
                    reviewed=False, reviewer="unknown", review_date="unknown", conversion_chain=[])
    metadata.setdefault("command", [sys.executable, *sys.argv])
    metadata.setdefault("platform", {"os": platform.platform(), "cpu": platform.processor() or platform.machine(), "blas": "see native runtime log"})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    for name, records in bundle.get("extra_tables", {}).items():
        (output / name).write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return load_bundle(output, case_id=bundle["canonical_id"], tool=tool)
