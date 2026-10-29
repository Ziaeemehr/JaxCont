"""Reviewed bundles must stay reproducible by the committed drivers (Loop 2, L2-F7).

Every v2 bundle records the hashes of the source files that produced it. If a
committed driver no longer has that hash, the reviewed numbers can no longer
be regenerated from the repository; this offline check catches that without
MATLAB or Julia (G8 then regenerates and compares numerically).
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "examples" / "validation"
BUNDLES = sorted(p.parent for p in (VALIDATION / "reference").glob("*/*/*/promotion-manifest.json"))


def _sources(name):
    return [p for p in VALIDATION.rglob(name) if "generated" not in p.parts and "reference" not in p.parts]


@pytest.mark.parametrize("bundle", BUNDLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_recorded_sources_match_the_committed_files(bundle):
    recorded = json.loads((bundle / "metadata.json").read_text()).get("source_hashes") or {}
    for name, digest in recorded.items():
        files = _sources(name)
        assert files, f"{name}, recorded by {bundle.name}, is not in the repository"
        current = {"sha256:" + hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        assert digest in current, f"{name} changed since {bundle.parent.name}/{bundle.name} was reviewed"


def test_every_promoted_bundle_is_checked():
    assert len(BUNDLES) >= 14


# --- verify_v2_bundles: only differences without meaning are ignored ---------

import csv
import shutil

from examples.validation.artifacts import ValidationMismatch, file_hash, verify_v2_bundles

SOURCE = VALIDATION / "reference" / "lorenz84" / "bifurcationkit" / "lorenz84.hopf-ft"
TOLERANCES = {"parameter_atol": 1e-9}


def _copy(tmp_path, name):
    target = tmp_path / name
    shutil.copytree(SOURCE, target)
    return target


def _rewrite(bundle, table, edit):
    path = bundle / f"{table}.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    fields, rows = edit(fields, rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = json.loads((bundle / "metadata.json").read_text())
    metadata["output_hashes"][f"{table}.csv"] = file_hash(path)
    (bundle / "metadata.json").write_text(json.dumps(metadata))


def _swap_first_conjugate_pair(fields, rows):
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        same_point = all(a.get(k) == b.get(k) for k in ("segment_id", "point_id", "event_id"))
        if same_point and float(a["imag"]) != 0 and float(a["imag"]) == -float(b["imag"]):
            for key in ("real", "imag", "logabs", "phase"):
                if key in a:
                    a[key], b[key] = b[key], a[key]
            return fields, rows
    raise AssertionError("fixture has no conjugate pair")


def test_conjugate_pair_order_is_not_a_difference(tmp_path):
    reordered = _copy(tmp_path, "reordered")
    _rewrite(reordered, "spectra", _swap_first_conjugate_pair)
    assert verify_v2_bundles(reordered, SOURCE, TOLERANCES)["max_numeric_error"] == 0


def test_a_changed_eigenvalue_is_still_a_difference(tmp_path):
    changed = _copy(tmp_path, "changed")

    def nudge(fields, rows):
        rows[0]["real"] = repr(float(rows[0]["real"]) + 1e-3)
        return fields, rows

    _rewrite(changed, "spectra", nudge)
    with pytest.raises(ValidationMismatch):
        verify_v2_bundles(changed, SOURCE, TOLERANCES)


def test_derived_event_kind_column_alone_is_not_a_difference(tmp_path):
    without = _copy(tmp_path, "without-kind")
    _rewrite(without, "events", lambda fields, rows: ([f for f in fields if f != "kind"],
                                                       [{k: v for k, v in r.items() if k != "kind"} for r in rows]))
    verify_v2_bundles(without, SOURCE, TOLERANCES)


def test_nan_on_one_side_only_is_a_difference(tmp_path):
    one_sided = _copy(tmp_path, "one-sided-nan")

    def blank(fields, rows):
        rows[0]["parameter"] = "nan"
        return fields, rows

    _rewrite(one_sided, "events", blank)
    with pytest.raises(ValidationMismatch, match="NaN on one side only"):
        verify_v2_bundles(one_sided, SOURCE, TOLERANCES)
