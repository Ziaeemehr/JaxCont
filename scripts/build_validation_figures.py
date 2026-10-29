"""Build every comparison-page figure and check it against its report (gate G4).

  python3 scripts/build_validation_figures.py --output-dir /tmp/jaxcont-evidence --check

Each page in ``examples/validation/gallery`` runs in a fresh process with
``--output-dir <dir>/<page>`` and reads only committed bundles. With
``--check`` every PNG must have a ``.checks.json`` sidecar whose
``report_hash`` matches its own contents, and the report record embedded in
the PNG metadata must equal the sidecar, so a figure cannot carry a status or
numbers its report does not. ``--check-only`` checks an existing directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "examples" / "validation" / "gallery"


def build(output_dir: Path, pages: list[Path], timeout: float) -> list[str]:
    problems = []
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for page in pages:
        target = output_dir / page.stem
        target.mkdir(parents=True, exist_ok=True)
        run = subprocess.run([sys.executable, str(page), "--output-dir", str(target)], cwd=ROOT, env=env,
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        print(f"{'BUILT' if run.returncode == 0 else 'FAIL '} {page.name}")
        if run.returncode != 0:
            problems.append(f"{page.name}: exit {run.returncode}: {run.stderr.strip()[-500:]}")
    return problems


def _report_hash(record: dict) -> str:
    """The hash ValidationReport.record() computes over everything but the hash itself."""
    payload = {key: value for key, value in record.items() if key != "report_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def check(output_dir: Path) -> list[str]:
    from PIL import Image

    problems = []
    figures = sorted(output_dir.rglob("*.png"))
    if not figures:
        return [f"no figures under {output_dir}"]
    for figure in figures:
        name = figure.relative_to(output_dir)
        sidecar = figure.with_suffix(".checks.json")
        if not sidecar.is_file():
            problems.append(f"{name}: no .checks.json sidecar")
            continue
        record = json.loads(sidecar.read_text())
        if record.get("status") not in ("PASS", "FAIL", "PENDING"):
            problems.append(f"{name}: sidecar status {record.get('status')!r}")
        if record.get("report_hash") != _report_hash(record):
            problems.append(f"{name}: sidecar report_hash does not match its contents")
        embedded = Image.open(figure).info.get("Description")
        if embedded is None:
            problems.append(f"{name}: PNG carries no report record")
        elif json.loads(embedded) != record:
            problems.append(f"{name}: PNG record differs from its sidecar")
    print(f"{'PASS' if not problems else 'FAIL'}  {len(figures)} figures in {output_dir}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="check the figures after building them")
    parser.add_argument("--check-only", action="store_true", help="check an existing directory without building")
    parser.add_argument("--page", action="append", help="build only these pages (file names)")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)
    pages = sorted(GALLERY.glob("plot_*.py"))
    if args.page:
        pages = [page for page in pages if page.name in set(args.page)]
    problems = [] if args.check_only else build(args.output_dir, pages, args.timeout)
    if args.check or args.check_only:
        problems += check(args.output_dir)
    for problem in problems:
        print(f"FAIL  {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
