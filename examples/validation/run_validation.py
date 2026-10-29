"""Command-line entry point for the MatCont validation suite."""

from __future__ import annotations

import argparse
import importlib
import json
import inspect
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .artifacts import (
    enrich_generated_metadata,
    case_result_payload,
    candidate_directory,
    load_bundle,
    verify_v2_bundles,
    reference_directory,
    validate_equilibrium_artifacts,
    validate_periodic_artifacts,
    validate_reference_metadata,
    verify_case_references,
)
from .registry import load_registry, select_cases
from .validation import build_validation_report, evaluate_case_result

_SUITE_DIR = Path(__file__).resolve().parent
_DEFAULT_REFERENCE_DIR = _SUITE_DIR / "reference"
_DEFAULT_GENERATED_DIR = _SUITE_DIR / "generated"
_EQUATION_FILES = {
    "MC-EQ-001": _SUITE_DIR / "regenerate" / "matcont" / "systems" / "cubic_fold.m",
    "MC-EQ-002": _SUITE_DIR / "regenerate" / "matcont" / "systems" / "vanderpol_hopf.m",
    "MC-EQ-003": _SUITE_DIR / "regenerate" / "matcont" / "systems" / "adaptive_control.m",
    "MC-LC-001": _SUITE_DIR / "regenerate" / "matcont" / "systems" / "radial_cycle_system.m",
}


def resolve_matlab_bin(override: str | None = None) -> str | None:
    """Prefer an explicit override, then MATLAB_BIN, then MATLAB on PATH."""
    configured = override or os.environ.get("MATLAB_BIN")
    if configured:
        return str(Path(configured).expanduser().resolve())
    return shutil.which("matlab")


def build_parser(*, runtime_defaults=True) -> argparse.ArgumentParser:
    """Build the validation CLI parser without running validation."""
    parser = argparse.ArgumentParser(description="Run JaxCont's MatCont validation suite.")
    parser.add_argument(
        "--case",
        action="append",
        help="Validate only this case ID; may be repeated.",
    )
    parser.add_argument("--regenerate", choices=["matcont", "bifurcationkit"])
    parser.add_argument("--julia-bin")
    parser.add_argument("--bifurcationkit-project", type=Path)
    parser.add_argument("--julia-depot")
    parser.add_argument(
        "--regenerate-matcont",
        action="store_true",
        help="Regenerate MatCont artifacts before validation.",
    )
    parser.add_argument(
        "--verify-references",
        action="store_true",
        help="Compare regenerated artifacts against reviewed references.",
    )
    parser.add_argument(
        "--include-unsupported",
        action="store_true",
        help="Include MatCont-only cases that JaxCont does not yet support.",
    )
    parser.add_argument(
        "--matlab-bin",
        default=(os.environ.get("MATLAB_BIN") or None) if runtime_defaults else None,
        help="Path to the MATLAB executable (default: %(default)s).",
    )
    parser.add_argument(
        "--matcont-root",
        default=(os.environ.get("MATCONT_ROOT") or None) if runtime_defaults else None,
        help="Path to the MatCont installation (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=_DEFAULT_REFERENCE_DIR,
        help="Reviewed CSV/JSON reference directory (default: %(default)s).",
    )
    parser.add_argument(
        "--generated-dir",
        "--output-dir",
        type=Path,
        default=_DEFAULT_GENERATED_DIR,
        help="Regenerated MatCont artifact directory (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Report case selection without running Python or MATLAB " "producers."),
    )
    return parser


def resolve_runtime_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve paths before MATLAB changes its working directory."""
    args.reference_dir = Path(args.reference_dir).expanduser().resolve()
    args.generated_dir = Path(args.generated_dir).expanduser().resolve()
    args.matlab_bin = resolve_matlab_bin(args.matlab_bin)
    args.matcont_root = args.matcont_root or os.environ.get("MATCONT_ROOT")
    if args.matcont_root:
        args.matcont_root = str(Path(args.matcont_root).expanduser().resolve())
    return args


def validate_case_metadata(metadata_path: Path, case_id: str, reference_dir: Path) -> dict:
    """Require reviewed metadata only for the committed oracle."""
    require_reviewed = Path(reference_dir).resolve() == _DEFAULT_REFERENCE_DIR.resolve()
    return validate_reference_metadata(metadata_path, case_id, require_reviewed=require_reviewed)


def _matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _run_matlab_function(
    function_name: str,
    *,
    matlab_bin: str | None,
    matcont_root: str | None,
    output_dir: Path | None,
) -> None:
    if not matlab_bin or not shutil.which(matlab_bin):
        raise RuntimeError("MATLAB is unavailable (set MATLAB_BIN)")
    if not matcont_root:
        raise RuntimeError("set MATCONT_ROOT")
    matlab_dir = _SUITE_DIR / "regenerate" / "matcont"
    capability_dir = matlab_dir / "capability_gaps"
    arguments = "" if output_dir is None else f"('{_matlab_quote(output_dir)}')"
    expression = (
        f"addpath('{_matlab_quote(matlab_dir)}',"
        f"'{_matlab_quote(capability_dir)}'); "
        f"{function_name}{arguments}"
    )
    environment = os.environ.copy()
    environment["MATCONT_ROOT"] = matcont_root
    import tempfile

    if output_dir is not None:
        output_dir = candidate_directory(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jaxcont-matlab-prefs-") as prefs:
        environment["MATLAB_PREFDIR"] = prefs
        try:
            completed = subprocess.run(
                [matlab_bin, "-batch", expression],
                cwd=matlab_dir,
                env=environment,
                text=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=600 if function_name == "run_adaptx_pd" else 300,
            )
        except subprocess.TimeoutExpired as exc:
            _save_matlab_logs(output_dir, function_name, exc.stdout, exc.stderr)
            raise RuntimeError(
                f"MATLAB producer {function_name} timed out; logs: {output_dir}"
            ) from exc
    _save_matlab_logs(output_dir, function_name, completed.stdout, completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"MATLAB producer {function_name} exited {completed.returncode}")
    return completed


def _save_matlab_logs(directory, name, stdout, stderr):
    if directory is None:
        directory = _DEFAULT_GENERATED_DIR / "logs"
    Path(directory).mkdir(parents=True, exist_ok=True)
    for stream, value in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        (Path(directory) / f"{name}.{stream}.log").write_text(value or "", encoding="utf-8")


def _equation_path(case: dict, matcont_root: str) -> Path:
    if case["id"] == "MC-LC-002":
        return Path(matcont_root) / "Testruns" / "TestSystems" / "torBPC.m"
    if case["id"] == "MC-PRC-001":
        return Path(matcont_root) / "Testruns" / "TestSystems" / "adaptx.m"
    return _EQUATION_FILES[case["id"]]


def _regenerate(cases: list[dict], args: argparse.Namespace) -> bool:
    candidate_directory(args.generated_dir, args.reference_dir)
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for case in cases:
        function_name = case["matlab"]
        if not function_name:
            print(f"PENDING {case['id']}: no MATLAB producer", file=sys.stderr)
            ok = False
            continue
        output = None
        if case["canonical_id"].startswith("lorenz84.") or case["canonical_id"] == "adaptx.cycle-pd-switch":
            output = args.generated_dir / case["canonical_id"] / "matcont"
            output.mkdir(parents=True, exist_ok=True)
            # A failure-only driver invocation must never refresh old evidence.
            for stale in ("raw.json", "metadata.json"):
                (output / stale).unlink(missing_ok=True)
        try:
            if case["support"] == "unsupported":
                if case.get("unsupported_execution") == "template":
                    raise RuntimeError("NON_EXECUTABLE_TEMPLATE")
                _run_matlab_function(
                    function_name,
                    matlab_bin=args.matlab_bin,
                    matcont_root=args.matcont_root,
                    output_dir=None,
                )
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: " "MatCont wrapper completed")
                continue
            completed = _run_matlab_function(
                function_name,
                matlab_bin=args.matlab_bin,
                matcont_root=args.matcont_root,
                output_dir=args.generated_dir,
            )
            if case["canonical_id"].startswith("lorenz84.") or case["canonical_id"] == "adaptx.cycle-pd-switch":
                from .candidates import write_candidate
                output = args.generated_dir / case["canonical_id"] / "matcont"
                raw = output / "raw.json"
                if not raw.exists():
                    failure = output / "failure.json"
                    raise RuntimeError(f"No complete native candidate; see {failure}")
                native = json.loads(raw.read_text())
                # MATLAB's jsonencode writes a one-element struct array (the
                # single-point lorenz84.bt-ft branch) as an object, not a list.
                for table in ("branch", "events", "spectra", "orbits"):
                    if isinstance(native.get(table), dict):
                        native[table] = [native[table]]
                if completed is not None:
                    native["metadata"]["command"] = completed.args
                write_candidate(native, output, tool="matcont",
                    driver=_SUITE_DIR / "regenerate" / "matcont" / (function_name + ".m"),
                    equation=(_SUITE_DIR / "regenerate" / "matcont" / "systems" / "lorenz84.m" if case["canonical_id"].startswith("lorenz84.") else Path(args.matcont_root) / "Testruns/TestSystems/adaptx.m"))
                (output / "failure.json").unlink(missing_ok=True)
                print(f"REGENERATED {case['canonical_id']}")
                continue
            metadata_path = args.generated_dir / f"{case['id']}_metadata.json"
            enrich_generated_metadata(metadata_path, case, _equation_path(case, args.matcont_root))
            print(f"REGENERATED {case['id']}")
        except Exception as exc:  # every producer failure must remain visible
            if output is not None:
                (output / "metadata.json").unlink(missing_ok=True)
                (output / "failure.json").write_text(json.dumps({"canonical_id": case["canonical_id"], "status": "FAIL", "error": str(exc)}, indent=2) + "\n")
            print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
            ok = False
        finally:
            if output is not None:
                for stream in ("stdout", "stderr"):
                    log = args.generated_dir / f"{function_name}.{stream}.log"
                    if log.exists():
                        shutil.copy2(log, output / log.name)
    return ok


def _regenerate_julia(cases, args):
    from .regenerate.bifurcationkit.runner import run_julia

    candidate_directory(args.generated_dir, args.reference_dir)
    success = True
    for case in cases:
        script = case.get("bifurcationkit")
        if not script:
            print(f"PENDING {case['canonical_id']}: no headless Julia producer", file=sys.stderr)
            success = False
            continue
        output = args.generated_dir / case["canonical_id"] / "bifurcationkit"
        try:
            run_julia(
                _SUITE_DIR / "regenerate" / "bifurcationkit" / script,
                ["--case", case["canonical_id"], "--output-dir", str(output)],
                600 if script in ("adaptx_pd.jl", "torbpc_ns.jl") else 300,
                cli=args.julia_bin,
                project=args.bifurcationkit_project,
                depot=args.julia_depot,
                log_dir=output,
            )
            load_bundle(output, case_id=case["canonical_id"], tool="bifurcationkit")
            print(f"REGENERATED {case['canonical_id']}")
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=True)
            (output / "metadata.json").unlink(missing_ok=True)
            (output / "failure.json").write_text(json.dumps({"canonical_id": case["canonical_id"], "status": "FAIL", "error": str(exc)}, indent=2) + "\n")
            print(f"FAIL {case['canonical_id']}: {exc}", file=sys.stderr)
            success = False
    return success


def _load_case_callable(specification: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError(f"invalid Python case entry point: {specification}")
    return getattr(importlib.import_module(module_name), function_name)


def _validate_case(case: dict, reference_dir: Path) -> tuple[bool, dict]:
    if case.get("evidence_status") == "PENDING" and not any(
        (Path(reference_dir) / case["system"] / tool / case["canonical_id"] / "metadata.json").is_file()
        for tool in case.get("tool_coverage", {}) if tool != "jaxcont"
    ):
        return False, {"status": "PENDING", "reason": "reviewed evidence pending Task 2.12"}
    v2_paths = {
        tool: Path(reference_dir) / case["system"] / tool / case["canonical_id"]
        for tool in case.get("tool_coverage", {})
        if tool != "jaxcont"
    }
    if any((path / "metadata.json").is_file() for path in v2_paths.values()):
        results = {
            tool: load_bundle(
                path,
                case_id=case["canonical_id"],
                tool=tool,
                require_reviewed=Path(reference_dir).resolve() == _DEFAULT_REFERENCE_DIR.resolve(),
            )
            for tool, path in v2_paths.items()
            if (path / "metadata.json").is_file()
        }
        if case.get("python"):
            fresh = _load_case_callable(case.get("v2_python", case["python"]))()
            results["jaxcont"] = (fresh if isinstance(fresh, dict) else
                                  fresh.artifacts.get("bundle") or case_result_payload(case, fresh))
        report = build_validation_report(case, results)
        return report.status == "PASS", {
            "status": report.status,
            "validation_report": report.to_dict(),
            "reason": (
                "required evidence incomplete" if report.status == "PENDING" else report.status
            ),
        }
    bundle_dir = reference_directory(reference_dir, case["id"])
    for filename in case["references"]:
        path = bundle_dir / filename
        if filename.endswith("_metadata.json"):
            validate_case_metadata(path, case["id"], reference_dir)
        elif not path.is_file():
            raise FileNotFoundError(f"missing MatCont reference artifact: {path}")
    if case["id"].startswith("MC-EQ-"):
        validate_equilibrium_artifacts(reference_dir, case["id"])
    elif case["id"] in {"MC-LC-001", "MC-LC-002"}:
        validate_periodic_artifacts(reference_dir, case["id"])
        if case['id'] == 'MC-LC-001':
            # This control has an exact period at every native sample, including
            # endpoint rows outside the pairwise interpolation samples.
            import csv
            import math
            with (bundle_dir / 'MC-LC-001_branch.csv').open(newline='') as stream:
                if any(abs(float(row['period']) - 2 * math.pi) > case['tolerances']['period_atol']
                       for row in csv.DictReader(stream)):
                    raise ValueError('radial reference period exceeds exact period budget')

    function = _load_case_callable(case["python"])
    parameters = inspect.signature(function).parameters
    result = function(reference_dir) if "reference_dir" in parameters else function()
    validation = evaluate_case_result(case, result, reference_dir)
    return validation.passed, validation.checks


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate, verify, or execute selected validation cases.

    Reviewed references are read-only; regeneration targets ``generated``.
    """
    args = build_parser(runtime_defaults=False).parse_args(argv)
    if args.regenerate_matcont:
        if args.regenerate and args.regenerate != "matcont":
            print("conflicting regeneration flags", file=sys.stderr)
            return 2
        args.regenerate = "matcont"
    args.regenerate_matcont = args.regenerate == "matcont"
    if args.regenerate:
        try:
            candidate_directory(args.generated_dir, args.reference_dir)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.generated_dir = args.generated_dir.expanduser().resolve()
    if args.regenerate_matcont or args.include_unsupported:
        args = resolve_runtime_paths(args)
    registry = load_registry()
    if args.case and not args.include_unsupported:
        unsupported_requested = sorted(
            case["id"]
            for case in registry["cases"]
            if case["id"] in set(args.case) and case["support"] == "unsupported"
        )
        if unsupported_requested:
            print(
                "Unsupported case(s) "
                + ", ".join(unsupported_requested)
                + " requires --include-unsupported",
                file=sys.stderr,
            )
            return 2
    cases = select_cases(registry, ids=args.case, include_unsupported=args.include_unsupported)
    if args.dry_run:
        for case in cases:
            if case["support"] == "unsupported":
                suffix = (
                    " NON_EXECUTABLE_TEMPLATE"
                    if case.get("unsupported_execution") == "template"
                    else ""
                )
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: " f"{case['title']}{suffix}")
            else:
                print(f"WOULD_RUN {case['id']}: {case['title']}")
        return 0

    success = True
    if args.regenerate:
        if args.regenerate == "matcont":
            success = _regenerate(cases, args) and success
        else:
            success = _regenerate_julia(cases, args) and success
    if args.verify_references:
        for case in cases:
            if case["support"] == "unsupported":
                continue
            try:
                candidate = (
                    args.generated_dir
                    / case["canonical_id"]
                    / (args.regenerate or "matcont")
                )
                reference = (
                    args.reference_dir
                    / case["system"]
                    / (args.regenerate or "matcont")
                    / case["canonical_id"]
                )
                if not (candidate / "metadata.json").exists() and not case["references"]:
                    raise ValueError("no candidate or reviewed bundle is available")
                diagnostics = (
                    verify_v2_bundles(candidate, reference, case["tolerances"])
                    if (candidate / "metadata.json").exists()
                    else verify_case_references(case, args.generated_dir, args.reference_dir)
                )
                print(
                    f"REFERENCE_OK {case['id']}: "
                    f"{diagnostics['artifact_count']} artifacts, "
                    f"max error {diagnostics['max_numeric_error']:.3e}"
                )
            except Exception as exc:
                print(f"REFERENCE_FAIL {case['id']}: {exc}", file=sys.stderr)
                success = False
        return int(not success)

    if args.regenerate:
        return int(not success)

    for case in cases:
        if case["support"] == "unsupported":
            if case.get("unsupported_execution") == "template":
                print(
                    f"NON_EXECUTABLE_TEMPLATE {case['id']}: {case['title']}",
                    file=sys.stderr,
                )
                success = False
                continue
            try:
                _run_matlab_function(
                    case["matlab"],
                    matlab_bin=args.matlab_bin,
                    matcont_root=args.matcont_root,
                    output_dir=None,
                )
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: " "MatCont wrapper completed")
            except Exception as exc:
                print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
                success = False
            continue
        try:
            passed, checks = _validate_case(case, args.reference_dir)
            if checks.get("status") == "PENDING":
                print(f"PENDING {case['id']}: {checks['reason']}")
                success = False
            elif passed:
                print(f"PASS {case['id']}: {case['title']}")
                report = checks.get("validation_report", {})
                if report.get("status") == "PENDING":
                    print(
                        f"PENDING {case.get('canonical_id', case['id'])}: full tool coverage; legacy case checks above passed"
                    )
            else:
                print(f"FAIL {case['id']}: {checks}", file=sys.stderr)
                success = False
        except Exception as exc:
            print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
            success = False
    return int(not success)


if __name__ == "__main__":
    raise SystemExit(main())
