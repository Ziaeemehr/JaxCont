"""Command-line entry point for the MatCont validation suite."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .artifacts import (
    compare_case_result_to_reference,
    enrich_generated_metadata,
    validate_equilibrium_artifacts,
    validate_periodic_artifacts,
    validate_reference_metadata,
    verify_case_references,
)
from .registry import load_registry, select_cases


_DEFAULT_MATLAB_BIN = "/home/ziaee/prog/Matlab/R2020a/bin/matlab"
_DEFAULT_MATCONT_ROOT = "/home/ziaee/prog/MatCont/MatCont7p6"
_SUITE_DIR = Path(__file__).resolve().parent
_DEFAULT_REFERENCE_DIR = _SUITE_DIR / "reference"
_DEFAULT_GENERATED_DIR = _SUITE_DIR / "generated"
_EQUATION_FILES = {
    "MC-EQ-001": _SUITE_DIR / "matlab" / "systems" / "cubic_fold.m",
    "MC-EQ-002": _SUITE_DIR / "matlab" / "systems" / "vanderpol_hopf.m",
    "MC-EQ-003": _SUITE_DIR / "matlab" / "systems" / "adaptive_control.m",
    "MC-LC-001": _SUITE_DIR / "matlab" / "systems" / "radial_cycle_system.m",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the validation CLI parser without performing any validation work."""
    parser = argparse.ArgumentParser(description="Run JaxCont's MatCont validation suite.")
    parser.add_argument(
        "--case", action="append", help="Validate only this case ID; may be repeated."
    )
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
        default=os.environ.get("MATLAB_BIN", _DEFAULT_MATLAB_BIN),
        help="Path to the MATLAB executable (default: %(default)s).",
    )
    parser.add_argument(
        "--matcont-root",
        default=os.environ.get("MATCONT_ROOT", _DEFAULT_MATCONT_ROOT),
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
        type=Path,
        default=_DEFAULT_GENERATED_DIR,
        help="Regenerated MatCont artifact directory (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report case selection without running Python or MATLAB producers.",
    )
    return parser


def resolve_runtime_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve every filesystem override before MATLAB changes its working directory."""
    args.reference_dir = Path(args.reference_dir).expanduser().resolve()
    args.generated_dir = Path(args.generated_dir).expanduser().resolve()
    args.matlab_bin = str(Path(args.matlab_bin).expanduser().resolve())
    args.matcont_root = str(Path(args.matcont_root).expanduser().resolve())
    return args


def validate_case_metadata(
    metadata_path: Path, case_id: str, reference_dir: Path
) -> dict:
    """Validate metadata, requiring review only for the committed oracle directory."""
    require_reviewed = Path(reference_dir).resolve() == _DEFAULT_REFERENCE_DIR.resolve()
    return validate_reference_metadata(
        metadata_path, case_id, require_reviewed=require_reviewed
    )


def _matlab_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _run_matlab_function(
    function_name: str,
    *,
    matlab_bin: str,
    matcont_root: str,
    output_dir: Path | None,
) -> None:
    matlab_dir = _SUITE_DIR / "matlab"
    unsupported_dir = matlab_dir / "unsupported"
    arguments = "" if output_dir is None else f"('{_matlab_quote(output_dir)}')"
    expression = (
        f"addpath('{_matlab_quote(matlab_dir)}','{_matlab_quote(unsupported_dir)}'); "
        f"{function_name}{arguments}"
    )
    environment = os.environ.copy()
    environment["MATCONT_ROOT"] = matcont_root
    completed = subprocess.run(
        [matlab_bin, "-batch", expression],
        cwd=matlab_dir,
        env=environment,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"MATLAB producer {function_name} exited {completed.returncode}")


def _equation_path(case: dict, matcont_root: str) -> Path:
    if case["id"] == "MC-LC-002":
        return Path(matcont_root) / "Testruns" / "TestSystems" / "torBPC.m"
    return _EQUATION_FILES[case["id"]]


def _regenerate(cases: list[dict], args: argparse.Namespace) -> bool:
    args.generated_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for case in cases:
        function_name = case["matlab"]
        if not function_name:
            continue
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
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: MatCont wrapper completed")
                continue
            _run_matlab_function(
                function_name,
                matlab_bin=args.matlab_bin,
                matcont_root=args.matcont_root,
                output_dir=args.generated_dir,
            )
            metadata_path = args.generated_dir / f"{case['id']}_metadata.json"
            enrich_generated_metadata(
                metadata_path, case, _equation_path(case, args.matcont_root)
            )
            print(f"REGENERATED {case['id']}")
        except Exception as exc:  # every producer failure must remain visible
            print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
            ok = False
    return ok


def _load_case_callable(specification: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError(f"invalid Python case entry point: {specification}")
    return getattr(importlib.import_module(module_name), function_name)


def _case_result_passes(case_id: str, checks: dict) -> bool:
    predicates = {
        "MC-EQ-001": lambda: (
            checks["fold_count"] == 2
            and checks["max_fold_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["natural_stalled_at_fold"]
            and checks["palc_traversed_both_folds"]
            and checks["max_fold_coefficient_error"] < 1e-4
            and checks["stability_transition_count"] == 2
        ),
        "MC-EQ-002": lambda: (
            checks["hopf_count"] == 1
            and checks["max_hopf_error"] < 5e-4
            and checks["frequency_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["lyapunov_error"] < 1e-4
            and checks["stable_for_negative_parameter"]
            and checks["unstable_for_positive_parameter"]
        ),
        "MC-EQ-003": lambda: (
            checks["hopf_count"] == 1
            and checks["max_hopf_error"] < 5e-4
            and checks["frequency_error"] < 5e-4
            and checks["max_residual"] < 2e-5
            and checks["lyapunov_error"] < 5e-4
        ),
        "MC-JAX-001": lambda: (
            checks["all_finite"]
            and checks["max_analytic_gradient_error"] < 2e-3
            and checks["max_finite_difference_error"] < 2e-3
            and checks["jit_matches_eager"]
            and checks["vmap_valid_masks_present"]
            and checks["vmap_valid_masks_match_serial"]
            and checks["vmap_valid_parameters_match_serial"]
            and checks["vmap_valid_states_match_serial"]
            and checks["permutation_invariant"]
        ),
        "MC-C2-001": lambda: (
            checks["all_converged"]
            and checks["max_parameter_error"] < 1e-3
            and checks["bt_bifurcationkit_error"] < 1e-3
            and checks["frequency_error"] < 1e-3
            and checks["gh_lyapunov_error"] < 1e-4
            and checks["parameter_gradients_finite"]
            and checks["max_analytic_gradient_error"] < 2e-3
            and checks["max_finite_difference_error"] < 2e-3
        ),
        "MC-LC-001": lambda: (
            checks["max_radius_error"] < 5e-3
            and checks["max_period_error"] < 5e-3
            and checks["max_collocation_residual"] < 2e-5
            and checks["max_multiplier_error"] < 5e-3
            and checks["all_stable"]
        ),
        "MC-LC-002": lambda: bool(checks["all_comparisons_pass"]),
    }
    return bool(predicates[case_id]())


def _validate_case(case: dict, reference_dir: Path) -> tuple[bool, dict]:
    for filename in case["references"]:
        path = reference_dir / filename
        if filename.endswith("_metadata.json"):
            validate_case_metadata(path, case["id"], reference_dir)
        elif not path.is_file():
            raise FileNotFoundError(f"missing MatCont reference artifact: {path}")
    if case["id"].startswith("MC-EQ-"):
        validate_equilibrium_artifacts(reference_dir, case["id"])
    elif case["id"] in {"MC-LC-001", "MC-LC-002"}:
        validate_periodic_artifacts(reference_dir, case["id"])
    function = _load_case_callable(case["python"])
    parameters = inspect.signature(function).parameters
    result = function(reference_dir) if parameters else function()
    checks = dict(result.checks)
    if case["id"] in {"MC-EQ-001", "MC-EQ-002", "MC-EQ-003", "MC-LC-001"}:
        checks["matcont_reference"] = compare_case_result_to_reference(
            case, result, reference_dir
        )
    return _case_result_passes(case["id"], checks), checks


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate, verify, or execute selected validation cases.

    Reviewed references are read-only; regeneration always targets ``generated``.
    """
    args = resolve_runtime_paths(build_parser().parse_args(argv))
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
    cases = select_cases(
        registry, ids=args.case, include_unsupported=args.include_unsupported
    )
    if args.dry_run:
        for case in cases:
            if case["support"] == "unsupported":
                suffix = (
                    " NON_EXECUTABLE_TEMPLATE"
                    if case.get("unsupported_execution") == "template"
                    else ""
                )
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: {case['title']}{suffix}")
            else:
                print(f"WOULD_RUN {case['id']}: {case['title']}")
        return 0

    success = True
    if args.regenerate_matcont:
        success = _regenerate(cases, args) and success
    if args.verify_references:
        for case in cases:
            if case["support"] == "unsupported" or not case["references"]:
                continue
            try:
                diagnostics = verify_case_references(
                    case, args.generated_dir, args.reference_dir
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

    if args.regenerate_matcont:
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
                print(f"UNSUPPORTED_BY_JAXCONT {case['id']}: MatCont wrapper completed")
            except Exception as exc:
                print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
                success = False
            continue
        try:
            passed, checks = _validate_case(case, args.reference_dir)
            if passed:
                print(f"PASS {case['id']}: {case['title']}")
            else:
                print(f"FAIL {case['id']}: {checks}", file=sys.stderr)
                success = False
        except Exception as exc:
            print(f"FAIL {case['id']}: {exc}", file=sys.stderr)
            success = False
    return int(not success)


if __name__ == "__main__":
    raise SystemExit(main())
