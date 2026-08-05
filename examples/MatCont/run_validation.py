"""Command-line entry point for the MatCont validation suite."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from .registry import load_registry, select_cases


_DEFAULT_MATLAB_BIN = "/home/ziaee/prog/Matlab/R2020a/bin/matlab"
_DEFAULT_MATCONT_ROOT = "/home/ziaee/prog/MatCont/MatCont7p6"


def build_parser() -> argparse.ArgumentParser:
    """Build the validation CLI parser without performing any validation work."""
    parser = argparse.ArgumentParser(description="Run JaxCont's MatCont validation suite.")
    parser.add_argument("--case", action="append", help="Validate only this case ID; may be repeated.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options and report the selected foundation cases.

    Case execution and reference verification are added by later validation-suite tasks.
    """
    args = build_parser().parse_args(argv)
    cases = select_cases(
        load_registry(), ids=args.case, include_unsupported=args.include_unsupported
    )
    for case in cases:
        print(f"SELECTED {case['id']}: {case['title']}")
    if args.regenerate_matcont:
        print("MATCONT_REGENERATION_REQUESTED")
    if args.verify_references:
        print("REFERENCE_VERIFICATION_REQUESTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
