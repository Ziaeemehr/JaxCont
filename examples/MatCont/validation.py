"""Shared case-level gates for CLI and visual MatCont validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import compare_case_result_to_reference
from .compare import ValidationMismatch
from .python_cases import CaseResult

_REFERENCE_COMPARISON_CASES = {
    "MC-EQ-001",
    "MC-EQ-002",
    "MC-EQ-003",
    "MC-LC-001",
}


@dataclass(frozen=True)
class CaseValidation:
    """Outcome from the validation gate shared by every presentation layer."""

    passed: bool
    producer_passed: bool
    checks: dict[str, Any]
    diagnostics: dict[str, float] | None
    numerical_mismatch: str | None


def case_result_passes(case_id: str, checks: dict[str, Any]) -> bool:
    """Apply the authoritative producer-specific predicate for one case."""
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
        "MC-PRC-001": lambda: checks["prc_matches_matcont"],
    }
    return bool(predicates[case_id]())


def evaluate_case_result(
    case: dict[str, Any],
    result: CaseResult,
    reference_dir: Path,
) -> CaseValidation:
    """Combine producer checks with the applicable MatCont comparison.

    Numerical ``ValidationMismatch`` failures are returned as diagnostic state
    so a caller can render them. Structural exceptions retain their native
    types and are expected to be rejected by artifact validation or plotting.
    """
    checks = dict(result.checks)
    diagnostics = None
    numerical_mismatch = None
    if case["id"] in _REFERENCE_COMPARISON_CASES:
        try:
            diagnostics = compare_case_result_to_reference(
                case,
                result,
                reference_dir,
            )
        except ValidationMismatch as exc:
            numerical_mismatch = str(exc)
        checks["matcont_reference"] = (
            diagnostics
            if diagnostics is not None
            else {"mismatch": numerical_mismatch}
        )

    producer_passed = case_result_passes(case["id"], checks)
    return CaseValidation(
        passed=producer_passed and numerical_mismatch is None,
        producer_passed=producer_passed,
        checks=checks,
        diagnostics=diagnostics,
        numerical_mismatch=numerical_mismatch,
    )


__all__ = ["CaseValidation", "case_result_passes", "evaluate_case_result"]
