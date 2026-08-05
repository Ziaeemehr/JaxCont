"""Focused contract tests for the MatCont validation-suite foundation."""

import numpy as np
import pytest

from examples.MatCont.compare import (
    ValidationMismatch,
    interpolate_observable,
    match_events,
    match_spectrum,
    scaled_close,
)
from examples.MatCont.registry import load_registry, select_cases
from examples.MatCont.run_validation import build_parser
from examples.MatCont.python_cases.codim2 import run_codim2_points
from examples.MatCont.python_cases.equilibrium import (
    run_adaptive_control_hopf,
    run_cubic_fold,
    run_vanderpol_hopf,
)
from examples.MatCont.python_cases.transforms import run_transform_checks


REQUIRED_CASE_FIELDS = {
    "id",
    "title",
    "support",
    "features",
    "python",
    "matlab",
    "references",
    "manual_source",
    "tolerances",
}


def test_cubic_case_finds_both_analytic_folds():
    """Dropping either turning point would leave half the S-curve unvalidated."""
    result = run_cubic_fold()

    assert result.checks["fold_count"] == 2
    assert result.checks["max_fold_error"] < 5e-4
    assert result.checks["max_residual"] < 2e-5
    assert result.checks["natural_stalled_at_fold"]
    assert result.checks["palc_traversed_both_folds"]
    assert result.checks["max_fold_coefficient_error"] < 1e-4
    assert result.checks["stability_transition_count"] == 2


def test_vanderpol_case_recovers_the_analytic_hopf():
    """A crossing away from mu=0 or omega=1 is not the Van der Pol Hopf."""
    result = run_vanderpol_hopf()

    assert result.checks["hopf_count"] == 1
    assert result.checks["max_hopf_error"] < 5e-4
    assert result.checks["frequency_error"] < 5e-4
    assert result.checks["max_residual"] < 2e-5
    assert result.checks["lyapunov_error"] < 1e-4
    assert result.checks["stable_for_negative_parameter"]
    assert result.checks["unstable_for_positive_parameter"]


def test_adaptive_control_case_recovers_the_analytic_hopf():
    """Changing the adaptx characteristic polynomial must move this check."""
    result = run_adaptive_control_hopf()

    assert result.checks["hopf_count"] == 1
    assert result.checks["max_hopf_error"] < 5e-4
    assert result.checks["frequency_error"] < 5e-4
    assert result.checks["max_residual"] < 2e-5
    assert result.checks["lyapunov_error"] < 5e-4


def test_transform_case_matches_analytic_and_finite_difference_gradients():
    """A broken custom derivative can return right points but wrong sensitivities."""
    result = run_transform_checks()

    assert result.checks["all_finite"]
    assert result.checks["max_analytic_gradient_error"] < 2e-3
    assert result.checks["max_finite_difference_error"] < 2e-3
    assert result.checks["jit_matches_eager"]
    assert result.checks["vmap_valid_masks_present"]
    assert result.checks["permutation_invariant"]


def test_codim2_case_recovers_all_shifted_points():
    """Returning a seed or zero must not pass shifted codimension-two fixtures."""
    result = run_codim2_points()

    assert result.checks["all_converged"]
    assert result.checks["max_parameter_error"] < 1e-3
    assert result.checks["bt_bifurcationkit_error"] < 1e-3
    assert result.checks["frequency_error"] < 1e-3
    assert result.checks["gh_lyapunov_error"] < 1e-4
    assert result.checks["parameter_gradients_finite"]
    assert result.checks["max_analytic_gradient_error"] < 2e-3
    assert result.checks["max_finite_difference_error"] < 2e-3


def test_default_selection_excludes_unsupported():
    """Removing the support filter would select MatCont-only cases by default."""
    selected = select_cases(load_registry())

    assert selected
    assert all(case["support"] != "unsupported" for case in selected)


def test_registry_cases_have_the_required_comparison_metadata():
    """Removing a required registry field would make later validators ambiguous."""
    cases = load_registry()["cases"]

    assert cases
    assert all(REQUIRED_CASE_FIELDS <= set(case) for case in cases)


def test_select_cases_filters_ids_without_reordering_registry_entries():
    """Replacing ID selection with set-based filtering would lose requested registry order."""
    registry = {
        "cases": [
            {"id": "MC-EQ-001", "support": "supported"},
            {"id": "MC-EQ-002", "support": "supported"},
        ]
    }

    assert select_cases(registry, ids=["MC-EQ-002", "MC-EQ-001"]) == registry["cases"]


def test_scaled_close_reports_the_largest_absolute_error():
    """Dropping absolute-error diagnostics would hide the scale of an accepted mismatch."""
    diagnostics = scaled_close(
        np.array([1.0, 2.0]), np.array([1.0, 2.001]), atol=0.002
    )

    assert diagnostics["max_error"] == pytest.approx(0.001)


def test_scaled_close_rejects_errors_beyond_the_scaled_tolerance():
    """Ignoring the tolerance threshold would accept a numerically wrong comparison."""
    with pytest.raises(ValidationMismatch):
        scaled_close(np.array([1.0]), np.array([1.1]), atol=1e-3, rtol=1e-3)


def test_interpolate_observable_interpolates_each_observable_column():
    """Using only the first observable column would corrupt vector branch comparisons."""
    interpolated = interpolate_observable(
        np.array([0.0, 1.0, 2.0]),
        np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]),
        np.array([0.5, 1.5]),
    )

    np.testing.assert_allclose(interpolated, np.array([[1.0, 2.0], [3.0, 6.0]]))


def test_match_events_rejects_duplicate_reuse():
    """Reusing one actual event for two references would mask a missing event."""
    reference = [{"kind": "LP", "parameter": -1.0}, {"kind": "LP", "parameter": 1.0}]
    actual = [{"kind": "LP", "parameter": -1.0}]

    with pytest.raises(ValidationMismatch):
        match_events(actual, reference, atol=1e-3)


def test_match_events_returns_unique_assignments_and_maximum_error():
    """Greedy reuse or missing diagnostics would obscure event correspondence."""
    actual = [{"kind": "LP", "parameter": 1.002}, {"kind": "LP", "parameter": -1.001}]
    reference = [{"kind": "LP", "parameter": -1.0}, {"kind": "LP", "parameter": 1.0}]

    diagnostics = match_events(actual, reference, atol=0.01)

    assert diagnostics["assignments"] == [(1, 0), (0, 1)]
    assert diagnostics["max_error"] == pytest.approx(0.002)


def test_match_spectrum_ignores_only_one_trivial_multiplier():
    """Leaving the trivial multiplier in matching makes otherwise equal spectra disagree."""
    actual = np.array([1.0, 0.5, -1.0])
    reference = np.array([1.0, -1.0, 0.5])

    assert match_spectrum(actual, reference, atol=1e-8)["max_error"] == pytest.approx(0.0)


def test_match_spectrum_retains_a_second_unit_multiplier():
    """Removing every unit multiplier would hide a genuine spectral mismatch."""
    actual = np.array([1.0, 1.0, 0.5])
    reference = np.array([1.0, 0.8, 0.5])

    with pytest.raises(ValidationMismatch):
        match_spectrum(actual, reference, atol=1e-3)


def test_match_spectrum_prefers_a_tolerance_feasible_assignment():
    """Minimizing raw distance first can reject a spectrum with a valid matching."""
    actual = np.array([1.0, 0.672222 + 0.870699j, 0.9])
    reference = np.array([1.0, 0.0, 0.9])

    diagnostics = match_spectrum(actual, reference, atol=1.0)

    assert diagnostics["max_error"] == pytest.approx(0.9, abs=1e-5)
    assert diagnostics["assignments"] == [(2, 1), (1, 2)]


def test_cli_parser_exposes_the_foundation_options_and_environment_defaults(monkeypatch):
    """Removing a CLI option or environment override would block reproducible execution."""
    monkeypatch.setenv("MATLAB_BIN", "/custom/matlab")
    monkeypatch.setenv("MATCONT_ROOT", "/custom/matcont")

    args = build_parser().parse_args(
        [
            "--case",
            "MC-EQ-001",
            "--regenerate-matcont",
            "--verify-references",
            "--include-unsupported",
        ]
    )

    assert args.case == ["MC-EQ-001"]
    assert args.regenerate_matcont
    assert args.verify_references
    assert args.include_unsupported
    assert args.matlab_bin == "/custom/matlab"
    assert args.matcont_root == "/custom/matcont"
