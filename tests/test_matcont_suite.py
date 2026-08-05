"""Focused contract tests for the MatCont validation-suite foundation."""

import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

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
from examples.MatCont.run_validation import build_parser, resolve_runtime_paths
from examples.MatCont.artifacts import (
    validate_equilibrium_artifacts,
    enrich_generated_metadata,
    validate_reference_metadata,
    verify_case_references,
)
from examples.MatCont.python_cases.codim2 import run_codim2_points
from examples.MatCont.python_cases.equilibrium import (
    run_adaptive_control_hopf,
    run_cubic_fold,
    run_vanderpol_hopf,
)
from examples.MatCont.python_cases.transforms import run_transform_checks
from examples.MatCont.python_cases.periodic import (
    load_torbpc_reference,
    run_radial_cycle,
    run_torbpc_cycle,
)


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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_adaptive_control_case_recovers_the_analytic_hopf():
    """Changing the adaptx characteristic polynomial must move this check."""
    result = run_adaptive_control_hopf()

    assert result.checks["hopf_count"] == 1
    assert result.checks["max_hopf_error"] < 5e-4
    assert result.checks["frequency_error"] < 5e-4
    assert result.checks["max_residual"] < 2e-5
    assert result.checks["lyapunov_error"] < 5e-4


@pytest.mark.slow
def test_transform_case_matches_analytic_and_finite_difference_gradients():
    """A broken custom derivative can return right points but wrong sensitivities."""
    result = run_transform_checks()

    assert result.checks["all_finite"]
    assert result.checks["max_analytic_gradient_error"] < 2e-3
    assert result.checks["max_finite_difference_error"] < 2e-3
    assert result.checks["jit_matches_eager"]
    assert result.checks["vmap_valid_masks_present"]
    assert result.checks["permutation_invariant"]


@pytest.mark.slow
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


@pytest.mark.slow
def test_radial_cycle_matches_exact_floquet_formula():
    """Using equilibrium eigenvalues in place of Floquet multipliers must fail."""
    result = run_radial_cycle()

    assert result.checks["max_radius_error"] < 5e-3
    assert result.checks["max_period_error"] < 5e-3
    assert result.checks["max_collocation_residual"] < 2e-5
    assert result.checks["max_multiplier_error"] < 5e-3
    assert result.checks["all_stable"]


def _write_torbpc_reference(path: Path) -> None:
    (path / "MC-LC-002_branch.csv").write_text(
        "case_id,point,parameter,period,residual_norm,state_0_min,state_0_max,"
        "state_1_min,state_1_max,state_2_min,state_2_max\n"
        "MC-LC-002,0,-0.5844928424,8.411866,1e-8,-0.20,0.20,-0.10,0.10,-0.30,0.30\n"
        "MC-LC-002,1,-0.5957504315,8.861099,1e-8,-0.30,0.30,-0.20,0.20,-0.40,0.40\n"
        "MC-LC-002,2,-0.6146816596,9.256846,1e-8,-0.40,0.40,-0.30,0.30,-0.50,0.50\n",
        encoding="utf-8",
    )
    (path / "MC-LC-002_events.csv").write_text(
        "case_id,event_index,event_type,point,parameter,period\n"
        "MC-LC-002,0,LPC,0,-0.5844928424,8.411866\n"
        "MC-LC-002,1,NS,1,-0.5957504315,8.861099\n"
        "MC-LC-002,2,PD,2,-0.6146816596,9.256846\n",
        encoding="utf-8",
    )
    (path / "MC-LC-002_multipliers.csv").write_text(
        "case_id,point,event_index,event_type,spectrum_kind,multiplier_index,real,imag\n"
        "MC-LC-002,0,0,LPC,FLOQUET,0,2.8,0.0\n"
        "MC-LC-002,0,0,LPC,FLOQUET,1,1.0,0.0\n"
        "MC-LC-002,0,0,LPC,FLOQUET,2,1.0,0.0\n"
        "MC-LC-002,1,1,NS,FLOQUET,0,1.0,0.0\n"
        "MC-LC-002,1,1,NS,FLOQUET,1,0.6,0.8\n"
        "MC-LC-002,1,1,NS,FLOQUET,2,0.6,-0.8\n"
        "MC-LC-002,2,2,PD,FLOQUET,0,1.0,0.0\n"
        "MC-LC-002,2,2,PD,FLOQUET,1,-1.0,0.0\n"
        "MC-LC-002,2,2,PD,FLOQUET,2,-0.5,0.0\n",
        encoding="utf-8",
    )


def test_torbpc_reference_schema_covers_events_periods_extrema_and_multipliers(tmp_path):
    """Missing normalized periodic diagnostics must fail before a slow JaxCont sweep."""
    _write_torbpc_reference(tmp_path)

    reference = load_torbpc_reference(tmp_path)

    assert set(reference["events_by_type"]) == {"LPC", "NS", "PD"}
    assert all(len(reference["spectra"][kind]) == 3 for kind in ("LPC", "NS", "PD"))
    assert all(len(reference["extrema"][kind]) == 3 for kind in ("LPC", "NS", "PD"))
    assert reference["max_event_parameter_error"] < 2e-3
    assert reference["all_periods_finite_positive"]
    assert reference["critical_multipliers_match"]


def test_equilibrium_reference_schema_requires_spectra_and_normal_forms(tmp_path):
    """An empty equilibrium spectrum file must not masquerade as a reviewed reference."""
    (tmp_path / "MC-EQ-003_branch.csv").write_text(
        "case_id,point,parameter,residual_norm,stable,unstable_dimension,state_0,state_1,state_2\n"
        "MC-EQ-003,0,1.0,0.0,0,2,0.0,0.0,0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "MC-EQ-003_events.csv").write_text(
        "case_id,event_index,event_type,point,parameter,frequency,fold_coefficient,first_lyapunov\n"
        "MC-EQ-003,0,H,0,1.0,1.0,nan,-0.3\n",
        encoding="utf-8",
    )
    (tmp_path / "MC-EQ-003_multipliers.csv").write_text(
        "case_id,point,event_index,event_type,spectrum_kind,multiplier_index,real,imag\n"
        "MC-EQ-003,0,-1,BRANCH,EIGENVALUE,0,0.0,1.0\n"
        "MC-EQ-003,0,-1,BRANCH,EIGENVALUE,1,0.0,-1.0\n"
        "MC-EQ-003,0,-1,BRANCH,EIGENVALUE,2,-1.0,0.0\n",
        encoding="utf-8",
    )

    diagnostics = validate_equilibrium_artifacts(tmp_path, "MC-EQ-003")

    assert diagnostics["spectrum_rows"] == 3
    assert diagnostics["hopf_frequency"] == pytest.approx(1.0)
    assert diagnostics["first_lyapunov"] == pytest.approx(-0.3)


@pytest.fixture(scope="module")
def matcont_generated_dir(tmp_path_factory):
    matlab = Path("/home/ziaee/prog/Matlab/R2020a/bin/matlab")
    if not matlab.is_file():
        pytest.skip("MATLAB R2020a is unavailable")
    output = tmp_path_factory.mktemp("matcont-generated")
    matlab_dir = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "matlab"
    completed = subprocess.run(
        [str(matlab), "-batch", f"cd('{matlab_dir}'); run_supported('{output}')"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


@pytest.mark.slow
def test_matlab_equilibrium_exporters_preserve_spectra_and_normal_forms(matcont_generated_dir):
    cubic = validate_equilibrium_artifacts(matcont_generated_dir, "MC-EQ-001")
    vanderpol = validate_equilibrium_artifacts(matcont_generated_dir, "MC-EQ-002")
    adaptive = validate_equilibrium_artifacts(matcont_generated_dir, "MC-EQ-003")

    assert cubic["spectrum_rows"] > 0
    assert sorted(cubic["fold_coefficients"]) == pytest.approx([-1.0, 1.0], abs=5e-4)
    assert vanderpol["hopf_frequency"] == pytest.approx(1.0, abs=5e-4)
    assert vanderpol["first_lyapunov"] == pytest.approx(0.0, abs=1e-4)
    assert adaptive["hopf_frequency"] == pytest.approx(1.0, abs=5e-4)
    assert adaptive["first_lyapunov"] == pytest.approx(-0.3, abs=5e-4)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Known JaxCont torBPC LPC/PD event and Floquet comparison mismatch",
)
def test_torbpc_jaxcont_matches_all_matcont_diagnostics(torbpc_result):
    checks = torbpc_result.checks

    assert {
        "sweep": checks["jaxcont_sweep_completed"],
        "lpc_parameter": checks["jaxcont_lpc_parameter_error"] < 2e-3,
        "ns_parameter": checks["jaxcont_ns_parameter_error"] < 2e-3,
        "pd_parameter": checks["jaxcont_pd_parameter_error"] < 2e-3,
        "periods": checks["jaxcont_max_period_error"] < 5e-3,
        "extrema": checks["jaxcont_max_extrema_error"] < 5e-3,
        "multipliers": checks["jaxcont_critical_multipliers_match"],
        "events": checks["jaxcont_missing_event_types"] == [],
        "overall": checks["all_comparisons_pass"],
    } == {
        "sweep": True,
        "lpc_parameter": True,
        "ns_parameter": True,
        "pd_parameter": True,
        "periods": True,
        "extrema": True,
        "multipliers": True,
        "events": True,
        "overall": True,
    }


@pytest.fixture(scope="module")
def torbpc_result(matcont_generated_dir):
    return run_torbpc_cycle(matcont_generated_dir)


@pytest.mark.slow
def test_torbpc_known_disagreements_are_explicit_failures(torbpc_result):
    checks = torbpc_result.checks

    assert checks["jaxcont_pd_parameter_error"] < 2e-3
    assert checks["jaxcont_max_period_error"] < 5e-3
    assert checks["jaxcont_max_extrema_error"] < 5e-3
    assert not checks["jaxcont_critical_multipliers_match"]
    assert {"LPC", "PD"} <= set(checks["jaxcont_missing_event_types"])
    assert not checks["all_comparisons_pass"]


def test_cycle_reference_has_normalized_columns(tmp_path):
    """Renaming a portable branch column must be rejected at the reader boundary."""
    (tmp_path / "MC-LC-002_branch.csv").write_text(
        "case_id,point,parameter,period,residual_norm,state_0_min,state_0_max\n"
        "MC-LC-002,0,-0.5844928424,6.3,1e-8,-0.1,0.1\n",
        encoding="utf-8",
    )

    branch = np.genfromtxt(
        tmp_path / "MC-LC-002_branch.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    assert {"point", "parameter", "period", "residual_norm"} <= set(branch.dtype.names)


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


def test_registry_exposes_both_periodic_validation_cases():
    """Dropping either periodic entry would silently remove it from default runs."""
    periodic = {
        case["id"]: case
        for case in load_registry()["cases"]
        if "periodic-orbit" in case["features"]
    }

    assert set(periodic) == {"MC-LC-001", "MC-LC-002"}
    assert periodic["MC-LC-001"]["python"].endswith(":run_radial_cycle")
    assert periodic["MC-LC-002"]["python"].endswith(":run_torbpc_cycle")


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
    diagnostics = scaled_close(np.array([1.0, 2.0]), np.array([1.0, 2.001]), atol=0.002)

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


def _run_matcont_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "examples.MatCont.run_validation", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_offline_cli_validates_supported_reference_case():
    """A supported case with reviewed artifacts must execute, not stop at selection."""
    reference_dir = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "reference"

    completed = _run_matcont_cli(
        "--case", "MC-EQ-001", "--reference-dir", str(reference_dir)
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS MC-EQ-001" in completed.stdout


def test_unsupported_case_is_explicitly_reported():
    """Including an unsupported case must never imply JaxCont validated it."""
    completed = _run_matcont_cli(
        "--case", "US-BP-001", "--include-unsupported", "--dry-run"
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "UNSUPPORTED_BY_JAXCONT US-BP-001" in completed.stdout


def test_unsupported_registry_covers_every_deferred_feature_family():
    """Omitting a deferred family would make the capability matrix incomplete."""
    unsupported_features = {
        feature
        for case in load_registry()["cases"]
        if case["support"] == "unsupported"
        for feature in case["features"]
    }

    assert {
        "branch-switching",
        "two-parameter-fold-curve",
        "two-parameter-hopf-curve",
        "two-parameter-pd-curve",
        "two-parameter-lpc-curve",
        "two-parameter-ns-curve",
        "general-bvp-continuation",
        "homoclinic-continuation",
        "heteroclinic-continuation",
        "prc",
        "dprc",
    } <= unsupported_features


def _complete_metadata(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "provenance": "regenerated with the committed standalone producer",
        "source": "analytic fixture",
        "source_section": "validation test",
        "matlab_version": "9.8.0.1323502 (R2020a)",
        "matcont_version": "7.6",
        "jaxcont_version": "0.1.0",
        "python_version": "3.11.0",
        "jax_version": "0.4.0",
        "precision": "double",
        "mesh": {"ntst": 25, "ncol": 4},
        "solver_settings": {"MaxNumPoints": 80},
        "generated_utc": "2026-08-05T00:00:00Z",
        "equation_hash": "sha256:" + "1" * 64,
        "reviewed": True,
    }


def test_reference_metadata_requires_complete_reproducibility_provenance(tmp_path):
    """A reference without an equation hash cannot identify the equations it validates."""
    path = tmp_path / "MC-EQ-001_metadata.json"
    metadata = _complete_metadata("MC-EQ-001")
    del metadata["equation_hash"]
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="equation_hash"):
        validate_reference_metadata(path, "MC-EQ-001")

    metadata["equation_hash"] = "sha256:" + "1" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert validate_reference_metadata(path, "MC-EQ-001")["reviewed"] is True


def test_reference_metadata_rejects_malformed_sha256(tmp_path):
    """A prefix-only or non-hex equation hash must not identify a reviewed equation."""
    path = tmp_path / "MC-EQ-001_metadata.json"
    for malformed in ("sha256:", "sha256:" + "z" * 64, "sha256:" + "1" * 63):
        metadata = _complete_metadata("MC-EQ-001")
        metadata["equation_hash"] = malformed
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="equation_hash"):
            validate_reference_metadata(path, "MC-EQ-001")


def test_generated_metadata_is_enriched_with_runtime_and_equation_provenance(tmp_path):
    """Promotion must identify the exact equations and Python/JAX runtime used."""
    metadata_path = tmp_path / "MC-EQ-001_metadata.json"
    equation_path = tmp_path / "cubic_fold.m"
    equation_source = "function y = cubic_fold(x)\ny = x;\nend\n"
    equation_path.write_text(equation_source, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "case_id": "MC-EQ-001",
                "source": "analytic cubic fold",
                "matlab_version": "9.8.0 (R2020a)",
                "matcont_version": "7.6",
                "precision": "double",
                "mesh": {"kind": "equilibrium"},
                "solver_settings": {"MaxNumPoints": 500},
                "generated_utc": "2026-08-05T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    case = {
        "id": "MC-EQ-001",
        "manual_source": "MatCont manual section 3",
        "python": "examples.MatCont.python_cases.equilibrium:run_cubic_fold",
    }

    enriched = enrich_generated_metadata(metadata_path, case, equation_path)

    assert enriched["equation_hash"] == "sha256:" + hashlib.sha256(
        equation_source.encode()
    ).hexdigest()
    assert enriched["source_section"] == "MatCont manual section 3"
    assert enriched["python_version"]
    assert enriched["jax_version"]
    assert enriched["jaxcont_version"]
    assert enriched["reviewed"] is False


def test_reference_verification_is_tolerant_and_never_overwrites_reviewed_files(tmp_path):
    """Verification must accept roundoff, detect drift, and leave the oracle unchanged."""
    reviewed = tmp_path / "reviewed"
    generated = tmp_path / "generated"
    reviewed.mkdir()
    generated.mkdir()
    reference_csv = "case_id,point,parameter\nMC-EQ-001,0,0.6666666667\n"
    generated_csv = "case_id,point,parameter\nMC-EQ-001,0,0.6666667667\n"
    (reviewed / "branch.csv").write_text(reference_csv, encoding="utf-8")
    (generated / "branch.csv").write_text(generated_csv, encoding="utf-8")
    reviewed_metadata = _complete_metadata("MC-EQ-001")
    generated_metadata = dict(reviewed_metadata, generated_utc="2026-08-06T00:00:00Z")
    (reviewed / "metadata.json").write_text(json.dumps(reviewed_metadata), encoding="utf-8")
    (generated / "metadata.json").write_text(json.dumps(generated_metadata), encoding="utf-8")
    case = {
        "id": "MC-EQ-001",
        "references": ["branch.csv", "metadata.json"],
        "tolerances": {"parameter_atol": 5e-4},
    }

    before = (reviewed / "branch.csv").read_bytes()
    diagnostics = verify_case_references(case, generated, reviewed)

    assert diagnostics["max_numeric_error"] == pytest.approx(1e-7)
    assert (reviewed / "branch.csv").read_bytes() == before

    (generated / "branch.csv").write_text(
        "case_id,point,parameter\nMC-EQ-001,0,0.7\n", encoding="utf-8"
    )
    with pytest.raises(ValidationMismatch):
        verify_case_references(case, generated, reviewed)


def _write_topology_fixture(directory: Path, *, reordered: bool, midpoint: bool) -> None:
    branch_rows = [
        "MC-EQ-X,0,-1.0,0.0,1,0,-1.0",
        "MC-EQ-X,1,0.0,0.0,0,1,0.0",
        "MC-EQ-X,2,1.0,0.0,0,1,1.0",
    ]
    if midpoint:
        branch_rows.insert(2, "MC-EQ-X,8,0.5,0.0,0,1,0.5")
    if reordered:
        branch_rows = branch_rows[::-1]
    (directory / "branch.csv").write_text(
        "case_id,point,parameter,residual_norm,stable,unstable_dimension,state_0\n"
        + "\n".join(branch_rows)
        + "\n",
        encoding="utf-8",
    )
    event_rows = [
        "MC-EQ-X,0,LP,1,0.0,NaN,1.0,NaN",
        "MC-EQ-X,1,H,2,1.0,2.0,NaN,-0.3",
    ]
    if reordered:
        event_rows = event_rows[::-1]
    (directory / "events.csv").write_text(
        "case_id,event_index,event_type,point,parameter,frequency,fold_coefficient,first_lyapunov\n"
        + "\n".join(event_rows)
        + "\n",
        encoding="utf-8",
    )
    spectrum_rows = [
        "MC-EQ-X,0,-1,BRANCH,EIGENVALUE,0,-1.0,0.0",
        "MC-EQ-X,1,-1,BRANCH,EIGENVALUE,0,0.0,0.0",
        "MC-EQ-X,2,-1,BRANCH,EIGENVALUE,0,1.0,0.0",
        "MC-EQ-X,1,0,LP,EIGENVALUE,0,0.0,0.0",
        "MC-EQ-X,2,1,H,EIGENVALUE,0,0.0,2.0",
        "MC-EQ-X,2,1,H,EIGENVALUE,1,0.0,-2.0",
    ]
    if midpoint:
        spectrum_rows.insert(3, "MC-EQ-X,8,-1,BRANCH,EIGENVALUE,0,0.5,0.0")
    if reordered:
        spectrum_rows = spectrum_rows[::-1]
    (directory / "multipliers.csv").write_text(
        "case_id,point,event_index,event_type,spectrum_kind,multiplier_index,real,imag\n"
        + "\n".join(spectrum_rows)
        + "\n",
        encoding="utf-8",
    )
    metadata = _complete_metadata("MC-EQ-X")
    metadata["reviewed"] = not reordered
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_reference_verification_matches_topology_not_row_order_or_mesh(tmp_path):
    """Reordered events/spectra and an adaptive branch point are the same result."""
    reviewed = tmp_path / "reviewed"
    generated = tmp_path / "generated"
    reviewed.mkdir()
    generated.mkdir()
    _write_topology_fixture(reviewed, reordered=False, midpoint=False)
    _write_topology_fixture(generated, reordered=True, midpoint=True)
    case = {
        "id": "MC-EQ-X",
        "references": ["branch.csv", "events.csv", "multipliers.csv", "metadata.json"],
        "tolerances": {
            "parameter_atol": 1e-6,
            "state_atol": 1e-6,
            "frequency_atol": 1e-6,
            "fold_coefficient_atol": 1e-6,
            "first_lyapunov_atol": 1e-6,
            "spectrum_atol": 1e-6,
        },
    }

    diagnostics = verify_case_references(case, generated, reviewed)

    assert diagnostics["event_max_error"] == pytest.approx(0.0)
    assert diagnostics["spectrum_max_error"] == pytest.approx(0.0)

    event_path = generated / "events.csv"
    event_path.write_text(
        event_path.read_text(encoding="utf-8").replace(",1.0,NaN", ",1.1,NaN"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationMismatch):
        verify_case_references(case, generated, reviewed)


def test_folded_branch_comparison_preserves_duplicate_parameter_arms(tmp_path):
    """Sorting/deduplicating parameter values must not hide a wrong folded arm."""
    reviewed = tmp_path / "reviewed"
    generated = tmp_path / "generated"
    reviewed.mkdir()
    generated.mkdir()
    header = "case_id,point,parameter,period,residual_norm\n"
    (reviewed / "branch.csv").write_text(
        header
        + "MC-LC-X,0,0,1,0\n"
        + "MC-LC-X,1,1,2,0\n"
        + "MC-LC-X,2,0,10,0\n",
        encoding="utf-8",
    )
    (generated / "branch.csv").write_text(
        header
        + "MC-LC-X,0,0,1,0\n"
        + "MC-LC-X,1,1,2,0\n"
        + "MC-LC-X,2,0,999,0\n",
        encoding="utf-8",
    )
    reviewed_metadata = _complete_metadata("MC-LC-X")
    generated_metadata = dict(reviewed_metadata, reviewed=False)
    (reviewed / "metadata.json").write_text(json.dumps(reviewed_metadata), encoding="utf-8")
    (generated / "metadata.json").write_text(json.dumps(generated_metadata), encoding="utf-8")
    case = {
        "id": "MC-LC-X",
        "references": ["branch.csv", "metadata.json"],
        "tolerances": {"parameter_atol": 1e-9, "period_atol": 1e-6},
    }

    with pytest.raises(ValidationMismatch):
        verify_case_references(case, generated, reviewed)


def test_folded_branch_comparison_accepts_reversed_adaptive_segments(tmp_path):
    """Reversing branch direction and adding points must preserve both folded arms."""
    reviewed = tmp_path / "reviewed"
    generated = tmp_path / "generated"
    reviewed.mkdir()
    generated.mkdir()
    header = "case_id,point,parameter,period,residual_norm\n"
    (reviewed / "branch.csv").write_text(
        header
        + "MC-LC-X,0,0,1,0\n"
        + "MC-LC-X,1,1,2,0\n"
        + "MC-LC-X,2,0,10,0\n",
        encoding="utf-8",
    )
    (generated / "branch.csv").write_text(
        header
        + "MC-LC-X,0,0,10,0\n"
        + "MC-LC-X,1,0.5,6,0\n"
        + "MC-LC-X,2,1,2,0\n"
        + "MC-LC-X,3,0.5,1.5,0\n"
        + "MC-LC-X,4,0,1,0\n",
        encoding="utf-8",
    )
    reviewed_metadata = _complete_metadata("MC-LC-X")
    generated_metadata = dict(reviewed_metadata, reviewed=False)
    (reviewed / "metadata.json").write_text(json.dumps(reviewed_metadata), encoding="utf-8")
    (generated / "metadata.json").write_text(json.dumps(generated_metadata), encoding="utf-8")
    case = {
        "id": "MC-LC-X",
        "references": ["branch.csv", "metadata.json"],
        "tolerances": {"parameter_atol": 1e-9, "period_atol": 1e-9},
    }

    diagnostics = verify_case_references(case, generated, reviewed)

    assert diagnostics["branch_max_error"] == pytest.approx(0.0)


def test_relative_generated_directory_resolves_before_matlab_changes_directory(
    tmp_path, monkeypatch
):
    """A relative artifact path must remain relative to the caller, not matlab/."""
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--generated-dir", "relative-output"])

    resolve_runtime_paths(args)

    assert args.generated_dir == tmp_path / "relative-output"
    assert args.reference_dir.is_absolute()


def test_explicit_unsupported_case_requires_opt_in():
    """Silently selecting nothing hides that the requested capability is unsupported."""
    completed = _run_matcont_cli("--case", "US-BP-001", "--dry-run")

    assert completed.returncode != 0
    assert "requires --include-unsupported" in completed.stderr


def test_unsupported_template_is_never_reported_as_executed():
    """A setup template must not become a false MatCont execution success."""
    completed = _run_matcont_cli(
        "--case",
        "US-BVP-001",
        "--include-unsupported",
        "--matlab-bin",
        "/definitely/not/matlab",
    )

    assert completed.returncode != 0
    assert "NON_EXECUTABLE_TEMPLATE" in completed.stderr
    assert "wrapper completed" not in completed.stdout


@pytest.mark.slow
def test_executable_unsupported_wrapper_runs_only_after_opt_in():
    """Executable MatCont-only cases must run and remain labeled unsupported."""
    matlab = Path("/home/ziaee/prog/Matlab/R2020a/bin/matlab")
    if not matlab.is_file():
        pytest.skip("MATLAB R2020a is unavailable")

    completed = _run_matcont_cli(
        "--case", "US-BP-001", "--include-unsupported", "--matlab-bin", str(matlab)
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "UNSUPPORTED_BY_JAXCONT US-BP-001: MatCont wrapper completed" in completed.stdout


@pytest.mark.slow
def test_offline_cli_rejects_corrupted_matcont_fold_event(tmp_path):
    """Analytic fold checks must not let a corrupted committed MatCont event pass."""
    reference = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "reference"
    mutated = tmp_path / "reference"
    shutil.copytree(reference, mutated)
    event_path = mutated / "MC-EQ-001_events.csv"
    event_path.write_text(
        event_path.read_text(encoding="utf-8").replace(",0.666666666666084,", ",0.5,"),
        encoding="utf-8",
    )

    completed = _run_matcont_cli("--case", "MC-EQ-001", "--reference-dir", str(mutated))

    assert completed.returncode != 0
    assert "FAIL MC-EQ-001" in completed.stderr


@pytest.mark.slow
def test_offline_cli_rejects_corrupted_radial_matcont_branch(tmp_path):
    """Exact radial checks must not let a corrupted committed MatCont period pass."""
    reference = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "reference"
    mutated = tmp_path / "reference"
    shutil.copytree(reference, mutated)
    branch_path = mutated / "MC-LC-001_branch.csv"
    rows = branch_path.read_text(encoding="utf-8").splitlines()
    fields = rows[1].split(",")
    fields[3] = "7.0"
    rows[1] = ",".join(fields)
    branch_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    completed = _run_matcont_cli("--case", "MC-LC-001", "--reference-dir", str(mutated))

    assert completed.returncode != 0
    assert "FAIL MC-LC-001" in completed.stderr


def test_offline_cli_rejects_corrupted_radial_branch_schema(tmp_path):
    """A radial reference without period data must fail before numerical execution."""
    reference = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "reference"
    mutated = tmp_path / "reference"
    shutil.copytree(reference, mutated)
    branch_path = mutated / "MC-LC-001_branch.csv"
    branch_path.write_text(
        branch_path.read_text(encoding="utf-8").replace("period", "missing_period", 1),
        encoding="utf-8",
    )

    completed = _run_matcont_cli("--case", "MC-LC-001", "--reference-dir", str(mutated))

    assert completed.returncode != 0
    assert "missing columns" in completed.stderr
