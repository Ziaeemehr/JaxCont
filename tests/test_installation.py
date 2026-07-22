"""Tests for the installation diagnostics."""

from jaxcont.utils.config import test_package_imports as check_package_imports


def test_package_import_diagnostic_uses_current_package_tree():
    results = check_package_imports()

    assert "jaxcont.core.scan_continuation" in results
    assert "jaxcont.core.natural_continuation" not in results
    assert "jaxcont.core.predictor_corrector" not in results
    assert "jaxcont.core.pseudo_arclength" not in results

    failures = {
        module_name: result["error"]
        for module_name, result in results.items()
        if not result["success"]
    }
    assert failures == {}
