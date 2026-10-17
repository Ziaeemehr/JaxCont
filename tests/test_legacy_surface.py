"""The removed OO API must not shadow the functional public API."""

import importlib
import importlib.util

import pytest


@pytest.mark.parametrize("module,names", [
    ("jaxcont", ("NewtonSolver", "Corrector", "EquilibriumProblem", "Config")),
    ("jaxcont.solvers", ("NewtonSolver", "Corrector")),
    ("jaxcont.problems", ("EquilibriumProblem", "BoundaryValueProblem")),
    ("jaxcont.utils", ("Config",)),
    ("jaxcont.utils.config", ("Config", "get_config", "set_config", "reset_config")),
])
def test_removed_exports(module, names):
    namespace = importlib.import_module(module)
    for name in names:
        assert not hasattr(namespace, name), f"{module}.{name} remains public"


@pytest.mark.parametrize("module", [
    "jaxcont.solvers.newton", "jaxcont.solvers.corrector",
    "jaxcont.problems.equilibrium", "jaxcont.problems.bvp",
])
def test_removed_modules(module):
    assert importlib.util.find_spec(module) is None


def test_installation_diagnostics_and_solution_remain_available():
    from jaxcont import ContinuationSolution
    from jaxcont.utils.config import test_package_imports as check_imports

    assert ContinuationSolution is not None
    results = check_imports()
    assert results
    assert all(result["success"] for result in results.values()), results
