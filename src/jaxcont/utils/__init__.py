"""Utility functions and helpers."""

from jaxcont.utils.config import (
    Config,
    print_jax_cuda_info,
    print_package_import_test,
    run_installation_tests,
    test_jax_cuda,
    test_package_imports,
)

__all__ = [
    "Config",
    "plot_bifurcation_diagram",
    "plot_continuation",
    "print_jax_cuda_info",
    "print_package_import_test",
    "run_installation_tests",
    "test_jax_cuda",
    "test_package_imports",
]


def __getattr__(name):
    # viz imports the engine, which imports utils.precision; resolving the
    # plotting names lazily keeps `jaxcont.utils.precision` import-cycle free.
    if name in ("plot_bifurcation_diagram", "plot_continuation"):
        from jaxcont import viz

        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
