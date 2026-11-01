"""Test configuration."""

import os

# JaxCont is float64 only; select it before jax (or jaxcont) is imported
# anywhere, so importing jaxcont does not have to warn.
import jax

jax.config.update("jax_enable_x64", True)


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )

    config.addinivalue_line("markers", "matlab: opt-in tests requiring MATLAB and MatCont")

    config.addinivalue_line("markers", "julia: opt-in tests requiring pinned Julia/BifurcationKit")

