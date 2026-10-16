"""Test configuration."""

import os

# Precision must be selected before jax (or jaxcont) is imported anywhere.
if os.environ.get("JAXCONT_TEST_X64", "0") == "1":
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
