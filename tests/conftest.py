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

    config.addinivalue_line("markers", "matlab: opt-in tests requiring MATLAB and MatCont")

    config.addinivalue_line("markers", "julia: opt-in tests requiring pinned Julia/BifurcationKit")

    config.addinivalue_line(
        "markers",
        "float64_only: validated in float64 only (precision policy); skipped in float32 runs",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``float64_only`` tests when the run is float32."""
    import jax
    import pytest

    if jax.config.x64_enabled:
        return
    skip = pytest.mark.skip(reason="float64_only: validation runs in float64 (precision policy)")
    for item in items:
        if "float64_only" in item.keywords:
            item.add_marker(skip)
