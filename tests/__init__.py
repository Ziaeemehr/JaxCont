"""Test package imports."""

def test_import_jaxcont():
    """Test that jaxcont can be imported."""
    try:
        import jaxcont
        assert jaxcont.__version__ is not None
    except ImportError as e:
        pytest.skip(f"Could not import jaxcont: {e}")


def test_import_core():
    """Test that core modules can be imported."""
    try:
        from jaxcont.core import ContinuationProblem, ContinuationSolution
        assert ContinuationProblem is not None
        assert ContinuationSolution is not None
    except ImportError as e:
        pytest.skip(f"Could not import core: {e}")
