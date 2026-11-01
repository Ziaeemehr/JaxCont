"""The published endpoint must select valid points under transformations."""
import ast
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jaxcont as jc
import numpy as np

from tests.test_check_docs import check_docs


def _documented_endpoint():
    page = Path(__file__).resolve().parents[1] / "docs/source/migration.rst"
    code = next(code for _, code, _ in check_docs.python_blocks(page) if "def endpoint(a, loop):" in code)
    function = next(node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef))
    namespace = dict(jc=jc, jnp=jnp, jax=jax)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(page), "exec"), namespace)
    return namespace["endpoint"]


def test_migration_endpoint_and_gradient_agree_under_jit_and_vmap():
    endpoint = _documented_endpoint()
    run = lambda a: endpoint(a, "scan")
    value, gradient = jax.value_and_grad(run)(0.1)
    assert float(value) > 1.0
    assert float(gradient) > 0.1
    tolerance = 256 * np.finfo(np.asarray(value).dtype).eps
    compiled = jax.jit(jax.value_and_grad(run))(0.1)
    np.testing.assert_allclose(compiled, (value, gradient), rtol=tolerance, atol=0)
    inputs = jnp.array([0.1, 0.2])
    serial = [jax.value_and_grad(run)(a) for a in inputs]
    batched = jax.vmap(jax.value_and_grad(run))(inputs)
    np.testing.assert_allclose(batched[0], [pair[0] for pair in serial], rtol=tolerance, atol=0)
    np.testing.assert_allclose(batched[1], [pair[1] for pair in serial], rtol=tolerance, atol=0)


def test_migration_endpoint_returns_nan_when_no_point_is_valid(monkeypatch):
    monkeypatch.setattr(jc, "continuation", lambda *a, **kw: SimpleNamespace(
        branch=SimpleNamespace(states=jnp.zeros((41, 1)), n_valid=jnp.array(0))))
    assert np.isnan(_documented_endpoint()(0.1, "scan"))
