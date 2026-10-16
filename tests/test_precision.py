"""Tests for the precision helpers and dtype propagation."""

import jax.numpy as jnp

from jaxcont.utils.precision import default_tol, is_x64


def test_default_tol_values():
    assert default_tol(jnp.float64) == 1e-9
    assert default_tol(jnp.float32) == 1e-5
    assert default_tol(jnp.float16) == 1e-5


def test_is_x64_matches_default_float():
    assert is_x64() == (jnp.zeros(()).dtype == jnp.float64)


def test_scan_float32_inputs_under_x64():
    import jax
    from jaxcont.core.scan_continuation import pseudo_arclength_scan

    def f(u, p):
        return p * u - u**3

    f32 = jnp.float32
    with jax.enable_x64(True):
        res = pseudo_arclength_scan(
            f, jnp.array([0.1], f32), jnp.array(0.5, f32), jnp.array(1.5, f32),
            jnp.array(0.05, f32), jnp.array(1e-5, f32), jnp.array(0.2, f32),
            jnp.array(1e-6, f32), 20, jnp.array(20),
        )
    assert res.states.dtype == jnp.float32
    assert res.params.dtype == jnp.float32


def test_newton_tol_none_resolves_by_dtype():
    import jax
    import jaxcont as jc

    assert jc.ContinuationPar().newton_tol is None

    def f(u, p, args):
        return u**2 + p

    problem = jc.bif_problem(f, u0=jnp.array([1.0]), p0=jnp.array(-1.0))
    result = jc.continuation(
        problem, jc.PseudoArclength(), p_span=(-1.0, -0.2),
        settings=jc.ContinuationPar(ds=0.05, max_steps=40, compute_stability=False),
    )
    assert result.branch.states.shape[0] > 5
    resid = jnp.abs(result.branch.states[:, 0] ** 2 + result.branch.params)
    if jax.config.jax_enable_x64:
        assert float(jnp.max(resid)) < 1e-9
    else:
        assert float(jnp.max(resid)) < 1e-4


def test_solver_tol_none_resolves_by_dtype():
    from jaxcont.solvers.implicit import differentiable_root_checked

    x0 = jnp.array([1.5])
    x, ok = differentiable_root_checked(lambda x, th: x**2 - th, x0, jnp.asarray(2.0, x0.dtype))
    assert bool(ok)
    assert float(jnp.abs(x[0] ** 2 - 2.0)) < default_tol(x0.dtype)
