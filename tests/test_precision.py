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
