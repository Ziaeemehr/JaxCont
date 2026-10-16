"""Dtype-derived numerical tolerances."""

import jax
import jax.numpy as jnp


def default_tol(dtype) -> float:
    """Return the default tolerance for ``dtype``: 1e-9 for float64, else 1e-5."""
    return 1e-9 if jnp.dtype(dtype) == jnp.float64 else 1e-5


def collocation_tol(dtype) -> float:
    """Return the residual tolerance for collocation systems.

    Collocation residuals of stiff orbits stay near ``1e-6`` even in float64,
    so this never drops below the float32 default.
    """
    return max(default_tol(dtype), default_tol(jnp.float32))


def is_x64() -> bool:
    """Return True when JAX 64-bit mode is enabled."""
    return bool(jax.config.jax_enable_x64)
