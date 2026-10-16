"""Dtype-derived numerical tolerances."""

import jax
import jax.numpy as jnp


def default_tol(dtype) -> float:
    """Return the default tolerance for ``dtype``: 1e-9 for float64, else 1e-5."""
    return 1e-9 if jnp.dtype(dtype) == jnp.float64 else 1e-5


def is_x64() -> bool:
    """Return True when JAX 64-bit mode is enabled."""
    return bool(jax.config.jax_enable_x64)
