"""Dtype-derived numerical tolerances."""

import jax
import jax.numpy as jnp


def default_tol(dtype) -> float:
    """Return the default tolerance for ``dtype``: 1e-9 for float64, else 1e-5."""
    return 1e-9 if jnp.dtype(dtype) == jnp.float64 else 1e-5


def collocation_tol(dtype) -> float:
    """Return the Newton residual tolerance for collocation systems.

    Not a float32 artifact: the float64 residual of a well-conditioned orbit
    reaches ~1e-15. Collocation Jacobians of small orbits near a Hopf point
    are ill-conditioned (cond ~1e8 measured on the torBPC seed), and there
    Newton stalls near 1e-7 in float64, so the tolerance never drops below
    the float32 default.
    """
    return max(default_tol(dtype), default_tol(jnp.float32))


def is_x64() -> bool:
    """Return True when JAX 64-bit mode is enabled."""
    return bool(jax.config.jax_enable_x64)
