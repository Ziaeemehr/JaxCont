"""Dtype-derived numerical tolerances and the float64-only precision policy.

JaxCont computes in float64 only (see
docs/superpowers/specs/2026-10-29-precision-policy-design.md): importing the
package turns on ``jax_enable_x64`` unless ``JAXCONT_NO_AUTO_X64=1``, and the
entry points convert floating inputs to float64.
"""

import os
import warnings

import jax
import jax.numpy as jnp

OPT_OUT = "JAXCONT_NO_AUTO_X64"


class PrecisionWarning(UserWarning):
    """JaxCont changed JAX's global precision setting to float64."""


def default_tol(dtype) -> float:
    """Return the default tolerance for ``dtype``: 1e-9 for float64, else 1e-5."""
    return 1e-9 if jnp.dtype(dtype) == jnp.float64 else 1e-5


def is_x64() -> bool:
    """Return True when JAX 64-bit mode is enabled."""
    return bool(jax.config.jax_enable_x64)


def ensure_x64() -> None:
    """Turn on ``jax_enable_x64`` (once, with a warning) unless opted out.

    The setting is global: it also makes the caller's other JAX code default
    to 64-bit. ``JAXCONT_NO_AUTO_X64=1`` leaves it alone; JaxCont entry points
    then refuse to run.
    """
    if is_x64() or os.environ.get(OPT_OUT) == "1":
        return
    jax.config.update("jax_enable_x64", True)
    warnings.warn(
        "JaxCont computes in float64 only and has turned on jax_enable_x64 for "
        "this process (a global JAX setting). Set it yourself before importing "
        f"jaxcont to silence this, or set {OPT_OUT}=1 to keep float32; JaxCont "
        "then refuses to run.",
        PrecisionWarning,
        stacklevel=3,
    )


def as_float64(tree, *, scalars=True):
    """Floating leaves of ``tree`` as float64 (complex as complex128).

    Integer, boolean and non-array leaves are left alone; with
    ``scalars=False`` (user ``args``) Python floats are left alone too, so a
    caller's own Python logic on them is unchanged. Raises ``RuntimeError``
    when float64 is unavailable (``JAXCONT_NO_AUTO_X64=1``).
    """
    if not is_x64():
        raise RuntimeError(
            "JaxCont requires float64, but jax_enable_x64 is off "
            f"({OPT_OUT}=1 kept it off). Enable it with "
            "jax.config.update('jax_enable_x64', True) before creating arrays."
        )

    def convert(leaf):
        if isinstance(leaf, (bool, int)) or leaf is None:
            return leaf
        if isinstance(leaf, (float, complex)):
            if not scalars:
                return leaf
            return jnp.asarray(leaf, jnp.complex128 if isinstance(leaf, complex) else jnp.float64)
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or dtype in (jnp.float64, jnp.complex128):
            return leaf  # unchanged, weak type included (no recompiles)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            return jnp.asarray(leaf, jnp.complex128)
        if jnp.issubdtype(dtype, jnp.floating):
            return jnp.asarray(leaf, jnp.float64)
        return leaf

    return jax.tree_util.tree_map(convert, tree)


__all__ = ["OPT_OUT", "PrecisionWarning", "as_float64", "default_tol", "ensure_x64", "is_x64"]
