"""
Proves the eqx.Module static/traced field-split pattern that v0.2's real
periodic-orbit types (Collocation predictor, mesh/ntst/ncol) will need,
ahead of those types existing. See
docs/superpowers/specs/2026-07-22-equinox-adoption-design.md.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxcont.core._periodic_eqx_scaffold import CollocationMeshScaffold


def _make_scaffold(ntst=4, ncol=3):
    return CollocationMeshScaffold(
        ntst=ntst, ncol=ncol, mesh=jnp.linspace(0.0, 1.0, 5)
    )


def _eval(m: CollocationMeshScaffold, period):
    # Uses ntst/ncol in a *shape* position -- only legal if they are static,
    # not traced. A wrong static/traced split raises a
    # TracerArrayConversionError/ConcretizationTypeError here.
    pad = jnp.zeros((m.ntst, m.ncol))
    return m.mesh * period + pad.sum()


def test_scaffold_runs_eagerly():
    m = _make_scaffold()
    result = _eval(m, jnp.array(2.0))
    expected = jnp.linspace(0.0, 1.0, 5) * 2.0
    assert jnp.allclose(result, expected)


def test_scaffold_jits():
    m = _make_scaffold()
    result = jax.jit(_eval)(m, jnp.array(2.0))
    expected = jnp.linspace(0.0, 1.0, 5) * 2.0
    assert jnp.allclose(result, expected)


def test_scaffold_vmaps_over_traced_input_with_static_config_fixed():
    m = _make_scaffold()
    periods = jnp.array([1.0, 2.0, 3.0])

    batch = jax.vmap(lambda p: _eval(m, p))(periods)

    expected = jnp.stack([jnp.linspace(0.0, 1.0, 5) * p for p in [1.0, 2.0, 3.0]])
    assert batch.shape == (3, 5)
    assert jnp.allclose(batch, expected)


def test_ntst_ncol_are_static_python_ints_not_traced():
    m = _make_scaffold(ntst=4, ncol=3)
    assert isinstance(m.ntst, int)
    assert isinstance(m.ncol, int)

    # Changing a static field changes the pytree's structure (it's part of
    # the jit cache key), unlike a traced field -- confirm this holds.
    m2 = _make_scaffold(ntst=6, ncol=3)
    leaves1, treedef1 = jax.tree_util.tree_flatten(m)
    leaves2, treedef2 = jax.tree_util.tree_flatten(m2)
    assert treedef1 != treedef2
