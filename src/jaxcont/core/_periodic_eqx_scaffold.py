"""
Throwaway scaffold proving the `eqx.Module` static/traced field split works
end-to-end (jit + vmap) for the v0.2 periodic-orbit types.

This is NOT a real predictor and is not exported from `jaxcont.__init__`.
Delete and replace with a real `Collocation` predictor once periodic-orbit
continuation is actually built. See
docs/superpowers/specs/2026-07-22-equinox-adoption-design.md for the
rationale and docs/superpowers/plans/2026-07-22-equinox-adoption.md for how
this was built.
"""

import equinox as eqx
from jax import Array


class CollocationMeshScaffold(eqx.Module):
    """Static mesh-size config (ntst/ncol) + a traced mesh-point array."""

    ntst: int = eqx.field(static=True)
    ncol: int = eqx.field(static=True)
    mesh: Array
