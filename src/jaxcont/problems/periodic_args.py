"""The ``args`` pytree of a periodic-orbit :class:`~jaxcont.api.BifProblem`."""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
from jax import Array

from jaxcont.core.collocation import Collocation

PyTree = Any


class PeriodicArgs(eqx.Module):
    """Phase-condition reference data, the user's own ``args`` and the static
    description of a periodic collocation problem.

    ``u_ref_coll`` / ``uref_prime_coll`` are the phase-condition reference orbit
    and its time derivative at the collocation points; ``user_args`` is the
    pytree handed to the right-hand side. These three are dynamic leaves.
    ``raw_f`` and ``mesh`` are static (part of the treedef, hence of the jit
    cache key). ``raw_f`` is always called as ``raw_f(u, p, user_args)``.
    """

    u_ref_coll: Array
    uref_prime_coll: Array
    user_args: PyTree
    raw_f: Callable[[Array, Array, PyTree], Array] = eqx.field(static=True)
    mesh: Collocation = eqx.field(static=True)
