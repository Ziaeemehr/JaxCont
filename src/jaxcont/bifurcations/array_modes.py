"""Trace-safe predictive spectral assignment and owned-zero crossings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax


def _assignment(cost: Array) -> Array:
    """Square Hungarian assignment with bounded paths and lowest-column ties."""
    n = cost.shape[0]
    if n == 0:
        return jnp.empty((0,), dtype=jnp.int32)
    cost = lax.stop_gradient(cost)
    limit = jnp.finfo(cost.dtype).max / 4
    cost = jnp.where(jnp.isfinite(cost), cost, limit)
    zeros = jnp.zeros(n + 1, dtype=cost.dtype)
    indices = jnp.zeros(n + 1, dtype=jnp.int32)

    def insert(row, state):
        u, v, matching = state
        matching = matching.at[0].set(row + 1)
        # Column zero roots the alternating tree; real columns are one-based.
        initial = (u, v, jnp.full((n + 1,), jnp.inf, cost.dtype),
                   jnp.zeros(n + 1, bool), indices, jnp.int32(0), jnp.bool_(True))

        def search(_, search_state):
            def advance(s):
                u, v, minimum, used, previous, column, _ = s
                used = used.at[column].set(True)
                current_row = matching[column]
                reduced = cost[current_row - 1] - u[current_row] - v[1:]
                improve = (~used[1:]) & (reduced < minimum[1:])
                minimum = minimum.at[1:].set(jnp.where(improve, reduced, minimum[1:]))
                previous = previous.at[1:].set(jnp.where(improve, column, previous[1:]))
                candidates = jnp.where(used[1:], jnp.inf, minimum[1:])
                next_column = jnp.argmin(candidates).astype(jnp.int32) + 1
                delta = candidates[next_column - 1]
                u = u.at[matching].add(jnp.where(used, delta, 0))
                v = v - jnp.where(used, delta, 0)
                minimum = minimum - jnp.where(used, 0, delta)
                return u, v, minimum, used, previous, next_column, matching[next_column] != 0
            return lax.cond(search_state[-1], advance, lambda s: s, search_state)

        u, v, _, _, previous, column, _ = lax.fori_loop(0, n, search, initial)

        def augment(_, state):
            matching, column = state
            parent = previous[column]
            matching = lax.cond(column != 0,
                                lambda p: p.at[column].set(p[parent]),
                                lambda p: p, matching)
            return matching, parent

        matching, _ = lax.fori_loop(0, n, augment, (matching, column))
        return u, v, matching

    _, _, matching = lax.fori_loop(0, n, insert, (zeros, zeros, indices))
    return indices[1:].at[matching[1:] - 1].set(jnp.arange(n, dtype=jnp.int32))


def track_modes_array(spectra: Array, valid: Array) -> tuple[Array, Array]:
    """Track ``(B,K)`` spectra using the previous two valid samples.

    Invalid rows retain raw order and do not update the predictor. The first
    valid row retains raw order; permutations index raw columns at each row.
    Assignment decisions have stopped gradients.
    """
    spectra = jnp.asarray(spectra, dtype=jnp.result_type(spectra, jnp.complex64))
    valid = jnp.asarray(valid, dtype=bool)
    k = spectra.shape[1]
    identity = jnp.arange(k, dtype=jnp.int32)
    if spectra.shape[0] == 0 or k == 0:
        return spectra, jnp.broadcast_to(identity, spectra.shape)

    def step(carry, sample):
        older, previous, count = carry
        raw, is_valid = sample
        prediction = jnp.where(count < 2, previous, 2 * previous - older)
        permutation = lax.cond(
            is_valid & (count > 0),
            lambda _: _assignment(jnp.abs(prediction[:, None] - raw[None, :])),
            lambda _: identity, operand=None)
        tracked = raw[permutation]
        carry = (jnp.where(is_valid, previous, older),
                 jnp.where(is_valid, tracked, previous),
                 jnp.minimum(count + is_valid.astype(jnp.int32), 2))
        return carry, (tracked, lax.stop_gradient(permutation))

    initial = (jnp.zeros(k, spectra.dtype), jnp.zeros(k, spectra.dtype), jnp.int32(0))
    _, result = lax.scan(step, initial, (spectra, valid))
    return result


def crossing_mask(values: Array, eligible: Array, valid: Array) -> Array:
    """Return ``(B-1,K)`` brackets, assigning sampled zeros to one owner.

    A preceding eligible bracket owns a zero before a following bracket can.
    Nonfinite endpoints, padding, and identically zero pairs never bracket.
    """
    values = jnp.asarray(values)
    eligible = jnp.asarray(eligible, dtype=bool) & jnp.asarray(valid, bool)[:, None]
    a, b = values[:-1], values[1:]
    hit = ((jnp.sign(a) != jnp.sign(b)) & ((a != 0) | (b != 0))
           & jnp.isfinite(a) & jnp.isfinite(b) & eligible[:-1] & eligible[1:])
    preceding = jnp.concatenate((jnp.zeros_like(hit[:1]), hit[:-1]), axis=0)
    return lax.stop_gradient(hit & ~((a == 0) & preceding))
