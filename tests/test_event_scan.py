"""Array-only mode assignment and crossing ownership."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.bifurcations.array_modes import crossing_mask, track_modes_array
from jaxcont.bifurcations.modes import track_modes


@pytest.mark.parametrize("compiled", [False, True])
def test_array_assignment_matches_host_predictor(compiled):
    run = jax.jit(track_modes_array) if compiled else track_modes_array
    p = jnp.linspace(-1, 1, 21)
    modes = jnp.stack((p, -p, .2 + .1*p + 2j, -.2 + .1*p + 3j), axis=1)
    rng = np.random.default_rng(61)
    raw = jnp.asarray(np.stack([row[rng.permutation(4)] for row in np.asarray(modes)]))
    tracked, permutation = run(raw, jnp.ones(21, dtype=bool))
    np.testing.assert_allclose(tracked, track_modes(np.asarray(raw)))
    np.testing.assert_array_equal(tracked, jnp.take_along_axis(raw, permutation, axis=1))
    assert jnp.issubdtype(permutation.dtype, jnp.integer)
    tied, identity = run(jnp.zeros((3, 4), dtype=modes.dtype), jnp.ones(3, bool))
    np.testing.assert_array_equal(identity, np.tile(np.arange(4), (3, 1)))
    np.testing.assert_array_equal(tied, 0)
    # Invalid rows cannot perturb the two-valid-sample predictor.
    padded = jnp.concatenate((raw[:8], jnp.full((2, 4), jnp.nan), raw[8:]))
    valid = jnp.concatenate((jnp.ones(8, bool), jnp.zeros(2, bool), jnp.ones(13, bool)))
    result, _ = run(padded, valid)
    np.testing.assert_allclose(result[valid], tracked)
    # Batched random branches exercise augmenting paths beyond greedy matching.
    batch = jnp.asarray(rng.normal(size=(6, 5, 4)) + 1j*rng.normal(size=(6, 5, 4)))
    batched, _ = jax.jit(jax.vmap(track_modes_array, in_axes=(0, None)))(
        batch, jnp.ones(5, bool))
    for raw_branch, tracked_branch in zip(np.asarray(batch), np.asarray(batched)):
        np.testing.assert_allclose(tracked_branch, track_modes(raw_branch))
    for size in (0, 1):
        result, perm = run(raw[:size], jnp.ones(size, bool))
        assert result.shape == perm.shape == (size, 4)
    nonfinite = jnp.array([[1, jnp.inf], [jnp.nan, 2]], dtype=modes.dtype)
    result, perm = run(nonfinite, jnp.ones(2, bool))
    np.testing.assert_allclose(result, track_modes(np.asarray(nonfinite)), equal_nan=True)


@pytest.mark.parametrize("compiled", [False, True])
def test_crossing_masks_zeros_padding_nonfinite(compiled):
    run = jax.jit(crossing_mask) if compiled else crossing_mask
    cases = [
        ([-1, 0, 1, 0, np.nan], [1, 1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 0, 0], [1, 1, 1], [0, 0]),
        ([0, 1], [1, 1], [1]),
        ([-1, 1], [1, 1], [1]),
        ([np.inf, -1], [1, 1], [0]),
        ([1], [1], []),
        ([], [], []),
    ]
    for values, valid, expected in cases:
        values = jnp.asarray(values, dtype=float)[:, None]
        mask = run(values, jnp.ones(values.shape, bool), jnp.array(valid, bool))
        np.testing.assert_array_equal(mask, np.asarray(expected, bool).reshape(-1, 1))
    # An ineligible predecessor cannot own the zero; the following bracket can.
    values = jnp.array([[-1., 1.], [0., 0.], [1., -1.]])
    eligible = jnp.array([[False, True], [True, True], [True, True]])
    np.testing.assert_array_equal(run(values, eligible, jnp.ones(3, bool)),
                                  [[False, True], [True, False]])
