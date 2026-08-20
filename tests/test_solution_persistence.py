"""
Tests for ContinuationSolution.save()/.load() (2026-08-19 review finding #7:
persistence was broken for the common case of optional fields set to None,
and unsafe by defaulting to allow_pickle=True).
"""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc


def _sample_solution(*, with_optional_fields: bool) -> jc.ContinuationSolution:
    states = jnp.array([[0.0], [0.1], [0.2]])
    parameters = jnp.array([0.0, 0.5, 1.0])
    if not with_optional_fields:
        return jc.ContinuationSolution(states=states, parameters=parameters)
    return jc.ContinuationSolution(
        states=states,
        parameters=parameters,
        eigenvalues=jnp.array([[-1.0], [-0.5], [0.1]]),
        stability=jnp.array([True, True, False]),
        tangent_vectors=jnp.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
        bifurcations=[
            {
                "type": "fold",
                "parameter": 0.9,
                "state": jnp.array([0.19]),
                "index": 2,
                "null_vector": jnp.array([1.0, 0.0]),
            }
        ],
        convergence_info=[
            {"step": 0, "converged": True, "newton_iters": 3, "ds": 0.1},
            {"step": 1, "converged": True, "newton_iters": 2, "ds": 0.1},
        ],
        state_names=("x",),
        param_name="p",
    )


def test_round_trip_preserves_all_fields(tmp_path):
    sol = _sample_solution(with_optional_fields=True)
    path = tmp_path / "sol.npz"
    sol.save(str(path))
    loaded = jc.ContinuationSolution.load(str(path))

    assert jnp.allclose(loaded.states, sol.states)
    assert jnp.allclose(loaded.parameters, sol.parameters)
    assert jnp.allclose(loaded.eigenvalues, sol.eigenvalues)
    assert jnp.array_equal(loaded.stability, sol.stability)
    assert jnp.allclose(loaded.tangent_vectors, sol.tangent_vectors)
    assert loaded.state_names == sol.state_names
    assert loaded.param_name == sol.param_name
    assert loaded.convergence_info == sol.convergence_info

    assert len(loaded.bifurcations) == 1
    bif = loaded.bifurcations[0]
    assert bif["type"] == "fold"
    assert bif["parameter"] == pytest.approx(0.9)
    assert bif["index"] == 2
    assert bif["state"] == pytest.approx([0.19])
    assert bif["null_vector"] == pytest.approx([1.0, 0.0])


def test_round_trip_with_none_optional_fields_does_not_raise(tmp_path):
    """Direct regression test for the reviewed TypeError: saving/loading a
    solution with eigenvalues=None/stability=None/tangent_vectors=None must
    round-trip those fields as None, not crash."""
    sol = _sample_solution(with_optional_fields=False)
    assert sol.eigenvalues is None
    assert sol.stability is None
    assert sol.tangent_vectors is None

    path = tmp_path / "sol.npz"
    sol.save(str(path))
    loaded = jc.ContinuationSolution.load(str(path))

    assert loaded.eigenvalues is None
    assert loaded.stability is None
    assert loaded.tangent_vectors is None
    assert jnp.allclose(loaded.states, sol.states)

    assert loaded.bifurcations == []
    assert loaded.convergence_info is None
    assert loaded.state_names is None
    assert loaded.param_name is None


def test_load_rejects_a_file_with_no_format_version(tmp_path):
    path = tmp_path / "not_a_solution.npz"
    np.savez(path, states=np.zeros((1, 1)), parameters=np.zeros((1,)))
    with pytest.raises(ValueError, match="format_version"):
        jc.ContinuationSolution.load(str(path))


def test_load_rejects_pickled_array_payloads(tmp_path):
    """allow_pickle=False must be the load() default: an untrusted archive
    that smuggles a pickled object into a field this code actually reads
    must fail to load, not silently execute the pickle."""
    path = tmp_path / "malicious.npz"
    metadata = json.dumps(
        {
            "bifurcations": [],
            "convergence_info": None,
            "state_names": None,
            "param_name": None,
        }
    )
    np.savez(
        path,
        format_version=np.array(1),
        states=np.array([object()], dtype=object),
        parameters=np.zeros((1,)),
        metadata_json=np.array(metadata),
    )
    with pytest.raises(ValueError):
        jc.ContinuationSolution.load(str(path))
