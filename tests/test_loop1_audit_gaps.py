"""Tests for documented behaviour that had no test (Loop 1 audit L1-F40, L1-F42)."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
import jaxcont.bifurcations.pd_switch as pd_switch
from jaxcont.bifurcations.events import EventHit


def test_load_reads_a_valid_version_1_archive(tmp_path):
    """The 0.4 format (format_version=1): no trivial_index, Floquet, layout, status or rejected events."""
    states = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    params = np.array([-1.0, -0.5, 0.0])
    eigenvalues = np.array([[-1.0 + 0.5j, -1.0 - 0.5j]] * 3)
    stability = np.array([True, True, False])
    metadata = {"bifurcations": [{"type": "hopf", "parameter": -0.25, "index": 1}],
                "convergence_info": None, "state_names": ["x", "y"], "param_name": "mu"}
    path = tmp_path / "v1.npz"
    np.savez(path, format_version=np.array(1), states=states, parameters=params,
             eigenvalues=eigenvalues, stability=stability, metadata_json=np.array(json.dumps(metadata)))

    loaded = jc.ContinuationSolution.load(str(path))

    # Loading converts to the session's JAX dtype; values are otherwise exact.
    as_loaded = lambda reference, got: reference.astype(np.asarray(got).dtype)
    np.testing.assert_array_equal(np.asarray(loaded.states), as_loaded(states, loaded.states))
    np.testing.assert_array_equal(np.asarray(loaded.parameters), as_loaded(params, loaded.parameters))
    np.testing.assert_array_equal(np.asarray(loaded.eigenvalues), as_loaded(eigenvalues, loaded.eigenvalues))
    np.testing.assert_array_equal(np.asarray(loaded.stability), stability)
    assert loaded.bifurcations == metadata["bifurcations"]
    assert loaded.state_names == ("x", "y") and loaded.param_name == "mu"
    # Fields introduced in version 2 take their defaults.
    assert loaded.tangent_vectors is None and loaded.trivial_index is None
    assert loaded.curve_layout is None and loaded.rejected_events == [] and loaded.status is None


def _pd_event():
    """The period-doubling point of the fixture in tests/test_pd_switch.py (alpha = 0)."""
    from tests.test_pd_switch import base_problem
    prob, mesh = base_problem(0.0)
    return EventHit(kind="period_doubling", p=float(prob.p0), u=prob.u0, index=None), mesh


# An explicit value differing from the dtype default (1e-9 / 1e-5) and reachable in that precision.
_EXPLICIT = 1e-8


@pytest.mark.parametrize("refine_tol", [None, _EXPLICIT])
def test_branch_switch_pd_forwards_refine_tol_to_candidate_correction(monkeypatch, refine_tol):
    from tests.test_pd_switch import rhs
    seen = []
    original = pd_switch.periodic_orbit_problem

    def recording(*args, **kwargs):
        seen.append(kwargs.get("tol"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pd_switch, "periodic_orbit_problem", recording)
    event, mesh = _pd_event()
    pd_switch.branch_switch_pd(rhs, event, mesh, dp=0.05, refine_tol=refine_tol)
    # Every candidate correction received exactly the caller's value (None resolves by dtype inside).
    assert seen and all(tol == refine_tol for tol in seen)
