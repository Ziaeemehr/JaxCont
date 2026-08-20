"""
Regression test locking in BoundaryValueProblem's documented placeholder
status (2026-08-19 review finding #13): solve_collocation/solve_shooting
must keep raising NotImplementedError, and the class docstring must say so
up front, until a real implementation lands.
"""

import jax.numpy as jnp
import pytest

from jaxcont.problems.bvp import BoundaryValueProblem


def _sample_problem() -> BoundaryValueProblem:
    return BoundaryValueProblem(
        rhs=lambda t, u, params: u,
        boundary_conditions=lambda u0, uT: u0 - uT,
        params={},
        t_span=(0.0, 1.0),
        initial_guess=jnp.zeros(2),
    )


def test_class_docstring_states_placeholder_status():
    assert "placeholder" in BoundaryValueProblem.__doc__.lower()


def test_solve_collocation_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _sample_problem().solve_collocation()


def test_solve_shooting_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        _sample_problem().solve_shooting()
