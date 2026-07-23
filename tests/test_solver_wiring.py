"""
Tests proving the LinearSolver/EigenSolver protocol seam wired into
core/scan_continuation.py is real (routes actual calls through a supplied
solver), not decorative -- see
docs/superpowers/specs/2026-07-23-linear-eigen-solver-protocols-design.md.
"""

import jax
import jax.numpy as jnp

from jaxcont.core.scan_continuation import (
    branch_eigenvalues,
    natural_scan,
    pseudo_arclength_scan,
)
from jaxcont.solvers.protocols import Dense


def pitchfork(u, p):
    return jnp.array([p * u[0] - u[0] ** 3])


class _CountingLinearSolver:
    """Delegates to jnp.linalg.solve but records how many times it ran.

    Deliberately a plain class, not a frozen dataclass like Dense --
    LinearSolver implementations only need to be valid jax.jit static
    arguments (hashable + equality-comparable), which the default
    identity-based __eq__/__hash__ on a normal class already satisfies for
    a single-use, stateful test double. (A frozen dataclass with a list
    field would generate a __hash__ that crashes -- lists aren't hashable --
    the exact trap Dense's "no fields" design in Task 1 exists to avoid.)
    """

    def __init__(self):
        self.calls = []

    def __call__(self, A, b):
        self.calls.append(1)
        return jnp.linalg.solve(A, b)


class _CountingEigenSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, A):
        self.calls.append(1)
        return jnp.linalg.eigvals(A)


_SCAN_ARGS = (
    pitchfork, jnp.array([0.1]), jnp.array(0.5), jnp.array(1.5),
    jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
    jnp.array(1e-6), 60, jnp.array(20),
)


def test_pseudo_arclength_scan_routes_through_custom_linear_solver():
    solver = _CountingLinearSolver()
    res = pseudo_arclength_scan(*_SCAN_ARGS, solver)
    assert int(res.n_valid) > 1
    assert len(solver.calls) > 0


def test_natural_scan_routes_through_custom_linear_solver():
    solver = _CountingLinearSolver()
    res = natural_scan(*_SCAN_ARGS, solver)
    assert int(res.n_valid) > 1
    assert len(solver.calls) > 0


def test_branch_eigenvalues_routes_through_custom_eigen_solver():
    solver = _CountingEigenSolver()
    states = jnp.array([[0.5], [0.6]])
    params = jnp.array([1.0, 1.0])
    branch_eigenvalues(pitchfork, states, params, eigen_solver=solver)
    assert len(solver.calls) > 0


def test_pseudo_arclength_scan_accepts_independently_constructed_dense_instances():
    # Two separately-constructed Dense() instances (not the same object)
    # must be usable interchangeably as a jax.jit static argument -- no
    # recompile-related error, and identical results.
    res_a = pseudo_arclength_scan(*_SCAN_ARGS, Dense())
    res_b = pseudo_arclength_scan(*_SCAN_ARGS, Dense())
    assert jnp.array_equal(res_a.states, res_b.states)
    assert jnp.array_equal(res_a.params, res_b.params)


def test_pseudo_arclength_scan_vmap_with_explicit_linear_solver():
    def run(p0):
        return pseudo_arclength_scan(
            pitchfork, jnp.array([0.1]), p0, p0 + 1.0,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-6), 80, jnp.array(20), Dense(),
        )

    batch = jax.vmap(run)(jnp.linspace(0.5, 3.0, 16))
    assert batch.params.shape == (16, 81)
    assert batch.n_valid.shape == (16,)


def test_pseudo_arclength_scan_matches_pre_protocol_baseline():
    # Regression guard: captured from the unmodified (pre-Task-2) engine on
    # 2026-07-23 by running pseudo_arclength_scan directly on this exact
    # pitchfork problem. Dense() must reproduce these numbers exactly.
    res = pseudo_arclength_scan(*_SCAN_ARGS)
    n = int(res.n_valid)
    assert n == 9
    expected_params = [
        0.5, 0.5298426151275635, 0.6048426032066345, 0.7173426151275635,
        0.8860926032066345, 1.0860925912857056, 1.2860926389694214,
        1.4860926866531372, 1.686092734336853,
    ]
    expected_states0 = [
        0.10000000149011612, 9.813811630010605e-08, 8.424652264693577e-08,
        6.857676737581642e-08, 5.244454825970024e-08, 4.0607286422300604e-08,
        3.3129602172721206e-08, 2.7977623773267624e-08, 2.4212363669562365e-08,
    ]
    assert res.params[:n].tolist() == expected_params
    assert res.states[:n, 0].tolist() == expected_states0
