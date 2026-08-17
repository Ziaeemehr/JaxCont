"""
Tests for jaxcont.stability.floquet: Floquet multipliers via the collocation
monodromy matrix (core/collocation.py's monodromy_matrix), for the periodic
orbit r' = r*(rho - r^2), theta' = 1, which has an exact closed-form limit
cycle x=cos(t), y=sin(t), T=2*pi at rho=1, with exact Floquet multipliers
{1, exp(-4*pi)}. See
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.floquet import (
    branch_floquet_multipliers,
    floquet_multipliers,
    floquet_stable,
)


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _coarse_wrong_trajectory():
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    return jnp.asarray(u_traj), jnp.asarray(t_traj)


def _circle_problem(rho=1.0):
    u_traj, t_traj = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_traj, t_traj, 5.5, rho, mesh)
    return prob, mesh


def test_monodromy_matrix_matches_interval_propagators_scan():
    """interval_propagators must be extractable from monodromy_matrix without
    changing monodromy_matrix's own numerics -- see
    docs/superpowers/specs/2026-08-05-prc-dprc-design.md."""
    from jaxcont.core.collocation import (
        collocation_matrices,
        interval_propagators,
        monodromy_matrix,
    )

    prob, mesh = _circle_problem(rho=1.0)
    ntst, ncol = mesh.ntst, mesh.ncol
    n = 2
    h = 1.0 / ntst
    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = prob.u0[-1]

    M_all = interval_propagators(_rhs, D, E, h, mesh_states, coll_states, T, prob.p0)
    assert M_all.shape == (ntst, n, n)

    Phi_from_blocks, _ = jax.lax.scan(
        lambda carry, M: (M @ carry, None), jnp.eye(n), M_all
    )
    Phi_direct = monodromy_matrix(_rhs, D, E, h, mesh_states, coll_states, T, prob.p0)
    assert jnp.max(jnp.abs(Phi_from_blocks - Phi_direct)) < 1e-6


def test_floquet_multipliers_matches_closed_form_at_rho_1():
    # Verified during design: JAX result [3.4570694e-06, 1.0000001] vs
    # exact {1, exp(-4*pi)} = {1, 3.4873423562089973e-06}.
    prob, mesh = _circle_problem(rho=1.0)
    multipliers = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    got = jnp.sort(jnp.abs(multipliers))
    expected = jnp.sort(jnp.array([np.exp(-4 * np.pi), 1.0]))
    assert float(jnp.max(jnp.abs(got - expected))) < 1e-5


def test_floquet_stable_true_at_rho_1_with_correct_trivial_multiplier():
    prob, mesh = _circle_problem(rho=1.0)
    multipliers = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    stable = floquet_stable(multipliers[None, :])
    assert bool(stable[0]) is True

    trivial_idx = int(jnp.argmin(jnp.abs(multipliers - 1.0)))
    assert abs(float(multipliers[trivial_idx].real) - 1.0) < 1e-4


def test_branch_floquet_multipliers_matches_per_point_calls():
    prob1, mesh = _circle_problem(rho=1.0)
    prob2, _ = _circle_problem(rho=1.5)
    states = jnp.stack([prob1.u0, prob2.u0])
    params = jnp.stack([prob1.p0, prob2.p0])

    batched = branch_floquet_multipliers(_rhs, mesh, states, params)
    individual = jnp.stack(
        [
            floquet_multipliers(_rhs, mesh, prob1.u0, prob1.p0),
            floquet_multipliers(_rhs, mesh, prob2.u0, prob2.p0),
        ]
    )
    assert batched.shape == (2, 2)
    assert jnp.allclose(jnp.sort(jnp.abs(batched), axis=1), jnp.sort(jnp.abs(individual), axis=1), atol=1e-4)
