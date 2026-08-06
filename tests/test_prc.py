"""
Tests for jaxcont.stability.prc: infinitesimal phase response curves (iPRC)
via the collocation adjoint method, for the periodic orbit
r' = r*(rho - r^2), theta' = 1 -- the same closed-form circle system
tests/test_floquet.py uses. Because theta' = 1 independent of r, the
asymptotic phase is exactly theta, so Z = grad(phase) in Cartesian
coordinates is closed-form: Z(theta) = (-sin(theta), cos(theta)) / sqrt(rho).
See docs/superpowers/specs/2026-08-05-prc-dprc-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.prc import prc_curve


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


def _expected_Z(prob, mesh, rho):
    n = 2
    mesh_states = np.asarray(prob.u0[: mesh.ntst * n]).reshape(mesh.ntst, n)
    theta = np.arctan2(mesh_states[:, 1], mesh_states[:, 0])
    return np.stack([-np.sin(theta), np.cos(theta)], axis=1) / np.sqrt(rho)


def test_prc_curve_matches_closed_form_at_rho_1():
    prob, mesh = _circle_problem(rho=1.0)
    with jax.default_matmul_precision("float32"):
        Z = prc_curve(_rhs, mesh, prob.u0, prob.p0)
    assert Z.shape == (mesh.ntst, 2)
    expected = _expected_Z(prob, mesh, rho=1.0)
    assert float(jnp.max(jnp.abs(np.asarray(Z) - expected))) < 1e-5


def test_prc_curve_periodicity_seed_matches_chain_endpoint():
    """The backward adjoint chain must land back on its own seed Z(0) --
    the numerical expression of (Phi(T)^T - I) Z(0) = 0."""
    from jaxcont.core.collocation import collocation_matrices, interval_propagators

    prob, mesh = _circle_problem(rho=1.0)
    ntst, ncol, n = mesh.ntst, mesh.ncol, 2
    h = 1.0 / ntst
    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = prob.u0[-1]

    with jax.default_matmul_precision("float32"):
        M_all = interval_propagators(_rhs, D, E, h, mesh_states, coll_states, T, prob.p0)
        Phi, _ = jax.lax.scan(lambda c, M: (M @ c, None), jnp.eye(n), M_all)
        Z = prc_curve(_rhs, mesh, prob.u0, prob.p0)
    # Z[0], as returned by prc_curve, is not literally the bordered-solve
    # seed Z0 before any propagation -- prc_curve's backward jax.lax.scan
    # starts at Z0 and chains M_i^T through every interval in reverse
    # order, and Z[0] is that scan's *last* step (after reversing the scan
    # output back into forward order), i.e. Z0 having already made one
    # full round trip through Phi^T = M_all[0]^T @ ... @ M_all[-1]^T.
    # Periodicity (Phi(T)^T Z(0) = Z(0)) means this round-tripped value
    # should equal Z0 again, which is exactly what this residual checks --
    # it is not checking Z[0] against itself trivially.
    residual = (Phi.T - jnp.eye(n)) @ Z[0]
    assert float(jnp.max(jnp.abs(residual))) < 1e-5


def test_branch_prc_matches_per_point_calls():
    from jaxcont.stability.prc import branch_prc

    prob1, mesh = _circle_problem(rho=1.0)
    prob2, _ = _circle_problem(rho=1.5)
    states = jnp.stack([prob1.u0, prob2.u0])
    params = jnp.stack([prob1.p0, prob2.p0])

    with jax.default_matmul_precision("float32"):
        batched = branch_prc(_rhs, mesh, states, params)
        individual = jnp.stack(
            [
                prc_curve(_rhs, mesh, prob1.u0, prob1.p0),
                prc_curve(_rhs, mesh, prob2.u0, prob2.p0),
            ]
        )
    assert batched.shape == (2, mesh.ntst, 2)
    assert jnp.max(jnp.abs(batched - individual)) < 1e-6


def test_dprc_curve_matches_finite_difference_of_reconverged_orbits():
    """dprc_curve must differentiate through a re-solve of U(p) -- NOT
    jax.jacfwd(prc_curve, argnums=p) at a frozen U, which prototyping showed
    is differentiable but not meaningful (see the design spec's "Design
    findings from prototyping"). Verified against a central finite
    difference built from two INDEPENDENTLY re-converged periodic orbits at
    rho +/- eps (not the naive closed form alone -- that omits the phase
    condition's own small theta-drift with rho)."""
    from jaxcont.stability.prc import dprc_curve

    rho0, eps = 1.0, 0.01
    prob0, mesh = _circle_problem(rho0)
    prob_hi, _ = _circle_problem(rho0 + eps)
    prob_lo, _ = _circle_problem(rho0 - eps)

    with jax.default_matmul_precision("float32"):
        dZ = dprc_curve(prob0)
        Z_hi = prc_curve(_rhs, mesh, prob_hi.u0, prob_hi.p0)
        Z_lo = prc_curve(_rhs, mesh, prob_lo.u0, prob_lo.p0)

    dZ_fd = (Z_hi - Z_lo) / (2 * eps)
    assert dZ.shape == (mesh.ntst, 2)
    assert float(jnp.max(jnp.abs(np.asarray(dZ) - np.asarray(dZ_fd)))) < 0.01
