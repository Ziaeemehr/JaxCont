"""
Tests for jaxcont.problems.periodic.periodic_orbit_problem -- fixed-mesh
collocation applied to r' = r*(rho - r^2), theta' = 1 (Cartesian), which has
an exact closed-form limit cycle x(t)=cos(t), y(t)=sin(t), T=2*pi at
rho=1 -- independent of any external reference tool. See
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _coarse_wrong_trajectory():
    # Deliberately wrong: radius 0.8 (true circle has radius 1.0), phase
    # offset 0.3, and a period guess of 5.5 (true period is 2*pi ~ 6.283).
    # Discrete, irregularly-spaced samples -- not a closed form -- to
    # exercise jnp.interp resampling with real data, matching what was
    # verified during design.
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    return jnp.asarray(u_traj), jnp.asarray(t_traj)


def test_periodic_orbit_problem_refines_to_exact_circle():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)

    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    n = 2
    mesh_states = prob.u0[: mesh.ntst * n].reshape(mesh.ntst, n)
    T = prob.u0[-1]

    # Tolerances here (1e-5) are float32-achievable, not float64-tight --
    # this project runs float32 by default (no jax_enable_x64 anywhere).
    # Verified during design under real float32 (with the matmul-precision
    # fix in residual() -- without it, this system doesn't converge at
    # all, see periodic.py): T error ~3.0e-7, radius error ~2.4e-7 -- both
    # comfortably inside 1e-5 with margin to spare.
    assert abs(float(T) - 2 * np.pi) < 1e-5
    radii = jnp.linalg.norm(mesh_states, axis=1)
    assert float(jnp.max(jnp.abs(radii - 1.0))) < 1e-5
    assert prob.kind == "periodic"


def test_periodic_orbit_problem_residual_is_near_zero_at_u0():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    r = prob.f(prob.u0, prob.p0, prob.args)
    # Verified during design under real float32: residual norm ~3.4e-6 --
    # the achievable floor for this ~100-dim system at this precision
    # (hence differentiable_root's tol=1e-5 in periodic.py, not the
    # unreachable default 1e-8).
    assert float(jnp.linalg.norm(r)) < 1e-5


def test_periodic_orbit_problem_mesh_size_scaling_sanity():
    # Regression for the mesh-size sanity check verified during design:
    # both a coarser (ntst=10) and finer (ntst=15) mesh converge to the
    # same exact circle to float32-achievable precision. This does NOT
    # assert finer-is-more-accurate (err_fine <= err_coarse): verified
    # during design, at this precision both mesh sizes sit at the float32
    # noise floor (~1e-7 range) where floating-point rounding, not
    # discretization error, dominates -- coarse measured ~2.4e-7, fine
    # ~3.6e-7 (fine slightly *worse*, not better). Asserting an ordering
    # here would be asserting noise, not a real discretization-order
    # property; that requires float64 to observe meaningfully, which this
    # project doesn't run by default.
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh_coarse = Collocation(ntst=10, ncol=4)
    mesh_fine = Collocation(ntst=15, ncol=4)

    prob_coarse = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh_coarse)
    prob_fine = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh_fine)

    n = 2
    mesh_states_coarse = prob_coarse.u0[: mesh_coarse.ntst * n].reshape(mesh_coarse.ntst, n)
    mesh_states_fine = prob_fine.u0[: mesh_fine.ntst * n].reshape(mesh_fine.ntst, n)

    err_coarse = float(jnp.max(jnp.abs(jnp.linalg.norm(mesh_states_coarse, axis=1) - 1.0)))
    err_fine = float(jnp.max(jnp.abs(jnp.linalg.norm(mesh_states_fine, axis=1) - 1.0)))

    assert err_coarse < 1e-5
    assert err_fine < 1e-5
