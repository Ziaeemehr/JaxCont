"""
Tests for jaxcont.problems.periodic.periodic_orbit_problem -- fixed-mesh
collocation applied to r' = r*(rho - r^2), theta' = 1 (Cartesian), which has
an exact closed-form limit cycle x(t)=cos(t), y(t)=sin(t), T=2*pi at
rho=1 -- independent of any external reference tool. See
docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.problems.periodic_args import PeriodicArgs


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


_AD_MESH = Collocation(ntst=10, ncol=4)
_AD_T = jnp.linspace(0.0, 2 * jnp.pi, 120, endpoint=False)


def _cycle_radius(p, *, parameter_dependent_guess):
    guess_radius = jnp.sqrt(p) if parameter_dependent_guess else 1.0
    u_guess = guess_radius * jnp.stack([jnp.cos(_AD_T), jnp.sin(_AD_T)], axis=1)
    problem = periodic_orbit_problem(
        _rhs, u_guess, _AD_T, 2 * jnp.pi, p, _AD_MESH
    )
    mesh_states = problem.u0[: 2 * _AD_MESH.ntst].reshape(_AD_MESH.ntst, 2)
    return jnp.mean(jnp.linalg.norm(mesh_states, axis=1))


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


def test_periodic_orbit_problem_args_carries_raw_f_and_mesh():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)

    assert isinstance(prob.args, PeriodicArgs)
    assert prob.args.raw_f is _rhs
    assert prob.args.mesh is mesh
    assert prob.args.user_args is None
    # raw_f and mesh are static: only the reference data are pytree leaves.
    assert len(jax.tree_util.tree_leaves(prob.args)) == 2
    r = prob.f(prob.u0, prob.p0, prob.args)
    assert float(jnp.linalg.norm(r)) < 1e-5


def test_periodic_problem_is_jit_argument():
    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)
    total = jax.jit(lambda pr: pr.u0.sum())(prob)
    assert float(total) == pytest.approx(float(prob.u0.sum()))


def _hopf_omega(u, p, omega):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([(p - r2) * x - omega * y, (p - r2) * y + omega * x])


_OMEGA_MESH = Collocation(ntst=10, ncol=4)


def _period_for(omega, p=1.0):
    t = jnp.linspace(0.0, 2 * jnp.pi / 1.5, 120, endpoint=False)
    u = jnp.stack([jnp.cos(1.5 * t), jnp.sin(1.5 * t)], axis=1)
    prob = periodic_orbit_problem(
        _hopf_omega, u, t, 2 * jnp.pi / 1.5, p, _OMEGA_MESH, args=omega
    )
    return prob.u0[-1]


def test_periodic_rhs_receives_user_args():
    omega = jnp.asarray(1.5)
    assert float(_period_for(omega)) == pytest.approx(2 * np.pi / 1.5, rel=1e-3)


def test_grad_period_wrt_user_args():
    omega = jnp.asarray(1.5)
    g = jax.grad(_period_for)(omega)
    assert float(g) == pytest.approx(-2 * np.pi / 1.5**2, rel=1e-3)


def test_periodic_continuation_reuses_compiled_kernel():
    from jaxcont.core.scan_continuation import pseudo_arclength_scan

    u_trajectory, t_trajectory = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    settings = jc.ContinuationPar(max_steps=3, ds=0.02, compute_stability=False)

    def run():
        prob = periodic_orbit_problem(_rhs, u_trajectory, t_trajectory, 5.5, 1.0, mesh)
        return jc.continuation(prob, p_span=(1.0, 1.1), settings=settings)

    run()
    size = pseudo_arclength_scan._cache_size()
    run()
    run()
    assert pseudo_arclength_scan._cache_size() == size


def test_construction_grad_matches_exact_radius_derivative_with_fixed_guess():
    parameters = jnp.array([0.5, 1.0, 1.5])
    actual = jax.vmap(jax.grad(lambda p: _cycle_radius(p, parameter_dependent_guess=False)))(
        parameters
    )
    expected = 1.0 / (2.0 * jnp.sqrt(parameters))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=0.0)


def test_construction_grad_matches_exact_radius_derivative_with_parameter_dependent_guess():
    parameters = jnp.array([0.5, 1.0, 1.5])
    actual = jax.vmap(jax.grad(lambda p: _cycle_radius(p, parameter_dependent_guess=True)))(
        parameters
    )
    expected = 1.0 / (2.0 * jnp.sqrt(parameters))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=0.0)


def test_vmap_over_construction_matches_eager_radii():
    parameters = jnp.array([0.5, 1.0, 1.5])
    eager = jnp.stack(
        [_cycle_radius(p, parameter_dependent_guess=True) for p in parameters]
    )
    batched = jax.vmap(
        lambda p: _cycle_radius(p, parameter_dependent_guess=True)
    )(parameters)
    np.testing.assert_array_equal(batched, eager)


def test_jit_over_construction_matches_eager_radius():
    p = jnp.asarray(1.25)
    eager = _cycle_radius(p, parameter_dependent_guess=True)
    compiled = jax.jit(
        lambda value: _cycle_radius(value, parameter_dependent_guess=True)
    )(p)
    np.testing.assert_allclose(compiled, eager, atol=2e-7, rtol=0.0)


def test_floquet_and_prc_receive_user_args():
    from jaxcont.stability.floquet import floquet_multipliers
    from jaxcont.stability.prc import prc_curve

    omega = jnp.asarray(1.5)
    t = jnp.linspace(0.0, 2 * jnp.pi / 1.5, 120, endpoint=False)
    u = jnp.stack([jnp.cos(1.5 * t), jnp.sin(1.5 * t)], axis=1)
    prob = periodic_orbit_problem(
        _hopf_omega, u, t, 2 * jnp.pi / 1.5, 1.0, _OMEGA_MESH, args=omega
    )
    pargs = prob.args
    assert float(pargs.user_args) == 1.5
    T = float(prob.u0[-1])
    mult = floquet_multipliers(
        _hopf_omega, _OMEGA_MESH, prob.u0, prob.p0, args=pargs.user_args
    )
    nontrivial = mult[jnp.argmax(jnp.abs(mult - 1.0))]
    assert float(jnp.abs(nontrivial)) == pytest.approx(float(np.exp(-2.0 * T)), rel=1e-2)
    Z = prc_curve(_hopf_omega, _OMEGA_MESH, prob.u0, prob.p0, args=pargs.user_args)
    x0 = prob.u0[:2]
    f0 = _hopf_omega(x0, prob.p0, pargs.user_args)
    assert float(Z[0] @ f0) == pytest.approx(2 * np.pi / T, rel=1e-3)


def test_trajectory_time_offset_is_rebased():
    from jaxcont.utils.precision import default_tol

    u, t = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    original = periodic_orbit_problem(_rhs, u, t, 5.5, 1.0, mesh)
    shifted = periodic_orbit_problem(_rhs, u, t + 100, 5.5, 1.0, mesh)
    np.testing.assert_allclose(
        shifted.u0, original.u0, atol=default_tol(u.dtype), rtol=0.0
    )


def test_collapsed_orbit_raises():
    t = jnp.linspace(0.0, 2 * jnp.pi, 120)
    with pytest.raises(ValueError, match="amplitude"):
        periodic_orbit_problem(
            _rhs, jnp.zeros((t.size, 2)), t, 2 * jnp.pi, 1.0, _AD_MESH
        )


def test_nonconverged_construction_raises():
    u, t = _coarse_wrong_trajectory()
    with pytest.raises(ValueError, match="converge"):
        periodic_orbit_problem(_rhs, u, t, 5.5, 1.0, _AD_MESH, max_iter=1)


def test_short_trajectory_raises():
    u, t = _coarse_wrong_trajectory()
    with pytest.raises(ValueError, match="0.95"):
        periodic_orbit_problem(_rhs, u, t * 0.9, 5.5, 1.0, _AD_MESH)


def test_nonpositive_refined_period_raises():
    u, t = _coarse_wrong_trajectory()
    with pytest.raises(ValueError, match="positive"):
        periodic_orbit_problem(_rhs, u, t, -5.5, 1.0, _AD_MESH)


def test_nonpositive_period0_raises_before_resampling():
    u, t = _coarse_wrong_trajectory()
    for bad in (0.0, -5.5):
        with pytest.raises(ValueError, match="period0 must be positive"):
            periodic_orbit_problem(_rhs, u, t, bad, 1.0, _AD_MESH)


def test_trajectory_longer_than_period0_is_not_folded():
    # 1.5 laps of the exact circle with a 3% wrong period guess. Resampling
    # must read the first lap only: folding the later samples back modulo
    # period0 interleaves points from a phase-shifted second lap.
    t = np.linspace(0.0, 3 * np.pi, 2000)
    u = np.stack([np.cos(t), np.sin(t)], axis=1)
    period0 = 1.03 * 2 * np.pi
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(
        _rhs, jnp.asarray(u), jnp.asarray(t), period0, 1.0, mesh
    )

    seed = np.asarray(prob.args.u_ref_coll)  # resampled (unrefined) seed
    _, _, gauss, _ = jc.core.collocation.collocation_matrices(mesh.ncol)
    tau = (np.arange(mesh.ntst)[:, None] + np.asarray(gauss)[None, :]) / mesh.ntst
    reference = np.stack([np.cos(tau * period0), np.sin(tau * period0)], axis=-1)
    np.testing.assert_allclose(seed, reference, atol=1e-4)

    assert float(prob.u0[-1]) == pytest.approx(2 * np.pi, rel=1e-3)
    radius = np.linalg.norm(np.asarray(prob.u0[: 2 * mesh.ntst]).reshape(-1, 2), axis=1)
    np.testing.assert_allclose(radius, 1.0, atol=1e-3)
