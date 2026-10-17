"""
Integration tests: periodic_orbit_problem's returned BifProblem run through
the real jc.continuation() -- the compute_stability guard, and a
false-positive check for events=[Fold()] on a system with no fold-of-cycles.
See docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _periodic_problem():
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    mesh = Collocation(ntst=10, ncol=4)
    return periodic_orbit_problem(
        _rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), 5.5, 1.0, mesh
    )


def test_compute_stability_true_computes_floquet_multipliers_for_periodic_problem():
    # Regression for the removed guard clause: compute_stability=True is now
    # the whole point for periodic problems, not an error case. r' = r*(rho -
    # r^2), theta' = 1 has limit-cycle radius sqrt(rho) for all rho > 0, so
    # every branch point should read stable=True (exp(-4*pi*rho) < 1
    # always) -- see
    # docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.05, max_steps=50, newton_tol=1e-5
        ),
    )
    assert sol.branch.n_valid > 1
    assert sol.branch.eigenvalues is not None
    assert sol.branch.eigenvalues.shape == (sol.branch.n_valid, 2)
    assert sol.branch.trivial_index.shape == (sol.branch.n_valid,)
    leaves, tree = jax.tree_util.tree_flatten(sol.branch)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert jnp.array_equal(restored.trivial_index, sol.branch.trivial_index)
    assert sol.branch.stable is not None
    assert bool(jnp.all(sol.branch.stable))


def test_compute_stability_true_default_still_works_for_equilibrium_problem_after_dispatch():
    # Same as the existing equilibrium regression test, run again after the
    # problem.kind dispatch is added, to confirm the equilibrium path
    # (real-part condition via branch_eigenvalues) is provably untouched.
    def pitchfork(u, p, args):
        return jnp.array([p * u[0] - u[0] ** 3])

    prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
    sol = jc.continuation(prob, p_span=(0.5, 1.5))
    assert sol.branch.n_valid > 1
    assert sol.branch.eigenvalues is not None
    assert sol.branch.stable is not None


def test_compute_stability_false_runs_cleanly_for_periodic_problem():
    # newton_tol=1e-5, not ContinuationPar's default 1e-6: the collocation
    # residual's achievable float32 precision floor (~3.4e-6, verified
    # during design -- see periodic.py's periodic_orbit_problem docstring)
    # is tighter than the default corrector tolerance can reliably satisfy
    # every step, which stalls continuation (every step rejected, n_valid
    # stuck at 1) rather than raising -- this test would otherwise fail
    # silently on the n_valid assertion below, not on an obvious error.
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(
            compute_stability=False, ds=0.05, max_steps=50, newton_tol=1e-5
        ),
    )
    assert sol.branch.n_valid > 1


def test_compute_stability_true_default_still_works_for_equilibrium_problem():
    # The guard must only fire for kind="periodic" -- equilibrium
    # continuation (compute_stability=True is the default) must be
    # completely unaffected.
    def pitchfork(u, p, args):
        return jnp.array([p * u[0] - u[0] ** 3])

    prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
    sol = jc.continuation(prob, p_span=(0.5, 1.5))
    assert sol.branch.n_valid > 1


def test_fold_events_zero_false_positives_on_periodic_branch():
    # r' = r*(rho - r^2) has limit-cycle radius sqrt(rho), smooth and
    # monotonic in rho -- no fold-of-cycles anywhere on this branch.
    # Verified during design directly against jc.continuation(): 34 valid
    # points from rho=1.0 to rho~2.018, zero Fold detections, final radius
    # matching sqrt(rho) to 7 significant figures.
    prob = _periodic_problem()
    sol = jc.continuation(
        prob, p_span=(1.0, 2.0),
        settings=jc.ContinuationPar(
            compute_stability=False, ds=0.05, max_steps=50, newton_tol=1e-5
        ),
        events=[jc.Fold()],
    )
    assert sol.branch.n_valid > 1
    assert sol.events == []


def _brusselator(u, p, args):
    x, y = u[0], u[1]
    return jnp.array([1.0 - (p + 1.0) * x + x**2 * y, p * x - x**2 * y])


def test_long_branch_non_symmetric_orbit():
    # Example 11's Brusselator over its full documented range, b in [2.5, 4]:
    # the cycle is not symmetric and its shape changes strongly.
    from scipy.integrate import solve_ivp
    from scipy.signal import find_peaks

    def rhs_np(t, u):
        x, y = u
        return [1.0 - 3.5 * x + x**2 * y, 2.5 * x - x**2 * y]

    raw = solve_ivp(rhs_np, [0.0, 80.0], [1.1, 1.1], max_step=0.01, dense_output=True)
    tail = np.linspace(60.0, 80.0, 6000)
    peaks, _ = find_peaks(raw.sol(tail)[0])
    peak_times = tail[peaks]
    period = float(np.mean(np.diff(peak_times)))
    t_traj = np.linspace(0.0, period, 60, endpoint=False)
    u_traj = raw.sol(t_traj + peak_times[-2]).T

    prob = periodic_orbit_problem(
        _brusselator, jnp.asarray(u_traj), jnp.asarray(t_traj), period, 2.5,
        Collocation(ntst=20, ncol=4), tol=1e-4,
    )
    sol = jc.continuation(
        prob, p_span=(2.5, 4.0),
        settings=jc.ContinuationPar(
            ds=0.05, max_steps=350, newton_tol=1e-4, compute_stability=False
        ),
    )
    assert sol.status == "reached_end"
    assert float(sol.branch.params[-1]) >= 4.0


def _flip_angle(p):
    """Rotation angle of the cycle's plane: 0 at p <= 0, pi at p >= 1 (C^2)."""
    s = jnp.clip(p, 0.0, 1.0)
    return jnp.pi * s**3 * (10.0 - 15.0 * s + 6.0 * s**2)


def _rot_x(a):
    c, s = jnp.cos(a), jnp.sin(a)
    o, z = jnp.ones_like(c), jnp.zeros_like(c)
    return jnp.array([[o, z, z], [z, c, -s], [z, s, c]])


def _flipping_cycle(u, p, args):
    """A unit-circle cycle of period 2*pi whose plane turns by pi about the
    x axis as p goes from 0 to 1: at p >= 1 the same circle runs backwards."""
    R = _rot_x(_flip_angle(p))
    z = R.T @ u
    r2 = z[0] ** 2 + z[1] ** 2
    dz = jnp.array([(1 - r2) * z[0] - z[1], (1 - r2) * z[1] + z[0], -z[2]])
    return R @ dz


def test_periodic_branch_survives_reversal_of_the_cycle():
    # A phase condition anchored to the starting cycle vanishes identically
    # once the cycle runs backwards, so the reference must follow the branch.
    t = np.linspace(0.0, 2 * np.pi, 60, endpoint=False)
    u = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    ntst = 10
    prob = periodic_orbit_problem(
        _flipping_cycle, jnp.asarray(u), jnp.asarray(t), 2 * np.pi, 0.0,
        Collocation(ntst=ntst, ncol=4),
    )
    sol = jc.continuation(
        prob, p_span=(0.0, 2.0),
        settings=jc.ContinuationPar(
            # The jump bound below measures Euclidean distance.
            theta=1.0, ds=0.05, max_steps=200, newton_tol=1e-5, compute_stability=False
        ),
    )
    assert sol.status == "reached_end"
    # Consecutive points stay within two ds_max (= 0.1) of each other.
    steps = np.diff(np.column_stack([sol.branch.states, sol.branch.params]), axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) <= 0.2
    last = np.asarray(sol.branch.states[-1])
    mesh_states = last[: ntst * 3].reshape(ntst, 3)
    np.testing.assert_allclose(last[-1], 2 * np.pi, rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(mesh_states, axis=1), 1.0, atol=1e-4)
    # Reversed sense of rotation: x * y' - y * x' < 0 around the cycle.
    x, y = mesh_states[:, 0], mesh_states[:, 1]
    assert np.all(x * np.roll(y, -1) - y * np.roll(x, -1) < 0)
