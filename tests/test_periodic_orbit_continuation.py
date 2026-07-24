"""
Integration tests: periodic_orbit_problem's returned BifProblem run through
the real jc.continuation() -- the compute_stability guard, and a
false-positive check for events=[Fold()] on a system with no fold-of-cycles.
See docs/superpowers/specs/2026-07-24-periodic-orbit-collocation-design.md.
"""

import jax.numpy as jnp
import numpy as np
import pytest

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
