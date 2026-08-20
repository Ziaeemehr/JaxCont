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
    # pitchfork problem. Dense() must reproduce these numbers to
    # float32-achievable precision -- NOT bit-for-bit: exact `==` equality
    # on floats compares the baseline-capturing machine's specific
    # GPU/driver/XLA rounding behavior, not this test's actual intent (that
    # the LinearSolver/EigenSolver refactor didn't change behavior). Found
    # via a real CI failure on different hardware than this baseline was
    # captured on: index 4 of states0 differed by ~3.6e-15 absolute
    # (~6.8e-8 relative) -- squarely float32 cross-hardware noise, not a
    # regression (this test passes bit-exact on the original machine).
    #
    # Re-captured 2026-08-19 (Phase 1 correctness hardening, Task 1): slot 0
    # is now Newton-corrected instead of the raw, invalid seed (see
    # scan_continuation.py's u0_seed/_natural_correct call). u0=[0.1] at
    # p0=0.5 isn't itself an equilibrium (residual 0.049); plain Newton
    # collapses it onto the trivial branch u=0 -- the same branch points
    # 1..8 of the *old* baseline were already tracking (their states0 were
    # already ~1e-7, not near 0.1). Starting step 0 from an exact root
    # instead of an off-branch guess changes the first Newton correction's
    # iteration count, which cascades through _adapt_ds into a different
    # step-size/point-count trajectory (n=8 instead of 9) -- expected, not a
    # regression.
    res = pseudo_arclength_scan(*_SCAN_ARGS)
    n = int(res.n_valid)
    assert n == 8
    expected_params = [
        0.5, 0.550000011920929, 0.625, 0.737500011920929,
        0.90625, 1.1062500476837158, 1.3062500953674316, 1.5062501430511475,
    ]
    expected_states0 = [
        3.082677721977234e-07, 2.7744098929360916e-07, 2.396081129063532e-07,
        1.9647865201477543e-07, 1.515216752068227e-07, 1.1808241140442988e-07,
        9.67341833302271e-08, 8.192321132582947e-08,
    ]
    assert jnp.allclose(res.params[:n], jnp.array(expected_params), atol=1e-6)
    assert jnp.allclose(res.states[:n, 0], jnp.array(expected_states0), atol=1e-6)


def test_continuation_routes_through_custom_solvers_bundle():
    # The spec-required proof that continuation() itself -- the public
    # entry point, not just the lower-level scan functions -- actually
    # routes through a user-supplied Solvers bundle.
    import jaxcont as jc

    linear_solver = _CountingLinearSolver()
    eigen_solver = _CountingEigenSolver()

    prob = jc.bif_problem(lambda u, p, args: pitchfork(u, p), u0=jnp.array([0.1]), p0=0.5)
    result = jc.continuation(
        prob, p_span=(0.5, 1.5),
        solvers=jc.Solvers(linear=linear_solver, eigen=eigen_solver),
    )

    assert result.branch.n_valid > 1
    assert len(linear_solver.calls) > 0
    assert len(eigen_solver.calls) > 0


def test_continuation_default_solvers_matches_prior_behavior():
    # Exercises the default Solvers() end to end through continuation() --
    # the public-API analogue of Task 2's pseudo_arclength_scan baseline
    # test. Settings are pinned to match that captured baseline exactly
    # (ds=0.05, ds_min=1e-5, ds_max=0.2, newton_tol=1e-6, max_steps=60,
    # newton_max_iter=20) rather than ContinuationPar's defaults, so the
    # same reference numbers apply.
    #
    # Re-captured 2026-08-19 (Phase 1 correctness hardening, Task 1) along
    # with test_pseudo_arclength_scan_matches_pre_protocol_baseline above --
    # see that test's comment for why slot-0 seed correction shifts this
    # trajectory (same u0/p0/settings, so the same new numbers apply).
    import jaxcont as jc

    prob = jc.bif_problem(lambda u, p, args: pitchfork(u, p), u0=jnp.array([0.1]), p0=0.5)
    settings = jc.ContinuationPar(
        ds=0.05, ds_min=1e-5, ds_max=0.2, max_steps=60,
        newton_tol=1e-6, newton_max_iter=20,
    )
    result = jc.continuation(prob, p_span=(0.5, 1.5), settings=settings)

    expected_params = [
        0.5, 0.550000011920929, 0.625, 0.737500011920929,
        0.90625, 1.1062500476837158, 1.3062500953674316, 1.5062501430511475,
    ]
    assert result.branch.params.tolist() == expected_params
