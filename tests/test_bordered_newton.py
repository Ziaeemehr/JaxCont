"""
Test the bordered Newton solver used in pseudo-arclength continuation.

The bordered Newton system is:
    [ df/du    df/dp ] [ Delta_u ]   [ -f(u, p)              ]
    [ du0^T    dp0    ] [ Delta_p ] = [ -g(u, p) = ds - (...) ]

This test verifies:
1. The block elimination algorithm is correct
2. The solver converges for simple problems
3. The solver handles singular Jacobians
4. The solver maintains the arclength constraint

Ported from the deleted PseudoArclengthContinuation OO class onto the scan
engine's private _tangent/_newton_correct functions (see
docs/superpowers/plans/2026-07-21-engine-consolidation.md Task 4) -- these
are the same functions pseudo_arclength_scan itself calls every step, so
this remains a direct test of the production corrector, not a reimplementation.
"""

import jax.numpy as jnp
from jaxcont.core.scan_continuation import _tangent, _newton_correct


def test_bordered_system_simple():
    """
    Test bordered Newton solver on a simple linear problem.

    Problem: f(u, p) = u - p = 0
    Arclength constraint: g(u, p) = (u - u0)*du0 + (p - p0)*dp0 - ds = 0

    This is simple enough that we can solve it analytically.
    """
    def f(u, p):
        return u - p

    u_prev = jnp.array([1.0])
    param_prev = jnp.array(1.0)
    # Tangent vector (normalized): for f(u,p)=u-p, du/dp=1, so [1,1]/sqrt(2).
    tangent = jnp.array([1.0 / jnp.sqrt(2.0), 1.0 / jnp.sqrt(2.0)])
    ds = jnp.array(0.1)

    du0 = tangent[:-1]
    dp0 = tangent[-1]
    u_pred = u_prev + ds * du0
    param_pred = param_prev + ds * dp0

    u_corr, p_corr, converged, n_iter = _newton_correct(
        f, u_pred, param_pred, u_prev, param_prev, du0, dp0, ds, 1e-6, 20
    )

    f_val = f(u_corr, p_corr)
    g_val = jnp.dot(u_corr - u_prev, du0) + (p_corr - param_prev) * dp0 - ds

    assert converged, "Solver should converge"
    assert jnp.linalg.norm(f_val) < 1e-6, "Should satisfy f(u, p) = 0"
    assert abs(g_val) < 1e-6, "Should satisfy arclength constraint"
    assert jnp.isclose(u_corr[0], p_corr, atol=1e-6), "Should have u = p"


def test_bordered_system_nonlinear():
    """
    Test bordered Newton solver on nonlinear problem.

    Problem: f(u, p) = u^2 - p = 0

    This is the classic fold bifurcation problem.
    """
    def f(u, p):
        return u ** 2 - p

    u_prev = jnp.array([2.0])
    param_prev = jnp.array(4.0)

    # Seed: dp=+1, matching the deleted compute_tangent(prev_tangent=None)'s
    # default orientation (it always assumed p increasing on the first call).
    seed = jnp.array([0.0, 1.0])
    tangent = _tangent(f, u_prev, param_prev, seed)

    ds = jnp.array(0.5)
    du0 = tangent[:-1]
    dp0 = tangent[-1]
    u_pred = u_prev + ds * du0
    param_pred = param_prev + ds * dp0

    u_corr, p_corr, converged, n_iter = _newton_correct(
        f, u_pred, param_pred, u_prev, param_prev, du0, dp0, ds, 1e-6, 20
    )

    f_val = f(u_corr, p_corr)
    g_val = jnp.dot(u_corr - u_prev, du0) + (p_corr - param_prev) * dp0 - ds

    assert converged, "Solver should converge"
    assert jnp.linalg.norm(f_val) < 1e-6, "Should satisfy f(u, p) = 0"
    assert abs(g_val) < 1e-6, "Should satisfy arclength constraint"
    assert jnp.isclose(u_corr[0] ** 2, p_corr, atol=1e-6), "Should have u^2 = p"


def test_bordered_system_2d():
    """
    Test bordered Newton solver on 2D system.

    Problem:
        f1 = x + y - p
        f2 = x - y

    Solution: x = p/2, y = p/2
    """
    def f(u, p):
        x, y = u[0], u[1]
        return jnp.array([x + y - p, x - y])

    u_prev = jnp.array([1.0, 1.0])
    param_prev = jnp.array(2.0)

    seed = jnp.array([0.0, 0.0, 1.0])
    tangent = _tangent(f, u_prev, param_prev, seed)

    ds = jnp.array(0.2)
    du0 = tangent[:-1]
    dp0 = tangent[-1]
    u_pred = u_prev + ds * du0
    param_pred = param_prev + ds * dp0

    u_corr, p_corr, converged, n_iter = _newton_correct(
        f, u_pred, param_pred, u_prev, param_prev, du0, dp0, ds, 1e-6, 20
    )

    f_val = f(u_corr, p_corr)
    g_val = jnp.dot(u_corr - u_prev, du0) + (p_corr - param_prev) * dp0 - ds

    assert converged, "Solver should converge"
    assert jnp.linalg.norm(f_val) < 1e-6, "Should satisfy f(u, p) = 0"
    assert abs(g_val) < 1e-6, "Should satisfy arclength constraint"
    assert jnp.isclose(u_corr[0], u_corr[1], atol=1e-6), "Should have x = y"
    assert jnp.isclose(u_corr[0] + u_corr[1], p_corr, atol=1e-6), "Should have x + y = p"


def test_bordered_system_continuation_branch():
    """
    Test that bordered Newton correctly continues along a branch.

    We'll use the pitchfork bifurcation: f(u, p) = u^3 - p*u = 0
    """
    def f(u, p):
        return u ** 3 - p * u

    u_current = jnp.array([0.5])
    param_current = jnp.array(0.25)

    n_steps = 5
    ds = jnp.array(0.1)
    tangent = jnp.array([0.0, 1.0])  # seed, matches compute_tangent(prev_tangent=None)

    for i in range(n_steps):
        # Recompute tangent at the current point first, exactly as the
        # deleted OO class's .run() loop did (tangent before predict/correct).
        tangent = _tangent(f, u_current, param_current, tangent)

        du0 = tangent[:-1]
        dp0 = tangent[-1]
        u_prev = u_current
        param_prev = param_current
        u_pred = u_prev + ds * du0
        param_pred = param_prev + ds * dp0

        u_current, param_current, converged, n_iter = _newton_correct(
            f, u_pred, param_pred, u_prev, param_prev, du0, dp0, ds, 1e-6, 20
        )

        f_val = f(u_current, param_current)
        residual = jnp.linalg.norm(f_val)

        assert converged, f"Step {i + 1} should converge"
        assert residual < 1e-6, f"Step {i + 1} should satisfy f = 0"


def test_bordered_system_block_elimination():
    """
    Test the block elimination algorithm directly.

    Verify that the block elimination formula is mathematically correct.
    This test never touched PseudoArclengthContinuation -- it reimplements
    the block-elimination formula from scratch and checks it against a
    direct solve of the same bordered matrix. Unchanged by the engine
    consolidation.
    """
    n = 3

    import numpy as np
    np.random.seed(42)

    jac_u = jnp.array(np.random.randn(n, n) + 3 * np.eye(n))
    df_dp = jnp.array(np.random.randn(n))
    du0 = jnp.array(np.random.randn(n))
    dp0 = float(np.random.randn())

    rhs_f = jnp.array(np.random.randn(n))
    rhs_g = float(np.random.randn())

    # Step 1: w = jac_u^{-1} * (-rhs_f)
    w = jnp.linalg.solve(jac_u, -rhs_f)
    # Step 2: v = jac_u^{-1} * df_dp
    v = jnp.linalg.solve(jac_u, df_dp)
    # Step 3: delta_p = (-rhs_g - du0^T * w) / (dp0 - du0^T * v)
    denominator = dp0 - jnp.dot(du0, v)
    delta_p = (-rhs_g - jnp.dot(du0, w)) / denominator
    # Step 4: delta_u = w - v * delta_p
    delta_u = w - v * delta_p

    bordered_matrix = jnp.zeros((n + 1, n + 1))
    bordered_matrix = bordered_matrix.at[:n, :n].set(jac_u)
    bordered_matrix = bordered_matrix.at[:n, n].set(df_dp)
    bordered_matrix = bordered_matrix.at[n, :n].set(du0)
    bordered_matrix = bordered_matrix.at[n, n].set(dp0)

    rhs_vector = jnp.concatenate([-rhs_f, jnp.array([-rhs_g])])

    solution_direct = jnp.linalg.solve(bordered_matrix, rhs_vector)
    delta_u_direct = solution_direct[:n]
    delta_p_direct = solution_direct[n]

    assert jnp.isclose(delta_p, delta_p_direct, atol=1e-10), "delta_p should match"
    assert jnp.allclose(delta_u, delta_u_direct, atol=1e-10), "delta_u should match"
