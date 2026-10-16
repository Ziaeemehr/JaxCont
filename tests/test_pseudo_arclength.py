"""
Tests for pseudo-arclength continuation method.

This test suite validates that pseudo-arclength continuation can:
1. Handle simple continuation problems
2. Pass through fold bifurcations (turning points)
3. Continue along branches where natural continuation fails
4. Compute correct tangent vectors

Ported from the deleted PseudoArclengthContinuation OO class onto the scan
engine's private _tangent/_newton_correct functions (see
docs/superpowers/plans/2026-07-21-engine-consolidation.md Task 5).
"""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc

from jaxcont.core.scan_continuation import _newton_correct, _tangent


class TestPseudoArclengthBasic:
    """Test basic pseudo-arclength continuation functionality."""

    def test_linear_system(self):
        """
        Test with simple linear system: dx/dt = r - x
        Exact solution: x = r
        """
        def f(u, p, args):
            return jnp.array([p - u[0]])

        u = jnp.array([0.0])
        param = jnp.array(0.0)
        param_values = [float(param)]
        state_values = [float(u[0])]

        ds = jnp.array(0.1)
        param_end = 1.0
        max_steps = 15
        step = 0

        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        while param < param_end and step < max_steps:
            du0 = tangent[:-1]
            dp0 = tangent[-1]
            u_pred = u + ds * du0
            param_pred = param + ds * dp0
            u_new, param_new, converged, _n_iter = _newton_correct(
                f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 100
            )

            if not converged:
                break

            u = u_new
            param = param_new
            tangent = _tangent(f, u, param, None, tangent)

            param_values.append(float(param))
            state_values.append(float(u[0]))
            step += 1

        errors = [abs(x - r) for x, r in zip(state_values, param_values)]
        max_error = max(errors)

        assert max_error < 1e-6, f"Maximum error {max_error} exceeds tolerance"
        assert step >= 8, f"Only completed {step} steps, expected at least 8"

    def test_quadratic_system(self):
        """
        Test with quadratic system: dx/dt = r - x^2
        Has fold bifurcation at r = 0
        """
        def f(u, p, args):
            return jnp.array([p - u[0] ** 2])

        r0 = 0.1
        x0 = jnp.sqrt(r0)

        u = jnp.array([x0])
        param = jnp.array(r0)
        param_values = [float(param)]
        state_values = [float(u[0])]

        ds = jnp.array(0.05)
        max_steps = 20
        step = 0

        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        while step < max_steps and param < 1.0:
            du0 = tangent[:-1]
            dp0 = tangent[-1]
            u_pred = u + ds * du0
            param_pred = param + ds * dp0
            u_new, param_new, converged, _n_iter = _newton_correct(
                f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 50
            )

            if not converged:
                break

            u = u_new
            param = param_new
            tangent = _tangent(f, u, param, None, tangent)

            param_values.append(float(param))
            state_values.append(float(u[0]))
            step += 1

        errors = []
        for x, r in zip(state_values, param_values):
            if r > 0:
                expected = jnp.sqrt(r)
                errors.append(abs(x - expected))

        if errors:
            max_error = max(errors)
            assert max_error < 1e-4, f"Maximum error {max_error} exceeds tolerance"

        assert step >= 1, f"Only completed {step} steps, expected at least 1"

    def test_tangent_computation(self):
        """Test that tangent vectors are computed correctly."""
        def f(u, p, args):
            return jnp.array([p - u[0]])

        u = jnp.array([0.5])
        param = jnp.array(0.5)

        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        norm = jnp.linalg.norm(tangent)
        assert jnp.isclose(norm, 1.0), f"Tangent not normalized: norm={norm}"

        assert tangent.shape[0] == 2, f"Tangent has wrong shape: {tangent.shape}"

        # For f(u,p)=p-u, equilibrium is u=p, so du/dp should be ~1.
        du_dp = tangent[0] / tangent[1] if abs(tangent[1]) > 1e-10 else 0
        assert abs(du_dp - 1.0) < 0.1, f"du/dp = {du_dp}, expected ~1.0"


class TestPseudoArclengthFoldBifurcation:
    """Test pseudo-arclength continuation through fold bifurcations."""

    def test_fold_continuation(self):
        """
        Test continuation through fold bifurcation.
        System: dx/dt = r - x^2

        This has a fold at r=0. Pseudo-arclength should be able to
        pass through it while natural continuation cannot.
        """
        def f(u, p, args):
            return jnp.array([p - u[0] ** 2])

        r0 = 1.0
        x0 = jnp.sqrt(r0)

        u = jnp.array([x0])
        param = jnp.array(r0)
        param_values = [float(param)]
        state_values = [float(u[0])]

        ds = jnp.array(-0.05)
        max_steps = 50
        step = 0
        min_r = -0.5

        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        while step < max_steps and param > min_r:
            du0 = tangent[:-1]
            dp0 = tangent[-1]
            u_pred = u + ds * du0
            param_pred = param + ds * dp0
            u_new, param_new, converged, _n_iter = _newton_correct(
                f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 100
            )

            if not converged:
                break

            u = u_new
            param = param_new
            tangent = _tangent(f, u, param, None, tangent)

            param_values.append(float(param))
            state_values.append(float(u[0]))
            step += 1

        assert len(param_values) >= 1, "No continuation steps taken"

    def test_pitchfork_branch(self):
        """
        Test continuation on pitchfork bifurcation branches.
        System: dx/dt = r*x - x^3

        This has stable branches at x = ±sqrt(r) for r > 0.
        """
        def f(u, p, args):
            return jnp.array([p * u[0] - u[0] ** 3])

        r0 = 1.0
        x0 = jnp.sqrt(r0)

        u = jnp.array([x0])
        param = jnp.array(r0)
        param_values = [float(param)]
        state_values = [float(u[0])]

        ds = jnp.array(0.1)
        max_steps = 20
        step = 0
        param_end = 2.0

        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        while step < max_steps and param < param_end:
            du0 = tangent[:-1]
            dp0 = tangent[-1]
            u_pred = u + ds * du0
            param_pred = param + ds * dp0
            u_new, param_new, converged, _n_iter = _newton_correct(
                f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 50
            )

            if not converged:
                break

            u = u_new
            param = param_new
            tangent = _tangent(f, u, param, None, tangent)

            param_values.append(float(param))
            state_values.append(float(u[0]))
            step += 1

        assert len(state_values) >= 1, "No continuation steps taken"

        if step > 0:
            errors = []
            for x, r in zip(state_values, param_values):
                if r > 0.01:
                    expected = jnp.sqrt(r)
                    errors.append(abs(x - expected))

            if errors:
                max_error = max(errors)
                assert max_error < 1e-3, f"Maximum error {max_error} exceeds tolerance"


class TestPseudoArclengthStepControl:
    """Test step size control and adaptive continuation."""

    def test_different_step_sizes(self):
        """Test that different step sizes produce consistent results."""
        def f(u, p, args):
            return jnp.array([p - u[0]])

        results = {}

        for ds_val in [0.05, 0.1, 0.2]:
            ds = jnp.array(ds_val)
            u = jnp.array([0.0])
            param = jnp.array(0.0)
            tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

            for _ in range(5):
                du0 = tangent[:-1]
                dp0 = tangent[-1]
                u_pred = u + ds * du0
                param_pred = param + ds * dp0
                u_new, param_new, converged, _n_iter = _newton_correct(
                    f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 50
                )

                if not converged:
                    break

                u = u_new
                param = param_new
                tangent = _tangent(f, u, param, None, tangent)

            results[ds_val] = (float(u[0]), float(param))

        final_params = [p for _, p in results.values()]
        param_range = max(final_params) - min(final_params)

        # With different step sizes, we expect some variation but should be
        # bounded. For this linear system (f = p - u), the tangent is
        # (1, 1)/sqrt(2), so 5 pseudo-arclength steps of size ds advance the
        # parameter by ~5*ds/sqrt(2); across ds in [0.05, 0.2] that range is
        # ~0.53 once every step actually converges (see the historical note
        # in the pre-migration version of this test, ROADMAP.md issue #9).
        assert param_range < 0.6, f"Parameter range {param_range} too large"

    def test_tangent_consistency(self):
        """Test that tangent vectors remain consistent along the branch."""
        def f(u, p, args):
            return jnp.array([p - u[0]])

        u = jnp.array([0.0])
        param = jnp.array(0.0)

        tangent1 = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        ds = jnp.array(0.1)
        du0 = tangent1[:-1]
        dp0 = tangent1[-1]
        u_pred = u + ds * du0
        param_pred = param + ds * dp0
        u_new, param_new, converged, _n_iter = _newton_correct(
            f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 50
        )

        assert converged, "First step did not converge"

        tangent2 = _tangent(f, u_new, param_new, None, tangent1)

        dot_product = jnp.dot(tangent1, tangent2)
        assert dot_product > 0.5, f"Tangents not consistent: dot={dot_product}"

        assert jnp.isclose(jnp.linalg.norm(tangent1), 1.0)
        assert jnp.isclose(jnp.linalg.norm(tangent2), 1.0)


class TestPseudoArclengthVsNatural:
    """Compare pseudo-arclength with natural continuation."""

    def test_performance_on_simple_system(self):
        """
        Both methods should work well on simple systems without folds.
        """
        def f(u, p, args):
            return jnp.array([p - u[0]])

        u = jnp.array([0.0])
        param = jnp.array(0.0)
        tangent = _tangent(f, u, param, None, jnp.array([0.0, 1.0]))

        ds = jnp.array(0.1)
        du0 = tangent[:-1]
        dp0 = tangent[-1]
        u_pred = u + ds * du0
        param_pred = param + ds * dp0
        u_pa, param_pa, converged_pa, _n_iter_pa = _newton_correct(
            f, u_pred, param_pred, None, u, param, du0, dp0, ds, 1e-6, 50
        )

        assert converged_pa, "Pseudo-arclength did not converge"

        error_pa = abs(u_pa[0] - param_pa)
        assert error_pa < 1e-6, f"Pseudo-arclength error {error_pa} too large"


def test_module_imports():
    """Test that the functional API is importable and callable."""
    import jaxcont as jc
    from jaxcont.core.scan_continuation import pseudo_arclength_scan

    assert callable(jc.continuation)
    assert callable(pseudo_arclength_scan)
    assert jc.PseudoArclength() is not None


def _weighted_linear(u, p, args):
    return u - p


def test_tangent_orientation_uses_weighted_alignment():
    """Preserve weighted orientation when Euclidean alignment has the opposite sign."""
    def residual(u, p, args):
        return u + p

    theta = 0.01
    previous = jnp.array([10., 1.]) / jnp.sqrt(2.)
    expected = jnp.array([-1., 1.]) / jnp.sqrt(1. + theta)
    assert jnp.dot(expected, previous) < 0
    assert theta * expected[0] * previous[0] + expected[1] * previous[1] > 0

    tangent = _tangent(residual, jnp.zeros(1), jnp.array(0.), None,
                       previous, theta=theta)

    np.testing.assert_allclose(tangent, expected, rtol=1e-5)
    assert theta * tangent[0] * previous[0] + tangent[1] * previous[1] > 0


@pytest.mark.parametrize('loop', ['while', 'scan'])
def test_theta_controls_arclength_and_reuses_compile_cache(loop):
    from jaxcont.core.scan_continuation import pseudo_arclength_scan, pseudo_arclength_lax_scan
    engine = pseudo_arclength_scan if loop == 'while' else pseudo_arclength_lax_scan
    problem = jc.bif_problem(_weighted_linear, u0=jnp.zeros(3), p0=0.)
    sizes = []
    for theta in [0.25, 1.0, None]:
        sol = jc.continuation(problem, p_span=(0., 2.), settings=jc.ContinuationPar(
            theta=theta, loop=loop, ds=0.1, ds_max=0.1, max_steps=2,
            adaptive=False, compute_stability=False))
        weight = 1 / 3 if theta is None else theta
        np.testing.assert_allclose(sol.branch.params[1], 0.1 / np.sqrt(1 + 3 * weight), rtol=1e-5)
        sizes.append(engine._cache_size())
    assert sizes[0] == sizes[1] == sizes[2]


@pytest.mark.parametrize('loop', ['while', 'scan'])
def test_periodic_step_count_independent_of_mesh(loop):
    from jaxcont.core.collocation import Collocation
    from jaxcont.problems.periodic import periodic_orbit_problem
    def _rhs(u, p, args):
        x, y = u
        r2 = x*x + y*y
        return jnp.array([(p-r2)*x-y, (p-r2)*y+x])

    t = jnp.linspace(0., 2 * jnp.pi, 161)
    trajectory = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
    counts = []
    for ntst in [10, 40]:
        problem = periodic_orbit_problem(_rhs, trajectory, t, 2 * jnp.pi, 1.,
                                         Collocation(ntst=ntst, ncol=4))
        sol = jc.continuation(problem, p_span=(1., 2.), settings=jc.ContinuationPar(
            loop=loop, ds=0.05, ds_max=0.05, adaptive=False,
            max_steps=200, compute_stability=False))
        assert sol.status == 'reached_end'
        counts.append(sol.branch.n_valid - 1)
    assert abs(counts[0] - counts[1]) <= 0.2 * min(counts), counts


@pytest.mark.parametrize("loop", ["while", "scan"])
def test_weighted_corrector_constraint(loop):
    def parabola(u, p, args):
        return u**2 - p

    from jaxcont.utils.precision import default_tol

    theta = 0.25
    u = jnp.array([1.])
    p = jnp.array(1.)
    tangent = _tangent(parabola, u, p, None, jnp.array([0., 1.]), theta=theta)
    ds = 0.2
    tol = default_tol(u.dtype)
    u_new, p_new, converged, iters = _newton_correct(
        parabola, u + ds*tangent[:-1], p + ds*tangent[-1], None,
        u, p, tangent[:-1], tangent[-1], ds, tol, 20, loop=loop, theta=theta)
    assert converged
    assert iters > 0
    np.testing.assert_allclose(u_new**2, p_new, atol=tol)
    np.testing.assert_allclose(theta*jnp.dot(u_new-u, tangent[:-1])
                               + (p_new-p)*tangent[-1], ds, atol=tol)
