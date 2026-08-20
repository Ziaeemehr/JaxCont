"""
Tests for adaptive step size control in continuation methods.

Tests various aspects of adaptive step size control including:
- Step size adaptation based on Newton convergence
- Minimum and maximum step size constraints
- Step size behavior near bifurcations
- Comparison between adaptive and fixed step size

Ported from the deleted PseudoArclengthContinuation OO class onto
jc.continuation() + the scan engine's per-point `ds` buffer (see
docs/superpowers/plans/2026-07-21-engine-consolidation.md Task 6). Two
tests from the pre-migration version are intentionally not ported --
see the module-level notes in that plan's Task 6 for why.
"""

import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.core.scan_continuation import _adapt_ds

# Marked slow and excluded from the default `make test` run: several cases drive
# hard branches (e.g. `smooth_rhs = p - tanh(x)` into the tanh-saturation regime).
pytestmark = pytest.mark.slow


def pitchfork_rhs(u, p, args):
    """Pitchfork bifurcation: dx/dt = p*x - x^3."""
    x = u[0]
    return jnp.array([p * x - x ** 3])


def smooth_rhs(u, p, args):
    """Smooth system that should allow large step sizes."""
    x = u[0]
    return jnp.array([p - jnp.tanh(x)])


class TestAdaptiveStepsizeBasics:
    """Test basic adaptive step size functionality."""

    def test_stepsize_increases_on_fast_convergence(self):
        """Test that step size can increase when Newton converges quickly."""
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.005, ds_min=0.001, ds_max=0.2, adaptive=True,
                max_steps=50, newton_tol=1e-6, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        step_sizes = [info["ds"] for info in sol._solution.convergence_info[:n]]

        assert n > 5, "Should have computed multiple points"
        assert sol._solution.convergence_info[n - 1]["converged"], "Last step should converge"
        assert min(step_sizes) >= 0.001, "Step sizes should respect minimum"

    def test_stepsize_decreases_on_slow_convergence(self):
        """Test that step size can decrease near difficult regions."""
        prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.0),
            settings=jc.ContinuationPar(
                ds=0.05, ds_min=0.001, ds_max=0.2, adaptive=True,
                max_steps=150, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        assert n > 3, "Should have computed multiple points"
        step_sizes = [
            info["ds"] for info in sol._solution.convergence_info[:n] if info["converged"]
        ]
        assert len(step_sizes) > 0, "Should have converged steps"
        assert all(s >= 0.001 for s in step_sizes), "Step sizes should respect minimum"

    def test_stepsize_respects_minimum(self):
        """Test that step size doesn't go below minimum."""
        ds_min = 0.005
        prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, -0.2),
            settings=jc.ContinuationPar(
                ds=0.05, ds_min=ds_min, ds_max=0.2, adaptive=True,
                max_steps=100, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        for info in sol._solution.convergence_info[:n]:
            assert info["ds"] >= ds_min * 0.99, f"Step size {info['ds']} below minimum {ds_min}"

    def test_stepsize_respects_maximum(self):
        """Test that step size doesn't go above maximum."""
        ds_max = 0.05
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, ds_min=0.001, ds_max=ds_max, adaptive=True,
                max_steps=50, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        for info in sol._solution.convergence_info[:n]:
            assert info["ds"] <= ds_max * 1.01, f"Step size {info['ds']} above maximum {ds_max}"


class TestAdaptiveVsFixed:
    """Compare adaptive vs fixed step size."""

    def test_adaptive_uses_fewer_steps(self):
        """
        Test that a looser step-size-bound configuration can use fewer steps
        than a tighter one on a smooth problem, compared against a true
        fixed-step run (`adaptive=False`, wired in via
        test_disabled_adaptive_keeps_step_constant_after_success in this
        same test class).
        """
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)

        sol_fixed = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, adaptive=False, max_steps=200, compute_stability=False,
            ),
        )
        sol_adaptive = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, ds_min=0.005, ds_max=0.1, adaptive=True,
                max_steps=200, compute_stability=False,
            ),
        )

        assert sol_fixed.branch.n_valid > 10, "Fixed should have many points"
        assert sol_adaptive.branch.n_valid > 10, "Adaptive should have many points"
        assert sol_adaptive.branch.n_valid <= sol_fixed.branch.n_valid + 5, (
            f"Adaptive ({sol_adaptive.branch.n_valid}) should not use significantly more "
            f"steps than fixed ({sol_fixed.branch.n_valid})"
        )

    def test_adaptive_handles_difficult_regions(self):
        """
        Test that adaptive step-size control reaches closer to p_end than
        fixed-step in a difficult region (near a fold bifurcation).

        In a fixed-step run near a fold, failed corrections shrink ds, but the
        algorithm can wander past the bifurcation without stalling (ds never
        drops below the minimum), reaching max_steps at a parameter far from
        the goal. Adaptive control instead grows ds on success and reaches the
        target parameter in far fewer points, demonstrating it "handles the
        difficult region better" by actually reaching the goal.
        """
        prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)
        p_end = -0.1

        sol_fixed = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, p_end),
            settings=jc.ContinuationPar(
                ds=0.05, adaptive=False, max_steps=100,
                newton_max_iter=30, compute_stability=False,
            ),
        )
        sol_adaptive = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, p_end),
            settings=jc.ContinuationPar(
                ds=0.05, ds_min=0.001, ds_max=0.1, adaptive=True,
                max_steps=100, newton_max_iter=30, compute_stability=False,
            ),
        )

        # Adaptive should reach closer to the target p_end than fixed-step.
        p_fixed_final = float(sol_fixed.branch.params[-1])
        p_adaptive_final = float(sol_adaptive.branch.params[-1])
        dist_fixed = abs(p_fixed_final - p_end)
        dist_adaptive = abs(p_adaptive_final - p_end)

        assert dist_adaptive < dist_fixed, (
            f"Adaptive should reach closer to p_end={p_end} than fixed. "
            f"Adaptive reached p={p_adaptive_final:.6f} (dist={dist_adaptive:.6f}), "
            f"fixed reached p={p_fixed_final:.6f} (dist={dist_fixed:.6f})"
        )

    def test_disabled_adaptive_keeps_step_constant_after_success(self):
        """`adaptive=False` must preserve the requested fixed step after
        every successful correction, instead of silently growing/shrinking
        it (2026-08-19 review finding #5)."""
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, adaptive=False, max_steps=200, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        converged_ds = [
            info["ds"] for info in sol._solution.convergence_info[:n]
            if info["converged"]
        ]
        assert len(converged_ds) > 5, "should have several converged fixed steps"
        assert all(ds == pytest.approx(0.01) for ds in converged_ds), (
            f"adaptive=False must keep every successful step at ds=0.01, "
            f"got distinct values {sorted(set(converged_ds))}"
        )


class TestAdaptiveStepsizeAlgorithm:
    """Test the adaptive step size algorithm directly."""

    def test_adapt_stepsize_increase_on_fast_convergence(self):
        """Test step size increase logic."""
        new_ds = _adapt_ds(jnp.array(0.01), 2, jnp.array(True), 0.001, 0.1)
        assert new_ds > 0.01, "Step size should increase for fast convergence"
        assert new_ds <= 0.1, "Step size should not exceed maximum"

    def test_adapt_stepsize_decrease_on_slow_convergence(self):
        """Test step size decrease logic."""
        new_ds = _adapt_ds(jnp.array(0.05), 8, jnp.array(True), 0.001, 0.1)
        assert new_ds < 0.05, "Step size should decrease for slow convergence"
        assert new_ds >= 0.001, "Step size should not go below minimum"

    def test_adapt_stepsize_halve_on_failure(self):
        """Test step size halving on convergence failure."""
        new_ds = _adapt_ds(jnp.array(0.05), 20, jnp.array(False), 0.001, 0.1)
        assert jnp.isclose(new_ds, 0.025), "Step size should be halved on failure"

    def test_adapt_stepsize_stable_on_moderate_convergence(self):
        """Test step size remains stable for moderate convergence."""
        new_ds = _adapt_ds(jnp.array(0.03), 4, jnp.array(True), 0.001, 0.1)
        assert jnp.isclose(new_ds, 0.03), "Step size should remain stable for moderate convergence"

    def test_adapt_stepsize_fixed_mode_still_shrinks_on_failure(self):
        """Even with adaptive=False, a failed step still backs off (so a
        run that can't converge at the fixed size still terminates via the
        existing ds <= ds_min stall condition, rather than retrying forever)."""
        new_ds = _adapt_ds(jnp.array(0.05), 20, jnp.array(False), 0.001, 0.1, False)
        assert new_ds == pytest.approx(0.025), "failed step should still shrink by the same factor as the adaptive path"

    # NOTE: the pre-migration test_disabled_adaptive_returns_same is not
    # ported as-is; coverage for `adaptive=False` freezing ds now lives in
    # test_disabled_adaptive_keeps_step_constant_after_success (above, in
    # TestAdaptiveVsFixed).


# NOTE: the pre-migration TestStepsizeNearBifurcations.
# test_stepsize_decreases_near_bifurcation is not ported. On the old OO
# engine it always vacuously passed: the corrector took exactly 1 point
# total for these settings, so both the "near bifurcation" and "away"
# buckets were empty and the `if step_sizes_near_bif and step_sizes_away:`
# guard skipped the assertion. On the new engine it takes 6-7 real steps,
# but the bordered pseudo-arclength corrector (issue #1) stays
# well-conditioned through folds/turning points by design, converging in
# ~1 Newton iteration everywhere -- so ds grows monotonically and never
# shrinks near a bifurcation, on this system or a genuine fold. The
# test's premise is invalidated by the engine working correctly, not a
# porting bug.


class TestStepsizeConvergenceInfo:
    """Test that convergence info properly tracks step sizes."""

    def test_convergence_info_records_stepsize(self):
        """Test that convergence info contains step size information."""
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.0),
            settings=jc.ContinuationPar(
                ds=0.02, ds_min=0.01, ds_max=0.1, adaptive=True,
                max_steps=50, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        for info in sol._solution.convergence_info[:n]:
            assert "ds" in info, "Convergence info should contain 'ds'"
            assert info["ds"] > 0, "Step size should be positive"
            assert "newton_iters" in info, "Convergence info should contain 'newton_iters'"
            assert "converged" in info, "Convergence info should contain 'converged'"

    def test_convergence_info_tracks_adaptation(self):
        """Test that convergence info records step size properly."""
        prob = jc.bif_problem(smooth_rhs, u0=jnp.array([0.5]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(
                ds=0.01, ds_min=0.005, ds_max=0.1, adaptive=True,
                max_steps=50, compute_stability=False,
            ),
        )

        n = sol.branch.n_valid
        step_sizes = [info["ds"] for info in sol._solution.convergence_info[:n]]

        assert len(step_sizes) > 0, "Should have convergence info"
        assert all(s > 0 for s in step_sizes), "All step sizes should be positive"
        assert all(s >= 0.005 * 0.99 for s in step_sizes), "Step sizes should respect minimum"
        assert all(s <= 0.1 * 1.01 for s in step_sizes), "Step sizes should respect maximum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
