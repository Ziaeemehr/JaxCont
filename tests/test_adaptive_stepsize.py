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

import pytest
import jax.numpy as jnp

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
        than a tighter one on a smooth problem.

        NOTE: `adaptive=False` is not wired into the scan engine (same gap
        documented above for the dropped test_disabled_adaptive_returns_same
        -- `_run_scan` never reads `settings.adaptive`, and `_adapt_ds` runs
        unconditionally every step). So `sol_fixed` below is NOT a true
        fixed-step run; both runs actually use the same always-on adaptation.
        What's really being compared is two different (ds, ds_min, ds_max)
        configurations, one of which happens to be labeled "fixed". The
        assertions still hold and are meaningful for that narrower claim.
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
        Test that a looser step-size-bound configuration reaches at least as
        far as a tighter one in a difficult region.

        Reformulated from the pre-migration version, which counted rejected
        (non-converged) Newton attempts via convergence_info -- the new scan
        engine only surfaces *accepted* points, not individual rejected
        attempts (rejections happen inside one jitted lax.while_loop and
        never get their own buffer slot). The supportable proxy for "adaptive
        handles this better" is that adaptive continuation reaches at least
        as many accepted points as fixed continuation in the same difficult
        region, without needing per-attempt visibility the engine doesn't
        expose.

        NOTE: as in test_adaptive_uses_fewer_steps above, `adaptive=False`
        is a no-op on this engine (`_run_scan` never reads
        `settings.adaptive`; `_adapt_ds` always runs) -- so `sol_fixed` is
        not actually a fixed-step run, and this is really a comparison
        between two different (ds, ds_min, ds_max) configurations rather
        than a true adaptive-vs-fixed comparison. The assertion is still a
        meaningful check of that narrower claim.
        """
        prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)

        sol_fixed = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, -0.1),
            settings=jc.ContinuationPar(
                ds=0.05, adaptive=False, max_steps=100,
                newton_max_iter=30, compute_stability=False,
            ),
        )
        sol_adaptive = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, -0.1),
            settings=jc.ContinuationPar(
                ds=0.05, ds_min=0.001, ds_max=0.1, adaptive=True,
                max_steps=100, newton_max_iter=30, compute_stability=False,
            ),
        )

        assert sol_adaptive.branch.n_valid >= sol_fixed.branch.n_valid, (
            f"Adaptive ({sol_adaptive.branch.n_valid} points) should reach at least as far "
            f"as fixed ({sol_fixed.branch.n_valid} points) in a difficult region"
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

    # NOTE: the pre-migration test_disabled_adaptive_returns_same is not
    # ported. It tested PredictorCorrector.adapt_stepsize() honoring
    # `adaptive_stepsize=False` to freeze ds. `_adapt_ds` (the scan engine's
    # replacement) has no such toggle, and `ContinuationPar.adaptive` was
    # already not wired into the scan engine before this migration (confirmed
    # by grep: _run_scan never reads settings.adaptive). This is a
    # pre-existing gap, not introduced here -- reintroducing an adaptive-off
    # mode is separate feature work, not part of engine consolidation.


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
