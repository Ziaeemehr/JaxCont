"""
Tests for the functional API (`jaxcont.api`) and the JIT scan engine
(`jaxcont.core.scan_continuation`).

Kept fast: small `max_steps`, `tol=1e-6` (float32-reachable), tiny systems.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.core.scan_continuation import Status, natural_scan, pseudo_arclength_scan
from jaxcont.solvers.protocols import Dense


def test_bp_and_branch_switch_are_exported_at_top_level():
    assert hasattr(jc, "BP")
    assert hasattr(jc, "branch_switch")
    assert jc.BP().kind == "branch_point"


def test_branch_switch_pd_is_exported_at_top_level():
    import jaxcont as jc

    assert hasattr(jc, "branch_switch_pd")
    assert "branch_switch_pd" in jc.__all__


# --- test systems ---------------------------------------------------------

def pitchfork(u, p, args):
    return p * u - u**3


def saddle_node(u, p, args):
    # equilibria u = ±sqrt(-p); fold (turning point) at p = 0
    return u**2 + p


def hopf_normal_form(u, p, args):
    # Linearization at u=0 has eigenvalues p ± i.
    return jnp.array([p * u[0] - u[1], u[0] + p * u[1]])


def _max_residual(f, states, params, args=None, skip_first=False):
    start = 1 if skip_first else 0
    vals = [jnp.abs(f(states[i], params[i], args)).max() for i in range(start, states.shape[0])]
    return float(jnp.max(jnp.array(vals))) if vals else 0.0


# --- BifProblem -----------------------------------------------------------

class TestBifProblem:
    def test_factory_coerces_arrays(self):
        prob = jc.bif_problem(pitchfork, u0=[0.1], p0=0.5)
        assert isinstance(prob.u0, jax.Array)
        assert prob.p0.dtype == prob.u0.dtype
        assert prob.kind == "equilibrium"

    def test_at_override_is_functional(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        prob2 = prob.at(p0=0.8)
        assert float(prob2.p0) == pytest.approx(0.8)
        assert float(prob.p0) == pytest.approx(0.5)  # original unchanged

    def test_at_keep_sentinel_preserves_args(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5, args={"k": 3.0})
        prob2 = prob.at(p0=0.9)  # args not passed -> kept
        assert prob2.args == {"k": 3.0}

    def test_as_rhs_bridge(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.2]), p0=0.7)
        rhs = prob.as_rhs(0.7)
        u = jnp.array([0.2])
        assert jnp.allclose(rhs(u), pitchfork(u, 0.7, None))

    def test_pytree_roundtrip(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        leaves, treedef = jax.tree_util.tree_flatten(prob)
        prob2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(prob2.u0, prob.u0)
        assert float(prob2.p0) == float(prob.p0)


# --- continuation() (both engines) ---------------------------------------

class TestContinuation:
    def test_pitchfork_basic(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        )
        assert sol.branch.n_valid > 5
        assert _max_residual(pitchfork, sol.branch.states, sol.branch.params) < 1e-5

    def test_scan_computes_stability(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60),
        )
        assert sol.branch.stable is not None
        assert sol.branch.stable.shape[0] == sol.branch.n_valid

    def test_natural_dispatches_to_scan_engine(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.Natural(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        )
        assert sol.branch.n_valid > 5
        assert _max_residual(pitchfork, sol.branch.states, sol.branch.params) < 1e-5

    def test_pseudo_arclength_has_no_engine_field(self):
        assert not hasattr(jc.PseudoArclength(), "engine")

    def test_convergence_info_has_ds(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        )
        assert sol._solution.convergence_info[0]["ds"] > 0


class TestFolds:
    def test_scan_passes_and_detects_fold(self):
        prob = jc.bif_problem(saddle_node, u0=jnp.array([1.0]), p0=-1.0)
        sol = jc.continuation(
            prob, jc.PseudoArclength(), p_span=(-1.0, 0.2),
            settings=jc.ContinuationPar(ds=0.05, max_steps=200, newton_tol=1e-6),
            events=[jc.Fold()],
        )
        # branch turns around at the fold: parameter should not exceed ~0
        assert float(sol.branch.params.max()) < 0.05
        folds = [e for e in sol.events if e.kind == "fold"]
        assert len(folds) >= 1
        assert abs(folds[0].p) < 1e-3  # true fold at p = 0
        assert folds[0].info["method"] == "extended_system"


class TestHopf:
    def test_scan_detects_and_refines_hopf(self):
        prob = jc.bif_problem(hopf_normal_form, u0=jnp.zeros(2), p0=-1.0)
        sol = jc.continuation(
            prob,
            p_span=(-1.0, 1.0),
            settings=jc.ContinuationPar(ds=0.05, max_steps=80, newton_tol=1e-6),
            events=[jc.Hopf()],
        )
        hopf = [event for event in sol.events if event.kind == "hopf"]
        assert len(hopf) == 1
        assert abs(hopf[0].p) < 1e-4
        assert hopf[0].info["method"] == "extended_system"


# --- scan engine: vmap + boundedness -------------------------------------

class TestDifferentiableFold:
    # f(u, p; theta) = u^2 - theta*u + p
    #   fold at u = theta/2, p* = theta^2/4  ->  dp*/dtheta = theta/2
    @staticmethod
    def _f(u, p, theta):
        return jnp.array([u[0] ** 2 - theta * u[0] + p])

    def test_fold_location(self):
        u, p, v, _converged = jc.fold_point(self._f, jnp.array([0.4]), jnp.array(0.2),
                                jnp.array(1.0))
        assert float(u[0]) == pytest.approx(0.5, abs=1e-4)
        assert float(p) == pytest.approx(0.25, abs=1e-4)
        assert abs(float(v[0])) == pytest.approx(1.0, abs=1e-4)

    def test_reverse_mode_grad_matches_analytic(self):
        pstar = lambda th: jc.fold_parameter(self._f, jnp.array([0.4]),
                                             jnp.array(0.2), th)
        for theta in (1.0, 2.5):
            g = jax.grad(pstar)(jnp.array(theta))
            assert float(g) == pytest.approx(theta / 2.0, abs=1e-4)

    def test_vector_parameter_jacobian(self):
        # f(u,p;a,b) = u^2 - a u + b p  ->  p* = a^2/(4b)
        def f2(u, p, ab):
            return jnp.array([u[0] ** 2 - ab[0] * u[0] + ab[1] * p])

        ab = jnp.array([1.0, 2.0])
        J = jax.jacobian(
            lambda x: jc.fold_parameter(f2, jnp.array([0.4]), jnp.array(0.2), x)
        )(ab)
        # d/da = a/(2b) = 0.25 ; d/db = -a^2/(4b^2) = -0.0625
        assert float(J[0]) == pytest.approx(0.25, abs=1e-4)
        assert float(J[1]) == pytest.approx(-0.0625, abs=1e-4)


def imperfect_pitchfork(u, p, b):
    return jnp.array([p * u[0] - u[0] ** 3 + b])


class TestVmapSafety:
    """Issue #13: jc.continuation() must not crash inside jax.vmap/jax.jit."""

    def _run(self, b, *, events=()):
        prob = jc.bif_problem(imperfect_pitchfork, u0=jnp.array([0.05]), p0=-1.0, args=b)
        return jc.continuation(
            prob, p_span=(-1.0, 1.0),
            settings=jc.ContinuationPar(ds=0.05, max_steps=40, newton_tol=1e-6),
            events=events,
        )

    def test_vmap_returns_fixed_size_buffers_with_valid_mask(self):
        bs = jnp.linspace(-0.2, 0.2, 5)
        res = jax.vmap(self._run)(bs)

        buf = 41  # max_steps + 1
        assert res.branch.params.shape == (5, buf)
        assert res.branch.states.shape == (5, buf, 1)
        assert res.branch.valid.shape == (5, buf)
        assert res.branch.valid.dtype == jnp.bool_
        assert bool(jnp.all(res.branch.valid[:, 0]))  # slot 0 is always the seed point
        assert res.events == []
        assert res.stats["n_valid"].shape == (5,)
        assert jnp.all(res.stats["n_valid"] > 0)

    def test_vmap_with_events_returns_candidates(self):
        bs = jnp.linspace(-0.2, 0.2, 3)
        result = jax.vmap(lambda b: self._run(b, events=[jc.Fold()]))(bs)
        assert result.events == []
        assert result.event_candidates[0].bracket_mask.shape == (3, 40, 1)

    def test_eager_call_unaffected(self):
        sol = self._run(0.1)
        assert sol.branch.valid is None
        assert sol.branch.n_valid > 0
        assert sol.branch.params.shape == (sol.branch.n_valid,)


class TestScanEngine:
    def test_vmap_batch(self):
        f = pitchfork

        def run(p0):
            return pseudo_arclength_scan(
                f, None, jnp.array([0.1]), p0, p0 + 1.0, None,
                jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
                jnp.array(1e-6), 80, jnp.array(20),
            )

        batch = jax.vmap(run)(jnp.linspace(0.5, 3.0, 16))
        assert batch.params.shape == (16, 81)
        assert batch.n_valid.shape == (16,)

    def test_bounded_on_saturating_branch(self):
        # r - tanh(x): the branch that hung the interim corrector must terminate.
        f = lambda u, p, args: jnp.array([p - jnp.tanh(u[0])])
        res = pseudo_arclength_scan(
            f, None, jnp.array([0.5]), jnp.array(0.5), jnp.array(1.5), None,
            jnp.array(0.005), jnp.array(0.001), jnp.array(0.2),
            jnp.array(1e-6), 200, jnp.array(20),
        )
        # n_valid is a concrete int and never exceeds the buffer -> it terminated
        assert 0 < int(res.n_valid) <= 201

    def test_ds_buffer_records_stepsize_per_point(self):
        f = pitchfork

        res = pseudo_arclength_scan(
            f, None, jnp.array([0.1]), jnp.array(0.5), jnp.array(1.5), None,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-6), 60, jnp.array(20),
        )
        n = int(res.n_valid)
        assert res.ds.shape == (61,)
        assert float(res.ds[0]) == pytest.approx(0.05)  # slot 0 = initial ds0
        assert bool(jnp.all(res.ds[:n] >= 1e-5 - 1e-9))
        assert bool(jnp.all(res.ds[:n] <= 0.2 + 1e-9))


class TestNaturalScanEngine:
    def test_tracks_linear_branch(self):
        # f(u, p) = p - u  ->  equilibrium u = p exactly, no fold anywhere.
        f = lambda u, p, args: jnp.array([p - u[0]])

        res = natural_scan(
            f, None, jnp.array([0.0]), jnp.array(0.0), jnp.array(1.0), None,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-6), 40, jnp.array(20),
        )
        n = int(res.n_valid)
        assert n > 5
        assert bool(jnp.all(res.converged[:n]))
        # accuracy: u should equal p at every accepted point
        assert float(jnp.max(jnp.abs(res.states[:n, 0] - res.params[:n]))) < 1e-5

    def test_stalls_at_fold(self):
        # f(u, p) = p - u^2  ->  fold at p=0; natural continuation (fixed p,
        # solve for u) cannot pass it -- the branch must stop short of p=0.
        f = lambda u, p, args: jnp.array([p - u[0] ** 2])

        res = natural_scan(
            f, None, jnp.array([1.0]), jnp.array(1.0), jnp.array(-1.0), None,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-6), 60, jnp.array(20),
        )
        n = int(res.n_valid)
        last_p = float(res.params[n - 1])
        assert last_p > 0.0, (
            f"natural continuation should stall before reaching the fold at "
            f"p=0, but reached p={last_p}"
        )

    def test_vmap_batch(self):
        f = lambda u, p, args: jnp.array([p - u[0]])

        def run(p0):
            return natural_scan(
                f, None, jnp.array([0.0]), p0, p0 + 1.0, None,
                jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
                jnp.array(1e-6), 40, jnp.array(20),
            )

        batch = jax.vmap(run)(jnp.linspace(0.0, 2.0, 8))
        assert batch.params.shape == (8, 41)
        assert batch.n_valid.shape == (8,)


def test_p_span_mismatch_raises_for_equilibrium_problem():
    # Regression for finding #1: p_span[0] == problem.p0 used to be checked
    # only for fold_curve/hopf_curve kinds, so an ordinary equilibrium
    # problem silently started its branch at a point that was never on the
    # refined curve.
    def f(u, p, args):
        return u - p

    prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=jnp.array(0.0))
    with pytest.raises(ValueError, match="p_span"):
        jc.continuation(prob, p_span=(1.0, 1.2))


def test_invalid_seed_is_newton_corrected_not_silently_accepted():
    # Regression for finding #1: an invalid u0 (residual != 0 at p0) used to
    # be copied into branch slot 0 and marked converged unconditionally.
    def f(u, p, args):
        return u - p - 1.0  # true equilibrium: u = p + 1

    prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=jnp.array(0.0))  # f(0, 0) = -1, invalid
    result = jc.continuation(
        prob, p_span=(0.0, 0.5), settings=jc.ContinuationPar(max_steps=20),
    )
    assert result.branch.n_valid >= 1
    assert float(result.branch.states[0, 0]) == pytest.approx(1.0, abs=1e-4)
    assert _max_residual(f, result.branch.states, result.branch.params) < 1e-5


@pytest.mark.parametrize("scan_fn", [pseudo_arclength_scan, natural_scan])
def test_seed_correction_failure_falls_back_to_finite_u0(scan_fn):
    # Regression for the Important #1 review finding: when the seed's own
    # Newton correction fails to converge, its raw (possibly non-finite)
    # output used to be written into branch slot 0 anyway, with only
    # `converged=False` as an out-of-band signal. u0=0 is a genuine
    # zero-derivative starting point for f(u, p) = u**2 + p (df/du = 2u = 0),
    # so the uncorrected Newton step is a division by zero -- the raw
    # corrected output is -inf. states[0] must instead fall back to the
    # caller's original, finite u0, with seed_converged still False.
    def f(u, p, args):
        return u ** 2 + p

    u0 = jnp.array([0.0])
    p0 = jnp.array(1.0)  # no real root exists for p > 0
    result = scan_fn(
        f, None, u0, p0, jnp.array(2.0), None,
        jnp.array(0.01), jnp.array(1e-6), jnp.array(0.1),
        jnp.array(1e-8), 5, jnp.array(50),
    )
    assert bool(jnp.all(jnp.isfinite(result.states[0])))
    assert jnp.allclose(result.states[0], u0)
    assert not bool(result.converged[0])


def test_convergence_info_reports_real_newton_iteration_counts():
    """Regression test for the fabricated newton_iters=0 diagnostic
    (2026-08-19 review finding #10)."""
    def rhs(u, p, args):
        return u ** 3 - p * u

    prob = jc.bif_problem(rhs, u0=jnp.array([0.01]), p0=-1.0)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=50, compute_stability=False),
    )

    iters = [info["newton_iters"] for info in sol._solution.convergence_info]
    assert any(i > 0 for i in iters), (
        "at least one accepted step should report a nonzero Newton iteration "
        "count -- every entry was 0"
    )


def test_verbose_prints_a_bifurcation_summary(capsys):
    """Regression test for the unused verbose=True flag (2026-08-19 review
    finding #10)."""
    def rhs(u, p, args):
        return u ** 2 - p

    prob = jc.bif_problem(rhs, u0=jnp.array([0.5]), p0=0.25)
    jc.continuation(
        prob, jc.PseudoArclength(), p_span=(0.25, -0.25),
        settings=jc.ContinuationPar(ds=0.05, max_steps=50, compute_stability=True),
        events=[jc.Fold()],
        verbose=True,
    )

    captured = capsys.readouterr()
    assert captured.out.strip() != "", "verbose=True should print a summary"


class TestBothside:
    """Issue #8: continuation() only explored one direction from p0 per call.

    ``ContinuationPar(bothside=True)`` runs the existing single-direction
    scan engine twice (toward each bound of ``p_span``) and stitches the
    two legs into one branch -- no changes to core/scan_continuation.py.
    """

    def test_covers_both_directions_from_p0(self):
        # f(u, p) = p*u - u**3: u=0 is a trivial equilibrium for every p,
        # so both legs continue cleanly with no bifurcation in the way.
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.0]), p0=-0.5)
        sol = jc.continuation(
            prob, p_span=(-1.5, 0.5),
            settings=jc.ContinuationPar(
                ds=0.05, max_steps=60, newton_tol=1e-6, bothside=True,
            ),
        )
        params = sol.branch.params
        assert float(jnp.min(params)) <= -1.4
        assert float(jnp.max(params)) >= 0.4
        assert _max_residual(pitchfork, sol.branch.states, params) < 1e-5
        # exactly one point should sit at the shared seed p0 -- stitching
        # must not duplicate it.
        assert int(jnp.sum(jnp.isclose(params, -0.5, atol=1e-6))) == 1

    def test_asymmetric_p_span(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.0]), p0=-0.5)
        sol = jc.continuation(
            prob, p_span=(-2.0, 0.5),
            settings=jc.ContinuationPar(
                ds=0.05, max_steps=80, newton_tol=1e-6, bothside=True,
            ),
        )
        params = sol.branch.params
        assert float(jnp.min(params)) <= -1.9
        assert float(jnp.max(params)) >= 0.4

    def test_p0_outside_span_raises(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.0]), p0=1.0)
        with pytest.raises(ValueError, match="p0"):
            jc.continuation(
                prob, p_span=(-1.0, 0.5),
                settings=jc.ContinuationPar(bothside=True),
            )

    def test_events_detected_on_both_legs(self):
        # f(u, p) = u**2 - p has a fold at p=0; starting past it (p0=1) and
        # sweeping bothside through p_span=(-1, 2) should still find it, even
        # though bothside's stitch seam sits at p0=1, nowhere near the fold.
        def f(u, p, args):
            return jnp.array([u[0] ** 2 - p])

        prob = jc.bif_problem(f, u0=jnp.array([1.0]), p0=1.0)
        sol = jc.continuation(
            prob, p_span=(-1.0, 2.0),
            settings=jc.ContinuationPar(
                ds=0.05, max_steps=60, newton_tol=1e-6, bothside=True,
            ),
            events=[jc.Fold()],
        )
        folds = [e for e in sol.events if e.kind == "fold"]
        assert len(folds) == 1
        assert folds[0].p == pytest.approx(0.0, abs=0.02)

    def test_raises_under_vmap(self):
        def f(u, p, args):
            return u - p

        def run(p0):
            prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=p0)
            return jc.continuation(
                prob, p_span=(-1.0, 1.0),
                settings=jc.ContinuationPar(bothside=True, max_steps=20),
            )

        with pytest.raises(NotImplementedError, match="bothside"):
            jax.vmap(run)(jnp.array([0.0, 0.1]))


def _quad_p(u, p, args):
    return u ** 2 + p


def test_repeated_calls_hit_compile_cache():
    prob = jc.bif_problem(_quad_p, u0=jnp.array([1.0]), p0=-1.0, args={"a": jnp.array(1.0)})
    settings = jc.ContinuationPar(max_steps=10, compute_stability=False)

    def run(prob):
        return jc.continuation(prob, p_span=(-1.0, -0.5), settings=settings)

    run(prob)
    size = pseudo_arclength_scan._cache_size()
    run(prob)
    run(prob)
    assert pseudo_arclength_scan._cache_size() == size
    run(prob.at(args={"a": jnp.array(2.0)}))
    assert pseudo_arclength_scan._cache_size() == size


def test_grad_wrt_args_through_engine_forward():
    # The seed u0 is deliberately off-root: Newton must act on it for the
    # derivative to reach slot 0.
    # u - a*p = 0  ->  u(p) = a*p, so states/params == a at every valid point
    # and d/da sum(valid * states/params) == n_valid.
    def f(u, p, args):
        return u - args * p

    def run(a):
        return pseudo_arclength_scan(
            f, None, jnp.array([0.3]), jnp.array(0.5), jnp.array(1.5), a,
            jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
            jnp.array(1e-4), 30, jnp.array(20),
        )

    def observable(a):
        res = run(a)
        valid = jnp.arange(res.states.shape[0]) < res.n_valid
        ratio = res.states[:, 0] / jnp.where(valid, res.params, 1.0)
        return jnp.sum(ratio * valid)

    a = jnp.array(2.0)
    n = int(run(a).n_valid)
    assert n > 3
    assert float(jax.jacfwd(observable)(a)) == pytest.approx(float(n), rel=1e-3)


def _mixed_f(u, p, args):
    scale, fn = args
    return fn(u) + p * scale


def test_mixed_array_and_callable_args_cache_and_match_reference():
    sq = lambda u: u ** 2
    prob = jc.bif_problem(_mixed_f, u0=jnp.array([1.0]), p0=-1.0, args=(jnp.array(1.0), sq))
    settings = jc.ContinuationPar(max_steps=10, compute_stability=False)

    def run(pr):
        return jc.continuation(pr, p_span=(-1.0, -0.5), settings=settings)

    r1 = run(prob)
    size = pseudo_arclength_scan._cache_size()
    r2 = run(prob)
    assert pseudo_arclength_scan._cache_size() == size
    ref = jc.bif_problem(lambda u, p, a: u ** 2 + p * 1.0, u0=jnp.array([1.0]), p0=-1.0)
    r3 = run(ref)
    assert jnp.allclose(r1.branch.states, r2.branch.states)
    assert jnp.allclose(r1.branch.states, r3.branch.states)
    assert jnp.allclose(r1.branch.params, r3.branch.params)


def test_python_int_arg_leaf_is_concrete_inside_engine():
    def f(u, p, args):
        return u ** 2 + p + jnp.zeros(args["n"])  # needs a concrete int

    prob = jc.bif_problem(f, u0=jnp.array([1.0]), p0=-1.0, args={"n": 1})
    sol = jc.continuation(
        prob, p_span=(-1.0, -0.5),
        settings=jc.ContinuationPar(max_steps=10, compute_stability=False),
    )
    assert sol.branch.n_valid > 1


def _seed_failing_problem():
    # u0 = 0 is a zero-derivative start for u**2 + p, and no real root exists at p0 = 1.
    return jc.bif_problem(lambda u, p, args: u ** 2 + p, u0=jnp.array([0.0]), p0=1.0)


def test_seed_failure_raises_eagerly():
    with pytest.raises(ValueError, match="seed"):
        jc.continuation(_seed_failing_problem(), p_span=(1.0, 2.0))


def test_seed_failure_traced_marks_nothing_valid():
    res = jax.jit(
        lambda: jc.continuation(
            _seed_failing_problem(), p_span=(1.0, 2.0),
            settings=jc.ContinuationPar(max_steps=5, compute_stability=False),
        )
    )()
    assert int(res.branch.valid.sum()) == 0
    assert int(res.stats["status"]) == Status.SEED_FAILED


def _curved_run(**kw):
    # u = 17.5 p**2 with a single Newton iteration: the corrector fails above ds ~ 0.012.
    f = lambda u, p, args: jnp.array([u[0] - 17.5 * p ** 2])
    prob = jc.bif_problem(f, u0=jnp.array([0.0]), p0=0.0)
    base = dict(ds=0.015, ds_min=0.01, ds_max=0.015, max_steps=30, newton_max_iter=1,
                newton_tol=1e-5, compute_stability=False)
    base.update(kw)
    return jc.continuation(prob, p_span=(0.0, 0.15), settings=jc.ContinuationPar(**base))


def test_stall_only_after_failure_at_ds_min():
    res = _curved_run()
    steps = [c["ds"] for c in res._solution.convergence_info[1:] if c["converged"]]
    assert any(d == pytest.approx(0.01) for d in steps)
    assert res.status == "reached_end"


def test_initial_ds_is_clipped():
    prob = jc.bif_problem(lambda u, p, args: u - p, u0=jnp.array([0.0]), p0=0.0)
    res = jc.continuation(
        prob, p_span=(0.0, 1.0),
        settings=jc.ContinuationPar(ds=0.5, ds_max=0.1, max_steps=5, compute_stability=False),
    )
    assert res._solution.convergence_info[1]["ds"] <= 0.1 * (1 + 1e-6)


def test_max_attempts_bounds_rejections():
    res = _curved_run(max_attempts=5)
    assert res.status == "max_attempts"
    assert res.stats["n_attempts"] == 5
    assert res.stats["status"] == "max_attempts"


def test_traced_n_valid_counts_mask():
    def run():
        prob = jc.bif_problem(lambda u, p, args: u - p, jnp.array([0.0]), 0.0)
        out = jc.continuation(
            prob, p_span=(0.0, 1.0),
            settings=jc.ContinuationPar(max_steps=400, compute_stability=False),
        )
        return out, out.branch.n_valid

    out, traced_count = jax.jit(run)()
    assert int(out.branch.valid.sum()) < out.branch.params.size
    assert int(out.branch.n_valid) == int(out.branch.valid.sum())
    assert int(traced_count) == int(out.branch.valid.sum())


def test_at_param_ignores_padding():
    branch = jc.Branch(
        params=jnp.array([1.0, 2.0, 0.0]),
        states=jnp.array([[10.0], [20.0], [0.0]]),
        valid=jnp.array([True, True, False]),
    )
    p, u = branch.at_param(0.0)
    assert p == 1.0
    assert float(u[0]) == 10.0


@pytest.mark.parametrize("crossing", [-0.5, 0.5])
def test_bothside_event_index_points_at_event(crossing):
    def f(u, p, args):
        return hopf_normal_form(u, p - crossing, args)

    prob = jc.bif_problem(f, jnp.zeros(2), 0.0)
    out = jc.continuation(
        prob, p_span=(-2.0, 1.0), events=[jc.Hopf()],
        settings=jc.ContinuationPar(ds=0.05, max_steps=60, bothside=True),
    )
    assert len(out.events) == 1
    for e in out.events:
        params = out.branch.params
        assert abs(params[e.index[0]] - e.p) <= abs(params[e.index[0]] - params[e.index[1]])


def test_bothside_tangents_follow_stitched_direction():
    prob = jc.bif_problem(lambda u, p, args: u - p, jnp.array([0.0]), 0.0)
    out = jc.continuation(
        prob, p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(max_steps=60, bothside=True, compute_stability=False),
    )
    tangents = out.branch.tangents
    assert bool(jnp.all(jnp.sum(tangents[:-1] * tangents[1:], axis=1) > 0))


def _shifted(u, p, args):
    return u - args["c"] * args["k"]


def _shift_by_one(args, u, p):
    return {"c": u / args["k"] + 1.0, "k": args["k"]}


def test_scan_applies_step_update_to_the_seed_and_each_accepted_step():
    # u = c, and step_update sets c = u + 1: the seed is corrected with the
    # original c, every later point with the update of the point before it.
    res = natural_scan(
        _shifted, _shift_by_one, jnp.array([0.3]), jnp.array(0.0), jnp.array(1.0),
        {"c": jnp.array([2.0]), "k": jnp.array(1.0)},
        jnp.array(0.1), jnp.array(1e-5), jnp.array(0.1),
        jnp.array(1e-4), 5, jnp.array(20), Dense(), jnp.array(False),
    )
    n = int(res.n_valid)
    assert n == 6
    np.testing.assert_allclose(np.asarray(res.states[:n, 0]), 2.0 + np.arange(n), rtol=1e-5)


def test_step_update_sees_full_args_and_keeps_compile_cache():
    # "k" is a Python scalar, bound statically; step_update still sees it.
    prob = jc.bif_problem(
        _shifted, u0=jnp.array([0.0]), p0=0.0,
        args={"c": jnp.array([1.0]), "k": 2.0}, step_update=_shift_by_one,
    )
    settings = jc.ContinuationPar(ds=0.1, ds_max=0.1, max_steps=4, compute_stability=False)
    res = jc.continuation(prob, jc.Natural(), p_span=(0.0, 1.0), settings=settings)
    np.testing.assert_allclose(
        np.asarray(res.branch.states[:, 0]), 2.0 * (1.0 + np.arange(5)), rtol=1e-5
    )
    size = natural_scan._cache_size()
    jc.continuation(prob, jc.Natural(), p_span=(0.0, 1.0), settings=settings)
    assert natural_scan._cache_size() == size


def test_step_update_may_not_change_static_args():
    prob = jc.bif_problem(
        _shifted, u0=jnp.array([1.0]), p0=0.0,
        args={"c": jnp.array([1.0]), "k": 1.0},
        step_update=lambda a, u, p: {"c": a["c"], "k": 3.0},
    )
    with pytest.raises(ValueError, match="only change the array leaves"):
        jc.continuation(
            prob, p_span=(0.0, 1.0),
            settings=jc.ContinuationPar(max_steps=3, compute_stability=False),
        )


@partial(jax.jit, static_argnums=(1, 2))
def _loop_objective(a, loop, alg=jc.PseudoArclength()):
    problem = jc.bif_problem(lambda u, p, a: u - a * p,
                             jnp.array([0.0]), 0.0, args=a)
    result = jc.continuation(
        problem, alg, p_span=(0.0, 0.3),
        settings=jc.ContinuationPar(loop=loop, max_steps=8, max_attempts=12,
                                   compute_stability=False, adaptive=False),
    )
    b = result.branch
    return jnp.sum(b.states * b.valid[:, None])


@pytest.mark.parametrize('alg', [jc.PseudoArclength(), jc.Natural()])
def test_grad_through_scan_loop(alg):
    a = jnp.array(1.3)
    expected = jax.jacfwd(lambda a: _loop_objective(a, 'while', alg))(a)
    actual = jax.grad(lambda a: _loop_objective(a, 'scan', alg))(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_grad_through_while_loop_raises_clear_error():
    with pytest.raises(ValueError, match="loop='scan'"):
        jax.grad(lambda a: _loop_objective(a, 'while'))(jnp.array(1.3))


@pytest.mark.parametrize('alg', [jc.PseudoArclength(), jc.Natural()])
@pytest.mark.parametrize('loop', ['while', 'scan'])
def test_continuation_loop_jit_vmap(alg, loop):
    values = jnp.array([0.8, 1.3])
    actual = jax.jit(jax.vmap(lambda a: _loop_objective(a, loop, alg)))(values)
    expected = jnp.stack([_loop_objective(a, loop, alg) for a in values])
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


@pytest.mark.parametrize('engine', [pseudo_arclength_scan, natural_scan])
@pytest.mark.parametrize('case', ['end', 'steps', 'attempts', 'seed', 'nonfinite', 'stalled'])
def test_scan_loop_preserves_status_and_step_update(engine, case):
    from jaxcont.core.scan_continuation import pseudo_arclength_lax_scan, natural_lax_scan
    scan = pseudo_arclength_lax_scan if engine is pseudo_arclength_scan else natural_lax_scan
    def f(u, p, args):
        if case == 'seed':
            return jnp.ones_like(u)
        if case == 'nonfinite':
            return jnp.where(p > 0, jnp.nan, u - p)
        if case in ('attempts', 'stalled'):
            return u**2 - (1 - p)
        return u - p
    def update(args, u, p):
        return args + 1
    kwargs = dict(f=f, step_update=update, u0=jnp.array([1.0 if case in ('attempts', 'stalled') else 0.0]),
                  p0=0.0, p_end=0.01 if case == 'end' else 10.0,
                  args=jnp.array(0.0), ds0=0.1, ds_min=0.1 if case == 'stalled' else 0.001,
                  ds_max=0.1, tol=1e-5, max_steps=3,
                  max_iter=0 if case in ('attempts', 'stalled') else 5,
                  max_attempts=2 if case == 'attempts' else 12)
    before, after = engine(**kwargs), scan(**kwargs)
    for x, y in zip(before, after):
        np.testing.assert_allclose(x, y, rtol=1e-5, equal_nan=True)


def test_scan_forward_derivative_matches_while():
    a = jnp.array(1.3)
    np.testing.assert_allclose(
        jax.jacfwd(lambda a: _loop_objective(a, 'scan'))(a),
        jax.jacfwd(lambda a: _loop_objective(a, 'while'))(a), rtol=1e-5,
    )


def test_scan_step_update_changes_next_corrector_only_after_acceptance():
    from jaxcont.core.scan_continuation import natural_lax_scan
    def f(u, p, args):
        return u - p - args
    def update(args, u, p):
        return args + 1
    result = natural_lax_scan(
        f, update, jnp.array([0.0]), 0.0, 10.0, jnp.array(0.0),
        0.1, 0.001, 0.1, 1e-5, 2, 4, Dense(), False, 8,
    )
    # The seed refreshes args to 1; the first accept refreshes it to 2.
    np.testing.assert_allclose(result.states[:, 0], [0, 1.1, 2.2], rtol=1e-5)
    assert int(result.n_attempts) == 2
