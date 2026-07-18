"""
Tests for the functional API (`jaxcont.api`) and the JIT scan engine
(`jaxcont.core.scan_continuation`).

Kept fast: small `max_steps`, `tol=1e-6` (float32-reachable), tiny systems.
"""

import jax
import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.core.scan_continuation import pseudo_arclength_scan


# --- test systems ---------------------------------------------------------

def pitchfork(u, p, args):
    return p * u - u**3


def saddle_node(u, p, args):
    # equilibria u = ±sqrt(-p); fold (turning point) at p = 0
    return u**2 + p


def hopf_normal_form(u, p, args):
    # Linearization at u=0 has eigenvalues p ± i.
    return jnp.array([p * u[0] - u[1], u[0] + p * u[1]])


def _max_residual(f, states, params, args=None, skip_first=True):
    start = 1 if skip_first else 0  # slot 0 is the uncorrected initial guess
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
    def test_scan_is_default(self):
        assert jc.PseudoArclength().engine == "scan"

    @pytest.mark.parametrize("engine", ["legacy", "scan"])
    def test_pitchfork_basic(self, engine):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(engine=engine), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6),
        )
        assert sol.branch.n_valid > 5
        assert _max_residual(pitchfork, sol.branch.states, sol.branch.params) < 1e-5

    def test_legacy_scan_parity(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        kw = dict(p_span=(0.5, 1.5),
                  settings=jc.ContinuationPar(ds=0.05, max_steps=60, newton_tol=1e-6))
        legacy = jc.continuation(prob, jc.PseudoArclength(engine="legacy"), **kw)
        scan = jc.continuation(prob, jc.PseudoArclength(engine="scan"), **kw)
        # same terminal parameter to a few digits
        assert float(scan.branch.params[-1]) == pytest.approx(
            float(legacy.branch.params[-1]), abs=0.1
        )

    def test_scan_computes_stability(self):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=0.5)
        sol = jc.continuation(
            prob, jc.PseudoArclength(engine="scan"), p_span=(0.5, 1.5),
            settings=jc.ContinuationPar(ds=0.05, max_steps=60),
        )
        assert sol.branch.stable is not None
        assert sol.branch.stable.shape[0] == sol.branch.n_valid


class TestFolds:
    def test_scan_passes_and_detects_fold(self):
        prob = jc.bif_problem(saddle_node, u0=jnp.array([1.0]), p0=-1.0)
        sol = jc.continuation(
            prob, jc.PseudoArclength(engine="scan"), p_span=(-1.0, 0.2),
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
        assert hopf[0].info["method"] == "bisection"


# --- scan engine: vmap + boundedness -------------------------------------

class TestDifferentiableFold:
    # f(u, p; theta) = u^2 - theta*u + p
    #   fold at u = theta/2, p* = theta^2/4  ->  dp*/dtheta = theta/2
    @staticmethod
    def _f(u, p, theta):
        return jnp.array([u[0] ** 2 - theta * u[0] + p])

    def test_fold_location(self):
        u, p, v = jc.fold_point(self._f, jnp.array([0.4]), jnp.array(0.2),
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


class TestScanEngine:
    def test_vmap_batch(self):
        f = lambda u, p: pitchfork(u, p, None)

        def run(p0):
            return pseudo_arclength_scan(
                f, jnp.array([0.1]), p0, p0 + 1.0,
                jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
                jnp.array(1e-6), 80, jnp.array(20),
            )

        batch = jax.vmap(run)(jnp.linspace(0.5, 3.0, 16))
        assert batch.params.shape == (16, 81)
        assert batch.n_valid.shape == (16,)

    def test_bounded_on_saturating_branch(self):
        # r - tanh(x): the branch that hung the interim corrector must terminate.
        f = lambda u, p: jnp.array([p - jnp.tanh(u[0])])
        res = pseudo_arclength_scan(
            f, jnp.array([0.5]), jnp.array(0.5), jnp.array(1.5),
            jnp.array(0.005), jnp.array(0.001), jnp.array(0.2),
            jnp.array(1e-6), 200, jnp.array(20),
        )
        # n_valid is a concrete int and never exceeds the buffer -> it terminated
        assert 0 < int(res.n_valid) <= 201
