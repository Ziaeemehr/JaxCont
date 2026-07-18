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
        assert abs(folds[0].p) < 0.1  # true fold at p = 0


# --- scan engine: vmap + boundedness -------------------------------------

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
