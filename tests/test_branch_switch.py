"""branch_switch() -- bifurcations/branch_switch.py."""

import jax
import jax.numpy as jnp
import pytest

import jaxcont as jc
from jaxcont.bifurcations.branch_switch import branch_switch
from jaxcont.bifurcations.events import BP


def _pitchfork(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 3])


def _transcritical(u, p, args):
    return jnp.array([p * u[0] - u[0] ** 2])


def _find_bp(f, u0, p0, p_span):
    prob = jc.bif_problem(f, u0=u0, p0=p0)
    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=p_span,
        settings=jc.ContinuationPar(ds=0.02, max_steps=60, newton_tol=1e-6),
        events=[BP()],
    )
    hits = [e for e in sol.events if e.kind == "branch_point"]
    assert len(hits) == 1
    return hits[0]


def test_switch_lands_on_the_pitchfork_second_branch():
    event = _find_bp(_pitchfork, jnp.array([0.0]), -0.5, (-0.5, 0.5))
    new_prob = branch_switch(_pitchfork, event, ds=0.05)
    # u = +-sqrt(p) -- sign is a genuine ambiguity (SVD's arbitrary sign
    # convention), so check magnitude only.
    assert abs(float(new_prob.u0[0])) == pytest.approx(
        float(jnp.sqrt(jnp.abs(new_prob.p0))), rel=0.2
    )
    assert float(new_prob.p0) > 0.0  # the pitchfork's second branch only exists for p > 0

    # The seed must actually be a valid equilibrium, not just "off the
    # trivial branch".
    residual = _pitchfork(new_prob.u0, new_prob.p0, None)
    assert float(jnp.max(jnp.abs(residual))) < 1e-4

    # And it must be continuable: continuing further should stay on
    # u != 0 (the new branch), not snap back onto the trivial one.
    sol = jc.continuation(
        new_prob, jc.PseudoArclength(), p_span=(float(new_prob.p0), float(new_prob.p0) + 0.3),
        settings=jc.ContinuationPar(ds=0.02, max_steps=30, newton_tol=1e-6),
    )
    assert float(jnp.min(jnp.abs(sol.branch.states[:, 0]))) > 0.05


def test_switch_lands_on_the_transcritical_second_branch():
    event = _find_bp(_transcritical, jnp.array([0.0]), -0.5, (-0.5, 0.5))
    new_prob = branch_switch(_transcritical, event, ds=0.05)
    # second branch: u = p
    assert float(new_prob.u0[0]) == pytest.approx(float(new_prob.p0), abs=0.1)
    residual = _transcritical(new_prob.u0, new_prob.p0, None)
    assert float(jnp.max(jnp.abs(residual))) < 1e-4


def test_branch_switch_rejects_a_non_branch_point_event():
    from jaxcont.bifurcations.events import EventHit
    fake = EventHit(kind="fold", p=0.0, u=jnp.array([0.0]), index=(0, 1), info={})
    with pytest.raises(ValueError, match="branch_point"):
        branch_switch(_pitchfork, fake)
