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
    assert float(jnp.min(jnp.abs(sol.branch.states[:, 0]))) >= 0.05 - 1e-6


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


def test_branch_switch_is_differentiable_in_args():
    def imperfect_pitchfork(u, p, b):
        return jnp.array([p * u[0] - u[0] ** 3 + b])

    event = _find_bp(lambda u, p, a: imperfect_pitchfork(u, p, 0.0), jnp.array([0.0]), -0.5, (-0.5, 0.5))

    def new_branch_start_state(b):
        new_prob = branch_switch(imperfect_pitchfork, event, args=b, ds=0.05)
        return new_prob.u0[0]

    b0 = jnp.array(0.0)
    analytic = jax.grad(new_branch_start_state)(b0)

    eps = 1e-3
    finite_diff = (new_branch_start_state(b0 + eps) - new_branch_start_state(b0 - eps)) / (2 * eps)

    assert jnp.isfinite(analytic)
    assert float(jnp.abs(analytic - finite_diff)) < 2e-2


def test_branch_switch_vmap_batches_over_a_second_parameter():
    def shifted_pitchfork(u, p, b):
        shifted_u = u[0] - b
        return jnp.array([p * shifted_u - shifted_u ** 3])

    def switch_for_b(b):
        event = _find_bp(lambda u, p, a: shifted_pitchfork(u, p, b), jnp.array([b]), -0.5, (-0.5, 0.5))
        new_prob = branch_switch(shifted_pitchfork, event, args=b, ds=0.05)
        return new_prob.u0[0], new_prob.p0

    # _find_bp calls continuation() with events=[...], which is eager-only
    # (see events.py's module docstring) -- vmap only the branch_switch()
    # call itself, using a pre-detected event per batch element built
    # outside the trace. This matches how events=[...] under vmap is
    # documented as unsupported elsewhere in this codebase.
    bs = jnp.linspace(-0.05, 0.05, 4)
    events = [
        _find_bp(lambda u, p, a: shifted_pitchfork(u, p, float(b)), jnp.array([b]), -0.5, (-0.5, 0.5))
        for b in bs
    ]
    u_stars = jnp.array([e.u[0] for e in events])
    p_stars = jnp.array([e.p for e in events])
    tangents = jnp.array([e.info["tangent"] for e in events])

    def switch_one(u_star, p_star, tangent, b):
        from jaxcont.bifurcations.events import EventHit
        fake_event = EventHit(kind="branch_point", p=p_star, u=jnp.array([u_star]),
                              index=(0, 1), info={"tangent": tangent})
        new_prob = branch_switch(shifted_pitchfork, fake_event, args=b, ds=0.05)
        return new_prob.u0[0], new_prob.p0

    batched = jax.vmap(switch_one)(u_stars, p_stars, tangents, bs)
    assert batched[0].shape == (4,)
    for i, b in enumerate(bs):
        serial_u, serial_p = switch_for_b(float(b))
        assert float(batched[0][i]) == pytest.approx(float(serial_u), abs=1e-4)
        assert float(batched[1][i]) == pytest.approx(float(serial_p), abs=1e-4)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e6])
@pytest.mark.parametrize("branch_point", [False, True], ids=["fold", "transcritical"])
def test_bp_invariant_under_rhs_scaling(scale, branch_point):
    from jaxcont.bifurcations.events import BranchPoint

    def rhs(u, p):
        value = u[0] * (p - u[0]) if branch_point else u[0] ** 2 + p
        return jnp.array([scale * value])

    # The midpoint is the exact singular equilibrium for both normal forms.
    left = BranchPoint(p=-0.1, u=jnp.zeros(1))
    right = BranchPoint(p=0.1, u=jnp.zeros(1), tangent=jnp.array([0.0, 1.0]))
    hit = BP().refine(
        left, right, (0, 1), rhs, tolerance=None, max_iterations=50,
    )
    hits = [] if hit is None else [hit]
    assert len(hits) == int(branch_point)
    if branch_point:
        assert hit.status == "validated"
        assert hit.kind == "branch_point"


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("rank_tolerance", [None, 0.005, 0.02])
def test_bp_rank_tolerance_controls_relative_rank(scale, rank_tolerance):
    from jaxcont.bifurcations.events import BranchPoint

    def rhs(u, p):
        return scale * jnp.array([u[0] ** 2 + 0.01 * p, u[1]])

    # At the exact fold the augmented singular values are scale * [1, 0.01].
    left = BranchPoint(p=-0.1, u=jnp.zeros(2))
    right = BranchPoint(p=0.1, u=jnp.zeros(2))
    hit = BP(rank_tolerance=rank_tolerance).refine(
        left, right, (0, 1), rhs, tolerance=None, max_iterations=50,
    )
    if rank_tolerance is None:
        accepted = False
    else:
        accepted = rank_tolerance == 0.02
    assert (hit is not None) == accepted
    if accepted:
        assert hit.status == "validated"
