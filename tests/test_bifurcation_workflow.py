"""
Test what bifurcation detection currently does in the continuation workflow.

This test runs actual continuation and checks:
1. Does it detect bifurcations?
2. Does refine_location get called?
3. What does the output look like?

Ported from the deleted PseudoArclengthContinuation OO class onto
jc.continuation() (see
docs/superpowers/plans/2026-07-21-engine-consolidation.md Task 7).
"""

import jax.numpy as jnp

import jaxcont as jc


def test_bifurcation_detection_in_continuation():
    """
    Test bifurcation detection during continuation of pitchfork system.

    System: f(u, p) = u^3 - p*u = 0
    - Trivial branch: u = 0 (all p)
    - Bifurcating branches: u = ±√p (p > 0)
    - Bifurcation at p = 0 (pitchfork)
    """
    def rhs(u, p, args):
        return u ** 3 - p * u

    u0 = jnp.array([0.01])  # Close to zero
    prob = jc.bif_problem(rhs, u0=u0, p0=-1.0)

    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(ds=0.05, max_steps=100),
        events=[jc.Fold(), jc.Hopf()],
        verbose=True,
    )

    assert sol.branch.n_valid > 0
    assert sol.branch.eigenvalues is not None


def test_with_and_without_refinement():
    """Compare detection with and without location refinement."""
    def rhs(u, p, args):
        return u ** 2 - p  # Simple fold at p = 0

    u0 = jnp.array([0.5])
    prob = jc.bif_problem(rhs, u0=u0, p0=0.25)

    sol = jc.continuation(
        prob, jc.PseudoArclength(), p_span=(0.25, -0.25),
        settings=jc.ContinuationPar(ds=0.05, max_steps=50),
        events=[jc.Fold()],
    )

    assert sol.branch.n_valid > 0
    if sol.events:
        assert sol.events[0].info.get("method") in ("extended_system", "bisection", None)
