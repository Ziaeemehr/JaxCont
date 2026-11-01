"""Eager event refinement reuses compiled code across calls (Loop 2, L2-F1).

A Python side effect in the right-hand side runs only while JAX traces it, so
counting calls counts traces. A second identical continuation must not trace
the right-hand side again, including in fold/Hopf refinement.
"""

import jax.numpy as jnp

import jaxcont as jc


def _counting(rhs):
    calls = []

    def f(u, p, args):
        calls.append(None)
        return rhs(u, p, args)

    return f, calls


def _cubic(u, p, args):
    return jnp.array([p + u[0] - u[0] ** 3 / 3])


def _hopf(u, p, args):
    x, y = u
    r2 = x * x + y * y
    return jnp.array([p * x - y - x * r2, x + p * y - y * r2])


def _traces_of_second_run(rhs, u0, p0, span, event):
    f, calls = _counting(rhs)
    problem = jc.bif_problem(f, u0=u0, p0=p0)
    settings = jc.ContinuationPar(ds=0.02, max_steps=200)
    first = jc.continuation(problem, p_span=span, settings=settings, events=[event])
    before = len(calls)
    second = jc.continuation(problem, p_span=span, settings=settings, events=[event])
    assert [e.p for e in first.events] == [e.p for e in second.events] and first.events
    return len(calls) - before


def test_fold_refinement_is_not_retraced():
    assert _traces_of_second_run(_cubic, jnp.array([-2.1038034027355366]), -1.0, (-1.0, 1.0), jc.Fold()) == 0


def test_hopf_refinement_is_not_retraced():
    assert _traces_of_second_run(_hopf, jnp.zeros(2), -0.5, (-0.5, 0.5), jc.Hopf()) == 0
