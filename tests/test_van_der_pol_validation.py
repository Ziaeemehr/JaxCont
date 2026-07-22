"""Analytic and MatCont cross-validation for example 03."""

import jax
import jax.numpy as jnp

import jaxcont as jc


def _van_der_pol_rhs(u, mu, args):
    x, y = u
    return jnp.array([y, mu * (1.0 - x**2) * y - x])


def test_van_der_pol_hopf_matches_matcont_and_analytic_invariants():
    """MatCont 7.6 reference: one H at (x, y, mu) = (0, 0, 0)."""
    problem = jc.bif_problem(
        _van_der_pol_rhs,
        u0=jnp.array([0.0, 0.0]),
        p0=-2.0,
    )
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(-2.0, 2.0),
        settings=jc.ContinuationPar(
            ds=0.02,
            ds_max=0.05,
            max_steps=160,
            newton_tol=1e-7,
            compute_stability=True,
        ),
        events=[jc.Hopf()],
    )
    solution = result._solution
    hopf_events = [event for event in result.events if event.kind == "hopf"]

    assert len(hopf_events) == 1
    hopf = hopf_events[0]
    assert abs(hopf.p) < 5e-4
    assert float(jnp.linalg.norm(hopf.u, ord=jnp.inf)) < 1e-6

    jacobian = jax.jacfwd(_van_der_pol_rhs, argnums=0)(
        jnp.array([0.0, 0.0]), jnp.array(0.0), None,
    )
    eigenvalues = jnp.linalg.eigvals(jacobian)
    assert bool(
        jnp.allclose(
            jnp.sort(jnp.imag(eigenvalues)), jnp.array([-1.0, 1.0]),
        )
    )
    assert bool(jnp.allclose(jnp.real(eigenvalues), 0.0, atol=1e-7))

    negative_mu = solution.parameters < -0.1
    positive_mu = solution.parameters > 0.1
    assert bool(jnp.all(solution.stability[negative_mu]))
    assert bool(jnp.all(~solution.stability[positive_mu]))

    residuals = jax.vmap(_van_der_pol_rhs, in_axes=(0, 0, None))(
        solution.states, solution.parameters, None,
    )
    assert float(jnp.max(jnp.abs(residuals))) < 2e-5
