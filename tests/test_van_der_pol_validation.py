"""Analytic and MatCont cross-validation for example 03."""

import jax
import jax.numpy as jnp

import jaxcont as jc


def _eps_J():
    return float(jnp.finfo(jnp.asarray(0.).dtype).eps)

_POINT_ATOL = 1e-9 if jnp.asarray(0.).dtype == jnp.float64 else 1e-5



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
    # Measured |error| x32/x64: 1.24e-10/4.42e-19; analytic Hopf parameter, default Newton budget.
    assert abs(hopf.p) < _POINT_ATOL
    # Measured |error| x32/x64: 0/0; analytic Hopf state, default Newton budget.
    assert float(jnp.linalg.norm(hopf.u, ord=jnp.inf)) < min(1e-06, _POINT_ATOL)

    jacobian = jax.jacfwd(_van_der_pol_rhs, argnums=0)(
        jnp.array([0.0, 0.0]), jnp.array(0.0), None,
    )
    eigenvalues = jnp.linalg.eigvals(jacobian)
    # Measured |error| x32/x64: 0/0; analytic unit Hopf eigenfrequency.
    assert bool(jnp.allclose(jnp.sort(jnp.imag(eigenvalues)), jnp.array([-1.0, 1.0]), atol=0, rtol=32 * _eps_J()))
    # Measured |error| x32/x64: 0/0; analytic imaginary Hopf pair has exactly zero real part.
    assert bool(jnp.allclose(jnp.real(eigenvalues), 0.0, atol=0, rtol=0))

    negative_mu = solution.parameters < -0.1
    positive_mu = solution.parameters > 0.1
    assert bool(jnp.all(solution.stability[negative_mu]))
    assert bool(jnp.all(~solution.stability[positive_mu]))

    residuals = jax.vmap(_van_der_pol_rhs, in_axes=(0, 0, None))(
        solution.states, solution.parameters, None,
    )
    assert float(jnp.max(jnp.abs(residuals))) < 2e-5
