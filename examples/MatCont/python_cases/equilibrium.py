"""Analytic equilibrium-continuation validation cases."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxcont as jc

from . import CaseResult


def _cubic_rhs(u, parameter, args):
    del args
    x = u[0]
    return jnp.array([parameter + x - x**3 / 3.0])


def _van_der_pol_rhs(u, parameter, args):
    del args
    x, y = u
    return jnp.array([y, parameter * (1.0 - x**2) * y - x])


def _adaptive_control_rhs(u, alpha, beta):
    """MatCont ``adaptx`` equilibrium system, with ``alpha`` free."""
    beta = 1.0 if beta is None else beta
    x, y, z = u
    return jnp.array([y, z, -alpha * z - beta * y - x + x**2])


def _maximum_residual(rhs, states, parameters, args=None) -> float:
    residuals = jax.vmap(rhs, in_axes=(0, 0, None))(states, parameters, args)
    return float(jnp.max(jnp.abs(residuals)))


def run_cubic_fold() -> CaseResult:
    """Continue the cubic S-curve and refine both analytic folds."""
    # The historical example uses x=-2 as a convenient Newton guess.  Case
    # artifacts include the seed itself, so use the refined analytic root to
    # keep every exported branch point on f(x, -1)=0.
    problem = jc.bif_problem(
        _cubic_rhs, u0=jnp.array([-2.1038034027355366]), p0=-1.0
    )
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(
            ds=0.01,
            max_steps=300,
            newton_tol=1e-6,
            compute_stability=True,
        ),
        events=[jc.Fold()],
    )
    fold_events = [event for event in result.events if event.kind == "fold"]
    refined = [jc.fold_point(_cubic_rhs, event.u, event.p) for event in fold_events]
    actual = sorted(
        [(float(state[0]), float(parameter), vector) for state, parameter, vector in refined],
        key=lambda item: item[0],
    )
    expected = [(-1.0, 2.0 / 3.0), (1.0, -2.0 / 3.0)]
    fold_errors = [
        max(abs(state - expected_state), abs(parameter - expected_parameter))
        for (state, parameter, _), (expected_state, expected_parameter) in zip(actual, expected)
    ]
    coefficients = [
        float(jc.fold_coefficient(_cubic_rhs, jnp.array([state]), parameter, vector))
        for state, parameter, vector in actual
    ]
    solution = result._solution
    return CaseResult(
        case_id="MC-EQ-001",
        checks={
            "fold_count": len(fold_events),
            "max_fold_error": max(fold_errors, default=float("inf")),
            "max_residual": _maximum_residual(
                _cubic_rhs, solution.states, solution.parameters
            ),
            "fold_coefficients_nonzero": all(abs(value) > 0.1 for value in coefficients),
        },
        observations={
            "folds": [(state, parameter) for state, parameter, _ in actual],
            "fold_coefficients": coefficients,
            "n_points": solution.n_points,
        },
        artifacts={
            "states": solution.states,
            "parameters": solution.parameters,
            "stability": solution.stability,
        },
    )


def run_vanderpol_hopf() -> CaseResult:
    """Validate the degenerate Van der Pol equilibrium Hopf at ``mu=0``."""
    problem = jc.bif_problem(
        _van_der_pol_rhs, u0=jnp.array([0.0, 0.0]), p0=-2.0
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
    hopf_events = [event for event in result.events if event.kind == "hopf"]
    if hopf_events:
        state, parameter, q1, q2, omega = jc.hopf_point(
            _van_der_pol_rhs, hopf_events[0].u, hopf_events[0].p, tol=1e-7
        )
        lyapunov = jc.lyapunov_coefficient(
            _van_der_pol_rhs, state, parameter, q1, q2, omega
        )
        hopf_error = max(float(jnp.max(jnp.abs(state))), abs(float(parameter)))
        frequency_error = abs(abs(float(omega)) - 1.0)
    else:
        state = jnp.full((2,), jnp.nan)
        parameter = omega = lyapunov = jnp.array(jnp.nan)
        hopf_error = frequency_error = float("inf")
    solution = result._solution
    return CaseResult(
        case_id="MC-EQ-002",
        checks={
            "hopf_count": len(hopf_events),
            "max_hopf_error": hopf_error,
            "frequency_error": frequency_error,
            "max_residual": _maximum_residual(
                _van_der_pol_rhs, solution.states, solution.parameters
            ),
        },
        observations={
            "hopf_state": state,
            "hopf_parameter": parameter,
            "frequency": omega,
            "first_lyapunov": lyapunov,
            "n_points": solution.n_points,
        },
        artifacts={
            "states": solution.states,
            "parameters": solution.parameters,
            "stability": solution.stability,
        },
    )


def run_adaptive_control_hopf() -> CaseResult:
    """Validate MatCont's adaptive-control equilibrium Hopf analytically."""
    beta = jnp.array(1.0)
    problem = jc.bif_problem(
        _adaptive_control_rhs,
        u0=jnp.zeros(3),
        p0=-2.0,
        args=beta,
    )
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(-2.0, 2.0),
        settings=jc.ContinuationPar(
            ds=0.03,
            ds_max=0.08,
            max_steps=100,
            newton_tol=1e-7,
            compute_stability=True,
        ),
        events=[jc.Hopf()],
    )
    hopf_events = [event for event in result.events if event.kind == "hopf"]
    if hopf_events:
        state, parameter, q1, q2, omega = jc.hopf_point(
            _adaptive_control_rhs,
            hopf_events[0].u,
            hopf_events[0].p,
            beta,
            tol=1e-7,
        )
        lyapunov = jc.lyapunov_coefficient(
            _adaptive_control_rhs, state, parameter, q1, q2, omega, beta
        )
        hopf_error = max(
            float(jnp.max(jnp.abs(state))), abs(float(parameter) - 1.0)
        )
        frequency_error = abs(abs(float(omega)) - 1.0)
    else:
        state = jnp.full((3,), jnp.nan)
        parameter = omega = lyapunov = jnp.array(jnp.nan)
        hopf_error = frequency_error = float("inf")
    solution = result._solution
    return CaseResult(
        case_id="MC-EQ-003",
        checks={
            "hopf_count": len(hopf_events),
            "max_hopf_error": hopf_error,
            "frequency_error": frequency_error,
            "max_residual": _maximum_residual(
                _adaptive_control_rhs,
                solution.states,
                solution.parameters,
                beta,
            ),
        },
        observations={
            "hopf_state": state,
            "hopf_parameter": parameter,
            "frequency": omega,
            "first_lyapunov": lyapunov,
            "n_points": solution.n_points,
        },
        artifacts={
            "states": solution.states,
            "parameters": solution.parameters,
            "stability": solution.stability,
        },
    )


__all__ = [
    "run_adaptive_control_hopf",
    "run_cubic_fold",
    "run_vanderpol_hopf",
]
