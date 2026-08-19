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
    settings = jc.ContinuationPar(
        ds=0.01,
        ds_min=1e-5,
        ds_max=0.05,
        max_steps=300,
        newton_tol=1e-6,
        compute_stability=True,
    )
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(-1.0, 1.0),
        settings=settings,
        events=[jc.Fold()],
    )
    natural_result = jc.continuation(
        problem,
        jc.Natural(),
        p_span=(-1.0, 1.0),
        settings=settings,
    )
    fold_events = [event for event in result.events if event.kind == "fold"]
    refined = [jc.fold_point(_cubic_rhs, event.u, event.p) for event in fold_events]
    actual = sorted(
        [(float(state[0]), float(parameter), vector)
         for state, parameter, vector, _converged in refined],
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
    expected_coefficients = [1.0, -1.0]
    coefficient_error = max(
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(coefficients, expected_coefficients)
    )
    branch = result.branch
    natural_branch = natural_result.branch
    natural_max_parameter = float(jnp.max(natural_branch.params))
    stability_transitions = int(jnp.sum(branch.stable[1:] != branch.stable[:-1]))
    return CaseResult(
        case_id="MC-EQ-001",
        checks={
            "fold_count": len(fold_events),
            "max_fold_error": max(fold_errors, default=float("inf")),
            "max_residual": _maximum_residual(
                _cubic_rhs, branch.states, branch.params
            ),
            "natural_stalled_at_fold": (
                abs(natural_max_parameter - 2.0 / 3.0) < 5e-3
                and natural_max_parameter < 1.0 - 5e-3
            ),
            "palc_traversed_both_folds": (
                len(fold_events) == 2
                and float(jnp.min(branch.states[:, 0])) < -1.0
                and float(jnp.max(branch.states[:, 0])) > 1.0
            ),
            "max_fold_coefficient_error": coefficient_error,
            "stability_transition_count": stability_transitions,
        },
        observations={
            "folds": [(state, parameter) for state, parameter, _ in actual],
            "fold_coefficients": coefficients,
            "n_points": branch.n_valid,
            "natural_n_points": natural_branch.n_valid,
            "natural_max_parameter": natural_max_parameter,
        },
        artifacts={
            "states": branch.states,
            "parameters": branch.params,
            "stability": branch.stable,
            "spectra": branch.eigenvalues,
            "events": [
                {
                    "kind": "LP",
                    "parameter": parameter,
                    "state": [state],
                    "fold_coefficient": coefficient,
                }
                for (state, parameter, _), coefficient in zip(actual, coefficients)
            ],
            "event_spectra": [
                {
                    "kind": "LP",
                    "parameter": parameter,
                    "values": jnp.linalg.eigvals(
                        jax.jacfwd(_cubic_rhs, argnums=0)(
                            jnp.array([state]), parameter, None
                        )
                    ),
                }
                for state, parameter, _ in actual
            ],
            "natural_states": natural_branch.states,
            "natural_parameters": natural_branch.params,
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
        state, parameter, q1, q2, omega, _converged = jc.hopf_point(
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
    branch = result.branch
    negative_parameter = branch.params < -0.1
    positive_parameter = branch.params > 0.1
    return CaseResult(
        case_id="MC-EQ-002",
        checks={
            "hopf_count": len(hopf_events),
            "max_hopf_error": hopf_error,
            "frequency_error": frequency_error,
            "max_residual": _maximum_residual(
                _van_der_pol_rhs, branch.states, branch.params
            ),
            "lyapunov_error": abs(float(lyapunov)),
            "stable_for_negative_parameter": bool(
                jnp.any(negative_parameter) and jnp.all(branch.stable[negative_parameter])
            ),
            "unstable_for_positive_parameter": bool(
                jnp.any(positive_parameter) and jnp.all(~branch.stable[positive_parameter])
            ),
        },
        observations={
            "hopf_state": state,
            "hopf_parameter": parameter,
            "frequency": omega,
            "first_lyapunov": lyapunov,
            "n_points": branch.n_valid,
        },
        artifacts={
            "states": branch.states,
            "parameters": branch.params,
            "stability": branch.stable,
            "spectra": branch.eigenvalues,
            "events": [
                {
                    "kind": "H",
                    "parameter": float(parameter),
                    "state": state,
                    "frequency": abs(float(omega)),
                    "first_lyapunov": float(lyapunov),
                }
            ] if hopf_events else [],
            "event_spectra": [
                {
                    "kind": "H",
                    "parameter": float(parameter),
                    "values": jnp.linalg.eigvals(
                        jax.jacfwd(_van_der_pol_rhs, argnums=0)(state, parameter, None)
                    ),
                }
            ] if hopf_events else [],
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
        state, parameter, q1, q2, omega, _converged = jc.hopf_point(
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
    branch = result.branch
    # MatCont reports its cubic Hopf coefficient ``b = 2*l1`` for this
    # system.  Record both conventions and compare the MatCont value against
    # the analytic -0.3 reference.
    matcont_lyapunov = 2.0 * float(lyapunov)
    return CaseResult(
        case_id="MC-EQ-003",
        checks={
            "hopf_count": len(hopf_events),
            "max_hopf_error": hopf_error,
            "frequency_error": frequency_error,
            "max_residual": _maximum_residual(
                _adaptive_control_rhs,
                branch.states,
                branch.params,
                beta,
            ),
            "lyapunov_error": abs(matcont_lyapunov - (-0.3)),
        },
        observations={
            "hopf_state": state,
            "hopf_parameter": parameter,
            "frequency": omega,
            "first_lyapunov": lyapunov,
            "matcont_hopf_coefficient": matcont_lyapunov,
            "n_points": branch.n_valid,
        },
        artifacts={
            "states": branch.states,
            "parameters": branch.params,
            "stability": branch.stable,
            "spectra": branch.eigenvalues,
            "events": [
                {
                    "kind": "H",
                    "parameter": float(parameter),
                    "state": state,
                    "frequency": abs(float(omega)),
                    "first_lyapunov": matcont_lyapunov,
                }
            ] if hopf_events else [],
            "event_spectra": [
                {
                    "kind": "H",
                    "parameter": float(parameter),
                    "values": jnp.linalg.eigvals(
                        jax.jacfwd(_adaptive_control_rhs, argnums=0)(state, parameter, beta)
                    ),
                }
            ] if hopf_events else [],
        },
    )


__all__ = [
    "run_adaptive_control_hopf",
    "run_cubic_fold",
    "run_vanderpol_hopf",
]
