"""JAX transformation and differentiability validation case."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxcont as jc

from . import CaseResult


def _fold_family(u, parameter, theta):
    return jnp.array([u[0] ** 2 - theta * u[0] + parameter])


def _fold_coefficient_family(u, parameter, scale):
    x, y = u
    return jnp.array([parameter[0] + parameter[1] * x + scale * x**2, -y])


def _hopf_family(u, parameter, shift):
    x, y = u
    radius_squared = x**2 + y**2
    effective_parameter = parameter - shift
    return jnp.array(
        [
            -y + x * (effective_parameter - radius_squared),
            x + y * (effective_parameter - radius_squared),
        ]
    )


def _lyapunov_family(u, parameter, scale):
    x, y = u
    radius_squared = x**2 + y**2
    return jnp.array(
        [
            -y + x * (parameter - scale * radius_squared),
            x + y * (parameter - scale * radius_squared),
        ]
    )


def _imperfect_pitchfork(u, parameter, imperfection):
    return jnp.array([parameter * u[0] - u[0] ** 3 + imperfection])


def _batched_branch(imperfection):
    problem = jc.bif_problem(
        _imperfect_pitchfork,
        u0=jnp.array([0.05]),
        p0=-1.0,
        args=imperfection,
    )
    return jc.continuation(
        problem,
        p_span=(-1.0, 1.0),
        settings=jc.ContinuationPar(
            ds=0.05,
            ds_min=1e-5,
            ds_max=0.2,
            max_steps=40,
            newton_tol=1e-6,
        ),
    )


def _centered_difference(function, point, step=1e-3):
    return (function(point + step) - function(point - step)) / (2.0 * step)


def run_transform_checks() -> CaseResult:
    """Compare JAX derivatives with exact and finite-difference values."""
    theta = jnp.array(1.0)

    def fold_parameter(value):
        return jc.fold_point(
            _fold_family, jnp.array([0.4]), jnp.array(0.2), value
        )[1]

    fold_gradient = jax.grad(fold_parameter)(theta)
    fold_fd = _centered_difference(fold_parameter, theta)

    def fold_normal_form_coefficient(scale):
        return jc.fold_coefficient(
            _fold_coefficient_family,
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.array([1.0, 0.0]),
            scale,
        )

    coefficient_gradient = jax.grad(fold_normal_form_coefficient)(jnp.array(2.0))
    coefficient_fd = _centered_difference(
        fold_normal_form_coefficient, jnp.array(2.0)
    )

    def hopf_parameter(shift):
        return jc.hopf_point(
            _hopf_family, jnp.zeros(2), jnp.array(0.05), shift, tol=1e-7
        )[1]

    hopf_gradient = jax.grad(hopf_parameter)(jnp.array(0.1))
    hopf_fd = _centered_difference(hopf_parameter, jnp.array(0.1))

    def lyapunov(scale):
        state, parameter, q1, q2, omega = jc.hopf_point(
            _lyapunov_family,
            jnp.zeros(2),
            jnp.array(0.05),
            scale,
            tol=1e-7,
        )
        return jc.lyapunov_coefficient(
            _lyapunov_family, state, parameter, q1, q2, omega, scale
        )

    lyapunov_gradient = jax.grad(lyapunov)(jnp.array(1.0))
    lyapunov_fd = _centered_difference(lyapunov, jnp.array(1.0), step=1e-4)

    gradients = jnp.array(
        [fold_gradient, coefficient_gradient, hopf_gradient, lyapunov_gradient]
    )
    finite_differences = jnp.array([fold_fd, coefficient_fd, hopf_fd, lyapunov_fd])
    analytic = jnp.array([0.5, 1.0, 1.0, -1.0])

    imperfections = jnp.array([-0.15, 0.0, 0.15])
    batched = jax.vmap(_batched_branch)(imperfections)
    permuted = jax.vmap(_batched_branch)(imperfections[::-1])
    serial = [_batched_branch(imperfection) for imperfection in imperfections]
    serial_masks = jnp.stack(
        [
            jnp.arange(batched.branch.valid.shape[1]) < result.branch.n_valid
            for result in serial
        ]
    )
    serial_masks_match = bool(jnp.array_equal(batched.branch.valid, serial_masks))
    serial_parameters_match = bool(
        all(
            jnp.allclose(batched.branch.params[index][serial_masks[index]], result.branch.params)
            for index, result in enumerate(serial)
        )
    )
    serial_states_match = bool(
        all(
            jnp.allclose(batched.branch.states[index][serial_masks[index]], result.branch.states)
            for index, result in enumerate(serial)
        )
    )
    permutation_invariant = bool(
        jnp.allclose(batched.branch.params, permuted.branch.params[::-1])
        and jnp.allclose(batched.branch.states, permuted.branch.states[::-1])
        and jnp.array_equal(batched.branch.valid, permuted.branch.valid[::-1])
    )
    eager = _batched_branch(jnp.array(0.1))
    jitted = jax.jit(_batched_branch)(jnp.array(0.1))
    jitted_n_valid = int(jitted.stats["n_valid"])
    jit_matches_eager = bool(
        jitted.branch.valid is not None
        and int(jnp.sum(jitted.branch.valid)) == eager.branch.n_valid
        and jitted_n_valid == eager.branch.n_valid
        and jnp.allclose(jitted.branch.params[:jitted_n_valid], eager.branch.params)
        and jnp.allclose(jitted.branch.states[:jitted_n_valid], eager.branch.states)
    )
    return CaseResult(
        case_id="MC-JAX-001",
        checks={
            "all_finite": bool(
                jnp.all(jnp.isfinite(gradients))
                and jnp.all(jnp.isfinite(finite_differences))
            ),
            "max_analytic_gradient_error": float(jnp.max(jnp.abs(gradients - analytic))),
            "max_finite_difference_error": float(
                jnp.max(jnp.abs(gradients - finite_differences))
            ),
            "vmap_valid_masks_present": batched.branch.valid is not None,
            "vmap_valid_masks_match_serial": serial_masks_match,
            "vmap_valid_parameters_match_serial": serial_parameters_match,
            "vmap_valid_states_match_serial": serial_states_match,
            "permutation_invariant": permutation_invariant,
            "jit_matches_eager": jit_matches_eager,
        },
        observations={
            "gradients": gradients,
            "finite_differences": finite_differences,
            "analytic_gradients": analytic,
            "n_valid": batched.stats["n_valid"],
        },
        artifacts={
            "parameters": batched.branch.params,
            "states": batched.branch.states,
            "valid": batched.branch.valid,
        },
    )


__all__ = ["run_transform_checks"]
