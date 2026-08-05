"""Shifted direct codimension-two solver validation case."""

from __future__ import annotations

import jax.numpy as jnp

import jaxcont as jc

from . import CaseResult


def _cusp_shifted(u, parameter, args):
    scale = 1.0 if args is None else args
    state = u[0] - 2.0
    first = parameter[0] - 1.0
    second = parameter[1] + 4.0
    return jnp.array([first + second * state + scale * state**3])


def _bt_shifted(u, parameter, args):
    del args
    x, y = u[0] - 5.0, u[1] - 2.0
    first = parameter[0] - 3.0
    second = parameter[1] + 1.0
    return jnp.array([y, first + second * x + x**2 + x * y])


def _gh_shifted(u, parameter, args):
    del args
    x, y = u
    first = parameter[0] - 2.0
    second = parameter[1] + 3.0
    radius_squared = x**2 + y**2
    radial = first + second * radius_squared - radius_squared**2
    return jnp.array([-y + x * radial, x + y * radial])


def _zh_shifted(u, parameter, args):
    del args
    w, x, y = u[0] - 1.0, u[1], u[2]
    first = parameter[0] - 4.0
    second = parameter[1] + 2.0
    return jnp.array([first + w**2, second * x - y, x + second * y])


def _hh_shifted(u, parameter, args):
    del args
    x1, y1, x2, y2 = u
    first = parameter[0] - 5.0
    second = parameter[1] + 6.0
    return jnp.array(
        [
            first * x1 - y1,
            x1 + first * y1,
            second * x2 - 2.0 * y2,
            2.0 * x2 + second * y2,
        ]
    )


_BK_BT_STATE = jnp.array(
    [1.2256404959862783, -0.03632079327106889, 0.197288323112333, -0.12339001198518851]
)
_BK_BT_PARAMETER = jnp.array([1.4467167011153617, 0.020940169683322584])


def _lorenz84(u, parameter, args):
    del args
    x, y, z, fourth = u
    forcing, temperature = parameter
    alpha = 0.25
    beta = 1.0
    gravity = 0.25
    delta = 1.04
    gamma = 0.987
    return jnp.array(
        [
            -y**2 - z**2 - alpha * x + alpha * forcing - gamma * fourth**2,
            x * y - beta * x * z - y + gravity,
            beta * x * y + x * z - z,
            -delta * fourth + gamma * fourth * x + temperature,
        ]
    )


def run_codim2_points() -> CaseResult:
    """Recover shifted CP/BT/GH/ZH/HH points and an independent BT reference."""
    cusp = jc.cusp_point(
        _cusp_shifted, jnp.array([2.2]), jnp.array([0.8, -3.7])
    )
    bt = jc.bogdanov_takens_point(
        _bt_shifted, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8])
    )
    gh = jc.generalized_hopf_point(
        _gh_shifted, jnp.array([0.02, -0.03]), jnp.array([2.05, -2.90])
    )
    zh = jc.zero_hopf_point(
        _zh_shifted,
        jnp.array([1.05, 0.03, -0.02]),
        jnp.array([4.04, -1.94]),
    )
    hh = jc.double_hopf_point(
        _hh_shifted,
        jnp.array([0.03, -0.02, 0.04, 0.01]),
        jnp.array([5.05, -5.93]),
        seed_b=jnp.array([0.0, 0.0, 1.0, 0.0]),
    )
    bk_bt = jc.bogdanov_takens_point(
        _lorenz84,
        _BK_BT_STATE + jnp.array([0.05, -0.03, 0.02, -0.01]),
        _BK_BT_PARAMETER + jnp.array([0.03, -0.02]),
    )

    solved_parameters = {
        "CP": cusp[1],
        "BT": bt[1],
        "GH": gh[1],
        "ZH": zh[1],
        "HH": hh[1],
    }
    expected_parameters = {
        "CP": jnp.array([1.0, -4.0]),
        "BT": jnp.array([3.0, -1.0]),
        "GH": jnp.array([2.0, -3.0]),
        "ZH": jnp.array([4.0, -2.0]),
        "HH": jnp.array([5.0, -6.0]),
    }
    parameter_errors = {
        name: float(jnp.max(jnp.abs(solved_parameters[name] - expected)))
        for name, expected in expected_parameters.items()
    }
    convergence = {
        "CP": bool(cusp[-1]),
        "BT": bool(bt[-1]),
        "GH": bool(gh[-1]),
        "ZH": bool(zh[-1]),
        "HH": bool(hh[-1]),
        "BK-BT": bool(bk_bt[-1]),
    }
    bk_error = max(
        float(jnp.max(jnp.abs(bk_bt[0] - _BK_BT_STATE))),
        float(jnp.max(jnp.abs(bk_bt[1] - _BK_BT_PARAMETER))),
    )
    frequencies = {
        "GH": float(gh[4]),
        "ZH": float(zh[5]),
        "HH": sorted([float(hh[4]), float(hh[7])]),
    }
    return CaseResult(
        case_id="MC-C2-001",
        checks={
            "all_converged": all(convergence.values()),
            "max_parameter_error": max(parameter_errors.values()),
            "bt_bifurcationkit_error": bk_error,
            "frequency_error": max(
                abs(frequencies["GH"] - 1.0),
                abs(frequencies["ZH"] - 1.0),
                abs(frequencies["HH"][0] - 1.0),
                abs(frequencies["HH"][1] - 2.0),
            ),
        },
        observations={
            "parameters": solved_parameters,
            "parameter_errors": parameter_errors,
            "convergence": convergence,
            "frequencies": frequencies,
            "bifurcationkit_bt_state": bk_bt[0],
            "bifurcationkit_bt_parameter": bk_bt[1],
        },
        artifacts={},
    )


__all__ = ["run_codim2_points"]
