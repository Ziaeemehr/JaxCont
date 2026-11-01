"""Numerical assertions on the data plotted by every public-API lesson."""

import runpy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxcont as jc
import matplotlib.pyplot as plt
import numpy as np
import pytest

def _eps_J():
    return float(jnp.finfo(jnp.float64).eps)

_POINT_ATOL = 1e-9


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lesson(tmp_path, monkeypatch):
    def run(number):
        path = next((ROOT / "examples/tutorials").glob(f"plot_{number:02d}_*.py"))
        monkeypatch.setattr(sys, "argv", [str(path), "--output-dir", str(tmp_path)])
        return runpy.run_path(str(path), run_name="__main__")

    yield run
    plt.close("all")
    jax.clear_caches()


def test_cubic_fold_has_two_folds(lesson):
    m = lesson(1)
    events = [e for e in m["result"].events if e.kind == "fold"]
    assert len(events) == 2
    # Measured |error|: 1.41e-11; analytic folds, default Newton budget.
    np.testing.assert_allclose(sorted((e.p for e in events)), [-2 / 3, 2 / 3], atol=_POINT_ATOL, rtol=0)
    # Measured positive ceiling excess: 0; analytic natural-branch fold ceiling, explicit Newton budget.
    assert np.max(m['natural'].branch.params) < 2 / 3 + 1e-5


def test_pitchfork_switch_finds_both_arms(lesson):
    m = lesson(2)
    endpoints = [float(a.branch.states[-1, 0]) for a in m["arms"]]
    assert min(endpoints) < -0.5 and max(endpoints) > 0.5
    for arm in m["arms"]:
        np.testing.assert_allclose(arm.branch.states[:, 0] ** 2, arm.branch.params, atol=2e-5)


def test_lorenz84_branch_has_fold_and_hopf(lesson):
    m = lesson(3)
    kinds = [e.kind for e in m["result"].events]
    assert "fold" in kinds and "hopf" in kinds
    values = jax.vmap(m["lorenz84_rhs"], in_axes=(0, 0, None))(
        m["result"].branch.states, m["result"].branch.params, m["args"]
    )
    assert np.max(np.abs(values)) < 2e-5


def test_vanderpol_figure_shows_stability_crossing(lesson):
    m = lesson(4)
    b = m["result"].branch
    assert np.all(np.asarray(b.stable)[np.asarray(b.params) < -0.1])
    assert not np.any(np.asarray(b.stable)[np.asarray(b.params) > 0.1])
    # Measured |error|: 4.42e-19; analytic Hopf parameter, default Newton budget.
    assert abs(next((e.p for e in m['result'].events if e.kind == 'hopf'))) < _POINT_ATOL


def test_neural_mass_figure_contains_corrected_equilibria(lesson):
    m = lesson(5)
    b = m["result"].branch
    residual = jax.jit(jax.vmap(m["TMvf"], in_axes=(0, 0, None)))(b.states, b.params, m["args"])
    # Check stationary state balances in their state units. Dividing the E
    # balance by tau=0.013 magnifies float32 cancellation in the vector field.
    balances = residual * np.array([m["args"][2], m["args"][3], m["args"][4]])
    assert np.max(np.abs(balances)) < 2e-5
    assert {"fold", "hopf"} <= {e.kind for e in m["result"].events}


def test_two_parameter_figure_matches_exact_fold_curve(lesson):
    m = lesson(6)
    # Measured |error|: 2.31e-08; exact fold curve, explicit Newton budget.
    np.testing.assert_allclose(m['solved'], 3 + (m['result'].branch.params + 1) ** 2 / 4, atol=1e-05, rtol=0)


def test_vmap_sweep_matches_serial_valid_points(lesson):
    m = lesson(7)
    for i, b in enumerate(m["bs"]):
        serial = m["run_one"](b).branch
        mask = np.asarray(m["result"].branch.valid[i])
        serial_mask = np.ones(serial.n_valid, dtype=bool)
        assert int(mask.sum()) == serial.n_valid
        # Measured |error|: 4.44e-16; serial/vmap consistency at unit scale.
        np.testing.assert_allclose(m['result'].branch.params[i][mask], serial.params[serial_mask], atol=32 * _eps_J(), rtol=0)
        # Measured |error|: 2.22e-16; serial/vmap consistency at unit scale.
        np.testing.assert_allclose(m['result'].branch.states[i][mask], serial.states[serial_mask], atol=32 * _eps_J(), rtol=0)


def test_fold_gradient_figure_matches_analytic_sensitivity(lesson):
    m = lesson(8)
    # Measured |error|: 0; analytic derivative of a quadratic fold parameter.
    np.testing.assert_allclose(m['gradients'], m['thetas'] / 2, atol=0, rtol=32 * _eps_J())
    # Measured |error|: 0; analytic fold parameter, default Newton budget.
    np.testing.assert_allclose(m['folds'], m['thetas'] ** 2 / 4, atol=_POINT_ATOL, rtol=0)


@pytest.mark.parametrize("number", [9, 10, 12, 13, 15, 16])
def test_periodic_tutorials_have_positive_period_and_amplitude(lesson, number):
    m = lesson(number)
    problem = m.get("problem", m.get("prob"))
    if number == 16:
        assert np.all(np.asarray(m["result"].branch.states)[:, -1] > 0)
        # Measured |error|: 5.24e-08; Gauss collocation ntst=10 ncol=4 plus Newton error.
        np.testing.assert_allclose(m['radii'], np.sqrt(m['parameters']), atol=max(_POINT_ATOL, 1e-06), rtol=0)
        # Measured |error|: 8.83e-08; Gauss collocation ntst=10 ncol=4 plus Newton error.
        np.testing.assert_allclose(m['radius_derivatives'], 1 / (2 * np.sqrt(m['parameters'])), atol=max(_POINT_ATOL, 1e-06), rtol=0)
        return
    assert float(problem.u0[-1]) > 0
    orbit = np.asarray(problem.u0[: 2 * m["mesh"].ntst]).reshape(-1, 2)
    assert np.ptp(orbit[:, 0]) > 0.1
    if number in (9, 10):
        assert np.all(m["periods"] > 0)
        assert m["periods"][-1] > m["periods"][0]
        assert np.min(m["amplitude"]) > 0.1
    elif number == 12:
        states = np.asarray(problem.u0[: 2 * m["mesh"].ntst]).reshape(-1, 2)
        np.testing.assert_allclose(
            m["Z"], np.column_stack([-states[:, 1], states[:, 0]]), atol=1e-3
        )

        # Phase anchoring drifts with rho; compare against re-converged orbits,
        # rather than differentiating a closed form at frozen phase.
        def response(rho):
            prob = jc.periodic_orbit_problem(
                m["rhs"], m["u_traj"], jnp.asarray(m["t_traj"]), 5.5, rho, m["mesh"]
            )
            return jc.prc_curve(m["rhs"], m["mesh"], prob.u0, prob.p0)

        fd = (response(1.01) - response(0.99)) / 0.02
        np.testing.assert_allclose(m["dZ"], fd, atol=2e-3)
    else:
        # Measured |error|: 3.35e-10; analytic periodic event, mesh and bisection/Newton budget.
        assert abs(m['event'].p) < (1e-08)
        # Measured |error|: 1.35e-09; fixed-mesh orbit comparison, collocation and Newton budget.
        assert m['max_error'] < 0.0005


def test_phase_plane_figures_bracket_hopf(lesson):
    m = lesson(11)
    event = next(e for e in m["result"].events if e.kind == "hopf")
    assert 0 < event.p < 0.5
    assert len(plt.get_fignums()) == 2
    for p, expected in [(0.0, -1), (0.5, 1)]:
        i = np.argmin(np.abs(np.asarray(m["result"].branch.params) - p))
        state = m["result"].branch.states[i]
        jac = jax.jacfwd(m["fitzhugh_nagumo_rhs"])(state, p, m["args"])
        assert float(np.max(np.real(np.linalg.eigvals(jac)))) * expected > 0


def test_pd_switch_period_ratio_near_two(lesson):
    m = lesson(14)
    # Measured |error|: 1.89e-09; doubled Gauss period ratio, accumulation and mesh truncation.
    assert abs(float(m['switched'].u0[-1]) / m['T_EXACT'] - 2) < max(128 * _eps_J(), 1e-07)
    # Measured |error|: 1.7e-09; Gauss cycle amplitude, Newton conditioning and mesh truncation.
    np.testing.assert_allclose(m['amps'], np.sqrt(m['alphas']), atol=1e-07, rtol=0)
    orbit = np.asarray(m["switched"].u0[:80]).reshape(20, 4)
    centered = orbit - orbit.mean(axis=0)
    assert np.max(np.abs(centered[:10] - centered[10:])) > 0.1
