"""Tests for period-doubling branch switching (jc.branch_switch_pd).

Fixture: a cubic-saturated variant of the repo's 4D "circle (+) transverse"
system. The repo's EXISTING PD fixture in
tests/test_period_doubling_neimark_sacker.py has a LINEAR transverse block, so
w=0 solves it for every parameter and NO period-doubled branch exists to switch
onto -- correct for detection, useless here. Adding the cubic saturation gives a
genuine doubled branch with closed-form ground truth:

    x' = (1 - r^2) x - y            r^2 = x^2 + y^2
    y' = (1 - r^2) y + x            -> circle, period T = 2*pi
    w' = (alpha + i*beta) w - c |w|^2 w    w = w1 + i*w2, beta = pi/T, c = 1

The linear part rotates w by exactly pi per period, so w(T) = -w(0):
anti-periodic, i.e. a period-doubling at alpha = 0. For alpha > 0 the exact
doubled orbit is |w| = sqrt(alpha/c) with period 2T = 4*pi.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.bifurcations.pd_switch import _doubling_measure, _pd_eigenvector
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

T_EXACT = 2 * np.pi
BETA = np.pi / T_EXACT
C_SAT = 1.0
N_STATE = 4


def rhs(u, p, args):
    x, y, w1, w2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    m2 = w1 * w1 + w2 * w2
    return jnp.array([
        (1.0 - r2) * x - y,
        (1.0 - r2) * y + x,
        p * w1 - BETA * w2 - C_SAT * w1 * m2,
        BETA * w1 + p * w2 - C_SAT * w2 * m2,
    ])


def base_problem(alpha, ntst=10, ncol=4):
    """The w=0 circle orbit at parameter `alpha`, period T."""
    t = np.linspace(0.0, T_EXACT, 120, endpoint=False)
    u = np.stack([np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1)
    mesh = Collocation(ntst=ntst, ncol=ncol)
    prob = periodic_orbit_problem(
        rhs, jnp.asarray(u), jnp.asarray(t), T_EXACT, alpha, mesh
    )
    return prob, mesh


def test_pd_eigenvector_is_supported_on_the_transverse_block():
    # At alpha=0 the transverse multiplier is -exp(0)= -1, and its eigenvector
    # lies entirely in the (w1, w2) plane -- components 2 and 3.
    prob, mesh = base_problem(0.0)
    v = _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)
    v = np.asarray(v)
    assert v.shape == (N_STATE,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)
    assert abs(v[0]) < 1e-3
    assert abs(v[1]) < 1e-3
    assert np.linalg.norm(v[2:]) > 0.99


def test_pd_eigenvector_rejects_a_point_that_is_not_period_doubling():
    # At alpha=0.5 the transverse multiplier is -exp(0.5*T) ~ -23, nowhere
    # near -1, so this is not a PD point and must be rejected loudly.
    prob, mesh = base_problem(0.5)
    with pytest.raises(ValueError, match="period-doubling"):
        _pd_eigenvector(rhs, prob.u0, prob.p0, mesh)


def _pack_mesh_only(mesh_states, mesh, n):
    """Build a U vector whose mesh-state block is `mesh_states`.

    _doubling_measure only reads the mesh-state block, so the collocation
    block and period can be zeros for this unit test.
    """
    ntst, ncol = mesh.ntst, mesh.ncol
    return jnp.concatenate([
        jnp.asarray(mesh_states).flatten(),
        jnp.zeros(ntst * ncol * n),
        jnp.zeros(1),
    ])


def test_doubling_measure_separates_a_doubled_orbit_from_a_retraced_one():
    mesh = Collocation(ntst=20, ncol=4)
    t = np.linspace(0.0, 2 * T_EXACT, 20, endpoint=False)

    # The ORIGINAL orbit traversed twice: identical under a t -> t+T shift.
    retraced = np.stack(
        [np.cos(t), np.sin(t), np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    d_retraced = _doubling_measure(
        _pack_mesh_only(retraced, mesh, N_STATE), mesh, N_STATE
    )
    assert d_retraced < 1e-3

    # A genuine doubled orbit: w rotates by pi per period, so w(t+T) = -w(t).
    amp = np.sqrt(0.05 / C_SAT)
    doubled = np.stack([
        np.cos(t), np.sin(t),
        amp * np.cos(BETA * t), amp * np.sin(BETA * t),
    ], axis=1)
    d_doubled = _doubling_measure(
        _pack_mesh_only(doubled, mesh, N_STATE), mesh, N_STATE
    )
    # Closed form: d = 2*amp, scale = sqrt(1 + amp^2)  ->  ~0.436
    assert d_doubled == pytest.approx(2 * amp / np.sqrt(1 + amp**2), abs=1e-3)
    assert d_doubled > 0.4
