"""
End-to-end tests for jc.PeriodDoubling/jc.NeimarkSacker against a shared 4D
"circle (+) transverse" system: the existing verified 2D circle system
(r'=r(rho-r^2), theta'=1, rho=1 fixed, T=2*pi) plus a decoupled linear
transverse block (w1'=alpha*w1-beta*w2, w2'=beta*w1+alpha*w2). w=0 is an
exact periodic solution for any alpha/beta, and the transverse block's exact
Floquet-multiplier contribution is exp((alpha +/- i*beta)*T) (matrix
exponential of a constant matrix):
  - beta = pi/T: multiplier = -exp(alpha*T), real, crosses -1 at alpha=0
    (period-doubling ground truth).
  - beta = 0.3 (not a multiple of pi/T): multiplier = exp(alpha*T)*exp(+/-i*beta*T),
    complex pair, |multiplier| crosses 1 at alpha=0 (Neimark-Sacker ground truth).
alpha is the continuation parameter (p) in every test below. See
docs/superpowers/specs/2026-07-24-period-doubling-neimark-sacker-design.md.
"""

import numpy as np
import jax.numpy as jnp

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

RHO = 1.0
T_EXACT = 2 * np.pi
BETA_PD = np.pi / T_EXACT
BETA_NS = 0.3


def _make_rhs(beta):
    def rhs(u, p, args):
        x, y, w1, w2 = u[0], u[1], u[2], u[3]
        r2 = x * x + y * y
        alpha = p
        dx = (RHO - r2) * x - y
        dy = (RHO - r2) * y + x
        dw1 = alpha * w1 - beta * w2
        dw2 = beta * w1 + alpha * w2
        return jnp.array([dx, dy, dw1, dw2])
    return rhs


def _build_problem(beta, alpha0):
    t_traj = np.linspace(0, T_EXACT, 60, endpoint=False)
    x = np.sqrt(RHO) * np.cos(t_traj)
    y = np.sqrt(RHO) * np.sin(t_traj)
    u_traj = np.stack([x, y, np.zeros_like(t_traj), np.zeros_like(t_traj)], axis=1)
    mesh = Collocation(ntst=10, ncol=4)
    rhs = _make_rhs(beta)
    prob = periodic_orbit_problem(
        rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), T_EXACT, alpha0, mesh
    )
    return prob, mesh, rhs


def _sweep(beta, event_cls, span):
    prob, mesh, rhs = _build_problem(beta, alpha0=span[0])
    sol = jc.continuation(
        prob, p_span=span,
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.02, max_steps=50, newton_tol=1e-5
        ),
        events=[event_cls(raw_f=rhs, mesh=mesh)],
    )
    return sol


def test_period_doubling_detects_bifurcation_at_alpha_zero():
    # Verified during design: narrow sweep -0.05..0.05 detects exactly one
    # hit at p~-4.6e-7.
    sol = _sweep(BETA_PD, jc.PeriodDoubling, span=(-0.05, 0.05))
    assert sol.branch.n_valid > 1
    assert len(sol.events) == 1
    assert sol.events[0].kind == "period_doubling"
    assert abs(sol.events[0].p) < 1e-4


def test_neimark_sacker_detects_bifurcation_at_alpha_zero():
    # Verified during design: narrow sweep -0.05..0.05 detects exactly one
    # hit at p~-4.6e-7.
    sol = _sweep(BETA_NS, jc.NeimarkSacker, span=(-0.05, 0.05))
    assert sol.branch.n_valid > 1
    assert len(sol.events) == 1
    assert sol.events[0].kind == "neimark_sacker"
    assert abs(sol.events[0].p) < 1e-4


def test_period_doubling_zero_false_positives_on_neimark_sacker_system():
    sol = _sweep(BETA_NS, jc.PeriodDoubling, span=(-0.05, 0.05))
    assert sol.events == []


def test_neimark_sacker_zero_false_positives_on_period_doubling_system():
    sol = _sweep(BETA_PD, jc.NeimarkSacker, span=(-0.05, 0.05))
    assert sol.events == []


def test_period_doubling_near_unit_circle_filter_prevents_double_detection():
    # Regression for the false-positive bug found during design: a wider
    # sweep (-0.1..0.3) that pushes the transverse multiplier well past -1
    # must still report exactly ONE detection (at alpha=0), not two --
    # without the near_unit_circle filter this produced a spurious second
    # hit at p~0.110, where the "closest to -1" argmin silently switched to
    # tracking the unrelated, always-decaying xy multiplier (~3.4e-6)
    # instead of the true transverse candidate.
    #
    # This span also found a second, false-*negative* bug (v0.3.0): with the
    # old near_unit_circle=0.5, the adaptive stepper can sample the true
    # candidate at magnitude ~1.48 (distance 0.48 from 1, a 0.02 margin
    # inside the 0.5 window) -- thin enough that ordinary collocation/FP
    # differences across hardware push it outside the window, turning that
    # bracket point's test_function to nan and silently dropping the real
    # crossing (0 events instead of 1; reproduced on CI, not locally).
    # near_unit_circle=0.9 gives a ~20x wider margin at that same point while
    # still safely excluding the ~3.4e-6 decaying candidate (distance ~1.0,
    # comfortably outside 0.9) -- see events.py's near_unit_circle comment.
    sol = _sweep(BETA_PD, jc.PeriodDoubling, span=(-0.1, 0.3))
    assert len(sol.events) == 1
    assert abs(sol.events[0].p) < 1e-4
