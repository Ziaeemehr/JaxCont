"""Two-parameter fold/Hopf-curve validation against a real MatCont 7.6 run.

Promotes ``cases.json``'s ``US-C2-LP-001``/``US-C2-H-001`` from
``unsupported`` to ``supported``: cross-validates
``bifurcations/curves.py``'s ``fold_curve_problem``/``hopf_curve_problem``
against MatCont 7.6's own catalytic-oscillator LP/Hopf curves
(``Testruns/testLPcataloscill.m``/``testLPHopfcataloscill.m``, matching
``examples/MatCont/matlab/unsupported/run_two_parameter_{fold,hopf}_curve.m``).

Unlike ``MC-EQ-*``/``MC-LC-001`` (compared via ``compare.py``'s
monotone-parameter branch interpolation), a fold/Hopf *curve* varies two
parameters at once -- there is no single axis to interpolate against, and
generalizing ``compare.py`` into a full 2D-curve comparison framework for
two cases would be over-building. Instead this follows the same pattern
``MC-C2-001`` already uses for the direct codim-2 point solvers: a handful
of numeric reference values, read directly off a real MatCont run and
hardcoded below, checked against JaxCont's own output.

Reference values (``_MATCONT_*`` below) were captured 2026-08-26 from a
live run of the two MATLAB drivers above against the local MatCont 7.6
install -- not fabricated, not back-solved from JaxCont's own output.

The catalytic oscillator (Kevrekidis et al.; MatCont's
``Testruns/TestSystems/cataloscill.m``) is a 3-state, 7-parameter ODE.
Following MatCont's own testruns, five of the seven parameters
(``q1, q3, q4, q5, q6``) stay fixed; ``(q2, k)`` are the two free
parameters both curves live in.

Direction note: the LP curve passes through a cusp (MatCont's own console
output labels it ``CP`` mid-run) at k~=0.356, where the curve doubles back
on itself in the (q2, k) plane -- for k in roughly (0.356, 0.4) the curve
has two branches. Continuing from the shared LP/H seed at k=0.4 in the
*decreasing*-k direction (matching MatCont's own arc-length choice, verified
by comparing against the raw MatCont CSV during development) traces the
same branch MatCont's reference points below sit on; continuing toward
increasing k instead traces a different, equally valid stretch of the same
overall curve that MatCont's 25-point run never sampled. This is a real
property of the curve, not a bug in either tool.
"""

from __future__ import annotations

import jax.numpy as jnp

import jaxcont as jc
from jaxcont.bifurcations.curves import (
    fold_curve_problem,
    hopf_curve_problem,
    unpack_fold_curve,
    unpack_hopf_curve,
)

from . import CaseResult

# Fixed catalytic-oscillator parameters (q1, q3, q4, q5, q6), matching the
# MATLAB drivers' `p = [2.5; 2.204678; 10; 0.0675; 1; 0.1; 0.4]`
# (p(1)=q1, p(3)=q3, p(4)=q4, p(5)=q5, p(6)=q6; p(2)=q2 and p(7)=k are free).
_Q1, _Q3, _Q4, _Q5, _Q6 = 2.5, 10.0, 0.0675, 1.0, 0.1

_EQ_U0 = jnp.array([0.001137, 0.891483, 0.062345])
_EQ_P0 = 2.204678
_K0 = 0.4

# The LP/H points MatCont's own drivers select via `find(strcmp(labels,
# 'LP'|'H'), 1)` along the equilibrium branch -- read from the raw
# MATLAB run's console output / exported curve seed rows.
_MATCONT_LP_P2 = 1.04204587467179
_MATCONT_H_P2 = 1.05158600128154

# Two points each, safely inside the pre-cusp/monotone stretch both curves
# share with the JaxCont run below (k, q2) -- from
# examples/MatCont/matlab/unsupported/run_two_parameter_fold_curve.m /
# run_two_parameter_hopf_curve.m's real MatCont 7.6 output.
_MATCONT_LP_CURVE_REFS = [
    (0.382240667895003, 1.02900241954427),
    (0.365411487710214, 1.01511414557068),
]
_MATCONT_H_CURVE_REFS = [
    (0.344826449242455, 1.0145538982455),
    (0.279715710421856, 0.954127904211814),
]


def _cataloscill(u, p, args):
    del args
    x1, x2, x3 = u
    q2, k = p
    z = 1.0 - x1 - x2 - x3
    return jnp.array([
        2 * _Q1 * z**2 - 2 * _Q5 * x1**2 - _Q3 * x1 * x2,
        q2 * z - _Q6 * x2 - _Q3 * x1 * x2,
        _Q4 * z - k * _Q4 * x3,
    ])


def _find_equilibrium_events():
    """Continue the catalytic oscillator in q2 (k=0.4 fixed) and detect
    Fold/Hopf events, matching the equilibrium sweep both MATLAB drivers
    run before seeding their codim-2 curve."""
    rhs = lambda u, q2, args: _cataloscill(u, jnp.array([q2, _K0]), args)
    problem = jc.bif_problem(rhs, u0=_EQ_U0, p0=_EQ_P0)
    result = jc.continuation(
        problem, jc.PseudoArclength(), p_span=(_EQ_P0, 0.5),
        settings=jc.ContinuationPar(ds=0.005, ds_max=0.01, max_steps=400, newton_tol=1e-6),
        events=[jc.Fold(), jc.Hopf()],
    )
    return result.events


def _nearest_by_parameter(events, kind: str, target_p: float):
    """The detected event of `kind` closest to `target_p`.

    MatCont's `find(strcmp(labels, ...), 1)` picks the first LP/H labeled
    during ITS OWN continuation trajectory, which is not simply "smallest
    p2" (the equilibrium branch's own folds make p2 non-monotonic).
    Selecting by nearness to the known MatCont value both locates the same
    physical point and doubles as the location check itself.
    """
    candidates = [e for e in events if e.kind == kind]
    return min(candidates, key=lambda e: abs(e.p - target_p))


def _min_distance_to_curve(k_points, q2_points, k_ref: float, q2_ref: float) -> float:
    distances = jnp.sqrt((k_points - k_ref) ** 2 + (q2_points - q2_ref) ** 2)
    return float(jnp.min(distances))


def run_two_parameter_fold_curve() -> CaseResult:
    """Continue the catalytic-oscillator LP curve and compare against a
    real MatCont 7.6 run (`US-C2-LP-001`)."""
    events = _find_equilibrium_events()
    lp_event = _nearest_by_parameter(events, "fold", _MATCONT_LP_P2)
    lp_seed_error = abs(lp_event.p - _MATCONT_LP_P2)

    problem = fold_curve_problem(
        _cataloscill, lp_event.u, jnp.array([lp_event.p, _K0]), free=1, tol=1e-6,
    )
    result = jc.continuation(
        problem, p_span=(_K0, 0.36),
        # Preserve dense Euclidean sampling for nearest-sample comparisons.
        settings=jc.ContinuationPar(theta=1.0, ds=0.002, ds_max=0.005, max_steps=100, newton_tol=1e-6, compute_stability=False),
    )
    branch = result.branch
    q2_points = jnp.array([unpack_fold_curve(state, 3)[1] for state in branch.states])
    k_points = branch.params

    curve_errors = [
        _min_distance_to_curve(k_points, q2_points, k_ref, q2_ref)
        for k_ref, q2_ref in _MATCONT_LP_CURVE_REFS
    ]

    return CaseResult(
        case_id="US-C2-LP-001",
        checks={
            "lp_found": lp_event is not None,
            "lp_seed_error": lp_seed_error,
            "n_curve_points": branch.n_valid,
            "max_curve_error": max(curve_errors),
        },
        observations={
            "lp_seed": (float(lp_event.p), [float(x) for x in lp_event.u]),
            "curve_errors": curve_errors,
        },
        artifacts={"k": k_points, "q2": q2_points},
    )


def run_two_parameter_hopf_curve() -> CaseResult:
    """Continue the catalytic-oscillator Hopf curve and compare against a
    real MatCont 7.6 run (`US-C2-H-001`)."""
    events = _find_equilibrium_events()
    h_event = _nearest_by_parameter(events, "hopf", _MATCONT_H_P2)
    h_seed_error = abs(h_event.p - _MATCONT_H_P2)

    problem = hopf_curve_problem(
        _cataloscill, h_event.u, jnp.array([h_event.p, _K0]), free=1, tol=1e-6,
        formulation="extended",
    )
    result = jc.continuation(
        problem, p_span=(_K0, 0.2),
        # Preserve dense Euclidean sampling for nearest-sample comparisons.
        settings=jc.ContinuationPar(theta=1.0, ds=0.001, ds_max=0.002, max_steps=400, newton_tol=1e-6, compute_stability=False),
    )
    branch = result.branch
    q2_points = jnp.array([unpack_hopf_curve(state, 3)[1] for state in branch.states])
    k_points = branch.params

    curve_errors = [
        _min_distance_to_curve(k_points, q2_points, k_ref, q2_ref)
        for k_ref, q2_ref in _MATCONT_H_CURVE_REFS
    ]

    return CaseResult(
        case_id="US-C2-H-001",
        checks={
            "hopf_found": h_event is not None,
            "hopf_seed_error": h_seed_error,
            "n_curve_points": branch.n_valid,
            "max_curve_error": max(curve_errors),
        },
        observations={
            "hopf_seed": (float(h_event.p), [float(x) for x in h_event.u]),
            "curve_errors": curve_errors,
        },
        artifacts={"k": k_points, "q2": q2_points},
    )
