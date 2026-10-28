"""MatCont ``testadapt2`` cross-validation for period-doubling switching.

The reference values below were captured from a live MatCont 7.6 run of
``Testruns/testadapt2.m`` on 2026-08-27.  The JaxCont path transcribes only
``TestSystems/adaptx.m`` and independently continues its limit-cycle branch;
it never consumes MatCont's orbit data as a seed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jaxcont as jc
from jaxcont.bifurcations.events import EventHit
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

from . import CaseResult

# MatCont 7.6's `testadapt2` output, exported from a live run.
#
# `BASE_PD` is the PD point on the base limit-cycle branch that `init_PD_LC`
# switches from. `DOUBLED_START` is the first point of the doubled branch
# `init_PD_LC` produces -- MatCont's own answer to "what does the switched
# orbit look like", and therefore the like-for-like reference for this
# feature. Note that MatCont's doubled branch starts at *exactly* twice the
# base period (12.7281426646 = 2 x 6.3640713323), which is what makes the
# period-ratio check below a real structural check rather than a restatement
# of JaxCont's own arithmetic.
#
# `DOUBLED_AT` is a second point MatCont reports further along the same
# doubled branch (it is that branch's own next PD). Its *parameter* is NOT a
# valid target for the switch -- comparing JaxCont's switched alpha to it
# would only test that 0.6303 - dp lands near an unrelated bifurcation. Its
# *period* is valid and is the tightest evidence here: this case's switched
# point lands within ~6e-5 of that alpha, and two points at the same alpha on
# the same branch must share a period. `_ALPHA_MATCH_ATOL` guards exactly
# that precondition, so a future `dp` change turns this into a loud failure
# rather than a silently meaningless comparison.
_MATCONT_BASE_PD_ALPHA = 0.630302040837
_MATCONT_BASE_PD_PERIOD = 6.3640713323
_MATCONT_DOUBLED_START_PERIOD = 12.7281426646
_MATCONT_DOUBLED_AT_ALPHA = 0.579629869689
_MATCONT_DOUBLED_AT_PERIOD = 12.7343655405
_ALPHA_MATCH_ATOL = 1e-3


def _adaptx_rhs(u, alpha, args):
    """``Testruns/TestSystems/adaptx.m`` with beta fixed to MatCont's one."""
    del args
    x, y, z = u
    return jnp.array([y, z, -alpha * z - y - x + x * x])


def _base_cycle_problem():
    """Build a local Hopf-direction seed without using a MatCont orbit."""
    alpha = 0.9
    period = 2.0 * np.pi
    t = jnp.linspace(0.0, period, 160, endpoint=False)
    # Leading Hopf amplitude: harmonic balance gives
    # 1 - alpha = (9/10) * amplitude**2 + O(amplitude**4).
    # An arbitrary smaller amplitude sends undamped Newton close to the
    # equilibrium, where the period is undetermined, making convergence
    # depend on rounding in the collocation residual/Jacobian.
    amplitude = np.sqrt(10.0 * (1.0 - alpha) / 9.0)
    trajectory = jnp.stack(
        [
            amplitude * jnp.cos(t),
            -amplitude * jnp.sin(t),
            -amplitude * jnp.cos(t),
        ],
        axis=1,
    )
    mesh = Collocation(ntst=20, ncol=4)
    return periodic_orbit_problem(
        _adaptx_rhs, trajectory, t, period, alpha, mesh
    ), mesh


def _period_doubling_event(result) -> EventHit:
    """Locate the flip crossing directly from JaxCont's Floquet branch data."""
    branch = result.branch
    n_valid = branch.n_valid
    parameters = np.asarray(branch.params[:n_valid])
    spectra = np.asarray(branch.eigenvalues[:n_valid])

    candidates = []
    for index, spectrum in enumerate(spectra):
        trivial = int(np.argmin(np.abs(spectrum - 1.0)))
        nontrivial = np.delete(spectrum, trivial)
        multiplier = nontrivial[int(np.argmin(np.abs(nontrivial + 1.0)))]
        candidates.append(float(np.real(multiplier) + 1.0))

    crossings = [
        index
        for index, (left, right) in enumerate(zip(candidates, candidates[1:]))
        if left * right <= 0.0
    ]
    if not crossings:
        raise RuntimeError("JaxCont periodic continuation found no adaptx PD crossing")
    left = crossings[0]
    index = min((left, left + 1), key=lambda candidate: abs(candidates[candidate]))
    return EventHit(
        kind="period_doubling",
        p=float(parameters[index]),
        u=branch.states[index],
        index=index,
        info={"method": "Floquet sign-change on JaxCont periodic branch"},
    )


def _half_period_difference(problem, mesh: Collocation) -> float:
    n_state = (problem.u0.shape[-1] - 1) // (mesh.ntst * (1 + mesh.ncol))
    states = np.asarray(problem.u0[: mesh.ntst * n_state]).reshape(mesh.ntst, n_state)
    spread = np.max(np.linalg.norm(states - np.roll(states, mesh.ntst // 2, axis=0), axis=1))
    scale = np.max(np.linalg.norm(states, axis=1))
    return float(spread / scale)


def run_adaptx_pd_switch() -> CaseResult:
    """Find adaptx's PD point, switch it, and check MatCont live references."""
    dp = 0.05
    problem, mesh = _base_cycle_problem()
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(0.9, 0.55),
        settings=jc.ContinuationPar(
            # PD seeding selects a raw crossing endpoint at this Euclidean spacing.
            theta=1.0,
            compute_stability=True,
            ds=0.01,
            ds_max=0.02,
            max_steps=300,
            newton_tol=1e-5,
        ),
    )
    event = _period_doubling_event(result)
    switched = jc.branch_switch_pd(
        _adaptx_rhs, event, mesh, dp=dp, dp_sign=-1
    )
    switched_mesh = Collocation(ntst=2 * mesh.ntst, ncol=mesh.ncol)
    residual = float(jnp.linalg.norm(switched.f(switched.u0, switched.p0, switched.args)))
    difference = _half_period_difference(switched, switched_mesh)

    return CaseResult(
        case_id="MC-LC-003",
        checks={
            "pd_found": True,
            "pd_parameter_error": abs(float(event.p) - _MATCONT_BASE_PD_ALPHA),
            "pd_period_error": abs(float(event.u[-1]) - _MATCONT_BASE_PD_PERIOD),
            # Internal consistency, not a MatCont comparison: the switch is
            # seeded a fixed `dp` off the detected PD, so this only confirms
            # the factory stepped where it was told.
            "switched_parameter_offset_error": abs(
                float(switched.p0) - (float(event.p) - dp)
            ),
            # The real cross-validation of the switch: MatCont's own doubled
            # branch starts at exactly twice the base period, so a correct
            # switch must land on an orbit of that period. Phase-invariant and
            # independent of `dp`.
            "doubled_period_error": abs(
                float(switched.u0[-1]) - _MATCONT_DOUBLED_START_PERIOD
            ),
            "doubled_period_ratio_error": abs(
                float(switched.u0[-1]) / float(event.u[-1]) - 2.0
            ),
            # Tightest evidence that JaxCont's doubled branch IS MatCont's:
            # two points at the same alpha on the same branch share a period.
            # Only meaningful while the alphas actually match -- hence the
            # guard immediately below, which must be asserted alongside it.
            "doubled_branch_alpha_gap": abs(
                float(switched.p0) - _MATCONT_DOUBLED_AT_ALPHA
            ),
            "doubled_branch_alphas_match": (
                abs(float(switched.p0) - _MATCONT_DOUBLED_AT_ALPHA)
                < _ALPHA_MATCH_ATOL
            ),
            "doubled_branch_period_error": abs(
                float(switched.u0[-1]) - _MATCONT_DOUBLED_AT_PERIOD
            ),
            "switch_residual": residual,
            "genuine_doubled_orbit": difference > 1e-3,
        },
        observations={
            "jaxcont_pd": {"alpha": float(event.p), "period": float(event.u[-1])},
            "jaxcont_switched": {
                "alpha": float(switched.p0),
                "period": float(switched.u0[-1]),
                "half_period_difference": difference,
            },
            "matcont_base_pd": {
                "alpha": _MATCONT_BASE_PD_ALPHA,
                "period": _MATCONT_BASE_PD_PERIOD,
            },
            "matcont_doubled_start": {
                "alpha": _MATCONT_BASE_PD_ALPHA,
                "period": _MATCONT_DOUBLED_START_PERIOD,
            },
        },
        artifacts={
            "base_parameters": result.branch.params[: result.branch.n_valid],
            "base_states": result.branch.states[: result.branch.n_valid],
            "switched_state": switched.u0,
        },
    )


__all__ = ["run_adaptx_pd_switch"]
