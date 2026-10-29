"""MatCont ``testadapt2`` cross-validation for period-doubling switching.

External values are read from an explicit candidate or a reviewed v2 bundle.
The JaxCont path transcribes only
``TestSystems/adaptx.m`` and independently continues its limit-cycle branch;
it never consumes MatCont's orbit data as a seed.
"""

from __future__ import annotations

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

import jaxcont as jc
from jaxcont.bifurcations.events import EventHit
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

from . import CaseResult

_CONTRACT = json.loads((Path(__file__).parents[1] / "systems/periodic.json").read_text())["adaptx"]

def _matcont_pd_reference(reference_dir=None):
    """Only explicit candidates or reviewed bundles may supply external values.

    Promotion is Task 2.12. Default offline runs remain PENDING until the
    owner has reviewed that bundle; candidates are never discovered implicitly.
    """
    from ...artifacts import load_bundle
    path = Path(reference_dir) if reference_dir else Path(__file__).resolve().parents[2] / 'reference/adaptx/matcont/adaptx.cycle-pd-switch'
    if not (path / 'metadata.json').exists():
        return None
    bundle = load_bundle(path, case_id='adaptx.cycle-pd-switch', tool='matcont')
    if reference_dir is None and bundle['metadata'].get('reviewed') is not True:
        raise ValueError('Default adaptx MatCont reference must be reviewed')
    contract = json.loads((Path(__file__).parents[1] / 'systems/periodic.json').read_text())['adaptx']
    def event(segment, window):
        rows = [e for e in bundle['events'] if e['segment_id']==segment and e['event_type']=='PD' and window[0]<=float(e['parameter'])<=window[1]]
        if len(rows)!=1:
            raise ValueError('Missing or duplicate MatCont PD in computed window')
        return rows[0]
    pd = event('base', contract['pd_window'])
    doubled_pd = event('doubled', contract['doubled_pd_window'])
    start = next(r for r in bundle['branch'] if r['segment_id']=='doubled')
    return dict(alpha=float(pd['parameter']),period=float(pd['period']),doubled_start_period=float(start['period']),doubled_alpha=float(doubled_pd['parameter']),doubled_period=float(doubled_pd['period']))


_ALPHA_MATCH_ATOL = 1e-3

def _adaptx_rhs(u, alpha, args):
    """``Testruns/TestSystems/adaptx.m`` with beta fixed to MatCont's one."""
    del args
    x, y, z = u
    return jnp.array([y, z, -alpha * z - y - x + x * x])


def _base_cycle_problem():
    """Build a local Hopf-direction seed without using a MatCont orbit."""
    alpha = _CONTRACT["base_domain"][1]
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
        and _CONTRACT["pd_window"][0] <= parameters[index] <= _CONTRACT["pd_window"][1]
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
    from .periodic_checks import doubled_shape
    return doubled_shape(states)


def run_adaptx_pd_switch(reference_dir=None) -> CaseResult:
    """Find adaptx's PD point, switch it, and check MatCont live references."""
    reference = _matcont_pd_reference(reference_dir)
    reference_status = "PENDING" if reference is None else "CANDIDATE_OR_REVIEWED"
    ref = reference or dict.fromkeys(("alpha","period","doubled_start_period","doubled_alpha","doubled_period"), float("inf"))
    dp = 0.05
    problem, mesh = _base_cycle_problem()
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=tuple(reversed(_CONTRACT["base_domain"])),
        settings=jc.ContinuationPar(
            # PD seeding selects a raw crossing endpoint at this Euclidean spacing.
            theta=1.0,
            compute_stability=True,
            ds=_CONTRACT["ds"],
            ds_min=_CONTRACT["ds_min"],
            ds_max=_CONTRACT["ds_max"],
            max_steps=_CONTRACT["max_steps"],
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
    from .periodic_checks import independent_defect

    case_result = CaseResult(
        case_id="MC-LC-003",
        checks={
            "pd_found": True,
            "pd_parameter_error": abs(float(event.p) - ref["alpha"]),
            "pd_period_error": abs(float(event.u[-1]) - ref["period"]),
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
                float(switched.u0[-1]) - ref["doubled_start_period"]
            ),
            "doubled_period_ratio_error": abs(
                float(switched.u0[-1]) / float(event.u[-1]) - 2.0
            ),
            # Tightest evidence that JaxCont's doubled branch IS MatCont's:
            # two points at the same alpha on the same branch share a period.
            # Only meaningful while the alphas actually match -- hence the
            # guard immediately below, which must be asserted alongside it.
            "doubled_branch_alpha_gap": abs(
                float(switched.p0) - ref["doubled_alpha"]
            ),
            "doubled_branch_alphas_match": (
                abs(float(switched.p0) - ref["doubled_alpha"])
                < _ALPHA_MATCH_ATOL
            ),
            "doubled_branch_period_error": abs(
                float(switched.u0[-1]) - ref["doubled_period"]
            ),
            "switch_residual": residual,
            "genuine_doubled_orbit": difference > 1e-3,
            **independent_defect(_adaptx_rhs, switched_mesh, switched.u0, switched.p0),
        },
        observations={
            "reference_status": reference_status,
            "hopf": {"alpha": 1.0, "period": float(2*np.pi), "method": "exact Jacobian of adaptx at origin"},
            "jaxcont_pd": {"alpha": float(event.p), "period": float(event.u[-1])},
            "jaxcont_switched": {
                "alpha": float(switched.p0),
                "period": float(switched.u0[-1]),
                "half_period_difference": difference,
            },
            "matcont_base_pd": {
                "alpha": ref["alpha"],
                "period": ref["period"],
            },
            "matcont_doubled_start": {
                "alpha": ref["alpha"],
                "period": ref["doubled_start_period"],
            },
        },
        artifacts={
            "base_parameters": result.branch.params[: result.branch.n_valid],
            "base_states": result.branch.states[: result.branch.n_valid],
            "switched_state": switched.u0,
            "base_result": result,
            "base_problem": problem,
            "mesh": mesh,
            "switched_problem": switched,
            "pd_event": event,
        },
    )

    case_result.checks['matcont_reference_present'] = reference is not None
    if reference is None:
        for key in ('pd_parameter_error','pd_period_error','doubled_period_error','doubled_branch_alpha_gap','doubled_branch_period_error'):
            case_result.checks[key] = None
        case_result.observations['matcont_base_pd'] = None
        case_result.observations['matcont_doubled_start'] = None
    return case_result



__all__ = ["run_adaptx_pd_switch"]


def _refined_pd_event(result, problem, window=None):
    """Public event refinement on the computed branch, using its flow modes."""
    from jaxcont.bifurcations.events import BranchPoint
    branch=result.branch
    window = _CONTRACT['pd_window'] if window is None else window
    points=[BranchPoint(p=float(p),u=u,eigenvalues=ev,trivial_index=triv) for p,u,ev,triv in zip(branch.params,branch.states,branch.eigenvalues,branch.trivial_index)]
    detector=jc.PeriodDoubling(corrector_tol=1e-9 if problem.u0.dtype==jnp.float64 else 1e-5)
    brackets=[(i,j,mode) for i,j,mode in detector.scan(points) if window[0] <= (points[i].p+points[j].p)/2 <= window[1]]
    if len(brackets)!=1:
        raise RuntimeError('Expected one native adaptx PD bracket in frozen window')
    i,j,mode=brackets[0]
    args=problem.step_update(problem.args,points[i].u,points[i].p)
    rhs=jax.jit(lambda u,p:problem.f(u,p,args))
    hit=detector.refine(points[i],points[j],(i,j),rhs,tolerance=1e-7,max_iterations=40,mode=mode,args=args)
    if hit is None or hit.status!='validated':
        raise RuntimeError(f'Adaptx PD refinement failed: {hit}')
    return hit


def produce_adaptx_periodic(reference_dir=None):
    """Fresh base, PD and doubled-branch tables; external evidence is optional."""
    from .periodic_checks import periodic_records, mesh_refinement
    result=run_adaptx_pd_switch(reference_dir)
    a=result.artifacts; mesh=a['mesh']; switched=a['switched_problem']
    from types import SimpleNamespace
    refined=_refined_pd_event(a['base_result'],a['base_problem'])
    native=SimpleNamespace(branch=a['base_result'].branch,events=[refined])
    tables=periodic_records(native,mesh,_adaptx_rhs,segment='base')
    doubled_mesh=Collocation(ntst=2*mesh.ntst,ncol=mesh.ncol)
    from types import SimpleNamespace
    settings=jc.ContinuationPar(theta=1.,compute_stability=True,ds=_CONTRACT['ds'],ds_min=_CONTRACT['ds_min'],ds_max=_CONTRACT['ds_max'],max_steps=_CONTRACT['max_steps'],newton_tol=1e-5)
    down=jc.continuation(switched,jc.PseudoArclength(),p_span=(float(switched.p0),_CONTRACT['doubled_domain'][0]),settings=settings)
    up=jc.continuation(switched,jc.PseudoArclength(),p_span=(float(switched.p0),_CONTRACT['doubled_domain'][1]),settings=settings)
    # Join the two computed directions at their shared seed, preserving
    # continuation order. No sorting/deduplication across a fold is used.
    nu,nd=int(up.branch.n_valid),int(down.branch.n_valid)
    states=jnp.concatenate([up.branch.states[:nu][::-1],down.branch.states[1:nd]])
    params=jnp.concatenate([up.branch.params[:nu][::-1],down.branch.params[1:nd]])
    doubled=SimpleNamespace(branch=SimpleNamespace(states=states,params=params,n_valid=len(params),eigenvalues=jnp.concatenate([up.branch.eigenvalues[:nu][::-1],down.branch.eigenvalues[1:nd]]),trivial_index=jnp.concatenate([up.branch.trivial_index[:nu][::-1],down.branch.trivial_index[1:nd]]),floquet_logabs=jnp.concatenate([up.branch.floquet_logabs[:nu][::-1],down.branch.floquet_logabs[1:nd]])),events=[])
    doubled.events = [_refined_pd_event(doubled, switched, _CONTRACT['doubled_pd_window'])]
    second=periodic_records(doubled,doubled_mesh,_adaptx_rhs,segment='doubled')
    for key in tables:
        tables[key].extend(second[key])
    e=refined
    if float(jnp.min(params))>_CONTRACT['doubled_domain'][0] or float(jnp.max(params))<_CONTRACT['doubled_domain'][1]:
        raise RuntimeError('JaxCont doubled branch did not cover the full frozen domain')
    tables['canonical_id']='adaptx.cycle-pd-switch'
    tables['metadata']=dict(canonical_id='adaptx.cycle-pd-switch',tool='jaxcont',precision=str(a['base_problem'].u0.dtype),hopf=result.observations['hopf'],period_ratio=float(switched.u0[-1]/e.u[-1]),refinement=mesh_refinement(_adaptx_rhs,mesh,a['base_problem'].u0,a['base_problem'].p0),reference_status=result.observations['reference_status'],switch_seed_pd_alpha=float(a['pd_event'].p),refined_pd_alpha=float(refined.p))
    import platform
    tables['metadata'].update(runtime_version=platform.python_version(),package_versions={'JaxCont':jc.__version__,'JAX':jax.__version__},parameter_order=['alpha','beta'],scaling={'time':'original','metric':'theta=1 Euclidean collocation coordinates'},seeds={'alpha':float(a['base_problem'].p0),'beta':1,'method':'Hopf direction harmonic balance'},mesh={'ntst':mesh.ntst,'ncol':mesh.ncol,'doubled_ntst':doubled_mesh.ntst},resolved_controls={k:_CONTRACT[k] for k in ('ds','ds_min','ds_max','max_steps')},solver_strategies={'method':'JaxCont PseudoArclength','event_selection':'real Floquet -1 crossing and frozen window'})
    tables['metadata']['resolved_controls'].update(newton_atol=1e-5,event_atol=1e-7,event_corrector_atol=1e-9 if a['base_problem'].u0.dtype==jnp.float64 else 1e-5,event_bisection_iterations=40)
    return tables
