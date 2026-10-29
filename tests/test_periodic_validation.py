"""Periodic evidence must reject repeated traversals and missing labels."""
import numpy as np
import pytest
from jaxcont.core.collocation import Collocation
from examples.validation.cases.python_cases.periodic import _radial_problem, _radial_rhs
from examples.validation.cases.python_cases.periodic_checks import doubled_shape, require_events, independent_defect


def test_pd_switch_requires_genuine_doubled_shape():
    t = np.linspace(0,2*np.pi,80,endpoint=False)
    base = np.c_[np.cos(2*t), np.sin(2*t)]
    assert doubled_shape(base) < 1e-12
    assert doubled_shape(base + np.c_[.1*np.cos(t), .1*np.sin(t)]) > .1
    assert doubled_shape(np.ones((80,2))) == 0
    assert doubled_shape(base+100) < 1e-12


def test_periodic_reference_requires_complete_event_set():
    events = [dict(event_type=k) for k in ('LPC','PD','NS')]
    require_events(events, ('LPC','PD','NS'))
    for kind in ('LPC','PD','NS'):
        with pytest.raises(ValueError):
            require_events([e for e in events if e['event_type']!=kind], ('LPC','PD','NS'))
    with pytest.raises(ValueError):
        require_events(events + [events[0]], ('LPC','PD','NS'))


def test_radial_mesh_refinement_reduces_independent_defect():
    defects = []
    for ntst in (4,8):
        mesh = Collocation(ntst=ntst,ncol=4)
        problem = _radial_problem(.5,mesh)
        defects.append(independent_defect(_radial_rhs,mesh,problem.u0,problem.p0)['independent_dense_defect'])
    assert defects[1] < defects[0]


def test_prc_parameter_derivative_is_not_time_derivative():
    # Circle: Z=(-sin(t),cos(t))/sqrt(rho); dZ/drho=-Z/(2rho),
    # while dZ/dt rotates Z. Compare API output with independently re-solved
    # rho +/- eps orbits as well as this analytic orthogonality condition.
    from tests.test_prc import _circle_problem
    from jaxcont.stability.prc import prc_curve, dprc_curve
    from examples.validation.cases.python_cases.prc import prc_derivative_conventions
    assert prc_derivative_conventions() == {"jaxcont_dprc": "parameter", "matcont_dprc": "absolute time"}
    prob, mesh = _circle_problem(1.)
    from tests.test_prc import _rhs
    z = np.asarray(prc_curve(_rhs,mesh,prob.u0,prob.p0))
    dp = np.asarray(dprc_curve(prob))
    hi,_ = _circle_problem(1.01)
    lo,_ = _circle_problem(.99)
    fd = (np.asarray(prc_curve(_rhs,mesh,hi.u0,hi.p0))-np.asarray(prc_curve(_rhs,mesh,lo.u0,lo.p0)))/.02
    np.testing.assert_allclose(dp,fd,atol=.01,rtol=0)
    dt = np.c_[-z[:,1],z[:,0]]
    assert np.max(np.abs(dp-dt)) > .5


@pytest.mark.slow
def test_fresh_adaptx_record_preserves_branch_identity():
    from examples.validation.cases.python_cases.pd_switch import produce_adaptx_periodic
    bundle=produce_adaptx_periodic()
    assert {r['segment_id'] for r in bundle['branch']}=={'base','doubled'}
    assert abs(bundle['metadata']['period_ratio']-2)<.005
    pd=next(e for e in bundle['events'] if e['event_type']=='PD')
    critical=[complex(m['real'],m['imag']) for m in bundle['spectra'] if m['event_id']==pd['event_id']]
    assert sum(abs(z+1)<5e-3 for z in critical)==1
    assert all(np.isfinite(r['residual_norm']) for r in bundle['branch'])
    assert max(r['half_period_difference'] for r in bundle['branch'] if r['segment_id']=='doubled')>1e-3


def test_failed_native_flow_mode_stays_failed_in_full_record():
    from types import SimpleNamespace
    from examples.validation.cases.python_cases.periodic_checks import periodic_records
    mesh=Collocation(ntst=4,ncol=4)
    problem=_radial_problem(.5,mesh)
    branch=SimpleNamespace(n_valid=1,states=problem.u0[None,:],params=np.array([.5]),eigenvalues=np.full((1,2),complex(np.nan,np.nan)),trivial_index=np.array([-1]),floquet_logabs=np.full((1,2),np.nan))
    result=SimpleNamespace(branch=branch,events=[])
    with pytest.raises(RuntimeError,match='flow mode'):
        periodic_records(result,mesh,_radial_rhs)
    record=periodic_records(result,mesh,_radial_rhs,allow_failed=True)
    assert record['branch'][0]['status']=='FLOQUET_FAILED'
    assert all(not m['is_trivial'] and np.isnan(m['real']) for m in record['spectra'])
