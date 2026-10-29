"""Native Julia candidate exports, explicitly opt in to runtime execution."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from examples.validation.artifacts import load_bundle
from examples.validation.cases.systems.lorenz84 import lorenz84_rhs, lorenz84_jacobian, FIXED_ARGS, select_event, F_SWEEP
from examples.validation.regenerate.bifurcationkit.runner import run_julia

ROOT=Path(__file__).resolve().parents[1]
JULIA=ROOT/'examples/validation/regenerate/bifurcationkit'

def check_samples(samples):
    for sample in samples:
        args=(*FIXED_ARGS[:5],sample['T'])
        np.testing.assert_allclose(sample['rhs'],lorenz84_rhs(np.array(sample['state']), sample['F'],args), atol=2e-7,rtol=1e-6)
        np.testing.assert_allclose(sample['jacobian'],lorenz84_jacobian(np.array(sample['state']),sample['F'],args),atol=2e-7,rtol=1e-6)

def test_julia_rhs_matches_contract_samples():
    # Independent polynomial evaluation at a non-equilibrium sample.
    check_samples([dict(state=[1.,.2,-.3,.1],F=2.,T=.04,
        rhs=[.11013,.55,.2,.0347],jacobian=[[-.25,-.4,.6,-.1974],[.5,0,-1,0],[-.1,1,0,0],[.0987,0,0,-.053]])])

def test_equilibrium_export_schema_and_conventions(tmp_path):
    from tests.test_validation_schema import make_bundle
    bundle=load_bundle(make_bundle(tmp_path/'fixture',tool='bifurcationkit'))
    assert bundle['metadata']['tool']=='bifurcationkit'
    assert bundle['metadata']['reviewed'] is False

@pytest.mark.julia
@pytest.mark.parametrize('case,script', [('cubic.eq-fold','cubic_fold.jl'),('radial.cycle-floquet','radial_cycle.jl'),('lorenz84.eq-f','lorenz84_equilibria.jl'),('hopf-normal-form.l1','hopf_normal_form.jl')])
def test_native_exports(case,script):
    output=ROOT/'examples/validation/generated/candidates'/case/'bifurcationkit'
    run_julia(JULIA/script,['--case',case,'--output-dir',str(output)],600,log_dir=output)
    bundle=load_bundle(output,case_id=case,tool='bifurcationkit')
    assert len(bundle['branch'])>1
    if case=='lorenz84.eq-f':
        check_samples(json.loads((output/'samples.json').read_text()))
        for e in F_SWEEP['expected_events']:
            select_event(bundle['events'],e['kind'],e['window'])
    if case=='hopf-normal-form.l1':
        assert len(bundle['events'])==1
        assert float(bundle['events'][0]['l1_native_real']) < 0
        assert bundle['metadata']['conventions']['normal_form']
    if case=='radial.cycle-floquet':
        check_radial_modes(bundle)
        assert (output/'orbits.csv').is_file()
        assert max(float(r['residual_norm']) for r in bundle['branch']) < 1e-7

@pytest.mark.julia
@pytest.mark.parametrize('case',['lorenz84.fold-ft','lorenz84.hopf-ft','lorenz84.bt-ft'])
def test_codim2_export_preserves_parameter_order_and_event_identity(case):
    output=ROOT/'examples/validation/generated/candidates'/case/'bifurcationkit'
    run_julia(JULIA/'lorenz84_codim2.jl',['--case',case,'--output-dir',str(output)],600,log_dir=output)
    bundle=load_bundle(output,case_id=case,tool='bifurcationkit')
    assert bundle['metadata']['parameter_order']==['F','T']
    assert bundle['metadata']['chart_status']
    if case.endswith('bt-ft'):
        point=select_event(bundle['events'],'BT',{'F':[1.40,1.50], 'T':[.01,.03]})
        assert float(point['residual_norm'])<1e-7
        assert int(point['geometric_nullity'])==1


def check_radial_modes(bundle):
    """Native radial spectra identify the flow mode by eigenvector alignment."""
    modes=bundle['spectra']
    assert all(r['trivial_mode_method']=='flow alignment at orbit phase zero' for r in modes)
    for point in bundle['branch']:
        selected=[r for r in modes if r['point_id']==point['point_id'] and int(r['is_trivial'])]
        assert len(selected)==1
        assert float(selected[0]['flow_alignment'])>.99


def run_periodic_native(case, script):
    output=ROOT/'examples/validation/generated/candidates'/('periodic-test-x'+os.environ.get('JAXCONT_TEST_X64','0'))/case/'bifurcationkit'
    result=run_julia(JULIA/script,['--case',case,'--output-dir',str(output)],600,log_dir=output)
    assert 'soft scope' not in result.stderr.lower()
    bundle=load_bundle(output,case_id=case,tool='bifurcationkit')
    assert (output/'orbits.csv').is_file()
    assert bundle['metadata']['coverage_status']=='COMPLETE'
    assert bundle['metadata']['comparable'] is False
    refinement=bundle['metadata']['refinement']
    assert refinement['fine']['independent_dense_defect'] < refinement['coarse']['independent_dense_defect']
    return bundle


@pytest.mark.julia
def test_pd_export_has_base_and_doubled_orbit_identity():
    bundle=run_periodic_native('adaptx.cycle-pd-switch','adaptx_pd.jl')
    assert {r['segment_id'] for r in bundle['branch']}=={'base','doubled'}
    assert abs(bundle['metadata']['period_ratio']-2)<.005
    doubled=[r for r in bundle['branch'] if r['segment_id']=='doubled']
    assert max(float(r['half_period_difference']) for r in doubled)>1e-3
    pd=[e for e in bundle['events'] if e['event_type']=='PD' and e['segment_id']=='base']
    assert len(pd)==1
    critical=[complex(float(m['real']),float(m['imag'])) for m in bundle['spectra'] if m['event_id']==pd[0]['event_id']]
    assert sum(abs(z+1)<1e-5 for z in critical)==1
    window=json.loads((ROOT/'examples/validation/cases/systems/periodic.json').read_text())['adaptx']['doubled_pd_window']
    dp=[e for e in bundle['events'] if e['event_type']=='PD' and e['segment_id']=='doubled' and window[0]<=float(e['parameter'])<=window[1]]
    assert len(dp)==1
    dm=[complex(float(m['real']),float(m['imag'])) for m in bundle['spectra'] if m['event_id']==dp[0]['event_id']]
    assert sum(abs(z+1)<1e-5 for z in dm)==1


@pytest.mark.julia
def test_torbpc_export_requires_full_expected_event_domain():
    bundle=run_periodic_native('torbpc.cycle-lpc-pd-ns','torbpc_ns.jl')
    from examples.validation.cases.python_cases.periodic_checks import require_events
    contract=json.loads((ROOT/'examples/validation/cases/systems/periodic.json').read_text())['torbpc']
    expected=[e for e in bundle['events'] if e['event_type'] in ('LPC','PD','NS') and contract[e['event_type'].lower()+'_window'][0]<=float(e['parameter'])<=contract[e['event_type'].lower()+'_window'][1]]
    require_events(expected,('LPC','PD','NS'))
    assert min(float(r['parameter']) for r in bundle['branch'])<=-.747263543778405
    ns=next(e for e in expected if e['event_type']=='NS')
    lpc=next(e for e in expected if e['event_type']=='LPC')
    lm=[complex(float(m['real']),float(m['imag'])) for m in bundle['spectra'] if m['event_id']==lpc['event_id']]
    assert sum(abs(z-1)<1e-4 for z in lm)==2
    assert .01<float(ns['ns_angle'])<np.pi-.01
    assert float(ns['crossing_logabs_before'])*float(ns['crossing_logabs_after'])<0
