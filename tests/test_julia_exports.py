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
    if case.endswith('bt-ft'):
        with pytest.raises(RuntimeError):
            run_julia(JULIA/'lorenz84_codim2.jl',['--case',case,'--output-dir',str(output)],600,log_dir=output)
        failure=json.loads((output/'failure.json').read_text())
        assert failure['status']=='FAIL'
        assert 'unique BT' in failure['reason']
        assert not (output/'metadata.json').exists()
        return
    run_julia(JULIA/'lorenz84_codim2.jl',['--case',case,'--output-dir',str(output)],600,log_dir=output)
    bundle=load_bundle(output,case_id=case,tool='bifurcationkit')
    assert bundle['metadata']['parameter_order']==['F','T']
    assert bundle['metadata']['chart_status']
    if case.endswith('bt-ft'):
        point=select_event(bundle['events'],'BT',{'F':[1.5,2.], 'T':[.01,.03]})
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
