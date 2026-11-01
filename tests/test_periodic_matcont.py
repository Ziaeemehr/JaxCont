"""Real adaptx Hopf/PD/doubled route, writing candidates only."""
import json
from pathlib import Path
import numpy as np
import pytest
from examples.validation.artifacts import load_bundle
from examples.validation.run_validation import main

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.matlab
def test_adaptx_pd_candidate_has_base_and_doubled_branches():
    output=ROOT/'examples/validation/generated/candidates'/'periodic-matlab-test'
    assert main(['--regenerate','matcont','--case','adaptx.cycle-pd-switch','--output-dir',str(output)])==0
    path=output/'adaptx.cycle-pd-switch/matcont'
    bundle=load_bundle(path,case_id='adaptx.cycle-pd-switch',tool='matcont')
    assert {r['segment_id'] for r in bundle['branch']}=={'base','doubled'}
    # Measured |error|: 0; native float64 seed copies exactly twice the period.
    assert abs(bundle['metadata']['period_ratio']-2)<32 * np.finfo(float).eps
    assert max(float(r['half_period_difference']) for r in bundle['branch'] if r['segment_id']=='doubled')>1e-3
    event=next(e for e in bundle['events'] if e['event_type']=='PD' and e['segment_id']=='base')
    modes=[complex(float(m['real']),float(m['imag'])) for m in bundle['spectra'] if m['event_id']==event['event_id']]
    # Measured |error|: 6.00e-9; native MatCont TestTolerance=1e-7.
    assert min(abs(m+1) for m in modes)<1e-7
    assert (path/'orbits.csv').is_file()


@pytest.mark.matlab
@pytest.mark.slow
def test_jaxcont_switch_matches_explicit_matcont_candidate():
    from examples.validation.cases.python_cases.pd_switch import run_adaptx_pd_switch
    from examples.validation.validation import case_result_passes
    output=ROOT/'examples/validation/generated/candidates'/'periodic-explicit-test'
    assert main(['--regenerate','matcont','--case','adaptx.cycle-pd-switch','--output-dir',str(output)])==0
    result=run_adaptx_pd_switch(output/'adaptx.cycle-pd-switch/matcont')
    assert case_result_passes('MC-LC-003',result.checks)
