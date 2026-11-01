"""Candidate completeness and source identity are distinct from successful I/O."""
from pathlib import Path
from examples.validation.candidates import write_candidate
from examples.validation.artifacts import load_bundle
from tests.test_validation_schema import make_bundle


def test_step_budget_shortfall_is_partial_even_when_rows_converge(tmp_path):
    native=load_bundle(make_bundle(tmp_path/'fixture'))
    native['canonical_id']='lorenz84.fold-ft'
    for row in native['branch']:
        row['T']=.04-float(row['order'])*.01
    driver=tmp_path/'driver.m'; driver.write_text('% native driver')
    result=write_candidate(native,tmp_path/'candidate',tool='matcont',driver=driver,equation=driver)
    metadata=result['metadata']
    assert metadata['coverage_status']=='PARTIAL'
    assert metadata['observed_domain']==[.03,.04]
    assert metadata['expected_domain']==[-.01,.04]
    assert 'step budget' in metadata['chart_status']


def test_numerical_matlab_helper_changes_candidate_source_identity(tmp_path):
    native=load_bundle(make_bundle(tmp_path/'fixture'))
    native['canonical_id']='synthetic.eq'
    driver=tmp_path/'driver.m';driver.write_text('% native driver')
    helper=tmp_path/'lorenz84_seed.m';helper.write_text('% seed controls v1')
    one=write_candidate(native,tmp_path/'candidate1',tool='matcont',driver=driver,equation=driver)
    helper.write_text('% seed controls v2')
    two=write_candidate(native,tmp_path/'candidate2',tool='matcont',driver=driver,equation=driver)
    assert one['metadata']['source_hashes']['lorenz84_seed.m'] != two['metadata']['source_hashes']['lorenz84_seed.m']


def test_python_numerical_failure_invalidates_previous_candidate(tmp_path,monkeypatch):
    import json
    import pytest
    from examples.validation.cases.python_cases import lorenz84
    output=tmp_path/'candidate';output.mkdir()
    (output/'metadata.json').write_text('{"schema_version":2}')
    def fail(case):
        raise ValueError('strict Newton tolerance could not be met')
    monkeypatch.setattr(lorenz84,'produce',fail)
    with pytest.raises(ValueError):
        lorenz84.export('lorenz84.eq-f',output)
    assert not (output/'metadata.json').exists()
    assert json.loads((output/'failure.json').read_text())['status']=='FAIL'
