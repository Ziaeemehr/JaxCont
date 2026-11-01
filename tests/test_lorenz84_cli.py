"""Candidate routing leaves reviewed evidence untouched."""
import argparse
import json
from pathlib import Path
from examples.validation import run_validation as cli
from examples.validation.registry import load_registry, select_cases
from tests.test_validation_schema import make_bundle
from examples.validation.artifacts import load_bundle


def test_matcont_native_candidate_routes_to_canonical_tool_directory(tmp_path,monkeypatch):
    case=select_cases(load_registry(),['lorenz84.eq-f'])[0]
    fixture=load_bundle(make_bundle(tmp_path/'source'))
    fixture['canonical_id']='lorenz84.eq-f'
    def native(name,**kw):
        path=kw['output_dir']/'lorenz84.eq-f'/'matcont'
        path.mkdir(parents=True,exist_ok=True)
        (path/'raw.json').write_text(json.dumps(fixture))
    monkeypatch.setattr(cli,'_run_matlab_function',native)
    args=argparse.Namespace(generated_dir=tmp_path/'candidates', reference_dir=cli._DEFAULT_REFERENCE_DIR,matlab_bin='matlab',matcont_root='not-used')
    assert cli._regenerate([case],args)
    result=load_bundle(args.generated_dir/'lorenz84.eq-f'/'matcont',case_id='lorenz84.eq-f',tool='matcont')
    assert not result['metadata']['reviewed']


def test_failure_only_matlab_run_cannot_reuse_stale_native_data(tmp_path,monkeypatch):
    case=select_cases(load_registry(),['lorenz84.bt-ft'])[0]
    output=tmp_path/'candidates'
    path=output/'lorenz84.bt-ft'/'matcont'
    fixture=load_bundle(make_bundle(tmp_path/'source'))
    fixture['canonical_id']='lorenz84.bt-ft'
    path.mkdir(parents=True,exist_ok=True)
    (path/'raw.json').write_text(json.dumps(fixture))
    (path/'metadata.json').write_text(json.dumps(fixture['metadata']))
    def fail(name,**kw):
        (path/'failure.json').write_text(json.dumps({'status':'FAIL','reason':'immutable contract'}))
    monkeypatch.setattr(cli,'_run_matlab_function',fail)
    args=argparse.Namespace(generated_dir=output,reference_dir=cli._DEFAULT_REFERENCE_DIR,matlab_bin='matlab',matcont_root='not-used')
    assert cli._regenerate([case],args) is False
    assert not (path/'metadata.json').exists()


def test_registry_preserves_the_reviewed_lorenz_contract():
    from examples.validation.cases.systems.lorenz84 import F_SWEEP, EVENT_WINDOWS
    for case in select_cases(load_registry(),['lorenz84.eq-f','lorenz84.fold-ft','lorenz84.hopf-ft','lorenz84.bt-ft']):
        assert case['model_contract']['event_windows']==EVENT_WINDOWS
        assert case['model_contract']['f_sweep']==F_SWEEP
