"""Figures cannot contradict their numerical evidence or publish partial data."""
import json
from dataclasses import replace

import matplotlib.pyplot as plt
from PIL import Image
import pytest

from examples.validation.reporting import ValidationReport, save_checked_figure
from examples.validation.visualize import _annotate_summary


def report(status='PASS'):
    return ValidationReport('synthetic', status,
        [dict(metric='domain coverage', measured=1., limit=1., passed=True,
              sample_count=5, coverage_fraction=1., source_hashes={'matcont': {'branch.csv': 'abc'}})],
        {}, None, None, 'float64', {'matcont': {'branch.csv': 'abc'}}, {'parameter': [0., 1.]})


def test_figure_and_sidecar_share_status_and_hashes(tmp_path):
    evidence = report()
    figure, _ = plt.subplots()
    target = tmp_path / 'figure.png'
    save_checked_figure(figure, evidence, target)
    sidecar = json.loads(target.with_suffix('.checks.json').read_text())
    with Image.open(target) as png:
        assert json.loads(png.info['Description']) == sidecar == evidence.record()
    assert sidecar['reference_hashes'] == evidence.reference_hashes
    assert sidecar['report_hash'][:12] in ' '.join(t.get_text() for t in figure.texts)
    plt.close(figure)


def test_failed_report_cannot_render_pass_banner(tmp_path):
    evidence = replace(report(), metrics=[dict(metric='difference', measured=2., limit=1., passed=False)])
    assert evidence.status == 'FAIL'
    figure, _ = plt.subplots()
    _annotate_summary(figure, status='PASS', passed=True, details=[])
    save_checked_figure(figure, evidence, tmp_path / 'failed.png')
    texts = ' '.join(t.get_text() for t in figure.texts)
    assert 'FAIL' in texts and 'PASS' not in texts
    plt.close(figure)


def test_pending_tool_is_visibly_pending(tmp_path):
    evidence = report('PENDING')
    figure, _ = plt.subplots()
    save_checked_figure(figure, evidence, tmp_path / 'pending.png')
    assert 'PENDING' in ' '.join(t.get_text() for t in figure.texts)
    assert json.loads((tmp_path / 'pending.checks.json').read_text())['status'] == 'PENDING'
    plt.close(figure)


def test_publication_requires_complete_domain(tmp_path):
    evidence = replace(report(), metrics=[dict(metric='endpoint gap', measured=.5, limit=.01,
        passed=False, coverage_fraction=.5)])
    figure, _ = plt.subplots()
    with pytest.raises(ValueError, match='complete domain'):
        save_checked_figure(figure, evidence, tmp_path / 'partial.png', publication=True)
    assert not (tmp_path / 'partial.png').exists()
    misleading = replace(report(), metrics=[{**report().metrics[0], 'coverage_fraction': .5}])
    with pytest.raises(ValueError, match='complete domain'):
        save_checked_figure(figure, misleading, tmp_path / 'misleading.png', publication=True)
    save_checked_figure(figure, report(), tmp_path / 'complete.png', publication=True)
    plt.close(figure)


def _three_tool_report(monkeypatch, *, missing=None, start=0.):
    """Exercise the real numerical gate without expensive continuation."""
    from examples.validation import reporting
    from examples.validation.cases.python_cases import CaseResult
    case = dict(id='synthetic', canonical_id='synthetic', comparison_window=[0., 1.],
        expected_events=[], spectrum_required=False,
        tool_coverage={tool: 'PASS' for tool in ('jaxcont', 'matcont', 'bifurcationkit')},
        tolerances=dict(parameter_atol=1e-5, state_atol=1e-5, residual_atol=1e-7))
    tools = {}
    for tool in case['tool_coverage']:
        if tool == missing:
            continue
        lower = start if tool == 'matcont' else 0.
        tools[tool] = dict(metadata={'output_hashes': {'branch': tool}, 'package_versions': {tool: 'test'}},
            branch=[dict(segment_id='a', point_id=str(i), parameter=x, state_0=x,
                         residual_norm=0., status='CONVERGED') for i,x in enumerate([lower, .5, 1.])],
            events=[], spectra=[])
    monkeypatch.setattr(reporting, 'comparison_inputs', lambda result, directory: (case, tools))
    return reporting._make_three_tool_report(case, CaseResult('synthetic', {}, {}, {}), '.')


def test_missing_bundle_is_pending_in_actual_gate_and_figure(monkeypatch, tmp_path):
    evidence = _three_tool_report(monkeypatch, missing='bifurcationkit')
    assert evidence.status == 'PENDING'
    assert evidence.evidence['gate']['coverage']['bifurcationkit'] == 'PENDING'
    figure, _ = plt.subplots()
    save_checked_figure(figure, evidence, tmp_path / 'pending-tool.png')
    assert 'PENDING' in ' '.join(t.get_text() for t in figure.texts)
    plt.close(figure)


def test_actual_partial_domain_cannot_publish(monkeypatch, tmp_path):
    evidence = _three_tool_report(monkeypatch, start=.2)
    assert evidence.status == 'FAIL'
    assert any('endpoint_gap' in r['metric'] and not r['passed'] for r in evidence.metrics)
    figure, _ = plt.subplots()
    with pytest.raises(ValueError, match='complete domain'):
        save_checked_figure(figure, evidence, tmp_path / 'partial-domain.png', publication=True)
    plt.close(figure)


def test_renderer_samples_are_the_report_metric_samples(monkeypatch):
    evidence = _three_tool_report(monkeypatch)
    assert evidence.status == 'PASS'
    assert len(evidence.evidence['samples']) == 3
    for pair, curves in evidence.evidence['samples'].items():
        for name, sampled in curves.items():
            row = next(r for r in evidence.metrics if r['metric'] == f'{pair}: {name}')
            assert row['sample_count'] == len(sampled['x']) == 3
            assert max(sampled['error']) == row['measured'] == 0.


def test_renderer_rejects_series_changed_after_report(monkeypatch):
    from types import SimpleNamespace
    from examples.validation import reporting
    from examples.validation.visualize import _checked_comparison_inputs
    evidence = _three_tool_report(monkeypatch)
    case, tools = reporting.comparison_inputs(None, '.')
    tools['matcont']['metadata']['output_hashes'] = {'branch': 'changed'}
    with pytest.raises(ValueError, match='hashes differ'):
        _checked_comparison_inputs(SimpleNamespace(case_id='synthetic'), evidence, '.')
