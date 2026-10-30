"""Opt-in independent MatCont Lorenz-84 numerical exports."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest
from examples.validation.cases.systems import lorenz84 as contract

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ROOT / 'examples/validation/regenerate/matcont'
OUT = ROOT / 'examples/validation/generated/candidates'

@pytest.fixture(scope='module')
def native():
    env = os.environ.copy()
    env.setdefault('MATCONT_ROOT', str(Path('~/programms/MatCont/MatCont7p6').expanduser()))
    env['MATLAB_PREFDIR'] = tempfile.mkdtemp(prefix='jc-lorenz84-prefs-')
    matlab = env.get('MATLAB_BIN', str(Path('~/programms/Matlab/R2024a/bin/matlab').expanduser()))
    OUT.mkdir(parents=True, exist_ok=True)
    command = f"addpath('{DRIVERS}');run_lorenz84_equilibria('{OUT}');run_lorenz84_codim2('{OUT}');"
    try:
        result = subprocess.run([matlab, '-batch', command], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired as exc:
        (OUT / 'lorenz84-matcont.log').write_bytes((exc.stdout or b'') + (exc.stderr or b''))
        pytest.fail('MATLAB timed out; see generated/candidates/lorenz84-matcont-test.log')
    (OUT / 'lorenz84-matcont.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return {run: json.loads((OUT / run / 'matcont/native.json').read_text()) for run in ('lorenz84.eq-f','lorenz84.fold-ft','lorenz84.hopf-ft','lorenz84.bt-ft')}

@pytest.mark.matlab
def test_flagship_exports_all_four_runs(native):
    for run, data in native.items():
        assert data['states'] and data['parameters']
        # Measured |error|: 6.66e-16; native float64 Newton stopping tolerance 1e-12.
        assert max(np.atleast_1d(data['residuals'])) < 1e-11
    assert native['lorenz84.hopf-ft']['parameters'][-1][1] < .04
    assert native['lorenz84.fold-ft']['parameters'][-1][1] < .04
    events = native['lorenz84.eq-f']['events']
    assert sorted(e['kind'] for e in events) == ['H', 'H', 'H', 'LP']
    for expected in contract.F_SWEEP['expected_events']:
        contract.select_event(events, expected['kind'], expected['window'])
    assert all(-1.5 <= p[0] <= 3 for p in native['lorenz84.eq-f']['parameters'])
    # A single-point run: MATLAB's jsonencode writes the event list as one object.
    bt = native['lorenz84.bt-ft']['events']
    bt = [bt] if isinstance(bt, dict) else bt
    contract.select_event(bt, 'BT', contract.EVENT_WINDOWS['bt_positive_T'])
    assert not (OUT / 'lorenz84.bt-ft/matcont/failure.json').exists()

@pytest.mark.matlab
def test_matlab_rhs_samples_match_contract(native):
    for sample in native['lorenz84.eq-f']['samples']:
        state, parameters = np.array(sample['state']), sample['parameters']
        np.testing.assert_allclose(sample['rhs'], contract.lorenz84_ft_rhs(state, parameters), atol=3e-7, rtol=3e-7)
        np.testing.assert_allclose(sample['jacobian'], contract.lorenz84_jacobian(state, parameters[0], (*contract.CONSTANTS, parameters[1])), atol=3e-7, rtol=3e-7)


def test_promoted_matlab_rhs_samples_match_contract():
    """Use promoted samples offline once the separately reviewed promotion exists."""
    metadata = ROOT / 'examples/validation/reference/lorenz84/matcont/lorenz84.eq-f/metadata.json'
    if not metadata.exists():
        pytest.skip('Lorenz-84 MatCont promotion is pending Task 2.12')
    for sample in json.loads(metadata.read_text())['samples']:
        np.testing.assert_allclose(sample['rhs'], contract.lorenz84_ft_rhs(np.array(sample['state']), sample['parameters']), atol=3e-7, rtol=3e-7)

@pytest.mark.matlab
def test_refined_events_preserve_spectra_residuals_and_native_platform(native):
    for run in ('lorenz84.eq-f', 'lorenz84.fold-ft', 'lorenz84.hopf-ft'):
        raw = json.loads((OUT / run / 'matcont/raw.json').read_text())
        for event in raw['events']:
            modes = [m for m in raw['spectra'] if m['event_id'] == event['event_id']]
            assert len(modes) == 4
            assert {m['mode_id'] for m in modes} == set(range(4))
            # Measured |error|: 5.55e-17; native float64 Newton stopping tolerance 1e-12.
            assert event['residual_norm'] < 1e-11
            assert event['parameter'] == event['F' if run.endswith('eq-f') else 'T']
            assert event['bracket_status'] in ('native', 'unknown')
            if event['bracket_status'] == 'unknown':
                assert event['bracket_lower'] is None and event['bracket_upper'] is None
            state = np.array([event[f'state_{i}'] for i in range(4)])
            # Near BT, compare spectra in float64 to preserve the double zero eigenvalue.
            expected = np.linalg.eigvals(np.asarray(contract.lorenz84_jacobian(state, event['F'], (*contract.CONSTANTS, event['T'])), dtype=float))
            actual = np.array([complex(m['real'], m['imag']) for m in modes])
            # Measured |error|: 4.56e-09; native versus contract spectra, double-zero eigenvalue conditioning.
            np.testing.assert_allclose(np.sort_complex(actual), np.sort_complex(expected), atol=1e-07, rtol=0)
        platform = raw['metadata']['platform']
        assert platform['os'] and platform['cpu'] and platform['blas']
        assert platform['source'] == 'MATLAB native runtime'
