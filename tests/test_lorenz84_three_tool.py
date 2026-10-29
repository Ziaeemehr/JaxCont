"""Offline evidence gates for the flagship three-tool Lorenz84 runs."""
from copy import deepcopy
import json
import pytest
from examples.validation.artifacts import load_bundle
from examples.validation.registry import load_registry, select_cases
from examples.validation.validation import build_validation_report
from tests.reference_bundles import REFERENCE


@pytest.mark.parametrize('canonical_id', ['lorenz84.hopf-ft', 'lorenz84.bt-ft'])
def test_flagship_requires_all_three_bundles(canonical_id):
    case, = select_cases(load_registry(), [canonical_id])
    bundles = {tool: load_bundle(REFERENCE / 'lorenz84' / tool / canonical_id,
                               tool=tool, require_reviewed=True)
               for tool in ('matcont', 'bifurcationkit')}
    # Offline runs must remain pending without fresh JaxCont evidence.
    report = build_validation_report(case, bundles)
    assert report.status == 'PENDING'
    for tool, data in bundles.items():
        manifest = json.loads((REFERENCE / 'lorenz84' / tool / canonical_id / 'promotion-manifest.json').read_text())
        assert manifest['verdict'] == manifest['validation_report']['status'] == 'PASS'
        assert set(manifest['independent']) == {'jaxcont', 'matcont', 'bifurcationkit'}
        assert len(manifest['pairwise']) == 3
        assert all(metric['verdict'] == 'PASS' for group in manifest['pairwise'].values()
                   for metric in group.values())
        assert data['metadata']['reviewer'] == 'Claude (review delegated by owner, 2026-10-28)'
    missing = deepcopy(bundles)
    del missing['bifurcationkit']
    assert build_validation_report(case, missing).status != 'PASS'


def test_aliases_and_canonical_ids_select_identical_runs():
    registry = load_registry()
    for case in registry['cases']:
        if not case['canonical_id'].startswith('lorenz84.'):
            continue
        expected = select_cases(registry, [case['canonical_id']])
        for alias in {case['id'], *case.get('aliases', [])}:
            assert select_cases(registry, [alias]) == expected
