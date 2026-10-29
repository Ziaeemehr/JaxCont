"""Offline readers for reviewed expectations and explicitly historical evidence."""
import json
from pathlib import Path
from examples.validation.artifacts import file_hash, load_bundle

REFERENCE = Path(__file__).resolve().parents[1] / 'examples/validation/reference'


def historical_values(name):
    path = REFERENCE / 'historical/bifurcationkit' / name
    meta = json.loads((path / 'metadata.json').read_text())
    assert meta['schema_version'] == 1 and meta['tool'] == 'bifurcationkit'
    assert meta['reviewed'] is False and meta['source_revision']
    assert file_hash(path / 'values.json') == meta['output_hashes']['values.json']
    return json.loads((path / 'values.json').read_text())


def reviewed_bt():
    data = load_bundle(REFERENCE / 'lorenz84/bifurcationkit/lorenz84.bt-ft',
                       tool='bifurcationkit', require_reviewed=True)
    row, = data['branch']
    return tuple(float(row[f'state_{i}']) for i in range(4)), tuple(float(row[k]) for k in ('F', 'T'))
