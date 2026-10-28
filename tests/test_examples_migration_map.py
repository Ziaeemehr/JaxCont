"""The migration audit rejects incomplete plans and false completion claims."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'docs/source/examples-migration.json'


def check(path, mode='--planned'):
    return subprocess.run(
        [sys.executable, str(ROOT / 'scripts/check_examples_migration_map.py'), mode,
         '--map', str(path)], capture_output=True, text=True, cwd=ROOT,
    )


def test_full_inventory_has_a_valid_plan():
    result = check(MAP)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mutation,diagnostic', [
    ('source', 'unmapped source'), ('alias', 'alias collision'),
    ('merge', 'rationale'), ('case', 'unmapped case'),
    ('page', 'comparison page'), ('hash', 'SHA-256'),
    ('tutorial', 'tutorial destination'), ('redirect', 'gallery URL'),
])
def test_invalid_plan_is_rejected(tmp_path, mutation, diagnostic):
    data = json.loads(MAP.read_text())
    if mutation == 'source':
        data['entries'] = [e for e in data['entries'] if e['old_path'] != 'examples/example_02_lorenz.py']
    elif mutation == 'alias':
        data['cases'].append(data['cases'][0])
    elif mutation == 'merge':
        next(e for e in data['entries'] if e['verdict'] == 'merge').pop('rationale')
    elif mutation == 'case':
        data['cases'].pop()
    elif mutation == 'page':
        data['entries'] = [e for e in data['entries'] if 'examples/validation/gallery/plot_lorenz84.py' not in e['new_paths']]
    elif mutation == 'hash':
        next(e for e in data['entries'] if 'old_sha256' in e)['new_sha256'] = '0' * 64
    elif mutation == 'tutorial':
        next(e for e in data['entries'] if e['old_path'] == 'examples/example_02_lorenz.py')['new_paths'] = ['examples/tutorials/wrong.py']
    else:
        data['gallery_redirects'].pop()
    path = tmp_path / 'map.json'
    path.write_text(json.dumps(data))
    result = check(path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert diagnostic in result.stderr


def test_unfinished_tutorials_are_not_reported_as_completed():
    result = check(MAP, '--completed')
    assert result.returncode == 1
    assert 'missing destination' in result.stderr


def test_new_canonical_runs_need_no_legacy_alias_but_unknown_runs_are_rejected(tmp_path):
    from scripts.check_examples_migration_map import audit
    path = tmp_path / "examples/validation/cases/cases.json"
    path.parent.mkdir(parents=True)
    data = json.loads(MAP.read_text())
    run = {"id": "lorenz84.eq-f", "canonical_id": "lorenz84.eq-f", "aliases": []}
    path.write_text(json.dumps({"cases": [run]}))
    assert audit(data, root=tmp_path) == []
    run.update(id="unknown.new", canonical_id="unknown.new")
    path.write_text(json.dumps({"cases": [run]}))
    assert "unmapped case ID: unknown.new" in audit(data, root=tmp_path)
