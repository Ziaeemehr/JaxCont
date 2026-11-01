"""Audit the planned inventory or completed examples migration from the repo root."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / 'docs/source/examples-migration.json'
# Frozen pre-migration inventory; checking only today's files would silently lose
# coverage as sources move or disappear.
SOURCES = frozenset(['examples/BifurcationKit/01_pitchfork.jl',
 'examples/BifurcationKit/02_lorenz84.jl',
 'examples/BifurcationKit/03_neural_mass.jl',
 'examples/BifurcationKit/04_hopf_normal_form.jl',
 'examples/BifurcationKit/05_codim2.jl',
 'examples/GALLERY_HEADER.rst',
 'examples/MatCont/GALLERY_HEADER.txt',
 'examples/MatCont/README.md',
 'examples/MatCont/__init__.py',
 'examples/MatCont/artifacts.py',
 'examples/MatCont/cases.json',
 'examples/MatCont/compare.py',
 'examples/MatCont/generated/.gitignore',
 'examples/MatCont/matlab/export_cycle_run.m',
 'examples/MatCont/matlab/export_equilibrium_run.m',
 'examples/MatCont/matlab/export_prc_run.m',
 'examples/MatCont/matlab/run_adaptive_control.m',
 'examples/MatCont/matlab/run_cubic_fold.m',
 'examples/MatCont/matlab/run_prc_dprc.m',
 'examples/MatCont/matlab/run_radial_cycle.m',
 'examples/MatCont/matlab/run_supported.m',
 'examples/MatCont/matlab/run_torbpc_cycle.m',
 'examples/MatCont/matlab/run_vanderpol_hopf.m',
 'examples/MatCont/matlab/setup_matcont.m',
 'examples/MatCont/matlab/systems/adaptive_control.m',
 'examples/MatCont/matlab/systems/cubic_fold.m',
 'examples/MatCont/matlab/systems/radial_cycle_system.m',
 'examples/MatCont/matlab/systems/vanderpol_hopf.m',
 'examples/MatCont/matlab/unsupported/run_bratu_branch_switching.m',
 'examples/MatCont/matlab/unsupported/run_connecting_orbits.m',
 'examples/MatCont/matlab/unsupported/run_general_bvp_continuation.m',
 'examples/MatCont/matlab/unsupported/run_heteroclinic_continuation.m',
 'examples/MatCont/matlab/unsupported/run_two_parameter_fold_curve.m',
 'examples/MatCont/matlab/unsupported/run_two_parameter_hopf_curve.m',
 'examples/MatCont/matlab/unsupported/run_two_parameter_lpc_curve.m',
 'examples/MatCont/matlab/unsupported/run_two_parameter_ns_curve.m',
 'examples/MatCont/matlab/unsupported/run_two_parameter_pd_curve.m',
 'examples/MatCont/python_cases/__init__.py',
 'examples/MatCont/python_cases/codim2.py',
 'examples/MatCont/python_cases/codim2_curves.py',
 'examples/MatCont/python_cases/equilibrium.py',
 'examples/MatCont/python_cases/pd_switch.py',
 'examples/MatCont/python_cases/periodic.py',
 'examples/MatCont/python_cases/prc.py',
 'examples/MatCont/python_cases/transforms.py',
 'examples/MatCont/reference/MC-EQ-001_branch.csv',
 'examples/MatCont/reference/MC-EQ-001_events.csv',
 'examples/MatCont/reference/MC-EQ-001_metadata.json',
 'examples/MatCont/reference/MC-EQ-001_multipliers.csv',
 'examples/MatCont/reference/MC-EQ-002_branch.csv',
 'examples/MatCont/reference/MC-EQ-002_events.csv',
 'examples/MatCont/reference/MC-EQ-002_metadata.json',
 'examples/MatCont/reference/MC-EQ-002_multipliers.csv',
 'examples/MatCont/reference/MC-EQ-003_branch.csv',
 'examples/MatCont/reference/MC-EQ-003_events.csv',
 'examples/MatCont/reference/MC-EQ-003_metadata.json',
 'examples/MatCont/reference/MC-EQ-003_multipliers.csv',
 'examples/MatCont/reference/MC-LC-001_branch.csv',
 'examples/MatCont/reference/MC-LC-001_events.csv',
 'examples/MatCont/reference/MC-LC-001_metadata.json',
 'examples/MatCont/reference/MC-LC-001_multipliers.csv',
 'examples/MatCont/reference/MC-LC-002_branch.csv',
 'examples/MatCont/reference/MC-LC-002_events.csv',
 'examples/MatCont/reference/MC-LC-002_metadata.json',
 'examples/MatCont/reference/MC-LC-002_multipliers.csv',
 'examples/MatCont/reference/MC-PRC-001_metadata.json',
 'examples/MatCont/reference/MC-PRC-001_prc.csv',
 'examples/MatCont/registry.py',
 'examples/MatCont/run_validation.py',
 'examples/MatCont/validation.py',
 'examples/MatCont/visualize.py',
 'examples/__init__.py',
 'examples/check_installation.py',
 'examples/demo_jit_speedup.py',
 'examples/example_01_pitchfork.py',
 'examples/example_02_lorenz.py',
 'examples/example_03_van_der_pol.py',
 'examples/example_04_continuation_methods.py',
 'examples/example_05_neural_mass.py',
 'examples/example_06_vmap_sweep.py',
 'examples/example_07_differentiable.py',
 'examples/example_08_period_doubling.py',
 'examples/example_09_neimark_sacker.py',
 'examples/example_10_van_der_pol_limit_cycle.py',
 'examples/example_11_brusselator_limit_cycle.py',
 'examples/example_12_fitzhugh_nagumo_phase_plane.py',
 'examples/example_13_phase_response_curve.py',
 'examples/example_14_prc_shooting_validation.py',
 'examples/example_15_two_parameter_diagram.py',
 'examples/example_16_matcont_cubic_overlay.py',
 'examples/example_17_matcont_vanderpol_overlay.py',
 'examples/example_18_matcont_adaptive_control_overlay.py',
 'examples/example_19_matcont_radial_cycle_overlay.py',
 'examples/example_20_matcont_torbpc_overlay.py',
 'examples/example_21_pitchfork_branch_switch.py',
 'examples/example_22_period_doubling_switch.py',
 'examples/example_23_periodic_vmap_grad.py',
 'examples/profile_continuation.py'])
CASE_ALIASES = {'MC-C2-001': 'analytic.codim2',
 'MC-EQ-001': 'cubic.eq-fold',
 'MC-EQ-002': 'vanderpol.eq-degenerate-hopf',
 'MC-EQ-003': 'adaptx.eq-hopf',
 'MC-JAX-001': 'jax.transforms',
 'MC-LC-001': 'radial.cycle-floquet',
 'MC-LC-002': 'torbpc.cycle-lpc-pd-ns',
 'MC-LC-003': 'adaptx.cycle-pd-switch',
 'MC-PRC-001': 'adaptx.prc',
 'US-BP-001': 'US-BP-001',
 'US-BVP-001': 'US-BVP-001',
 'US-C2-H-001': 'catalytic.hopf-curve',
 'US-C2-LP-001': 'catalytic.fold-curve',
 'US-C2-LPC-001': 'US-C2-LPC-001',
 'US-C2-NS-001': 'US-C2-NS-001',
 'US-C2-PD-001': 'US-C2-PD-001',
 'US-HET-001': 'US-HET-001',
 'US-HOM-001': 'US-HOM-001'}
CANONICAL_IDS = frozenset(['adaptx.cycle-pd-switch',
 'adaptx.eq-hopf',
 'adaptx.prc',
 'analytic.codim2',
 'catalytic.fold-curve',
 'catalytic.hopf-curve',
 'cubic.eq-fold',
 'hopf-normal-form.l1',
 'jax.transforms',
 'lorenz84.bt-ft',
 'lorenz84.eq-f',
 'lorenz84.fold-ft',
 'lorenz84.hopf-ft',
 'radial.cycle-floquet',
 'shooting.prc-check',
 'torbpc.cycle-lpc-pd-ns',
 'vanderpol.eq-degenerate-hopf'])
TUTORIAL_DESTINATIONS = {'examples/example_01_pitchfork.py': ['examples/tutorials/plot_01_cubic_fold.py'],
 'examples/example_02_lorenz.py': ['examples/tutorials/plot_03_lorenz84.py'],
 'examples/example_03_van_der_pol.py': ['examples/tutorials/plot_04_vanderpol_equilibrium.py'],
 'examples/example_04_continuation_methods.py': ['examples/tutorials/plot_01_cubic_fold.py'],
 'examples/example_05_neural_mass.py': ['examples/tutorials/plot_05_neural_mass.py'],
 'examples/example_06_vmap_sweep.py': ['examples/tutorials/plot_07_vmap_sweep.py'],
 'examples/example_07_differentiable.py': ['examples/tutorials/plot_08_differentiable.py'],
 'examples/example_08_period_doubling.py': ['examples/tutorials/plot_13_period_doubling.py'],
 'examples/example_09_neimark_sacker.py': ['examples/tutorials/plot_15_neimark_sacker.py'],
 'examples/example_10_van_der_pol_limit_cycle.py': ['examples/tutorials/plot_09_vanderpol_cycle.py'],
 'examples/example_11_brusselator_limit_cycle.py': ['examples/tutorials/plot_10_brusselator_cycle.py'],
 'examples/example_12_fitzhugh_nagumo_phase_plane.py': ['examples/tutorials/plot_11_phase_plane.py'],
 'examples/example_13_phase_response_curve.py': ['examples/tutorials/plot_12_prc.py'],
 'examples/example_15_two_parameter_diagram.py': ['examples/tutorials/plot_06_two_parameter.py'],
 'examples/example_21_pitchfork_branch_switch.py': ['examples/tutorials/plot_02_pitchfork_switch.py'],
 'examples/example_22_period_doubling_switch.py': ['examples/tutorials/plot_14_pd_switch.py'],
 'examples/example_23_periodic_vmap_grad.py': ['examples/tutorials/plot_16_periodic_transforms.py']}
COMPARISON_PAGES = frozenset(['examples/validation/gallery/plot_adaptx.py',
 'examples/validation/gallery/plot_catalytic.py',
 'examples/validation/gallery/plot_cubic.py',
 'examples/validation/gallery/plot_hopf_normal_form.py',
 'examples/validation/gallery/plot_lorenz84.py',
 'examples/validation/gallery/plot_prc_shooting.py',
 'examples/validation/gallery/plot_radial.py',
 'examples/validation/gallery/plot_torbpc.py',
 'examples/validation/gallery/plot_vanderpol.py'])


def audit(data: dict, *, completed: bool = False, root: Path = ROOT) -> list[str]:
    """Return all inventory, disposition, alias, and byte-identity violations."""
    errors = []
    entries = data['entries']
    sources = [e['old_path'] for e in entries if e['old_path'] is not None]
    destinations = {p for e in entries for p in e['new_paths']}
    for source in sorted(SOURCES - set(sources)):
        errors.append(f'unmapped source: {source}')
    # Also reject newly added legacy files that were not in the frozen inventory.
    for directory in ('examples/MatCont', 'examples/BifurcationKit'):
        for path in (root / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and 'generated' not in path.parts:
                relative = path.relative_to(root).as_posix()
                if relative not in sources:
                    errors.append(f'unmapped source: {relative}')
    for source, count in Counter(sources).items():
        if count > 1:
            errors.append(f'duplicate source: {source}')
    for source, expected in TUTORIAL_DESTINATIONS.items():
        row = next((e for e in entries if e['old_path'] == source), None)
        if row is not None and row['new_paths'] != expected:
            errors.append(f'incorrect tutorial destination: {source}')
    for page in sorted(COMPARISON_PAGES - destinations):
        errors.append(f'unmapped comparison page: {page}')
    for entry in entries:
        old, new = entry['old_path'], entry['new_paths']
        verdict = entry['verdict']
        if verdict not in {'move', 'merge', 'split', 'rewrite', 'drop', 'create'}:
            errors.append(f'invalid verdict: {old}')
        if not entry.get('coverage_ids') or (not new and verdict != 'drop'):
            errors.append(f'missing destination or coverage ID: {old}')
        if verdict in {'merge', 'split', 'drop'} and not entry.get('rationale', '').strip():
            errors.append(f'missing {verdict} rationale: {old}')
        for path in ([old] if old else []) + new:
            if Path(path).is_absolute() or '..' in Path(path).parts:
                errors.append(f'path must be repository-relative: {path}')
        if old and old.startswith('examples/MatCont/reference/'):
            hashes = [entry.get(k, '') for k in ('old_sha256', 'new_sha256')]
            if len(new) != 1 or not all(re.fullmatch(r'[0-9a-f]{64}', h) for h in hashes) or hashes[0] != hashes[1]:
                errors.append(f'invalid reference SHA-256 pair: {old}')
            else:
                for path in [old, *new]:
                    candidate = root / path
                    if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() != hashes[0]:
                        errors.append(f'reference SHA-256 mismatch: {path}')
        if completed:
            for path in new:
                if not (root / path).is_file():
                    errors.append(f'missing destination: {path}')
            if old and old not in new and (root / old).exists():
                errors.append(f'source still exists: {old}')
    # Multiple source files can converge only with an explicit merge/rewrite.
    owners = {}
    for entry in entries:
        for path in entry['new_paths']:
            owners.setdefault(path, []).append(entry)
    for path, rows in owners.items():
        if len(rows) > 1 and any(e['verdict'] not in {'merge', 'split', 'rewrite'} or not e.get('rationale') for e in rows):
            errors.append(f'merge without a rationale: {path}')
    aliases = data['cases']
    names = [e['legacy_id'] for e in aliases]
    canonical = [e['canonical_id'] for e in aliases]
    if len(names) != len(set(names)) or len(canonical) != len(set(canonical)):
        errors.append('alias collision')
    actual = {e['legacy_id']: e['canonical_id'] for e in aliases}
    for legacy, expected in CASE_ALIASES.items():
        if actual.get(legacy) != expected:
            errors.append(f'unmapped case ID or incorrect alias: {legacy} -> {expected}')
    if set(data['canonical_ids']) != CANONICAL_IDS:
        errors.append('unmapped case canonical IDs')
    registry_path = root / 'examples/validation/cases/cases.json'
    if not registry_path.exists():
        registry_path = root / 'examples/MatCont/cases.json'
    if registry_path.exists():
        for case in json.loads(registry_path.read_text())['cases']:
            # New runs have no pre-migration ID; the approved canonical
            # inventory already accounts for them. Legacy aliases still need
            # their original mapping, and unknown new runs remain errors.
            new_canonical = (
                case['id'] in data['canonical_ids']
                and case.get('canonical_id') == case['id']
                and not case.get('aliases')
            )
            if case['id'] not in actual and not new_canonical:
                errors.append(f'unmapped case ID: {case["id"]}')
    redirects = data['gallery_redirects']
    expected_urls = {'auto_examples/' + Path(s).stem + '.html' for s in SOURCES if s.startswith('examples/example_')}
    if {r['old_url'] for r in redirects} != expected_urls or len(redirects) != 23:
        errors.append('unmapped or duplicate gallery URL')
    for redirect in redirects:
        source = 'examples/' + Path(redirect['old_url']).stem + '.py'
        row = next((e for e in entries if e['old_path'] == source), None)
        targets = [] if row is None else [
            ('auto_tutorials/' if '/tutorials/' in p else 'auto_validation/') + Path(p).stem + '.html'
            for p in row['new_paths'] if '/plot_' in p
        ]
        if redirect['new_url'] not in targets:
            errors.append(f'incorrect gallery URL destination: {source}')
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--planned', action='store_true')
    mode.add_argument('--completed', action='store_true')
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP)
    args = parser.parse_args(argv)
    try:
        errors = audit(json.loads(args.map.read_text(encoding='utf-8')), completed=args.completed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [f'invalid migration map: {exc}']
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print('Migration map OK: 23 numbered examples, 3 other scripts, 5 Julia files, 18 cases, 22 reference files.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
