"""Regression coverage for the loop-3 review rulings."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.bifurcations.curve_layout import CurveLayout
from jaxcont.bifurcations.events import BranchPoint, EventHit, detect_events, _test_values, _refined_points_match
from jaxcont.core.continuation import ContinuationSolution
from tests.test_period_doubling_neimark_sacker import _sweep, BETA_PD, BETA_NS


def test_distinct_cusps_keep_all_parameter_coordinates():
    def f(u, p, args):
        return jnp.array([u[0]**3 + p[1]*u[0] + p[0]*(1-p[0])])
    us = np.array([-.5, .1, -.5])
    fixed = np.array([(1-np.sqrt(1-8*us[0]**3))/2,
                      (1-np.sqrt(1-8*us[1]**3))/2,
                      (1+np.sqrt(1-8*us[2]**3))/2])
    states = jnp.asarray(np.column_stack([us, fixed, np.ones(3)]))
    hits = detect_events([jc.Cusp(f)], jnp.asarray(-3*us**2), states,
                         None, None, lambda u, p: u, ds=1.,
                         curve_layout=CurveLayout("fold", "extended", 1, 1))
    assert len(hits) == 2
    np.testing.assert_allclose(sorted(float(h.info['p'][0]) for h in hits), [0., 1.], atol=1e-4)


@pytest.mark.parametrize('key', ['p', 'p_fixed'])
def test_dedup_includes_parameter_metadata(key):
    a = EventHit('cusp', 0., jnp.zeros(1), (0, 1), info={key: jnp.asarray(0.)})
    b = EventHit('cusp', 0., jnp.zeros(1), (1, 2), info={key: jnp.asarray(1.)})
    assert not _refined_points_match(a, b)
    assert _refined_points_match(a, a)


def test_bp_refinement_preserves_both_candidate_modes():
    def f(u, p, args):
        return jnp.array([(p+.03)*u[0]-u[0]**3, (p-.04)*u[1]-u[1]**3])
    result = jc.continuation(jc.bif_problem(f, jnp.zeros(2), -.1), p_span=(-.1, .11),
        settings=jc.ContinuationPar(ds=.2, ds_max=.2, adaptive=False, max_steps=2), events=[jc.BP()])
    assert result.status == 'reached_end'
    np.testing.assert_allclose(sorted(h.p for h in result.events), [-.03, .04], atol=1e-4)


@pytest.mark.parametrize('beta,event', [(BETA_PD, jc.PeriodDoubling), (BETA_NS, jc.NeimarkSacker)])
def test_periodic_event_save_load_roundtrip(tmp_path, beta, event):
    result = _sweep(beta, event, (-.05, .05))
    assert len(result.events) == 1
    filename = tmp_path / 'branch.npz'
    result._solution.save(str(filename))
    loaded = ContinuationSolution.load(str(filename))
    np.testing.assert_array_equal(loaded.states, result._solution.states)
    np.testing.assert_array_equal(loaded.parameters, result._solution.parameters)
    original = result._solution.bifurcations[0]
    restored = loaded.bifurcations[0]
    assert restored['type'] == original['type']
    # Metadata complex values use the documented [real, imaginary] encoding.
    multiplier = original['info']['multiplier'] if 'info' in original else original['multiplier']
    restored_multiplier = restored['info']['multiplier'] if 'info' in restored else restored['multiplier']
    assert complex(*restored_multiplier) == multiplier


def test_batched_custom_test_keeps_trivial_index():
    class CustomPD:
        kind = 'custom_pd'
        def test_function(self, pt):
            return jnp.sum(jnp.where(jnp.arange(pt.eigenvalues.size) == pt.trivial_index,
                                    0., pt.eigenvalues.real + 1.))
        def refine(self, left, right, index, rhs, **kwargs):
            return EventHit(self.kind, float((left.p+right.p)/2), (left.u+right.u)/2, index)
    event = CustomPD()
    params, states = jnp.array([0., 1.]), jnp.array([[0.], [1.]])
    eigs = jnp.array([[1., -1.2], [1., -.8]])
    points = [BranchPoint(params[i], states[i], eigenvalues=eigs[i], trivial_index=jnp.array(0)) for i in range(2)]
    np.testing.assert_allclose(_test_values(event, points), [-.2, .2], atol=1e-6)
    assert len(detect_events([event], params, states, None, eigs, lambda u, p: u-p,
                            ds=1., trivial_index=jnp.array([0, 0]))) == 1


def test_changelog_signature_and_dedup_coverage():
    text = (Path(__file__).resolve().parents[1] / 'CHANGELOG.md').read_text().split('## [Unreleased]')[0]
    for term in ['eigenvalue=', 'interval_propagators', 'monodromy_matrix']:
        assert term in text
    assert '2 * max(ds_i, ds_j)' not in text
