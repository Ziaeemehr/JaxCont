"""Regressions for the binding loop-2 rulings."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import jaxcont as jc
from jaxcont.bifurcations.events import Fold, Hopf, detect_events
from jaxcont.core.collocation import Collocation


def test_bothside_preserves_hopf_in_first_backward_bracket():
    def rhs(u, p, args):
        x, y = u
        return jnp.array([(p + .005) * x - y, x + (p + .005) * y])
    result = jc.continuation(
        jc.bif_problem(rhs, u0=jnp.zeros(2), p0=0.), p_span=(-.1, .1),
        settings=jc.ContinuationPar(bothside=True, ds=.02, max_steps=20),
        events=[Hopf()],
    )
    assert len(result.events) == 1
    assert result.events[0].p == pytest.approx(-.005, abs=2e-5)


def test_adjacent_distinct_folds_survive_dedup():
    hits = detect_events(
        [Fold()], jnp.zeros(3), jnp.array([[-1.], [0.], [1.]]),
        jnp.array([[1., -1.], [1., 1.], [1., -1.]]), None,
        lambda u, p: u**3 - u - p, ds=1.,
    )
    good = sorted(h.p for h in hits if h.status == 'validated')
    assert good == pytest.approx([-.38490018, .38490018], abs=2e-5)


@pytest.mark.parametrize('algorithm', [jc.Natural(), jc.PseudoArclength()])
@pytest.mark.parametrize('span', [(-.1, 0.), (0., .1)])
def test_bothside_endpoint_skips_empty_leg(algorithm, span):
    result = jc.continuation(
        jc.bif_problem(lambda u, p, args: u - p, u0=jnp.zeros(1), p0=0.),
        algorithm, p_span=span,
        settings=jc.ContinuationPar(bothside=True, ds=.02, max_steps=20),
    )
    assert result.status == 'reached_end'
    assert np.isfinite(result.branch.tangents).all()
    assert np.count_nonzero(np.asarray(result.branch.params) == 0.) == 1
    assert np.all(np.diff(result.branch.params) > 0)


@pytest.mark.parametrize('bothside', [False, True])
@pytest.mark.parametrize('algorithm', [jc.Natural(), jc.PseudoArclength()])
def test_zero_total_span_rejected(algorithm, bothside):
    with pytest.raises(ValueError, match='span'):
        jc.continuation(
            jc.bif_problem(lambda u, p, args: u - p, u0=jnp.zeros(1), p0=0.),
            algorithm, p_span=(0., 0.), settings=jc.ContinuationPar(bothside=bothside),
        )


def test_event_setup_avoids_scalar_device_conversions(monkeypatch):
    params = jnp.linspace(0., 1., 100)
    array_type = type(params)
    original = array_type.__float__
    calls = []
    def counted(value):
        calls.append(value.shape)
        return original(value)
    monkeypatch.setattr(array_type, '__float__', counted)
    class PositiveEvent:
        def test_function(self, point):
            return point.p + 1
        def refine(self, *args, **kwargs):
            pytest.fail('no crossing expected')
    assert detect_events([PositiveEvent()], params, jnp.zeros((100, 1)), None,
                         None, lambda u, p: u, ds=.01) == []
    assert calls == []


def test_zero_interval_mesh_rejected():
    with pytest.raises(ValueError, match='ntst'):
        Collocation(ntst=0, ncol=4)


@pytest.mark.slow
def test_example14_completes_in_scratch(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'examples'
    scratch = tmp_path / 'examples'
    shutil.copytree(source, scratch)
    run = subprocess.run(
        [sys.executable, str(scratch / 'example_14_prc_shooting_validation.py')],
        cwd=scratch, env={**os.environ, 'MPLBACKEND': 'Agg'},
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
