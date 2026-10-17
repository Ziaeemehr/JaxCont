"""Regression gates for the fourth remediation audit."""
from pathlib import Path
import subprocess
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jaxcont as jc
from jaxcont.bifurcations.curve_layout import CurveLayout, decode_curve_state
from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
from jaxcont.stability.periodic_schur import periodic_spectrum

ROOT = Path(__file__).resolve().parents[1]


def _rotation(u, p, a):
    x, y = u - jnp.array([a, -.3])
    r2 = x*x + y*y
    return jnp.array([(p[0]-r2)*x-p[1]*y, p[1]*x+(p[0]-r2)*y])


def _seed(a):
    return jc.hopf_curve_problem(_rotation, jnp.array([.5, -.3]),
                                 jnp.array([0., 1.]), args=a, formulation='bt')


def test_isotropic_bt_seed_and_scan_gradient():
    np.testing.assert_allclose(jax.grad(lambda a: _seed(a).u0.sum())(.5), 1., atol=2e-4)
    def branch(a):
        result = jc.continuation(_seed(a), jc.Natural(), p_span=(1., 1.1),
            settings=jc.ContinuationPar(loop='scan', ds=.05, ds_max=.05, max_steps=2,
                                       compute_stability=False, adaptive=False))
        return (result.branch.states[:, 0]*result.branch.valid).sum()
    np.testing.assert_allclose(jax.jit(jax.grad(branch))(.5), 3., atol=2e-4)


def _fold(u, p, a):
    x, y, z = u
    return jnp.array([p[0]+p[1]*x+x**3, -a*y, -a*z])


def test_repeated_stable_fold_decoder_gradient():
    layout = CurveLayout('fold', 'minimal', 3, 1)
    def decoded(t):
        return decode_curve_state(jnp.array([t, 0., 0., 2*t**3]), -3*t*t,
                                  layout, _fold, 1.)[2].v[0]
    np.testing.assert_allclose(jax.jit(jax.grad(decoded))(.2), 0., atol=2e-4)


def test_repeated_stable_cusp_scan_gradient():
    layout = CurveLayout('fold', 'minimal', 3, 1)
    xs = jnp.array([-.2, .2])
    states = jnp.stack([xs, jnp.zeros(2), jnp.zeros(2), 2*xs**3], axis=1)
    data = EventScanInput(-3*xs**2, states, None, None, None, jnp.ones(2, bool), layout)
    def coefficient(a):
        rhs = lambda u, p, args: _fold(u, p, a)
        return scan_event_candidates((jc.Cusp(rhs),), data, None)[0].values[1, 0]
    np.testing.assert_allclose(coefficient(1.), .6, atol=2e-4)
    np.testing.assert_allclose(jax.jit(jax.grad(coefficient))(1.), 0., atol=2e-4)


def test_cancelling_local_growth_derivatives():
    strength = 400. if jax.config.x64_enabled else 80.
    def logs(t):
        large, small = jnp.exp(strength), jnp.exp(-strength)
        factors = jnp.stack([jnp.array([[large*jnp.exp(t), .1], [0., small]]),
                             jnp.diag(jnp.array([2*small, .5*large]))])
        return periodic_spectrum(factors)[1]
    np.testing.assert_allclose(jax.jacfwd(logs)(0.), [1., 0.], atol=2e-4)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda t: logs(t).sum()))(0.), 1., atol=2e-4)


@pytest.mark.slow
@pytest.mark.parametrize('number', [10, 11, 15])
def test_shipped_gallery_example(number, tmp_path):
    """Run the full gallery script with its documented float64 setting."""
    (tmp_path/'images').mkdir()
    script = next((ROOT/'examples').glob(f'example_{number:02d}_*.py'))
    env = dict(os.environ, MPLBACKEND='Agg', JAX_PLATFORMS='cpu',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    runner = ("import runpy, sys; ns = runpy.run_path(sys.argv[1]); "
              "r = ns.get('result'); "
              "assert r is None or r.status == 'reached_end', getattr(r, 'status', None)")
    result = subprocess.run([sys.executable, '-c', runner, str(script)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
