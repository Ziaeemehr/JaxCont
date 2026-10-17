"""Analytic regressions for the loop-5 audit."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.stability.periodic_schur import periodic_schur, periodic_spectrum

schur_module = importlib.import_module('jaxcont.stability.periodic_schur')


def _shear():
    return 1e-15 if jax.config.x64_enabled else 1e-7


def _reciprocal_shear(t, diagonal, roll):
    """Factors whose cyclic product is ``[[1, 1/s], [s*exp(t), exp(t) + diagonal]]``.

    The small coupling ``s`` in one factor meets ``1/s`` in the other, so it
    moves the multipliers by O(1) although it is tiny relative to its factor.
    """
    s = _shear()
    factors = jnp.stack([jnp.array([[1., 1 / s], [0., 1.]]),
                         jnp.array([[1., 0.], [s * jnp.exp(t), diagonal]])])
    return jnp.roll(factors, roll, axis=0)


@pytest.mark.parametrize('roll', [0, 1])
@pytest.mark.parametrize('diagonal', [1., 2.])
def test_reciprocal_shears_are_accurate_or_rejected(diagonal, roll):
    zero = jnp.array(0.)
    # The unresolved coupling is not accepted as a deflation.
    assert not bool(periodic_schur(_reciprocal_shear(zero, diagonal, roll), max_sweeps=0)[-1])

    def spectrum(t):
        return periodic_spectrum(_reciprocal_shear(t, diagonal, roll))

    def trace(t):
        phase, logs, _ = spectrum(t)
        return jnp.sum(jnp.real(phase * jnp.exp(logs)))

    phase, logs, ok = spectrum(zero)
    derivative = jax.jit(jax.grad(trace))(zero)
    if not bool(ok):
        np.testing.assert_array_equal(derivative, 0.)
        return
    trace0, det0 = 2. + diagonal, diagonal
    disc = np.sqrt(trace0**2 - 4 * det0)
    expected = [(trace0 - disc) / 2, (trace0 + disc) / 2]
    np.testing.assert_allclose(np.sort(np.real(phase * jnp.exp(logs))), expected, rtol=2e-3)
    np.testing.assert_allclose(derivative, 1., rtol=2e-3)
    np.testing.assert_allclose(jax.jacfwd(trace)(zero), 1., rtol=2e-3)


def test_premature_deflation_publishes_no_sensitivity(monkeypatch):
    """Modes rebuilt from a falsely deflated Schur form do not reproduce the
    multipliers through the true factors, so no derivative is published."""
    def local_only(H, eps):
        H_last = H[-1] if H.ndim == 3 else H
        k = jnp.arange(H_last.shape[0] - 1)
        sub = jnp.abs(H_last[k + 1, k])
        diagonal = jnp.abs(H_last[k, k]) + jnp.abs(H_last[k + 1, k + 1])
        return H_last.at[k + 1, k].set(jnp.where(sub <= 100 * eps * diagonal, 0, H_last[k + 1, k]))

    monkeypatch.setattr(schur_module, '_deflate', local_only)
    zero = jnp.array(0.)
    phase, logs, converged = periodic_spectrum(_reciprocal_shear(zero, 2., 0))
    assert bool(converged)
    np.testing.assert_allclose(np.sort(np.real(phase * jnp.exp(logs))), [1., 2.], rtol=1e-3)
    tangent = jax.jvp(lambda t: periodic_spectrum(_reciprocal_shear(t, 2., 0))[1],
                      (zero,), (jnp.ones_like(zero),))[1]
    assert bool(jnp.all(jnp.isnan(tangent)))


def _affine_objective(algorithm, guess, loop):
    def objective(a):
        problem = jc.BifProblem(lambda u, p, a: u - a - p, jnp.array([guess]), 0., args=a)
        result = jc.continuation(problem, algorithm, p_span=(0., .3),
            settings=jc.ContinuationPar(loop=loop, ds=.1, ds_max=.1,
                max_steps=3, adaptive=False, compute_stability=False))
        return result.branch.states[:, 0].sum()
    return objective


@pytest.mark.parametrize('algorithm', [jc.Natural(), jc.PseudoArclength()])
@pytest.mark.parametrize('guess', [0., .01])
def test_scan_affine_translation_sensitivity(algorithm, guess):
    objective = _affine_objective(algorithm, guess, 'scan')
    a = jnp.array(0.)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(a), 4., rtol=2e-4)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(objective))(a), 4., rtol=2e-4)


@pytest.mark.parametrize('algorithm', [jc.Natural(), jc.PseudoArclength()])
@pytest.mark.parametrize('guess', [0., .01])
def test_while_affine_translation_sensitivity(algorithm, guess):
    objective = _affine_objective(algorithm, guess, 'while')
    np.testing.assert_allclose(jax.jacfwd(objective)(jnp.array(0.)), 4., rtol=2e-4)


@pytest.mark.parametrize('algorithm', [jc.Natural(), jc.PseudoArclength()])
def test_exact_seed_sensitivity_through_closure_and_vmap(algorithm):
    """A right-hand side closing over the traced parameter keeps its implicit
    derivative, also under ``vmap``."""
    def objective(a):
        problem = jc.BifProblem(lambda u, p, _: u - a - p, jnp.array([0.]), 0.)
        result = jc.continuation(problem, algorithm, p_span=(0., .3),
            settings=jc.ContinuationPar(loop='scan', ds=.1, ds_max=.1,
                max_steps=3, adaptive=False, compute_stability=False))
        return result.branch.states[:, 0].sum()

    np.testing.assert_allclose(jax.grad(objective)(jnp.array(0.)), 4., rtol=2e-4)
    batch = jax.jit(jax.vmap(jax.grad(objective)))(jnp.array([0., 0.]))
    np.testing.assert_allclose(batch, [4., 4.], rtol=2e-4)


@pytest.mark.parametrize('loop', ['scan', 'while'])
def test_natural_exact_seed_nonlinear_sensitivity(loop):
    """``u^3 + u = a + p``: ``du/da = 1 / (3u^2 + 1)`` at every stored state."""
    def states(a):
        problem = jc.BifProblem(lambda u, p, a: u**3 + u - a - p, jnp.array([0.]), 0., args=a)
        result = jc.continuation(problem, jc.Natural(), p_span=(0., .3),
            settings=jc.ContinuationPar(loop=loop, ds=.1, ds_max=.1,
                max_steps=3, adaptive=False, compute_stability=False))
        return result.branch.states[:, 0]

    a = jnp.array(0.)
    u = states(a)
    expected = np.sum(1 / (3 * np.asarray(u)**2 + 1))
    objective = lambda a: jnp.sum(states(a))
    np.testing.assert_allclose(jax.jacfwd(objective)(a), expected, rtol=2e-4)
    if loop == 'scan':
        np.testing.assert_allclose(jax.jit(jax.grad(objective))(a), expected, rtol=2e-4)


def test_exact_singular_scan_seed_keeps_acceptance():
    from jaxcont.core.scan_continuation import _natural_correct

    def correct(u):
        return _natural_correct(lambda u, p, a: u * u, u, jnp.array(0.), None,
                                jnp.array(.001), 4, loop='scan')

    value, converged, iterations = jax.jit(correct)(jnp.array([0.]))
    assert bool(converged)
    assert int(iterations) == 0
    np.testing.assert_array_equal(value, [0.])
    # No implicit derivative exists; the seed's own tangent passes through.
    np.testing.assert_array_equal(jax.jacfwd(lambda u: correct(u)[0])(jnp.array([0.])), [[1.]])
