"""
Tests for jaxcont.stability.periodic_schur: periodic Hessenberg-triangular
reduction and periodic Schur decomposition of ordered interval factors, see
docs/superpowers/specs/2026-10-17-periodic-schur-floquet-design.md.

Machine-epsilon checks (``500*eps`` backward error) are the spec's explicit
algorithmic exceptions to the ``default_tol`` policy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.utils.precision import default_tol


def _real_dtype():
    return jnp.asarray(0.0).dtype


def _eps():
    return float(jnp.finfo(_real_dtype()).eps)


def _rotation13(angle):
    """3x3 rotation acting in the first/third coordinate plane."""
    c, s = jnp.cos(angle), jnp.sin(angle)
    zero, one = jnp.zeros_like(c), jnp.ones_like(c)
    return jnp.array([[c, zero, -s], [zero, one, zero], [s, zero, c]])


def _rotating_scaled_factors(a=2.0, m=32):
    """``M_i = Q_(i+1) diag(exp(-a), 1, exp(2)) Q_i.T`` with ``Q_i`` rotations
    by ``2*pi*i/m``; the cyclic product is ``diag(exp(-m*a), 1, exp(2*m))``."""
    a = jnp.asarray(a, dtype=_real_dtype())
    diag = jnp.diag(jnp.stack([jnp.exp(-a), jnp.ones_like(a), jnp.exp(2.0 * jnp.ones_like(a))]))
    rotations = [_rotation13(2.0 * jnp.pi * i / m) for i in range(m + 1)]
    return jnp.stack([rotations[i + 1] @ diag @ rotations[i].T for i in range(m)])


def _general_factors():
    """Nonnormal, nontriangular ``(4, 3, 3)`` fixture."""
    raw = jnp.sin(jnp.arange(36, dtype=_real_dtype()).reshape(4, 3, 3) + 1.0)
    return raw + 2.0 * jnp.eye(3, dtype=_real_dtype())


def _rel_fro(x, ref):
    return float(jnp.linalg.norm(x) / jnp.maximum(jnp.linalg.norm(ref), jnp.finfo(_real_dtype()).tiny))


def _assert_cyclic_reconstruction(factors, H, Q):
    eps = _eps()
    m, n = factors.shape[0], factors.shape[1]
    eye = jnp.eye(n, dtype=Q.dtype)
    for i in range(m):
        M = factors[i].astype(Q.dtype)
        residual = M @ Q[i] - Q[(i + 1) % m] @ H[i]
        assert _rel_fro(residual, M) < 500 * eps, (i, _rel_fro(residual, M))
        assert float(jnp.linalg.norm(Q[i].conj().T @ Q[i] - eye)) < 500 * eps


def _assert_periodic_hessenberg_structure(H):
    eps = _eps()
    m = H.shape[0]
    for i in range(m):
        forbidden = jnp.tril(H[i], -2 if i == m - 1 else -1)
        assert float(jnp.max(jnp.abs(forbidden))) < 500 * eps * float(jnp.linalg.norm(H[i])), i


_HESSENBERG_FIXTURES = {
    "general": _general_factors,
    "rotating_scaled": _rotating_scaled_factors,
}


@pytest.mark.parametrize("fixture", sorted(_HESSENBERG_FIXTURES))
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_periodic_hessenberg_factor_reconstruction(fixture, mode):
    from jaxcont.stability.periodic_schur import periodic_hessenberg

    factors = _HESSENBERG_FIXTURES[fixture]()
    if mode == "eager":
        H, Q = periodic_hessenberg(factors)
    elif mode == "jit":
        H, Q = jax.jit(periodic_hessenberg)(factors)
    else:
        batch = jnp.stack([factors, 0.5 * factors[::-1]])
        Hb, Qb = jax.vmap(periodic_hessenberg)(batch)
        _assert_cyclic_reconstruction(batch[1], Hb[1], Qb[1])
        _assert_periodic_hessenberg_structure(Hb[1])
        H, Q = Hb[0], Qb[0]
    complex_dtype = jnp.result_type(factors.dtype, jnp.complex64)
    assert H.dtype == complex_dtype and Q.dtype == complex_dtype
    assert H.shape == factors.shape and Q.shape == factors.shape
    _assert_cyclic_reconstruction(factors, H, Q)
    _assert_periodic_hessenberg_structure(H)


def test_periodic_hessenberg_single_factor():
    from jaxcont.stability.periodic_schur import periodic_hessenberg

    factors = _general_factors()[:1]
    H, Q = periodic_hessenberg(factors)
    _assert_cyclic_reconstruction(factors, H, Q)
    _assert_periodic_hessenberg_structure(H)


def _err():
    """Exact-factor log/phase acceptance tolerance of spec 6.3."""
    return 200 * default_tol(_real_dtype())


def _rotation2(angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def _assert_schur_form(factors, T, Q):
    eps = _eps()
    _assert_cyclic_reconstruction(factors, T, Q)
    for i in range(T.shape[0]):
        forbidden = jnp.tril(T[i], -1)
        assert float(jnp.max(jnp.abs(forbidden))) <= 500 * eps * float(jnp.linalg.norm(T[i])), i


def _match(expected, actual):
    """Permutation matching ``actual`` to ``expected`` by nearest complex value."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    order = []
    free = list(range(actual.size))
    for value in expected:
        k = min(free, key=lambda j: abs(actual[j] - value))
        order.append(k)
        free.remove(k)
    return np.asarray(order)


def test_periodic_schur_rotating_scaled_factors():
    from jaxcont.stability.periodic_schur import periodic_schur

    factors = _rotating_scaled_factors()
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    _assert_schur_form(factors, T, Q)
    expected = np.array([-64.0, 0.0, 64.0])
    order = _match(expected, logabs)
    np.testing.assert_array_less(np.abs(np.asarray(logabs)[order] - expected), _err())
    np.testing.assert_array_less(np.abs(np.asarray(phase) - 1.0), _err())
    values = phase * jnp.exp(logabs)
    smallest = values[order[0]]
    assert float(jnp.abs(smallest)) > 0.0
    assert abs(float(jnp.log(jnp.abs(smallest))) + 64.0) < _err()


def test_periodic_schur_complex_equal_modulus():
    from jaxcont.stability.periodic_schur import periodic_schur

    block = 0.9 * _rotation2(jnp.asarray(0.2, dtype=_real_dtype()))
    factors = jnp.stack([block] * 16)
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    _assert_schur_form(factors, T, Q)
    np.testing.assert_array_less(np.abs(np.asarray(logabs) - 16 * np.log(0.9)), _err())
    expected = np.exp(np.array([3.2j, -3.2j]))
    order = _match(expected, phase)
    np.testing.assert_array_less(np.abs(np.asarray(phase)[order] - expected), _err())


def test_periodic_schur_zero_and_budget_status():
    from jaxcont.stability.periodic_schur import periodic_schur

    factors = jnp.diag(jnp.array([0.0, 1.0, 2.0], dtype=_real_dtype()))[None]
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    np.testing.assert_array_equal(np.asarray(logabs)[0], -np.inf)
    np.testing.assert_array_less(np.abs(np.asarray(logabs)[1:] - [0.0, np.log(2.0)]), _err())
    np.testing.assert_array_equal(np.asarray(phase), [0.0, 1.0, 1.0])

    general = _general_factors()
    *_, converged = periodic_schur(general, max_sweeps=0)
    assert not bool(converged)
    T, Q, _, _, converged = periodic_schur(general)
    assert bool(converged)
    _assert_schur_form(general, T, Q)


def test_periodic_schur_general_matches_product_eigenvalues():
    """Host reference: eigenvalues of the explicitly formed product of a
    well-conditioned nonnormal fixture."""
    from jaxcont.stability.periodic_schur import periodic_schur

    for factors in (_general_factors(), _general_factors()[:1], _general_factors()[:2, :2, :2]):
        product = np.eye(factors.shape[1])
        for M in np.asarray(factors, dtype=np.float64):
            product = M @ product
        expected = np.linalg.eigvals(product)
        _, _, phase, logabs, converged = periodic_schur(factors)
        assert bool(converged)
        values = np.asarray(phase * jnp.exp(logabs))
        order = _match(expected, values)
        np.testing.assert_array_less(
            np.abs(values[order] - expected), _err() * np.max(np.abs(expected))
        )


def test_periodic_schur_jit_vmap():
    from jaxcont.stability.periodic_schur import periodic_schur

    batch = jnp.stack([_general_factors(), _general_factors()[::-1]])
    T, Q, phase, logabs, converged = jax.jit(jax.vmap(periodic_schur))(batch)
    assert bool(jnp.all(converged))
    for lane in range(2):
        _assert_schur_form(batch[lane], T[lane], Q[lane])
        _, _, phase1, logabs1, _ = periodic_schur(batch[lane])
        order = _match(phase1 * jnp.exp(logabs1), phase[lane] * jnp.exp(logabs[lane]))
        np.testing.assert_allclose(np.asarray(logabs[lane])[order], logabs1, atol=_err())
