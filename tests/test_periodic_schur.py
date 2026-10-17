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


def _complex_dtype():
    return jnp.result_type(_real_dtype(), jnp.complex64)


def _rel_scalar(got, expected):
    return abs(complex(got) - complex(expected)) / max(abs(complex(expected)), 1.0)


def _local_multipliers(factors, right, left):
    """``d_i = l_(i+1)^H M_i r_i`` per phase and mode, ``(m, n)``."""
    left_next = jnp.roll(left, -1, axis=0)
    return jnp.einsum("ikj,ikl,ilj->ij", jnp.conj(left_next), factors.astype(right.dtype), right)


def _assert_periodic_eigenvectors(factors, right, left, values):
    """Right ``M_i r_i = d_i r_(i+1)`` and left ``M_i^H l_(i+1) = conj(d_i) l_i``
    residuals and the local dual pairing ``l_i^H r_i = 1`` at every phase."""
    eps = _eps()
    m, n = factors.shape[0], factors.shape[1]
    M = factors.astype(right.dtype)
    d = _local_multipliers(factors, right, left)
    assert bool(jnp.all(jnp.isfinite(d))) and bool(jnp.all(jnp.abs(d) > 0))
    for i in range(m):
        nxt = (i + 1) % m
        scale = float(jnp.linalg.norm(M[i], 2))
        for j in range(n):
            r, r_next = right[i, :, j], right[nxt, :, j]
            l, l_next = left[i, :, j], left[nxt, :, j]
            right_res = jnp.linalg.norm(M[i] @ r - d[i, j] * r_next)
            left_res = jnp.linalg.norm(jnp.conj(M[i]).T @ l_next - jnp.conj(d[i, j]) * l)
            assert float(right_res) < 500 * eps * scale * float(jnp.linalg.norm(r)), (i, j)
            assert float(left_res) < 500 * eps * scale * float(jnp.linalg.norm(l_next)), (i, j)
            assert abs(complex(jnp.vdot(l, r)) - 1.0) < 500 * eps, (i, j)
    products = jnp.prod(d, axis=0)
    for j in range(n):
        assert _rel_scalar(products[j], values[j]) < _err() * max(1.0, 1.0 / abs(complex(values[j])))


def _conjugated_triangular_factors(m=8):
    """``M_i = R_(i+1) U R_i.T`` with ``U = [[1, 2], [0, .5]]`` and rotations
    ``R_i`` by ``2*pi*i/m``: product ``U^m`` with nonorthogonal eigenvectors."""
    U = jnp.array([[1.0, 2.0], [0.0, 0.5]], dtype=_real_dtype())
    R = [_rotation2(jnp.asarray(2.0 * np.pi * i / m, dtype=_real_dtype())) for i in range(m + 1)]
    return jnp.stack([R[i + 1] @ U @ R[i].T for i in range(m)])


def test_periodic_eigenvectors_are_not_schur_columns():
    from jaxcont.stability.periodic_schur import periodic_eigenvectors, periodic_schur

    factors = _conjugated_triangular_factors()
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    right, left = periodic_eigenvectors(factors, T, Q, phase, logabs)
    product = jnp.eye(2, dtype=_complex_dtype())
    for M in factors:
        product = M.astype(product.dtype) @ product
    values = phase * jnp.exp(logabs)
    eps = _eps()
    scale = float(jnp.linalg.norm(product, 2))
    schur_residuals = []
    for j in range(2):
        r = right[0, :, j]
        residual = jnp.linalg.norm(product @ r - values[j] * r)
        assert float(residual) < 500 * eps * scale * float(jnp.linalg.norm(r)), j
        q = Q[0, :, j]
        schur_residuals.append(float(jnp.linalg.norm(product @ q - values[j] * q)) / scale)
    assert max(schur_residuals) > 0.1
    _assert_periodic_eigenvectors(factors, right, left, values)


def test_periodic_schur_jvp_log_multiplier():
    from jaxcont.stability.periodic_schur import periodic_spectrum

    def contracting(a):
        phase, logabs, _ = periodic_spectrum(_rotating_scaled_factors(a))
        k = jnp.argmin(logabs)
        return logabs[k], jnp.real(phase[k] * jnp.exp(logabs[k]))

    a = jnp.asarray(1.0, dtype=_real_dtype())
    (log_value, value), (dlog, dvalue) = jax.jvp(contracting, (a,), (jnp.ones_like(a),))
    expected_value = -32.0 * np.exp(-32.0)
    assert abs(float(log_value) + 32.0) < _err()
    assert _rel_scalar(dlog, -32.0) < 20 * _err()
    assert abs(float(dvalue) - expected_value) < 20 * _err() * abs(expected_value)
    grad_log = jax.grad(lambda a: contracting(a)[0])(a)
    grad_value = jax.grad(lambda a: contracting(a)[1])(a)
    assert _rel_scalar(grad_log, -32.0) < 20 * _err()
    assert abs(float(grad_value) - expected_value) < 20 * _err() * abs(expected_value)


_NONNORMAL_RATES = (-0.3 + 0.2j, 0.1 - 0.4j)


def _nonnormal_complex_factors(a, variant="spec"):
    """``M_i(a) = S_(i+1)(a) D(a) inv(S_i(a))``, ``i = 0..3``, ``S_4 = S_0``.

    ``variant="spec"`` is the spec 6.3 fixture, whose first ``S_i`` column does
    not depend on ``a``; ``"both_columns"`` also moves that column, so the
    Schur basis rotates with ``a`` and Schur columns give wrong derivatives."""
    cdt = _complex_dtype()
    a = jnp.asarray(a).astype(cdt)
    lower = 0.1j + (a * (0.1 - 0.03j) if variant == "both_columns" else 0.0)

    def S(i):
        return jnp.array(
            [[1.0, 0.4 + 0.1 * i + a * (0.2 + 0.05j)], [lower, 1.0 + 0.05 * i]], dtype=cdt
        )

    D = jnp.diag(jnp.exp(jnp.array(_NONNORMAL_RATES, dtype=cdt) * a))
    return jnp.stack([S((i + 1) % 4) @ D @ jnp.linalg.inv(S(i)) for i in range(4)])


@pytest.mark.parametrize("variant", ["spec", "both_columns"])
def test_periodic_schur_nonnormal_complex_derivatives(variant):
    from jaxcont.stability.periodic_schur import (
        periodic_eigenvectors,
        periodic_schur,
        periodic_spectrum,
    )

    a0 = 0.7
    rates = np.asarray(_NONNORMAL_RATES)
    exact_values = np.exp(4 * rates * a0)
    exact_phase = exact_values / np.abs(exact_values)
    d_logabs = 4 * rates.real
    d_phase = 4j * rates.imag * exact_phase
    d_values = 4 * rates * exact_values

    a = jnp.asarray(a0, dtype=_real_dtype())
    factors = _nonnormal_complex_factors(a, variant)
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    values = phase * jnp.exp(logabs)
    order = _match(exact_values, values)
    np.testing.assert_array_less(np.abs(np.asarray(values)[order] - exact_values), _err())
    right, left = periodic_eigenvectors(factors, T, Q, phase, logabs)
    _assert_periodic_eigenvectors(factors, right, left, values)

    def spectrum(a):
        phase, logabs, _ = periodic_spectrum(_nonnormal_complex_factors(a, variant))
        phase, logabs = phase[order], logabs[order]
        return phase, logabs, phase * jnp.exp(logabs)

    _, (t_phase, t_logabs, t_values) = jax.jvp(spectrum, (a,), (jnp.ones_like(a),))
    for k in range(2):
        assert _rel_scalar(t_logabs[k], d_logabs[k]) < 20 * _err(), k
        assert _rel_scalar(t_phase[k], d_phase[k]) < 20 * _err(), k
        assert _rel_scalar(t_values[k], d_values[k]) < 20 * _err(), k

    objectives = {
        "logabs": (lambda s, k: s[1][k], lambda k: d_logabs[k]),
        "phase_re": (lambda s, k: jnp.real(s[0][k]), lambda k: d_phase[k].real),
        "phase_im": (lambda s, k: jnp.imag(s[0][k]), lambda k: d_phase[k].imag),
        "value_re": (lambda s, k: jnp.real(s[2][k]), lambda k: d_values[k].real),
        "value_im": (lambda s, k: jnp.imag(s[2][k]), lambda k: d_values[k].imag),
    }
    for name, (objective, expected) in objectives.items():
        for k in range(2):
            g = jax.grad(lambda a: objective(spectrum(a), k))(a)
            scale = max(abs(d_values[k]), abs(d_phase[k]), abs(d_logabs[k]))
            assert abs(float(g) - expected(k)) < 20 * _err() * scale, (name, k)


def test_periodic_schur_mixed_status_batch():
    from jaxcont.stability.periodic_schur import periodic_spectrum

    def lanes(params):
        decay = jnp.exp(-params[0])
        diagonal = jnp.diag(jnp.stack([decay, jnp.ones_like(decay), 2.0 * jnp.ones_like(decay)]))
        converging = jnp.stack([diagonal] * 4)
        failing = params[1] * _general_factors()
        return jnp.stack([converging, failing])

    def batch(params):
        return jax.vmap(lambda f: periodic_spectrum(f, max_sweeps=0))(lanes(params))

    params = jnp.asarray([0.5, 1.0], dtype=_real_dtype())
    phase, logabs, converged = jax.jit(batch)(params)
    np.testing.assert_array_equal(np.asarray(converged), [True, False])
    expected = [-2.0, 0.0, 4 * np.log(2.0)]
    np.testing.assert_array_less(np.abs(np.sort(np.asarray(logabs[0])) - expected), _err())
    np.testing.assert_array_less(np.abs(np.asarray(phase[0]) - 1.0), _err())

    grad_good = jax.jit(jax.grad(lambda p: jnp.min(batch(p)[1][0])))(params)
    np.testing.assert_allclose(np.asarray(grad_good), [-4.0, 0.0], atol=20 * _err() * 4.0)
    grad_bad = jax.jit(jax.grad(lambda p: jnp.sum(batch(p)[1][1])))(params)
    assert bool(jnp.all(jnp.isfinite(grad_bad)))
    np.testing.assert_array_equal(np.asarray(grad_bad), 0.0)


def test_periodic_schur_repeated_modes_have_no_derivative():
    """Repeated multipliers get a NaN tangent instead of an advertised
    derivative; a simple mode of the same spectrum keeps its gradient."""
    from jaxcont.stability.periodic_schur import periodic_spectrum

    def factors(a):
        return jnp.stack([jnp.diag(jnp.stack([jnp.exp(a), jnp.exp(a), jnp.exp(2.0 * a)]))] * 4)

    a = jnp.asarray(0.3, dtype=_real_dtype())
    (_, logabs, converged), (_, t_logabs, _) = jax.jvp(
        lambda a: periodic_spectrum(factors(a)), (a,), (jnp.ones_like(a),)
    )
    assert bool(converged)
    simple = int(jnp.argmax(logabs))
    assert bool(jnp.all(jnp.isnan(jnp.delete(t_logabs, simple))))
    assert _rel_scalar(t_logabs[simple], 8.0) < 20 * _err()
    grad = jax.grad(lambda a: periodic_spectrum(factors(a))[1][simple])(a)
    assert _rel_scalar(grad, 8.0) < 20 * _err()


def test_periodic_schur_zero_underflow_overflow():
    from jaxcont.stability.periodic_schur import periodic_spectrum

    rdt = _real_dtype()
    zero = jnp.zeros((3, 3), dtype=rdt)
    diagonal_chain = jnp.stack(
        [jnp.diag(jnp.array([1.0, 2.0, 3.0], dtype=rdt)), zero, jnp.diag(jnp.array([2.0, 1.0, 0.5], dtype=rdt))]
    )
    general_chain = jnp.concatenate([_general_factors()[:2], zero[None]])
    for factors in (diagonal_chain, general_chain):
        phase, logabs, converged = periodic_spectrum(factors)
        assert bool(converged)
        np.testing.assert_array_equal(np.asarray(logabs), -np.inf)
        np.testing.assert_array_equal(np.asarray(phase), 0.0)
        np.testing.assert_array_equal(np.asarray(phase * jnp.exp(logabs)), 0.0)

    def extreme(a):
        block = jnp.diag(jnp.stack([jnp.exp(30.0 * a), jnp.exp(-30.0 * a)]))
        return jnp.stack([block] * 32)

    a = jnp.asarray(1.0, dtype=rdt)
    phase, logabs, converged = periodic_spectrum(extreme(a))
    assert bool(converged)
    assert not bool(jnp.any(jnp.isnan(logabs))) and not bool(jnp.any(jnp.isnan(phase)))
    order = np.argsort(np.asarray(logabs))
    np.testing.assert_array_less(np.abs(np.asarray(logabs)[order] - [-960.0, 960.0]), 960 * _err())
    np.testing.assert_array_less(np.abs(np.asarray(phase) - 1.0), _err())
    values = np.asarray(phase * jnp.exp(logabs))[order]
    assert values[0] == 0.0 and np.isinf(values[1])
    # An underflowed zero keeps a finite logabs; an exact zero has -inf.
    assert np.isfinite(np.asarray(logabs)).all()
    _, (_, t_logabs, _) = jax.jvp(lambda a: periodic_spectrum(extreme(a)), (a,), (jnp.ones_like(a),))
    np.testing.assert_allclose(np.asarray(t_logabs)[order], [-960.0, 960.0], rtol=20 * _err())


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("compiled", [False, True])
def test_periodic_schur_zero_factor_every_position(position, compiled):
    from jaxcont.stability.periodic_schur import periodic_schur

    factors = _general_factors().at[position].set(0)
    solve = jax.jit(periodic_schur) if compiled else periodic_schur
    T, Q, phase, logabs, converged = solve(factors)
    assert bool(converged)
    assert bool(jnp.all(phase == 0))
    assert bool(jnp.all(jnp.isneginf(logabs)))
    _assert_cyclic_reconstruction(factors, T, Q)
    assert bool(jnp.all(jnp.tril(T, -1) == 0))


@pytest.mark.parametrize("n,coupling", [(2, 1e25), (3, 1e12)])
def test_periodic_eigenvectors_highly_nonnormal_finite(n, coupling):
    from jaxcont.stability.periodic_schur import periodic_eigenvectors, periodic_schur

    factors = (
        jnp.diag(jnp.arange(1, n + 1, dtype=_real_dtype()))
        + coupling * jnp.eye(n, k=1, dtype=_real_dtype())
    )[None]
    T, Q, phase, logabs, converged = periodic_schur(factors)
    assert bool(converged)
    right, left = jax.jit(periodic_eigenvectors)(factors, T, Q, phase, logabs)
    assert bool(jnp.all(jnp.isfinite(right)))
    assert bool(jnp.all(jnp.isfinite(left)))
    np.testing.assert_allclose(jnp.sum(jnp.abs(right) ** 2, axis=1), 1, atol=500 * _eps())
    np.testing.assert_allclose(jnp.sum(jnp.conj(left) * right, axis=1), 1, atol=500 * _eps())
    values = phase * jnp.exp(logabs)
    for vectors, matrix, eigenvalues in [
        (right, factors, values),
        (left, factors.swapaxes(-1, -2), values.conj()),
    ]:
        scale = jnp.max(jnp.abs(vectors), axis=1, keepdims=True)
        scaled = vectors / scale
        residual = matrix @ scaled - scaled * eigenvalues[None, None, :]
        assert float(jnp.max(jnp.abs(residual)) / coupling) < 500 * _eps()
