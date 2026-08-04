"""
Differentiable Hopf-point solver + first Lyapunov coefficient (Kuznetsov's
normal-form formula), via the extended system + implicit diff.

A Hopf point of ``f(u, p; args) = 0`` is the solution of the extended system

    G1: f(u, p)                  = 0   (n eqs)  equilibrium
    G2: J(u,p)*q1 + omega*q2     = 0   (n eqs)  Re[(J - i*omega)(q1+i*q2)] = 0
    G3: J(u,p)*q2 - omega*q1     = 0   (n eqs)  Im[(J - i*omega)(q1+i*q2)] = 0
    G4: q1.q1 + q2.q2 - 1        = 0   (1 eq)   unit norm
    G5: q1_seed.q2 - q2_seed.q1  = 0   (1 eq)   phase anchored to the seed

in the unknowns ``x = (u, p, q1, q2, omega)`` (dimension ``3n+2``), where
``q1 + i*q2`` is the right eigenvector for the critical eigenvalue
``i*omega``. The extended-system Newton solve and its implicit-function-
theorem gradient live in :func:`jaxcont.solvers.implicit.differentiable_root`,
shared with ``fold_solve.py``; this module only builds the Hopf-specific
``G`` and initial guess.

G5's phase constraint uses the FIXED seed eigenvector (``q1_seed``,
``q2_seed``), not the live ``(q1, q2)``: the natural-looking self-
referential alternative ``G5 = q1.q2 = 0`` has a Jacobian (with respect to
``(q1, q2)``) that can become exactly singular depending on the
eigenvector's own components -- found and fixed during design, see
``docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md``. Anchoring
to a fixed reference makes G5's gradient the constant vector ``(q2_seed,
-q1_seed)``, which cannot vanish. Inside ``G``, the seed is recomputed
under ``jax.lax.stop_gradient`` -- not bare ``theta`` -- because the seed
comes from ``jnp.linalg.eig``, and JAX has no gradient rule for
non-symmetric eigenvectors; the seed is an arbitrary fixed reference, not a
differentiated quantity, so cutting its gradient path is correct, not a
workaround.

Public entry points:
- :func:`hopf_point`     -> (u*, p*, q1*, q2*, omega0*), differentiable in ``args``
- :func:`hopf_parameter` -> p*,                          differentiable in ``args``
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, jvp, lax

from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _pack(u, p, q1, q2, omega):
    return jnp.concatenate([u, jnp.reshape(p, (1,)), q1, q2, jnp.reshape(omega, (1,))])


def _unpack(x, n):
    return x[:n], x[n], x[n + 1:2 * n + 1], x[2 * n + 1:3 * n + 1], x[3 * n + 1]


def _seed(f, u, p, args, n):
    """Undifferentiated Newton seed: eigenvector of J(u,p) with smallest
    |Re(eigenvalue)|, matching bifurcations.events.Hopf's own selection
    rule. jnp.linalg.eig (not just eigvals) is needed for the eigenvector;
    like fold_solve.py's SVD-based null-vector seed, this is never itself
    differentiated."""
    jac = jacfwd(f, argnums=0)(u, p, args)
    evals, evecs = jnp.linalg.eig(jac)
    idx = jnp.argmin(jnp.abs(jnp.real(evals)))
    vec = evecs[:, idx]
    vec = vec / jnp.linalg.norm(vec)
    q1 = jnp.real(vec)
    q2 = jnp.imag(vec)
    omega = jnp.abs(jnp.imag(evals[idx]))
    return q1, q2, omega


def _extended_residual(x, f, args, n, u_guess, p_guess):
    """G(x, args) for the Hopf extended system."""
    u, p, q1, q2, omega = _unpack(x, n)
    q1_seed, q2_seed, _ = _seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac = jacfwd(f, argnums=0)(u, p, args)
    g1 = f(u, p, args)
    g2 = jac @ q1 + omega * q2
    g3 = jac @ q2 - omega * q1
    g4 = jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0
    g5 = jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1)
    return jnp.concatenate([g1, g2, g3, jnp.reshape(g4, (1,)), jnp.reshape(g5, (1,))])


def hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Locate a Hopf point near ``(u_guess, p_guess)``, differentiable in ``args``.

    Returns ``(u*, p*, q1*, q2*, omega0*)``: the equilibrium, parameter,
    real and imaginary parts of the (unit) critical eigenvector, and the
    critical frequency.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _extended_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        q1_0, q2_0, omega_0 = _seed(f, u_guess, p_guess, theta, n)
        return _pack(u_guess, p_guess, q1_0, q2_0, omega_0)

    x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1, q2, omega = _unpack(x_star, n)
    return u, p, q1, q2, omega


def hopf_parameter(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: float | Array,
    args: PyTree = None,
    *,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> Array:
    """
    Parameter value ``p*`` at the Hopf point -- a scalar, differentiable in ``args``.
    """
    _, p, _, _, _ = hopf_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p


def lyapunov_coefficient(
    f: Callable[[Array, Array, PyTree], Array],
    u: Array,
    p: Array,
    q1: Array,
    q2: Array,
    omega0: Array,
    args: PyTree = None,
) -> Array:
    """
    First Lyapunov coefficient l1 at a Hopf point (u, p) with critical
    eigenvector q1 + i*q2 and frequency omega0 -- Kuznetsov's formula
    (Elements of Applied Bifurcation Theory, 3rd ed., eq. 3.19/10.16).
    l1 < 0 is supercritical (stable branching limit cycle), l1 > 0 is
    subcritical. Pure algebra (directional derivatives via jax.jvp plus two
    linear solves) -- no Newton iteration, differentiable wherever its
    inputs are, so composes with hopf_point's implicit-diff gradients via
    ordinary reverse-mode chain-rule composition; never calls
    jnp.linalg.eig itself.

    The left eigenvector p_left (Aᵀ p_left = -i*omega0*p_left, normalized
    so conj(p_left)@q == 1) is solved directly in complex arithmetic on the
    n x n matrix (Aᵀ + i*omega0*I) -- NOT via a real 2n x 2n block
    representation, which structurally always has repeated singular-value
    pairs and makes jnp.linalg.svd's gradient nan (found during design,
    see docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md).

    Verified: the standard supercritical textbook example (f = (-y + x(mu -
    x^2 - y^2), x + y(mu - x^2 - y^2)), exact l1 = -1 at the origin) and an
    independent BifurcationKit.jl v0.5.2 cross-check on a less-symmetric
    example -- see examples/BifurcationKit/04_hopf_normal_form.jl.
    """
    n = u.shape[0]
    complex_dtype = jnp.complex128 if u.dtype == jnp.float64 else jnp.complex64
    u_c = u.astype(complex_dtype)
    q = (q1 + 1j * q2).astype(complex_dtype)
    qbar = jnp.conj(q)

    A = jacfwd(lambda uu: f(uu, p, args))(u)
    Ac = A.astype(complex_dtype)
    eye_n = jnp.eye(n, dtype=complex_dtype)

    m_complex = Ac.T + 1j * omega0 * eye_n
    _, _, vh = jnp.linalg.svd(m_complex)
    p_left = jnp.conj(vh[-1, :])
    p_left = p_left / jnp.conj(jnp.vdot(p_left, q))

    def f_c(uu):
        return f(uu, p, args).astype(complex_dtype)

    def d1(uu, y):
        return jvp(f_c, (uu,), (y,))[1]

    def d2(uu, y, z):
        return jvp(lambda uu_: d1(uu_, y), (uu,), (z,))[1]

    def d3(uu, y, z, w):
        return jvp(lambda uu_: d2(uu_, y, z), (uu,), (w,))[1]

    B = lambda x, y: d2(u_c, x, y)
    C = lambda x, y, z: d3(u_c, x, y, z)

    term1 = jnp.vdot(p_left, C(q, q, qbar))
    a_inv_b = jnp.linalg.solve(Ac, B(q, qbar))
    term2 = 2.0 * jnp.vdot(p_left, B(q, a_inv_b))
    m2 = 2j * omega0 * eye_n - Ac
    m2_inv_b = jnp.linalg.solve(m2, B(q, q))
    term3 = jnp.vdot(p_left, B(qbar, m2_inv_b))

    return jnp.real(term1 - term2 + term3) / (4.0 * omega0)
