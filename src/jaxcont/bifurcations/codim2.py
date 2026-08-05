"""
Direct codim-2 bifurcation point solvers.

Each solver builds a square extended system ``G(x, theta) = 0`` whose root
is a codim-2 point, and solves it with
``solvers/implicit.py:differentiable_root`` -- the same Newton-in-
``lax.while_loop`` + ``jax.custom_vjp`` machinery ``fold_solve.py`` and
``hopf_normal_form.py`` already use for codim-1. The result is therefore
differentiable in ``args`` via the implicit function theorem, so
``jax.grad`` of a codim-2 *location* with respect to design parameters
works without differentiating through the iteration.

Codim-2 points need TWO free parameters, so throughout this module ``p``
has shape ``(2,)`` rather than being a scalar. The right-hand side keeps
the usual ``f(u, p, args)`` signature; only ``p``'s shape changes.

These are refinement tools: every solver needs a guess already near the
codim-2 point and none of them search. Finding codim-2 points you cannot
already approximate requires two-parameter continuation, which is a
separate (unstarted) roadmap item.

See docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array, jacfwd, lax

from jaxcont.bifurcations.fold_normal_form import fold_coefficient
from jaxcont.bifurcations.fold_solve import _initial_v
from jaxcont.bifurcations.hopf_normal_form import _seed as _hopf_seed
from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient
from jaxcont.solvers.implicit import differentiable_root

PyTree = Any


def _solve_and_check(
    G: Callable[[Array, PyTree], Array],
    x0: Callable[[PyTree], Array],
    args: PyTree,
    *,
    tol: float,
    max_iter: int,
) -> Tuple[Array, Array]:
    """
    Solve ``G(x, args) = 0`` and report whether the result is trustworthy.

    ``differentiable_root`` returns only the root, and its Newton loop exits
    on non-finite iterates as well as on convergence, so the caller cannot
    tell success from failure without re-checking. ``converged`` is a JAX
    boolean (not a Python bool) so callers stay ``jit``/``vmap``-safe.
    """
    x_star = differentiable_root(G, x0, args, tol=tol, max_iter=max_iter)
    residual = jnp.linalg.norm(G(x_star, args))
    converged = (
        jnp.isfinite(residual)
        & (residual < tol)
        & jnp.all(jnp.isfinite(x_star))
    )
    return x_star, converged


def _normalize_omega(q2: Array, omega: Array) -> Tuple[Array, Array]:
    """
    Pin the critical frequency to ``omega >= 0``.

    The Hopf block of every extended system below is EXACTLY invariant under
    ``(omega, q2) -> (-omega, -q2)``: substituting flips the sign of the
    ``J q2 - omega q1`` row and leaves ``J q1 + omega q2`` unchanged, so both
    signs are genuine roots and Newton picks whichever the seed falls toward
    (during planning it chose the negative one from positive seeds on more
    than one system). Flipping afterwards selects the conjugate of the same
    eigenvector, so this is exact rather than a heuristic.

    It matters beyond cosmetics: ``bifurcations/events.py:Hopf.refine()``
    already treats ``omega0 <= 0`` as a failed solve. Applied AFTER the
    solve, never as an extra equation, which would break squareness.
    """
    flip = omega < 0
    return jnp.where(flip, -q2, q2), jnp.where(flip, -omega, omega)


# --------------------------------------------------------------------------
# CP -- cusp
# --------------------------------------------------------------------------

def _cp_unpack(x: Array, n: int) -> Tuple[Array, Array, Array]:
    return x[:n], x[n:n + 2], x[n + 2:]


def _cp_residual(x, f, args, n):
    """
    Cusp extended system, ``2n+2`` equations in ``2n+2`` unknowns
    ``(u, p, v)``: the fold system plus ``a = 0``.
    """
    u, p, v = _cp_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                                        # n
        jac_u @ v,                                            # n
        jnp.reshape(jnp.dot(v, v) - 1.0, (1,)),               # 1
        jnp.reshape(fold_coefficient(f, u, p, v, args), (1,)),  # 1
    ])


def cusp_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array]:
    """
    Locate a cusp (``CP``) near ``(u_guess, p_guess)``, differentiable in
    ``args``. Kuznetsov, 3rd ed., Sec. 8.2.

    A cusp is a fold at which the quadratic normal-form coefficient ``a``
    also vanishes. Returns ``(u*, p*, v*, converged)`` where ``v*`` is the
    unit right null vector of ``f_u`` and ``p*`` has shape ``(2,)``.

    ``tol`` defaults to ``1e-6`` rather than ``1e-8`` because the achievable
    float32 residual floor for this system is around ``4e-8``; a tighter
    tolerance reports ``converged=False`` even when the answer is exact.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _cp_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        return jnp.concatenate([u_guess, p_guess, v0])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v = _cp_unpack(x_star, n)
    return u, p, v, converged


def cusp_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the cusp -- differentiable in
    ``args``. Returns a bare array with no convergence flag so
    ``jax.grad(...)`` applies directly; use :func:`cusp_point` when you need
    the flag.
    """
    _, p, _, _ = cusp_point(f, u_guess, p_guess, args, tol=tol, max_iter=max_iter)
    return p


# --------------------------------------------------------------------------
# BT -- Bogdanov-Takens
# --------------------------------------------------------------------------

def _bt_unpack(x: Array, n: int) -> Tuple[Array, Array, Array, Array]:
    return x[:n], x[n:n + 2], x[n + 2:2 * n + 2], x[2 * n + 2:]


def _bt_residual(x, f, args, n):
    """
    Bogdanov-Takens extended system, ``3n+2`` equations in ``3n+2`` unknowns
    ``(u, p, v0, v1)``.

    Uses the JORDAN-CHAIN formulation (``v1`` is the generalized
    eigenvector). The textbook left/right-null-vector alternative
    ``f=0, J v=0, J^T w=0, |v|=1, |w|=1, w.v=0`` is OVERDETERMINED -- it has
    ``3n+3`` equations for ``3n+2`` unknowns -- which is why it is not used
    here. Confirmed by direct count during planning.
    """
    u, p, v0, v1 = _bt_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                              # n
        jac_u @ v0,                                 # n
        jac_u @ v1 - v0,                            # n
        jnp.reshape(jnp.dot(v0, v0) - 1.0, (1,)),   # 1
        jnp.reshape(jnp.dot(v0, v1), (1,)),         # 1
    ])


def bogdanov_takens_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Locate a Bogdanov-Takens point (``BT``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.4.

    ``BT`` is where ``f_u`` has a double zero eigenvalue with a single
    eigenvector -- the organizing centre at which fold and Hopf curves meet.
    Returns ``(u*, p*, v0*, v1*, converged)`` with ``f_u v0 = 0`` and
    ``f_u v1 = v0`` (the Jordan chain), ``p*`` of shape ``(2,)``.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _bt_residual(x, f, theta, n)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        # A UNIFORM nonzero seed for the generalized eigenvector. Seeding
        # v1 with zeros makes the very first Newton step produce nan --
        # verified during planning, and NOT obvious from the residual being
        # linear in v1. Do not "simplify" this to jnp.zeros(n).
        v1 = jnp.ones(n, u_guess.dtype) / jnp.sqrt(jnp.asarray(n, u_guess.dtype))
        return jnp.concatenate([u_guess, p_guess, v0, v1])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v0, v1 = _bt_unpack(x_star, n)
    return u, p, v0, v1, converged


def bogdanov_takens_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the Bogdanov-Takens point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _ = bogdanov_takens_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p


# --------------------------------------------------------------------------
# GH -- generalized Hopf (Bautin)
# --------------------------------------------------------------------------

def _gh_unpack(x, n):
    return (x[:n], x[n:n + 2], x[n + 2:2 * n + 2],
            x[2 * n + 2:3 * n + 2], x[-1])


def _gh_residual(x, f, args, n, u_guess, p_guess):
    """
    Generalized-Hopf extended system, ``3n+3`` equations in ``3n+3``
    unknowns ``(u, p, q1, q2, omega)``: the Hopf system plus ``l1 = 0``.
    """
    u, p, q1, q2, omega = _gh_unpack(x, n)
    # Seed recomputed inside the traced primal on every iteration, exactly
    # as hopf_normal_form.py does -- differentiable_root's contract requires
    # theta-dependent seeds to be resolved here, not hoisted out. The seed
    # comes from jnp.linalg.eig, which has no gradient rule for non-symmetric
    # eigenvectors, so args is stop_gradient'd here -- matching
    # hopf_normal_form.py's _extended_residual exactly. Confirmed necessary:
    # omitting this reproduces `NotImplementedError: Derivatives of
    # non-symmetric eigenvectors...` in the parameters-grad test.
    q1_seed, q2_seed, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    l1 = lyapunov_coefficient(f, u, p, q1, q2, omega, args)
    return jnp.concatenate([
        f(u, p, args),                                                   # n
        jac_u @ q1 + omega * q2,                                         # n
        jac_u @ q2 - omega * q1,                                         # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),      # 1
        # Seed-based phase condition, matching hopf_normal_form.py's g5.
        # The naive `q1 . q2 == 0` alternative is recorded as broken in the
        # Hopf design spec -- do not substitute it.
        jnp.reshape(jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1), (1,)),  # 1
        jnp.reshape(l1, (1,)),                                           # 1
    ])


def generalized_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Locate a generalized Hopf / Bautin point (``GH``) near
    ``(u_guess, p_guess)``, differentiable in ``args``. Kuznetsov, 3rd ed.,
    Sec. 8.3.

    ``GH`` is a Hopf point at which the first Lyapunov coefficient ``l1``
    also vanishes -- the point separating supercritical from subcritical
    Hopf along a Hopf curve. Returns
    ``(u*, p*, q1*, q2*, omega*, converged)``; ``omega* >= 0`` by
    construction (see :func:`_normalize_omega`) and ``p*`` has shape ``(2,)``.

    ``f`` must be complex-analytic (holomorphic) in ``u`` near the point,
    inherited from :func:`~jaxcont.bifurcations.hopf_normal_form.lyapunov_coefficient`.

    ``tol`` defaults to ``1e-6``: the achievable float32 residual floor for
    this system is about ``6e-8`` (measured), so ``1e-8`` would report
    ``converged=False`` even where the answer is exact, while ``1e-4`` stops
    early with visible parameter error.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _gh_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        q1_0, q2_0, omega_0 = _hopf_seed(f, u_guess, p_guess, theta, n)
        return jnp.concatenate(
            [u_guess, p_guess, q1_0, q2_0, jnp.reshape(omega_0, (1,))]
        )

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1, q2, omega = _gh_unpack(x_star, n)
    q2, omega = _normalize_omega(q2, omega)
    return u, p, q1, q2, omega, converged


def generalized_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the generalized Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _, _ = generalized_hopf_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p


# --------------------------------------------------------------------------
# ZH -- zero-Hopf
# --------------------------------------------------------------------------

def _zh_unpack(x, n):
    return (x[:n], x[n:n + 2], x[n + 2:2 * n + 2],
            x[2 * n + 2:3 * n + 2], x[3 * n + 2:4 * n + 2], x[-1])


def _zh_residual(x, f, args, n, u_guess, p_guess):
    """
    Zero-Hopf extended system, ``4n+3`` equations in ``4n+3`` unknowns
    ``(u, p, v, q1, q2, omega)``: one equilibrium condition carrying both a
    fold block and a Hopf block.
    """
    u, p, v, q1, q2, omega = _zh_unpack(x, n)
    # stop_gradient required here -- see the identical comment in
    # _gh_residual above. Task 4 found this the hard way (a missing
    # stop_gradient here fails the gradient test with a NotImplementedError
    # from jnp.linalg.eig's vjp on non-symmetric eigenvectors); `lax` is
    # already imported in codim2.py by Task 4's fix, no new import needed.
    q1_seed, q2_seed, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(args), n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),                                                   # n
        jac_u @ v,                                                       # n
        jnp.reshape(jnp.dot(v, v) - 1.0, (1,)),                          # 1
        jac_u @ q1 + omega * q2,                                         # n
        jac_u @ q2 - omega * q1,                                         # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),      # 1
        jnp.reshape(jnp.dot(q1_seed, q2) - jnp.dot(q2_seed, q1), (1,)),  # 1
    ])


def zero_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Locate a zero-Hopf point (``ZH``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.5.

    ``ZH`` is where ``f_u`` simultaneously has a zero eigenvalue and a pair
    of purely imaginary eigenvalues -- the intersection of a fold curve and
    a Hopf curve. Requires ``n >= 3``. Returns
    ``(u*, p*, v*, q1*, q2*, omega*, converged)`` with ``omega* >= 0`` and
    ``p*`` of shape ``(2,)``.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)

    def G(x, theta):
        return _zh_residual(x, f, theta, n, u_guess, p_guess)

    def x0(theta):
        v0 = _initial_v(f, u_guess, p_guess, theta, n)
        q1_0, q2_0, omega_0 = _hopf_seed(f, u_guess, p_guess, theta, n)
        return jnp.concatenate(
            [u_guess, p_guess, v0, q1_0, q2_0, jnp.reshape(omega_0, (1,))]
        )

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, v, q1, q2, omega = _zh_unpack(x_star, n)
    q2, omega = _normalize_omega(q2, omega)
    return u, p, v, q1, q2, omega, converged


def zero_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the zero-Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    _, p, _, _, _, _, _ = zero_hopf_point(
        f, u_guess, p_guess, args, tol=tol, max_iter=max_iter
    )
    return p


# --------------------------------------------------------------------------
# HH -- double Hopf
# --------------------------------------------------------------------------

def _hh_unpack(x, n):
    o = n + 2
    o2 = o + 2 * n + 1
    return (
        x[:n], x[n:o],
        x[o:o + n], x[o + n:o + 2 * n], x[o + 2 * n],
        x[o2:o2 + n], x[o2 + n:o2 + 2 * n], x[o2 + 2 * n],
    )


def _hopf_block(jac_u, q1, q2, omega, s1, s2):
    """One Hopf block: eigenvector relations, normalization, phase."""
    return jnp.concatenate([
        jac_u @ q1 + omega * q2,                                     # n
        jac_u @ q2 - omega * q1,                                     # n
        jnp.reshape(jnp.dot(q1, q1) + jnp.dot(q2, q2) - 1.0, (1,)),  # 1
        jnp.reshape(jnp.dot(s1, q2) - jnp.dot(s2, q1), (1,)),        # 1
    ])


def _hh_residual(x, f, args, n, seed_a, seed_b):
    """
    Double-Hopf extended system, ``5n+4`` equations in ``5n+4`` unknowns:
    one ``f(u,p) = 0`` plus two independent Hopf blocks, each with its own
    normalization and its own DISTINCT phase seed.
    """
    u, p, q1a, q2a, oa, q1b, q2b, ob = _hh_unpack(x, n)
    jac_u = jacfwd(f, argnums=0)(u, p, args)
    return jnp.concatenate([
        f(u, p, args),
        _hopf_block(jac_u, q1a, q2a, oa, seed_a[0], seed_a[1]),
        _hopf_block(jac_u, q1b, q2b, ob, seed_b[0], seed_b[1]),
    ])


def double_hopf_point(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    seed_b: Array,
    tol: float = 1e-6,
    max_iter: int = 50,
    separation_tolerance: float = 1e-3,
) -> Tuple[Array, ...]:
    """
    Locate a double-Hopf point (``HH``) near ``(u_guess, p_guess)``,
    differentiable in ``args``. Kuznetsov, 3rd ed., Sec. 8.6.

    ``HH`` is where ``f_u`` has TWO distinct pairs of purely imaginary
    eigenvalues. Requires ``n >= 4``. Returns
    ``(u*, p*, q1a, q2a, omega_a, q1b, q2b, omega_b, converged)`` with both
    frequencies normalized non-negative and ``p*`` of shape ``(2,)``.

    ``seed_b`` (required, keyword-only) is a real direction seeding the
    SECOND Hopf block, and it must point at a different physical pair than
    the first. The first block is seeded from the usual eigen-decomposition
    heuristic, which by construction finds only one pair; without an
    independent ``seed_b`` both blocks would converge onto that same pair,
    making the extended system structurally singular. That case is detected
    -- ``converged`` is ``False`` when
    ``abs(|omega_a| - |omega_b|) <= separation_tolerance`` -- rather than
    returned as a plausible-looking "double Hopf" that is one pair counted
    twice.
    """
    u_guess = jnp.asarray(u_guess)
    n = u_guess.shape[0]
    p_guess = jnp.asarray(p_guess, u_guess.dtype)
    seed_b = jnp.asarray(seed_b, u_guess.dtype)

    def G(x, theta):
        # stop_gradient required here -- see the identical comment in
        # _gh_residual/_zh_residual above: _hopf_seed uses jnp.linalg.eig,
        # which has no gradient rule for non-symmetric eigenvectors.
        q1_a, q2_a, _ = _hopf_seed(f, u_guess, p_guess, lax.stop_gradient(theta), n)
        # Phase seeds: block A from the eigen-decomposition, block B from
        # the caller-supplied direction and its image under f_u, so the two
        # phase conditions are genuinely independent.
        jac_g = jacfwd(f, argnums=0)(u_guess, p_guess, theta)
        s_b2 = jac_g @ seed_b
        s_b2 = s_b2 / (jnp.linalg.norm(s_b2) + jnp.finfo(u_guess.dtype).eps)
        return _hh_residual(
            x, f, theta, n, (q1_a, q2_a), (seed_b, s_b2)
        )

    def x0(theta):
        q1_a, q2_a, omega_a = _hopf_seed(f, u_guess, p_guess, theta, n)
        jac_g = jacfwd(f, argnums=0)(u_guess, p_guess, theta)
        q2_b = jac_g @ seed_b
        q2_b = q2_b / (jnp.linalg.norm(q2_b) + jnp.finfo(u_guess.dtype).eps)
        # Unit-normalize the (q1, q2) pair as the block's own condition wants.
        scale = jnp.sqrt(jnp.dot(seed_b, seed_b) + jnp.dot(q2_b, q2_b))
        q1_b = seed_b / scale
        q2_b = q2_b / scale
        omega_b = 2.0 * omega_a
        return jnp.concatenate([
            u_guess, p_guess,
            q1_a, q2_a, jnp.reshape(omega_a, (1,)),
            q1_b, q2_b, jnp.reshape(jnp.asarray(omega_b, u_guess.dtype), (1,)),
        ])

    x_star, converged = _solve_and_check(G, x0, args, tol=tol, max_iter=max_iter)
    u, p, q1a, q2a, oa, q1b, q2b, ob = _hh_unpack(x_star, n)
    q2a, oa = _normalize_omega(q2a, oa)
    q2b, ob = _normalize_omega(q2b, ob)
    separated = jnp.abs(oa - ob) > separation_tolerance
    converged = converged & separated
    return u, p, q1a, q2a, oa, q1b, q2b, ob, converged


def double_hopf_parameters(
    f: Callable[[Array, Array, PyTree], Array],
    u_guess: Array,
    p_guess: Array,
    args: PyTree = None,
    *,
    seed_b: Array,
    tol: float = 1e-6,
    max_iter: int = 50,
    separation_tolerance: float = 1e-3,
) -> Array:
    """
    Parameter pair ``p*`` (shape ``(2,)``) at the double-Hopf point --
    differentiable in ``args``, no convergence flag (see
    :func:`cusp_parameters`).
    """
    result = double_hopf_point(
        f, u_guess, p_guess, args, seed_b=seed_b, tol=tol, max_iter=max_iter,
        separation_tolerance=separation_tolerance,
    )
    return result[1]
