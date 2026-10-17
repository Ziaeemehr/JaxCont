"""Static curve representations and normalized diagnostic modes."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxcont.utils.precision import default_tol


def minimal_hopf_floor(dtype, scale=1.0):
    """Lowest frequency of the minimal Hopf chart,
    ``sqrt(default_tol(dtype)) * scale``.

    ``hopf_curve_problem`` uses ``scale = |omega_seed|``, the critical pair's
    frequency at the seed, so the floor follows that pair's time scale. The
    chart's border residual is O(omega) near a Bogdanov-Takens point, so
    below this floor it no longer separates Hopf points from their
    neighbourhood at the corrector tolerance.
    """
    return default_tol(dtype) ** 0.5 * scale


@dataclass(frozen=True)
class CurveLayout:
    kind: str
    formulation: str
    n: int
    free: int

    def __post_init__(self):
        if (self.kind not in ("fold", "hopf")
                or self.formulation not in ("minimal", "extended", "bt")
                or (self.kind == "fold" and self.formulation == "bt")):
            raise ValueError("Unsupported curve layout")
        if self.n < 1 or self.free not in (0, 1):
            raise ValueError("Invalid curve layout dimensions or free parameter")


class CurveModes(NamedTuple):
    v: object = None
    w: object = None
    q1: object = None
    q2: object = None
    omega: object = None
    kappa: object = None
    oscillatory: object = None


def _unit(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.where(norm > 0, norm, 1)


def decode_curve_state(X, q, layout, raw_f, user_args):
    """Decode an explicitly described curve state in original coordinates.

    Ordinary modes are diagnostic unit vectors; BT returns stored real v,w.
    Packed Newton unknowns are unchanged.
    With no RHS, minimal states decode coordinates only (for plotting).
    """
    if layout is None:
        raise ValueError("curve layout is required for curve decoding")
    n = layout.n
    width = (n + 1 if layout.formulation == "minimal" else 2 * n + 1) if layout.kind == "fold" else (n + 2 if layout.formulation == "minimal" else 3 * n + 2)
    if X.shape[-1] != width:
        raise ValueError("State width does not match curve layout")
    p = jnp.stack((jnp.asarray(q), X[n])) if layout.free == 0 else jnp.stack((X[n], jnp.asarray(q)))
    if layout.formulation == "bt":
        from jaxcont.bifurcations.curves import unpack_bt_hopf_curve, hopf_curve_frequency
        u, _, v, w, kappa = unpack_bt_hopf_curve(X, n)
        omega, oscillatory = hopf_curve_frequency(kappa)
        return u, p, CurveModes(v=v, w=w, q1=v, q2=omega*w,
                                omega=omega, kappa=kappa, oscillatory=oscillatory)
    if layout.kind == "fold":
        if layout.formulation == "minimal":
            if raw_f is None:
                return X[:n], p, CurveModes()
            A = jax.jacfwd(raw_f)(X[:n], p, user_args)
            left, _, vh = jnp.linalg.svd(A, full_matrices=False)
            b = jax.lax.stop_gradient(left[:, -1])
            c = jax.lax.stop_gradient(vh[-1])
            M = jnp.block([[A, b[:, None]], [c[None, :], jnp.zeros((1, 1), dtype=A.dtype)]])
            rhs = jnp.zeros(n + 1, dtype=X.dtype).at[-1].set(1)
            v = _unit(jnp.linalg.solve(M, rhs)[:-1])
        else:
            v = _unit(X[n + 1 :])
        pivot = v[jnp.argmax(jnp.abs(v))]
        v = jnp.where(pivot < 0, -v, v)
        modes = CurveModes(v=v)
    else:
        if layout.formulation == "minimal":
            if raw_f is None:
                return X[:n], p, CurveModes(omega=X[-1], oscillatory=X[-1] > 0)
            from jaxcont.bifurcations.curves import _hopf_border_matrix
            A = jax.jacfwd(raw_f)(X[:n], p, user_args)
            C = A - 1j * X[-1] * jnp.eye(n, dtype=A.dtype)
            left, _, vh = jnp.linalg.svd(C, full_matrices=False)
            # Borders define a fixed local chart; mode derivatives use the solve.
            b = jax.lax.stop_gradient(left[:, -1])
            c = jax.lax.stop_gradient(vh[-1].conj())
            M = _hopf_border_matrix(A, X[-1], b, c)
            rhs = jnp.zeros(n + 1, dtype=M.dtype).at[-1].set(1)
            z = _unit(jnp.linalg.solve(M, rhs)[:-1])
        else:
            z = _unit(X[n + 1 : 2 * n + 1] + 1j * X[2 * n + 1 : 3 * n + 1])
        modes = CurveModes(q1=z.real, q2=z.imag, omega=X[-1], oscillatory=X[-1] > 0)
    return X[:n], p, modes


def orient_curve_modes(modes, valid):
    """Align one branch; invalid slots preserve carry and disable brackets.

    Returns aligned modes and eligibility for each adjacent bracket. Each
    vmapped invocation owns an independent carry.
    """
    fold = modes.v is not None
    raw = modes.v if fold else modes.q1 + 1j * modes.q2
    vectors = _unit(raw)
    finite = jnp.all(jnp.isfinite(raw), axis=-1) & (jnp.linalg.norm(raw, axis=-1) > 0)
    valid = valid & finite
    tol = default_tol(raw.real.dtype)

    def step(carry, item):
        previous, seen = carry
        vector, ok = item
        overlap = jnp.vdot(previous, vector)
        pivot = vector[jnp.argmax(jnp.abs(vector))]
        if fold:
            first = jnp.where(pivot < 0, -1.0, 1.0)
            factor = jnp.where(seen, jnp.where(overlap < 0, -1.0, 1.0), first)
        else:
            phase = jnp.where(seen & (jnp.abs(overlap) > tol), overlap, pivot)
            factor = jnp.conj(phase) / jnp.where(jnp.abs(phase) > 0, jnp.abs(phase), 1)
        aligned = vector * factor
        reliable = ok & (~seen | (jnp.abs(overlap) > tol))
        return (jnp.where(ok, aligned, previous), seen | ok), (aligned, reliable)

    _, (aligned, reliable) = jax.lax.scan(
        step, (jnp.zeros_like(vectors[0]), jnp.array(False)), (vectors, valid)
    )
    eligible = valid[:-1] & valid[1:] & reliable[1:]
    if fold:
        return modes._replace(v=aligned), eligible
    return modes._replace(q1=aligned.real, q2=aligned.imag), eligible
