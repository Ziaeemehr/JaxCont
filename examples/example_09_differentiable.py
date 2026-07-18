"""
Differentiable bifurcation analysis
===================================

Use ``jax.grad`` and ``jax.jacfwd`` for inverse design and sensitivity.

The capability that Julia/MATLAB continuation tools don't offer natively
(notes/ARCHITECTURE.md §3.2): because everything is JAX, we can *differentiate*
through the analysis and use gradients for inverse design and sensitivity.

  Part A — Reverse-mode `jax.grad` of a FOLD LOCATION (`jc.fold_parameter`).
    A fold is solved as an extended system G(u,p,v;θ)=0 and wrapped in
    `custom_vjp` (implicit function theorem), so the fold parameter p*(θ) is an
    exact, reverse-mode-differentiable function of the design parameters θ. We
    verify the gradient against the analytic value, then run gradient descent to
    *place the fold at a target parameter value* (inverse design).

  Part B — Forward-mode `jacfwd` sensitivity THROUGH the continuation engine.
    The whole-loop engine uses `lax.while_loop` (forward-mode AD only), so
    `jacfwd` gives the sensitivity of a computed branch to a parameter directly.
"""

import jax
import jax.numpy as jnp

import jaxcont as jc
from jaxcont.core.scan_continuation import pseudo_arclength_scan


# =========================================================================
# Part A — reverse-mode grad of a fold location + inverse design
# =========================================================================
#
# f(u, p; θ) = u^2 - θ*u + p
#   fold at u = θ/2,  p*(θ) = θ^2/4    ->    dp*/dθ = θ/2   (analytic check)

def f_fold(u, p, theta):
    return jnp.array([u[0] ** 2 - theta * u[0] + p])


def fold_p(theta):
    """Fold parameter as a differentiable function of θ."""
    return jc.fold_parameter(f_fold, jnp.array([0.4]), jnp.array(0.2), theta)


def part_a():
    print("=" * 72)
    print("Part A — reverse-mode jax.grad of a fold location + inverse design")
    print("=" * 72)

    theta0 = jnp.array(1.0)
    u, p, v = jc.fold_point(f_fold, jnp.array([0.4]), jnp.array(0.2), theta0)
    print(f"\n  fold at θ={float(theta0):.2f}:  u*={float(u[0]):.4f}  p*={float(p):.4f}"
          f"   (analytic u=θ/2={float(theta0)/2:.4f}, p=θ²/4={float(theta0)**2/4:.4f})")

    g = jax.grad(fold_p)(theta0)
    print(f"  dp*/dθ  (jax.grad)  = {float(g):.5f}    "
          f"(analytic θ/2 = {float(theta0)/2:.5f})")

    # Inverse design: choose θ so the fold sits at a target parameter value.
    target_p = 1.00                     # want p*(θ) = 1  ->  θ = 2
    theta = theta0
    lr = 0.6
    loss_grad = jax.grad(lambda th: (fold_p(th) - target_p) ** 2)
    print(f"\n  inverse design: place fold at p* -> {target_p}")
    for it in range(20):
        theta = theta - lr * loss_grad(theta)
        if it % 4 == 0 or it == 19:
            print(f"    it {it:2d}:  θ={float(theta):.4f}  p*={float(fold_p(theta)):.4f}")
    print(f"\n  final θ={float(theta):.4f}  p*={float(fold_p(theta)):.4f} "
          f"(target {target_p};  analytic θ=√(4·target)={ (4*target_p)**0.5:.4f})")


# =========================================================================
# Part B — forward-mode jacfwd through the continuation engine
# =========================================================================

def branch_observable(b, slot=8):
    """A smooth scalar read off a computed branch of the imperfect pitchfork."""
    rhs = lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])
    res = pseudo_arclength_scan(
        rhs, jnp.array([0.6]), jnp.array(0.5), jnp.array(2.0),
        jnp.array(0.05), jnp.array(1e-5), jnp.array(0.2),
        jnp.array(1e-6), 80, jnp.array(20),
    )
    return res.states[slot, 0]


def part_b():
    print("\n" + "=" * 72)
    print("Part B — forward-mode jacfwd through the whole-loop engine")
    print("=" * 72)

    b0 = jnp.array(0.1)
    d_fwd = jax.jacfwd(branch_observable)(b0)   # works with lax.while_loop
    h = 1e-3
    fd = (branch_observable(b0 + h) - branch_observable(b0 - h)) / (2 * h)
    print(f"\n  d(branch state)/db  (jacfwd)      = {float(d_fwd):+.5f}")
    print(f"  d(branch state)/db  (finite diff) = {float(fd):+.5f}   [cross-check]")
    print("\n  (reverse-mode grad through the *sweep* is unsupported by lax.while_loop;")
    print("   use jc.fold_parameter / implicit diff for reverse-mode — see Part A.)")


def main():
    part_a()
    part_b()


if __name__ == "__main__":
    main()
