"""
Two-parameter continuation: curves of folds and Hopf points
============================================================

Traces a *curve* of fold points and a *curve* of Hopf points through the
(p0, p1) plane, marks the codim-2 points where they degenerate, and then
shows the two things MatCont and BifurcationKit.jl cannot do at all: the
whole curve batched under ``jax.vmap``, and the exact gradient of a
codim-2 location under ``jax.grad``.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jaxcont as jc

os.makedirs("images", exist_ok=True)

def bt_system(u, p, args):
    """x' = y ; y' = b1 + b2*x + x^2 + x*y, shifted so the BT sits at
    u = (5, 2), p = (3, -1). Its fold curve is the exact parabola
    p0 = 3 + (p1 + 1)^2 / 4."""
    X, Y = u[0], u[1]
    x, y = X - 5.0, Y - 2.0
    b1 = p[0] - 3.0
    b2 = p[1] + 1.0
    return jnp.array([y, b1 + b2 * x + x**2 + x * y])


# %%
# Trace the fold curve and detect the Bogdanov-Takens point
# ---------------------------------------------------------
prob = jc.fold_curve_problem(
    bt_system, jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0]), free=1,
)
sol = jc.continuation(
    prob,
    p_span=(-2.0, 0.0),
    settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    events=[jc.BogdanovTakens(raw_f=bt_system, free=1, curve="fold")],
)

print(f"curve points: {sol.branch.params.shape[0]}")
for hit in sol.events:
    print(f"  {hit.kind:18s} p = {hit.info['p']}  (exact: [3, -1])")

ax = jc.plot_two_parameter_diagram([(sol, "fold")], free=1)
ax.set_title("Fold curve with its Bogdanov-Takens point")
plt.savefig("images/example_15_two_parameter_diagram.png", dpi=140,
            bbox_inches="tight")

# %%
# Batched: 64 curves in one kernel
# --------------------------------
# The curve is an ordinary BifProblem, so the existing scan engine's vmap
# support applies -- but not by reconstructing the problem inside the
# vmapped function. ``fold_curve_problem`` does an eager Newton seed
# refinement (``fold_point``) at construction time; tracing that call under
# ``jax.vmap`` doesn't raise, but it silently returns an all-zero result
# (confirmed by direct comparison against the eager values). The fix,
# exactly as flagged: build ONE ``BifProblem`` outside the vmapped function,
# thread the swept quantity through ``args`` (not a Python closure), and
# vary it per batch element with ``prob.at(args=...)``, vmapping only
# ``jc.continuation()`` itself.
#
# A second wrinkle: under ``jax.vmap``/``jax.jit`` the branch buffers are
# NOT trimmed to the number of real points (only eager calls trim to
# ``n_valid`` -- see ``api.py``'s ``Branch`` docstring), so ``states[-1]``
# grabs unwritten buffer padding, not the curve's actual endpoint. The
# fix is the same ``Branch.valid`` mask the library ships for exactly this
# situation: the last ``True`` index is the real endpoint.
def bt_system_shiftable(u, p, args):
    shift = args
    return bt_system(u, p + jnp.array([shift, 0.0]), None)


base_prob = jc.fold_curve_problem(
    bt_system_shiftable, jnp.array([5.5, 2.0]), jnp.array([3.25, -2.0]),
    free=1, args=0.0,
)


def curve_endpoint(shift):
    res = jc.continuation(
        base_prob.at(args=shift), p_span=(-2.0, 0.0),
        settings=jc.ContinuationPar(compute_stability=False,
                                    newton_tol=1e-5, max_steps=200),
    )
    last = jnp.sum(res.branch.valid.astype(jnp.int32)) - 1
    return jnp.take(res.branch.states, last, axis=0)[2]   # solved p0 at the curve's end


shifts = jnp.linspace(0.0, 0.5, 64)
batched = jax.vmap(curve_endpoint)(shifts)
print(f"64 curves batched in one kernel; endpoint spread: "
      f"{float(batched.min()):.4f} .. {float(batched.max()):.4f}")

# %%
# Differentiable: exact sensitivity of a codim-2 location
# -------------------------------------------------------
# bogdanov_takens_parameters is built on the same implicit-function-theorem
# primitive as the rest of the library, so jax.grad skips the iteration --
# but only for the variable it is told about explicitly via ``args``
# (``differentiable_root``'s ``theta``). Closing over the swept ``shift``
# in a Python closure, as the fold-curve seed above initially did, leaks a
# tracer across the underlying ``custom_vjp`` boundary
# (``UnexpectedTracerError``) instead of differentiating correctly; routing
# ``shift`` through ``args`` fixes it the same way the batched section
# above does.
def bt_p0_given_shift(shift):
    p_star = jc.bogdanov_takens_parameters(
        bt_system_shiftable, jnp.array([5.3, 1.7]), jnp.array([2.6, -0.8]),
        args=shift,
    )
    return p_star[0]


g = jax.grad(bt_p0_given_shift)(0.0)
fd = (bt_p0_given_shift(1e-3) - bt_p0_given_shift(-1e-3)) / 2e-3
print(f"d(BT p0)/d(shift): grad = {float(g):.5f}, "
      f"finite-diff = {float(fd):.5f}")
assert abs(float(g) - float(fd)) < 1e-2, "gradient disagrees with FD"
