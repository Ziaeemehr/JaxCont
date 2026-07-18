"""
Batched continuation with ``vmap``
======================================

The capability that most sets JaxCont apart from BifurcationKit.jl / MATCONT:
because the whole-loop engine is a pure function of its inputs, an entire
*family* of bifurcation diagrams computes
as ONE compiled, vectorized kernel via ``jax.vmap`` -- ideal for GPUs,
parameter scans, ensembles, and multistart.

Here we sweep a second parameter :math:`b` (an imperfection) of the
imperfect pitchfork

.. math::

    f(u, p; b) = p u - u^3 + b

and compute one branch per :math:`b` value, comparing a single ``vmap`` call
against a Python loop over the same engine.
"""

# %%
# Setup

import os
import time

import jax
import jax.numpy as jnp

from jaxcont.core.scan_continuation import pseudo_arclength_scan

MAX_STEPS = 120

# %%
# Define the system and a single-run helper
# ---------------------------------------------
# ``run_one`` continues one branch for a given imperfection ``b``, using the
# fully JIT-compiled whole-loop engine -- it's the *whole loop* being one
# compiled program (rather than many small dispatched ops) that makes
# ``vmap`` batching possible. The companion differentiable-analysis example
# looks at what this same engine can and can't be differentiated through.


def make_rhs(b):
    return lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])


def run_one(b):
    return pseudo_arclength_scan(
        make_rhs(b),
        jnp.array([0.05]),   # u0
        jnp.array(-1.0),     # p0
        jnp.array(2.0),      # p_end
        jnp.array(0.05),     # ds
        jnp.array(1e-5),     # ds_min
        jnp.array(0.2),      # ds_max
        jnp.array(1e-6),     # tol (float32-reachable)
        MAX_STEPS,           # max_steps (static)
        jnp.array(20),       # newton_max_iter
    )


# %%
# Run 256 diagrams with a single ``vmap`` call
# -------------------------------------------------

n_diagrams = 256
bs = jnp.linspace(-0.4, 0.4, n_diagrams)

batched = jax.vmap(run_one)
res = batched(bs)  # trigger compilation
jax.block_until_ready(res)

t0 = time.perf_counter()
res = batched(bs)
jax.block_until_ready(res)
t_vmap = time.perf_counter() - t0

print(f"vmap (one kernel): {t_vmap * 1e3:.2f} ms for {n_diagrams} diagrams")

# %%
# Compare against a sequential Python loop over the same engine
# -------------------------------------------------------------------
# Same computation, same JIT-compiled ``run_one`` -- the only difference is
# whether JAX batches the calls into one kernel or dispatches them one at a
# time.

_ = run_one(bs[0])  # compile once, outside the timed loop

t0 = time.perf_counter()
for b in bs:
    r = run_one(b)
    jax.block_until_ready(r)
t_loop = time.perf_counter() - t0

print(f"Python loop (engine): {t_loop * 1e3:.2f} ms for {n_diagrams} diagrams")
print(f"speedup: {t_loop / t_vmap:.1f}x")

# %%
# Inspect the batched result
# -------------------------------
# ``res.params``/``res.states`` are stacked across the batch dimension;
# ``n_valid`` tells you how many of the fixed-size buffer entries are real
# points for each diagram.

print(f"result shapes: params {tuple(res.params.shape)}, states {tuple(res.states.shape)}")
print(f"points per diagram (n_valid): min={int(res.n_valid.min())} max={int(res.n_valid.max())}")

# %%
# Plot a sample of the batched diagrams
# -------------------------------------------

import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))
idx = jnp.linspace(0, n_diagrams - 1, 9).astype(int)
for i in idx:
    n = int(res.n_valid[i])
    ax.plot(res.params[i, :n], res.states[i, :n, 0], lw=1.2, label=f"b={float(bs[i]):+.2f}")

ax.set_xlabel("parameter p")
ax.set_ylabel("state u")
ax.set_title("Imperfect pitchfork: one vmap, many diagrams")
ax.legend(fontsize=8, ncol=3)
plt.savefig("images/example_06_vmap_sweep.png", dpi=110, bbox_inches="tight")
plt.show()
