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
against a Python loop, using the public ``jc.continuation()`` API directly --
no need to drop to the lower-level ``pseudo_arclength_scan`` engine, since
``jc.continuation()`` is itself ``vmap``-safe for the branch/stability data.
"""

# %%
# Setup

import os
import time

import jax
import jax.numpy as jnp

import jaxcont as jc

MAX_STEPS = 120

# %%
# Define the system and a single-run helper
# ---------------------------------------------
# ``run_one`` continues one branch for a given imperfection ``b`` via
# ``jc.continuation()``. Under the hood this still runs the fully
# JIT-compiled whole-loop engine -- it's the *whole loop* being one compiled
# program (rather than many small dispatched ops) that makes ``vmap``
# batching possible. The companion differentiable-analysis example looks at
# what this same engine can and can't be differentiated through.


def imperfect_pitchfork(u, p, b):
    return jnp.array([p * u[0] - u[0] ** 3 + b])


def run_one(b):
    prob = jc.bif_problem(imperfect_pitchfork, u0=jnp.array([0.05]), p0=-1.0, args=b)
    return jc.continuation(
        prob,
        p_span=(-1.0, 2.0),
        settings=jc.ContinuationPar(
            ds=0.05, ds_min=1e-5, ds_max=0.2,
            max_steps=MAX_STEPS, newton_tol=1e-6, newton_max_iter=20,
        ),
        # events=[...] is not yet supported inside jax.vmap -- see
        # notes/ROADMAP.md issue #13. Omitting `events` gives the
        # vmap-safe branch/stability data used below.
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
# Compare against a sequential Python loop over the same call
# -------------------------------------------------------------------
# Same computation, same JIT-compiled engine underneath -- the only
# difference is whether JAX batches the calls into one kernel or dispatches
# them one at a time.

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
# ``res.branch.params``/``res.branch.states`` are stacked across the batch
# dimension at the engine's fixed buffer size (``MAX_STEPS + 1``);
# ``res.branch.valid`` is a boolean mask (and ``res.stats["n_valid"]`` the
# equivalent count) telling you which of those fixed-size entries are real
# points for each diagram -- since ``vmap`` requires uniform output shapes,
# trimming to the true (ragged) branch length only happens after the trace
# exits, per diagram, exactly as below.

print(f"result shapes: params {tuple(res.branch.params.shape)}, "
      f"states {tuple(res.branch.states.shape)}")
n_valid = res.stats["n_valid"]
print(f"points per diagram (n_valid): min={int(n_valid.min())} max={int(n_valid.max())}")

# %%
# Plot a sample of the batched diagrams
# -------------------------------------------

import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 5))
idx = jnp.linspace(0, n_diagrams - 1, 9).astype(int)
for i in idx:
    mask = res.branch.valid[i]
    ax.plot(res.branch.params[i][mask], res.branch.states[i, :, 0][mask],
            lw=1.2, label=f"b={float(bs[i]):+.2f}")

ax.set_xlabel("parameter p")
ax.set_ylabel("state u")
ax.set_title("Imperfect pitchfork: one vmap, many diagrams")
ax.legend(fontsize=8, ncol=3)
plt.savefig("images/example_06_vmap_sweep.png", dpi=110, bbox_inches="tight")
plt.show()
