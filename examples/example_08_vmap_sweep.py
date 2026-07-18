"""
Example 08 — Batched continuation with `vmap` (the JAX flagship).

The single capability that most sets JaxCont apart from BifurcationKit.jl / MATCONT
(see notes/ARCHITECTURE.md §3.1): because the whole-loop engine is a pure function
of its inputs, an entire *family* of bifurcation diagrams computes as ONE compiled,
vectorized kernel via `jax.vmap` — ideal for GPUs, parameter scans, ensembles, and
multistart.

Here we sweep a second parameter `b` (an imperfection) of the imperfect pitchfork

    f(u, p; b) = p * u - u^3 + b

and compute one branch per `b` value. We compare a single `vmap` call against a
Python loop over the same engine to show the batching win.
"""

import time

import jax
import jax.numpy as jnp

from jaxcont.core.scan_continuation import pseudo_arclength_scan

MAX_STEPS = 120


def make_rhs(b):
    """Imperfect-pitchfork RHS with imperfection `b` closed over."""
    return lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])


def run_one(b):
    """Continue the branch for a single imperfection value `b`."""
    return pseudo_arclength_scan(
        make_rhs(b),
        jnp.array([0.05]),          # u0
        jnp.array(-1.0),            # p0
        jnp.array(2.0),             # p_end
        jnp.array(0.05),            # ds
        jnp.array(1e-5),            # ds_min
        jnp.array(0.2),             # ds_max
        jnp.array(1e-6),            # tol (float32-reachable)
        MAX_STEPS,                  # max_steps (static)
        jnp.array(20),              # newton_max_iter
    )


def main():
    n_diagrams = 256
    bs = jnp.linspace(-0.4, 0.4, n_diagrams)

    print("=" * 72)
    print(f"Batched continuation: {n_diagrams} bifurcation diagrams")
    print("=" * 72)

    # --- one vmapped kernel ------------------------------------------------
    batched = jax.vmap(run_one)
    res = batched(bs)                       # compile
    jax.block_until_ready(res)
    t0 = time.perf_counter()
    res = batched(bs)
    jax.block_until_ready(res)
    t_vmap = time.perf_counter() - t0

    # --- sequential Python loop over the same engine -----------------------
    _ = run_one(bs[0])                      # compile once
    t0 = time.perf_counter()
    for b in bs:
        r = run_one(b)
        jax.block_until_ready(r)
    t_loop = time.perf_counter() - t0

    print(f"\n  vmap (one kernel):     {t_vmap * 1e3:8.2f} ms")
    print(f"  Python loop (engine):  {t_loop * 1e3:8.2f} ms")
    print(f"  speedup:               {t_loop / t_vmap:8.1f}x")
    print(f"\n  result shapes: params {tuple(res.params.shape)}, "
          f"states {tuple(res.states.shape)}")
    print(f"  points computed per diagram (n_valid): "
          f"min={int(res.n_valid.min())} max={int(res.n_valid.max())}")

    # --- optional plot -----------------------------------------------------
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        idx = jnp.linspace(0, n_diagrams - 1, 9).astype(int)
        for i in idx:
            n = int(res.n_valid[i])
            ax.plot(res.params[i, :n], res.states[i, :n, 0],
                    lw=1.2, label=f"b={float(bs[i]):+.2f}")
        ax.set_xlabel("parameter p")
        ax.set_ylabel("state u")
        ax.set_title("Imperfect pitchfork: one vmap, many diagrams")
        ax.legend(fontsize=8, ncol=3)
        out = os.path.join(os.path.dirname(__file__), "images",
                           "example_08_vmap_sweep.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\n  saved plot -> {out}")
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"\n  (plot skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
