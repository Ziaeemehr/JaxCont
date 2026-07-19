"""
GPU smoke test.

Not a correctness test (that's covered on CPU by the rest of the suite) -- this
exists purely to answer "does the differentiable/`vmap` story actually execute
on a GPU device", per notes/ROADMAP.md's v0.1.0 release-engineering checklist.

Skipped automatically (not failed) whenever no GPU device is usable, which
covers two distinct situations found while writing this file:
  1. No GPU hardware at all (e.g. GitHub Actions' default `ubuntu-latest`
     runner) -- `jax.devices("gpu")` raises/returns empty.
  2. GPU hardware present but not actually usable (e.g. a driver/cuDNN version
     mismatch) -- `jax.devices("gpu")` lists a device, but a real computation
     on it fails at dispatch time. Verified on this project's own dev machine:
     an RTX A5000 is listed by `jax.devices()`, but cuDNN refuses to
     initialize against driver 535.183, so most kernels silently fall back to
     wrong/aborted results rather than raising a clean ImportError. See
     ROADMAP.md issue #11 -- `JAX_PLATFORMS=cpu` sidesteps this on such a
     machine; this file exists to give a *real* GPU environment (this dev
     machine's is not one) something to actually run.

Excluded from the default test run via the `gpu` marker (see conftest.py and
pyproject.toml's `addopts`), mirroring how `slow` tests are opt-in. Run
explicitly with:

    pytest -m gpu -v

on a machine/CI runner with a genuinely working GPU.
"""

import pytest
import jax
import jax.numpy as jnp

import jaxcont as jc

pytestmark = pytest.mark.gpu


def _gpu_device_or_skip():
    """Return a usable GPU device, or skip the test with a clear reason."""
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if not gpu_devices:
        pytest.skip("No GPU device visible to JAX (jax.devices('gpu') is empty).")

    device = gpu_devices[0]
    try:
        # A real (if tiny) linear solve, not just an add -- cheap enough to be
        # a smoke test, expensive enough to exercise cuBLAS/cuDNN dispatch,
        # which is exactly what silently breaks under a driver mismatch.
        A = jax.device_put(jnp.eye(3), device)
        b = jax.device_put(jnp.ones(3), device)
        x = jnp.linalg.solve(A, b)
        x.block_until_ready()
        if not bool(jnp.all(jnp.isfinite(x))):
            raise RuntimeError("GPU linear solve returned non-finite values.")
    except Exception as exc:  # pragma: no cover - depends on GPU/driver state
        pytest.skip(f"GPU device visible but not usable ({exc!r}).")

    return device


def test_gpu_device_present_and_computes():
    """Baseline: a GPU device exists and can run a real linear algebra op."""
    device = _gpu_device_or_skip()
    assert device.platform == "gpu"


def test_continuation_runs_on_gpu():
    """The default `continuation()` path executes on GPU, not silently on CPU."""
    device = _gpu_device_or_skip()

    def pitchfork(u, p, args):
        return p * u - u**3

    with jax.default_device(device):
        prob = jc.bif_problem(pitchfork, u0=jnp.array([0.1]), p0=-1.0)
        sol = jc.continuation(prob, p_span=(-1.0, 1.0), events=[jc.Fold()])

    assert sol.branch.n_valid > 1
    assert bool(jnp.all(jnp.isfinite(sol.branch.states)))
    assert sol.branch.states.devices() == {device}


def test_vmap_sweep_runs_on_gpu():
    """The flagship `vmap`-batched sweep (example_06) executes on GPU.

    Deliberately calls `pseudo_arclength_scan` directly rather than the
    higher-level `jc.continuation()`, matching `example_06_vmap_sweep.py` --
    for the same reason that example does: `jc.continuation()`'s `_run_scan`
    concretizes `n_valid` with a bare `int(...)` to trim the fixed-size
    buffer, which raises `jax.errors.ConcretizationTypeError` under `vmap`
    (confirmed while writing this test). `pseudo_arclength_scan` itself
    returns the untrimmed fixed-size buffer and is genuinely vmap-safe. See
    ROADMAP.md issue #13 for the fix (make `_run_scan` vmap-safe, tracked
    under the v0.2 engine-consolidation work) -- this test exercises the path
    that actually works today rather than the one that's supposed to.
    """
    device = _gpu_device_or_skip()

    from jaxcont.core.scan_continuation import pseudo_arclength_scan

    def make_rhs(b):
        return lambda u, p: jnp.array([p * u[0] - u[0] ** 3 + b])

    def run_one(b):
        res = pseudo_arclength_scan(
            make_rhs(b),
            jnp.array([0.05]),   # u0
            jnp.array(-1.0),     # p0
            jnp.array(2.0),      # p_end
            jnp.array(0.05),     # ds
            jnp.array(1e-5),     # ds_min
            jnp.array(0.2),      # ds_max
            jnp.array(1e-6),     # tol
            60,                  # max_steps (static)
            jnp.array(20),       # newton_max_iter
        )
        return res.states[:, 0]

    with jax.default_device(device):
        b_values = jnp.linspace(-0.2, 0.2, 8)
        branches = jax.vmap(run_one)(b_values)

    assert branches.shape == (8, 61)
    assert bool(jnp.all(jnp.isfinite(branches)))
    assert branches.devices() == {device}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
