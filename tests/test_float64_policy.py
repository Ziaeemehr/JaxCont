"""JaxCont is float64 only (docs/superpowers/specs/2026-10-29-precision-policy-design.md)."""

import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fresh(code, **env):
    """Run ``code`` in a new interpreter, where x64 has not been set yet."""
    environment = {k: v for k, v in os.environ.items()
                   if k not in ("JAX_ENABLE_X64", "JAXCONT_NO_AUTO_X64", "JAXCONT_TEST_X64")}
    environment.update(JAX_PLATFORMS="cpu", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]), **env)
    return subprocess.run([sys.executable, "-W", "default", "-c", code], env=environment,
                          capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)


def test_import_turns_on_x64_and_says_so():
    done = _fresh("import jax, jaxcont; print(jax.config.jax_enable_x64)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "True"
    assert "PrecisionWarning" in done.stderr and "JAXCONT_NO_AUTO_X64" in done.stderr


def test_no_warning_when_x64_is_already_on():
    done = _fresh("import jax, jaxcont; print(jax.config.jax_enable_x64)", JAX_ENABLE_X64="1")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "True"
    assert "PrecisionWarning" not in done.stderr


def test_opt_out_keeps_the_setting_and_entry_points_refuse_float32():
    code = ("import jax, jax.numpy as jnp, jaxcont as jc\n"
            "print(jax.config.jax_enable_x64)\n"
            "jc.bif_problem(lambda u, p, a: u + p, u0=jnp.array([1.]), p0=0.)\n")
    done = _fresh(code, JAXCONT_NO_AUTO_X64="1")
    assert done.stdout.strip() == "False"
    assert done.returncode != 0
    assert "RuntimeError" in done.stderr and "requires float64" in done.stderr


def test_float32_inputs_are_computed_in_float64():
    import jaxcont as jc
    f = lambda u, p, args: jnp.array([p + u[0] - u[0] ** 3 / 3])
    problem = jc.bif_problem(f, u0=jnp.array([-2.1], jnp.float32), p0=jnp.float32(-1.0))
    assert problem.u0.dtype == jnp.float64
    assert jnp.asarray(problem.p0).dtype == jnp.float64
    # A Newton tolerance of 1e-12 is reachable in float64 only.
    u, p, _, converged = jc.fold_point(f, jnp.array([-1.1], jnp.float32), jnp.float32(.6), tol=1e-12)
    assert bool(converged) and u.dtype == jnp.float64
    assert abs(float(p) - 2 / 3) < 1e-12  # the analytic fold is at p = 2/3


def test_float64_inputs_are_returned_unchanged():
    """Converting an array that is already float64 must keep it, weak type
    included; otherwise later calls with ordinary arrays recompile."""
    from jaxcont.utils.precision import as_float64
    weak = jnp.array(1.0)
    assert as_float64(weak) is weak


def test_complex_inputs_become_complex128():
    from jaxcont.utils.precision import as_float64
    tree = as_float64({"z": jnp.array([1 + 1j], jnp.complex64), "n": np.int32(3), "x": np.float32(1.5)})
    assert tree["z"].dtype == jnp.complex128 and tree["x"].dtype == jnp.float64
    assert tree["n"] == 3  # integers are left alone
