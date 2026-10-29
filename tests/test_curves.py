"""Two-parameter curve factories (bifurcations/curves.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.bifurcations.curves import fold_curve_problem, unpack_fold_curve


def _cusp_shifted(u, p, args):
    """u' = (p0-0.3) + (p1-1.2)*(u-0.7) - (u-0.7)**3.

    The cusp normal form, affinely shifted off the origin (origin-centred
    normal forms have no discriminating power -- a stub returning zeros
    passes them all). Its fold set is the exact discriminant
        27*(p0-0.3)**2 == 4*(p1-1.2)**3
    derived by eliminating u from f=0 and df/du=0.
    """
    x = u[0] - 0.7
    a = p[0] - 0.3
    b = p[1] - 1.2
    return jnp.array([a + b * x - x**3])


def _cusp_discriminant(p0, p1):
    return 27.0 * (p0 - 0.3) ** 2 - 4.0 * (p1 - 1.2) ** 3


def test_fold_curve_traces_the_exact_cusp_discriminant():
    # Seed: b = p1-1.2 = 3  ->  x = sqrt(b/3) = 1 -> u = 1.7,
    # a = p0-0.3 = -2*x*b/3 = -2 -> p0 = -1.7.
    prob = fold_curve_problem(
        _cusp_shifted,
        jnp.array([1.7]),
        jnp.array([-1.7, 4.2]), formulation="extended",
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 6.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _ = unpack_fold_curve(sol.branch.states[i], n=1)
        assert abs(_cusp_discriminant(float(p0), p1)) < 1e-2


def test_fold_curve_problem_rejects_wrong_p_shape():
    with pytest.raises(ValueError, match="shape"):
        fold_curve_problem(_cusp_shifted, jnp.array([1.7]), jnp.array([-1.7]), formulation="extended")


from jaxcont.bifurcations.curves import hopf_curve_problem, unpack_hopf_curve


def _hopf_parabola(u, p, args):
    """Hopf normal form with equilibrium shifted to (0.5, -0.3) and
    mu = p0 + p1**2 - 2, so the exact Hopf curve is the parabola
    p0 = 2 - p1**2 and the critical frequency is exactly omega = 1."""
    x = u[0] - 0.5
    y = u[1] + 0.3
    mu = p[0] + p[1] ** 2 - 2.0
    r2 = x * x + y * y
    return jnp.array([mu * x - y - x * r2, x + mu * y - y * r2])


def test_hopf_curve_traces_the_exact_parabola():
    prob = hopf_curve_problem(
        _hopf_parabola,
        jnp.array([0.5, -0.3]),
        jnp.array([2.0, 0.0]), formulation="extended",
        free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 1.2),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
    )
    assert sol.branch.params.shape[0] > 5
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _, _, omega = unpack_hopf_curve(sol.branch.states[i], n=2)
        assert abs(float(p0) + p1**2 - 2.0) < 1e-3
        assert abs(float(omega) - 1.0) < 1e-3


def test_fold_curve_continuation_under_jit():
    """Regression test: fold_curve continuation wrapped in jax.jit should not
    raise TracerBoolConversionError from the p_span[0] == problem.p0 guard.
    Exercises the traced code path that was broken by the original guard."""
    @jax.jit
    def run_continuation(u_guess, p_guess, p_span_start, p_span_end):
        prob = fold_curve_problem(
            _cusp_shifted,
            u_guess,
            p_guess, formulation="extended",
            free=1,
        )
        sol = jc.continuation(
            prob,
            p_span=(p_span_start, p_span_end),
            settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        )
        valid = sol.branch.valid
        residuals = jax.vmap(lambda u, p: jnp.linalg.norm(prob.f(u, p, prob.args)))(
            sol.branch.states, sol.branch.params,
        )
        return valid, jnp.max(jnp.where(valid, residuals, 0.0))

    # Call with concrete values to ensure p_span is traced
    valid, max_residual = run_continuation(
        jnp.array([1.7]),
        jnp.array([-1.7, 4.2]),
        4.2,
        5.2,
    )
    assert valid.sum() > 10
    assert max_residual < 1e-4


def test_hopf_curve_continuation_under_jit():
    """Regression test: hopf_curve continuation wrapped in jax.jit should not
    raise TracerBoolConversionError from the p_span[0] == problem.p0 guard."""
    @jax.jit
    def run_continuation(u_guess, p_guess, p_span_start, p_span_end):
        prob = hopf_curve_problem(
            _hopf_parabola,
            u_guess,
            p_guess, formulation="extended",
            free=1,
        )
        sol = jc.continuation(
            prob,
            p_span=(p_span_start, p_span_end),
            settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        )
        valid = sol.branch.valid
        residuals = jax.vmap(lambda u, p: jnp.linalg.norm(prob.f(u, p, prob.args)))(
            sol.branch.states, sol.branch.params,
        )
        return valid, jnp.max(jnp.where(valid, residuals, 0.0))

    # Call with concrete values to ensure p_span is traced
    valid, max_residual = run_continuation(
        jnp.array([0.5, -0.3]),
        jnp.array([2.0, 0.0]),
        0.0,
        0.5,
    )
    assert valid.sum() > 10
    assert max_residual < 1e-4


def test_two_parameter_surface_is_exported_top_level():
    for name in [
        "fold_curve_problem", "hopf_curve_problem",
        "Cusp", "BogdanovTakens", "ZeroHopf", "GeneralizedHopf", "DoubleHopf",
    ]:
        assert hasattr(jc, name), f"jc.{name} missing"
        assert name in jc.__all__, f"{name} missing from __all__"


def test_every_codim2_event_kind_has_a_plot_style():
    from jaxcont.viz.styles import BIFURCATION_STYLES
    for kind in [
        "cusp", "bogdanov_takens", "zero_hopf",
        "generalized_hopf", "double_hopf",
    ]:
        assert kind in BIFURCATION_STYLES, f"no style for {kind!r}"


# --------------------------------------------------------------------------
# Cross-validation against BifurcationKit.jl v0.5.2 (detector, not solver)
# --------------------------------------------------------------------------
#
# tests/test_codim2.py::test_bogdanov_takens_matches_bifurcationkit_jl_independent_run
# already asserts jaxcont's DIRECT SOLVER (bogdanov_takens_point) lands on
# BifurcationKit.jl's Newton-refined Lorenz-84 BT point when given a
# hand-supplied guess near it. This test asserts the DETECTOR
# (BogdanovTakens event) finds the same point by tracing the fold curve
# from a point BifurcationKit.jl located far away on the SAME curve --
# no guess anywhere near the BT is ever supplied.
#
# Preserve the original independent seed from the historical v1 bundle.
# Stop before the symmetry-related second BT point.
from tests.reference_bundles import historical_values

_fold_seed = historical_values("fold-seed")
U_SEED = jnp.array(_fold_seed["U_SEED"])
P_SEED = jnp.array(_fold_seed["P_SEED"])
P_SEED_FREE = _fold_seed["P_SEED_FREE"]
P_TARGET_FREE = -0.01


def test_lorenz84_bt_matches_bifurcationkit():
    """Detection along a traced fold curve must land on the BT point
    BifurcationKit.jl v0.5.2 finds independently (examples/validation/regenerate/bifurcationkit/
    lorenz84_codim2.jl). tests/test_codim2.py asserts the direct SOLVER matches
    these values; this asserts the DETECTOR finds them without a guess."""
    from tests.test_codim2 import BK_BT_P, BK_BT_U, _bk_model

    # Seed the fold curve at a fold point away from the BT, then continue
    # toward it. Exact seed values come from lorenz84_codim2.jl's printed output.
    prob = jc.fold_curve_problem(
        _bk_model, U_SEED, P_SEED, formulation="extended", free=1,
    )
    sol = jc.continuation(
        prob, p_span=(P_SEED_FREE, P_TARGET_FREE),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[jc.BogdanovTakens(raw_f=_bk_model, free=1, curve="fold")],
    )
    hits = [h for h in sol.events if h.kind == "bogdanov_takens"]
    assert len(hits) == 1
    assert hits[0].info["converged"] is True
    assert jnp.allclose(hits[0].u, jnp.array(BK_BT_U), atol=1e-3)
    assert jnp.allclose(hits[0].info["p"], jnp.array(BK_BT_P), atol=1e-3)


# --------------------------------------------------------------------------
# Review finding 2: events=[...] on a curve problem must not eigendecompose
# the extended-system Jacobian -- no codim-2 event reads Branch.eigenvalues.
# --------------------------------------------------------------------------


def test_fold_curve_events_do_not_eigendecompose_the_extended_jacobian():
    """Regression: passing events=[...] to a fold_curve problem must leave
    Branch.eigenvalues unset (None) -- codim-2 events carry their own
    raw_f and compute the ORIGINAL system's spectrum themselves, they
    never read Branch/BranchPoint.eigenvalues, so eigendecomposing the
    extended-system Jacobian here was pure waste. Detection must still
    work with eigenvalues left None."""
    prob = fold_curve_problem(
        _cusp_shifted, jnp.array([1.7]), jnp.array([-1.7, 4.2]), formulation="extended", free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(4.2, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[jc.Cusp(raw_f=_cusp_shifted, free=1)],
    )
    assert sol.branch.eigenvalues is None
    hits = [h for h in sol.events if h.kind == "cusp"]
    assert len(hits) == 1
    assert abs(hits[0].p - 1.2) < 5e-2


def test_hopf_curve_events_do_not_eigendecompose_the_extended_jacobian():
    """Same regression as above, for a hopf_curve problem."""
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf

    def _bautin(u, p, args):
        x = u[0] - 0.4
        y = u[1] + 0.6
        mu = p[0]
        b = p[1] - 0.4
        r2 = x * x + y * y
        return jnp.array([
            mu * x - y + b * x * r2 - x * r2 * r2,
            x + mu * y + b * y * r2 - y * r2 * r2,
        ])

    prob = hopf_curve_problem(
        _bautin, jnp.array([0.4, -0.6]), jnp.array([0.0, 0.0]), formulation="extended", free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 0.9),
        settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        events=[GeneralizedHopf(raw_f=_bautin, free=1)],
    )
    assert sol.branch.eigenvalues is None
    hits = [h for h in sol.events if h.kind == "generalized_hopf"]
    assert len(hits) == 1
    assert abs(hits[0].p - 0.4) < 5e-2


_FLIP_CENTRE = jnp.array([0.5, -0.3, 0.2])


def _flipping_hopf(u, p, args):
    """Hopf point at mu = p0 - p1 = 0 (exact curve p0 == p1, omega == 1).

    The oscillation plane turns by pi about the x axis as p0 goes from 0 to
    1, so the critical eigenvector at p0 >= 1 is the complex conjugate of
    the one at p0 = 0, i.e. orthogonal to it.
    """
    s = jnp.clip(p[0], 0.0, 1.0)
    a = jnp.pi * s**3 * (10.0 - 15.0 * s + 6.0 * s**2)
    c, sn = jnp.cos(a), jnp.sin(a)
    o, z0 = jnp.ones_like(c), jnp.zeros_like(c)
    R = jnp.array([[o, z0, z0], [z0, c, -sn], [z0, sn, c]])
    z = R.T @ (u - _FLIP_CENTRE)
    mu = p[0] - p[1]
    r2 = z[0] ** 2 + z[1] ** 2
    dz = jnp.array([mu * z[0] - z[1] - z[0] * r2, z[0] + mu * z[1] - z[1] * r2, -z[2]])
    return R @ dz


def test_hopf_curve_survives_eigenvector_rotation():
    prob = hopf_curve_problem(
        _flipping_hopf, _FLIP_CENTRE, jnp.array([0.0, 0.0]), formulation="extended", free=1,
    )
    sol = jc.continuation(
        prob,
        p_span=(0.0, 2.0),
        # The jump bound below measures Euclidean distance, so use that metric.
        settings=jc.ContinuationPar(theta=1.0, compute_stability=False, newton_tol=1e-5, max_steps=200),
    )
    assert sol.status == "reached_end"
    # Consecutive points stay within two ds_max (= 0.1) of each other: no
    # jump across a degenerate phase condition.
    steps = np.diff(np.column_stack([sol.branch.states, sol.branch.params]), axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) <= 0.2
    for i in range(sol.branch.params.shape[0]):
        p1 = float(sol.branch.params[i])
        _, p0, _, _, omega = unpack_hopf_curve(sol.branch.states[i], n=3)
        assert abs(float(p0) - p1) < 1e-3
        assert abs(float(omega) - 1.0) < 1e-3


def test_fold_curve_seed_grad_wrt_start_parameter():
    with jax.enable_x64():
        def f(u, p, args):
            return jnp.array([(u[0] - p[1]) ** 2 + p[0] - p[1]])

        def seed(q):
            return fold_curve_problem(
                f, jnp.array([0.6]), jnp.array([0.6, q]), formulation="extended",
            ).u0[0]

        q = jnp.array(0.7)
        derivative = jax.grad(seed)(q)
        h = 1e-5
        central = (seed(q + h) - seed(q - h)) / (2 * h)
        assert jnp.isfinite(derivative)
        assert jnp.allclose(derivative, central, rtol=1e-4, atol=1e-9)


def test_hopf_curve_seed_grad_wrt_start_parameter():
    with jax.enable_x64():
        def f(u, p, args):
            x, y = u - p[1]
            mu = p[0] - p[1]
            return jnp.array([mu * x - y, x + mu * y])

        def seed(q):
            return hopf_curve_problem(
                f, jnp.array([0.6, 0.6]), jnp.array([0.6, q]), formulation="extended",
            ).u0[0]

        q = jnp.array(0.7)
        derivative = jax.grad(seed)(q)
        h = 1e-5
        central = (seed(q + h) - seed(q - h)) / (2 * h)
        assert jnp.isfinite(derivative)
        assert jnp.allclose(derivative, central, rtol=1e-4, atol=1e-9)
