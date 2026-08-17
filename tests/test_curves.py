"""Two-parameter curve factories (bifurcations/curves.py)."""

import jax
import jax.numpy as jnp
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
        jnp.array([-1.7, 4.2]),
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
        fold_curve_problem(_cusp_shifted, jnp.array([1.7]), jnp.array([-1.7]))


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
        jnp.array([2.0, 0.0]),
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
            p_guess,
            free=1,
        )
        sol = jc.continuation(
            prob,
            p_span=(p_span_start, p_span_end),
            settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        )
        return sol.branch.n_valid

    # Call with concrete values to ensure p_span is traced
    n_valid = run_continuation(
        jnp.array([1.7]),
        jnp.array([-1.7, 4.2]),
        4.2,
        5.2,
    )
    assert n_valid > 0  # Continuation should produce at least one point


def test_hopf_curve_continuation_under_jit():
    """Regression test: hopf_curve continuation wrapped in jax.jit should not
    raise TracerBoolConversionError from the p_span[0] == problem.p0 guard."""
    @jax.jit
    def run_continuation(u_guess, p_guess, p_span_start, p_span_end):
        prob = hopf_curve_problem(
            _hopf_parabola,
            u_guess,
            p_guess,
            free=1,
        )
        sol = jc.continuation(
            prob,
            p_span=(p_span_start, p_span_end),
            settings=jc.ContinuationPar(compute_stability=False, newton_tol=1e-5),
        )
        return sol.branch.n_valid

    # Call with concrete values to ensure p_span is traced
    n_valid = run_continuation(
        jnp.array([0.5, -0.3]),
        jnp.array([2.0, 0.0]),
        0.0,
        0.5,
    )
    assert n_valid > 0  # Continuation should produce at least one point


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
# Seed values below are BifurcationKit.jl's own first fold-curve point
# (`sn_codim2.sol[1]`), printed by examples/BifurcationKit/05_codim2.jl:
#     first point x = [1.3117855253168738, -0.04287499818491644,
#                      0.18038939415099514, -0.15702758496684757,
#                      1.5466483725981275]     # (X, Y, Z, U, F)
#     first point p = 0.04                     # T
# The curve's augmented state there is (u, F) with T as the externally
# continued parameter -- exactly jaxcont's fold_curve_problem(..., free=1)
# convention, where p[0]=F is solved for (p_fixed) and p[1]=T is continued.
#
# P_TARGET_FREE is NOT BifurcationKit's full curve endpoint (T=-0.0735).
# examples/BifurcationKit/05_codim2.jl's specialpoint list shows TWO BT
# points on that curve (T=0.018390565929128334 and T=-0.023705593679238822,
# from `sp.param`) -- the Lorenz-84 model here is exactly symmetric under
# (U, T) -> (-U, -T) (U and T enter the RHS only as U**2 and as the U-row's
# additive +T, so the second BT is this model's own mirror image of the
# first, at u=(X,Y,Z,-U), p=(F,-T)). Continuing all the way to the
# endpoint makes the detector correctly report BOTH real BT points (2
# hits), which would defeat `len(hits) == 1` for reasons that have nothing
# to do with detector correctness. P_TARGET_FREE = -0.01 stops well short
# of the second BT (and of BifurcationKit's two ZH points at T ~= 0) while
# clearing the first BT at T = 0.0209 by a wide margin -- verified
# empirically to yield exactly one hit, landing on BK_BT_U/BK_BT_P.
U_SEED = jnp.array([
    1.3117855253168738, -0.04287499818491644,
    0.18038939415099514, -0.15702758496684757,
])
P_SEED = jnp.array([1.5466483725981275, 0.04])  # (F, T) at the seed
P_SEED_FREE = 0.04    # T at the seed (first curve point)
P_TARGET_FREE = -0.01  # T target; see margin note above


def test_lorenz84_bt_matches_bifurcationkit():
    """Detection along a traced fold curve must land on the BT point
    BifurcationKit.jl v0.5.2 finds independently (examples/BifurcationKit/
    05_codim2.jl). tests/test_codim2.py asserts the direct SOLVER matches
    these values; this asserts the DETECTOR finds them without a guess."""
    from tests.test_codim2 import _bk_model, BK_BT_U, BK_BT_P

    # Seed the fold curve at a fold point away from the BT, then continue
    # toward it. Exact seed values come from 05_codim2.jl's printed output.
    prob = jc.fold_curve_problem(
        _bk_model, U_SEED, P_SEED, free=1,
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
