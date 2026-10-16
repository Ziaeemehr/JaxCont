"""Regressions for the loop-1 numerical review rulings."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcont as jc
from jaxcont.bifurcations.events import (
    BP,
    BranchPoint,
    Hopf,
    NeimarkSacker,
    PeriodDoubling,
    detect_events,
)
from jaxcont.bifurcations.modes import crossings
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic_args import PeriodicArgs


def test_bp_rejects_coarsely_bracketed_scalar_fold():
    hit = BP().refine(
        BranchPoint(-4., jnp.array([-2.])), BranchPoint(-4., jnp.array([2.])),
        (0, 1), lambda u, p: u**2 + p, tolerance=None, max_iterations=50,
    )
    assert hit is None


@pytest.mark.parametrize("values, expected", [
    ([-1e-30, 1e-30], [0]), ([-np.inf, 1.], []),
    ([np.nan, 0.], []), ([0., np.inf], []),
])
def test_crossings_scale_and_finiteness(values, expected):
    assert crossings(np.array(values, dtype=np.float32), np.ones(2, bool)) == expected


def test_distinct_hopf_modes_in_one_bracket():
    def rhs(u, p):
        x, y, z, w = u
        return jnp.array([
            (p + .02) * x - y, x + (p + .02) * y,
            (p - .02) * z - 2 * w, 2 * z + (p - .02) * w,
        ])

    params = jnp.array([-.1, .1])
    states = jnp.zeros((2, 4))
    spectra = jnp.array([
        [p + .02 + 1j, p + .02 - 1j, p - .02 + 2j, p - .02 - 2j]
        for p in [-.1, .1]
    ])
    hits = detect_events([Hopf()], params, states, None, spectra, rhs, ds=.2)
    good = sorted(h.p for h in hits if h.status == "validated")
    assert good == pytest.approx([-.02, .02], abs=2e-5)


def test_bothside_shared_seed_hopf_has_one_owner():
    def rhs(u, p, args):
        x, y = u
        return jnp.array([p * x - y, x + p * y])

    result = jc.continuation(
        jc.bif_problem(rhs, u0=jnp.zeros(2), p0=0.), p_span=(-.1, .1),
        settings=jc.ContinuationPar(bothside=True, ds=.02, max_steps=20),
        events=[Hopf()],
    )
    assert len(result.events) == 1
    assert result.events[0].p == pytest.approx(0., abs=2e-5)


@pytest.mark.parametrize("engine", [jc.Natural(), jc.PseudoArclength()])
def test_nonfinite_trial_status(engine):
    def rhs(u, p, args):
        return jnp.where(p > 0, jnp.full_like(u, jnp.nan), u)

    result = jc.continuation(
        jc.bif_problem(rhs, u0=jnp.zeros(1), p0=0.), engine, p_span=(0., .1),
        settings=jc.ContinuationPar(ds=.02, max_steps=10, compute_stability=False),
    )
    assert result.status == "nonfinite"
    assert bool(jnp.all(jnp.isfinite(result.branch.states)))


@pytest.mark.parametrize("event", [PeriodDoubling(), NeimarkSacker()])
def test_floquet_refine_requires_periodic_args(event):
    left = BranchPoint(
        -1., jnp.zeros(1), eigenvalues=jnp.array([1., -0.5]), trivial_index=0,
    )
    right = BranchPoint(
        2., jnp.zeros(1), eigenvalues=jnp.array([1., -2.]), trivial_index=0,
    )
    with pytest.raises(ValueError, match="PeriodicArgs"):
        event.refine(left, right, (0, 1), lambda u, p: u, tolerance=None, max_iterations=1)


@pytest.mark.parametrize("event", [PeriodDoubling(), NeimarkSacker()])
def test_floquet_refine_rejects_exhausted_bracket(event, monkeypatch):
    def spectrum(u, p, args):
        critical = -np.exp(p) if event.kind == "period_doubling" else np.exp(p + 0.7j)
        return np.array([1., critical]), 0

    monkeypatch.setattr(type(event), "_multipliers", staticmethod(spectrum))
    args = PeriodicArgs(
        jnp.zeros((1, 1, 1)), jnp.zeros((1, 1, 1)), None,
        lambda u, p, a: u, Collocation(ntst=1, ncol=1),
    )

    def pt(p):
        return BranchPoint(
            p, jnp.zeros(1), eigenvalues=jnp.asarray(spectrum(None, p, None)[0]),
            trivial_index=0,
        )

    hit = event.refine(
        pt(-1.), pt(2.), (0, 1), lambda u, p: u,
        tolerance=None, max_iterations=1, args=args,
    )
    assert hit.status == "failed"
    assert hit.info["reason"] == "not_converged"
