"""
Tests for jaxcont.stability.floquet: Floquet multipliers via the collocation
monodromy matrix (core/collocation.py's monodromy_matrix), for the periodic
orbit r' = r*(rho - r^2), theta' = 1, which has an exact closed-form limit
cycle x=cos(t), y=sin(t), T=2*pi at rho=1, with exact Floquet multipliers
{1, exp(-4*pi)}. See
docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.floquet import (
    PeriodicSchur,
    ProductFloquet,
    branch_floquet_multipliers,
    floquet_multipliers,
    floquet_spectrum,
    floquet_stable,
)


def _rhs(u, p, args):
    x, y = u[0], u[1]
    r2 = x * x + y * y
    rho = p
    return jnp.array([(rho - r2) * x - y, (rho - r2) * y + x])


def _coarse_wrong_trajectory():
    rng = np.random.default_rng(0)
    t_traj = np.sort(rng.uniform(0, 5.5, size=40))
    t_traj[0] = 0.0
    theta = lambda t: 2 * np.pi * t / 5.5 + 0.3
    u_traj = np.stack(
        [0.8 * np.cos(theta(t_traj)), 0.8 * np.sin(theta(t_traj))], axis=1
    )
    return jnp.asarray(u_traj), jnp.asarray(t_traj)


def _circle_problem(rho=1.0):
    u_traj, t_traj = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=10, ncol=4)
    prob = periodic_orbit_problem(_rhs, u_traj, t_traj, 5.5, rho, mesh)
    return prob, mesh


def test_monodromy_matrix_matches_interval_propagators_scan():
    """interval_propagators must be extractable from monodromy_matrix without
    changing monodromy_matrix's own numerics -- see
    docs/superpowers/specs/2026-08-05-prc-dprc-design.md."""
    from jaxcont.core.collocation import (
        collocation_matrices,
        interval_propagators,
        monodromy_matrix,
    )

    prob, mesh = _circle_problem(rho=1.0)
    ntst, ncol = mesh.ntst, mesh.ncol
    n = 2
    h = 1.0 / ntst
    D_np, E_np, _, _ = collocation_matrices(ncol)
    D, E = jnp.asarray(D_np), jnp.asarray(E_np)
    mesh_states = prob.u0[: ntst * n].reshape(ntst, n)
    coll_states = prob.u0[ntst * n : ntst * n + ntst * ncol * n].reshape(ntst, ncol, n)
    T = prob.u0[-1]

    M_all = interval_propagators(_rhs, D, E, h, mesh_states, coll_states, T, prob.p0)
    assert M_all.shape == (ntst, n, n)

    Phi_from_blocks, _ = jax.lax.scan(
        lambda carry, M: (M @ carry, None), jnp.eye(n), M_all
    )
    Phi_direct = monodromy_matrix(_rhs, D, E, h, mesh_states, coll_states, T, prob.p0)
    assert jnp.max(jnp.abs(Phi_from_blocks - Phi_direct)) < 1e-6


def test_floquet_multipliers_matches_closed_form_at_rho_1():
    prob, mesh = _circle_problem(rho=1.0)
    multipliers, trivial_idx = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    got = jnp.sort(jnp.abs(multipliers))
    expected = jnp.sort(jnp.array([np.exp(-4 * np.pi), 1.0]))
    if got.dtype == jnp.float64:
        assert float(jnp.max(jnp.abs(jnp.log(got) - jnp.log(expected)))) < 1e-3
    else:
        # A multiplier of size ~3.5e-6 carries ~eps*||Phi||/|mu| relative error in float32.
        assert jnp.max(jnp.abs(got - expected)) < (
            1e3 * jnp.finfo(jnp.float32).eps * jnp.max(jnp.abs(expected))
        )


def test_floquet_stable_true_at_rho_1_with_correct_trivial_multiplier():
    prob, mesh = _circle_problem(rho=1.0)
    multipliers, trivial_idx = floquet_multipliers(_rhs, mesh, prob.u0, prob.p0)
    stable = floquet_stable(multipliers[None, :], trivial_idx[None])
    assert bool(stable[0]) is True

    assert abs(float(multipliers[trivial_idx].real) - 1.0) < 1e-4


def test_branch_floquet_multipliers_matches_per_point_calls():
    prob1, mesh = _circle_problem(rho=1.0)
    prob2, _ = _circle_problem(rho=1.5)
    states = jnp.stack([prob1.u0, prob2.u0])
    params = jnp.stack([prob1.p0, prob2.p0])

    batched, indices = branch_floquet_multipliers(_rhs, mesh, states, params)
    individual = jnp.stack(
        [
            floquet_multipliers(_rhs, mesh, prob1.u0, prob1.p0)[0],
            floquet_multipliers(_rhs, mesh, prob2.u0, prob2.p0)[0],
        ]
    )
    assert batched.shape == (2, 2)
    assert jnp.array_equal(indices, jnp.stack([
        floquet_multipliers(_rhs, mesh, prob1.u0, prob1.p0)[1],
        floquet_multipliers(_rhs, mesh, prob2.u0, prob2.p0)[1],
    ]))
    assert jnp.allclose(jnp.sort(jnp.abs(batched), axis=1), jnp.sort(jnp.abs(individual), axis=1), atol=1e-4)


def test_trivial_multiplier_by_eigenvector(monkeypatch):
    import importlib
    module = importlib.import_module('jaxcont.stability.floquet')
    monkeypatch.setattr(module, 'monodromy_matrix',
                        lambda *a: jnp.diag(jnp.array([0.9999, 1.00001])))
    mesh = Collocation(ntst=1, ncol=1)
    U = jnp.array([1., 0., 1., 0., 1.])
    multipliers, trivial_index = floquet_multipliers(
        lambda u, p, args: jnp.array([1., 0.]), mesh, U, jnp.array(0.), ProductFloquet())
    assert int(trivial_index) == 0
    assert not bool(floquet_stable(multipliers[None, :], trivial_index[None])[0])


def test_flow_index_respects_custom_solver_order_under_jit(monkeypatch):
    import importlib
    module = importlib.import_module('jaxcont.stability.floquet')
    monkeypatch.setattr(module, 'monodromy_matrix',
                        lambda *a: jnp.diag(jnp.array([0.9999, 1.00001])))
    mesh = Collocation(ntst=1, ncol=1)
    def at(U):
        return floquet_multipliers(
            lambda u, p, args: jnp.array([1., 0.]), mesh, U, jnp.array(0.),
            ProductFloquet(eigen_solver=lambda phi: jnp.linalg.eigvals(phi)[::-1]))
    values, index = jax.jit(at)(jnp.array([1., 0., 1., 0., 1.]))
    assert int(index) == 1
    assert not bool(floquet_stable(values[None, :], index[None])[0])


def test_floquet_multiplier_derivatives_keep_eigensolver_path(monkeypatch):
    import importlib
    module = importlib.import_module('jaxcont.stability.floquet')
    monkeypatch.setattr(module, 'monodromy_matrix',
                        lambda *a: jnp.diag(jnp.array([0.9999, a[-2]])))
    mesh = Collocation(ntst=1, ncol=1)
    def transverse(p):
        values, index = floquet_multipliers(
            lambda u, p, args: jnp.array([1., 0.]), mesh,
            jnp.array([1., 0., 1., 0., 1.]), p, ProductFloquet())
        return jnp.real(values[1 - index])
    assert jnp.allclose(jax.grad(transverse)(jnp.array(1.00001)), 1.)


# --- periodic Schur strategy (docs/superpowers/specs/2026-10-17-periodic-schur-floquet-design.md)


def _eps():
    return float(jnp.finfo(jnp.asarray(0.0).dtype).eps)


def _circle_on_mesh(ntst, ncol):
    u_traj, t_traj = _coarse_wrong_trajectory()
    mesh = Collocation(ntst=ntst, ncol=ncol)
    return periodic_orbit_problem(_rhs, u_traj, t_traj, 5.5, 1.0, mesh), mesh


def test_circle_log_multipliers_mesh_convergence():
    expected = np.array([-12.566370614359172, 0.0])
    errors = []
    for ntst, ncol in ((20, 4), (40, 4)):
        prob, mesh = _circle_on_mesh(ntst, ncol)
        spectrum = floquet_spectrum(_rhs, mesh, prob.u0, prob.p0, PeriodicSchur())
        assert bool(spectrum.converged)
        logabs = np.sort(np.asarray(spectrum.logabs))
        errors.append(float(np.max(np.abs(logabs - expected))))
        # The trivial mode is the flow direction, not the smallest |log|.
        assert abs(float(spectrum.logabs[spectrum.trivial_index])) < 1e-3
    coarse, fine = errors
    assert fine < 2e-3
    assert fine <= coarse + 500 * _eps()


def test_periodic_schur_flow_index_with_near_unit_physical_mode(monkeypatch):
    import importlib
    module = importlib.import_module('jaxcont.stability.floquet')
    # Four diagonal factors whose cyclic product is diag(.9999, 1.00001).
    root = jnp.array([0.9999, 1.00001]) ** 0.25
    monkeypatch.setattr(module, 'interval_propagators',
                        lambda *a: jnp.stack([jnp.diag(root)] * 4))
    mesh = Collocation(ntst=1, ncol=1)
    U = jnp.array([1., 0., 1., 0., 1.])
    flow = lambda u, p, args: jnp.array([1., 0.])
    for run in (lambda U: floquet_multipliers(flow, mesh, U, jnp.array(0.), PeriodicSchur()),
                jax.jit(lambda U: floquet_multipliers(flow, mesh, U, jnp.array(0.), PeriodicSchur()))):
        values, index = run(U)
        assert int(index) == 0
        np.testing.assert_allclose(np.abs(values), [0.9999, 1.00001], rtol=10 * _eps())
        assert not bool(floquet_stable(values[None, :], index[None])[0])
    spectrum = floquet_spectrum(flow, mesh, U, jnp.array(0.), PeriodicSchur())
    assert not bool(floquet_stable(spectrum.values[None], spectrum.trivial_index[None],
                                   logabs=spectrum.logabs[None])[0])
    # A zero flow identifies no trivial mode.
    zero = lambda u, p, args: jnp.zeros(2)
    values, index = jax.jit(lambda U: floquet_multipliers(zero, mesh, U, jnp.array(0.),
                                                          PeriodicSchur()))(U)
    assert int(index) == -1 and np.all(np.isnan(np.asarray(values)))


# Circle (+) transverse rotation blocks: a PD block (multiplier -exp(alpha*T),
# double) and an NS block (multipliers exp(alpha*T +/- 0.3j*T)) both cross the
# unit circle at alpha = p - shift = 0.
_T = 2 * np.pi
_BETA_PD = np.pi / _T
_BETA_NS = 0.3


def _transverse_rhs(blocks):
    def rhs(u, p, args):
        alpha = p - args
        x, y = u[0], u[1]
        r2 = x * x + y * y
        out = [(1.0 - r2) * x - y, (1.0 - r2) * y + x]
        for k, beta in enumerate(blocks):
            w1, w2 = u[2 + 2 * k], u[3 + 2 * k]
            out += [alpha * w1 - beta * w2, beta * w1 + alpha * w2]
        return jnp.stack(out)
    return rhs


def _transverse_problem(blocks, p0, shift=0.0):
    t = np.linspace(0, _T, 60, endpoint=False)
    zeros = [np.zeros_like(t)] * (2 * len(blocks))
    u = np.stack([np.cos(t), np.sin(t), *zeros], axis=1)
    mesh = Collocation(ntst=10, ncol=4)
    rhs = _transverse_rhs(blocks)
    prob = periodic_orbit_problem(rhs, jnp.asarray(u), jnp.asarray(t), _T, p0, mesh,
                                  args=jnp.asarray(shift, jnp.asarray(0.).dtype))
    return prob, mesh, rhs


def _fixed_steps(max_steps=5, **kw):
    import jaxcont as jc
    return jc.ContinuationPar(ds=0.02, ds_max=0.02, adaptive=False, max_steps=max_steps, **kw)


def test_periodic_schur_branch_jit_vmap_and_pd_ns():
    import equinox as eqx
    import jaxcont as jc
    prob, mesh, rhs = _transverse_problem((_BETA_PD,), -0.05)
    solvers = jc.Solvers(floquet=PeriodicSchur())

    def run(shift):
        lane = prob.at(p0=shift - 0.05, u0=prob.u0,
                       args=eqx.tree_at(lambda a: a.user_args, prob.args, shift))
        return jc.continuation(lane, jc.Natural(), p_span=(shift - 0.05, shift + 0.05),
                               events=(jc.PeriodDoubling(),), solvers=solvers,
                               settings=_fixed_steps(max_steps=7))

    shifts = jnp.array([-0.01, 0.01])
    result = jax.jit(jax.vmap(run))(shifts)
    branch = result.branch
    B = branch.params.shape[1]
    assert branch.eigenvalues.shape == (2, B, 4)
    assert branch.floquet_logabs.shape == (2, B, 4)
    assert branch.floquet_converged.shape == (2, B)
    assert branch.trivial_index.shape == branch.stable.shape == (2, B)
    valid = np.asarray(branch.valid)
    assert valid.sum(axis=1).min() >= 6 and not valid.all()
    np.testing.assert_array_equal(np.asarray(branch.floquet_converged), valid)
    assert np.all(np.asarray(branch.trivial_index)[~valid] == -1)
    # Both members of the double PD multiplier cross in one bracket.
    mask = np.asarray(result.event_candidates[0].bracket_mask)
    np.testing.assert_array_equal(mask.sum(axis=(1, 2)), [2, 2])
    np.testing.assert_array_equal(mask.any(axis=2).sum(axis=1), [1, 1])
    for k, shift in enumerate(shifts):
        single = jax.tree_util.tree_map(lambda x: x[k], result)
        lane = prob.at(args=eqx.tree_at(lambda a: a.user_args, prob.args, shift))
        refined = jc.refine_event_candidates(single, lane, (jc.PeriodDoubling(),),
                                             solvers=solvers)
        assert len(refined.events) == 1
        assert abs(refined.events[0].p - float(shift)) < 1e-4

    ns_prob, _, _ = _transverse_problem((_BETA_NS,), -0.05)
    ns = jc.continuation(ns_prob, p_span=(-0.05, 0.05), events=[jc.NeimarkSacker()],
                         solvers=solvers, settings=jc.ContinuationPar(ds=0.02, max_steps=50))
    assert [e.kind for e in ns.events] == ["neimark_sacker"]
    assert abs(ns.events[0].p) < 1e-4
    assert ns.branch.floquet_logabs.shape == ns.branch.eigenvalues.shape
    assert bool(np.all(np.asarray(ns.branch.floquet_converged)))


class _FlowLast:
    """Product eigensolver ordering by decreasing distance from 1."""

    def __call__(self, matrix):
        values = jnp.linalg.eigvals(matrix)
        return values[jnp.argsort(-jnp.abs(values - 1))]


def _instrument(monkeypatch):
    """Record branch/point spectrum strategies and raw scanned spectra."""
    import importlib
    floquet = importlib.import_module('jaxcont.stability.floquet')
    scan = importlib.import_module('jaxcont.bifurcations.event_scan')
    from jaxcont.bifurcations.events import NeimarkSacker, PeriodDoubling
    log = {"branch": [], "point": [], "scanned": [], "kind": None}
    branch, point, track = floquet._branch_spectrum, floquet._eager_point_spectrum, scan.track_modes_array

    def counted_branch(strategy, *a, **k):
        log["branch"].append(strategy)
        return branch(strategy, *a, **k)

    def counted_point(strategy, *a, **k):
        log["point"].append((log["kind"], strategy))
        return point(strategy, *a, **k)

    def recorded_track(raw, valid):
        log["scanned"].append(np.asarray(raw))
        return track(raw, valid)

    def tagged(cls):
        refine = cls.refine
        def wrapper(self, *a, **k):
            log["kind"] = self.kind
            return refine(self, *a, **k)
        monkeypatch.setattr(cls, 'refine', wrapper)

    monkeypatch.setattr(floquet, '_branch_spectrum', counted_branch)
    monkeypatch.setattr(floquet, '_eager_point_spectrum', counted_point)
    monkeypatch.setattr(scan, 'track_modes_array', recorded_track)
    tagged(PeriodDoubling)
    tagged(NeimarkSacker)
    return log


def test_floquet_event_override_detection_and_refinement(monkeypatch):
    import jaxcont as jc
    prob, mesh, rhs = _transverse_problem((_BETA_PD, _BETA_NS), -0.05)
    product, schur = ProductFloquet(eigen_solver=_FlowLast()), PeriodicSchur()
    settings = _fixed_steps()
    for bundle, override in ((product, schur), (schur, product)):
        log = _instrument(monkeypatch)
        events = (jc.PeriodDoubling(floquet_solver=override), jc.NeimarkSacker(),
                  jc.NeimarkSacker(floquet_solver=bundle))
        result = jc.continuation(prob, jc.Natural(), p_span=(-0.05, 0.05), events=events,
                                 solvers=jc.Solvers(floquet=bundle), settings=settings)
        branch = result.branch
        log["kind"] = None
        # Shared effective strategies are computed once per point.
        assert log["branch"] == [bundle, override]
        own = {s: jax.jit(lambda U, p, s=s: jax.vmap(lambda u, q: floquet_spectrum(
                   rhs, mesh, u, q, s, args=prob.args.user_args))(U, p))(branch.states, branch.params)
               for s in (bundle, override)}
        trivial = {s: np.asarray(own[s].trivial_index) for s in own}
        assert np.all(trivial[product] == 5)
        assert np.all(trivial[schur] != 5)
        # Branch reporting follows the bundle strategy.
        np.testing.assert_array_equal(branch.trivial_index, trivial[bundle])
        np.testing.assert_allclose(branch.eigenvalues,
                                   np.asarray(own[bundle].values), atol=1e3 * _eps())
        # Scan inputs: PD sees the override spectrum, both NS see the bundle's.
        assert len(log["scanned"]) == 3
        for raw, strategy in zip(log["scanned"], (override, bundle, bundle)):
            np.testing.assert_allclose(raw, np.asarray(own[strategy].values),
                                       atol=1e3 * _eps())
        for candidate, strategy in zip(result.event_candidates, (override, bundle, bundle)):
            assert int(np.asarray(candidate.bracket_mask).sum()) in (1, 2)
            raw = np.asarray(own[strategy].values)
            perm = np.asarray(candidate.permutation)
            np.testing.assert_allclose(candidate.spectra, np.take_along_axis(raw, perm, 1),
                                       atol=1e3 * _eps())
            # Raw-column exclusion uses the event strategy's trivial index.
            flow = perm == trivial[strategy][:, None]
            assert not np.asarray(candidate.eligible)[flow].any()
        kinds = sorted(e.kind for e in result.events)
        assert kinds == ["neimark_sacker", "period_doubling"]
        assert all(abs(e.p) < 1e-4 for e in result.events)
        # Refinement uses the identical effective event strategy.
        pd_calls = {id(s) for k, s in log["point"] if k == "period_doubling"}
        ns_calls = {id(s) for k, s in log["point"] if k == "neimark_sacker"}
        assert pd_calls == {id(override)} and ns_calls == {id(bundle)}
        monkeypatch.undo()

    # A failed strategy masks only the events that use it.
    failing = PeriodicSchur(max_sweeps=0)
    for bundle, override, alive in ((product, failing, "neimark_sacker"),
                                    (failing, product, "period_doubling")):
        result = jc.continuation(prob, jc.Natural(), p_span=(-0.05, 0.05),
                                 events=(jc.PeriodDoubling(floquet_solver=override),
                                         jc.NeimarkSacker()),
                                 solvers=jc.Solvers(floquet=bundle), settings=settings)
        assert [e.kind for e in result.events] == [alive]
        dead = 0 if alive == "neimark_sacker" else 1
        assert not np.asarray(result.event_candidates[dead].bracket_mask).any()
        converged = np.asarray(result.branch.floquet_converged)
        if bundle is failing:
            assert not converged.any() and not np.asarray(result.branch.stable).any()
            assert np.all(np.asarray(result.branch.trivial_index) == -1)
        else:
            assert converged.all()


def test_failed_schur_masks_events_and_padding(monkeypatch):
    import importlib
    import jaxcont as jc
    floquet = importlib.import_module('jaxcont.stability.floquet')
    prob, mesh, rhs = _transverse_problem((_BETA_PD,), -0.05)
    U, p = prob.u0, prob.p0
    failing = PeriodicSchur(max_sweeps=0)
    try:
        floquet_multipliers(rhs, mesh, U, p, failing, args=prob.args.user_args)
    except RuntimeError as error:
        assert "converge" in str(error)
    else:
        raise AssertionError("eager nonconvergence must raise")
    values, index = jax.jit(lambda U: floquet_multipliers(
        rhs, mesh, U, p, failing, args=prob.args.user_args))(U)
    assert int(index) == -1 and np.all(np.isnan(np.asarray(values)))

    periods = []
    branch_spectrum = floquet._branch_spectrum
    def recorded(strategy, raw_f, mesh, states, params, args):
        jax.debug.callback(lambda T: periods.append(np.asarray(T)), states[:, -1])
        return branch_spectrum(strategy, raw_f, mesh, states, params, args)
    monkeypatch.setattr(floquet, '_branch_spectrum', recorded)
    for solvers in (jc.Solvers(floquet=failing), jc.Solvers(floquet=ProductFloquet())):
        periods.clear()
        result = jax.jit(lambda q: jc.continuation(
            q, jc.Natural(), p_span=(-0.05, 0.05), events=(jc.PeriodDoubling(),),
            solvers=solvers, settings=_fixed_steps(max_steps=8)))(prob)
        jax.block_until_ready(result)
        branch = result.branch
        valid = np.asarray(branch.valid)
        assert not valid.all()
        # Padding is evaluated at a valid seed, never as a collapsed orbit.
        assert periods and all(np.all(np.asarray(T) > 1) for T in periods)
        assert not np.asarray(branch.stable)[~valid].any()
        assert not np.asarray(branch.floquet_converged)[~valid].any()
        if solvers.floquet is failing:
            assert not np.asarray(branch.floquet_converged).any()
            assert not np.asarray(branch.stable).any()
            assert np.all(np.asarray(branch.trivial_index) == -1)
            assert np.all(np.isnan(np.asarray(branch.eigenvalues)[valid]))
            assert not np.asarray(result.event_candidates[0].bracket_mask).any()
            assert not np.asarray(result.event_candidates[0].eligible).any()
        else:
            assert np.asarray(result.event_candidates[0].bracket_mask).sum() >= 1
    eager = jc.continuation(prob, jc.Natural(), p_span=(-0.05, 0.05),
                            events=(jc.PeriodDoubling(),), solvers=jc.Solvers(floquet=failing),
                            settings=_fixed_steps())
    assert eager.events == [] and eager.rejected_events == []
    assert not np.asarray(eager.branch.stable).any()


def test_repeated_unit_flow_does_not_select_overlapping_physical_mode(monkeypatch):
    """Unavailable unit modes must not make (1,0,1) the e0 flow mode."""
    import importlib
    import pytest
    module = importlib.import_module('jaxcont.stability.floquet')
    # Unit eigenspace span(e0,e1); the .5 eigenvector is (1,0,1).
    factor = jnp.array([[1., 0., -.5], [0., 1., 0.], [0., 0., .5]])
    monkeypatch.setattr(module, 'interval_propagators', lambda *a: factor[None])
    reconstruct = module.periodic_spectrum_modes

    def unavailable_unit_modes(*a, **kw):
        phase, logabs, converged, right = reconstruct(*a, **kw)
        # Model unavailable reconstructed columns, retaining the real spectrum
        # and the finite nonnormal physical mode from the Schur computation.
        return phase, logabs, converged, right.at[:, :2].set(jnp.nan)

    monkeypatch.setattr(module, 'periodic_spectrum_modes', unavailable_unit_modes)
    mesh = Collocation(ntst=1, ncol=1)
    U = jnp.array([1., 0., 0., 1., 0., 0., 1.])
    flow = lambda u, p, args: jnp.array([1., 0., 0.])
    evaluate = lambda U: floquet_spectrum(flow, mesh, U, jnp.array(0.), PeriodicSchur())
    spectrum = jax.jit(evaluate)(U)
    assert not bool(spectrum.converged)
    assert int(spectrum.trivial_index) == -1
    assert np.all(np.isnan(np.asarray(spectrum.values)))
    assert not bool(floquet_stable(spectrum.values, spectrum.trivial_index,
                                   logabs=spectrum.logabs))
    with pytest.raises(RuntimeError, match='did not converge'):
        evaluate(U)


def test_flow_alignment_requires_direction_not_partial_overlap():
    from jaxcont.stability.floquet import _aligned
    vectors = jnp.array([[jnp.nan, jnp.nan, 1.],
                         [jnp.nan, jnp.nan, 0.],
                         [jnp.nan, jnp.nan, 1.]], dtype=complex)
    flow = jnp.array([1., 0., 0.])
    for run in (_aligned, jax.jit(_aligned)):
        _, valid = run(vectors, flow)
        assert not bool(valid)
        # Arbitrary complex phase and column scaling preserve identity.
        index, valid = run(jnp.diag(jnp.array([2j, 3., 4.])), flow)
        assert bool(valid) and int(index) == 0
