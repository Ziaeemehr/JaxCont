"""Array-only mode assignment and crossing ownership."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcont.bifurcations.array_modes import crossing_mask, track_modes_array
from jaxcont.bifurcations.modes import track_modes


@pytest.mark.parametrize("compiled", [False, True])
def test_array_assignment_matches_host_predictor(compiled):
    run = jax.jit(track_modes_array) if compiled else track_modes_array
    p = jnp.linspace(-1, 1, 21)
    modes = jnp.stack((p, -p, .2 + .1*p + 2j, -.2 + .1*p + 3j), axis=1)
    rng = np.random.default_rng(61)
    raw = jnp.asarray(np.stack([row[rng.permutation(4)] for row in np.asarray(modes)]))
    tracked, permutation = run(raw, jnp.ones(21, dtype=bool))
    np.testing.assert_allclose(tracked, track_modes(np.asarray(raw)))
    np.testing.assert_array_equal(tracked, jnp.take_along_axis(raw, permutation, axis=1))
    assert jnp.issubdtype(permutation.dtype, jnp.integer)
    tied, identity = run(jnp.zeros((3, 4), dtype=modes.dtype), jnp.ones(3, bool))
    np.testing.assert_array_equal(identity, np.tile(np.arange(4), (3, 1)))
    np.testing.assert_array_equal(tied, 0)
    # Invalid rows cannot perturb the two-valid-sample predictor.
    padded = jnp.concatenate((raw[:8], jnp.full((2, 4), jnp.nan), raw[8:]))
    valid = jnp.concatenate((jnp.ones(8, bool), jnp.zeros(2, bool), jnp.ones(13, bool)))
    result, _ = run(padded, valid)
    np.testing.assert_allclose(result[valid], tracked)
    # Batched random branches exercise augmenting paths beyond greedy matching.
    batch = jnp.asarray(rng.normal(size=(6, 5, 4)) + 1j*rng.normal(size=(6, 5, 4)))
    batched, _ = jax.jit(jax.vmap(track_modes_array, in_axes=(0, None)))(
        batch, jnp.ones(5, bool))
    for raw_branch, tracked_branch in zip(np.asarray(batch), np.asarray(batched)):
        np.testing.assert_allclose(tracked_branch, track_modes(raw_branch))
    for size in (0, 1):
        result, perm = run(raw[:size], jnp.ones(size, bool))
        assert result.shape == perm.shape == (size, 4)
    nonfinite = jnp.array([[1, jnp.inf], [jnp.nan, 2]], dtype=modes.dtype)
    result, perm = run(nonfinite, jnp.ones(2, bool))
    np.testing.assert_allclose(result, track_modes(np.asarray(nonfinite)), equal_nan=True)


@pytest.mark.parametrize("compiled", [False, True])
def test_crossing_masks_zeros_padding_nonfinite(compiled):
    run = jax.jit(crossing_mask) if compiled else crossing_mask
    cases = [
        ([-1, 0, 1, 0, np.nan], [1, 1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 0, 0], [1, 1, 1], [0, 0]),
        ([0, 1], [1, 1], [1]),
        ([-1, 1], [1, 1], [1]),
        ([np.inf, -1], [1, 1], [0]),
        ([1], [1], []),
        ([], [], []),
    ]
    for values, valid, expected in cases:
        values = jnp.asarray(values, dtype=float)[:, None]
        mask = run(values, jnp.ones(values.shape, bool), jnp.array(valid, bool))
        np.testing.assert_array_equal(mask, np.asarray(expected, bool).reshape(-1, 1))
    # An ineligible predecessor cannot own the zero; the following bracket can.
    values = jnp.array([[-1., 1.], [0., 0.], [1., -1.]])
    eligible = jnp.array([[False, True], [True, True], [True, True]])
    np.testing.assert_array_equal(run(values, eligible, jnp.ones(3, bool)),
                                  [[False, True], [True, False]])


def test_scan_preserves_raw_trivial_identity():
    from jaxcont.bifurcations.event_scan import nontrivial_mask
    raw = jnp.array([[1., 1., 1.1]])
    permutation = jnp.array([[2, 0, 1]])
    tracked = jnp.take_along_axis(raw, permutation, axis=1)
    np.testing.assert_array_equal(tracked, jnp.array([[1.1, 1., 1.]]))
    mask = jax.jit(nontrivial_mask)(permutation, jnp.array([1]))
    np.testing.assert_array_equal(mask, [[True, True, False]])


def test_array_scan_two_modes_one_bracket():
    from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
    from jaxcont.bifurcations.events import PeriodDoubling
    data = EventScanInput(jnp.array([0., 1.]), jnp.zeros((2, 1)), None,
                          jnp.array([[-.8, -.9, 1.], [-1.2, -1.1, 1.]]),
                          jnp.array([2, 2]), jnp.ones(2, bool))
    candidate, = jax.jit(lambda d: scan_event_candidates((PeriodDoubling(),), d, None))(data)
    np.testing.assert_array_equal(candidate.bracket_mask, [[True, True, False]])
    np.testing.assert_array_equal(candidate.eligible, [[True, True, False]] * 2)


@pytest.mark.parametrize('event_name', ['Hopf', 'BP', 'NeimarkSacker'])
def test_array_spectral_masks_match_eager(event_name):
    from jaxcont.bifurcations import events
    from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
    event = getattr(events, event_name)()
    if event_name == 'NeimarkSacker':
        raw = jnp.array([[.8j, -.8j, 1.], [1.2j, -1.2j, 1.]])
    else:
        raw = jnp.array([[-.2+2j, -.2-2j, -.3], [.2+2j, .2-2j, .3]])
    data = EventScanInput(jnp.array([0., 1.]), jnp.zeros((2, 1)), None, raw,
                          jnp.array([2, 2]), jnp.ones(2, bool))
    candidate, = jax.jit(lambda d: scan_event_candidates((event,), d, None))(data)
    points = [events.BranchPoint(data.params[i], data.states[i], eigenvalues=raw[i],
                                 trivial_index=data.trivial_index[i]) for i in range(2)]
    expected = np.zeros((1, 3), bool)
    for i, _, mode in event.scan(points):
        expected[i, mode.column] = True
    np.testing.assert_array_equal(candidate.bracket_mask, expected)


def test_array_fold_and_custom_protocol():
    from jaxcont.bifurcations.events import Fold
    from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
    data = EventScanInput(jnp.arange(3.), jnp.zeros((3, 1)),
                          jnp.array([[0., -1.], [0., 0.], [0., 1.]]), None, None,
                          jnp.ones(3, bool))
    class Scalar:
        def test_function(self, point):
            return point.p - 1
    class PythonScanner:
        def scan(self, points):
            return []
    candidates = jax.jit(lambda d: scan_event_candidates((Fold(), Scalar()), d, None))(data)
    for candidate in candidates:
        np.testing.assert_array_equal(candidate.bracket_mask, [[True], [False]])
    with pytest.raises(TypeError, match='PythonScanner.*scan_arrays'):
        jax.jit(lambda d: scan_event_candidates((PythonScanner(),), d, None))(data)
    # Custom eager scanners retain their original hook.
    assert PythonScanner().scan([]) == []


def test_array_cusp_orientation_padding_and_vmap():
    from jaxcont.bifurcations.codim2_events import Cusp
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventScanInput
    def rhs(u, p, args):
        return jnp.array([p[0] + p[1]*u[0] - u[0]**3])
    x = jnp.array([-.5, -.25, .25, .5, jnp.nan])
    states = jnp.stack((x, -2*x**3, jnp.array([1., -1., 1., -1., jnp.nan])), axis=1)
    data = EventScanInput(3*x*x, states, None, None, None,
                          jnp.array([1, 1, 1, 1, 0], bool), CurveLayout('fold', 'extended', 1, 1))
    event = Cusp(rhs)
    def run(sign):
        from dataclasses import replace
        return event.scan_arrays(replace(data, states=states.at[:, 2].multiply(sign)), None)
    result = jax.jit(jax.vmap(run))(jnp.array([1., -1.]))
    np.testing.assert_allclose(result.values[:, :4, 0], [[1.5, .75, -.75, -1.5]]*2)
    np.testing.assert_array_equal(result.bracket_mask[:, :, 0], [[False, True, False, False]]*2)


def test_pinned_pair_distinct_under_jit_vmap():
    from jaxcont.bifurcations.codim2_events import _drop_nearest_pinned_pair
    rows = jnp.array([[0j, 0j, 2j], [1j, -1j, 3j]])
    mask = jax.jit(jax.vmap(_drop_nearest_pinned_pair))(rows, jnp.array([0., 1.]))
    np.testing.assert_array_equal(mask, [[False, False, True]]*2)


@pytest.mark.parametrize('name,curve,count', [
    ('BogdanovTakens', 'fold', 1), ('ZeroHopf', 'fold', 1),
    ('ZeroHopf', 'hopf', 1), ('DoubleHopf', 'hopf', 1),
])
def test_curve_array_spectrum_uses_original_system(name, curve, count):
    from jaxcont.bifurcations import codim2_events as ce
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventScanInput
    n = 4 if curve == 'fold' else 5
    def rhs(u, p, args):
        a = p[1]
        if curve == 'fold':
            matrix = jnp.array([[0., 0., 0., 0.], [0., a, 0., 0.],
                                [0., 0., a, -2.], [0., 0., 2., a]])
        else:
            matrix = jnp.array([[0., -1., 0., 0., 0.], [1., 0., 0., 0., 0.],
                                [0., 0., a, 0., 0.], [0., 0., 0., a, -2.],
                                [0., 0., 0., 2., a]])
        return matrix @ u
    if curve == 'fold':
        state = jnp.concatenate((jnp.zeros(n+1), jnp.eye(n)[0]))
    else:
        state = jnp.concatenate((jnp.zeros(n+1), jnp.eye(n)[0], -jnp.eye(n)[1], jnp.ones(1)))
    data = EventScanInput(jnp.array([-.2, .2]), jnp.stack((state, state)), None,
                          jnp.zeros((2, 2)), None, jnp.ones(2, bool),
                          CurveLayout(curve, 'extended', n, 1))
    event = getattr(ce, name)(rhs, curve=curve)
    result = jax.jit(lambda d: event.scan_arrays(d, None))(data)
    assert result.spectra.shape == (2, n)
    assert int(result.bracket_mask.sum()) == count
    assert int(result.eligible[0].sum()) == count


def test_array_gh_normalizes_modes_and_bt_frequency_is_scalar():
    from jaxcont.bifurcations.codim2_events import GeneralizedHopf, BogdanovTakens
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventScanInput
    def rhs(u, p, args):
        x, y = u
        r2 = x*x + y*y
        return jnp.array([-y + p[1]*x*r2, x + p[1]*y*r2])
    states = jnp.array([[0., 0., 0., 3., 0., 0., -3., 1.],
                        [0., 0., 0., 0., 2., 2., 0., 1.]])
    data = EventScanInput(jnp.array([-.2, .2]), states, None, None, None,
                          jnp.ones(2, bool), CurveLayout('hopf', 'extended', 2, 1))
    result = jax.jit(lambda d: GeneralizedHopf(rhs).scan_arrays(d, None))(data)
    assert result.spectra is result.permutation is None
    np.testing.assert_array_equal(result.bracket_mask, [[True]])
    from dataclasses import replace
    unit_states = states.at[:, 3:7].divide(jnp.array([3., 2.])[:, None])
    unit_result = GeneralizedHopf(rhs).scan_arrays(replace(data, states=unit_states), None)
    np.testing.assert_allclose(result.values, unit_result.values, rtol=1e-5)
    bt = BogdanovTakens(rhs, curve='hopf')
    data = replace(data, states=states.at[:, -1].set(jnp.array([-.2, .2])))
    result = jax.jit(lambda d: bt.scan_arrays(d, None))(data)
    np.testing.assert_array_equal(result.bracket_mask, [[True]])
    assert result.spectra is result.permutation is None


@pytest.mark.parametrize('frequencies,valid', [
    ([20., 1., np.nan, np.nan], [1, 1, 0, 0]),
    ([20., -1., np.nan, np.nan], [1, 1, 0, 0]),
    ([-20., 20., 1., np.nan], [1, 1, 1, 0]),
    ([20., 11., np.nan, np.nan], [1, 1, 0, 0]),
    ([1., np.nan, np.nan, np.nan], [1, 0, 0, 0]),
    ([np.nan]*4, [0]*4),
    ([], []),
])
def test_bt_terminal_frequency_matches_eager_with_padding(frequencies, valid):
    from jaxcont.bifurcations.codim2_events import BogdanovTakens
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventScanInput
    from jaxcont.bifurcations.events import BranchPoint
    from jaxcont.utils.precision import default_tol

    def rhs(u, p, args):
        return jnp.array([-u[1], u[0]])

    omega = jnp.asarray(frequencies, dtype=float) * default_tol(jnp.asarray(0.).dtype)
    states = jnp.tile(jnp.array([0., 0., 0., 1., 0., 0., -1., 1.]), (len(valid), 1))
    states = states.at[:, -1].set(omega)
    data = EventScanInput(jnp.arange(len(valid), dtype=float), states, None, None,
                          None, jnp.asarray(valid, bool), CurveLayout('hopf', 'extended', 2, 1))
    event = BogdanovTakens(rhs, curve='hopf')
    result = jax.jit(lambda d: event.scan_arrays(d, None))(data)
    points = [BranchPoint(data.params[i], states[i], curve_layout=data.curve_layout)
              for i in range(len(valid)) if valid[i]]
    expected_crossings = np.zeros((max(len(valid)-1, 0), 1), bool)
    expected_terminal = expected_crossings.copy()
    for i, _, mode in event.scan(points):
        (expected_terminal if mode == 'curve_terminated' else expected_crossings)[i, 0] = True
    np.testing.assert_array_equal(result.bracket_mask, expected_crossings)
    np.testing.assert_array_equal(result.terminal_mask, expected_terminal)


def _shifted_hopf(a, *, bothside=False):
    import jaxcont as jc
    def f(u, p, a):
        return jnp.array([[p-a, -2.], [2., p-a]]) @ (u-jnp.array([.5, -.3]))
    return jc.continuation(jc.bif_problem(f, jnp.array([.5, -.3]), 0. if bothside else -.5, args=a),
        jc.Natural(), p_span=(-.5, .5), events=(jc.Hopf(),),
        settings=jc.ContinuationPar(ds=.125, ds_max=.125, adaptive=False, max_steps=10,
                                   compute_stability=False, bothside=bothside))


def test_jit_vmap_detects_shifted_hopf():
    import jaxcont as jc
    from jaxcont.utils.precision import default_tol
    shifts = jnp.array([-.25, .25])
    result = jax.jit(jax.vmap(_shifted_hopf))(shifts)
    assert result.events == result.rejected_events == []
    np.testing.assert_array_equal(result.event_candidates[0].bracket_mask.sum(axis=(1, 2)), [1, 1])
    np.testing.assert_allclose(result.branch.params[:, :9],
                               np.tile(np.arange(-.5, .501, .125), (2, 1)))
    np.testing.assert_array_equal(np.where(result.event_candidates[0].bracket_mask)[1], [1, 5])
    for k, a in enumerate(shifts):
        single = jax.tree_util.tree_map(lambda x: x[k], result)
        problem = jc.bif_problem(lambda u, p, a: jnp.array([[p-a, -2.], [2., p-a]]) @
                                 (u-jnp.array([.5, -.3])), jnp.array([.5, -.3]), -.5, args=a)
        refined = jc.refine_event_candidates(single, problem, (jc.Hopf(),))
        assert len(refined.events) == 1
        assert abs(refined.events[0].p-a) < 20*default_tol(shifts.dtype)
        assert not np.asarray(single.event_candidates[0].bracket_mask)[8:].any()


def test_custom_python_scan_rejected_under_jit():
    import jaxcont as jc
    class PythonScanner:
        def scan(self, points):
            return []
    with pytest.raises(TypeError, match='PythonScanner.*scan_arrays'):
        jax.jit(lambda a: jc.continuation(jc.bif_problem(lambda u,p,a: u-a*p,
            jnp.zeros(1), 0., args=a), jc.Natural(), p_span=(0., 1.),
            settings=jc.ContinuationPar(max_steps=4), events=(PythonScanner(),)))(jnp.array(1.))


def test_loop_scan_event_detection_keeps_smooth_branch_grad():
    import jaxcont as jc
    class Scalar:
        def test_function(self, point):
            return point.p-.5
    def run(a):
        result = jc.continuation(jc.bif_problem(lambda u,p,a: u-a*p, jnp.zeros(1), 0., args=a),
            jc.Natural(), p_span=(0., 1.), events=(Scalar(),),
            settings=jc.ContinuationPar(ds=.25, ds_max=.25, adaptive=False, max_steps=4, loop='scan',
                                       compute_stability=False))
        return result.branch.states.sum()
    np.testing.assert_allclose(jax.jit(jax.grad(run))(jnp.array(2.)), 2.5)


def test_candidates_refine_with_left_anchor():
    import jaxcont as jc
    from jaxcont.bifurcations.events import EventHit
    seen = []
    class Scalar:
        needs_args = True
        def test_function(self, point):
            return point.p-.5
        def refine(self, left, right, index, rhs, *, args, **kwargs):
            seen.append((float(left.p), float(args)))
            np.testing.assert_allclose(rhs(jnp.array([args]), left.p), 0.)
            return EventHit('custom', .5, jnp.array([.5]), index)
    problem = jc.bif_problem(lambda u,p,a: u-a, jnp.zeros(1), 0., args=jnp.array(0.),
                             step_update=lambda a,u,p: p)
    result = jax.jit(lambda: jc.continuation(problem, jc.Natural(), p_span=(0., 1.),
        events=(Scalar(),), settings=jc.ContinuationPar(ds=.25, ds_max=.25, adaptive=False, max_steps=4,
                                                       compute_stability=False)))()
    refined = jc.refine_event_candidates(result, problem, (Scalar(),))
    assert seen == [(.25, .25)]
    assert len(refined.events) == 1


def test_bothside_candidate_assembly():
    import jaxcont as jc
    from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
    def f(u, p, args):
        blocks = [jnp.array([[p-a, -w], [w, p-a]])
                  for a, w in zip((-.25, 0., .25), (1., 2., 3.))]
        return jax.scipy.linalg.block_diag(*blocks) @ (u-.5)
    class PermutedEigen:
        def __call__(self, matrix):
            values = jnp.linalg.eigvals(matrix)
            return jnp.roll(values, jnp.where(matrix[0, 0] < .25, 1, 3))
    problem = jc.bif_problem(f, jnp.full(6, .5), 0.)
    result = jc.continuation(problem, jc.Natural(), p_span=(-.5, .375), events=(jc.Hopf(),),
        solvers=jc.Solvers(eigen=PermutedEigen()),
        settings=jc.ContinuationPar(ds=.125, ds_max=.125, adaptive=False, max_steps=6, bothside=True,
                                   compute_stability=False))
    branch = result.branch
    assert int((branch.params == 0).sum()) == 1
    candidate, = result.event_candidates
    direct, = scan_event_candidates((jc.Hopf(),), EventScanInput(branch.params, branch.states,
        branch.tangents, branch.eigenvalues, None, jnp.ones(branch.params.shape, bool)), None)
    np.testing.assert_array_equal(candidate.bracket_mask, direct.bracket_mask)
    np.testing.assert_array_equal(candidate.permutation, direct.permutation)
    assert np.any(np.asarray(candidate.permutation) != np.arange(6))
    rows, modes = np.where(candidate.bracket_mask)
    np.testing.assert_array_equal(rows, [1, 3, 5])
    assert len(set(modes)) == 3
    np.testing.assert_allclose([hit.p for hit in result.events], [-.25, 0., .25], atol=1e-7)
    # Extra buffer slots in a traced single arm never open brackets.
    padded = jax.jit(lambda: _shifted_hopf(jnp.array(.25)))()
    mask = np.asarray(padded.branch.valid)
    assert not np.asarray(padded.event_candidates[0].bracket_mask)[~(mask[:-1]&mask[1:])].any()


@pytest.mark.parametrize('name', ['Cusp', 'BogdanovTakens'])
def test_scalar_candidate_refinement_dispatch(name, monkeypatch):
    import jaxcont as jc
    from jaxcont.bifurcations.curve_layout import CurveLayout
    from jaxcont.bifurcations.event_scan import EventCandidates
    from jaxcont.bifurcations.events import EventHit
    event = getattr(jc, name)(lambda u,p,a: u, **({'curve': 'hopf'} if name == 'BogdanovTakens' else {}))
    calls = []
    def refine(self, left, right, index, rhs, *, tolerance, max_iterations, args=None):
        calls.append(index)
        return EventHit(name, 0., jnp.zeros(1), index)
    monkeypatch.setattr(type(event), 'refine', refine)
    layout = CurveLayout('hopf' if name == 'BogdanovTakens' else 'fold', 'extended', 1, 1)
    branch = jc.Branch(jnp.array([-.25, .25]), jnp.zeros((2, 5)), curve_layout=layout)
    candidate = EventCandidates(jnp.array([[-1.], [1.]]), jnp.ones((2,1), bool),
                                jnp.ones((1,1), bool), None, None)
    result = jc.ContinuationResult(branch, event_candidates=(candidate,))
    refined = jc.refine_event_candidates(result, jc.bif_problem(lambda u,p,a: u, jnp.zeros(5), -.25), (event,))
    assert calls == [(0, 1)]
    assert len(refined.events) == 1


def test_eager_custom_scanner_receives_assembled_branch():
    import jaxcont as jc
    from jaxcont.bifurcations.events import EventHit
    seen = []
    class PythonScanner:
        def scan(self, points):
            seen.extend(float(point.p) for point in points)
            return [(1, 2, None)]
        def refine(self, left, right, index, rhs, *, mode, tolerance, max_iterations):
            assert mode is None
            return EventHit('custom', -.0625, jnp.array([-.0625]), index)
    result = jc.continuation(jc.bif_problem(lambda u,p,a: u-p, jnp.zeros(1), 0.),
        jc.Natural(), p_span=(-.25, .5), events=(PythonScanner(),),
        settings=jc.ContinuationPar(ds=.125, ds_max=.125, adaptive=False, max_steps=6,
                                   bothside=True, compute_stability=False))
    np.testing.assert_allclose(seen, np.arange(-.25, .501, .125))
    assert result.event_candidates == (None,)
    assert len(result.events) == 1


@pytest.mark.parametrize('bothside', [False, True])
def test_eager_python_scalar_event_falls_back_to_pointwise_tests(bothside):
    import jaxcont as jc
    from jaxcont.bifurcations.events import EventHit

    class PythonScalarEvent:
        def test_function(self, point):
            return float(point.p) - .5

        def refine(self, left, right, index, rhs, *, tolerance, max_iterations):
            assert float(left.p) <= .5 <= float(right.p)
            return EventHit('custom_scalar', .5, jnp.array([.5]), index)

    problem = jc.bif_problem(lambda u, p, args: u-p, jnp.zeros(1), 0.)
    settings = jc.ContinuationPar(ds=.25, ds_max=.25, adaptive=False,
                                 max_steps=6, bothside=bothside,
                                 compute_stability=False)
    result = jc.continuation(problem, jc.Natural(), p_span=(-.5, 1.) if bothside else (0., 1.),
                             settings=settings, events=(PythonScalarEvent(),))
    assert result.event_candidates == (None,)
    assert len(result.events) == 1
    assert result.events[0].p == .5
    np.testing.assert_allclose(result.events[0].u, [.5])
    assert result.rejected_events == []


def test_python_scalar_event_rejected_under_jit():
    import jaxcont as jc

    class PythonScalarEvent:
        def test_function(self, point):
            return float(point.p) - .5

    def run(u):
        return jc.continuation(jc.bif_problem(lambda u, p, args: u-p, u, 0.),
            jc.Natural(), p_span=(0., 1.), events=(PythonScalarEvent(),),
            settings=jc.ContinuationPar(max_steps=2, compute_stability=False))

    with pytest.raises(TypeError, match='array-pure'):
        jax.jit(run)(jnp.zeros(1))
