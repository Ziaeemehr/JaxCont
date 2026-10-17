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
