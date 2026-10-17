"""Fixed-shape, array-only event detection; candidates are unvalidated."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxcont.bifurcations.array_modes import crossing_mask, track_modes_array
from jaxcont.bifurcations.curve_layout import (
    decode_curve_state, minimal_hopf_floor, orient_curve_modes,
)
from jaxcont.utils.precision import default_tol


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EventScanInput:
    """One branch buffer with static layout and effective Floquet strategy keys.

    ``floquet_spectra[k]`` is the branch ``FloquetSpectrum`` computed with the
    static strategy ``floquet_strategies[k]``. A PD/NS event whose
    ``floquet_solver`` is set scans its matching entry (values, trivial index
    and convergence); with no strategies supplied, Floquet events scan
    ``eigenvalues`` and ``trivial_index``.

    ``status`` is the run's integer ``Status`` code, or ``None`` when unknown;
    a minimal-chart ``BogdanovTakens(curve="hopf")`` scan uses it to tell a
    curve that stopped short of its span from one that completed it.
    """

    params: object
    states: object
    tangents: object
    eigenvalues: object
    trivial_index: object
    valid: object
    curve_layout: object = None
    floquet_strategies: tuple = ()
    floquet_spectra: tuple = ()
    status: object = None

    def tree_flatten(self):
        return ((self.params, self.states, self.tangents, self.eigenvalues,
                 self.trivial_index, self.valid, self.floquet_spectra, self.status),
                (self.curve_layout, self.floquet_strategies))

    @classmethod
    def tree_unflatten(cls, metadata, children):
        layout, strategies = metadata
        params, states, tangents, eigenvalues, trivial, valid, spectra, status = children
        return cls(params, states, tangents, eigenvalues, trivial, valid,
                   layout, strategies, spectra, status)


def terminated_status(status):
    """Whether a run's ``Status`` code means it stopped before its span ended
    (stalled, nonfinite or out of attempts); ``None`` counts as terminated."""
    from jaxcont.core.scan_continuation import Status

    if status is None:
        return jnp.asarray(True)
    stopped = (int(Status.STALLED), int(Status.NONFINITE), int(Status.MAX_ATTEMPTS))
    return jnp.isin(jnp.asarray(status), jnp.asarray(stopped))


class EventCandidates(NamedTuple):
    """Crossings plus optional non-root ``curve_terminated`` bracket flags.

    ``terminal_mask`` has the same shape as ``bracket_mask`` when supplied;
    its entries require failed-hit reporting rather than root refinement.
    """
    values: object
    eligible: object
    bracket_mask: object
    spectra: object
    permutation: object
    terminal_mask: object = None


def nontrivial_mask(permutation, trivial_index):
    """Exclude raw integer identity, including repeated unit multipliers."""
    return permutation != jnp.asarray(trivial_index)[:, None]


def _candidates(values, eligible, data, spectra=None, permutation=None, adjacent=None):
    eligible = eligible & data.valid[:, None] & jnp.isfinite(values)
    mask = crossing_mask(values, eligible, data.valid)
    if adjacent is not None:
        a, b = values[:-1], values[1:]
        possible = ((jnp.sign(a) != jnp.sign(b)) & jnp.isfinite(a) & jnp.isfinite(b)
                    & eligible[:-1] & eligible[1:] & adjacent[:, None])
        preceding = jnp.concatenate((jnp.zeros_like(possible[:1]), possible[:-1]), axis=0)
        mask = possible & ~((a == 0) & preceding)
    return EventCandidates(values, jax.lax.stop_gradient(eligible),
                           jax.lax.stop_gradient(mask), spectra, permutation)


def _scalar(values, data, adjacent=None):
    values = jnp.asarray(values).reshape((data.params.shape[0], 1))
    return _candidates(values, jnp.ones_like(values, dtype=bool), data, adjacent=adjacent)


def _floquet_inputs(event, data):
    """``(raw, trivial_index, usable)`` for the event's effective strategy."""
    strategy = getattr(event, 'floquet_solver', None)
    if strategy is None or not data.floquet_strategies:
        return data.eigenvalues, data.trivial_index, data.valid
    for key, spectrum in zip(data.floquet_strategies, data.floquet_spectra):
        if key == strategy:
            return spectrum.values, spectrum.trivial_index, data.valid & spectrum.converged
    raise ValueError(f'{type(event).__name__}: no Floquet spectrum supplied for {strategy!r}')


def _spectral(event, data, raw, pinned=None, trivial_index=None, usable=None):
    from jaxcont.bifurcations.events import Hopf, PeriodDoubling, NeimarkSacker
    from jaxcont.bifurcations.codim2_events import DoubleHopf, ZeroHopf

    usable = data.valid if usable is None else usable
    # Failed spectra neither update mode prediction nor open brackets.
    spectra, permutation = track_modes_array(raw, usable)
    tolerance = getattr(event, 'tolerance', None)
    tol = default_tol(spectra.real.dtype) ** .5 if tolerance is None else tolerance
    real = jnp.abs(spectra.imag) <= tol
    complex_upper = spectra.imag > tol
    values = spectra.real
    if isinstance(event, (PeriodDoubling, NeimarkSacker)):
        if trivial_index is None:
            raise ValueError('Floquet events require flow-direction trivial_index at every point')
        keep = nontrivial_mask(permutation, trivial_index) & usable[:, None]
        if isinstance(event, PeriodDoubling):
            values, eligible = spectra.real + 1, real & keep
        else:
            values, eligible = jnp.abs(spectra) - 1, complex_upper & keep
    else:
        upper = isinstance(event, (Hopf, DoubleHopf)) or (
            isinstance(event, ZeroHopf) and event.curve == 'fold')
        eligible = complex_upper if upper else real
    if pinned is not None:
        eligible = eligible & pinned(spectra)
    return _candidates(values, eligible, data, spectra, permutation)


def scan_builtin(event, data, args):
    """Dispatch on static event types; all sample-dependent work stays in JAX."""
    from jaxcont.bifurcations.events import Fold, Hopf, BP, PeriodDoubling, NeimarkSacker
    from jaxcont.bifurcations.codim2_events import (
        _CurveEvent, Cusp, GeneralizedHopf, BogdanovTakens,
        _drop_nearest, _drop_nearest_pinned_pair, _bt_coefficient,
    )
    from jaxcont.bifurcations.fold_normal_form import fold_coefficient
    from jaxcont.bifurcations.hopf_normal_form import lyapunov_coefficient

    if isinstance(event, Fold):
        if data.tangents is None:
            raise ValueError('Fold array scanning requires tangents')
        return _scalar(data.tangents[:, -1], data)
    if isinstance(event, _CurveEvent):
        layout = data.curve_layout
        if layout is None:
            raise ValueError('curve layout is required for curve decoding')
        if layout.kind != event.curve or layout.free != event.free:
            raise ValueError('Event and curve layout disagree')
        if data.params.shape[0] == 0:
            if isinstance(event, (Cusp, GeneralizedHopf)) or (
                    isinstance(event, BogdanovTakens) and event.curve == 'hopf'):
                result = _scalar(jnp.empty((0,), data.states.dtype), data)
                if isinstance(event, BogdanovTakens):
                    result = result._replace(terminal_mask=jnp.zeros_like(result.bracket_mask))
                return result
            raw = jnp.empty((0, layout.n), jnp.result_type(data.states, jnp.complex64))
            return _spectral(event, data, raw)
        # Padding evaluates a valid branch state before any spectral or coefficient solve.
        first = jnp.argmax(data.valid)
        states = jnp.where(data.valid[:, None], data.states, data.states[first])
        params = jnp.where(data.valid, data.params, data.params[first])
        u, p, modes = jax.vmap(lambda X, q: decode_curve_state(
            X, q, layout, event.raw_f, event.args))(states, params)
        if isinstance(event, (Cusp, GeneralizedHopf)):
            if isinstance(event, GeneralizedHopf) and layout.formulation == 'bt':
                values = jax.vmap(lambda x, q, m: _bt_coefficient(event, x, q, m))(u, p, modes)
                return _candidates(values[:, None], (modes.kappa > 0)[:, None], data)
            modes, adjacent = orient_curve_modes(modes, data.valid)
            if isinstance(event, Cusp):
                values = jax.vmap(lambda x, q, v: fold_coefficient(
                    event.raw_f, x, q, v, event.args))(u, p, modes.v)
            else:
                values = jax.vmap(lambda x, q, a, b, w: lyapunov_coefficient(
                    event.raw_f, x, q, a, b, w, event.args))(
                        u, p, modes.q1, modes.q2, modes.omega)
            return _scalar(values, data, adjacent)
        if isinstance(event, BogdanovTakens) and event.curve == 'hopf':
            if layout.formulation == 'bt':
                result = _scalar(modes.kappa, data)
                return result._replace(terminal_mask=jnp.zeros_like(result.bracket_mask))
            result = _scalar(modes.omega, data)
            indices = jnp.arange(data.params.shape[0])
            last = jnp.max(jnp.where(data.valid, indices, -1))
            terminal = ((indices[:-1] == last - 1) & data.valid[:-1]
                        & data.valid[1:])[:, None]
            last_omega = jnp.abs(modes.omega[jnp.maximum(last, 0)])
            if layout.formulation == 'minimal':
                # The chart cannot reach omega = 0: a run that stopped just
                # above its frequency floor ran into the BT point.
                near_zero = ((last_omega < 10 * minimal_hopf_floor(data.states.dtype))
                             & terminated_status(data.status))
            else:
                near_zero = last_omega < 10 * default_tol(data.states.dtype)
            terminal = terminal & near_zero & ~jnp.any(result.bracket_mask)
            return result._replace(terminal_mask=jax.lax.stop_gradient(terminal))
        raw = jax.vmap(event._eigenvalues)(u, p)
        if event.curve == 'fold':
            pinned = lambda rows: jax.vmap(lambda row: _drop_nearest(row, 0j))(rows)
        else:
            pinned = lambda rows: jax.vmap(_drop_nearest_pinned_pair)(rows, modes.omega)
        if layout.formulation == 'bt':
            def pinned(rows):
                roots = jnp.sqrt(modes.kappa.astype(rows.dtype))
                return jax.vmap(_drop_nearest_pinned_pair)(rows, roots)
            result = _spectral(event, data, raw, pinned)
            eligible = result.eligible & (modes.kappa > 0)[:, None]
            return _candidates(result.values, eligible, data, result.spectra, result.permutation)
        return _spectral(event, data, raw, pinned)
    if isinstance(event, (PeriodDoubling, NeimarkSacker)):
        raw, trivial, usable = _floquet_inputs(event, data)
        if raw is None:
            raise ValueError(f'{type(event).__name__} array scanning requires eigenvalues')
        return _spectral(event, data, raw, trivial_index=trivial, usable=usable)
    if isinstance(event, (Hopf, BP)):
        if data.eigenvalues is None:
            raise ValueError(f'{type(event).__name__} array scanning requires eigenvalues')
        return _spectral(event, data, data.eigenvalues)
    raise TypeError(f'{type(event).__name__} has no built-in array scanner')


def scan_event_candidates(events, data, args):
    """Return one fixed-shape candidate buffer per event.

    Custom eager scanners remain supported by ``detect_events``. Array-pure
    custom scalar tests and explicit ``scan_arrays`` hooks support transforms.
    """
    from jaxcont.bifurcations.events import BranchPoint

    result = []
    for event in events:
        scanner = getattr(event, 'scan_arrays', None)
        if scanner is not None:
            result.append(scanner(data, args))
            continue
        name = type(event).__name__
        if getattr(event, 'scan', None) is not None:
            raise TypeError(f'{name}: Python scan(points) requires eager detect_events; implement scan_arrays(data, args) for traced detection')
        try:
            values = jax.vmap(lambda p, u, t, e, i: event.test_function(
                BranchPoint(p, u, t, e, i, data.curve_layout)))(
                    data.params, data.states, data.tangents, data.eigenvalues, data.trivial_index)
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError) as error:
            raise TypeError(f'{name}: test_function must be array-pure or implement scan_arrays(data, args)') from error
        result.append(_scalar(values, data))
    return tuple(result)
