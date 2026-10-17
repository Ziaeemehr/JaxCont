"""
Functional, diffrax-style public API for JaxCont (the "Sketch A" spine).

This is the blessed surface going forward:

    prob = jaxcont.bif_problem(f, u0, p0)
    sol  = jaxcont.continuation(prob, p_span=(0.0, 1.0), events=[jaxcont.Fold()])

Both algorithms (`PseudoArclength()`, the default, and `Natural()`) run on
the fully JIT-compiled whole-loop engines in `core/scan_continuation.py`,
which support `jit`, `vmap` and `jax.jacfwd`. Reverse-mode `jax.grad`
requires `ContinuationPar(loop="scan")`; the default `loop="while"`
uses dynamic termination. See docs/source/transformations.rst for tested coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxcont.bifurcations.events import (
    BP,
    Event,
    EventHit,
    Fold,
    Hopf,
    NeimarkSacker,
    PeriodDoubling,
    detect_events,
)
from jaxcont.bifurcations.event_scan import EventCandidates
from jaxcont.core.continuation import ContinuationSolution
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver
from jaxcont.stability import floquet as _floquet
from jaxcont.stability.floquet import FloquetSpectrum, PeriodicSchur, ProductFloquet
from jaxcont.utils.precision import default_tol

__all__ = [
    "BP",
    "BifProblem",
    "Branch",
    "ContinuationAlgorithm",
    "ContinuationPar",
    "ContinuationResult",
    "Event",
    "EventHit",
    "FloquetSpectrum",
    "Fold",
    "Hopf",
    "Natural",
    "NeimarkSacker",
    "PeriodDoubling",
    "PeriodicSchur",
    "ProductFloquet",
    "PseudoArclength",
    "Solvers",
    "bif_problem",
    "continuation",
    "refine_event_candidates",
]

PyTree = Any

# Internal name for the continuation parameter inside the legacy params dict.
_P_KEY = "__jaxcont_p__"

# Sentinel for "leave this field unchanged" (distinct from args=None, which is
# a legitimate value meaning "no extra args").
_KEEP = object()


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BifProblem:
    """
    A continuation/bifurcation problem: find ``u`` with ``f(u, p, args) = 0``
    and continue it in the scalar parameter ``p``.

    ``f`` is a pure function ``f(u, p, args) -> residual`` (same shape as ``u``).
    Extra parameters live in ``args`` (any PyTree) — this is the axis you
    ``vmap``/``grad`` over (a second parameter, a design vector, NN weights).

    ``state_names``/``param_name`` are display labels only (e.g. for
    ``plot_continuation``); they carry no numerical meaning.

    ``step_update(args, u, p) -> args`` (optional) refreshes auxiliary data in
    ``args`` from each accepted point -- the periodic phase-condition
    reference, the Hopf-curve eigenvector anchor. It must return ``args``
    with the same structure, shapes and dtypes, and may only change data that
    selects among equivalent solutions (a phase or normalization anchor), not
    the solution set itself: eigenvalues and event test functions are
    evaluated with ``args``, while each event bracket is refined with
    ``step_update(args, u_left, p_left)``.

    Registered as a JAX PyTree: ``u0``, ``p0``, ``args`` are dynamic leaves;
    ``f``, ``kind``, ``state_names``, ``param_name``, ``curve_layout`` and ``step_update`` are
    static. This makes ``jax.vmap(..., in_axes=...)`` over a batched problem
    structurally valid (full end-to-end vmap awaits the lax.scan loop
    rewrite; see ARCHITECTURE.md §3.1).
    """

    f: Callable[[Array, Array, PyTree], Array]
    u0: Array
    p0: Array
    args: PyTree = None
    kind: Literal[
        "equilibrium", "periodic", "bvp", "fold_curve", "hopf_curve"
    ] = "equilibrium"
    state_names: Optional[tuple[str, ...]] = None
    param_name: Optional[str] = None
    step_update: Optional[Callable[[PyTree, Array, Array], PyTree]] = None

    curve_layout: object = None

    def at(
        self,
        *,
        u0: Optional[Array] = None,
        p0: Optional[Array] = None,
        args: PyTree = _KEEP,
    ) -> BifProblem:
        """Return a copy with selected fields overridden (cheap, functional)."""
        return replace(
            self,
            u0=self.u0 if u0 is None else u0,
            p0=self.p0 if p0 is None else p0,
            args=self.args if args is _KEEP else args,
        )

    def as_rhs(self, p) -> Callable[[Array], Array]:
        """
        Return an autonomous ``rhs(u)`` frozen at parameter value ``p``.

        This is the lyapax bridge (ARCHITECTURE.md §9): hand a branch point to
        ``lyapax.ode_problem(prob.as_rhs(p_star), state0=u_star, ...)``.
        """
        args = self.args
        f = self.f
        return lambda u: f(u, p, args)


def _bifproblem_flatten(prob: BifProblem):
    children = (prob.u0, prob.p0, prob.args)
    aux = (prob.f, prob.kind, prob.state_names, prob.param_name, prob.step_update, prob.curve_layout)
    return children, aux


def _bifproblem_unflatten(aux, children):
    f, kind, state_names, param_name, step_update, curve_layout = aux
    u0, p0, args = children
    return BifProblem(
        f=f, u0=u0, p0=p0, args=args, kind=kind,
        state_names=state_names, param_name=param_name, step_update=step_update, curve_layout=curve_layout,
    )


jax.tree_util.register_pytree_node(
    BifProblem, _bifproblem_flatten, _bifproblem_unflatten
)


def bif_problem(
    f: Callable[[Array, Array, PyTree], Array],
    u0: Array,
    p0: float | Array,
    *,
    args: PyTree = None,
    kind: str = "equilibrium",
    state_names: Optional[Sequence[str]] = None,
    param_name: Optional[str] = None,
    step_update: Optional[Callable[[PyTree, Array, Array], PyTree]] = None,
) -> BifProblem:
    """
    Front-door factory for a :class:`BifProblem` (mirrors lyapax's
    ``ode_problem``). Coerces ``u0``/``p0`` to JAX arrays. ``step_update``
    is described on :class:`BifProblem`.

    ``state_names``/``param_name`` are optional display labels (used by
    ``plot_continuation`` and friends) -- purely cosmetic, no effect on the
    numerics.
    """
    u0 = jnp.asarray(u0)
    if state_names is not None and len(state_names) != len(u0):
        raise ValueError(
            f"state_names has {len(state_names)} entries but u0 has "
            f"{len(u0)} components"
        )
    return BifProblem(
        f=f,
        u0=u0,
        p0=jnp.asarray(p0, dtype=u0.dtype),
        args=args,
        kind=kind,
        state_names=tuple(state_names) if state_names is not None else None,
        param_name=param_name,
        step_update=step_update,
    )


# ---------------------------------------------------------------------------
# Algorithms & settings
# ---------------------------------------------------------------------------

class ContinuationAlgorithm:
    """Marker base for continuation algorithms."""


@dataclass(frozen=True)
class PseudoArclength(ContinuationAlgorithm):
    """
    Pseudo-arclength continuation (default; passes fold points).

    Runs on the fully JIT-compiled whole-loop engine
    (``core/scan_continuation.pseudo_arclength_scan``): it is ``vmap``-able
    and structurally bounded. Detection/stability are computed as a
    vectorized post-pass and refined with the same detector.
    """


@dataclass(frozen=True)
class Natural(ContinuationAlgorithm):
    """
    Natural-parameter continuation (simple; stalls at folds).

    Runs on ``core/scan_continuation.natural_scan`` -- the same whole-loop,
    ``vmap``-safe engine design as :class:`PseudoArclength`, with the
    fixed-parameter predictor/corrector instead of the bordered one.
    """


@dataclass(frozen=True)
class ContinuationPar:
    """Numerical settings for a continuation run."""

    ds: float = 0.01
    ds_min: float = 1e-5
    ds_max: float = 0.1
    max_steps: int = 1000
    adaptive: bool = True
    newton_tol: Optional[float] = None
    newton_max_iter: int = 20
    loop: Literal["while", "scan"] = "while"
    max_attempts: Optional[int] = None   # total corrector attempts; None -> 4 * max_steps
    compute_stability: bool = True
    bothside: bool = False
    theta: Optional[float] = None  # state weight; None -> 1 / n_state


@dataclass(frozen=True)
class Solvers:
    """Pluggable linear-algebra bundle for continuation() (ARCHITECTURE.md §4.6).

    Dense()/DenseEigen() are the only implementations today; the boundary
    exists so a future GMRES()/Arnoldi() (large systems) or ChebyshevDDE()
    (the DDE eigensolver seam, ARCHITECTURE.md §10.2) can swap in without
    touching the continuation loop. Never itself crosses a jax.jit boundary
    -- only its .linear/.eigen fields do, as static arguments to
    pseudo_arclength_scan/natural_scan.

    ``eigen`` serves equilibrium spectra. ``floquet`` is the periodic-orbit
    Floquet strategy (``ProductFloquet(eigen_solver)`` or
    ``PeriodicSchur(max_sweeps)``) for branch stability and for PD/NS events
    without their own ``floquet_solver``.
    """

    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()
    floquet: Any = _floquet.DEFAULT_FLOQUET


# ---------------------------------------------------------------------------
# Events -- Event, Fold, Hopf, EventHit are imported from
# jaxcont.bifurcations.events (see top of file) and re-exported here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Branch:
    """The computed solution branch (one connected curve).

    Under a normal (eager) call, ``params``/``states``/... are already
    trimmed to the ``n_valid`` real points and ``valid`` is ``None``. When
    ``continuation()`` runs inside ``jax.vmap``/``jax.jit``, the true point
    count can't be turned into a concrete trim length, so these arrays stay
    the full fixed-size engine buffer and ``valid`` is a boolean mask over
    that buffer's first axis (``True`` for real points) -- mirrors how
    ``examples/example_06_vmap_sweep.py`` uses ``pseudo_arclength_scan``
    directly and trims per-batch-element with ``n_valid`` after the trace.

    Periodic branches report the ``Solvers.floquet`` spectrum: ``eigenvalues``
    are multipliers, ``trivial_index`` the flow-direction column,
    ``floquet_logabs`` their log-magnitudes (usable where values under/overflow)
    and ``floquet_converged`` the per-point status. A failed point has NaN
    multipliers, index ``-1`` and ``stable`` false; padding has zero
    multipliers, ``-inf`` logs, index ``-1`` and false status.
    """

    params: Array           # (n_valid,) eager, (buffer_len,) traced
    states: Array           # (n_valid, state_dim) eager, (buffer_len, state_dim) traced
    tangents: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    stable: Optional[Array] = None
    valid: Optional[Array] = None   # bool mask, only set when traced
    trivial_index: Optional[Array] = None  # periodic flow-direction indices
    floquet_logabs: Optional[Array] = None     # periodic log|multiplier|, (N, n)
    floquet_converged: Optional[Array] = None  # periodic spectrum status, (N,)

    curve_layout: object = None

    @property
    def n_valid(self) -> int | Array:
        if self.valid is None:
            return int(self.params.shape[0])
        count = self.valid.sum()
        return count if isinstance(count, jax.core.Tracer) else int(count)

    def at_param(self, p: float) -> tuple[float, Array]:
        """Return the ``(param, state)`` on the branch closest to ``p``."""
        distance = jnp.abs(self.params - p)
        if self.valid is not None:
            distance = jnp.where(self.valid, distance, jnp.inf)
        idx = int(jnp.argmin(distance))
        return float(self.params[idx]), self.states[idx]


def _branch_flatten(b: Branch):
    children = (b.params, b.states, b.tangents, b.eigenvalues, b.stable, b.valid, b.trivial_index,
                b.floquet_logabs, b.floquet_converged)
    return children, b.curve_layout


def _branch_unflatten(aux, children):
    (params, states, tangents, eigenvalues, stable, valid, trivial_index,
     floquet_logabs, floquet_converged) = children
    return Branch(
        params=params, states=states, tangents=tangents,
        eigenvalues=eigenvalues, stable=stable, valid=valid, trivial_index=trivial_index,
        floquet_logabs=floquet_logabs, floquet_converged=floquet_converged, curve_layout=aux,
    )


jax.tree_util.register_pytree_node(Branch, _branch_flatten, _branch_unflatten)


@dataclass
class ContinuationResult:
    """Return value of :func:`continuation`."""

    branch: Branch
    # ``events`` holds only validated hits; refinements that failed (see
    # ``EventHit.status``) are kept in ``rejected_events`` with ``info["reason"]``.
    events: list[EventHit] = field(default_factory=list)
    rejected_events: list[EventHit] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _solution: Optional[ContinuationSolution] = None
    # Lower-case ``Status`` name (e.g. "reached_end"); None under jit/vmap,
    # where ``stats["status"]`` holds the integer ``Status`` code instead.
    status: Optional[str] = None
    event_candidates: tuple[EventCandidates | None, ...] = ()

    def plot(self, **kwargs):
        """Plot the bifurcation diagram (delegates to the legacy solution)."""
        if self._solution is None:
            raise RuntimeError("No underlying solution to plot.")
        return self._solution.plot(**kwargs)


def _continuation_result_flatten(r: ContinuationResult):
    children = (r.branch, r.events, r.rejected_events, r.stats, r._solution, r.event_candidates)
    return children, r.status


def _continuation_result_unflatten(aux, children):
    branch, events, rejected_events, stats, solution, candidates = children
    return ContinuationResult(
        branch=branch, events=events, rejected_events=rejected_events,
        stats=stats, _solution=solution, status=aux, event_candidates=candidates,
    )


jax.tree_util.register_pytree_node(
    ContinuationResult, _continuation_result_flatten, _continuation_result_unflatten
)


# ---------------------------------------------------------------------------
# Adapters (BifProblem <-> ContinuationSolution)
# ---------------------------------------------------------------------------


class _BoundF:
    """Static engine ``f`` that closes over the non-array leaves of ``args``.

    Invariant: equal ``(f, treedef, mask, static)`` compare and hash equal, so
    repeated ``continuation()`` calls on one problem reuse the compiled kernel.
    The array leaves of ``args`` stay dynamic and are passed as a list.
    """

    def __init__(self, f, treedef, mask, static):
        self.f, self.treedef, self.mask, self.static = f, treedef, mask, static

    def __call__(self, u, p, dyn):
        dyn_it, static_it = iter(dyn), iter(self.static)
        leaves = [next(dyn_it) if m else next(static_it) for m in self.mask]
        return self.f(u, p, jax.tree_util.tree_unflatten(self.treedef, leaves))

    def _key(self):
        return (self.f, self.treedef, self.mask, self.static)

    def __eq__(self, other):
        return isinstance(other, _BoundF) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class _BoundStepUpdate(_BoundF):
    """Static engine ``step_update`` over the same split of ``args`` as :class:`_BoundF`.

    Takes and returns the list of dynamic leaves; the wrapped ``step_update``
    sees the full ``args``. Its result must keep the treedef and the static
    leaves, so the engine's loop carry keeps its structure.
    """

    def __call__(self, dyn, u, p):
        dyn_it, static_it = iter(dyn), iter(self.static)
        leaves = [next(dyn_it) if m else next(static_it) for m in self.mask]
        new = self.f(jax.tree_util.tree_unflatten(self.treedef, leaves), u, p)
        new_leaves, new_treedef = jax.tree_util.tree_flatten(new)
        new_static = tuple(x for x, m in zip(new_leaves, self.mask) if not m)
        if new_treedef != self.treedef or any(
            a is not b and a != b for a, b in zip(new_static, self.static)
        ):
            raise ValueError(
                "step_update may only change the array leaves of args; it "
                "must return args with the same structure and the same "
                "non-array leaves"
            )
        return [x for x, m in zip(new_leaves, self.mask) if m]

    def __eq__(self, other):
        return isinstance(other, _BoundStepUpdate) and self._key() == other._key()

    def __hash__(self):
        return hash(("step_update",) + self._key())


def _engine_f_args(problem: BifProblem):
    """Return ``(f, step_update, args)`` for the scan engine.

    Only ``jax.Array`` / ``np.ndarray`` leaves of ``problem.args`` are dynamic
    engine inputs (compile-cache reuse across changed values). Every other leaf
    -- Python scalars, callables such as the one periodic problems carry -- is
    bound statically and stays concrete inside ``f``, so it may be used as a
    shape, loop bound or ``if`` flag; changing its value recompiles.
    ``step_update`` (``None`` when the problem has none) is wrapped over the
    same split.
    """
    leaves, treedef = jax.tree_util.tree_flatten(problem.args)
    mask = tuple(isinstance(x, (jax.Array, np.ndarray)) for x in leaves)
    if all(mask):
        return problem.f, problem.step_update, problem.args
    dyn = [x for x, m in zip(leaves, mask) if m]
    static = tuple(x for x, m in zip(leaves, mask) if not m)
    step_update = (
        None if problem.step_update is None
        else _BoundStepUpdate(problem.step_update, treedef, mask, static)
    )
    return _BoundF(problem.f, treedef, mask, static), step_update, dyn


def _call_scan(scan_fn, problem: BifProblem, u0, p_from, p_to, settings: ContinuationPar, solvers: Solvers, dtype):
    f, step_update, args = _engine_f_args(problem)
    from jaxcont.core.scan_continuation import natural_scan, natural_lax_scan
    weighting = {} if scan_fn in (natural_scan, natural_lax_scan) else {
        "theta": jnp.asarray(1 / u0.shape[0] if settings.theta is None else settings.theta, dtype)
    }
    return scan_fn(
        f,
        step_update,
        u0,
        jnp.asarray(p_from, dtype),
        jnp.asarray(p_to, dtype),
        args,
        jnp.asarray(settings.ds, dtype),
        jnp.asarray(settings.ds_min, dtype),
        jnp.asarray(settings.ds_max, dtype),
        jnp.asarray(
            default_tol(dtype) if settings.newton_tol is None else settings.newton_tol, dtype
        ),
        int(settings.max_steps),
        int(settings.newton_max_iter) if settings.loop == "scan" else jnp.asarray(settings.newton_max_iter),
        solvers.linear,
        jnp.asarray(settings.adaptive),
        (4 * int(settings.max_steps) if settings.max_attempts is None
         else int(settings.max_attempts)) if settings.loop == "scan" else jnp.asarray(
            4 * int(settings.max_steps) if settings.max_attempts is None
            else settings.max_attempts, jnp.int32),
        **weighting,
    )


def _status_name(code) -> str:
    from jaxcont.core.scan_continuation import Status

    return Status(int(code)).name.lower()


def _periodic_spectrum(problem: BifProblem, strategy, states, params, valid=None) -> FloquetSpectrum:
    """Branch Floquet spectrum with one static strategy.

    Padding (``valid`` false) is replaced by a valid seed before evaluation,
    so no collapsed orbit is decomposed, and masked afterwards.
    """
    pargs = problem.args
    if valid is not None:
        first = jnp.argmax(valid)
        states = jnp.where(valid[:, None], states, states[first])
        params = jnp.where(valid, params, params[first])
    spectrum = _floquet._branch_spectrum(
        strategy, pargs.raw_f, pargs.mesh, states, params, pargs.user_args)
    if valid is None:
        return spectrum
    column = valid[:, None]
    return FloquetSpectrum(
        values=jnp.where(column, spectrum.values, 0),
        phase=jnp.where(column, spectrum.phase, 0),
        logabs=jnp.where(column, spectrum.logabs, -jnp.inf),
        trivial_index=jnp.where(valid, spectrum.trivial_index, -1),
        converged=spectrum.converged & valid,
    )


def _branch_spectra(problem, settings, events, solvers, states, params, valid=None):
    """``(eigenvalues, stability, trivial_index, floquet_logabs, floquet_converged)``."""
    from jaxcont.core.scan_continuation import branch_eigenvalues

    # Curve-kind problems never need Branch.eigenvalues: the branch state is
    # an extended-system vector, and codim-2 events compute the original
    # spectrum from their own raw_f.
    if problem.kind in ("fold_curve", "hopf_curve") or not (
            settings.compute_stability or len(events) > 0) or states.shape[0] == 0:
        return None, None, None, None, None
    if problem.kind == "periodic":
        spectrum = _periodic_spectrum(problem, solvers.floquet, states, params, valid)
        stability = _floquet.floquet_stable(
            spectrum.values, spectrum.trivial_index, logabs=spectrum.logabs) & spectrum.converged
        return (spectrum.values, stability, spectrum.trivial_index,
                spectrum.logabs, spectrum.converged)
    if valid is not None:
        states = jnp.where(valid[:, None], states, states[0])
        params = jnp.where(valid, params, params[0])
    eigenvalues = branch_eigenvalues(
        problem.f, states, params, problem.args, eigen_solver=solvers.eigen
    )
    stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)
    if valid is not None:
        stability = stability & valid
        eigenvalues = jnp.where(valid[:, None], eigenvalues, 0)
    return eigenvalues, stability, None, None, None


def _assemble_leg(
    res,
    problem: BifProblem,
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
):
    """Trim one engine leg and compute its branch spectra."""
    from jaxcont.core.scan_continuation import Status

    n = int(res.n_valid)
    if int(res.status) == int(Status.SEED_FAILED):
        raise ValueError(
            "continuation() seed failed: Newton did not converge at the "
            "starting point (u0, p0) -- the initial guess is not on a "
            "solution branch (singular Jacobian or too far away). Refine u0 "
            "first or pass a better initial guess."
        )
    states = res.states[:n]
    params = res.params[:n]
    tangents = res.tangents[:n]

    eigenvalues, stability, trivial_index, logabs, floquet_ok = _branch_spectra(
        problem, settings, events, solvers, states, params)

    converged = np.asarray(res.converged[:n])
    iters = np.asarray(res.iters[:n])
    ds = np.asarray(res.ds[:n])
    convergence_info = [
        {
            "step": i,
            "converged": bool(converged[i]),
            "newton_iters": int(iters[i]),
            "ds": float(ds[i]),
        }
        for i in range(n)
    ]

    hits: list[EventHit] = []

    floquet = None if logabs is None else (logabs, floquet_ok)
    return states, params, tangents, eigenvalues, stability, trivial_index, floquet, convergence_info, hits


def _finish_result(
    problem: BifProblem,
    states,
    params,
    tangents,
    eigenvalues,
    stability,
    trivial_index,
    floquet,
    convergence_info,
    hits: list[EventHit],
    verbose: bool,
    status: str,
    n_attempts: int,
) -> ContinuationResult:
    sol = ContinuationSolution(
        states=states,
        parameters=params,
        tangent_vectors=tangents,
        eigenvalues=eigenvalues,
        stability=stability,
        convergence_info=convergence_info,
        state_names=problem.state_names,
        param_name=problem.param_name,
    )
    # sol.bifurcations stays dict-shaped: viz/core.py's plotting and
    # ContinuationSolution.get_bifurcations_by_type both read
    # bif.get("type")/bif.get("parameter")/bif.get("state") directly.
    validated = [h for h in hits if h.status == "validated"]
    rejected = [h for h in hits if h.status == "failed"]
    sol.bifurcations = [
        {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
        for h in validated
    ]

    if verbose:
        n_converged = sum(1 for c in convergence_info if c["converged"])
        kind_counts: dict[str, int] = {}
        for b in sol.bifurcations:
            kind_counts[b["type"]] = kind_counts.get(b["type"], 0) + 1
        bif_summary = (
            ", ".join(f"{count} {kind}" for kind, count in sorted(kind_counts.items()))
            or "none"
        )
        print(
            f"continuation(): {len(params)} points ({n_converged} converged), "
            f"bifurcations: {bif_summary}"
        )

    result = _to_result(sol)
    result.branch.curve_layout = problem.curve_layout
    result.branch.trivial_index = trivial_index
    if floquet is not None:
        result.branch.floquet_logabs, result.branch.floquet_converged = floquet
    result.rejected_events = rejected
    result.status = status
    result.stats["n_attempts"] = n_attempts
    result.stats["status"] = status
    return result


def _run_scan(
    scan_fn,
    problem: BifProblem,
    p_span: tuple[float, float],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
    verbose: bool,
) -> ContinuationResult:
    """
    Run ``scan_fn`` (``pseudo_arclength_scan`` or ``natural_scan``) and
    reassemble a legacy-shaped :class:`ContinuationSolution` so
    detection/plotting reuse existing code.
    """
    p_start, p_end = p_span
    if not isinstance(p_start, jax.core.Tracer) and not isinstance(p_end, jax.core.Tracer):
        if float(p_start) == float(p_end):
            raise ValueError("p_span must have nonzero length")
    u0 = jnp.asarray(problem.u0)
    dtype = u0.dtype

    if problem.kind in ("fold_curve", "hopf_curve"):
        if settings.compute_stability:
            raise ValueError(
                f"compute_stability=True is not meaningful for "
                f"kind={problem.kind!r}: the branch state is an extended-"
                f"system vector, so eigendecomposing its Jacobian does not "
                f"give the original system's spectrum. Pass "
                f"ContinuationPar(compute_stability=False); the codim-2 "
                f"events in bifurcations/codim2_events.py carry their own "
                f"raw_f and compute the original spectrum themselves."
            )

    if settings.bothside:
        return _run_scan_bothside(
            scan_fn, problem, u0, p_span, settings, events, solvers, verbose, dtype,
        )

    try:
        # Check p_span[0] == problem.p0 only in eager mode; traced calls
        # (jax.vmap/jax.jit) have p_start as a tracer and would raise
        # TracerBoolConversionError on the jnp.allclose result.
        if not jnp.allclose(jnp.asarray(p_start, dtype), problem.p0):
            raise ValueError(
                f"p_span[0]={float(p_start)} must equal the problem's starting "
                f"parameter p0={float(problem.p0)}. continuation() treats "
                f"p_span[0] as the literal starting value rather than reading "
                f"it off the problem, so a mismatch would start the run at a "
                f"point that is not the problem's actual seed."
            )
    except jax.errors.ConcretizationTypeError:
        # Traced call: p_start cannot be concretized to test equality, so
        # the check is deferred to eager execution. Skip it here.
        pass

    res = _call_scan(scan_fn, problem, u0, p_start, p_end, settings, solvers, dtype)

    try:
        int(res.n_valid)
    except jax.errors.ConcretizationTypeError:
        # Traced call (jax.vmap/jax.jit over this problem/settings): n_valid
        # can't become a concrete Python int, so there is no single trim
        # length. Fall back to the fixed-size-buffer + mask representation.
        return _run_scan_traced(res, problem, settings, events, solvers)

    states, params, tangents, eigenvalues, stability, trivial_index, floquet, convergence_info, hits = _assemble_leg(
        res, problem, settings, events, solvers
    )
    return _finish_result(
        problem, states, params, tangents, eigenvalues, stability, trivial_index, floquet,
        convergence_info, hits, verbose,
        _status_name(res.status), int(res.n_attempts),
    )


def _run_scan_bothside(
    scan_fn,
    problem: BifProblem,
    u0: Array,
    p_span: tuple[float, float],
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
    verbose: bool,
    dtype,
) -> ContinuationResult:
    """
    ``ContinuationPar(bothside=True)``: explore both directions from
    ``problem.p0`` in one call (issue #8), by running ``scan_fn`` twice --
    once toward each bound of ``p_span`` -- and stitching the two legs into
    one branch. No changes to core/scan_continuation.py: direction there is
    already just ``sign(p_end - p0)``.

    Eager-only: assemble states before scanning so candidate indices, mode
    orientation and sampled-zero ownership follow the assembled branch.
    """
    p_a, p_b = p_span
    try:
        p_a_f = float(p_a)
        p_b_f = float(p_b)
        p0_f = float(problem.p0)
    except jax.errors.ConcretizationTypeError:
        raise NotImplementedError(
            "ContinuationPar(bothside=True) is not supported when "
            "continuation() runs inside jax.vmap/jax.jit: it needs two "
            "full Python-level scan_fn calls plus list-based stitching of "
            "the two legs, none of which is traceable. Call continuation() "
            "eagerly per problem instead."
        )

    p_lo, p_hi = min(p_a_f, p_b_f), max(p_a_f, p_b_f)
    if not (p_lo <= p0_f <= p_hi):
        raise ValueError(
            f"bothside=True requires problem.p0={p0_f} to lie within "
            f"p_span's bounds [{p_lo}, {p_hi}]."
        )

    if p0_f == p_lo or p0_f == p_hi:
        bound = p_hi if p0_f == p_lo else p_lo
        res = _call_scan(scan_fn, problem, u0, p0_f, bound, settings, solvers, dtype)
        states, params, tangents, eig, stab, trivial, floquet, conv, hits = _assemble_leg(
            res, problem, settings, events, solvers
        )
        if p0_f == p_hi:
            n = len(conv) - 1
            def reverse(array):
                return None if array is None else jnp.flip(array, axis=0)
            states, params, tangents = reverse(states), reverse(params), -reverse(tangents)
            eig, stab, trivial = reverse(eig), reverse(stab), reverse(trivial)
            floquet = None if floquet is None else tuple(reverse(a) for a in floquet)
            conv = [{**c, "step": i} for i, c in enumerate(reversed(conv))]
            hits = [replace(h, index=None if h.index is None else tuple(n - i for i in h.index))
                    for h in hits]
        return _finish_result(
            problem, states, params, tangents, eig, stab, trivial, floquet, conv, hits,
            verbose, _status_name(res.status), int(res.n_attempts),
        )

    fwd_res = _call_scan(scan_fn, problem, u0, p0_f, p_hi, settings, solvers, dtype)
    bwd_res = _call_scan(scan_fn, problem, u0, p0_f, p_lo, settings, solvers, dtype)

    f_states, f_params, f_tangents, f_eig, f_stab, f_trivial, f_floquet, f_conv, f_hits = _assemble_leg(
        fwd_res, problem, settings, events, solvers
    )
    b_states, b_params, b_tangents, b_eig, b_stab, b_trivial, b_floquet, b_conv, b_hits = _assemble_leg(
        bwd_res, problem, settings, events, solvers
    )

    # Reverse the backward leg (dropping its index-0 point -- the same seed
    # the forward leg's index 0 already is) so arc length runs low -> high
    # across the stitch, then prepend it to the forward leg.
    def _stitch(back, fwd):
        if back is None or fwd is None:
            return None
        return jnp.concatenate([jnp.flip(back[1:], axis=0), fwd], axis=0)

    states = _stitch(b_states, f_states)
    params = _stitch(b_params, f_params)
    tangents = _stitch(-b_tangents, f_tangents)
    eigenvalues = _stitch(b_eig, f_eig)
    stability = _stitch(b_stab, f_stab)
    trivial_index = _stitch(b_trivial, f_trivial)
    floquet = None if f_floquet is None or b_floquet is None else tuple(
        _stitch(b, f) for b, f in zip(b_floquet, f_floquet))

    n_back = len(b_conv) - 1  # excludes the shared seed point
    convergence_info = [
        {**c, "step": i} for i, c in enumerate(reversed(b_conv[1:]))
    ] + [
        {**c, "step": i + n_back} for i, c in enumerate(f_conv)
    ]

    hits = []

    # Report the first leg that did not simply reach its end.
    status = next(
        (n for n in (_status_name(fwd_res.status), _status_name(bwd_res.status))
         if n != "reached_end"),
        "reached_end",
    )

    return _finish_result(
        problem, states, params, tangents, eigenvalues, stability, trivial_index, floquet,
        convergence_info, hits, verbose,
        status, int(fwd_res.n_attempts) + int(bwd_res.n_attempts),
    )


def _run_scan_traced(
    res,
    problem: BifProblem,
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
) -> ContinuationResult:
    """
    ``_run_scan``'s path when ``res.n_valid`` is a tracer (called inside
    ``jax.vmap``/``jax.jit``). No concrete trim length exists, so the fixed-
    size engine buffers are returned as-is with a ``valid`` mask instead of
    the legacy ``ContinuationSolution``/``detect_events`` machinery,
    neither of which is traceable (Python loops, ``float()``, ``list.sort()``).
    """
    states, params, tangents = res.states, res.params, res.tangents
    valid = jnp.arange(states.shape[0]) < res.n_valid
    eigenvalues, stability, trivial_index, logabs, floquet_ok = _branch_spectra(
        problem, settings, events, solvers, states, params, valid)

    branch = Branch(
        params=params,
        states=states,
        tangents=tangents,
        eigenvalues=eigenvalues,
        stable=stability,
        trivial_index=trivial_index,
        floquet_logabs=logabs,
        floquet_converged=floquet_ok,
        valid=valid,
        curve_layout=problem.curve_layout,
    )
    return ContinuationResult(
        branch=branch, events=[], stats={
            "n_valid": res.n_valid,
            "n_attempts": res.n_attempts,
            "status": res.status,
        },
        _solution=None,
    )


def _to_result(sol: ContinuationSolution) -> ContinuationResult:
    branch = Branch(
        params=sol.parameters,
        states=sol.states,
        tangents=sol.tangent_vectors,
        eigenvalues=sol.eigenvalues,
        stable=sol.stability,
    )
    events = [
        EventHit(
            kind=b.get("type", "unknown"),
            p=float(b.get("parameter", 0.0)),
            u=b.get("state"),
            index=b.get("index"),
            info={k: v for k, v in b.items()
                  if k not in ("type", "parameter", "state", "index")},
        )
        for b in (sol.bifurcations or [])
    ]
    stats = {
        "n_points": sol.n_points,
        "converged_steps": sum(
            1 for c in (sol.convergence_info or []) if c.get("converged")
        ),
    }
    return ContinuationResult(branch=branch, events=events, stats=stats, _solution=sol)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def continuation(
    problem: BifProblem,
    alg: ContinuationAlgorithm = PseudoArclength(),
    *,
    p_span: tuple[float, float],
    settings: ContinuationPar = ContinuationPar(),
    events: Sequence[Event] = (),
    solvers: Solvers = Solvers(),
    verbose: bool = False,
) -> ContinuationResult:
    """
    Continue a solution branch of ``problem`` across ``p_span``.

    Args:
        problem: the :class:`BifProblem` to continue.
        alg: :class:`PseudoArclength` (default) or :class:`Natural`.
        p_span: ``(p_start, p_end)`` range for the continuation parameter.
        settings: numerical settings (:class:`ContinuationPar`).
        events: detectors to run along the branch (e.g. ``[Fold(), Hopf()]``).
            An empty list disables detection.
        solvers: linear-algebra bundle (:class:`Solvers`); defaults to dense
            direct solves (``Dense()``/``DenseEigen()``).
        verbose: print a bifurcation summary.

    Returns:
        :class:`ContinuationResult` with ``.branch`` and ``.events``.
    """
    from jaxcont.core.scan_continuation import natural_scan, pseudo_arclength_scan

    if settings.loop not in ("while", "scan"):
        raise ValueError("loop must be 'while' or 'scan'")
    if settings.loop == "scan":
        from jaxcont.core.scan_continuation import natural_lax_scan, pseudo_arclength_lax_scan
        natural_scan, pseudo_arclength_scan = natural_lax_scan, pseudo_arclength_lax_scan
    if isinstance(alg, Natural):
        result = _run_scan(natural_scan, problem, p_span, settings, events, solvers, False)
    elif isinstance(alg, PseudoArclength):
        result = _run_scan(pseudo_arclength_scan, problem, p_span, settings, events, solvers, False)
    else:
        raise TypeError(f"Unknown continuation algorithm: {alg!r}")

    from jaxcont.bifurcations.event_scan import EventScanInput, scan_event_candidates
    branch = result.branch
    traced = isinstance(branch.states, jax.core.Tracer)
    events = _bind_floquet(events, solvers)
    # Eager custom events without array hooks use the Python compatibility path.
    use_arrays = tuple(traced or getattr(e, "scan_arrays", None) is not None
                       for e in events)
    array_events = tuple(e for e, use in zip(events, use_arrays) if use)
    strategies, spectra = _event_floquet_spectra(problem, branch, array_events, solvers)
    data = EventScanInput(branch.params, branch.states, branch.tangents,
                          branch.eigenvalues, branch.trivial_index,
                          jnp.ones(branch.params.shape, bool) if branch.valid is None else branch.valid,
                          branch.curve_layout, strategies, spectra)
    scanned = iter(scan_event_candidates(array_events, data, problem.args))
    result.event_candidates = tuple(next(scanned) if use else None for use in use_arrays)
    if not traced and events:
        result = refine_event_candidates(result, problem, events, solvers=solvers)
    if verbose and not traced:
        counts: dict[str, int] = {}
        for hit in result.events:
            counts[hit.kind] = counts.get(hit.kind, 0) + 1
        summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items())) or "none"
        print(f"continuation(): {branch.n_valid} points "
              f"({result.stats['converged_steps']} converged), bifurcations: {summary}")
    return result


def _bind_floquet(events: Sequence[Event], solvers: Solvers) -> tuple:
    """PD/NS events without an explicit ``floquet_solver`` use ``solvers.floquet``."""
    return tuple(
        replace(e, floquet_solver=solvers.floquet)
        if isinstance(e, (PeriodDoubling, NeimarkSacker)) and e.floquet_solver is None else e
        for e in events)


def _event_floquet_spectra(problem, branch, events, solvers):
    """Static strategy keys and one branch spectrum per distinct effective
    Floquet strategy; the bundle strategy reuses the branch spectrum."""
    if problem.kind != "periodic" or branch.floquet_converged is None:
        return (), ()
    values = branch.eigenvalues
    magnitude = jnp.abs(values)
    phase = jnp.where(magnitude > 0, values / jnp.where(magnitude > 0, magnitude, 1), 0)
    strategies = [solvers.floquet]
    spectra = [FloquetSpectrum(values, phase, branch.floquet_logabs, branch.trivial_index,
                               branch.floquet_converged)]
    for event in events:
        strategy = getattr(event, "floquet_solver", None)
        if isinstance(event, (PeriodDoubling, NeimarkSacker)) and strategy not in strategies:
            strategies.append(strategy)
            spectra.append(_periodic_spectrum(problem, strategy, branch.states, branch.params,
                                              branch.valid))
    return tuple(strategies), tuple(spectra)


def refine_event_candidates(result: ContinuationResult, problem: BifProblem,
                            events: Sequence[Event], *, tolerance=None,
                            max_iterations=50, solvers=Solvers()) -> ContinuationResult:
    """Refine unvalidated candidates on one unbatched, eager result.

    Slice batched result pytrees before calling. Use the detection solver bundle:
    PD/NS events without their own ``floquet_solver`` refine with
    ``solvers.floquet``, explicit event strategies are kept.
    Bracket args are reconstructed from the left endpoint's step hook.
    """
    branch = result.branch
    if isinstance(branch.states, jax.core.Tracer) or branch.states.ndim != 2:
        raise TypeError("refine_event_candidates requires one unbatched eager result")
    if len(events) != len(result.event_candidates):
        raise ValueError("events must match the candidate tuple used for detection")
    events = _bind_floquet(events, solvers)
    hits = detect_events(events, branch.params, branch.states, branch.tangents,
        branch.eigenvalues, lambda u,p: problem.f(u,p,problem.args), ds=0.,
        trivial_index=branch.trivial_index, curve_layout=branch.curve_layout,
        args=problem.args, step_update=problem.step_update,
        f=problem.f, tolerance=tolerance, max_iterations=max_iterations,
        candidates=result.event_candidates)
    refined = replace(result, events=[h for h in hits if h.status == "validated"],
                      rejected_events=[h for h in hits if h.status == "failed"])
    if result._solution is not None:
        import copy
        solution = copy.copy(result._solution)
        solution.bifurcations = [
            {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
            for h in refined.events]
        refined._solution = solution
    return refined
