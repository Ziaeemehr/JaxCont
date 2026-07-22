"""
Functional, diffrax-style public API for JaxCont (the "Sketch A" spine).

This is the blessed surface going forward:

    prob = jaxcont.bif_problem(f, u0, p0)
    sol  = jaxcont.continuation(prob, p_span=(0.0, 1.0), events=[jaxcont.Fold()])

Both algorithms (`PseudoArclength()`, the default, and `Natural()`) run on
the fully JIT-compiled whole-loop engines in `core/scan_continuation.py`,
which is what makes `vmap`-batched continuation and `jax.grad`/`jax.jacfwd`
through the analysis possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution

__all__ = [
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "Branch",
    "ContinuationResult",
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

    Registered as a JAX PyTree: ``u0``, ``p0``, ``args`` are dynamic leaves;
    ``f``, ``kind``, ``state_names`` and ``param_name`` are static. This makes
    ``jax.vmap(..., in_axes=...)`` over a batched problem structurally valid
    (full end-to-end vmap awaits the lax.scan loop rewrite; see
    ARCHITECTURE.md §3.1).
    """

    f: Callable[[Array, Array, PyTree], Array]
    u0: Array
    p0: Array
    args: PyTree = None
    kind: Literal["equilibrium", "periodic", "bvp"] = "equilibrium"
    state_names: Optional[Tuple[str, ...]] = None
    param_name: Optional[str] = None

    def at(
        self,
        *,
        u0: Optional[Array] = None,
        p0: Optional[Array] = None,
        args: PyTree = _KEEP,
    ) -> "BifProblem":
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
    aux = (prob.f, prob.kind, prob.state_names, prob.param_name)
    return children, aux


def _bifproblem_unflatten(aux, children):
    f, kind, state_names, param_name = aux
    u0, p0, args = children
    return BifProblem(
        f=f, u0=u0, p0=p0, args=args, kind=kind,
        state_names=state_names, param_name=param_name,
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
) -> BifProblem:
    """
    Front-door factory for a :class:`BifProblem` (mirrors lyapax's
    ``ode_problem``). Coerces ``u0``/``p0`` to JAX arrays.

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
    newton_tol: float = 1e-6
    newton_max_iter: int = 20
    compute_stability: bool = True


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class Event:
    """Marker base for a bifurcation/event detector.

    Naming follows the standard abbreviations used throughout the
    bifurcation-theory literature (see
    ``jaxcont.bifurcations.taxonomy.BIFURCATION_TYPES``) rather than
    inventing new names -- e.g. a fold is ``jc.Fold`` (abbreviation **LP**)
    and a Hopf point is ``jc.Hopf`` (abbreviation **H**).
    """

    #: legacy detector key this event maps onto ("fold" | "hopf")
    _kind: str = ""


@dataclass(frozen=True)
class Fold(Event):
    """A limit point / fold bifurcation of equilibria.

    Abbreviation: **LP** -- ``jaxcont.bifurcations.taxonomy.describe("LP")``.
    """

    _kind: str = "fold"


@dataclass(frozen=True)
class Hopf(Event):
    """A Hopf bifurcation of equilibria.

    Abbreviation: **H** -- ``jaxcont.bifurcations.taxonomy.describe("H")``.
    """

    _kind: str = "hopf"


@dataclass(frozen=True)
class EventHit:
    """A detected event along the branch."""

    kind: str
    p: float
    u: Array
    index: Optional[Tuple[int, int]] = None
    info: dict = field(default_factory=dict)


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
    """

    params: Array           # (n_valid,) eager, (buffer_len,) traced
    states: Array           # (n_valid, state_dim) eager, (buffer_len, state_dim) traced
    tangents: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    stable: Optional[Array] = None
    valid: Optional[Array] = None   # bool mask, only set when traced

    @property
    def n_valid(self) -> int:
        return int(self.params.shape[0])

    def at_param(self, p: float) -> Tuple[float, Array]:
        """Return the ``(param, state)`` on the branch closest to ``p``."""
        idx = int(jnp.argmin(jnp.abs(self.params - p)))
        return float(self.params[idx]), self.states[idx]


def _branch_flatten(b: Branch):
    children = (b.params, b.states, b.tangents, b.eigenvalues, b.stable, b.valid)
    return children, None


def _branch_unflatten(aux, children):
    params, states, tangents, eigenvalues, stable, valid = children
    return Branch(
        params=params, states=states, tangents=tangents,
        eigenvalues=eigenvalues, stable=stable, valid=valid,
    )


jax.tree_util.register_pytree_node(Branch, _branch_flatten, _branch_unflatten)


@dataclass
class ContinuationResult:
    """Return value of :func:`continuation`."""

    branch: Branch
    events: list[EventHit] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _solution: Optional[ContinuationSolution] = None

    def plot(self, **kwargs):
        """Plot the bifurcation diagram (delegates to the legacy solution)."""
        if self._solution is None:
            raise RuntimeError("No underlying solution to plot.")
        return self._solution.plot(**kwargs)


def _continuation_result_flatten(r: ContinuationResult):
    children = (r.branch, r.events, r.stats, r._solution)
    return children, None


def _continuation_result_unflatten(aux, children):
    branch, events, stats, solution = children
    return ContinuationResult(branch=branch, events=events, stats=stats, _solution=solution)


jax.tree_util.register_pytree_node(
    ContinuationResult, _continuation_result_flatten, _continuation_result_unflatten
)


# ---------------------------------------------------------------------------
# Adapters (BifProblem <-> legacy ContinuationProblem/Solution)
# ---------------------------------------------------------------------------

def _to_legacy_problem(problem: BifProblem) -> ContinuationProblem:
    """Wrap a BifProblem's ``f(u, p, args)`` as a legacy ``rhs(u, params)``."""
    f = problem.f
    args = problem.args

    def rhs(u, params):
        return f(u, params[_P_KEY], args)

    return ContinuationProblem(
        rhs=rhs,
        u0=problem.u0,
        params={_P_KEY: float(problem.p0)},
        continuation_param=_P_KEY,
        problem_type=problem.kind,
    )


def _run_scan(
    scan_fn,
    problem: BifProblem,
    p_span: Tuple[float, float],
    settings: ContinuationPar,
    events: Sequence[Event],
    verbose: bool,
) -> ContinuationResult:
    """
    Run ``scan_fn`` (``pseudo_arclength_scan`` or ``natural_scan``) and
    reassemble a legacy-shaped :class:`ContinuationSolution` so
    detection/plotting reuse existing code.
    """
    from jaxcont.core.scan_continuation import branch_eigenvalues

    args = problem.args
    rhs2 = lambda u, p: problem.f(u, p, args)

    p_start, p_end = p_span
    u0 = jnp.asarray(problem.u0)
    dtype = u0.dtype

    res = scan_fn(
        rhs2,
        u0,
        jnp.asarray(p_start, dtype),
        jnp.asarray(p_end, dtype),
        jnp.asarray(settings.ds, dtype),
        jnp.asarray(settings.ds_min, dtype),
        jnp.asarray(settings.ds_max, dtype),
        jnp.asarray(settings.newton_tol, dtype),
        int(settings.max_steps),
        jnp.asarray(settings.newton_max_iter),
    )

    try:
        n = int(res.n_valid)
    except jax.errors.ConcretizationTypeError:
        # Traced call (jax.vmap/jax.jit over this problem/settings): n_valid
        # can't become a concrete Python int, so there is no single trim
        # length. Fall back to the fixed-size-buffer + mask representation.
        return _run_scan_traced(res, rhs2, settings, events)

    states = res.states[:n]
    params = res.params[:n]
    tangents = res.tangents[:n]

    eigenvalues = None
    stability = None
    want_eigs = settings.compute_stability or len(events) > 0
    if want_eigs and states.shape[0] > 0:
        eigenvalues = branch_eigenvalues(rhs2, states, params)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)

    convergence_info = [
        {
            "step": i,
            "converged": bool(res.converged[i]),
            "newton_iters": 0,
            "ds": float(res.ds[i]),
        }
        for i in range(n)
    ]

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

    # Reuse the existing detector on the reassembled solution.
    if len(events) > 0 and eigenvalues is not None:
        from jaxcont.bifurcations.detector import BifurcationDetector

        requested = {e._kind for e in events if getattr(e, "_kind", "")}
        detector = BifurcationDetector(
            detect_fold="fold" in requested,
            detect_hopf="hopf" in requested,
            tolerance=1e-6,
        )
        sol.bifurcations = detector.detect_along_branch(
            sol,
            eigenvalues,
            refine_location=True,
            problem=_to_legacy_problem(problem),
            fold_extended_system=True,
        )

    result = _to_result(sol)
    requested = {e._kind for e in events if getattr(e, "_kind", "")}
    if requested:
        result.events = [h for h in result.events if h.kind in requested]
    return result


def _run_scan_traced(
    res,
    rhs2: Callable[[Array, Array], Array],
    settings: ContinuationPar,
    events: Sequence[Event],
) -> ContinuationResult:
    """
    ``_run_scan``'s path when ``res.n_valid`` is a tracer (called inside
    ``jax.vmap``/``jax.jit``). No concrete trim length exists, so the fixed-
    size engine buffers are returned as-is with a ``valid`` mask instead of
    the legacy ``ContinuationSolution``/``BifurcationDetector`` machinery,
    neither of which is traceable (Python loops, ``float()``, ``list.sort()``).
    """
    if len(events) > 0:
        raise NotImplementedError(
            "events=[...] is not supported when continuation() runs inside "
            "jax.vmap/jax.jit: BifurcationDetector uses Python-level control "
            "flow (loops, list.sort(), float()) that isn't traceable. Call "
            "continuation() without events inside the trace -- e.g. inspect "
            "branch.states/branch.params/branch.valid -- or run it eagerly "
            "per point of interest outside the trace to get events."
        )

    from jaxcont.core.scan_continuation import branch_eigenvalues

    states, params, tangents = res.states, res.params, res.tangents
    valid = jnp.arange(states.shape[0]) < res.n_valid

    eigenvalues = None
    stability = None
    if settings.compute_stability:
        eigenvalues = branch_eigenvalues(rhs2, states, params)
        stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)

    branch = Branch(
        params=params,
        states=states,
        tangents=tangents,
        eigenvalues=eigenvalues,
        stable=stability,
        valid=valid,
    )
    return ContinuationResult(
        branch=branch, events=[], stats={"n_valid": res.n_valid}, _solution=None,
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
    p_span: Tuple[float, float],
    settings: ContinuationPar = ContinuationPar(),
    events: Sequence[Event] = (),
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
        verbose: print a bifurcation summary.

    Returns:
        :class:`ContinuationResult` with ``.branch`` and ``.events``.
    """
    from jaxcont.core.scan_continuation import natural_scan, pseudo_arclength_scan

    if isinstance(alg, Natural):
        return _run_scan(natural_scan, problem, p_span, settings, events, verbose)
    elif isinstance(alg, PseudoArclength):
        return _run_scan(pseudo_arclength_scan, problem, p_span, settings, events, verbose)
    else:
        raise TypeError(f"Unknown continuation algorithm: {alg!r}")
