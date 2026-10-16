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
from jaxcont.core.continuation import ContinuationSolution
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver
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
    "Fold",
    "Hopf",
    "Natural",
    "NeimarkSacker",
    "PeriodDoubling",
    "PseudoArclength",
    "Solvers",
    "bif_problem",
    "continuation",
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
    kind: Literal[
        "equilibrium", "periodic", "bvp", "fold_curve", "hopf_curve"
    ] = "equilibrium"
    state_names: Optional[tuple[str, ...]] = None
    param_name: Optional[str] = None

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
    newton_tol: Optional[float] = None
    newton_max_iter: int = 20
    max_attempts: Optional[int] = None   # total corrector attempts; None -> 4 * max_steps
    compute_stability: bool = True
    bothside: bool = False


@dataclass(frozen=True)
class Solvers:
    """Pluggable linear-algebra bundle for continuation() (ARCHITECTURE.md §4.6).

    Dense()/DenseEigen() are the only implementations today; the boundary
    exists so a future GMRES()/Arnoldi() (large systems) or ChebyshevDDE()
    (the DDE eigensolver seam, ARCHITECTURE.md §10.2) can swap in without
    touching the continuation loop. Never itself crosses a jax.jit boundary
    -- only its .linear/.eigen fields do, as static arguments to
    pseudo_arclength_scan/natural_scan.
    """

    linear: LinearSolver = Dense()
    eigen: EigenSolver = DenseEigen()


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
    """

    params: Array           # (n_valid,) eager, (buffer_len,) traced
    states: Array           # (n_valid, state_dim) eager, (buffer_len, state_dim) traced
    tangents: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    stable: Optional[Array] = None
    valid: Optional[Array] = None   # bool mask, only set when traced

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
    # ``events`` holds only validated hits; refinements that failed (see
    # ``EventHit.status``) are kept in ``rejected_events`` with ``info["reason"]``.
    events: list[EventHit] = field(default_factory=list)
    rejected_events: list[EventHit] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _solution: Optional[ContinuationSolution] = None
    # Lower-case ``Status`` name (e.g. "reached_end"); None under jit/vmap,
    # where ``stats["status"]`` holds the integer ``Status`` code instead.
    status: Optional[str] = None

    def plot(self, **kwargs):
        """Plot the bifurcation diagram (delegates to the legacy solution)."""
        if self._solution is None:
            raise RuntimeError("No underlying solution to plot.")
        return self._solution.plot(**kwargs)


def _continuation_result_flatten(r: ContinuationResult):
    children = (r.branch, r.events, r.rejected_events, r.stats, r._solution)
    return children, r.status


def _continuation_result_unflatten(aux, children):
    branch, events, rejected_events, stats, solution = children
    return ContinuationResult(
        branch=branch, events=events, rejected_events=rejected_events,
        stats=stats, _solution=solution, status=aux,
    )


jax.tree_util.register_pytree_node(
    ContinuationResult, _continuation_result_flatten, _continuation_result_unflatten
)


# ---------------------------------------------------------------------------
# Adapters (BifProblem <-> legacy ContinuationProblem/Solution)
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


def _engine_f_args(problem: BifProblem):
    """Return ``(f, args)`` for the scan engine.

    Only ``jax.Array`` / ``np.ndarray`` leaves of ``problem.args`` are dynamic
    engine inputs (compile-cache reuse across changed values). Every other leaf
    -- Python scalars, callables such as the one periodic problems carry -- is
    bound statically and stays concrete inside ``f``, so it may be used as a
    shape, loop bound or ``if`` flag; changing its value recompiles.
    """
    leaves, treedef = jax.tree_util.tree_flatten(problem.args)
    mask = tuple(isinstance(x, (jax.Array, np.ndarray)) for x in leaves)
    if all(mask):
        return problem.f, problem.args
    dyn = [x for x, m in zip(leaves, mask) if m]
    static = tuple(x for x, m in zip(leaves, mask) if not m)
    return _BoundF(problem.f, treedef, mask, static), dyn


def _call_scan(scan_fn, problem: BifProblem, u0, p_from, p_to, settings: ContinuationPar, solvers: Solvers, dtype):
    f, args = _engine_f_args(problem)
    return scan_fn(
        f,
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
        jnp.asarray(settings.newton_max_iter),
        solvers.linear,
        jnp.asarray(settings.adaptive),
        jnp.asarray(
            4 * int(settings.max_steps) if settings.max_attempts is None
            else settings.max_attempts,
            jnp.int32,
        ),
    )


def _status_name(code) -> str:
    from jaxcont.core.scan_continuation import Status

    return Status(int(code)).name.lower()


def _assemble_leg(
    res,
    problem: BifProblem,
    settings: ContinuationPar,
    events: Sequence[Event],
    solvers: Solvers,
):
    """
    Trim one ``scan_fn`` result to its ``n_valid`` real points, compute
    eigenvalues/stability, and detect events on it in isolation.

    Isolation matters for :func:`_run_scan_bothside`: each leg starts at the
    problem's own seed point, so scanning it for events independently (as
    opposed to scanning a single array stitched from both legs) never lets a
    fold/Hopf test function see the artificial tangent-direction reversal at
    the stitch seam.
    """
    from jaxcont.core.scan_continuation import Status, branch_eigenvalues
    from jaxcont.stability.floquet import branch_floquet_multipliers, floquet_stable

    rhs2 = lambda u, p: problem.f(u, p, problem.args)
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

    # Curve-kind problems (fold_curve/hopf_curve) never need Branch.eigen-
    # values: the branch state is an extended-system vector, so eigen-
    # decomposing its Jacobian wouldn't give the original system's
    # spectrum anyway (see the compute_stability=True guard in _run_scan),
    # and every codim-2 event in bifurcations/codim2_events.py carries its
    # own raw_f and computes the original spectrum itself via
    # BranchPoint.u -- it never reads BranchPoint.eigenvalues. Computing
    # eigenvalues here whenever events=[...] was non-empty (the old
    # `len(events) > 0` clause) wasted real time re-eigendecomposing a
    # Jacobian nothing reads.
    is_curve_kind = problem.kind in ("fold_curve", "hopf_curve")
    eigenvalues = None
    stability = None
    want_eigs = not is_curve_kind and (settings.compute_stability or len(events) > 0)
    if want_eigs and states.shape[0] > 0:
        if problem.kind == "periodic":
            _, _, raw_f, mesh = problem.args
            eigenvalues = branch_floquet_multipliers(
                raw_f, mesh, states, params, eigen_solver=solvers.eigen
            )
            stability = floquet_stable(eigenvalues)
        else:
            eigenvalues = branch_eigenvalues(
                problem.f, states, params, problem.args, eigen_solver=solvers.eigen
            )
            stability = jnp.all(jnp.real(eigenvalues) < 0.0, axis=1)

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

    # Curve-kind problems intentionally have eigenvalues=None (see above)
    # but their codim-2 events are self-contained and never read
    # BranchPoint.eigenvalues, so detection must still run for them.
    hits: list[EventHit] = []
    if len(events) > 0 and (eigenvalues is not None or is_curve_kind):
        hits = detect_events(
            events, params, states, tangents, eigenvalues, rhs2,
            ds=float(settings.ds),
        )

    return states, params, tangents, eigenvalues, stability, convergence_info, hits


def _finish_result(
    problem: BifProblem,
    states,
    params,
    tangents,
    eigenvalues,
    stability,
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

    states, params, tangents, eigenvalues, stability, convergence_info, hits = _assemble_leg(
        res, problem, settings, events, solvers
    )
    return _finish_result(
        problem, states, params, tangents, eigenvalues, stability,
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

    Eager-only, like ``events=[...]``: needs two full Python-level
    ``scan_fn`` calls plus list-based stitching (and, with events, of their
    independently-detected hit lists), none of which is traceable.
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

    fwd_res = _call_scan(scan_fn, problem, u0, p0_f, p_hi, settings, solvers, dtype)
    bwd_res = _call_scan(scan_fn, problem, u0, p0_f, p_lo, settings, solvers, dtype)

    f_states, f_params, f_tangents, f_eig, f_stab, f_conv, f_hits = _assemble_leg(
        fwd_res, problem, settings, events, solvers
    )
    b_states, b_params, b_tangents, b_eig, b_stab, b_conv, b_hits = _assemble_leg(
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

    n_back = len(b_conv) - 1  # excludes the shared seed point
    convergence_info = [
        {**c, "step": i} for i, c in enumerate(reversed(b_conv[1:]))
    ] + [
        {**c, "step": i + n_back} for i, c in enumerate(f_conv)
    ]

    # Each leg is scanned for events in isolation (see _assemble_leg's
    # docstring) -- both start at the same p0, so their hit lists never
    # need cross-leg deduping; a genuine bifurcation can only land in
    # whichever leg's disjoint parameter interval actually contains it.
    hits = [
        replace(h, index=None if h.index is None else tuple(n_back - i for i in h.index))
        for h in b_hits
    ] + [
        replace(h, index=None if h.index is None else tuple(n_back + i for i in h.index))
        for h in f_hits
    ]

    # Report the first leg that did not simply reach its end.
    status = next(
        (n for n in (_status_name(fwd_res.status), _status_name(bwd_res.status))
         if n != "reached_end"),
        "reached_end",
    )

    return _finish_result(
        problem, states, params, tangents, eigenvalues, stability,
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
    if len(events) > 0:
        raise NotImplementedError(
            "events=[...] is not supported when continuation() runs inside "
            "jax.vmap/jax.jit: detect_events uses Python-level control "
            "flow (loops, list.sort(), float()) that isn't traceable. Call "
            "continuation() without events inside the trace -- e.g. inspect "
            "branch.states/branch.params/branch.valid -- or run it eagerly "
            "per point of interest outside the trace to get events."
        )

    from jaxcont.core.scan_continuation import branch_eigenvalues
    from jaxcont.stability.floquet import branch_floquet_multipliers, floquet_stable

    states, params, tangents = res.states, res.params, res.tangents
    valid = jnp.arange(states.shape[0]) < res.n_valid

    eigenvalues = None
    stability = None
    if settings.compute_stability:
        if problem.kind == "periodic":
            _, _, raw_f, mesh = problem.args
            eigenvalues = branch_floquet_multipliers(
                raw_f, mesh, states, params, eigen_solver=solvers.eigen
            )
            stability = floquet_stable(eigenvalues)
        else:
            eigenvalues = branch_eigenvalues(
                problem.f, states, params, problem.args, eigen_solver=solvers.eigen
            )
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

    if isinstance(alg, Natural):
        return _run_scan(natural_scan, problem, p_span, settings, events, solvers, verbose)
    elif isinstance(alg, PseudoArclength):
        return _run_scan(pseudo_arclength_scan, problem, p_span, settings, events, solvers, verbose)
    else:
        raise TypeError(f"Unknown continuation algorithm: {alg!r}")
