"""
Functional, diffrax-style public API for JaxCont (the "Sketch A" spine).

This is the blessed surface going forward:

    prob = jaxcont.bif_problem(f, u0, p0)
    sol  = jaxcont.continuation(prob, p_span=(0.0, 1.0), events=[jaxcont.Fold()])

For now it is a thin adapter over the existing predictor-corrector loop
(`core/pseudo_arclength.py`, `core/natural_continuation.py`); the numerics are
unchanged. Later versions migrate the loop internals to `lax.scan` and expose
`vmap`/`grad` end-to-end (see notes/ARCHITECTURE.md §2, §3) without changing
these signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution
from jaxcont.core.pseudo_arclength import PseudoArclengthContinuation
from jaxcont.core.natural_continuation import NaturalContinuation

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

    Registered as a JAX PyTree: ``u0``, ``p0``, ``args`` are dynamic leaves;
    ``f`` and ``kind`` are static. This makes ``jax.vmap(..., in_axes=...)`` over
    a batched problem structurally valid (full end-to-end vmap awaits the
    lax.scan loop rewrite; see ARCHITECTURE.md §3.1).
    """

    f: Callable[[Array, Array, PyTree], Array]
    u0: Array
    p0: Array
    args: PyTree = None
    kind: Literal["equilibrium", "periodic", "bvp"] = "equilibrium"

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
    aux = (prob.f, prob.kind)
    return children, aux


def _bifproblem_unflatten(aux, children):
    f, kind = aux
    u0, p0, args = children
    return BifProblem(f=f, u0=u0, p0=p0, args=args, kind=kind)


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
) -> BifProblem:
    """
    Front-door factory for a :class:`BifProblem` (mirrors lyapax's
    ``ode_problem``). Coerces ``u0``/``p0`` to JAX arrays.
    """
    return BifProblem(
        f=f,
        u0=jnp.asarray(u0),
        p0=jnp.asarray(p0, dtype=jnp.asarray(u0).dtype),
        args=args,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Algorithms & settings
# ---------------------------------------------------------------------------

class ContinuationAlgorithm:
    """Marker base for continuation algorithms."""


@dataclass(frozen=True)
class PseudoArclength(ContinuationAlgorithm):
    """Pseudo-arclength continuation (default; passes fold points)."""


@dataclass(frozen=True)
class Natural(ContinuationAlgorithm):
    """Natural-parameter continuation (simple; stalls at folds)."""


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
    """Marker base for a bifurcation/event detector."""

    #: legacy detector key this event maps onto ("fold" | "hopf")
    _kind: str = ""


@dataclass(frozen=True)
class Fold(Event):
    _kind: str = "fold"


@dataclass(frozen=True)
class Hopf(Event):
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
    """The computed solution branch (one connected curve)."""

    params: Array           # (n_valid,)
    states: Array           # (n_valid, state_dim)
    tangents: Optional[Array] = None
    eigenvalues: Optional[Array] = None
    stable: Optional[Array] = None

    @property
    def n_valid(self) -> int:
        return int(self.params.shape[0])

    def at_param(self, p: float) -> Tuple[float, Array]:
        """Return the ``(param, state)`` on the branch closest to ``p``."""
        idx = int(jnp.argmin(jnp.abs(self.params - p)))
        return float(self.params[idx]), self.states[idx]


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
    if isinstance(alg, Natural):
        runner_cls = NaturalContinuation
    elif isinstance(alg, PseudoArclength):
        runner_cls = PseudoArclengthContinuation
    else:
        raise TypeError(f"Unknown continuation algorithm: {alg!r}")

    detect = len(events) > 0
    runner = runner_cls(
        ds=settings.ds,
        ds_min=settings.ds_min,
        ds_max=settings.ds_max,
        max_steps=settings.max_steps,
        adaptive_stepsize=settings.adaptive,
        newton_tol=settings.newton_tol,
        newton_max_iter=settings.newton_max_iter,
        detect_bifurcations=detect,
        compute_stability=settings.compute_stability,
        verbose=verbose,
    )

    legacy_problem = _to_legacy_problem(problem)
    sol = runner.run(legacy_problem, param_range=p_span)
    result = _to_result(sol)

    # Filter detected events to those requested, if the user narrowed the set.
    requested = {e._kind for e in events if getattr(e, "_kind", "")}
    if requested:
        result.events = [h for h in result.events if h.kind in requested]

    return result
