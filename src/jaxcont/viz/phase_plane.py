"""
2D phase-plane visualization for jaxcont.viz: nullclines, vector fields,
streamlines, continuation equilibria, and trajectories.

Grid evaluation is batched with ``jax.vmap``; rendering is ordinary
matplotlib. Nothing in this module is jittable end-to-end or differentiable --
matplotlib is NumPy-land, so this module is too. See
docs/superpowers/specs/2026-07-28-phase-plane-visualization-design.md for the
design rationale, in particular why only 2D systems are supported.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

#: The x-nullcline (f_0 = 0) and y-nullcline (f_1 = 0). Same colorblind-safe
#: pair plot_continuation uses for stable/unstable branches.
X_NULLCLINE_COLOR = "#0072B2"
Y_NULLCLINE_COLOR = "#D55E00"

#: Equilibrium markers; NEUTRAL_COLOR is used when stability is unknown and
#: matches styles.DEFAULT_STYLE.color.
STABLE_COLOR = "#0072B2"
UNSTABLE_COLOR = "#D55E00"
NEUTRAL_COLOR = "#262626"

DEFAULT_FIGSIZE = (6.5, 6.0)


def _require_2d(problem) -> None:
    """Reject problems that are not two-dimensional.

    Slices of higher-dimensional systems are deliberately not drawn: on a
    slice the zero-contours are not nullclines of the full system, their
    intersections are not generally equilibria, and the arrows are a
    projection of the true flow.
    """
    u0 = jnp.asarray(problem.u0)
    if u0.shape != (2,):
        raise NotImplementedError(
            "Phase-plane visualization supports 2D autonomous systems; this "
            f"problem has n={int(u0.size)}. Slices of higher-dimensional "
            "systems are not drawn because their zero-contours are not "
            "nullclines of the full system."
        )


def _state_names(problem) -> Tuple[str, str]:
    """Display names for the two state components, with a neutral fallback."""
    names = getattr(problem, "state_names", None)
    if names is None:
        return ("state[0]", "state[1]")
    return (str(names[0]), str(names[1]))


def _prepare_axes(ax: Optional[plt.Axes], figsize) -> Tuple[plt.Figure, plt.Axes]:
    """Return ``(figure, axes)``, creating a figure when ``ax`` is None."""
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.get_figure(), ax


def _evaluate_field(problem, p, xlim, ylim, resolution: int):
    """Evaluate the frozen right-hand side on a ``resolution**2`` grid.

    Returns ``(X, Y, F)`` as NumPy arrays, with ``X``/``Y`` shaped
    ``(resolution, resolution)`` in matplotlib's "xy" indexing (x varies along
    columns) and ``F`` shaped ``(resolution, resolution, 2)``.
    """
    _require_2d(problem)
    xs = jnp.linspace(xlim[0], xlim[1], resolution)
    ys = jnp.linspace(ylim[0], ylim[1], resolution)
    X, Y = jnp.meshgrid(xs, ys)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    field = jax.vmap(problem.as_rhs(p))(points)
    F = field.reshape(X.shape + (2,))
    return np.asarray(X), np.asarray(Y), np.asarray(F)


def plot_nullclines(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    resolution: int = 200,
    colors: Optional[Tuple[str, str]] = None,
    labels: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Plot the nullclines of a 2D autonomous system at parameter ``p``.

    The nullclines are the zero level sets of each right-hand-side component;
    their intersections are the equilibria.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        resolution: Grid points per axis. Raise it if a nullcline looks jagged.
        colors: ``(x_nullcline_color, y_nullcline_color)``; defaults to the
            module's colorblind-safe pair.
        labels: Whether to set axis labels and draw the legend.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.contour``.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    X, Y, F = _evaluate_field(problem, p, xlim, ylim, resolution)
    fig, ax = _prepare_axes(ax, figsize)

    if colors is None:
        colors = (X_NULLCLINE_COLOR, Y_NULLCLINE_COLOR)
    names = _state_names(problem)

    contour_options = {"linewidths": 2.0}
    contour_options.update(kwargs)

    for index, color in enumerate(colors):
        ax.contour(X, Y, F[..., index], levels=[0.0], colors=[color], **contour_options)
        if labels:
            # A ContourSet carries no legend handle, so add an invisible proxy
            # line whose only job is to appear in the legend.
            ax.plot(
                [], [],
                color=color,
                linewidth=contour_options["linewidths"],
                label=r"$\dot{%s} = 0$" % names[index],
            )

    if labels:
        ax.set_xlabel(names[0], fontsize=12)
        ax.set_ylabel(names[1], fontsize=12)
        ax.legend()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig


def plot_vector_field(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    density: int = 20,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Plot the vector field of a 2D autonomous system as arrows.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        density: Arrows per axis. Coarser than the nullcline grid on purpose.
        normalize: Scale arrows to unit length so direction stays legible
            where the field is stiff, and color them by ``log10`` of the true
            magnitude so speed information is not lost. When False, arrows
            carry their true magnitude in a single color.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.quiver``.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    X, Y, F = _evaluate_field(problem, p, xlim, ylim, density)
    fig, ax = _prepare_axes(ax, figsize)

    U, V = F[..., 0], F[..., 1]
    magnitude = np.hypot(U, V)

    quiver_options = {"pivot": "mid", "width": 0.004}
    quiver_options.update(kwargs)

    if normalize:
        # Leave true zeros at zero rather than dividing by them; a zero-length
        # arrow is the honest rendering of an equilibrium.
        scale = np.where(magnitude == 0.0, 1.0, magnitude)
        speed = np.ma.masked_invalid(
            np.log10(np.where(magnitude == 0.0, np.nan, magnitude))
        )
        quiver_options.setdefault("cmap", "viridis")
        ax.quiver(X, Y, U / scale, V / scale, speed, **quiver_options)
    else:
        quiver_options.setdefault("color", "#7F7F7F")
        ax.quiver(X, Y, U, V, **quiver_options)

    names = _state_names(problem)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig


def plot_streamlines(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    resolution: int = 100,
    color: str = "#7F7F7F",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Plot the flow of a 2D autonomous system as streamlines.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        resolution: Grid points per axis used to seed the integrator.
        color: Streamline color.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.streamplot``.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    X, Y, F = _evaluate_field(problem, p, xlim, ylim, resolution)
    fig, ax = _prepare_axes(ax, figsize)

    # streamplot needs 1-D, evenly spaced coordinates -- exactly the rows and
    # columns _evaluate_field built the grid from.
    stream_options = {"density": 1.0, "linewidth": 0.8, "arrowsize": 0.9}
    stream_options.update(kwargs)
    ax.streamplot(X[0, :], Y[:, 0], F[..., 0], F[..., 1], color=color, **stream_options)

    names = _state_names(problem)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig


def _branch_of(result):
    """Accept a ContinuationResult or a bare Branch, return the Branch."""
    from jaxcont.api import Branch

    if isinstance(result, Branch):
        return result
    branch = getattr(result, "branch", None)
    if branch is None:
        raise TypeError(
            "plot_equilibria expects a ContinuationResult or Branch (this "
            "function consumes continuation output; it does not solve for "
            f"equilibria itself); got {type(result).__name__}."
        )
    return branch


def _valid_arrays(branch):
    """Branch params/states/stability as NumPy, with the buffer mask applied.

    A traced (``jax.vmap``/``jax.jit``) result keeps the full fixed-size engine
    buffer and marks the real points with ``branch.valid``; an eager one is
    already trimmed and has ``valid is None``.
    """
    params = np.asarray(branch.params)
    states = np.asarray(branch.states)
    stable = None if branch.stable is None else np.asarray(branch.stable)
    if branch.valid is not None:
        mask = np.asarray(branch.valid)
        params, states = params[mask], states[mask]
        if stable is not None:
            stable = stable[mask]
    return params, states, stable


def _equilibria_at(branch, p, atol: Optional[float] = None):
    """Every equilibrium on ``branch`` at parameter ``p``.

    Handles a branch that has passed a fold, where two or more points share
    the same ``p`` -- the case ``Branch.at_param`` cannot cover, since it
    returns only the single nearest point.

    Returns ``(states, stable_flags)``: states shaped ``(k, 2)`` and a list of
    ``True``/``False``/``None`` flags, one per state.
    """
    params, states, stable = _valid_arrays(branch)
    if params.size == 0:
        return np.empty((0, states.shape[-1])), []

    d = params - p
    found = []
    flags = []
    crossing_indices = set()

    # Interior crossings, linearly interpolated.
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0.0:
            theta = d[i] / (d[i] - d[i + 1])
            found.append(states[i] + theta * (states[i + 1] - states[i]))
            crossing_indices.add(i)
            crossing_indices.add(i + 1)
            if stable is None:
                flags.append(None)
            elif bool(stable[i]) == bool(stable[i + 1]):
                flags.append(bool(stable[i]))
            else:
                # A stability change between adjacent points means a
                # bifurcation lies in between; draw the conservative style.
                flags.append(False)

    # Points sitting on p without a sign change (e.g. a branch endpoint).
    # Skip indices already claimed by an interior crossing above -- a point
    # adjacent to a crossing is that crossing's own equilibrium, not a second
    # one, and treating it as "sitting on p" would double-count it. The
    # default atol comes from the *local* minimum step, not a trajectory-wide
    # median: a branch that overshoots far past p_span after reversing
    # through a fold (pseudo_arclength_scan has no stop condition for the
    # reversed direction) inflates a global median to the point where it
    # spans the local spacing near p, which is exactly the failure mode this
    # exclusion plus the local-step default both close off.
    if atol is None:
        steps = np.abs(np.diff(params))
        atol = 0.25 * float(np.min(steps)) if steps.size else 0.0
    for i in np.flatnonzero(np.abs(d) <= atol):
        if i in crossing_indices:
            continue
        found.append(states[i])
        flags.append(None if stable is None else bool(stable[i]))

    if not found:
        return np.empty((0, states.shape[-1])), []

    # Dedupe: an exact hit next to a crossing describes the same equilibrium.
    scale = float(np.max(np.abs(states))) if states.size else 1.0
    dedupe_tol = 1e-8 + 1e-6 * max(scale, 1.0)
    kept_states = []
    kept_flags = []
    for point, flag in zip(found, flags):
        if any(np.linalg.norm(point - seen) <= dedupe_tol for seen in kept_states):
            continue
        kept_states.append(point)
        kept_flags.append(flag)

    return np.asarray(kept_states), kept_flags


def plot_equilibria(
    result,
    p,
    *,
    atol: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Mark the equilibria a continuation run found at parameter ``p``.

    Stable equilibria render as filled circles, unstable ones as hollow
    circles -- the convention ``plot_continuation`` already uses. When the
    branch carries no stability information, all points render neutral.

    Both sides of a fold are marked: this selects every branch point at ``p``,
    not just the nearest one.

    Args:
        result: A :class:`~jaxcont.api.ContinuationResult` or
            :class:`~jaxcont.api.Branch`. This function consumes continuation
            output; it does not solve for equilibria itself.
        p: Parameter value at which to mark equilibria.
        atol: Tolerance for treating a branch point as sitting exactly on
            ``p``, excluding points already claimed by an interior crossing.
            Defaults to a quarter of the local minimum parameter step.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.plot``.

    Returns:
        Matplotlib figure.
    """
    branch = _branch_of(result)
    states, flags = _equilibria_at(branch, p, atol=atol)
    fig, ax = _prepare_axes(ax, figsize)

    seen_labels = set()
    for point, flag in zip(states, flags):
        if flag is None:
            color, face, label = NEUTRAL_COLOR, NEUTRAL_COLOR, "equilibrium"
        elif flag:
            color, face, label = STABLE_COLOR, STABLE_COLOR, "stable equilibrium"
        else:
            color, face, label = UNSTABLE_COLOR, "none", "unstable equilibrium"

        marker_options = {
            "marker": "o",
            "markersize": 9,
            "markeredgewidth": 2.0,
            "linestyle": "none",
            "zorder": 5,
        }
        marker_options.update(kwargs)
        ax.plot(
            [point[0]], [point[1]],
            markeredgecolor=color,
            markerfacecolor=face,
            label=None if label in seen_labels else label,
            **marker_options,
        )
        seen_labels.add(label)

    return fig
