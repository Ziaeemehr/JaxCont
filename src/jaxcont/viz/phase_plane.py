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
