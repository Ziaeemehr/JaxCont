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
