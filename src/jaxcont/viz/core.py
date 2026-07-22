"""
Core continuation-diagram plots for jaxcont.viz: plot_continuation (single
state variable vs. the parameter) and plot_bifurcation_diagram (alias).
"""

from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont.core.continuation import ContinuationSolution
from jaxcont.viz.styles import style_for


def plot_continuation(
    solution: ContinuationSolution,
    state_index: int = 0,
    state_name: Optional[str] = None,
    param_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_bifurcations: bool = True,
    annotate: bool = False,
    stable_color: str = "blue",
    unstable_color: str = "red",
    **kwargs,
) -> plt.Figure:
    """
    Plot continuation diagram.

    Args:
        solution: Continuation solution
        state_index: Which state variable to plot
        state_name: Label for the plotted state variable. Defaults to
            ``solution.state_names[state_index]`` if the problem defined one,
            else ``"State[<index>]"``.
        param_name: Label for the continuation parameter on the x-axis.
            Defaults to ``solution.param_name`` if the problem defined one,
            else ``"Parameter"``.
        ax: Matplotlib axes (creates new figure if None)
        show_bifurcations: Whether to mark bifurcation points
        annotate: If True, draw a text-box + arrow label next to each
            bifurcation marker showing its type and (parameter, state) value.
        stable_color: Color for stable branches
        unstable_color: Color for unstable branches
        **kwargs: Additional plotting options

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    if state_name is None:
        state_names = getattr(solution, "state_names", None)
        state_name = (
            state_names[state_index] if state_names is not None
            else f"State[{state_index}]"
        )
    if param_name is None:
        param_name = getattr(solution, "param_name", None) or "Parameter"

    params = solution.parameters
    states = solution.states[:, state_index] if solution.state_dim > 1 else solution.states

    if solution.stability is not None:
        stable_mask = solution.stability

        def plot_segments(mask, color, linestyle, label):
            if not jnp.any(mask):
                return

            mask_int = mask.astype(int)
            transitions = jnp.diff(jnp.concatenate([jnp.array([0]), mask_int, jnp.array([0])]))
            starts = jnp.where(transitions == 1)[0]
            ends = jnp.where(transitions == -1)[0]

            label_used = False
            for start, end in zip(starts, ends):
                segment_label = label if not label_used else None
                ax.plot(
                    params[start:end],
                    states[start:end],
                    marker='o',
                    color=color,
                    linestyle=linestyle,
                    label=segment_label,
                    markersize=3,
                    **kwargs,
                )
                label_used = True

        plot_segments(stable_mask, stable_color, '-', 'Stable')
        plot_segments(~stable_mask, unstable_color, '--', 'Unstable')
    else:
        ax.plot(params, states, 'o-', markersize=3, **kwargs)

    if show_bifurcations and solution.bifurcations:
        labeled_types = set()

        for bif in solution.bifurcations:
            bif_type = bif.get("type", "unknown")
            param = bif.get("parameter")
            state = bif.get("state")

            if state is not None and len(state) > state_index:
                state_val = state[state_index]
            else:
                idx = jnp.searchsorted(params, param)
                if idx < len(states):
                    state_val = states[idx]
                else:
                    continue

            style = style_for(bif_type)
            label = style.label if bif_type not in labeled_types else None
            labeled_types.add(bif_type)

            ax.plot(
                param, state_val, style.marker, color=style.color, markersize=10,
                label=label, markeredgecolor='black', markeredgewidth=1,
            )

            if annotate:
                ax.annotate(
                    f"{style.label or bif_type}\n"
                    f"{param_name}={float(param):.3f}\n"
                    f"{state_name}={float(state_val):.3f}",
                    xy=(param, state_val), xytext=(15, 15), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9,
                )

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(state_name, fontsize=12)
    ax.set_title("Continuation Diagram", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_bifurcation_diagram(
    solution: ContinuationSolution,
    state_index: int = 0,
    **kwargs,
) -> plt.Figure:
    """
    Plot bifurcation diagram (alias for plot_continuation).

    Args:
        solution: Continuation solution
        state_index: Which state variable to plot
        **kwargs: Additional plotting options

    Returns:
        Matplotlib figure
    """
    return plot_continuation(solution, state_index=state_index, **kwargs)
