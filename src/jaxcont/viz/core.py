"""
Core continuation-diagram plots for jaxcont.viz: plot_continuation (single
state variable vs. the parameter) and plot_bifurcation_diagram (alias).
"""

from typing import Optional, Sequence, Tuple

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
    stable_color: str = "#0072B2",
    unstable_color: str = "#D55E00",
    figsize: Tuple[float, float] = (8.5, 5.25),
    title: Optional[str] = "Continuation Diagram",
    legend: bool = True,
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
        figsize: Figure size when ``ax`` is not supplied
        title: Axes title; pass ``None`` to omit it
        legend: Whether to draw the legend when labeled artists are present
        **kwargs: Additional Matplotlib line options. These override the
            branch defaults, including ``marker``, ``linestyle``, and
            ``linewidth``.

    Returns:
        Matplotlib figure
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=figsize)
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

    line_options = {
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 3.5,
        "markeredgewidth": 0,
        "solid_capstyle": "round",
    }
    line_options.update(kwargs)

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
                options = dict(line_options)
                options.setdefault("color", color)
                options.setdefault("linestyle", linestyle)
                options.setdefault("label", segment_label)
                ax.plot(
                    params[start:end],
                    states[start:end],
                    **options,
                )
                label_used = True

        plot_segments(stable_mask, stable_color, "-", "Stable")
        plot_segments(~stable_mask, unstable_color, "--", "Unstable")
    else:
        options = dict(line_options)
        options.setdefault("color", stable_color)
        options.setdefault("linestyle", "-")
        ax.plot(params, states, **options)

    if show_bifurcations and solution.bifurcations:
        labeled_types = set()
        plotted_points = set()
        plotted_index = 0

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

            # Refinement can occasionally report the same event more than
            # once.  Preserve the solution records, but do not overplot an
            # identical marker and annotation at the same visible point.
            point_key = (bif_type, float(param), float(state_val))
            if point_key in plotted_points:
                continue
            plotted_points.add(point_key)

            style = style_for(bif_type)
            label = style.label if bif_type not in labeled_types else None
            labeled_types.add(bif_type)

            ax.plot(
                param, state_val, style.marker, color=style.color, markersize=10,
                label=label, markeredgecolor="white", markeredgewidth=1.25,
                zorder=5,
            )

            if annotate:
                vertical_offset = 18 if plotted_index % 2 == 0 else -42
                ax.annotate(
                    f"{style.label or bif_type}\n"
                    f"{param_name}={float(param):.3f}\n"
                    f"{state_name}={float(state_val):.3f}",
                    xy=(param, state_val),
                    xytext=(14, vertical_offset),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor=style.color,
                        alpha=0.95,
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=style.color, lw=1.1,
                        shrinkA=3, shrinkB=5,
                    ),
                    color="#262626",
                    fontsize=8.5,
                    linespacing=1.3,
                    zorder=6,
                )
            plotted_index += 1

    ax.set_xlabel(param_name, fontsize=11, labelpad=8)
    ax.set_ylabel(state_name, fontsize=11, labelpad=8)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.65)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6B7280")
    ax.spines["bottom"].set_color("#6B7280")
    ax.tick_params(colors="#374151", labelsize=9.5)
    ax.margins(x=0.025, y=0.08)

    handles, labels = ax.get_legend_handles_labels()
    if legend and labels:
        ax.legend(
            handles,
            labels,
            loc="best",
            frameon=False,
            fontsize=9,
            handlelength=2.4,
            borderaxespad=0.8,
        )

    if created_figure:
        fig.tight_layout(pad=1.2)
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


def plot_all_states(
    solution: ContinuationSolution,
    param_name: Optional[str] = None,
    state_names: Optional[Sequence[str]] = None,
    show_bifurcations: bool = True,
    stable_color: str = "blue",
    unstable_color: str = "red",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Plot every state variable against the continuation parameter, one per
    subplot (replaces hand-rolled per-example subplot loops), sharing a single
    figure-level legend.

    Args:
        solution: Continuation solution
        param_name: Label for the continuation parameter (x-axis of the
            bottom subplot). Defaults to ``solution.param_name`` if set, else
            "Parameter".
        state_names: Per-state y-axis labels; length must equal
            ``solution.state_dim``. Defaults to ``solution.state_names`` if
            set, else ``"State[<index>]"`` for each.
        show_bifurcations: Whether to mark bifurcation points on every subplot
        stable_color: Color for stable branches
        unstable_color: Color for unstable branches
        figsize: Figure size; defaults to ``(8, 3 * state_dim)``

    Returns:
        Matplotlib figure with one subplot per state variable
    """
    n = solution.state_dim
    if state_names is not None and len(state_names) != n:
        raise ValueError(
            f"state_names has {len(state_names)} entries but solution has "
            f"state_dim={n}"
        )

    if figsize is None:
        figsize = (8, 3 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False, sharex=True)
    axes = axes[:, 0]

    for i, ax in enumerate(axes):
        name = state_names[i] if state_names is not None else None
        plot_continuation(
            solution,
            state_index=i,
            state_name=name,
            param_name=param_name,
            ax=ax,
            show_bifurcations=show_bifurcations,
            stable_color=stable_color,
            unstable_color=unstable_color,
        )
        ax.set_title("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if i < n - 1:
            ax.set_xlabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()
    return fig
