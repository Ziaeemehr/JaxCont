"""
Phase-portrait and eigenvalue-trajectory plots for jaxcont.viz.
"""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxcont.core.continuation import ContinuationSolution
from jaxcont.viz.styles import style_for


@dataclass(frozen=True)
class EigenvalueReference:
    """A vertical reference marker from an external or analytic result.

    ``label`` should identify the source and event, for example
    ``"MatCont H: mu=0"`` or ``"analytic crossing"``.
    """

    parameter: float
    label: str
    color: str = "#009E73"
    linestyle: str = ":"
    linewidth: float = 1.8


def plot_phase_portrait(
    solution: ContinuationSolution,
    state_indices: Tuple[int, int] = (0, 1),
    param_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot phase portraits for selected parameter values.

    Args:
        solution: Continuation solution
        state_indices: Which state variables to plot (x, y)
        param_indices: Which parameter indices to show (None = all)
        ax: Matplotlib axes (creates new figure if None). Previously this
            parameter didn't exist, so a caller-supplied ax (e.g.
            example_03_van_der_pol.py's ax=ax2) silently landed in **kwargs
            and was never used -- the function always drew onto its own new
            figure instead.
        **kwargs: Additional plotting options

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    if param_indices is None:
        param_indices = jnp.linspace(0, solution.n_points - 1, min(10, solution.n_points), dtype=int)

    idx_x, idx_y = state_indices

    for i in param_indices:
        state = solution.states[i]
        param = solution.parameters[i]

        ax.plot(state[idx_x], state[idx_y], 'o', markersize=8,
               label=f"p={param:.3f}")

    ax.set_xlabel(f"State[{idx_x}]", fontsize=12)
    ax.set_ylabel(f"State[{idx_y}]", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_eigenvalues(
    solution: Any,
    ax: Optional[Union[plt.Axes, Sequence[plt.Axes]]] = None,
    *,
    parameters: Optional[Any] = None,
    events: Optional[Sequence[Any]] = None,
    show_events: bool = True,
    shade_stability: bool = False,
    references: Optional[Sequence[Union[EigenvalueReference, Mapping[str, Any]]]] = None,
    param_name: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (12.0, 5.0),
    titles: Tuple[Optional[str], Optional[str]] = (
        "Real Part of Eigenvalues",
        "Imaginary Part of Eigenvalues",
    ),
    legend: bool = True,
    stable_color: str = "#0072B2",
    unstable_color: str = "#D55E00",
    **kwargs,
) -> plt.Figure:
    """Plot eigenvalue trajectories and continuation-validation overlays.

    Args:
        solution: A ``ContinuationResult``, ``Branch``, legacy
            ``ContinuationSolution``, or raw eigenvalue array. Raw arrays have
            shape ``(n_points, n_eigenvalues)``; a 1-D array is one trajectory.
        ax: None to create the standard real/imag panels, one axes for the
            legacy real-only view, or a two-axes sequence for both panels.
        parameters: Parameter coordinates for raw arrays. Defaults to point
            indices. Overrides inferred coordinates when supplied.
        events: JaxCont ``EventHit`` objects or bifurcation dictionaries.
            Defaults to events stored on the input object.
        show_events: Highlight eigenvalues at JaxCont event parameters.
        shade_stability: Shade contiguous stable/unstable parameter regions.
        references: External or analytic vertical reference markers.
        param_name: X-axis label. Defaults to solution metadata or
            ``"Parameter"``.
        labels: Optional label for each eigenvalue trajectory.
        figsize: Figure size when creating a figure.
        titles: Titles for the real and imaginary panels; use None to omit one.
        legend: Draw a de-duplicated legend when labeled artists exist.
        stable_color: Stable-region shading color.
        unstable_color: Unstable-region shading color.
        **kwargs: Matplotlib options applied to every trajectory line.

    Returns:
        Matplotlib figure
    """
    params, eigenvalues, stability, inferred_events, inferred_name = _eigenvalue_data(
        solution, parameters
    )
    if events is None:
        events = inferred_events
    if param_name is None:
        param_name = inferred_name or "Parameter"

    if labels is not None and len(labels) != eigenvalues.shape[1]:
        raise ValueError("labels must contain one label per eigenvalue")
    if len(titles) != 2:
        raise ValueError("titles must contain the real and imaginary panel titles")
    if shade_stability and stability is None:
        raise ValueError("Stability information is required when shade_stability=True")

    created_figure = ax is None
    if created_figure:
        fig, axes_array = plt.subplots(1, 2, figsize=figsize)
        axes = list(axes_array)
    elif isinstance(ax, plt.Axes):
        fig = ax.get_figure()
        axes = [ax]
    else:
        axes = list(np.asarray(ax, dtype=object).reshape(-1))
        if len(axes) != 2:
            raise ValueError("ax must be one Matplotlib axes or a two-axes sequence")
        if not all(isinstance(item, plt.Axes) for item in axes):
            raise TypeError("ax must contain Matplotlib axes")
        fig = axes[0].get_figure()
        if any(item.get_figure() is not fig for item in axes[1:]):
            raise ValueError("supplied axes must belong to the same figure")

    components = (np.real, np.imag)
    ylabels = ("Re(λ)", "Im(λ)")
    line_options = {"linewidth": 1.8, "alpha": 0.8}
    line_options.update(kwargs)

    for panel, axis in enumerate(axes):
        if shade_stability:
            _shade_stability(axis, params, stability, stable_color, unstable_color)

        component = components[panel]
        for index in range(eigenvalues.shape[1]):
            options = dict(line_options)
            if labels is not None:
                options.setdefault("label", labels[index])
            axis.plot(params, component(eigenvalues[:, index]), **options)

        axis.axhline(y=0, color="#262626", linestyle="--", linewidth=1.0, alpha=0.35)
        axis.set_xlabel(param_name, fontsize=11, labelpad=8)
        axis.set_ylabel(ylabels[panel], fontsize=11, labelpad=8)
        if titles[panel]:
            axis.set_title(titles[panel], fontsize=13, fontweight="bold", pad=12)
        axis.set_axisbelow(True)
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#6B7280")
        axis.spines["bottom"].set_color("#6B7280")
        axis.tick_params(colors="#374151", labelsize=9.5)

    if show_events:
        _plot_events(axes, params, eigenvalues, events or ())
    _plot_references(axes, references or ())

    if legend:
        for axis in axes:
            handles, legend_labels = axis.get_legend_handles_labels()
            unique = dict(zip(legend_labels, handles))
            if unique:
                axis.legend(
                    unique.values(), unique.keys(), frameon=False,
                    fontsize=9, handlelength=2.4,
                )

    if created_figure:
        fig.tight_layout(pad=1.2)
    return fig


def _eigenvalue_data(data: Any, parameters: Optional[Any]):
    """Normalize supported public containers to NumPy plotting arrays."""
    stability = None
    inferred_events = ()
    inferred_name = None

    if hasattr(data, "branch") and hasattr(data.branch, "params"):
        branch = data.branch
        params = branch.params
        eigenvalues = branch.eigenvalues
        stability = branch.stable
        valid = branch.valid
        inferred_events = getattr(data, "events", ())
        legacy = getattr(data, "_solution", None)
        inferred_name = getattr(legacy, "param_name", None)
    elif hasattr(data, "params") and hasattr(data, "eigenvalues"):
        params = data.params
        eigenvalues = data.eigenvalues
        stability = getattr(data, "stable", None)
        valid = getattr(data, "valid", None)
    elif hasattr(data, "parameters") and hasattr(data, "eigenvalues"):
        params = data.parameters
        eigenvalues = data.eigenvalues
        stability = getattr(data, "stability", None)
        valid = None
        inferred_events = getattr(data, "bifurcations", ())
        inferred_name = getattr(data, "param_name", None)
    else:
        eigenvalues = data
        params = parameters
        valid = None

    if eigenvalues is None:
        raise ValueError("Solution does not contain eigenvalue information")

    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim == 1:
        eigenvalues = eigenvalues[:, np.newaxis]
    elif eigenvalues.ndim != 2:
        raise ValueError("eigenvalues must be a one- or two-dimensional array")
    if eigenvalues.shape[0] == 0 or eigenvalues.shape[1] == 0:
        raise ValueError("eigenvalues must contain at least one point and one trajectory")

    if parameters is not None:
        params = parameters
    elif params is None:
        params = np.arange(eigenvalues.shape[0])
    params = np.asarray(params)
    if params.ndim != 1 or params.shape[0] != eigenvalues.shape[0]:
        raise ValueError("parameters and eigenvalues must have the same number of points")

    if stability is not None:
        stability = np.asarray(stability, dtype=bool)
        if stability.ndim != 1 or stability.shape[0] != eigenvalues.shape[0]:
            raise ValueError("stability and eigenvalues must have the same number of points")

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim != 1 or valid.shape[0] != eigenvalues.shape[0]:
            raise ValueError("valid and eigenvalues must have the same number of points")
        params = params[valid]
        eigenvalues = eigenvalues[valid]
        if stability is not None:
            stability = stability[valid]

    return params, eigenvalues, stability, inferred_events, inferred_name


def _shade_stability(axis, params, stability, stable_color, unstable_color):
    if len(params) == 0:
        return
    transitions = np.flatnonzero(stability[1:] != stability[:-1]) + 1
    starts = np.concatenate(([0], transitions))
    ends = np.concatenate((transitions, [len(params)]))
    used = {True: False, False: False}
    for start, end in zip(starts, ends):
        state = bool(stability[start])
        left = params[start] if start == 0 else (params[start - 1] + params[start]) / 2
        right = params[end - 1] if end == len(params) else (params[end - 1] + params[end]) / 2
        label = ("Stable" if state else "Unstable") if not used[state] else None
        axis.axvspan(
            min(left, right), max(left, right),
            color=stable_color if state else unstable_color,
            alpha=0.08, linewidth=0, label=label, zorder=0,
        )
        used[state] = True


def _event_kind_and_parameter(event):
    if isinstance(event, Mapping):
        return event.get("kind", event.get("type", "event")), event.get(
            "p", event.get("parameter")
        )
    return getattr(event, "kind", getattr(event, "type", "event")), getattr(
        event, "p", getattr(event, "parameter", None)
    )


def _plot_events(axes, params, eigenvalues, events):
    seen = set()
    for event in events:
        kind, parameter = _event_kind_and_parameter(event)
        if parameter is None:
            continue
        key = (str(kind), float(parameter))
        if key in seen:
            continue
        seen.add(key)
        point_index = int(np.argmin(np.abs(params - float(parameter))))
        style = style_for(str(kind))
        for panel, axis in enumerate(axes):
            values = (np.real, np.imag)[panel](eigenvalues[point_index])
            axis.scatter(
                np.full(values.shape, float(parameter)), values,
                marker=style.marker, s=58, color=style.color,
                edgecolor="white", linewidth=1.0, zorder=5,
                label=f"JaxCont {style.label}",
            )


def _plot_references(axes, references):
    for reference in references:
        if isinstance(reference, Mapping):
            reference = EigenvalueReference(**reference)
        if not isinstance(reference, EigenvalueReference):
            raise TypeError("references must contain EigenvalueReference objects or mappings")
        for axis in axes:
            axis.axvline(
                reference.parameter,
                color=reference.color,
                linestyle=reference.linestyle,
                linewidth=reference.linewidth,
                label=reference.label,
            )
