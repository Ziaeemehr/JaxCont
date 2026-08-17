"""Two-parameter (codim-2) bifurcation diagrams: curves of folds and Hopf
points in the (p[0], p[1]) plane, with codim-2 points marked.

Reuses viz/styles.py's shared BIFURCATION_STYLES table so these plots use
the same markers/colors/abbreviations as every other diagram.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from jaxcont.bifurcations.curves import unpack_fold_curve, unpack_hopf_curve
from jaxcont.viz.styles import style_for

_CURVE_STYLE = {
    "fold": {"color": "#009E73", "label": "LP curve"},
    "hopf": {"color": "#CC79A7", "label": "H curve"},
}


def _curve_points(result, curve_kind: str, free: int):
    """Return ``(p_free, p_fixed)`` arrays for a traced curve."""
    states = result.branch.states
    params = result.branch.params
    n_state = states.shape[1]
    fixed = []
    for i in range(states.shape[0]):
        if curve_kind == "fold":
            n = (n_state - 1) // 2
            _u, p_fixed, _v = unpack_fold_curve(states[i], n)
        elif curve_kind == "hopf":
            n = (n_state - 2) // 3
            _u, p_fixed, _q1, _q2, _w = unpack_hopf_curve(states[i], n)
        else:
            raise ValueError(
                f"curve_kind must be 'fold' or 'hopf', got {curve_kind!r}"
            )
        fixed.append(float(p_fixed))
    return [float(p) for p in params], fixed


def plot_two_parameter_diagram(
    results: Sequence[Tuple[object, str]],
    *,
    free: int = 1,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
) -> plt.Axes:
    """
    Plot one or more two-parameter curves with their codim-2 points.

    ``results`` is a sequence of ``(ContinuationResult, curve_kind)`` pairs
    where ``curve_kind`` is ``"fold"`` or ``"hopf"``. ``free`` must match
    the ``free`` passed to the curve factory.

    ``ax`` is a real parameter: pass an existing axis to compose this into a
    multi-panel figure.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 5))

    for i, (result, curve_kind) in enumerate(results):
        p_free, p_fixed = _curve_points(result, curve_kind, free)
        style = _CURVE_STYLE[curve_kind]
        label = labels[i] if labels is not None else style["label"]
        ax.plot(p_free, p_fixed, "-", color=style["color"], lw=1.8, label=label)

        for hit in result.events:
            bif = style_for(hit.kind)
            p_vec = hit.info.get("p")
            if p_vec is None:
                continue
            x, y = float(p_vec[free]), float(p_vec[1 - free])
            ax.plot(
                x, y, bif.marker, color=bif.color, markersize=11,
                markeredgecolor="black", markeredgewidth=0.6,
                linestyle="None", label=bif.label, zorder=5,
            )
            if annotate:
                ax.annotate(
                    bif.label, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=bif.color,
                )

    ax.set_xlabel(f"p[{free}]")
    ax.set_ylabel(f"p[{1 - free}]")
    ax.grid(alpha=0.3)
    # De-duplicate legend entries (each codim-2 marker adds its own label).
    handles, legend_labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, legend_labels):
        seen.setdefault(lab, h)
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best", fontsize=9)
    return ax
