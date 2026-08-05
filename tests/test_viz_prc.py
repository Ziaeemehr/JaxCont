"""Tests for jaxcont.viz.portraits.plot_prc."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from jaxcont.viz import plot_prc


def test_plot_prc_returns_figure_with_one_line_per_component():
    curve = np.stack([np.cos(np.linspace(0, 2 * np.pi, 10, endpoint=False)),
                       np.sin(np.linspace(0, 2 * np.pi, 10, endpoint=False))], axis=1)
    fig = plot_prc(curve)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == curve.shape[1]


def test_plot_prc_accepts_explicit_phase_and_labels():
    curve = np.zeros((5, 2))
    phase = np.linspace(0, 1, 5)
    fig = plot_prc(curve, phase=phase, labels=["x", "y"])
    ax = fig.axes[0]
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["x", "y"]
