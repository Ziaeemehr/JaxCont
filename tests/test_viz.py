"""Tests for jaxcont.viz."""

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from jaxcont.core.continuation import ContinuationSolution
from jaxcont.viz.styles import BIFURCATION_STYLES, DEFAULT_STYLE, style_for


def test_bifurcation_styles_cover_detector_types():
    """Every bif_type string bifurcations/detector.py emits has a style entry.

    BifurcationDetector tags points with bif_type == 'fold' or 'hopf' (see
    detector.py's detect_along_branch); both must resolve to a real
    BIFURCATION_STYLES entry, not silently fall through to DEFAULT_STYLE, so a
    marker/color change in one place can't accidentally stop applying to one
    of the two shipped detectors.
    """
    for bif_type in ("fold", "hopf"):
        assert bif_type in BIFURCATION_STYLES


def test_style_for_known_type_returns_table_entry():
    assert style_for("fold") == BIFURCATION_STYLES["fold"]


def test_style_for_unknown_type_falls_back_with_raw_label():
    style = style_for("some-future-type")
    assert style.marker == DEFAULT_STYLE.marker
    assert style.color == DEFAULT_STYLE.color
    assert style.label == "some-future-type"


from jaxcont.viz.core import plot_bifurcation_diagram, plot_continuation


def _simple_solution(with_stability=False, with_bifurcation=False):
    states = jnp.array([[0.0], [0.5], [1.0], [0.5], [0.0]])
    parameters = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    stability = jnp.array([True, True, False, True, True]) if with_stability else None
    bifurcations = (
        [{"type": "fold", "parameter": 1.0, "state": jnp.array([1.0])}]
        if with_bifurcation else []
    )
    return ContinuationSolution(
        states=states, parameters=parameters, stability=stability,
        bifurcations=bifurcations,
    )


def test_plot_continuation_returns_figure_with_one_axes():
    fig = plot_continuation(_simple_solution())
    assert len(fig.axes) == 1


def test_plot_continuation_default_labels_are_generic():
    fig = plot_continuation(_simple_solution())
    ax = fig.axes[0]
    assert ax.get_ylabel() == "State[0]"
    assert ax.get_xlabel() == "Parameter"


def test_plot_continuation_marks_fold_with_shared_style():
    fig = plot_continuation(_simple_solution(with_bifurcation=True))
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Fold" in labels


def test_plot_bifurcation_diagram_is_an_alias():
    fig = plot_bifurcation_diagram(_simple_solution())
    assert len(fig.axes) == 1


def test_plot_continuation_annotate_false_by_default_no_text_boxes():
    fig = plot_continuation(_simple_solution(with_bifurcation=True))
    assert len(fig.axes[0].texts) == 0


def test_plot_continuation_annotate_true_adds_one_box_per_bifurcation():
    solution = _simple_solution(with_bifurcation=True)
    fig = plot_continuation(solution, annotate=True)
    assert len(fig.axes[0].texts) == len(solution.bifurcations)
