"""Tests for jaxcont.viz."""

import warnings

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from jaxcont.api import Branch, ContinuationResult, EventHit
from jaxcont.core.continuation import ContinuationSolution
from jaxcont.viz.styles import BIFURCATION_STYLES, DEFAULT_STYLE, style_for


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


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
    # "LP" (Limit Point), matching bifurcations/taxonomy.py's standard
    # abbreviation for a fold bifurcation -- not the full word "Fold".
    assert "LP" in labels


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


def test_plot_continuation_has_clean_publication_defaults():
    fig = plot_continuation(_simple_solution(with_stability=True))
    ax = fig.axes[0]

    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.get_axisbelow()
    assert {line.get_label() for line in ax.lines} >= {"Stable", "Unstable"}


def test_plot_continuation_style_kwargs_override_defaults_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = plot_continuation(
            _simple_solution(), marker="x", linestyle=":", linewidth=3.0,
        )

    assert not caught
    line = fig.axes[0].lines[0]
    assert line.get_marker() == "x"
    assert line.get_linestyle() == ":"
    assert line.get_linewidth() == 3.0


def test_plot_continuation_uses_bifurcation_color_for_annotation():
    fig = plot_continuation(_simple_solution(with_bifurcation=True), annotate=True)
    annotation = fig.axes[0].texts[0]
    expected = BIFURCATION_STYLES["fold"].color

    assert annotation.arrow_patch.get_edgecolor() == matplotlib.colors.to_rgba(expected)


def test_plot_continuation_deduplicates_coincident_bifurcations():
    solution = _simple_solution(with_bifurcation=True)
    solution.bifurcations.append(dict(solution.bifurcations[0]))

    fig = plot_continuation(solution, annotate=True)
    ax = fig.axes[0]

    assert len(ax.texts) == 1
    assert [text.get_text() for text in ax.get_legend().get_texts()].count("LP") == 1


def test_plot_continuation_custom_title_uses_only_left_title_slot():
    fig = plot_continuation(_simple_solution(), title="A custom title")
    ax = fig.axes[0]

    assert ax.get_title(loc="left") == "A custom title"
    assert ax.get_title(loc="center") == ""


from jaxcont.viz.core import plot_all_states


def _two_state_solution():
    states = jnp.array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    parameters = jnp.array([0.0, 0.5, 1.0])
    return ContinuationSolution(
        states=states, parameters=parameters,
        state_names=("E", "x"), param_name="E0",
    )


def test_plot_all_states_one_subplot_per_state():
    fig = plot_all_states(_two_state_solution())
    assert len(fig.axes) == 2


def test_plot_all_states_uses_problem_names_by_default():
    fig = plot_all_states(_two_state_solution())
    assert fig.axes[0].get_ylabel() == "E"
    assert fig.axes[1].get_ylabel() == "x"


def test_plot_all_states_only_bottom_subplot_has_xlabel():
    fig = plot_all_states(_two_state_solution())
    assert fig.axes[0].get_xlabel() == ""
    assert fig.axes[1].get_xlabel() == "E0"


def test_plot_all_states_only_one_legend_on_figure():
    fig = plot_all_states(_two_state_solution())
    per_axes_legends = [ax.get_legend() for ax in fig.axes]
    assert all(legend is None for legend in per_axes_legends)
    assert len(fig.legends) == 1


def test_plot_all_states_rejects_mismatched_state_names():
    with pytest.raises(ValueError):
        plot_all_states(_two_state_solution(), state_names=["only-one-name"])


def test_plot_all_states_single_state_keeps_its_xlabel():
    """With state_dim == 1 there's only one subplot, so the "clear all but
    the last subplot's xlabel" logic (`if i < n - 1: ax.set_xlabel("")`) must
    never trigger -- the sole subplot should keep its x-axis label.
    """
    solution = _simple_solution()
    assert solution.state_dim == 1

    fig = plot_all_states(solution)

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() != ""


from jaxcont.viz.portraits import EigenvalueReference, plot_eigenvalues, plot_phase_portrait


def test_plot_phase_portrait_draws_onto_supplied_ax_not_a_new_figure():
    solution = _two_state_solution()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("placeholder")

    returned_fig = plot_phase_portrait(solution, ax=ax2)

    assert returned_fig is fig
    assert ax2.get_title() == "Phase Portrait"
    assert ax1.get_title() == "placeholder"


def test_plot_phase_portrait_creates_own_figure_when_no_ax_given():
    fig = plot_phase_portrait(_two_state_solution())
    assert len(fig.axes) == 1


def test_plot_eigenvalues_raises_without_eigenvalues():
    solution = _two_state_solution()
    with pytest.raises(ValueError):
        plot_eigenvalues(solution)


def test_plot_eigenvalues_plots_real_and_imag_parts():
    states = jnp.array([[0.0, 1.0], [0.5, 1.5]])
    parameters = jnp.array([0.0, 1.0])
    eigenvalues = jnp.array([[1.0 + 1.0j, -1.0 + 0.5j], [0.9 + 0.9j, -0.8 + 0.4j]])
    solution = ContinuationSolution(
        states=states, parameters=parameters, eigenvalues=eigenvalues,
    )
    fig = plot_eigenvalues(solution)
    assert len(fig.axes) == 2


def test_plot_eigenvalues_draws_onto_supplied_ax_not_a_new_figure():
    states = jnp.array([[0.0, 1.0], [0.5, 1.5]])
    parameters = jnp.array([0.0, 1.0])
    eigenvalues = jnp.array([[1.0 + 1.0j, -1.0 + 0.5j], [0.9 + 0.9j, -0.8 + 0.4j]])
    solution = ContinuationSolution(
        states=states, parameters=parameters, eigenvalues=eigenvalues,
    )

    fig, ax = plt.subplots()

    returned_fig = plot_eigenvalues(solution, ax=ax)

    assert returned_fig is fig
    assert ax.get_title() == "Real Part of Eigenvalues"
    assert len(ax.lines) > 0


def _eigenvalue_result():
    parameters = jnp.array([-1.0, 0.0, 1.0])
    eigenvalues = jnp.array(
        [
            [-0.5 + 1.0j, -0.5 - 1.0j],
            [0.0 + 1.0j, 0.0 - 1.0j],
            [0.5 + 1.0j, 0.5 - 1.0j],
        ]
    )
    stable = jnp.array([True, False, False])
    branch = Branch(
        params=parameters,
        states=jnp.zeros((3, 2)),
        eigenvalues=eigenvalues,
        stable=stable,
    )
    solution = ContinuationSolution(
        states=branch.states,
        parameters=parameters,
        eigenvalues=eigenvalues,
        stability=stable,
        param_name="mu",
    )
    return ContinuationResult(
        branch=branch,
        events=[EventHit(kind="hopf", p=0.0, u=jnp.zeros(2))],
        _solution=solution,
    )


def test_plot_eigenvalues_accepts_result_and_infers_metadata():
    fig = plot_eigenvalues(_eigenvalue_result())

    assert len(fig.axes) == 2
    assert all(ax.get_xlabel() == "mu" for ax in fig.axes)
    assert all("JaxCont H" in ax.get_legend_handles_labels()[1] for ax in fig.axes)


def test_plot_eigenvalues_shades_stable_and_unstable_regions():
    fig = plot_eigenvalues(_eigenvalue_result(), shade_stability=True)

    for ax in fig.axes:
        labels = ax.get_legend_handles_labels()[1]
        assert "Stable" in labels
        assert "Unstable" in labels
        assert len(ax.patches) == 2


def test_plot_eigenvalues_adds_external_references_to_both_panels():
    reference = EigenvalueReference(0.0, "MatCont H: mu=0")
    fig = plot_eigenvalues(_eigenvalue_result(), references=[reference])

    for ax in fig.axes:
        assert "MatCont H: mu=0" in ax.get_legend_handles_labels()[1]


def test_plot_eigenvalues_accepts_raw_arrays_and_custom_labels():
    eigenvalues = jnp.array(
        [[-1.0 + 0.5j, -2.0], [-0.5 + 0.25j, -1.5]]
    )
    fig = plot_eigenvalues(
        eigenvalues,
        parameters=jnp.array([2.0, 3.0]),
        param_name="gain",
        labels=["critical", "secondary"],
        titles=("Growth rates", "Frequencies"),
    )

    real, imag = fig.axes
    assert real.get_xlabel() == "gain"
    assert real.get_title() == "Growth rates"
    assert imag.get_title() == "Frequencies"
    assert {"critical", "secondary"} <= set(real.get_legend_handles_labels()[1])


def test_plot_eigenvalues_accepts_pair_of_supplied_axes():
    fig, axes = plt.subplots(2, 1)

    returned = plot_eigenvalues(_eigenvalue_result(), ax=axes)

    assert returned is fig
    assert axes[0].get_title() == "Real Part of Eigenvalues"
    assert axes[1].get_title() == "Imaginary Part of Eigenvalues"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"parameters": jnp.array([0.0])}, "same number of points"),
        ({"labels": ["only one"]}, "one label per eigenvalue"),
    ],
)
def test_plot_eigenvalues_validates_generalized_inputs(kwargs, message):
    eigenvalues = jnp.array([[1.0 + 1.0j, 2.0], [0.5 + 0.5j, 1.5]])

    with pytest.raises(ValueError, match=message):
        plot_eigenvalues(eigenvalues, **kwargs)


def test_viz_package_exports_public_surface():
    import jaxcont.viz as viz

    for name in (
        "plot_continuation", "plot_bifurcation_diagram", "plot_all_states",
        "plot_phase_portrait", "plot_eigenvalues", "EigenvalueReference",
    ):
        assert hasattr(viz, name), f"jaxcont.viz missing {name}"
