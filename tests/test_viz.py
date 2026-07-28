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


def test_bifurcation_styles_cover_event_types():
    """Every bif_type string bifurcations/events.py emits has a style entry.

    detect_events tags points with bif_type == 'fold' or 'hopf' (see
    events.py's Fold/Hopf); both must resolve to a real
    BIFURCATION_STYLES entry, not silently fall through to DEFAULT_STYLE, so a
    marker/color change in one place can't accidentally stop applying to one
    of the two shipped event kinds.
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


from jaxcont.viz.portraits import (
    EigenvalueReference, plot_branch_states, plot_eigenvalues, plot_phase_portrait,
)


def test_plot_branch_states_draws_onto_supplied_ax_not_a_new_figure():
    solution = _two_state_solution()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("placeholder")

    returned_fig = plot_branch_states(solution, ax=ax2)

    assert returned_fig is fig
    assert ax2.get_title() == "Phase Portrait"
    assert ax1.get_title() == "placeholder"


def test_plot_branch_states_creates_own_figure_when_no_ax_given():
    fig = plot_branch_states(_two_state_solution())
    assert len(fig.axes) == 1


def test_plot_phase_portrait_alias_warns_and_forwards():
    with pytest.warns(DeprecationWarning, match="plot_branch_states"):
        fig = plot_phase_portrait(_two_state_solution())

    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Phase Portrait"


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
        "plot_branch_states", "plot_phase_portrait", "plot_eigenvalues",
        "EigenvalueReference", "plot_phase_plane", "plot_nullclines",
        "plot_vector_field", "plot_streamlines", "plot_equilibria",
        "plot_trajectory",
    ):
        assert hasattr(viz, name), f"jaxcont.viz missing {name}"


import numpy as np

import jaxcont as jc
from jaxcont.viz.phase_plane import _evaluate_field, _require_2d, _state_names


def _linear_spiral_problem():
    """dx/dt = y - x, dy/dt = -x - y.

    Nullclines are the straight lines y = x and y = -x. The only equilibrium
    is the origin, a stable spiral (eigenvalues -1 +/- i).
    """
    def rhs(u, p, args):
        x, y = u
        return jnp.array([y - x, -x - y])

    return jc.bif_problem(
        rhs, u0=jnp.array([0.0, 0.0]), p0=0.0,
        state_names=["x", "y"], param_name="mu",
    )


def _three_state_problem():
    def rhs(u, p, args):
        return -u

    return jc.bif_problem(rhs, u0=jnp.array([1.0, 1.0, 1.0]), p0=0.0)


def test_require_2d_accepts_two_state_problem():
    _require_2d(_linear_spiral_problem())  # must not raise


def test_require_2d_rejects_higher_dimensional_problem_naming_n():
    with pytest.raises(NotImplementedError, match="n=3"):
        _require_2d(_three_state_problem())


def test_evaluate_field_shapes_and_values():
    problem = _linear_spiral_problem()

    X, Y, F = _evaluate_field(problem, 0.0, (-1.0, 1.0), (-2.0, 2.0), resolution=5)

    assert X.shape == (5, 5)
    assert Y.shape == (5, 5)
    assert F.shape == (5, 5, 2)
    assert isinstance(F, np.ndarray)
    # X varies along columns, Y along rows (matplotlib's "xy" indexing).
    np.testing.assert_allclose(X[0, :], np.linspace(-1.0, 1.0, 5))
    np.testing.assert_allclose(Y[:, 0], np.linspace(-2.0, 2.0, 5))
    # F must equal the right-hand side evaluated pointwise.
    np.testing.assert_allclose(F[..., 0], Y - X, atol=1e-6)
    np.testing.assert_allclose(F[..., 1], -X - Y, atol=1e-6)


def test_state_names_falls_back_when_problem_has_none():
    def rhs(u, p, args):
        return u

    problem = jc.bif_problem(rhs, u0=jnp.array([0.0, 0.0]), p0=0.0)

    assert _state_names(problem) == ("state[0]", "state[1]")
    assert _state_names(_linear_spiral_problem()) == ("x", "y")


from jaxcont.viz.phase_plane import plot_nullclines


def test_plot_nullclines_zero_set_matches_analytic_lines():
    """For dx/dt = y - x the x-nullcline is exactly y = x."""
    problem = _linear_spiral_problem()

    X, Y, F = _evaluate_field(problem, 0.0, (-2.0, 2.0), (-2.0, 2.0), resolution=41)

    on_diagonal = np.isclose(X, Y)
    assert np.all(np.abs(F[..., 0][on_diagonal]) < 1e-6)
    on_antidiagonal = np.isclose(X, -Y)
    assert np.all(np.abs(F[..., 1][on_antidiagonal]) < 1e-6)


def test_plot_nullclines_draws_onto_supplied_ax_and_returns_its_figure():
    problem = _linear_spiral_problem()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("placeholder")

    returned_fig = plot_nullclines(
        problem, 0.0, (-2.0, 2.0), (-2.0, 2.0), resolution=25, ax=ax2
    )

    assert returned_fig is fig
    assert len(ax2.get_children()) > len(ax1.get_children())
    assert ax1.get_title() == "placeholder"


def test_plot_nullclines_creates_own_figure_when_no_ax_given():
    fig = plot_nullclines(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0), resolution=25
    )
    assert len(fig.axes) == 1


def test_plot_nullclines_labels_axes_from_state_names():
    fig = plot_nullclines(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0), resolution=25
    )
    ax = fig.axes[0]

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(legend_labels) == 2


def test_plot_nullclines_rejects_three_state_problem():
    with pytest.raises(NotImplementedError, match="n=3"):
        plot_nullclines(_three_state_problem(), 0.0, (-1.0, 1.0), (-1.0, 1.0))


from jaxcont.viz.phase_plane import plot_streamlines, plot_vector_field


def test_plot_vector_field_normalizes_arrows_to_unit_length():
    problem = _linear_spiral_problem()

    fig = plot_vector_field(
        problem, 0.0, (-2.0, 2.0), (-2.0, 2.0), density=7, normalize=True
    )
    quiver = fig.axes[0].collections[0]

    lengths = np.hypot(quiver.U, quiver.V)
    finite = lengths[np.isfinite(lengths)]
    # Every arrow is unit length except at the origin, where the field is zero.
    assert np.all((np.abs(finite - 1.0) < 1e-6) | (finite < 1e-12))


def test_plot_vector_field_unnormalized_keeps_true_magnitudes():
    problem = _linear_spiral_problem()

    fig = plot_vector_field(
        problem, 0.0, (-2.0, 2.0), (-2.0, 2.0), density=7, normalize=False
    )
    quiver = fig.axes[0].collections[0]

    assert np.max(np.hypot(quiver.U, quiver.V)) > 1.5


def test_plot_vector_field_draws_onto_supplied_ax():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    returned_fig = plot_vector_field(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0), density=7, ax=ax2
    )

    assert returned_fig is fig
    assert len(ax2.collections) == 1
    assert len(ax1.collections) == 0


def test_plot_streamlines_draws_onto_supplied_ax():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    returned_fig = plot_streamlines(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0), resolution=25, ax=ax2
    )

    assert returned_fig is fig
    assert len(ax2.get_children()) > len(ax1.get_children())


def test_vector_field_and_streamlines_reject_three_state_problem():
    problem = _three_state_problem()
    with pytest.raises(NotImplementedError, match="n=3"):
        plot_vector_field(problem, 0.0, (-1.0, 1.0), (-1.0, 1.0))
    with pytest.raises(NotImplementedError, match="n=3"):
        plot_streamlines(problem, 0.0, (-1.0, 1.0), (-1.0, 1.0))


from jaxcont.viz.phase_plane import _equilibria_at, plot_equilibria


def _fold_problem():
    """dx/dt = x^2 + p, dy/dt = -y.

    Equilibria are (+/-sqrt(-p), 0), meeting at a fold at p = 0. The branch
    with x > 0 is unstable (Jacobian eigenvalues 2x, -1) and the x < 0 branch
    is stable, so one continuation run produces both stability classes.
    """
    def rhs(u, p, args):
        x, y = u
        return jnp.array([x**2 + p, -y])

    return jc.bif_problem(
        rhs, u0=jnp.array([1.0, 0.0]), p0=-1.0,
        state_names=["x", "y"], param_name="p",
    )


def _fold_result():
    return jc.continuation(
        _fold_problem(),
        p_span=(-1.0, 0.1),
        settings=jc.ContinuationPar(ds=0.02, max_steps=400),
    )


def test_equilibria_at_returns_both_sides_of_a_fold():
    """The case Branch.at_param cannot cover: two equilibria at one p."""
    result = _fold_result()

    states, _ = _equilibria_at(result.branch, -0.5)

    assert states.shape == (2, 2)
    found = np.sort(states[:, 0])
    np.testing.assert_allclose(found, [-np.sqrt(0.5), np.sqrt(0.5)], atol=1e-3)
    np.testing.assert_allclose(states[:, 1], [0.0, 0.0], atol=1e-6)


def test_equilibria_at_marks_the_positive_branch_unstable():
    result = _fold_result()

    states, stable_flags = _equilibria_at(result.branch, -0.5)

    by_x = {round(float(x), 3): flag for x, flag in zip(states[:, 0], stable_flags)}
    assert by_x[round(float(np.sqrt(0.5)), 3)] is False
    assert by_x[round(float(-np.sqrt(0.5)), 3)] is True


def test_equilibria_lie_on_both_nullclines():
    """The defining property: an equilibrium is a nullcline intersection."""
    problem = _fold_problem()
    result = _fold_result()

    states, _ = _equilibria_at(result.branch, -0.5)

    rhs = problem.as_rhs(-0.5)
    for point in states:
        residual = np.asarray(rhs(jnp.asarray(point)))
        assert np.max(np.abs(residual)) < 1e-3


def test_equilibria_at_returns_nothing_outside_the_branch_range():
    result = _fold_result()

    states, flags = _equilibria_at(result.branch, 5.0)

    assert states.shape[0] == 0
    assert flags == []


def test_plot_equilibria_rejects_a_three_state_branch():
    branch = Branch(
        params=jnp.array([0.0, 1.0]),
        states=jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    with pytest.raises(NotImplementedError, match="n=3"):
        plot_equilibria(branch, 0.5)


def test_plot_equilibria_accepts_marker_kwargs_without_colliding():
    branch = Branch(
        params=jnp.array([0.0, 1.0]),
        states=jnp.array([[0.0, 0.0], [1.0, 1.0]]),
    )
    fig = plot_equilibria(branch, 0.5, markersize=20)
    assert fig.axes[0].get_lines()[0].get_markersize() == 20


def test_plot_equilibria_marks_both_fold_branches_with_distinct_styles():
    result = _fold_result()
    fig, ax = plt.subplots()

    returned_fig = plot_equilibria(result, -0.5, ax=ax)

    assert returned_fig is fig
    lines = ax.get_lines()
    assert len(lines) == 2
    face_colors = {line.get_markerfacecolor() for line in lines}
    # Stable renders filled, unstable renders hollow.
    assert "none" in face_colors
    assert len(face_colors) == 2


def test_plot_equilibria_uses_neutral_style_without_stability_data():
    branch = Branch(
        params=jnp.array([-1.0, 0.0, 1.0]),
        states=jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    )
    fig, ax = plt.subplots()

    plot_equilibria(branch, 0.5, ax=ax)

    line = ax.get_lines()[0]
    assert line.get_markeredgecolor() == "#262626"


def test_plot_equilibria_rejects_a_problem():
    with pytest.raises(TypeError, match="ContinuationResult or Branch"):
        plot_equilibria(_linear_spiral_problem(), 0.0)


from jaxcont.viz.phase_plane import plot_trajectory


def test_plot_trajectory_integrates_toward_the_stable_spiral():
    """dx/dt = y - x, dy/dt = -x - y has a stable spiral at the origin."""
    problem = _linear_spiral_problem()
    fig, ax = plt.subplots()

    returned_fig = plot_trajectory(
        problem, u0=jnp.array([1.5, 0.0]), p=0.0, t_span=(0.0, 20.0),
        n_points=400, ax=ax,
    )

    assert returned_fig is fig
    line = ax.get_lines()[0]
    xs, ys = line.get_xdata(), line.get_ydata()
    np.testing.assert_allclose([xs[0], ys[0]], [1.5, 0.0], atol=1e-6)
    assert np.hypot(xs[-1], ys[-1]) < 1e-4


def test_plot_trajectory_accepts_a_precomputed_array():
    states = np.column_stack([np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)])
    fig, ax = plt.subplots()

    plot_trajectory(states, ax=ax, arrow=False)

    line = ax.get_lines()[0]
    np.testing.assert_allclose(line.get_xdata(), states[:, 0])
    np.testing.assert_allclose(line.get_ydata(), states[:, 1])


def test_plot_trajectory_rejects_mixing_the_two_calling_forms():
    states = np.zeros((10, 2))
    with pytest.raises(TypeError, match="takes no u0"):
        plot_trajectory(states, u0=jnp.array([1.0, 0.0]))


def test_plot_trajectory_requires_all_integration_arguments():
    with pytest.raises(TypeError, match="requires u0, p and t_span"):
        plot_trajectory(_linear_spiral_problem(), u0=jnp.array([1.0, 0.0]))


def test_plot_trajectory_rejects_a_wrongly_shaped_array():
    with pytest.raises(ValueError, match=r"\(n_steps, 2\)"):
        plot_trajectory(np.zeros((10, 3)))


def test_plot_trajectory_rejects_three_state_problem():
    with pytest.raises(NotImplementedError, match="n=3"):
        plot_trajectory(
            _three_state_problem(), u0=jnp.ones(3), p=0.0, t_span=(0.0, 1.0)
        )


from jaxcont.viz.phase_plane import plot_phase_plane


def test_plot_phase_plane_composes_requested_layers_only():
    problem = _linear_spiral_problem()

    fig = plot_phase_plane(
        problem, 0.0, (-2.0, 2.0), (-2.0, 2.0),
        resolution=25, density=7,
        nullclines=True, vector_field=True, streamlines=False,
    )
    ax = fig.axes[0]

    # One quiver collection from the vector field; the two nullclines add
    # contour artists plus their two invisible legend proxies.
    assert len(ax.collections) >= 1
    assert len(ax.get_lines()) == 2


def test_plot_phase_plane_legend_false_removes_any_sublayer_legend():
    fig = plot_phase_plane(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0),
        resolution=25, density=7, legend=False,
    )
    assert fig.axes[0].get_legend() is None


def test_plot_phase_plane_without_any_layer_still_returns_a_figure():
    fig = plot_phase_plane(
        _linear_spiral_problem(), 0.0, (-2.0, 2.0), (-2.0, 2.0),
        nullclines=False, vector_field=False, streamlines=False,
    )

    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlim() == (-2.0, 2.0)


def test_plot_phase_plane_draws_equilibria_when_given_a_result():
    result = _fold_result()
    problem = _fold_problem()

    fig = plot_phase_plane(
        problem, -0.5, (-2.0, 2.0), (-2.0, 2.0),
        resolution=25, density=7, vector_field=False, result=result,
    )
    ax = fig.axes[0]

    # Two nullcline legend proxies plus one marker per fold branch.
    assert len(ax.get_lines()) == 4


def test_plot_phase_plane_accepts_trajectory_pairs_and_arrays():
    problem = _linear_spiral_problem()
    precomputed = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.linspace(1.0, 0.0, 20)]
    )

    fig = plot_phase_plane(
        problem, 0.0, (-2.0, 2.0), (-2.0, 2.0),
        resolution=25, nullclines=False, vector_field=False,
        trajectories=[(jnp.array([1.5, 0.0]), (0.0, 10.0)), precomputed],
    )

    assert len(fig.axes[0].get_lines()) == 2


def test_plot_phase_plane_titles_with_the_parameter_name():
    fig = plot_phase_plane(
        _linear_spiral_problem(), 0.25, (-2.0, 2.0), (-2.0, 2.0),
        resolution=25, density=7,
    )

    assert fig.axes[0].get_title() == "mu = 0.25"
