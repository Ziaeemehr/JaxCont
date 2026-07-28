# 2D Phase-Plane Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `jaxcont.viz.phase_plane` module that draws nullclines, vector fields, streamlines, continuation equilibria, and trajectories for 2D autonomous systems.

**Architecture:** A new file `src/jaxcont/viz/phase_plane.py` holds composable plotting primitives plus one `plot_phase_plane` wrapper that composes them. Everything is built on the existing `BifProblem.as_rhs(p)` bridge (`src/jaxcont/api.py`), which returns an autonomous `rhs(u)` frozen at parameter `p`. Grid evaluation is batched with `jax.vmap`, then converted to NumPy once for matplotlib. No changes to `jaxcont.core`, `jaxcont.api`, or the solver packages — the viz layer stays a pure consumer.

**Tech Stack:** JAX (grid evaluation), matplotlib (rendering), scipy `solve_ivp` (trajectory integration), pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-phase-plane-visualization-design.md`

**Branch:** `feat/phase-plane-viz` (already created; the spec is committed there as `f6ab687`).

## Global Constraints

- **No new dependencies.** scipy is already a hard dependency (`pyproject.toml`, `scipy>=1.7.0`). Do not add diffrax, do not add a `jaxcont[ode]` extra, do not write an ODE integrator.
- **Strict 2D only.** Every public entry point rejects `problem.u0.shape != (2,)` with `NotImplementedError`. No `state_indices=` or `fixed=` slicing arguments.
- **Python `>=3.9`.** No `match` statements, no PEP 604 `X | Y` annotations at runtime in this module. Note `src/jaxcont/viz/styles.py` uses `dict[str, BifStyle]`, which is fine as a 3.9 *variable* annotation, but prefer `typing.Optional`/`Tuple` in signatures to match `core.py` and `portraits.py`.
- **matplotlib `>=3.5`.** `ContourSet.collections` was removed in matplotlib 3.10 (3.10.7 is installed here), so **no test may touch `.collections` on a `ContourSet`, and no code may rely on it.** Nullcline *correctness* is asserted at the data level via `_evaluate_field`; nullcline *rendering* is asserted only as "artists were added to the axes".
- **Existing viz conventions** (`src/jaxcont/viz/core.py`, `portraits.py`): every public plot function takes `ax: Optional[plt.Axes] = None`, creates its own figure when `ax` is None, returns the `plt.Figure`, and accepts `**kwargs` forwarded to the underlying matplotlib call.
- **Colorblind-safe palette already in use** — reuse these exact hex values: stable/x-nullcline `#0072B2`, unstable/y-nullcline `#D55E00`, neutral `#262626` (`DEFAULT_STYLE.color` in `styles.py`).
- **Tests** live in `tests/test_viz.py`, which already sets `matplotlib.use("Agg")` at import and has an autouse `plt.close("all")` fixture. Append to that file; do not create a new test module.
- Run tests with `python -m pytest tests/test_viz.py -v`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/jaxcont/viz/phase_plane.py` | **Create.** All phase-plane plotting: private helpers (`_require_2d`, `_evaluate_field`, `_state_names`, `_prepare_axes`, `_branch_of`, `_valid_arrays`, `_equilibria_at`, `_integrate`) plus the six public functions. |
| `src/jaxcont/viz/portraits.py` | **Modify.** Rename `plot_phase_portrait` → `plot_branch_states`; add a deprecated forwarding alias. |
| `src/jaxcont/viz/__init__.py` | **Modify.** Export the new functions and both rename spellings. |
| `tests/test_viz.py` | **Modify.** Append phase-plane tests; update the two existing `plot_phase_portrait` tests and the `__all__` assertion. |
| `examples/example_12_fitzhugh_nagumo_phase_plane.py` | **Create.** Sphinx-gallery example. |
| `docs/source/api/index.rst` | **Modify.** Add a "Visualization" section. |
| `CHANGELOG.md`, `notes/ROADMAP.md` | **Modify.** Record the feature and the rename. |

Everything lives in one new module because these functions share the same private grid-evaluation and axes-preparation helpers and always change together. The file lands at roughly 400 lines, comparable to the existing `portraits.py` (~370).

---

### Task 1: Module scaffold — dimension guard and grid evaluation

**Files:**
- Create: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `BifProblem.as_rhs(p) -> Callable[[Array], Array]` and `BifProblem.u0`, `BifProblem.state_names` from `src/jaxcont/api.py`.
- Produces:
  - `_require_2d(problem) -> None` — raises `NotImplementedError` when `problem.u0.shape != (2,)`.
  - `_evaluate_field(problem, p, xlim, ylim, resolution) -> Tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(X, Y, F)` with `X`/`Y` of shape `(resolution, resolution)` and `F` of shape `(resolution, resolution, 2)`.
  - `_state_names(problem) -> Tuple[str, str]`.
  - `_prepare_axes(ax, figsize) -> Tuple[plt.Figure, plt.Axes]`.
  - Module constants `X_NULLCLINE_COLOR = "#0072B2"`, `Y_NULLCLINE_COLOR = "#D55E00"`, `STABLE_COLOR = "#0072B2"`, `UNSTABLE_COLOR = "#D55E00"`, `NEUTRAL_COLOR = "#262626"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k "require_2d or evaluate_field or state_names" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jaxcont.viz.phase_plane'`

- [ ] **Step 3: Write the module scaffold**

Create `src/jaxcont/viz/phase_plane.py`:

```python
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


def _prepare_axes(ax, figsize) -> Tuple[plt.Figure, plt.Axes]:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k "require_2d or evaluate_field or state_names" -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): phase-plane module scaffold with 2D guard and grid evaluation"
```

---

### Task 2: `plot_nullclines`

**Files:**
- Modify: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `_evaluate_field`, `_require_2d`, `_state_names`, `_prepare_axes`, `X_NULLCLINE_COLOR`, `Y_NULLCLINE_COLOR`, `DEFAULT_FIGSIZE` from Task 1.
- Produces: `plot_nullclines(problem, p, xlim, ylim, *, resolution=200, colors=None, labels=True, ax=None, figsize=DEFAULT_FIGSIZE, **kwargs) -> plt.Figure`.

**Note on what the tests assert.** Nullcline correctness is asserted through `_evaluate_field` (the zero set of `F[..., i]` is the nullcline, and it is a plain NumPy array), *not* by walking `ContourSet` geometry. `ContourSet.collections` was removed in matplotlib 3.10, and `ContourSet` is an axes child only in matplotlib >= 3.8 — any geometry-walking test would break on one end of the supported `matplotlib>=3.5` range. Rendering is asserted as "the axes gained artists", which holds in both regimes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k nullclines -v`
Expected: FAIL — `ImportError: cannot import name 'plot_nullclines'`

- [ ] **Step 3: Implement `plot_nullclines`**

Append to `src/jaxcont/viz/phase_plane.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k nullclines -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): add plot_nullclines for 2D systems"
```

---

### Task 3: `plot_vector_field` and `plot_streamlines`

**Files:**
- Modify: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `_evaluate_field`, `_state_names`, `_prepare_axes`, `DEFAULT_FIGSIZE` from Task 1.
- Produces:
  - `plot_vector_field(problem, p, xlim, ylim, *, density=20, normalize=True, ax=None, figsize=DEFAULT_FIGSIZE, **kwargs) -> plt.Figure`
  - `plot_streamlines(problem, p, xlim, ylim, *, resolution=100, color="#7F7F7F", ax=None, figsize=DEFAULT_FIGSIZE, **kwargs) -> plt.Figure`

Both ship despite the visual overlap: matplotlib provides them almost free, streamlines read better for smooth flows, and quiver reads better layered under nullclines.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k "vector_field or streamlines" -v`
Expected: FAIL — `ImportError: cannot import name 'plot_vector_field'`

- [ ] **Step 3: Implement both functions**

Append to `src/jaxcont/viz/phase_plane.py`:

```python
def plot_vector_field(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    density: int = 20,
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Plot the vector field of a 2D autonomous system as arrows.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        density: Arrows per axis. Coarser than the nullcline grid on purpose.
        normalize: Scale arrows to unit length so direction stays legible
            where the field is stiff, and color them by ``log10`` of the true
            magnitude so speed information is not lost. When False, arrows
            carry their true magnitude in a single color.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.quiver``.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    X, Y, F = _evaluate_field(problem, p, xlim, ylim, density)
    fig, ax = _prepare_axes(ax, figsize)

    U, V = F[..., 0], F[..., 1]
    magnitude = np.hypot(U, V)

    quiver_options = {"pivot": "mid", "width": 0.004}
    quiver_options.update(kwargs)

    if normalize:
        # Leave true zeros at zero rather than dividing by them; a zero-length
        # arrow is the honest rendering of an equilibrium.
        scale = np.where(magnitude == 0.0, 1.0, magnitude)
        speed = np.ma.masked_invalid(
            np.log10(np.where(magnitude == 0.0, np.nan, magnitude))
        )
        quiver_options.setdefault("cmap", "viridis")
        ax.quiver(X, Y, U / scale, V / scale, speed, **quiver_options)
    else:
        quiver_options.setdefault("color", "#7F7F7F")
        ax.quiver(X, Y, U, V, **quiver_options)

    names = _state_names(problem)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig


def plot_streamlines(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    resolution: int = 100,
    color: str = "#7F7F7F",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Plot the flow of a 2D autonomous system as streamlines.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        resolution: Grid points per axis used to seed the integrator.
        color: Streamline color.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.streamplot``.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    X, Y, F = _evaluate_field(problem, p, xlim, ylim, resolution)
    fig, ax = _prepare_axes(ax, figsize)

    # streamplot needs 1-D, evenly spaced coordinates -- exactly the rows and
    # columns _evaluate_field built the grid from.
    stream_options = {"density": 1.0, "linewidth": 0.8, "arrowsize": 0.9}
    stream_options.update(kwargs)
    ax.streamplot(X[0, :], Y[:, 0], F[..., 0], F[..., 1], color=color, **stream_options)

    names = _state_names(problem)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k "vector_field or streamlines" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): add plot_vector_field and plot_streamlines"
```

---

### Task 4: `plot_equilibria`

**Files:**
- Modify: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `_prepare_axes`, `STABLE_COLOR`, `UNSTABLE_COLOR`, `NEUTRAL_COLOR`, `DEFAULT_FIGSIZE` from Task 1; `Branch` and `ContinuationResult` from `jaxcont.api`.
- Produces:
  - `_branch_of(result) -> Branch`
  - `_valid_arrays(branch) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]`
  - `_equilibria_at(branch, p, atol=None) -> Tuple[np.ndarray, List[Optional[bool]]]` — states shaped `(k, 2)` and one stability flag per state.
  - `plot_equilibria(result, p, *, atol=None, ax=None, figsize=DEFAULT_FIGSIZE, **kwargs) -> plt.Figure`

**Why not `Branch.at_param`.** `Branch.at_param(p)` (`src/jaxcont/api.py`) returns only the single nearest point. A branch that has passed a fold has two or more equilibria at the same `p`, and this function must mark all of them. Instead scan `params - p` for sign changes between consecutive points and linearly interpolate the states at each crossing, then add any point within `atol` of `p` (catching a branch endpoint that touches `p` without crossing), then dedupe in state space.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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

    # Key through plain Python floats: states is float32 (this project never
    # enables jax_enable_x64), and an np.float32 key can compare equal to a
    # Python float while hashing differently, breaking dict lookup.
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k "equilibria" -v`
Expected: FAIL — `ImportError: cannot import name '_equilibria_at'`

- [ ] **Step 3: Implement the equilibrium selection and plotting**

Append to `src/jaxcont/viz/phase_plane.py` (no new imports needed):

```python
def _branch_of(result):
    """Accept a ContinuationResult or a bare Branch, return the Branch."""
    from jaxcont.api import Branch

    if isinstance(result, Branch):
        return result
    branch = getattr(result, "branch", None)
    if branch is None:
        raise TypeError(
            "plot_equilibria expects a ContinuationResult or Branch (this "
            "function consumes continuation output; it does not solve for "
            f"equilibria itself); got {type(result).__name__}."
        )
    return branch


def _valid_arrays(branch):
    """Branch params/states/stability as NumPy, with the buffer mask applied.

    A traced (``jax.vmap``/``jax.jit``) result keeps the full fixed-size engine
    buffer and marks the real points with ``branch.valid``; an eager one is
    already trimmed and has ``valid is None``.
    """
    params = np.asarray(branch.params)
    states = np.asarray(branch.states)
    stable = None if branch.stable is None else np.asarray(branch.stable)
    if branch.valid is not None:
        mask = np.asarray(branch.valid)
        params, states = params[mask], states[mask]
        if stable is not None:
            stable = stable[mask]
    return params, states, stable


def _equilibria_at(branch, p, atol: Optional[float] = None):
    """Every equilibrium on ``branch`` at parameter ``p``.

    Handles a branch that has passed a fold, where two or more points share
    the same ``p`` -- the case ``Branch.at_param`` cannot cover, since it
    returns only the single nearest point.

    Returns ``(states, stable_flags)``: states shaped ``(k, 2)`` and a list of
    ``True``/``False``/``None`` flags, one per state.
    """
    params, states, stable = _valid_arrays(branch)
    if params.size == 0:
        return np.empty((0, states.shape[-1])), []

    d = params - p
    found = []
    flags = []
    crossing_indices = set()

    # Interior crossings, linearly interpolated.
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0.0:
            theta = d[i] / (d[i] - d[i + 1])
            found.append(states[i] + theta * (states[i + 1] - states[i]))
            crossing_indices.add(i)
            crossing_indices.add(i + 1)
            if stable is None:
                flags.append(None)
            elif bool(stable[i]) == bool(stable[i + 1]):
                flags.append(bool(stable[i]))
            else:
                # A stability change between adjacent points means a
                # bifurcation lies in between; draw the conservative style.
                flags.append(False)

    # Points sitting on p without a sign change (e.g. a branch endpoint).
    # Skip indices already claimed by an interior crossing above -- a point
    # adjacent to a crossing is that crossing's own equilibrium, not a second
    # one, and treating it as "sitting on p" would double-count it. The
    # default atol comes from the *local* minimum step, not a trajectory-wide
    # median: a branch that overshoots far past p_span after reversing
    # through a fold (pseudo_arclength_scan has no stop condition for the
    # reversed direction) inflates a global median to the point where it
    # spans the local spacing near p, which is exactly the failure mode this
    # exclusion plus the local-step default both close off.
    if atol is None:
        steps = np.abs(np.diff(params))
        atol = 0.25 * float(np.min(steps)) if steps.size else 0.0
    for i in np.flatnonzero(np.abs(d) <= atol):
        if i in crossing_indices:
            continue
        found.append(states[i])
        flags.append(None if stable is None else bool(stable[i]))

    if not found:
        return np.empty((0, states.shape[-1])), []

    # Dedupe: an exact hit next to a crossing describes the same equilibrium.
    scale = float(np.max(np.abs(states))) if states.size else 1.0
    dedupe_tol = 1e-8 + 1e-6 * max(scale, 1.0)
    kept_states = []
    kept_flags = []
    for point, flag in zip(found, flags):
        if any(np.linalg.norm(point - seen) <= dedupe_tol for seen in kept_states):
            continue
        kept_states.append(point)
        kept_flags.append(flag)

    return np.asarray(kept_states), kept_flags


def plot_equilibria(
    result,
    p,
    *,
    atol: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Mark the equilibria a continuation run found at parameter ``p``.

    Stable equilibria render as filled circles, unstable ones as hollow
    circles -- the convention ``plot_continuation`` already uses. When the
    branch carries no stability information, all points render neutral.

    Both sides of a fold are marked: this selects every branch point at ``p``,
    not just the nearest one.

    Args:
        result: A :class:`~jaxcont.api.ContinuationResult` or
            :class:`~jaxcont.api.Branch`. This function consumes continuation
            output; it does not solve for equilibria itself.
        p: Parameter value at which to mark equilibria.
        atol: Tolerance for treating a branch point as sitting exactly on
            ``p``, excluding points already claimed by an interior crossing.
            Defaults to a quarter of the local minimum parameter step.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.plot``.

    Returns:
        Matplotlib figure.
    """
    branch = _branch_of(result)
    states, flags = _equilibria_at(branch, p, atol=atol)
    fig, ax = _prepare_axes(ax, figsize)

    seen_labels = set()
    for point, flag in zip(states, flags):
        if flag is None:
            color, face, label = NEUTRAL_COLOR, NEUTRAL_COLOR, "equilibrium"
        elif flag:
            color, face, label = STABLE_COLOR, STABLE_COLOR, "stable equilibrium"
        else:
            color, face, label = UNSTABLE_COLOR, "none", "unstable equilibrium"

        marker_options = {
            "marker": "o",
            "markersize": 9,
            "markeredgewidth": 2.0,
            "linestyle": "none",
            "zorder": 5,
        }
        marker_options.update(kwargs)
        ax.plot(
            [point[0]], [point[1]],
            markeredgecolor=color,
            markerfacecolor=face,
            label=None if label in seen_labels else label,
            **marker_options,
        )
        seen_labels.add(label)

    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k "equilibria" -v`
Expected: PASS (7 tests)

`branch.stable` is populated without passing any `events`, because `ContinuationPar.compute_stability` defaults to `True` (`src/jaxcont/api.py:215`, consumed at line 379).

If `test_equilibria_at_returns_both_sides_of_a_fold` finds only one point, the continuation did not pass the fold — check that `p_span=(-1.0, 0.1)` overshoots `p=0` and that the default `PseudoArclength` algorithm is in use, not `Natural`.

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): add plot_equilibria, marking both sides of a fold"
```

---

### Task 5: `plot_trajectory`

**Files:**
- Modify: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `_require_2d`, `_state_names`, `_prepare_axes`, `DEFAULT_FIGSIZE` from Task 1.
- Produces:
  - `_integrate(problem, u0, p, t_span, n_points, rtol, atol) -> np.ndarray` of shape `(n_points, 2)`.
  - `plot_trajectory(problem, u0=None, p=None, t_span=None, *, n_points=1000, rtol=1e-8, atol=1e-10, arrow=True, color="#262626", ax=None, figsize=DEFAULT_FIGSIZE, **kwargs) -> plt.Figure`

**Two calling forms.** `plot_trajectory(problem, u0, p, t_span)` integrates with `scipy.integrate.solve_ivp`; `plot_trajectory(states_array)` draws a precomputed `(n_steps, 2)` array from diffrax, lyapax, or anywhere else. Mixing them raises `TypeError`. scipy is already a hard dependency, so this adds nothing to the dependency set — do not add diffrax and do not write an integrator.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k trajectory -v`
Expected: FAIL — `ImportError: cannot import name 'plot_trajectory'`

- [ ] **Step 3: Implement `plot_trajectory`**

Append to `src/jaxcont/viz/phase_plane.py`:

```python
def _integrate(problem, u0, p, t_span, n_points, rtol, atol) -> np.ndarray:
    """Integrate the frozen right-hand side with scipy's solve_ivp.

    scipy is already a hard dependency, so this needs no optional extra. The
    right-hand side is jitted once; the per-step JAX -> NumPy round trip is
    negligible for a 2D system over a plotting-length interval.
    """
    from scipy.integrate import solve_ivp

    _require_2d(problem)
    rhs = jax.jit(problem.as_rhs(p))

    def scipy_rhs(_t, y):
        return np.asarray(rhs(jnp.asarray(y)), dtype=float)

    solution = solve_ivp(
        scipy_rhs,
        (float(t_span[0]), float(t_span[1])),
        np.asarray(u0, dtype=float),
        t_eval=np.linspace(float(t_span[0]), float(t_span[1]), n_points),
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Trajectory integration failed: {solution.message}")
    return solution.y.T


def plot_trajectory(
    problem,
    u0=None,
    p=None,
    t_span: Optional[Tuple[float, float]] = None,
    *,
    n_points: int = 1000,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    arrow: bool = True,
    color: str = "#262626",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    **kwargs,
) -> plt.Figure:
    """
    Draw a trajectory of a 2D autonomous system on the phase plane.

    Two calling forms::

        plot_trajectory(problem, u0, p, t_span)   # integrates with scipy
        plot_trajectory(states_array)             # draws a precomputed orbit

    The second form takes any ``(n_steps, 2)`` array, so an orbit computed
    with diffrax, lyapax, or anything else drops straight in. JaxCont does not
    own an integrator; ``problem.as_rhs(p)`` is the bridge to whichever solver
    you prefer.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`, or an
            ``(n_steps, 2)`` array of precomputed states.
        u0: Initial condition (integration form only).
        p: Value of the continuation parameter (integration form only).
        t_span: ``(t_start, t_end)`` (integration form only).
        n_points: Number of output samples (integration form only).
        rtol: Relative tolerance passed to ``solve_ivp``.
        atol: Absolute tolerance passed to ``solve_ivp``.
        arrow: Draw a direction arrow at the trajectory midpoint.
        color: Trajectory color.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.
        **kwargs: Additional options forwarded to ``ax.plot``.

    Returns:
        Matplotlib figure.

    Raises:
        TypeError: If the two calling forms are mixed.
        ValueError: If a precomputed array is not shaped ``(n_steps, 2)``.
        NotImplementedError: If the problem is not two-dimensional.
    """
    from jaxcont.api import BifProblem

    if isinstance(problem, BifProblem):
        if u0 is None or p is None or t_span is None:
            raise TypeError(
                "plot_trajectory(problem, u0, p, t_span) requires u0, p and "
                "t_span; or call plot_trajectory(states_array) with a "
                "precomputed (n_steps, 2) array."
            )
        states = _integrate(problem, u0, p, t_span, n_points, rtol, atol)
        names = _state_names(problem)
    else:
        if u0 is not None or p is not None or t_span is not None:
            raise TypeError(
                "plot_trajectory(states_array) takes no u0, p or t_span; pass "
                "a BifProblem as the first argument to integrate instead."
            )
        states = np.asarray(problem, dtype=float)
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError(
                "A precomputed trajectory must have shape (n_steps, 2); got "
                f"{states.shape}."
            )
        names = None

    fig, ax = _prepare_axes(ax, figsize)

    line_options = {"linewidth": 1.6, "zorder": 4}
    line_options.update(kwargs)
    ax.plot(states[:, 0], states[:, 1], color=color, **line_options)

    if arrow and len(states) >= 3:
        mid = len(states) // 2
        ax.annotate(
            "",
            xy=states[mid + 1], xytext=states[mid],
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.6},
            zorder=4,
        )

    if names is not None:
        ax.set_xlabel(names[0], fontsize=12)
        ax.set_ylabel(names[1], fontsize=12)
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k trajectory -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): add plot_trajectory backed by scipy solve_ivp"
```

---

### Task 6: `plot_phase_plane` wrapper

**Files:**
- Modify: `src/jaxcont/viz/phase_plane.py`
- Test: `tests/test_viz.py` (append)

**Interfaces:**
- Consumes: `plot_nullclines` (Task 2), `plot_vector_field`/`plot_streamlines` (Task 3), `plot_equilibria` (Task 4), `plot_trajectory` (Task 5).
- Produces: `plot_phase_plane(problem, p, xlim, ylim, *, result=None, nullclines=True, vector_field=True, streamlines=False, trajectories=None, resolution=200, density=20, title=None, legend=True, ax=None, figsize=DEFAULT_FIGSIZE) -> plt.Figure`

**`trajectories` accepts `(u0, t_span)` pairs or precomputed `(n_steps, 2)` arrays — never bare initial conditions.** No default `t_span` is defensible across models; guessing one produces either an empty transient or an unreadable tangle depending on the system's timescale.

Draw order is fixed: streamlines, vector field, nullclines, trajectories, equilibria — cheapest to read on top.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k phase_plane -v`
Expected: FAIL — `ImportError: cannot import name 'plot_phase_plane'`

- [ ] **Step 3: Implement `plot_phase_plane`**

Append to `src/jaxcont/viz/phase_plane.py`:

```python
def plot_phase_plane(
    problem,
    p,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    result=None,
    nullclines: bool = True,
    vector_field: bool = True,
    streamlines: bool = False,
    trajectories=None,
    resolution: int = 200,
    density: int = 20,
    title: Optional[str] = None,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
) -> plt.Figure:
    """
    Compose a phase plane for a 2D autonomous system at parameter ``p``.

    A convenience wrapper over :func:`plot_streamlines`,
    :func:`plot_vector_field`, :func:`plot_nullclines`,
    :func:`plot_trajectory`, and :func:`plot_equilibria`, drawn in that order
    so the most informative layers sit on top. Call those directly when you
    need finer control.

    Args:
        problem: A 2D :class:`~jaxcont.api.BifProblem`.
        p: Value of the continuation parameter to freeze the system at.
        xlim: ``(min, max)`` range for the first state component.
        ylim: ``(min, max)`` range for the second state component.
        result: Optional :class:`~jaxcont.api.ContinuationResult` whose
            equilibria at ``p`` are marked. Omit to draw no equilibria.
        nullclines: Draw the nullclines.
        vector_field: Draw the vector-field arrows.
        streamlines: Draw streamlines. Off by default, since they overlap
            visually with the vector field.
        trajectories: Sequence of orbits, each either an ``(u0, t_span)``
            pair to integrate or a precomputed ``(n_steps, 2)`` array. Bare
            initial conditions are deliberately not accepted: no default
            ``t_span`` is defensible across models.
        resolution: Grid points per axis for nullclines and streamlines.
        density: Arrows per axis for the vector field.
        title: Axes title. Defaults to ``"<param_name> = <p>"``; pass ``None``
            for that default and an empty string to omit it.
        legend: Draw the legend when labeled artists are present.
        ax: Matplotlib axes (creates a new figure if None).
        figsize: Figure size when ``ax`` is not supplied.

    Returns:
        Matplotlib figure.

    Raises:
        NotImplementedError: If the problem is not two-dimensional.
    """
    _require_2d(problem)
    fig, ax = _prepare_axes(ax, figsize)

    if streamlines:
        plot_streamlines(problem, p, xlim, ylim, resolution=resolution, ax=ax)
    if vector_field:
        plot_vector_field(problem, p, xlim, ylim, density=density, ax=ax)
    if nullclines:
        plot_nullclines(problem, p, xlim, ylim, resolution=resolution, ax=ax)

    for trajectory in trajectories or ():
        if isinstance(trajectory, tuple) and len(trajectory) == 2:
            u0, t_span = trajectory
            plot_trajectory(problem, u0=u0, p=p, t_span=t_span, ax=ax)
        else:
            plot_trajectory(trajectory, ax=ax)

    if result is not None:
        plot_equilibria(result, p, ax=ax)

    names = _state_names(problem)
    ax.set_xlabel(names[0], fontsize=12)
    ax.set_ylabel(names[1], fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if title is None:
        param_name = getattr(problem, "param_name", None) or "p"
        title = f"{param_name} = {float(p):g}"
    if title:
        ax.set_title(title, fontsize=13)

    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=9)

    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz.py -k phase_plane -v`
Expected: PASS (5 tests)

Note `test_plot_phase_plane_composes_requested_layers_only` asserts exactly 2 lines: the nullcline legend proxies. If it fails with 0, the proxy `ax.plot([], [], ...)` calls in `plot_nullclines` were dropped.

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/phase_plane.py tests/test_viz.py
git commit -m "feat(viz): add plot_phase_plane composing the phase-plane layers"
```

---

### Task 7: Rename `plot_phase_portrait` → `plot_branch_states`, wire up exports

**Files:**
- Modify: `src/jaxcont/viz/portraits.py:31-78` (the `plot_phase_portrait` definition)
- Modify: `src/jaxcont/viz/__init__.py`
- Modify: `tests/test_viz.py:207-225` (the two existing tests) and `tests/test_viz.py:360-366` (the `__all__` assertion)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `jaxcont.viz.plot_branch_states` (the existing behavior, unchanged) and a deprecated `jaxcont.viz.plot_phase_portrait` that warns and forwards. `jaxcont.viz` also exports `plot_phase_plane`, `plot_nullclines`, `plot_vector_field`, `plot_streamlines`, `plot_equilibria`, `plot_trajectory`.

**Why.** The existing `plot_phase_portrait` scatters one dot per branch point in state space, one per parameter value — not a phase portrait in the dynamical-systems sense, and it occupies a name this feature's domain needs. The project is `Development Status :: 3 - Alpha` and pre-1.0, so the deprecation cycle is cheap. Removal target: v0.4.0.

There are **no callers in `examples/`** — `example_03_van_der_pol.py` imports only `plot_eigenvalues`. The only in-repo callers are in `tests/test_viz.py`.

- [ ] **Step 1: Update the existing tests to the new name and add alias tests**

In `tests/test_viz.py`, replace the import at line 207 and the two tests that follow it:

```python
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
```

And update the export assertion near line 360:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz.py -k "branch_states or phase_portrait or exports_public_surface" -v`
Expected: FAIL — `ImportError: cannot import name 'plot_branch_states'`

- [ ] **Step 3: Rename in `portraits.py` and add the alias**

In `src/jaxcont/viz/portraits.py`, add `import warnings` at the top of the imports, rename the function on line 31 from `plot_phase_portrait` to `plot_branch_states`, and replace its summary line so the docstring reads:

```python
def plot_branch_states(
    solution: ContinuationSolution,
    state_indices: Tuple[int, int] = (0, 1),
    param_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Scatter two state components of a continuation branch, one point per
    parameter value.

    This is a picture of the *branch* in state space, not a phase portrait of
    the flow -- for nullclines and vector fields of a 2D system, see
    :func:`jaxcont.viz.plot_phase_plane`. The function was named
    ``plot_phase_portrait`` before v0.3.0.
    ...
```

Leave the rest of the docstring and the entire body unchanged.

Then append the deprecated alias immediately after the function:

```python
def plot_phase_portrait(*args, **kwargs) -> plt.Figure:
    """Deprecated alias for :func:`plot_branch_states`.

    Renamed in v0.3.0: this function scatters branch points in state space, so
    the old name misdescribed it and collided with the real phase-plane plots
    in :mod:`jaxcont.viz.phase_plane`. Scheduled for removal in v0.4.0.
    """
    warnings.warn(
        "plot_phase_portrait is deprecated and will be removed in v0.4.0; use "
        "plot_branch_states instead. For nullclines and vector fields of a 2D "
        "system, see jaxcont.viz.plot_phase_plane.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_branch_states(*args, **kwargs)
```

- [ ] **Step 4: Update the package exports**

Replace the body of `src/jaxcont/viz/__init__.py`:

```python
"""
jaxcont.viz -- consolidated visualization for continuation/bifurcation
diagrams and 2D phase planes. See
docs/superpowers/specs/2026-07-22-viz-module-design.md and
docs/superpowers/specs/2026-07-28-phase-plane-visualization-design.md for the
design rationale behind this module's structure.
"""

from jaxcont.viz.core import plot_all_states, plot_bifurcation_diagram, plot_continuation
from jaxcont.viz.phase_plane import (
    plot_equilibria,
    plot_nullclines,
    plot_phase_plane,
    plot_streamlines,
    plot_trajectory,
    plot_vector_field,
)
from jaxcont.viz.portraits import (
    EigenvalueReference,
    plot_branch_states,
    plot_eigenvalues,
    plot_phase_portrait,
)

__all__ = [
    # Continuation diagrams
    "plot_continuation",
    "plot_bifurcation_diagram",
    "plot_all_states",
    "plot_branch_states",
    "plot_eigenvalues",
    "EigenvalueReference",
    # 2D phase planes
    "plot_phase_plane",
    "plot_nullclines",
    "plot_vector_field",
    "plot_streamlines",
    "plot_equilibria",
    "plot_trajectory",
    # Deprecated, removal target v0.4.0
    "plot_phase_portrait",
]
```

- [ ] **Step 5: Run the full viz suite**

Run: `python -m pytest tests/test_viz.py -v`
Expected: PASS — every test, old and new.

- [ ] **Step 6: Record the change in `CHANGELOG.md`**

Insert a new section directly below the `## [0.2.0] - 2026-07-24` heading's preceding blank line — that is, between the intro paragraph block ending at line 6 and the `## [0.2.0]` heading:

```markdown
## [Unreleased]

### Added
- 2D phase-plane visualization in `jaxcont.viz`: `plot_phase_plane`, `plot_nullclines`,
  `plot_vector_field`, `plot_streamlines`, `plot_equilibria`, and `plot_trajectory`.
  Supports 2D autonomous systems only; trajectories integrate with `scipy.integrate.solve_ivp`
  (already a dependency) or accept a precomputed `(n_steps, 2)` array from any solver.
- Example: FitzHugh–Nagumo phase plane (`example_12`).

### Changed
- `plot_phase_portrait` is renamed to `plot_branch_states`, which describes what it does:
  scatter branch points in state space. The old name remains as a deprecated alias and will be
  removed in v0.4.0.
```

- [ ] **Step 7: Commit**

```bash
git add src/jaxcont/viz/portraits.py src/jaxcont/viz/__init__.py tests/test_viz.py CHANGELOG.md
git commit -m "refactor(viz): rename plot_phase_portrait to plot_branch_states, export phase-plane API"
```

---

### Task 8: FitzHugh–Nagumo example and docs

**Files:**
- Create: `examples/example_12_fitzhugh_nagumo_phase_plane.py`
- Modify: `docs/source/api/index.rst`
- Modify: `notes/ROADMAP.md`

**Interfaces:**
- Consumes: the full public surface from Task 7.
- Produces: no code interfaces — documentation and a runnable example.

The example follows the sphinx-gallery conventions in `examples/example_03_van_der_pol.py`: a module docstring with a reStructuredText title and `.. math::` block, then `# %%` cell markers with comment headings.

- [ ] **Step 1: Write the example**

Create `examples/example_12_fitzhugh_nagumo_phase_plane.py`:

```python
r"""
FitzHugh-Nagumo phase plane
===========================

Continue the equilibrium of the FitzHugh-Nagumo neuron model through its Hopf
bifurcation, then read the same transition off the phase plane.

.. math::

    \dot{v} &= v - v^3/3 - w + I \\
    \dot{w} &= 0.08 (v + 0.7 - 0.8 w)

The cubic :math:`v`-nullcline and the linear :math:`w`-nullcline intersect at
the single equilibrium. As the input current :math:`I` grows the intersection
slides up the cubic's middle branch; where it crosses the local maximum the
equilibrium loses stability at a Hopf bifurcation and a limit cycle appears.
The bifurcation diagram reports the crossing; the phase plane shows the
geometry that produces it.
"""

# %%
# Setup

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import plot_phase_plane

# %%
# Define the system


def fitzhugh_nagumo_rhs(u, p, args):
    v, w = u
    current = p
    a, b, tau = args
    return jnp.array([
        v - v**3 / 3.0 - w + current,
        tau * (v + a - b * w),
    ])


args = (0.7, 0.8, 0.08)

problem = jc.bif_problem(
    fitzhugh_nagumo_rhs,
    u0=jnp.array([-1.2, -0.6]),
    p0=0.0,
    args=args,
    state_names=["v", "w"],
    param_name="I",
)

# %%
# Continue the equilibrium and detect the Hopf bifurcation

result = jc.continuation(
    problem,
    p_span=(0.0, 1.0),
    settings=jc.ContinuationPar(ds=0.01, max_steps=400),
    events=[jc.Hopf()],
)

for event in result.events:
    print(f"{event.kind} at I = {float(event.p):.4f}")

# %%
# Bifurcation diagram beside the phase plane
# ------------------------------------------
#
# The right panel freezes the system at I = 0.5, past the Hopf point: the
# equilibrium is unstable (hollow marker) and the trajectory spirals outward
# onto the limit cycle.

fig, (ax_diagram, ax_plane) = plt.subplots(1, 2, figsize=(13.0, 5.5))

result.plot(state_index=0, ax=ax_diagram, annotate=True)

plot_phase_plane(
    problem,
    p=0.5,
    xlim=(-2.5, 2.5),
    ylim=(-1.0, 2.0),
    result=result,
    vector_field=True,
    trajectories=[(jnp.array([-1.0, -0.5]), (0.0, 200.0))],
    ax=ax_plane,
)

plt.tight_layout()
plt.show()

# %%
# Below the Hopf point the same picture shows a stable equilibrium
# ----------------------------------------------------------------
#
# At I = 0.0 the intersection sits on the cubic's left branch, the marker is
# filled, and the trajectory spirals inward.

fig = plot_phase_plane(
    problem,
    p=0.0,
    xlim=(-2.5, 2.5),
    ylim=(-1.0, 2.0),
    result=result,
    trajectories=[(jnp.array([1.0, 0.0]), (0.0, 200.0))],
)

plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Run the example**

Run: `python examples/example_12_fitzhugh_nagumo_phase_plane.py`
Expected: it completes without error and prints a `hopf` event with `I` somewhere in `(0.3, 0.5)`.

If no Hopf event is reported, widen `p_span` to `(0.0, 1.5)`. If the continuation stalls, lower `ds` to `0.005`. `result.plot(**kwargs)` forwards to `plot_continuation` (`src/jaxcont/core/continuation.py:148`), so `ax=` and `annotate=` pass straight through.

- [ ] **Step 3: Add the docs API section**

Append to `docs/source/api/index.rst`:

```rst
Visualization
-------------

.. autofunction:: jaxcont.viz.plot_continuation

.. autofunction:: jaxcont.viz.plot_eigenvalues

.. autofunction:: jaxcont.viz.plot_branch_states

2D phase planes
~~~~~~~~~~~~~~~

Phase-plane plots support two-dimensional autonomous systems only. Slices of
higher-dimensional systems are deliberately not drawn: on a slice the
zero-contours are not nullclines of the full system and their intersections
are not generally equilibria.

.. autofunction:: jaxcont.viz.plot_phase_plane

.. autofunction:: jaxcont.viz.plot_nullclines

.. autofunction:: jaxcont.viz.plot_vector_field

.. autofunction:: jaxcont.viz.plot_streamlines

.. autofunction:: jaxcont.viz.plot_equilibria

.. autofunction:: jaxcont.viz.plot_trajectory
```

- [ ] **Step 4: Record it in the roadmap**

In `notes/ROADMAP.md`, add to the v0.3.0 section:

```markdown
- [x] Phase-plane visualization for 2D autonomous systems: nullclines, vector fields,
      streamlines, continuation equilibria, and trajectories.
```

- [ ] **Step 5: Run the whole test suite**

Run: `python -m pytest tests/ -x -q`
Expected: PASS. Nothing outside `tests/test_viz.py` should change behavior — this feature touches only the viz layer.

- [ ] **Step 6: Commit**

```bash
git add examples/example_12_fitzhugh_nagumo_phase_plane.py docs/source/api/index.rst notes/ROADMAP.md
git commit -m "docs: add FitzHugh-Nagumo phase-plane example and viz API reference"
```

---

## Verification

After Task 8, confirm the whole feature:

```bash
python -m pytest tests/ -q
python examples/example_12_fitzhugh_nagumo_phase_plane.py
python -c "
import jaxcont.viz as viz
print(sorted(viz.__all__))
"
```

All three must succeed before the branch is considered done. Do not claim completion without the actual output of these commands.
