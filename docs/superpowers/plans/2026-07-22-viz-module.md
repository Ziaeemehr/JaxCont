# jaxcont/viz/ visualization module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate JaxCont's duplicated per-example plotting code (annotated bifurcation
labels in `example_02`, a per-state subplot grid in `example_05`) into a shared `jaxcont/viz/`
subpackage, and fix a latent bug in `plot_phase_portrait` discovered along the way.

**Architecture:** New subpackage `jaxcont/viz/` with three files (`styles.py`, `core.py`,
`portraits.py`) replacing `jaxcont/utils/plotting.py` (deleted outright, no shim). One shared
`BIFURCATION_STYLES` table drives marker/color/label for every bifurcation point everywhere.
`plot_continuation` gains an opt-in `annotate` flag; a new `plot_all_states` thinly loops over
`plot_continuation` for a multi-panel grid. Top-level `jc.plot_continuation`/
`jc.plot_bifurcation_diagram` names are unchanged.

**Tech Stack:** Python, matplotlib, JAX (`jax.numpy`), pytest.

## Global Constraints

- `annotate` on `plot_continuation` defaults to `False` — `example_01`/`06`/`07`'s existing plots
  must render identically unless a caller explicitly opts in.
- `plot_continuation`/`plot_bifurcation_diagram` are **not** renamed — they are top-level
  published API (`jc.plot_continuation`, `jc.plot_bifurcation_diagram`, exported from
  `jaxcont/__init__.py`, on PyPI as of v0.1.0).
- `jaxcont/utils/plotting.py` is deleted outright once its contents are moved — no re-export shim
  (matches this project's established pre-1.0 practice; see
  `docs/superpowers/specs/2026-07-21-engine-consolidation-design.md`).
- `BIFURCATION_STYLES` is keyed by the lowercase `bif_type` strings the detector emits
  (`"fold"`, `"hopf"`, `"period-doubling"`, `"branch-point"`) — **not**
  `bifurcations/taxonomy.py`'s short codes (`"LP"`, `"H"`), which are a separate vocabulary.
  Unknown types fall back to a marker of `"x"`, color `"black"`, with the raw `bif_type` string
  as the label (matches the pre-consolidation behavior being replaced).
- No multi-branch/multi-solution overlay API — out of scope, per the approved spec. Overlaying
  is achieved by calling `plot_continuation(..., ax=shared_ax)` twice, which already works.
- No repo-wide headless matplotlib backend is configured. Every new test file in this plan must
  start with `import matplotlib; matplotlib.use("Agg")` **before** `import matplotlib.pyplot`,
  so tests don't try to open a GUI window in CI or this sandbox.
- Full spec: `docs/superpowers/specs/2026-07-22-viz-module-design.md`.

---

### Task 1: `jaxcont/viz/styles.py` — shared bifurcation style table

**Files:**
- Create: `src/jaxcont/viz/__init__.py` (empty for now — populated in Task 6)
- Create: `src/jaxcont/viz/styles.py`
- Test: `tests/test_viz.py` (new file)

**Interfaces:**
- Produces: `BifStyle` (frozen dataclass, fields `marker: str`, `color: str`, `label: str`),
  `BIFURCATION_STYLES: dict[str, BifStyle]`, `DEFAULT_STYLE: BifStyle`,
  `style_for(bif_type: str) -> BifStyle` — all in `jaxcont.viz.styles`. Later tasks import
  `style_for` from here; nothing else in this task is consumed elsewhere.

- [ ] **Step 1: Create the empty package marker**

Create `src/jaxcont/viz/__init__.py` with just:

```python
"""jaxcont.viz -- consolidated visualization for continuation/bifurcation diagrams."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_viz.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.viz.styles'`

- [ ] **Step 4: Write the implementation**

Create `src/jaxcont/viz/styles.py`:

```python
"""
Shared bifurcation-point styling for jaxcont.viz.

One canonical marker/color/label table, so plot_continuation, plot_all_states,
and any future plotting function style bifurcation markers identically
instead of each hardcoding its own dict (previously duplicated across
utils/plotting.py, example_02_lorenz.py, and example_05_neural_mass.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BifStyle:
    """Marker/color/label for one bifurcation type, e.g. BIFURCATION_STYLES["fold"]."""

    marker: str
    color: str
    label: str


#: Keyed by the lowercase bif_type strings BifurcationDetector emits (see
#: bifurcations/detector.py) -- NOT taxonomy.py's short codes ("LP"/"H"),
#: which are a separate, human-facing vocabulary.
BIFURCATION_STYLES: dict[str, BifStyle] = {
    "fold": BifStyle("s", "green", "Fold"),
    "hopf": BifStyle("^", "magenta", "Hopf"),
    "period-doubling": BifStyle("v", "orange", "PD"),
    "branch-point": BifStyle("D", "purple", "BP"),
}

#: Fallback for a bif_type with no entry above.
DEFAULT_STYLE = BifStyle("x", "black", None)


def style_for(bif_type: str) -> BifStyle:
    """Style for a bifurcation type, falling back to DEFAULT_STYLE (with the
    raw ``bif_type`` string as its label) for unknown types."""
    style = BIFURCATION_STYLES.get(bif_type)
    if style is not None:
        return style
    return BifStyle(DEFAULT_STYLE.marker, DEFAULT_STYLE.color, bif_type)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/viz/__init__.py src/jaxcont/viz/styles.py tests/test_viz.py
git commit -m "feat: add jaxcont.viz.styles shared bifurcation style table"
```

---

### Task 2: `jaxcont/viz/core.py` — move `plot_continuation`/`plot_bifurcation_diagram`

**Files:**
- Create: `src/jaxcont/viz/core.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `style_for` from `jaxcont.viz.styles` (Task 1).
- Produces: `plot_continuation(solution, state_index=0, state_name=None, param_name=None,
  ax=None, show_bifurcations=True, stable_color="blue", unstable_color="red", **kwargs) ->
  plt.Figure` and `plot_bifurcation_diagram(solution, state_index=0, **kwargs) -> plt.Figure`,
  both in `jaxcont.viz.core`. Task 3 adds `annotate` to `plot_continuation`'s signature; Task 4
  calls `plot_continuation` from `plot_all_states`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.viz.core'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/viz/core.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/core.py tests/test_viz.py
git commit -m "feat: move plot_continuation/plot_bifurcation_diagram into jaxcont.viz.core"
```

---

### Task 3: `annotate` option on `plot_continuation`

**Files:**
- Modify: `src/jaxcont/viz/core.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `style_for` (Task 1).
- Produces: `plot_continuation(..., annotate: bool = False, ...)` — the new keyword-only-in-effect
  parameter later tasks/examples pass explicitly (e.g. `example_02` will call
  `plot_continuation(solution, annotate=True)`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
def test_plot_continuation_annotate_false_by_default_no_text_boxes():
    fig = plot_continuation(_simple_solution(with_bifurcation=True))
    assert len(fig.axes[0].texts) == 0


def test_plot_continuation_annotate_true_adds_one_box_per_bifurcation():
    solution = _simple_solution(with_bifurcation=True)
    fig = plot_continuation(solution, annotate=True)
    assert len(fig.axes[0].texts) == len(solution.bifurcations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL — first test passes (no annotate arg = no texts today either), second test FAILS
with `TypeError: plot_continuation() got an unexpected keyword argument 'annotate'`

- [ ] **Step 3: Modify the implementation**

In `src/jaxcont/viz/core.py`, add `annotate` to `plot_continuation`'s signature (right after
`show_bifurcations`):

```python
    show_bifurcations: bool = True,
    annotate: bool = False,
    stable_color: str = "blue",
```

Update the docstring's `Args:` block, inserting after the `show_bifurcations` line:

```python
        annotate: If True, draw a text-box + arrow label next to each
            bifurcation marker showing its type and (parameter, state) value.
```

Inside the `if show_bifurcations and solution.bifurcations:` loop, right after the existing
`ax.plot(...)` call for the marker, add:

```python
            if annotate:
                ax.annotate(
                    f"{style.label or bif_type}\n"
                    f"{param_name}={float(param):.3f}\n"
                    f"{state_name}={float(state_val):.3f}",
                    xy=(param, state_val), xytext=(15, 15), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/core.py tests/test_viz.py
git commit -m "feat: add opt-in annotate flag to plot_continuation"
```

---

### Task 4: `plot_all_states` — multi-panel state grid

**Files:**
- Modify: `src/jaxcont/viz/core.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `plot_continuation(solution, state_index, state_name, param_name, ax,
  show_bifurcations, stable_color, unstable_color)` (Task 2/3, same file).
- Produces: `plot_all_states(solution, param_name=None, state_names=None,
  show_bifurcations=True, stable_color="blue", unstable_color="red", figsize=None) ->
  plt.Figure`, in `jaxcont.viz.core`. `example_05` (Task 9) calls this directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL with `ImportError: cannot import name 'plot_all_states'`

- [ ] **Step 3: Write the implementation**

Append to `src/jaxcont/viz/core.py`:

```python
from typing import Sequence, Tuple


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/core.py tests/test_viz.py
git commit -m "feat: add plot_all_states multi-panel state grid"
```

---

### Task 5: `jaxcont/viz/portraits.py` — move + fix `plot_phase_portrait`, move `plot_eigenvalues`

**Files:**
- Create: `src/jaxcont/viz/portraits.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Produces: `plot_phase_portrait(solution, state_indices=(0, 1), param_indices=None,
  ax=None, **kwargs) -> plt.Figure` (now with a real `ax` parameter — the bug fix) and
  `plot_eigenvalues(solution, ax=None, **kwargs) -> plt.Figure`, both in
  `jaxcont.viz.portraits`. `example_03` (Task 10) passes `ax=` to `plot_phase_portrait` and now
  gets the fix.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
from jaxcont.viz.portraits import plot_eigenvalues, plot_phase_portrait


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jaxcont.viz.portraits'`

- [ ] **Step 3: Write the implementation**

Create `src/jaxcont/viz/portraits.py`:

```python
"""
Phase-portrait and eigenvalue-trajectory plots for jaxcont.viz.
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont.core.continuation import ContinuationSolution


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
    solution: ContinuationSolution,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot eigenvalue trajectories along the branch.

    Args:
        solution: Continuation solution (must have eigenvalues)
        ax: Matplotlib axes
        **kwargs: Additional plotting options

    Returns:
        Matplotlib figure
    """
    if solution.eigenvalues is None:
        raise ValueError("Solution does not contain eigenvalue information")

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None

    params = solution.parameters
    eigenvalues = solution.eigenvalues

    for i in range(eigenvalues.shape[1]):
        ax1.plot(params, jnp.real(eigenvalues[:, i]), '-', alpha=0.7)

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Parameter", fontsize=12)
    ax1.set_ylabel("Re(λ)", fontsize=12)
    ax1.set_title("Real Part of Eigenvalues", fontsize=14)
    ax1.grid(True, alpha=0.3)

    if ax2 is not None:
        for i in range(eigenvalues.shape[1]):
            ax2.plot(params, jnp.imag(eigenvalues[:, i]), '-', alpha=0.7)

        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel("Parameter", fontsize=12)
        ax2.set_ylabel("Im(λ)", fontsize=12)
        ax2.set_title("Imaginary Part of Eigenvalues", fontsize=14)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (18 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/portraits.py tests/test_viz.py
git commit -m "feat: move plot_phase_portrait/plot_eigenvalues, fix missing ax param"
```

---

### Task 6: `jaxcont/viz/__init__.py` — public re-exports

**Files:**
- Modify: `src/jaxcont/viz/__init__.py`
- Modify: `tests/test_viz.py`

**Interfaces:**
- Consumes: `plot_continuation`, `plot_bifurcation_diagram`, `plot_all_states` (Tasks 2-4),
  `plot_phase_portrait`, `plot_eigenvalues` (Task 5).
- Produces: `jaxcont.viz.__all__` — the package's public surface. Task 7 repoints
  `jaxcont/__init__.py`/`jaxcont/utils/__init__.py`/`core/continuation.py` to import from here.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_viz.py`:

```python
def test_viz_package_exports_public_surface():
    import jaxcont.viz as viz

    for name in (
        "plot_continuation", "plot_bifurcation_diagram", "plot_all_states",
        "plot_phase_portrait", "plot_eigenvalues",
    ):
        assert hasattr(viz, name), f"jaxcont.viz missing {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: FAIL — `AssertionError: jaxcont.viz missing plot_continuation`

- [ ] **Step 3: Write the implementation**

Replace `src/jaxcont/viz/__init__.py` entirely with:

```python
"""
jaxcont.viz -- consolidated visualization for continuation/bifurcation
diagrams. See docs/superpowers/specs/2026-07-22-viz-module-design.md for the
design rationale behind this module's structure.
"""

from jaxcont.viz.core import plot_all_states, plot_bifurcation_diagram, plot_continuation
from jaxcont.viz.portraits import plot_eigenvalues, plot_phase_portrait

__all__ = [
    "plot_continuation",
    "plot_bifurcation_diagram",
    "plot_all_states",
    "plot_phase_portrait",
    "plot_eigenvalues",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python -m pytest tests/test_viz.py -v`
Expected: PASS (19 passed)

- [ ] **Step 5: Commit**

```bash
git add src/jaxcont/viz/__init__.py tests/test_viz.py
git commit -m "feat: export jaxcont.viz public surface"
```

---

### Task 7: Delete `jaxcont/utils/plotting.py`, repoint every import site

**Files:**
- Delete: `src/jaxcont/utils/plotting.py`
- Modify: `src/jaxcont/utils/__init__.py`
- Modify: `src/jaxcont/__init__.py`
- Modify: `src/jaxcont/core/continuation.py` (the `ContinuationSolution.plot()` method)
- Modify: `docs/source/user_guide/index.rst`

**Interfaces:**
- Consumes: `jaxcont.viz` public surface (Task 6).
- Produces: nothing new — this task only repoints existing import sites. No further tasks
  depend on this one beyond "the old module no longer exists."

- [ ] **Step 1: Delete the old module**

```bash
rm src/jaxcont/utils/plotting.py
```

- [ ] **Step 2: Update `src/jaxcont/utils/__init__.py`**

Change:

```python
from jaxcont.utils.plotting import plot_bifurcation_diagram, plot_continuation
```

to:

```python
from jaxcont.viz import plot_bifurcation_diagram, plot_continuation
```

(The rest of the file — the `Config`/`test_jax_cuda`/etc. imports and `__all__` list — is
unchanged.)

- [ ] **Step 3: Update `src/jaxcont/__init__.py`**

Change:

```python
from jaxcont.utils.plotting import plot_bifurcation_diagram, plot_continuation
```

to:

```python
from jaxcont.viz import plot_bifurcation_diagram, plot_continuation
```

(Unchanged: the `__all__` list already includes `"plot_bifurcation_diagram"` and
`"plot_continuation"` — no edit needed there.)

- [ ] **Step 4: Update `src/jaxcont/core/continuation.py`**

Find the `ContinuationSolution.plot()` method (around line 144-152):

```python
    def plot(self, **kwargs):
        """
        Plot the continuation diagram.

        Args:
            **kwargs: Additional plotting options
        """
        from jaxcont.utils.plotting import plot_continuation
        return plot_continuation(self, **kwargs)
```

Change the inline import to:

```python
        from jaxcont.viz import plot_continuation
```

- [ ] **Step 5: Update `docs/source/user_guide/index.rst`**

Find (around line 83-84):

```rst
   # 6. Visualize
   from jaxcont.utils.plotting import plot_continuation
   plot_continuation(solution)
```

Change to:

```rst
   # 6. Visualize
   from jaxcont.viz import plot_continuation
   plot_continuation(solution)
```

- [ ] **Step 6: Run the full test suite**

Run: `/home/ziaee/envs/jaxcont/bin/python -m pytest -q`
Expected: same pass count as before this task (no regressions), plus the new `tests/test_viz.py`
tests still passing.

- [ ] **Step 7: Grep sweep for stale references**

Run:

```bash
grep -rn "utils.plotting\|utils import plotting" src/ tests/ examples/ docs/source/user_guide/
```

Expected: no output (empty). If anything appears outside `docs/source/auto_examples/` or
`docs/build/` (Sphinx-Gallery auto-generated, regenerated on the next docs build — not
hand-edited), fix it before continuing.

- [ ] **Step 8: Commit**

```bash
git add -u src/jaxcont/utils/__init__.py src/jaxcont/__init__.py \
  src/jaxcont/core/continuation.py docs/source/user_guide/index.rst
git commit -m "refactor: delete jaxcont.utils.plotting, repoint imports to jaxcont.viz"
```

---

### Task 8: Migrate `example_02_lorenz.py`

**Files:**
- Modify: `examples/example_02_lorenz.py`

**Interfaces:**
- Consumes: `jc.plot_continuation` (top-level, from `jaxcont.viz` via Task 7), which now accepts
  `annotate=True`; `bif_problem(..., state_names=..., param_name=...)` (already existing feature
  from the earlier `state_names`/`param_name` work this session).

- [ ] **Step 1: Add `state_names`/`param_name` to the problem definition**

Find (around line 69):

```python
prob = jc.bif_problem(lorenz84_rhs, u0=u0, p0=F0, args=args)
```

Change to:

```python
prob = jc.bif_problem(
    lorenz84_rhs, u0=u0, p0=F0, args=args,
    state_names=["X", "Y", "Z", "U"], param_name="F",
)
```

- [ ] **Step 2: Delete the hand-rolled plotting function and replace the final plot call**

Find the entire block from the `# %%` comment through the end of the file (around lines 127-177):

```python
# %%
# Plot the bifurcation diagram (X variable)
# --------------------------------------------
# With a 4-D state we pick one variable (X) to plot against the parameter;
# detected bifurcations are annotated on the branch.


def plot_lorenz84_diagram(solution):
    fig, ax = plt.subplots(figsize=(10, 6))

    params_arr = solution.parameters
    X_states = solution.states[:, 0]

    ax.plot(params_arr, X_states, "b-", linewidth=2, alpha=0.7, label="X(F)")
    ax.plot(params_arr, X_states, "b.", markersize=4, alpha=0.5)

    for bif in solution.bifurcations:
        param = bif["parameter"]
        state_X = bif["state"][0]
        bif_type = bif.get("type", "unknown")
        marker, mcolor, label = {
            "fold": ("s", "red", "Fold"),
            "hopf": ("^", "magenta", "Hopf"),
        }.get(bif_type, ("o", "orange", bif_type))

        ax.plot(
            param, state_X, marker, color=mcolor, markersize=12, markeredgewidth=2,
            markerfacecolor=mcolor, markeredgecolor="darkred", label=label, zorder=10,
        )
        ax.annotate(
            f"{label}\nF={param:.3f}\nX={state_X:.3f}",
            xy=(param, state_X), xytext=(15, 15), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9,
        )

    ax.set_xlabel("Parameter F (External Forcing)", fontweight="bold")
    ax.set_ylabel("X", fontweight="bold")
    ax.set_title("Lorenz-84 System: Bifurcation Diagram (X variable)", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    plt.tight_layout()
    return fig


fig = plot_lorenz84_diagram(solution)
plt.savefig("images/lorenz84_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
```

Replace with:

```python
# %%
# Plot the bifurcation diagram (X variable)
# --------------------------------------------
# With a 4-D state we pick one variable (X) to plot against the parameter;
# detected bifurcations are annotated on the branch.

fig = plot_continuation(solution, annotate=True)
fig.axes[0].set_title("Lorenz-84 System: Bifurcation Diagram (X variable)", fontweight="bold")
plt.savefig("images/lorenz84_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 3: Add the import**

Find the import block near the top of the file:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
```

Change to:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.viz import plot_continuation
```

- [ ] **Step 4: Run the example headless and inspect the output**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_02_lorenz.py`
Expected: exits 0, prints the same bifurcation comparison table as before (data/detection is
untouched — only plotting changed), writes `images/lorenz84_bifurcation.png`.

Then view `images/lorenz84_bifurcation.png` and confirm: X vs. F is now colored by stability
(solid=stable/dashed=unstable, since `settings.compute_stability=True` was already set — this is
a genuine, expected visual change from the original always-blue line, documented in the spec's
Risks section) and fold/Hopf points are marked with yellow annotation boxes.

- [ ] **Step 5: Commit**

```bash
git add examples/example_02_lorenz.py
git commit -m "refactor: migrate example_02_lorenz onto jaxcont.viz.plot_continuation"
```

---

### Task 9: Migrate `example_05_neural_mass.py`

**Files:**
- Modify: `examples/example_05_neural_mass.py`

**Interfaces:**
- Consumes: `jc.plot_all_states` — needs adding to the top-level `jc.*` namespace, OR imported
  directly from `jaxcont.viz` (this task imports directly from `jaxcont.viz`, matching
  `example_02`'s pattern from Task 8 and avoiding an unplanned top-level API addition).

- [ ] **Step 1: Add `state_names`/`param_name` to the problem definition**

Find (around line 87):

```python
prob = jc.bif_problem(TMvf, u0=z0, p0=E0_0, args=args)
```

Change to:

```python
prob = jc.bif_problem(
    TMvf, u0=z0, p0=E0_0, args=args,
    state_names=["E", "x", "u"], param_name="E0",
)
```

- [ ] **Step 2: Replace the manual subplot loop**

Find (around lines 143-166):

```python
# %%
# Plot all three state variables against E0
# ---------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 12))
var_names = ["E", "x", "u"]
var_labels = ["Neural Activity E", "Recovery Variable x", "Adaptation Variable u"]
colors = ["blue", "green", "red"]

for i, (ax, name, label, color) in enumerate(zip(axes, var_names, var_labels, colors)):
    ax.plot(solution.parameters, solution.states[:, i], color=color, linewidth=2,
            label=f"{name} equilibrium", alpha=0.7)
    for bif in solution.bifurcations:
        ax.plot(bif["parameter"], bif["state"][i], "rs", markersize=10,
                markeredgewidth=2, markerfacecolor="red", markeredgecolor="darkred", zorder=10)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best")

axes[-1].set_xlabel("External Input E0")
plt.suptitle("Neural Mass Model - Bifurcation Diagram")
plt.tight_layout()
plt.savefig("images/neural_mass_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
```

Replace with:

```python
# %%
# Plot all three state variables against E0
# ---------------------------------------------

fig = plot_all_states(solution)
plt.suptitle("Neural Mass Model - Bifurcation Diagram")
plt.savefig("images/neural_mass_bifurcation.png", dpi=150, bbox_inches="tight")
plt.show()
```

- [ ] **Step 3: Add the import**

Find:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.solvers.newton import NewtonSolver
```

Change to:

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxcont as jc
from jaxcont.solvers.newton import NewtonSolver
from jaxcont.viz import plot_all_states
```

- [ ] **Step 4: Run the example headless and inspect the output**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_05_neural_mass.py`
Expected: exits 0, same printed comparison table as before, writes
`images/neural_mass_bifurcation.png` with 3 stacked subplots (E, x, u vs. E0), fold/Hopf markers
per subplot, one shared legend instead of three repeated ones.

- [ ] **Step 5: Commit**

```bash
git add examples/example_05_neural_mass.py
git commit -m "refactor: migrate example_05_neural_mass onto jaxcont.viz.plot_all_states"
```

---

### Task 10: Migrate `example_03_van_der_pol.py`, confirm the `ax` bug fix

**Files:**
- Modify: `examples/example_03_van_der_pol.py`

**Interfaces:**
- Consumes: `plot_phase_portrait(solution, state_indices=(0, 1), ax=ax2)` — now honors `ax`
  (Task 5's fix).

- [ ] **Step 1: Update the import**

Find:

```python
import jaxcont as jc
from jaxcont.utils.plotting import plot_phase_portrait
```

Change to:

```python
import jaxcont as jc
from jaxcont.viz import plot_phase_portrait
```

- [ ] **Step 2: Run the example headless and inspect the output**

Run: `MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python examples/example_03_van_der_pol.py`
Expected: exits 0, same printed output as before.

Then view `van_der_pol.png` and confirm it is now a genuine **two-panel** figure — "Bifurcation
Diagram" on the left (`ax1`) and "Phase Portrait" on the right (`ax2`) — rather than only the
single phase-portrait panel it silently produced before this plan's fix (see the bug found while
writing this plan: `plot_phase_portrait` previously had no `ax` parameter, so `ax=ax2` was
dropped and `plt.savefig()` grabbed `plot_phase_portrait`'s own standalone figure instead of the
intended two-panel one).

- [ ] **Step 3: Commit**

```bash
git add examples/example_03_van_der_pol.py van_der_pol.png
git commit -m "refactor: migrate example_03_van_der_pol to jaxcont.viz (fixes two-panel plot)"
```

---

### Task 11: Final verification sweep and roadmap update

**Files:**
- Modify: `notes/ROADMAP.md`

**Interfaces:**
- Consumes: everything from Tasks 1-10 (no new production code).

- [ ] **Step 1: Run the full test suite**

Run: `/home/ziaee/envs/jaxcont/bin/python -m pytest -q`
Expected: all tests pass (previous count + the ~19 new `tests/test_viz.py` tests), 0 failures.

- [ ] **Step 2: Re-run every migrated/touched example headless**

Run:

```bash
for f in examples/example_01_pitchfork.py examples/example_02_lorenz.py \
         examples/example_03_van_der_pol.py examples/example_05_neural_mass.py \
         examples/example_06_vmap_sweep.py; do
  echo "=== $f ==="
  MPLBACKEND=Agg /home/ziaee/envs/jaxcont/bin/python "$f" || echo "FAILED: $f"
done
```

Expected: every script exits 0, no `FAILED` lines. `example_02`/`example_05`'s printed
BifurcationKit.jl comparison tables must show the same matches as before (data/detection unchanged
by this refactor — only presentation changed).

- [ ] **Step 3: Grep sweep for any remaining old-module references**

Run:

```bash
grep -rln "jaxcont.utils.plotting" src/ tests/ examples/ docs/source/ 2>/dev/null | \
  grep -v "docs/source/auto_examples\|docs/build"
```

Expected: empty output.

- [ ] **Step 4: Update `notes/ROADMAP.md`**

Find the status table row (updated earlier this session):

```markdown
| Plotting | ⚠️ Works, 9% cov | Under-tested — consolidation into `jaxcont/viz/` planned, see below |
```

Change to:

```markdown
| Plotting | ✅ Works, tested | Consolidated into `jaxcont/viz/` (2026-07-22) |
```

Find the "Visualization module consolidation (planned, 2026-07-22)" section added earlier this
session and check off every box, e.g.:

```markdown
- [x] Move `plot_continuation`/`plot_bifurcation_diagram`/`plot_phase_portrait`/`plot_eigenvalues`
  into a new `jaxcont/viz/` subpackage (`core.py`/`styles.py`/`portraits.py`), delete
  `jaxcont/utils/plotting.py` outright (matches this project's "remove, don't deprecate" pre-1.0
  practice — see the engine-consolidation entry above). Top-level `jc.plot_continuation`/
  `jc.plot_bifurcation_diagram` names are unaffected.
- [x] Add a single shared `BIFURCATION_STYLES` table (`viz/styles.py`), replacing the three
  independently-hardcoded marker/color dicts in `plotting.py`, `example_02`, and `example_05`.
- [x] Add `annotate: bool = False` to `plot_continuation` (the `example_02` text-box+arrow style,
  opt-in — existing plots unaffected by default) and a new `plot_all_states()` (the `example_05`
  multi-panel style), both consuming the shared style table.
- [x] Migrate `example_02`/`example_05` onto the shared functions; `example_03` gets an import-path
  update only.
- [x] Add `tests/test_viz.py` — closes part of the "Plotting ... Under-tested" gap above (currently
  zero dedicated plotting tests exist).
- [x] Update this table's "Plotting" row once done.
```

Also append one sentence noting the extra fix found along the way, right before the "Design spec"
link line:

```markdown
**Found along the way:** `plot_phase_portrait` had no `ax` parameter, so
`example_03_van_der_pol.py`'s `ax=ax2` call silently dropped into an unused `**kwargs`, and the
function always built its own standalone figure — meaning the script's intended two-panel image
(bifurcation diagram + phase portrait) never actually saved correctly; only the phase-portrait
panel did, because `plt.savefig()` grabs the most-recently-created figure. Fixed alongside this
consolidation (`viz/portraits.py` now has a real `ax` parameter).
```

- [ ] **Step 5: Commit**

```bash
git add notes/ROADMAP.md
git commit -m "docs: mark jaxcont/viz/ consolidation done on the roadmap"
```
