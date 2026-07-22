# `jaxcont/viz/` — consolidated visualization module — design spec

**Date:** 2026-07-22
**Roadmap reference:** `notes/ROADMAP.md`, status table row "Plotting — ⚠️ Works, 9% cov |
Under-tested". No prior roadmap item proposed consolidating per-example plotting code; this spec
introduces that item (see "Roadmap update" below).
**Related work this builds on:** `state_names`/`param_name` fields added to `BifProblem` and
`ContinuationSolution` earlier this session (2026-07-22) — `plot_continuation` already defaults
its axis labels from these when the problem sets them. This spec reuses that plumbing rather than
introducing a second naming mechanism.

## Problem

Bifurcation-diagram plotting is duplicated, not shared, across the example gallery:

1. `src/jaxcont/utils/plotting.py`'s `plot_continuation()` handles exactly one state variable vs.
   the parameter, on one axis, with plain (non-annotated) fold/Hopf markers styled from a
   hardcoded `markers` dict local to the function.
2. `examples/example_02_lorenz.py` defines its own 40-line `plot_lorenz84_diagram()` because it
   wants annotated bifurcation labels (text box + arrow, via `ax.annotate`) — a feature
   `plot_continuation` doesn't have. It hardcodes its own, slightly different marker/color dict.
3. `examples/example_05_neural_mass.py` hand-rolls a 20-line loop building one
   `plt.subplots(3, 1, ...)` panel per state variable, because `plot_continuation` only ever
   plots a single `state_index` per call. It hardcodes a third marker/color scheme (plain red
   squares, no per-type distinction).

Consequence: three independent, drifting copies of "what does a fold/Hopf marker look like," and
two real capabilities (annotated labels, multi-panel state grids) that exist only as one-off
example code instead of reusable library functions — exactly the maintenance risk the user asked
to fix ("we need a module for visualization instead of writing one per example").

`plot_continuation`/`plot_bifurcation_diagram` are also **top-level public API**
(`jc.plot_continuation`, `jc.plot_bifurcation_diagram`, exported from `jaxcont/__init__.py`,
published to PyPI as part of v0.1.0) — any restructuring must preserve these names at the
top-level `jaxcont` namespace, even though the file they live in changes.

## Decision: new `jaxcont/viz/` subpackage, existing names kept, `jaxcont/utils/plotting.py` deleted

Per this project's own established practice for pre-1.0 internal reorganization (see
`docs/superpowers/specs/2026-07-21-engine-consolidation-design.md`'s "full removal, not
deprecation" decision): move the implementation, delete the old file outright, do not leave a
compatibility shim module. This is safe here specifically because the *public* import path
(`jaxcont.plot_continuation` / `from jaxcont import plot_continuation`) does not change — only the
internal `jaxcont.utils.plotting` module path does. `docs/source/quickstart.rst` and the README
already import through the top-level `jaxcont` name, so neither needs touching. The one place that
does reference the internal path directly is `docs/source/user_guide/index.rst`, updated as part
of this work — see below.

Function names are **not** renamed (`plot_continuation` stays `plot_continuation`, not
`plot_branch`) — the earlier "Approach A" brainstorm draft considered a rename but that was before
confirming these are top-level published exports; renaming them would be a gratuitous breaking
change the user didn't ask for.

## Architecture

### New files

- **`src/jaxcont/viz/__init__.py`** — re-exports the public surface: `plot_continuation`,
  `plot_bifurcation_diagram`, `plot_all_states`, `plot_phase_portrait`, `plot_eigenvalues`.
- **`src/jaxcont/viz/styles.py`** — single source of truth for bifurcation-point styling:

  ```python
  @dataclass(frozen=True)
  class BifStyle:
      marker: str
      color: str
      label: str

  BIFURCATION_STYLES: dict[str, BifStyle] = {
      "fold": BifStyle("s", "green", "Fold"),
      "hopf": BifStyle("^", "magenta", "Hopf"),
      "period-doubling": BifStyle("v", "orange", "PD"),
      "branch-point": BifStyle("D", "purple", "BP"),
  }
  DEFAULT_STYLE = BifStyle("x", "black", None)  # falls back to bif_type string as label
  ```

  Keyed by the same lowercase `bif_type` strings the detector already emits (`"fold"`, `"hopf"`,
  etc. — confirmed in `bifurcations/detector.py`), **not** `taxonomy.py`'s short codes (`"LP"`,
  `"H"`) — those are a different, human-facing vocabulary used for `describe()`, and translating
  between the two isn't needed for this work.
- **`src/jaxcont/viz/core.py`** — `plot_continuation()` (moved from `utils/plotting.py`, plus the
  new `annotate` parameter below), `plot_bifurcation_diagram()` (existing alias, moved unchanged),
  and the new `plot_all_states()`.
- **`src/jaxcont/viz/portraits.py`** — `plot_phase_portrait()` and `plot_eigenvalues()`, moved
  unchanged (no new features requested for these in this pass).

### `plot_continuation`: new `annotate` parameter

```python
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
```

When `annotate=True` (default `False` — existing plots in `example_01/03/06/07` render identically
unless they opt in), each bifurcation marker additionally gets an `ax.annotate(...)` text box +
arrow, in the style currently hand-rolled in `example_02`'s `plot_lorenz84_diagram`:

```python
if annotate:
    ax.annotate(
        f"{style.label}\n{param_name}={param:.3f}\n{state_name}={state_val:.3f}",
        xy=(param, state_val), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5), fontsize=9,
    )
```

Marker styling (currently a hardcoded `markers` dict inside the function) is replaced by a lookup
into `BIFURCATION_STYLES`, so this is also the point where the three drifting style copies
converge onto one.

### New function: `plot_all_states`

```python
def plot_all_states(
    solution: ContinuationSolution,
    param_name: Optional[str] = None,
    state_names: Optional[Sequence[str]] = None,
    show_bifurcations: bool = True,
    stable_color: str = "blue",
    unstable_color: str = "red",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
```

Builds a `solution.state_dim`-row, 1-column figure and calls `plot_continuation(solution,
state_index=i, state_name=..., param_name=..., ax=axes[i], show_bifurcations=..., ...)` once per
row — it is a thin loop over the existing single-panel function, not a parallel implementation, so
it cannot drift from `plot_continuation`'s stability-coloring/marker behavior. Only the bottom
subplot gets an x-axis label; a single shared legend is built from the first subplot's handles
(deduplicated) rather than repeating "Stable"/"Unstable"/"Fold" in every row (an improvement over
`example_05`'s current per-panel-repeated legend).

`state_names`/`param_name` follow the same resolution order already implemented for
`plot_continuation`: explicit argument → `solution.state_names`/`solution.param_name` (from the
originating `bif_problem(...)`) → generic `"State[i]"`/`"Parameter"` fallback.

### Multi-branch overlay: explicitly out of scope

Per the brainstorm discussion: `plot_continuation(solution, ax=my_ax)` already composes for
overlaying two solutions (call it twice against the same `ax`). No dedicated
`plot_continuation(solutions=[...])` API is added — nothing in the current examples needs it, and
building it now would be speculative surface area (consistent with the roadmap's own
demand-driven-scope stance elsewhere).

### Deletions and import updates

- `src/jaxcont/utils/plotting.py` deleted (whole file).
- `src/jaxcont/utils/__init__.py`: `from jaxcont.utils.plotting import ...` →
  `from jaxcont.viz import ...`.
- `src/jaxcont/__init__.py`: same import-path change; the two exported names
  (`plot_bifurcation_diagram`, `plot_continuation`) are unchanged.
- `src/jaxcont/core/continuation.py`: `ContinuationSolution.plot()`'s inline
  `from jaxcont.utils.plotting import plot_continuation` → `from jaxcont.viz import
  plot_continuation`.
- `docs/source/user_guide/index.rst`: its code sample `from jaxcont.utils.plotting import
  plot_continuation` → `from jaxcont.viz import plot_continuation` (the one place, besides the
  package internals, with a direct reference to the old module path).

## Example migration

- **`example_01_pitchfork.py`** — no change needed; already calls bare `plot_continuation(solution)`
  and gets its `state_name="x"`/`param_name="r"` from `bif_problem(...)` (this session's earlier
  change). Confirms the new module is a drop-in for existing call sites.
- **`example_02_lorenz.py`** — delete the entire `plot_lorenz84_diagram()` function (~40 lines).
  Add `state_names=["X", "Y", "Z", "U"], param_name="F"` to its existing `bif_problem(...)` call.
  Replace the final plotting block with:
  ```python
  fig = plot_continuation(solution, annotate=True)
  ```
  (imports `plot_continuation` from `jaxcont.viz` instead of defining a local function). Since this
  example never sets `compute_stability` meaningfully for its plot today (the hand-rolled function
  plots one solid blue line regardless of stability), verify after migration whether
  `settings.compute_stability=True` (already set, per the `ContinuationPar` call) now produces a
  stable/unstable-colored plot where the original was uniformly blue — this is a **visible behavior
  change** to confirm looks correct (arguably more informative than the original), not silently
  accept as a side effect.
- **`example_05_neural_mass.py`** — delete the `fig, axes = plt.subplots(3, 1, ...)` loop (~20
  lines). Add `state_names=["E", "x", "u"], param_name="E0"` to its existing `bif_problem(...)`
  call. Replace with:
  ```python
  fig = plot_all_states(solution)
  plt.suptitle("Neural Mass Model - Bifurcation Diagram")
  ```
- **`example_03_van_der_pol.py`** — import path only: `from jaxcont.utils.plotting import
  plot_phase_portrait` → `from jaxcont.viz import plot_phase_portrait`. No behavior change.

## Testing

`notes/ROADMAP.md` already flags plotting at 9% coverage with zero dedicated test file found in
`tests/` (confirmed: no `test_plot*.py`/`test_viz*.py` exists). This is new test-writing, not
migration of existing assertions. New `tests/test_viz.py` covers, headless (`MPLBACKEND=Agg`):

- `BIFURCATION_STYLES` has an entry for every `bif_type` string the detector can emit (`"fold"`,
  `"hopf"`; cross-check against `bifurcations/detector.py`'s literals) — a regression guard so a
  new bifurcation type added later doesn't silently fall through to `DEFAULT_STYLE` unnoticed.
- `plot_continuation(solution, annotate=True)` does not raise, and produces exactly
  `len(solution.bifurcations)` annotation artists on the returned axes.
- `plot_all_states(solution)` produces exactly `solution.state_dim` subplots (`len(fig.axes)`),
  and each subplot's y-label matches the corresponding `state_names` entry when set on the
  originating problem.
- `state_name`/`param_name` resolution order (explicit arg → problem-supplied → generic fallback)
  for both `plot_continuation` and `plot_all_states`, using a synthetic `ContinuationSolution` for
  the no-names case and one built via `bif_problem(..., state_names=..., param_name=...)` →
  `jc.continuation(...)` for the names-supplied case.

## Roadmap update

`notes/ROADMAP.md` changes as part of this work (not deferred to a follow-up):

- Status table "Plotting" row: updated from "⚠️ Works, 9% cov | Under-tested" to reflect the
  `viz/` consolidation and new test coverage once implemented.
- A new dated entry recording: what was duplicated, the `jaxcont/viz/` restructuring, and the
  `annotate`/`plot_all_states` additions — so a future contributor sees this was a deliberate
  consolidation, not an oversight, matching this roadmap's existing style for past sessions.

## Verification plan

- Full test suite green (`pytest`), including the new `tests/test_viz.py`.
- Each of `example_01/02/03/05/06` re-run headless (`MPLBACKEND=Agg`) after migration; visually
  confirm (read the saved PNG) that `example_02`'s and `example_05`'s output still convey the
  intended information after switching to the shared functions — not just "runs without crashing."
  `example_02`'s stability-coloring change (noted above) is specifically checked, not assumed.
- `grep` sweep confirming zero remaining references to `jaxcont.utils.plotting` anywhere in `src/`,
  `tests/`, `examples/`, `docs/source/user_guide/` (excluding this spec, the roadmap, and
  Sphinx-Gallery's auto-generated `docs/source/auto_examples/`/`docs/build/`, which regenerate from
  the examples on the next docs build and are not hand-edited).

## Risks

- **`example_02`'s stability-coloring behavior change** (described above) could be read as a
  regression if the resulting plot looks meaningfully different from the cross-validated original
  figure, even though the underlying data and detected bifurcation points are unchanged (only the
  line's stable/unstable coloring is new). Mitigate by checking the rendered PNG, not just that the
  script exits 0.
- **`BIFURCATION_STYLES` missing a future bifurcation type silently falling back to `DEFAULT_STYLE`
  (black "x", no legend label)** rather than erroring — acceptable for now (matches the existing
  `markers.get(bif_type, ("x", "black", bif_type))` fallback behavior being consolidated), but worth
  the explicit coverage-guard test above so it's a visible, tested default rather than an
  unconsidered gap.
