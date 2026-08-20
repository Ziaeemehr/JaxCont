# MatCont Visual Comparison Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build five runnable Sphinx-Gallery examples that honestly visualize JaxCont results against reviewed MatCont 7.6 equilibrium and periodic-orbit references.

**Architecture:** Keep model narration in one gallery script per validation case and centralize all registry execution, artifact parsing, plotting, status annotation, and PNG output in `examples/MatCont/visualize.py`. Equilibrium Hopf cases add a spectral-abscissa panel; radial and torBPC periodic cases use dedicated three-panel renderers because their artifact shapes and useful stability views differ.

**Tech Stack:** Python 3, JAX/JaxCont registered validation producers, NumPy, Matplotlib, CSV reference artifacts, pytest, Sphinx-Gallery.

**Spec:** `docs/superpowers/specs/2026-08-19-matcont-visual-gallery-design.md`

## Global Constraints

- Use the committed reviewed MatCont 7.6 CSV artifacts; do not run MATLAB during gallery execution.
- Plot JaxCont and MatCont on their native adaptive meshes; do not fabricate paired sample points.
- Restrict parameter-based panels to the shared parameter domain.
- The first four cases must display `Systematic comparison: PASS` only after their existing numerical comparison passes.
- `MC-LC-002` must display `Systematic comparison: FAIL (known limitation)` and retain the existing event and multiplier tolerances.
- Reviewed files under `examples/MatCont/reference/` are read-only.
- Every script must run as a repository-root module and in the Sphinx-Gallery execution context.

---

### Task 1: Establish the Named Cubic Example and Shared Registered-Case Boundary

**Files:**
- Modify: `examples/MatCont/visualize.py`
- Rename: `examples/example_16_matcont_overlay.py` to `examples/example_16_matcont_cubic_overlay.py`
- Modify: `tests/test_matcont_visualization.py`
- Modify: `docs/source/conf.py`

**Interfaces:**
- Consumes: `load_registry()`, registry entry points such as `examples.MatCont.python_cases.equilibrium:run_cubic_fold`, and `CaseResult`.
- Produces: `_run_registered_case(case_id: str, reference_dir: Path) -> tuple[dict, CaseResult]`, `_save_figure(figure, output_path: Path | str | None) -> None`, and the existing `render_case_overlay(...)` compatibility entry point.

- [ ] **Step 1: Change the gallery smoke test to require the named cubic module**

Update the final test in `tests/test_matcont_visualization.py` so it executes:

```python
script = repository / "examples" / "example_16_matcont_cubic_overlay.py"
completed = subprocess.run(
    [sys.executable, str(script)],
    cwd=tmp_path,
    env=environment,
    text=True,
    capture_output=True,
    timeout=120,
)

assert completed.returncode == 0, completed.stdout + completed.stderr
assert (tmp_path / "images" / "matcont_cubic_overlay.png").is_file()
```

- [ ] **Step 2: Run the smoke test and verify that the new filename is absent**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py::test_gallery_example_runs_from_outside_repository_root -v
```

Expected: FAIL because `example_16_matcont_cubic_overlay.py` does not exist.

- [ ] **Step 3: Rename the script and extract registry execution/output helpers**

Rename the script with `mv` or the repository-safe equivalent. In
`examples/MatCont/visualize.py`, replace the inline registry/import logic with:

```python
def _run_registered_case(case_id: str, reference_dir: Path):
    registry = load_registry()
    try:
        case = next(item for item in registry["cases"] if item["id"] == case_id)
    except StopIteration as exc:
        raise ValueError(f"unknown MatCont validation case: {case_id}") from exc
    if case["support"] != "supported" or not case.get("python"):
        raise ValueError(f"case has no supported JaxCont producer: {case_id}")
    module_name, separator, function_name = case["python"].partition(":")
    if not separator:
        raise ValueError(f"invalid Python case entry point: {case['python']}")
    function = getattr(importlib.import_module(module_name), function_name)
    parameters = inspect.signature(function).parameters
    result = function(reference_dir) if parameters else function()
    return case, result


def _save_figure(figure, output_path):
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
```

Keep `render_case_overlay` working for `MC-EQ-001`, and have it call both
helpers. Update `docs/source/conf.py` only if the renamed module changes the
gallery ignore pattern or path assumptions.

- [ ] **Step 4: Run the cubic visualization tests**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -v
```

Expected: all current cubic tests PASS.

- [ ] **Step 5: Commit the cubic baseline**

```bash
git add docs/source/conf.py examples/MatCont/visualize.py examples/example_16_matcont_cubic_overlay.py tests/test_matcont_visualization.py
git commit -m "feat: add MatCont cubic visual comparison"
```

---

### Task 2: Add Spectral Panels and Gallery Pages for Hopf Equilibria

**Files:**
- Modify: `examples/MatCont/visualize.py`
- Create: `examples/example_17_matcont_vanderpol_overlay.py`
- Create: `examples/example_18_matcont_adaptive_control_overlay.py`
- Modify: `tests/test_matcont_visualization.py`

**Interfaces:**
- Consumes: `_run_registered_case`, `_save_figure`, equilibrium `CaseResult.artifacts["parameters" | "states" | "spectra" | "events"]`, and normalized `*_multipliers.csv` rows.
- Produces: `plot_equilibrium_overlay(..., include_spectrum: bool = False)` returning a one- or two-panel `Figure`, and `render_equilibrium_overlay(case_id: str, ..., include_spectrum: bool | None = None)`.

- [ ] **Step 1: Write failing tests for Hopf spectral overlays**

Import `pytest`, then use both registered Hopf cases to assert that the second
panel contains both solvers and the Hopf marker:

```python
@pytest.mark.parametrize("case_id", ["MC-EQ-002", "MC-EQ-003"])
def test_registered_hopf_case_renders_spectral_pass_figure(case_id, tmp_path):
    from examples.MatCont.visualize import render_equilibrium_overlay

    output = tmp_path / f"{case_id}.png"
    figure = render_equilibrium_overlay(case_id, output_path=output)

    assert len(figure.axes) == 2
    labels = {
        artist.get_label()
        for artist in [*figure.axes[1].lines, *figure.axes[1].collections]
    }
    assert {"JaxCont spectral abscissa", "MatCont 7.6 spectral abscissa"} <= labels
    assert figure.axes[1].get_ylabel() == "Largest Re(eigenvalue)"
    assert any("H" in label for label in labels)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Systematic comparison: PASS" in " ".join(
        text.get_text() for text in figure.texts
    )
```

- [ ] **Step 2: Run the new Hopf tests and verify the missing interface**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -k hopf -v
```

Expected: FAIL because `include_spectrum` and `render_equilibrium_overlay` do
not exist.

- [ ] **Step 3: Implement branch-spectrum parsing and spectral-abscissa plotting**

Add a helper that groups branch spectra by `point` and computes the maximum
real component:

```python
def _branch_spectral_abscissa(rows):
    values_by_point = {}
    for row in rows:
        if int(row["event_index"]) != -1:
            continue
        values_by_point.setdefault(int(row["point"]), []).append(float(row["real"]))
    return {
        point: max(real_parts)
        for point, real_parts in values_by_point.items()
    }
```

When `include_spectrum=True`, create `plt.subplots(2, 1, sharex=True,
figsize=(9, 8))`. Plot JaxCont's `max(real(spectra), axis=1)` and MatCont's
point-linked values in the second panel, add a horizontal zero line, and mark
the registered Hopf event from each solver. Preserve the current single-axis
cubic layout when `include_spectrum=False`.

Implement `render_equilibrium_overlay` using `_run_registered_case`, reject
non-`MC-EQ-` IDs, default `include_spectrum` to true when the registry features
contain `hopf`, run `compare_case_result_to_reference`, annotate `PASS` with
branch/event/spectrum maximum errors, and call `_save_figure`.

- [ ] **Step 4: Create the Van der Pol and adaptive-control gallery scripts**

Each script follows the cubic page's Sphinx-Gallery structure and calls:

```python
figure = render_equilibrium_overlay(
    "MC-EQ-002",  # MC-EQ-003 in example 18
    output_path=Path("images") / "matcont_vanderpol_overlay.png",
    parameter_name=r"$\mu$",
    state_name=r"$x$",
    title="Van der Pol Hopf: JaxCont vs MatCont 7.6",
)
plt.show()
```

For `MC-EQ-003`, use parameter label `r"$\alpha$"`, state label `r"$x_1$"`,
and output `images/matcont_adaptive_control_overlay.png`. Explain in each page
that the stability crossing is more informative than the geometrically trivial
equilibrium branch.

- [ ] **Step 5: Run Hopf and cubic tests together**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -v
```

Expected: PASS, with the cubic renderer still producing one panel.

- [ ] **Step 6: Commit the equilibrium gallery expansion**

```bash
git add examples/MatCont/visualize.py examples/example_17_matcont_vanderpol_overlay.py examples/example_18_matcont_adaptive_control_overlay.py tests/test_matcont_visualization.py
git commit -m "feat: visualize MatCont Hopf equilibrium comparisons"
```

---

### Task 3: Add the Radial Periodic-Orbit Multipanel Comparison

**Files:**
- Modify: `examples/MatCont/visualize.py`
- Create: `examples/example_19_matcont_radial_cycle_overlay.py`
- Modify: `tests/test_matcont_visualization.py`

**Interfaces:**
- Consumes: `_run_registered_case`, `_save_figure`, `run_radial_cycle()` artifacts (`parameters`, `periods`, `multipliers`, `state_min`, `state_max`), and `MC-LC-001` branch/multiplier CSV files.
- Produces: `plot_radial_cycle_overlay(result: CaseResult, reference_dir: Path | str, *, parameter_name: str = "Continuation parameter", title: str | None = None)` and `render_periodic_overlay("MC-LC-001", ...)`.

- [ ] **Step 1: Write a failing low-level radial multipanel test**

Use the registered radial producer and reviewed files, then assert:

```python
def test_registered_radial_cycle_renders_three_panel_pass_figure(tmp_path):
    from examples.MatCont.visualize import render_periodic_overlay

    output = tmp_path / "radial.png"
    figure = render_periodic_overlay(
        "MC-LC-001", output_path=output, parameter_name=r"$\rho$"
    )

    assert len(figure.axes) == 3
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "Orbit amplitude",
        "Period",
        "Nontrivial |Floquet multiplier|",
    ]
    for axis in figure.axes:
        labels = axis.get_legend_handles_labels()[1]
        assert any(label.startswith("JaxCont") for label in labels)
        assert any(label.startswith("MatCont 7.6") for label in labels)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Systematic comparison: PASS" in " ".join(
        text.get_text() for text in figure.texts
    )
```

- [ ] **Step 2: Run the radial test and verify the plotter is missing**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py::test_registered_radial_cycle_renders_three_panel_pass_figure -v
```

Expected: FAIL importing `plot_radial_cycle_overlay`.

- [ ] **Step 3: Implement radial artifact mapping and plotting**

Read `state_0_min`, `state_0_max`, `period`, and branch Floquet rows. For each
MatCont point, remove exactly one multiplier nearest `+1` and plot the modulus
of the remaining multiplier. Apply the same trivial-multiplier removal to every
JaxCont spectrum row. Draw three shared-x panels, use min/max envelope lines in
the first panel, and apply the intersection of both parameter ranges to all
three axes.

Extend `render_periodic_overlay` to accept only `MC-LC-001` at this stage. Run
`compare_case_result_to_reference`, annotate `PASS` with branch and spectrum
errors plus the producer's period/radius errors, save, and return the figure.

- [ ] **Step 4: Write the radial gallery page**

Create `examples/example_19_matcont_radial_cycle_overlay.py` with the analytic
model `r' = r(rho-r^2), theta' = 1`, explain the expected radius `sqrt(rho)`,
period `2*pi`, and nontrivial multiplier `exp(-4*pi*rho)`, then call:

```python
figure = render_periodic_overlay(
    "MC-LC-001",
    output_path=Path("images") / "matcont_radial_cycle_overlay.png",
    parameter_name=r"$\rho$",
    title="Radial periodic orbit: JaxCont vs MatCont 7.6",
)
plt.show()
```

- [ ] **Step 5: Run the radial and existing visualization tests**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -k "radial or equilibrium or cubic" -v
```

Expected: PASS.

- [ ] **Step 6: Commit the radial comparison**

```bash
git add examples/MatCont/visualize.py examples/example_19_matcont_radial_cycle_overlay.py tests/test_matcont_visualization.py
git commit -m "feat: visualize radial periodic orbit against MatCont"
```

---

### Task 4: Add the Honest torBPC Diagnostic Multipanel Comparison

**Files:**
- Modify: `examples/MatCont/visualize.py`
- Create: `examples/example_20_matcont_torbpc_overlay.py`
- Modify: `tests/test_matcont_visualization.py`

**Interfaces:**
- Consumes: `run_torbpc_cycle(reference_dir)` artifacts (`branch`, `events`, `multipliers`, `jaxcont_parameters`, `jaxcont_states`, `jaxcont_multipliers`, `jaxcont_orbits`) and checks (`all_comparisons_pass`, event, period, extrema, and multiplier errors).
- Produces: `plot_torbpc_overlay(result: CaseResult, reference_dir: Path | str, *, state_index: int = 0, parameter_name: str = r"$\nu$", state_name: str | None = None, title: str | None = None)` and `render_periodic_overlay("MC-LC-002", ...)`.

- [ ] **Step 1: Write failing torBPC multipanel/status tests**

Use the registered producer and reviewed references so the test also locks in
the honest failing status:

```python
def test_registered_torbpc_renderer_reports_known_failure(tmp_path):
    from examples.MatCont.visualize import render_periodic_overlay

    figure = render_periodic_overlay(
        "MC-LC-002", output_path=tmp_path / "torbpc.png"
    )
    summary = " ".join(text.get_text() for text in figure.texts)

    assert "Systematic comparison: FAIL (known limitation)" in summary
    assert "multiplier" in summary
    assert "event" in summary
    assert (tmp_path / "torbpc.png").is_file()
    assert len(figure.axes) == 3
    assert figure.axes[2].get_xlabel() == "Re(Floquet multiplier)"
    assert figure.axes[2].get_ylabel() == "Im(Floquet multiplier)"
    labels = {
        label
        for axis in figure.axes
        for label in axis.get_legend_handles_labels()[1]
    }
    for event_type in ("LPC", "NS", "PD"):
        assert f"MatCont 7.6 {event_type}" in labels
        assert f"JaxCont near {event_type}" in labels
```

- [ ] **Step 2: Run the torBPC tests and verify the missing plotter/dispatch**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -k torbpc -v
```

Expected: FAIL importing `plot_torbpc_overlay` or rejecting `MC-LC-002`.

- [ ] **Step 3: Implement torBPC envelope, period, and event-spectrum panels**

Map JaxCont orbit minima/maxima from `jaxcont_orbits`; map MatCont extrema and
periods from branch rows. For each MatCont LPC/NS/PD event parameter, select the
nearest JaxCont branch point and label its spectrum `JaxCont near <type>`; use
`event_index`/`event_type` for the MatCont event spectrum. Plot actual detected
JaxCont hits separately in the parameter panels so missed or displaced labels
remain visible. Plot the unit circle in the complex panel:

```python
angle = np.linspace(0.0, 2.0 * np.pi, 361)
spectrum_axis.plot(np.cos(angle), np.sin(angle), color="#94a3b8", linestyle="--", label="Unit circle")
spectrum_axis.set_aspect("equal", adjustable="box")
```

Use event-specific colors consistently across all panels. Preserve actual
computed locations and values even when they do not overlap.

Extend `render_periodic_overlay` to dispatch `MC-LC-002`, build the status from
`result.checks["all_comparisons_pass"]`, and annotate the expected current
failure with `jaxcont_lpc_parameter_error`, `jaxcont_ns_parameter_error`,
`jaxcont_pd_parameter_error`, `jaxcont_max_period_error`,
`jaxcont_max_extrema_error`, and `jaxcont_max_multiplier_error`. Do not call
`compare_case_result_to_reference` for this specialized producer.

- [ ] **Step 4: Write the torBPC gallery page**

Create `examples/example_20_matcont_torbpc_overlay.py`. Explain that the page
is intentionally diagnostic and that the automated validator currently fails;
do not describe the plot as agreement. Call:

```python
figure = render_periodic_overlay(
    "MC-LC-002",
    output_path=Path("images") / "matcont_torbpc_overlay.png",
    parameter_name=r"$\nu$",
    state_name=r"$x$ envelope",
    title="torBPC periodic branch: JaxCont and MatCont 7.6 diagnostic",
)
plt.show()
```

- [ ] **Step 5: Run torBPC tests and confirm the known failure remains visible**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -k torbpc -v
python -m examples.MatCont.run_validation --case MC-LC-002
```

Expected: visualization tests PASS; the validation CLI exits nonzero and
prints `FAIL MC-LC-002`, confirming the plot did not relax the comparison.

- [ ] **Step 6: Commit the torBPC diagnostic**

```bash
git add examples/MatCont/visualize.py examples/example_20_matcont_torbpc_overlay.py tests/test_matcont_visualization.py
git commit -m "feat: visualize known torBPC MatCont mismatches"
```

---

### Task 5: Document, Execute, and Visually Verify the Complete Gallery

**Files:**
- Modify: `examples/MatCont/README.md`
- Modify: `tests/test_matcont_visualization.py`
- Modify: `docs/source/conf.py` only if gallery discovery needs adjustment
- Generate (ignored): `images/matcont_cubic_overlay.png`
- Generate (ignored): `images/matcont_vanderpol_overlay.png`
- Generate (ignored): `images/matcont_adaptive_control_overlay.png`
- Generate (ignored): `images/matcont_radial_cycle_overlay.png`
- Generate (ignored): `images/matcont_torbpc_overlay.png`

**Interfaces:**
- Consumes: all five gallery scripts and their public render functions.
- Produces: documented commands and an end-to-end gallery execution contract.

- [ ] **Step 1: Generalize the subprocess smoke test across all five scripts**

Replace the single-script integration test with:

```python
def _gallery_environment(repository, tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(repository / "src"), str(repository)]
            ),
        }
    )
    return environment


@pytest.mark.parametrize(
    ("script_name", "image_name"),
    [
        ("example_16_matcont_cubic_overlay.py", "matcont_cubic_overlay.png"),
        ("example_17_matcont_vanderpol_overlay.py", "matcont_vanderpol_overlay.png"),
        ("example_18_matcont_adaptive_control_overlay.py", "matcont_adaptive_control_overlay.png"),
        ("example_19_matcont_radial_cycle_overlay.py", "matcont_radial_cycle_overlay.png"),
        ("example_20_matcont_torbpc_overlay.py", "matcont_torbpc_overlay.png"),
    ],
)
def test_gallery_example_runs_from_outside_repository_root(
    tmp_path, script_name, image_name
):
    repository = Path(__file__).resolve().parents[1]
    environment = _gallery_environment(repository, tmp_path)
    completed = subprocess.run(
        [sys.executable, str(repository / "examples" / script_name)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "images" / image_name).is_file()
```

- [ ] **Step 2: Run the gallery smoke tests before documentation changes**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py -k gallery_example -v
```

Expected: all five parameterized cases PASS.

- [ ] **Step 3: Update the MatCont README**

Document the five module commands, the generated filenames, the meaning of
each panel, native adaptive mesh behavior, and the distinction between visual
inspection and the authoritative numerical validator. Explicitly state that
`MC-LC-002` is a known failing diagnostic and link that status to the existing
supported-coverage warning instead of claiming complete correspondence.

- [ ] **Step 4: Run focused quality checks**

Run:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_matcont_visualization.py tests/test_matcont_suite.py -v
python -m flake8 examples/MatCont/visualize.py examples/example_16_matcont_cubic_overlay.py examples/example_17_matcont_vanderpol_overlay.py examples/example_18_matcont_adaptive_control_overlay.py examples/example_19_matcont_radial_cycle_overlay.py examples/example_20_matcont_torbpc_overlay.py tests/test_matcont_visualization.py
git diff --check
```

Expected: tests PASS, Flake8 has no findings, and `git diff --check` is silent.

- [ ] **Step 5: Generate and visually inspect all five PNGs**

Run from the feature worktree root:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m examples.example_16_matcont_cubic_overlay
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m examples.example_17_matcont_vanderpol_overlay
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m examples.example_18_matcont_adaptive_control_overlay
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m examples.example_19_matcont_radial_cycle_overlay
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m examples.example_20_matcont_torbpc_overlay
```

Inspect each image for readable axes and legends, coincident passing overlays,
correct event placement, and a prominent non-misleading torBPC failure banner.

- [ ] **Step 6: Run the full regression suite**

Run:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu python -m pytest -p no:cacheprovider
```

Expected: all tests PASS, apart from tests explicitly deselected by the
repository's default configuration.

- [ ] **Step 7: Commit the documentation and end-to-end tests**

```bash
git add examples/MatCont/README.md docs/source/conf.py tests/test_matcont_visualization.py
git commit -m "docs: publish MatCont visual comparison gallery"
```
