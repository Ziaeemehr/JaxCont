"""Visual comparisons between JaxCont results and reviewed MatCont artifacts."""

from __future__ import annotations

import csv
import importlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import compare_case_result_to_reference
from .python_cases import CaseResult
from .registry import load_registry

_DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_registered_case(
    case_id: str, reference_dir: Path
) -> tuple[dict, CaseResult]:
    """Run the JaxCont producer declared for one supported registry case."""
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


def _save_figure(figure, output_path: Path | str | None) -> None:
    """Save a figure when the caller requested a file output."""
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")


def plot_equilibrium_overlay(
    result: CaseResult,
    reference_dir: Path | str,
    *,
    state_index: int = 0,
    parameter_name: str = "Continuation parameter",
    state_name: str | None = None,
    title: str | None = None,
):
    """Overlay one JaxCont equilibrium branch on its reviewed MatCont branch."""
    reference_dir = Path(reference_dir)
    branch_rows = _read_csv(reference_dir / f"{result.case_id}_branch.csv")
    event_rows = _read_csv(reference_dir / f"{result.case_id}_events.csv")
    state_column = f"state_{state_index}"

    jax_parameters = np.asarray(result.artifacts["parameters"])
    jax_states = np.asarray(result.artifacts["states"])
    matcont_parameters = np.asarray([float(row["parameter"]) for row in branch_rows])
    matcont_states = np.asarray([float(row[state_column]) for row in branch_rows])
    branch_by_point = {row["point"]: row for row in branch_rows}

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        jax_parameters,
        jax_states[:, state_index],
        color="#2563eb",
        linewidth=2.4,
        label="JaxCont branch",
        zorder=3,
    )
    axis.plot(
        matcont_parameters,
        matcont_states,
        linestyle="none",
        marker="o",
        markersize=3.5,
        markerfacecolor="none",
        markeredgecolor="#f97316",
        alpha=0.75,
        label="MatCont 7.6 branch",
        zorder=2,
    )

    labeled_jaxcont_events: set[str] = set()
    for event in result.artifacts.get("events", []):
        kind = str(event["kind"])
        state = np.asarray(event["state"])
        label = f"JaxCont {kind}"
        axis.scatter(
            [float(event["parameter"])],
            [float(state[state_index])],
            marker="o",
            s=85,
            facecolor="#2563eb",
            edgecolor="white",
            linewidth=1.2,
            label=label if kind not in labeled_jaxcont_events else "_nolegend_",
            zorder=5,
        )
        labeled_jaxcont_events.add(kind)

    labeled_matcont_events: set[str] = set()
    for event in event_rows:
        kind = event["event_type"]
        event_state = event.get(state_column)
        if event_state is None:
            event_state = branch_by_point[event["point"]][state_column]
        label = f"MatCont 7.6 {kind}"
        axis.scatter(
            [float(event["parameter"])],
            [float(event_state)],
            marker="x",
            s=95,
            color="#c2410c",
            linewidth=2.2,
            label=label if kind not in labeled_matcont_events else "_nolegend_",
            zorder=6,
        )
        labeled_matcont_events.add(kind)

    axis.set_xlabel(parameter_name)
    axis.set_ylabel(state_name or f"state[{state_index}]")
    axis.set_title(title or f"{result.case_id}: JaxCont and MatCont branch overlay")
    shared_min = max(float(np.min(jax_parameters)), float(np.min(matcont_parameters)))
    shared_max = min(float(np.max(jax_parameters)), float(np.max(matcont_parameters)))
    axis.set_xlim(shared_min, shared_max)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def render_case_overlay(
    case_id: str,
    *,
    reference_dir: Path | str = _DEFAULT_REFERENCE_DIR,
    output_path: Path | str | None = None,
    parameter_name: str = "Continuation parameter",
    state_name: str | None = None,
    title: str | None = None,
):
    """Run a registered equilibrium case and render its MatCont overlay."""
    reference_dir = Path(reference_dir)
    case, result = _run_registered_case(case_id, reference_dir)
    if not case_id.startswith("MC-EQ-"):
        raise ValueError(
            f"visual comparison currently supports equilibrium cases, got {case_id}"
        )
    diagnostics = compare_case_result_to_reference(case, result, reference_dir)
    figure = plot_equilibrium_overlay(
        result,
        reference_dir,
        parameter_name=parameter_name,
        state_name=state_name,
        title=title,
    )
    figure.text(
        0.5,
        0.015,
        "Systematic comparison: PASS  •  "
        f"branch max error {diagnostics['branch_max_error']:.2e}  •  "
        f"event max error {diagnostics['event_max_error']:.2e}  •  "
        f"spectrum max error {diagnostics['spectrum_max_error']:.2e}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    figure.subplots_adjust(bottom=0.13)

    _save_figure(figure, output_path)
    return figure


__all__ = ["plot_equilibrium_overlay", "render_case_overlay"]
