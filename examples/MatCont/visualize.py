"""Visual comparisons between JaxCont and reviewed MatCont artifacts."""

from __future__ import annotations

import csv
import importlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import (
    validate_equilibrium_artifacts,
    validate_periodic_artifacts,
)
from .python_cases import CaseResult
from .registry import load_registry
from .validation import CaseValidation, evaluate_case_result

_DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_TORBPC_EVENT_COLORS = {
    "LPC": "#16a34a",
    "NS": "#7c3aed",
    "PD": "#dc2626",
}
_TORBPC_JAXCONT_EVENT_TYPES = {
    "fold": "LPC",
    "neimark_sacker": "NS",
    "period_doubling": "PD",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_registered_case(
    case_id: str, reference_dir: Path
) -> tuple[dict, CaseResult]:
    """Run the JaxCont producer declared for one supported registry case."""
    registry = load_registry()
    try:
        case = next(
            item for item in registry["cases"] if item["id"] == case_id
        )
    except StopIteration as exc:
        raise ValueError(
            f"unknown MatCont validation case: {case_id}"
        ) from exc
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


def _branch_spectral_abscissa(rows: list[dict[str, str]]) -> dict[int, float]:
    """Return the largest real eigenvalue at each non-event MatCont point."""
    values_by_point: dict[int, list[float]] = {}
    for row in rows:
        if int(row["event_index"]) != -1:
            continue
        values_by_point.setdefault(int(row["point"]), []).append(
            float(row["real"])
        )
    return {
        point: max(real_parts) for point, real_parts in values_by_point.items()
    }


def _event_spectral_abscissa(
    event: dict,
    event_spectra: list[dict],
    *,
    source: str,
) -> float:
    """Find an event's spectral abscissa from its registered event spectrum."""
    for event_spectrum in event_spectra:
        if event_spectrum.get("kind") == event.get("kind") and np.isclose(
            float(event_spectrum["parameter"]), float(event["parameter"])
        ):
            return float(np.max(np.real(np.asarray(event_spectrum["values"]))))
    raise ValueError(
        f"missing {source} {event.get('kind')} event spectrum at parameter "
        f"{float(event['parameter']):.12g}"
    )


def _matcont_event_spectral_abscissa(
    event: dict[str, str],
    spectrum_rows: list[dict[str, str]],
) -> float:
    values = [
        float(row["real"])
        for row in spectrum_rows
        if int(row["event_index"]) == int(event["event_index"])
    ]
    if not values:
        raise ValueError(
            "missing MatCont 7.6 "
            f"{event['event_type']} event spectrum at parameter "
            f"{float(event['parameter']):.12g}"
        )
    return max(values)


def _validation_failure_details(validation: CaseValidation) -> list[str]:
    details = []
    if not validation.producer_passed:
        details.append("producer checks failed")
    if validation.numerical_mismatch is not None:
        details.append(f"numerical mismatch: {validation.numerical_mismatch}")
    return details


def _annotate_summary(
    figure,
    *,
    status: str,
    passed: bool,
    details: list[str],
    fontsize: float = 9.0,
) -> None:
    text = figure.text(
        0.5,
        0.015,
        f"Systematic comparison: {status}\n" + "  •  ".join(details),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#334155" if passed else "#991b1b",
        fontweight="normal" if passed else "bold",
        bbox=(
            None
            if passed
            else {
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fee2e2",
                "edgecolor": "#dc2626",
                "alpha": 0.95,
            }
        ),
    )
    if passed:
        text.set_bbox(None)
    figure.subplots_adjust(bottom=0.16)


def _nontrivial_floquet_multipliers(values: np.ndarray) -> np.ndarray:
    """Remove one multiplier nearest the trivial value ``+1`` per row."""
    values = np.asarray(values, dtype=complex)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(
            "each periodic spectrum must include a trivial and "
            "nontrivial multiplier"
        )
    trivial_indices = np.argmin(np.abs(values - 1.0), axis=1)
    keep = np.ones(values.shape, dtype=bool)
    keep[np.arange(values.shape[0]), trivial_indices] = False
    return values[keep].reshape(values.shape[0], values.shape[1] - 1)


def plot_radial_cycle_overlay(
    result: CaseResult,
    reference_dir: Path | str,
    *,
    parameter_name: str = "Continuation parameter",
    title: str | None = None,
):
    """Overlay radial extrema, periods, and nontrivial multipliers."""
    reference_dir = Path(reference_dir)
    branch_rows = _read_csv(reference_dir / f"{result.case_id}_branch.csv")
    multiplier_rows = _read_csv(
        reference_dir / f"{result.case_id}_multipliers.csv"
    )

    jax_parameters = np.asarray(result.artifacts["parameters"])
    jax_periods = np.asarray(result.artifacts["periods"])
    jax_state_min = np.asarray(result.artifacts["state_min"])[:, 0]
    jax_state_max = np.asarray(result.artifacts["state_max"])[:, 0]
    jax_multipliers = _nontrivial_floquet_multipliers(
        np.asarray(result.artifacts["multipliers"])
    )

    matcont_parameters = np.asarray(
        [float(row["parameter"]) for row in branch_rows]
    )
    matcont_periods = np.asarray([float(row["period"]) for row in branch_rows])
    matcont_state_min = np.asarray(
        [float(row["state_0_min"]) for row in branch_rows]
    )
    matcont_state_max = np.asarray(
        [float(row["state_0_max"]) for row in branch_rows]
    )
    multipliers_by_point: dict[int, list[complex]] = {}
    for row in multiplier_rows:
        if int(row["event_index"]) == -1:
            multipliers_by_point.setdefault(int(row["point"]), []).append(
                complex(float(row["real"]), float(row["imag"]))
            )
    matcont_multipliers = np.asarray(
        [multipliers_by_point[int(row["point"])] for row in branch_rows]
    )
    matcont_nontrivial = _nontrivial_floquet_multipliers(matcont_multipliers)

    shared_min = max(
        float(np.min(jax_parameters)),
        float(np.min(matcont_parameters)),
    )
    shared_max = min(
        float(np.max(jax_parameters)),
        float(np.max(matcont_parameters)),
    )
    if shared_min > shared_max:
        raise ValueError(
            "JaxCont and MatCont branches have no shared parameter domain"
        )
    jax_in_domain = (jax_parameters >= shared_min) & (
        jax_parameters <= shared_max
    )
    matcont_in_domain = (matcont_parameters >= shared_min) & (
        matcont_parameters <= shared_max
    )

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 10))
    amplitude_axis, period_axis, multiplier_axis = axes
    amplitude_axis.plot(
        jax_parameters[jax_in_domain],
        jax_state_min[jax_in_domain],
        color="#2563eb",
        linewidth=2.4,
        label="JaxCont minimum",
    )
    amplitude_axis.plot(
        jax_parameters[jax_in_domain],
        jax_state_max[jax_in_domain],
        color="#2563eb",
        linewidth=2.4,
        linestyle="--",
        label="JaxCont maximum",
    )
    amplitude_axis.plot(
        matcont_parameters[matcont_in_domain],
        matcont_state_min[matcont_in_domain],
        color="#f97316",
        linewidth=1.4,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 minimum",
    )
    amplitude_axis.plot(
        matcont_parameters[matcont_in_domain],
        matcont_state_max[matcont_in_domain],
        color="#f97316",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 maximum",
    )
    amplitude_axis.set_ylabel("Orbit amplitude")
    amplitude_axis.set_title(
        title or f"{result.case_id}: JaxCont and MatCont periodic overlay"
    )

    period_axis.plot(
        jax_parameters[jax_in_domain],
        jax_periods[jax_in_domain],
        color="#2563eb",
        linewidth=2.4,
        label="JaxCont period",
    )
    period_axis.plot(
        matcont_parameters[matcont_in_domain],
        matcont_periods[matcont_in_domain],
        color="#f97316",
        linewidth=1.4,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 period",
    )
    period_axis.set_ylabel("Period")

    # Auto-scaling alone can zoom in on a numerically negligible gap (e.g.
    # both curves agreeing with the true period to ~1e-9, well inside the
    # tolerance already reported in the footer) and make two curves that
    # agree to 9+ significant figures look like they disagree outright.
    # Floor the y-range so a near-exact match reads as visually overlapping.
    period_values = np.concatenate(
        [jax_periods[jax_in_domain], matcont_periods[matcont_in_domain]]
    )
    period_center = float(np.mean(period_values))
    period_half_range = max(
        float(np.max(period_values) - np.min(period_values)) / 2.0,
        abs(period_center) * 0.005,
        1e-6,
    )
    period_axis.set_ylim(
        period_center - period_half_range, period_center + period_half_range
    )

    multiplier_axis.plot(
        jax_parameters[jax_in_domain],
        np.abs(jax_multipliers[jax_in_domain, 0]),
        color="#2563eb",
        linewidth=2.4,
        label="JaxCont nontrivial multiplier",
    )
    multiplier_axis.plot(
        matcont_parameters[matcont_in_domain],
        np.abs(matcont_nontrivial[matcont_in_domain, 0]),
        color="#f97316",
        linewidth=1.4,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 nontrivial multiplier",
    )
    multiplier_axis.set_xlabel(parameter_name)
    multiplier_axis.set_ylabel("Nontrivial |Floquet multiplier|")

    for axis in axes:
        axis.set_xlim(shared_min, shared_max)
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    return figure


def plot_torbpc_overlay(
    result: CaseResult,
    reference_dir: Path | str,
    *,
    state_index: int = 0,
    parameter_name: str = r"$\nu$",
    state_name: str | None = None,
    title: str | None = None,
):
    """Plot torBPC envelopes, periods, and event-centered Floquet spectra."""
    branch_rows = result.artifacts["branch"]
    event_rows = result.artifacts["events"]
    multiplier_rows = result.artifacts["multipliers"]

    jax_parameters = np.asarray(result.artifacts["jaxcont_parameters"])
    jax_states = np.asarray(result.artifacts["jaxcont_states"])
    jax_periods = jax_states[:, -1]
    jax_orbits = np.asarray(result.artifacts["jaxcont_orbits"])
    jax_state_min = np.min(jax_orbits[:, :, state_index], axis=1)
    jax_state_max = np.max(jax_orbits[:, :, state_index], axis=1)
    jax_multipliers = np.asarray(result.artifacts["jaxcont_multipliers"])

    matcont_parameters = np.asarray(
        [float(row["parameter"]) for row in branch_rows]
    )
    matcont_periods = np.asarray([float(row["period"]) for row in branch_rows])
    matcont_state_min = np.asarray(
        [float(row[f"state_{state_index}_min"]) for row in branch_rows]
    )
    matcont_state_max = np.asarray(
        [float(row[f"state_{state_index}_max"]) for row in branch_rows]
    )
    branch_by_point = {int(row["point"]): row for row in branch_rows}

    figure = plt.figure(figsize=(13, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.4, 1.0))
    amplitude_axis = figure.add_subplot(grid[0, 0])
    period_axis = figure.add_subplot(grid[1, 0], sharex=amplitude_axis)
    spectrum_axis = figure.add_subplot(grid[:, 1])

    amplitude_axis.plot(
        jax_parameters,
        jax_state_min,
        color="#2563eb",
        linewidth=2.2,
        label="JaxCont minimum",
    )
    amplitude_axis.plot(
        jax_parameters,
        jax_state_max,
        color="#2563eb",
        linewidth=2.2,
        linestyle="--",
        label="JaxCont maximum",
    )
    amplitude_axis.plot(
        matcont_parameters,
        matcont_state_min,
        color="#f97316",
        linewidth=1.3,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 minimum",
    )
    amplitude_axis.plot(
        matcont_parameters,
        matcont_state_max,
        color="#f97316",
        linewidth=1.3,
        linestyle="--",
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 maximum",
    )
    amplitude_axis.set_ylabel(state_name or f"state[{state_index}] envelope")
    amplitude_axis.set_title(
        title or f"{result.case_id}: JaxCont and MatCont 7.6 diagnostic"
    )

    period_axis.plot(
        jax_parameters,
        jax_periods,
        color="#2563eb",
        linewidth=2.2,
        label="JaxCont period",
    )
    period_axis.plot(
        matcont_parameters,
        matcont_periods,
        color="#f97316",
        linewidth=1.3,
        marker="o",
        markersize=2.8,
        markerfacecolor="none",
        label="MatCont 7.6 period",
    )
    period_axis.set_xlabel(parameter_name)
    period_axis.set_ylabel("Period")

    for event in event_rows:
        event_type = event["event_type"].strip()
        if event_type not in _TORBPC_EVENT_COLORS:
            continue
        color = _TORBPC_EVENT_COLORS[event_type]
        event_parameter = float(event["parameter"])
        branch_row = branch_by_point[int(event["point"])]
        amplitude_axis.scatter(
            [event_parameter],
            [float(branch_row[f"state_{state_index}_max"])],
            marker="x",
            s=85,
            color=color,
            linewidth=2.0,
            label=f"MatCont 7.6 {event_type}",
            zorder=6,
        )
        period_axis.scatter(
            [event_parameter],
            [float(event["period"])],
            marker="x",
            s=85,
            color=color,
            linewidth=2.0,
            label=f"MatCont 7.6 {event_type}",
            zorder=6,
        )

        nearest_index = int(
            np.argmin(np.abs(jax_parameters - event_parameter))
        )
        event_index = int(event["event_index"])
        matcont_spectrum = np.asarray(
            [
                complex(float(row["real"]), float(row["imag"]))
                for row in multiplier_rows
                if int(row["event_index"]) == event_index
                and row["event_type"].strip() == event_type
            ]
        )
        spectrum_axis.scatter(
            np.real(matcont_spectrum),
            np.imag(matcont_spectrum),
            marker="x",
            s=75,
            color=color,
            linewidth=2.0,
            label=f"MatCont 7.6 {event_type}",
            zorder=6,
        )
        spectrum_axis.scatter(
            np.real(jax_multipliers[nearest_index]),
            np.imag(jax_multipliers[nearest_index]),
            marker="o",
            s=55,
            facecolor="none",
            edgecolor=color,
            linewidth=1.8,
            label=f"JaxCont near {event_type}",
            zorder=5,
        )

    detected_event_labels: set[str] = set()
    for event in result.observations.get("jaxcont_events", []):
        event_type = _TORBPC_JAXCONT_EVENT_TYPES.get(str(event["kind"]))
        if event_type is None:
            continue
        parameter = float(event["parameter"])
        nearest_index = int(np.argmin(np.abs(jax_parameters - parameter)))
        label = f"JaxCont detected {event_type}"
        legend_label = (
            label if event_type not in detected_event_labels else "_nolegend_"
        )
        amplitude_axis.scatter(
            [parameter],
            [jax_state_max[nearest_index]],
            marker="D",
            s=55,
            facecolor=_TORBPC_EVENT_COLORS[event_type],
            edgecolor="white",
            linewidth=0.9,
            label=legend_label,
            zorder=7,
        )
        period_axis.scatter(
            [parameter],
            [jax_periods[nearest_index]],
            marker="D",
            s=55,
            facecolor=_TORBPC_EVENT_COLORS[event_type],
            edgecolor="white",
            linewidth=0.9,
            label=legend_label,
            zorder=7,
        )
        detected_event_labels.add(event_type)

    shared_min = max(
        float(np.min(jax_parameters)), float(np.min(matcont_parameters))
    )
    shared_max = min(
        float(np.max(jax_parameters)), float(np.max(matcont_parameters))
    )
    if shared_min > shared_max:
        raise ValueError(
            "JaxCont and MatCont branches have no shared parameter domain"
        )
    for axis in (amplitude_axis, period_axis):
        axis.set_xlim(shared_min, shared_max)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncols=2)

    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    spectrum_axis.plot(
        np.cos(angle),
        np.sin(angle),
        color="#94a3b8",
        linestyle="--",
        label="Unit circle",
    )
    spectrum_axis.axhline(0.0, color="#cbd5e1", linewidth=0.8, zorder=1)
    spectrum_axis.axvline(0.0, color="#cbd5e1", linewidth=0.8, zorder=1)
    spectrum_axis.set_xlabel("Re(Floquet multiplier)")
    spectrum_axis.set_ylabel("Im(Floquet multiplier)")
    spectrum_axis.set_aspect("equal", adjustable="box")
    spectrum_axis.grid(alpha=0.2)
    spectrum_axis.legend(fontsize=8)

    figure.tight_layout()
    return figure


def plot_equilibrium_overlay(
    result: CaseResult,
    reference_dir: Path | str,
    *,
    state_index: int = 0,
    parameter_name: str = "Continuation parameter",
    state_name: str | None = None,
    title: str | None = None,
    include_spectrum: bool = False,
):
    """Overlay one JaxCont branch on its reviewed MatCont branch."""
    reference_dir = Path(reference_dir)
    branch_rows = _read_csv(reference_dir / f"{result.case_id}_branch.csv")
    event_rows = _read_csv(reference_dir / f"{result.case_id}_events.csv")
    state_column = f"state_{state_index}"

    jax_parameters = np.asarray(result.artifacts["parameters"])
    jax_states = np.asarray(result.artifacts["states"])
    matcont_parameters = np.asarray(
        [float(row["parameter"]) for row in branch_rows]
    )
    matcont_states = np.asarray(
        [float(row[state_column]) for row in branch_rows]
    )
    branch_by_point = {row["point"]: row for row in branch_rows}

    if include_spectrum:
        figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 8))
        axis, spectral_axis = axes
    else:
        figure, axis = plt.subplots(figsize=(9, 6))
        spectral_axis = None
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
            label=(
                label if kind not in labeled_jaxcont_events else "_nolegend_"
            ),
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
            label=(
                label if kind not in labeled_matcont_events else "_nolegend_"
            ),
            zorder=6,
        )
        labeled_matcont_events.add(kind)

    axis.set_ylabel(state_name or f"state[{state_index}]")
    axis.set_title(
        title or f"{result.case_id}: JaxCont and MatCont branch overlay"
    )
    shared_min = max(
        float(np.min(jax_parameters)), float(np.min(matcont_parameters))
    )
    shared_max = min(
        float(np.max(jax_parameters)), float(np.max(matcont_parameters))
    )
    if shared_min > shared_max:
        raise ValueError(
            "JaxCont and MatCont branches have no shared parameter domain"
        )
    axis.set_xlim(shared_min, shared_max)
    axis.grid(alpha=0.2)
    axis.legend()

    if spectral_axis is not None:
        spectrum_rows = _read_csv(
            reference_dir / f"{result.case_id}_multipliers.csv"
        )
        matcont_spectral_abscissa = _branch_spectral_abscissa(spectrum_rows)
        matcont_spectrum_points = [
            (
                float(row["parameter"]),
                matcont_spectral_abscissa[int(row["point"])],
            )
            for row in branch_rows
            if int(row["point"]) in matcont_spectral_abscissa
        ]
        jax_spectra = np.asarray(result.artifacts["spectra"])
        spectral_axis.plot(
            jax_parameters,
            np.max(np.real(jax_spectra), axis=1),
            color="#2563eb",
            linewidth=2.4,
            label="JaxCont spectral abscissa",
            zorder=3,
        )
        spectral_axis.plot(
            [point[0] for point in matcont_spectrum_points],
            [point[1] for point in matcont_spectrum_points],
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="none",
            markeredgecolor="#f97316",
            alpha=0.75,
            label="MatCont 7.6 spectral abscissa",
            zorder=2,
        )
        spectral_axis.axhline(0.0, color="#64748b", linewidth=1.0, zorder=1)

        jax_event_spectra = result.artifacts.get("event_spectra", [])
        labeled_jaxcont_spectral_events: set[str] = set()
        for event in result.artifacts.get("events", []):
            kind = str(event["kind"])
            spectral_axis.scatter(
                [float(event["parameter"])],
                [
                    _event_spectral_abscissa(
                        event,
                        jax_event_spectra,
                        source="JaxCont",
                    )
                ],
                marker="o",
                s=85,
                facecolor="#2563eb",
                edgecolor="white",
                linewidth=1.2,
                label=(
                    f"JaxCont {kind}"
                    if kind not in labeled_jaxcont_spectral_events
                    else "_nolegend_"
                ),
                zorder=5,
            )
            labeled_jaxcont_spectral_events.add(kind)

        labeled_matcont_spectral_events: set[str] = set()
        for event in event_rows:
            kind = event["event_type"]
            spectral_axis.scatter(
                [float(event["parameter"])],
                [_matcont_event_spectral_abscissa(event, spectrum_rows)],
                marker="x",
                s=95,
                color="#c2410c",
                linewidth=2.2,
                label=(
                    f"MatCont 7.6 {kind}"
                    if kind not in labeled_matcont_spectral_events
                    else "_nolegend_"
                ),
                zorder=6,
            )
            labeled_matcont_spectral_events.add(kind)

        spectral_axis.set_xlabel(parameter_name)
        spectral_axis.set_ylabel("Largest Re(eigenvalue)")
        spectral_axis.set_xlim(shared_min, shared_max)
        spectral_axis.grid(alpha=0.2)
        spectral_axis.legend()
    else:
        axis.set_xlabel(parameter_name)

    figure.tight_layout()
    return figure


def render_equilibrium_overlay(
    case_id: str,
    *,
    reference_dir: Path | str = _DEFAULT_REFERENCE_DIR,
    output_path: Path | str | None = None,
    parameter_name: str = "Continuation parameter",
    state_name: str | None = None,
    title: str | None = None,
    include_spectrum: bool | None = None,
):
    """Run a registered equilibrium case and render its MatCont overlay."""
    if not case_id.startswith("MC-EQ-"):
        raise ValueError(
            "visual comparison currently supports equilibrium cases, "
            f"got {case_id}"
        )
    reference_dir = Path(reference_dir)
    case, result = _run_registered_case(case_id, reference_dir)
    validate_equilibrium_artifacts(reference_dir, case_id)
    validation = evaluate_case_result(case, result, reference_dir)
    if include_spectrum is None:
        include_spectrum = "hopf" in case["features"]
    figure = plot_equilibrium_overlay(
        result,
        reference_dir,
        parameter_name=parameter_name,
        state_name=state_name,
        title=title,
        include_spectrum=include_spectrum,
    )
    details = _validation_failure_details(validation)
    if validation.diagnostics is not None:
        details.extend(
            [
                "branch max error "
                f"{validation.diagnostics['branch_max_error']:.2e}",
                "event max error "
                f"{validation.diagnostics['event_max_error']:.2e}",
                "spectrum max error "
                f"{validation.diagnostics['spectrum_max_error']:.2e}",
            ]
        )
    _annotate_summary(
        figure,
        status="PASS" if validation.passed else "FAIL",
        passed=validation.passed,
        details=details,
    )

    _save_figure(figure, output_path)
    return figure


def render_periodic_overlay(
    case_id: str,
    *,
    reference_dir: Path | str = _DEFAULT_REFERENCE_DIR,
    output_path: Path | str | None = None,
    state_index: int = 0,
    parameter_name: str = "Continuation parameter",
    state_name: str | None = None,
    title: str | None = None,
):
    """Run a registered periodic case and render its MatCont overlay."""
    if case_id not in {"MC-LC-001", "MC-LC-002"}:
        raise ValueError(
            "visual comparison currently supports MC-LC-001 and MC-LC-002, "
            f"got {case_id}"
        )
    reference_dir = Path(reference_dir)
    case, result = _run_registered_case(case_id, reference_dir)
    validate_periodic_artifacts(reference_dir, case_id)
    validation = evaluate_case_result(case, result, reference_dir)
    if case_id == "MC-LC-002":
        figure = plot_torbpc_overlay(
            result,
            reference_dir,
            state_index=state_index,
            parameter_name=parameter_name,
            state_name=state_name,
            title=title,
        )
        checks = validation.checks
        missing_event_types = ", ".join(checks["jaxcont_missing_event_types"])
        _annotate_summary(
            figure,
            status=(
                "PASS" if validation.passed else "FAIL (known limitation)"
            ),
            passed=validation.passed,
            details=[
                "critical-signature proxy parameter discrepancies "
                f"LPC {checks['jaxcont_lpc_parameter_error']:.2e}, "
                f"NS {checks['jaxcont_ns_parameter_error']:.2e}, "
                f"PD {checks['jaxcont_pd_parameter_error']:.2e}",
                "missing correctly located event types "
                f"{missing_event_types or 'none'}",
                f"period {checks['jaxcont_max_period_error']:.2e}",
                f"extrema {checks['jaxcont_max_extrema_error']:.2e}",
                f"multiplier {checks['jaxcont_max_multiplier_error']:.2e}",
            ],
            fontsize=8.5,
        )
        _save_figure(figure, output_path)
        return figure

    figure = plot_radial_cycle_overlay(
        result,
        reference_dir,
        parameter_name=parameter_name,
        title=title,
    )
    details = _validation_failure_details(validation)
    if validation.diagnostics is not None:
        details.extend(
            [
                "branch max error "
                f"{validation.diagnostics['branch_max_error']:.2e}",
                "spectrum max error "
                f"{validation.diagnostics['spectrum_max_error']:.2e}",
            ]
        )
    details.extend(
        [
            f"period max error {result.checks['max_period_error']:.2e}",
            f"radius max error {result.checks['max_radius_error']:.2e}",
        ]
    )
    _annotate_summary(
        figure,
        status="PASS" if validation.passed else "FAIL",
        passed=validation.passed,
        details=details,
    )

    _save_figure(figure, output_path)
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
    """Render the original one-panel equilibrium overlay compatibility view."""
    return render_equilibrium_overlay(
        case_id,
        reference_dir=reference_dir,
        output_path=output_path,
        parameter_name=parameter_name,
        state_name=state_name,
        title=title,
        include_spectrum=False,
    )


__all__ = [
    "plot_equilibrium_overlay",
    "plot_radial_cycle_overlay",
    "plot_torbpc_overlay",
    "render_case_overlay",
    "render_equilibrium_overlay",
    "render_periodic_overlay",
]
