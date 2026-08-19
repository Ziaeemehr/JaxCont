"""Behavioral tests for visual MatCont/JaxCont comparisons."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import matplotlib
import pytest

from examples.MatCont.python_cases import CaseResult

matplotlib.use("Agg")


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("case_id", ["MC-EQ-002", "MC-EQ-003"])
def test_registered_hopf_case_renders_spectral_pass_figure(case_id, tmp_path):
    """Hopf views compare stability crossings as well as flat equilibria."""
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


def test_registered_radial_cycle_renders_three_panel_pass_figure(tmp_path):
    """The periodic overlay compares amplitude, period, and Floquet stability."""
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
    assert {"JaxCont detected LPC", "JaxCont detected NS"} <= labels
    assert "JaxCont detected PD" not in labels


def test_equilibrium_overlay_draws_both_branches_and_event_sources(tmp_path):
    """Dropping either solver's curve or markers must make the visual comparison fail."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {"case_id": "MC-EQ-001", "point": 0, "parameter": -1.0, "stable": 1, "state_0": -2.0},
            {"case_id": "MC-EQ-001", "point": 1, "parameter": 0.5, "stable": 0, "state_0": -1.0},
            {"case_id": "MC-EQ-001", "point": 2, "parameter": -0.5, "stable": 1, "state_0": 1.0},
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "parameter", "state_0"],
        [
            {"case_id": "MC-EQ-001", "event_index": 0, "event_type": "LP", "parameter": 0.5, "state_0": -1.0},
            {"case_id": "MC-EQ-001", "event_index": 1, "event_type": "LP", "parameter": -0.5, "state_0": 1.0},
        ],
    )
    result = CaseResult(
        case_id="MC-EQ-001",
        checks={},
        observations={},
        artifacts={
            "parameters": jnp.array([-1.0, 0.5, -0.5]),
            "states": jnp.array([[-2.0], [-1.0], [1.0]]),
            "stability": jnp.array([True, False, True]),
            "events": [
                {"kind": "LP", "parameter": 0.5, "state": jnp.array([-1.0])},
                {"kind": "LP", "parameter": -0.5, "state": jnp.array([1.0])},
            ],
        },
    )

    figure = plot_equilibrium_overlay(result, tmp_path, state_index=0)
    axis = figure.axes[0]
    artists = {artist.get_label(): artist for artist in [*axis.lines, *axis.collections]}

    assert set(artists) >= {
        "JaxCont branch",
        "MatCont 7.6 branch",
        "JaxCont LP",
        "MatCont 7.6 LP",
    }
    assert list(artists["JaxCont branch"].get_xdata()) == [-1.0, 0.5, -0.5]
    assert list(artists["MatCont 7.6 branch"].get_xdata()) == [-1.0, 0.5, -0.5]
    assert axis.get_xlabel() == "Continuation parameter"
    assert axis.get_ylabel() == "state[0]"
    legend_labels = axis.get_legend_handles_labels()[1]
    assert legend_labels.count("JaxCont LP") == 1
    assert legend_labels.count("MatCont 7.6 LP") == 1


def test_equilibrium_overlay_resolves_matcont_event_state_from_branch_point(tmp_path):
    """MatCont event artifacts refer to branch rows; ignoring that link loses markers."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {"case_id": "MC-EQ-001", "point": 3, "parameter": 0.49, "stable": 1, "state_0": -1.1},
            {"case_id": "MC-EQ-001", "point": 4, "parameter": 0.5, "stable": 0, "state_0": -1.0},
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "point", "parameter"],
        [
            {"case_id": "MC-EQ-001", "event_index": 0, "event_type": "LP", "point": 4, "parameter": 0.5},
        ],
    )
    result = CaseResult(
        case_id="MC-EQ-001",
        checks={},
        observations={},
        artifacts={
            "parameters": jnp.array([0.49, 0.5]),
            "states": jnp.array([[-1.1], [-1.0]]),
            "events": [],
        },
    )

    figure = plot_equilibrium_overlay(result, tmp_path)
    marker = next(
        collection
        for collection in figure.axes[0].collections
        if collection.get_label() == "MatCont 7.6 LP"
    )

    assert marker.get_offsets().tolist() == [[0.5, -1.0]]


def test_render_case_overlay_reuses_registered_case_and_writes_png(tmp_path):
    """The user-facing renderer must run the registered case, not a frozen JaxCont copy."""
    from examples.MatCont.visualize import render_case_overlay

    reference_dir = Path(__file__).resolve().parents[1] / "examples" / "MatCont" / "reference"
    output_path = tmp_path / "mc-eq-001-overlay.png"

    figure = render_case_overlay(
        "MC-EQ-001",
        reference_dir=reference_dir,
        output_path=output_path,
    )

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.axes[0].get_title() == "MC-EQ-001: JaxCont and MatCont branch overlay"
    summary = " ".join(text.get_text() for text in figure.texts)
    assert "Systematic comparison: PASS" in summary
    assert "branch max error" in summary
    assert "event max error" in summary
    assert "spectrum max error" in summary


def test_equilibrium_overlay_limits_view_to_shared_parameter_domain(tmp_path):
    """A longer run from one package must not look like branch disagreement."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {"case_id": "MC-EQ-001", "point": 0, "parameter": -2.0, "stable": 1, "state_0": -2.0},
            {"case_id": "MC-EQ-001", "point": 1, "parameter": 2.0, "stable": 0, "state_0": 2.0},
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "point", "parameter"],
        [],
    )
    result = CaseResult(
        case_id="MC-EQ-001",
        checks={},
        observations={},
        artifacts={
            "parameters": jnp.array([-1.0, 1.0]),
            "states": jnp.array([[-1.0], [1.0]]),
            "events": [],
        },
    )

    figure = plot_equilibrium_overlay(result, tmp_path)

    assert figure.axes[0].get_xlim() == (-1.0, 1.0)


def test_equilibrium_overlay_accepts_domain_labels_and_title(tmp_path):
    """A gallery figure must describe the model rather than expose array indices."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {"case_id": "MC-EQ-001", "point": 0, "parameter": -1.0, "stable": 1, "state_0": -2.0},
            {"case_id": "MC-EQ-001", "point": 1, "parameter": 1.0, "stable": 0, "state_0": 2.0},
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "point", "parameter"],
        [],
    )
    result = CaseResult(
        case_id="MC-EQ-001",
        checks={},
        observations={},
        artifacts={
            "parameters": jnp.array([-1.0, 1.0]),
            "states": jnp.array([[-2.0], [2.0]]),
            "events": [],
        },
    )

    figure = plot_equilibrium_overlay(
        result,
        tmp_path,
        parameter_name=r"$r$",
        state_name=r"$x$",
        title="Cubic S-curve: JaxCont vs MatCont 7.6",
    )
    axis = figure.axes[0]

    assert axis.get_xlabel() == r"$r$"
    assert axis.get_ylabel() == r"$x$"
    assert axis.get_title() == "Cubic S-curve: JaxCont vs MatCont 7.6"


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
        (
            "example_18_matcont_adaptive_control_overlay.py",
            "matcont_adaptive_control_overlay.png",
        ),
        (
            "example_19_matcont_radial_cycle_overlay.py",
            "matcont_radial_cycle_overlay.png",
        ),
        ("example_20_matcont_torbpc_overlay.py", "matcont_torbpc_overlay.png"),
    ],
)
def test_gallery_example_runs_from_outside_repository_root(
    tmp_path, script_name, image_name
):
    """Sphinx-Gallery execution must not depend on the repository root in sys.path."""
    repository = Path(__file__).resolve().parents[1]
    obsolete_script = repository / "examples" / "example_16_matcont_overlay.py"
    assert not obsolete_script.exists()
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
