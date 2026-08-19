"""Behavioral tests for visual MatCont/JaxCont comparisons."""

from __future__ import annotations

import csv
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

from examples.MatCont.python_cases import CaseResult

matplotlib.use("Agg")

_REPOSITORY = Path(__file__).resolve().parents[1]
_REFERENCE_DIR = _REPOSITORY / "examples" / "MatCont" / "reference"


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("case_id", "hopf_parameter"),
    [("MC-EQ-002", 0.0), ("MC-EQ-003", 1.0)],
)
def test_registered_hopf_case_renders_spectral_pass_figure(
    case_id, hopf_parameter, tmp_path
):
    """Hopf views compare stability crossings as well as flat equilibria."""
    from examples.MatCont.visualize import render_equilibrium_overlay

    output = tmp_path / f"{case_id}.png"
    figure = render_equilibrium_overlay(case_id, output_path=output)

    assert len(figure.axes) == 2
    labels = {
        artist.get_label()
        for artist in [*figure.axes[1].lines, *figure.axes[1].collections]
    }
    assert {
        "JaxCont spectral abscissa",
        "MatCont 7.6 spectral abscissa",
    } <= labels
    assert figure.axes[1].get_ylabel() == "Largest Re(eigenvalue)"
    assert any("H" in label for label in labels)
    for axis in figure.axes:
        artists = {artist.get_label(): artist for artist in axis.collections}
        np.testing.assert_allclose(
            np.asarray(artists["JaxCont H"].get_offsets()[0], dtype=float),
            [hopf_parameter, 0.0],
            atol=5e-4,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(
                artists["MatCont 7.6 H"].get_offsets()[0],
                dtype=float,
            ),
            [hopf_parameter, 0.0],
            atol=5e-4,
            rtol=0.0,
        )
    spectral_lines = {line.get_label(): line for line in figure.axes[1].lines}
    for solver in (
        "JaxCont spectral abscissa",
        "MatCont 7.6 spectral abscissa",
    ):
        x_data = np.asarray(spectral_lines[solver].get_xdata())
        y_data = np.asarray(spectral_lines[solver].get_ydata())
        assert x_data.shape == y_data.shape
        assert np.min(y_data) < 0.0 < np.max(y_data)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "Systematic comparison: PASS" in " ".join(
        text.get_text() for text in figure.texts
    )


def test_registered_radial_cycle_renders_three_panel_pass_figure(tmp_path):
    """The periodic overlay compares all three radial diagnostics."""
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
    assert "critical-signature proxy parameter discrepancies" in summary
    assert "missing correctly located event types LPC, PD" in summary
    assert "event errors" not in summary
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

    parameter_labels = {
        label
        for axis in figure.axes[:2]
        for label in axis.get_legend_handles_labels()[1]
    }
    spectrum_labels = set(figure.axes[2].get_legend_handles_labels()[1])
    assert not any(
        label.startswith("JaxCont near") for label in parameter_labels
    )
    assert not any(
        label.startswith("JaxCont detected") for label in spectrum_labels
    )
    assert {
        "JaxCont near LPC",
        "JaxCont near NS",
        "JaxCont near PD",
    } <= spectrum_labels

    amplitude_markers = {
        collection.get_label(): collection
        for collection in figure.axes[0].collections
    }
    matcont_lpc = amplitude_markers["MatCont 7.6 LPC"].get_offsets()[0, 0]
    detected_lpc = amplitude_markers["JaxCont detected LPC"].get_offsets()[
        0, 0
    ]
    assert abs(float(detected_lpc) - float(matcont_lpc)) > 2e-3


@pytest.fixture(scope="module")
def cubic_registered_case():
    from examples.MatCont.python_cases.equilibrium import run_cubic_fold
    from examples.MatCont.registry import load_registry

    case = next(
        case for case in load_registry()["cases"] if case["id"] == "MC-EQ-001"
    )
    return case, run_cubic_fold()


def test_equilibrium_renderer_reports_failed_producer_predicate(
    monkeypatch, cubic_registered_case
):
    """A MatCont match cannot override a failed producer self-check."""
    from examples.MatCont import visualize

    case, result = cubic_registered_case
    failed_result = replace(
        result,
        checks={**result.checks, "max_residual": 1.0},
    )
    monkeypatch.setattr(
        visualize,
        "_run_registered_case",
        lambda *_arguments: (case, failed_result),
    )

    figure = visualize.render_equilibrium_overlay("MC-EQ-001")
    summary = next(
        text
        for text in figure.texts
        if "Systematic comparison" in text.get_text()
    )

    assert "Systematic comparison: FAIL" in summary.get_text()
    assert "producer checks failed" in summary.get_text()
    assert summary.get_color() == "#991b1b"
    assert summary.get_weight() == "bold"


def test_equilibrium_renderer_turns_numerical_mismatch_into_diagnostic(
    monkeypatch, cubic_registered_case
):
    """A numerical mismatch must remain drawable for diagnosis."""
    from examples.MatCont import visualize

    case, result = cubic_registered_case
    shifted_states = np.asarray(result.artifacts["states"]).copy()
    shifted_states[:, 0] += 0.1
    mismatched_result = replace(
        result,
        artifacts={**result.artifacts, "states": shifted_states},
    )
    monkeypatch.setattr(
        visualize,
        "_run_registered_case",
        lambda *_arguments: (case, mismatched_result),
    )

    figure = visualize.render_equilibrium_overlay("MC-EQ-001")
    summary = next(
        text
        for text in figure.texts
        if "Systematic comparison" in text.get_text()
    )

    assert "Systematic comparison: FAIL" in summary.get_text()
    assert "numerical mismatch:" in summary.get_text()
    assert summary.get_color() == "#991b1b"


def test_equilibrium_renderer_keeps_missing_artifacts_exceptional(
    monkeypatch, tmp_path, cubic_registered_case
):
    """Diagnostic FAIL handling must not hide a missing structural artifact."""
    from examples.MatCont import visualize

    case, result = cubic_registered_case
    monkeypatch.setattr(
        visualize,
        "_run_registered_case",
        lambda *_arguments: (case, result),
    )

    with pytest.raises(FileNotFoundError, match="missing .*artifact"):
        visualize.render_equilibrium_overlay(
            "MC-EQ-001",
            reference_dir=tmp_path,
        )


def test_equilibrium_renderer_rejects_family_before_running_producer(
    monkeypatch,
):
    """An invalid family must not execute an unrelated registered producer."""
    from examples.MatCont import visualize

    producer_called = False

    def unexpected_producer(*_arguments):
        nonlocal producer_called
        producer_called = True
        raise AssertionError("producer executed")

    monkeypatch.setattr(visualize, "_run_registered_case", unexpected_producer)

    with pytest.raises(ValueError, match="supports equilibrium cases"):
        visualize.render_equilibrium_overlay("MC-LC-001")

    assert not producer_called


@pytest.mark.parametrize(
    ("missing_source", "message"),
    [
        ("jaxcont", "missing JaxCont H event spectrum"),
        ("matcont", "missing MatCont 7.6 H event spectrum"),
    ],
)
def test_hopf_overlay_rejects_missing_event_spectrum(
    tmp_path, missing_source, message
):
    """A missing Hopf spectrum must not become a fabricated zero marker."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-002_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {
                "case_id": "MC-EQ-002",
                "point": 0,
                "parameter": -1.0,
                "stable": 1,
                "state_0": 0.0,
            },
            {
                "case_id": "MC-EQ-002",
                "point": 1,
                "parameter": 1.0,
                "stable": 0,
                "state_0": 0.0,
            },
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-002_events.csv",
        [
            "case_id",
            "event_index",
            "event_type",
            "point",
            "parameter",
            "state_0",
        ],
        [
            {
                "case_id": "MC-EQ-002",
                "event_index": 0,
                "event_type": "H",
                "point": 1,
                "parameter": 0.0,
                "state_0": 0.0,
            }
        ],
    )
    spectrum_rows = [
        {
            "case_id": "MC-EQ-002",
            "point": point,
            "event_index": -1,
            "event_type": "BRANCH",
            "real": real,
            "imag": imag,
        }
        for point, real in [(0, -0.5), (1, 0.5)]
        for imag in (-1.0, 1.0)
    ]
    if missing_source != "matcont":
        spectrum_rows.extend(
            {
                "case_id": "MC-EQ-002",
                "point": 1,
                "event_index": 0,
                "event_type": "H",
                "real": 0.0,
                "imag": imag,
            }
            for imag in (-1.0, 1.0)
        )
    _write_rows(
        tmp_path / "MC-EQ-002_multipliers.csv",
        [
            "case_id",
            "point",
            "event_index",
            "event_type",
            "real",
            "imag",
        ],
        spectrum_rows,
    )
    event_spectra = []
    if missing_source != "jaxcont":
        event_spectra = [
            {
                "kind": "H",
                "parameter": 0.0,
                "values": np.asarray([1j, -1j]),
            }
        ]
    result = CaseResult(
        case_id="MC-EQ-002",
        checks={},
        observations={},
        artifacts={
            "parameters": np.asarray([-1.0, 1.0]),
            "states": np.zeros((2, 1)),
            "spectra": np.asarray(
                [[-0.5 + 1j, -0.5 - 1j], [0.5 + 1j, 0.5 - 1j]]
            ),
            "events": [{"kind": "H", "parameter": 0.0, "state": np.zeros(1)}],
            "event_spectra": event_spectra,
        },
    )

    with pytest.raises(ValueError, match=message):
        plot_equilibrium_overlay(
            result,
            tmp_path,
            include_spectrum=True,
        )


def test_radial_overlay_filters_native_samples_before_y_autoscaling(tmp_path):
    """Out-of-domain sentinels must not flatten any radial comparison panel."""
    from examples.MatCont.visualize import plot_radial_cycle_overlay

    branch_rows = [
        {
            "case_id": "MC-LC-001",
            "point": point,
            "parameter": parameter,
            "period": period,
            "state_0_min": minimum,
            "state_0_max": maximum,
        }
        for point, parameter, period, minimum, maximum in [
            (10, 1.1, 6.1, -1.1, 1.1),
            (20, 1.3, 6.3, -1.3, 1.3),
            (30, 1.5, 6.5, -1.5, 1.5),
        ]
    ]
    _write_rows(
        tmp_path / "MC-LC-001_branch.csv",
        [
            "case_id",
            "point",
            "parameter",
            "period",
            "state_0_min",
            "state_0_max",
        ],
        branch_rows,
    )
    multiplier_rows = []
    for point, nontrivial in [(10, 0.21), (20, 0.25), (30, 0.31)]:
        for index, value in enumerate((1.0, nontrivial)):
            multiplier_rows.append(
                {
                    "case_id": "MC-LC-001",
                    "point": point,
                    "event_index": -1,
                    "event_type": "BRANCH",
                    "multiplier_index": index,
                    "real": value,
                    "imag": 0.0,
                }
            )
    _write_rows(
        tmp_path / "MC-LC-001_multipliers.csv",
        [
            "case_id",
            "point",
            "event_index",
            "event_type",
            "multiplier_index",
            "real",
            "imag",
        ],
        multiplier_rows,
    )
    result = CaseResult(
        case_id="MC-LC-001",
        checks={},
        observations={},
        artifacts={
            "parameters": np.asarray([0.5, 1.2, 1.4, 2.0]),
            "periods": np.asarray([1e6, 6.2, 6.4, -1e6]),
            "state_min": np.asarray([[-1e6], [-1.2], [-1.4], [-1e6]]),
            "state_max": np.asarray([[1e6], [1.2], [1.4], [1e6]]),
            "multipliers": np.asarray(
                [
                    [1.0, 1e6],
                    [1.0, 0.2],
                    [1.0, 0.3],
                    [1.0, -1e6],
                ],
                dtype=complex,
            ),
        },
    )

    figure = plot_radial_cycle_overlay(result, tmp_path)
    amplitude_lines = {line.get_label(): line for line in figure.axes[0].lines}
    period_lines = {line.get_label(): line for line in figure.axes[1].lines}
    multiplier_lines = {
        line.get_label(): line for line in figure.axes[2].lines
    }

    assert amplitude_lines["JaxCont minimum"].get_xdata() == pytest.approx(
        [1.2, 1.4]
    )
    assert amplitude_lines["JaxCont minimum"].get_ydata() == pytest.approx(
        [-1.2, -1.4]
    )
    assert amplitude_lines["MatCont 7.6 maximum"].get_xdata() == pytest.approx(
        [1.1, 1.3, 1.5]
    )
    assert amplitude_lines["MatCont 7.6 maximum"].get_ydata() == pytest.approx(
        [1.1, 1.3, 1.5]
    )
    assert period_lines["JaxCont period"].get_ydata() == pytest.approx(
        [6.2, 6.4]
    )
    assert period_lines["MatCont 7.6 period"].get_ydata() == pytest.approx(
        [6.1, 6.3, 6.5]
    )
    assert multiplier_lines[
        "JaxCont nontrivial multiplier"
    ].get_ydata() == pytest.approx([0.2, 0.3])
    assert multiplier_lines[
        "MatCont 7.6 nontrivial multiplier"
    ].get_ydata() == pytest.approx([0.21, 0.25, 0.31])
    for axis in figure.axes:
        assert axis.get_xlim() == pytest.approx((1.1, 1.5))
        assert max(abs(bound) for bound in axis.get_ylim()) < 10.0


def test_equilibrium_overlay_draws_both_branches_and_event_sources(tmp_path):
    """The overlay must retain both solver curves and event markers."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {
                "case_id": "MC-EQ-001",
                "point": 0,
                "parameter": -1.0,
                "stable": 1,
                "state_0": -2.0,
            },
            {
                "case_id": "MC-EQ-001",
                "point": 1,
                "parameter": 0.5,
                "stable": 0,
                "state_0": -1.0,
            },
            {
                "case_id": "MC-EQ-001",
                "point": 2,
                "parameter": -0.5,
                "stable": 1,
                "state_0": 1.0,
            },
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "parameter", "state_0"],
        [
            {
                "case_id": "MC-EQ-001",
                "event_index": 0,
                "event_type": "LP",
                "parameter": 0.5,
                "state_0": -1.0,
            },
            {
                "case_id": "MC-EQ-001",
                "event_index": 1,
                "event_type": "LP",
                "parameter": -0.5,
                "state_0": 1.0,
            },
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
    artists = {
        artist.get_label(): artist
        for artist in [*axis.lines, *axis.collections]
    }

    assert set(artists) >= {
        "JaxCont branch",
        "MatCont 7.6 branch",
        "JaxCont LP",
        "MatCont 7.6 LP",
    }
    assert list(artists["JaxCont branch"].get_xdata()) == [-1.0, 0.5, -0.5]
    assert list(artists["MatCont 7.6 branch"].get_xdata()) == [
        -1.0,
        0.5,
        -0.5,
    ]
    assert axis.get_xlabel() == "Continuation parameter"
    assert axis.get_ylabel() == "state[0]"
    legend_labels = axis.get_legend_handles_labels()[1]
    assert legend_labels.count("JaxCont LP") == 1
    assert legend_labels.count("MatCont 7.6 LP") == 1


def test_equilibrium_overlay_resolves_matcont_event_state_from_branch_point(
    tmp_path,
):
    """MatCont event artifacts resolve their state from branch rows."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {
                "case_id": "MC-EQ-001",
                "point": 3,
                "parameter": 0.49,
                "stable": 1,
                "state_0": -1.1,
            },
            {
                "case_id": "MC-EQ-001",
                "point": 4,
                "parameter": 0.5,
                "stable": 0,
                "state_0": -1.0,
            },
        ],
    )
    _write_rows(
        tmp_path / "MC-EQ-001_events.csv",
        ["case_id", "event_index", "event_type", "point", "parameter"],
        [
            {
                "case_id": "MC-EQ-001",
                "event_index": 0,
                "event_type": "LP",
                "point": 4,
                "parameter": 0.5,
            },
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
    """The renderer must execute the registered JaxCont case."""
    from examples.MatCont.visualize import render_case_overlay

    reference_dir = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "MatCont"
        / "reference"
    )
    output_path = tmp_path / "mc-eq-001-overlay.png"

    figure = render_case_overlay(
        "MC-EQ-001",
        reference_dir=reference_dir,
        output_path=output_path,
    )

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        figure.axes[0].get_title()
        == "MC-EQ-001: JaxCont and MatCont branch overlay"
    )
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
            {
                "case_id": "MC-EQ-001",
                "point": 0,
                "parameter": -2.0,
                "stable": 1,
                "state_0": -2.0,
            },
            {
                "case_id": "MC-EQ-001",
                "point": 1,
                "parameter": 2.0,
                "stable": 0,
                "state_0": 2.0,
            },
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
    """A gallery figure must use model-specific labels."""
    from examples.MatCont.visualize import plot_equilibrium_overlay

    _write_rows(
        tmp_path / "MC-EQ-001_branch.csv",
        ["case_id", "point", "parameter", "stable", "state_0"],
        [
            {
                "case_id": "MC-EQ-001",
                "point": 0,
                "parameter": -1.0,
                "stable": 1,
                "state_0": -2.0,
            },
            {
                "case_id": "MC-EQ-001",
                "point": 1,
                "parameter": 1.0,
                "stable": 0,
                "state_0": 2.0,
            },
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
        (
            "example_17_matcont_vanderpol_overlay.py",
            "matcont_vanderpol_overlay.png",
        ),
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
    """Sphinx-Gallery execution must work outside the repository root."""
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
