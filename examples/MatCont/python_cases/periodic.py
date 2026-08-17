"""Periodic-orbit validation cases with analytic and MatCont references."""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

from ..compare import ValidationMismatch, match_spectrum
from . import CaseResult

_TORBPC_EVENT_PARAMETERS = {
    "LPC": -0.5844928424,
    "NS": -0.5957504315,
    "PD": -0.6146816596,
}

_TORBPC_FIXED = {
    "beta": 0.5,
    "gamma": -0.6,
    "r": 0.6,
    "a3": 0.32858,
    "b3": 0.93358,
    "epsilon": 0.001,
}


def _radial_rhs(u, rho, args):
    del args
    x, y = u
    radius_squared = x * x + y * y
    return jnp.array([(rho - radius_squared) * x - y, (rho - radius_squared) * y + x])


def _radial_problem(rho: float, mesh: Collocation):
    period = 2.0 * np.pi
    time = np.linspace(0.0, period, 80, endpoint=False)
    radius = np.sqrt(rho)
    trajectory = np.stack([radius * np.cos(time), radius * np.sin(time)], axis=1)
    return periodic_orbit_problem(
        _radial_rhs,
        jnp.asarray(trajectory),
        jnp.asarray(time),
        period,
        rho,
        mesh,
    )


def run_radial_cycle() -> CaseResult:
    """Validate collocation and Floquet stability for an exact radial cycle."""
    mesh = Collocation(ntst=10, ncol=4)
    problem = _radial_problem(0.5, mesh)
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(0.5, 1.5),
        settings=jc.ContinuationPar(
            compute_stability=True,
            ds=0.05,
            ds_max=0.08,
            max_steps=35,
            newton_tol=1e-5,
        ),
    )
    branch = result.branch
    states = branch.states
    parameters = branch.params
    dimension = 2
    mesh_count = mesh.ntst * dimension
    collocation_count = mesh.ntst * mesh.ncol * dimension
    mesh_states = states[:, :mesh_count].reshape((-1, mesh.ntst, dimension))
    collocation_states = states[:, mesh_count : mesh_count + collocation_count].reshape(
        (-1, mesh.ntst, mesh.ncol, dimension)
    )
    periods = states[:, -1]
    all_orbit_states = jnp.concatenate(
        [
            mesh_states.reshape((states.shape[0], -1, dimension)),
            collocation_states.reshape((states.shape[0], -1, dimension)),
        ],
        axis=1,
    )
    radii = jnp.linalg.norm(all_orbit_states, axis=-1)
    radius_errors = jnp.abs(radii - jnp.sqrt(parameters)[:, None])
    residuals = jax.vmap(lambda unknown, parameter: problem.f(unknown, parameter, problem.args))(
        states, parameters
    )
    expected_nontrivial = jnp.exp(-4.0 * jnp.pi * parameters)
    multipliers = branch.eigenvalues
    trivial_indices = jnp.argmin(jnp.abs(multipliers - 1.0), axis=1)
    nontrivial_indices = 1 - trivial_indices
    nontrivial = multipliers[jnp.arange(multipliers.shape[0]), nontrivial_indices]
    multiplier_errors = jnp.abs(nontrivial - expected_nontrivial)
    return CaseResult(
        case_id="MC-LC-001",
        checks={
            "max_radius_error": float(jnp.max(radius_errors)),
            "max_period_error": float(jnp.max(jnp.abs(periods - 2.0 * jnp.pi))),
            "max_collocation_residual": float(jnp.max(jnp.abs(residuals))),
            "max_multiplier_error": float(jnp.max(multiplier_errors)),
            "all_stable": bool(jnp.all(branch.stable)),
        },
        observations={
            "n_points": branch.n_valid,
            "mesh": {"ntst": mesh.ntst, "ncol": mesh.ncol},
            "parameter_range": (float(jnp.min(parameters)), float(jnp.max(parameters))),
        },
        artifacts={
            "states": states,
            "parameters": parameters,
            "periods": periods,
            "multipliers": multipliers,
            "stability": branch.stable,
            "state_min": jnp.min(all_orbit_states, axis=1),
            "state_max": jnp.max(all_orbit_states, axis=1),
            "events": [],
        },
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing MatCont reference artifact: {path}") from exc


def _critical_multiplier_matches(event_type: str, values: np.ndarray) -> bool:
    if values.size < 2:
        return False
    trivial = int(np.argmin(np.abs(values - 1.0)))
    critical = np.delete(values, trivial)
    if event_type == "LPC":
        return bool(np.min(np.abs(critical - 1.0)) < 5e-3)
    if event_type == "PD":
        return bool(np.min(np.abs(critical + 1.0)) < 5e-3)
    if event_type == "NS":
        candidates = critical[np.abs(np.imag(critical)) > 1e-4]
        if candidates.size < 2:
            return False
        conjugacy_error = np.min(
            np.abs(candidates[:, None] - np.conjugate(candidates)[None, :])
            + np.eye(candidates.size) * 10.0
        )
        return bool(np.max(np.abs(np.abs(candidates) - 1.0)) < 5e-3 and conjugacy_error < 5e-3)
    return False


def _torbpc_rhs(u, nu, args):
    del args
    beta = _TORBPC_FIXED["beta"]
    gamma = _TORBPC_FIXED["gamma"]
    scale = _TORBPC_FIXED["r"]
    a3 = _TORBPC_FIXED["a3"]
    b3 = _TORBPC_FIXED["b3"]
    epsilon = _TORBPC_FIXED["epsilon"]
    x, y, z = u
    difference = y - x
    return jnp.array(
        [
            (-(beta + nu) * x + beta * y - a3 * x**3 + b3 * difference**3) / scale,
            beta * x - (beta + gamma) * y - z - b3 * difference**3,
            y + epsilon,
        ]
    )


def _torbpc_hopf_seed(nu: float) -> tuple[np.ndarray, float]:
    """Return the small equilibrium and Hopf frequency used by testtorBPC1."""
    beta = _TORBPC_FIXED["beta"]
    gamma = _TORBPC_FIXED["gamma"]
    scale = _TORBPC_FIXED["r"]
    a3 = _TORBPC_FIXED["a3"]
    b3 = _TORBPC_FIXED["b3"]
    y = -_TORBPC_FIXED["epsilon"]
    roots = np.roots(
        [
            -(a3 + b3),
            3.0 * b3 * y,
            -(beta + nu) - 3.0 * b3 * y**2,
            beta * y + b3 * y**3,
        ]
    )
    real_roots = roots[np.abs(np.imag(roots)) < 1e-10].real
    x = float(real_roots[np.argmin(np.abs(real_roots - 0.0056))])
    difference = y - x
    z = beta * x - (beta + gamma) * y - b3 * difference**3
    jacobian = np.array(
        [
            [
                (-(beta + nu) - 3.0 * a3 * x**2 - 3.0 * b3 * difference**2) / scale,
                (beta + 3.0 * b3 * difference**2) / scale,
                0.0,
            ],
            [
                beta + 3.0 * b3 * difference**2,
                -(beta + gamma) - 3.0 * b3 * difference**2,
                -1.0,
            ],
            [0.0, 1.0, 0.0],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    index = int(np.argmax(np.imag(eigenvalues)))
    vector = eigenvectors[:, index] / np.linalg.norm(eigenvectors[:, index])
    omega = float(np.imag(eigenvalues[index]))
    period = 2.0 * np.pi / omega
    time = np.linspace(0.0, period, 100, endpoint=False)
    trajectory = np.array([x, y, z])[None, :] + 0.002 * np.real(
        vector[None, :] * np.exp(1j * omega * time[:, None])
    )
    return trajectory, period


def _run_torbpc_jaxcont():
    start_parameter = -0.589286291695516
    trajectory, period = _torbpc_hopf_seed(start_parameter)
    time = np.linspace(0.0, period, trajectory.shape[0], endpoint=False)
    mesh = Collocation(ntst=10, ncol=4)
    problem = periodic_orbit_problem(
        _torbpc_rhs,
        jnp.asarray(trajectory),
        jnp.asarray(time),
        period,
        start_parameter,
        mesh,
    )
    result = jc.continuation(
        problem,
        jc.PseudoArclength(),
        p_span=(start_parameter, -0.75),
        settings=jc.ContinuationPar(
            compute_stability=True,
            ds=0.005,
            ds_min=1e-5,
            ds_max=0.02,
            max_steps=300,
            newton_tol=1e-5,
        ),
        events=[
            jc.Fold(),
            jc.NeimarkSacker(raw_f=_torbpc_rhs, mesh=mesh),
            jc.PeriodDoubling(raw_f=_torbpc_rhs, mesh=mesh),
        ],
    )
    return result, mesh


def _require_columns(rows: list[dict[str, str]], required: set[str], artifact: str) -> None:
    if not rows:
        raise ValueError(f"{artifact} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{artifact} is missing columns: {sorted(missing)}")


def load_torbpc_reference(reference_dir: Path) -> dict:
    """Load and validate normalized MatCont torBPC branch/event/spectrum data."""
    reference_dir = Path(reference_dir)
    branch = _read_csv(reference_dir / "MC-LC-002_branch.csv")
    events = _read_csv(reference_dir / "MC-LC-002_events.csv")
    multiplier_rows = _read_csv(reference_dir / "MC-LC-002_multipliers.csv")
    _require_columns(
        branch,
        {
            "case_id",
            "point",
            "parameter",
            "period",
            "residual_norm",
            "state_0_min",
            "state_0_max",
            "state_1_min",
            "state_1_max",
            "state_2_min",
            "state_2_max",
        },
        "MC-LC-002_branch.csv",
    )
    _require_columns(
        events,
        {"case_id", "event_index", "event_type", "point", "parameter", "period"},
        "MC-LC-002_events.csv",
    )
    _require_columns(
        multiplier_rows,
        {
            "case_id",
            "point",
            "event_index",
            "event_type",
            "spectrum_kind",
            "multiplier_index",
            "real",
            "imag",
        },
        "MC-LC-002_multipliers.csv",
    )
    branch_by_point = {int(row["point"]): row for row in branch}
    events_by_type: dict[str, dict[str, str]] = {}
    spectra: dict[str, np.ndarray] = {}
    extrema: dict[str, np.ndarray] = {}
    event_errors = []
    for event in events:
        event_type = event["event_type"].strip()
        if event_type not in _TORBPC_EVENT_PARAMETERS:
            continue
        if event_type in events_by_type:
            raise ValueError(f"duplicate torBPC event type: {event_type}")
        events_by_type[event_type] = event
        event_errors.append(
            abs(float(event["parameter"]) - _TORBPC_EVENT_PARAMETERS[event_type])
        )
        event_index = int(event["event_index"])
        values = np.asarray(
            [
                complex(float(row["real"]), float(row["imag"]))
                for row in multiplier_rows
                if int(row["event_index"]) == event_index
                and row["event_type"].strip() == event_type
            ],
            dtype=complex,
        )
        spectra[event_type] = values
        point = int(event["point"])
        try:
            branch_row = branch_by_point[point]
        except KeyError as exc:
            raise ValueError(f"event {event_type} refers to missing branch point {point}") from exc
        extrema[event_type] = np.asarray(
            [
                (float(branch_row[f"state_{dimension}_min"]),
                 float(branch_row[f"state_{dimension}_max"]))
                for dimension in range(3)
            ]
        )
    if set(events_by_type) != set(_TORBPC_EVENT_PARAMETERS):
        raise ValueError("torBPC reference must contain exactly one LPC, NS, and PD event")
    if any(
        row["spectrum_kind"].strip().upper() != "FLOQUET"
        for row in multiplier_rows
    ):
        raise ValueError("torBPC spectrum contains non-Floquet rows")
    periods = np.asarray([float(row["period"]) for row in branch + events])
    extrema_ordered = all(np.all(value[:, 0] <= value[:, 1]) for value in extrema.values())
    critical_matches = {
        kind: _critical_multiplier_matches(kind, spectra[kind])
        for kind in _TORBPC_EVENT_PARAMETERS
    }
    return {
        "branch": branch,
        "events": events,
        "multiplier_rows": multiplier_rows,
        "events_by_type": events_by_type,
        "spectra": spectra,
        "extrema": extrema,
        "max_event_parameter_error": max(event_errors),
        "all_periods_finite_positive": bool(
            np.all(np.isfinite(periods)) and np.all(periods > 0)
        ),
        "all_extrema_ordered": extrema_ordered,
        "critical_multipliers_match": all(critical_matches.values()),
    }


def run_torbpc_cycle(reference_dir: Path) -> CaseResult:
    """Compare JaxCont's torBPC branch with normalized MatCont artifacts."""
    reference = load_torbpc_reference(reference_dir)
    branch = reference["branch"]
    events = reference["events"]
    multiplier_rows = reference["multiplier_rows"]
    jaxcont_result, mesh = _run_torbpc_jaxcont()
    jaxcont_branch = jaxcont_result.branch
    jaxcont_parameters = np.asarray(jaxcont_branch.params)
    jaxcont_states = np.asarray(jaxcont_branch.states)
    jaxcont_spectra = np.asarray(jaxcont_branch.eigenvalues)
    dimension = 3
    orbit_count = mesh.ntst * (mesh.ncol + 1)
    orbit_states = jaxcont_states[:, :-1].reshape((-1, orbit_count, dimension))
    observed_indices = {
        "LPC": int(np.argmax(jaxcont_parameters)),
        "NS": int(
            np.nanargmin(
                np.where(
                    np.abs(np.imag(jaxcont_spectra)) > 1e-3,
                    np.abs(np.abs(jaxcont_spectra) - 1.0),
                    np.nan,
                )
            )
            // jaxcont_spectra.shape[1]
        ),
        "PD": int(
            np.nanargmin(
                np.where(
                    np.abs(np.imag(jaxcont_spectra)) < 1e-3,
                    np.abs(np.real(jaxcont_spectra) + 1.0),
                    np.nan,
                )
            )
            // jaxcont_spectra.shape[1]
        ),
    }
    parameter_errors = {
        kind: abs(float(jaxcont_parameters[index]) - expected)
        for kind, expected in _TORBPC_EVENT_PARAMETERS.items()
        for index in [observed_indices[kind]]
    }
    reference_events = {row["event_type"].strip(): row for row in events}
    period_errors = {
        kind: abs(float(jaxcont_states[index, -1]) - float(reference_events[kind]["period"]))
        for kind, index in observed_indices.items()
    }
    extrema_errors = {}
    spectrum_errors = {}
    spectrum_matches = {}
    for kind, index in observed_indices.items():
        actual_extrema = np.stack(
            [np.min(orbit_states[index], axis=0), np.max(orbit_states[index], axis=0)],
            axis=1,
        )
        extrema_errors[kind] = float(
            np.max(np.abs(actual_extrema - reference["extrema"][kind]))
        )
        try:
            diagnostics = match_spectrum(
                jaxcont_spectra[index], reference["spectra"][kind], atol=5e-3
            )
            spectrum_matches[kind] = True
        except ValidationMismatch:
            diagnostics = match_spectrum(
                jaxcont_spectra[index], reference["spectra"][kind], atol=1e6
            )
            spectrum_matches[kind] = False
        spectrum_errors[kind] = diagnostics["max_error"]
    matched_event_types = set()
    for hit in jaxcont_result.events:
        kind = {
            "fold": "LPC",
            "neimark_sacker": "NS",
            "period_doubling": "PD",
        }.get(hit.kind)
        if kind and abs(hit.p - _TORBPC_EVENT_PARAMETERS[kind]) < 2e-3:
            matched_event_types.add(kind)
    missing_event_types = sorted(set(_TORBPC_EVENT_PARAMETERS) - matched_event_types)
    all_comparisons_pass = bool(
        max(parameter_errors.values()) < 2e-3
        and max(period_errors.values()) < 5e-3
        and max(extrema_errors.values()) < 5e-3
        and all(spectrum_matches.values())
        and not missing_event_types
    )
    return CaseResult(
        case_id="MC-LC-002",
        checks={
            "event_count": len(reference["events_by_type"]),
            "max_event_parameter_error": reference["max_event_parameter_error"],
            "all_periods_finite_positive": reference["all_periods_finite_positive"],
            "all_extrema_ordered": reference["all_extrema_ordered"],
            "critical_multipliers_match": reference["critical_multipliers_match"],
            "jaxcont_sweep_completed": jaxcont_branch.n_valid > 1,
            "jaxcont_event_parameter_errors": parameter_errors,
            "jaxcont_lpc_parameter_error": parameter_errors["LPC"],
            "jaxcont_ns_parameter_error": parameter_errors["NS"],
            "jaxcont_pd_parameter_error": parameter_errors["PD"],
            "jaxcont_max_period_error": max(period_errors.values()),
            "jaxcont_max_extrema_error": max(extrema_errors.values()),
            "jaxcont_max_multiplier_error": max(spectrum_errors.values()),
            "jaxcont_critical_multipliers_match": all(spectrum_matches.values()),
            "jaxcont_missing_event_types": missing_event_types,
            "all_comparisons_pass": all_comparisons_pass,
        },
        observations={
            "events": events,
            "branch_points": len(branch),
            "jaxcont_events": [
                {"kind": hit.kind, "parameter": hit.p} for hit in jaxcont_result.events
            ],
            "jaxcont_period_errors": period_errors,
            "jaxcont_extrema_errors": extrema_errors,
            "jaxcont_spectrum_errors": spectrum_errors,
        },
        artifacts={
            "branch": branch,
            "events": events,
            "multipliers": multiplier_rows,
            "jaxcont_states": jaxcont_branch.states,
            "jaxcont_parameters": jaxcont_branch.params,
            "jaxcont_multipliers": jaxcont_branch.eigenvalues,
            "jaxcont_orbits": orbit_states,
        },
    )


__all__ = ["load_torbpc_reference", "run_radial_cycle", "run_torbpc_cycle"]
