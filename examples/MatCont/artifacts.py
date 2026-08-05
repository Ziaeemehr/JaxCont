"""Schema validation for normalized MatCont reference artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def _read_rows(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or ())
            missing = set(required) - columns
            if missing:
                raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
            return list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing MatCont reference artifact: {path}") from exc


def _finite_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric {column!r} value") from exc
        if np.isfinite(value):
            values.append(value)
    return values


def validate_equilibrium_artifacts(reference_dir: Path, case_id: str) -> dict:
    """Validate equilibrium branch/event/spectrum files and expose diagnostics."""
    reference_dir = Path(reference_dir)
    branch = _read_rows(
        reference_dir / f"{case_id}_branch.csv",
        {
            "case_id",
            "point",
            "parameter",
            "residual_norm",
            "stable",
            "unstable_dimension",
        },
    )
    events = _read_rows(
        reference_dir / f"{case_id}_events.csv",
        {
            "case_id",
            "event_index",
            "event_type",
            "point",
            "parameter",
            "frequency",
            "fold_coefficient",
            "first_lyapunov",
        },
    )
    spectra = _read_rows(
        reference_dir / f"{case_id}_multipliers.csv",
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
    )
    if not branch:
        raise ValueError(f"{case_id} equilibrium branch is empty")
    if not spectra:
        raise ValueError(f"{case_id} equilibrium spectrum is empty")
    if any(row["spectrum_kind"].strip().upper() != "EIGENVALUE" for row in spectra):
        raise ValueError(f"{case_id} equilibrium spectrum contains non-eigenvalue rows")
    if any(row["case_id"] != case_id for row in branch + events + spectra):
        raise ValueError(f"{case_id} artifacts contain a different case_id")
    state_columns = sorted(column for column in branch[0] if column.startswith("state_"))
    branch_spectra = [row for row in spectra if int(row["event_index"]) == -1]
    if len(branch_spectra) != len(branch) * len(state_columns):
        raise ValueError(f"{case_id} branch does not have one eigenvalue per state and point")
    for row in branch:
        stable = int(row["stable"])
        unstable_dimension = int(row["unstable_dimension"])
        if stable not in (0, 1) or unstable_dimension < 0:
            raise ValueError(f"{case_id} branch has invalid stability diagnostics")
        if stable != int(unstable_dimension == 0):
            raise ValueError(f"{case_id} stable flag disagrees with unstable_dimension")

    hopf_events = [row for row in events if row["event_type"].strip() == "H"]
    fold_events = [row for row in events if row["event_type"].strip() == "LP"]
    frequencies = _finite_values(hopf_events, "frequency")
    lyapunov = _finite_values(hopf_events, "first_lyapunov")
    fold_coefficients = _finite_values(fold_events, "fold_coefficient")
    if hopf_events and (len(frequencies) != len(hopf_events) or len(lyapunov) != len(hopf_events)):
        raise ValueError(f"{case_id} Hopf diagnostics are incomplete")
    if fold_events and len(fold_coefficients) != len(fold_events):
        raise ValueError(f"{case_id} fold coefficients are incomplete")
    return {
        "branch_rows": len(branch),
        "event_rows": len(events),
        "spectrum_rows": len(spectra),
        "fold_coefficients": fold_coefficients,
        "hopf_frequency": frequencies[0] if frequencies else None,
        "first_lyapunov": lyapunov[0] if lyapunov else None,
    }


__all__ = ["validate_equilibrium_artifacts"]
