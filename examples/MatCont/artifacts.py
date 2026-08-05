"""Schema validation for normalized MatCont reference artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import jax

from .compare import ValidationMismatch


_REQUIRED_METADATA_FIELDS = {
    "case_id",
    "provenance",
    "source",
    "source_section",
    "matlab_version",
    "matcont_version",
    "jaxcont_version",
    "python_version",
    "jax_version",
    "precision",
    "mesh",
    "solver_settings",
    "generated_utc",
    "equation_hash",
    "reviewed",
}
_VOLATILE_METADATA_FIELDS = {"generated_utc", "reviewed", "provenance"}


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


def validate_reference_metadata(path: Path, case_id: str) -> dict[str, Any]:
    """Load a reviewed metadata file and enforce its reproducibility contract."""
    try:
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing MatCont reference artifact: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{Path(path).name} must contain a JSON object")
    missing = _REQUIRED_METADATA_FIELDS - set(metadata)
    if missing:
        raise ValueError(
            f"{Path(path).name} is missing metadata fields: {', '.join(sorted(missing))}"
        )
    if metadata["case_id"] != case_id:
        raise ValueError(f"{Path(path).name} contains a different case_id")
    if metadata["reviewed"] is not True:
        raise ValueError(f"{Path(path).name} is not marked as a reviewed reference")
    equation_hash = metadata["equation_hash"]
    if not isinstance(equation_hash, str) or not equation_hash.startswith("sha256:"):
        raise ValueError(f"{Path(path).name} has an invalid equation_hash")
    return metadata


def enrich_generated_metadata(
    metadata_path: Path, case: dict[str, Any], equation_path: Path
) -> dict[str, Any]:
    """Add cross-runtime provenance to a raw MATLAB producer metadata file."""
    metadata_path = Path(metadata_path)
    equation_path = Path(equation_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    equation_bytes = equation_path.read_bytes()
    try:
        jaxcont_version = importlib.metadata.version("jaxcont")
    except importlib.metadata.PackageNotFoundError:
        jaxcont_version = "source-tree"
    metadata.update(
        {
            "provenance": "regenerated with the committed standalone MATLAB producer",
            "source_section": case["manual_source"],
            "jaxcont_version": jaxcont_version,
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "mesh": metadata.get("mesh", {"kind": "equilibrium"}),
            "equation_hash": "sha256:" + hashlib.sha256(equation_bytes).hexdigest(),
            "reviewed": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _column_atol(column: str, tolerances: dict[str, float]) -> float:
    if column == "parameter":
        return float(tolerances.get("parameter_atol", 1e-9))
    if column == "period":
        return float(tolerances.get("period_atol", 1e-9))
    if column == "residual_norm":
        return float(tolerances.get("residual_atol", 1e-9))
    if column in {"real", "imag"}:
        return float(tolerances.get("multiplier_atol", 1e-9))
    if column.startswith("state_"):
        return float(tolerances.get("extrema_atol", tolerances.get("state_atol", 1e-9)))
    return 0.0 if column in {"point", "event_index", "multiplier_index"} else 1e-9


def _verify_csv(
    generated_path: Path,
    reviewed_path: Path,
    tolerances: dict[str, float],
) -> float:
    with generated_path.open(newline="", encoding="utf-8") as stream:
        generated_reader = csv.DictReader(stream)
        generated_fields = generated_reader.fieldnames
        generated_rows = list(generated_reader)
    with reviewed_path.open(newline="", encoding="utf-8") as stream:
        reviewed_reader = csv.DictReader(stream)
        reviewed_fields = reviewed_reader.fieldnames
        reviewed_rows = list(reviewed_reader)
    if generated_fields != reviewed_fields:
        raise ValidationMismatch(
            f"{reviewed_path.name}: generated and reviewed columns differ"
        )
    if len(generated_rows) != len(reviewed_rows):
        raise ValidationMismatch(
            f"{reviewed_path.name}: generated {len(generated_rows)} rows, "
            f"reviewed {len(reviewed_rows)}"
        )

    max_error = 0.0
    for row_index, (generated, reviewed) in enumerate(zip(generated_rows, reviewed_rows)):
        for column in generated_fields or ():
            generated_value = generated[column]
            reviewed_value = reviewed[column]
            try:
                generated_number = float(generated_value)
                reviewed_number = float(reviewed_value)
            except ValueError:
                if generated_value != reviewed_value:
                    raise ValidationMismatch(
                        f"{reviewed_path.name}:{row_index + 2} {column} differs"
                    )
                continue
            if np.isnan(generated_number) and np.isnan(reviewed_number):
                continue
            if not (np.isfinite(generated_number) and np.isfinite(reviewed_number)):
                if generated_number != reviewed_number:
                    raise ValidationMismatch(
                        f"{reviewed_path.name}:{row_index + 2} {column} differs"
                    )
                continue
            error = abs(generated_number - reviewed_number)
            max_error = max(max_error, error)
            atol = _column_atol(column, tolerances)
            rtol = float(tolerances.get("rtol", 1e-9))
            threshold = atol + rtol * max(abs(generated_number), abs(reviewed_number))
            if error > threshold:
                raise ValidationMismatch(
                    f"{reviewed_path.name}:{row_index + 2} {column} error "
                    f"{error:.6g} exceeds {threshold:.6g}"
                )
    return max_error


def _verify_metadata(generated_path: Path, reviewed_path: Path, case_id: str) -> None:
    reviewed = validate_reference_metadata(reviewed_path, case_id)
    try:
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing generated MatCont artifact: {generated_path}") from exc
    if not isinstance(generated, dict):
        raise ValidationMismatch(f"{generated_path.name} must contain a JSON object")
    stable_fields = set(reviewed) - _VOLATILE_METADATA_FIELDS
    missing = stable_fields - set(generated)
    if missing:
        raise ValidationMismatch(
            f"{generated_path.name} is missing metadata fields: {', '.join(sorted(missing))}"
        )
    for field in sorted(stable_fields):
        if generated[field] != reviewed[field]:
            raise ValidationMismatch(f"{reviewed_path.name}: metadata field {field} differs")


def verify_case_references(
    case: dict[str, Any], generated_dir: Path, reviewed_dir: Path
) -> dict[str, Any]:
    """Compare regenerated text artifacts with reviewed references without writing either."""
    generated_dir = Path(generated_dir)
    reviewed_dir = Path(reviewed_dir)
    references = case.get("references", [])
    if not references:
        return {"artifact_count": 0, "max_numeric_error": 0.0}
    max_numeric_error = 0.0
    for filename in references:
        generated_path = generated_dir / filename
        reviewed_path = reviewed_dir / filename
        if filename.endswith(".csv"):
            try:
                error = _verify_csv(generated_path, reviewed_path, case.get("tolerances", {}))
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"missing validation artifact: {exc.filename}") from exc
            max_numeric_error = max(max_numeric_error, error)
        elif filename.endswith(".json"):
            _verify_metadata(generated_path, reviewed_path, case["id"])
        else:
            raise ValueError(f"unsupported reference artifact type: {filename}")
    return {"artifact_count": len(references), "max_numeric_error": max_numeric_error}


__all__ = [
    "validate_equilibrium_artifacts",
    "enrich_generated_metadata",
    "validate_reference_metadata",
    "verify_case_references",
]
