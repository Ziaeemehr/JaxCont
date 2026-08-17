"""Schema validation for normalized MatCont reference artifacts."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jax
import numpy as np
from scipy.optimize import linear_sum_assignment

from .compare import (
    ValidationMismatch,
    interpolate_observable,
    match_events,
    match_spectrum,
    scaled_close,
)

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


def validate_periodic_artifacts(reference_dir: Path, case_id: str) -> dict[str, int]:
    """Validate the normalized periodic branch/event/Floquet schemas."""
    reference_dir = Path(reference_dir)
    branch = _read_rows(
        reference_dir / f"{case_id}_branch.csv",
        {"case_id", "point", "parameter", "period", "residual_norm"},
    )
    events = _read_rows(
        reference_dir / f"{case_id}_events.csv",
        {"case_id", "event_index", "event_type", "point", "parameter", "period"},
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
    if not branch or not spectra:
        raise ValueError(f"{case_id} periodic branch/spectrum must be non-empty")
    if any(row["case_id"] != case_id for row in branch + events + spectra):
        raise ValueError(f"{case_id} artifacts contain a different case_id")
    if any(row["spectrum_kind"].strip().upper() != "FLOQUET" for row in spectra):
        raise ValueError(f"{case_id} periodic spectrum contains non-Floquet rows")
    return {"branch_rows": len(branch), "event_rows": len(events), "spectrum_rows": len(spectra)}


def validate_reference_metadata(
    path: Path, case_id: str, *, require_reviewed: bool = True
) -> dict[str, Any]:
    """Load metadata and enforce reproducibility plus optional review state."""
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
    if require_reviewed and metadata["reviewed"] is not True:
        raise ValueError(f"{Path(path).name} is not marked as a reviewed reference")
    equation_hash = metadata["equation_hash"]
    valid_hash = isinstance(equation_hash, str) and re.fullmatch(
        r"sha256:[0-9a-fA-F]{64}", equation_hash
    )
    if not valid_hash:
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
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _column_atol(column: str, tolerances: dict[str, float]) -> float:
    if column == "parameter":
        return float(tolerances.get("parameter_atol", 1e-9))
    if column == "period":
        return float(tolerances.get("period_atol", 1e-9))
    if column == "frequency":
        return float(tolerances.get("frequency_atol", 1e-9))
    if column == "fold_coefficient":
        return float(tolerances.get("fold_coefficient_atol", 1e-9))
    if column == "first_lyapunov":
        return float(tolerances.get("first_lyapunov_atol", 1e-9))
    if column == "residual_norm":
        return float(tolerances.get("residual_atol", 1e-9))
    if column in {"real", "imag"}:
        return float(tolerances.get("spectrum_atol", tolerances.get("multiplier_atol", 1e-9)))
    if column.startswith("state_"):
        return float(tolerances.get("extrema_atol", tolerances.get("state_atol", 1e-9)))
    return 0.0 if column in {"point", "event_index", "multiplier_index"} else 1e-9


def _read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or ())
            return fields, list(reader)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing validation artifact: {path}") from exc


def _as_float(row: dict[str, Any], column: str) -> float:
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationMismatch(f"invalid numeric {column!r} value") from exc


def _branch_coordinate(case_id: str, fields: set[str]) -> str:
    if case_id == "MC-EQ-001":
        return "state_0" if "state_0" in fields else "parameter"
    if "parameter" not in fields:
        raise ValidationMismatch("branch is missing parameter")
    return "parameter"


def _unique_sorted(rows: list[dict[str, Any]], coordinate: str) -> tuple[np.ndarray, list[dict]]:
    ordered = sorted(rows, key=lambda row: _as_float(row, coordinate))
    values = np.asarray([_as_float(row, coordinate) for row in ordered])
    unique, indices = np.unique(values, return_index=True)
    return unique, [ordered[int(index)] for index in indices]


def _monotone_segments(
    rows: list[dict[str, Any]], coordinate: str
) -> list[list[dict[str, Any]]]:
    """Split continuation order at coordinate turning points, retaining the turn."""
    if len(rows) < 2:
        return [rows]
    values = np.asarray([_as_float(row, coordinate) for row in rows])
    segments: list[list[dict[str, Any]]] = []
    start = 0
    direction = 0
    for index, difference in enumerate(np.diff(values), start=1):
        new_direction = int(np.sign(difference))
        if new_direction == 0:
            continue
        if direction and new_direction != direction:
            segments.append(rows[start:index])
            start = index - 1
        direction = new_direction
    segments.append(rows[start:])
    return segments


def _compare_branch_rows(
    case: dict[str, Any],
    actual_fields: list[str],
    actual_rows: list[dict[str, Any]],
    reference_fields: list[str],
    reference_rows: list[dict[str, Any]],
    *,
    require_same_extent: bool,
    require_same_schema: bool,
    _allow_segments: bool = True,
) -> dict[str, float]:
    if not actual_rows or not reference_rows:
        raise ValidationMismatch("branch artifacts must be non-empty")
    actual_set = set(actual_fields)
    reference_set = set(reference_fields)
    if require_same_schema and actual_set != reference_set:
        raise ValidationMismatch("generated and reviewed branch columns differ")
    if not reference_set <= actual_set and require_same_schema:
        raise ValidationMismatch("generated branch is missing reviewed columns")
    coordinate = _branch_coordinate(case["id"], actual_set & reference_set)
    if _allow_segments and coordinate == "parameter":
        actual_segments = _monotone_segments(actual_rows, coordinate)
        reference_segments = _monotone_segments(reference_rows, coordinate)
        if len(actual_segments) != len(reference_segments):
            raise ValidationMismatch(
                "branch turning-point count differs: "
                f"actual has {len(actual_segments)} segments, "
                f"reference has {len(reference_segments)}"
            )
        if len(actual_segments) > 1:
            count = len(actual_segments)
            penalty = 1e100
            costs = np.full((count, count), penalty)
            pair_diagnostics: dict[tuple[int, int], dict[str, float]] = {}
            for actual_index, actual_segment in enumerate(actual_segments):
                for reference_index, reference_segment in enumerate(reference_segments):
                    try:
                        diagnostics = _compare_branch_rows(
                            case,
                            actual_fields,
                            actual_segment,
                            reference_fields,
                            reference_segment,
                            require_same_extent=require_same_extent,
                            require_same_schema=require_same_schema,
                            _allow_segments=False,
                        )
                    except ValidationMismatch:
                        continue
                    pair_diagnostics[(actual_index, reference_index)] = diagnostics
                    costs[actual_index, reference_index] = diagnostics["max_error"]
            actual_indices, reference_indices = linear_sum_assignment(costs)
            assignments = list(zip(actual_indices.tolist(), reference_indices.tolist()))
            infeasible = any(
                costs[actual_index, reference_index] >= penalty
                for actual_index, reference_index in assignments
            )
            if infeasible:
                raise ValidationMismatch(
                    "no tolerance-feasible matching for folded branch segments"
                )
            return {
                "max_error": max(
                    pair_diagnostics[assignment]["max_error"] for assignment in assignments
                ),
                "segment_count": count,
            }
    actual_x, actual_ordered = _unique_sorted(actual_rows, coordinate)
    reference_x, reference_ordered = _unique_sorted(reference_rows, coordinate)
    tolerances = case.get("tolerances", {})
    coordinate_atol = _column_atol(coordinate, tolerances)
    coordinate_error = 0.0
    if require_same_extent:
        extent_diagnostics = scaled_close(
            np.asarray([actual_x[0], actual_x[-1]]),
            np.asarray([reference_x[0], reference_x[-1]]),
            atol=coordinate_atol,
            rtol=float(tolerances.get("rtol", 1e-9)),
        )
        coordinate_error = extent_diagnostics["max_error"]
    lower = max(actual_x[0], reference_x[0])
    upper = min(actual_x[-1], reference_x[-1])
    if upper < lower:
        if actual_x.size == reference_x.size == 1 and coordinate_error <= coordinate_atol:
            lower = upper = actual_x[0]
        else:
            raise ValidationMismatch("branches have no overlapping continuation interval")
    if actual_x.size == reference_x.size == 1:
        query = np.asarray([lower])
    else:
        candidates = (
            np.unique(np.concatenate([actual_x, reference_x]))
            if require_same_extent
            else actual_x
        )
        if not require_same_extent and case["id"] == "MC-LC-001":
            candidates = np.unique(
                np.concatenate([candidates, [reference_x[0], reference_x[-1]]])
            )
        query = candidates[(candidates >= lower) & (candidates <= upper)]
        if query.size > 201:
            query = np.linspace(lower, upper, 201)

    ignored = {"case_id", "point", coordinate, "residual_norm"}
    discrete_columns = {"stable", "unstable_dimension"} & actual_set & reference_set
    if case["id"] == "MC-LC-001":
        ignored.update(
            column for column in actual_set | reference_set if column.startswith("state_")
        )
    numeric_columns = []
    for column in sorted((actual_set & reference_set) - ignored - discrete_columns):
        try:
            np.asarray([_as_float(row, column) for row in actual_ordered + reference_ordered])
        except ValidationMismatch:
            continue
        numeric_columns.append(column)
    max_error = coordinate_error
    for column in numeric_columns:
        actual_values = np.asarray([_as_float(row, column) for row in actual_ordered])
        reference_values = np.asarray([_as_float(row, column) for row in reference_ordered])
        if actual_x.size == 1:
            actual_interpolated = actual_values
        else:
            actual_interpolated = interpolate_observable(actual_x, actual_values, query)
        if reference_x.size == 1:
            reference_interpolated = reference_values
        else:
            reference_interpolated = interpolate_observable(reference_x, reference_values, query)
        diagnostics = scaled_close(
            actual_interpolated,
            reference_interpolated,
            atol=_column_atol(column, tolerances),
            rtol=float(tolerances.get("rtol", 1e-9)),
        )
        max_error = max(max_error, diagnostics["max_error"])
    for column in sorted(discrete_columns):
        actual_values = np.asarray([int(_as_float(row, column)) for row in actual_ordered])
        reference_values = np.asarray(
            [int(_as_float(row, column)) for row in reference_ordered]
        )
        actual_nearest = np.abs(actual_x[:, None] - query[None, :]).argmin(axis=0)
        reference_nearest = np.abs(reference_x[:, None] - query[None, :]).argmin(axis=0)
        if not np.array_equal(
            actual_values[actual_nearest], reference_values[reference_nearest]
        ):
            raise ValidationMismatch(f"branch {column} labels differ")
    if case["id"] == "MC-LC-001":
        extrema_columns = sorted(
            column
            for column in actual_set & reference_set
            if column.startswith("state_") and (column.endswith(("_min", "_max")))
        )
        if not extrema_columns:
            raise ValidationMismatch("radial branch is missing state extrema")
        actual_radius = np.asarray(
            [
                max(abs(_as_float(row, column)) for column in extrema_columns)
                for row in actual_ordered
            ]
        )
        reference_radius = np.asarray(
            [
                max(abs(_as_float(row, column)) for column in extrema_columns)
                for row in reference_ordered
            ]
        )
        actual_radius_query = (
            actual_radius
            if actual_x.size == 1
            else interpolate_observable(actual_x, actual_radius, query)
        )
        reference_radius_query = (
            reference_radius
            if reference_x.size == 1
            else interpolate_observable(reference_x, reference_radius, query)
        )
        diagnostics = scaled_close(
            actual_radius_query,
            reference_radius_query,
            atol=float(tolerances.get("radius_atol", 5e-3)),
            rtol=float(tolerances.get("rtol", 0.0)),
        )
        max_error = max(max_error, diagnostics["max_error"])
    for rows, label in ((actual_rows, "generated"), (reference_rows, "reviewed")):
        if "residual_norm" in rows[0] and "residual_atol" in tolerances:
            maximum = max(abs(_as_float(row, "residual_norm")) for row in rows)
            if maximum > tolerances["residual_atol"]:
                raise ValidationMismatch(f"{label} branch residual {maximum:.6g} exceeds tolerance")
    return {"max_error": max_error}


def _normalized_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"kind": str(row["event_type"]).strip(), "parameter": _as_float(row, "parameter")}
        for row in rows
    ]


def _compare_event_rows(
    case: dict[str, Any],
    actual_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    tolerances = case.get("tolerances", {})
    diagnostics = match_events(
        _normalized_events(actual_rows),
        _normalized_events(reference_rows),
        atol=float(tolerances.get("parameter_atol", 1e-9)),
        rtol=float(tolerances.get("rtol", 0.0)),
    )
    max_error = diagnostics["max_error"]
    for actual_index, reference_index in diagnostics["assignments"]:
        actual = actual_rows[actual_index]
        reference = reference_rows[reference_index]
        for column in ("period", "frequency", "fold_coefficient", "first_lyapunov"):
            if column not in actual or column not in reference:
                continue
            actual_value = _as_float(actual, column)
            reference_value = _as_float(reference, column)
            if np.isnan(actual_value) and np.isnan(reference_value):
                continue
            if not (np.isfinite(actual_value) and np.isfinite(reference_value)):
                raise ValidationMismatch(f"event {column} availability differs")
            comparison = scaled_close(
                actual_value,
                reference_value,
                atol=_column_atol(column, tolerances),
                rtol=float(tolerances.get("rtol", 0.0)),
            )
            max_error = max(max_error, comparison["max_error"])
    diagnostics["max_error"] = max_error
    return diagnostics, diagnostics["assignments"]


def _complex_values(rows: list[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: int(float(row["multiplier_index"])))
    return np.asarray(
        [complex(_as_float(row, "real"), _as_float(row, "imag")) for row in ordered]
    )


def _group_spectra(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        key = (int(float(row["event_index"])), str(row["event_type"]).strip())
        grouped.setdefault(key, []).append(row)
    return grouped


def _branch_spectrum_curve(
    branch_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    coordinate: str,
) -> list[dict[str, Any]]:
    by_point: dict[int, list[dict[str, Any]]] = {}
    for row in spectrum_rows:
        by_point.setdefault(int(float(row["point"])), []).append(row)
    curve = []
    for branch_row in branch_rows:
        point = int(float(branch_row["point"]))
        if point in by_point:
            curve.append(
                {
                    coordinate: _as_float(branch_row, coordinate),
                    "values": _complex_values(by_point[point]),
                }
            )
    if len(curve) != len(by_point):
        raise ValidationMismatch("branch spectrum points do not match branch rows")
    return curve


def _unique_spectrum_points(
    curve: list[dict[str, Any]], coordinate: str
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(curve, key=lambda row: _as_float(row, coordinate))
    coordinates = np.asarray([_as_float(row, coordinate) for row in ordered])
    unique, indices = np.unique(coordinates, return_index=True)
    values = np.stack([ordered[int(index)]["values"] for index in indices])
    return unique, values


def _interpolate_spectrum_roots(
    coordinates: np.ndarray, values: np.ndarray, query: np.ndarray
) -> list[np.ndarray]:
    """Interpolate symmetric characteristic coefficients, then recover roots."""
    coefficients = np.stack([np.poly(point_values) for point_values in values])
    if coordinates.size == 1:
        interpolated = np.repeat(coefficients, query.size, axis=0)
    else:
        real = interpolate_observable(coordinates, coefficients.real, query)
        imag = interpolate_observable(coordinates, coefficients.imag, query)
        interpolated = real + 1j * imag
    return [np.roots(point_coefficients) for point_coefficients in interpolated]


def _compare_spectrum_segments(
    case: dict[str, Any],
    actual_curve: list[dict[str, Any]],
    reference_curve: list[dict[str, Any]],
    coordinate: str,
    *,
    require_same_extent: bool,
    kind: str,
) -> dict[str, float]:
    actual_x, actual_values = _unique_spectrum_points(actual_curve, coordinate)
    reference_x, reference_values = _unique_spectrum_points(reference_curve, coordinate)
    if actual_values.shape[1] != reference_values.shape[1]:
        raise ValidationMismatch("branch spectrum dimensions differ")
    lower = max(actual_x[0], reference_x[0])
    upper = min(actual_x[-1], reference_x[-1])
    if upper < lower:
        raise ValidationMismatch("branch spectra have no overlapping continuation interval")
    if actual_x.size == reference_x.size == 1:
        query = np.asarray([lower])
    elif require_same_extent:
        query = np.unique(np.concatenate([actual_x, reference_x]))
        query = query[(query >= lower) & (query <= upper)]
        if query.size > 101:
            query = np.linspace(lower, upper, 101)
    else:
        query = actual_x[(actual_x >= lower) & (actual_x <= upper)]
        if query.size > 101:
            indices = np.linspace(0, query.size - 1, 101).round().astype(int)
            query = query[indices]
    actual_roots = _interpolate_spectrum_roots(actual_x, actual_values, query)
    reference_roots = _interpolate_spectrum_roots(reference_x, reference_values, query)
    tolerances = case.get("tolerances", {})
    atol = float(
        tolerances.get("spectrum_atol", tolerances.get("multiplier_atol", 1e-9))
    )
    max_error = 0.0
    for actual_at_point, reference_at_point in zip(actual_roots, reference_roots):
        diagnostics = match_spectrum(
            actual_at_point,
            reference_at_point,
            atol=atol,
            rtol=float(tolerances.get("rtol", 0.0)),
            exclude_trivial=kind == "FLOQUET",
        )
        max_error = max(max_error, diagnostics["max_error"])
    return {"max_error": max_error}


def _compare_spectrum_rows(
    case: dict[str, Any],
    actual_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    actual_events: list[dict[str, Any]],
    reference_events: list[dict[str, Any]],
    event_assignments: list[tuple[int, int]],
    actual_branch: list[dict[str, Any]],
    reference_branch: list[dict[str, Any]],
    *,
    require_same_extent: bool,
) -> dict[str, float]:
    if not actual_rows and not reference_rows:
        return {"max_error": 0.0}
    actual_groups = _group_spectra(actual_rows)
    reference_groups = _group_spectra(reference_rows)
    atol = float(
        case.get("tolerances", {}).get(
            "spectrum_atol", case.get("tolerances", {}).get("multiplier_atol", 1e-9)
        )
    )
    max_error = 0.0
    actual_branch_rows = [row for row in actual_rows if int(float(row["event_index"])) == -1]
    reference_branch_rows = [
        row for row in reference_rows if int(float(row["event_index"])) == -1
    ]
    if bool(actual_branch_rows) != bool(reference_branch_rows):
        raise ValidationMismatch("branch spectrum availability differs")
    if actual_branch_rows:
        coordinate = _branch_coordinate(
            case["id"], set(actual_branch[0]) & set(reference_branch[0])
        )
        kind = str(reference_branch_rows[0]["spectrum_kind"]).strip().upper()
        actual_curve = _branch_spectrum_curve(
            actual_branch, actual_branch_rows, coordinate
        )
        reference_curve = _branch_spectrum_curve(
            reference_branch, reference_branch_rows, coordinate
        )
        actual_segments = _monotone_segments(actual_curve, coordinate)
        reference_segments = _monotone_segments(reference_curve, coordinate)
        if len(actual_segments) != len(reference_segments):
            raise ValidationMismatch("branch spectrum turning-point count differs")
        segment_count = len(actual_segments)
        penalty = 1e100
        costs = np.full((segment_count, segment_count), penalty)
        pair_diagnostics: dict[tuple[int, int], dict[str, float]] = {}
        for actual_index, actual_segment in enumerate(actual_segments):
            for reference_index, reference_segment in enumerate(reference_segments):
                try:
                    diagnostics = _compare_spectrum_segments(
                        case,
                        actual_segment,
                        reference_segment,
                        coordinate,
                        require_same_extent=require_same_extent,
                        kind=kind,
                    )
                except ValidationMismatch:
                    continue
                pair_diagnostics[(actual_index, reference_index)] = diagnostics
                costs[actual_index, reference_index] = diagnostics["max_error"]
        actual_indices, reference_indices = linear_sum_assignment(costs)
        assignments = list(zip(actual_indices.tolist(), reference_indices.tolist()))
        if any(
            costs[actual_index, reference_index] >= penalty
            for actual_index, reference_index in assignments
        ):
            raise ValidationMismatch(
                "no tolerance-feasible matching for folded branch spectrum segments"
            )
        max_error = max(
            max_error,
            max(pair_diagnostics[assignment]["max_error"] for assignment in assignments),
        )
    for actual_event_index, reference_event_index in event_assignments:
        actual_event = actual_events[actual_event_index]
        reference_event = reference_events[reference_event_index]
        actual_key = (
            int(float(actual_event["event_index"])),
            str(actual_event["event_type"]).strip(),
        )
        reference_key = (
            int(float(reference_event["event_index"])),
            str(reference_event["event_type"]).strip(),
        )
        if actual_key not in actual_groups or reference_key not in reference_groups:
            raise ValidationMismatch("event-local spectrum is missing")
        kind = str(reference_groups[reference_key][0]["spectrum_kind"]).strip().upper()
        diagnostics = match_spectrum(
            _complex_values(actual_groups[actual_key]),
            _complex_values(reference_groups[reference_key]),
            atol=atol,
            rtol=float(case.get("tolerances", {}).get("rtol", 0.0)),
            exclude_trivial=kind == "FLOQUET",
        )
        max_error = max(max_error, diagnostics["max_error"])
    return {"max_error": max_error}


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
    """Compare regenerated topology with reviewed references without writing either."""
    generated_dir = Path(generated_dir)
    reviewed_dir = Path(reviewed_dir)
    references = case.get("references", [])
    if not references:
        return {"artifact_count": 0, "max_numeric_error": 0.0}

    def named(suffix: str) -> str | None:
        return next((filename for filename in references if filename.endswith(suffix)), None)

    branch_name = named("_branch.csv") or named("branch.csv")
    event_name = named("_events.csv") or named("events.csv")
    spectrum_name = named("_multipliers.csv") or named("multipliers.csv")
    metadata_name = named("_metadata.json") or named("metadata.json")
    max_numeric_error = 0.0
    branch_diagnostics = {"max_error": 0.0}
    event_diagnostics = {"max_error": 0.0, "assignments": []}
    spectrum_diagnostics = {"max_error": 0.0}
    actual_branch: list[dict[str, Any]] = []
    reference_branch: list[dict[str, Any]] = []
    actual_events: list[dict[str, Any]] = []
    reference_events: list[dict[str, Any]] = []
    if branch_name:
        actual_fields, actual_branch = _read_table(generated_dir / branch_name)
        reference_fields, reference_branch = _read_table(reviewed_dir / branch_name)
        branch_diagnostics = _compare_branch_rows(
            case,
            actual_fields,
            actual_branch,
            reference_fields,
            reference_branch,
            require_same_extent=True,
            require_same_schema=True,
        )
        max_numeric_error = max(max_numeric_error, branch_diagnostics["max_error"])
    if event_name:
        actual_event_fields, actual_events = _read_table(generated_dir / event_name)
        reference_event_fields, reference_events = _read_table(reviewed_dir / event_name)
        if set(actual_event_fields) != set(reference_event_fields):
            raise ValidationMismatch("generated and reviewed event columns differ")
        event_diagnostics, assignments = _compare_event_rows(
            case, actual_events, reference_events
        )
        max_numeric_error = max(max_numeric_error, event_diagnostics["max_error"])
    else:
        assignments = []
    if spectrum_name:
        actual_spectrum_fields, actual_spectra = _read_table(generated_dir / spectrum_name)
        reference_spectrum_fields, reference_spectra = _read_table(
            reviewed_dir / spectrum_name
        )
        if set(actual_spectrum_fields) != set(reference_spectrum_fields):
            raise ValidationMismatch("generated and reviewed spectrum columns differ")
        spectrum_diagnostics = _compare_spectrum_rows(
            case,
            actual_spectra,
            reference_spectra,
            actual_events,
            reference_events,
            assignments,
            actual_branch,
            reference_branch,
            require_same_extent=True,
        )
        max_numeric_error = max(max_numeric_error, spectrum_diagnostics["max_error"])
    if metadata_name:
        _verify_metadata(
            generated_dir / metadata_name, reviewed_dir / metadata_name, case["id"]
        )
    return {
        "artifact_count": len(references),
        "max_numeric_error": max_numeric_error,
        "branch_max_error": branch_diagnostics["max_error"],
        "event_max_error": event_diagnostics["max_error"],
        "spectrum_max_error": spectrum_diagnostics["max_error"],
    }


def _reference_names(case: dict[str, Any]) -> tuple[str, str, str]:
    references = case.get("references", [])
    try:
        branch = next(name for name in references if name.endswith("_branch.csv"))
        events = next(name for name in references if name.endswith("_events.csv"))
        spectra = next(name for name in references if name.endswith("_multipliers.csv"))
    except StopIteration as exc:
        raise ValidationMismatch(
            f"{case['id']} lacks normalized branch/event/spectrum references"
        ) from exc
    return branch, events, spectra


def _case_result_rows(case: dict[str, Any], result: Any) -> tuple:
    artifacts = result.artifacts
    parameters = np.asarray(artifacts["parameters"])
    states = np.asarray(artifacts["states"])
    branch_rows: list[dict[str, Any]] = []
    if case["id"].startswith("MC-EQ-"):
        spectra = np.asarray(artifacts["spectra"])
        unstable_dimensions = np.sum(np.real(spectra) > 0.0, axis=1)
        branch_fields = [
            "case_id",
            "point",
            "parameter",
            "stable",
            "unstable_dimension",
        ] + [
            f"state_{index}" for index in range(states.shape[1])
        ]
        for point, (parameter, state, stable, unstable_dimension) in enumerate(
            zip(
                parameters,
                states,
                np.asarray(artifacts["stability"]),
                unstable_dimensions,
            )
        ):
            row = {
                "case_id": case["id"],
                "point": point,
                "parameter": float(parameter),
                "stable": int(bool(stable)),
                "unstable_dimension": int(unstable_dimension),
            }
            row.update({f"state_{index}": float(value) for index, value in enumerate(state)})
            branch_rows.append(row)
        spectrum_kind = "EIGENVALUE"
    else:
        periods = np.asarray(artifacts["periods"])
        state_min = np.asarray(artifacts["state_min"])
        state_max = np.asarray(artifacts["state_max"])
        branch_fields = ["case_id", "point", "parameter", "period"]
        for dimension in range(state_min.shape[1]):
            branch_fields.extend([f"state_{dimension}_min", f"state_{dimension}_max"])
        for point, parameter in enumerate(parameters):
            row = {
                "case_id": case["id"],
                "point": point,
                "parameter": float(parameter),
                "period": float(periods[point]),
            }
            for dimension in range(state_min.shape[1]):
                row[f"state_{dimension}_min"] = float(state_min[point, dimension])
                row[f"state_{dimension}_max"] = float(state_max[point, dimension])
            branch_rows.append(row)
        spectra = np.asarray(artifacts["multipliers"])
        spectrum_kind = "FLOQUET"

    event_fields = [
        "case_id",
        "event_index",
        "event_type",
        "point",
        "parameter",
        "period",
        "frequency",
        "fold_coefficient",
        "first_lyapunov",
    ]
    event_rows = []
    for event_index, event in enumerate(artifacts.get("events", [])):
        event_rows.append(
            {
                "case_id": case["id"],
                "event_index": event_index,
                "event_type": event["kind"],
                "point": -1,
                "parameter": float(event["parameter"]),
                "period": float(event.get("period", np.nan)),
                "frequency": float(event.get("frequency", np.nan)),
                "fold_coefficient": float(event.get("fold_coefficient", np.nan)),
                "first_lyapunov": float(event.get("first_lyapunov", np.nan)),
            }
        )
    spectrum_fields = [
        "case_id",
        "point",
        "event_index",
        "event_type",
        "spectrum_kind",
        "multiplier_index",
        "real",
        "imag",
    ]
    spectrum_rows = []
    for point, values in enumerate(spectra):
        for multiplier_index, value in enumerate(values):
            spectrum_rows.append(
                {
                    "case_id": case["id"],
                    "point": point,
                    "event_index": -1,
                    "event_type": "BRANCH",
                    "spectrum_kind": spectrum_kind,
                    "multiplier_index": multiplier_index,
                    "real": float(np.real(value)),
                    "imag": float(np.imag(value)),
                }
            )
    for event_index, event_spectrum in enumerate(artifacts.get("event_spectra", [])):
        for multiplier_index, value in enumerate(np.asarray(event_spectrum["values"])):
            spectrum_rows.append(
                {
                    "case_id": case["id"],
                    "point": -1,
                    "event_index": event_index,
                    "event_type": event_spectrum["kind"],
                    "spectrum_kind": spectrum_kind,
                    "multiplier_index": multiplier_index,
                    "real": float(np.real(value)),
                    "imag": float(np.imag(value)),
                }
            )
    return (
        branch_fields,
        branch_rows,
        event_fields,
        event_rows,
        spectrum_fields,
        spectrum_rows,
    )


def compare_case_result_to_reference(
    case: dict[str, Any], result: Any, reference_dir: Path
) -> dict[str, float]:
    """Numerically compare a Python/JaxCont case result with committed MatCont data."""
    branch_name, event_name, spectrum_name = _reference_names(case)
    reference_branch_fields, reference_branch = _read_table(Path(reference_dir) / branch_name)
    _reference_event_fields, reference_events = _read_table(Path(reference_dir) / event_name)
    reference_spectrum_fields, reference_spectra = _read_table(
        Path(reference_dir) / spectrum_name
    )
    if case["id"].startswith("MC-EQ-"):
        validate_equilibrium_artifacts(reference_dir, case["id"])
    else:
        required_branch = {"case_id", "point", "parameter", "period", "residual_norm"}
        if not required_branch <= set(reference_branch_fields):
            raise ValidationMismatch("periodic branch reference schema is incomplete")
        required_spectrum = {
            "case_id",
            "point",
            "event_index",
            "event_type",
            "spectrum_kind",
            "multiplier_index",
            "real",
            "imag",
        }
        if not required_spectrum <= set(reference_spectrum_fields):
            raise ValidationMismatch("periodic spectrum reference schema is incomplete")
    (
        actual_branch_fields,
        actual_branch,
        _actual_event_fields,
        actual_events,
        _actual_spectrum_fields,
        actual_spectra,
    ) = _case_result_rows(case, result)
    branch = _compare_branch_rows(
        case,
        actual_branch_fields,
        actual_branch,
        reference_branch_fields,
        reference_branch,
        require_same_extent=False,
        require_same_schema=False,
    )
    events, assignments = _compare_event_rows(case, actual_events, reference_events)
    spectra = _compare_spectrum_rows(
        case,
        actual_spectra,
        reference_spectra,
        actual_events,
        reference_events,
        assignments,
        actual_branch,
        reference_branch,
        require_same_extent=False,
    )
    return {
        "branch_max_error": branch["max_error"],
        "event_max_error": events["max_error"],
        "spectrum_max_error": spectra["max_error"],
    }


__all__ = [
    "compare_case_result_to_reference",
    "enrich_generated_metadata",
    "validate_equilibrium_artifacts",
    "validate_periodic_artifacts",
    "validate_reference_metadata",
    "verify_case_references",
]
