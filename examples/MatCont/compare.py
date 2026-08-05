"""Numerical comparison primitives for MatCont validation artifacts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


class ValidationMismatch(AssertionError):
    """Raised when a validation artifact disagrees beyond its tolerance."""


def _absolute_errors(actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if actual.shape != reference.shape:
        raise ValidationMismatch(
            f"Shape mismatch: actual has shape {actual.shape}, reference has shape {reference.shape}"
        )
    return np.abs(actual - reference)


def _scaled_limits(actual: np.ndarray, reference: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative")
    return atol + rtol * np.maximum(np.abs(actual), np.abs(reference))


def scaled_close(
    actual: Any, reference: Any, *, atol: float, rtol: float = 0.0
) -> dict[str, Any]:
    """Compare equally shaped values with symmetric absolute and relative tolerances."""
    actual_values = np.asarray(actual)
    reference_values = np.asarray(reference)
    errors = _absolute_errors(actual_values, reference_values)
    limits = _scaled_limits(actual_values, reference_values, atol, rtol)
    max_error = float(np.max(errors)) if errors.size else 0.0
    max_tolerance = float(np.max(limits)) if limits.size else float(atol)
    diagnostics = {
        "max_error": max_error,
        "max_tolerance": max_tolerance,
        "errors": errors,
    }
    if not np.all(np.isfinite(errors)) or np.any(errors > limits):
        raise ValidationMismatch(
            f"Scaled comparison failed: max error {max_error:.6g} exceeds tolerance {max_tolerance:.6g}"
        )
    return diagnostics


def interpolate_observable(
    parameters: Any, observables: Any, query_parameters: Any
) -> np.ndarray:
    """Linearly interpolate scalar or vector observables on a monotone segment."""
    parameter_values = np.asarray(parameters, dtype=float)
    observable_values = np.asarray(observables)
    query_values = np.asarray(query_parameters, dtype=float)
    if parameter_values.ndim != 1 or parameter_values.size < 2:
        raise ValueError("parameters must be a one-dimensional sequence with at least two points")
    if observable_values.ndim == 0 or observable_values.shape[0] != parameter_values.size:
        raise ValueError("observables must have one value per parameter")

    differences = np.diff(parameter_values)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("parameters must be strictly monotone for interpolation")
    if np.any(query_values < parameter_values.min()) or np.any(query_values > parameter_values.max()):
        raise ValidationMismatch("Interpolation query lies outside the monotone parameter segment")

    if np.all(differences < 0):
        parameter_values = parameter_values[::-1]
        observable_values = observable_values[::-1]

    flattened = observable_values.reshape(parameter_values.size, -1)
    interpolated = np.stack(
        [np.interp(query_values, parameter_values, flattened[:, column]) for column in range(flattened.shape[1])],
        axis=-1,
    )
    return interpolated.reshape(query_values.shape + observable_values.shape[1:])


def _event_arrays(events: Sequence[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    try:
        kinds = [str(event["kind"]) for event in events]
        parameters = np.asarray([event["parameter"] for event in events], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Events must contain string 'kind' and numeric 'parameter' fields") from error
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Event parameters must be finite")
    return kinds, parameters


def match_events(
    actual: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    *,
    atol: float,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Uniquely match events by kind and parameter location."""
    if len(actual) != len(reference):
        raise ValidationMismatch(
            f"Event-count mismatch: actual has {len(actual)}, reference has {len(reference)}"
        )
    actual_kinds, actual_parameters = _event_arrays(actual)
    reference_kinds, reference_parameters = _event_arrays(reference)
    if not actual:
        return {"max_error": 0.0, "max_tolerance": float(atol), "assignments": [], "errors": []}

    errors = np.abs(actual_parameters[:, None] - reference_parameters[None, :])
    limits = _scaled_limits(
        actual_parameters[:, None], reference_parameters[None, :], atol, rtol
    )
    compatible = np.equal.outer(actual_kinds, reference_kinds)
    penalty = max(float(np.max(errors)), float(np.max(limits)), 1.0) * (len(actual) + 1)
    cost = np.where(compatible, errors, penalty)
    actual_indices, reference_indices = linear_sum_assignment(cost)
    assigned_actual = {reference_index: actual_index for actual_index, reference_index in zip(actual_indices, reference_indices)}
    assignments = [(int(assigned_actual[index]), index) for index in range(len(reference))]
    assignment_errors = np.asarray([errors[actual_index, reference_index] for actual_index, reference_index in assignments])
    assignment_limits = np.asarray([limits[actual_index, reference_index] for actual_index, reference_index in assignments])
    assignment_kinds_match = all(
        actual_kinds[actual_index] == reference_kinds[reference_index]
        for actual_index, reference_index in assignments
    )
    max_error = float(np.max(assignment_errors))
    max_tolerance = float(np.max(assignment_limits))
    diagnostics = {
        "max_error": max_error,
        "max_tolerance": max_tolerance,
        "assignments": assignments,
        "errors": assignment_errors.tolist(),
    }
    if not assignment_kinds_match or np.any(assignment_errors > assignment_limits):
        raise ValidationMismatch(
            f"Event matching failed: max error {max_error:.6g} exceeds tolerance {max_tolerance:.6g}"
        )
    return diagnostics


def match_spectrum(
    actual: Any,
    reference: Any,
    *,
    atol: float,
    rtol: float = 0.0,
    exclude_trivial: bool = True,
) -> dict[str, Any]:
    """Uniquely match spectra, optionally removing one multiplier nearest ``+1``."""
    actual_values = np.asarray(actual, dtype=complex)
    reference_values = np.asarray(reference, dtype=complex)
    if actual_values.ndim != 1 or reference_values.ndim != 1:
        raise ValueError("Spectra must be one-dimensional")
    if not actual_values.size or not reference_values.size:
        raise ValidationMismatch("Spectra must be non-empty")

    actual_trivial = int(np.argmin(np.abs(actual_values - 1.0))) if exclude_trivial else None
    reference_trivial = (
        int(np.argmin(np.abs(reference_values - 1.0))) if exclude_trivial else None
    )
    actual_indices = (
        np.delete(np.arange(actual_values.size), actual_trivial)
        if exclude_trivial
        else np.arange(actual_values.size)
    )
    reference_indices = (
        np.delete(np.arange(reference_values.size), reference_trivial)
        if exclude_trivial
        else np.arange(reference_values.size)
    )
    reduced_actual = actual_values[actual_indices]
    reduced_reference = reference_values[reference_indices]
    if reduced_actual.size != reduced_reference.size:
        raise ValidationMismatch(
            "Spectrum-count mismatch after excluding one trivial multiplier: "
            f"actual has {reduced_actual.size}, reference has {reduced_reference.size}"
        )
    if not reduced_actual.size:
        return {
            "max_error": 0.0,
            "max_tolerance": float(atol),
            "assignments": [],
            "errors": [],
            "excluded": {"actual": actual_trivial, "reference": reference_trivial},
        }

    errors = np.abs(reduced_actual[:, None] - reduced_reference[None, :])
    limits = _scaled_limits(
        reduced_actual[:, None], reduced_reference[None, :], atol, rtol
    )
    feasible = np.isfinite(errors) & (errors <= limits)
    penalty = max(float(np.max(errors)), float(np.max(limits)), 1.0) * (
        reduced_actual.size + 1
    )
    cost = np.where(feasible, errors, penalty)
    rows, columns = linear_sum_assignment(cost)
    assigned_actual = {column: row for row, column in zip(rows, columns)}
    assignments = [
        (int(actual_indices[assigned_actual[column]]), int(reference_indices[column]))
        for column in range(reduced_reference.size)
    ]
    assignment_errors = np.asarray([errors[row, column] for row, column in zip(rows, columns)])
    assignment_limits = np.asarray([limits[row, column] for row, column in zip(rows, columns)])
    max_error = float(np.max(assignment_errors))
    max_tolerance = float(np.max(assignment_limits))
    diagnostics = {
        "max_error": max_error,
        "max_tolerance": max_tolerance,
        "assignments": assignments,
        "errors": assignment_errors.tolist(),
        "excluded": {"actual": actual_trivial, "reference": reference_trivial},
    }
    if not np.all(np.isfinite(assignment_errors)) or np.any(assignment_errors > assignment_limits):
        raise ValidationMismatch(
            f"Spectrum matching failed: max error {max_error:.6g} exceeds tolerance {max_tolerance:.6g}"
        )
    return diagnostics
