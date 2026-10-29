"""Benchmark accuracy, computed by the harness from each tool's raw results.

Every tool on a row is measured against the same reference, never against its
own output: the analytic folds for ``cubic.eq-fold`` (p = -2/3 and 2/3), and
the committed MatCont bundle otherwise. Metrics a tool's worker does not
return stay ``None``, which keeps the row out of cross-tool tables.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "validation" / "reference"
ANALYTIC = {"cubic.eq-fold": {"LP": [-2.0 / 3.0, 2.0 / 3.0]}}
# Fold states and the Jacobian there: f = p + u - u^3/3 folds at u = 1 (p = -2/3) and u = -1 (p = 2/3).
ANALYTIC_STATES = {"cubic.eq-fold": {"LP": [1.0, -1.0]}}
JACOBIAN = {"cubic.eq-fold": lambda u: 1.0 - u**2}
BUNDLES = {"cubic.eq-fold": "cubic/matcont/cubic.eq-fold", "lorenz84.eq-f": "lorenz84/matcont/lorenz84.eq-f"}


def valid_accuracy(value):
    """Accuracy evidence must be a finite, nonnegative number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def meets_limit(value, limit):
    return valid_accuracy(value) and valid_accuracy(limit) and value <= limit


def eligible(row, limits):
    """Every required metric must be measured and meet its limit."""
    accuracy = row.get("accuracy") or {}
    return all(meets_limit(accuracy.get(metric), limit) for metric, limit in limits.items())


def branch_residual(states, params, rhs):
    """Maximum residual only for a complete, finite, nonempty branch."""
    states, params = np.asarray(states), np.asarray(params)
    if (not states.size or not params.size or len(states) != len(params)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(params))):
        return None
    residuals = [np.asarray(rhs(u, p)) for u, p in zip(states, params)]
    if any(not value.size or not np.all(np.isfinite(value)) for value in residuals):
        return None
    return max(float(np.max(np.abs(value))) for value in residuals)


def reference_events(canonical_id: str) -> dict[str, list[float]]:
    """Event parameters by type (``LP``, ``H``) for one workload."""
    if canonical_id in ANALYTIC:
        return ANALYTIC[canonical_id]
    with open(REFERENCE / BUNDLES[canonical_id] / "events.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events: dict[str, list[float]] = {}
    for row in rows:
        events.setdefault(row.get("event_type") or row.get("kind"), []).append(float(row["parameter"]))
    return events


def measure(case, events: dict[str, list[float]], residual: float | None,
            event_states: dict[str, list[float]] | None = None) -> dict:
    """Errors of one tool's results against the shared reference.

    ``parameter_atol``: events matched one-to-one in sorted parameter order
    within type. ``state_atol`` and ``spectrum_atol`` (where an analytic
    reference exists): the matched events' state error and the critical
    eigenvalue's distance from 0 at the returned state. ``residual_atol``: the
    largest |f| the harness evaluated on the returned branch.
    """
    measured = {metric: None for metric in case.accuracy_limits}
    # Validate all samples before sorting or reducing: max can hide later NaNs,
    # and zip would silently truncate mismatched state/event arrays.
    if any(not math.isfinite(value) for values in events.values() for value in values):
        return measured
    if event_states is not None:
        if any(kind not in events or len(values) != len(events[kind])
               or any(not math.isfinite(value) for value in values)
               for kind, values in event_states.items()):
            return measured
        if any(kind not in event_states for kind in events):
            return measured
    if residual is not None and not valid_accuracy(residual):
        return measured
    errors, state_errors, spectral = [], [], []
    for kind, values in reference_events(case.canonical_id).items():
        found = events.get(kind, [])
        if len(found) != len(values):
            return measured | {"missing_events": kind}
        order = sorted(range(len(found)), key=lambda i: found[i])
        errors += [abs(a - found[i]) for a, i in zip(sorted(values), order)]
        states = (event_states or {}).get(kind)
        reference_states = ANALYTIC_STATES.get(case.canonical_id, {}).get(kind)
        if states is not None and reference_states is not None:
            pairs = sorted(zip(values, reference_states))
            state_errors += [abs(want - states[i]) for (_, want), i in zip(pairs, order)]
            spectral += [abs(JACOBIAN[case.canonical_id](states[i])) for i in order]
    measured["parameter_atol"] = max(errors) if errors else None
    measured["residual_atol"] = residual
    if "state_atol" in measured and state_errors:
        measured["state_atol"] = max(state_errors)
    if "spectrum_atol" in measured and spectral:
        measured["spectrum_atol"] = max(spectral)
    return measured
