"""Benchmark accuracy, computed by the harness from each tool's raw results.

Every tool on a row is measured against the same reference, never against its
own output: the analytic folds for ``cubic.eq-fold`` (p = -2/3 and 2/3), and
the committed MatCont bundle otherwise. Metrics a tool's worker does not
return stay ``None``, which keeps the row out of cross-tool tables.
"""

from __future__ import annotations

import csv
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "examples" / "validation" / "reference"
ANALYTIC = {"cubic.eq-fold": {"LP": [-2.0 / 3.0, 2.0 / 3.0]}}
# Fold states and the Jacobian there: f = p + u - u^3/3 folds at u = 1 (p = -2/3) and u = -1 (p = 2/3).
ANALYTIC_STATES = {"cubic.eq-fold": {"LP": [1.0, -1.0]}}
JACOBIAN = {"cubic.eq-fold": lambda u: 1.0 - u**2}
BUNDLES = {"cubic.eq-fold": "cubic/matcont/cubic.eq-fold", "lorenz84.eq-f": "lorenz84/matcont/lorenz84.eq-f"}


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
