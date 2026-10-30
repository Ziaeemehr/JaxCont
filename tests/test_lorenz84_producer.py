"""Scientific tests for fresh Lorenz-84 candidates; no external reference input."""
import numpy as np
import pytest
from examples.validation.cases.systems.lorenz84 import F_SWEEP, select_event

pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def runs():
    from examples.validation.cases.python_cases.lorenz84 import produce
    return {name: produce("lorenz84." + name) for name in ("eq-f", "fold-ft", "hopf-ft")}

def test_branch_has_expected_domain_and_events(runs):
    bundle = runs["eq-f"]
    assert len(bundle["branch"]) > 20
    assert all(-1.5 <= r["F"] <= 3 for r in bundle["branch"])
    assert max(r["F"] for r in bundle["branch"]) == pytest.approx(3)
    assert len(bundle["events"]) == len(F_SWEEP["expected_events"])
    for event in F_SWEEP["expected_events"]:
        select_event(bundle["events"], event["kind"], event["window"])
    # Measured |error|: 5.17e-13; float64-only producer, explicit Newton tolerance 1e-12.
    assert max((r['residual_norm'] for r in bundle['branch'])) < 1e-11
    for name in ("fold-ft", "hopf-ft"):
        assert len(runs[name]["branch"]) > 2
        assert runs[name]["metadata"]["parameter_order"] == ["F", "T"]
        assert runs[name]["metadata"]["chart_status"]

def test_bt_has_jordan_structure(runs):
    from examples.validation.cases.python_cases.lorenz84 import produce
    from examples.validation.cases.systems.lorenz84 import EVENT_WINDOWS
    bundle = produce("lorenz84.bt-ft")
    event = select_event(bundle["events"], "BT", EVENT_WINDOWS["bt_positive_T"])
    # Measured |error|: 7.63e-17; float64-only producer, explicit Newton tolerance 1e-12.
    assert event['residual_norm'] < 1e-11
    # Measured |error|: 2.64e-16; float64-only producer, explicit Newton tolerance 1e-12.
    assert event['jordan_v0_residual'] < 1e-11
    # Measured |error|: 2.43e-14; float64-only producer, explicit Newton tolerance 1e-12.
    assert event['jordan_v1_residual'] < 1e-11
    assert event["geometric_nullity"] == 1
    eig = np.array([complex(r["real"], r["imag"]) for r in bundle["spectra"] if r["event_id"]])
    # Measured |error|: 5.59e-09; defective double-zero eigenvalue splitting scales with sqrt(float64 epsilon).
    assert sum(abs(eig) < 1e-07) == 2
