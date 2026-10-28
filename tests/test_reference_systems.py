"""Independent, hand-evaluated Lorenz-84 contract samples (no continuation)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.validation.cases.systems import lorenz84 as model
from examples.validation.registry import load_registry, select_cases


@pytest.mark.parametrize(
    "state,F,rhs,jacobian",
    [
        (
            [0, 0, 0, 0],
            3,
            [0.75, 0.25, 0, 0.04],
            [[-0.25, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1.04]],
        ),
        (
            [1, 2, -1, 0.5],
            2,
            [-4.99675, 1.25, 2, 0.0135],
            [[-0.25, -4, 2, -0.987], [3, 0, -1, 0], [1, 1, 0, 0], [0.4935, 0, 0, -0.053]],
        ),
        (
            [0.5, -0.25, 0.75, -0.2],
            1.5,
            [-0.41448, 0, -0.5, 0.1493],
            [
                [-0.25, 0.5, -1.5, 0.3948],
                [-1, -0.5, -0.5, 0],
                [0.5, 0.5, -0.5, 0],
                [-0.1974, 0, 0, -0.5465],
            ],
        ),
    ],
)
def test_lorenz84_rhs_matches_independent_samples(state, F, rhs, jacobian):
    x = jnp.array(state, dtype=float)
    np.testing.assert_allclose(model.lorenz84_rhs(x, F, model.FIXED_ARGS), rhs, atol=4e-7, rtol=0)
    np.testing.assert_allclose(
        model.lorenz84_jacobian(x, F, model.FIXED_ARGS), jacobian, atol=2e-7, rtol=0
    )
    np.testing.assert_allclose(
        jax.jacfwd(model.lorenz84_rhs)(x, F, model.FIXED_ARGS), jacobian, atol=2e-7, rtol=0
    )
    np.testing.assert_allclose(
        model.lorenz84_ft_rhs(x, jnp.array([F, 0.04]), model.CONSTANTS), rhs, atol=4e-7, rtol=0
    )


def test_event_selection_uses_type_and_window():
    events = [
        {"kind": "H", "F": 1.55, "T": 0.04},
        {"kind": "LP", "F": 1.55, "T": 0.04},
        {"kind": "LP", "F": 2.5, "T": 0.04},
    ]
    window = {"F": [1.5, 1.6], "T": [0.039, 0.041]}
    assert model.select_event(events[::-1], "LP", window) == events[1]
    for invalid in (events + [events[1]], events[:1]):
        with pytest.raises(ValueError, match="exactly one"):
            model.select_event(invalid, "LP", window)


def test_flagship_registry_and_conventions_are_ready_for_review():
    registry = load_registry()
    assert model.CODIM2_PARAMETER_NAMES == ("F", "T")
    assert model.F_SWEEP["fixed_T"] == 0.04
    assert model.F_SWEEP["bounds"] == [-1.5, 3.0]
    for run in ("eq-f", "fold-ft", "hopf-ft", "bt-ft"):
        case = select_cases(registry, ["lorenz84." + run])[0]
        assert case["system"] == "lorenz84"
        assert case["evidence_status"] == "PENDING"
        assert set(case["tool_coverage"]) == {"jaxcont", "matcont", "bifurcationkit"}
        assert case["model_contract"]["review_status"].startswith("REVIEWED")


@pytest.mark.slow
def test_f_sweep_windows_select_exactly_the_expected_events():
    """Every expected F-sweep event and both curve seeds match exactly one event."""
    import jaxcont as jc

    sweep = model.F_SWEEP
    problem = jc.bif_problem(
        model.lorenz84_rhs, u0=jnp.array(sweep["seed_state"]), p0=sweep["seed_F"],
        args=model.FIXED_ARGS,
    )
    result = jc.continuation(
        problem, jc.PseudoArclength(), p_span=(sweep["seed_F"], sweep["bounds"][0]),
        settings=jc.ContinuationPar(
            ds=sweep["ds"], ds_max=sweep["ds_max"],
            max_steps=sweep["max_steps_per_direction"], compute_stability=True,
        ),
        events=[jc.Fold(), jc.Hopf()],
    )
    low, high = sweep["bounds"]
    events = [
        {"kind": {"fold": "LP", "hopf": "H"}[event.kind], "F": float(event.p), "T": sweep["fixed_T"]}
        for event in result.events
        if low <= float(event.p) <= high
    ]
    assert sorted(e["kind"] for e in events) == sorted(e["kind"] for e in sweep["expected_events"])
    for expected in sweep["expected_events"]:
        model.select_event(events, expected["kind"], expected["window"])
    model.select_event(events, "LP", model.EVENT_WINDOWS["fold_seed"])
    model.select_event(events, "H", model.EVENT_WINDOWS["hopf_seed"])
