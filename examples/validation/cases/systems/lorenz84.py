"""Extended Lorenz-84 model: the shared contract for JaxCont, MatCont and BifurcationKit.

All three transcriptions must use these equations, constants, parameter order,
seeds, windows and conventions. ``SEED_U``/``SEED_P`` is an equilibrium at
T=0.04 used by the tutorial; ``F_SWEEP["seed_state"]`` seeds the reference
F-sweep at F=3.
"""

import jax.numpy as jnp

STATE_NAMES = ("X", "Y", "Z", "U")
PARAMETER_NAMES = ("F",)
FIXED_ARGUMENT_NAMES = ("alpha", "beta", "gamma", "delta", "G", "T")
FIXED_ARGS = (0.25, 1.0, 0.987, 1.04, 0.25, 0.04)
SEED_U = (1.6673192028567203, -0.05172586841139392, 0.12923880103788027, -0.0660453938041009)
SEED_P = 1.7620532879639


def lorenz84_rhs(state, F, args):
    X, Y, Z, U = state
    alpha, beta, gamma, delta, G, T = args
    dX = -(Y**2) - Z**2 - alpha * X + alpha * F - gamma * U**2
    dY = X * Y - beta * X * Z - Y + G
    dZ = beta * X * Y + X * Z - Z
    dU = -delta * U + gamma * U * X + T
    return jnp.array([dX, dY, dZ, dU])


# Contract for all three independent transcriptions. Equilibrium continuation
# retains the scalar-F API above; codimension two always orders parameters F,T.
CODIM2_PARAMETER_NAMES = ("F", "T")
CONSTANT_NAMES = ("alpha", "beta", "gamma", "delta", "G")
CONSTANTS = FIXED_ARGS[:5]
F_SWEEP = {
    "fixed_T": 0.04,
    "bounds": [-1.5, 3.0],
    "seed_F": 3.0,
    "seed_state": [
        2.9787004394953343,
        -0.03868302503393752,
        0.058232737694740085,
        -0.02105288273117459,
    ],
    "seed_policy": "Newton-correct at F=3,T=0.04 before continuation",
    "both_directions": True,
    "ds": 0.002,
    "ds_max": 0.05,
    "max_steps_per_direction": 200,
    "newton_atol": 1e-12,
    "source": "historical lorenz84_equilibria.jl (Julia 02)",
    # The comparison domain is the closed window ``bounds`` on every sheet.
    # JaxCont's p_span is (start, end) and only stops at the end, so after the
    # fold the returning sheet runs past F=3; every tool's branch is clipped
    # to ``bounds`` (per sheet, never re-sorted) before matching.
    "comparison_domain": "clip each sheet to bounds before matching",
    # Full expected event set on the clipped branch at T=0.04 (JaxCont x64,
    # Claude review 2026-10-28): one fold, then three Hopf points on the
    # returning sheet. Windows identify events; they are not reference values.
    "expected_events": [
        {"kind": "LP", "window": {"F": [1.50, 1.58]}},
        {"kind": "H", "window": {"F": [1.59, 1.65]}},
        {"kind": "H", "window": {"F": [2.40, 2.55]}},
        {"kind": "H", "window": {"F": [2.80, 2.95]}},
    ],
}
# Windows select event identity; they are not measured reference values.
# They are frozen for owner review before any new producer starts.
EVENT_WINDOWS = {
    "fold_seed": {"F": [1.50, 1.58], "T": [0.039, 0.041]},
    # The (F,T) Hopf curve is seeded at the Hopf point nearest the BT; the
    # two Hopf points at larger F belong to other Hopf curves.
    "hopf_seed": {"F": [1.59, 1.65], "T": [0.039, 0.041]},
    "bt_positive_T": {"F": [1.5, 2.0], "T": [0.01, 0.03]},
}
CODIM2_CONTROLS = {
    "parameter_order": ["F", "T"],
    "free_parameter": "T",
    "T_bounds": [-0.1, 3.2],
    "ds": -0.001,
    "ds_min": 1e-5,
    "ds_max": 0.015,
    "max_steps": 30,
    "expected_T_domain": [-0.01, 0.04],
    "source": "historical lorenz84_codim2.jl; shared positive-T BT chart",
}
CONVENTIONS = {
    "state_order": list(STATE_NAMES),
    "parameter_order": list(CODIM2_PARAMETER_NAMES),
    "time": "original nondimensional time; no time rescaling",
    "jacobian": "rows are RHS components; columns follow X,Y,Z,U",
    "frequency": "angular frequency in radians per original time unit",
    "fold_label": "LP (BK fold aliases normalized by the exporter)",
    "hopf_label": "H (BK hopf aliases normalized by the exporter)",
    "bt_label": "BT; two small eigenvalues and geometric nullity one",
    "normal_forms": "record tool-native definition and eigenvector normalization; no implicit l1 conversion",
    "arclength": "record physical metric and state/parameter weights per tool",
    "chart_boundary": "Hopf chart termination near BT is a boundary, not a BT detection",
    "symmetry": "(U,T) -> (-U,-T); select positive-T BT by window",
    "review_status": "REVIEWED (Claude review 1, 2026-10-28)",
}


def lorenz84_ft_rhs(state, parameters, constants=CONSTANTS):
    """Two-parameter RHS with explicit (F,T) order and fixed model constants."""
    F, T = parameters
    return lorenz84_rhs(state, F, (*constants, T))


def lorenz84_jacobian(state, F, args=FIXED_ARGS):
    """Analytic state Jacobian for independent exporter sample checks."""
    X, Y, Z, U = state
    alpha, beta, gamma, delta, G, T = args
    return jnp.array(
        [
            [-alpha, -2 * Y, -2 * Z, -2 * gamma * U],
            [Y - beta * Z, X - 1, -beta * X, 0],
            [beta * Y + Z, beta * X, X - 1, 0],
            [gamma * U, 0, 0, -delta + gamma * X],
        ]
    )


def select_event(events, kind, window):
    """Select exactly one normalized event by type and named parameter window.

    No numeric special-point indices are part of this contract. Exporters
    normalize their native labels and parameter names before calling this.
    """
    import math

    matches = []
    for event in events:
        if event.get("kind", event.get("event_type")) != kind:
            continue
        values = event.get("parameters", event)
        if all(
            name in values
            and math.isfinite(float(values[name]))
            and low <= float(values[name]) <= high
            for name, (low, high) in window.items()
        ):
            matches.append(event)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} event in {window}, found {len(matches)}")
    return matches[0]
