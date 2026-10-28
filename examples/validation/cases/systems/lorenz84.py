"""lorenz84 equations; parameter order and seeds are visible in the tutorial."""

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
    dX = -Y ** 2 - Z ** 2 - alpha * X + alpha * F - gamma * U ** 2
    dY = X * Y - beta * X * Z - Y + G
    dZ = beta * X * Y + X * Z - Z
    dU = -delta * U + gamma * U * X + T
    return jnp.array([dX, dY, dZ, dU])
