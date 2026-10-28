"""Cubic S-curve: x' = p + x - x**3/3.

The fold condition 1-x**2=0 gives x=+/-1; inserting these roots into
p=x**3/3-x gives the two analytic fold parameters. State and parameter
orders and the lower-branch equilibrium seed are recorded below.
"""

import jax.numpy as jnp

STATE_NAMES = ("x",)
PARAMETER_NAMES = ("p",)
SEED_U, SEED_P = (-2.1038034027355366,), -1.0

def rhs(u, p, args):
    return jnp.array([p + u[0] - u[0]**3 / 3])
