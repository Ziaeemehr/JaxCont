"""Van der Pol: x'=y, y'=mu*(1-x**2)*y-x.

At the origin the Jacobian has trace mu and determinant one, so its
critical eigenvalues are +/-i at mu=0. The vector field is linear there:
this is a degenerate Hopf/center, with zero first Lyapunov coefficient.
"""

import jax.numpy as jnp

STATE_NAMES = ("x", "y")
PARAMETER_NAMES = ("mu",)
SEED_U, SEED_P = (0.0, 0.0), -2.0

def rhs(u, mu, args):
    x, y = u
    return jnp.array([y, mu * (1 - x*x) * y - x])
