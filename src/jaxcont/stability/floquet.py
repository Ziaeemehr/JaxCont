"""
Floquet multipliers for periodic orbit stability analysis.
"""

from typing import Tuple
import jax.numpy as jnp
from jax import Array, jacfwd
from scipy.integrate import solve_ivp
import numpy as np


def compute_floquet_multipliers(
    rhs: callable,
    u0: Array,
    period: float,
    params: dict,
    **kwargs
) -> Array:
    """
    Compute Floquet multipliers for a periodic orbit.
    
    Floquet multipliers are eigenvalues of the monodromy matrix,
    which is the linearized flow map over one period.
    
    Args:
        rhs: Right-hand side function f(u, params)
        u0: Initial point on periodic orbit
        period: Period of the orbit
        params: System parameters
        **kwargs: Additional arguments for integration
    
    Returns:
        Array of Floquet multipliers (one will always be 1 due to time-translation)
    """
    n = len(u0)
    
    # Integrate the variational equation to get monodromy matrix
    # We need to solve: dΦ/dt = J(u(t)) * Φ, with Φ(0) = I
    
    def augmented_rhs(t, y):
        """Augmented system: [u, vec(Φ)]"""
        u = y[:n]
        phi_vec = y[n:]
        phi = phi_vec.reshape((n, n))
        
        # Compute du/dt
        dudt = np.array(rhs(jnp.array(u), params))
        
        # Compute Jacobian
        def f_jac(u_eval):
            return rhs(u_eval, params)
        jac = np.array(jacfwd(f_jac)(jnp.array(u)))
        
        # Compute dΦ/dt = J * Φ
        dphidt = jac @ phi
        
        # Return augmented vector
        return np.concatenate([dudt, dphidt.flatten()])
    
    # Initial conditions: [u0, I]
    y0 = np.concatenate([np.array(u0), np.eye(n).flatten()])
    
    # Integrate over one period
    sol = solve_ivp(
        augmented_rhs,
        (0, period),
        y0,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    # Extract monodromy matrix
    phi_final = sol.y[n:, -1].reshape((n, n))
    
    # Compute eigenvalues (Floquet multipliers)
    multipliers = jnp.linalg.eigvals(jnp.array(phi_final))
    
    # Sort by magnitude (descending)
    idx = jnp.argsort(-jnp.abs(multipliers))
    return multipliers[idx]


def analyze_periodic_orbit_stability(
    floquet_multipliers: Array,
    tolerance: float = 1e-6
) -> dict:
    """
    Analyze stability of periodic orbit based on Floquet multipliers.
    
    Args:
        floquet_multipliers: Floquet multipliers
        tolerance: Tolerance for unit circle check
    
    Returns:
        Dictionary with stability information
    """
    abs_multipliers = jnp.abs(floquet_multipliers)
    
    # Count multipliers outside unit circle (ignoring the trivial one at 1)
    # Find the trivial multiplier
    trivial_idx = jnp.argmin(jnp.abs(abs_multipliers - 1.0))
    
    # Check others
    max_mult = 0.0
    for i, mult in enumerate(abs_multipliers):
        if i != trivial_idx:
            max_mult = max(max_mult, float(mult))
    
    is_stable = max_mult < 1.0 + tolerance
    
    return {
        "is_stable": bool(is_stable),
        "max_multiplier": float(max_mult),
        "multipliers": floquet_multipliers,
        "n_unstable": int(jnp.sum((abs_multipliers > 1.0 + tolerance))),
    }
