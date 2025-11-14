"""
Eigenvalue computation and stability analysis for equilibria.
"""

from typing import Tuple, Dict
import jax.numpy as jnp
from jax import Array, jacfwd


def compute_eigenvalues(jacobian: Array) -> Array:
    """
    Compute eigenvalues of the Jacobian matrix.
    
    Args:
        jacobian: Jacobian matrix
    
    Returns:
        Array of eigenvalues (sorted by real part, descending)
    """
    eigenvalues = jnp.linalg.eigvals(jacobian)
    
    # Sort by real part (descending)
    idx = jnp.argsort(-jnp.real(eigenvalues))
    return eigenvalues[idx]


def analyze_stability(eigenvalues: Array, tolerance: float = 1e-10) -> Dict[str, any]:
    """
    Analyze stability based on eigenvalues.
    
    Args:
        eigenvalues: Array of eigenvalues
        tolerance: Tolerance for determining if eigenvalue is on imaginary axis
    
    Returns:
        Dictionary with stability information
    """
    real_parts = jnp.real(eigenvalues)
    imag_parts = jnp.imag(eigenvalues)
    
    # Count eigenvalues by location
    n_positive = jnp.sum(real_parts > tolerance)
    n_negative = jnp.sum(real_parts < -tolerance)
    n_zero = jnp.sum(jnp.abs(real_parts) <= tolerance)
    
    # Determine stability
    is_stable = n_positive == 0 and n_zero == 0
    is_unstable = n_positive > 0
    is_center = n_zero > 0 and n_positive == 0
    
    # Classify equilibrium type
    if is_stable:
        if jnp.all(jnp.abs(imag_parts) < tolerance):
            equilibrium_type = "stable node"
        else:
            equilibrium_type = "stable focus"
    elif is_unstable:
        if n_negative > 0:
            equilibrium_type = "saddle"
        else:
            if jnp.all(jnp.abs(imag_parts) < tolerance):
                equilibrium_type = "unstable node"
            else:
                equilibrium_type = "unstable focus"
    elif is_center:
        equilibrium_type = "center"
    else:
        equilibrium_type = "unknown"
    
    return {
        "is_stable": bool(is_stable),
        "is_unstable": bool(is_unstable),
        "type": equilibrium_type,
        "n_unstable": int(n_positive),
        "n_stable": int(n_negative),
        "n_center": int(n_zero),
        "eigenvalues": eigenvalues,
        "dominant_eigenvalue": eigenvalues[0],  # Rightmost eigenvalue
    }


def compute_eigenvalues_along_branch(
    problem,
    solution,
    **kwargs
) -> Array:
    """
    Compute eigenvalues along a continuation branch.
    
    Args:
        problem: Continuation problem
        solution: Continuation solution
        **kwargs: Additional arguments
    
    Returns:
        Array of eigenvalues at each point (n_points, state_dim)
    """
    from jaxcont.core.continuation import ContinuationProblem
    
    n_points = solution.n_points
    state_dim = solution.state_dim
    
    eigenvalues_list = []
    
    for i in range(n_points):
        u = solution.states[i]
        p = solution.parameters[i]
        
        # Compute Jacobian at this point
        def f(u_eval):
            return problem.evaluate_rhs(u_eval, p)
        
        jac = jacfwd(f)(u)
        eigs = compute_eigenvalues(jac)
        eigenvalues_list.append(eigs)
    
    return jnp.array(eigenvalues_list)


def compute_stability_along_branch(problem, solution) -> Array:
    """
    Compute stability indicators along continuation branch.
    
    Args:
        problem: Continuation problem
        solution: Continuation solution
    
    Returns:
        Boolean array indicating stability at each point
    """
    eigenvalues = compute_eigenvalues_along_branch(problem, solution)
    
    stability = []
    for eigs in eigenvalues:
        analysis = analyze_stability(eigs)
        stability.append(analysis["is_stable"])
    
    return jnp.array(stability)
