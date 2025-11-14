"""
Plotting utilities for continuation and bifurcation diagrams.
"""

from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import Array

from jaxcont.core.continuation import ContinuationSolution


def plot_continuation(
    solution: ContinuationSolution,
    state_index: int = 0,
    ax: Optional[plt.Axes] = None,
    show_bifurcations: bool = True,
    stable_color: str = "blue",
    unstable_color: str = "red",
    **kwargs
) -> plt.Figure:
    """
    Plot continuation diagram.
    
    Args:
        solution: Continuation solution
        state_index: Which state variable to plot
        ax: Matplotlib axes (creates new figure if None)
        show_bifurcations: Whether to mark bifurcation points
        stable_color: Color for stable branches
        unstable_color: Color for unstable branches
        **kwargs: Additional plotting options
    
    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Extract data
    params = solution.parameters
    states = solution.states[:, state_index] if solution.state_dim > 1 else solution.states
    
    # Plot based on stability if available
    if solution.stability is not None:
        # Separate stable and unstable branches
        stable_mask = solution.stability
        
        # Plot stable points
        ax.plot(
            params[stable_mask],
            states[stable_mask],
            'o-',
            color=stable_color,
            label='Stable',
            markersize=3,
            **kwargs
        )
        
        # Plot unstable points
        if jnp.any(~stable_mask):
            ax.plot(
                params[~stable_mask],
                states[~stable_mask],
                'o--',
                color=unstable_color,
                label='Unstable',
                markersize=3,
                **kwargs
            )
    else:
        # Just plot the branch
        ax.plot(params, states, 'o-', markersize=3, **kwargs)
    
    # Mark bifurcations
    if show_bifurcations and solution.bifurcations:
        for bif in solution.bifurcations:
            bif_type = bif.get("type", "unknown")
            param = bif.get("parameter")
            state = bif.get("state")
            
            if state is not None and len(state) > state_index:
                state_val = state[state_index]
            else:
                # Interpolate state value at bifurcation parameter
                idx = jnp.searchsorted(params, param)
                if idx < len(states):
                    state_val = states[idx]
                else:
                    continue
            
            # Choose marker based on bifurcation type
            markers = {
                "fold": ("s", "green", "Fold"),
                "hopf": ("^", "magenta", "Hopf"),
                "period-doubling": ("v", "orange", "PD"),
                "branch-point": ("D", "purple", "BP"),
            }
            
            marker, color, label = markers.get(bif_type, ("x", "black", bif_type))
            ax.plot(param, state_val, marker, color=color, markersize=10, 
                   label=label, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel(f"State[{state_index}]", fontsize=12)
    ax.set_title("Continuation Diagram", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_bifurcation_diagram(
    solution: ContinuationSolution,
    state_index: int = 0,
    **kwargs
) -> plt.Figure:
    """
    Plot bifurcation diagram (alias for plot_continuation).
    
    Args:
        solution: Continuation solution
        state_index: Which state variable to plot
        **kwargs: Additional plotting options
    
    Returns:
        Matplotlib figure
    """
    return plot_continuation(solution, state_index=state_index, **kwargs)


def plot_phase_portrait(
    solution: ContinuationSolution,
    state_indices: Tuple[int, int] = (0, 1),
    param_indices: Optional[List[int]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot phase portraits for selected parameter values.
    
    Args:
        solution: Continuation solution
        state_indices: Which state variables to plot (x, y)
        param_indices: Which parameter indices to show (None = all)
        **kwargs: Additional plotting options
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if param_indices is None:
        # Sample evenly across the branch
        param_indices = jnp.linspace(0, solution.n_points - 1, min(10, solution.n_points), dtype=int)
    
    idx_x, idx_y = state_indices
    
    for i in param_indices:
        state = solution.states[i]
        param = solution.parameters[i]
        
        ax.plot(state[idx_x], state[idx_y], 'o', markersize=8, 
               label=f"p={param:.3f}")
    
    ax.set_xlabel(f"State[{idx_x}]", fontsize=12)
    ax.set_ylabel(f"State[{idx_y}]", fontsize=12)
    ax.set_title("Phase Portrait", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_eigenvalues(
    solution: ContinuationSolution,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot eigenvalue trajectories along the branch.
    
    Args:
        solution: Continuation solution (must have eigenvalues)
        ax: Matplotlib axes
        **kwargs: Additional plotting options
    
    Returns:
        Matplotlib figure
    """
    if solution.eigenvalues is None:
        raise ValueError("Solution does not contain eigenvalue information")
    
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig = ax.get_figure()
        ax1 = ax
        ax2 = None
    
    params = solution.parameters
    eigenvalues = solution.eigenvalues
    
    # Plot real parts
    for i in range(eigenvalues.shape[1]):
        ax1.plot(params, jnp.real(eigenvalues[:, i]), '-', alpha=0.7)
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Parameter", fontsize=12)
    ax1.set_ylabel("Re(λ)", fontsize=12)
    ax1.set_title("Real Part of Eigenvalues", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot imaginary parts if second axis available
    if ax2 is not None:
        for i in range(eigenvalues.shape[1]):
            ax2.plot(params, jnp.imag(eigenvalues[:, i]), '-', alpha=0.7)
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel("Parameter", fontsize=12)
        ax2.set_ylabel("Im(λ)", fontsize=12)
        ax2.set_title("Imaginary Part of Eigenvalues", fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
