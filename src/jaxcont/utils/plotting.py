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
    state_name: Optional[str] = None,
    param_name: Optional[str] = None,
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
        state_name: Label for the plotted state variable. Defaults to
            ``solution.state_names[state_index]`` if the problem defined one,
            else ``"State[<index>]"``.
        param_name: Label for the continuation parameter on the x-axis.
            Defaults to ``solution.param_name`` if the problem defined one,
            else ``"Parameter"``.
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

    if state_name is None:
        state_names = getattr(solution, "state_names", None)
        state_name = (
            state_names[state_index] if state_names is not None
            else f"State[{state_index}]"
        )
    if param_name is None:
        param_name = getattr(solution, "param_name", None) or "Parameter"

    # Extract data
    params = solution.parameters
    states = solution.states[:, state_index] if solution.state_dim > 1 else solution.states
    
    # Plot based on stability if available
    if solution.stability is not None:
        # Plot continuous segments to avoid connecting separated regions
        stable_mask = solution.stability
        
        # Find continuous segments
        def plot_segments(mask, color, linestyle, label):
            """Plot continuous segments where mask is True."""
            if not jnp.any(mask):
                return
            
            # Find transitions
            mask_int = mask.astype(int)
            transitions = jnp.diff(jnp.concatenate([jnp.array([0]), mask_int, jnp.array([0])]))
            starts = jnp.where(transitions == 1)[0]
            ends = jnp.where(transitions == -1)[0]
            
            # Plot each continuous segment
            label_used = False
            for start, end in zip(starts, ends):
                segment_label = label if not label_used else None
                ax.plot(
                    params[start:end],
                    states[start:end],
                    marker='o',
                    color=color,
                    linestyle=linestyle,
                    label=segment_label,
                    markersize=3,
                    **kwargs
                )
                label_used = True
        
        # Plot stable and unstable segments
        plot_segments(stable_mask, stable_color, '-', 'Stable')
        plot_segments(~stable_mask, unstable_color, '--', 'Unstable')
    else:
        # Just plot the branch
        ax.plot(params, states, 'o-', markersize=3, **kwargs)
    
    # Mark bifurcations
    if show_bifurcations and solution.bifurcations:
        # Track which bifurcation types have been labeled
        labeled_types = set()
        
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
            
            marker, color, label_text = markers.get(bif_type, ("x", "black", bif_type))
            
            # Only add label for the first occurrence of each bifurcation type
            label = label_text if bif_type not in labeled_types else None
            labeled_types.add(bif_type)
            
            ax.plot(param, state_val, marker, color=color, markersize=10, 
                   label=label, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel(state_name, fontsize=12)
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
