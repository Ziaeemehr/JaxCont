"""
Phase-portrait and eigenvalue-trajectory plots for jaxcont.viz.
"""

from typing import List, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxcont.core.continuation import ContinuationSolution


def plot_phase_portrait(
    solution: ContinuationSolution,
    state_indices: Tuple[int, int] = (0, 1),
    param_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot phase portraits for selected parameter values.

    Args:
        solution: Continuation solution
        state_indices: Which state variables to plot (x, y)
        param_indices: Which parameter indices to show (None = all)
        ax: Matplotlib axes (creates new figure if None). Previously this
            parameter didn't exist, so a caller-supplied ax (e.g.
            example_03_van_der_pol.py's ax=ax2) silently landed in **kwargs
            and was never used -- the function always drew onto its own new
            figure instead.
        **kwargs: Additional plotting options

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    if param_indices is None:
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
    **kwargs,
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

    for i in range(eigenvalues.shape[1]):
        ax1.plot(params, jnp.real(eigenvalues[:, i]), '-', alpha=0.7)

    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel("Parameter", fontsize=12)
    ax1.set_ylabel("Re(λ)", fontsize=12)
    ax1.set_title("Real Part of Eigenvalues", fontsize=14)
    ax1.grid(True, alpha=0.3)

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
