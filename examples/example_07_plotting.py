"""
Plot stability and bifurcation labels
=====================================
"""

import os
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation
from jaxcont.utils import plot_continuation
import matplotlib.pyplot as plt

def pitchfork_rhs(state, params):
    """Modified ODE system: dx/dt = r + x - x^3/3"""
    x = state[0] if len(state.shape) == 1 and state.shape[0] > 1 else state
    r = params["r"]
    dxdt = r + x - x**3 / 3
    return jnp.array([dxdt]) if isinstance(dxdt, (int, float)) else dxdt


def main():
    print("Testing plotting fixes for stability and bifurcation labels...")
    
    # Define problem
    problem = ContinuationProblem(
        rhs=pitchfork_rhs,
        u0=jnp.array([-2.0]),
        params={"r": -1.0},
        continuation_param="r",
        problem_type="equilibrium",
    )

    # Run continuation with stability computation
    solution = equilibrium_continuation(
        problem,
        param_range=(-1.0, 1.0),
        ds=0.01,
        max_steps=300,
        detect_bifurcations=True,
        compute_stability=True,
        verbose=False,
        bifurcation_tolerance=1e-4,
    )

    print(f"Continuation completed: {solution.n_points} points")
    print(f"Bifurcations detected: {len(solution.bifurcations)}")
    
    # Print bifurcation details
    if solution.bifurcations:
        print("\nBifurcation points:")
        for i, bif in enumerate(solution.bifurcations, 1):
            print(f"  {i}. Type: {bif['type']}, r = {bif['parameter']:.6f}, x = {bif['state'][0]:.6f}")
    
    # Check stability information
    if solution.stability is not None:
        n_stable = jnp.sum(solution.stability)
        n_unstable = solution.n_points - n_stable
        print(f"\nStability statistics:")
        print(f"  Stable points: {n_stable}")
        print(f"  Unstable points: {n_unstable}")
        
        # Find where stability changes
        stability_array = jnp.array(solution.stability, dtype=int)
        changes = jnp.where(jnp.diff(stability_array) != 0)[0]
        if len(changes) > 0:
            print(f"\nStability transitions at indices: {changes}")
            for idx in changes:
                print(f"  Index {idx}->{idx+1}: r={solution.parameters[idx]:.6f} -> r={solution.parameters[idx+1]:.6f}")
                print(f"    Stable: {solution.stability[idx]} -> {solution.stability[idx+1]}")
        
        # Show a sample of stability near bifurcations
        if solution.bifurcations:
            print("\nStability around bifurcations:")
            for bif in solution.bifurcations:
                bif_param = bif['parameter']
                # Find closest index
                idx = jnp.argmin(jnp.abs(solution.parameters - bif_param))
                start = max(0, idx - 2)
                end = min(solution.n_points, idx + 3)
                print(f"\n  Around r={bif_param:.6f} (index {idx}):")
                for i in range(start, end):
                    marker = " <--" if i == idx else ""
                    print(f"    [{i}] r={solution.parameters[i]:.6f}, stable={solution.stability[i]}{marker}")
    
    # Use the updated plot_continuation function
    fig = plot_continuation(
        solution,
        state_index=0,
        show_bifurcations=True,
    )
    
    plt.savefig("images/test_plotting_fix.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to: images/test_plotting_fix.png")
    print("\nCheck the plot:")
    print("1. Fold bifurcations should appear only once in the legend")
    print("2. Unstable branch (between fold points) should be red and dashed")
    print("3. Stable branches should be blue and solid")
    print("4. No extra lines connecting separate stable branches")
    
    plt.show()


if __name__ == "__main__":
    main()
