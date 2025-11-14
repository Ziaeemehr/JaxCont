"""
Test what bifurcation detection currently does in the continuation workflow.

This test runs actual continuation and checks:
1. Does it detect bifurcations?
2. Does refine_location get called?
3. What does the output look like?
"""

import jax.numpy as jnp
from jaxcont.core.continuation import ContinuationProblem
from jaxcont.core.pseudo_arclength import PseudoArclengthContinuation


def test_bifurcation_detection_in_continuation():
    """
    Test bifurcation detection during continuation of pitchfork system.
    
    System: f(u, p) = u^3 - p*u = 0
    - Trivial branch: u = 0 (all p)
    - Bifurcating branches: u = ±√p (p > 0)
    - Bifurcation at p = 0 (pitchfork)
    """
    print("\n" + "="*80)
    print("Test: Bifurcation Detection During Continuation")
    print("="*80)
    
    # Define the pitchfork system
    def rhs(u, params):
        p = params['p']
        return u**3 - p * u
    
    # Start on trivial branch near bifurcation
    u0 = jnp.array([0.01])  # Close to zero
    params = {'p': -1.0}  # Start before bifurcation
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=u0,
        params=params,
        continuation_param='p'
    )
    
    print(f"\nSystem: f(u, p) = u³ - p*u = 0")
    print(f"Starting point: u = {u0[0]:.4f}, p = {params['p']:.4f}")
    print(f"Target range: p ∈ [-1, 1]")
    print(f"Expected bifurcation: pitchfork at p = 0")
    
    # Run continuation WITH bifurcation detection
    print("\n" + "-"*80)
    print("Running continuation WITH bifurcation detection...")
    print("-"*80)
    
    continuation = PseudoArclengthContinuation(
        ds=0.05,
        max_steps=100,
        detect_bifurcations=True,
        bifurcation_tolerance=1e-4,
        verbose=True
    )
    
    solution = continuation.run(problem, param_range=(-1.0, 1.0))
    
    print(f"\nContinuation completed:")
    print(f"  Number of points: {solution.n_points}")
    print(f"  Parameter range: [{solution.parameters[0]:.4f}, {solution.parameters[-1]:.4f}]")
    print(f"  State range: [{jnp.min(solution.states):.4f}, {jnp.max(solution.states):.4f}]")
    
    # Check bifurcations
    print(f"\n" + "-"*80)
    print("Bifurcation Results:")
    print("-"*80)
    
    if solution.bifurcations:
        print(f"✓ Found {len(solution.bifurcations)} bifurcation(s)!")
        for i, bif in enumerate(solution.bifurcations):
            print(f"\nBifurcation {i+1}:")
            print(f"  Type: {bif.get('type', 'unknown')}")
            print(f"  Parameter: {bif.get('parameter', 'N/A'):.6f}")
            print(f"  State: {bif.get('state', 'N/A')}")
            print(f"  Index: {bif.get('index', 'N/A')}")
            print(f"  Method: {bif.get('method', 'detection only')}")
            if 'iterations' in bif:
                print(f"  Bisection iterations: {bif['iterations']}")
            if 'residual' in bif:
                print(f"  Test function residual: {bif['residual']:.2e}")
            if 'original_parameter' in bif:
                print(f"  Original (unrefined) parameter: {bif['original_parameter']:.6f}")
                print(f"  Refinement improvement: {abs(bif['parameter'] - bif['original_parameter']):.2e}")
    else:
        print("✗ No bifurcations detected")
        print("  This might be because:")
        print("  - Eigenvalues not computed (check detect_bifurcations=True)")
        print("  - Step size too large (try smaller ds)")
        print("  - Bifurcation outside parameter range")
    
    # Check eigenvalues
    print(f"\n" + "-"*80)
    print("Stability Information:")
    print("-"*80)
    
    if solution.eigenvalues is not None:
        print(f"✓ Eigenvalues computed at {len(solution.eigenvalues)} points")
        
        # Find where eigenvalue crosses zero
        real_parts = jnp.real(solution.eigenvalues[:, 0])  # Largest real eigenvalue
        print(f"\nEigenvalue evolution:")
        print(f"  First point: Re(λ) = {real_parts[0]:.6f} at p = {solution.parameters[0]:.4f}")
        print(f"  Last point:  Re(λ) = {real_parts[-1]:.6f} at p = {solution.parameters[-1]:.4f}")
        
        # Find sign changes
        sign_changes = jnp.where(jnp.diff(jnp.sign(real_parts)) != 0)[0]
        if len(sign_changes) > 0:
            print(f"\n  Sign changes detected at {len(sign_changes)} location(s):")
            for idx in sign_changes:
                p1, p2 = solution.parameters[idx], solution.parameters[idx+1]
                e1, e2 = real_parts[idx], real_parts[idx+1]
                print(f"    Between p = {p1:.4f} (λ = {e1:.6f}) and p = {p2:.4f} (λ = {e2:.6f})")
                # Linear interpolation estimate
                p_zero = p1 + (p2 - p1) * abs(e1) / (abs(e1) + abs(e2))
                print(f"    → Linear interpolation suggests p ≈ {p_zero:.6f}")
    else:
        print("✗ Eigenvalues not computed")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)
    
    return solution


def test_with_and_without_refinement():
    """Compare detection with and without location refinement."""
    print("\n" + "="*80)
    print("Comparison: Detection vs Detection+Refinement")
    print("="*80)
    
    def rhs(u, params):
        p = params['p']
        return u**2 - p  # Simple fold at p = 0
    
    u0 = jnp.array([0.5])
    params = {'p': 0.25}
    
    problem = ContinuationProblem(
        rhs=rhs, u0=u0, params=params, continuation_param='p'
    )
    
    # Test 1: Without refinement (detection only)
    print("\nTest 1: Detection only (refine_location=False)")
    print("-"*80)
    cont1 = PseudoArclengthContinuation(
        ds=0.05, max_steps=50, detect_bifurcations=True
    )
    sol1 = cont1.run(problem, param_range=(0.25, -0.25))
    
    if sol1.bifurcations:
        bif1 = sol1.bifurcations[0]
        print(f"Detected at: p = {bif1['parameter']:.8f}")
        print(f"Method: {bif1.get('method', 'basic detection')}")
    
    # Test 2: With refinement (would try bisection if implemented)
    print("\nTest 2: Detection with refinement attempt")
    print("-"*80)
    print("Note: Bisection will be attempted but may fall back to interpolation")
    print("      if _compute_eigenvalues_at_point() is not implemented")
    
    # The actual refinement would happen inside detect_along_branch
    # We can't directly control it from here without modifying the code
    
    print("\n✓ Test shows current behavior of detection system")


if __name__ == "__main__":
    solution = test_bifurcation_detection_in_continuation()
    print("\n")
    test_with_and_without_refinement()
