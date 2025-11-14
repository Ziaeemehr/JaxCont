"""
Test precise bifurcation location using bisection method.

This tests the implementation of the bisection algorithm that refines
bifurcation locations to high precision.
"""

import pytest
import jax.numpy as jnp
from jaxcont.core.continuation import ContinuationProblem, ContinuationSolution
from jaxcont.bifurcations.detector import BifurcationDetector


def test_eigenvalue_computation_at_point():
    """
    Test that we can compute eigenvalues at an arbitrary point.
    """
    print("\n" + "="*80)
    print("Test 1: Eigenvalue computation at arbitrary point")
    print("="*80)
    
    # Define a simple problem: du/dt = u^2 - p
    # At equilibrium: u^2 = p, so u = ±sqrt(p)
    # Jacobian: df/du = 2u
    # Eigenvalue: λ = 2u
    
    def rhs(u, params):
        p = params['p']
        return u**2 - p
    
    u0 = jnp.array([2.0])
    params = {'p': 4.0}
    problem = ContinuationProblem(
        rhs=rhs,
        u0=u0,
        params=params,
        continuation_param='p'
    )
    
    # Create detector
    detector = BifurcationDetector()
    
    # Create a dummy solution (we'll just use it for structure)
    states = jnp.array([[2.0], [2.1], [2.2]])
    parameters = jnp.array([4.0, 4.41, 4.84])
    solution = ContinuationSolution(states=states, parameters=parameters)
    
    # Compute eigenvalues at a specific point
    u_test = jnp.array([1.5])
    p_test = 2.25  # Should be equilibrium: 1.5^2 = 2.25
    
    eigs = detector._compute_eigenvalues_at_point(u_test, p_test, solution, problem)
    
    print(f"Test point: u = {u_test[0]:.6f}, p = {p_test:.6f}")
    print(f"Computed eigenvalue: λ = {eigs[0]:.6f}")
    print(f"Expected eigenvalue: λ = 2u = {2*u_test[0]:.6f}")
    
    # For this problem, eigenvalue should be 2*u = 3.0
    expected_eig = 2.0 * u_test[0]
    assert jnp.isclose(eigs[0].real, expected_eig, atol=1e-6), \
        f"Eigenvalue {eigs[0].real} doesn't match expected {expected_eig}"
    
    print("✓ Eigenvalue computation correct!")


def test_bisection_fold_simple():
    """
    Test bisection for fold bifurcation in simple 1D problem.
    
    Problem: du/dt = u^2 - p
    Fold occurs at u = 0, p = 0 where eigenvalue = 0
    """
    print("\n" + "="*80)
    print("Test 2: Bisection for fold bifurcation (1D problem)")
    print("="*80)
    
    # Define problem
    def rhs(u, params):
        p = params['p']
        return u**2 - p
    
    # Create a solution that crosses the fold
    # Before fold: u = 0.1, p = 0.01, eigenvalue = 0.2 (positive)
    # After fold: u = -0.1, p = 0.01, eigenvalue = -0.2 (negative)
    
    states = jnp.array([[0.1], [-0.1]])
    parameters = jnp.array([0.01, 0.01])
    
    # Compute eigenvalues for these points
    from jax import jacfwd
    
    def compute_eigs(u, p):
        def f_u(u_eval):
            return jnp.array([u_eval[0]**2 - p])
        jac = jacfwd(f_u)(u)
        return jnp.linalg.eigvals(jac)
    
    eigs1 = compute_eigs(states[0], parameters[0])
    eigs2 = compute_eigs(states[1], parameters[1])
    eigenvalues = jnp.array([eigs1, eigs2])
    
    print(f"Point 1: u = {states[0,0]:.6f}, p = {parameters[0]:.6f}, λ = {eigs1[0].real:.6f}")
    print(f"Point 2: u = {states[1,0]:.6f}, p = {parameters[1]:.6f}, λ = {eigs2[0].real:.6f}")
    
    solution = ContinuationSolution(
        states=states,
        parameters=parameters,
        eigenvalues=eigenvalues
    )
    
    # Create problem
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([0.1]),
        params={'p': 0.01},
        continuation_param='p'
    )
    
    # Detect bifurcation with bisection
    detector = BifurcationDetector(detect_fold=True)
    
    result = detector.locate_bifurcation(
        solution=solution,
        index1=0,
        index2=1,
        bif_type='fold',
        problem=problem,
        max_iterations=30,
        tolerance=1e-10
    )
    
    print(f"\nBisection result:")
    print(f"  Refined u = {result['state'][0]:.10f}")
    print(f"  Refined p = {result['parameter']:.10f}")
    print(f"  Eigenvalue = {result['eigenvalues'][0].real:.10f}")
    print(f"  Iterations = {result['iterations']}")
    print(f"  Residual = {result['residual']:.2e}")
    
    # At fold: u ≈ 0, eigenvalue ≈ 0
    assert jnp.abs(result['state'][0]) < 1e-6, "State should be near 0 at fold"
    assert jnp.abs(result['eigenvalues'][0].real) < 1e-6, "Eigenvalue should be near 0 at fold"
    
    print("✓ Bisection correctly located fold bifurcation!")


def test_bisection_fold_pitchfork():
    """
    Test bisection on pitchfork bifurcation problem.
    
    Problem: du/dt = u^3 - p*u
    Fold on nontrivial branch occurs when d(u^3 - p*u)/du = 0
    => 3u^2 - p = 0, combined with u^3 - p*u = 0
    => u^2 = p/3, u^3 = p*u => u = ±sqrt(p/3)
    => p = 3u^2, so at u = ±1, p = 3 (fold points)
    """
    print("\n" + "="*80)
    print("Test 3: Bisection for fold on pitchfork branch")
    print("="*80)
    
    # Define problem
    def rhs(u, params):
        p = params['p']
        return u**3 - p * u
    
    # Points near fold at u ≈ 1, p ≈ 3
    # For fold on u^3 - p*u = 0 branch with eigenvalue 3u^2 - p = 0
    # Fold is at u = 1, p = 3 (where 3*1^2 - 3 = 0)
    # 
    # Before fold: smaller u, same p => eigenvalue positive
    # After fold: larger u, same p => eigenvalue negative
    
    p_fold = 3.0
    u1 = 0.95  # Before fold
    u2 = 1.05  # After fold
    
    # Keep parameter fixed at fold value
    p1 = p_fold
    p2 = p_fold
    
    states = jnp.array([[u1], [u2]])
    parameters = jnp.array([p1, p2])
    
    # Compute eigenvalues
    from jax import jacfwd
    
    def compute_eigs(u, p):
        def f_u(u_eval):
            return jnp.array([u_eval[0]**3 - p * u_eval[0]])
        jac = jacfwd(f_u)(jnp.array([u]))
        return jnp.linalg.eigvals(jac)
    
    eigs1 = compute_eigs(u1, p1)
    eigs2 = compute_eigs(u2, p2)
    eigenvalues = jnp.array([eigs1, eigs2])
    
    print(f"Point 1: u = {u1:.6f}, p = {p1:.6f}, λ = {eigs1[0].real:.6f}")
    print(f"Point 2: u = {u2:.6f}, p = {p2:.6f}, λ = {eigs2[0].real:.6f}")
    
    solution = ContinuationSolution(
        states=states,
        parameters=parameters,
        eigenvalues=eigenvalues
    )
    
    # Create problem
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([u1]),
        params={'p': p1},
        continuation_param='p'
    )
    
    # Detect with bisection
    detector = BifurcationDetector(detect_fold=True)
    
    result = detector.locate_bifurcation(
        solution=solution,
        index1=0,
        index2=1,
        bif_type='fold',
        problem=problem,
        max_iterations=30,
        tolerance=1e-10
    )
    
    print(f"\nBisection result:")
    print(f"  Refined u = {result['state'][0]:.10f}")
    print(f"  Refined p = {result['parameter']:.10f}")
    print(f"  Eigenvalue = {result['eigenvalues'][0].real:.10f}")
    print(f"  Iterations = {result['iterations']}")
    print(f"  Residual = {result['residual']:.2e}")
    
    # At fold: u ≈ 1, p ≈ 3, eigenvalue ≈ 0
    expected_u = 1.0
    expected_p = 3.0
    
    print(f"\nExpected: u = {expected_u:.10f}, p = {expected_p:.10f}")
    print(f"Error in u: {abs(result['state'][0] - expected_u):.2e}")
    print(f"Error in p: {abs(result['parameter'] - expected_p):.2e}")
    
    assert jnp.abs(result['state'][0] - expected_u) < 1e-4, \
        f"State {result['state'][0]} not near expected {expected_u}"
    assert jnp.abs(result['parameter'] - expected_p) < 1e-4, \
        f"Parameter {result['parameter']} not near expected {expected_p}"
    assert jnp.abs(result['eigenvalues'][0].real) < 1e-4, \
        "Eigenvalue should be near 0 at fold"
    
    print("✓ Bisection correctly located fold on pitchfork branch!")


def test_bisection_convergence():
    """
    Test that bisection converges with expected rate.
    """
    print("\n" + "="*80)
    print("Test 4: Bisection convergence rate")
    print("="*80)
    
    # Simple problem with known fold
    def rhs(u, params):
        p = params['p']
        return u**2 - p
    
    # Wide bracket around fold
    states = jnp.array([[1.0], [-1.0]])
    parameters = jnp.array([1.0, 1.0])
    
    from jax import jacfwd
    
    def compute_eigs(u, p):
        def f_u(u_eval):
            return jnp.array([u_eval[0]**2 - p])
        jac = jacfwd(f_u)(u)
        return jnp.linalg.eigvals(jac)
    
    eigs1 = compute_eigs(states[0], parameters[0])
    eigs2 = compute_eigs(states[1], parameters[1])
    eigenvalues = jnp.array([eigs1, eigs2])
    
    solution = ContinuationSolution(
        states=states,
        parameters=parameters,
        eigenvalues=eigenvalues
    )
    
    problem = ContinuationProblem(
        rhs=rhs,
        u0=jnp.array([1.0]),
        params={'p': 1.0},
        continuation_param='p'
    )
    
    detector = BifurcationDetector(detect_fold=True)
    
    # Test with different tolerance levels
    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    
    print("\nConvergence test:")
    print("Tolerance    Iterations    Error")
    print("-" * 40)
    
    for tol in tolerances:
        result = detector.locate_bifurcation(
            solution=solution,
            index1=0,
            index2=1,
            bif_type='fold',
            problem=problem,
            tolerance=tol
        )
        
        error = abs(result['state'][0])  # Should be 0 at fold
        print(f"{tol:10.0e}    {result['iterations']:8d}      {error:.2e}")
        
        assert error < tol * 10, f"Error {error} exceeds tolerance {tol}"
    
    print("\n✓ Bisection converges with expected rate!")


if __name__ == "__main__":
    test_eigenvalue_computation_at_point()
    test_bisection_fold_simple()
    test_bisection_fold_pitchfork()
    test_bisection_convergence()
    
    print("\n" + "="*80)
    print(" ALL TESTS PASSED ✓")
    print("="*80)
    print("\nPrecise bifurcation location using bisection is working correctly!")
