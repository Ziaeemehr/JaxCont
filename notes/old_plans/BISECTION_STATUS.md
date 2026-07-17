# Bifurcation Location Using Bisection - Status Report

## Current Status: ⚠️ **PARTIALLY IMPLEMENTED**

**Date**: November 14, 2025

## What "Integration with Continuation Workflow" Means

The bisection method for precise bifurcation location has two parts:

### 1. ✅ The Bisection Algorithm (IMPLEMENTED)
**Location**: `src/jaxcont/bifurcations/detector.py`, lines 135-260

The bisection algorithm itself IS implemented:
- Takes two continuation points (before and after bifurcation)
- Uses test function evaluation
- Iteratively narrows down the exact parameter value
- Returns refined bifurcation location

```python
def locate_bifurcation(
    self,
    solution: ContinuationSolution,
    index1: int,  # Point before bifurcation
    index2: int,  # Point after bifurcation
    bif_type: str,
    max_iterations: int = 20,
    tolerance: float = 1e-8
) -> Dict[str, Any]:
    # Bisection algorithm here...
```

### 2. ⚠️ The Integration (PARTIALLY WORKING)

**The issue**: The bisection is called in `detect_along_branch()` BUT:

#### What IS working:
```python
# In detector.py line 86-106
if self.detect_fold and self.fold_detector is not None:
    fold_points = self.fold_detector.detect(solution, eigenvalues)
    
    # This part calls bisection:
    if refine_location:
        for fold in fold_points:
            refined = self.locate_bifurcation(
                solution, idx1, idx2, 'fold',
                tolerance=self.tolerance
            )
```

#### What's MISSING:
1. **Test function evaluation at interpolated points**
   - Line 220-225: `_compute_eigenvalues_at_point()` is called but NOT IMPLEMENTED
   - Line 228-230: `_evaluate_fold_test()` and `_evaluate_hopf_test()` are called but NOT IMPLEMENTED
   
2. **Connection to the continuation problem**
   - The bisection needs to re-solve `f(u, p) = 0` at interpolated parameter values
   - Currently it only does LINEAR INTERPOLATION of state (line 212-213)
   - This is NOT accurate for nonlinear systems!

## The Key Missing Methods

### Missing Method 1: `_compute_eigenvalues_at_point()`
```python
def _compute_eigenvalues_at_point(self, u, p, solution):
    # NEEDED: Solve f(u, p) = 0 at parameter p
    # Then compute eigenvalues of Jacobian
    # Currently NOT IMPLEMENTED - falls back to linear interpolation
```

**Why it's needed**: During bisection, we need eigenvalues at intermediate parameter values. We can't just interpolate eigenvalues linearly - we need to:
1. Solve the equilibrium equation at the new parameter
2. Compute Jacobian at that equilibrium
3. Compute eigenvalues

### Missing Method 2: `_evaluate_fold_test()` and `_evaluate_hopf_test()`
```python
def _evaluate_fold_test(self, eigenvalues):
    # NEEDED: Evaluate the test function for fold:
    # g(p) = det(Jacobian) or smallest eigenvalue
    
def _evaluate_hopf_test(self, eigenvalues):
    # NEEDED: Evaluate test function for Hopf:
    # g(p) = Re(eigenvalue) crossing zero with Im != 0
```

**Why they're needed**: The bisection algorithm needs to evaluate test functions to determine which side of the midpoint contains the bifurcation.

## What Works vs What Doesn't

### ✅ Works:
1. **Detection of bifurcations** - finds the interval containing a bifurcation
2. **Calling bisection** - the workflow tries to call bisection if `refine_location=True`
3. **Bisection structure** - the algorithm skeleton is there
4. **Fallback to interpolation** - if test functions fail, returns simple average

### ❌ Doesn't Work:
1. **Accurate state at intermediate parameters** - uses linear interpolation instead of solving
2. **Test function evaluation** - methods not implemented, falls back
3. **High precision refinement** - can't converge without proper test functions
4. **Verification** - no tests to check if bisection is working

## Example: What Should Happen

For a fold bifurcation at p ≈ 1.0:

**Current behavior** (linear interpolation):
```
Step 50: p = 0.98, eigenvalue = 0.05  (positive)
Step 51: p = 1.02, eigenvalue = -0.03 (negative)

With refine_location=True:
→ Returns p = 1.00 (just the average)
→ Uses linearly interpolated state
→ No iteration, no refinement
```

**Desired behavior** (true bisection):
```
Step 50: p = 0.98, eigenvalue = 0.05
Step 51: p = 1.02, eigenvalue = -0.03

With refine_location=True:
Iteration 1: Try p = 1.00
  → Solve f(u, 1.00) = 0 to get u
  → Compute eigenvalue = 0.008
  → Sign same as left, narrow to [1.00, 1.02]
  
Iteration 2: Try p = 1.01
  → Solve f(u, 1.01) = 0 to get u
  → Compute eigenvalue = -0.012
  → Sign same as right, narrow to [1.00, 1.01]
  
... continue until |p_right - p_left| < tolerance

Final: p = 1.00134 ± 1e-8
```

## What Needs to Be Done

### Priority 1: Implement Missing Methods
1. **Implement `_compute_eigenvalues_at_point()`**
   - Needs access to the continuation problem
   - Use Newton solver to find equilibrium at parameter p
   - Compute Jacobian and eigenvalues

2. **Implement `_evaluate_fold_test()` and `_evaluate_hopf_test()`**
   - Fold: Return smallest real eigenvalue or determinant
   - Hopf: Return real part of complex conjugate pair

### Priority 2: Fix Architecture
**Problem**: `BifurcationDetector.locate_bifurcation()` needs access to `ContinuationProblem` to solve equilibrium equations, but currently only gets `ContinuationSolution`.

**Solution Options**:
1. Store problem reference in solution
2. Pass problem as additional argument
3. Use callback function for equilibrium solving

### Priority 3: Add Tests
Create tests that verify:
1. Bisection converges to correct value
2. Test functions are evaluated correctly
3. Refinement improves accuracy over linear interpolation

## Recommendation

The bisection skeleton is there, but it needs these three methods implemented:

```python
# In BifurcationDetector class:

def _compute_eigenvalues_at_point(
    self, u: Array, p: float, solution: ContinuationSolution
) -> Array:
    """Solve equilibrium at p and compute eigenvalues."""
    # TODO: Need access to problem to solve f(u, p) = 0
    # For now, this is the blocker
    pass

def _evaluate_fold_test(self, eigenvalues: Array) -> float:
    """Test function for fold: smallest real eigenvalue."""
    return jnp.min(jnp.real(eigenvalues))

def _evaluate_hopf_test(self, eigenvalues: Array) -> float:
    """Test function for Hopf: real part of complex pair."""
    # Find complex conjugate pair closest to imaginary axis
    complex_pairs = eigenvalues[jnp.abs(jnp.imag(eigenvalues)) > 1e-8]
    if len(complex_pairs) > 0:
        # Return real part of pair closest to Re=0
        idx = jnp.argmin(jnp.abs(jnp.real(complex_pairs)))
        return jnp.real(complex_pairs[idx])
    return jnp.min(jnp.real(eigenvalues))
```

## Bottom Line

✅ **Bisection algorithm structure**: DONE  
❌ **Test function evaluation**: MISSING  
❌ **Equilibrium solving at intermediate points**: MISSING  
❌ **Full integration**: NOT WORKING  

**Status**: Can be marked as "partially implemented" but not "complete and verified"
