# JaxCont Profiling Results and Optimization Opportunities

**Date:** November 14, 2025  
**Status:** Initial profiling complete

## Executive Summary

Profiling identified several key bottlenecks and optimization opportunities:

1. **No JIT compilation** in critical continuation paths
2. **Repeated Jacobian computations** without caching
3. **Sequential operations** that could be vectorized with `vmap`
4. **Eigenvalue computation** overhead in stability analysis

## Detailed Findings

### 1. JIT Compilation Status ❌

**Current State:**
```
pseudo_arclength module:
  - JIT compiled: None
  - Not JIT compiled: jacobian, jacfwd

predictor_corrector module:
  - JIT compiled: None
  - Not JIT compiled: abstractmethod

newton solver module:
  - JIT compiled: jit (only the function named 'jit')
  - Not JIT compiled: jacfwd
```

**Impact:** Functions are being recompiled on every call, causing significant overhead.

### 2. Performance Metrics

#### Basic Operations (1D system)
| Operation | Size | Avg Time (ms) | First Compile (ms) |
|-----------|------|---------------|-------------------|
| Jacobian | 1 | 4.03 | 363.53 |
| Jacobian | 10 | 3.75 | 415.19 |
| Linear Solve | 10 | 0.033 | 91.23 |
| Linear Solve | 100 | 0.151 | 109.43 |
| Eigenvalues | 10 | 0.054 | 35.95 |
| Eigenvalues | 50 | 0.550 | 40.26 |

**Key Insight:** Compilation overhead is 100-1000x the execution time! Once compiled, operations are extremely fast.

#### Continuation Performance
| System | Dimension | Points | Total Time (s) | Time/Point (ms) | Avg Newton Iters |
|--------|-----------|--------|----------------|-----------------|------------------|
| Pitchfork | 1D | 12 | 0.32 | 26.96 | 0.18 |
| Lorenz | 3D | 51 | 1.67 | 32.80 | 0.08 |

### 3. Hot Spots (from cProfile)

Most time-consuming functions (total time):
1. **jacfun (0.191s, 68%)** - JAX Jacobian computation
2. **evaluate_rhs (0.134s, 48%)** - RHS function evaluation
3. **compute_tangent (0.107s, 38%)** - Tangent vector computation
4. **_compute_eigenvalues (0.089s, 32%)** - Eigenvalue calculation

### 4. Specific Issues Identified

#### Issue 1: No JIT on Critical Path
```python
# Current: No JIT decoration
def compute_tangent(self, ...):
    J = jax.jacfwd(...)  # Recompiled every call
    ...

# Should be:
@jax.jit
def compute_tangent(self, ...):
    ...
```

#### Issue 2: Repeated Jacobian Computations
The Jacobian is computed multiple times per continuation step:
- Once in predictor
- Once (or more) in corrector/Newton iterations
- Once for tangent computation

**Opportunity:** Cache and reuse Jacobian computations.

#### Issue 3: Sequential Operations
Many operations process points sequentially that could be batched:
```python
# Current: Sequential
for i in range(n_points):
    eigenvalues[i] = compute_eigenvalues(jacobian[i])

# Could be: Vectorized with vmap
eigenvalues = jax.vmap(compute_eigenvalues)(jacobians)
```

#### Issue 4: No Pre-compilation
Cold start overhead is significant. Functions should be pre-compiled during initialization.

## Optimization Opportunities

### Priority 1: Add JIT Compilation (High Impact, Low Effort) 🚀

**Files to modify:**
1. `src/jaxcont/core/predictor_corrector.py`
   - Add `@jax.jit` to `_newton_step()`
   - Add `@jax.jit` to `_compute_eigenvalues()`
   - Add `@jax.jit` to helper functions

2. `src/jaxcont/core/pseudo_arclength.py`
   - Add `@jax.jit` to `compute_tangent()`
   - Add `@jax.jit` to `_predict()`
   - Add `@jax.jit` to `_correct()`

3. `src/jaxcont/solvers/newton.py`
   - Add `@jax.jit` to `solve()`

**Expected speedup:** 2-10x for hot paths

### Priority 2: Use vmap for Vectorization (Medium Impact, Medium Effort) 📊

**Opportunities:**
1. Batch eigenvalue computations along a branch
2. Vectorize stability analysis
3. Batch bifurcation test function evaluations

**Example:**
```python
# Instead of loop
eigenvalues = []
for state in states:
    eigenvalues.append(compute_eig(state))

# Use vmap
compute_eig_batch = jax.vmap(compute_eig)
eigenvalues = compute_eig_batch(states)
```

**Expected speedup:** 2-5x for batch operations

### Priority 3: Jacobian Caching (Medium Impact, Medium Effort) 💾

**Strategy:**
- Cache Jacobian from corrector step
- Reuse in tangent computation
- Invalidate only when solution changes

**Expected speedup:** 1.5-2x reduction in Jacobian calls

### Priority 4: Pre-compilation Strategy (Low Impact, Low Effort) ⚡

**Implementation:**
```python
class PseudoArclengthContinuation:
    def __init__(self, ...):
        ...
        self._warm_up()
    
    def _warm_up(self):
        """Pre-compile critical functions."""
        # Dummy call to trigger compilation
        u_dummy = jnp.ones(1)
        params_dummy = {'param': 0.0}
        self._newton_step(u_dummy, params_dummy, ...)
```

**Expected benefit:** Faster first run, better user experience

### Priority 5: GPU Acceleration (High Impact, High Effort) 🎮

**When beneficial:**
- Large systems (n > 100)
- Many continuation points
- Expensive eigenvalue computations

**Implementation:**
- Already using JAX - just need CUDA setup
- Use `jax.device_put()` for explicit GPU placement
- Profile GPU vs CPU for different system sizes

**Expected speedup:** 10-100x for large systems

## Recommended Implementation Order

### Phase 1: Quick Wins (Week 1)
- [ ] Add `@jax.jit` to all hot path functions
- [ ] Add pre-compilation in `__init__`
- [ ] Profile again to measure improvement

### Phase 2: Vectorization (Week 2)
- [ ] Implement vmap for eigenvalue computations
- [ ] Vectorize stability analysis
- [ ] Add batch bifurcation detection

### Phase 3: Advanced Optimization (Week 3-4)
- [ ] Implement Jacobian caching
- [ ] Add GPU support
- [ ] Optimize memory allocation

## Benchmarking Plan

Create benchmark suite with:
1. Small systems (n=1-3) - test overhead reduction
2. Medium systems (n=10-50) - test overall performance
3. Large systems (n=100-1000) - test scalability
4. Long continuations (1000+ points) - test memory efficiency

Target metrics:
- 5-10x speedup from JIT compilation
- 2-3x speedup from vmap
- 2x speedup from Jacobian caching
- **Overall target: 10-20x speedup**

## Next Steps

1. Start with Priority 1 (JIT compilation)
2. Create minimal test case to verify improvements
3. Profile after each optimization
4. Document performance improvements
5. Update benchmarks

## Notes

- Current implementation is correct but not optimized
- JAX provides most tools needed - just need to use them
- Focus on hot paths first (80/20 rule)
- GPU acceleration should wait until CPU code is optimized
