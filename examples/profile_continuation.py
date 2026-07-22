"""
Profile JaxCont continuation and solver performance.

This script profiles the most time-consuming parts:
- Continuation loop
- Newton solver iterations
- Jacobian computations
- Linear solves
- Eigenvalue computations
"""

import jax
import jax.numpy as jnp
import time
import cProfile
import pstats
import io
from functools import wraps

import jaxcont as jc


# ============================================================================
# Test Problems
# ============================================================================

def pitchfork_rhs(u, p, args):
    """Pitchfork bifurcation: du/dt = r*u - u^3"""
    return p * u - u ** 3


def lorenz_rhs(u, p, args):
    """Lorenz system"""
    sigma, beta = args
    rho = p
    x, y, z = u[0], u[1], u[2]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return jnp.array([dx, dy, dz])


def van_der_pol_rhs(u, params):
    """Van der Pol oscillator"""
    mu = params['mu']
    x, y = u[0], u[1]
    
    dx = y
    dy = mu * (1 - x**2) * y - x
    
    return jnp.array([dx, dy])


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        

class ProfilingStats:
    """Collect profiling statistics."""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
        
    def add_timing(self, name, elapsed):
        """Add a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        self.timings[name].append(elapsed)
        self.counts[name] += 1
        
    def get_stats(self):
        """Get statistics for all timings."""
        stats = {}
        for name, times in self.timings.items():
            times_array = jnp.array(times)
            stats[name] = {
                'count': self.counts[name],
                'total': float(jnp.sum(times_array)),
                'mean': float(jnp.mean(times_array)),
                'std': float(jnp.std(times_array)),
                'min': float(jnp.min(times_array)),
                'max': float(jnp.max(times_array)),
            }
        return stats
        
    def print_report(self):
        """Print a formatted profiling report."""
        stats = self.get_stats()
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print("\n" + "="*80)
        print("PROFILING REPORT")
        print("="*80)
        
        total_time = sum(s['total'] for s in stats.values())
        
        print(f"\n{'Operation':<40} {'Count':>8} {'Total (s)':>12} {'Mean (ms)':>12} {'%':>8}")
        print("-"*80)
        
        for name, stat in sorted_stats:
            percentage = (stat['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<40} {stat['count']:>8} {stat['total']:>12.4f} "
                  f"{stat['mean']*1000:>12.4f} {percentage:>7.1f}%")
        
        print("-"*80)
        print(f"{'TOTAL':<40} {'':<8} {total_time:>12.4f} {'':<12} {100.0:>7.1f}%")
        print("="*80 + "\n")


# ============================================================================
# Profiling Functions
# ============================================================================

def profile_simple_continuation():
    """Profile a simple 1D continuation problem."""
    print("\n" + "=" * 80)
    print("PROFILING: Simple 1D Pitchfork Bifurcation")
    print("=" * 80)

    prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)
    settings = jc.ContinuationPar(
        ds=0.05, max_steps=100, adaptive=True, compute_stability=True,
    )

    print("Warming up JAX (first run compiles)...")
    _ = jc.continuation(prob, jc.PseudoArclength(), p_span=(0.5, 1.0), settings=settings,
                         events=[jc.Fold(), jc.Hopf()])

    print("Running profiled continuation...")
    start = time.perf_counter()
    result = jc.continuation(prob, jc.PseudoArclength(), p_span=(0.5, 1.5), settings=settings,
                              events=[jc.Fold(), jc.Hopf()])
    elapsed = time.perf_counter() - start

    n = result.branch.n_valid
    print(f"\nTotal continuation time: {elapsed:.4f} seconds")
    print(f"Number of points computed: {n}")
    print(f"Time per point: {elapsed / n * 1000:.2f} ms")

    # NOTE: the scan engine's convergence_info hardcodes newton_iters=0 (a
    # pre-existing limitation -- per-point Newton iteration counts aren't
    # tracked by pseudo_arclength_scan). This line is kept for structural
    # parity with the pre-migration profiling report but will always print 0.
    newton_iters = [info["newton_iters"] for info in result._solution.convergence_info[:n]]
    print(f"Average Newton iterations: {jnp.mean(jnp.array(newton_iters)):.2f}")

    return result, elapsed


def profile_3d_continuation():
    """Profile a 3D Lorenz system continuation."""
    print("\n" + "=" * 80)
    print("PROFILING: 3D Lorenz System")
    print("=" * 80)

    prob = jc.bif_problem(
        lorenz_rhs, u0=jnp.array([1.0, 1.0, 1.0]), p0=20.0, args=(10.0, 8.0 / 3.0),
    )
    settings = jc.ContinuationPar(
        ds=0.1, max_steps=50, adaptive=True, compute_stability=True, newton_max_iter=20,
    )

    print("Warming up JAX (first run compiles)...")
    _ = jc.continuation(prob, jc.PseudoArclength(), p_span=(20.0, 22.0), settings=settings)

    print("Running profiled continuation...")
    start = time.perf_counter()
    result = jc.continuation(prob, jc.PseudoArclength(), p_span=(20.0, 25.0), settings=settings)
    elapsed = time.perf_counter() - start

    n = result.branch.n_valid
    print(f"\nTotal continuation time: {elapsed:.4f} seconds")
    print(f"Number of points computed: {n}")
    print(f"Time per point: {elapsed / n * 1000:.2f} ms")

    newton_iters = [info["newton_iters"] for info in result._solution.convergence_info[:n]]
    print(f"Average Newton iterations: {jnp.mean(jnp.array(newton_iters)):.2f}")

    return result, elapsed


def profile_with_cprofile(func, *args, **kwargs):
    """Profile a function using cProfile."""
    print("\n" + "="*80)
    print("DETAILED cProfile ANALYSIS")
    print("="*80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    print(s.getvalue())
    
    return result


def profile_jacobian_computation():
    """Profile Jacobian computation specifically."""
    print("\n" + "="*80)
    print("PROFILING: Jacobian Computation")
    print("="*80)
    
    # Test with different system sizes
    sizes = [1, 2, 3, 5, 10]
    
    print(f"\n{'Size':>6} {'Time (ms)':>12} {'Compiled Time (ms)':>20}")
    print("-"*40)
    
    for n in sizes:
        # Create a simple n-dimensional system
        def rhs_n(u, params):
            r = params['r']
            return r * u - u**3
        
        u = jnp.ones(n) * 0.1
        params = {'r': 0.5}
        
        # Compute Jacobian using JAX
        jac_func = jax.jacfwd(lambda u: rhs_n(u, params))
        
        # First call (compilation)
        start = time.perf_counter()
        jac = jac_func(u)
        compile_time = (time.perf_counter() - start) * 1000
        
        # Subsequent calls (compiled)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            jac = jac_func(u)
            jac.block_until_ready()  # Wait for computation
            times.append(time.perf_counter() - start)
        
        avg_time = jnp.mean(jnp.array(times)) * 1000
        
        print(f"{n:>6} {avg_time:>12.4f} {compile_time:>20.4f}")


def profile_linear_solve():
    """Profile linear solve operations."""
    print("\n" + "="*80)
    print("PROFILING: Linear Solve")
    print("="*80)
    
    sizes = [2, 5, 10, 20, 50, 100]
    
    print(f"\n{'Size':>6} {'Time (ms)':>12} {'Compiled Time (ms)':>20}")
    print("-"*40)
    
    for n in sizes:
        # Create random linear system
        A = jnp.eye(n) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n, n))
        b = jax.random.normal(jax.random.PRNGKey(1), (n,))
        
        # First solve (compilation)
        start = time.perf_counter()
        x = jnp.linalg.solve(A, b)
        x.block_until_ready()
        compile_time = (time.perf_counter() - start) * 1000
        
        # Subsequent solves (compiled)
        times = []
        for i in range(100):
            b_i = jax.random.normal(jax.random.PRNGKey(i+10), (n,))
            start = time.perf_counter()
            x = jnp.linalg.solve(A, b_i)
            x.block_until_ready()
            times.append(time.perf_counter() - start)
        
        avg_time = jnp.mean(jnp.array(times)) * 1000
        
        print(f"{n:>6} {avg_time:>12.4f} {compile_time:>20.4f}")


def profile_eigenvalue_computation():
    """Profile eigenvalue computation."""
    print("\n" + "="*80)
    print("PROFILING: Eigenvalue Computation")
    print("="*80)
    
    sizes = [2, 3, 5, 10, 20, 50]
    
    print(f"\n{'Size':>6} {'Time (ms)':>12} {'Compiled Time (ms)':>20}")
    print("-"*40)
    
    for n in sizes:
        # Create random matrix
        A = jnp.eye(n) + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (n, n))
        A = (A + A.T) / 2  # Make symmetric for stability
        
        # First computation (compilation)
        start = time.perf_counter()
        eigvals = jnp.linalg.eigvals(A)
        eigvals.block_until_ready()
        compile_time = (time.perf_counter() - start) * 1000
        
        # Subsequent computations (compiled)
        times = []
        for i in range(50):
            A_i = A + 0.01 * jax.random.normal(jax.random.PRNGKey(i+10), (n, n))
            A_i = (A_i + A_i.T) / 2
            start = time.perf_counter()
            eigvals = jnp.linalg.eigvals(A_i)
            eigvals.block_until_ready()
            times.append(time.perf_counter() - start)
        
        avg_time = jnp.mean(jnp.array(times)) * 1000
        
        print(f"{n:>6} {avg_time:>12.4f} {compile_time:>20.4f}")


def check_jit_usage():
    """Check which functions are currently JIT compiled."""
    print("\n" + "="*80)
    print("JIT COMPILATION STATUS")
    print("="*80)
    
    from jaxcont.core import scan_continuation
    from jaxcont.solvers import newton
    import inspect

    modules_to_check = [
        ('scan_continuation', scan_continuation),
        ('newton', newton),
    ]
    
    print("\nChecking for @jit decorators in key modules...")
    
    for name, module in modules_to_check:
        print(f"\n{name}:")
        members = inspect.getmembers(module, inspect.isfunction)
        
        jitted = []
        not_jitted = []
        
        for func_name, func in members:
            if func_name.startswith('_'):
                continue
            source = inspect.getsource(func)
            if '@jit' in source or 'jax.jit' in source:
                jitted.append(func_name)
            else:
                not_jitted.append(func_name)
        
        if jitted:
            print(f"  JIT compiled: {', '.join(jitted)}")
        else:
            print(f"  JIT compiled: None")
        
        if not_jitted:
            print(f"  Not JIT compiled: {', '.join(not_jitted)}")


# ============================================================================
# Main Profiling Script
# ============================================================================

def main():
    """Run all profiling tests."""
    
    print("\n" + "="*80)
    print(" JaxCont Performance Profiling")
    print("="*80)
    print("\nThis script profiles the key performance bottlenecks:")
    print("  1. Continuation loop")
    print("  2. Newton solver")
    print("  3. Jacobian computation")
    print("  4. Linear solves")
    print("  5. Eigenvalue computation")
    print("  6. JIT compilation status")
    
    # Check JIT usage first
    check_jit_usage()
    
    # Profile basic operations
    profile_jacobian_computation()
    profile_linear_solve()
    profile_eigenvalue_computation()
    
    # Profile continuation runs
    sol1, time1 = profile_simple_continuation()
    sol3, time3 = profile_3d_continuation()
    
    # Detailed profiling with cProfile
    print("\n" + "="*80)
    print("DETAILED PROFILING (cProfile)")
    print("="*80)
    print("\nRunning 1D continuation with detailed profiling...")
    
    prob = jc.bif_problem(pitchfork_rhs, u0=jnp.array([0.1]), p0=0.5)
    settings = jc.ContinuationPar(ds=0.05, max_steps=100, adaptive=True, compute_stability=True)

    # Warm up first
    _ = jc.continuation(prob, jc.PseudoArclength(), p_span=(0.5, 1.0), settings=settings,
                         events=[jc.Fold(), jc.Hopf()])

    # Profile
    profile_with_cprofile(
        jc.continuation, prob, jc.PseudoArclength(), p_span=(0.5, 1.5), settings=settings,
        events=[jc.Fold(), jc.Hopf()],
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    print("\nKey findings:")
    print("  - Check above for functions without @jit decorators")
    print("  - Compare compilation time vs. execution time")
    print("  - Look for repeated operations that could be JIT compiled")
    print("  - Consider using vmap for batch operations")
    print("\nNext steps:")
    print("  1. Add @jit to frequently called functions")
    print("  2. Use vmap for vectorizable operations")
    print("  3. Pre-compile hot paths")
    print("  4. Consider GPU acceleration for large systems")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
