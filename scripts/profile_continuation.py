"""Developer diagnostic: synchronized continuation timing and host cProfile.

The first call includes tracing/compilation. Host cProfile measures Python
work and dispatch, not time spent inside compiled XLA kernels.
"""
import cProfile
import pstats
import time

import jax
import jax.numpy as jnp
import jaxcont as jc


def rhs(u, p, args):
    return jnp.array([p + u[0] - u[0]**3 / 3])


def main():
    problem = jc.bif_problem(rhs, u0=jnp.array([-2.1038034027355366]), p0=-1.)
    settings = jc.ContinuationPar(ds=0.01, max_steps=300, newton_tol=1e-6)

    def run():
        result = jc.continuation(problem, p_span=(-1., 1.), settings=settings)
        jax.block_until_ready(result)
        return result

    start = time.perf_counter()
    run()
    print(f'First call (includes compilation): {time.perf_counter() - start:.6f} s')
    profiler = cProfile.Profile()
    start = time.perf_counter()
    result = profiler.runcall(run)
    print(f'Warm synchronized call: {time.perf_counter() - start:.6f} s')
    print(f'Valid branch points: {result.branch.n_valid}')
    print('Host cProfile (Python and dispatch; not compiled-kernel attribution)')
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(20)


if __name__ == '__main__':
    main()
