"""
Pseudo-arclength continuation (most robust method).
"""

from functools import partial
from typing import Callable, Tuple, Optional
import jax.numpy as jnp
from jax import Array, jacfwd, jit, lax

from jaxcont.core.predictor_corrector import PredictorCorrector
from jaxcont.core.continuation import ContinuationProblem
from jaxcont.solvers.newton import NewtonSolver


@partial(jit, static_argnums=(0, 8, 9))
def _correct_jit(
    f: Callable[[Array, Array], Array],
    u_pred: Array,
    p_pred: Array,
    u_prev: Array,
    p_prev: Array,
    du0: Array,
    dp0: Array,
    ds: Array,
    tol: float,
    max_iter: int,
) -> Tuple[Array, Array, Array, Array]:
    """
    JIT-compiled pseudo-arclength corrector.

    Solves the full bordered Newton system at each iteration:

        [ df/du    df/dp ] [ Delta_u ]   [ -f(u, p) ]
        [ du0^T    dp0    ] [ Delta_p ] = [ -g(u, p) ]

    Solving the *full* (n+1)x(n+1) bordered system (rather than eliminating
    through df/du) keeps the solve well-posed at fold points, where df/du
    itself is singular. ``df/dp`` is obtained by autodiff, not finite
    differences. The Newton loop is a ``lax.while_loop`` so the whole
    corrector compiles to a single dispatched kernel.

    Args:
        f: Pure RHS f(u, p) -> residual, closed over the fixed parameters.
        u_pred, p_pred: Predicted state / parameter (Newton initial guess).
        u_prev, p_prev: Previous accepted point (anchors the arclength plane).
        du0, dp0: Tangent components [du0, dp0].
        ds: Arclength step.
        tol: Convergence tolerance on the combined residual norm.
        max_iter: Maximum Newton iterations.

    Returns:
        (u, p, converged, n_iter) with u/p JAX arrays and converged a bool array.
    """

    def residual(u, p):
        f_val = f(u, p)
        g_val = jnp.dot(u - u_prev, du0) + (p - p_prev) * dp0 - ds
        return f_val, g_val

    def res_norm(f_val, g_val):
        return jnp.sqrt(jnp.sum(f_val ** 2) + g_val ** 2)

    def cond_fun(carry):
        _, _, iteration, converged, _ = carry
        return jnp.logical_and(jnp.logical_not(converged), iteration < max_iter)

    def newton_step(carry):
        u, p, iteration, _, _ = carry

        f_val, g_val = residual(u, p)

        # Jacobian blocks via autodiff.
        jac_u = jacfwd(f, argnums=0)(u, p)      # (n, n)
        df_dp = jacfwd(f, argnums=1)(u, p)      # (n,)

        # Assemble and solve the full bordered system.
        top = jnp.concatenate([jac_u, df_dp[:, None]], axis=1)          # (n, n+1)
        bottom = jnp.concatenate([du0, jnp.reshape(dp0, (1,))])[None, :]  # (1, n+1)
        bordered = jnp.concatenate([top, bottom], axis=0)               # (n+1, n+1)
        rhs = -jnp.concatenate([f_val, jnp.reshape(g_val, (1,))])       # (n+1,)

        delta = jnp.linalg.solve(bordered, rhs)
        delta_u = delta[:-1]
        delta_p = delta[-1]

        u_new = u + delta_u
        p_new = p + delta_p

        f_new, g_new = residual(u_new, p_new)
        residual_new = res_norm(f_new, g_new)
        converged_new = residual_new < tol

        return u_new, p_new, iteration + 1, converged_new, residual_new

    f0, g0 = residual(u_pred, p_pred)
    res0 = res_norm(f0, g0)
    init = (u_pred, p_pred, 0, res0 < tol, res0)

    u_f, p_f, iter_f, conv_f, _ = lax.while_loop(cond_fun, newton_step, init)
    return u_f, p_f, conv_f, iter_f


class PseudoArclengthContinuation(PredictorCorrector):
    """
    Pseudo-arclength continuation method.
    
    This is the most robust continuation method that can pass turning points
    (fold bifurcations) by parametrizing the solution branch by arclength
    rather than by a single parameter.
    
    The method solves:
        f(u, p) = 0
        g(u, p) = (u - u0)^T * du0 + (p - p0) * dp0 - ds = 0
    
    where (du0, dp0) is the tangent vector and ds is the arclength step.
    """
    
    def predict(
        self,
        u: Array,
        param: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float]:
        """
        Predict next point by stepping along the tangent.
        
        Args:
            u: Current state
            param: Current parameter value
            tangent: Tangent vector [du, dp]
            ds: Step size in arclength
        
        Returns:
            (u + ds*du, param + ds*dp) tuple
        """
        # Split tangent into state and parameter components
        du = tangent[:-1]
        dp = tangent[-1]
        
        u_pred = u + ds * du
        param_pred = param + ds * dp
        
        return u_pred, param_pred
    
    def _get_rhs_fn(self, problem: ContinuationProblem) -> Callable[[Array, Array], Array]:
        """
        Return (and cache) a pure ``f(u, p)`` closed over the problem's fixed
        parameters, with the continuation parameter exposed as ``p``.

        The function identity is stable per problem, so ``_correct_jit`` (which
        takes ``f`` as a static argument) compiles once and is reused across all
        continuation steps rather than recompiling each call.
        """
        cache = self.__dict__.setdefault("_rhs_fn_cache", {})
        key = id(problem)
        fn = cache.get(key)
        if fn is None:
            base_params = dict(problem.params)
            cont_param = problem.continuation_param
            rhs = problem.rhs

            def f(u: Array, p: Array) -> Array:
                params = dict(base_params)
                params[cont_param] = p
                return rhs(u, params)

            cache[key] = f
            fn = f
        return fn

    def correct(
        self,
        problem: ContinuationProblem,
        u_pred: Array,
        param_pred: float,
        u_prev: Array,
        param_prev: float,
        tangent: Array,
        ds: float
    ) -> Tuple[Array, float, bool, int]:
        """
        Correct prediction using the JIT-compiled bordered Newton corrector.

        Delegates the inner Newton loop to :func:`_correct_jit`, which solves the
        full bordered system with autodiff Jacobians inside a ``lax.while_loop``.

        Args:
            problem: Continuation problem
            u_pred: Predicted state
            param_pred: Predicted parameter
            u_prev: Previous state
            param_prev: Previous parameter
            tangent: Tangent vector [du0, dp0]
            ds: Step size

        Returns:
            (corrected_state, corrected_param, converged, iterations) tuple
        """
        f = self._get_rhs_fn(problem)
        du0 = tangent[:-1]
        dp0 = tangent[-1]

        u_new, p_new, converged, n_iter = _correct_jit(
            f,
            u_pred,
            jnp.asarray(param_pred, dtype=u_pred.dtype),
            u_prev,
            jnp.asarray(param_prev, dtype=u_pred.dtype),
            du0,
            jnp.asarray(dp0, dtype=u_pred.dtype),
            jnp.asarray(ds, dtype=u_pred.dtype),
            self.newton_tol,
            self.newton_max_iter,
        )

        return u_new, float(p_new), bool(converged), int(n_iter)
    
    def compute_tangent(
        self,
        problem: ContinuationProblem,
        u: Array,
        param: float,
        prev_tangent: Optional[Array] = None
    ) -> Array:
        """
        Compute tangent vector to the solution branch.
        
        The tangent satisfies:
            df/du * du + df/dp * dp = 0
            ||[du, dp]|| = 1
        
        Args:
            problem: Continuation problem
            u: Current state
            param: Current parameter value
            prev_tangent: Previous tangent (for orientation)
        
        Returns:
            Normalized tangent vector [du, dp]
        """
        # Jacobian blocks via autodiff (df/du and df/dp).
        f = self._get_rhs_fn(problem)
        param_arr = jnp.asarray(param, dtype=u.dtype)
        jac_u = jacfwd(f, argnums=0)(u, param_arr)
        df_dp = jacfwd(f, argnums=1)(u, param_arr)
        
        # We need to find [du, dp] such that:
        # jac_u * du + df_dp * dp = 0 and ||[du, dp]|| = 1
        
        # From the first equation: du = -jac_u^{-1} * df_dp * dp
        # Let's set dp = 1 initially and normalize
        try:
            du_unnorm = -jnp.linalg.solve(jac_u, df_dp)
            dp_unnorm = 1.0
        except:
            # If singular, try dp = 0 and du along null space
            # For simplicity, use random direction
            du_unnorm = jnp.ones_like(u)
            dp_unnorm = 0.0
        
        # Combine and normalize
        tangent = jnp.concatenate([du_unnorm, jnp.array([dp_unnorm])])
        tangent = tangent / jnp.linalg.norm(tangent)
        
        # Orient with previous tangent if available
        if prev_tangent is not None:
            if jnp.dot(tangent, prev_tangent) < 0:
                tangent = -tangent
        
        return tangent
