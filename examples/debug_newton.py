"""Debug Newton solver on the failing test."""

import jax.numpy as jnp
from jaxcont.solvers.newton import NewtonSolver
from jax import jacfwd

def f(x):
    # f1 = x^2 + y^2 - 1
    # f2 = x - y
    return jnp.array([
        x[0]**2 + x[1]**2 - 1.0,
        x[0] - x[1]
    ])

x0 = jnp.array([0.5, 0.5])

print("Testing Newton solver on system:")
print("f1 = x^2 + y^2 - 1")
print("f2 = x - y")
print(f"\nInitial guess: x0 = {x0}")
print(f"f(x0) = {f(x0)}")
print(f"Jacobian at x0:")
jac = jacfwd(f)(x0)
print(jac)

solver = NewtonSolver(tol=1e-8, max_iter=20)

# Manually run iterations to see what happens
x = x0
for i in range(20):
    f_val = f(x)
    residual = jnp.linalg.norm(f_val)
    print(f"\nIteration {i}: x = {x}, residual = {residual:.2e}")
    
    if residual < 1e-8:
        print(f"Converged!")
        break
    
    jac = jacfwd(f)(x)
    print(f"Jacobian:\n{jac}")
    print(f"Condition number: {jnp.linalg.cond(jac):.2e}")
    
    try:
        dx = jnp.linalg.solve(jac, -f_val)
        print(f"dx = {dx}")
        x = x + dx
    except Exception as e:
        print(f"Error solving: {e}")
        break

print(f"\n\nFinal solution: x = {x}")
print(f"f(x) = {f(x)}")
print(f"Expected: x = y = 1/sqrt(2) = {1.0/jnp.sqrt(2.0)}")

# Now test with the solver
print("\n" + "="*70)
print("Testing with NewtonSolver:")
x_result, converged, n_iter = solver.solve(f, x0)
print(f"Result: x = {x_result}")
print(f"Converged: {converged}")
print(f"Iterations: {n_iter}")
print(f"Final residual: {jnp.linalg.norm(f(x_result)):.2e}")
