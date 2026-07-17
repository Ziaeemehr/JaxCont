# Installation and Quick Start

## Installation

### From Source (Development)

```bash
git clone https://github.com/yourusername/JaxCont.git
cd JaxCont
pip install -e ".[dev]"
```

### Dependencies

JaxCont requires:
- Python ≥ 3.9
- JAX ≥ 0.4.20
- NumPy ≥ 1.24.0
- SciPy ≥ 1.11.0
- Matplotlib ≥ 3.7.0

## Quick Start

### Example 1: Simple Bifurcation

```python
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation

# Define system: dx/dt = r*x - x^3 (pitchfork bifurcation)
def pitchfork(state, params):
    x = state[0]
    r = params['r']
    return jnp.array([r * x - x**3])

# Setup problem
problem = ContinuationProblem(
    rhs=pitchfork,
    u0=jnp.array([0.1]),
    params={'r': -1.0},
    continuation_param='r'
)

# Run continuation
solution = equilibrium_continuation(problem, param_range=(-1.0, 2.0))

# Plot results
solution.plot()
```

### Example 2: Lorenz System

```python
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation

def lorenz(state, params):
    x, y, z = state
    sigma, rho, beta = params['sigma'], params['rho'], params['beta']
    return jnp.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

problem = ContinuationProblem(
    rhs=lorenz,
    u0=jnp.array([0.0, 0.0, 0.0]),
    params={'sigma': 10.0, 'rho': 1.0, 'beta': 8/3},
    continuation_param='rho'
)

solution = equilibrium_continuation(problem, param_range=(1.0, 30.0))
solution.plot()
```

## Running Examples

The package includes several examples:

```bash
cd examples
python example_01_pitchfork.py
python example_02_lorenz.py
python example_03_van_der_pol.py
```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=jaxcont --cov-report=html
```

## Next Steps

- Read the full documentation (coming soon)
- Explore the examples directory
- Check out DEVELOPMENT.md for contributing
- See the roadmap in README.md
