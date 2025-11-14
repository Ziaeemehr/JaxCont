# JaxCont

A high-performance continuation and bifurcation analysis package implemented in JAX.

## Overview

JaxCont is a modern Python package for numerical continuation and bifurcation analysis of dynamical systems, leveraging JAX's automatic differentiation and JIT compilation for exceptional performance. Inspired by MATCONT, PyDSTool, and Julia's BifurcationKit, JaxCont aims to provide a user-friendly interface for analyzing:

- Equilibrium points and their stability
- Periodic orbits
- Bifurcations (fold, Hopf, period-doubling, etc.)
- Two-parameter continuation
- Boundary value problems

## Key Features

- **High Performance**: Leverages JAX's JIT compilation and automatic differentiation
- **GPU Support**: Seamless GPU acceleration through JAX
- **Modern API**: Clean, intuitive Python interface
- **Extensible**: Modular design for easy customization
- **Well-tested**: Comprehensive test suite with classical examples

## Installation

```bash
pip install jaxcont
```

For development:
```bash
git clone https://github.com/yourusername/JaxCont.git
cd JaxCont
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from jaxcont import ContinuationProblem, equilibrium_continuation

# Define your dynamical system
def lorenz(state, params):
    x, y, z = state
    sigma, rho, beta = params['sigma'], params['rho'], params['beta']
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return jnp.array([dx, dy, dz])

# Setup continuation problem
problem = ContinuationProblem(
    rhs=lorenz,
    u0=jnp.array([1.0, 1.0, 1.0]),
    params={'sigma': 10.0, 'rho': 28.0, 'beta': 8/3},
    continuation_param='rho'
)

# Run continuation
solution = equilibrium_continuation(problem, param_range=(0.0, 40.0))

# Plot results
solution.plot()
```

## Project Structure

```
jaxcont/
├── core/           # Core continuation algorithms
├── problems/       # Problem definitions and BVP solvers
├── bifurcations/   # Bifurcation detection and analysis
├── solvers/        # Numerical solvers (Newton, predictor-corrector)
├── stability/      # Stability analysis tools
├── examples/       # Classical examples and tutorials
└── utils/          # Utilities and helpers
```

## Roadmap

- [x] Project skeleton
- [ ] Core continuation engine (pseudo-arclength)
- [ ] Equilibrium continuation
- [ ] Fold bifurcation detection
- [ ] Hopf bifurcation detection
- [ ] Periodic orbit continuation (shooting/collocation)
- [ ] Period-doubling bifurcation detection
- [ ] Two-parameter continuation
- [ ] Stability analysis (Floquet multipliers)
- [ ] Interactive visualization tools
- [ ] Comprehensive documentation

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- MATCONT: Continuation toolbox for MATLAB
- PyDSTool: Python Dynamical Systems Toolbox
- BifurcationKit.jl: Julia package for bifurcation analysis
- AUTO: Classical continuation software

## Citation

If you use JaxCont in your research, please cite:

```bibtex
@software{jaxcont,
  title = {JaxCont: High-Performance Continuation and Bifurcation Analysis in JAX},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/JaxCont}
}
```
