# JaxCont Development Guide

## Project Structure

```
JaxCont/
├── src/jaxcont/          # Main package source code
│   ├── core/             # Core continuation algorithms
│   ├── problems/         # Problem definitions
│   ├── bifurcations/     # Bifurcation detection
│   ├── solvers/          # Numerical solvers
│   ├── stability/        # Stability analysis
│   └── utils/            # Utilities and plotting
├── tests/                # Test suite
├── examples/             # Example scripts
├── docs/                 # Documentation (to be added)
└── pyproject.toml        # Package configuration
```

## Key Modules

### Core (`src/jaxcont/core/`)
- `continuation.py`: Main problem and solution containers
- `predictor_corrector.py`: Base class for continuation methods
- `natural_continuation.py`: Natural parameter continuation
- `pseudo_arclength.py`: Pseudo-arclength continuation (most robust)

### Problems (`src/jaxcont/problems/`)
- `equilibrium.py`: Equilibrium point problems
- `periodic.py`: Periodic orbit problems
- `bvp.py`: Boundary value problems

### Bifurcations (`src/jaxcont/bifurcations/`)
- `detector.py`: Main bifurcation detection engine
- `fold.py`: Fold (saddle-node) bifurcations
- `hopf.py`: Hopf bifurcations
- `period_doubling.py`: Period-doubling bifurcations

### Solvers (`src/jaxcont/solvers/`)
- `newton.py`: Newton's method with automatic differentiation
- `corrector.py`: Corrector methods for continuation

### Stability (`src/jaxcont/stability/`)
- `eigenvalue.py`: Eigenvalue computation and stability analysis
- `floquet.py`: Floquet multipliers for periodic orbits

## Implementation Roadmap

### Phase 1: Core Functionality (Current)
- [x] Package structure
- [x] Basic continuation framework
- [x] Natural continuation
- [x] Pseudo-arclength continuation
- [x] Newton solver with JAX autodiff
- [ ] Full testing and validation

### Phase 2: Bifurcation Analysis
- [ ] Complete fold bifurcation detection
- [ ] Complete Hopf bifurcation detection
- [ ] Branch point detection
- [ ] Bifurcation point refinement
- [ ] Normal form computations

### Phase 3: Periodic Orbits
- [ ] Single shooting method
- [ ] Multiple shooting method
- [ ] Orthogonal collocation
- [ ] Floquet multiplier computation
- [ ] Period-doubling detection

### Phase 4: Advanced Features
- [ ] Two-parameter continuation
- [ ] Homoclinic orbits
- [ ] Limit point continuation
- [ ] Interactive plotting
- [ ] GPU acceleration

### Phase 5: Polish
- [ ] Comprehensive documentation
- [ ] Tutorial notebooks
- [ ] Performance benchmarks
- [ ] Comparison with MATCONT/PyDSTool
- [ ] Publication-ready examples

## Development Guidelines

### Using JAX Effectively

1. **JIT Compilation**: Use `@jit` for performance-critical functions
2. **Automatic Differentiation**: Leverage `jacfwd` and `grad` for derivatives
3. **Pure Functions**: Keep functions pure for JIT compatibility
4. **Array Operations**: Use JAX numpy for all array operations

### Testing

Run tests with:
```bash
pytest tests/ -v --cov=jaxcont
```

### Code Quality

Format code:
```bash
black src/
isort src/
```

Check types:
```bash
mypy src/
```

## Next Steps

1. Install dependencies and test the skeleton:
```bash
pip install -e ".[dev]"
pytest tests/
```

2. Start with a simple example:
```bash
python examples/example_01_pitchfork.py
```

3. Begin implementing missing functionality based on the roadmap

## References

- **MATCONT**: https://sourceforge.net/projects/matcont/
- **PyDSTool**: https://github.com/robclewley/pydstool
- **BifurcationKit.jl**: https://github.com/bifurcationkit/BifurcationKit.jl
- **AUTO**: http://indy.cs.concordia.ca/auto/
- **JAX Documentation**: https://jax.readthedocs.io/
