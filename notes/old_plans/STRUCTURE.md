# JaxCont Package Structure Summary

This document provides a comprehensive overview of the JaxCont package structure and implementation plan.

## Package Overview

JaxCont is a high-performance continuation and bifurcation analysis package implemented in JAX, inspired by:
- **MATCONT** (MATLAB): Classical continuation toolbox
- **PyDSTool** (Python): Comprehensive dynamical systems toolkit (unmaintained)
- **BifurcationKit.jl** (Julia): Modern Julia implementation
- **AUTO** (Fortran): Classical bifurcation analysis software

## Directory Structure

```
JaxCont/
├── src/jaxcont/              # Main source code
│   ├── __init__.py           # Package initialization and exports
│   │
│   ├── core/                 # Core continuation algorithms
│   │   ├── __init__.py
│   │   ├── continuation.py           # Main problem/solution containers
│   │   ├── predictor_corrector.py    # Base predictor-corrector class
│   │   ├── natural_continuation.py   # Natural parameter continuation
│   │   └── pseudo_arclength.py       # Pseudo-arclength continuation
│   │
│   ├── problems/             # Problem definitions
│   │   ├── __init__.py
│   │   ├── equilibrium.py    # Equilibrium continuation problems
│   │   ├── periodic.py       # Periodic orbit problems
│   │   └── bvp.py            # Boundary value problems
│   │
│   ├── bifurcations/         # Bifurcation detection
│   │   ├── __init__.py
│   │   ├── detector.py       # Main bifurcation detector
│   │   ├── fold.py           # Fold (saddle-node) bifurcations
│   │   ├── hopf.py           # Hopf bifurcations
│   │   └── period_doubling.py # Period-doubling bifurcations
│   │
│   ├── solvers/              # Numerical solvers
│   │   ├── __init__.py
│   │   ├── newton.py         # Newton's method with JAX autodiff
│   │   └── corrector.py      # Corrector methods
│   │
│   ├── stability/            # Stability analysis
│   │   ├── __init__.py
│   │   ├── eigenvalue.py     # Eigenvalue computation
│   │   └── floquet.py        # Floquet multipliers for periodic orbits
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       └── plotting.py       # Plotting utilities
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py           # Pytest configuration
│   ├── test_continuation.py # Core continuation tests
│   ├── test_newton.py        # Newton solver tests
│   ├── test_bifurcations.py  # Bifurcation detection tests
│   └── test_stability.py     # Stability analysis tests
│
├── examples/                 # Example scripts
│   ├── __init__.py
│   ├── example_01_pitchfork.py     # Pitchfork bifurcation
│   ├── example_02_lorenz.py        # Lorenz system
│   └── example_03_van_der_pol.py   # Van der Pol oscillator
│
├── docs/                     # Documentation (to be added)
│
├── pyproject.toml           # Package configuration (PEP 621)
├── setup.py                 # Setup script (for compatibility)
├── LICENSE                  # MIT License
├── README.md                # Main documentation
├── INSTALL.md               # Installation instructions
├── CONTRIBUTING.md          # Contribution guidelines
├── DEVELOPMENT.md           # Development guide
├── Makefile                 # Development commands
└── .gitignore              # Git ignore rules
```

## Key Components

### 1. Core Module (`src/jaxcont/core/`)

**ContinuationProblem** - Main problem container
- Stores RHS function, initial state, parameters
- Handles parameter management
- Problem type specification (equilibrium, periodic, BVP)

**ContinuationSolution** - Solution container
- Stores states and parameters along branch
- Optional eigenvalues and stability information
- Bifurcation point information
- Plotting and saving capabilities

**PredictorCorrector** - Base class for continuation methods
- Defines interface for predict/correct steps
- Adaptive step size control
- Tangent vector computation
- Main continuation loop

**NaturalContinuation** - Simplest method
- Increments parameter, fixes with Newton
- Cannot pass turning points
- Good for learning and simple problems

**PseudoArclengthContinuation** - Most robust method
- Parametrizes by arclength instead of parameter
- Can pass turning points (fold bifurcations)
- Uses bordered Newton system
- Industry standard method

### 2. Problems Module (`src/jaxcont/problems/`)

**EquilibriumProblem** - For equilibria
- Solves f(u, p) = 0
- Provides Jacobian computation
- Parameter derivatives

**PeriodicOrbitProblem** - For periodic orbits
- Shooting methods
- Multiple shooting for stability
- Collocation methods
- Phase conditions

**BoundaryValueProblem** - For BVPs
- Two-point boundary value problems
- Collocation solver
- Shooting method

### 3. Bifurcations Module (`src/jaxcont/bifurcations/`)

**BifurcationDetector** - Main detector
- Monitors test functions along branch
- Detects sign changes
- Coordinates specific detectors
- Locates bifurcation precisely

**FoldBifurcation** - Fold/saddle-node
- Detects eigenvalue crossing zero
- Normal form computation
- Tangent vector at bifurcation

**HopfBifurcation** - Hopf bifurcation
- Detects complex pair crossing imaginary axis
- Frequency estimation
- First Lyapunov coefficient (criticality)

**PeriodDoublingBifurcation** - For periodic orbits
- Monitors Floquet multiplier crossing -1
- Period-doubling cascades

### 4. Solvers Module (`src/jaxcont/solvers/`)

**NewtonSolver** - Newton's method
- Uses JAX automatic differentiation
- Handles systems of equations
- Configurable tolerance and iterations
- Optional JIT compilation

**Corrector** - Correction strategies
- Newton correction (standard)
- Moore-Penrose correction
- Extensible for other methods

### 5. Stability Module (`src/jaxcont/stability/`)

**eigenvalue.py** - Equilibrium stability
- Eigenvalue computation via JAX
- Stability classification
- Equilibrium type determination

**floquet.py** - Periodic orbit stability
- Monodromy matrix computation
- Floquet multiplier calculation
- Variational equations integration

### 6. Utils Module (`src/jaxcont/utils/`)

**Config** - Configuration management
- Global settings
- Presets (fast, accurate)
- Detection toggles

**plotting.py** - Visualization
- Bifurcation diagrams
- Phase portraits
- Eigenvalue trajectories
- Stability coloring

## Implementation Status

### ✅ Completed
- Package structure and organization
- Core data structures (Problem, Solution)
- Base predictor-corrector framework
- Natural continuation implementation
- Pseudo-arclength continuation implementation
- Newton solver with JAX autodiff
- Bifurcation detector framework
- Fold/Hopf detection (structure)
- Eigenvalue stability analysis
- Floquet multiplier computation (structure)
- Configuration system
- Plotting utilities
- Example scripts (pitchfork, Lorenz, Van der Pol)
- Test framework
- Documentation structure

### 🚧 In Progress / Needs Implementation
- Full validation and testing of all modules
- Refinement of bifurcation point location
- Normal form computations
- Complete periodic orbit solvers
- Two-parameter continuation
- Interactive visualization
- GPU acceleration optimizations
- Comprehensive documentation

### 📋 Future Enhancements
- Branch switching at bifurcations
- Homoclinic orbit detection
- Torus bifurcations
- Symmetry exploitation
- Parallel continuation
- Web-based visualization
- Integration with other tools

## Getting Started

1. **Install dependencies:**
```bash
pip install -e ".[dev]"
```

2. **Run tests:**
```bash
make test
```

3. **Try examples:**
```bash
make examples
```

4. **Start developing:**
See DEVELOPMENT.md for detailed guidelines

## Design Philosophy

1. **JAX-First**: Leverage automatic differentiation and JIT compilation
2. **Modular**: Easy to extend with new methods and bifurcations
3. **Type-Safe**: Use type hints throughout
4. **Well-Tested**: Comprehensive test coverage
5. **User-Friendly**: Simple API for common tasks
6. **Performance**: GPU-ready, optimized for large systems
7. **Educational**: Clear code that teaches continuation theory

## References and Inspiration

- **MATCONT Manual**: Excellent theoretical background
- **PyDSTool Documentation**: Good API design patterns
- **BifurcationKit.jl**: Modern Julia implementation
- **Kuznetsov's Book**: "Elements of Applied Bifurcation Theory"
- **Doedel et al.**: AUTO continuation software papers

## Contributing

See CONTRIBUTING.md for guidelines on how to contribute to the project.

## License

MIT License - See LICENSE file for details.
