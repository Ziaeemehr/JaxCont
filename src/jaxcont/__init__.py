"""
JaxCont: High-Performance Continuation and Bifurcation Analysis in JAX

A modern Python package for numerical continuation and bifurcation analysis
of dynamical systems, leveraging JAX's automatic differentiation and JIT
compilation for exceptional performance.
"""

from jaxcont._version import __version__

__author__ = "Abolfazl Ziaeemehr"
__license__ = "MIT"

# Functional API -- the blessed public surface: bif_problem() + continuation()
from jaxcont.api import (
    BifProblem,
    bif_problem,
    continuation,
    ContinuationPar,
    Solvers,
    ContinuationAlgorithm,
    PseudoArclength,
    Natural,
    Event,
    Fold,
    Hopf,
    EventHit,
    Branch,
    ContinuationResult,
)

# Core imports
from jaxcont.core.continuation import (
    ContinuationProblem,
    ContinuationSolution,
)

# Problem definitions
from jaxcont.problems.equilibrium import EquilibriumProblem

# Differentiable fold solver (reverse-mode grad of a fold location via the
# implicit function theorem -- see examples/example_07_differentiable.py)
from jaxcont.bifurcations.fold_solve import fold_point, fold_parameter

# Solvers
from jaxcont.solvers.newton import NewtonSolver
from jaxcont.solvers.corrector import Corrector
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver

# Stability analysis
from jaxcont.stability.eigenvalue import compute_eigenvalues, analyze_stability

# NOTE: v0.1.0 ships equilibria only. Periodic-orbit / Floquet / BVP / period-
# doubling APIs are experimental stubs and are intentionally NOT exported at
# the top level (see the project roadmap). They remain importable from their
# submodules for development, e.g.:
#     from jaxcont.problems.periodic import periodic_orbit_problem
#     from jaxcont.stability.floquet import compute_floquet_multipliers

# Utilities
from jaxcont.utils.config import Config
from jaxcont.viz import plot_bifurcation_diagram, plot_continuation

__all__ = [
    # Functional API (blessed surface)
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "Solvers",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "Branch",
    "ContinuationResult",
    "fold_point",
    "fold_parameter",
    # Core
    "ContinuationProblem",
    "ContinuationSolution",
    # Problems
    "EquilibriumProblem",
    # Solvers
    "NewtonSolver",
    "Corrector",
    "LinearSolver",
    "EigenSolver",
    "Dense",
    "DenseEigen",
    # Stability
    "compute_eigenvalues",
    "analyze_stability",
    # Utilities
    "Config",
    "plot_bifurcation_diagram",
    "plot_continuation",
]
