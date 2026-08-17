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
    Branch,
    ContinuationAlgorithm,
    ContinuationPar,
    ContinuationResult,
    Event,
    EventHit,
    Fold,
    Hopf,
    Natural,
    NeimarkSacker,
    PeriodDoubling,
    PseudoArclength,
    Solvers,
    bif_problem,
    continuation,
)
from jaxcont.bifurcations.codim2 import (
    bogdanov_takens_parameters,
    bogdanov_takens_point,
    cusp_parameters,
    cusp_point,
    double_hopf_parameters,
    double_hopf_point,
    generalized_hopf_parameters,
    generalized_hopf_point,
    zero_hopf_parameters,
    zero_hopf_point,
)
from jaxcont.bifurcations.codim2_events import (
    BogdanovTakens,
    Cusp,
    DoubleHopf,
    GeneralizedHopf,
    ZeroHopf,
)

# NOTE: v0.1.0 shipped equilibria only. Periodic-orbit continuation and
# Floquet-multiplier computation are real (not stubs) but intentionally not
# exported at the top level yet (see the project roadmap) -- import from
# their submodules, e.g.:
#     from jaxcont.problems.periodic import periodic_orbit_problem
#     from jaxcont.stability.floquet import floquet_multipliers
# The bifurcation-detection Events that consume them (Fold, Hopf,
# PeriodDoubling, NeimarkSacker) ARE exported at the top level above, since
# events=[...] is passed to the top-level continuation() call regardless of
# problem kind. BVP continuation remains an unimplemented stub.
# Two-parameter continuation: fold/Hopf curves + codim-2 events along them.
# See docs/superpowers/specs/2026-08-17-two-parameter-continuation-design.md
from jaxcont.bifurcations.curves import fold_curve_problem, hopf_curve_problem

# Fold normal-form coefficient + direct codim-2 point solvers (cusp,
# Bogdanov-Takens, generalized Hopf, zero-Hopf, double Hopf). These take p
# with shape (2,) -- codim-2 needs two free parameters -- and are
# differentiable in args like their codim-1 siblings above. See
# docs/superpowers/specs/2026-08-05-codim2-direct-solvers-design.md
from jaxcont.bifurcations.fold_normal_form import fold_coefficient

# Differentiable fold solver (reverse-mode grad of a fold location via the
# implicit function theorem -- see examples/example_07_differentiable.py)
from jaxcont.bifurcations.fold_solve import fold_parameter, fold_point

# Differentiable Hopf-point solver + first Lyapunov coefficient (Hopf
# criticality) -- see docs/superpowers/specs/2026-08-04-hopf-normal-form-design.md
from jaxcont.bifurcations.hopf_normal_form import (
    hopf_parameter,
    hopf_point,
    lyapunov_coefficient,
)

# Core imports
from jaxcont.core.continuation import (
    ContinuationProblem,
    ContinuationSolution,
)

# Problem definitions
from jaxcont.problems.equilibrium import EquilibriumProblem
from jaxcont.solvers.corrector import Corrector

# Solvers
from jaxcont.solvers.newton import NewtonSolver
from jaxcont.solvers.protocols import Dense, DenseEigen, EigenSolver, LinearSolver

# Stability analysis
from jaxcont.stability.eigenvalue import analyze_stability, compute_eigenvalues

# Utilities
from jaxcont.utils.config import Config
from jaxcont.viz import (
    plot_bifurcation_diagram,
    plot_continuation,
    plot_two_parameter_diagram,
)

__all__ = [
    # Functional API (blessed surface)
    "BifProblem",
    "BogdanovTakens",
    "Branch",
    # Utilities
    "Config",
    "ContinuationAlgorithm",
    "ContinuationPar",
    # Core
    "ContinuationProblem",
    "ContinuationResult",
    "ContinuationSolution",
    "Corrector",
    "Cusp",
    "Dense",
    "DenseEigen",
    "DoubleHopf",
    "EigenSolver",
    # Problems
    "EquilibriumProblem",
    "Event",
    "EventHit",
    "Fold",
    "GeneralizedHopf",
    "Hopf",
    "LinearSolver",
    "Natural",
    "NeimarkSacker",
    # Solvers
    "NewtonSolver",
    "PeriodDoubling",
    "PseudoArclength",
    "Solvers",
    "ZeroHopf",
    "analyze_stability",
    "bif_problem",
    "bogdanov_takens_parameters",
    "bogdanov_takens_point",
    # Stability
    "compute_eigenvalues",
    "continuation",
    "cusp_parameters",
    "cusp_point",
    "double_hopf_parameters",
    "double_hopf_point",
    "fold_coefficient",
    # Two-parameter continuation
    "fold_curve_problem",
    "fold_parameter",
    "fold_point",
    "generalized_hopf_parameters",
    "generalized_hopf_point",
    "hopf_curve_problem",
    "hopf_parameter",
    "hopf_point",
    "lyapunov_coefficient",
    "plot_bifurcation_diagram",
    "plot_continuation",
    "plot_two_parameter_diagram",
    "zero_hopf_parameters",
    "zero_hopf_point",
]
