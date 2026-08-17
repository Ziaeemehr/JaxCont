"""
jaxcont.viz -- consolidated visualization for continuation/bifurcation
diagrams and 2D phase planes. See
docs/superpowers/specs/2026-07-22-viz-module-design.md and
docs/superpowers/specs/2026-07-28-phase-plane-visualization-design.md for the
design rationale behind this module's structure.
"""

from jaxcont.viz.core import (
    plot_all_states,
    plot_bifurcation_diagram,
    plot_continuation,
)
from jaxcont.viz.phase_plane import (
    plot_equilibria,
    plot_nullclines,
    plot_phase_plane,
    plot_streamlines,
    plot_trajectory,
    plot_vector_field,
)
from jaxcont.viz.portraits import (
    EigenvalueReference,
    plot_branch_states,
    plot_eigenvalues,
    plot_phase_portrait,
    plot_prc,
)
from jaxcont.viz.two_parameter import plot_two_parameter_diagram

__all__ = [
    "EigenvalueReference",
    "plot_all_states",
    "plot_bifurcation_diagram",
    "plot_branch_states",
    # Continuation diagrams
    "plot_continuation",
    "plot_eigenvalues",
    "plot_equilibria",
    "plot_nullclines",
    # 2D phase planes
    "plot_phase_plane",
    # Deprecated, removal target v0.4.0
    "plot_phase_portrait",
    "plot_prc",
    "plot_streamlines",
    "plot_trajectory",
    # Two-parameter diagrams
    "plot_two_parameter_diagram",
    "plot_vector_field",
]
