"""
jaxcont.viz -- consolidated visualization for continuation/bifurcation
diagrams and 2D phase planes. See
docs/superpowers/specs/2026-07-22-viz-module-design.md and
docs/superpowers/specs/2026-07-28-phase-plane-visualization-design.md for the
design rationale behind this module's structure.
"""

from jaxcont.viz.core import plot_all_states, plot_bifurcation_diagram, plot_continuation
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
)

__all__ = [
    # Continuation diagrams
    "plot_continuation",
    "plot_bifurcation_diagram",
    "plot_all_states",
    "plot_branch_states",
    "plot_eigenvalues",
    "EigenvalueReference",
    # 2D phase planes
    "plot_phase_plane",
    "plot_nullclines",
    "plot_vector_field",
    "plot_streamlines",
    "plot_equilibria",
    "plot_trajectory",
    # Deprecated, removal target v0.4.0
    "plot_phase_portrait",
]
