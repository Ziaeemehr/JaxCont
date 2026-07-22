"""
jaxcont.viz -- consolidated visualization for continuation/bifurcation
diagrams. See docs/superpowers/specs/2026-07-22-viz-module-design.md for the
design rationale behind this module's structure.
"""

from jaxcont.viz.core import plot_all_states, plot_bifurcation_diagram, plot_continuation
from jaxcont.viz.portraits import EigenvalueReference, plot_eigenvalues, plot_phase_portrait

__all__ = [
    "plot_continuation",
    "plot_bifurcation_diagram",
    "plot_all_states",
    "plot_phase_portrait",
    "plot_eigenvalues",
    "EigenvalueReference",
]
