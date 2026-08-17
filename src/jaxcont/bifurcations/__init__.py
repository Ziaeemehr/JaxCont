"""Bifurcation detection and analysis."""

from jaxcont.bifurcations.events import (
    BranchPoint,
    Event,
    EventHit,
    Fold,
    Hopf,
    detect_events,
)
from jaxcont.bifurcations.taxonomy import (
    BIFURCATION_TYPES,
    LABELS,
    BifurcationLabel,
    describe,
)

__all__ = [
    "BIFURCATION_TYPES",
    "LABELS",
    "BifurcationLabel",
    "BranchPoint",
    "Event",
    "EventHit",
    "Fold",
    "Hopf",
    "describe",
    "detect_events",
]
