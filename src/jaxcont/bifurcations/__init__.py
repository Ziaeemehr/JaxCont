"""Bifurcation detection and analysis."""

from jaxcont.bifurcations.events import BranchPoint, Event, Fold, Hopf, EventHit, detect_events
from jaxcont.bifurcations.period_doubling import PeriodDoublingBifurcation
from jaxcont.bifurcations.taxonomy import LABELS, BIFURCATION_TYPES, BifurcationLabel, describe

__all__ = [
    "BranchPoint",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "detect_events",
    "PeriodDoublingBifurcation",
    "LABELS",
    "BIFURCATION_TYPES",
    "BifurcationLabel",
    "describe",
]
