"""Bifurcation detection and analysis."""

from jaxcont.bifurcations.detector import BifurcationDetector
from jaxcont.bifurcations.fold import FoldBifurcation
from jaxcont.bifurcations.hopf import HopfBifurcation
from jaxcont.bifurcations.period_doubling import PeriodDoublingBifurcation
from jaxcont.bifurcations.taxonomy import LABELS, BIFURCATION_TYPES, BifurcationLabel, describe

__all__ = [
    "BifurcationDetector",
    "FoldBifurcation",
    "HopfBifurcation",
    "PeriodDoublingBifurcation",
    "LABELS",
    "BIFURCATION_TYPES",
    "BifurcationLabel",
    "describe",
]
