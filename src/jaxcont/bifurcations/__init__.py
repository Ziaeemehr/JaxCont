"""Bifurcation detection and analysis."""

from jaxcont.bifurcations.detector import BifurcationDetector
from jaxcont.bifurcations.fold import FoldBifurcation
from jaxcont.bifurcations.hopf import HopfBifurcation
from jaxcont.bifurcations.period_doubling import PeriodDoublingBifurcation

__all__ = [
    "BifurcationDetector",
    "FoldBifurcation",
    "HopfBifurcation",
    "PeriodDoublingBifurcation",
]
