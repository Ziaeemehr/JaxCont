"""Tests for jaxcont.viz."""

import matplotlib
matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from jaxcont.core.continuation import ContinuationSolution
from jaxcont.viz.styles import BIFURCATION_STYLES, DEFAULT_STYLE, style_for


def test_bifurcation_styles_cover_detector_types():
    """Every bif_type string bifurcations/detector.py emits has a style entry.

    BifurcationDetector tags points with bif_type == 'fold' or 'hopf' (see
    detector.py's detect_along_branch); both must resolve to a real
    BIFURCATION_STYLES entry, not silently fall through to DEFAULT_STYLE, so a
    marker/color change in one place can't accidentally stop applying to one
    of the two shipped detectors.
    """
    for bif_type in ("fold", "hopf"):
        assert bif_type in BIFURCATION_STYLES


def test_style_for_known_type_returns_table_entry():
    assert style_for("fold") == BIFURCATION_STYLES["fold"]


def test_style_for_unknown_type_falls_back_with_raw_label():
    style = style_for("some-future-type")
    assert style.marker == DEFAULT_STYLE.marker
    assert style.color == DEFAULT_STYLE.color
    assert style.label == "some-future-type"
