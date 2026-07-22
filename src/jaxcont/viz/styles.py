"""
Shared bifurcation-point styling for jaxcont.viz.

One canonical marker/color/label table, so plot_continuation, plot_all_states,
and any future plotting function style bifurcation markers identically
instead of each hardcoding its own dict (previously duplicated across
utils/plotting.py, example_02_lorenz.py, and example_05_neural_mass.py).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BifStyle:
    """Marker/color/label for one bifurcation type, e.g. BIFURCATION_STYLES["fold"]."""

    marker: str
    color: str
    label: str


#: Keyed by the lowercase bif_type strings BifurcationDetector emits (see
#: bifurcations/detector.py) -- NOT taxonomy.py's short codes ("LP"/"H"),
#: which are a separate, human-facing vocabulary.
BIFURCATION_STYLES: dict[str, BifStyle] = {
    "fold": BifStyle("s", "green", "Fold"),
    "hopf": BifStyle("^", "magenta", "Hopf"),
    "period-doubling": BifStyle("v", "orange", "PD"),
    "branch-point": BifStyle("D", "purple", "BP"),
}

#: Fallback for a bif_type with no entry above.
DEFAULT_STYLE = BifStyle("x", "black", None)


def style_for(bif_type: str) -> BifStyle:
    """Style for a bifurcation type, falling back to DEFAULT_STYLE (with the
    raw ``bif_type`` string as its label) for unknown types."""
    style = BIFURCATION_STYLES.get(bif_type)
    if style is not None:
        return style
    return BifStyle(DEFAULT_STYLE.marker, DEFAULT_STYLE.color, bif_type)
