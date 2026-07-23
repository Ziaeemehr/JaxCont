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


#: Keyed by the lowercase bif_type strings detect_events emits (see
#: bifurcations/events.py). Labels use bifurcations/taxonomy.py's standard
#: abbreviations (BIFURCATION_TYPES), not full words, so every plot uses the
#: same naming convention as the rest of the project: "fold" -> LP, "hopf" ->
#: H, matching taxonomy.py's own comments (`# jc.Fold`, `# jc.Hopf`).
BIFURCATION_STYLES: dict[str, BifStyle] = {
    "fold": BifStyle("s", "#009E73", "LP"),
    "hopf": BifStyle("^", "#CC79A7", "H"),
    "period-doubling": BifStyle("v", "#E69F00", "PD"),
    "branch-point": BifStyle("D", "#7B61A8", "BP"),
}

#: Fallback for a bif_type with no entry above.
DEFAULT_STYLE = BifStyle("x", "#262626", None)


def style_for(bif_type: str) -> BifStyle:
    """Style for a bifurcation type, falling back to DEFAULT_STYLE (with the
    raw ``bif_type`` string as its label) for unknown types."""
    style = BIFURCATION_STYLES.get(bif_type)
    if style is not None:
        return style
    return BifStyle(DEFAULT_STYLE.marker, DEFAULT_STYLE.color, bif_type)
