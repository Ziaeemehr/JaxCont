"""
Bifurcation/object naming and abbreviations.

A single reference table of the standard short labels for equilibrium- and
cycle-related continuation objects (the same names used throughout the
bifurcation-theory literature, e.g. Kuznetsov's *Elements of Applied
Bifurcation Theory*), so every ``Event``/detector JaxCont adds over time
uses a consistent abbreviation instead of inventing a new one each time.

Each entry also records whether JaxCont currently implements it, and if not,
which roadmap milestone is expected to -- so this table doubles as a compact
cross-reference between "what this bifurcation type is called" and "where in
JaxCont's own roadmap it lives" (see ``notes/ROADMAP.md``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BifurcationLabel:
    """One entry in the naming table: an object/bifurcation type and its abbreviation."""

    label: str
    name: str
    #: "equilibrium" | "cycle" -- which continuation object this label attaches to.
    of: str
    #: "v0.1" (implemented today), "v0.2"/"v0.3" (planned), or "out of scope".
    status: str


#: Codimension-0 objects (Point/Orbit/Equilibrium/Limit cycle) are the
#: *continued curves* themselves; the rest are codim-1/codim-2 singularities
#: detected/continued along those curves.
BIFURCATION_TYPES = (
    BifurcationLabel("P", "Point", "equilibrium", "v0.1"),
    BifurcationLabel("O", "Orbit", "equilibrium", "out of scope"),  # time integration, not continuation
    BifurcationLabel("EP", "Equilibrium", "equilibrium", "v0.1"),
    BifurcationLabel("LC", "Limit cycle", "cycle", "v0.2"),
    BifurcationLabel("LP", "Limit Point (fold) bifurcation", "equilibrium", "v0.1"),  # jc.Fold
    BifurcationLabel("H", "Hopf bifurcation", "equilibrium", "v0.1"),  # jc.Hopf
    BifurcationLabel("LPC", "Limit Point bifurcation of cycles", "cycle", "v0.2"),
    BifurcationLabel("NS", "Neimark-Sacker (torus) bifurcation", "cycle", "v0.2"),
    BifurcationLabel("PD", "Period Doubling (flip) bifurcation", "cycle", "v0.2"),
    BifurcationLabel("BP", "Branch Point", "equilibrium", "v0.3"),
    BifurcationLabel("CP", "Cusp bifurcation", "equilibrium", "v0.3"),
    BifurcationLabel("BT", "Bogdanov-Takens bifurcation", "equilibrium", "v0.3"),
    BifurcationLabel("ZH", "Zero-Hopf bifurcation", "equilibrium", "v0.3"),
    BifurcationLabel("HH", "Double Hopf bifurcation", "equilibrium", "v0.3"),
    BifurcationLabel("GH", "Generalized Hopf (Bautin) bifurcation", "equilibrium", "v0.3"),
    BifurcationLabel("BPC", "Branch Point of Cycles", "cycle", "out of scope"),
    BifurcationLabel("CPC", "Cusp bifurcation of Cycles", "cycle", "out of scope"),
    BifurcationLabel("R1", "1:1 Resonance", "cycle", "out of scope"),
    BifurcationLabel("R2", "1:2 Resonance", "cycle", "out of scope"),
    BifurcationLabel("R3", "1:3 Resonance", "cycle", "out of scope"),
    BifurcationLabel("R4", "1:4 Resonance", "cycle", "out of scope"),
    BifurcationLabel("CH", "Chenciner (generalized Neimark-Sacker) bifurcation", "cycle", "out of scope"),
    BifurcationLabel("LPNS", "Fold-Neimark-Sacker bifurcation", "cycle", "out of scope"),
    BifurcationLabel("PDNS", "Flip-Neimark-Sacker bifurcation", "cycle", "out of scope"),
    BifurcationLabel("LPPD", "Fold-flip", "cycle", "out of scope"),
    BifurcationLabel("NSNS", "Double Neimark-Sacker", "cycle", "out of scope"),
    BifurcationLabel("GPD", "Generalized Period Doubling", "cycle", "out of scope"),
)

#: Convenience lookup: label -> :class:`BifurcationLabel`, e.g. ``LABELS["H"]``.
LABELS = {row.label: row for row in BIFURCATION_TYPES}


def describe(label: str) -> str:
    """Human-readable one-liner for a bifurcation label, e.g. ``describe("H")``
    -> ``"H — Hopf bifurcation (equilibrium; implemented)"``.

    Raises ``KeyError`` with the full table's labels listed if ``label`` isn't
    one of the known abbreviations (a likely typo, not a missing feature).
    """
    try:
        row = LABELS[label]
    except KeyError:
        raise KeyError(
            f"{label!r} is not a known bifurcation label. Known labels: "
            f"{', '.join(LABELS)}"
        ) from None
    return f"{row.label} — {row.name} ({row.of}; {_status_phrase(row.status)})"


def _status_phrase(status: str) -> str:
    if status == "v0.1":
        return "implemented"
    if status == "out of scope":
        return "out of scope, see ROADMAP.md"
    return f"planned {status}"
