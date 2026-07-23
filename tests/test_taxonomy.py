"""
Tests for the bifurcation naming/abbreviation reference table.

See jaxcont/bifurcations/taxonomy.py for the table and rationale.
"""

import pytest

from jaxcont.bifurcations.taxonomy import LABELS, BIFURCATION_TYPES, describe
from jaxcont import Fold, Hopf


def test_table_has_no_duplicate_labels():
    labels = [row.label for row in BIFURCATION_TYPES]
    assert len(labels) == len(set(labels))


def test_table_has_expected_row_count():
    assert len(BIFURCATION_TYPES) == 27


@pytest.mark.parametrize(
    "label,expected_name",
    [
        ("LP", "Limit Point (fold) bifurcation"),
        ("H", "Hopf bifurcation"),
        ("LC", "Limit cycle"),
        ("BT", "Bogdanov-Takens bifurcation"),
        ("GPD", "Generalized Period Doubling"),
    ],
)
def test_known_labels(label, expected_name):
    assert LABELS[label].name == expected_name


def test_describe_formats_a_known_label():
    text = describe("H")
    assert "Hopf bifurcation" in text
    assert "implemented" in text


def test_describe_raises_on_unknown_label():
    with pytest.raises(KeyError, match="not a known bifurcation label"):
        describe("NOT_A_REAL_LABEL")


def test_fold_and_hopf_events_correspond_to_lp_and_h():
    """jc.Fold/jc.Hopf are the standard LP/H abbreviations, not ad hoc names."""
    assert Fold().kind == "fold"
    assert Hopf().kind == "hopf"
    assert LABELS["LP"].name.startswith("Limit Point (fold)")
    assert LABELS["H"].name == "Hopf bifurcation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
