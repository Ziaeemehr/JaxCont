"""docs/source/validation.md is generated and agrees with the registry and the bundles."""

import json

from examples.validation import docs_page
from examples.validation.registry import load_registry


def test_validation_page_is_generated_from_the_registry():
    """Regenerate with `python3 -m examples.validation.docs_page`."""
    assert docs_page.PAGE.read_text() == docs_page.render()


def test_every_registered_case_is_on_the_page_with_its_status():
    text = docs_page.render()
    for case in load_registry()["cases"]:
        row = next(line for line in text.splitlines() if line.startswith(f"| `{case.get('canonical_id', case['id'])}` |"))
        assert f"| {case['evidence_status']} |" in row


def test_a_passing_case_has_no_failing_promoted_bundle():
    """evidence_status PASS must not contradict a reviewed promotion verdict."""
    for case in load_registry()["cases"]:
        canonical = case.get("canonical_id", case["id"])
        for manifest in docs_page.REFERENCE.glob(f"*/*/{canonical}/promotion-manifest.json"):
            verdict = json.loads(manifest.read_text())["verdict"]
            if case["evidence_status"] == "PASS":
                assert verdict == "PASS", f"{canonical}: {manifest.parent} is {verdict}"
