"""Load and select declarative MatCont validation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


_REQUIRED_CASE_FIELDS = {
    "id",
    "title",
    "support",
    "features",
    "python",
    "matlab",
    "references",
    "manual_source",
    "tolerances",
}
_DEFAULT_REGISTRY_PATH = Path(__file__).with_name("cases.json")


def load_registry(path: Path | None = None) -> dict[str, Any]:
    """Load the case registry and reject incomplete or ambiguous entries."""
    registry_path = _DEFAULT_REGISTRY_PATH if path is None else Path(path)
    with registry_path.open(encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    cases = registry.get("cases") if isinstance(registry, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("MatCont registry must contain a non-empty 'cases' list")

    case_ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("MatCont registry cases must be objects")
        missing_fields = _REQUIRED_CASE_FIELDS - set(case)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"MatCont registry case is missing required fields: {missing}")
        case_ids.append(case["id"])

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("MatCont registry case IDs must be unique")
    return registry


def select_cases(
    registry: dict[str, Any],
    ids: Iterable[str] | None = None,
    include_unsupported: bool = False,
) -> list[dict[str, Any]]:
    """Return requested cases in registry order, excluding unsupported ones by default."""
    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise ValueError("MatCont registry must contain a 'cases' list")

    requested_ids = None if ids is None else set(ids)
    known_ids = {case["id"] for case in cases}
    unknown_ids = set() if requested_ids is None else requested_ids - known_ids
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise ValueError(f"Unknown MatCont case ID(s): {unknown}")

    return [
        case
        for case in cases
        if (requested_ids is None or case["id"] in requested_ids)
        and (include_unsupported or case["support"] != "unsupported")
    ]
