"""Python-side numerical cases for the MatCont validation suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseResult:
    """Normalized result returned by every Python validation case."""

    case_id: str
    checks: dict[str, Any]
    observations: dict[str, Any]
    artifacts: dict[str, Any]


__all__ = ["CaseResult"]
