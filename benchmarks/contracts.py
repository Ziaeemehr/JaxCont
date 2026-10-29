"""Benchmark workload and controls contract.

A benchmark row is only comparable across tools when every tool ran the same
validation workload with the same *resolved* controls: the values each tool
actually used after its own defaults and unit conversions, not the numbers
passed in. Comparison uses exact equality of normalized decimals, and every
mismatch is recorded with its field, tool and value.

Accuracy limits are separate from solver stopping tolerances: they name
validation-report metrics (errors against the committed reference bundles),
never a control field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping, Optional

TOOLS = ("jaxcont", "matcont", "bifurcationkit")
STAGES = ("startup", "first_call", "compile", "warm", "end_to_end", "batched")
CASES_FILE = Path(__file__).resolve().parent / "cases.json"


@dataclass(frozen=True)
class ResolvedControls:
    """Controls one tool actually used, after defaults and unit conversion.

    ``None`` means the tool has no such control. A field that is ``None`` for
    one tool and set for another makes the row non-comparable.

    The arclength metric is ``state_weight * |du|^2 + parameter_weight * dp^2``.
    JaxCont weights the state by 1/n_state and the parameter by 1, so an equal
    numeric ``ds`` under a different metric is a different physical step.
    """

    newton_atol: Optional[float]
    newton_rtol: Optional[float]
    newton_max_iter: Optional[int]
    event_atol: Optional[float]
    ds: float
    ds_min: float
    ds_max: float
    adaptive: bool
    max_steps: int
    direction: int
    bounds: Optional[tuple[float, float]]  # None: the tool cannot stop at parameter bounds
    state_weight: float | str  # a number, or the target rule "1/n_state" before resolution
    parameter_weight: float
    stability_every_step: bool
    ntst: Optional[int] = None
    ncol: Optional[int] = None
    # What the Newton tolerance bounds. JaxCont: the 2-norm of the augmented
    # residual (equations plus arclength constraint); MatCont also bounds the
    # step (VarTolerance). An equal number under another criterion differs.
    newton_criterion: Optional[str] = None


def _normalized(value: Any) -> Any:
    """Exact, representation-independent form of a control value."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return Decimal(repr(float(value))).normalize()
    if isinstance(value, (tuple, list)):
        return tuple(_normalized(item) for item in value)
    return value


def _finite_control(value):
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, tuple):
        return all(_finite_control(item) for item in value)
    return True


@dataclass(frozen=True)
class Mismatch:
    """``tool`` used ``value`` for ``field`` where ``baseline_tool`` used ``baseline_value``."""

    field: str
    tool: str
    value: Any
    baseline_tool: Optional[str] = None
    baseline_value: Any = None


@dataclass(frozen=True)
class Comparability:
    comparable: bool
    mismatches: tuple[Mismatch, ...] = ()
    reason: str = ""


def compare_controls(resolved: Mapping[str, Optional[ResolvedControls]]) -> Comparability:
    """Compare resolved controls across tools by exact normalized equality.

    A tool whose controls could not be resolved (``None``) makes the set
    non-comparable. Otherwise every field is compared with the first tool's
    value, and each differing tool is recorded with its own value.
    """
    if len(resolved) < 2:
        return Comparability(False, reason="fewer than two tools")
    missing = [tool for tool, controls in resolved.items() if controls is None]
    if missing:
        return Comparability(
            False,
            tuple(Mismatch("resolved_controls", tool, None) for tool in missing),
            "controls could not be resolved",
        )
    tools = list(resolved)
    baseline = resolved[tools[0]]
    mismatches = []
    for control in fields(ResolvedControls):
        expected = _normalized(getattr(baseline, control.name))
        if not _finite_control(expected):
            mismatches.append(Mismatch(control.name, tools[0], getattr(baseline, control.name)))
        for tool in tools[1:]:
            value = getattr(resolved[tool], control.name)
            normalized = _normalized(value)
            if not _finite_control(normalized) or normalized != expected:
                mismatches.append(Mismatch(control.name, tool, value, tools[0], getattr(baseline, control.name)))
    if mismatches:
        return Comparability(False, tuple(mismatches), "resolved controls differ")
    return Comparability(True)


@dataclass(frozen=True)
class Sampling:
    first_calls: int = 5  # fresh process per replicate
    warm_repetitions: int = 20
    untimed_warmups: int = 3
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128)
    timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class BenchmarkCase:
    """One workload row: a validation run, frozen controls and accuracy limits."""

    benchmark_id: str
    canonical_id: str
    step_mode: str  # "fixed" or "adaptive"
    controls: ResolvedControls
    accuracy_limits: Mapping[str, float]
    stages: tuple[str, ...]
    sampling: Sampling = field(default_factory=Sampling)

    def __post_init__(self):
        if self.step_mode not in ("fixed", "adaptive"):
            raise ValueError(f"{self.benchmark_id}: step_mode must be 'fixed' or 'adaptive'")
        if self.step_mode == "fixed" and (
            self.controls.adaptive or not self.controls.ds_min == self.controls.ds == self.controls.ds_max
        ):
            raise ValueError(f"{self.benchmark_id}: a fixed-step row needs ds_min == ds == ds_max and adaptive=False")
        unknown = set(self.stages) - set(STAGES)
        if unknown:
            raise ValueError(f"{self.benchmark_id}: unknown stages {sorted(unknown)}")
        if not self.accuracy_limits:
            raise ValueError(f"{self.benchmark_id}: accuracy_limits are required")
        control_names = {control.name for control in fields(ResolvedControls)}
        overlap = control_names & set(self.accuracy_limits)
        if overlap:
            # A stopping tolerance is not evidence of accuracy.
            raise ValueError(
                f"{self.benchmark_id}: accuracy_limits must name report metrics, not controls {sorted(overlap)}"
            )
        for metric, limit in self.accuracy_limits.items():
            if not (isinstance(limit, (int, float)) and limit > 0):
                raise ValueError(f"{self.benchmark_id}: accuracy limit {metric}={limit!r} must be positive")


@dataclass(frozen=True)
class StageTiming:
    """One timing measurement. An unavailable stage is ``None`` with a reason, never 0."""

    stage: str
    seconds: Optional[float]
    reason: str = ""
    estimate: bool = False  # True for "first minus warm" overhead estimates

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.seconds is None:
            if not self.reason:
                raise ValueError(f"{self.stage}: an unavailable timing needs a reason")
        elif not self.seconds > 0:
            raise ValueError(f"{self.stage}: a measured timing must be positive; use None with a reason")


def _sampling(values: dict) -> Sampling:
    return Sampling(**{**values, "batch_sizes": tuple(values.get("batch_sizes", Sampling.batch_sizes))})


def load_cases(path: Path = CASES_FILE) -> list[BenchmarkCase]:
    """Read the workload manifest into validated ``BenchmarkCase`` rows."""
    data = json.loads(Path(path).read_text())
    defaults = data["defaults"]
    cases = []
    for row in data["cases"]:
        controls = {**defaults["controls"], **row.get("controls", {})}
        controls["bounds"] = tuple(controls["bounds"])
        cases.append(
            BenchmarkCase(
                benchmark_id=row["benchmark_id"],
                canonical_id=row["canonical_id"],
                step_mode=row["step_mode"],
                controls=ResolvedControls(**controls),
                accuracy_limits=row["accuracy_limits"],
                stages=tuple(row.get("stages", defaults["stages"])),
                sampling=_sampling({**defaults["sampling"], **row.get("sampling", {})}),
            )
        )
    ids = [case.benchmark_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark_id")
    return cases


def controls_record(controls: ResolvedControls) -> dict:
    """JSON-ready resolved controls, as stored per tool in each raw record."""
    record = asdict(controls)
    record["bounds"] = None if record["bounds"] is None else list(record["bounds"])
    return record
