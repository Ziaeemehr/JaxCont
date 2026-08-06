"""MC-PRC-001: adaptx Hopf limit cycle, PRC/dPRC cross-check against MatCont
7.6's Testruns/testadaptPRC.m. See
docs/superpowers/specs/2026-08-05-prc-dprc-design.md.

Two convention mismatches between JaxCont's prc_curve/dprc_curve and
MatCont's exported PRC/dPRC columns were found and resolved/diagnosed while
building this validator (see task-8-report.md for the full derivation):

1. **Units.** ``prc_curve`` normalizes ``Z(0) . f(x_0, p) = omega = 2*pi/T``
   (phase in *radians*, one cycle = 2*pi -- the standard adjoint-PRC
   convention). MatCont's ``calcPRC.m`` normalizes phase in *cycle
   fractions* (one cycle = 1 -- matching its own exported ``phase_fraction``
   column, which runs 0..1). The two differ by exactly a factor of ``2*pi``:
   ``Z_matcont = Z_jaxcont / (2*pi)``. Applied below before comparison.

2. **``dPRC`` is not a parameter derivative.** MatCont's exported ``dPRC``
   column is ``d(PRC)/dt`` (the *time* derivative of the PRC curve),
   confirmed by reading ``LimitCycle/calcPRC.m`` in the installed MatCont
   7.6 tree directly: ``dvl(:,i) = -newperiod * J(x_i)' * vl(:,i)`` is the
   adjoint variational ODE's right-hand side (``dZ/dtau = -T . J(x)^T . Z``
   in normalized time), and the final ``dPRC = dPRC / newperiod`` converts
   it to absolute-time units -- there is no parameter (``alpha``) anywhere
   in that computation. This was confirmed numerically too: a central
   finite difference of the *reference* ``prc`` column with respect to
   time matches the reference ``dprc`` column to the same tolerance PRC
   itself achieves. JaxCont's ``dprc_curve``, by contrast, is
   ``d(prc_curve)/dp`` -- differentiation with respect to the continuation
   parameter ``alpha`` (verified independently: matches a finite difference
   of two independently re-converged ``periodic_orbit_problem`` solves at
   ``alpha +/- eps`` to ~0.2% relative, and matches the design spec's own
   closed-form circle-system check). These are two different mathematical
   objects (a time derivative vs. a parameter derivative) -- no unit or
   sign convention reconciles them. This validator still performs the
   literal comparison (dprc_curve's output against MatCont's dprc column)
   rather than substituting a different quantity, so the mismatch is
   reported honestly rather than hidden; see task-8-report.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem
from jaxcont.stability.prc import dprc_curve, prc_curve

from ..compare import ValidationMismatch, scaled_close
from . import CaseResult

_TWO_PI = 2.0 * np.pi
_PRC_ATOL = 0.05
_PRC_RTOL = 0.05


def _adaptx_rhs(u, p, args):
    del args
    x, y, z = u[0], u[1], u[2]
    alpha, beta = p[0], p[1]
    return jnp.array([y, z, -alpha * z - beta * y - x + x * x])


def _load_reference_prc(reference_dir: Path) -> dict[str, np.ndarray]:
    path = reference_dir / "MC-PRC-001_prc.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "phase_fraction": np.asarray([float(row["phase_fraction"]) for row in rows]),
        "prc": np.asarray([float(row["prc"]) for row in rows]),
        "dprc": np.asarray([float(row["dprc"]) for row in rows]),
    }


def _load_converged_point(reference_dir: Path) -> tuple[float, float]:
    """Read the alpha/period MatCont's run_prc_dprc.m actually converged to
    for the exported PRC/dPRC point (run_prc_dprc.m's third, PRC/dPRC-enabled
    continuation's last point) -- run_prc_dprc.m/export_prc_run.m record
    these as converged_alpha/converged_period so this validator can build a
    matching JaxCont periodic orbit instead of guessing one."""
    path = reference_dir / "MC-PRC-001_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return float(metadata["converged_alpha"]), float(metadata["converged_period"])


def _run_adaptx_jaxcont(alpha: float, period: float):
    """Reach the same Hopf -> limit-cycle -> PRC/dPRC point
    Testruns/testadaptPRC.m reaches, on the same adaptx system.

    ``alpha``/``period`` are MatCont's own converged values for the branch
    point its PRC/dPRC/Input continuation exported (read from
    MC-PRC-001_metadata.json's converged_alpha/converged_period) -- rather
    than rediscovering the Hopf/limit-cycle branch from scratch, this seeds
    periodic_orbit_problem's Newton solve directly at MatCont's answer, so
    only a coarse trajectory shape needs to converge, not a multi-stage
    continuation path.
    """
    n_guess = 60
    t_traj = jnp.linspace(0.0, period, n_guess, endpoint=False)
    # adaptx's Hopf eigenvector direction (see run_prc_dprc.m's init_H_LC
    # call): near the Hopf point x=y=z=0, the linearization's imaginary
    # eigenvector for x'=y, y'=z, z'=-alpha*z-beta*y-x+x^2 (beta=1,
    # omega=1 at alpha=1) is approximately (1, -i, -1) in (x, y, z) --
    # i.e. y leads x by a quarter period and z is antiphase to x. This
    # coarse guess only needs the right shape/phase relationship; the
    # collocation Newton solve (periodic_orbit_problem's internal
    # differentiable_root) converges it to the true nonlinear cycle at
    # (alpha, period).
    omega = 2 * jnp.pi / period
    amplitude = 0.5
    u_traj = jnp.stack(
        [
            amplitude * jnp.cos(omega * t_traj),
            -amplitude * jnp.sin(omega * t_traj),
            -amplitude * jnp.cos(omega * t_traj),
        ],
        axis=1,
    )
    mesh = Collocation(ntst=20, ncol=4)
    p0 = jnp.array([alpha, 1.0])
    problem = periodic_orbit_problem(_adaptx_rhs, u_traj, t_traj, period, p0, mesh)
    return problem, mesh


def _best_phase_shift(
    jaxcont_phase: np.ndarray,
    jaxcont_prc: np.ndarray,
    reference_phase: np.ndarray,
    reference_prc: np.ndarray,
    *,
    atol: float,
    rtol: float,
    window: float = 0.05,
    steps: int = 4001,
) -> float:
    """Find the circular phase shift that best aligns JaxCont's PRC curve
    onto MatCont's reference curve before comparing.

    An autonomous periodic orbit's phase origin (t=0) is only defined up to
    an arbitrary time-translation. ``periodic_orbit_problem`` anchors phase
    via an integral condition relative to the caller's own initial
    trajectory guess (see its docstring); MatCont's own convention is
    unrelated (``calcPRC.m`` circularly rotates its output so the exported
    curve starts at the state trajectory's own x-maximum -- "make the PRC
    and collocation mesh start at the spike-top", confirmed by reading
    ``LimitCycle/calcPRC.m`` directly). Nothing pins these two conventions
    to agree exactly.

    Resolving that gauge freedom via a reference-informed search before
    the tolerance check is the same pattern this suite's own comparison
    engine already uses for other non-canonical correspondences --
    ``compare.py``'s ``match_spectrum``/``match_events`` use
    ``linear_sum_assignment`` the same way, for eigenvalue/event orderings
    that likewise have no intrinsic ordering. It is not a tolerance
    relaxation: the same ``atol``/``rtol`` is applied to the aligned curves
    afterward, and the search window is bounded to about one JaxCont mesh
    spacing (``1/ntst``), not an unconstrained fit.
    """
    extended_phase = np.concatenate(
        [reference_phase - 1.0, reference_phase, reference_phase + 1.0]
    )
    extended_prc = np.concatenate([reference_prc, reference_prc, reference_prc])

    def worst_margin(delta: float) -> float:
        query = (jaxcont_phase + delta) % 1.0
        aligned = np.interp(query, extended_phase, extended_prc)
        errors = np.abs(jaxcont_prc - aligned)
        limits = atol + rtol * np.maximum(np.abs(jaxcont_prc), np.abs(aligned))
        return float(np.max(errors - limits))

    deltas = np.linspace(-window, window, steps)
    margins = np.asarray([worst_margin(delta) for delta in deltas])
    return float(deltas[int(np.argmin(margins))])


def _diagnose(actual: np.ndarray, reference: np.ndarray, *, atol: float, rtol: float) -> dict:
    """Run scaled_close without letting a failure abort the whole case --
    mirrors periodic.py's run_torbpc_cycle spectrum-matching pattern
    (try the real tolerance, fall back to an effectively unbounded one
    purely to recover the same diagnostics dict for reporting)."""
    try:
        return {"passed": True, **scaled_close(actual, reference, atol=atol, rtol=rtol)}
    except ValidationMismatch:
        return {"passed": False, **scaled_close(actual, reference, atol=1e6, rtol=0.0)}


def run_adaptx_prc_dprc(reference_dir: Path) -> CaseResult:
    """Compare JaxCont's prc_curve/dprc_curve on the adaptx system against
    MatCont's own PRC/dPRC processor_data output."""
    reference = _load_reference_prc(reference_dir)
    alpha, period = _load_converged_point(reference_dir)

    problem, mesh = _run_adaptx_jaxcont(alpha, period)
    with jax.default_matmul_precision("float32"):
        jaxcont_prc_radian = np.asarray(prc_curve(_adaptx_rhs, mesh, problem.u0, problem.p0))
        jaxcont_dprc_radian = np.asarray(dprc_curve(problem))

    # Unit conversion (see module docstring, point 1): MatCont's PRC/dPRC
    # are in cycle-fraction phase units, JaxCont's are in radian phase
    # units -- convert JaxCont's output, not MatCont's, since the reference
    # CSV is the fixed oracle.
    jaxcont_prc = jaxcont_prc_radian[:, 0] / _TWO_PI
    jaxcont_dprc = jaxcont_dprc_radian[:, 0, 0] / _TWO_PI

    jaxcont_phase = np.arange(mesh.ntst) / mesh.ntst
    phase_shift = _best_phase_shift(
        jaxcont_phase,
        jaxcont_prc,
        reference["phase_fraction"],
        reference["prc"],
        atol=_PRC_ATOL,
        rtol=_PRC_RTOL,
    )
    aligned_phase = (jaxcont_phase + phase_shift) % 1.0

    extended_phase = np.concatenate(
        [reference["phase_fraction"] - 1.0, reference["phase_fraction"], reference["phase_fraction"] + 1.0]
    )
    extended_prc = np.tile(reference["prc"], 3)
    extended_dprc = np.tile(reference["dprc"], 3)
    reference_prc_aligned = np.interp(aligned_phase, extended_phase, extended_prc)
    reference_dprc_aligned = np.interp(aligned_phase, extended_phase, extended_dprc)

    prc_diagnostics = _diagnose(jaxcont_prc, reference_prc_aligned, atol=_PRC_ATOL, rtol=_PRC_RTOL)
    # dPRC comparison kept literal (dprc_curve's actual output against
    # MatCont's actual dprc column) -- see module docstring, point 2: these
    # are different physical quantities, so this is expected to fail, and
    # is reported as such rather than papered over.
    dprc_diagnostics = _diagnose(jaxcont_dprc, reference_dprc_aligned, atol=_PRC_ATOL, rtol=_PRC_RTOL)

    return CaseResult(
        case_id="MC-PRC-001",
        checks={
            "prc_matches_matcont": prc_diagnostics["passed"],
            "dprc_matches_matcont": dprc_diagnostics["passed"],
        },
        observations={
            "converged_alpha": alpha,
            "converged_period": period,
            "phase_shift_applied": phase_shift,
            "prc_max_error": prc_diagnostics["max_error"],
            "prc_max_tolerance": prc_diagnostics["max_tolerance"],
            "dprc_max_error": dprc_diagnostics["max_error"],
            "dprc_max_tolerance": dprc_diagnostics["max_tolerance"],
            "dprc_quantity_mismatch": (
                "MatCont's dPRC is d(PRC)/dt (time derivative, per "
                "LimitCycle/calcPRC.m); JaxCont's dprc_curve is "
                "d(PRC)/d(alpha) (parameter derivative). Not reconcilable "
                "by rescaling -- see task-8-report.md."
            ),
        },
        artifacts={
            "jaxcont_prc": jaxcont_prc,
            "jaxcont_dprc": jaxcont_dprc,
            "reference_prc": reference_prc_aligned,
            "reference_dprc": reference_dprc_aligned,
        },
    )


__all__ = ["run_adaptx_prc_dprc"]
