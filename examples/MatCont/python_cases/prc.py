"""MC-PRC-001: adaptx Hopf limit cycle, PRC cross-check against MatCont 7.6's
Testruns/testadaptPRC.m. See docs/superpowers/specs/2026-08-05-prc-dprc-design.md.

This case cross-validates ``prc_curve`` only. ``dprc_curve`` (JaxCont's
``d(PRC)/d(alpha)``) is deliberately **not** compared against MatCont's
exported ``dPRC`` column here -- they are different mathematical quantities,
not just different units/phase conventions:

- MatCont's exported ``dPRC`` is ``d(PRC)/dt``, a *time* derivative.
  Confirmed by reading ``LimitCycle/calcPRC.m`` in the installed MatCont 7.6
  tree directly: ``dvl(:,i) = -newperiod * J(x_i)' * vl(:,i)`` is the adjoint
  variational ODE's right-hand side (``dZ/dtau = -T . J(x)^T . Z`` in
  normalized time), and the final ``dPRC = dPRC / newperiod`` converts it to
  absolute-time units -- there is no parameter (``alpha``) anywhere in that
  computation. Confirmed numerically too: a central finite difference of the
  *reference* ``prc`` column with respect to time (using the corrected,
  periodicity-consistent ``MC-PRC-001_prc.csv``; see
  ``matlab/export_prc_run.m``) matches the reference ``dprc`` column with
  max abs error ~0.0063 (~0.5% of ``dprc``'s own ~1.3 max magnitude). That
  is *not* the same (tighter) tolerance the PRC-vs-JaxCont cross-check
  itself achieves (~0.0008, see ``_PRC_ATOL`` below) -- it is looser, which
  is expected: this is a first-order-accurate central difference over
  MatCont's own non-uniform (``'Adapt', 1``) mesh, not an exact identity,
  so some truncation error is normal. It is still small enough to confirm
  MatCont's own ``prc``/``dprc`` columns describe the same underlying
  curve (one the time derivative of the other), which is the only claim
  this check is used to support.
- JaxCont's ``dprc_curve`` is ``d(prc_curve)/dp`` -- the derivative with
  respect to the continuation parameter ``alpha``. Its correctness is
  established independently of this case, by ``tests/test_prc.py`` (a
  closed-form circle-system check and a finite-difference-of-independently-
  re-converged-orbits check, both from Task 4) -- not by any MatCont
  cross-check, since MatCont's own PRC/dPRC/Input processor never computes
  this quantity to check against.

Since no unit or sign convention reconciles a time derivative with a
parameter derivative, comparing them here would either always fail for the
wrong reason or (worse) require silently substituting some other quantity
in place of ``dprc_curve``'s actual output -- which would stop testing the
function this validator exists to validate. Full derivation:
task-8-report.md.

**Units (still relevant for PRC itself).** ``prc_curve`` normalizes
``Z(0) . f(x_0, p) = omega = 2*pi/T`` (phase in *radians*, one cycle =
``2*pi`` -- the standard adjoint-PRC convention). MatCont's ``calcPRC.m``
normalizes phase in *cycle fractions* (one cycle = 1 -- matching its own
exported ``phase_fraction`` column, which runs 0..1). The two differ by
exactly a factor of ``2*pi``: ``Z_matcont = Z_jaxcont / (2*pi)``. Applied
below before comparison.
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

# atol tightened from 0.065 to 0.005 (rtol unchanged at 0.05) after fixing
# a real bug in matlab/export_prc_run.m: it paired calcPRC.m's *rotated*
# PRC/dPRC value output (circularly shifted to start at the trajectory's
# x-maximum -- "make the PRC and collocation mesh start at the spike-top")
# with an *un-rotated* phase_fraction label built straight from lds.msh.
# That mislabeling was the actual cause of the previous ~0.10 max error
# this comparison saw (and of MC-PRC-001_prc.csv's periodic endpoint,
# prc(phase=0) != prc(phase=1), not agreeing even though they are the same
# physical point) -- it was NOT a genuine JaxCont-vs-MatCont discretization
# difference, contrary to what the previous version of this comment
# claimed. The fix un-rotates the *value* columns back to their natural
# (pre-calcPRC-rotation) order to match the naturally-unrotated
# phase_fraction, using the same rotation index calcPRC.m itself derives
# from the trajectory's x-maximum (see export_prc_run.m).
#
# With that fix, the true phase offset between MatCont's own (natural,
# un-rotated) time origin and JaxCont's periodic_orbit_problem's time
# origin turned out to be about a quarter-period, not near zero -- so
# _best_phase_shift's search window was widened from +/-0.05 to the full
# circle (window=0.5) to actually find it; see that function's docstring.
#
# Re-swept JaxCont's own mesh resolution (ntst in {20, 40, 60, 80}) against
# the corrected reference and found the worst-case max abs PRC error over
# that sweep is 0.0012 (individual values: 0.00082, 0.00082, 0.00121,
# 0.00090 at ntst=20/40/60/80 respectively, under float32 matmul
# precision, matching this case's own default). Re-ran ntst in {20, 40}
# under JAX_ENABLE_X64=1 (float64 throughout) and got 0.00076/0.00093 --
# essentially unchanged, so this is not a float32-precision artifact
# either. The error does not shrink monotonically with ntst, so it is not
# straightforwardly JaxCont's own discretization error; the most likely
# remaining explanation is the small, expected difference between two
# independently Newton-converged representations of the same continuous
# orbit (JaxCont's uniform ntst mesh vs MatCont's own 'Adapt', 1
# non-uniform mesh), plus each solver's own residual tolerance. 0.005
# covers the observed worst case (0.0012) with a ~4x buffer -- tight
# relative to the old 0.065, but not shaved to the observed number itself.
_PRC_ATOL = 0.005
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


def _circular_interp(
    query_phase: np.ndarray, reference_phase: np.ndarray, reference_values: np.ndarray
) -> np.ndarray:
    """Interpolate a phase-periodic reference curve (``reference_phase`` in
    ``[0, 1)``) at arbitrary query phases, wrapping around the period
    boundary -- shared by ``_best_phase_shift``'s per-shift search and
    ``run_adaptx_prc_dprc``'s final aligned-curve lookups, which would
    otherwise repeat the same circular-extension-then-``np.interp``
    pattern three times."""
    extended_phase = np.concatenate(
        [reference_phase - 1.0, reference_phase, reference_phase + 1.0]
    )
    extended_values = np.tile(reference_values, 3)
    return np.interp(query_phase, extended_phase, extended_values)


def _best_phase_shift(
    jaxcont_phase: np.ndarray,
    jaxcont_prc: np.ndarray,
    reference_phase: np.ndarray,
    reference_prc: np.ndarray,
    *,
    atol: float,
    rtol: float,
    window: float = 0.5,
    steps: int = 20001,
) -> float:
    """Find the circular phase shift that best aligns JaxCont's PRC curve
    onto MatCont's reference curve before comparing.

    An autonomous periodic orbit's phase origin (t=0) is only defined up to
    an arbitrary time-translation. ``periodic_orbit_problem`` anchors phase
    via an integral condition relative to the caller's own initial
    trajectory guess (see its docstring); MatCont's own convention is
    unrelated -- the exported ``phase_fraction`` column is MatCont's
    natural (unrotated) mesh labeling, i.e. wherever its own LC
    continuation happened to place ``t=0`` for this point, several
    continuation stages removed from the Hopf point. Nothing pins these
    two conventions to agree exactly, or even approximately: there is no
    a priori reason the offset between them should be small.

    Resolving that gauge freedom via a reference-informed search before
    the tolerance check is the same pattern this suite's own comparison
    engine already uses for other non-canonical correspondences --
    ``compare.py``'s ``match_spectrum``/``match_events`` use
    ``linear_sum_assignment`` the same way, for eigenvalue/event orderings
    that likewise have no intrinsic ordering. It is not a tolerance
    relaxation: the same ``atol``/``rtol`` is applied to the aligned curves
    afterward, and the search only chooses a single rigid circular shift
    applied uniformly to every point -- it cannot fit away a genuine
    per-point disagreement. The search spans the full circular offset
    range (``window=0.5``, i.e. the whole circle mod 1.0) rather than a
    small neighborhood of zero: an earlier, narrower window (bounded to
    about one JaxCont mesh spacing) happened to work only because a
    since-fixed export bug (see ``export_prc_run.m``'s phase-rotation fix)
    coincidentally left MatCont's mislabeled phase close to JaxCont's own
    convention; the correctly-labeled reference needs the full range (the
    true best offset for MC-PRC-001 is about a quarter-period away).
    """

    def worst_margin(delta: float) -> float:
        query = (jaxcont_phase + delta) % 1.0
        aligned = _circular_interp(query, reference_phase, reference_prc)
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
    """Compare JaxCont's prc_curve on the adaptx system against MatCont's
    own PRC processor output. dprc_curve is computed too (returned in
    observations/artifacts for anyone inspecting the case), but is not
    checked against MatCont's dPRC column -- see the module docstring for
    why that comparison would be meaningless."""
    reference = _load_reference_prc(reference_dir)
    alpha, period = _load_converged_point(reference_dir)

    problem, mesh = _run_adaptx_jaxcont(alpha, period)
    with jax.default_matmul_precision("float32"):
        jaxcont_prc_radian = np.asarray(prc_curve(_adaptx_rhs, mesh, problem.u0, problem.p0))
        jaxcont_dprc_radian = np.asarray(dprc_curve(problem))

    # Unit conversion (see module docstring): MatCont's PRC is in
    # cycle-fraction phase units, JaxCont's is in radian phase units --
    # convert JaxCont's output, not MatCont's, since the reference CSV is
    # the fixed oracle. dprc_curve is converted the same way purely for
    # display/diagnostic consistency (it is not compared against MatCont).
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

    reference_prc_aligned = _circular_interp(
        aligned_phase, reference["phase_fraction"], reference["prc"]
    )
    # Reference dPRC is aligned the same way purely so it can sit alongside
    # jaxcont_dprc in artifacts for anyone inspecting the case (e.g. to see
    # for themselves that the two curves have different shapes, not just
    # different scales) -- not used in any check.
    reference_dprc_aligned = _circular_interp(
        aligned_phase, reference["phase_fraction"], reference["dprc"]
    )

    prc_diagnostics = _diagnose(jaxcont_prc, reference_prc_aligned, atol=_PRC_ATOL, rtol=_PRC_RTOL)

    return CaseResult(
        case_id="MC-PRC-001",
        checks={
            "prc_matches_matcont": prc_diagnostics["passed"],
        },
        observations={
            "converged_alpha": alpha,
            "converged_period": period,
            "phase_shift_applied": phase_shift,
            "prc_max_error": prc_diagnostics["max_error"],
            "prc_max_tolerance": prc_diagnostics["max_tolerance"],
            "dprc_not_cross_validated": (
                "dprc_curve is d(PRC)/d(alpha); MatCont's exported dPRC is "
                "d(PRC)/dt (confirmed via LimitCycle/calcPRC.m). Different "
                "quantities -- see module docstring and task-8-report.md. "
                "dprc_curve's own correctness is validated by "
                "tests/test_prc.py, not by this MatCont cross-check."
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
