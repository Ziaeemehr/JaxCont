"""Fresh Lorenz-84 numerical candidates using only the JaxCont public API."""
from functools import lru_cache
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import jaxcont as jc
from jaxcont.bifurcations.curves import unpack_fold_curve, unpack_hopf_curve

from ..systems import lorenz84 as model
from . import CaseResult


class ContractViolation(ValueError):
    """A numerical route cannot satisfy the immutable reviewed selector."""
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _row(u, F, T, segment, point, coordinate):
    u = np.asarray(u)
    J = np.asarray(model.lorenz84_jacobian(u, F, (*model.CONSTANTS, T)))
    return dict(segment_id=segment, point_id=str(point), order=point, parameter=float(coordinate),
                F=float(F), T=float(T), **{f"state_{i}": float(x) for i, x in enumerate(u)},
                residual_norm=float(np.max(np.abs(model.lorenz84_rhs(u, F, (*model.CONSTANTS, T))))),
                unstable_dimension=int(sum(np.linalg.eigvals(J).real > 1e-7)), status="CONVERGED")


def _spectra(row, event_id=""):
    u = np.array([row[f"state_{i}"] for i in range(4)])
    eigenvalues = np.linalg.eigvals(np.asarray(model.lorenz84_jacobian(u, row["F"], (*model.CONSTANTS, row["T"]))))
    return [dict(segment_id=row["segment_id"], point_id=row["point_id"], event_id=event_id,
                 kind="EIGENVALUE", mode_id=str(i), real=float(z.real), imag=float(z.imag),
                 logabs=float(np.log(max(abs(z), np.finfo(float).tiny))), phase=float(np.angle(z)),
                 trivial_mode_method="not-applicable") for i, z in enumerate(eigenvalues)]


def _event(row, kind, identity, **info):
    return dict(row, kind=kind, event_type=kind, event_id=identity,
                bracket_lower=row["parameter"], bracket_upper=row["parameter"], **info)


def _metadata():
    return dict(runtime_version=sys.version.split()[0], package_versions={"jaxcont": jc.__version__, "jax": jax.__version__},
                precision="float64" if jax.config.x64_enabled else "float32", parameter_order=["F", "T"],
                scaling={"time": "original", "arclength": "theta=1; Euclidean augmented state and parameter"},
                seeds=dict(model.F_SWEEP), mesh={"kind": "equilibrium"},
                resolved_controls={"equilibrium": dict(model.F_SWEEP), "codim2": dict(model.CODIM2_CONTROLS)},
                solver_strategies={"method": "JaxCont PALC; extended fold/Hopf charts", "event_selection": "type and named window"},
                conventions=dict(model.CONVENTIONS), chart_status="completed")


@lru_cache(maxsize=2)
def _equilibrium(x64):
    seed = jnp.asarray(model.F_SWEEP["seed_state"])
    # Correct the contract seed without substituting another tool's solution.
    for _ in range(5):
        seed = seed - jnp.linalg.solve(model.lorenz84_jacobian(seed, 3.), model.lorenz84_rhs(seed, 3., model.FIXED_ARGS))
    branches, events = [], []
    for label, endpoint in (("decreasing", -1.5), ("increasing", 3.0000001)):
        result = jc.continuation(jc.bif_problem(model.lorenz84_rhs, u0=seed, p0=3., args=model.FIXED_ARGS),
            p_span=(3., endpoint), settings=jc.ContinuationPar(ds=.002, ds_max=.05,
            max_steps=200, newton_tol=1e-12, theta=1.), events=[jc.Fold(), jc.Hopf()])
        rows = [_row(u, p, .04, label, i, p) for i, (u, p) in enumerate(zip(result.branch.states, result.branch.params))
                if -1.5 <= float(p) <= 3.]
        branches.extend(rows)
        for e in result.events:
            if not (-1.5 <= float(e.p) <= 3):
                continue
            kind = {"fold": "LP", "hopf": "H"}.get(e.kind)
            if kind is None:
                continue
            if kind == "LP":
                u, p, v, ok = jc.fold_point(model.lorenz84_rhs, e.u, e.p, args=model.FIXED_ARGS, tol=1e-12)
            else:
                u, p, q1, q2, omega, ok = jc.hopf_point(model.lorenz84_rhs, e.u, e.p, args=model.FIXED_ARGS, tol=1e-12)
            if not bool(ok):
                raise RuntimeError(f"{kind} refinement failed")
            nearest = min(rows, key=lambda r: np.linalg.norm(np.array([r[f'state_{i}'] for i in range(4)])-np.asarray(u)))
            row = _row(u, p, .04, label, int(nearest["point_id"]), p)
            extra = {"fold_coefficient": float(jc.fold_coefficient(model.lorenz84_rhs, u, p, v, args=model.FIXED_ARGS))} if kind == "LP" else {
                "frequency": float(omega), "first_lyapunov": float(jc.lyapunov_coefficient(model.lorenz84_rhs,u,p,q1,q2,omega,args=model.FIXED_ARGS))}
            events.append(_event(row, kind, f"e{len(events)}", **extra))
    for expected in model.F_SWEEP["expected_events"]:
        model.select_event(events, expected["kind"], expected["window"])
    return branches, events


@lru_cache(maxsize=2)
def _fold(x64):
    _, events = _equilibrium(x64)
    seed = model.select_event(events, "LP", model.EVENT_WINDOWS["fold_seed"])
    u = jnp.array([seed[f"state_{i}"] for i in range(4)])
    prob = jc.fold_curve_problem(model.lorenz84_ft_rhs, u, jnp.array([seed["F"], .04]), free=1, formulation="extended", tol=1e-12, args=model.CONSTANTS)
    result = jc.continuation(prob, p_span=(.04, -.01),
        settings=jc.ContinuationPar(ds=.001, ds_min=1e-5, ds_max=.015, max_steps=30, theta=1., newton_tol=1e-12, compute_stability=False),
        events=[jc.BogdanovTakens(raw_f=model.lorenz84_ft_rhs, free=1, curve="fold", args=model.CONSTANTS)])
    rows=[]
    for i, (state, T) in enumerate(zip(result.branch.states, result.branch.params)):
        u, F, *_ = unpack_fold_curve(state, 4)
        rows.append(_row(u, F, T, "fold", i, T))
    return result, rows


def produce(case_id):
    """Return v2 tables; failures raise and never masquerade as complete data."""
    x64 = jax.config.x64_enabled
    if not x64:
        # The contract's Newton tolerance (1e-12) is below float32 resolution.
        raise RuntimeError("Lorenz-84 reference producers run in float64; set jax_enable_x64")
    metadata = _metadata()
    if case_id == "lorenz84.eq-f":
        rows, events = _equilibrium(x64)
    elif case_id == "lorenz84.fold-ft":
        result, rows = _fold(x64)
        events=[]
        for e in result.events:
            if e.kind == "bogdanov_takens":
                p = e.info["p"]
                nearest=min(rows, key=lambda r: abs(r["T"]-float(p[1])))
                events.append(_event(_row(e.u, *p, "fold", int(nearest["point_id"]), p[1]), "BT", f"bt{len(events)}"))
    elif case_id == "lorenz84.hopf-ft":
        _, eq_events = _equilibrium(x64)
        seed = model.select_event(eq_events, "H", model.EVENT_WINDOWS["hopf_seed"])
        u = jnp.array([seed[f"state_{i}"] for i in range(4)])
        prob = jc.hopf_curve_problem(model.lorenz84_ft_rhs, u, jnp.array([seed["F"], .04]), free=1, formulation="extended", tol=1e-12, args=model.CONSTANTS)
        result = jc.continuation(prob, p_span=(.04, -.01), settings=jc.ContinuationPar(ds=.001, ds_min=1e-5, ds_max=.015,
            max_steps=30, theta=1., newton_tol=1e-12, compute_stability=False))
        rows=[]
        for i, (state, T) in enumerate(zip(result.branch.states, result.branch.params)):
            u, F, *_ = unpack_hopf_curve(state, 4)
            rows.append(_row(u, F, T, "hopf", i, T))
        events=[]
        metadata["chart_status"] = "Hopf chart boundary near BT" if rows[-1]["T"] > -.01 else "completed"
    elif case_id == "lorenz84.bt-ft":
        result, curve = _fold(x64)
        candidates=[e for e in result.events if e.kind == "bogdanov_takens" and .01 <= float(e.info["p"][1]) <= .03]
        if len(candidates) != 1:
            raise RuntimeError(f"expected unique positive-T BT on fold curve, got {len(candidates)}")
        seed=candidates[0]
        u,p,v0,v1,ok=jc.bogdanov_takens_point(model.lorenz84_ft_rhs, seed.u, seed.info["p"], args=model.CONSTANTS, tol=1e-12)
        if not bool(ok):
            raise RuntimeError("BT Newton refinement failed")
        row=_row(u,*p,"bt",0,p[1]); rows=[row]
        J=np.asarray(model.lorenz84_jacobian(u,p[0],(*model.CONSTANTS,p[1])))
        events=[_event(row,"BT","bt", jordan_v0_residual=float(np.max(np.abs(J@v0))),
                     jordan_v1_residual=float(np.max(np.abs(J@v1-v0))),
                     geometric_nullity=int(sum(np.linalg.svd(J,compute_uv=False)<1e-6)))]
        try:
            model.select_event(events,"BT",model.EVENT_WINDOWS["bt_positive_T"])
        except ValueError as exc:
            evidence = dict(canonical_id=case_id, branch=rows, events=events,
                            spectra=_spectra(row, "bt"), metadata=metadata)
            raise ContractViolation(f"Refined BT F={float(p[0])}, T={float(p[1])} outside reviewed BT window", evidence) from exc
    else:
        raise ValueError(case_id)
    if case_id in ("lorenz84.fold-ft", "lorenz84.hopf-ft"):
        domain = [min(r["T"] for r in rows), max(r["T"] for r in rows)]
        metadata.update(observed_domain=domain, expected_domain=model.CODIM2_CONTROLS["expected_T_domain"],
                        coverage_status="COMPLETE" if domain[0] <= -.01 and domain[1] >= .04 else "PARTIAL")
        if metadata["coverage_status"] == "PARTIAL":
            metadata["chart_status"] = "Hopf chart boundary or configured step budget reached" if case_id.endswith("hopf-ft") else "Configured step budget reached before expected T domain"
    spectra=[s for row in rows for s in _spectra(row)] + [s for e in events for s in _spectra(e,e["event_id"])]
    return dict(canonical_id=case_id, branch=rows, events=events, spectra=spectra, metadata=metadata)


def export(case_id, output_dir):
    import json
    from ...artifacts import candidate_directory
    from ...candidates import write_candidate
    output = candidate_directory(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").unlink(missing_ok=True)
    try:
        bundle = produce(case_id)
    except Exception as exc:
        failure = dict(status="FAIL", reason=str(exc))
        if isinstance(exc, ContractViolation):
            failure["evidence"] = exc.evidence
        (output / "failure.json").write_text(json.dumps(failure, indent=2))
        raise
    return write_candidate(bundle, output, tool="jaxcont", driver=__file__, equation=model.__file__)


def run(case_id):
    bundle=produce(case_id)
    return CaseResult(case_id, {"max_residual": max(r["residual_norm"] for r in bundle["branch"])},
                      {"v2": bundle}, {"bundle": bundle, "events": bundle["events"], "states": [[r[f"state_{i}"] for i in range(4)] for r in bundle["branch"]],
                      "parameters": [r["parameter"] for r in bundle["branch"]]})


def run_equilibria(): return run("lorenz84.eq-f")
def run_fold(): return run("lorenz84.fold-ft")
def run_hopf(): return run("lorenz84.hopf-ft")
def run_bt(): return run("lorenz84.bt-ft")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["lorenz84." + name for name in ("eq-f", "fold-ft", "hopf-ft", "bt-ft")])
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args()
    export(options.case, options.output_dir)
