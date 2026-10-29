"""Numerical promotion/report metrics on explicit comparison windows.

All sheets retain continuation order. Only orientation within a monotone sheet
is reversed for interpolation; different fold arms are never merged.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .artifacts import _monotone_segments

def _metric(value, limit, reason=None):
    finite = value is not None and np.isfinite(value)
    return dict(value=float(value) if finite else None, limit=limit,
                verdict="PENDING" if limit is None else "PASS" if finite and value <= limit else "FAIL", reason=reason)


def window(case, segment=None):
    value = case.get("comparison_window", case.get("expected_domain"))
    if isinstance(value, dict):
        return value.get(segment)
    return value


def unrefined(row):
    return str(row.get("native_status", row.get("status", ""))).lower().startswith("guess")


def applicable(column, row):
    kind = row.get("event_type") or row.get("kind")
    return (column != "frequency" or kind in ("H", "GH", "ZH", "HH")) and (column != "fold_coefficient" or kind == "LP") and (column != "first_lyapunov" or kind in ("H", "GH"))


def event_rows(case, data, *, refined=True):
    result = []
    for row in data.get("events", []):
        bounds = window(case, row.get("segment_id"))
        if bounds and not bounds[0] <= float(row["parameter"]) <= bounds[1]:
            continue
        if refined and unrefined(row):
            continue
        result.append(row)
    return result


def _independent(case, data):
    tol = case["tolerances"]
    if case.get("comparison_window") == {"kind": "prc"}:
        prc = data.get("prc", [])
        return {"prc_present": _metric(0 if prc else None, 0),
                "prc_comparison": _metric(None, None, "phase data present; no complete fresh JaxCont PRC bundle for comparison")}
    sheets = _sheets(data, case)
    rows = [r for sheet in sheets for r in sheet]
    # Bracketing nodes support boundary interpolation; do not count their
    # out-of-window diagnostics as in-window failures.
    inside = [r for r in rows if not window(case, r.get("segment_id")) or
              window(case, r.get("segment_id"))[0] <= float(r["parameter"]) <= window(case, r.get("segment_id"))[1]]
    events = event_rows(case, data)
    metrics = {}
    required = ["parameter", "residual_norm"]
    if "period_atol" in tol: required.append("period")
    if "state_atol" in tol and rows:
        states = sorted(k for k in rows[0] if k.startswith("state_") and not k.endswith(("_min","_max")))
        if not states: metrics["state_present"] = _metric(None, 0, "state budget requires state columns")
        required += states
    for column in required:
        if any(r.get(column) in (None, "") for r in inside):
            metrics[f"missing.{column}"] = _metric(None, 0, "required applicable column is missing")
    if case["canonical_id"] == "lorenz84.eq-f":
        bounds = window(case)
        metrics["required_sheet_count"] = _metric(abs(len(sheets)-2),0)
        for i,sheet in enumerate(sheets):
            coords = [float(r["parameter"]) for r in sheet]
            metrics[f"sheet_{i}.endpoint_gap"] = _metric(max(min(coords)-bounds[0],bounds[1]-max(coords),0),tol["parameter_atol"])
    good = ("CONVERGED", "converged", "PASS", "REFINED")
    unresolved = sum(r.get("status", "CONVERGED") not in good for r in { (r.get("segment_id"), r.get("point_id", r.get("point")), r.get("event_id", "")): r for r in inside + events }.values())
    metrics["unresolved_points"] = _metric(unresolved, 0)
    if data["metadata"].get("coverage_status") == "PARTIAL" and "comparison_window" not in case:
        metrics["partial_coverage"] = _metric(1, 0, "native chart or step budget did not cover the contract")
    if not rows:
        metrics["branch_present"] = _metric(None, 0, "no branch evidence")
    domains = case.get("comparison_window", case.get("expected_domain"))
    if isinstance(domains, list):
        domains = {None: domains}
    elif isinstance(domains, dict) and "kind" in domains:
        domains = {}
    for segment, bounds in (domains or {}).items():
        subset = [r for r in rows if segment is None or r.get("segment_id") == segment]
        gap = max(min(float(r["parameter"]) for r in subset)-bounds[0], bounds[1]-max(float(r["parameter"]) for r in subset), 0.) if subset else None
        metrics[f"{segment or 'domain'}.endpoint_gap"] = _metric(gap, tol.get("parameter_atol", 0))
    from collections import Counter
    expected = Counter(case.get("comparison_expected_events", case.get("expected_events", [])))
    if case["canonical_id"] == "lorenz84.eq-f" and "comparison_expected_events" not in case:
        from .cases.systems.lorenz84 import F_SWEEP
        bounds = window(case)
        expected = Counter(e["kind"] for e in F_SWEEP["expected_events"]
                           if max(e["window"]["F"][0], bounds[0]) <= min(e["window"]["F"][1], bounds[1]))
        # Fold is below the comparison window; its contract selection interval
        # overlaps it, but the event itself is explicitly outside.
        expected.pop("LP", None)
    present = Counter(r.get("event_type") or r.get("kind") for r in events)
    guesses = Counter(r.get("event_type") or r.get("kind") for r in event_rows(case, data, refined=False) if unrefined(r))
    missing = expected-present
    metrics["missing_expected_events"] = _metric(sum((missing-guesses).values()), 0)
    if missing & guesses:
        metrics["unrefined_detections"] = _metric(sum((missing & guesses).values()), None, "unrefined detection; excluded from event location comparisons")
    if inside and all(r.get("residual_norm") not in (None, "") for r in inside):
        metrics["residual"] = _metric(max(abs(float(r["residual_norm"])) for r in inside), tol.get("residual_atol", 1e-7))
    if case.get("spectrum_required", True) and not data.get("spectra"):
        metrics["spectra_present"] = _metric(None, 0, "required spectra absent")
    for table, records in (("branch", inside), ("event", events)):
        for row in records:
            for column, value in row.items():
                limit = _limit(column, tol)
                if limit is None or (table == "event" and not applicable(column, row)):
                    continue
                if value in (None, ""):
                    # Sparse optional fields are absent, not float conversion errors.
                    continue
                try:
                    finite = np.isfinite(float(value))
                except (ValueError, TypeError):
                    finite = False
                if not finite:
                    metrics[f"{table}.finite.{column}"] = _metric(None, 0, "required applicable value is nonfinite")
    if case["canonical_id"] == "lorenz84.bt-ft" and rows:
        # Independently recompute the Jacobian, never infer geometric nullity
        # from a count of eigenvalues alone.
        from .cases.systems.lorenz84 import lorenz84_jacobian, FIXED_ARGS
        r = rows[0]
        jac = np.asarray(lorenz84_jacobian(np.array([float(r[f"state_{i}"]) for i in range(4)]), float(r["F"]), (*FIXED_ARGS[:-1], float(r["T"]))))
        ev = np.sort(np.abs(np.linalg.eigvals(jac)))
        metrics["two_small_eigenvalues"] = _metric(ev[1], tol["spectrum_atol"])
        metrics["geometric_nullity_difference"] = _metric(abs(np.count_nonzero(np.linalg.svd(jac,compute_uv=False) <= tol["spectrum_atol"])-1), 0)
    return metrics


def _limit(column, tol):
    if column in ("parameter", "F", "T"):
        return tol.get("parameter_atol")
    if column == "period":
        return tol.get("period_atol")
    if column == "frequency":
        return tol.get("frequency_atol")
    if column.startswith("state_"):
        return tol.get("extrema_atol") if column.endswith(("_min", "_max")) else tol.get("state_atol")
    return tol.get(column + "_atol")


def _sheets(data, case=None):
    groups = {}
    for row in data.get("branch", []):
        groups.setdefault(row.get("segment_id", "legacy"), []).append(row)
    sheets = [sheet for rows in groups.values() for sheet in _monotone_segments(rows, "parameter")]
    if case is None:
        return sheets
    result = []
    for sheet in sheets:
        bounds = window(case, sheet[0].get("segment_id"))
        if not bounds:
            result.append(sheet)
            continue
        indices = [i for i,r in enumerate(sheet) if bounds[0] <= float(r["parameter"]) <= bounds[1]]
        # Retain adjacent native nodes for interpolation at exact boundaries.
        crossings = [i for i in range(len(sheet)-1) if max(float(sheet[i]["parameter"]),float(sheet[i+1]["parameter"])) >= bounds[0] and min(float(sheet[i]["parameter"]),float(sheet[i+1]["parameter"])) <= bounds[1]]
        if crossings:
            indices += crossings + [i+1 for i in crossings]
        if indices:
            clipped = sheet[min(indices):max(indices)+1]
            # A duplicate seed on an outward run is not an additional arm.
            if len(clipped)>1 or case.get("comparison_window",{} ) == {"kind":"point"}:
                result.append(clipped)
    return result


def _spectrum_error(a, b, x, y, query, coordinate="parameter"):
    from examples.validation.artifacts import _interpolate_spectrum_roots, _unique_spectrum_points
    curves = []
    for data, sheet in ((a, x), (b, y)):
        curve = []
        for row in sheet:
            values = [complex(float(s["real"]), float(s["imag"])) for s in data.get("spectra", [])
                      if str(s.get("point_id", s.get("point"))) == str(row.get("point_id", row.get("point")))
                      and str(s.get("segment_id", "legacy")) == str(row.get("segment_id", "legacy"))
                      and not s.get("event_id") and str(s.get("event_index", -1)) == "-1"]
            if not values:
                linked = [s for s in data.get("spectra", [])
                          if str(s.get("point_id",s.get("point"))) == str(row.get("point_id",row.get("point")))
                          and str(s.get("segment_id","legacy")) == str(row.get("segment_id","legacy"))]
                identity = linked[0].get("event_id",linked[0].get("event_index")) if linked else None
                values = [complex(float(s["real"]),float(s["imag"])) for s in linked if s.get("event_id",s.get("event_index")) == identity]
            if not values or not np.all(np.isfinite(values)):
                return None
            curve.append({coordinate: float(row[coordinate]), "values": values})
        coordinates, values = _unique_spectrum_points(curve, coordinate)
        if coordinate == "state_0" and len(coordinates) >= 3:
            # Cubic control: J=x^2-1 is a quadratic polynomial in the
            # regular state coordinate. Interpolating against p is singular
            # at LP and linear interpolation in x adds a spurious O(ds^2) error.
            from scipy.interpolate import make_interp_spline
            curves.append(make_interp_spline(coordinates, np.asarray(values), k=2)(query))
        else:
            curves.append(_interpolate_spectrum_roots(coordinates, values, query))
    error = 0.
    for va, vb in zip(*curves):
        if len(va) != len(vb):
            return None
        ii, jj = linear_sum_assignment(abs(va[:, None]-vb[None, :]))
        error = max(error, float(max(abs(va[ii]-vb[jj]))))
    return error


def _pairwise(case, a, b):
    """Compare each monotone sheet and each event without merging fold arms."""
    tol = case["tolerances"]
    result = {}
    if case["canonical_id"] == "hopf-normal-form.l1":
        def coefficient(data):
            events = [e for e in data.get("events", []) if e.get("event_type", e.get("kind")) == "H"]
            return float(events[0].get("first_lyapunov", events[0].get("l1_native_real", "nan"))) if events else float("nan")
        result["normal_form_coefficient"] = _metric(abs(coefficient(a)-coefficient(b)),
            tol.get("first_lyapunov_atol"),
            "Native BK b is not implicitly converted; registry must supply the convention and coefficient budget")
    sa, sb = _sheets(a, case), _sheets(b, case)
    result["sheet_count_difference"] = _metric(abs(len(sa) - len(sb)), 0)
    if sa and len(sa) == len(sb):
        # Assignment uses observable values, not unrelated native segment names.
        costs = np.full((len(sa), len(sb)), 1e100)
        measured = {}
        for i, x in enumerate(sa):
            for j, y in enumerate(sb):
                columns = {k for k in x[0].keys() & y[0].keys()
                           if _limit(k, tol) is not None and k not in ("parameter", "residual_norm")}
                if not columns:
                    columns = {"parameter"}
                coordinate = "state_0" if case["canonical_id"] == "cubic.eq-fold" else "parameter"
                xx = x if float(x[0][coordinate]) <= float(x[-1][coordinate]) else x[::-1]
                yy = y if float(y[0][coordinate]) <= float(y[-1][coordinate]) else y[::-1]
                if coordinate != "parameter":
                    columns.discard(coordinate)
                    columns.add("parameter")
                xp = np.array([float(r[coordinate]) for r in xx])
                yp = np.array([float(r[coordinate]) for r in yy])
                bounds = window(case, x[0].get("segment_id")) if coordinate == "parameter" else None
                low, high = bounds if bounds else (-np.inf, np.inf)
                lo, hi = max(xp[0], yp[0], low), min(xp[-1], yp[-1], high)
                p_bounds = window(case, x[0].get("segment_id"))
                if coordinate != "parameter" and p_bounds:
                    # Restrict the regular coordinate to the image of the
                    # requested parameter window on this individual arm.
                    endpoints = []
                    for rr in (xx, yy):
                        pr = np.array([float(r["parameter"]) for r in rr])
                        st = np.array([float(r[coordinate]) for r in rr])
                        if pr[0] > pr[-1]: pr,st = pr[::-1],st[::-1]
                        endpoints.append(sorted(np.interp(np.clip(p_bounds,pr[0],pr[-1]),pr,st)))
                    lo,hi = max(lo,*(e[0] for e in endpoints)), min(hi,*(e[1] for e in endpoints))
                if lo > hi:
                    # Refined point candidates can differ within parameter atol.
                    if len(xx) == len(yy) == 1 and lo - hi <= tol["parameter_atol"]:
                        query = np.array([lo])
                    else:
                        continue
                else:
                    query = np.unique(np.r_[lo, xp[(xp >= lo) & (xp <= hi)], yp[(yp >= lo) & (yp <= hi)], hi])
                px = [float(r["parameter"]) for r in x]
                py = [float(r["parameter"]) for r in y]
                plow, phigh = p_bounds if p_bounds else (-np.inf,np.inf)
                metrics = {"endpoint_difference": _metric(max(abs(max(min(px),plow)-max(min(py),plow)),abs(min(max(px),phigh)-min(max(py),phigh))), tol["parameter_atol"])}
                for column in sorted(columns):
                    xa = np.array([float(r[column]) for r in xx])
                    ya = np.array([float(r[column]) for r in yy])
                    if coordinate == "state_0" and len(xp)>=4 and len(yp)>=4:
                        from scipy.interpolate import CubicSpline
                        av, bv = CubicSpline(xp,xa)(query), CubicSpline(yp,ya)(query)
                    else:
                        av, bv = np.interp(query, xp, xa), np.interp(query, yp, ya)
                    # Registry's existing mixed absolute/relative rule.
                    error = np.max(np.maximum(abs(av-bv) - tol.get("rtol", 0.)*np.maximum(abs(av),abs(bv)), 0.))
                    metrics[column] = _metric(error, _limit(column, tol))
                if case["canonical_id"] == "radial.cycle-floquet":
                    def radius(rows):
                        return np.array([max(abs(float(v)) for k,v in r.items() if k.startswith("state_")) for r in rows])
                    metrics["radius"] = _metric(np.max(abs(np.interp(query,xp,radius(xx))-np.interp(query,yp,radius(yy)))), tol["radius_atol"], "Phase 1 observable: maximum absolute sampled coordinate; radius_atol is not a per-axis extrema budget")
                spectral_limit = tol.get("spectrum_atol", tol.get("multiplier_atol"))
                if spectral_limit is not None:
                    metrics["spectrum"] = _metric(_spectrum_error(a, b, xx, yy, query, coordinate), spectral_limit,
                                                  "all native modes; no failed modes omitted")
                cost = max((m["value"] / m["limit"] if m["limit"] else 0.)
                           if m["value"] is not None else 1e100 for m in metrics.values())
                costs[i, j] = cost
                measured[i, j] = metrics
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            metrics = measured.get((i, j), {"observable": _metric(None, 0, "no comparable physical observable or overlap")})
            result.update({f"sheet_{i}.{k}": v for k, v in metrics.items()})
    # Event matching is one to one within type. Extra native diagnostic labels
    # are retained in the bundle and reported, never relabelled here.
    for kind in sorted(set(case.get("comparison_expected_events", case.get("expected_events", [])))):
        ea = [e for e in event_rows(case, a) if e.get("event_type", e.get("kind")) == kind]
        eb = [e for e in event_rows(case, b) if e.get("event_type", e.get("kind")) == kind]
        has_guess = any(unrefined(e) and e.get("event_type") == kind for d in (a,b) for e in event_rows(case,d,refined=False))
        result[f"{kind}.count_difference"] = _metric(abs(len(ea)-len(eb)), None if has_guess else 0,
            "unrefined detection excluded; refined BT is compared in lorenz84.bt-ft" if has_guess else None)
        if not ea or len(ea) != len(eb):
            continue
        costs = np.array([[abs(float(x["parameter"])-float(y["parameter"])) for y in eb] for x in ea])
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            x, y = ea[i], eb[j]
            for column in sorted(x.keys() & y.keys()):
                limit = _limit(column, tol)
                if limit is not None and applicable(column, x) and applicable(column, y):
                    try:
                        av, bv = float(x[column]), float(y[column])
                    except (ValueError, TypeError):
                        continue
                    result[f"{kind}_{i}.{column}"] = _metric(max(abs(av-bv)-tol.get("rtol",0)*max(abs(av),abs(bv)),0), limit)
            def modes(data, event):
                identity = str(event.get("event_id", event.get("event_index", "")))
                return np.array([complex(float(r["real"]), float(r["imag"])) for r in data.get("spectra", [])
                                 if str(r.get("event_id", r.get("event_index", ""))) == identity])
            va, vb = modes(a, x), modes(b, y)
            limit = tol.get("spectrum_atol", tol.get("multiplier_atol"))
            if limit is not None:
                error = None
                if len(va) == len(vb) and len(va) and np.all(np.isfinite(va)) and np.all(np.isfinite(vb)):
                    ii2, jj2 = linear_sum_assignment(abs(va[:,None]-vb[None,:]))
                    error = max(abs(va[ii2]-vb[jj2]))
                result[f"{kind}_{i}.spectrum"] = _metric(error, limit, "all modes, including flow mode")
    return result

