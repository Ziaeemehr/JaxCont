"""Numerical promotion/report metrics on explicit comparison windows.

All sheets retain continuation order. Only orientation within a monotone sheet
is reversed for interpolation; different fold arms are never merged.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import make_interp_spline
from .artifacts import _monotone_segments, _sparse_native_query

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
    def identity(row):
        return row.get("segment_id", "legacy"), str(row.get("point_id", row.get("point")))
    excluded = set()
    if case.get("comparison_passage") == "first":
        selected, _ = _passages(data, case)
        excluded = {identity(r) for r in data.get("branch", [])} - {identity(r) for r in selected}
    for row in data.get("events", []):
        if identity(row) in excluded:
            continue
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
        metrics["unrefined_detections"] = _metric(sum((missing & guesses).values()), None,
            "; ".join(_unrefined_reason(case, kind) for kind in sorted(missing & guesses)))
    # Every residual must be finite (a NaN after a finite value must not be
    # dropped by max), and refined events meet the same budget as branch
    # points (review B-01).
    for name, records in (("residual", inside), ("event_residual", events)):
        values = [r.get("residual_norm") for r in records if r.get("residual_norm") not in (None, "")]
        if not values or (name == "residual" and len(values) != len(records)):
            continue
        values = np.abs(np.asarray(values, dtype=float))
        metrics[name] = (_metric(float(values.max()), tol.get("residual_atol", 1e-7)) if np.all(np.isfinite(values))
                         else _metric(None, tol.get("residual_atol", 1e-7), "nonfinite residual"))
    if case.get("spectrum_required", True) and not data.get("spectra"):
        metrics["spectra_present"] = _metric(None, 0, "required spectra absent")
    # Spectra of the compared points must be finite whether or not the case
    # declares a spectral budget (review B-03).
    compared = {(r.get("segment_id", "legacy"), str(r.get("point_id", r.get("point")))) for r in inside}
    compared_events = {str(e.get("event_id", e.get("event_index", ""))) for e in events}
    def finite_mode(m):
        try:
            return np.isfinite(float(m["real"])) and np.isfinite(float(m["imag"]))
        except (KeyError, ValueError, TypeError):
            return False
    nonfinite = sum(not finite_mode(m) for m in data.get("spectra", [])
                    if (m.get("segment_id", "legacy"), str(m.get("point_id", m.get("point")))) in compared
                    or (m.get("event_id") and str(m["event_id"]) in compared_events))
    if nonfinite:
        metrics["spectra_finite"] = _metric(None, 0, f"{nonfinite} nonfinite modes at compared points")
    # A period-doubled branch must be a genuine doubling, not the base orbit
    # traversed twice: its half-period difference must reach the declared
    # minimum somewhere on the branch (it vanishes at the PD point itself).
    doubled = case.get("doubled_shape")
    if doubled:
        values = [float(r["half_period_difference"]) for r in data.get("branch", [])
                  if r.get("segment_id") == doubled["segment"] and r.get("half_period_difference") not in (None, "")]
        deficit = max(doubled["min_half_period_difference"] - max(values), 0.) if values else None
        metrics["doubled_shape"] = _metric(deficit, 0, "max half-period difference on the doubled branch")
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
    if column == "residual_norm":
        return tol.get("residual_atol")
    if column.startswith("state_"):
        return tol.get("extrema_atol") if column.endswith(("_min", "_max")) else tol.get("state_atol")
    return tol.get(column + "_atol")


def _passages(data, case):
    """Preserve native order across chart subdivisions of a seeded curve."""
    rows = data.get("branch", [])
    if case.get("comparison_passage") != "first":
        return rows, []
    bounds = window(case)
    inside = [bounds[0] <= float(r["parameter"]) <= bounds[1] for r in rows]
    runs = []
    for i, yes in enumerate(inside):
        if yes and (i == 0 or not inside[i-1]):
            runs.append([i, i])
        elif yes:
            runs[-1][1] = i
    if not runs:
        return [], []
    start, end = runs[0]
    return rows[max(0, start-1):min(len(rows), end+2)], [
        {"point_ids": [r.get("point_id", r.get("point")) for r in rows[a:b+1]],
         "parameter_range": [min(float(r["parameter"]) for r in rows[a:b+1]),
                             max(float(r["parameter"]) for r in rows[a:b+1])]}
        for a, b in runs[1:]]


def passage_information(data, case):
    return {"policy": case.get("comparison_passage", "all"),
            "later_reentries": _passages(data, case)[1]}


def _regular_coordinate(x, y, preferred=None):
    """Maximize range / max(1, magnitude) on both compared sheets."""
    candidates = [k for k in x[0].keys() & y[0].keys()
                  if k == "parameter" or k.startswith("state_") or k == preferred]
    def distinct(rows):
        result = []
        for row in rows:
            if not result or any(float(row[k]) != float(result[-1][k]) for k in candidates):
                result.append(row)
        return result
    x, y = distinct(x), distinct(y)
    scores = {}
    for k in candidates:
        arrays = [np.array([float(r[k]) for r in rows]) for rows in (x, y)]
        if all(len(v) > 1 and (np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0)) for v in arrays):
            scores[k] = min(np.ptp(v) / max(1., np.max(abs(v))) for v in arrays)
    if preferred in scores:
        return preferred, x, y
    if scores:
        return max(sorted(scores), key=lambda k: (scores[k], k != "parameter")), x, y
    # No globally monotone physical coordinate: use normalized geometric length.
    fields = sorted(candidates)
    values = [np.array([[float(r[k]) for k in fields] for r in rows]) for rows in (x,y)]
    scale = np.ptp(np.concatenate(values), axis=0)
    scale[scale == 0] = 1
    output = []
    for rows, v in zip((x,y), values):
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(v,axis=0)/scale, axis=1))]
        if arc[-1] > 0: arc /= arc[-1]
        output.append([{**r, "arclength": float(a)} for r,a in zip(rows,arc)])
    return "arclength", *output


def _cubic(coordinates, values, query):
    if not np.all(np.isfinite(values)):
        return np.full((len(query), *np.asarray(values).shape[1:]), np.nan)
    if len(coordinates) == 1:
        return np.repeat(np.asarray(values)[:1], len(query), axis=0)
    return make_interp_spline(coordinates, values, k=min(3, len(coordinates)-1))(query)


def _sheets(data, case=None):
    groups = {}
    for row in (_passages(data, case)[0] if case else data.get("branch", [])):
        groups.setdefault("passage" if case and case.get("comparison_passage") == "first" else row.get("segment_id", "legacy"), []).append(row)
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
        if indices and max(float(r["parameter"]) for r in sheet) > bounds[0] and min(float(r["parameter"]) for r in sheet) < bounds[1]:
            clipped = sheet[min(indices):max(indices)+1]
            # A duplicate seed on an outward run is not an additional arm.
            if len(clipped)>1 or case.get("comparison_window",{} ) == {"kind":"point"}:
                result.append(clipped)
    return result


def _mode_error(va, vb, relative):
    """Largest matched difference between two spectra.

    ``relative`` (a case that declares ``multiplier_rtol``) divides by
    ``max(1, |mu|)``: absolute up to the unit circle, relative above it,
    because a multiplier ``exp(lambda*T)`` carries an error proportional to
    its size (torBPC: MatCont and BifurcationKit differ by 0.033 at -53).
    Without the declaration the difference stays absolute.
    """
    ii, jj = linear_sum_assignment(abs(va[:, None]-vb[None, :]))
    difference = abs(va[ii]-vb[jj])
    if relative:
        difference = difference / np.maximum(1., np.maximum(abs(va[ii]), abs(vb[jj])))
    return float(max(difference))


def _spectral_budget(tol):
    """``(limit, relative)``: a declared ``multiplier_rtol`` is relative above
    ``|mu| = 1``; ``spectrum_atol``/``multiplier_atol`` stay absolute."""
    if tol.get("multiplier_rtol") is not None:
        return tol["multiplier_rtol"], True
    return tol.get("spectrum_atol", tol.get("multiplier_atol")), False


def _spectral_curve(data, sheet, coordinate):
    """``(coordinates, values)`` of the native branch spectra on one sheet, or
    ``None`` when a point has no finite spectrum."""
    from examples.validation.artifacts import _unique_spectrum_points
    curve = []
    for row in sheet:
        values = [complex(float(s["real"]), float(s["imag"])) for s in data.get("spectra", [])
                  if str(s.get("point_id", s.get("point"))) == str(row.get("point_id", row.get("point")))
                  and str(s.get("segment_id", "legacy")) == str(row.get("segment_id", "legacy"))
                  and not s.get("event_id") and str(s.get("event_index", -1)) == "-1"]
        if not values or not np.all(np.isfinite(values)):
            return None
        curve.append({coordinate: float(row[coordinate]), "values": values})
    return _unique_spectrum_points(curve, coordinate)


def _spectral_roots(coordinates, values, query):
    coefficients = np.stack([np.poly(v) for v in values])
    return [np.roots(c) for c in _cubic(coordinates, coefficients, query)]


def _dense_spectrum_metric(a, b, x, y, lo, hi, query, coordinate, limit, relative):
    """Spectra at the denser tool's own nodes against the sparser interpolant."""
    curves = [_spectral_curve(d, sheet, coordinate) for d, sheet in ((a, x), (b, y))]
    if any(c is None for c in curves):
        return None  # the sparse-node spectrum metric already reports this
    (dc, dv), (sc, sv) = sorted(curves, key=lambda c: -len(c[0]))
    dc = np.asarray(dc)
    nodes = (dc >= lo) & (dc <= hi) & ~np.isin(dc, query)
    if not np.any(nodes) or len(sc) < 2:
        return None  # no node of the denser tool lies between sparse nodes
    sc = np.asarray(sc)
    full = _spectral_roots(sc, sv, dc[nodes])
    half_c, half_v = _half_grid(sc, np.arange(len(sc)))
    half = _spectral_roots(half_c, [sv[k] for k in half_v], dc[nodes])
    dense = [np.asarray(v) for v, keep in zip(dv, nodes) if keep]
    differences = [_mode_error(f, d, relative) if len(f) == len(d) else np.inf for f, d in zip(full, dense)]
    change = [_mode_error(f, h, relative) if len(f) == len(h) else np.inf for f, h in zip(full, half)]
    return _dense_node_metric(differences, limit, _interval_allowance(sc, dc[nodes], change), "spectrum")


def _spectrum_error(a, b, x, y, query, coordinate="parameter", relative=False):
    from examples.validation.artifacts import _unique_spectrum_points
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
        # Symmetric coefficients remain smooth at eigenvalue collisions and
        # do not depend on the native mode order.
        coefficients = np.stack([np.poly(v) for v in values])
        curves.append([np.roots(c) for c in _cubic(coordinates, coefficients, query)])
    error = 0.
    for va, vb in zip(*curves):
        if len(va) != len(vb):
            return None
        error = max(error, _mode_error(va, vb, relative))
    return error


def _half_grid(nodes, values):
    """Every other native node (always keeping both ends)."""
    keep = np.r_[np.arange(0, len(nodes), 2), len(nodes) - 1]
    keep = np.unique(keep)
    return nodes[keep], np.asarray(values)[keep]


def _interval_allowance(sparse, nodes, change):
    """``change`` (full minus half-grid interpolant) is a pointwise estimate
    that crosses zero inside an interval; use its maximum over the sparse
    interval holding each node and its two neighbours instead."""
    change = np.abs(np.asarray(change, float))
    interval = np.clip(np.searchsorted(sparse, nodes) - 1, 0, len(sparse) - 2)
    peak = np.zeros(len(sparse) - 1)
    np.maximum.at(peak, interval, np.where(np.isfinite(change), change, np.inf))
    local = np.maximum(peak, np.maximum(np.r_[peak[1:], 0.], np.r_[0., peak[:-1]]))
    return local[interval]


def _dense_node_metric(differences, limit, allowance, what):
    """Compare at the denser tool's own nodes (review B-04).

    ``allowance`` estimates the sparser series' interpolation error at each
    node: the change of its cubic interpolant when every other node is
    dropped (an O(h^4) interpolant changes by ~15x its own error). A
    difference within ``limit`` passes; one only explained by the allowance
    is unresolved (PENDING, the sparse series cannot support the budget
    there); a larger one fails."""
    differences, allowance = np.asarray(differences, float), np.asarray(allowance, float)
    if not len(differences) or limit is None:
        return None
    worst = float(np.max(differences))
    if not np.all(np.isfinite(differences)):
        verdict = "FAIL"
    elif worst <= limit:
        verdict = "PASS"
    elif np.all(differences <= limit + allowance):
        verdict = "PENDING"
    else:
        verdict = "FAIL"
    return dict(value=worst if np.isfinite(worst) else None, limit=limit, verdict=verdict,
                reason=f"{what} at the denser tool's nodes; sparse interpolation allowance up to "
                       f"{float(np.max(allowance)) if np.all(np.isfinite(allowance)) else float('inf'):.3g}")


def _pairwise(case, a, b, *, coordinates=None, samples=None):
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
        sampled = {}
        for i, x in enumerate(sa):
            for j, y in enumerate(sb):
                # A budgeted column exported by one tool only is missing
                # evidence, not an inapplicable check (review B-02).
                columns = {k for k in x[0].keys() | y[0].keys()
                           if _limit(k, tol) is not None and k not in ("parameter", "residual_norm")}
                one_sided = {k for k in columns if k not in x[0] or k not in y[0]}
                columns -= one_sided
                if not columns:
                    columns = {"parameter"}
                coordinate, x, y = _regular_coordinate(x, y, preferred=case.get(
                    "comparison_coordinate", "state_0" if case["canonical_id"] == "cubic.eq-fold" else None))
                xx = x if float(x[0][coordinate]) <= float(x[-1][coordinate]) else x[::-1]
                yy = y if float(y[0][coordinate]) <= float(y[-1][coordinate]) else y[::-1]
                if coordinate != "parameter":
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
                    query = _sparse_native_query(xp, yp, lo, hi)
                px = [float(r["parameter"]) for r in x]
                py = [float(r["parameter"]) for r in y]
                plow, phigh = p_bounds if p_bounds else (-np.inf,np.inf)
                metrics = {"endpoint_difference": _metric(max(abs(max(min(px),plow)-max(min(py),plow)),abs(min(max(px),phigh)-min(max(py),phigh))), tol["parameter_atol"])}
                curves = {}
                for column in sorted(one_sided):
                    # Not compared, so no verdict is claimed for it.
                    metrics[column] = _metric(None, None, f"{column} exported by one tool only; not compared")
                for column in sorted(columns):
                    xa = np.array([float(r[column]) for r in xx])
                    ya = np.array([float(r[column]) for r in yy])
                    av, bv = _cubic(xp, xa, query), _cubic(yp, ya, query)
                    # Registry's existing mixed absolute/relative rule.
                    error = np.max(np.maximum(abs(av-bv) - tol.get("rtol", 0.)*np.maximum(abs(av),abs(bv)), 0.))
                    metrics[column] = _metric(error, _limit(column, tol))
                    # The sparse-node comparison cannot see a disagreement
                    # between sparse nodes; compare at the dense nodes too.
                    (dp, dv), (sp, sv) = sorted(((xp, xa), (yp, ya)), key=lambda c: -len(c[0]))
                    nodes = (dp >= lo) & (dp <= hi) & ~np.isin(dp, query)
                    if np.any(nodes) and len(sp) > 1:
                        full = _cubic(sp, sv, dp[nodes])
                        half = _cubic(*_half_grid(sp, sv), dp[nodes])
                        difference = np.maximum(abs(full - dv[nodes]) - tol.get("rtol", 0.) * np.maximum(abs(full), abs(dv[nodes])), 0.)
                        dense = _dense_node_metric(difference, _limit(column, tol),
                                                   _interval_allowance(sp, dp[nodes], full - half), column)
                        if dense is not None:
                            metrics[f"{column}.dense_nodes"] = dense
                    curves[column] = dict(coordinate=coordinate, x=query.tolist(),
                        error=abs(av-bv).tolist(), limit=_limit(column, tol),
                        budget=(_limit(column, tol) + tol.get("rtol", 0.)*np.maximum(abs(av),abs(bv))).tolist())
                if case["canonical_id"] == "radial.cycle-floquet":
                    def radius(rows):
                        return np.array([max(abs(float(v)) for k,v in r.items() if k.startswith("state_")) for r in rows])
                    av, bv = np.interp(query,xp,radius(xx)), np.interp(query,yp,radius(yy))
                    metrics["radius"] = _metric(np.max(abs(av-bv)), tol["radius_atol"], "Phase 1 observable: maximum absolute sampled coordinate; radius_atol is not a per-axis extrema budget")
                    curves["radius"] = dict(coordinate=coordinate, x=query.tolist(),
                        error=abs(av-bv).tolist(), limit=tol["radius_atol"],
                        budget=np.full(len(query), tol["radius_atol"]).tolist())
                spectral_limit, relative = _spectral_budget(tol)
                if spectral_limit is not None:
                    dense = _dense_spectrum_metric(a, b, xx, yy, lo, hi, query, coordinate, spectral_limit, relative)
                    if dense is not None:
                        metrics["spectrum.dense_nodes"] = dense
                    metrics["spectrum"] = _metric(_spectrum_error(a, b, xx, yy, query, coordinate, relative), spectral_limit,
                                                  "all native modes; no failed modes omitted"
                                                  + ("; multiplier_rtol: relative above |mu| = 1" if relative else ""))
                cost = max((m["value"] / m["limit"] if m["limit"] else 0.)
                           if m["value"] is not None else 1e100 for m in metrics.values())
                costs[i, j] = cost
                measured[i, j] = metrics, coordinate
                sampled[i, j] = curves
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            metrics, coordinate = measured.get((i, j), ({"observable": _metric(None, 0, "no comparable physical observable or overlap")}, None))
            if samples is not None:
                samples.update({f"sheet_{i}.{k}": v for k,v in sampled.get((i,j), {}).items()})
            if coordinates is not None:
                coordinates[f"sheet_{i}"] = coordinate
            result.update({f"sheet_{i}.{k}": v for k, v in metrics.items()})
    # Event matching is one to one within type. Extra native diagnostic labels
    # are retained in the bundle and reported, never relabelled here.
    for kind in sorted(set(case.get("comparison_expected_events", case.get("expected_events", [])))):
        ea = [e for e in event_rows(case, a) if e.get("event_type", e.get("kind")) == kind]
        eb = [e for e in event_rows(case, b) if e.get("event_type", e.get("kind")) == kind]
        has_guess = any(unrefined(e) and e.get("event_type") == kind for d in (a,b) for e in event_rows(case,d,refined=False))
        result[f"{kind}.count_difference"] = _metric(abs(len(ea)-len(eb)), None if has_guess else 0,
            _unrefined_reason(case, kind) if has_guess else None)
        if not ea or len(ea) != len(eb):
            continue
        costs = np.array([[abs(float(x["parameter"])-float(y["parameter"])) for y in eb] for x in ea])
        ii, jj = linear_sum_assignment(costs)
        for i, j in zip(ii, jj):
            x, y = ea[i], eb[j]
            def number(row, column):
                if row.get(column) in (None, ""):
                    return None
                try:
                    return float(row[column])
                except (ValueError, TypeError):
                    return float("nan")  # present but not a number: a failure
            for column in sorted(x.keys() | y.keys()):
                limit = _limit(column, tol)
                if limit is None or column == "residual_norm" or not (applicable(column, x) and applicable(column, y)):
                    continue
                av, bv = number(x, column), number(y, column)
                if av is None and bv is None:
                    continue  # exported by neither tool for this event
                if av is None or bv is None:
                    # One tool reports a budgeted value the other does not
                    # export: missing evidence, never silently dropped (B-02).
                    result[f"{kind}_{i}.{column}"] = _metric(None, None, f"{column} reported by one tool only; not compared")
                    continue
                result[f"{kind}_{i}.{column}"] = _metric(max(abs(av-bv)-tol.get("rtol",0)*max(abs(av),abs(bv)),0), limit)
            def modes(data, event):
                identity = str(event.get("event_id", event.get("event_index", "")))
                return np.array([complex(float(r["real"]), float(r["imag"])) for r in data.get("spectra", [])
                                 if str(r.get("event_id", r.get("event_index", ""))) == identity])
            va, vb = modes(a, x), modes(b, y)
            limit, relative = _spectral_budget(tol)
            if limit is not None:
                error = None
                if len(va) == len(vb) and len(va) and np.all(np.isfinite(va)) and np.all(np.isfinite(vb)):
                    error = _mode_error(va, vb, relative)
                result[f"{kind}_{i}.spectrum"] = _metric(error, limit, "all modes, including flow mode")
    return result


def _unrefined_reason(case, kind):
    """Why an unrefined detection of ``kind`` is excluded, naming the run that
    compares the refined point when the registry declares one."""
    elsewhere = case.get("refined_events_elsewhere", {}).get(kind)
    if elsewhere:
        return f"unrefined {kind} detection excluded; refined {kind} is compared in {elsewhere}"
    return f"unrefined {kind} detection excluded; no refined comparison declared"


def covered_elsewhere(metric):
    """A PENDING item delegated to another run ("... compared in <run-id>") is
    information, not a gap, once that run's registry evidence status is PASS
    (e.g. an unrefined BifurcationKit BT on lorenz84.fold-ft, whose refined BT
    is compared in lorenz84.bt-ft)."""
    import re
    from .registry import load_registry, select_cases
    match = re.search(r"compared in ([a-z0-9-]+\.[a-z0-9-]+)", metric.get("reason") or "")
    if not match:
        return False
    try:
        (run,) = select_cases(load_registry(), [match.group(1)])
    except (ValueError, KeyError):
        return False
    return run.get("evidence_status") == "PASS"
