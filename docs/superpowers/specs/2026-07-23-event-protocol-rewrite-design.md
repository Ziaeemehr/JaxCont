# Replace `BifurcationDetector` with an `Event` protocol (fixes issue #7)

**Date:** 2026-07-23

## Goal

Resolve v0.2 engineering-prep item 4 (`ROADMAP.md` "Engineering / architecture
recommendations for v0.2"): replace the monolithic `BifurcationDetector` —
one class doing sign-change scanning, bisection, and dedup for every
bifurcation type at once — with small, independently-testable `Event`
implementations (`Fold`, `Hopf`) matching `ARCHITECTURE.md` §4.7's sketch.
This is also the fix for issue #7 (`ROADMAP.md` "Known issues" #7):
`BifurcationDetector` produces duplicate/spurious fold-vs-Hopf flags near
closely-spaced or lower-quality crossings, found 2026-07-18 while
cross-validating `example_02_lorenz.py`/`example_05_neural_mass.py` against
real BifurcationKit.jl v0.5.2 runs.

**Root cause of issue #7:** `FoldBifurcation.test_function` picks the
eigenvalue closest to zero by magnitude and returns its real part, without
checking whether that eigenvalue is real or part of a complex pair. Near a
low-frequency Hopf point, the crossing complex pair's magnitude can be the
smallest among all eigenvalues, so the same crossing gets flagged as both a
correct Hopf and a spurious Fold. `ARCHITECTURE.md` §4.7 already sketches a
better Fold test (tangent's `dp` component / `det(df/du)` sign change) that
doesn't touch eigenvalues at all — the standard AUTO/MatCont technique — and
this rewrite adopts it, fixing the false positive at the root instead of
papering over it with dedup alone.

**Scope decision (made during design, not in the original roadmap wording):**
this task covers the `Event` rewrite and the issue #7 fix, staying
**eager-only** — `events=[...]` under `jax.vmap`/`jax.jit` keeps raising the
same `NotImplementedError` it does today (`api.py:476-484`). Making
detection itself trace-safe is a materially different, larger piece of work
(rewriting Python-loop bisection/sign-scanning into `lax`-traceable control
flow) and becomes its own future roadmap item instead of being bundled in.

## Scope

**In scope:**
1. New `src/jaxcont/bifurcations/events.py`: `BranchPoint`, the `Event`
   protocol, `Fold`, `Hopf`, `EventHit`, and a `detect_events(...)`
   orchestrator (scan for sign changes → refine each crossing → dedup
   near-coincident hits across all requested kinds).
2. `Fold`'s test function becomes the pseudo-arclength tangent's
   parameter-component sign change (no eigenvalues); its refinement always
   uses the extended-system solve (`fold_solve.fold_point`, itself built on
   `solvers/implicit.py:differentiable_root` — see the 2026-07-23
   differentiable-root-primitive plan) instead of bisection.
3. `Hopf`'s test function and bisection-based refinement are carried over
   from today's `HopfBifurcation`, with one bug fix found during design (see
   Design §3): the "no complex eigenvalues" sentinel changes from `inf` to
   `nan`, because `inf` produces a false sign-change (and therefore a
   spurious Hopf flag) at every point where the branch's eigenvalue
   structure transitions from all-real to complex, regardless of whether
   the resulting pair is anywhere near the imaginary axis.
4. A dedup step: after refining all crossings, group by kind, sort each
   group by parameter, and within a kind drop any hit within `2 * abs(ds)`
   of the previous kept hit of the **same kind**. (Verified during design,
   not cross-kind: `example_05_neural_mass.py`'s real fold at E0=-1.865224
   and real Hopf at E0=-1.850125 are only 0.015 apart — both confirmed by
   BifurcationKit.jl, well inside a naive `2*ds=0.04` window. A kind-agnostic
   merge silently drops one of two genuinely distinct, independently-verified
   bifurcations; restricting the merge to same-kind hits avoids that while
   still collapsing same-kind duplicate detections, which is what's actually
   observed today — see "Testing" below.)
5. Delete `src/jaxcont/bifurcations/detector.py` (`BifurcationDetector`) and
   the detection logic in `fold.py`/`hopf.py` (`FoldBifurcation`,
   `HopfBifurcation`, including their non-functional
   `compute_normal_form`/`compute_first_lyapunov_coefficient` placeholder
   stubs). Update `bifurcations/__init__.py` and `jaxcont/__init__.py` to
   drop these exports.
6. `api.py`'s `Event`/`Fold`/`Hopf`/`EventHit` (currently thin `_kind`-tag
   dataclasses at `api.py:215-257`) are deleted; `api.py` re-exports the
   real implementations from `bifurcations/events.py` instead. `_run_scan`'s
   detection call site (`api.py:439-454`) calls `detect_events(...)` with
   the plain `rhs2` callable it already builds, instead of
   `BifurcationDetector(...).detect_along_branch(...)` via
   `_to_legacy_problem`. The `_kind` attribute is renamed to the public
   `kind` (Design §1); `tests/test_taxonomy.py:49-50`, the one other place
   in the tree reading `Fold()._kind`/`Hopf()._kind` directly, updates to
   `.kind` alongside this.
7. Tests: a synthetic ground-truth regression reproducing the exact old-bug
   mechanism (see "Testing" below); `tests/test_bifurcations.py` rewritten
   against the new `Fold`/`Hopf` classes (it currently unit-tests the
   deleted `FoldBifurcation`/`HopfBifurcation` directly); confirming
   `example_02_lorenz.py`/`example_05_neural_mass.py` no longer show the
   documented false positives against their hardcoded BifurcationKit.jl
   reference tables; full existing suite green, including
   `TestVmapSafety`'s `NotImplementedError` behavior under `vmap`.

**Explicitly out of scope:**
- Making event detection trace-safe under `jax.vmap`/`jax.jit` (own future
  roadmap item — see "Scope decision" above).
- `PeriodDoubling`/`LPC`/`NS` `Event` implementations (v0.2/v0.3 feature
  work, not prep).
- A Hopf extended-system solver built on `differentiable_root` (Hopf keeps
  its existing bisection refinement here; the extended-system upgrade is
  separate future work already noted in the differentiable-root-primitive
  spec's "Non-goals" section).
- `LinearSolver`/`EigenSolver` protocols (v0.2 prep item 5, separate task).
- Real normal-form coefficients (`compute_normal_form` et al.) — v0.3 work;
  today's placeholders are deleted as dead code, not replaced.
- Any change to `BifProblem`, `Branch`, `ContinuationResult`'s pytree
  registration, or the `vmap`-safety fix from issue #13.

## Design

### 1. Data model (`bifurcations/events.py`)

```python
@dataclass(frozen=True)
class BranchPoint:
    """One point along a continuation branch, as seen by an Event."""
    p: float
    u: Array
    tangent: Optional[Array]       # (n+1,); last entry is the dp/ds component
    eigenvalues: Optional[Array]   # (n,) complex, or None


@runtime_checkable  # lets isinstance(x, Event) work; verified this raises
                    # TypeError without the decorator (Python's Protocol
                    # default) during design
class Event(Protocol):
    kind: str

    def test_function(self, point: BranchPoint) -> float:
        """Scalar; a sign change between consecutive points signals an event."""
        ...

    def refine(
        self,
        left: BranchPoint,
        right: BranchPoint,
        index: Tuple[int, int],
        rhs: Callable[[Array, float], Array],
        *,
        tolerance: float,
        max_iterations: int,
    ) -> "EventHit":
        """Precisely locate the event between `left` and `right`."""
        ...


@dataclass(frozen=True)
class EventHit:
    kind: str
    p: float
    u: Array
    index: Optional[Tuple[int, int]] = None
    info: dict = field(default_factory=dict)
```

`BranchPoint` and the `Event` protocol depend only on `jax`/`jax.numpy` and
(for `Fold.refine`) `jaxcont.bifurcations.fold_solve` — no dependency on
`api.py` or `core.continuation`, keeping the dependency direction one-way
(`api.py` depends on `bifurcations/events.py`, never the reverse).

### 2. `Fold`

```python
@dataclass(frozen=True)
class Fold(Event):
    kind: str = "fold"

    def test_function(self, point: BranchPoint) -> float:
        return float(point.tangent[-1])

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        u_guess = (left.u + right.u) / 2
        p_guess = (left.p + right.p) / 2
        # fold_point expects f(u, p, args) (3-arg, per fold_solve.py); `rhs`
        # here is the 2-arg (u, p) -> Array callable used throughout this
        # module (matches api.py's rhs2), so adapt with an ignored 3rd arg.
        u_bif, p_bif, null_vector = fold_point(
            lambda u, p, _args: rhs(u, p),
            u_guess, p_guess, tol=tolerance, max_iter=max_iterations,
        )
        return EventHit(
            kind="fold", p=float(p_bif), u=u_bif, index=index,
            info={"null_vector": null_vector, "method": "extended_system"},
        )
```

No eigenvalues anywhere in this class — the whole point. (The 3-arg adapter
is a signature-compatibility detail only; `fold_point`'s `args` is unused
here since `rhs`/`rhs2` already has any problem-level `args` baked in via
closure, same as `detector.py`'s current `fold_extended_system=True` path
does with its own `def rhs(u_eval, p_eval, _args): return
problem.evaluate_rhs(u_eval, p_eval)` wrapper.)

### 3. `Hopf`

```python
@dataclass(frozen=True)
class Hopf(Event):
    kind: str = "hopf"
    tolerance: float = 1e-6

    def test_function(self, point: BranchPoint) -> float:
        eigs = point.eigenvalues
        complex_mask = jnp.abs(jnp.imag(eigs)) > self.tolerance
        if not jnp.any(complex_mask):
            return float("nan")
        complex_eigs = eigs[complex_mask]
        idx = jnp.argmin(jnp.abs(jnp.real(complex_eigs)))
        return float(jnp.real(complex_eigs[idx]))

    def refine(self, left, right, index, rhs, *, tolerance, max_iterations) -> EventHit:
        p_left, p_right = left.p, right.p
        u_left, u_right = left.u, right.u
        t_left = self.test_function(left)
        t_right = self.test_function(right)
        for _ in range(max_iterations):
            if abs(p_right - p_left) < tolerance:
                break
            p_mid = (p_left + p_right) / 2
            alpha = (p_mid - p_left) / (p_right - p_left)
            u_mid = u_left + alpha * (u_right - u_left)
            mid_point = BranchPoint(
                p=p_mid, u=u_mid, tangent=None,
                eigenvalues=_eigenvalues_at(rhs, u_mid, p_mid),
            )
            t_mid = self.test_function(mid_point)
            # Three-way branch (not just "left-half or else"): a two-way
            # version was tried during design and verified broken -- if
            # t_mid lands on an exact zero, "t_left * t_mid < 0" is never
            # true, and a bare `else` then marches p_left monotonically
            # toward p_right every iteration instead of bisecting, converging
            # to the wrong endpoint. `break` on the degenerate case (matches
            # today's BifurcationDetector.locate_bifurcation) avoids that.
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="hopf", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )
```

`_eigenvalues_at(rhs, u, p)` is a small module-level helper: `jacfwd` of
`rhs(·, p)` at `u`, then `jaxcont.stability.eigenvalue.compute_eigenvalues`
— the same two calls `BifurcationDetector._compute_eigenvalues_at_point`
makes today when a problem is available, extracted so both the orchestrator
(building each `BranchPoint`'s `eigenvalues` from the branch's stored
per-step values) and `Hopf.refine` (recomputing at bisection midpoints not
on the original branch) share one implementation.

**`nan`, not `inf`, for "no complex eigenvalues" (found during design,
verified against a real example):** `today's HopfBifurcation.test_function`
returns `float("inf")` when no eigenvalue is complex. Feeding that into a
naive `test_vals[i] * test_vals[i+1] < 0` sign-change scan is a bug:
`inf * (any negative finite number) = -inf < 0`, so the scan reports a
"crossing" at every bracket where the eigenvalue *structure* switches from
all-real to complex — even when the resulting pair's real part is nowhere
near zero. Reproduced directly on `example_05_neural_mass.py`: at the branch
point just past index 11, eigenvalues go from three real values
`(-0.33, -13.03, -9.72)` to a genuine complex pair `-9.455 ± 4.56i` (a real,
non-noise imaginary part) plus one real value. `inf` (no complex pair) times
`-9.455` (that pair's real part, far from the imaginary axis) is negative,
so the old sentinel logic flags a phantom Hopf at that bracket. `nan`
doesn't have this problem: `nan * anything` is `nan`, and `nan < 0` is
`False` in IEEE 754, so the scan skips these transitions for free, with no
change needed to `detect_events`'s generic scanning loop. Confirmed this
was the exact source of the one remaining spurious detection in
`example_05_neural_mass.py`'s comparison table (see "Testing" below) —
switching the sentinel removed it entirely, with no other code changes.

### 4. `detect_events` orchestrator

```python
def detect_events(
    events: Sequence[Event],
    params: Array,
    states: Array,
    tangents: Optional[Array],
    eigenvalues: Optional[Array],
    rhs: Callable[[Array, float], Array],
    *,
    ds: float,
    tolerance: float = 1e-6,
    max_iterations: int = 20,
) -> List[EventHit]:
    points = [
        BranchPoint(
            p=float(params[i]), u=states[i],
            tangent=tangents[i] if tangents is not None else None,
            eigenvalues=eigenvalues[i] if eigenvalues is not None else None,
        )
        for i in range(params.shape[0])
    ]

    hits: List[EventHit] = []
    for event in events:
        test_vals = [event.test_function(pt) for pt in points]
        for i in range(len(points) - 1):
            if test_vals[i] * test_vals[i + 1] < 0:
                hits.append(event.refine(
                    points[i], points[i + 1], (i, i + 1), rhs,
                    tolerance=tolerance, max_iterations=max_iterations,
                ))

    hits.sort(key=lambda h: h.p)
    merge_window = 2.0 * abs(ds)
    deduped: List[EventHit] = []
    last_p_by_kind: dict = {}
    for hit in hits:
        prev_p = last_p_by_kind.get(hit.kind)
        if prev_p is not None and abs(hit.p - prev_p) < merge_window:
            continue
        last_p_by_kind[hit.kind] = hit.p
        deduped.append(hit)
    return deduped
```

Dedup is **same-kind only** (`last_p_by_kind`, not a single running
"previous hit" across all kinds) — see the note under "In scope" item 4
above for why a kind-agnostic merge was tried and found to silently drop
real, independently-verified bifurcations that just happen to sit close
together in parameter space.

This is a direct, eager Python-loop implementation — the same style
`BifurcationDetector` already used, just reorganized into per-kind
`test_function`/`refine` methods plus one shared dedup pass instead of
duplicated per-type loops with ad hoc merge logic. No `jax.jit`/`vmap`
compatibility is claimed or attempted here (see "Scope decision" above).

### 5. `api.py` integration

`api.py:439-454`'s call site becomes:

```python
if len(events) > 0 and eigenvalues is not None:
    from jaxcont.bifurcations.events import detect_events

    hits = detect_events(
        events, params, states, tangents, eigenvalues, rhs2,
        ds=float(settings.ds), tolerance=1e-6,
    )
```

`sol.bifurcations` (on the legacy `ContinuationSolution`) stays dict-shaped
— confirmed required, not just legacy inertia: `viz/core.py:119-124`
(`ContinuationResult.plot()` delegates to `solution.plot()`, which reads
`bif.get("type")`/`bif.get("parameter")`/`bif.get("state")`) and
`ContinuationSolution.get_bifurcations_by_type` (`core/continuation.py:132`,
`bif.get("type")`) both depend on it. So `_run_scan` converts `detect_events`'s
`EventHit`s to that dict shape before assigning `sol.bifurcations`:

```python
sol.bifurcations = [
    {"type": h.kind, "parameter": h.p, "state": h.u, "index": h.index, **h.info}
    for h in hits
]
```

`_to_result` (`api.py:510-535`) is **unchanged** — it already converts
`sol.bifurcations` dicts back into `EventHit`s for `ContinuationResult.events`
(`api.py:518-528`), so this round-trip (`EventHit` → dict → `EventHit`) keeps
`viz/core.py` and `get_bifurcations_by_type` working with zero changes to
either, at the cost of one small, harmless conversion step `_run_scan` now
owns instead of `BifurcationDetector`.

## Testing

1. **Synthetic ground-truth repro of the exact old-bug mechanism:** a 3D
   linear(ized) system with eigenvalues `p ± 0.1i` (a Hopf pair crossing
   `Re=0` at `p=0`) and a third eigenvalue fixed at `-5`. Near `p=0`, the
   Hopf pair's magnitude (~0.1) is far smaller than `5`, so the *old*
   eigenvalue-closest-to-zero fold test would have fired at `p≈0` too (the
   documented issue #7 mechanism). Assert: `Hopf().test_function` crosses
   zero at `p=0` (as expected), `Fold().test_function` (tangent-based) does
   **not** spuriously cross there, and end-to-end `detect_events` on this
   system returns exactly one `hopf` hit and no `fold` hit near `p=0`.
2. **Real cross-validated examples, verified during design (not just
   asserted):** re-running `example_02_lorenz.py` and
   `example_05_neural_mass.py` against this design's actual code (fold
   tangent test + same-kind dedup + `nan` Hopf sentinel, all three fixes
   together) produces, currently on `main`: `example_05` goes from 5 raw
   detections (2 exact-duplicate folds at E0=-1.865224, 1 hopf, 2
   exact-duplicate folds at E0=-1.463027 — the actual current bug shape,
   which has drifted from the 2026-07-18 issue #7 write-up's description of
   *mislabeled*/*spurious-with-no-counterpart* flags, likely because
   subsequent commits like the engine consolidation changed exactly how the
   detector receives branch data) down to exactly 3 clean detections, all
   matching BifurcationKit.jl to the same precision as today
   (fold=-1.865224, hopf=-1.850125≈-1.849986, fold=-1.463027). `example_02`
   goes from 6 raw detections (the F=1.546648 fold alone appearing 3 times)
   down to exactly 4, matching all four `bk_reference` entries. Zero
   unmatched/spurious rows in either comparison table. The plan captures
   these exact verified counts as its acceptance numbers instead of the
   original issue write-up's now-stale description.
3. **`tests/test_bifurcations.py` rewritten** against `Fold`/`Hopf`'s
   `test_function` (replacing its current direct tests of the deleted
   `FoldBifurcation`/`HopfBifurcation`), covering: fold test function on a
   real-eigenvalue case (using a `BranchPoint` with a synthetic tangent),
   Hopf test function on a complex-pair case and a no-complex-eigenvalues
   case (both carried over from today's tests, retargeted).
4. **Existing regression suite unmodified and green:**
   `tests/test_bifurcation_workflow.py`, the `events=[...]` cases in
   `tests/test_functional_api.py` (including `TestVmapSafety`'s
   `NotImplementedError` assertion under `vmap`), `tests/test_gpu_smoke.py`,
   `tests/test_van_der_pol_validation.py`, and the `events=` usages across
   `examples/*.py`.

## Error handling

Unchanged in kind from today: `detect_events` assumes `eigenvalues` is
non-`None` whenever `Hopf` is requested (mirrors today's
`BifurcationDetector` early-return-with-warning when eigenvalues are
unavailable — `api.py`'s caller already gates this via
`if len(events) > 0 and eigenvalues is not None`). No new validation is
introduced; `Fold.refine`'s extended-system solve inherits
`differentiable_root`'s existing non-convergence behavior (silently returns
the last Newton iterate at `max_iterations`, per the 2026-07-23
differentiable-root-primitive spec).

## Non-goals / future work this unblocks

- Trace-safe (`vmap`/`jit`-compatible) event detection — its own future
  roadmap item, now that `Branch.valid` (issue #13) and this clean
  `Event`/`BranchPoint` boundary both exist as groundwork.
- `PeriodDoubling`/`LPC`/`NS` as new `Event` implementations, once collocation
  (the actual v0.2.0 feature) exists to produce periodic-orbit branches for
  them to run against.
- A Hopf extended-system solver on `differentiable_root`, replacing `Hopf`'s
  bisection refinement the same way `Fold`'s already was.
