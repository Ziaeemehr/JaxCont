# Period-Doubling / Neimark–Sacker Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PeriodDoubling` and `NeimarkSacker` `Event` implementations that detect the two
codim-1 bifurcations Floquet multipliers signal along a periodic branch — a real multiplier crossing
`-1`, and a complex-conjugate pair crossing the unit circle.

**Architecture:** Both events are ordinary `Event` implementations added to the existing
`bifurcations/events.py` (alongside `Fold`/`Hopf`), consuming `BranchPoint.eigenvalues` (which for
periodic branches now holds Floquet multipliers) exactly the way `Hopf` already consumes equilibrium
eigenvalues. No changes to the `Event` protocol, `BranchPoint`, `EventHit`, or `detect_events`. Each
event carries `raw_f`/`mesh` as its own constructor fields (mirroring `Hopf`'s `tolerance` field) so
`refine()` can call `stability.floquet.floquet_multipliers` directly at bisection midpoints, since
`detect_events`'s generic `rhs` parameter is the assembled collocation residual, not the raw ODE.

**Tech Stack:** JAX (`jnp.argmin`/`jnp.nanargmin`/`jnp.where` for candidate selection), reusing
`stability.floquet.floquet_multipliers` (unmodified) and the existing `Event`/`detect_events`
machinery (unmodified).

## Global Constraints

- `Event`, `BranchPoint`, `EventHit`, and `detect_events` are unchanged — both new events are
  ordinary implementations of the existing protocol, not a protocol extension.
- `core/scan_continuation.py`, `core/collocation.py`, `stability/floquet.py`, `solvers/protocols.py`,
  `problems/periodic.py` are not touched in this plan.
- No branch switching / continuation onto the bifurcating branch — detection only.
- No enforcement (raise) of the equilibrium-branch footgun for `PeriodDoubling`/`NeimarkSacker` —
  documented only (docstring), matching the existing `Hopf`-on-periodic precedent exactly.
- No limit-cycle example scripts — separate future sub-project.
- The near-unit-circle candidate filter (`near_unit_circle: float = 0.5`) is **required, not
  optional** — found necessary by end-to-end verification during design (see Task 1). Without it,
  `argmin`/`nanargmin` over "closest to -1"/"closest to the unit circle" can silently switch which
  physical multiplier it tracks once the true one moves far enough away, producing a real false
  positive (verified: two detections instead of one on a sweep that should only cross once). Do not
  simplify this filter away.
- All code in this plan was prototyped and numerically verified before being written here — both the
  closed-form multiplier predictions and the actual `Event` classes run end-to-end through
  `jc.continuation()` with `events=[...]`, including the false-positive bug above being found, fixed,
  and re-verified. Exact values from that verification appear in this plan's tests.

---

## Background: reading the existing code

Before starting, the engineer should know:

- `src/jaxcont/bifurcations/events.py` has the `Event` protocol, `BranchPoint`, `EventHit`, and two
  concrete implementations: `Fold` (tangent-based, no eigenvalues) and `Hopf` (eigenvalue-based,
  three-way bisection `refine`). Both are `@dataclass(frozen=True)`. `BranchPoint.eigenvalues` is an
  `Optional[Array]` — for `kind="periodic"` branches with `compute_stability=True`, this now holds
  real Floquet multipliers (a prior sub-project). `detect_events(events, params, states, tangents,
  eigenvalues, rhs, *, ds, tolerance, max_iterations)` builds a `BranchPoint` per branch point, scans
  each event's `test_function` for sign changes between consecutive points, and calls `refine` on each
  crossing.
- `src/jaxcont/stability/floquet.py` has `floquet_multipliers(raw_f, mesh, U, p, eigen_solver=...) ->
  Array` (`raw_f` is the ODE right-hand side, `mesh` a `Collocation`, `U` the flat collocation state —
  same convention `periodic_orbit_problem` uses for `BifProblem.u0`).
- `src/jaxcont/bifurcations/period_doubling.py` is a dead, pre-`Event`-protocol stub
  (`PeriodDoublingBifurcation`, operating on the legacy `ContinuationSolution`) — deleted in Task 2.
- `tests/test_bifurcations.py` is where `Fold`/`Hopf`'s own unit tests already live (hand-built
  `BranchPoint`s, no full `continuation()` run needed for `test_function`/`refine` correctness).
- `tests/test_periodic_orbit_continuation.py` shows the pattern for full `jc.continuation()`
  integration tests against a periodic `BifProblem` (`periodic_orbit_problem` + `Collocation` +
  `settings=jc.ContinuationPar(compute_stability=..., ds=..., newton_tol=1e-5, ...)`).

---

### Task 1: `PeriodDoubling`/`NeimarkSacker` classes + test-function unit tests

**Files:**
- Modify: `src/jaxcont/bifurcations/events.py`
- Test: `tests/test_bifurcations.py`

**Interfaces:**
- Consumes: `stability.floquet.floquet_multipliers` (existing, unmodified) — imported into
  `events.py` for `refine()` to call.
- Produces: `PeriodDoubling(raw_f, mesh, kind="period_doubling", tolerance=1e-6,
  near_unit_circle=0.5)` and `NeimarkSacker(raw_f, mesh, kind="neimark_sacker", tolerance=1e-6,
  near_unit_circle=0.5)` — both `Event` implementations with `test_function(point) -> float` and
  `refine(left, right, index, rhs, *, tolerance, max_iterations) -> EventHit`. Task 2 imports both
  from `jaxcont.api`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bifurcations.py` (add `PeriodDoubling, NeimarkSacker` to the existing
`from jaxcont.bifurcations.events import ...` line at the top of the file):

```python
def test_period_doubling_test_function_finds_real_multiplier_near_minus_one():
    pd = PeriodDoubling(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, -0.8 + 0j, -0.8 + 0j]),
    )
    assert jnp.isclose(pd.test_function(point), 0.2, atol=1e-6)


def test_period_doubling_test_function_at_exact_bifurcation():
    pd = PeriodDoubling(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, -1.0 + 0j, -1.0 + 0j]),
    )
    assert jnp.isclose(pd.test_function(point), 0.0, atol=1e-6)


def test_period_doubling_near_unit_circle_filter_excludes_far_multipliers():
    # Regression for the false-positive bug found during design: once a real
    # multiplier moves far enough past -1 (here -2.776, |real+1|=1.776), an
    # unrelated multiplier that merely sits near a roughly constant distance
    # from -1 (the decaying xy multiplier ~3.4e-6, |3.4e-6+1|~1.0) must NOT
    # be picked as "closer to -1" just because 1.0 < 1.776 -- both are
    # outside near_unit_circle=0.5 (|mag-1| = 1.776 and ~1.0 respectively),
    # so nothing should be selected and test_function must return nan, not a
    # spurious finite value that could register a false sign-change.
    pd = PeriodDoubling(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, -2.776 + 0j, -2.776 + 0j]),
    )
    assert jnp.isnan(pd.test_function(point))


def test_period_doubling_test_function_no_real_candidate_returns_nan():
    pd = PeriodDoubling(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 0.5 + 0.8j, 0.5 - 0.8j]),
    )
    assert jnp.isnan(pd.test_function(point))


def test_neimark_sacker_test_function_finds_complex_pair_near_unit_circle():
    ns = NeimarkSacker(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, 0.6 + 0.8j, 0.6 - 0.8j]),
    )
    assert jnp.isclose(ns.test_function(point), 0.0, atol=1e-6)


def test_neimark_sacker_test_function_below_unit_circle():
    ns = NeimarkSacker(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, 0.3 + 0.4j, 0.3 - 0.4j]),
    )
    assert jnp.isclose(ns.test_function(point), -0.5, atol=1e-6)


def test_neimark_sacker_test_function_no_complex_candidate_returns_nan():
    ns = NeimarkSacker(raw_f=lambda u, p, args: u, mesh=None)
    point = BranchPoint(
        p=0.0, u=jnp.zeros(1),
        eigenvalues=jnp.array([1.0 + 0j, 3.4e-6 + 0j, -0.8 + 0j, -0.8 + 0j]),
    )
    assert jnp.isnan(ns.test_function(point))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bifurcations.py -k "period_doubling or neimark_sacker" -v`
Expected: FAIL with `ImportError: cannot import name 'PeriodDoubling'`

- [ ] **Step 3: Add `PeriodDoubling`/`NeimarkSacker` to `events.py`**

Add this import near the top of `src/jaxcont/bifurcations/events.py`, alongside the existing
`from jaxcont.stability.eigenvalue import compute_eigenvalues` line:

```python
from jaxcont.stability.floquet import floquet_multipliers
```

Append these two classes at the end of `src/jaxcont/bifurcations/events.py`:

```python
@dataclass(frozen=True)
class PeriodDoubling(Event):
    """A period-doubling (flip) bifurcation of a periodic orbit.

    Test function: a real Floquet multiplier crosses ``-1`` -- the
    periodic-orbit analogue of ``Hopf``'s imaginary-axis crossing, but a
    magnitude/sign condition on multipliers, not a real-part condition (see
    ``docs/superpowers/specs/2026-07-24-floquet-multipliers-design.md``).
    Only meaningful for ``kind="periodic"`` branches (``point.eigenvalues``
    must be Floquet multipliers, not equilibrium eigenvalues) -- using this
    on an equilibrium branch is the mirror image of the existing
    ``Hopf()``-on-periodic footgun: not enforced by a raise, just
    meaningless.

    ``raw_f``/``mesh`` are required (no meaningful default) because
    ``refine`` must recompute Floquet multipliers at bisection midpoints via
    ``stability.floquet.floquet_multipliers``, whose signature
    (``raw_f, mesh, U, p``) is incompatible with ``detect_events``'s generic
    ``rhs`` parameter (the assembled collocation *residual*, not the raw
    ODE) -- so this event carries its own copy of what
    ``periodic_orbit_problem`` was built with.
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    mesh: Any
    kind: str = "period_doubling"
    tolerance: float = 1e-6
    near_unit_circle: float = 0.5

    def test_function(self, point: BranchPoint) -> float:
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        near_unit = jnp.abs(jnp.abs(mult) - 1.0) < self.near_unit_circle
        candidates_mask = keep & near_unit & (jnp.abs(jnp.imag(mult)) < self.tolerance)
        if not jnp.any(candidates_mask):
            return float("nan")
        candidates = jnp.where(candidates_mask, mult, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.real(candidates) + 1.0))
        return float(jnp.real(mult[idx]) + 1.0)

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
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            t_mid = self.test_function(mid_point)
            # Three-way branch, not "left-half or else" -- see this file's
            # existing Global Constraints (Hopf has the same shape, for the
            # same reason: a two-way version degenerates whenever t_mid
            # lands on an exact zero).
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="period_doubling", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )


@dataclass(frozen=True)
class NeimarkSacker(Event):
    """A Neimark-Sacker (torus) bifurcation of a periodic orbit.

    Test function: a complex-conjugate pair of Floquet multipliers crosses
    the unit circle away from the real axis. See ``PeriodDoubling``'s
    docstring for the shared rationale (equilibrium-branch footgun,
    ``raw_f``/``mesh`` fields, why ``detect_events``'s generic ``rhs`` isn't
    used).
    """

    raw_f: Callable[[Array, Array, PyTree], Array]
    mesh: Any
    kind: str = "neimark_sacker"
    tolerance: float = 1e-6
    near_unit_circle: float = 0.5

    def test_function(self, point: BranchPoint) -> float:
        mult = point.eigenvalues
        trivial_idx = jnp.argmin(jnp.abs(mult - 1.0))
        keep = jnp.arange(mult.shape[0]) != trivial_idx
        near_unit = jnp.abs(jnp.abs(mult) - 1.0) < self.near_unit_circle
        candidates_mask = keep & near_unit & (jnp.abs(jnp.imag(mult)) > self.tolerance)
        if not jnp.any(candidates_mask):
            return float("nan")
        candidates = jnp.where(candidates_mask, mult, jnp.nan)
        idx = jnp.nanargmin(jnp.abs(jnp.abs(candidates) - 1.0))
        return float(jnp.abs(mult[idx]) - 1.0)

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
            mult_mid = floquet_multipliers(self.raw_f, self.mesh, u_mid, p_mid)
            mid_point = BranchPoint(p=p_mid, u=u_mid, eigenvalues=mult_mid)
            t_mid = self.test_function(mid_point)
            if t_left * t_mid < 0:
                p_right, u_right, t_right = p_mid, u_mid, t_mid
            elif t_mid * t_right < 0:
                p_left, u_left, t_left = p_mid, u_mid, t_mid
            else:
                break
        p_bif, u_bif = (p_left + p_right) / 2, (u_left + u_right) / 2
        return EventHit(
            kind="neimark_sacker", p=float(p_bif), u=u_bif, index=index,
            info={"method": "bisection"},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bifurcations.py -k "period_doubling or neimark_sacker" -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Run the full `test_bifurcations.py` file**

Run: `python -m pytest tests/test_bifurcations.py -v`
Expected: all PASS (no regression to existing `Fold`/`Hopf` tests)

- [ ] **Step 6: Commit**

```bash
git add src/jaxcont/bifurcations/events.py tests/test_bifurcations.py
git commit -m "feat: add PeriodDoubling/NeimarkSacker Event implementations"
```

---

### Task 2: Delete dead stub, re-export from `api.py`, end-to-end verification

**Files:**
- Delete: `src/jaxcont/bifurcations/period_doubling.py`
- Modify: `src/jaxcont/api.py`
- Test: `tests/test_period_doubling_neimark_sacker.py` (new)

**Interfaces:**
- Consumes: `PeriodDoubling`, `NeimarkSacker` from `jaxcont.bifurcations.events` (Task 1).
- Produces: `jc.PeriodDoubling`, `jc.NeimarkSacker` — public re-exports, usable in
  `events=[jc.PeriodDoubling(raw_f=..., mesh=...)]` the same way `jc.Fold()`/`jc.Hopf()` are today.

- [ ] **Step 1: Delete the dead stub**

```bash
rm src/jaxcont/bifurcations/period_doubling.py
```

(Read it first if you want to confirm what's being removed: a pre-`Event`-protocol,
`ContinuationSolution`-based `PeriodDoublingBifurcation` class. Its only reusable idea, the
`real(multiplier) + 1` test-function formula, is already in Task 1's `PeriodDoubling`.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_period_doubling_neimark_sacker.py`:

```python
"""
End-to-end tests for jc.PeriodDoubling/jc.NeimarkSacker against a shared 4D
"circle (+) transverse" system: the existing verified 2D circle system
(r'=r(rho-r^2), theta'=1, rho=1 fixed, T=2*pi) plus a decoupled linear
transverse block (w1'=alpha*w1-beta*w2, w2'=beta*w1+alpha*w2). w=0 is an
exact periodic solution for any alpha/beta, and the transverse block's exact
Floquet-multiplier contribution is exp((alpha +/- i*beta)*T) (matrix
exponential of a constant matrix):
  - beta = pi/T: multiplier = -exp(alpha*T), real, crosses -1 at alpha=0
    (period-doubling ground truth).
  - beta = 0.3 (not a multiple of pi/T): multiplier = exp(alpha*T)*exp(+/-i*beta*T),
    complex pair, |multiplier| crosses 1 at alpha=0 (Neimark-Sacker ground truth).
alpha is the continuation parameter (p) in every test below. See
docs/superpowers/specs/2026-07-24-period-doubling-neimark-sacker-design.md.
"""

import numpy as np
import jax.numpy as jnp

import jaxcont as jc
from jaxcont.core.collocation import Collocation
from jaxcont.problems.periodic import periodic_orbit_problem

RHO = 1.0
T_EXACT = 2 * np.pi
BETA_PD = np.pi / T_EXACT
BETA_NS = 0.3


def _make_rhs(beta):
    def rhs(u, p, args):
        x, y, w1, w2 = u[0], u[1], u[2], u[3]
        r2 = x * x + y * y
        alpha = p
        dx = (RHO - r2) * x - y
        dy = (RHO - r2) * y + x
        dw1 = alpha * w1 - beta * w2
        dw2 = beta * w1 + alpha * w2
        return jnp.array([dx, dy, dw1, dw2])
    return rhs


def _build_problem(beta, alpha0):
    t_traj = np.linspace(0, T_EXACT, 60, endpoint=False)
    x = np.sqrt(RHO) * np.cos(t_traj)
    y = np.sqrt(RHO) * np.sin(t_traj)
    u_traj = np.stack([x, y, np.zeros_like(t_traj), np.zeros_like(t_traj)], axis=1)
    mesh = Collocation(ntst=10, ncol=4)
    rhs = _make_rhs(beta)
    prob = periodic_orbit_problem(
        rhs, jnp.asarray(u_traj), jnp.asarray(t_traj), T_EXACT, alpha0, mesh
    )
    return prob, mesh, rhs


def _sweep(beta, event_cls, span):
    prob, mesh, rhs = _build_problem(beta, alpha0=span[0])
    sol = jc.continuation(
        prob, p_span=span,
        settings=jc.ContinuationPar(
            compute_stability=True, ds=0.02, max_steps=50, newton_tol=1e-5
        ),
        events=[event_cls(raw_f=rhs, mesh=mesh)],
    )
    return sol


def test_period_doubling_detects_bifurcation_at_alpha_zero():
    # Verified during design: narrow sweep -0.05..0.05 detects exactly one
    # hit at p~-4.6e-7.
    sol = _sweep(BETA_PD, jc.PeriodDoubling, span=(-0.05, 0.05))
    assert sol.branch.n_valid > 1
    assert len(sol.events) == 1
    assert sol.events[0].kind == "period_doubling"
    assert abs(sol.events[0].p) < 1e-4


def test_neimark_sacker_detects_bifurcation_at_alpha_zero():
    # Verified during design: narrow sweep -0.05..0.05 detects exactly one
    # hit at p~-4.6e-7.
    sol = _sweep(BETA_NS, jc.NeimarkSacker, span=(-0.05, 0.05))
    assert sol.branch.n_valid > 1
    assert len(sol.events) == 1
    assert sol.events[0].kind == "neimark_sacker"
    assert abs(sol.events[0].p) < 1e-4


def test_period_doubling_zero_false_positives_on_neimark_sacker_system():
    sol = _sweep(BETA_NS, jc.PeriodDoubling, span=(-0.05, 0.05))
    assert sol.events == []


def test_neimark_sacker_zero_false_positives_on_period_doubling_system():
    sol = _sweep(BETA_PD, jc.NeimarkSacker, span=(-0.05, 0.05))
    assert sol.events == []


def test_period_doubling_near_unit_circle_filter_prevents_double_detection():
    # Regression for the false-positive bug found during design: a wider
    # sweep (-0.1..0.3) that pushes the transverse multiplier well past -1
    # must still report exactly ONE detection (at alpha=0), not two --
    # without the near_unit_circle filter this produced a spurious second
    # hit at p~0.110, where the "closest to -1" argmin silently switched to
    # tracking the unrelated, always-decaying xy multiplier (~3.4e-6)
    # instead of the true transverse candidate.
    sol = _sweep(BETA_PD, jc.PeriodDoubling, span=(-0.1, 0.3))
    assert len(sol.events) == 1
    assert abs(sol.events[0].p) < 1e-4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_period_doubling_neimark_sacker.py -v`
Expected: FAIL with `AttributeError: module 'jaxcont' has no attribute 'PeriodDoubling'`

- [ ] **Step 4: Re-export from `api.py`**

In `src/jaxcont/api.py`, change:

```python
from jaxcont.bifurcations.events import Event, Fold, Hopf, EventHit, detect_events
```

to:

```python
from jaxcont.bifurcations.events import (
    Event, Fold, Hopf, PeriodDoubling, NeimarkSacker, EventHit, detect_events,
)
```

and change:

```python
__all__ = [
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "Solvers",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "EventHit",
    "Branch",
    "ContinuationResult",
]
```

to:

```python
__all__ = [
    "BifProblem",
    "bif_problem",
    "continuation",
    "ContinuationPar",
    "Solvers",
    "ContinuationAlgorithm",
    "PseudoArclength",
    "Natural",
    "Event",
    "Fold",
    "Hopf",
    "PeriodDoubling",
    "NeimarkSacker",
    "EventHit",
    "Branch",
    "ContinuationResult",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_period_doubling_neimark_sacker.py -v`
Expected: all PASS (5 passed)

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest -v`
Expected: all PASS, zero regressions (confirms the dead-stub deletion and `api.py` re-export didn't
break anything, and `Fold`/`Hopf`/equilibrium behavior is untouched)

- [ ] **Step 7: Commit**

```bash
git add -A src/jaxcont/bifurcations/period_doubling.py src/jaxcont/api.py tests/test_period_doubling_neimark_sacker.py
git commit -m "feat: expose PeriodDoubling/NeimarkSacker, delete dead pre-Event-protocol stub"
```

---

## Post-plan state

After this plan: `jc.continuation(periodic_prob, events=[jc.PeriodDoubling(raw_f=..., mesh=...)])` and
`events=[jc.NeimarkSacker(raw_f=..., mesh=...)]` detect the two remaining codim-1 bifurcations of
periodic orbits, alongside the existing `jc.Fold()` (already meaningful for periodic branches) and
`jc.Hopf()` (equilibrium-only, documented footgun). This closes the second-to-last v0.2.0 checklist
item. Remaining: limit-cycle example scripts (Van der Pol, Brusselator) — the last v0.2.0 item, a
separate future sub-project.
