# 2D Phase-Plane Visualization

**Date:** 2026-07-28
**Status:** Approved, ready for implementation planning
**Scope:** `src/jaxcont/viz/phase_plane.py` (new), `src/jaxcont/viz/portraits.py` (rename),
`src/jaxcont/viz/__init__.py`, `tests/test_viz.py`, one new example.

## Problem

JaxCont continues equilibria and plots bifurcation diagrams and eigenvalue trajectories, but
offers no way to see the state-space geometry those diagrams summarize. For 2D autonomous
systems the phase plane is the natural companion: nullcline intersections *are* the equilibria,
the local vector field *is* the stability the eigenvalues report, and a post-Hopf trajectory
*is* the limit cycle a Hopf marker announces. Users currently build this by hand with
`meshgrid` + `contour` in every notebook.

This is diagnostic visualization, not a continuation algorithm. It adds no numerics to the
solver core.

## Non-goals

Explicitly out of scope, so a future reader knows these were decided and not overlooked:

- **Systems with `n != 2`.** No slices or projections of higher-dimensional systems. See
  "Dimension guard" below for why.
- **DDEs.** A delayed system's state is a history function, so a 2D plot of `(x, y)` is a
  projection, not a phase portrait, and its nullclines describe only equilibria. DDE support is
  a continuation-formulation problem (history specification, characteristic roots,
  delay-dependent stability), not a plotting one.
- **Automatic equilibrium discovery.** No grid-seeded Newton inside `viz`. Equilibria come from
  a `ContinuationResult`.
- **Separatrices, invariant manifolds, basin classification, limit-cycle detection, symbolic
  nullclines.** These are what "phase-plane analysis" can imply; this feature is deliberately
  the narrower "phase-plane visualization".
- **A new ODE solver, and any new dependency.** See "Trajectories".

## Architecture

New module `src/jaxcont/viz/phase_plane.py`, alongside the existing `core.py`, `portraits.py`,
and `styles.py`. Nothing in `jaxcont.core`, `jaxcont.api`, or the solver packages changes: the
viz layer stays a pure consumer of `BifProblem` and `ContinuationResult`, which is the boundary
`docs/superpowers/specs/2026-07-22-viz-module-design.md` already established.

The whole feature rests on one function that already exists: `BifProblem.as_rhs(p)`
(`src/jaxcont/api.py`) returns an autonomous `rhs(u)` frozen at parameter `p`. Everything below
is grid evaluation of that callable plus matplotlib.

**Evaluation is JAX, rendering is matplotlib.** Grids are built with `jnp.meshgrid`, evaluated
with a single `jax.vmap` over the flattened `(resolution**2, 2)` point array, and reshaped back.
The result is converted once with `np.asarray` before it reaches matplotlib. Nothing in this
module is jittable end-to-end or differentiable, and its docstrings say so rather than implying
otherwise — matplotlib is NumPy-land, so the module is too.

**Dimension guard.** A shared private `_require_2d(problem)` raises `NotImplementedError` when
`problem.u0.shape != (2,)`, naming the actual dimension:

```
NotImplementedError: Phase-plane visualization supports 2D autonomous systems;
this problem has n=4. See docs for why slices of higher-dimensional systems are
not drawn.
```

Every public entry point calls it first. The rationale for strict 2D: on a slice through an
N-D system the zero-contours are not nullclines of the full system, their intersections are not
generally equilibria, and the arrows are a projection of the true flow. Every curve this module
draws is a true nullcline and every arrow a true tangent. The slice variant remains available
later as purely additive keyword arguments (`state_indices=`, `fixed=`), requiring no change to
the API defined here.

## Public API

Composable primitives, plus one convenience wrapper that composes them. Every function accepts
`ax=None` (creating a figure when omitted) and returns the `Figure`, matching the existing
signatures in `portraits.py` and `core.py`.

```python
plot_nullclines(problem, p, xlim, ylim, *, resolution=200, colors=None,
                labels=True, ax=None, **kwargs)

plot_vector_field(problem, p, xlim, ylim, *, density=20, normalize=True,
                  ax=None, **kwargs)

plot_streamlines(problem, p, xlim, ylim, *, resolution=100, ax=None, **kwargs)

plot_equilibria(result, p, *, atol=None, ax=None, **kwargs)

plot_trajectory(problem, u0, p, t_span, *, n_points=1000, rtol=1e-8, atol=1e-10,
                arrow=True, ax=None, **kwargs)

plot_phase_plane(problem, p, xlim, ylim, *, result=None, nullclines=True,
                 vector_field=True, streamlines=False, trajectories=None,
                 ax=None)
```

### `plot_nullclines`

Draws `ax.contour(X, Y, F[..., 0], levels=[0])` and the same for `F[..., 1]`. The two nullclines
get distinct colors from the project's existing colorblind-safe palette (the hex values already
used in `styles.py` and `portraits.py`): `#0072B2` for the `x`-nullcline, `#D55E00` for the
`y`-nullcline. With `labels=True`, axis labels and legend entries use `problem.state_names` when
present, falling back to `state[0]` / `state[1]` — the same convention `plot_continuation` uses.

### `plot_vector_field`

`ax.quiver` on a coarser `density x density` grid than the nullcline grid. `normalize=True`
scales arrows to unit length so direction stays legible where the field is stiff, and colors
them by `log10` of the true magnitude so the speed information is not lost. `normalize=False`
draws true-magnitude arrows.

### `plot_streamlines`

`ax.streamplot`, which needs strictly monotonic 1-D grid coordinates and NumPy arrays — both
already satisfied by the evaluation path. Kept alongside `plot_vector_field` despite the overlap
because matplotlib gives it to us for nearly free and streamlines read better for smooth flows
while quiver reads better on top of nullclines.

### `plot_equilibria`

Consumes a `ContinuationResult` (or a bare `Branch`), never a problem — no root-finding in the
viz layer.

Selecting the points at parameter `p` must handle a branch that has passed a fold, where two or
more branch points share the same `p`. `Branch.at_param(p)` returns only the single nearest
point, so it is insufficient here. Instead: scan `branch.params - p` for sign changes between
consecutive points and linearly interpolate `branch.states` at each crossing, which yields every
equilibrium on the branch at that `p`. Also include any point within `atol` of `p` to catch a
branch endpoint that touches `p` without crossing. `atol` defaults to a small multiple of the
median `|diff(params)|`.

Stability comes from `branch.stable` (interpolated crossings take the stability of the nearer
neighbor, and a crossing whose two neighbors disagree is drawn with the unstable style, since a
stability change between adjacent points means a bifurcation lies between them). Markers follow
the convention already used in `plot_continuation`: filled for stable, open/hollow for unstable.
When `branch.stable` is `None`, all points are drawn in the neutral `DEFAULT_STYLE` color from
`styles.py`.

If `branch.valid` is not `None` (a traced/`vmap`-ed result), mask the buffer to the real points
before doing any of the above.

### `plot_trajectory`

Integrates with `scipy.integrate.solve_ivp` — scipy is already a hard dependency
(`pyproject.toml`, `scipy>=1.7.0`), so this adds nothing to the dependency set and introduces no
`jaxcont[ode]` extra. Implementation:

```python
rhs = jax.jit(problem.as_rhs(p))
sol = solve_ivp(lambda t, y: np.asarray(rhs(jnp.asarray(y))),
                t_span, np.asarray(u0), t_eval=..., rtol=rtol, atol=atol)
```

The user never redefines their system: `as_rhs(p)` is the bridge. `jax.jit` is applied once so
the per-step JAX→NumPy round trip stays cheap; for a 2D system over a plotting-length interval
the overhead is negligible.

We do not write our own integrator. A fixed-step RK4 would be ~15 lines but a wrong default for
stiff neural models; an adaptive one with error control and dense output is 150+ lines plus its
own test suite, to reproduce something already installed.

`plot_trajectory` also accepts a pre-computed `(n_steps, 2)` array as its first argument, for
users who integrated with diffrax, lyapax, or anything else. In that form `u0`, `p`, and
`t_span` are unused and must be omitted; passing them raises `TypeError` with a message naming
the two calling forms. `arrow=True` adds a direction arrow at the trajectory midpoint.

### `plot_phase_plane`

Composes the primitives onto one axes in a fixed draw order (streamlines, vector field,
nullclines, trajectories, equilibria — cheapest to read on top). `result=` is optional; when
omitted, no equilibria are drawn. `trajectories=` accepts a list of `(u0, t_span)` pairs, or a
list of pre-computed `(n_steps, 2)` arrays. It deliberately does not accept bare initial
conditions: no default `t_span` is defensible across models, and silently picking one would
produce plots that are either empty transients or unreadable tangles depending on the system's
timescale.

Title defaults to `f"{problem.param_name or 'p'} = {p:g}"`.

## Renames

`plot_phase_portrait` (`src/jaxcont/viz/portraits.py`) currently scatters one dot per branch
point in state space, one per parameter value. That is not a phase portrait in the
dynamical-systems sense, and the name is needed for this feature's domain.

- Rename it to `plot_branch_states`, unchanged in behavior and signature.
- Keep `plot_phase_portrait` as a thin wrapper that emits `DeprecationWarning` and forwards.
- Export both from `jaxcont.viz`, with the deprecated one marked in `__all__` ordering and in the
  module docstring.
- Update the in-repo callers. There are no callers in `examples/` (`example_03_van_der_pol.py`
  imports only `plot_eigenvalues`); the only ones are in `tests/test_viz.py` — the two
  `ax`-handling tests and the `__all__` assertion.

The project is `Development Status :: 3 - Alpha` and pre-1.0, so a deprecation cycle here is
cheap. Removal target: v0.4.0. Record the rename in `CHANGELOG.md`.

## Testing

Extends `tests/test_viz.py`, following its existing use of the `Agg` backend and figure
teardown.

1. **Nullclines land where the analytic ones are.** For a system with known straight-line
   nullclines, assert the number of `ContourSet` paths and that sampled path vertices satisfy
   `f_i = 0` to tolerance.
2. **Equilibria coincide with nullcline intersections.** Continue a saddle-node problem, then
   assert each marker drawn by `plot_equilibria` sits on both nullclines.
3. **Both branches at a fold are drawn.** At a `p` where the continued branch has two states,
   assert `plot_equilibria` marks two points, not one — the case `Branch.at_param` cannot cover.
4. **Stability styling.** Stable points render filled, unstable hollow; `stable=None` renders
   neutral.
5. **Dimension guard.** A 3D problem raises `NotImplementedError` from every public entry point,
   with the dimension in the message.
6. **A caller-supplied `ax` is drawn onto.** Assert the returned figure is the caller's and that
   the caller's axes gained artists — the exact bug documented at `portraits.py` line 44, where
   an `ax` argument silently fell into `**kwargs`.
7. **Trajectory integration.** A linear system with a known stable spiral converges to the
   equilibrium; the array-input form draws the same vertices; passing both an array and `u0`
   raises `TypeError`.
8. **Deprecated alias.** `plot_phase_portrait` emits `DeprecationWarning` and produces a figure
   equivalent to `plot_branch_states`.

## Example

New `examples/example_12_fitzhugh_nagumo_phase_plane.py`: the FitzHugh–Nagumo model (2D, the
canonical teaching case), with a two-panel figure — bifurcation diagram with the Hopf marker on
the left, phase plane on the right showing cubic and linear nullclines, the equilibrium with its
stability, and a trajectory spiraling onto the limit cycle past the Hopf point.

`examples/example_03_van_der_pol.py` is also 2D; add a phase-plane overlay there once the module
lands.

Both are picked up automatically by the sphinx-gallery docs build.

## Roadmap placement

Slots after the equilibrium continuation API stabilized (v0.1) and alongside/before the
periodic-orbit work, since it is inexpensive and makes the limit-cycle examples far more
interpretable. Add to `notes/ROADMAP.md` as:

> Phase-plane visualization for 2D autonomous systems: nullclines, vector fields, streamlines,
> continuation equilibria, and trajectories.

## Prior art

Neither reference toolbox offers this as a first-class feature. MatCont plots trajectories and
state-space projections but has no built-in nullcline facility; users compose `contour` or
`fimplicit` themselves. BifurcationKit.jl provides branch plotting recipes and a user-supplied
`plot_solution` callback, with nullclines left to `Plots.jl`/`Makie.jl` composition at the
ecosystem level. A small, well-scoped phase-plane module is therefore a genuine differentiator
for teaching and low-dimensional neuroscience models, without diluting JaxCont's
differentiable-continuation identity.
