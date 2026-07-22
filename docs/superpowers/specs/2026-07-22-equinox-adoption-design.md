# Adopt `equinox` for v0.2 periodic-orbit types (prep + scaffold)

**Date:** 2026-07-22

## Goal

Resolve the open `eqx.Module` decision in `ARCHITECTURE.md` §4 — "adopt
`equinox` starting with the v0.2 periodic-orbit types, leave the shipped v0.1
equilibrium types as-is" — *before* periodic-orbit continuation itself is
built. The periodic-orbit types (`Collocation` predictor, mesh/`ntst`/`ncol`,
`PeriodDoubling`/`LPC`/`NS` events) don't exist yet, so this piece of work
cannot convert real types. Instead it: (a) adds the dependency, (b) proves
the specific pattern those future types will need — `eqx.Module` with a
static/traced field split — actually works end-to-end through `jit`/`vmap`,
and (c) leaves a concrete file for that future work to open and crib from
(or delete and replace) instead of starting from a blank page.

This is prep work, not feature work. It ships no new public API and changes
no existing behavior.

## Why `equinox`, briefly

`jit`/`vmap`/`grad` only trace JAX "pytrees." A plain Python class isn't one
until registered. JaxCont's v0.1 types (`BifProblem`, `Branch`) do this by
hand via `jax.tree_util.register_pytree_node` — fine for their flat shape.
Periodic-orbit types need richer objects with a mix of traced arrays (mesh
points, states) and compile-time constants (`ntst`/`ncol` — mesh size can't
be traced; it determines buffer shapes) plus several new `Event` subclasses.
`equinox.Module` + `eqx.field(static=True)` is the idiomatic way to express
that split without re-deriving `register_pytree_node` boilerplate for every
new type — and it's the diffrax-ecosystem-native choice this project already
aligns its API shape with (`ARCHITECTURE.md` §1, point 6).

## Scope

**In scope:**
1. Add `equinox` as a runtime dependency.
2. A small, explicitly-throwaway `eqx.Module` scaffold living in `src/` (not
   `tests/`), proving the static/traced split compiles, `jit`s, and `vmap`s
   correctly.
3. A test exercising that scaffold under `jit` and `vmap`.
4. Update `ARCHITECTURE.md` §4 and `ROADMAP.md` item 2 to mark the decision
   resolved, pointing here.

**Explicitly out of scope** (this is the actual v0.2 feature work, not prep):
- A real `Collocation` predictor.
- `PeriodDoubling` / `LPC` / `NS` `Event` implementations.
- Wiring `BifProblem.kind="periodic"` to anything.
- Any change to the v0.1 equilibrium types (`BifProblem`, `Branch`,
  `ContinuationResult`) — they stay hand-rolled `register_pytree_node`
  dataclasses, per the existing 2026-07-19 decision.

## Design

### 1. Dependency

Add `"equinox>=0.11"` to `pyproject.toml`'s main `dependencies` list (not
`dev`/`docs` extras) — the scaffold lives in `src/jaxcont/core/`, so it's a
real runtime import, matching how the eventual periodic-orbit code will use
it too.

### 2. Scaffold module

New file: `src/jaxcont/core/_periodic_eqx_scaffold.py`. Leading underscore
signals "internal, not part of the public surface" — consistent with how
`ARCHITECTURE.md` §8 already describes other not-yet-real v0.2/v0.3 stubs as
"importable from their submodule" but absent from top-level `__init__.py`.

Contents: one small `eqx.Module`, `CollocationMeshScaffold`, with:
- `ntst: int = eqx.field(static=True)` — number of mesh intervals; compile-
  time, since it determines buffer shapes (mirrors `max_steps` already being
  a static arg to the v0.1 scan engines).
- `ncol: int = eqx.field(static=True)` — collocation points per interval;
  compile-time for the same reason.
- `mesh: Array` — traced normalized mesh points in `[0, 1]`.

The module docstring states plainly that this is a throwaway pattern-proof
to be deleted and replaced once real periodic-orbit types are built, with a
pointer back to this design doc.

### 3. Test

New file: `tests/test_equinox_scaffold.py`. Covers:
- `jit` a function taking `(CollocationMeshScaffold, Array)` and returning an
  array derived from both fields; confirm it runs and gives the expected
  numeric result.
- `vmap` that same function over a batch of traced values (e.g. periods)
  with one `CollocationMeshScaffold` instance held fixed (closed over, not
  batched) — the exact "static config, batched dynamic input" pattern
  periodic-orbit continuation will need under `vmap` sweeps.
- Assert `ntst`/`ncol` are usable as Python-level shape constants inside the
  jitted function body (e.g. `jnp.zeros((m.ntst, m.ncol))`), proving they
  are actually static rather than silently traced (a static field used in a
  shape position would raise a `TracerArrayConversionError`/`ConcretizationTypeError`
  if the split were wrong — that failure mode is exactly what this test
  guards against).

### 4. Docs

- `ARCHITECTURE.md` §4: append one line to the existing "Decision
  (2026-07-19)" callout noting the dependency is now added and the pattern
  is proven in `_periodic_eqx_scaffold.py`, dated 2026-07-22.
- `ROADMAP.md`: mark item 2 of "Engineering / architecture recommendations
  for v0.2" ✅, in the same short-summary style already used for item 1
  (engine consolidation).

## Testing

One new pytest file (`tests/test_equinox_scaffold.py`), 2-3 test cases as
described above. Run via the existing `JAX_PLATFORMS=cpu pytest tests/ -q`
convention; expect all-green with the new tests added to the count.

## Error handling

None needed — a throwaway scaffold with no user-facing entry point has no
failure modes to guard against beyond what the test above already checks at
build time (the static/traced split either compiles correctly or the test
fails).
