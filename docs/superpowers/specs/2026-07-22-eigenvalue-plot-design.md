# Generalized `plot_eigenvalues` design

**Date:** 2026-07-22

## Goal

Turn the bespoke eigenvalue-plot assembly in `example_03_van_der_pol.py` into
one reusable `jaxcont.viz.plot_eigenvalues()` API. The plot must combine real
and imaginary trajectories, JaxCont events, optional stability shading, and
external validation markers without requiring callers to manipulate artists.

## Public API

`plot_eigenvalues(data, ax=None, *, parameters=None, events=None,
show_events=True, shade_stability=False, references=None, param_name=None,
labels=None, figsize=(12, 5), titles=(...), legend=True, **kwargs)` accepts:

- `ContinuationResult` (preferred): branch arrays, events, stability, and the
  legacy solution's display metadata are inferred.
- `Branch`: branch arrays and stability are inferred.
- `ContinuationSolution`: legacy arrays, bifurcations, and labels are inferred.
- A raw `(n_points, n_eigenvalues)` complex array, with optional `parameters`.

A one-dimensional eigenvalue array is treated as one trajectory. Invalid
array ranks or mismatched point/label counts raise `ValueError` early.

`ax=None` creates the standard real/imag pair. A single `Axes` preserves the
existing real-only embedding behavior. A two-element axes sequence embeds both
panels into a caller-owned figure.

## Overlays

- JaxCont events are inferred or supplied with `events=`. Each event highlights
  all eigenvalues at the nearest branch point on both panels and uses the
  canonical style from `viz.styles`; labels are `JaxCont <code>`.
- `shade_stability=True` shades contiguous stable and unstable parameter
  regions using the continuation stability mask. It is off by default because
  parameter-axis shading can overlap on folded/non-monotone branches.
- `EigenvalueReference(parameter, label, ...)` describes an external vertical
  reference line. References are source-agnostic and therefore work for
  MatCont, BifurcationKit, and analytic results without library-specific API.
- Legends are de-duplicated independently on each axes. Titles, parameter
  label, trajectory labels, line kwargs, and reference styles are configurable.

## Compatibility

Existing `plot_eigenvalues(solution)` and `plot_eigenvalues(solution, ax=ax)`
calls keep their return type and panel behavior. The previously ignored
`**kwargs` are now forwarded to every trajectory line.

## Example migration

Example 03 passes its `ContinuationResult` directly, enables stability
shading, and supplies the MatCont Hopf value as an `EigenvalueReference`.
Its manual `axvline`, `axvspan`, `scatter`, and legend-de-duplication block is
removed.
