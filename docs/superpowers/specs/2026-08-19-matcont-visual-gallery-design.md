# MatCont Visual Comparison Gallery Design

## Purpose

JaxCont already validates supported continuation cases against reviewed MatCont
7.6 artifacts numerically. The gallery should make that correspondence visible
without weakening or duplicating the systematic validation. A user should be
able to see the two independently sampled branches, their bifurcation points,
and their stability information in the same figures. Where the validator finds
a known mismatch, the gallery must expose it rather than imply agreement.

## Scope

Create one Sphinx-Gallery example for each supported MatCont equilibrium and
periodic-orbit case:

- `MC-EQ-001`: cubic S-curve and two folds.
- `MC-EQ-002`: Van der Pol equilibrium and Hopf point.
- `MC-EQ-003`: adaptive-control equilibrium and Hopf point.
- `MC-LC-001`: analytic radial periodic orbit.
- `MC-LC-002`: torBPC periodic branch with LPC, NS, and PD points.

The PRC, transformation, codimension-two, and unsupported registry cases are
outside this change because they do not share the equilibrium or periodic
branch artifact schemas.

## Gallery Structure

Use a separate, runnable page for each model:

```text
examples/example_16_matcont_cubic_overlay.py
examples/example_17_matcont_vanderpol_overlay.py
examples/example_18_matcont_adaptive_control_overlay.py
examples/example_19_matcont_radial_cycle_overlay.py
examples/example_20_matcont_torbpc_overlay.py
```

The current generic `example_16_matcont_overlay.py` becomes the explicitly
named cubic example. Each page explains the model, invokes a shared renderer,
saves one PNG under `images/`, and calls `plt.show()`. Each must run both with
`python -m examples.<module>` from the repository root and under
Sphinx-Gallery's execution directory.

## Figure Design

All figures use the same visual language: JaxCont is a blue solid curve,
MatCont 7.6 is represented by orange open markers, and bifurcation markers use
consistent colors and shapes by event type. Axes are restricted to the shared
parameter domain when both packages cover different continuation tails.

### Equilibrium cases

The cubic page retains its branch/state overlay and LP markers.

The Van der Pol and adaptive-control pages use two panels:

1. continuation parameter versus selected equilibrium state;
2. spectral abscissa (largest real eigenvalue part) versus parameter.

The second panel is essential because both equilibrium branches are
geometrically simple. It visibly demonstrates the stability crossing at the
Hopf point instead of relying on two coincident zero-state lines. Hopf markers
appear in both panels where meaningful.

### Periodic-orbit cases

The radial-cycle page uses three panels:

1. orbit amplitude envelope versus continuation parameter;
2. period versus continuation parameter;
3. nontrivial Floquet-multiplier magnitude versus continuation parameter.

The torBPC page uses three panels:

1. selected state minimum and maximum along the orbit versus parameter;
2. period versus parameter;
3. critical Floquet multipliers in the complex plane at LPC, NS, and PD.

LPC, NS, and PD locations are marked consistently in the parameter-based
panels. The complex-plane panel includes the unit circle and overlays JaxCont
and MatCont multipliers with event-specific colors, making the `+1`, `-1`, and
complex unit-circle crossings directly inspectable.

`MC-LC-002` is intentionally a diagnostic rather than a passing comparison.
The existing validator reports that JaxCont misses the correctly located
LPC/PD labels and exceeds the NS critical-multiplier tolerance. Its figure
therefore displays `Systematic comparison: FAIL (known limitation)` and the
measured discrepancies. It must not move event locations, relax tolerances, or
visually claim correspondence that the systematic comparison rejects.

## Components and Interfaces

`examples/MatCont/visualize.py` remains the shared implementation boundary. It
will contain:

- low-level CSV and spectrum readers;
- equilibrium branch and spectrum plotting;
- periodic envelope, period, and Floquet plotting;
- a registered-case runner that loads the producer declared in `cases.json`;
- output saving and numerical summary annotation.

Public render functions accept a case ID, optional reference directory, output
path, labels, and title. They return a Matplotlib figure so callers and tests
can inspect or further customize it. Gallery scripts contain narrative and
model-specific labels only; they do not parse artifacts themselves.

No MATLAB process runs while building the gallery. The renderers consume the
committed, reviewed MatCont artifacts and freshly execute the corresponding
JaxCont producer.

## Data Flow and Validation

For each page:

1. Load the case entry from `cases.json`.
2. Import and run its registered JaxCont producer.
3. Load the reviewed MatCont branch, event, and spectrum CSV files.
4. Invoke the existing comparison path appropriate to that case.
5. Derive the displayed `PASS` or `FAIL` status from the systematic comparison.
6. Plot each package on its native adaptive mesh without fabricating paired
   sample points.
7. Annotate the figure with the most relevant maximum errors.
8. Save the PNG and return the figure.

For the equilibrium cases and radial cycle, reuse
`compare_case_result_to_reference`. For torBPC, use the producer's existing
`all_comparisons_pass` result and its event, period, extrema, and multiplier
error diagnostics, because that case already performs its specialized
comparison inside `run_torbpc_cycle`.

## Error Handling

Renderers raise clear errors for unknown or unsupported case IDs, missing
artifacts, incompatible CSV columns, missing event-to-branch links, or absent
spectra. A failed numerical comparison remains renderable for diagnosis, but
must receive a prominent `FAIL` annotation; it is never silently presented as
agreement.

Output directories are created when necessary. Existing reviewed reference
files remain read-only; gallery output is written only beneath the requested
image directory.

## Testing and Acceptance Criteria

Tests will verify:

- all five registered cases render with the expected number and type of axes;
- JaxCont and MatCont artists are present in every comparison panel;
- LP, H, LPC, NS, and PD markers appear at the appropriate locations;
- periodic extrema, periods, and Floquet spectra are mapped from both artifact
  schemas correctly;
- the shared parameter domain and model-specific labels are correct;
- the first four validation summaries report `PASS` and applicable errors;
- the torBPC summary reports `FAIL (known limitation)` and its event, period,
  extrema, and multiplier discrepancies without changing tolerances;
- every renderer writes a non-empty PNG;
- each gallery script executes in a subprocess with the documentation-style
  import path and working directory.

The existing MatCont validation suite and full project test suite must remain
green. The generated figures will also receive a visual inspection for
legibility, coincident overlays, event placement, and uncluttered legends.
