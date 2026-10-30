# Precision policy: float64 only

Status: approved by the owner on 2026-10-29, replacing the same day's
"limited scope" float32 policy (which kept float32 as best effort with a
`PrecisionWarning` per feature).

## Decision

JaxCont computes in float64 only.

- `import jaxcont` turns on `jax_enable_x64` when it is off and emits one
  `jaxcont.PrecisionWarning` saying it did so and how to opt out.
- `JAXCONT_NO_AUTO_X64=1` leaves the global JAX setting untouched. JaxCont
  entry points then raise `RuntimeError` ("JaxCont requires float64 ...")
  instead of computing in float32.
- Floating inputs created as float32 (before the import, or explicitly) are
  converted to float64 (complex to complex128) at the entry points:
  `bif_problem`, `periodic_orbit_problem`, `fold_curve_problem`,
  `hopf_curve_problem`, `fold_point`, `hopf_point`, the codim-2 point
  solvers, `prc_curve`/`dprc_curve`/`branch_prc`, the Floquet spectrum
  functions and `branch_switch_pd`.

## Why

All validation against MatCont and BifurcationKit is float64; the reviews of
2026-10-29 found that float32 results for periodic orbits, Floquet
multipliers and codim-2 points are not trustworthy, several defaults
(Newton `1e-8`..`1e-12`) are below float32 resolution, and running the suite
in both precisions doubled CI time and put precision-dependent tolerances on
many assertions. `jax_enable_x64` is global, so turning it on also affects
the caller's other JAX code (memory, GPU speed); the warning and the opt-out
make that explicit instead of silent.

## Changes

1. `jaxcont.utils.precision`: `PrecisionWarning`, `ensure_x64()` (called at
   the top of `jaxcont/__init__.py`), `as_float64(tree)` used by the entry
   points.
2. Tests always run in float64; the `JAXCONT_TEST_X64` matrix, the float32
   CI/publish jobs and the `float64_only` marker are removed. Tests that pass
   float32 arrays now check their conversion.
3. Docs: installation/quickstart precision note, migration guide entry,
   CHANGELOG 0.5.0 Breaking entry.

Not changed in this step: dtype-dependent tolerances inside low-level
helpers (`default_tol`) stay; they are simply always float64 now.
